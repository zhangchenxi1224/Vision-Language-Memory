from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


from launch_background import STRICT_ENVIRONMENT, validate_command
from model_snapshot_manifest import verify_snapshot_binding
from r3_dag_contract import (
    PLAN_PROTOCOL,
    SHA256_PATTERN,
    STAGE_EVIDENCE_PROTOCOL,
    STAGE_SPEC_PROTOCOL,
    atomic_json,
    is_within,
    load_json_object,
    require_json_values,
    sha256_file,
    verify_bound_artifact,
    verify_clean_commit,
    verify_sha_sidecar,
)


def _load_worker_binding(
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    spec_sha256: str,
    plan: Mapping[str, Any],
) -> tuple[Path, str]:
    required_environment = (
        "VLM_STAGE_WORKER_INPUT",
        "VLM_STAGE_CONFIGURATION_SHA256",
        "VLM_STAGE_PREFLIGHT",
        "VLM_STAGE_PREFLIGHT_SHA256",
    )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        raise ValueError(f"R3 materialized stages must run through launch_background.py; missing {missing}")
    worker_input_path = Path(os.environ["VLM_STAGE_WORKER_INPUT"]).resolve()
    configuration_sha256 = os.environ["VLM_STAGE_CONFIGURATION_SHA256"]
    if SHA256_PATTERN.fullmatch(configuration_sha256) is None:
        raise ValueError("Launcher configuration SHA256 is malformed")
    if not worker_input_path.is_file() or sha256_file(worker_input_path) != configuration_sha256:
        raise ValueError("Launcher worker_input.json does not match its bound SHA256")
    configuration = load_json_object(worker_input_path)
    expected_runner_command = [
        str(plan["python"]),
        str(Path(spec["repo"]) / "scripts" / "inspire" / "run_r3_stage.py"),
        "--spec",
        str(spec_path),
        "--spec-sha256",
        spec_sha256,
    ]
    require_json_values(
        configuration,
        {
            "stage": spec["launcher_stage"],
            "repo": spec["repo"],
            "run_root": spec["run_root"],
            "run_dir": spec["run_dir"],
            "expected_commit": spec["expected_commit"],
            "preflight": spec["formal_preflight"]["path"],
            "preflight_sha256": spec["formal_preflight"]["sha256"],
            "infrastructure_stage": False,
            "strict_environment": STRICT_ENVIRONMENT,
            "command": expected_runner_command,
        },
        "launcher worker input",
    )
    if worker_input_path.parent != Path(spec["run_dir"]):
        raise ValueError("Launcher worker input is not inside the unique bound stage run directory")
    if Path(os.environ["VLM_STAGE_PREFLIGHT"]).resolve() != Path(spec["formal_preflight"]["path"]):
        raise ValueError("Launcher formal preflight path does not match the stage specification")
    if os.environ["VLM_STAGE_PREFLIGHT_SHA256"] != spec["formal_preflight"]["sha256"]:
        raise ValueError("Launcher formal preflight SHA256 does not match the stage specification")
    return worker_input_path, configuration_sha256


def _validate_spec_against_plan(spec: Mapping[str, Any], plan: Mapping[str, Any]) -> Mapping[str, Any]:
    require_json_values(
        spec,
        {
            "schema_version": 1,
            "protocol": STAGE_SPEC_PROTOCOL,
            "expected_commit": plan["expected_commit"],
            "repo": plan["repo"],
            "run_root": plan["run_root"],
            "formal_preflight": plan["formal_preflight"],
        },
        "stage specification",
    )
    stage = spec.get("stage")
    if stage not in plan.get("strict_order", []):
        raise ValueError(f"Stage {stage!r} is not part of the immutable DAG plan")
    definition = plan["stages"][stage]
    require_json_values(
        spec,
        {
            "stage_index": definition["index"],
            "stage_slug": definition["slug"],
            "launcher_stage": definition["launcher_stage"],
            "run_dir": definition["run_dir"],
            "commands": definition["commands"],
            "outputs": definition["outputs"],
            "evidence_path": definition["evidence_path"],
        },
        "stage specification versus plan",
    )
    return definition


def run_bound_stage(spec_path: Path, spec_sha256: str) -> dict[str, Any]:
    spec_path = spec_path.resolve()
    if SHA256_PATTERN.fullmatch(spec_sha256) is None:
        raise ValueError("spec_sha256 must be a lowercase SHA256 digest")
    verify_sha_sidecar(spec_path, expected_sha256=spec_sha256)
    spec = load_json_object(spec_path)
    plan_path = Path(spec["plan_path"]).resolve()
    plan_sha256 = str(spec["plan_sha256"])
    verify_sha_sidecar(plan_path, expected_sha256=plan_sha256)
    plan = load_json_object(plan_path)
    require_json_values(plan, {"schema_version": 1, "protocol": PLAN_PROTOCOL}, "DAG plan")
    _validate_spec_against_plan(spec, plan)

    repo = Path(spec["repo"]).resolve()
    run_root = Path(spec["run_root"]).resolve()
    run_dir = Path(spec["run_dir"]).resolve()
    evidence_path = Path(spec["evidence_path"]).resolve()
    if not is_within(spec_path, run_root) or not is_within(run_dir, run_root) or not is_within(evidence_path, run_root):
        raise ValueError("Stage specification, run directory, and evidence must remain inside the immutable run root")
    verify_clean_commit(repo, str(spec["expected_commit"]))
    preflight_path = Path(spec["formal_preflight"]["path"]).resolve()
    verify_sha_sidecar(preflight_path, expected_sha256=str(spec["formal_preflight"]["sha256"]))
    preflight = load_json_object(preflight_path)
    require_json_values(preflight, {"passed": True, "formal_ready": True}, "formal preflight")
    preflight_git = preflight.get("git")
    if not isinstance(preflight_git, Mapping) or preflight_git.get("commit") != spec["expected_commit"]:
        raise ValueError("Formal preflight is not bound to the stage Git commit")
    worker_input_path, configuration_sha256 = _load_worker_binding(
        spec=spec,
        spec_path=spec_path,
        spec_sha256=spec_sha256,
        plan=plan,
    )
    model_snapshots = plan.get("model_snapshots")
    production_plan = plan.get("kind") in {"technical", "teacher-preparation", "micro-extension"}
    if production_plan and (
        not isinstance(model_snapshots, Mapping)
        or set(model_snapshots) != {"dreamlite_mobile", "qwen_reader"}
    ):
        raise ValueError("Immutable DAG plan does not bind both model snapshot manifests")
    if not production_plan:
        model_snapshots = {}

    def verify_model_entry_paths() -> None:
        model_root = Path(str(plan.get("model_root", "")))
        if model_root.is_symlink() or not model_root.is_dir():
            raise ValueError("DAG model_root must remain a real directory, not a symlink")
        expected_directories = {
            "dreamlite_mobile": model_root / "DreamLite-mobile",
            "qwen_reader": model_root / "Qwen3-VL-4B-Instruct",
        }
        for name, entry in expected_directories.items():
            binding = model_snapshots[name]
            if entry.is_symlink() or not entry.is_dir() or str(entry) != binding.get("model_dir"):
                raise ValueError(f"Model command entry path drifted from the {name} snapshot binding")

    def verify_runtime_integrity(*, verify_models: bool = False) -> None:
        verify_clean_commit(repo, str(spec["expected_commit"]))
        verify_sha_sidecar(spec_path, expected_sha256=spec_sha256)
        verify_sha_sidecar(plan_path, expected_sha256=plan_sha256)
        verify_sha_sidecar(preflight_path, expected_sha256=str(spec["formal_preflight"]["sha256"]))
        if sha256_file(worker_input_path) != configuration_sha256:
            raise ValueError("Launcher worker_input.json changed while the stage was running")
        for bound_input in spec.get("prerequisites", []):
            if not isinstance(bound_input, Mapping):
                raise ValueError("Every prerequisite must be an object")
            verify_bound_artifact(bound_input)
        if verify_models and model_snapshots:
            verify_model_entry_paths()
            for model_binding in model_snapshots.values():
                if not isinstance(model_binding, Mapping):
                    raise ValueError("Model snapshot binding must be an object")
                verify_snapshot_binding(model_binding)

    if evidence_path.exists() or evidence_path.with_suffix(evidence_path.suffix + ".sha256").exists():
        raise ValueError(f"Stage evidence already exists and cannot be overwritten: {evidence_path}")
    for prerequisite in spec.get("prerequisites", []):
        if not isinstance(prerequisite, Mapping):
            raise ValueError("Every prerequisite must be an object")
        verify_bound_artifact(prerequisite)
    for output in spec["outputs"]:
        output_path = Path(output["path"]).resolve()
        if not is_within(output_path, run_root):
            raise ValueError(f"Stage output escapes the immutable run root: {output_path}")
        if output_path.exists():
            raise ValueError(f"Stage refuses a stale/pre-existing output: {output_path}")

    commands = spec["commands"]
    command_results: list[dict[str, Any]] = []
    verify_runtime_integrity(verify_models=True)
    command_environment = os.environ.copy()
    if model_snapshots:
        command_environment.update(
            {
            "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256": str(
                model_snapshots["dreamlite_mobile"]["manifest_sha256"]
            ),
            "VLM_READER_SNAPSHOT_MANIFEST_SHA256": str(
                model_snapshots["qwen_reader"]["manifest_sha256"]
            ),
            }
        )
    for index, command in enumerate(commands):
        if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
            raise ValueError("Every stage command must be an argv string list")
        validate_command(command)
        verify_runtime_integrity()
        print(json.dumps({"event": "command_started", "index": index, "argv": command}, sort_keys=True), flush=True)
        process = subprocess.run(command, cwd=repo, env=command_environment, check=False)
        command_results.append({"index": index, "exit_code": process.returncode})
        if process.returncode != 0:
            raise RuntimeError(f"Stage command {index} failed with exit code {process.returncode}")
        verify_runtime_integrity()

    verify_runtime_integrity()
    materialized_outputs: list[dict[str, Any]] = []
    for output in spec["outputs"]:
        output_path = Path(output["path"]).resolve()
        if not output_path.is_file():
            raise ValueError(f"Required stage output is missing: {output_path}")
        binding = {
            "label": output["label"],
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "required_values": output.get("required_values", {}),
        }
        verify_bound_artifact(binding)
        materialized_outputs.append(binding)

    verify_runtime_integrity(verify_models=True)
    evidence = {
        "schema_version": 1,
        "protocol": STAGE_EVIDENCE_PROTOCOL,
        "stage": spec["stage"],
        "stage_slug": spec["stage_slug"],
        "launcher_stage": spec["launcher_stage"],
        "passed": True,
        "expected_commit": spec["expected_commit"],
        "plan_sha256": plan_sha256,
        "stage_spec_sha256": spec_sha256,
        "worker_input_path": str(worker_input_path),
        "configuration_sha256": configuration_sha256,
        "formal_preflight_sha256": spec["formal_preflight"]["sha256"],
        "model_snapshots": {name: dict(binding) for name, binding in model_snapshots.items()},
        "prerequisites": spec.get("prerequisites", []),
        "commands": command_results,
        "outputs": materialized_outputs,
    }
    evidence_sha256 = atomic_json(evidence_path, evidence)
    return {**evidence, "evidence_path": str(evidence_path), "evidence_sha256": evidence_sha256}


def _write_failure(spec_path: Path, error: BaseException) -> None:
    worker_input = os.environ.get("VLM_STAGE_WORKER_INPUT")
    if not worker_input:
        return
    run_dir = Path(worker_input).resolve().parent
    failure = {
        "schema_version": 1,
        "protocol": "r3-inspire-stage-wrapper-failure.v1",
        "passed": False,
        "spec_path": str(spec_path.resolve()),
        "error": f"{type(error).__name__}: {error}",
    }
    atomic_json(run_dir / "wrapper_failure.json", failure)


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one immutable SHA-bound R3 stage inside launch_background.py")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    args = parser.parse_args()
    try:
        report = run_bound_stage(args.spec, args.spec_sha256)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        _write_failure(args.spec, exc)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 3
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
