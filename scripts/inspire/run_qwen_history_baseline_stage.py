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
from qwen_history_baseline_contract import (
    PLAN_PROTOCOL,
    STAGE_EVIDENCE_PROTOCOL,
    STAGE_SPEC_PROTOCOL,
    validate_replica_pair,
    validate_scientific_command,
    verify_declared_output,
)
from r3_dag_contract import (
    SHA256_PATTERN,
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
    required = (
        "VLM_STAGE_WORKER_INPUT",
        "VLM_STAGE_CONFIGURATION_SHA256",
        "VLM_STAGE_PREFLIGHT",
        "VLM_STAGE_PREFLIGHT_SHA256",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ValueError(f"Baseline stages must run through launch_background.py; missing {missing}")
    worker_input = Path(os.environ["VLM_STAGE_WORKER_INPUT"]).resolve()
    configuration_sha256 = os.environ["VLM_STAGE_CONFIGURATION_SHA256"]
    if SHA256_PATTERN.fullmatch(configuration_sha256) is None:
        raise ValueError("Launcher configuration SHA256 is malformed")
    if not worker_input.is_file() or sha256_file(worker_input) != configuration_sha256:
        raise ValueError("Launcher worker_input.json does not match its bound SHA256")
    configuration = load_json_object(worker_input)
    expected_runner = [
        str(plan["python"]),
        str(Path(spec["repo"]) / "scripts" / "inspire" / "run_qwen_history_baseline_stage.py"),
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
            "command": expected_runner,
        },
        "launcher worker input",
    )
    if worker_input.parent != Path(spec["run_dir"]).resolve():
        raise ValueError("Launcher worker input must be inside the unique stage directory")
    if Path(os.environ["VLM_STAGE_PREFLIGHT"]).resolve() != Path(spec["formal_preflight"]["path"]).resolve():
        raise ValueError("Launcher preflight path differs from the baseline stage specification")
    if os.environ["VLM_STAGE_PREFLIGHT_SHA256"] != spec["formal_preflight"]["sha256"]:
        raise ValueError("Launcher preflight SHA256 differs from the baseline stage specification")
    return worker_input, configuration_sha256


def _validate_spec(spec: Mapping[str, Any], plan: Mapping[str, Any]) -> Mapping[str, Any]:
    require_json_values(
        plan,
        {
            "schema_version": 1,
            "protocol": PLAN_PROTOCOL,
            "kind": "qwen-full-history-baseline",
            "strict_order": ["BH0", "BH1", "BH2", "BH3"],
        },
        "baseline DAG plan",
    )
    require_json_values(
        spec,
        {
            "schema_version": 1,
            "protocol": STAGE_SPEC_PROTOCOL,
            "expected_commit": plan["expected_commit"],
            "repo": plan["repo"],
            "run_root": plan["run_root"],
            "formal_preflight": plan["formal_preflight"],
            "amendment": plan["amendment"],
        },
        "baseline stage specification",
    )
    stage = spec.get("stage")
    if stage not in plan["strict_order"]:
        raise ValueError(f"Unknown baseline stage {stage!r}")
    definition = plan["stages"][stage]
    require_json_values(
        spec,
        {
            "stage_index": definition["index"],
            "stage_slug": definition["slug"],
            "launcher_stage": definition["launcher_stage"],
            "run_dir": definition["run_dir"],
            "parallel_groups": definition["parallel_groups"],
            "serial_commands": definition["serial_commands"],
            "outputs": definition["outputs"],
            "evidence_path": definition["evidence_path"],
        },
        "baseline stage specification versus plan",
    )
    for pair in spec["parallel_groups"]:
        validate_replica_pair(pair)
    for command in spec["serial_commands"]:
        validate_scientific_command(command)
    return definition


def _run_parallel_pair(
    pair: list[dict[str, Any]],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> list[dict[str, Any]]:
    validate_replica_pair(pair)
    processes: list[tuple[dict[str, Any], subprocess.Popen[bytes]]] = []
    for entry in pair:
        argv = list(entry["argv"])
        validate_command(argv)
        print(
            json.dumps(
                {"event": "replica_started", "replica": entry["replica"], "device": entry["device"], "argv": argv},
                sort_keys=True,
            ),
            flush=True,
        )
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
        )
        processes.append((entry, process))
    results: list[dict[str, Any]] = []
    failure = False
    for entry, process in processes:
        exit_code = process.wait()
        failure = failure or exit_code != 0
        results.append(
            {
                "replica": entry["replica"],
                "device": entry["device"],
                "exit_code": exit_code,
            }
        )
    if failure:
        raise RuntimeError(f"One or more baseline replicas failed: {results}")
    return results


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
    _validate_spec(spec, plan)

    repo = Path(spec["repo"]).resolve()
    run_root = Path(spec["run_root"]).resolve()
    run_dir = Path(spec["run_dir"]).resolve()
    evidence_path = Path(spec["evidence_path"]).resolve()
    if not all(is_within(path, run_root) for path in (spec_path, run_dir, evidence_path)):
        raise ValueError("Baseline stage artifacts must remain inside the immutable run root")
    verify_clean_commit(repo, str(spec["expected_commit"]))
    preflight_path = Path(spec["formal_preflight"]["path"]).resolve()
    verify_sha_sidecar(preflight_path, expected_sha256=str(spec["formal_preflight"]["sha256"]))
    preflight = load_json_object(preflight_path)
    require_json_values(preflight, {"passed": True, "formal_ready": True}, "formal preflight")
    preflight_git = preflight.get("git")
    if not isinstance(preflight_git, Mapping) or preflight_git.get("commit") != spec["expected_commit"]:
        raise ValueError("Formal preflight was produced from a different commit")
    worker_input, configuration_sha256 = _load_worker_binding(
        spec=spec,
        spec_path=spec_path,
        spec_sha256=spec_sha256,
        plan=plan,
    )
    reader_binding = plan.get("qwen_reader_snapshot")
    if not isinstance(reader_binding, Mapping):
        raise ValueError("Baseline plan must bind exactly one Qwen Reader snapshot")

    def verify_runtime(*, verify_reader: bool = False) -> None:
        verify_clean_commit(repo, str(spec["expected_commit"]))
        verify_sha_sidecar(spec_path, expected_sha256=spec_sha256)
        verify_sha_sidecar(plan_path, expected_sha256=plan_sha256)
        verify_sha_sidecar(preflight_path, expected_sha256=str(spec["formal_preflight"]["sha256"]))
        if sha256_file(worker_input) != configuration_sha256:
            raise ValueError("Launcher worker input changed while the baseline stage was running")
        for prerequisite in spec.get("prerequisites", []):
            verify_bound_artifact(prerequisite)
        if verify_reader:
            verify_snapshot_binding(reader_binding)

    if evidence_path.exists() or evidence_path.with_suffix(evidence_path.suffix + ".sha256").exists():
        raise ValueError(f"Baseline stage evidence already exists: {evidence_path}")
    for output in spec["outputs"]:
        output_path = Path(output["path"]).resolve()
        if not is_within(output_path, run_root):
            raise ValueError(f"Baseline output escapes the immutable run root: {output_path}")
        if output_path.exists():
            raise ValueError(f"Baseline stage refuses stale output: {output_path}")

    environment = os.environ.copy()
    environment.pop("VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256", None)
    environment["VLM_READER_SNAPSHOT_MANIFEST_SHA256"] = str(reader_binding["manifest_sha256"])
    parallel_results: list[dict[str, Any]] = []
    serial_results: list[dict[str, Any]] = []
    verify_runtime(verify_reader=True)
    for index, pair in enumerate(spec["parallel_groups"]):
        verify_runtime()
        results = _run_parallel_pair(pair, cwd=repo, environment=environment)
        parallel_results.append({"index": index, "replicas": results})
        verify_runtime()
    for index, command in enumerate(spec["serial_commands"]):
        validate_command(command)
        validate_scientific_command(command)
        verify_runtime()
        print(json.dumps({"event": "serial_command_started", "index": index, "argv": command}, sort_keys=True), flush=True)
        process = subprocess.run(command, cwd=repo, env=environment, stdin=subprocess.DEVNULL, check=False)
        serial_results.append({"index": index, "exit_code": process.returncode})
        if process.returncode != 0:
            raise RuntimeError(f"Baseline serial command {index} failed with exit code {process.returncode}")
        verify_runtime()

    materialized_outputs: list[dict[str, Any]] = []
    for output in spec["outputs"]:
        verify_declared_output(output)
        path = Path(output["path"]).resolve()
        materialized_outputs.append(
            {
                "label": output["label"],
                "path": str(path),
                "sha256": sha256_file(path),
                "required_values": output.get("required_values", {}),
                "validator": output.get("validator"),
                "suite": output.get("suite"),
            }
        )
    verify_runtime(verify_reader=True)
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
        "worker_input_path": str(worker_input),
        "configuration_sha256": configuration_sha256,
        "formal_preflight_sha256": spec["formal_preflight"]["sha256"],
        "amendment_sha256": spec["amendment"]["sha256"],
        "qwen_reader_snapshot": dict(reader_binding),
        "dreamlite_loaded": False,
        "training_performed": False,
        "prerequisites": spec.get("prerequisites", []),
        "parallel_groups": parallel_results,
        "serial_commands": serial_results,
        "outputs": materialized_outputs,
    }
    evidence_sha256 = atomic_json(evidence_path, evidence)
    return {**evidence, "evidence_path": str(evidence_path), "evidence_sha256": evidence_sha256}


def _write_failure(spec_path: Path, error: BaseException) -> None:
    worker_input = os.environ.get("VLM_STAGE_WORKER_INPUT")
    if not worker_input:
        return
    run_dir = Path(worker_input).resolve().parent
    atomic_json(
        run_dir / "wrapper_failure.json",
        {
            "schema_version": 1,
            "protocol": "r3-inspire-qwen-history-baseline-wrapper-failure.v1",
            "passed": False,
            "spec_path": str(spec_path.resolve()),
            "error": f"{type(error).__name__}: {error}",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one SHA-bound Qwen history baseline stage")
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
