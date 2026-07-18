from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
PROBES = ROOT / "scripts" / "probes"
sys.path.insert(0, str(PROBES))
sys.path.insert(0, str(ROOT / "src"))

from launch_background import STRICT_ENVIRONMENT, validate_command, verify_preflight  # noqa: E402
from model_snapshot_manifest import verify_snapshot_binding, verify_snapshot_manifest  # noqa: E402
from r3_dag_contract import (  # noqa: E402
    LAUNCH_COMMAND_PROTOCOL,
    COMMIT_PATTERN,
    MICRO_COMMAND_PROTOCOL,
    PLAN_PROTOCOL,
    RUN_NAME_PATTERN,
    SHA256_PATTERN,
    STAGE_EVIDENCE_PROTOCOL,
    STAGE_SPEC_PROTOCOL,
    atomic_json,
    git,
    is_within,
    load_json_object,
    require_absolute,
    require_absolute_executable,
    require_file_sha,
    require_json_values,
    sha256_file,
    verify_bound_artifact,
    verify_clean_commit,
    verify_sha_sidecar,
)
from validate_r3_micro_prerequisites import validate_prerequisites  # noqa: E402
from validate_r3_technical_gates import CHOICES, OVERWRITE_EVENT, QUERY, SET_EVENT  # noqa: E402
from vision_memory.reader import R3_QWEN_READER_RESIZE_CONTRACT  # noqa: E402
from vision_memory.teacher import load_teacher_calibration_input_lock  # noqa: E402


TECHNICAL_ORDER = ("R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L", "DL-S")
TECHNICAL_SLUGS = {
    "R3-R0": "00-r3-r0",
    "R3-S0": "01-r3-s0",
    "G4-L": "02-g4-l",
    "G5-L": "03-g5-l",
    "G6-L": "04-g6-l",
    "DL-S": "05-dl-s",
}
TECHNICAL_LAUNCHER_STAGES = {
    "R3-R0": "r3-r0",
    "R3-S0": "r3-s0",
    "G4-L": "g4-l",
    "G5-L": "g5-l",
    "G6-L": "g6-l",
    "DL-S": "dl-s",
}
TEACHER_PREPARATION_ORDER = (
    "R3-TC0",
    "R3-TF0",
    "T0",
    "CAL-Set8",
    "CAL-Transition16",
)
TEACHER_PREPARATION_SLUGS = {
    "R3-TC0": "00-r3-tc0",
    "R3-TF0": "01-r3-tf0",
    "T0": "02-t0",
    "CAL-Set8": "03-cal-set8",
    "CAL-Transition16": "04-cal-transition16",
}
TEACHER_PREPARATION_LAUNCHER_STAGES = {
    "R3-TC0": "r3-teacher-tc0",
    "R3-TF0": "r3-teacher-tf0",
    "T0": "r3-teacher-t0",
    "CAL-Set8": "r3-teacher-cal-set8",
    "CAL-Transition16": "r3-teacher-cal-transition16",
}


def _command(*parts: str | Path) -> list[str]:
    command = [str(part) for part in parts]
    validate_command(command)
    return command


def _output(label: str, path: Path, *, json_passed: bool = True) -> dict[str, Any]:
    binding: dict[str, Any] = {"label": label, "path": str(path), "required_values": {}}
    if json_passed:
        binding["required_values"] = {"passed": True}
    return binding


def _probe_command(
    *,
    python: Path,
    repo: Path,
    model_root: Path,
    events: tuple[str, ...],
    target_index: int,
    output: Path,
    detach: bool = False,
) -> list[str]:
    command = _command(
        python,
        repo / "scripts" / "probes" / "e2e_episode_grad.py",
        "--dreamlite",
        model_root / "DreamLite-mobile",
        "--reader",
        model_root / "Qwen3-VL-4B-Instruct",
    )
    for event in events:
        command.extend(("--event", event))
    command.extend(("--query", QUERY, "--reader-loss-mode", "listwise-choice"))
    for choice in CHOICES:
        command.extend(("--choice", choice))
    command.extend(
        (
            "--target-index",
            str(target_index),
            "--resolution",
            "1024",
            "--adapter-seed",
            "0",
            "--noise-seed",
            "0",
            "--lora-rank",
            "4",
            "--checkpoint-unet",
            "--dreamlite-device",
            "cuda:0",
            "--reader-device",
            "cuda:1",
        )
    )
    if detach:
        command.append("--detach-between-events")
    command.extend(("--output-json", str(output)))
    validate_command(command)
    return command


def _verified_model_snapshots(
    *,
    repo: Path,
    model_root: Path,
    preflight: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    lock = load_json_object(repo / "models.lock.json")
    model_reports = preflight.get("models")
    if not isinstance(model_reports, Mapping):
        raise ValueError("Formal preflight does not contain model snapshot reports")
    bindings: dict[str, dict[str, Any]] = {}
    for name, specification in lock.get("models", {}).items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"models.lock.json entry {name!r} is malformed")
        model_dir = model_root / Path(str(specification["local_dir"])).name
        current = verify_snapshot_manifest(
            manifest_path=model_dir / str(specification["snapshot_manifest"]),
            model_dir=model_dir,
            expected_repo_id=str(specification["repo_id"]),
            expected_revision=str(specification["revision"]),
        )
        reported_model = model_reports.get(name)
        reported = reported_model.get("snapshot_manifest") if isinstance(reported_model, Mapping) else None
        if not isinstance(reported, Mapping) or dict(reported) != current:
            raise ValueError(f"Formal preflight model snapshot binding drifted for {name}")
        verify_snapshot_binding(current)
        bindings[str(name)] = current
    if set(bindings) != set(lock.get("models", {})):
        raise ValueError("Formal preflight does not bind every locked model snapshot")
    return bindings


def _technical_validation_command(
    *,
    python: Path,
    repo: Path,
    results: Path,
    through: str,
    output: Path,
) -> list[str]:
    command = _command(
        python,
        repo / "scripts" / "probes" / "validate_r3_technical_gates.py",
        "--through",
        through,
        "--resize-contract",
        results / "R3_R0_qwen_resize_contract.json",
    )
    if through != "R3-R0":
        command.extend(("--g4", str(results / "G4_L.json")))
    if through in {"G5-L", "G6-L", "DL-S"}:
        command.extend(("--g5", str(results / "G5_L.json")))
    if through in {"G6-L", "DL-S"}:
        command.extend(("--g6", str(results / "G6_L_detached.json")))
    if through == "DL-S":
        command.extend(
            (
                "--scorer-s0",
                str(results / "R3_S0_qwen_scorer_contract.json"),
                "--resume-report",
                str(results / "DL_S_resume_equivalence.json"),
            )
        )
    command.extend(("--pair-atol", "1e-5", "--pair-rtol", "1e-4", "--output", str(output)))
    validate_command(command)
    return command


def _training_command(
    *,
    python: Path,
    repo: Path,
    model_root: Path,
    train: Path,
    dev: Path,
    output_dir: Path,
    resume: Path | None = None,
) -> list[str]:
    command = _command(
        python,
        repo / "scripts" / "train" / "dreamlite_episode.py",
        "--train",
        train,
        "--dev",
        dev,
        "--dataset-format",
        "synthetic",
        "--dreamlite",
        model_root / "DreamLite-mobile",
        "--reader",
        model_root / "Qwen3-VL-4B-Instruct",
        "--reader-loss-mode",
        "listwise-choice",
        "--choice-view-schedule",
        "cyclic4",
        "--training-regime",
        "qa_only",
        "--objective-stage",
        "qa",
        "--initial-state-mode",
        "blank",
        "--output-dir",
        output_dir,
        "--resolution",
        "1024",
        "--seed",
        "0",
        "--adapter-seed",
        "0",
        "--learning-rate",
        "1e-4",
        "--weight-decay",
        "0.01",
        "--lora-rank",
        "4",
        "--epochs",
        "2",
        "--gradient-accumulation",
        "1",
        "--gradient-clip",
        "1.0",
        "--checkpoint-every",
        "8",
        "--eval-every",
        "100000",
        "--eval-limit",
        "1",
        "--early-stopping-patience",
        "3",
        "--max-train-episodes",
        "16",
        "--max-optimizer-steps",
        "17",
        "--audit-gradient-sha",
        "--strict-determinism",
        "--require-mixed-delayed-probe",
        "--recurrence-mode",
        "direct_latent",
        "--noop-policy",
        "update",
        "--curriculum",
        "full",
        "--checkpoint-unet",
        "--dreamlite-device",
        "cuda:0",
        "--reader-device",
        "cuda:1",
    )
    if resume is not None:
        command.extend(("--resume", str(resume)))
    validate_command(command)
    return command


def build_technical_plan(
    *,
    repo: Path,
    python: Path,
    model_root: Path,
    train: Path,
    train_sha256: str,
    dev: Path,
    dev_sha256: str,
    run_root: Path,
    preflight: Path,
    preflight_sha256: str,
    expected_commit: str,
    model_snapshots: Mapping[str, Mapping[str, Any]],
    through: str = "DL-S",
) -> dict[str, Any]:
    if through not in TECHNICAL_ORDER:
        raise ValueError(f"Unknown technical gate: {through}")
    repo = require_absolute(repo, "repo")
    python = require_absolute_executable(python, "python")
    model_root = require_absolute(model_root, "model_root")
    train = require_absolute(train, "train")
    dev = require_absolute(dev, "dev")
    run_root = require_absolute(run_root, "run_root")
    preflight = require_absolute(preflight, "preflight")
    for model_name in ("DreamLite-mobile", "Qwen3-VL-4B-Instruct"):
        if not (model_root / model_name).is_dir():
            raise ValueError(f"Locked model directory is missing: {model_root / model_name}")
    train_binding = require_file_sha(train, train_sha256, "train")
    dev_binding = require_file_sha(dev, dev_sha256, "dev")
    if SHA256_PATTERN.fullmatch(preflight_sha256) is None:
        raise ValueError("preflight_sha256 must be a lowercase SHA256 digest")

    results = run_root / "results"
    reference_dir = run_root / "dl_s" / "reference_16"
    resumed_dir = run_root / "dl_s" / "resumed_from_8"
    prefix_checkpoint = reference_dir / "checkpoint-000008.pt"
    reference_checkpoint = reference_dir / "checkpoint-000016.pt"
    resumed_checkpoint = resumed_dir / "checkpoint-000016.pt"

    r0_raw = results / "R3_R0_qwen_resize_contract.json"
    r0_validation = results / "R3_R0_validation.json"
    s0_raw = results / "R3_S0_qwen_scorer_contract.json"
    g4_raw = results / "G4_L.json"
    g4_validation = results / "G4_L_validation.json"
    g5_raw = results / "G5_L.json"
    g5_validation = results / "G5_L_validation.json"
    g6_raw = results / "G6_L_detached.json"
    g6_validation = results / "G6_L_validation.json"
    resume_report = results / "DL_S_resume_equivalence.json"
    final_report = results / "R3_technical_gates_final.json"

    stage_definitions: dict[str, dict[str, Any]] = {
        "R3-R0": {
            "commands": [
                _command(
                    python,
                    repo / "scripts" / "probes" / "qwen_resize_contract.py",
                    "--reader",
                    model_root / "Qwen3-VL-4B-Instruct",
                    "--device",
                    "cuda:0",
                    "--seed",
                    "0",
                    "--output-json",
                    r0_raw,
                ),
                _technical_validation_command(
                    python=python,
                    repo=repo,
                    results=results,
                    through="R3-R0",
                    output=r0_validation,
                ),
            ],
            "prerequisite_output_labels": [],
            "outputs": [_output("r0_raw", r0_raw), _output("r0_validation", r0_validation)],
        },
        "R3-S0": {
            "commands": [
                _command(
                    python,
                    repo / "scripts" / "probes" / "qwen_scorer_contract.py",
                    "--reader",
                    model_root / "Qwen3-VL-4B-Instruct",
                    "--device",
                    "cuda:0",
                    "--output-json",
                    s0_raw,
                )
            ],
            "prerequisite_output_labels": ["r0_raw", "r0_validation"],
            "outputs": [_output("s0_raw", s0_raw)],
        },
        "G4-L": {
            "commands": [
                _probe_command(
                    python=python,
                    repo=repo,
                    model_root=model_root,
                    events=(SET_EVENT,),
                    target_index=0,
                    output=g4_raw,
                ),
                _technical_validation_command(
                    python=python,
                    repo=repo,
                    results=results,
                    through="G4-L",
                    output=g4_validation,
                ),
            ],
            "prerequisite_output_labels": ["r0_raw", "r0_validation", "s0_raw"],
            "outputs": [
                _output("g4_raw", g4_raw, json_passed=False),
                _output("g4_validation", g4_validation),
            ],
        },
        "G5-L": {
            "commands": [
                _probe_command(
                    python=python,
                    repo=repo,
                    model_root=model_root,
                    events=(SET_EVENT, OVERWRITE_EVENT),
                    target_index=1,
                    output=g5_raw,
                ),
                _technical_validation_command(
                    python=python,
                    repo=repo,
                    results=results,
                    through="G5-L",
                    output=g5_validation,
                ),
            ],
            "prerequisite_output_labels": [
                "r0_raw",
                "r0_validation",
                "s0_raw",
                "g4_raw",
                "g4_validation",
            ],
            "outputs": [
                _output("g5_raw", g5_raw, json_passed=False),
                _output("g5_validation", g5_validation),
            ],
        },
        "G6-L": {
            "commands": [
                _probe_command(
                    python=python,
                    repo=repo,
                    model_root=model_root,
                    events=(SET_EVENT, OVERWRITE_EVENT),
                    target_index=1,
                    output=g6_raw,
                    detach=True,
                ),
                _technical_validation_command(
                    python=python,
                    repo=repo,
                    results=results,
                    through="G6-L",
                    output=g6_validation,
                ),
            ],
            "prerequisite_output_labels": [
                "r0_raw",
                "r0_validation",
                "s0_raw",
                "g4_raw",
                "g4_validation",
                "g5_raw",
                "g5_validation",
            ],
            "outputs": [
                _output("g6_raw", g6_raw, json_passed=False),
                _output("g6_validation", g6_validation),
            ],
        },
        "DL-S": {
            "commands": [
                _training_command(
                    python=python,
                    repo=repo,
                    model_root=model_root,
                    train=train,
                    dev=dev,
                    output_dir=reference_dir,
                ),
                _training_command(
                    python=python,
                    repo=repo,
                    model_root=model_root,
                    train=train,
                    dev=dev,
                    output_dir=resumed_dir,
                    resume=prefix_checkpoint,
                ),
                _command(
                    python,
                    repo / "scripts" / "probes" / "validate_r3_resume_equivalence.py",
                    "--prefix",
                    prefix_checkpoint,
                    "--reference",
                    reference_checkpoint,
                    "--resumed",
                    resumed_checkpoint,
                    "--reference-next",
                    reference_dir / "last.pt",
                    "--resumed-next",
                    resumed_dir / "last.pt",
                    "--reference-metrics",
                    reference_dir / "metrics.jsonl",
                    "--resumed-metrics",
                    resumed_dir / "metrics.jsonl",
                    "--output",
                    resume_report,
                ),
                _technical_validation_command(
                    python=python,
                    repo=repo,
                    results=results,
                    through="DL-S",
                    output=final_report,
                ),
            ],
            "prerequisite_output_labels": [
                "r0_raw",
                "r0_validation",
                "s0_raw",
                "g4_raw",
                "g4_validation",
                "g5_raw",
                "g5_validation",
                "g6_raw",
                "g6_validation",
            ],
            "static_input_labels": ["train", "dev"],
            "outputs": [
                _output("dl_reference_manifest", reference_dir / "manifest.json", json_passed=False),
                _output("dl_reference_summary", reference_dir / "summary.json", json_passed=False),
                _output(
                    "dl_reference_state_gradient_audit",
                    reference_dir / "state_gradient_audit.json",
                    json_passed=False,
                ),
                _output("dl_prefix_checkpoint", prefix_checkpoint, json_passed=False),
                _output("dl_reference_checkpoint", reference_checkpoint, json_passed=False),
                _output("dl_resumed_manifest", resumed_dir / "manifest.json", json_passed=False),
                _output("dl_resumed_summary", resumed_dir / "summary.json", json_passed=False),
                _output(
                    "dl_resumed_state_gradient_audit",
                    resumed_dir / "state_gradient_audit.json",
                    json_passed=False,
                ),
                _output("dl_resumed_lineage", resumed_dir / "resume_lineage.json", json_passed=False),
                _output("dl_resumed_checkpoint", resumed_checkpoint, json_passed=False),
                _output("dl_reference_next", reference_dir / "last.pt", json_passed=False),
                _output("dl_resumed_next", resumed_dir / "last.pt", json_passed=False),
                _output("dl_reference_metrics", reference_dir / "metrics.jsonl", json_passed=False),
                _output("dl_resumed_metrics", resumed_dir / "metrics.jsonl", json_passed=False),
                _output("dl_resume_report", resume_report),
                _output("technical_final", final_report),
            ],
        },
    }

    through_index = TECHNICAL_ORDER.index(through)
    strict_order = list(TECHNICAL_ORDER[: through_index + 1])
    stages: dict[str, dict[str, Any]] = {}
    for index, stage in enumerate(strict_order):
        definition = stage_definitions[stage]
        stages[stage] = {
            **definition,
            "index": index,
            "slug": TECHNICAL_SLUGS[stage],
            "launcher_stage": TECHNICAL_LAUNCHER_STAGES[stage],
            "dependency": None if index == 0 else strict_order[index - 1],
            "run_dir": str(run_root / "stages" / TECHNICAL_SLUGS[stage]),
            "evidence_path": str(run_root / "evidence" / f"{TECHNICAL_SLUGS[stage]}.json"),
        }

    return {
        "schema_version": 1,
        "protocol": PLAN_PROTOCOL,
        "kind": "technical",
        "execution_backend": "inspire-notebook-background",
        "submission_backend": "scripts/inspire/launch_background.py",
        "external_scheduler_submission": False,
        "expected_commit": expected_commit,
        "repo": str(repo),
        "python": str(python),
        "model_root": str(model_root),
        "model_snapshots": {name: dict(binding) for name, binding in model_snapshots.items()},
        "run_root": str(run_root),
        "formal_preflight": {"path": str(preflight), "sha256": preflight_sha256},
        "static_inputs": {"train": train_binding, "dev": dev_binding},
        "strict_order": strict_order,
        "failure_policy": "authorize exactly one next stage only after predecessor terminal and SHA-bound evidence pass",
        "stages": stages,
        "micro_extension": {
            "command_protocol": MICRO_COMMAND_PROTOCOL,
            "supported_suites": ["set8", "transition16"],
            "requires_complete_technical_final": True,
            "teacher_assisted_requires_t0": True,
        },
    }


def _plan_paths(run_root: Path) -> tuple[Path, Path]:
    plan_path = run_root / "dag_plan.json"
    return plan_path, plan_path.with_suffix(plan_path.suffix + ".sha256")


def _load_verified_plan(run_root: Path) -> tuple[dict[str, Any], Path, str]:
    run_root = run_root.resolve()
    plan_path, _ = _plan_paths(run_root)
    plan_sha256 = verify_sha_sidecar(plan_path)
    plan = load_json_object(plan_path)
    require_json_values(
        plan,
        {"schema_version": 1, "protocol": PLAN_PROTOCOL, "run_root": str(run_root)},
        "DAG plan",
    )
    return plan, plan_path, plan_sha256


def _all_planned_outputs(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    outputs: dict[str, Mapping[str, Any]] = {}
    for stage in plan["strict_order"]:
        for output in plan["stages"][stage]["outputs"]:
            label = str(output["label"])
            if label in outputs:
                raise ValueError(f"Duplicate planned output label: {label}")
            outputs[label] = output
    return outputs


def _verify_completed_stage(plan: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    """Revalidate one completed stage against the immutable plan and every bound artifact."""

    definition = plan["stages"][stage]
    run_dir = Path(definition["run_dir"])
    plan_path, _ = _plan_paths(Path(plan["run_root"]))
    plan_sha256 = verify_sha_sidecar(plan_path)
    stage_spec_path = Path(plan["run_root"]) / "authorizations" / f"{definition['slug']}.json"
    stage_spec_sha256 = verify_sha_sidecar(stage_spec_path)
    stage_spec = load_json_object(stage_spec_path)
    require_json_values(
        stage_spec,
        {
            "schema_version": 1,
            "protocol": STAGE_SPEC_PROTOCOL,
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha256,
            "stage": stage,
            "launcher_stage": definition["launcher_stage"],
            "run_dir": definition["run_dir"],
            "expected_commit": plan["expected_commit"],
        },
        f"{stage} stage specification",
    )
    if stage_spec.get("commands") != definition.get("commands"):
        raise ValueError(f"{stage} stage specification commands drifted from the immutable plan")
    if stage_spec.get("outputs") != definition.get("outputs"):
        raise ValueError(f"{stage} stage specification outputs drifted from the immutable plan")
    terminal_path = run_dir / "terminal.json"
    terminal = load_json_object(terminal_path)
    require_json_values(
        terminal,
        {
            "status": "succeeded",
            "passed": True,
            "exit_code": 0,
            "expected_commit": plan["expected_commit"],
        },
        f"{stage} terminal",
    )
    for log_name, digest_field in (("stdout.log", "stdout_sha256"), ("stderr.log", "stderr_sha256")):
        log_path = run_dir / log_name
        digest = terminal.get(digest_field)
        if SHA256_PATTERN.fullmatch(str(digest or "")) is None or sha256_file(log_path) != digest:
            raise ValueError(f"{stage} terminal does not bind {log_name}")
    worker_input = run_dir / "worker_input.json"
    configuration_sha256 = str(terminal.get("configuration_sha256", ""))
    if SHA256_PATTERN.fullmatch(configuration_sha256) is None or sha256_file(worker_input) != configuration_sha256:
        raise ValueError(f"{stage} terminal does not bind its worker_input.json")
    worker_configuration = load_json_object(worker_input)
    require_json_values(
        worker_configuration,
        {
            "stage": definition["launcher_stage"],
            "repo": plan["repo"],
            "run_root": plan["run_root"],
            "run_dir": definition["run_dir"],
            "expected_commit": plan["expected_commit"],
            "preflight": plan["formal_preflight"]["path"],
            "preflight_sha256": plan["formal_preflight"]["sha256"],
            "infrastructure_stage": False,
            "strict_environment": STRICT_ENVIRONMENT,
        },
        f"{stage} worker input",
    )
    expected_runner_command = [
        str(plan["python"]),
        str(Path(plan["repo"]) / "scripts" / "inspire" / "run_r3_stage.py"),
        "--spec",
        str(stage_spec_path),
        "--spec-sha256",
        stage_spec_sha256,
    ]
    if worker_configuration.get("command") != expected_runner_command:
        raise ValueError(f"{stage} worker input does not bind the immutable stage specification")

    evidence_path = Path(definition["evidence_path"])
    evidence_sha256 = verify_sha_sidecar(evidence_path)
    evidence = load_json_object(evidence_path)
    require_json_values(
        evidence,
        {
            "schema_version": 1,
            "protocol": STAGE_EVIDENCE_PROTOCOL,
            "passed": True,
            "stage": stage,
            "stage_slug": definition["slug"],
            "launcher_stage": definition["launcher_stage"],
            "expected_commit": plan["expected_commit"],
            "configuration_sha256": configuration_sha256,
            "stage_spec_sha256": stage_spec_sha256,
            "plan_sha256": plan_sha256,
            "worker_input_path": str(worker_input),
            "formal_preflight_sha256": plan["formal_preflight"]["sha256"],
        },
        f"{stage} evidence",
    )
    if evidence.get("prerequisites") != stage_spec.get("prerequisites"):
        raise ValueError(f"{stage} evidence prerequisites drifted from the stage specification")
    if evidence.get("model_snapshots") != plan.get("model_snapshots"):
        raise ValueError(f"{stage} evidence model snapshot bindings drifted from the immutable plan")
    for model_binding in plan.get("model_snapshots", {}).values():
        if not isinstance(model_binding, Mapping):
            raise ValueError(f"{stage} plan contains a malformed model snapshot binding")
        verify_snapshot_binding(model_binding)
    for prerequisite in evidence.get("prerequisites", []):
        if not isinstance(prerequisite, Mapping):
            raise ValueError(f"{stage} evidence contains a malformed prerequisite")
        verify_bound_artifact(prerequisite)
    command_results = evidence.get("commands")
    expected_command_results = [
        {"index": index, "exit_code": 0} for index, _ in enumerate(definition.get("commands", []))
    ]
    if command_results != expected_command_results:
        raise ValueError(f"{stage} evidence does not prove every planned command exited successfully")
    actual_outputs = evidence.get("outputs")
    if not isinstance(actual_outputs, list):
        raise ValueError(f"{stage} evidence outputs must be a list")
    planned_outputs = {str(output["label"]): output for output in definition.get("outputs", [])}
    actual_by_label = {str(output.get("label")): output for output in actual_outputs if isinstance(output, Mapping)}
    if len(actual_by_label) != len(actual_outputs) or set(actual_by_label) != set(planned_outputs):
        raise ValueError(f"{stage} evidence does not bind every planned output exactly once")
    for label, output in actual_by_label.items():
        planned = planned_outputs[label]
        if output.get("path") != planned.get("path") or output.get("required_values") != planned.get("required_values"):
            raise ValueError(f"{stage} evidence output {label!r} drifted from the immutable plan")
        verify_bound_artifact(output)
    return [
        {
            "label": f"{stage}:terminal",
            "path": str(terminal_path),
            "sha256": sha256_file(terminal_path),
            "required_values": {"status": "succeeded", "passed": True},
        },
        {
            "label": f"{stage}:evidence",
            "path": str(evidence_path),
            "sha256": evidence_sha256,
            "required_values": {
                "passed": True,
                "stage": stage,
                "expected_commit": plan["expected_commit"],
            },
        },
    ]


def _verify_predecessor(plan: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    definition = plan["stages"][stage]
    dependency = definition["dependency"]
    if dependency is None:
        return []
    return _verify_completed_stage(plan, str(dependency))


def _next_authorizable_stage(plan: Mapping[str, Any], run_root: Path) -> str:
    for stage in plan["strict_order"]:
        slug = plan["stages"][stage]["slug"]
        spec_path = run_root / "authorizations" / f"{slug}.json"
        if not spec_path.exists():
            return str(stage)
    raise ValueError("Every planned stage is already materialized")


def authorize_stage(run_root: Path, *, stage: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    run_root = run_root.resolve()
    plan, plan_path, plan_sha256 = _load_verified_plan(run_root)
    repo = Path(plan["repo"])
    expected_commit = str(plan["expected_commit"])
    verify_clean_commit(repo, expected_commit)
    preflight = Path(plan["formal_preflight"]["path"])
    _, preflight_sha256 = verify_preflight(
        preflight,
        expected_commit=expected_commit,
        infrastructure_stage=False,
    )
    if preflight_sha256 != plan["formal_preflight"]["sha256"]:
        raise ValueError("Formal preflight SHA256 drifted after DAG initialization")

    next_stage = _next_authorizable_stage(plan, run_root)
    if stage is None:
        stage = next_stage
    if stage != next_stage:
        raise ValueError(f"Only the next fail-stop stage {next_stage} may be materialized, not {stage}")
    definition = plan["stages"][stage]
    run_dir = Path(definition["run_dir"])
    if run_dir.exists():
        raise ValueError(f"Unique stage run directory already exists: {run_dir}")

    prerequisites = _verify_predecessor(plan, stage)
    planned_outputs = _all_planned_outputs(plan)
    for label in definition.get("prerequisite_output_labels", []):
        output = planned_outputs[label]
        path = Path(output["path"])
        binding = {
            "label": f"upstream:{label}",
            "path": str(path),
            "sha256": sha256_file(path),
            "required_values": output.get("required_values", {}),
        }
        verify_bound_artifact(binding)
        prerequisites.append(binding)
    for label in definition.get("static_input_labels", []):
        source = plan["static_inputs"][label]
        binding = {
            "label": f"static:{label}",
            "path": source["path"],
            "sha256": source["sha256"],
            "required_values": {},
        }
        verify_bound_artifact(binding)
        prerequisites.append(binding)
    for external in definition.get("external_prerequisites", []):
        binding = dict(external)
        verify_bound_artifact(binding)
        prerequisites.append(binding)

    spec_path = run_root / "authorizations" / f"{definition['slug']}.json"
    evidence_path = Path(definition["evidence_path"])
    spec = {
        "schema_version": 1,
        "protocol": STAGE_SPEC_PROTOCOL,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "stage": stage,
        "stage_index": definition["index"],
        "stage_slug": definition["slug"],
        "launcher_stage": definition["launcher_stage"],
        "run_root": str(run_root),
        "run_dir": str(run_dir),
        "repo": str(repo),
        "expected_commit": expected_commit,
        "formal_preflight": {"path": str(preflight), "sha256": preflight_sha256},
        "prerequisites": prerequisites,
        "commands": definition["commands"],
        "outputs": definition["outputs"],
        "evidence_path": str(evidence_path),
        "immutable_after_materialization": True,
    }
    if dry_run:
        return {"dry_run": True, "stage_spec": spec, "launch": None}

    if spec_path.exists() or spec_path.with_suffix(spec_path.suffix + ".sha256").exists():
        raise ValueError(f"Stage authorization already exists: {spec_path}")
    spec_sha256 = atomic_json(spec_path, spec)
    python = Path(plan["python"])
    runner = repo / "scripts" / "inspire" / "run_r3_stage.py"
    runner_command = _command(
        python,
        runner,
        "--spec",
        spec_path,
        "--spec-sha256",
        spec_sha256,
    )
    launch_argv = _command(
        python,
        repo / "scripts" / "inspire" / "launch_background.py",
        "--repo",
        repo,
        "--run-root",
        run_root,
        "--run-dir",
        run_dir,
        "--stage",
        definition["launcher_stage"],
        "--expected-commit",
        expected_commit,
        "--preflight",
        preflight,
        "--",
        *runner_command,
    )
    launch = {
        "schema_version": 1,
        "protocol": LAUNCH_COMMAND_PROTOCOL,
        "stage": stage,
        "stage_spec": str(spec_path),
        "stage_spec_sha256": spec_sha256,
        "run_dir": str(run_dir),
        "argv": launch_argv,
        "shell_preview": shlex.join(launch_argv),
        "executed": False,
    }
    launch_path = run_root / "launch_commands" / f"{definition['slug']}.json"
    launch_sha256 = atomic_json(launch_path, launch)
    return {
        "dry_run": False,
        "stage_spec": spec,
        "stage_spec_sha256": spec_sha256,
        "launch_path": str(launch_path),
        "launch_sha256": launch_sha256,
        "launch": launch,
    }


def initialize_technical_dag(
    *,
    repo: Path,
    python: Path,
    model_root: Path,
    train: Path,
    train_sha256: str,
    dev: Path,
    dev_sha256: str,
    run_root: Path,
    preflight: Path,
    expected_commit: str,
    through: str = "DL-S",
    dry_run: bool = False,
) -> dict[str, Any]:
    verify_clean_commit(repo.resolve(), expected_commit)
    preflight_report, preflight_sha256 = verify_preflight(
        preflight.resolve(),
        expected_commit=expected_commit,
        infrastructure_stage=False,
    )
    reported_model_root = preflight_report.get("paths", {}).get("VLM_MODEL_ROOT", {}).get("value")
    if not isinstance(reported_model_root, str) or Path(reported_model_root).resolve() != model_root.resolve():
        raise ValueError("Formal preflight VLM_MODEL_ROOT does not match the technical DAG model root")
    reported_runs_root = preflight_report.get("paths", {}).get("VLM_RUN_ROOT", {}).get("value")
    if not isinstance(reported_runs_root, str) or not is_within(run_root.resolve(), Path(reported_runs_root)):
        raise ValueError("Technical DAG run root is not inside the formal preflight VLM_RUN_ROOT")
    reported_python = preflight_report.get("python", {}).get("executable")
    if not isinstance(reported_python, str) or Path(os.path.abspath(reported_python)) != python:
        raise ValueError("Formal preflight Python executable does not match the technical DAG Python")
    model_snapshots = _verified_model_snapshots(
        repo=repo.resolve(),
        model_root=model_root.resolve(),
        preflight=preflight_report,
    )
    plan = build_technical_plan(
        repo=repo,
        python=python,
        model_root=model_root,
        train=train,
        train_sha256=train_sha256,
        dev=dev,
        dev_sha256=dev_sha256,
        run_root=run_root,
        preflight=preflight,
        preflight_sha256=preflight_sha256,
        expected_commit=expected_commit,
        model_snapshots=model_snapshots,
        through=through,
    )
    if dry_run:
        return {"dry_run": True, "plan": plan, "first_stage": plan["strict_order"][0]}
    if run_root.exists():
        raise ValueError(f"Unique DAG run root already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "stages").mkdir(parents=False, exist_ok=False)
    plan_path, _ = _plan_paths(run_root)
    plan_sha256 = atomic_json(plan_path, plan)
    first = authorize_stage(run_root, stage=plan["strict_order"][0])
    return {
        "dry_run": False,
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "first_stage": first,
    }


def _mapping_field(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"R3 preregistration field {field} must be an object")
    return value


def _sha_field(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"R3 preregistration field {field} must be a lowercase SHA256 digest")
    return value


def _teacher_preregistered_contract(preregistration: Path) -> dict[str, Any]:
    value = load_json_object(preregistration)
    if value.get("schema") != "vision_memory.r3-preregistration.v1":
        raise ValueError("Teacher preparation requires the locked R3 preregistration schema")
    set8 = load_teacher_calibration_input_lock(preregistration, suite="set8")
    transition16 = load_teacher_calibration_input_lock(preregistration, suite="transition16")
    micro = _mapping_field(value.get("micro_data"), field="micro_data")
    micro_execution = _mapping_field(value.get("micro_execution"), field="micro_execution")
    if micro_execution.get("teacher_preparation_order") != list(TEACHER_PREPARATION_ORDER):
        raise ValueError("R3 preregistration micro_execution.teacher_preparation_order differs from the canonical DAG")
    transition_data = _mapping_field(micro.get("transition16"), field="micro_data.transition16")
    teacher = _mapping_field(value.get("teacher_contract"), field="teacher_contract")
    builds = _mapping_field(teacher.get("cache_builds"), field="teacher_contract.cache_builds")
    set8_build = _mapping_field(builds.get("set8"), field="teacher_contract.cache_builds.set8")
    transition16_build = _mapping_field(builds.get("transition16"), field="teacher_contract.cache_builds.transition16")
    models = _mapping_field(value.get("models"), field="models")
    reader = _mapping_field(models.get("reader"), field="models.reader")
    updater = _mapping_field(models.get("updater"), field="models.updater")
    reader_revision = reader.get("revision")
    dreamlite_revision = updater.get("revision")
    if (
        COMMIT_PATTERN.fullmatch(str(reader_revision or "")) is None
        or COMMIT_PATTERN.fullmatch(str(dreamlite_revision or "")) is None
    ):
        raise ValueError("Teacher preparation model revisions must be full immutable commits")
    return {
        "preregistration_sha256": sha256_file(preregistration),
        "reader_revision": reader_revision,
        "dreamlite_revision": dreamlite_revision,
        "font_sha256": _sha_field(teacher.get("font_sha256"), field="teacher_contract.font_sha256"),
        "transition16_gate_sha256": _sha_field(
            transition_data.get("gate_sha256"), field="micro_data.transition16.gate_sha256"
        ),
        "transition16_raw_sidecar_sha256": _sha_field(
            transition_data.get("raw_teacher_sidecar_sha256"),
            field="micro_data.transition16.raw_teacher_sidecar_sha256",
        ),
        "set8": {
            **set8.to_dict(),
            "build_report_sha256": _sha_field(
                set8_build.get("build_report_sha256"),
                field="teacher_contract.cache_builds.set8.build_report_sha256",
            ),
        },
        "transition16": {
            **transition16.to_dict(),
            "build_report_sha256": _sha_field(
                transition16_build.get("build_report_sha256"),
                field="teacher_contract.cache_builds.transition16.build_report_sha256",
            ),
        },
    }


def _required_output(label: str, path: Path, required_values: Mapping[str, Any]) -> dict[str, Any]:
    return {"label": label, "path": str(path), "required_values": dict(required_values)}


def build_teacher_preparation_plan(
    *,
    technical: Mapping[str, Any],
    technical_run_root: Path,
    technical_chain_bindings: list[dict[str, Any]],
    teacher_run_root: Path,
    preregistration: Path,
    set8_train: Path,
    transition16_train: Path,
    transition16_gate: Path,
    transition16_raw_sidecar: Path,
    set8_cache: Path,
    transition16_cache: Path,
    font: Path,
) -> dict[str, Any]:
    repo = require_absolute(Path(str(technical["repo"])), "repo")
    python = require_absolute_executable(Path(str(technical["python"])), "python")
    model_root = require_absolute(Path(str(technical["model_root"])), "model_root")
    teacher_run_root = require_absolute(teacher_run_root, "teacher_run_root")
    preregistration = require_absolute(preregistration, "preregistration")
    set8_train = require_absolute(set8_train, "set8_train")
    transition16_train = require_absolute(transition16_train, "transition16_train")
    transition16_gate = require_absolute(transition16_gate, "transition16_gate")
    transition16_raw_sidecar = require_absolute(transition16_raw_sidecar, "transition16_raw_sidecar")
    set8_cache = require_absolute(set8_cache, "set8_cache")
    transition16_cache = require_absolute(transition16_cache, "transition16_cache")
    font = require_absolute(font, "font")
    if not set8_cache.is_dir() or not transition16_cache.is_dir() or set8_cache == transition16_cache:
        raise ValueError("Teacher preparation requires two distinct existing cache directories")
    for cache in (set8_cache, transition16_cache):
        if is_within(teacher_run_root, cache) or is_within(cache, teacher_run_root):
            raise ValueError("Teacher preparation run root and immutable cache roots must be disjoint")

    lock = _teacher_preregistered_contract(preregistration)
    reader = model_root / "Qwen3-VL-4B-Instruct"
    dreamlite = model_root / "DreamLite-mobile"
    for path, revision, label in (
        (reader, lock["reader_revision"], "Reader"),
        (dreamlite, lock["dreamlite_revision"], "DreamLite"),
    ):
        marker = path / ".locked_revision"
        if not path.is_dir() or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != revision:
            raise ValueError(f"{label} snapshot does not match the preregistered revision")

    static_inputs = {
        "preregistration": require_file_sha(preregistration, lock["preregistration_sha256"], "R3 preregistration"),
        "set8_train": require_file_sha(set8_train, lock["set8"]["train_sha256"], "Set8 train"),
        "transition16_train": require_file_sha(
            transition16_train,
            lock["transition16"]["train_sha256"],
            "Transition16 train",
        ),
        "transition16_gate": require_file_sha(transition16_gate, lock["transition16_gate_sha256"], "Transition16 gate"),
        "transition16_raw_sidecar": require_file_sha(
            transition16_raw_sidecar,
            lock["transition16_raw_sidecar_sha256"],
            "Transition16 raw teacher sidecar",
        ),
        "font": require_file_sha(font, lock["font_sha256"], "embedded teacher font"),
        "set8_manifest": require_file_sha(
            set8_cache / "manifest.json", lock["set8"]["manifest_sha256"], "Set8 manifest"
        ),
        "set8_sidecar": require_file_sha(
            set8_cache / "transitions.jsonl", lock["set8"]["sidecar_sha256"], "Set8 sidecar"
        ),
        "set8_build_report": require_file_sha(
            set8_cache / "build_report.json",
            lock["set8"]["build_report_sha256"],
            "Set8 cache build report",
        ),
        "transition16_manifest": require_file_sha(
            transition16_cache / "manifest.json",
            lock["transition16"]["manifest_sha256"],
            "Transition16 manifest",
        ),
        "transition16_sidecar": require_file_sha(
            transition16_cache / "transitions.jsonl",
            lock["transition16"]["sidecar_sha256"],
            "Transition16 sidecar",
        ),
        "transition16_build_report": require_file_sha(
            transition16_cache / "build_report.json",
            lock["transition16"]["build_report_sha256"],
            "Transition16 cache build report",
        ),
        "reader_revision_marker": require_file_sha(
            reader / ".locked_revision", sha256_file(reader / ".locked_revision"), "Reader revision marker"
        ),
        "dreamlite_revision_marker": require_file_sha(
            dreamlite / ".locked_revision",
            sha256_file(dreamlite / ".locked_revision"),
            "DreamLite revision marker",
        ),
    }
    for binding in technical_chain_bindings:
        verify_bound_artifact(binding)

    results = teacher_run_root / "results"
    tc0_raw = results / "TC0_raw.json"
    tc0_validation = results / "TC0_validation.json"
    tf0_raw = results / "TF0_raw.json"
    tf0_validation = results / "TF0_validation.json"
    t0_report = results / "T0.json"
    set8_calibration = results / "set8" / "calibration.json"
    set8_calibration_report = results / "set8" / "calibration_report.json"
    transition16_calibration = results / "transition16" / "calibration.json"
    transition16_calibration_report = results / "transition16" / "calibration_report.json"
    final_report = results / "teacher_preparation_final.json"
    helper = repo / "scripts" / "inspire" / "run_r3_teacher_prep_step.py"
    expected_commit = str(technical["expected_commit"])

    stage_definitions: dict[str, dict[str, Any]] = {
        "R3-TC0": {
            "commands": [
                _command(
                    python,
                    repo / "scripts" / "probes" / "r3_teacher_cache_compatibility.py",
                    "--set8-cache",
                    set8_cache,
                    "--transition16-cache",
                    transition16_cache,
                    "--reader",
                    reader,
                    "--preregistration",
                    preregistration,
                    "--device",
                    "cuda:0",
                    "--output-json",
                    tc0_raw,
                ),
                _command(
                    python,
                    helper,
                    "validate-tc0",
                    "--report",
                    tc0_raw,
                    "--preregistration",
                    preregistration,
                    "--expected-commit",
                    expected_commit,
                    "--output",
                    tc0_validation,
                ),
            ],
            "prerequisite_output_labels": [],
            "static_input_labels": [
                "preregistration",
                "set8_manifest",
                "set8_sidecar",
                "set8_build_report",
                "transition16_manifest",
                "transition16_sidecar",
                "transition16_build_report",
                "reader_revision_marker",
            ],
            "external_prerequisites": technical_chain_bindings,
            "outputs": [
                _required_output(
                    "teacher_tc0_raw",
                    tc0_raw,
                    {"passed": True, "probe": "r3_tc0_teacher_cache_forward_compatibility"},
                ),
                _required_output(
                    "teacher_tc0_validation",
                    tc0_validation,
                    {
                        "passed": True,
                        "protocol": "R3-TC0-cache-forward-compatibility-validation.v1",
                        "expected_commit": expected_commit,
                        "teacher_calibration_unlocked": False,
                    },
                ),
            ],
        },
        "R3-TF0": {
            "commands": [
                _command(
                    python,
                    helper,
                    "run-tf0",
                    "--set8-cache",
                    set8_cache,
                    "--transition16-cache",
                    transition16_cache,
                    "--reader",
                    reader,
                    "--tc0-validation",
                    tc0_validation,
                    "--preregistration",
                    preregistration,
                    "--device",
                    "cuda:0",
                    "--output",
                    tf0_raw,
                ),
                _command(
                    python,
                    helper,
                    "validate-tf0",
                    "--report",
                    tf0_raw,
                    "--tc0-validation",
                    tc0_validation,
                    "--preregistration",
                    preregistration,
                    "--expected-commit",
                    expected_commit,
                    "--output",
                    tf0_validation,
                ),
            ],
            "prerequisite_output_labels": ["teacher_tc0_raw", "teacher_tc0_validation"],
            "static_input_labels": [
                "preregistration",
                "set8_manifest",
                "set8_sidecar",
                "set8_build_report",
                "transition16_manifest",
                "transition16_sidecar",
                "transition16_build_report",
                "reader_revision_marker",
            ],
            "outputs": [
                _required_output(
                    "teacher_tf0_raw",
                    tf0_raw,
                    {"passed": True, "probe": "r3_tf0_teacher_feature_backend_compatibility"},
                ),
                _required_output(
                    "teacher_tf0_validation",
                    tf0_validation,
                    {
                        "passed": True,
                        "protocol": "R3-TF0-feature-backend-compatibility-validation.v1",
                        "expected_commit": expected_commit,
                        "teacher_assisted_training_unlocked": True,
                    },
                ),
            ],
        },
        "T0": {
            "commands": [
                _command(
                    python,
                    repo / "scripts" / "probes" / "teacher_t0_upper_bound.py",
                    "--gate-jsonl",
                    transition16_gate,
                    "--raw-sidecar",
                    transition16_raw_sidecar,
                    "--teacher-manifest",
                    transition16_cache / "manifest.json",
                    "--teacher-cache-root",
                    transition16_cache,
                    "--reader",
                    reader,
                    "--font",
                    font,
                    "--reader-device",
                    "cuda:0",
                    "--output-json",
                    t0_report,
                )
            ],
            "prerequisite_output_labels": ["teacher_tf0_raw", "teacher_tf0_validation"],
            "static_input_labels": [
                "preregistration",
                "transition16_gate",
                "transition16_raw_sidecar",
                "transition16_manifest",
                "font",
                "reader_revision_marker",
            ],
            "outputs": [
                _required_output(
                    "teacher_t0",
                    t0_report,
                    {
                        "passed": True,
                        "probe": "teacher_t0_real_qwen_integrity_upper_bound",
                        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
                    },
                )
            ],
        },
        "CAL-Set8": {
            "commands": [
                _command(
                    python,
                    repo / "scripts" / "probes" / "calibrate_r3_teacher_loss.py",
                    "--train",
                    set8_train,
                    "--cache-dir",
                    set8_cache,
                    "--suite",
                    "set8",
                    "--preregistration",
                    preregistration,
                    "--dreamlite",
                    dreamlite,
                    "--reader",
                    reader,
                    "--output",
                    set8_calibration,
                    "--report",
                    set8_calibration_report,
                    "--resolution",
                    "1024",
                    "--seed",
                    "0",
                    "--adapter-seed",
                    "0",
                    "--lora-rank",
                    "4",
                    "--dreamlite-device",
                    "cuda:0",
                    "--reader-device",
                    "cuda:1",
                )
            ],
            "prerequisite_output_labels": ["teacher_tf0_validation", "teacher_t0"],
            "static_input_labels": [
                "preregistration",
                "set8_train",
                "set8_manifest",
                "set8_sidecar",
                "reader_revision_marker",
                "dreamlite_revision_marker",
            ],
            "outputs": [
                _required_output(
                    "teacher_set8_calibration",
                    set8_calibration,
                    {"schema": "vision_memory.teacher-calibration-file.v1", "split": "train"},
                ),
                _required_output(
                    "teacher_set8_calibration_report",
                    set8_calibration_report,
                    {
                        "schema": "vision_memory.r3-teacher-calibration-report.v1",
                        "suite": "set8",
                        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
                        "preregistration_sha256": lock["preregistration_sha256"],
                        "train_sha256": lock["set8"]["train_sha256"],
                        "manifest_sha256": lock["set8"]["manifest_sha256"],
                        "sidecar_sha256": lock["set8"]["sidecar_sha256"],
                    },
                ),
            ],
        },
        "CAL-Transition16": {
            "commands": [
                _command(
                    python,
                    repo / "scripts" / "probes" / "calibrate_r3_teacher_loss.py",
                    "--train",
                    transition16_train,
                    "--cache-dir",
                    transition16_cache,
                    "--suite",
                    "transition16",
                    "--preregistration",
                    preregistration,
                    "--dreamlite",
                    dreamlite,
                    "--reader",
                    reader,
                    "--output",
                    transition16_calibration,
                    "--report",
                    transition16_calibration_report,
                    "--resolution",
                    "1024",
                    "--seed",
                    "0",
                    "--adapter-seed",
                    "0",
                    "--lora-rank",
                    "4",
                    "--dreamlite-device",
                    "cuda:0",
                    "--reader-device",
                    "cuda:1",
                ),
                _command(
                    python,
                    helper,
                    "finalize",
                    "--expected-commit",
                    expected_commit,
                    "--tc0-raw",
                    tc0_raw,
                    "--tc0-validation",
                    tc0_validation,
                    "--tf0-raw",
                    tf0_raw,
                    "--tf0-validation",
                    tf0_validation,
                    "--t0",
                    t0_report,
                    "--set8-calibration",
                    set8_calibration,
                    "--set8-calibration-report",
                    set8_calibration_report,
                    "--transition16-calibration",
                    transition16_calibration,
                    "--transition16-calibration-report",
                    transition16_calibration_report,
                    "--output",
                    final_report,
                ),
            ],
            "prerequisite_output_labels": [
                "teacher_tf0_validation",
                "teacher_t0",
                "teacher_set8_calibration",
                "teacher_set8_calibration_report",
            ],
            "static_input_labels": [
                "preregistration",
                "transition16_train",
                "transition16_manifest",
                "transition16_sidecar",
                "reader_revision_marker",
                "dreamlite_revision_marker",
            ],
            "outputs": [
                _required_output(
                    "teacher_transition16_calibration",
                    transition16_calibration,
                    {"schema": "vision_memory.teacher-calibration-file.v1", "split": "train"},
                ),
                _required_output(
                    "teacher_transition16_calibration_report",
                    transition16_calibration_report,
                    {
                        "schema": "vision_memory.r3-teacher-calibration-report.v1",
                        "suite": "transition16",
                        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
                        "preregistration_sha256": lock["preregistration_sha256"],
                        "train_sha256": lock["transition16"]["train_sha256"],
                        "manifest_sha256": lock["transition16"]["manifest_sha256"],
                        "sidecar_sha256": lock["transition16"]["sidecar_sha256"],
                    },
                ),
                _required_output(
                    "teacher_preparation_final",
                    final_report,
                    {
                        "schema_version": 1,
                        "protocol": "r3-inspire-teacher-preparation-final.v1",
                        "passed": True,
                        "expected_commit": expected_commit,
                        "strict_order": list(TEACHER_PREPARATION_ORDER),
                    },
                ),
            ],
        },
    }
    stages: dict[str, dict[str, Any]] = {}
    for index, stage in enumerate(TEACHER_PREPARATION_ORDER):
        definition = stage_definitions[stage]
        slug = TEACHER_PREPARATION_SLUGS[stage]
        stages[stage] = {
            **definition,
            "index": index,
            "slug": slug,
            "launcher_stage": TEACHER_PREPARATION_LAUNCHER_STAGES[stage],
            "dependency": None if index == 0 else TEACHER_PREPARATION_ORDER[index - 1],
            "run_dir": str(teacher_run_root / "stages" / slug),
            "evidence_path": str(teacher_run_root / "evidence" / f"{slug}.json"),
        }

    technical_plan_path, _ = _plan_paths(technical_run_root)
    technical_plan_sha256 = verify_sha_sidecar(technical_plan_path)
    return {
        "schema_version": 1,
        "protocol": PLAN_PROTOCOL,
        "kind": "teacher-preparation",
        "execution_backend": "inspire-notebook-background",
        "submission_backend": "scripts/inspire/launch_background.py",
        "external_scheduler_submission": False,
        "expected_commit": expected_commit,
        "repo": str(repo),
        "python": str(python),
        "model_root": str(model_root),
        "model_snapshots": technical["model_snapshots"],
        "run_root": str(teacher_run_root),
        "formal_preflight": technical["formal_preflight"],
        "technical_parent": {
            "run_root": str(technical_run_root),
            "plan_sha256": technical_plan_sha256,
            "completed_stage_count": len(TECHNICAL_ORDER),
        },
        "teacher_contract": {
            "preregistration": static_inputs["preregistration"],
            "cache_roots": {"set8": str(set8_cache), "transition16": str(transition16_cache)},
            "reader_revision": lock["reader_revision"],
            "dreamlite_revision": lock["dreamlite_revision"],
            "calibration_input_locks": {"set8": lock["set8"], "transition16": lock["transition16"]},
        },
        "static_inputs": static_inputs,
        "strict_order": list(TEACHER_PREPARATION_ORDER),
        "failure_policy": "strict serial stop; never authorize a teacher stage after a failed predecessor",
        "stages": stages,
    }


def initialize_teacher_preparation_dag(
    *,
    technical_run_root: Path,
    teacher_run_root: Path,
    preregistration: Path,
    set8_train: Path,
    transition16_train: Path,
    transition16_gate: Path,
    transition16_raw_sidecar: Path,
    set8_cache: Path,
    transition16_cache: Path,
    font: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    technical_run_root = technical_run_root.resolve()
    technical, _, _ = _load_verified_plan(technical_run_root)
    if technical.get("kind") != "technical" or technical.get("strict_order") != list(TECHNICAL_ORDER):
        raise ValueError("Teacher preparation requires the complete six-stage technical parent")
    verify_clean_commit(Path(str(technical["repo"])), str(technical["expected_commit"]))
    technical_chain_bindings: list[dict[str, Any]] = []
    for stage in TECHNICAL_ORDER:
        technical_chain_bindings.extend(_verify_completed_stage(technical, stage))
    preflight_report, preflight_sha256 = verify_preflight(
        Path(str(technical["formal_preflight"]["path"])),
        expected_commit=str(technical["expected_commit"]),
        infrastructure_stage=False,
    )
    if preflight_sha256 != technical["formal_preflight"]["sha256"]:
        raise ValueError("Teacher preparation technical-parent preflight SHA256 drifted")
    runs_root = preflight_report.get("paths", {}).get("VLM_RUN_ROOT", {}).get("value")
    if not isinstance(runs_root, str) or not is_within(teacher_run_root.resolve(), Path(runs_root)):
        raise ValueError("Teacher preparation run root is outside the formal VLM_RUN_ROOT")
    plan = build_teacher_preparation_plan(
        technical=technical,
        technical_run_root=technical_run_root,
        technical_chain_bindings=technical_chain_bindings,
        teacher_run_root=teacher_run_root,
        preregistration=preregistration,
        set8_train=set8_train,
        transition16_train=transition16_train,
        transition16_gate=transition16_gate,
        transition16_raw_sidecar=transition16_raw_sidecar,
        set8_cache=set8_cache,
        transition16_cache=transition16_cache,
        font=font,
    )
    if dry_run:
        return {"dry_run": True, "plan": plan, "first_stage": TEACHER_PREPARATION_ORDER[0]}
    if teacher_run_root.exists():
        raise ValueError(f"Unique teacher-preparation run root already exists: {teacher_run_root}")
    teacher_run_root.mkdir(parents=True, exist_ok=False)
    (teacher_run_root / "stages").mkdir(parents=False, exist_ok=False)
    plan_path, _ = _plan_paths(teacher_run_root)
    plan_sha256 = atomic_json(plan_path, plan)
    first = authorize_stage(teacher_run_root, stage=TEACHER_PREPARATION_ORDER[0])
    return {
        "dry_run": False,
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "first_stage": first,
    }


_MICRO_ALLOWED_SCRIPTS = {
    "scripts/train/dreamlite_episode.py",
    "scripts/eval/dreamlite_mcq.py",
    "scripts/eval/score_r3_micro.py",
    "scripts/eval/teacher_state_retrieval.py",
    "scripts/probes/validate_r3_micro_replication.py",
    "scripts/eval/score_r3_teacher_attribution.py",
}
_MICRO_TEACHER_INPUT_FLAGS = {
    "--teacher-manifest",
    "--teacher-sidecar",
    "--teacher-calibration",
    "--teacher-cache",
}


def _command_flag_value(command: list[str], flag: str, *, required: bool = False) -> str | None:
    positions = [index for index, value in enumerate(command) if value == flag]
    if not positions:
        if required:
            raise ValueError(f"micro command is missing required flag {flag}")
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(command) or command[positions[0] + 1].startswith("--"):
        raise ValueError(f"micro command must provide {flag} exactly once with one value")
    return command[positions[0] + 1]


def _micro_script(command: list[str]) -> str:
    if len(command) < 2:
        raise ValueError("micro commands must invoke one audited repository Python script")
    script = Path(command[1])
    if not script.is_absolute() or not is_within(script, ROOT):
        raise ValueError("micro command script must be an absolute path inside the bound repository")
    relative = script.resolve().relative_to(ROOT.resolve()).as_posix()
    if relative not in _MICRO_ALLOWED_SCRIPTS:
        raise ValueError(f"micro command invokes an unaudited script: {relative}")
    return relative


def _preregistered_micro_data_binding(suite: str) -> dict[str, str]:
    preregistration = ROOT / "configs" / "experiments" / "r3_preregistration.json"
    value = load_json_object(preregistration)
    micro_data = value.get("micro_data")
    locked = micro_data.get(suite) if isinstance(micro_data, Mapping) else None
    if not isinstance(locked, Mapping):
        raise ValueError(f"R3 preregistration has no micro-data lock for {suite}")
    result = {
        "preregistration_sha256": sha256_file(preregistration),
        "train_sha256": str(locked.get("train_sha256", "")),
        "gate_sha256": str(locked.get("gate_sha256", "")),
    }
    if any(SHA256_PATTERN.fullmatch(value) is None for value in result.values()):
        raise ValueError(f"R3 preregistration micro-data lock for {suite} is malformed")
    return result


def _expected_micro_arm_contract(suite: str, regime: str) -> tuple[str, str, list[tuple[str, str]]]:
    if suite == "set8" and regime == "qa_only":
        return "single", "none", [("A", "none")]
    if suite == "transition16" and regime == "qa_only":
        return "paired-replica", "none", [("A", "none"), ("B", "none")]
    if suite == "set8" and regime == "teacher_assisted":
        return (
            "teacher-control-composite",
            "composite",
            [("correct", "correct"), ("shuffled", "shuffled"), ("random", "random-moment-matched")],
        )
    return "paired-replica", "correct", [("A", "correct"), ("B", "correct")]


def _validated_micro_arms(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    suite = str(contract["suite"])
    regime = str(contract["training_regime"])
    expected_shape, expected_top_control, expected_arms = _expected_micro_arm_contract(suite, regime)
    if contract.get("execution_shape") != expected_shape or contract.get("teacher_control") != expected_top_control:
        raise ValueError("micro execution shape/teacher control differs from the locked suite-regime DAG")
    arms = contract.get("arms")
    if not isinstance(arms, list) or len(arms) != len(expected_arms):
        raise ValueError("micro command arms do not match the locked execution shape")
    validated: list[dict[str, Any]] = []
    output_dirs: set[Path] = set()
    for value, (expected_id, expected_control) in zip(arms, expected_arms, strict=True):
        if not isinstance(value, Mapping):
            raise ValueError("micro arm must be an object")
        if value.get("arm_id") != expected_id or value.get("teacher_control") != expected_control:
            raise ValueError("micro arm identity/control/order drifted")
        qa_dir = Path(str(value.get("qa_output_dir", "")))
        if not qa_dir.is_absolute() or qa_dir.resolve() in output_dirs:
            raise ValueError("micro arm QA output directories must be unique absolute paths")
        qa_dir = qa_dir.resolve()
        output_dirs.add(qa_dir)
        distill_value = value.get("distill_output_dir")
        if regime == "qa_only":
            if distill_value is not None:
                raise ValueError("qa_only micro arms may not declare distillation output directories")
            distill_dir = None
        else:
            distill_path = Path(str(distill_value or ""))
            if not distill_path.is_absolute() or distill_path.resolve() in output_dirs:
                raise ValueError("teacher micro arm distillation directories must be unique absolute paths")
            distill_dir = distill_path.resolve()
            output_dirs.add(distill_dir)
        validated.append(
            {
                "arm_id": expected_id,
                "teacher_control": expected_control,
                "qa_output_dir": qa_dir,
                "distill_output_dir": distill_dir,
            }
        )
    return validated


def _validate_micro_command_semantics(contract: Mapping[str, Any]) -> None:
    regime = str(contract["training_regime"])
    suite = str(contract["suite"])
    arms = _validated_micro_arms(contract)
    data_binding = contract.get("data_binding")
    if not isinstance(data_binding, Mapping) or set(data_binding) != {
        "preregistration_sha256",
        "train_sha256",
        "gate_sha256",
    }:
        raise ValueError("micro command requires exact train/gate data SHA256 bindings")
    if any(SHA256_PATTERN.fullmatch(str(value)) is None for value in data_binding.values()):
        raise ValueError("micro command data SHA256 binding is malformed")
    if dict(data_binding) != _preregistered_micro_data_binding(suite):
        raise ValueError("micro command data binding differs from the preregistered suite lock")
    lineage_binding = contract.get("lineage_binding")
    if not isinstance(lineage_binding, Mapping) or set(lineage_binding) != {
        "git_commit",
        "reader_revision",
        "dreamlite_revision",
    }:
        raise ValueError("micro command requires a complete technical lineage binding")
    if COMMIT_PATTERN.fullmatch(str(lineage_binding.get("git_commit", ""))) is None or any(
        COMMIT_PATTERN.fullmatch(str(lineage_binding.get(field, ""))) is None
        for field in ("reader_revision", "dreamlite_revision")
    ):
        raise ValueError("micro command technical lineage binding is malformed")
    calibration_binding = contract.get("teacher_calibration_binding")
    if regime == "qa_only" and calibration_binding is not None:
        raise ValueError("qa_only micro command must not declare a teacher calibration binding")
    if regime == "teacher_assisted":
        expected_fields = {
            "suite",
            "preregistration_sha256",
            "train_sha256",
            "manifest_sha256",
            "sidecar_sha256",
            "calibration_sha256",
        }
        if not isinstance(calibration_binding, Mapping) or set(calibration_binding) != expected_fields:
            raise ValueError("teacher_assisted micro command requires the complete calibration input binding")
        if calibration_binding.get("suite") != suite or calibration_binding.get("train_sha256") != data_binding.get(
            "train_sha256"
        ):
            raise ValueError("teacher calibration suite/train binding differs from the micro data binding")
        if any(
            SHA256_PATTERN.fullmatch(str(value)) is None
            for field, value in calibration_binding.items()
            if field != "suite"
        ):
            raise ValueError("teacher calibration input SHA256 binding is malformed")

    arm_controls = {arm["teacher_control"] for arm in arms}
    for command in contract["commands"]:
        script = _micro_script(command)
        if script == "scripts/train/dreamlite_episode.py":
            if _command_flag_value(command, "--training-regime", required=True) != regime:
                raise ValueError("micro training command regime differs from its command contract")
            objective = _command_flag_value(command, "--objective-stage", required=True)
            control = str(_command_flag_value(command, "--teacher-control", required=True))
            train_path = Path(str(_command_flag_value(command, "--train", required=True)))
            gate_path = Path(str(_command_flag_value(command, "--dev", required=True)))
            if (
                not train_path.is_absolute()
                or not gate_path.is_absolute()
                or sha256_file(train_path) != data_binding["train_sha256"]
                or sha256_file(gate_path) != data_binding["gate_sha256"]
            ):
                raise ValueError("micro training command data paths do not match the bound train/gate SHA256")
            has_teacher_inputs = any(flag in command for flag in _MICRO_TEACHER_INPUT_FLAGS)
            if regime == "qa_only":
                if control != "correct" or objective != "qa" or has_teacher_inputs or "--initialize-from" in command:
                    raise ValueError("qa_only training must be fresh QA and must not receive teacher inputs")
            elif control not in arm_controls:
                raise ValueError("teacher training command control is absent from the arm contract")
            elif objective == "distill":
                if not all(
                    flag in command for flag in ("--teacher-manifest", "--teacher-sidecar", "--teacher-calibration")
                ):
                    raise ValueError("teacher distillation must receive manifest, sidecar, and calibration")
                manifest_path = Path(str(_command_flag_value(command, "--teacher-manifest", required=True)))
                sidecar_path = Path(str(_command_flag_value(command, "--teacher-sidecar", required=True)))
                if (
                    not manifest_path.is_absolute()
                    or not sidecar_path.is_absolute()
                    or sha256_file(manifest_path) != calibration_binding["manifest_sha256"]
                    or sha256_file(sidecar_path) != calibration_binding["sidecar_sha256"]
                ):
                    raise ValueError("teacher distillation command cache inputs differ from calibration binding")
            elif objective == "qa":
                if has_teacher_inputs or "--initialize-from" not in command:
                    raise ValueError("teacher-lineage QA must unload teacher inputs and initialize from distillation")
            else:
                raise ValueError("teacher_assisted training objective must be distill or qa")
        elif script == "scripts/eval/dreamlite_mcq.py":
            if _command_flag_value(command, "--expected-training-regime", required=True) != regime:
                raise ValueError("micro evaluation regime differs from its command contract")
            episode_path = Path(str(_command_flag_value(command, "--episodes", required=True)))
            if not episode_path.is_absolute() or sha256_file(episode_path) != data_binding["gate_sha256"]:
                raise ValueError("micro evaluation episodes differ from the bound gate SHA256")
        elif script == "scripts/eval/score_r3_micro.py":
            if _command_flag_value(command, "--suite", required=True) != suite:
                raise ValueError("micro scorer suite differs from its command contract")
            _command_flag_value(command, "--prediction-report", required=True)
            expected_score_bindings = {
                "--expected-git-commit": lineage_binding["git_commit"],
                "--expected-reader-revision": lineage_binding["reader_revision"],
                "--expected-dreamlite-revision": lineage_binding["dreamlite_revision"],
                "--expected-train-sha256": data_binding["train_sha256"],
                "--expected-dev-sha256": data_binding["gate_sha256"],
            }
            for flag, expected in expected_score_bindings.items():
                if _command_flag_value(command, flag, required=True) != expected:
                    raise ValueError(f"micro scorer {flag} differs from its command contract")
        elif script == "scripts/eval/teacher_state_retrieval.py":
            if regime != "teacher_assisted":
                raise ValueError("qa_only micro command may not invoke teacher diagnostics")
            calibration_path = Path(str(_command_flag_value(command, "--teacher-calibration", required=True)))
            if (
                not calibration_path.is_absolute()
                or sha256_file(calibration_path) != calibration_binding["calibration_sha256"]
            ):
                raise ValueError("teacher retrieval calibration differs from the arm calibration binding")
        elif script == "scripts/probes/validate_r3_micro_replication.py":
            expected_control = "none" if regime == "qa_only" else "correct"
            if (
                _command_flag_value(command, "--suite", required=True) != suite
                or _command_flag_value(command, "--training-regime", required=True) != regime
                or _command_flag_value(command, "--teacher-control", required=True) != expected_control
            ):
                raise ValueError("micro replication command differs from its suite/regime/control contract")
        elif script == "scripts/eval/score_r3_teacher_attribution.py" and regime != "teacher_assisted":
            raise ValueError("qa_only micro command may not invoke teacher diagnostics")
    _micro_command_dataflow(contract)


def _micro_command_dataflow(contract: Mapping[str, Any]) -> dict[str, Any]:
    suite = str(contract["suite"])
    regime = str(contract["training_regime"])
    episodes = 8 if suite == "set8" else 16
    arms = _validated_micro_arms(contract)
    indexed = list(enumerate(contract["commands"]))
    training = [(index, command) for index, command in indexed if Path(command[1]).name == "dreamlite_episode.py"]
    evaluations = [(index, command) for index, command in indexed if Path(command[1]).name == "dreamlite_mcq.py"]
    scorers = [(index, command) for index, command in indexed if Path(command[1]).name == "score_r3_micro.py"]
    retrievals = [
        (index, command) for index, command in indexed if Path(command[1]).name == "teacher_state_retrieval.py"
    ]
    replications = [
        (index, command) for index, command in indexed if Path(command[1]).name == "validate_r3_micro_replication.py"
    ]
    attributions = [
        (index, command) for index, command in indexed if Path(command[1]).name == "score_r3_teacher_attribution.py"
    ]

    train_by_key: dict[tuple[str, Path], tuple[int, list[str], dict[str, Any]]] = {}
    for index, command in training:
        objective = str(_command_flag_value(command, "--objective-stage", required=True))
        presentations = int(str(_command_flag_value(command, "--presentations-per-state", required=True)))
        epochs = int(str(_command_flag_value(command, "--epochs", required=True)))
        accumulation = int(str(_command_flag_value(command, "--gradient-accumulation", required=True)))
        episode_limit = int(str(_command_flag_value(command, "--max-train-episodes", required=True)))
        expected_presentations = 512 if regime == "qa_only" else 256
        if (
            presentations != expected_presentations
            or epochs != expected_presentations
            or accumulation != 8
            or episode_limit != episodes
        ):
            raise ValueError("micro training command presentations/epochs/accumulation/episode count drifted")
        output_dir = Path(str(_command_flag_value(command, "--output-dir", required=True))).resolve()
        key = (objective, output_dir)
        if key in train_by_key:
            raise ValueError("micro command duplicates one objective/output directory")
        optimizer_steps = presentations * episodes // accumulation
        train_by_key[key] = (
            index,
            command,
            {
                "output_dir": output_dir,
                "final_checkpoint": output_dir / f"checkpoint-{optimizer_steps:06d}.pt",
                "summary": output_dir / "summary.json",
                "metrics": output_dir / "metrics.jsonl",
            },
        )

    eval_by_checkpoint: dict[Path, tuple[int, list[str]]] = {}
    for index, command in evaluations:
        checkpoint = Path(str(_command_flag_value(command, "--checkpoint", required=True))).resolve()
        if checkpoint in eval_by_checkpoint:
            raise ValueError("micro command evaluates one checkpoint more than once")
        eval_by_checkpoint[checkpoint] = (index, command)
    score_by_predictions: dict[Path, tuple[int, list[str]]] = {}
    for index, command in scorers:
        predictions = Path(str(_command_flag_value(command, "--predictions", required=True))).resolve()
        if predictions in score_by_predictions:
            raise ValueError("micro command scores one prediction file more than once")
        score_by_predictions[predictions] = (index, command)
    retrieval_by_checkpoint: dict[Path, tuple[int, list[str]]] = {}
    for index, command in retrievals:
        checkpoint = Path(str(_command_flag_value(command, "--checkpoint", required=True))).resolve()
        if checkpoint in retrieval_by_checkpoint:
            raise ValueError("teacher micro command retrieves one checkpoint more than once")
        retrieval_by_checkpoint[checkpoint] = (index, command)

    required_outputs: dict[Path, str] = {}
    arm_records: dict[str, dict[str, Any]] = {}
    consumed_training: set[int] = set()
    consumed_evaluations: set[int] = set()
    consumed_scorers: set[int] = set()
    consumed_retrievals: set[int] = set()
    previous_score_index = -1
    objective_sequence: list[str] = []
    for arm in arms:
        arm_id = str(arm["arm_id"])
        control = str(arm["teacher_control"])
        qa_key = ("qa", arm["qa_output_dir"])
        if qa_key not in train_by_key:
            raise ValueError(f"micro arm {arm_id} is missing its QA training command")
        qa_index, qa_command, qa_record = train_by_key[qa_key]
        consumed_training.add(qa_index)
        arm_start = qa_index
        distill_record = None
        if regime == "teacher_assisted":
            distill_key = ("distill", arm["distill_output_dir"])
            if distill_key not in train_by_key:
                raise ValueError(f"teacher micro arm {arm_id} is missing distillation training")
            distill_index, distill_command, distill_record = train_by_key[distill_key]
            consumed_training.add(distill_index)
            arm_start = distill_index
            if distill_index >= qa_index:
                raise ValueError("teacher micro arm must finish distillation before QA")
            initialize_from = Path(str(_command_flag_value(qa_command, "--initialize-from", required=True))).resolve()
            if initialize_from != distill_record["final_checkpoint"]:
                raise ValueError("teacher QA does not initialize from its own final distillation checkpoint")
            if (
                _command_flag_value(distill_command, "--teacher-control", required=True) != control
                or _command_flag_value(qa_command, "--teacher-control", required=True) != control
            ):
                raise ValueError("teacher arm training controls drifted")
            objective_sequence.extend(("distill", "qa"))
        else:
            objective_sequence.append("qa")
        if arm_start <= previous_score_index:
            raise ValueError("micro arms must execute serially after the prior arm's scientific score")

        final_qa = qa_record["final_checkpoint"]
        if final_qa not in eval_by_checkpoint:
            raise ValueError(f"micro arm {arm_id} does not evaluate its final QA checkpoint")
        eval_index, evaluation = eval_by_checkpoint[final_qa]
        consumed_evaluations.add(eval_index)
        if eval_index <= qa_index:
            raise ValueError("micro arm evaluation must run after QA training")
        predictions = Path(str(_command_flag_value(evaluation, "--output", required=True))).resolve()
        prediction_report = predictions.with_suffix(predictions.suffix + ".report.json")
        if predictions not in score_by_predictions:
            raise ValueError(f"micro arm {arm_id} predictions are never scored")
        score_index, scorer = score_by_predictions[predictions]
        consumed_scorers.add(score_index)
        if score_index <= eval_index:
            raise ValueError("micro arm score must run after checkpoint evaluation")
        if Path(str(_command_flag_value(scorer, "--prediction-report", required=True))).resolve() != prediction_report:
            raise ValueError("micro scorer companion report does not come from its evaluation")
        gate_report = Path(str(_command_flag_value(scorer, "--output", required=True))).resolve()
        previous_score_index = score_index

        required_outputs.update(
            {
                qa_record["final_checkpoint"]: "artifact",
                qa_record["summary"]: "artifact",
                qa_record["metrics"]: "artifact",
                predictions: "artifact",
                prediction_report: "artifact",
                gate_report: "scientific_report" if suite == "set8" and regime == "qa_only" else "artifact",
            }
        )
        distill_retrieval = qa_retrieval = None
        if distill_record is not None:
            required_outputs.update(
                {
                    distill_record["final_checkpoint"]: "artifact",
                    distill_record["summary"]: "artifact",
                    distill_record["metrics"]: "artifact",
                }
            )
            for name, checkpoint in (("distill", distill_record["final_checkpoint"]), ("qa", final_qa)):
                if checkpoint not in retrieval_by_checkpoint:
                    raise ValueError(f"teacher arm {arm_id} is missing {name} teacher-state retrieval")
                retrieval_index, retrieval = retrieval_by_checkpoint[checkpoint]
                consumed_retrievals.add(retrieval_index)
                retrieval_output = Path(str(_command_flag_value(retrieval, "--output", required=True))).resolve()
                required_outputs[retrieval_output] = "artifact"
                if _command_flag_value(retrieval, "--expected-teacher-control", required=True) != control:
                    raise ValueError("teacher retrieval control differs from its arm")
                if name == "distill":
                    if not (distill_index < retrieval_index < qa_index):
                        raise ValueError("distillation retrieval must precede teacher QA")
                    distill_retrieval = retrieval_output
                else:
                    if not (qa_index < retrieval_index < score_index):
                        raise ValueError("QA retrieval must occur before the arm score")
                    qa_retrieval = retrieval_output
        arm_records[arm_id] = {
            "control": control,
            "distill": distill_record,
            "qa": qa_record,
            "distill_retrieval": distill_retrieval,
            "qa_retrieval": qa_retrieval,
            "predictions": predictions,
            "prediction_report": prediction_report,
            "gate_report": gate_report,
            "score_index": score_index,
        }

    if consumed_training != {index for index, _ in training}:
        raise ValueError("micro command contains training outside the declared arms")
    if consumed_evaluations != {index for index, _ in evaluations} or consumed_scorers != {
        index for index, _ in scorers
    }:
        raise ValueError("micro command contains evaluation/scoring outside the declared arms")
    if consumed_retrievals != {index for index, _ in retrievals}:
        raise ValueError("micro command contains teacher retrieval outside the declared arms")

    if suite == "transition16":
        if len(replications) != 1 or attributions:
            raise ValueError("Transition16 requires exactly one A/B replication command and no attribution command")
        replication_index, replication = replications[0]
        if replication_index <= previous_score_index:
            raise ValueError("Transition16 replication must run after both fresh replicas")
        a_report = arm_records["A"]["gate_report"]
        b_report = arm_records["B"]["gate_report"]
        if (
            Path(str(_command_flag_value(replication, "--a", required=True))).resolve() != a_report
            or Path(str(_command_flag_value(replication, "--b", required=True))).resolve() != b_report
        ):
            raise ValueError("Transition16 replication does not consume this stage's A/B gate reports")
        final_report = Path(str(_command_flag_value(replication, "--output", required=True))).resolve()
        required_outputs[final_report] = "scientific_report"
    elif regime == "teacher_assisted":
        if replications or len(attributions) != 1:
            raise ValueError("TD8 requires exactly one three-control attribution command")
        attribution_index, attribution = attributions[0]
        if attribution_index <= previous_score_index:
            raise ValueError("TD8 attribution must run after all three fresh control arms")
        cli_names = {"correct": "correct", "shuffled": "shuffled", "random": "random"}
        for arm_id, cli_name in cli_names.items():
            record = arm_records[arm_id]
            expected_inputs = {
                f"--{cli_name}-distill-summary": record["distill"]["summary"],
                f"--{cli_name}-qa-summary": record["qa"]["summary"],
                f"--{cli_name}-distill-retrieval": record["distill_retrieval"],
                f"--{cli_name}-qa-retrieval": record["qa_retrieval"],
                f"--{cli_name}-qa-gate": record["gate_report"],
            }
            for flag, expected in expected_inputs.items():
                if Path(str(_command_flag_value(attribution, flag, required=True))).resolve() != expected:
                    raise ValueError(f"TD8 attribution input {flag} does not come from its control arm")
        final_report = Path(str(_command_flag_value(attribution, "--output", required=True))).resolve()
        required_outputs[final_report] = "scientific_report"
    elif replications or attributions:
        raise ValueError("Set8 QA must not include replication or teacher attribution")
    else:
        final_report = arm_records["A"]["gate_report"]

    declared_outputs = {Path(str(output["path"])).resolve(): output for output in contract["outputs"]}
    missing = sorted(str(path) for path in required_outputs if path not in declared_outputs)
    if missing:
        raise ValueError(f"micro command dataflow artifacts are absent from audited outputs: {missing}")
    for path, kind in required_outputs.items():
        if declared_outputs[path].get("kind") != kind:
            raise ValueError(f"micro output {path} has the wrong artifact kind")
    return {
        "training_objectives": objective_sequence,
        "arms": arm_records,
        "final_report": final_report,
        "all_paths": list(required_outputs),
    }


def _validate_micro_runtime_bindings(contract: Mapping[str, Any], technical: Mapping[str, Any]) -> None:
    expected_python = Path(os.path.abspath(str(technical["python"])))
    model_root = Path(str(technical["model_root"])).resolve()
    expected_dreamlite = model_root / "DreamLite-mobile"
    expected_reader = model_root / "Qwen3-VL-4B-Instruct"
    for command in contract["commands"]:
        if Path(os.path.abspath(command[0])) != expected_python:
            raise ValueError("micro command Python differs from the technical parent's locked environment")
        script = _micro_script(command)
        if script in {"scripts/train/dreamlite_episode.py", "scripts/eval/dreamlite_mcq.py"}:
            dreamlite = Path(str(_command_flag_value(command, "--dreamlite", required=True))).resolve()
            reader = Path(str(_command_flag_value(command, "--reader", required=True))).resolve()
            if dreamlite != expected_dreamlite or reader != expected_reader:
                raise ValueError("micro command model paths differ from the technical parent's model root")
        elif script == "scripts/eval/teacher_state_retrieval.py":
            dreamlite = Path(str(_command_flag_value(command, "--dreamlite", required=True))).resolve()
            if dreamlite != expected_dreamlite:
                raise ValueError("teacher retrieval DreamLite path differs from the technical parent's model root")


def _load_micro_command_contract(path: Path) -> dict[str, Any]:
    contract = load_json_object(path)
    require_json_values(contract, {"schema_version": 2, "protocol": MICRO_COMMAND_PROTOCOL}, "micro command")
    if contract.get("suite") not in {"set8", "transition16"}:
        raise ValueError("micro command suite must be set8 or transition16")
    if contract.get("training_regime") not in {"qa_only", "teacher_assisted"}:
        raise ValueError("micro command training_regime must be qa_only or teacher_assisted")
    expected_fixed_protocol = {
        "reader_loss_mode": "listwise-choice",
        "train_choice_family": "cyclic4",
        "gate_choice_family": "reverse-cyclic4",
        "dreamlite": "DreamLite-mobile-4-step",
        "lora_rank": 4,
        "strict_determinism": True,
    }
    if contract.get("fixed_protocol") != expected_fixed_protocol:
        raise ValueError("micro command does not declare the locked R3 fixed protocol")
    if not isinstance(contract.get("commands"), list) or not contract["commands"]:
        raise ValueError("micro command requires at least one argv command")
    for command in contract["commands"]:
        if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
            raise ValueError("each micro command must be an argv string list")
        validate_command(command)
    if not isinstance(contract.get("outputs"), list) or not contract["outputs"]:
        raise ValueError("micro command requires audited outputs")
    labels: set[str] = set()
    paths: set[str] = set()
    for output in contract["outputs"]:
        if not isinstance(output, Mapping):
            raise ValueError("each micro output must be an object")
        label = output.get("label")
        path_value = output.get("path")
        required_values = output.get("required_values")
        kind = output.get("kind")
        if not isinstance(label, str) or not label or label in labels:
            raise ValueError("micro output labels must be non-empty and unique")
        if not isinstance(path_value, str) or not Path(path_value).is_absolute() or path_value in paths:
            raise ValueError("micro output paths must be absolute and unique")
        if kind not in {"artifact", "scientific_report"} or not isinstance(required_values, Mapping):
            raise ValueError("micro outputs must declare artifact/scientific_report kind and required_values")
        if kind == "artifact" and required_values:
            raise ValueError("micro binary/stream artifacts must use empty required_values")
        if kind == "scientific_report" and (
            required_values.get("passed") is not True
            or required_values.get("artifact_provenance_validated") is not True
        ):
            raise ValueError("micro scientific reports must require passed and artifact_provenance_validated")
        labels.add(label)
        paths.add(path_value)
    if not any(
        output.get("kind") == "scientific_report"
        and output["required_values"].get("suite") == contract["suite"]
        and output["required_values"].get("training_regime") == contract["training_regime"]
        for output in contract["outputs"]
    ):
        raise ValueError("micro outputs must bind at least one suite-and-training-regime scientific report")
    _validate_micro_command_semantics(contract)
    return contract


def _validate_micro_prerequisites(
    *,
    resize: Mapping[str, Any],
    scorer: Mapping[str, Any],
    technical: Mapping[str, Any],
    teacher_t0: Mapping[str, Any] | None,
    teacher_calibration_report: Mapping[str, Any] | None,
    teacher_calibration_file_sha256: str | None,
    teacher_tc0: Mapping[str, Any] | None,
    teacher_tc0_file_sha256: str | None,
    teacher_tf0: Mapping[str, Any] | None,
    teacher_tf0_file_sha256: str | None,
    training_regime: str,
    expected_commit: str,
    teacher_calibration_suite: str | None = None,
    teacher_calibration_preregistration_sha256: str | None = None,
    teacher_calibration_train_sha256: str | None = None,
    teacher_calibration_manifest_sha256: str | None = None,
    teacher_calibration_sidecar_sha256: str | None = None,
) -> dict[str, Any]:
    """Keep the Inspire adapter explicit when the shared prerequisite signature evolves."""

    return validate_prerequisites(
        resize_contract=resize,
        scorer_s0=scorer,
        technical=technical,
        teacher_t0=teacher_t0,
        teacher_calibration=teacher_calibration_report,
        teacher_calibration_file_sha256=teacher_calibration_file_sha256,
        training_regime=training_regime,
        expected_commit=expected_commit,
        teacher_tc0=teacher_tc0,
        teacher_tc0_file_sha256=teacher_tc0_file_sha256,
        teacher_tf0=teacher_tf0,
        teacher_tf0_file_sha256=teacher_tf0_file_sha256,
        teacher_calibration_suite=teacher_calibration_suite,
        teacher_calibration_preregistration_sha256=teacher_calibration_preregistration_sha256,
        teacher_calibration_train_sha256=teacher_calibration_train_sha256,
        teacher_calibration_manifest_sha256=teacher_calibration_manifest_sha256,
        teacher_calibration_sidecar_sha256=teacher_calibration_sidecar_sha256,
    )


def _completed_teacher_preparation_parent(
    *,
    teacher_run_root: Path,
    technical: Mapping[str, Any],
    technical_run_root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    teacher_run_root = teacher_run_root.resolve()
    teacher, teacher_plan_path, teacher_plan_sha256 = _load_verified_plan(teacher_run_root)
    if teacher.get("kind") != "teacher-preparation" or teacher.get("strict_order") != list(TEACHER_PREPARATION_ORDER):
        raise ValueError("teacher_assisted micro requires the complete immutable teacher-preparation DAG")
    expected_parent = {
        "expected_commit": technical["expected_commit"],
        "repo": technical["repo"],
        "python": technical["python"],
        "model_root": technical["model_root"],
        "model_snapshots": technical["model_snapshots"],
        "formal_preflight": technical["formal_preflight"],
    }
    for field, expected in expected_parent.items():
        if teacher.get(field) != expected:
            raise ValueError(f"teacher-preparation parent {field} differs from the technical parent")
    technical_parent = teacher.get("technical_parent")
    technical_plan_path, _ = _plan_paths(technical_run_root)
    if (
        not isinstance(technical_parent, Mapping)
        or Path(str(technical_parent.get("run_root", ""))).resolve() != technical_run_root
        or technical_parent.get("plan_sha256") != verify_sha_sidecar(technical_plan_path)
        or technical_parent.get("completed_stage_count") != len(TECHNICAL_ORDER)
    ):
        raise ValueError("teacher-preparation DAG is not bound to this exact completed technical parent")

    completion_bindings: list[dict[str, Any]] = []
    for stage in TEACHER_PREPARATION_ORDER:
        completion_bindings.extend(_verify_completed_stage(teacher, stage))
    outputs = _all_planned_outputs(teacher)
    final_output = outputs.get("teacher_preparation_final")
    if not isinstance(final_output, Mapping):
        raise ValueError("teacher-preparation parent is missing its final bound artifact index")
    final_binding = {
        "label": "teacher-parent:final",
        "path": final_output["path"],
        "sha256": sha256_file(Path(str(final_output["path"]))),
        "required_values": final_output.get("required_values", {}),
    }
    final_value = verify_bound_artifact(final_binding)
    final_artifacts = final_value.get("artifacts") if isinstance(final_value, Mapping) else None
    final_labels = {
        "tc0_raw": "teacher_tc0_raw",
        "tc0_validation": "teacher_tc0_validation",
        "tf0_raw": "teacher_tf0_raw",
        "tf0_validation": "teacher_tf0_validation",
        "t0": "teacher_t0",
        "set8_calibration": "teacher_set8_calibration",
        "set8_calibration_report": "teacher_set8_calibration_report",
        "transition16_calibration": "teacher_transition16_calibration",
        "transition16_calibration_report": "teacher_transition16_calibration_report",
    }
    if not isinstance(final_artifacts, Mapping) or set(final_artifacts) != set(final_labels):
        raise ValueError("teacher-preparation final index does not enumerate every scientific artifact")
    for final_name, output_label in final_labels.items():
        output = outputs.get(output_label)
        indexed = final_artifacts.get(final_name)
        if (
            not isinstance(output, Mapping)
            or not isinstance(indexed, Mapping)
            or indexed.get("path") != output.get("path")
            or indexed.get("sha256") != sha256_file(Path(str(output.get("path"))))
        ):
            raise ValueError(f"teacher-preparation final index has a drifted {final_name} binding")
    suite = str(contract["suite"])
    calibration_binding = contract.get("teacher_calibration_binding")
    teacher_contract = teacher.get("teacher_contract")
    locks = teacher_contract.get("calibration_input_locks") if isinstance(teacher_contract, Mapping) else None
    suite_lock = locks.get(suite) if isinstance(locks, Mapping) else None
    expected_input_binding = (
        {
            field: suite_lock[field]
            for field in (
                "suite",
                "preregistration_sha256",
                "train_sha256",
                "manifest_sha256",
                "sidecar_sha256",
            )
        }
        if isinstance(suite_lock, Mapping)
        else None
    )
    data_binding = contract.get("data_binding")
    if (
        not isinstance(data_binding, Mapping)
        or data_binding.get("preregistration_sha256") != expected_input_binding["preregistration_sha256"]
        or data_binding.get("train_sha256") != expected_input_binding["train_sha256"]
    ):
        raise ValueError("micro data binding differs from its teacher-preparation parent")

    calibration_prefix = "teacher_set8" if suite == "set8" else "teacher_transition16"
    selected_labels = {
        "tc0_validation": "teacher_tc0_validation",
        "tf0_validation": "teacher_tf0_validation",
        "t0": "teacher_t0",
        "calibration": f"{calibration_prefix}_calibration",
        "calibration_report": f"{calibration_prefix}_calibration_report",
    }
    artifact_bindings: dict[str, dict[str, Any]] = {}
    for key, label in selected_labels.items():
        output = outputs.get(label)
        if not isinstance(output, Mapping):
            raise ValueError(f"teacher-preparation parent is missing output {label}")
        binding = {
            "label": f"teacher-parent:{label}",
            "path": output["path"],
            "sha256": sha256_file(Path(str(output["path"]))),
            "required_values": output.get("required_values", {}),
        }
        verify_bound_artifact(binding)
        artifact_bindings[key] = binding
    expected_calibration_binding = {
        **expected_input_binding,
        "calibration_sha256": artifact_bindings["calibration"]["sha256"],
    }
    if calibration_binding != expected_calibration_binding:
        raise ValueError("micro calibration binding differs from its completed teacher-preparation parent")
    return {
        "plan": teacher,
        "completion_bindings": [*completion_bindings, final_binding],
        "artifacts": artifact_bindings,
        "calibration_binding": expected_calibration_binding,
        "identity": {
            "run_root": str(teacher_run_root),
            "plan_path": str(teacher_plan_path),
            "plan_sha256": teacher_plan_sha256,
            "final_index_path": final_binding["path"],
            "final_index_sha256": final_binding["sha256"],
            "required_complete_order": list(TEACHER_PREPARATION_ORDER),
        },
    }


def _validate_teacher_parent_policy(
    *,
    regime: str,
    teacher_preparation_run_root: Path | None,
    loose_teacher_inputs: Mapping[str, Any],
) -> None:
    has_loose = any(value is not None for value in loose_teacher_inputs.values())
    if regime == "qa_only":
        if teacher_preparation_run_root is not None or has_loose:
            raise ValueError("qa_only micro materialization rejects every teacher-preparation input")
        return
    if teacher_preparation_run_root is None:
        raise ValueError("teacher_assisted micro requires a completed teacher-preparation parent")
    if has_loose:
        raise ValueError(
            "teacher_assisted micro rejects loose T0/TC0/TF0/calibration evidence; use its immutable parent"
        )


def _require_run_child(path: Path, *, approved_root: Path, label: str) -> Path:
    value = require_absolute(path, label)
    root = require_absolute(approved_root, "formal VLM_RUN_ROOT")
    if value == root or not is_within(value, root):
        raise ValueError(f"{label} must be a strict child of the formal preflight VLM_RUN_ROOT")
    return value


def _validate_set8_parent_role(
    *,
    set8_contract: Mapping[str, Any],
    transition_contract: Mapping[str, Any],
    regime: str,
) -> None:
    """Validate cross-suite roles; Set8 teacher attribution is composite, TD16 uses correct teacher."""

    if set8_contract.get("suite") != "set8" or set8_contract.get("training_regime") != regime:
        raise ValueError("Transition16 Set8 parent has the wrong regime lineage")
    expected_parent_control = "none" if regime == "qa_only" else "composite"
    expected_child_control = "none" if regime == "qa_only" else "correct"
    if (
        set8_contract.get("teacher_control") != expected_parent_control
        or transition_contract.get("teacher_control") != expected_child_control
    ):
        raise ValueError("Transition16 Set8 parent/child teacher-control roles are malformed")
    if regime == "teacher_assisted" and not any(
        output.get("label") == "teacher-attribution"
        and output.get("kind") == "scientific_report"
        and output.get("required_values", {}).get("passed") is True
        and output.get("required_values", {}).get("artifact_provenance_validated") is True
        for output in set8_contract.get("outputs", [])
        if isinstance(output, Mapping)
    ):
        raise ValueError("Teacher-assisted Transition16 requires a passed composite Set8 attribution parent")


def _validate_set8_parent_dag_identity(
    *,
    set8_parent: Mapping[str, Any],
    technical_parent: Mapping[str, Any],
    teacher_parent: Mapping[str, Any] | None,
    regime: str,
) -> None:
    if set8_parent.get("technical_parent") != technical_parent:
        raise ValueError("Transition16 Set8 parent belongs to a different technical DAG")
    observed_teacher = set8_parent.get("teacher_preparation_parent")
    if regime == "teacher_assisted":
        if teacher_parent is None or observed_teacher != teacher_parent:
            raise ValueError("Transition16 Set8 parent belongs to a different teacher-preparation DAG")
    elif observed_teacher is not None:
        raise ValueError("QA-only Transition16 cannot use a teacher-lineage Set8 parent")


def initialize_micro_extension(
    *,
    technical_run_root: Path,
    micro_run_root: Path,
    command_contract_path: Path,
    teacher_preparation_run_root: Path | None = None,
    teacher_t0_path: Path | None = None,
    teacher_t0_sha256: str | None = None,
    teacher_calibration_path: Path | None = None,
    teacher_calibration_sha256: str | None = None,
    teacher_calibration_report_path: Path | None = None,
    teacher_calibration_report_sha256: str | None = None,
    tc0_validation_path: Path | None = None,
    tc0_validation_sha256: str | None = None,
    tf0_validation_path: Path | None = None,
    tf0_validation_sha256: str | None = None,
    set8_parent_run_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    technical_run_root = technical_run_root.resolve()
    technical, technical_plan_path, technical_plan_sha256 = _load_verified_plan(technical_run_root)
    if technical.get("kind") != "technical" or technical.get("strict_order") != list(TECHNICAL_ORDER):
        raise ValueError("Micro materialization requires the complete six-stage technical plan")
    formal_preflight_path = Path(technical["formal_preflight"]["path"])
    verify_sha_sidecar(
        formal_preflight_path,
        expected_sha256=str(technical["formal_preflight"]["sha256"]),
    )
    formal_preflight = load_json_object(formal_preflight_path)
    reported_runs_root = formal_preflight.get("paths", {}).get("VLM_RUN_ROOT", {}).get("value")
    if not isinstance(reported_runs_root, str):
        raise ValueError("Technical formal preflight does not bind VLM_RUN_ROOT")
    micro_run_root = _require_run_child(
        micro_run_root,
        approved_root=Path(reported_runs_root),
        label="micro_run_root",
    )
    verify_clean_commit(Path(technical["repo"]), str(technical["expected_commit"]))
    technical_chain_bindings: list[dict[str, Any]] = []
    for technical_stage in TECHNICAL_ORDER:
        technical_chain_bindings.extend(_verify_completed_stage(technical, technical_stage))

    results = technical_run_root / "results"
    resize_path = results / "R3_R0_qwen_resize_contract.json"
    scorer_path = results / "R3_S0_qwen_scorer_contract.json"
    technical_path = results / "R3_technical_gates_final.json"
    resize = load_json_object(resize_path)
    scorer = load_json_object(scorer_path)
    final = load_json_object(technical_path)
    contract = _load_micro_command_contract(command_contract_path.resolve())
    _validate_micro_runtime_bindings(contract, technical)
    regime = str(contract["training_regime"])
    calibration_input_binding = contract.get("teacher_calibration_binding")
    technical_checks = final.get("checks")
    dl_s_check = technical_checks.get("DL-S") if isinstance(technical_checks, Mapping) else None
    expected_lineage_binding = {
        "git_commit": technical["expected_commit"],
        "reader_revision": dl_s_check.get("reader_revision") if isinstance(dl_s_check, Mapping) else None,
        "dreamlite_revision": dl_s_check.get("dreamlite_revision") if isinstance(dl_s_check, Mapping) else None,
    }
    if contract.get("lineage_binding") != expected_lineage_binding:
        raise ValueError("micro command lineage does not match the completed technical parent")
    training_commands = [command for command in contract["commands"] if Path(command[1]).name == "dreamlite_episode.py"]
    train_paths = {
        Path(str(_command_flag_value(command, "--train", required=True))).resolve() for command in training_commands
    }
    gate_paths = {
        Path(str(_command_flag_value(command, "--dev", required=True))).resolve() for command in training_commands
    }
    evaluation_commands = [command for command in contract["commands"] if Path(command[1]).name == "dreamlite_mcq.py"]
    evaluation_episode_paths = {
        Path(str(_command_flag_value(command, "--episodes", required=True))).resolve()
        for command in evaluation_commands
    }
    if len(train_paths) != 1 or len(gate_paths) != 1 or evaluation_episode_paths not in (set(), gate_paths):
        raise ValueError("micro commands must share one exact train path and one exact train/evaluation gate path")
    train_path = next(iter(train_paths))
    gate_path = next(iter(gate_paths))
    loose_teacher_inputs = {
        "teacher_t0_path": teacher_t0_path,
        "teacher_t0_sha256": teacher_t0_sha256,
        "teacher_calibration_path": teacher_calibration_path,
        "teacher_calibration_sha256": teacher_calibration_sha256,
        "teacher_calibration_report_path": teacher_calibration_report_path,
        "teacher_calibration_report_sha256": teacher_calibration_report_sha256,
        "tc0_validation_path": tc0_validation_path,
        "tc0_validation_sha256": tc0_validation_sha256,
        "tf0_validation_path": tf0_validation_path,
        "tf0_validation_sha256": tf0_validation_sha256,
    }
    teacher_parent_bindings: list[dict[str, Any]] = []
    teacher_parent_identity: dict[str, Any] | None = None
    _validate_teacher_parent_policy(
        regime=regime,
        teacher_preparation_run_root=teacher_preparation_run_root,
        loose_teacher_inputs=loose_teacher_inputs,
    )
    if regime == "teacher_assisted":
        parent = _completed_teacher_preparation_parent(
            teacher_run_root=teacher_preparation_run_root,
            technical=technical,
            technical_run_root=technical_run_root,
            contract=contract,
        )
        teacher_parent_identity = dict(parent["identity"])
        artifacts = parent["artifacts"]
        teacher_parent_bindings = parent["completion_bindings"] + list(artifacts.values())
        teacher_t0_path = Path(artifacts["t0"]["path"])
        teacher_t0_sha256 = artifacts["t0"]["sha256"]
        teacher_calibration_path = Path(artifacts["calibration"]["path"])
        teacher_calibration_sha256 = artifacts["calibration"]["sha256"]
        teacher_calibration_report_path = Path(artifacts["calibration_report"]["path"])
        teacher_calibration_report_sha256 = artifacts["calibration_report"]["sha256"]
        tc0_validation_path = Path(artifacts["tc0_validation"]["path"])
        tc0_validation_sha256 = artifacts["tc0_validation"]["sha256"]
        tf0_validation_path = Path(artifacts["tf0_validation"]["path"])
        tf0_validation_sha256 = artifacts["tf0_validation"]["sha256"]
        calibration_input_binding = parent["calibration_binding"]

    teacher: Mapping[str, Any] | None = None
    calibration_report: Mapping[str, Any] | None = None
    tc0_validation: Mapping[str, Any] | None = None
    tf0_validation: Mapping[str, Any] | None = None
    teacher_bindings: list[dict[str, Any]] = []
    if regime == "teacher_assisted":
        expected_commit = str(technical["expected_commit"])
        if calibration_input_binding["calibration_sha256"] != teacher_calibration_sha256:
            raise ValueError("micro command calibration SHA256 differs from the explicitly bound file")
        typed_paths = {
            "teacher:T0": (teacher_t0_path, teacher_t0_sha256),
            "teacher:calibration-file": (teacher_calibration_path, teacher_calibration_sha256),
            "teacher:calibration-report": (
                teacher_calibration_report_path,
                teacher_calibration_report_sha256,
            ),
            "teacher:TC0-validation": (tc0_validation_path, tc0_validation_sha256),
            "teacher:TF0-validation": (tf0_validation_path, tf0_validation_sha256),
        }
        resolved: dict[str, Path] = {}
        for label, (path_value, sha_value) in typed_paths.items():
            if path_value is None or sha_value is None:  # Defensive after the complete-set check above.
                raise ValueError(f"{label} path/SHA256 binding is incomplete")
            path = path_value.resolve()
            if SHA256_PATTERN.fullmatch(sha_value) is None or sha256_file(path) != sha_value:
                raise ValueError(f"{label} does not match its explicit SHA256 binding")
            resolved[label] = path
        teacher = load_json_object(resolved["teacher:T0"])
        calibration_report = load_json_object(resolved["teacher:calibration-report"])
        tc0_validation = load_json_object(resolved["teacher:TC0-validation"])
        tf0_validation = load_json_object(resolved["teacher:TF0-validation"])
        require_json_values(
            tc0_validation,
            {
                "protocol": "R3-TC0-cache-forward-compatibility-validation.v1",
                "expected_commit": expected_commit,
                "passed": True,
                "cache_forward_compatibility_complete": True,
                "teacher_calibration_unlocked": False,
            },
            "R3-TC0 validation",
        )
        require_json_values(
            tf0_validation,
            {
                "protocol": "R3-TF0-feature-backend-compatibility-validation.v1",
                "expected_commit": expected_commit,
                "passed": True,
                "tc0_validation_sha256": tc0_validation_sha256,
                "teacher_t0_unlocked": True,
                "teacher_calibration_unlocked": True,
                "teacher_assisted_training_unlocked": True,
            },
            "R3-TF0 validation",
        )
        require_json_values(
            calibration_report,
            {
                "schema": "vision_memory.r3-teacher-calibration-report.v1",
                "calibration_file_sha256": teacher_calibration_sha256,
            },
            "teacher calibration report",
        )
        distill_commands = [
            command
            for command in contract["commands"]
            if Path(command[1]).name == "dreamlite_episode.py"
            and _command_flag_value(command, "--objective-stage", required=True) == "distill"
        ]
        expected_distill_commands = len(_validated_micro_arms(contract))
        if len(distill_commands) != expected_distill_commands:
            raise ValueError("teacher_assisted command contract must contain one fresh distillation command per arm")
        command_manifests: set[Path] = set()
        command_sidecars: set[Path] = set()
        for distill_command in distill_commands:
            command_calibration = Path(
                str(_command_flag_value(distill_command, "--teacher-calibration", required=True))
            ).resolve()
            if (
                command_calibration != resolved["teacher:calibration-file"]
                or sha256_file(command_calibration) != teacher_calibration_sha256
            ):
                raise ValueError("distillation command does not use the explicitly bound calibration file")
            command_manifests.add(
                Path(str(_command_flag_value(distill_command, "--teacher-manifest", required=True))).resolve()
            )
            command_sidecars.add(
                Path(str(_command_flag_value(distill_command, "--teacher-sidecar", required=True))).resolve()
            )
        if len(command_manifests) != 1 or len(command_sidecars) != 1:
            raise ValueError("all teacher arms must consume the same immutable manifest and sidecar")
        teacher_bindings = [
            {
                "label": label,
                "path": str(path),
                "sha256": typed_paths[label][1],
                "required_values": (
                    {"passed": True}
                    if label == "teacher:T0"
                    else (
                        {
                            "passed": True,
                            "expected_commit": expected_commit,
                        }
                        if label in {"teacher:TC0-validation", "teacher:TF0-validation"}
                        else (
                            {
                                "schema": "vision_memory.r3-teacher-calibration-report.v1",
                                "calibration_file_sha256": teacher_calibration_sha256,
                            }
                            if label == "teacher:calibration-report"
                            else {}
                        )
                    )
                ),
            }
            for label, path in resolved.items()
        ]
        teacher_bindings.extend(teacher_parent_bindings)
        teacher_bindings.extend(
            (
                {
                    "label": "teacher:manifest",
                    "path": str(next(iter(command_manifests))),
                    "sha256": calibration_input_binding["manifest_sha256"],
                    "required_values": {},
                },
                {
                    "label": "teacher:sidecar",
                    "path": str(next(iter(command_sidecars))),
                    "sha256": calibration_input_binding["sidecar_sha256"],
                    "required_values": {},
                },
            )
        )

    prerequisite_report = _validate_micro_prerequisites(
        resize=resize,
        scorer=scorer,
        technical=final,
        teacher_t0=teacher,
        teacher_calibration_report=calibration_report,
        teacher_calibration_file_sha256=(teacher_calibration_sha256 if regime == "teacher_assisted" else None),
        teacher_tc0=tc0_validation,
        teacher_tc0_file_sha256=(tc0_validation_sha256 if regime == "teacher_assisted" else None),
        teacher_tf0=tf0_validation,
        teacher_tf0_file_sha256=(tf0_validation_sha256 if regime == "teacher_assisted" else None),
        training_regime=regime,
        expected_commit=str(technical["expected_commit"]),
        teacher_calibration_suite=(calibration_input_binding["suite"] if regime == "teacher_assisted" else None),
        teacher_calibration_preregistration_sha256=(
            calibration_input_binding["preregistration_sha256"] if regime == "teacher_assisted" else None
        ),
        teacher_calibration_train_sha256=(
            calibration_input_binding["train_sha256"] if regime == "teacher_assisted" else None
        ),
        teacher_calibration_manifest_sha256=(
            calibration_input_binding["manifest_sha256"] if regime == "teacher_assisted" else None
        ),
        teacher_calibration_sidecar_sha256=(
            calibration_input_binding["sidecar_sha256"] if regime == "teacher_assisted" else None
        ),
    )
    if prerequisite_report.get("passed") is not True:
        raise ValueError(f"Micro prerequisites failed: {prerequisite_report.get('errors')}")

    contract_sha256 = sha256_file(command_contract_path)
    evidence_bindings = [
        {
            "label": "technical:R3-R0",
            "path": str(resize_path),
            "sha256": sha256_file(resize_path),
            "required_values": {"passed": True},
        },
        {
            "label": "technical:R3-S0",
            "path": str(scorer_path),
            "sha256": sha256_file(scorer_path),
            "required_values": {"passed": True},
        },
        {
            "label": "technical:final",
            "path": str(technical_path),
            "sha256": sha256_file(technical_path),
            "required_values": {"passed": True, "through": "DL-S"},
        },
        {
            "label": "micro-command-contract",
            "path": str(command_contract_path.resolve()),
            "sha256": contract_sha256,
            "required_values": {"schema_version": 2, "protocol": MICRO_COMMAND_PROTOCOL},
        },
    ]
    evidence_bindings.extend(
        (
            {
                "label": "micro-data:train",
                "path": str(train_path),
                "sha256": contract["data_binding"]["train_sha256"],
                "required_values": {},
            },
            {
                "label": "micro-data:gate",
                "path": str(gate_path),
                "sha256": contract["data_binding"]["gate_sha256"],
                "required_values": {},
            },
        )
    )
    evidence_bindings.extend(
        {
            **binding,
            "label": f"technical:{binding['label']}",
        }
        for binding in technical_chain_bindings
    )
    evidence_bindings.extend(teacher_bindings)
    if contract["suite"] == "transition16":
        if set8_parent_run_root is None:
            raise ValueError("transition16 materialization requires a passed same-regime Set8 parent")
        set8_parent, _, _ = _load_verified_plan(set8_parent_run_root.resolve())
        if set8_parent.get("kind") != "micro-extension" or len(set8_parent.get("strict_order", [])) != 1:
            raise ValueError("Set8 parent is not an immutable single-stage micro extension")
        set8_stage = set8_parent["strict_order"][0]
        set8_definition = set8_parent["stages"][set8_stage]
        if set8_definition.get("suite") != "set8" or set8_definition.get("training_regime") != regime:
            raise ValueError("Transition16 Set8 parent has the wrong suite or training regime")
        if set8_parent.get("expected_commit") != technical.get("expected_commit"):
            raise ValueError("Transition16 Set8 parent was not produced from the technical commit")
        expected_technical_parent = {
            "run_root": str(technical_run_root),
            "plan_path": str(technical_plan_path),
            "plan_sha256": technical_plan_sha256,
            "formal_preflight": technical["formal_preflight"],
        }
        _validate_set8_parent_dag_identity(
            set8_parent=set8_parent,
            technical_parent=expected_technical_parent,
            teacher_parent=teacher_parent_identity,
            regime=regime,
        )
        _verify_completed_stage(set8_parent, str(set8_stage))
        set8_terminal = Path(set8_definition["run_dir"]) / "terminal.json"
        set8_evidence = Path(set8_definition["evidence_path"])
        set8_contract_binding = next(
            (
                binding
                for binding in set8_definition.get("external_prerequisites", [])
                if binding.get("label") == "micro-command-contract"
            ),
            None,
        )
        if not isinstance(set8_contract_binding, Mapping):
            raise ValueError("Set8 parent does not bind its immutable micro command contract")
        set8_contract = _load_micro_command_contract(Path(str(set8_contract_binding["path"])))
        _validate_set8_parent_role(
            set8_contract=set8_contract,
            transition_contract=contract,
            regime=regime,
        )
        evidence_bindings.extend(
            (
                {
                    "label": "micro-parent:Set8-plan",
                    "path": str(_plan_paths(set8_parent_run_root.resolve())[0]),
                    "sha256": verify_sha_sidecar(_plan_paths(set8_parent_run_root.resolve())[0]),
                    "required_values": {
                        "kind": "micro-extension",
                        "expected_commit": technical["expected_commit"],
                    },
                },
                {
                    "label": "micro-parent:Set8-terminal",
                    "path": str(set8_terminal),
                    "sha256": sha256_file(set8_terminal),
                    "required_values": {"passed": True, "status": "succeeded"},
                },
                {
                    "label": "micro-parent:Set8-evidence",
                    "path": str(set8_evidence),
                    "sha256": sha256_file(set8_evidence),
                    "required_values": {"passed": True},
                },
            )
        )
    elif set8_parent_run_root is not None:
        raise ValueError("Set8 materialization must not receive a Set8 parent")
    for binding in evidence_bindings:
        verify_bound_artifact(binding)

    for output in contract["outputs"]:
        if not is_within(Path(output["path"]), micro_run_root):
            raise ValueError("Every micro command output must be inside the unique micro run root")
    stage = str(contract["stage"])
    slug = f"00-{contract['suite']}-{regime.replace('_', '-')}"
    plan = {
        "schema_version": 1,
        "protocol": PLAN_PROTOCOL,
        "kind": "micro-extension",
        "execution_backend": "inspire-notebook-background",
        "submission_backend": "scripts/inspire/launch_background.py",
        "external_scheduler_submission": False,
        "expected_commit": technical["expected_commit"],
        "repo": technical["repo"],
        "python": technical["python"],
        "model_root": technical["model_root"],
        "model_snapshots": technical["model_snapshots"],
        "run_root": str(micro_run_root),
        "formal_preflight": technical["formal_preflight"],
        "strict_order": [stage],
        "technical_parent": {
            "run_root": str(technical_run_root),
            "plan_path": str(technical_plan_path),
            "plan_sha256": technical_plan_sha256,
            "formal_preflight": technical["formal_preflight"],
        },
        "teacher_preparation_parent": teacher_parent_identity,
        "command_contract_sha256": contract_sha256,
        "prerequisite_report": prerequisite_report,
        "stages": {
            stage: {
                "index": 0,
                "slug": slug,
                "launcher_stage": slug,
                "dependency": None,
                "run_dir": str(micro_run_root / "stages" / slug),
                "evidence_path": str(micro_run_root / "evidence" / f"{slug}.json"),
                "commands": contract["commands"],
                "outputs": contract["outputs"],
                "prerequisite_output_labels": [],
                "external_prerequisites": evidence_bindings,
                "suite": contract["suite"],
                "training_regime": regime,
            }
        },
    }
    if dry_run:
        return {"dry_run": True, "plan": plan}
    if micro_run_root.exists():
        raise ValueError(f"Unique micro run root already exists: {micro_run_root}")
    micro_run_root.mkdir(parents=True, exist_ok=False)
    (micro_run_root / "stages").mkdir(parents=False, exist_ok=False)
    plan_path, _ = _plan_paths(micro_run_root)
    plan_sha256 = atomic_json(plan_path, plan)
    result = authorize_stage(micro_run_root, stage=stage)
    return {"dry_run": False, "plan": str(plan_path), "plan_sha256": plan_sha256, "first_stage": result}


def _default_run_name(prefix: str, commit: str) -> str:
    nonce = uuid.uuid4().hex[:8]
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{commit[:8]}-{nonce}"


def _require_run_name(name: str) -> str:
    if RUN_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError("run name must contain only lowercase letters, digits, dot, underscore, or dash")
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize immutable, SHA-bound Inspire R3 stages; this tool never launches a process"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    technical = subparsers.add_parser("init-technical")
    technical.add_argument("--repo", type=Path, required=True)
    technical.add_argument("--python", type=Path, required=True)
    technical.add_argument("--model-root", type=Path, required=True)
    technical.add_argument("--train", type=Path, required=True)
    technical.add_argument("--train-sha256", required=True)
    technical.add_argument("--dev", type=Path, required=True)
    technical.add_argument("--dev-sha256", required=True)
    technical.add_argument("--runs-root", type=Path, required=True)
    technical.add_argument("--run-name")
    technical.add_argument("--preflight", type=Path, required=True)
    technical.add_argument("--expected-commit", required=True)
    technical.add_argument("--through", choices=TECHNICAL_ORDER, default="DL-S")
    technical.add_argument("--dry-run", action="store_true")

    advance = subparsers.add_parser("authorize-next")
    advance.add_argument("--run-root", type=Path, required=True)
    advance.add_argument("--stage")
    advance.add_argument("--dry-run", action="store_true")

    teacher = subparsers.add_parser("init-teacher-preparation")
    teacher.add_argument("--technical-run-root", type=Path, required=True)
    teacher.add_argument("--runs-root", type=Path, required=True)
    teacher.add_argument("--run-name")
    teacher.add_argument("--preregistration", type=Path, required=True)
    teacher.add_argument("--set8-train", type=Path, required=True)
    teacher.add_argument("--transition16-train", type=Path, required=True)
    teacher.add_argument("--transition16-gate", type=Path, required=True)
    teacher.add_argument("--transition16-raw-sidecar", type=Path, required=True)
    teacher.add_argument("--set8-cache", type=Path, required=True)
    teacher.add_argument("--transition16-cache", type=Path, required=True)
    teacher.add_argument("--font", type=Path, required=True)
    teacher.add_argument("--dry-run", action="store_true")

    micro = subparsers.add_parser("init-micro")
    micro.add_argument("--technical-run-root", type=Path, required=True)
    micro.add_argument("--runs-root", type=Path, required=True)
    micro.add_argument("--run-name")
    micro.add_argument("--command-contract", type=Path, required=True)
    micro.add_argument("--teacher-preparation-run-root", type=Path)
    micro.add_argument("--teacher-t0", type=Path)
    micro.add_argument("--teacher-t0-sha256")
    micro.add_argument("--teacher-calibration", type=Path)
    micro.add_argument("--teacher-calibration-sha256")
    micro.add_argument("--teacher-calibration-report", type=Path)
    micro.add_argument("--teacher-calibration-report-sha256")
    micro.add_argument("--tc0-validation", type=Path)
    micro.add_argument("--tc0-validation-sha256")
    micro.add_argument("--tf0-validation", type=Path)
    micro.add_argument("--tf0-validation-sha256")
    micro.add_argument("--set8-parent-run-root", type=Path)
    micro.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "init-technical":
            if COMMIT_PATTERN.fullmatch(args.expected_commit) is None:
                raise ValueError("--expected-commit must be a lowercase full Git commit")
            commit = git(args.repo.resolve(), "rev-parse", "HEAD")
            if commit != args.expected_commit:
                raise ValueError(f"Commit mismatch: expected {args.expected_commit}, got {commit}")
            run_name = _require_run_name(args.run_name or _default_run_name("r3-technical", commit))
            result = initialize_technical_dag(
                repo=args.repo.resolve(),
                python=require_absolute_executable(args.python, "--python"),
                model_root=args.model_root.resolve(),
                train=args.train.resolve(),
                train_sha256=args.train_sha256,
                dev=args.dev.resolve(),
                dev_sha256=args.dev_sha256,
                run_root=args.runs_root.resolve() / run_name,
                preflight=args.preflight.resolve(),
                expected_commit=args.expected_commit,
                through=args.through,
                dry_run=args.dry_run,
            )
        elif args.action == "authorize-next":
            result = authorize_stage(args.run_root.resolve(), stage=args.stage, dry_run=args.dry_run)
        elif args.action == "init-teacher-preparation":
            technical, _, _ = _load_verified_plan(args.technical_run_root.resolve())
            commit = str(technical["expected_commit"])
            run_name = _require_run_name(args.run_name or _default_run_name("r3-teacher-preparation", commit))
            result = initialize_teacher_preparation_dag(
                technical_run_root=args.technical_run_root.resolve(),
                teacher_run_root=args.runs_root.resolve() / run_name,
                preregistration=args.preregistration.resolve(),
                set8_train=args.set8_train.resolve(),
                transition16_train=args.transition16_train.resolve(),
                transition16_gate=args.transition16_gate.resolve(),
                transition16_raw_sidecar=args.transition16_raw_sidecar.resolve(),
                set8_cache=args.set8_cache.resolve(),
                transition16_cache=args.transition16_cache.resolve(),
                font=args.font.resolve(),
                dry_run=args.dry_run,
            )
        else:
            command = _load_micro_command_contract(args.command_contract.resolve())
            technical, _, _ = _load_verified_plan(args.technical_run_root.resolve())
            commit = str(technical["expected_commit"])
            run_name = _require_run_name(
                args.run_name or _default_run_name(f"r3-{command['suite']}-{command['training_regime']}", commit)
            )
            result = initialize_micro_extension(
                technical_run_root=args.technical_run_root.resolve(),
                micro_run_root=args.runs_root.resolve() / run_name,
                command_contract_path=args.command_contract.resolve(),
                teacher_preparation_run_root=(
                    None if args.teacher_preparation_run_root is None else args.teacher_preparation_run_root.resolve()
                ),
                teacher_t0_path=None if args.teacher_t0 is None else args.teacher_t0.resolve(),
                teacher_t0_sha256=args.teacher_t0_sha256,
                teacher_calibration_path=(
                    None if args.teacher_calibration is None else args.teacher_calibration.resolve()
                ),
                teacher_calibration_sha256=args.teacher_calibration_sha256,
                teacher_calibration_report_path=(
                    None if args.teacher_calibration_report is None else args.teacher_calibration_report.resolve()
                ),
                teacher_calibration_report_sha256=args.teacher_calibration_report_sha256,
                tc0_validation_path=(None if args.tc0_validation is None else args.tc0_validation.resolve()),
                tc0_validation_sha256=args.tc0_validation_sha256,
                tf0_validation_path=(None if args.tf0_validation is None else args.tf0_validation.resolve()),
                tf0_validation_sha256=args.tf0_validation_sha256,
                set8_parent_run_root=(
                    None if args.set8_parent_run_root is None else args.set8_parent_run_root.resolve()
                ),
                dry_run=args.dry_run,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
