from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

from launch_background import STRICT_ENVIRONMENT, verify_preflight
from qwen_history_r4_contract import (
    ARM_METHODS,
    ARM_ORDER,
    COMPARISON_SCHEMA,
    DEVICES,
    PLAN_PROTOCOL,
    PREDICTION_REPORT_SCHEMA,
    SCORE_SCHEMA,
    STAGES,
    STAGE_EVIDENCE_PROTOCOL,
    STAGE_SPEC_PROTOCOL,
    bind_data_files,
    load_amendment,
    validate_replica_pair,
    validate_scientific_command,
    verified_reader_snapshot,
    verify_declared_output,
    verify_formal_preflight,
)
from r3_dag_contract import (
    COMMIT_PATTERN,
    LAUNCH_COMMAND_PROTOCOL,
    RUN_NAME_PATTERN,
    SHA256_PATTERN,
    atomic_json,
    git,
    is_within,
    load_json_object,
    require_absolute,
    require_absolute_executable,
    require_json_values,
    sha256_file,
    verify_bound_artifact,
    verify_clean_commit,
    verify_sha_sidecar,
)


ROOT = Path(__file__).resolve().parents[2]
KIND = "qwen-history-r4-three-arm-comparison"
SLUGS = {
    "BH0": "00-bh0-smoke4",
    "BH1": "01-bh1-transition32",
    "BH2": "02-bh2-formal-dev",
    "BH3": "03-bh3-formal-test",
}
LAUNCHER_STAGES = {stage: f"qwen-history-r4-{stage.lower()}" for stage in STAGES}
DATA_ARGUMENTS = {
    "lockbox_manifest": "lockbox-manifest",
    "smoke4": "smoke4",
    "transition32": "transition32",
    "formal_train": "formal-train",
    "formal_dev": "formal-dev",
    "formal_test_id": "formal-test-id",
    "formal_test_ood": "formal-test-ood",
}
TEST_DATA_ACCESS_AUDIT = {
    "test_file_sha256_bytes_read_during_materialization": True,
    "test_json_semantics_parsed_during_materialization": False,
    "test_predictions_or_metrics_accessed_during_materialization": False,
    "test_evaluation_executed_during_materialization": False,
}


def _prediction_outputs(
    results: Path, dataset: str, arm: str, replica: str
) -> tuple[Path, Path]:
    prediction = (
        results
        / dataset
        / arm
        / f"replica-{replica.lower()}"
        / "predictions.jsonl"
    )
    return prediction, prediction.with_suffix(prediction.suffix + ".report.json")


def _evaluator_command(
    *,
    python: Path,
    repo: Path,
    episodes: Path,
    reader: Path,
    output: Path,
    method: str,
    conditions: tuple[str, ...],
    probe_role: str,
    replica: str,
    device: str,
    seed: int,
    limit: int | None,
) -> list[str]:
    command = [
        str(python),
        str(repo / "scripts" / "eval" / "qwen_history_r4.py"),
        "--episodes",
        str(episodes),
        "--reader",
        str(reader),
        "--output",
        str(output),
        "--method",
        method,
        "--conditions",
        *conditions,
        "--probe-role",
        probe_role,
        "--choice-view-family",
        "reverse-cyclic4",
        "--seed",
        str(seed),
        "--replica-id",
        replica,
        "--device",
        device,
        "--strict-determinism",
    ]
    if limit is not None:
        command.extend(("--limit", str(limit)))
    validate_scientific_command(command)
    return command


def _score_command(
    *,
    python: Path,
    repo: Path,
    predictions_a: Path,
    report_a: Path,
    predictions_b: Path,
    report_b: Path,
    suite: str,
    method: str,
    output: Path,
    fail_on_gate: bool,
) -> list[str]:
    command = [
        str(python),
        str(repo / "scripts" / "eval" / "score_qwen_history_r4.py"),
        "--predictions",
        str(predictions_a),
        "--prediction-report",
        str(report_a),
        "--replica-b-predictions",
        str(predictions_b),
        "--replica-b-report",
        str(report_b),
        "--suite",
        suite,
        "--method",
        method,
        "--output",
        str(output),
        "--fail-on-gate" if fail_on_gate else "--no-fail-on-gate",
    ]
    validate_scientific_command(command)
    return command


def _comparison_command(
    *,
    python: Path,
    repo: Path,
    raw_predictions: Path,
    tagged_predictions: Path,
    last_effective_predictions: Path,
    suite: str,
    output: Path,
    iterations: int,
    seed: int,
) -> list[str]:
    command = [
        str(python),
        str(repo / "scripts" / "eval" / "compare_qwen_history_r4.py"),
        "--raw-predictions",
        str(raw_predictions),
        "--tagged-predictions",
        str(tagged_predictions),
        "--last-effective-predictions",
        str(last_effective_predictions),
        "--suite",
        suite,
        "--bootstrap-iterations",
        str(iterations),
        "--bootstrap-seed",
        str(seed),
        "--output",
        str(output),
    ]
    validate_scientific_command(command)
    return command


def _dataset_fragment(
    *,
    python: Path,
    repo: Path,
    reader: Path,
    episodes: Path,
    results: Path,
    dataset: str,
    inventory: Mapping[str, Any],
    score_suite: str,
    comparison_suite: str | None,
    seed: int,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    stage: str,
) -> tuple[list[dict[str, Any]], list[list[str]], list[dict[str, Any]]]:
    conditions = tuple(str(value) for value in inventory["conditions"])
    probe_role = str(inventory["probe_role"])
    query_states = int(inventory["query_states"])
    expected_records = int(inventory["prediction_records_per_arm"])
    limit_value = inventory.get("limit")
    limit = None if limit_value is None else int(limit_value)
    if expected_records != query_states * len(conditions) * 4:
        raise ValueError(f"R4 inventory arithmetic drifted for {dataset}")

    replica_pairs: list[dict[str, Any]] = []
    serial_commands: list[list[str]] = []
    outputs: list[dict[str, Any]] = []
    arm_predictions: dict[str, dict[str, tuple[Path, Path]]] = {}

    for arm in ARM_ORDER:
        method = ARM_METHODS[arm]
        pair: list[dict[str, Any]] = []
        replica_paths: dict[str, tuple[Path, Path]] = {}
        for replica, device in DEVICES.items():
            prediction, report = _prediction_outputs(results, dataset, arm, replica)
            replica_paths[replica] = (prediction, report)
            pair.append(
                {
                    "replica": replica,
                    "device": device,
                    "argv": _evaluator_command(
                        python=python,
                        repo=repo,
                        episodes=episodes,
                        reader=reader,
                        output=prediction,
                        method=method,
                        conditions=conditions,
                        probe_role=probe_role,
                        replica=replica,
                        device=device,
                        seed=seed,
                        limit=limit,
                    ),
                }
            )
        validate_replica_pair(pair, method=method)
        replica_pairs.append(
            {
                "dataset": dataset,
                "arm": arm,
                "method": method,
                "execution_mode": "sequential_a_then_b",
                "replicas": pair,
            }
        )
        arm_predictions[arm] = replica_paths

        score = results / dataset / arm / "score.json"
        fail_on_gate = arm == "last_effective" and stage in {"BH0", "BH1"}
        serial_commands.append(
            _score_command(
                python=python,
                repo=repo,
                predictions_a=replica_paths["A"][0],
                report_a=replica_paths["A"][1],
                predictions_b=replica_paths["B"][0],
                report_b=replica_paths["B"][1],
                suite=score_suite,
                method=method,
                output=score,
                fail_on_gate=fail_on_gate,
            )
        )

        for replica in DEVICES:
            prediction, report = replica_paths[replica]
            outputs.extend(
                (
                    {
                        "label": f"{dataset}:{arm}:replica-{replica.lower()}:predictions",
                        "path": str(prediction),
                        "required_values": {},
                        "validator": "prediction_jsonl",
                        "expected_records": expected_records,
                        "expected_method": method,
                        "expected_conditions": list(conditions),
                        "expected_probe_role": probe_role,
                    },
                    {
                        "label": f"{dataset}:{arm}:replica-{replica.lower()}:prediction-report",
                        "path": str(report),
                        "required_values": {
                            "schema_version": PREDICTION_REPORT_SCHEMA,
                            "status": "complete",
                            "method": method,
                            "input_mode": "blank_image",
                            "query_states": query_states,
                            "prediction_records": expected_records,
                            "conditions": list(conditions),
                            "probe_role": probe_role,
                            "choice_view_family": "reverse-cyclic4",
                            "replica_id": replica,
                        },
                    },
                )
            )

        if stage == "BH2" and arm == "last_effective":
            score_validator = "r4_bh2_last_effective_dev"
        else:
            score_validator = "r4_score"
        outputs.append(
            {
                "label": f"{dataset}:{arm}:score",
                "path": str(score),
                "required_values": {
                    "schema": SCORE_SCHEMA,
                    "method": method,
                    "suite": score_suite,
                    "passed": True,
                },
                "validator": score_validator,
                "suite": score_suite,
                "expected_method": method,
                "require_data_readability": arm == "last_effective"
                and stage in {"BH0", "BH1"},
                "expected_records": expected_records,
                "minimum_accuracy": 0.95,
                "performance_role": (
                    "blocking_data_readability"
                    if arm == "last_effective" and stage in {"BH0", "BH1", "BH2"}
                    else "descriptive_nonblocking"
                ),
            }
        )

    if comparison_suite is not None:
        comparison = results / dataset / "comparison.json"
        serial_commands.append(
            _comparison_command(
                python=python,
                repo=repo,
                raw_predictions=arm_predictions["raw"]["A"][0],
                tagged_predictions=arm_predictions["tagged"]["A"][0],
                last_effective_predictions=arm_predictions["last_effective"]["A"][0],
                suite=comparison_suite,
                output=comparison,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
        )
        outputs.append(
            {
                "label": f"{dataset}:comparison",
                "path": str(comparison),
                "required_values": {
                    "schema": COMPARISON_SCHEMA,
                    "suite": comparison_suite,
                    "passed": True,
                },
                "validator": "r4_comparison",
                "suite": comparison_suite,
            }
        )
    return replica_pairs, serial_commands, outputs


def build_plan(
    *,
    repo: Path,
    python: Path,
    model_root: Path,
    run_root: Path,
    preflight_path: Path,
    expected_commit: str,
    amendment_path: Path,
    data_paths: Mapping[str, Path],
) -> dict[str, Any]:
    repo = repo.resolve()
    python = require_absolute_executable(python, "--python")
    model_root = require_absolute(model_root, "--model-root")
    run_root = require_absolute(run_root, "--run-root")
    verify_clean_commit(repo, expected_commit)
    amendment, amendment_sha256 = load_amendment(amendment_path)
    preflight, preflight_sha256 = verify_formal_preflight(
        preflight_path, expected_commit=expected_commit
    )
    reader_snapshot = verified_reader_snapshot(
        repo=repo,
        model_root=model_root,
        preflight=preflight,
        amendment=amendment,
    )
    data_bindings = bind_data_files(amendment, data_paths)
    reported_runs_root = preflight.get("paths", {}).get("VLM_RUN_ROOT", {}).get("value")
    if reported_runs_root is not None and not is_within(run_root, Path(str(reported_runs_root))):
        raise ValueError("R4 run root must remain inside formal-preflight VLM_RUN_ROOT")
    if run_root.exists():
        raise ValueError(f"Unique R4 run root already exists: {run_root}")

    reader_path = Path(reader_snapshot["model_dir"])
    results = run_root / "results"
    seed = int(amendment["lockbox"]["seed"])
    bootstrap_iterations = int(amendment["execution"]["bootstrap_iterations"])
    bootstrap_seed = int(amendment["execution"]["bootstrap_seed"])
    stage_fragments: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        stage_entry = amendment["stages"][stage]
        replica_pairs: list[dict[str, Any]] = []
        serial_commands: list[list[str]] = []
        outputs: list[dict[str, Any]] = []
        for dataset in stage_entry["datasets"]:
            inventory = amendment["expected_inventory"][dataset]
            pairs, commands, declared = _dataset_fragment(
                python=python,
                repo=repo,
                reader=reader_path,
                episodes=Path(data_bindings[str(inventory["data_key"])]["path"]),
                results=results / stage,
                dataset=dataset,
                inventory=inventory,
                score_suite=str(stage_entry["score_suite"]),
                comparison_suite=stage_entry.get("comparison_suite"),
                seed=seed,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
                stage=stage,
            )
            replica_pairs.extend(pairs)
            serial_commands.extend(commands)
            outputs.extend(declared)
        static_keys = ["lockbox_manifest"] + [
            str(amendment["expected_inventory"][dataset]["data_key"])
            for dataset in stage_entry["datasets"]
        ]
        stage_fragments[stage] = {
            "replica_pairs": replica_pairs,
            "serial_commands": serial_commands,
            "outputs": outputs,
            "static_input_keys": list(dict.fromkeys(static_keys)),
            "score_suite": stage_entry["score_suite"],
            "comparison_suite": stage_entry.get("comparison_suite"),
            "last_effective_gate_required": stage_entry["last_effective_gate_required"],
        }

    stages: dict[str, Any] = {}
    for index, stage in enumerate(STAGES):
        slug = SLUGS[stage]
        stages[stage] = {
            "index": index,
            "slug": slug,
            "launcher_stage": LAUNCHER_STAGES[stage],
            "dependency": None if index == 0 else STAGES[index - 1],
            "run_dir": str(run_root / "stages" / slug),
            "evidence_path": str(run_root / "evidence" / f"{slug}.json"),
            **stage_fragments[stage],
        }

    return {
        "schema_version": 1,
        "protocol": PLAN_PROTOCOL,
        "kind": KIND,
        "execution_backend": "inspire-notebook-background",
        "submission_backend": "scripts/inspire/launch_background.py",
        "external_scheduler_submission": False,
        "expected_commit": expected_commit,
        "repo": str(repo),
        "python": str(python),
        "model_root": str(model_root),
        "run_root": str(run_root),
        "formal_preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": preflight_sha256,
        },
        "amendment": {
            "path": str(amendment_path.resolve()),
            "sha256": amendment_sha256,
            "schema": amendment["schema"],
        },
        "research_protocol": amendment["schema"],
        "research_role": amendment["research_role"]["name"],
        "qwen_reader_snapshot": reader_snapshot,
        "dreamlite_snapshot_bound": False,
        "dreamlite_loaded": False,
        "training_performed": False,
        **TEST_DATA_ACCESS_AUDIT,
        "data_bindings": data_bindings,
        "strict_order": list(STAGES),
        "arm_order": list(ARM_ORDER),
        "replica_order": ["A", "B"],
        "sequential_within_arm": True,
        "stages": stages,
    }


def _plan_paths(run_root: Path) -> tuple[Path, Path]:
    path = run_root / "dag_plan.json"
    return path, path.with_suffix(path.suffix + ".sha256")


def _load_plan(run_root: Path) -> tuple[dict[str, Any], Path, str]:
    plan_path, _ = _plan_paths(run_root)
    digest = verify_sha_sidecar(plan_path)
    plan = load_json_object(plan_path)
    require_json_values(
        plan,
        {
            "schema_version": 1,
            "protocol": PLAN_PROTOCOL,
            "kind": KIND,
            "strict_order": list(STAGES),
            "arm_order": list(ARM_ORDER),
            "replica_order": ["A", "B"],
            "sequential_within_arm": True,
        },
        "R4 DAG plan",
    )
    return plan, plan_path, digest


def _verify_completed_stage(plan: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    definition = plan["stages"][stage]
    run_root = Path(plan["run_root"])
    run_dir = Path(definition["run_dir"])
    plan_path, _ = _plan_paths(run_root)
    plan_sha256 = verify_sha_sidecar(plan_path)
    spec_path = run_root / "authorizations" / f"{definition['slug']}.json"
    spec_sha256 = verify_sha_sidecar(spec_path)
    spec = load_json_object(spec_path)
    require_json_values(
        spec,
        {
            "schema_version": 1,
            "protocol": STAGE_SPEC_PROTOCOL,
            "plan_path": str(plan_path),
            "plan_sha256": plan_sha256,
            "stage": stage,
            "launcher_stage": definition["launcher_stage"],
            "run_dir": definition["run_dir"],
            "expected_commit": plan["expected_commit"],
            "replica_pairs": definition["replica_pairs"],
            "serial_commands": definition["serial_commands"],
            "outputs": definition["outputs"],
        },
        f"R4 predecessor {stage} specification",
    )
    terminal_path = run_dir / "terminal.json"
    terminal = load_json_object(terminal_path)
    require_json_values(
        terminal,
        {
            "status": "succeeded",
            "passed": True,
            "exit_code": 0,
            "stage": definition["launcher_stage"],
            "expected_commit": plan["expected_commit"],
        },
        f"R4 predecessor {stage} terminal",
    )
    for log_name, digest_field in (("stdout.log", "stdout_sha256"), ("stderr.log", "stderr_sha256")):
        log_path = run_dir / log_name
        digest = str(terminal.get(digest_field, ""))
        if SHA256_PATTERN.fullmatch(digest) is None or sha256_file(log_path) != digest:
            raise ValueError(f"R4 predecessor {stage} terminal does not bind {log_name}")
    worker_input = run_dir / "worker_input.json"
    configuration_sha256 = str(terminal.get("configuration_sha256", ""))
    if (
        SHA256_PATTERN.fullmatch(configuration_sha256) is None
        or sha256_file(worker_input) != configuration_sha256
    ):
        raise ValueError(f"R4 predecessor {stage} terminal does not bind worker_input.json")
    expected_runner = [
        plan["python"],
        str(Path(plan["repo"]) / "scripts" / "inspire" / "run_qwen_history_r4_stage.py"),
        "--spec",
        str(spec_path),
        "--spec-sha256",
        spec_sha256,
    ]
    worker = load_json_object(worker_input)
    require_json_values(
        worker,
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
            "command": expected_runner,
        },
        f"R4 predecessor {stage} worker input",
    )
    evidence_path = Path(definition["evidence_path"])
    evidence_sha256 = verify_sha_sidecar(evidence_path)
    evidence = load_json_object(evidence_path)
    require_json_values(
        evidence,
        {
            "schema_version": 1,
            "protocol": STAGE_EVIDENCE_PROTOCOL,
            "stage": stage,
            "passed": True,
            "expected_commit": plan["expected_commit"],
            "plan_sha256": plan_sha256,
            "stage_spec_sha256": spec_sha256,
            "worker_input_path": str(worker_input),
            "configuration_sha256": configuration_sha256,
            "formal_preflight_sha256": plan["formal_preflight"]["sha256"],
            "amendment_sha256": plan["amendment"]["sha256"],
            "execution_mode": "sequential_within_arm",
            "dreamlite_loaded": False,
            "training_performed": False,
        },
        f"R4 predecessor {stage} evidence",
    )
    if evidence.get("qwen_reader_snapshot") != plan["qwen_reader_snapshot"]:
        raise ValueError(f"R4 predecessor {stage} Qwen snapshot binding drifted")
    if evidence.get("prerequisites") != spec.get("prerequisites"):
        raise ValueError(f"R4 predecessor {stage} prerequisite evidence drifted")
    expected_pairs = [
        {
            "index": index,
            "dataset": group["dataset"],
            "arm": group["arm"],
            "method": group["method"],
            "execution_mode": "sequential_a_then_b",
            "replicas": [
                {"replica": entry["replica"], "device": entry["device"], "exit_code": 0}
                for entry in group["replicas"]
            ],
        }
        for index, group in enumerate(definition["replica_pairs"])
    ]
    if evidence.get("replica_pairs") != expected_pairs:
        raise ValueError(f"R4 predecessor {stage} does not prove sequential A/B success")
    expected_serial = [
        {"index": index, "exit_code": 0}
        for index, _ in enumerate(definition["serial_commands"])
    ]
    if evidence.get("serial_commands") != expected_serial:
        raise ValueError(f"R4 predecessor {stage} does not prove every serial command succeeded")
    actual_outputs = evidence.get("outputs")
    if not isinstance(actual_outputs, list) or len(actual_outputs) != len(definition["outputs"]):
        raise ValueError(f"R4 predecessor {stage} evidence has incomplete outputs")
    actual_by_label = {
        str(output.get("label")): output
        for output in actual_outputs
        if isinstance(output, Mapping)
    }
    if set(actual_by_label) != {str(output["label"]) for output in definition["outputs"]}:
        raise ValueError(f"R4 predecessor {stage} evidence output labels drifted")
    scientific_bindings: list[dict[str, Any]] = []
    for output in definition["outputs"]:
        verify_declared_output(output)
        path = Path(output["path"])
        actual = actual_by_label[str(output["label"])]
        if actual.get("path") != str(path.resolve()) or actual.get("sha256") != sha256_file(path):
            raise ValueError(f"R4 predecessor {stage} output SHA drifted for {output['label']}")
        if output.get("validator") in {
            "r4_score",
            "r4_bh2_last_effective_dev",
            "r4_comparison",
        }:
            scientific_bindings.append(
                {
                    "label": f"{stage}:{output['label']}",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "required_values": output.get("required_values", {}),
                }
            )
    return [
        {
            "label": f"{stage}:terminal",
            "path": str(terminal_path),
            "sha256": sha256_file(terminal_path),
            "required_values": {"status": "succeeded", "passed": True, "exit_code": 0},
        },
        {
            "label": f"{stage}:evidence",
            "path": str(evidence_path),
            "sha256": evidence_sha256,
            "required_values": {
                "protocol": STAGE_EVIDENCE_PROTOCOL,
                "stage": stage,
                "passed": True,
            },
        },
        *scientific_bindings,
    ]


def _next_stage(plan: Mapping[str, Any], run_root: Path) -> str:
    for stage in STAGES:
        spec = run_root / "authorizations" / f"{plan['stages'][stage]['slug']}.json"
        if not spec.exists():
            return stage
    raise ValueError("Every R4 stage is already authorized")


def authorize_stage(
    run_root: Path, *, stage: str | None = None, dry_run: bool = False
) -> dict[str, Any]:
    run_root = run_root.resolve()
    plan, plan_path, plan_sha256 = _load_plan(run_root)
    repo = Path(plan["repo"])
    verify_clean_commit(repo, str(plan["expected_commit"]))
    _, preflight_sha256 = verify_preflight(
        Path(plan["formal_preflight"]["path"]),
        expected_commit=str(plan["expected_commit"]),
        infrastructure_stage=False,
    )
    if preflight_sha256 != plan["formal_preflight"]["sha256"]:
        raise ValueError("Formal preflight drifted after R4 plan materialization")
    verify_bound_artifact(
        {"label": "R4 amendment", **plan["amendment"], "required_values": {}}
    )

    next_stage = _next_stage(plan, run_root)
    selected = next_stage if stage is None else stage
    if selected != next_stage:
        raise ValueError(f"Only next fail-stop R4 stage {next_stage} may be authorized")
    definition = plan["stages"][selected]
    run_dir = Path(definition["run_dir"])
    if run_dir.exists():
        raise ValueError(f"Unique R4 stage directory already exists: {run_dir}")

    prerequisites: list[dict[str, Any]] = []
    dependency = definition["dependency"]
    if dependency is not None:
        prerequisites.extend(_verify_completed_stage(plan, str(dependency)))
    for key in definition["static_input_keys"]:
        binding = plan["data_bindings"][key]
        prerequisite = {
            "label": f"data:{key}",
            "path": binding["path"],
            "sha256": binding["sha256"],
            "required_values": {},
        }
        verify_bound_artifact(prerequisite)
        prerequisites.append(prerequisite)
    amendment_binding = {
        "label": "prospective-qwen-history-amendment",
        "path": plan["amendment"]["path"],
        "sha256": plan["amendment"]["sha256"],
        "required_values": {"schema": plan["amendment"]["schema"]},
    }
    verify_bound_artifact(amendment_binding)
    prerequisites.append(amendment_binding)

    spec_path = run_root / "authorizations" / f"{definition['slug']}.json"
    spec = {
        "schema_version": 1,
        "protocol": STAGE_SPEC_PROTOCOL,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha256,
        "stage": selected,
        "stage_index": definition["index"],
        "stage_slug": definition["slug"],
        "launcher_stage": definition["launcher_stage"],
        "run_root": str(run_root),
        "run_dir": str(run_dir),
        "repo": str(repo),
        "expected_commit": plan["expected_commit"],
        "formal_preflight": plan["formal_preflight"],
        "amendment": plan["amendment"],
        "prerequisites": prerequisites,
        "replica_pairs": definition["replica_pairs"],
        "serial_commands": definition["serial_commands"],
        "outputs": definition["outputs"],
        "evidence_path": definition["evidence_path"],
        "immutable_after_materialization": True,
    }
    if dry_run:
        return {"dry_run": True, "stage_spec": spec, "launch": None}
    if spec_path.exists() or spec_path.with_suffix(spec_path.suffix + ".sha256").exists():
        raise ValueError(f"R4 stage authorization already exists: {spec_path}")
    spec_sha256 = atomic_json(spec_path, spec)
    runner = repo / "scripts" / "inspire" / "run_qwen_history_r4_stage.py"
    runner_command = [
        plan["python"],
        str(runner),
        "--spec",
        str(spec_path),
        "--spec-sha256",
        spec_sha256,
    ]
    launch_argv = [
        plan["python"],
        str(repo / "scripts" / "inspire" / "launch_background.py"),
        "--repo",
        str(repo),
        "--run-root",
        str(run_root),
        "--run-dir",
        str(run_dir),
        "--stage",
        definition["launcher_stage"],
        "--expected-commit",
        plan["expected_commit"],
        "--preflight",
        plan["formal_preflight"]["path"],
        "--",
        *runner_command,
    ]
    launch = {
        "schema_version": 1,
        "protocol": LAUNCH_COMMAND_PROTOCOL,
        "stage": selected,
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


def initialize(
    *,
    repo: Path,
    python: Path,
    model_root: Path,
    runs_root: Path,
    run_name: str,
    preflight: Path,
    expected_commit: str,
    amendment: Path,
    data_paths: Mapping[str, Path],
    dry_run: bool = False,
) -> dict[str, Any]:
    runs_root = require_absolute(runs_root, "--runs-root")
    if RUN_NAME_PATTERN.fullmatch(run_name) is None:
        raise ValueError("run name must contain only lowercase letters, digits, dot, underscore, or dash")
    run_root = runs_root / run_name
    plan = build_plan(
        repo=repo,
        python=python,
        model_root=model_root,
        run_root=run_root,
        preflight_path=preflight,
        expected_commit=expected_commit,
        amendment_path=amendment,
        data_paths=data_paths,
    )
    if dry_run:
        return {"dry_run": True, "plan": plan}
    runs_root.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=False, exist_ok=False)
    (run_root / "stages").mkdir(parents=False, exist_ok=False)
    plan_path, _ = _plan_paths(run_root)
    plan_sha256 = atomic_json(plan_path, plan)
    first = authorize_stage(run_root, stage="BH0")
    return {
        "dry_run": False,
        "plan": str(plan_path),
        "plan_sha256": plan_sha256,
        "first_stage": first,
    }


def _default_run_name(commit: str) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    return f"qwen-history-r4-{timestamp}-{commit[:8]}-{uuid.uuid4().hex[:8]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the immutable R4 three-arm Qwen history DAG; never launch a process"
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    init = subparsers.add_parser("init")
    init.add_argument("--repo", type=Path, default=ROOT)
    init.add_argument("--python", type=Path, required=True)
    init.add_argument("--model-root", type=Path, required=True)
    init.add_argument("--runs-root", type=Path, required=True)
    init.add_argument("--run-name")
    init.add_argument("--preflight", type=Path, required=True)
    init.add_argument("--expected-commit", required=True)
    init.add_argument(
        "--amendment",
        type=Path,
        default=ROOT / "configs" / "experiments" / "r4_qwen_history_comparison_20260722.json",
    )
    for destination, option in DATA_ARGUMENTS.items():
        init.add_argument(f"--{option}", dest=destination, type=Path, required=True)
    init.add_argument("--dry-run", action="store_true")
    advance = subparsers.add_parser("authorize-next")
    advance.add_argument("--run-root", type=Path, required=True)
    advance.add_argument("--stage", choices=STAGES)
    advance.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.action == "authorize-next":
            result = authorize_stage(args.run_root.resolve(), stage=args.stage, dry_run=args.dry_run)
        else:
            if COMMIT_PATTERN.fullmatch(args.expected_commit) is None:
                raise ValueError("--expected-commit must be a lowercase full Git commit")
            commit = git(args.repo.resolve(), "rev-parse", "HEAD")
            if commit != args.expected_commit:
                raise ValueError(f"Commit mismatch: expected {args.expected_commit}, got {commit}")
            run_name = args.run_name or _default_run_name(commit)
            result = initialize(
                repo=args.repo.resolve(),
                python=args.python,
                model_root=args.model_root,
                runs_root=args.runs_root,
                run_name=run_name,
                preflight=args.preflight,
                expected_commit=args.expected_commit,
                amendment=args.amendment,
                data_paths={key: getattr(args, key) for key in DATA_ARGUMENTS},
                dry_run=args.dry_run,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
