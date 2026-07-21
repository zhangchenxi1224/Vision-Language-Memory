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
from qwen_history_baseline_contract import (
    DEVICES,
    PLAN_PROTOCOL,
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
SLUGS = {
    "BH0": "00-bh0-contract",
    "BH1": "01-bh1-micro",
    "BH2": "02-bh2-formal-dev",
    "BH3": "03-bh3-formal-test",
}
LAUNCHER_STAGES = {
    "BH0": "qwen-history-bh0",
    "BH1": "qwen-history-bh1",
    "BH2": "qwen-history-bh2",
    "BH3": "qwen-history-bh3",
}


def _command(*parts: str | Path) -> list[str]:
    command = [str(part) for part in parts]
    validate_scientific_command(command)
    return command


def _prediction_outputs(results: Path, dataset: str, replica: str) -> tuple[Path, Path]:
    prediction = results / dataset / f"replica-{replica.lower()}" / "predictions.jsonl"
    return prediction, prediction.with_suffix(prediction.suffix + ".report.json")


def _expected_query_states(episodes: Path, *, probe_role: str, limit: int | None) -> int:
    if probe_role not in {"all", "delayed"}:
        raise ValueError(f"Unsupported probe role: {probe_role}")
    records: list[Mapping[str, Any]] = []
    with episodes.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{episodes}:{line_number} must contain a JSON object")
            records.append(value)
    if limit is not None:
        if limit <= 0:
            raise ValueError("Evaluator episode limit must be positive")
        records = records[:limit]
    queries = 0
    for episode in records:
        turns = episode.get("turns")
        if not isinstance(turns, list):
            raise ValueError(f"Episode {episode.get('episode_id')!r} has no turns list")
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise ValueError("Episode turn must be an object")
            turn_type = turn.get("type")
            if turn_type == "mixed" and probe_role == "all":
                queries += 1
            elif turn_type == "query":
                queries += 1
            elif turn_type != "event":
                raise ValueError(f"Unsupported synthetic turn type {turn_type!r}")
    if queries <= 0:
        raise ValueError("Baseline dataset has no query states after the probe-role filter")
    return queries


def _evaluator_command(
    *,
    python: Path,
    repo: Path,
    episodes: Path,
    reader: Path,
    output: Path,
    replica: str,
    device: str,
    conditions: tuple[str, ...],
    probe_role: str,
    limit: int | None = None,
    method: str = "qwen_full_event_history",
    input_mode: str = "blank_image",
    micro_sensitivity: bool = False,
) -> list[str]:
    command = [
        str(python),
        str(repo / "scripts" / "eval" / "qwen_text_baselines.py"),
        "--episodes",
        str(episodes),
        "--format",
        "synthetic",
        "--reader",
        str(reader),
        "--output",
        str(output),
        "--method",
        method,
        "--probe-role",
        probe_role,
        "--choice-view-family",
        "reverse-cyclic4",
        "--input-mode",
        input_mode,
        "--replica-id",
        replica,
        "--device",
        device,
        "--strict-determinism",
        "--conditions",
        *conditions,
    ]
    if micro_sensitivity:
        command.append("--micro-sensitivity")
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
    output: Path,
    text_only_predictions: Path | None = None,
    text_only_report: Path | None = None,
) -> list[str]:
    command = _command(
        python,
        repo / "scripts" / "eval" / "score_qwen_history_baseline.py",
        "--predictions",
        predictions_a,
        "--prediction-report",
        report_a,
        "--replica-b-predictions",
        predictions_b,
        "--replica-b-report",
        report_b,
        "--suite",
        suite,
        "--bootstrap-iterations",
        "10000",
        "--bootstrap-seed",
        "2026",
        "--output",
        output,
    )
    if (text_only_predictions is None) != (text_only_report is None):
        raise ValueError("Text-only predictions and report must be supplied together")
    if text_only_predictions is not None and text_only_report is not None:
        command.extend(
            (
                "--text-only-predictions",
                str(text_only_predictions),
                "--text-only-report",
                str(text_only_report),
            )
        )
    validate_scientific_command(command)
    return command


def _dataset_commands(
    *,
    python: Path,
    repo: Path,
    reader: Path,
    episodes: Path,
    results: Path,
    dataset: str,
    score_suite: str,
    conditions: tuple[str, ...],
    probe_role: str,
    limit: int | None = None,
    expected_inventory: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    query_states = _expected_query_states(episodes, probe_role=probe_role, limit=limit)
    expected_records = query_states * len(conditions) * 4
    require_json_values(
        expected_inventory,
        {
            "query_states": query_states,
            "conditions": len(conditions),
            "records_per_condition": query_states * 4,
            "prediction_records": expected_records,
            "probe_role": probe_role,
        },
        f"{dataset} prospective inventory",
    )
    pair: list[dict[str, Any]] = []
    replica_paths: dict[str, tuple[Path, Path]] = {}
    for replica, device in DEVICES.items():
        prediction, report = _prediction_outputs(results, dataset, replica)
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
                    replica=replica,
                    device=device,
                    conditions=conditions,
                    probe_role=probe_role,
                    limit=limit,
                ),
            }
        )
    validate_replica_pair(pair)
    score = results / dataset / "score.json"
    score_command = _score_command(
        python=python,
        repo=repo,
        predictions_a=replica_paths["A"][0],
        report_a=replica_paths["A"][1],
        predictions_b=replica_paths["B"][0],
        report_b=replica_paths["B"][1],
        suite=score_suite,
        output=score,
    )
    outputs: list[dict[str, Any]] = []
    for replica in DEVICES:
        prediction, report = replica_paths[replica]
        outputs.extend(
            (
                {
                    "label": f"{dataset}:replica-{replica.lower()}:predictions",
                    "path": str(prediction),
                    "required_values": {},
                    "validator": "prediction_jsonl",
                    "expected_records": expected_records,
                    "expected_method": "qwen_full_event_history",
                    "expected_input_mode": "blank_image",
                    "expected_micro_sensitivity": False,
                    "expected_conditions": list(conditions),
                    "expected_probe_role": probe_role,
                },
                {
                    "label": f"{dataset}:replica-{replica.lower()}:prediction-report",
                    "path": str(report),
                    "required_values": {
                        "status": "complete",
                        "method": "qwen_full_event_history",
                        "input_mode": "blank_image",
                        "micro_sensitivity": False,
                        "query_states": query_states,
                        "prediction_records": expected_records,
                        "conditions": list(conditions),
                        "probe_role": probe_role,
                        "choice_view_family": "reverse-cyclic4",
                    },
                },
            )
        )
    outputs.append(
        {
            "label": f"{dataset}:score",
            "path": str(score),
            "required_values": {
                "schema": "vlm.qwen-history-baseline-score.v1",
                "method": "qwen_full_event_history",
                "passed": True,
            },
            "validator": "baseline_score",
            "suite": score_suite,
        }
    )
    return pair, score_command, outputs


def _text_only_sensitivity_commands(
    *,
    python: Path,
    repo: Path,
    reader: Path,
    episodes: Path,
    results: Path,
    dataset: str,
    suite: str,
    conditions: tuple[str, ...],
    probe_role: str,
    blank_predictions_a: Path,
    blank_report_a: Path,
    blank_predictions_b: Path,
    blank_report_b: Path,
    expected_inventory: Mapping[str, Any],
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    query_states = _expected_query_states(episodes, probe_role=probe_role, limit=None)
    expected_records = query_states * len(conditions) * 4
    require_json_values(
        expected_inventory,
        {
            "query_states": query_states,
            "conditions": len(conditions),
            "records_per_condition": query_states * 4,
            "prediction_records": expected_records,
            "probe_role": probe_role,
        },
        f"{dataset} text-only prospective inventory",
    )
    prediction, report = _prediction_outputs(results, f"{dataset}-text-only-sensitivity", "A")
    evaluator = _evaluator_command(
        python=python,
        repo=repo,
        episodes=episodes,
        reader=reader,
        output=prediction,
        replica="A",
        device="cuda:0",
        conditions=conditions,
        probe_role=probe_role,
        method="qwen_full_event_history_text_only",
        input_mode="text_only",
        micro_sensitivity=True,
    )
    score = results / f"{dataset}-text-only-sensitivity" / "score.json"
    score_command = _score_command(
        python=python,
        repo=repo,
        predictions_a=blank_predictions_a,
        report_a=blank_report_a,
        predictions_b=blank_predictions_b,
        report_b=blank_report_b,
        suite=suite,
        output=score,
        text_only_predictions=prediction,
        text_only_report=report,
    )
    outputs = [
        {
            "label": f"{dataset}:text-only-sensitivity:predictions",
            "path": str(prediction),
            "required_values": {},
            "validator": "prediction_jsonl",
            "role": "micro_sensitivity_not_formal_baseline",
            "expected_records": expected_records,
            "expected_method": "qwen_full_event_history_text_only",
            "expected_input_mode": "text_only",
            "expected_micro_sensitivity": True,
            "expected_conditions": list(conditions),
            "expected_probe_role": probe_role,
        },
        {
            "label": f"{dataset}:text-only-sensitivity:prediction-report",
            "path": str(report),
            "required_values": {
                "status": "complete",
                "method": "qwen_full_event_history_text_only",
                "input_mode": "text_only",
                "micro_sensitivity": True,
                "query_states": query_states,
                "prediction_records": expected_records,
                "conditions": list(conditions),
                "probe_role": probe_role,
                "choice_view_family": "reverse-cyclic4",
            },
            "role": "micro_sensitivity_not_formal_baseline",
        },
        {
            "label": f"{dataset}:text-only-sensitivity:score",
            "path": str(score),
            "required_values": {
                "schema": "vlm.qwen-history-baseline-score.v1",
                "method": "qwen_full_event_history",
                "passed": True,
            },
            "validator": "baseline_sensitivity_score",
            "suite": suite,
            "role": "micro_sensitivity_not_formal_baseline",
        },
    ]
    return [evaluator, score_command], outputs


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
    preflight, preflight_sha256 = verify_formal_preflight(preflight_path, expected_commit=expected_commit)
    reader_snapshot = verified_reader_snapshot(
        repo=repo,
        model_root=model_root,
        preflight=preflight,
        amendment=amendment,
    )
    data_bindings = bind_data_files(amendment, data_paths)
    reported_runs_root = preflight.get("paths", {}).get("VLM_RUN_ROOT", {}).get("value")
    if reported_runs_root is not None and not is_within(run_root, Path(str(reported_runs_root))):
        raise ValueError("Baseline run root must remain inside formal-preflight VLM_RUN_ROOT")
    if run_root.exists():
        raise ValueError(f"Unique baseline run root already exists: {run_root}")

    reader_path = Path(reader_snapshot["model_dir"])
    results = run_root / "results"
    stage_fragments: dict[str, dict[str, Any]] = {}

    bh0_pair, bh0_score, bh0_outputs = _dataset_commands(
        python=python,
        repo=repo,
        reader=reader_path,
        episodes=Path(data_bindings["set8_gate_sha256"]["path"]),
        results=results / "BH0",
        dataset="set8-smoke",
        score_suite="formal",
        conditions=("standard",),
        probe_role="all",
        limit=1,
        expected_inventory=amendment["expected_inventory"]["bh0_set8_smoke"],
    )
    stage_fragments["BH0"] = {
        "parallel_groups": [bh0_pair],
        "serial_commands": [bh0_score],
        "outputs": bh0_outputs,
        "static_input_keys": ["set8_gate_sha256"],
    }

    bh1_parallel: list[list[dict[str, Any]]] = []
    bh1_serial: list[list[str]] = []
    bh1_outputs: list[dict[str, Any]] = []
    for dataset, data_key, suite, probe_role, inventory_key in (
        ("set8", "set8_gate_sha256", "set8", "all", "set8"),
        ("transition16", "transition16_gate_sha256", "transition16", "delayed", "transition16"),
    ):
        pair, score, outputs = _dataset_commands(
            python=python,
            repo=repo,
            reader=reader_path,
            episodes=Path(data_bindings[data_key]["path"]),
            results=results / "BH1",
            dataset=dataset,
            score_suite=suite,
            conditions=("standard", "reset", "shuffle", "state_swap"),
            probe_role=probe_role,
            expected_inventory=amendment["expected_inventory"][inventory_key],
        )
        bh1_parallel.append(pair)
        bh1_serial.append(score)
        bh1_outputs.extend(outputs)
        blank_a, blank_a_report = _prediction_outputs(results / "BH1", dataset, "A")
        blank_b, blank_b_report = _prediction_outputs(results / "BH1", dataset, "B")
        sensitivity_commands, sensitivity_outputs = _text_only_sensitivity_commands(
            python=python,
            repo=repo,
            reader=reader_path,
            episodes=Path(data_bindings[data_key]["path"]),
            results=results / "BH1",
            dataset=dataset,
            suite=suite,
            conditions=("standard", "reset", "shuffle", "state_swap"),
            probe_role=probe_role,
            blank_predictions_a=blank_a,
            blank_report_a=blank_a_report,
            blank_predictions_b=blank_b,
            blank_report_b=blank_b_report,
            expected_inventory=amendment["expected_inventory"][inventory_key],
        )
        bh1_serial.extend(sensitivity_commands)
        bh1_outputs.extend(sensitivity_outputs)
    stage_fragments["BH1"] = {
        "parallel_groups": bh1_parallel,
        "serial_commands": bh1_serial,
        "outputs": bh1_outputs,
        "static_input_keys": ["set8_gate_sha256", "transition16_gate_sha256"],
    }

    bh2_pair, bh2_score, bh2_outputs = _dataset_commands(
        python=python,
        repo=repo,
        reader=reader_path,
        episodes=Path(data_bindings["formal_dev_sha256"]["path"]),
        results=results / "BH2",
        dataset="formal-dev",
        score_suite="formal",
        conditions=("standard", "reset", "shuffle", "state_swap"),
        probe_role="all",
        expected_inventory=amendment["expected_inventory"]["formal_dev"],
    )
    stage_fragments["BH2"] = {
        "parallel_groups": [bh2_pair],
        "serial_commands": [bh2_score],
        "outputs": bh2_outputs,
        "static_input_keys": ["formal_manifest_sha256", "formal_dev_sha256"],
    }

    bh3_parallel: list[list[dict[str, Any]]] = []
    bh3_serial: list[list[str]] = []
    bh3_outputs: list[dict[str, Any]] = []
    for dataset, data_key, inventory_key in (
        ("formal-test-id", "formal_test_id_sha256", "formal_test_id"),
        ("formal-test-ood", "formal_test_ood_sha256", "formal_test_ood"),
    ):
        pair, score, outputs = _dataset_commands(
            python=python,
            repo=repo,
            reader=reader_path,
            episodes=Path(data_bindings[data_key]["path"]),
            results=results / "BH3",
            dataset=dataset,
            score_suite="formal",
            conditions=("standard", "reset", "shuffle", "state_swap"),
            probe_role="all",
            expected_inventory=amendment["expected_inventory"][inventory_key],
        )
        bh3_parallel.append(pair)
        bh3_serial.append(score)
        bh3_outputs.extend(outputs)
    stage_fragments["BH3"] = {
        "parallel_groups": bh3_parallel,
        "serial_commands": bh3_serial,
        "outputs": bh3_outputs,
        "static_input_keys": [
            "formal_manifest_sha256",
            "formal_test_id_sha256",
            "formal_test_ood_sha256",
        ],
    }

    stages: dict[str, Any] = {}
    for index, stage in enumerate(STAGES):
        slug = SLUGS[stage]
        fragment = stage_fragments[stage]
        stages[stage] = {
            "index": index,
            "slug": slug,
            "launcher_stage": LAUNCHER_STAGES[stage],
            "dependency": None if index == 0 else STAGES[index - 1],
            "run_dir": str(run_root / "stages" / slug),
            "evidence_path": str(run_root / "evidence" / f"{slug}.json"),
            **fragment,
        }

    return {
        "schema_version": 1,
        "protocol": PLAN_PROTOCOL,
        "kind": "qwen-full-history-baseline",
        "execution_backend": "inspire-notebook-background",
        "submission_backend": "scripts/inspire/launch_background.py",
        "external_scheduler_submission": False,
        "expected_commit": expected_commit,
        "repo": str(repo),
        "python": str(python),
        "model_root": str(model_root),
        "run_root": str(run_root),
        "formal_preflight": {"path": str(preflight_path.resolve()), "sha256": preflight_sha256},
        "amendment": {"path": str(amendment_path.resolve()), "sha256": amendment_sha256},
        "qwen_reader_snapshot": reader_snapshot,
        "dreamlite_snapshot_bound": False,
        "dreamlite_loaded": False,
        "training_performed": False,
        "data_bindings": data_bindings,
        "strict_order": list(STAGES),
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
            "kind": "qwen-full-history-baseline",
            "strict_order": list(STAGES),
        },
        "baseline DAG plan",
    )
    return plan, plan_path, digest


def _verify_completed_stage(plan: Mapping[str, Any], stage: str) -> list[dict[str, Any]]:
    definition = plan["stages"][stage]
    run_dir = Path(definition["run_dir"])
    run_root = Path(plan["run_root"])
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
            "parallel_groups": definition["parallel_groups"],
            "serial_commands": definition["serial_commands"],
            "outputs": definition["outputs"],
        },
        f"baseline predecessor {stage} stage specification",
    )
    terminal_path = run_dir / "terminal.json"
    if not terminal_path.is_file():
        raise ValueError(f"Baseline predecessor {stage} has no terminal.json")
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
        f"baseline predecessor {stage} terminal",
    )
    for log_name, digest_field in (("stdout.log", "stdout_sha256"), ("stderr.log", "stderr_sha256")):
        log_path = run_dir / log_name
        digest = str(terminal.get(digest_field, ""))
        if SHA256_PATTERN.fullmatch(digest) is None or not log_path.is_file() or sha256_file(log_path) != digest:
            raise ValueError(f"Baseline predecessor {stage} terminal does not bind {log_name}")
    worker_input = run_dir / "worker_input.json"
    configuration_sha256 = str(terminal.get("configuration_sha256", ""))
    if (
        SHA256_PATTERN.fullmatch(configuration_sha256) is None
        or not worker_input.is_file()
        or sha256_file(worker_input) != configuration_sha256
    ):
        raise ValueError(f"Baseline predecessor {stage} terminal does not bind worker_input.json")
    worker = load_json_object(worker_input)
    expected_runner = [
        plan["python"],
        str(Path(plan["repo"]) / "scripts" / "inspire" / "run_qwen_history_baseline_stage.py"),
        "--spec",
        str(spec_path),
        "--spec-sha256",
        spec_sha256,
    ]
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
        f"baseline predecessor {stage} worker input",
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
            "dreamlite_loaded": False,
            "training_performed": False,
            "plan_sha256": plan_sha256,
            "stage_spec_sha256": spec_sha256,
            "worker_input_path": str(worker_input),
            "configuration_sha256": configuration_sha256,
            "formal_preflight_sha256": plan["formal_preflight"]["sha256"],
            "amendment_sha256": plan["amendment"]["sha256"],
        },
        f"baseline predecessor {stage} evidence",
    )
    if evidence.get("qwen_reader_snapshot") != plan["qwen_reader_snapshot"]:
        raise ValueError(f"Baseline predecessor {stage} Qwen snapshot binding drifted")
    if evidence.get("prerequisites") != spec.get("prerequisites"):
        raise ValueError(f"Baseline predecessor {stage} prerequisite evidence drifted")
    expected_parallel = [
        {
            "index": index,
            "replicas": [
                {"replica": entry["replica"], "device": entry["device"], "exit_code": 0}
                for entry in pair
            ],
        }
        for index, pair in enumerate(definition["parallel_groups"])
    ]
    if evidence.get("parallel_groups") != expected_parallel:
        raise ValueError(f"Baseline predecessor {stage} does not prove every A/B process succeeded")
    expected_serial = [
        {"index": index, "exit_code": 0}
        for index, _ in enumerate(definition["serial_commands"])
    ]
    if evidence.get("serial_commands") != expected_serial:
        raise ValueError(f"Baseline predecessor {stage} does not prove every serial command succeeded")
    actual_outputs = evidence.get("outputs")
    if not isinstance(actual_outputs, list) or len(actual_outputs) != len(definition["outputs"]):
        raise ValueError(f"Baseline predecessor {stage} evidence has incomplete outputs")
    actual_by_label = {str(output.get("label")): output for output in actual_outputs if isinstance(output, Mapping)}
    planned_by_label = {str(output["label"]): output for output in definition["outputs"]}
    if set(actual_by_label) != set(planned_by_label):
        raise ValueError(f"Baseline predecessor {stage} evidence output labels drifted")
    score_bindings: list[dict[str, Any]] = []
    for output in definition["outputs"]:
        verify_declared_output(output)
        actual = actual_by_label[str(output["label"])]
        path = Path(output["path"])
        if actual.get("path") != str(path) or actual.get("sha256") != sha256_file(path):
            raise ValueError(f"Baseline predecessor {stage} output SHA evidence drifted for {output['label']}")
        if output.get("validator") in {"baseline_score", "baseline_sensitivity_score"}:
            score_bindings.append(
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
            "required_values": {"protocol": STAGE_EVIDENCE_PROTOCOL, "stage": stage, "passed": True},
        },
        *score_bindings,
    ]


def _next_stage(plan: Mapping[str, Any], run_root: Path) -> str:
    for stage in STAGES:
        if not (run_root / "authorizations" / f"{plan['stages'][stage]['slug']}.json").exists():
            return stage
    raise ValueError("Every baseline stage is already authorized")


def authorize_stage(run_root: Path, *, stage: str | None = None, dry_run: bool = False) -> dict[str, Any]:
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
        raise ValueError("Formal preflight drifted after baseline plan materialization")
    verify_bound_artifact({"label": "amendment", **plan["amendment"], "required_values": {}})

    next_stage = _next_stage(plan, run_root)
    stage = next_stage if stage is None else stage
    if stage != next_stage:
        raise ValueError(f"Only next fail-stop baseline stage {next_stage} may be authorized")
    definition = plan["stages"][stage]
    run_dir = Path(definition["run_dir"])
    if run_dir.exists():
        raise ValueError(f"Unique baseline stage directory already exists: {run_dir}")

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
        "label": "prospective-amendment",
        "path": plan["amendment"]["path"],
        "sha256": plan["amendment"]["sha256"],
        "required_values": {"schema": "vision_memory.r3-qwen-full-history-baseline-amendment.v1"},
    }
    verify_bound_artifact(amendment_binding)
    prerequisites.append(amendment_binding)

    spec_path = run_root / "authorizations" / f"{definition['slug']}.json"
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
        "expected_commit": plan["expected_commit"],
        "formal_preflight": plan["formal_preflight"],
        "amendment": plan["amendment"],
        "prerequisites": prerequisites,
        "parallel_groups": definition["parallel_groups"],
        "serial_commands": definition["serial_commands"],
        "outputs": definition["outputs"],
        "evidence_path": definition["evidence_path"],
        "immutable_after_materialization": True,
    }
    if dry_run:
        return {"dry_run": True, "stage_spec": spec, "launch": None}
    if spec_path.exists() or spec_path.with_suffix(spec_path.suffix + ".sha256").exists():
        raise ValueError(f"Baseline stage authorization already exists: {spec_path}")
    spec_sha256 = atomic_json(spec_path, spec)
    runner = repo / "scripts" / "inspire" / "run_qwen_history_baseline_stage.py"
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
    return f"qwen-full-history-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{commit[:8]}-{uuid.uuid4().hex[:8]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize the immutable Qwen full-event-history baseline DAG; never launch a remote process"
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
        default=ROOT / "configs" / "experiments" / "r3_qwen_full_history_baseline_amendment.json",
    )
    init.add_argument("--set8-gate", type=Path, required=True)
    init.add_argument("--transition16-gate", type=Path, required=True)
    init.add_argument("--formal-manifest", type=Path, required=True)
    init.add_argument("--formal-dev", type=Path, required=True)
    init.add_argument("--formal-test-id", type=Path, required=True)
    init.add_argument("--formal-test-ood", type=Path, required=True)
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
                data_paths={
                    "set8_gate_sha256": args.set8_gate,
                    "transition16_gate_sha256": args.transition16_gate,
                    "formal_manifest_sha256": args.formal_manifest,
                    "formal_dev_sha256": args.formal_dev,
                    "formal_test_id_sha256": args.formal_test_id,
                    "formal_test_ood_sha256": args.formal_test_ood,
                },
                dry_run=args.dry_run,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
