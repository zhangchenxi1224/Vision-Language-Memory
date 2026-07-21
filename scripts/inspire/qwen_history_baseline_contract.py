from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from model_snapshot_manifest import verify_snapshot_binding, verify_snapshot_manifest
from r3_dag_contract import (
    COMMIT_PATTERN,
    SHA256_PATTERN,
    load_json_object,
    require_file_sha,
    require_json_values,
    sha256_file,
    verify_sha_sidecar,
)


AMENDMENT_SCHEMA = "vision_memory.r3-qwen-full-history-baseline-amendment.v1"
PLAN_PROTOCOL = "r3-inspire-qwen-history-baseline-dag.v1"
STAGE_SPEC_PROTOCOL = "r3-inspire-qwen-history-baseline-stage.v1"
STAGE_EVIDENCE_PROTOCOL = "r3-inspire-qwen-history-baseline-evidence.v1"
STAGES = ("BH0", "BH1", "BH2", "BH3")
DEVICES = {"A": "cuda:0", "B": "cuda:1"}
SCORE_SCHEMA = "vlm.qwen-history-baseline-score.v1"

FORBIDDEN_COMMAND_FRAGMENTS = (
    "dreamlite",
    "scripts/train/",
    "scripts\\train\\",
    "oracle_target",
    "full_history_reminder",
    "rag_top5",
    "teacher",
)


def load_amendment(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    amendment = load_json_object(path)
    require_json_values(
        amendment,
        {
            "schema": AMENDMENT_SCHEMA,
            "status": "prospective_before_any_qwen_full_history_baseline_gpu_prediction",
        },
        "Qwen history baseline amendment",
    )
    role = amendment.get("research_role")
    if not isinstance(role, Mapping):
        raise ValueError("Baseline amendment research_role must be an object")
    require_json_values(
        role,
        {
            "name": "qwen_full_event_history_blank_image",
            "training": False,
            "dreamlite_loaded": False,
        },
        "baseline research role",
    )
    reader = amendment.get("reader")
    if not isinstance(reader, Mapping):
        raise ValueError("Baseline amendment reader must be an object")
    require_json_values(
        reader,
        {
            "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
            "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
            "snapshot_key": "qwen_reader",
            "loss_mode": "listwise-choice",
            "choice_family": "reverse-cyclic4",
            "choice_family_sha256": "4cd725a443d8661dccccbff2d714876aee317c654ef2aa1bcb79aee307d64bbd",
            "parameters_frozen": True,
        },
        "baseline Reader contract",
    )
    blank = amendment.get("blank_image")
    if not isinstance(blank, Mapping):
        raise ValueError("Baseline amendment blank_image must be an object")
    require_json_values(
        blank,
        {
            "shape_chw": [3, 1024, 1024],
            "dtype": "float32",
            "rgb_value": 0.5,
            "constant_visual_input_bytes": 12_582_912,
        },
        "baseline blank image",
    )
    history = amendment.get("history_contract")
    if not isinstance(history, Mapping):
        raise ValueError("Baseline amendment history_contract must be an object")
    if history.get("silent_context_truncation_forbidden") is not True:
        raise ValueError("Baseline amendment must fail closed on context truncation")
    if amendment.get("conditions") != ["standard", "reset", "shuffle", "state_swap"]:
        raise ValueError("Baseline amendment conditions drifted")
    expected_inventory = {
        "bh0_set8_smoke": (1, 1, 4, 4, "all"),
        "set8": (8, 4, 32, 128, "all"),
        "transition16": (16, 4, 64, 256, "delayed"),
        "formal_dev": (1252, 4, 5008, 20032, "all"),
        "formal_test_id": (2488, 4, 9952, 39808, "all"),
        "formal_test_ood": (2488, 4, 9952, 39808, "all"),
    }
    inventory = amendment.get("expected_inventory")
    if not isinstance(inventory, Mapping) or set(inventory) != set(expected_inventory):
        raise ValueError("Baseline amendment expected_inventory drifted")
    for name, (queries, conditions, per_condition, total, probe_role) in expected_inventory.items():
        entry = inventory[name]
        if not isinstance(entry, Mapping):
            raise ValueError(f"Baseline expected_inventory.{name} must be an object")
        require_json_values(
            entry,
            {
                "query_states": queries,
                "conditions": conditions,
                "records_per_condition": per_condition,
                "prediction_records": total,
                "probe_role": probe_role,
            },
            f"baseline expected_inventory.{name}",
        )
    execution = amendment.get("execution")
    if not isinstance(execution, Mapping) or execution.get("strict_order") != list(STAGES):
        raise ValueError("Baseline amendment must lock BH0 -> BH1 -> BH2 -> BH3")
    replication = amendment.get("replication")
    if not isinstance(replication, Mapping):
        raise ValueError("Baseline amendment replication must be an object")
    require_json_values(
        replication,
        {
            "replica_a_device": "cuda:0",
            "replica_b_device": "cuda:1",
            "concurrent_within_dataset": True,
            "scientific_prediction_payload_exact": True,
        },
        "baseline replication contract",
    )
    sensitivity = amendment.get("text_only_micro_sensitivity")
    if not isinstance(sensitivity, Mapping):
        raise ValueError("Baseline amendment must define the text-only micro sensitivity")
    require_json_values(
        sensitivity,
        {
            "method": "qwen_full_event_history_text_only",
            "input_mode": "text_only",
            "micro_sensitivity": True,
            "suites": ["set8", "transition16"],
            "replica": "A",
            "device": "cuda:0",
            "role": "micro_sensitivity_not_formal_baseline",
            "formal_dev_or_test_forbidden": True,
            "does_not_authorize_formal_stages": True,
        },
        "text-only micro sensitivity",
    )
    data = amendment.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Baseline amendment data must be an object")
    required_data = {
        "set8_gate_sha256",
        "transition16_gate_sha256",
        "formal_manifest_sha256",
        "formal_dev_sha256",
        "formal_test_id_sha256",
        "formal_test_ood_sha256",
    }
    if set(data) != required_data:
        raise ValueError("Baseline amendment data lock has missing or unexpected entries")
    if any(SHA256_PATTERN.fullmatch(str(value)) is None for value in data.values()):
        raise ValueError("Baseline amendment data locks must be lowercase SHA256 digests")
    return amendment, sha256_file(path)


def verify_formal_preflight(path: Path, *, expected_commit: str) -> tuple[dict[str, Any], str]:
    if COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a lowercase full Git commit")
    path = path.resolve()
    digest = verify_sha_sidecar(path)
    preflight = load_json_object(path)
    require_json_values(preflight, {"passed": True, "formal_ready": True}, "formal preflight")
    git = preflight.get("git")
    if not isinstance(git, Mapping) or git.get("commit") != expected_commit:
        raise ValueError("Formal preflight is not bound to the baseline Git commit")
    return preflight, digest


def verified_reader_snapshot(
    *,
    repo: Path,
    model_root: Path,
    preflight: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    lock = load_json_object(repo / "models.lock.json")
    specification = lock.get("models", {}).get("qwen_reader")
    if not isinstance(specification, Mapping):
        raise ValueError("models.lock.json does not contain qwen_reader")
    reader_contract = amendment["reader"]
    require_json_values(
        specification,
        {
            "repo_id": reader_contract["repo_id"],
            "revision": reader_contract["revision"],
        },
        "locked Qwen Reader",
    )
    model_dir = model_root.resolve() / Path(str(specification["local_dir"])).name
    if model_root.resolve().is_symlink() or model_dir.is_symlink():
        raise ValueError("Baseline model root and Qwen snapshot must not be symlinks")
    current = verify_snapshot_manifest(
        manifest_path=model_dir / str(specification["snapshot_manifest"]),
        model_dir=model_dir,
        expected_repo_id=str(specification["repo_id"]),
        expected_revision=str(specification["revision"]),
    )
    preflight_models = preflight.get("models")
    reported_model = preflight_models.get("qwen_reader") if isinstance(preflight_models, Mapping) else None
    reported = reported_model.get("snapshot_manifest") if isinstance(reported_model, Mapping) else None
    if not isinstance(reported, Mapping) or dict(reported) != current:
        raise ValueError("Formal preflight Qwen snapshot binding drifted")
    verify_snapshot_binding(current)
    return current


def bind_data_files(
    amendment: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, dict[str, str]]:
    amendment_data = amendment["data"]
    if set(paths) != set(amendment_data):
        raise ValueError("Materializer data paths must exactly match the prospective data lock")
    return {
        name: require_file_sha(path.resolve(), str(amendment_data[name]), name)
        for name, path in paths.items()
    }


def validate_scientific_command(command: Sequence[str]) -> None:
    if not command or not all(isinstance(value, str) and value for value in command):
        raise ValueError("Every baseline command must be a non-empty argv string list")
    flattened = " ".join(command).lower()
    for fragment in FORBIDDEN_COMMAND_FRAGMENTS:
        if fragment in flattened:
            raise ValueError(f"Baseline command contains forbidden training/privileged fragment: {fragment}")
    if "--method" in command:
        index = list(command).index("--method")
        if index + 1 >= len(command):
            raise ValueError("Baseline evaluator --method has no value")
        method = command[index + 1]
        if method == "qwen_full_event_history":
            if "--micro-sensitivity" in command or "text_only" in command:
                raise ValueError("Primary blank-image baseline cannot be labeled as text-only sensitivity")
        elif method == "qwen_full_event_history_text_only":
            if "--micro-sensitivity" not in command:
                raise ValueError("Text-only evaluation is restricted to explicit micro sensitivity")
            if "--input-mode" not in command or command[list(command).index("--input-mode") + 1] != "text_only":
                raise ValueError("Text-only micro sensitivity must use input_mode=text_only")
        else:
            raise ValueError("Baseline evaluator received an unsupported method")


def validate_replica_pair(pair: Sequence[Mapping[str, Any]]) -> None:
    if len(pair) != 2:
        raise ValueError("Every baseline parallel group must contain exactly A and B")
    by_replica = {str(entry.get("replica")): entry for entry in pair}
    if set(by_replica) != set(DEVICES):
        raise ValueError("Every baseline parallel group must contain replicas A and B")
    for replica, device in DEVICES.items():
        entry = by_replica[replica]
        if entry.get("device") != device:
            raise ValueError(f"Replica {replica} must be bound to {device}")
        argv = entry.get("argv")
        if not isinstance(argv, list):
            raise ValueError("Replica argv must be a list")
        validate_scientific_command(argv)
        if "--device" not in argv or argv[argv.index("--device") + 1] != device:
            raise ValueError(f"Replica {replica} evaluator argv must explicitly use {device}")


def verify_score_report(path: Path, *, suite: str, require_text_only_sensitivity: bool = False) -> dict[str, Any]:
    report = load_json_object(path)
    require_json_values(
        report,
        {"schema": SCORE_SCHEMA, "suite": suite, "method": "qwen_full_event_history", "passed": True},
        f"baseline score {path.name}",
    )
    replication = report.get("replication")
    if not isinstance(replication, Mapping) or replication.get("passed") is not True:
        raise ValueError(f"Baseline score {path.name} did not pass A/B replication")
    if replication.get("bitwise_scientific_payload_match") is not True:
        raise ValueError(f"Baseline score {path.name} scientific payload is not exact across A/B")
    if suite in {"set8", "transition16"}:
        gate = report.get("micro_gate")
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            raise ValueError(f"Baseline score {path.name} did not pass its preregistered micro gate")
    sensitivity = report.get("text_only_sensitivity")
    if require_text_only_sensitivity:
        if not isinstance(sensitivity, Mapping):
            raise ValueError(f"Baseline score {path.name} is missing text-only micro sensitivity")
        if sensitivity.get("role") != "micro_sensitivity_not_formal_baseline":
            raise ValueError(f"Baseline score {path.name} mislabels text-only sensitivity")
    elif sensitivity is not None:
        raise ValueError(f"Baseline score {path.name} unexpectedly contains text-only sensitivity")
    return report


def verify_prediction_jsonl(output: Mapping[str, Any]) -> None:
    path = Path(str(output["path"])).resolve()
    expected_records = output.get("expected_records")
    if not isinstance(expected_records, int) or isinstance(expected_records, bool) or expected_records <= 0:
        raise ValueError("prediction_jsonl output requires a positive expected_records")
    expected_method = output.get("expected_method")
    expected_input_mode = output.get("expected_input_mode")
    expected_conditions = output.get("expected_conditions")
    expected_probe_role = output.get("expected_probe_role")
    expected_micro_sensitivity = output.get("expected_micro_sensitivity")
    if not isinstance(expected_conditions, list) or not expected_conditions:
        raise ValueError("prediction_jsonl output requires expected_conditions")
    records = 0
    conditions: set[str] = set()
    views_by_query_condition: dict[tuple[str, str], set[int]] = {}
    conditions_by_query: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            require_json_values(
                row,
                {
                    "method": expected_method,
                    "input_mode": expected_input_mode,
                    "micro_sensitivity": expected_micro_sensitivity,
                    "choice_view_family": "reverse-cyclic4",
                    "context_truncated": False,
                },
                f"{path.name}:{line_number}",
            )
            condition = row.get("condition")
            if condition not in expected_conditions:
                raise ValueError(f"{path}:{line_number} has an unexpected condition {condition!r}")
            if expected_probe_role == "delayed" and row.get("probe_role") != "delayed":
                raise ValueError(f"{path}:{line_number} is not a delayed probe")
            base_query_id = row.get("base_query_id")
            view_index = row.get("choice_view_index")
            if not isinstance(base_query_id, str) or not base_query_id:
                raise ValueError(f"{path}:{line_number} is missing base_query_id")
            if not isinstance(view_index, int) or isinstance(view_index, bool) or not 0 <= view_index < 4:
                raise ValueError(f"{path}:{line_number} has an invalid reverse-cyclic view index")
            key = (base_query_id, str(condition))
            views = views_by_query_condition.setdefault(key, set())
            if view_index in views:
                raise ValueError(f"{path}:{line_number} duplicates view {view_index} for {key}")
            views.add(view_index)
            conditions_by_query.setdefault(base_query_id, set()).add(str(condition))
            conditions.add(str(condition))
            records += 1
    if records != expected_records:
        raise ValueError(f"{path} expected {expected_records} records, observed {records}")
    if conditions != set(expected_conditions):
        raise ValueError(f"{path} did not materialize every expected condition")
    expected_views = {0, 1, 2, 3}
    incomplete = {
        key: sorted(expected_views - views)
        for key, views in views_by_query_condition.items()
        if views != expected_views
    }
    if incomplete:
        first = next(iter(sorted(incomplete.items())))
        raise ValueError(f"{path} has an incomplete reverse-cyclic view set: {first}")
    expected_condition_set = set(expected_conditions)
    missing_conditions = {
        query_id: sorted(expected_condition_set - observed)
        for query_id, observed in conditions_by_query.items()
        if observed != expected_condition_set
    }
    if missing_conditions:
        first = next(iter(sorted(missing_conditions.items())))
        raise ValueError(f"{path} has incomplete conditions for a query: {first}")
    expected_query_states = expected_records // (4 * len(expected_conditions))
    if len(conditions_by_query) != expected_query_states:
        raise ValueError(
            f"{path} expected {expected_query_states} base queries, observed {len(conditions_by_query)}"
        )


def verify_declared_output(output: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(str(output["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"Required baseline output is missing: {path}")
    required = output.get("required_values", {})
    if required:
        value = load_json_object(path)
        if not isinstance(required, Mapping):
            raise ValueError("required_values must be an object")
        require_json_values(value, required, str(output.get("label", path.name)))
    validator = output.get("validator")
    if validator is None:
        return None
    if validator == "prediction_jsonl":
        verify_prediction_jsonl(output)
        return None
    if validator not in {"baseline_score", "baseline_sensitivity_score"}:
        raise ValueError(f"Unknown baseline output validator: {validator}")
    suite = str(output.get("suite"))
    if suite not in {"set8", "transition16", "formal"}:
        raise ValueError("baseline_score output requires a locked suite")
    return verify_score_report(
        path,
        suite=suite,
        require_text_only_sensitivity=validator == "baseline_sensitivity_score",
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        return load_json_object(path)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON artifact {path}: {exc}") from exc


__all__ = [
    "AMENDMENT_SCHEMA",
    "DEVICES",
    "PLAN_PROTOCOL",
    "SCORE_SCHEMA",
    "STAGES",
    "STAGE_EVIDENCE_PROTOCOL",
    "STAGE_SPEC_PROTOCOL",
    "bind_data_files",
    "load_amendment",
    "read_json",
    "validate_replica_pair",
    "validate_scientific_command",
    "verified_reader_snapshot",
    "verify_declared_output",
    "verify_formal_preflight",
    "verify_prediction_jsonl",
    "verify_score_report",
]
