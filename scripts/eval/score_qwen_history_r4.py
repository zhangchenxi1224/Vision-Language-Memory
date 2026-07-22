from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.eval.r4_history_representations import (  # noqa: E402
    QWEN_R4_LAST_EFFECTIVE_EVENT,
    QWEN_R4_OPERATION_TAGGED_HISTORY,
    QWEN_R4_RAW_HISTORY,
    R4_HISTORY_METHODS,
    representation_contract_sha256,
)
from vision_memory.repro import canonical_object_sha256  # noqa: E402


SCHEMA = "vlm.qwen-history-r4-score.v1"
PREDICTION_SCHEMA = "vision_memory.qwen_r4_history_predictions.v1"
PREDICTION_REPORT_SCHEMA = "vision_memory.qwen_r4_history_report.v1"
SCIENTIFIC_PAYLOAD_SCHEMA = "vision_memory.qwen_r4_history_scientific_payload.v1"
EXPECTED_READER_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
EXPECTED_READER_RESIZE_CONTRACT = (
    "r3-qwen-reader-1024-to-256-bicubic-antialias-cpu-adjoint.v1"
)
EXPECTED_BLANK_IMAGE_CONTRACT = {
    "shape": [3, 1024, 1024],
    "dtype": "float32",
    "value": 0.5,
    "bytes": 12_582_912,
    "reader_resize_contract": EXPECTED_READER_RESIZE_CONTRACT,
}
CONDITIONS = ("standard", "reset", "shuffle", "state_swap")
RUNTIME_FIELDS = frozenset(
    {
        "replica_id",
        "event_latency_seconds",
        "reader_latency_seconds",
        "query_latency_seconds",
        "latency_seconds",
        "peak_reader_vram_gib",
        "peak_vram_gib",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def prediction_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("episode_id"),
        row.get("query_id"),
        row.get("query_ordinal"),
        row.get("probe_role", "delayed"),
        row.get("choice_view_family"),
        row.get("choice_view_index"),
        row.get("condition", "standard"),
    )


def state_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("episode_id"),
        row.get("query_ordinal"),
        row.get("probe_role", "delayed"),
        row.get("condition", "standard"),
    )


def scientific_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in RUNTIME_FIELDS}


def scientific_prediction_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    # File order is part of the evaluator payload. A/B must reproduce both
    # every scientific value and the deterministic traversal order exactly.
    records = [scientific_row(row) for row in rows]
    payload = {"schema_version": SCIENTIFIC_PAYLOAD_SCHEMA, "records": records}
    return {"sha256": canonical_object_sha256(payload), "payload": payload}


def validate_rows(rows: Sequence[Mapping[str, Any]], *, method: str) -> dict[str, Any]:
    if method not in R4_HISTORY_METHODS:
        raise ValueError(f"Unsupported R4 history method: {method!r}")
    expected_contract = representation_contract_sha256(method)
    identities: set[tuple[Any, ...]] = set()
    replica_ids: set[str] = set()
    dataset_sha256s: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"prediction row {index}"
        if row.get("schema_version") != PREDICTION_SCHEMA:
            raise ValueError(f"{prefix} has an unsupported prediction schema")
        if row.get("method") != method:
            raise ValueError(f"{prefix} method drifted")
        if row.get("input_mode") != "blank_image":
            raise ValueError(f"{prefix} must use the fixed blank image")
        if row.get("condition") not in CONDITIONS:
            raise ValueError(f"{prefix} has an unsupported condition")
        if row.get("choice_view_family") != "reverse-cyclic4":
            raise ValueError(f"{prefix} must use reverse-cyclic4")
        if row.get("deterministic_ce") is not True or row.get("context_truncated") is not False:
            raise ValueError(f"{prefix} violates deterministic/fail-closed scoring")
        if row.get("representation_contract_sha256") != expected_contract:
            raise ValueError(f"{prefix} representation contract SHA drifted")
        for field in (
            "query_text_sha256",
            "dataset_sha256",
            "episodes_sha256",
            "source_event_stream_sha256",
            "memory_text_sha256",
            "prompt_sha256",
            "chat_prompt_sha256",
        ):
            if not valid_sha256(row.get(field)):
                raise ValueError(f"{prefix}.{field} must be a lowercase SHA256")
        if row.get("dataset_sha256") != row.get("episodes_sha256"):
            raise ValueError(f"{prefix} dataset/episodes SHA binding drifted")
        dataset_sha256s.add(str(row["dataset_sha256"]))
        if row.get("blank_image") != EXPECTED_BLANK_IMAGE_CONTRACT:
            raise ValueError(f"{prefix} blank-image contract drifted")
        if row.get("reader_resize_contract") != EXPECTED_READER_RESIZE_CONTRACT:
            raise ValueError(f"{prefix} Reader resize contract drifted")
        choices = row.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 4
            or len(set(choices)) != 4
            or not all(isinstance(value, str) and value for value in choices)
        ):
            raise ValueError(f"{prefix} requires four distinct non-empty string choices")
        nll = row.get("choice_mean_nll")
        if not isinstance(nll, list) or len(nll) != 4:
            raise ValueError(f"{prefix} requires four choice mean NLLs")
        scores = [_finite(value, field=f"{prefix}.choice_mean_nll") for value in nll]
        target = _nonnegative_int(row.get("target_index"), field=f"{prefix}.target_index")
        prediction = _nonnegative_int(row.get("prediction_index"), field=f"{prefix}.prediction_index")
        if target >= 4 or prediction >= 4:
            raise ValueError(f"{prefix} target/prediction must be in [0,3]")
        if prediction != min(range(4), key=scores.__getitem__):
            raise ValueError(f"{prefix} prediction does not equal deterministic NLL argmin")
        if row.get("target_text") != choices[target] or row.get("prediction_text") != choices[prediction]:
            raise ValueError(f"{prefix} target/prediction text does not match its index")
        source_count = _nonnegative_int(row.get("source_event_count"), field=f"{prefix}.source_event_count")
        retained_count = _nonnegative_int(
            row.get("retained_event_count"), field=f"{prefix}.retained_event_count"
        )
        if method in {QWEN_R4_RAW_HISTORY, QWEN_R4_OPERATION_TAGGED_HISTORY}:
            if retained_count != source_count:
                raise ValueError(f"{prefix} chronological representation dropped an event")
        elif retained_count not in {0, 1} or retained_count > source_count:
            raise ValueError(f"{prefix} last-effective retained count is invalid")
        for field in (
            "memory_token_count",
            "memory_utf8_bytes",
            "prompt_token_count",
            "prompt_utf8_bytes",
            "state_bytes",
            "constant_visual_input_bytes",
        ):
            value = _nonnegative_int(row.get(field), field=f"{prefix}.{field}")
            if field in {"memory_token_count", "memory_utf8_bytes", "prompt_token_count", "prompt_utf8_bytes"} and value == 0:
                raise ValueError(f"{prefix}.{field} must be positive")
        if row.get("constant_visual_input_bytes") != EXPECTED_BLANK_IMAGE_CONTRACT["bytes"]:
            raise ValueError(f"{prefix} blank-image byte accounting drifted")
        if row.get("state_bytes") != row.get("memory_utf8_bytes"):
            raise ValueError(f"{prefix} state byte accounting drifted")
        if not isinstance(row.get("episode_id"), str) or not row["episode_id"]:
            raise ValueError(f"{prefix} requires a non-empty episode_id")
        identity = prediction_identity(row)
        if identity in identities:
            raise ValueError(f"Duplicate prediction identity: {identity!r}")
        identities.add(identity)
        replica_id = row.get("replica_id")
        if replica_id not in {"A", "B"}:
            raise ValueError(f"{prefix} replica_id must be A or B")
        replica_ids.add(str(replica_id))
    if len(replica_ids) != 1:
        raise ValueError("Each prediction artifact must contain exactly one replica ID")
    if len(dataset_sha256s) != 1:
        raise ValueError("Each prediction artifact must bind exactly one dataset SHA256")
    return {
        "passed": True,
        "prediction_records": len(rows),
        "unique_identities": len(identities),
        "method": method,
        "representation_contract_sha256": expected_contract,
        "replica_id": next(iter(replica_ids)),
        "dataset_sha256": next(iter(dataset_sha256s)),
    }


def validate_prediction_report(
    path: Path,
    rows_path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    report = load_json_object(path)
    if report.get("schema_version") != PREDICTION_REPORT_SCHEMA or report.get("status") != "complete":
        raise ValueError("Unsupported or incomplete R4 prediction report")
    if report.get("method") != method or report.get("input_mode") != "blank_image":
        raise ValueError("R4 prediction report method/input mode drifted")
    if report.get("reader_revision") != EXPECTED_READER_REVISION:
        raise ValueError("R4 prediction report Reader revision drifted")
    if report.get("reader_resize_contract") != EXPECTED_READER_RESIZE_CONTRACT:
        raise ValueError("R4 prediction report Reader resize contract drifted")
    if report.get("blank_image") != EXPECTED_BLANK_IMAGE_CONTRACT:
        raise ValueError("R4 prediction report blank-image contract drifted")
    if report.get("representation_contract_sha256") != representation_contract_sha256(method):
        raise ValueError("R4 prediction report representation contract drifted")
    if report.get("output_sha256") != sha256_file(rows_path):
        raise ValueError("R4 prediction report does not bind the prediction file")
    if report.get("prediction_records") != len(rows):
        raise ValueError("R4 prediction report record count drifted")
    row_dataset_sha256s = {row.get("dataset_sha256") for row in rows}
    if len(row_dataset_sha256s) != 1:
        raise ValueError("R4 prediction rows do not bind exactly one dataset SHA256")
    row_dataset_sha256 = next(iter(row_dataset_sha256s))
    if (
        not valid_sha256(report.get("dataset_sha256"))
        or report.get("dataset_sha256") != report.get("episodes_sha256")
        or report.get("dataset_sha256") != row_dataset_sha256
    ):
        raise ValueError("R4 prediction report dataset/episodes SHA binding drifted")
    expected_payload = scientific_prediction_payload(rows)["sha256"]
    if report.get("scientific_payload_sha256") != expected_payload:
        raise ValueError("R4 prediction report scientific payload SHA drifted")
    if report.get("deterministic_ce") is not True or report.get("context_truncation_policy") != "fail_closed":
        raise ValueError("R4 prediction report deterministic/context contract drifted")
    return report


def replication_report(
    rows_a: Sequence[Mapping[str, Any]], rows_b: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    identities_a = {prediction_identity(row) for row in rows_a}
    identities_b = {prediction_identity(row) for row in rows_b}
    payload_a = scientific_prediction_payload(rows_a)
    payload_b = scientific_prediction_payload(rows_b)
    identity_match = identities_a == identities_b and len(identities_a) == len(rows_a) == len(rows_b)
    payload_match = payload_a["sha256"] == payload_b["sha256"]
    return {
        "passed": identity_match and payload_match,
        "identity_sets_match": identity_match,
        "bitwise_scientific_payload_match": payload_match,
        "replica_a_scientific_payload_sha256": payload_a["sha256"],
        "replica_b_scientific_payload_sha256": payload_b["sha256"],
        "records_a": len(rows_a),
        "records_b": len(rows_b),
    }


def _correct(row: Mapping[str, Any]) -> int:
    return int(row["prediction_index"] == row["target_index"])


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(_correct(row) for row in rows)
    return {"correct": correct, "count": len(rows), "accuracy": correct / len(rows) if rows else None}


def _group_summary(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field))].append(row)
    return {name: _summary(values) for name, values in sorted(grouped.items())}


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def descriptive_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    standard = [row for row in rows if row.get("condition") == "standard"]
    conditions = {
        condition: _summary([row for row in rows if row.get("condition") == condition])
        for condition in CONDITIONS
    }
    base_accuracy = conditions["standard"]["accuracy"]
    for condition in ("reset", "shuffle"):
        condition_accuracy = conditions[condition]["accuracy"]
        conditions[condition]["accuracy_drop_from_standard"] = (
            None
            if base_accuracy is None or condition_accuracy is None
            else base_accuracy - condition_accuracy
        )
    rotations: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in standard:
        rotations[state_identity(row)].append(row)
    consistent = sum(
        len(values) == 4 and len({str(row["prediction_text"]) for row in values}) == 1
        for values in rotations.values()
    )
    swap_rows = [
        row
        for row in rows
        if row.get("condition") == "state_swap" and isinstance(row.get("donor_target_index"), int)
    ]
    donor_correct = sum(row["prediction_index"] == row["donor_target_index"] for row in swap_rows)
    return {
        "standard": _summary(standard),
        "conditions": conditions,
        "by_target_position": _group_summary(standard, "target_index"),
        "by_event_kind": _group_summary(standard, "subtype"),
        "by_form": _group_summary(standard, "form"),
        "by_ood_group": _group_summary(standard, "ood_group"),
        "rotation": {
            "consistent": consistent,
            "count": len(rotations),
            "agreement_rate": consistent / len(rotations) if rotations else None,
        },
        "state_swap_donor_answer": {
            "correct": donor_correct,
            "count": len(swap_rows),
            "rate": donor_correct / len(swap_rows) if swap_rows else None,
        },
        "efficiency": {
            "condition_scope": "standard",
            "memory_tokens": _numeric_summary([float(row["memory_token_count"]) for row in standard]),
            "memory_utf8_bytes": _numeric_summary([float(row["memory_utf8_bytes"]) for row in standard]),
            "retained_events": _numeric_summary([float(row["retained_event_count"]) for row in standard]),
            "latency_seconds": _numeric_summary([float(row["latency_seconds"]) for row in standard]),
            "peak_vram_gib": _numeric_summary([float(row["peak_vram_gib"]) for row in standard]),
        },
    }


def _delayed_reverse_rows(
    rows: Sequence[Mapping[str, Any]], condition: str
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row.get("condition") == condition
        and row.get("probe_role") == "delayed"
        and row.get("choice_view_family") == "reverse-cyclic4"
    ]


def _clean_noop_pair_summary(
    standard: Sequence[Mapping[str, Any]], *, expected_pairs: int
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in standard:
        pair_id = row.get("distractor_pair_id")
        variant = row.get("distractor_variant")
        if isinstance(pair_id, str) and pair_id and variant in {"clean", "distractor"}:
            grouped[(pair_id, int(row["choice_view_index"]))].append(row)
    if len(grouped) != expected_pairs or any(len(values) != 2 for values in grouped.values()):
        raise ValueError(f"Expected {expected_pairs} clean/noop view pairs")
    predicted_agreement = 0
    exact_memory = 0
    exact_prompt = 0
    exact_nll = 0
    exact_all = 0
    for identity, values in sorted(grouped.items()):
        variants = {str(row.get("distractor_variant")): row for row in values}
        if set(variants) != {"clean", "distractor"}:
            raise ValueError(f"Clean/noop pair {identity!r} has invalid variants")
        clean = variants["clean"]
        noop = variants["distractor"]
        for field in ("choices", "target_index", "target_text"):
            if clean.get(field) != noop.get(field):
                raise ValueError(f"Clean/noop pair {identity!r} changed {field}")
        prediction_match = clean.get("prediction_text") == noop.get("prediction_text")
        memory_match = clean.get("memory_text_sha256") == noop.get("memory_text_sha256")
        prompt_match = clean.get("prompt_sha256") == noop.get("prompt_sha256")
        nll_match = clean.get("choice_mean_nll") == noop.get("choice_mean_nll")
        predicted_agreement += int(prediction_match)
        exact_memory += int(memory_match)
        exact_prompt += int(prompt_match)
        exact_nll += int(nll_match)
        exact_all += int(prediction_match and memory_match and prompt_match and nll_match)
    return {
        "count": len(grouped),
        "predicted_text_agreement": predicted_agreement,
        "memory_sha_exact": exact_memory,
        "prompt_sha_exact": exact_prompt,
        "choice_nll_exact": exact_nll,
        "all_fields_exact": exact_all,
    }


def _smoke_gate(rows: Sequence[Mapping[str, Any]], *, method: str) -> dict[str, Any]:
    standard = _delayed_reverse_rows(rows, "standard")
    if len(standard) != 16:
        raise ValueError(f"Smoke4 requires 16 delayed reverse views, got {len(standard)}")
    kinds = {
        kind: [row for row in standard if row.get("subtype") == kind]
        for kind in ("set", "overwrite", "clear", "noop")
    }
    positions = {
        str(index): [row for row in standard if row.get("target_index") == index]
        for index in range(4)
    }
    if any(len(values) != 4 for values in kinds.values()) or any(
        len(values) != 4 for values in positions.values()
    ):
        raise ValueError("Smoke4 requires four views for every kind and target position")
    rotations: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in standard:
        rotations[state_identity(row)].append(row)
    if len(rotations) != 4 or any(len(values) != 4 for values in rotations.values()):
        raise ValueError("Smoke4 requires four states with four reverse-cyclic views")
    correct = sum(_correct(row) for row in standard)
    rotation_consistent = sum(
        len({str(row["prediction_text"]) for row in values}) == 1
        for values in rotations.values()
    )
    clean_noop = _clean_noop_pair_summary(standard, expected_pairs=4)
    required = method == QWEN_R4_LAST_EFFECTIVE_EVENT
    thresholds = {
        "overall": 15 if required else 14,
        "position": 3,
        "kind": 3,
        "rotation": 4 if required else 3,
        "clean_noop": 4 if required else 3,
    }
    checks = {
        "overall": correct >= thresholds["overall"],
        "positions": all(
            sum(_correct(row) for row in values) >= thresholds["position"]
            for values in positions.values()
        ),
        "event_kinds": all(
            sum(_correct(row) for row in values) >= thresholds["kind"]
            for values in kinds.values()
        ),
        "rotation": rotation_consistent >= thresholds["rotation"],
        "clean_noop": (
            clean_noop["all_fields_exact"] >= thresholds["clean_noop"]
            if required
            else clean_noop["predicted_text_agreement"] >= thresholds["clean_noop"]
        ),
    }
    payload = {
        "suite": "smoke",
        "method": method,
        "passed": all(checks.values()),
        "performance_only": not required,
        "data_readability_required": required,
        "correct": correct,
        "count": len(standard),
        "event_kinds": {name: _summary(values) for name, values in kinds.items()},
        "positions": {name: _summary(values) for name, values in positions.items()},
        "rotation": {"consistent": rotation_consistent, "count": len(rotations)},
        "clean_noop": clean_noop,
        "thresholds": thresholds,
        "checks": checks,
    }
    return {**payload, "scientific_payload_sha256": canonical_object_sha256(payload)}


def _transition32_gate(rows: Sequence[Mapping[str, Any]], *, method: str) -> dict[str, Any]:
    standard = _delayed_reverse_rows(rows, "standard")
    if len(standard) != 128:
        raise ValueError(f"Transition32 requires 128 standard delayed reverse views, got {len(standard)}")
    positions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    kinds: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    cells: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    mixed: list[Mapping[str, Any]] = []
    rotations: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in standard:
        positions[str(row["target_index"])].append(row)
        kind = str(row.get("subtype"))
        form = str(row.get("form"))
        kinds[kind].append(row)
        cells[f"{kind}:{form}"].append(row)
        if form == "mixed":
            mixed.append(row)
        rotations[state_identity(row)].append(row)
    if set(positions) != {"0", "1", "2", "3"} or any(len(values) != 32 for values in positions.values()):
        raise ValueError("Transition32 requires 32 target views in each position")
    if set(kinds) != {"set", "overwrite", "clear", "noop"} or any(
        len(values) != 32 for values in kinds.values()
    ):
        raise ValueError("Transition32 requires 32 views for each event kind")
    if len(cells) != 8 or any(len(values) != 16 for values in cells.values()):
        raise ValueError("Transition32 requires eight kind/form cells with 16 views each")
    if len(mixed) != 64:
        raise ValueError("Transition32 requires 64 mixed delayed views")
    if len(rotations) != 32 or any(len(values) != 4 for values in rotations.values()):
        raise ValueError("Transition32 requires 32 states with four reverse-cyclic views")
    swap_probe = [
        row
        for row in rows
        if row.get("condition") == "state_swap"
        and row.get("probe_role") == "delayed"
        and int(row.get("choice_view_index", -1)) == 0
    ]
    if len(swap_probe) != 32 or any(not isinstance(row.get("donor_target_index"), int) for row in swap_probe):
        raise ValueError("Transition32 requires the locked 32-record state-swap donor probe")

    reset = _delayed_reverse_rows(rows, "reset")
    shuffle = _delayed_reverse_rows(rows, "shuffle")
    state_swap = _delayed_reverse_rows(rows, "state_swap")
    if any(len(values) != 128 for values in (reset, shuffle, state_swap)):
        raise ValueError("Transition32 requires 128 delayed views in every intervention condition")
    if any(not isinstance(row.get("donor_target_index"), int) for row in state_swap):
        raise ValueError("Every Transition32 state-swap view must expose its donor target index")
    clean_noop = _clean_noop_pair_summary(standard, expected_pairs=32)

    correct = sum(_correct(row) for row in standard)
    reset_correct = sum(_correct(row) for row in reset)
    shuffle_correct = sum(_correct(row) for row in shuffle)
    position_summary = {name: _summary(values) for name, values in sorted(positions.items())}
    kind_summary = {name: _summary(values) for name, values in sorted(kinds.items())}
    cell_summary = {name: _summary(values) for name, values in sorted(cells.items())}
    mixed_summary = _summary(mixed)
    rotation_consistent = sum(
        len({str(row["prediction_text"]) for row in values}) == 1 for values in rotations.values()
    )
    donor_correct = sum(row["prediction_index"] == row["donor_target_index"] for row in swap_probe)
    if method == QWEN_R4_LAST_EFFECTIVE_EVENT:
        thresholds = {
            "overall": 122,
            "position": 30,
            "kind": 30,
            "mixed": 61,
            "cell": 15,
            "rotation": 31,
            "donor": 30,
            "reset_drop": 32,
            "shuffle_drop": 32,
            "clean_noop": 32,
        }
        performance_only = False
        data_readability_required = True
    else:
        thresholds = {
            "overall": 116,
            "position": 28,
            "kind": 28,
            "mixed": 58,
            "cell": 14,
            "rotation": 30,
            "donor": 28,
            "reset_drop": 32,
            "shuffle_drop": 32,
            "clean_noop": 30,
        }
        performance_only = True
        data_readability_required = False
    checks = {
        "overall": correct >= thresholds["overall"],
        "positions": all(value["correct"] >= thresholds["position"] for value in position_summary.values()),
        "event_kinds": all(value["correct"] >= thresholds["kind"] for value in kind_summary.values()),
        "mixed": mixed_summary["correct"] >= thresholds["mixed"],
        "cells": all(value["correct"] >= thresholds["cell"] for value in cell_summary.values()),
        "rotation": rotation_consistent >= thresholds["rotation"],
        "state_swap_donor": donor_correct >= thresholds["donor"],
        "reset_drop": correct - reset_correct >= thresholds["reset_drop"],
        "shuffle_drop": correct - shuffle_correct >= thresholds["shuffle_drop"],
        "clean_noop": (
            clean_noop["all_fields_exact"] >= thresholds["clean_noop"]
            if method == QWEN_R4_LAST_EFFECTIVE_EVENT
            else clean_noop["predicted_text_agreement"] >= thresholds["clean_noop"]
        ),
    }
    payload = {
        "suite": "transition32",
        "method": method,
        "passed": all(checks.values()),
        "performance_only": performance_only,
        "data_readability_required": data_readability_required,
        "correct": correct,
        "count": len(standard),
        "positions": position_summary,
        "event_kinds": kind_summary,
        "mixed": mixed_summary,
        "cells": cell_summary,
        "rotation": {"consistent": rotation_consistent, "count": len(rotations)},
        "state_swap": {"donor_answers": donor_correct, "count": len(swap_probe)},
        "reset": {"correct": reset_correct, "count": len(reset), "drop_from_standard": correct - reset_correct},
        "shuffle": {
            "correct": shuffle_correct,
            "count": len(shuffle),
            "drop_from_standard": correct - shuffle_correct,
        },
        "clean_noop": clean_noop,
        "thresholds": thresholds,
        "checks": checks,
    }
    return {**payload, "scientific_payload_sha256": canonical_object_sha256(payload)}


def score_r4_history(
    *,
    predictions: Path,
    prediction_report: Path,
    replica_b_predictions: Path,
    replica_b_report: Path,
    suite: str,
    method: str,
) -> dict[str, Any]:
    rows_a = load_jsonl(predictions)
    rows_b = load_jsonl(replica_b_predictions)
    integrity_a = validate_rows(rows_a, method=method)
    integrity_b = validate_rows(rows_b, method=method)
    report_a = validate_prediction_report(prediction_report, predictions, rows_a, method=method)
    report_b = validate_prediction_report(replica_b_report, replica_b_predictions, rows_b, method=method)
    if integrity_a["replica_id"] != "A" or integrity_b["replica_id"] != "B":
        raise ValueError("The primary and replica-B artifacts must be labelled A and B respectively")
    replication = replication_report(rows_a, rows_b)
    if suite == "smoke":
        scientific_gate = _smoke_gate(rows_a, method=method)
    elif suite == "transition32":
        scientific_gate = _transition32_gate(rows_a, method=method)
    elif suite == "formal":
        scientific_gate = {
            "suite": suite,
            "method": method,
            "passed": None,
            "performance_only": True,
            "data_readability_required": False,
            "checks": {},
            "role": "not_a_scientific_gate",
        }
    else:
        raise ValueError("suite must be smoke, transition32, or formal")
    blocking_scientific_gate = suite in {"smoke", "transition32"} and method == QWEN_R4_LAST_EFFECTIVE_EVENT
    passed = bool(
        integrity_a["passed"]
        and integrity_b["passed"]
        and replication["passed"]
        and (not blocking_scientific_gate or scientific_gate["passed"])
    )
    payload = {
        "schema": SCHEMA,
        "method": method,
        "suite": suite,
        "passed": passed,
        "execution_passed": passed,
        "integrity": {
            "passed": True,
            "replica_a": integrity_a,
            "replica_b": integrity_b,
            "prediction_sha256": sha256_file(predictions),
            "prediction_report_sha256": sha256_file(prediction_report),
            "replica_b_prediction_sha256": sha256_file(replica_b_predictions),
            "replica_b_report_sha256": sha256_file(replica_b_report),
            "episodes_sha256_match": report_a.get("episodes_sha256") == report_b.get("episodes_sha256"),
            "git_commit_match": report_a.get("git_commit") == report_b.get("git_commit"),
        },
        "replication": replication,
        "scientific_gate": scientific_gate,
        "descriptive_metrics": descriptive_metrics(rows_a),
        "blocking_policy": {
            "scientific_gate_blocks_execution": blocking_scientific_gate,
            "raw_and_tagged_transition_thresholds_are_report_only": method
            in {QWEN_R4_RAW_HISTORY, QWEN_R4_OPERATION_TAGGED_HISTORY},
            "formal_is_integrity_and_replication_only": suite == "formal",
        },
    }
    if not payload["integrity"]["episodes_sha256_match"] or not payload["integrity"]["git_commit_match"]:
        payload["integrity"]["passed"] = False
        payload["passed"] = False
        payload["execution_passed"] = False
    return {**payload, "report_sha256": canonical_object_sha256(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one replicated R4 Qwen history arm")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--replica-b-predictions", type=Path, required=True)
    parser.add_argument("--replica-b-report", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "transition32", "formal"), required=True)
    parser.add_argument("--method", choices=R4_HISTORY_METHODS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fail-on-gate", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing score report: {args.output}")
    report = score_r4_history(
        predictions=args.predictions,
        prediction_report=args.prediction_report,
        replica_b_predictions=args.replica_b_predictions,
        replica_b_report=args.replica_b_report,
        suite=args.suite,
        method=args.method,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] or not args.fail_on_gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
