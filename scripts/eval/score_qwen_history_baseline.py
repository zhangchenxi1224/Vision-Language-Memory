from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.eval.r3_micro import score_r3_micro  # noqa: E402
from vision_memory.repro import canonical_object_sha256  # noqa: E402


SCHEMA = "vlm.qwen-history-baseline-score.v1"
SCIENTIFIC_PAYLOAD_SCHEMA = "vlm.qwen-history-baseline-scientific-predictions.v1"
REPLICATION_SCHEMA = "vlm.qwen-history-baseline-replication.v1"
METHOD = "qwen_full_event_history"
TEXT_ONLY_METHOD = "qwen_full_event_history_text_only"
CONDITIONS = ("standard", "reset", "shuffle", "state_swap")
PREDICTION_SCHEMA = "vision_memory.qwen_full_event_history_predictions.v1"
PREDICTION_REPORT_SCHEMA = "vision_memory.qwen_full_event_history_report.v1"
EVALUATOR_SCIENTIFIC_SCHEMA = "vision_memory.qwen_full_event_history_scientific_payload.v1"
RUNTIME_FIELDS = {
    "replica_id",
    "event_latency_seconds",
    "reader_latency_seconds",
    "query_latency_seconds",
    "latency_seconds",
    "peak_reader_vram_gib",
    "peak_vram_gib",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def nonnegative_integer(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
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


def pair_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return prediction_identity(row)[:-1]


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_method: str = METHOD,
    expected_input_mode: str = "blank_image",
    expected_micro_sensitivity: bool = False,
) -> None:
    identities: set[tuple[Any, ...]] = set()
    methods: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"prediction row {index}"
        method = row.get("method")
        methods.add(str(method))
        if row.get("schema_version") != PREDICTION_SCHEMA:
            raise ValueError(f"{prefix} has an unsupported prediction schema")
        if method != expected_method:
            raise ValueError(f"{prefix} method must be {expected_method!r}, got {method!r}")
        if row.get("input_mode") != expected_input_mode:
            raise ValueError(f"{prefix} input_mode must be {expected_input_mode!r}")
        if row.get("micro_sensitivity") is not expected_micro_sensitivity:
            raise ValueError(f"{prefix} has the wrong micro_sensitivity flag")
        if not isinstance(row.get("episode_id"), str) or not row["episode_id"]:
            raise ValueError(f"{prefix} requires a non-empty episode_id")
        condition = row.get("condition", "standard")
        if condition not in CONDITIONS:
            raise ValueError(f"{prefix} has unsupported condition {condition!r}")
        choices = row.get("choices")
        if not isinstance(choices, list) or len(choices) != 4 or not all(isinstance(value, str) for value in choices):
            raise ValueError(f"{prefix} requires exactly four string choices")
        nll = row.get("choice_mean_nll")
        if not isinstance(nll, list) or len(nll) != 4:
            raise ValueError(f"{prefix} requires exactly four choice_mean_nll values")
        for choice_index, value in enumerate(nll):
            finite_number(value, field=f"{prefix}.choice_mean_nll[{choice_index}]")
        target = nonnegative_integer(row.get("target_index"), field=f"{prefix}.target_index")
        prediction = nonnegative_integer(row.get("prediction_index"), field=f"{prefix}.prediction_index")
        if target >= 4 or prediction >= 4:
            raise ValueError(f"{prefix} target/prediction index must be in [0, 3]")
        if row.get("target_text") != choices[target]:
            raise ValueError(f"{prefix} target_text does not match target_index")
        if row.get("prediction_text") != choices[prediction]:
            raise ValueError(f"{prefix} prediction_text does not match prediction_index")
        for field in ("history_sha256", "prompt_sha256"):
            if not valid_sha256(row.get(field)):
                raise ValueError(f"{prefix}.{field} must be a lowercase SHA256 digest")
        for field in ("history_token_count", "history_utf8_bytes"):
            nonnegative_integer(row.get(field), field=f"{prefix}.{field}")
        if row.get("latency_seconds") is not None:
            latency = finite_number(row["latency_seconds"], field=f"{prefix}.latency_seconds")
            if latency < 0:
                raise ValueError(f"{prefix}.latency_seconds cannot be negative")
        for field in ("peak_reader_vram_gib", "peak_vram_gib"):
            if row.get(field) is not None and finite_number(row[field], field=f"{prefix}.{field}") < 0:
                raise ValueError(f"{prefix}.{field} cannot be negative")
        identity = prediction_identity(row)
        if identity in identities:
            raise ValueError(f"Duplicate prediction identity: {identity}")
        identities.add(identity)
    if methods != {expected_method}:
        raise ValueError("Baseline score requires exactly one locked method")


def validate_prediction_report(
    *,
    predictions: Path,
    rows: Sequence[Mapping[str, Any]],
    report_path: Path,
    expected_method: str = METHOD,
    expected_input_mode: str = "blank_image",
    expected_micro_sensitivity: bool = False,
) -> dict[str, Any]:
    report = load_json_object(report_path)
    if report.get("schema_version") != PREDICTION_REPORT_SCHEMA:
        raise ValueError("Prediction companion report has an unsupported schema")
    actual_sha = sha256_file(predictions)
    if report.get("output_sha256") != actual_sha:
        raise ValueError("Prediction companion report does not bind the scored JSONL SHA256")
    reported_records = report.get("prediction_records", report.get("records"))
    if reported_records is not None and reported_records != len(rows):
        raise ValueError("Prediction companion report record count differs from the JSONL")
    reported_method = report.get("method")
    reported_methods = report.get("methods")
    if reported_method is not None and reported_method != expected_method:
        raise ValueError("Prediction companion report names the wrong baseline method")
    if reported_methods is not None and reported_methods != [expected_method]:
        raise ValueError("Prediction companion report must contain only the locked baseline method")
    if report.get("input_mode") != expected_input_mode:
        raise ValueError("Prediction companion report has the wrong input_mode")
    if report.get("micro_sensitivity") is not expected_micro_sensitivity:
        raise ValueError("Prediction companion report has the wrong micro_sensitivity flag")
    if not valid_sha256(report.get("episodes_sha256")):
        raise ValueError("Prediction companion report is missing episodes_sha256")
    revision = report.get("reader_revision")
    if not isinstance(revision, str) or not revision:
        raise ValueError("Prediction companion report is missing reader_revision")
    evaluator_payload = {
        "schema_version": EVALUATOR_SCIENTIFIC_SCHEMA,
        "records": [{key: value for key, value in row.items() if key not in RUNTIME_FIELDS} for row in rows],
    }
    observed_scientific_sha = canonical_object_sha256(evaluator_payload)
    if report.get("scientific_payload_sha256") != observed_scientific_sha:
        raise ValueError("Prediction companion report does not bind the complete evaluator scientific payload")
    report = {**report, "validated_scientific_payload_sha256": observed_scientific_sha}
    return report


def normalized_scientific_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in row.items() if key not in RUNTIME_FIELDS} for row in rows]


def scientific_prediction_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = normalized_scientific_rows(rows)
    return {
        "schema": SCIENTIFIC_PAYLOAD_SCHEMA,
        "row_count": len(normalized),
        "sha256": canonical_object_sha256(
            {"schema_version": EVALUATOR_SCIENTIFIC_SCHEMA, "records": normalized}
        ),
    }


def _correct(row: Mapping[str, Any]) -> int:
    return int(int(row["prediction_index"]) == int(row["target_index"]))


def _accuracy_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [_correct(row) for row in rows]
    return {"correct": sum(values), "count": len(values), "accuracy": statistics.fmean(values) if values else None}


def _group_accuracy(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            grouped[str(row[field])].append(row)
    return {key: _accuracy_summary(values) for key, values in sorted(grouped.items())}


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p05": _percentile(finite, 0.05),
        "p95": _percentile(finite, 0.95),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def _stratified_paired_bootstrap(
    pairs: Sequence[tuple[str, float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, difference in pairs:
        grouped[group].append(float(difference))
    group_ids = sorted(grouped)
    if not group_ids:
        return {"iterations": iterations, "seed": seed, "groups": 0, "ci95": [None, None]}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled = [rng.choice(group_ids) for _ in group_ids]
        values = [value for group in sampled for value in grouped[group]]
        estimates.append(statistics.fmean(values))
    return {
        "iterations": iterations,
        "seed": seed,
        "groups": len(group_ids),
        "ci95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
    }


def _condition_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, Any]]:
    by_condition: dict[str, dict[tuple[Any, ...], Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        condition = str(row.get("condition", "standard"))
        key = pair_identity(row)
        if key in by_condition[condition]:
            raise ValueError(f"Duplicate condition-pair key for {condition}: {key}")
        by_condition[condition][key] = row
    standard = by_condition.get("standard", {})
    result: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS[1:]:
        degraded = by_condition.get(condition)
        if not degraded:
            continue
        if set(standard) != set(degraded):
            missing = len(set(standard) - set(degraded))
            extra = len(set(degraded) - set(standard))
            raise ValueError(f"{condition} is not exactly paired with standard: missing={missing}, extra={extra}")
        differences: list[tuple[str, float]] = []
        standard_only_correct = 0
        condition_only_correct = 0
        both_correct = 0
        both_wrong = 0
        for key in sorted(standard, key=repr):
            left = _correct(standard[key])
            right = _correct(degraded[key])
            group = standard[key].get("semantic_group_id") or standard[key].get("episode_id")
            differences.append((str(group), float(left - right)))
            both_correct += int(left == 1 and right == 1)
            both_wrong += int(left == 0 and right == 0)
            standard_only_correct += int(left == 1 and right == 0)
            condition_only_correct += int(left == 0 and right == 1)
        values = [difference for _, difference in differences]
        result[condition] = {
            "n_pairs": len(values),
            "standard_accuracy": statistics.fmean(_correct(standard[key]) for key in standard),
            "condition_accuracy": statistics.fmean(_correct(degraded[key]) for key in degraded),
            "accuracy_drop": statistics.fmean(values),
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "standard_only_correct": standard_only_correct,
            "condition_only_correct": condition_only_correct,
            "bootstrap": _stratified_paired_bootstrap(
                differences,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            ),
        }
    return result


def _rotation_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    standard = [row for row in rows if row.get("condition", "standard") == "standard"]
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in standard:
        key = (
            row.get("episode_id"),
            row.get("base_query_id", row.get("query_id")),
            row.get("query_ordinal"),
            row.get("probe_role", "delayed"),
        )
        grouped[key].append(row)
    complete = 0
    consistent = 0
    per_group: dict[str, Any] = {}
    for key, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
        view_indices = {row.get("choice_view_index") for row in values}
        is_complete = view_indices == {0, 1, 2, 3}
        is_consistent = len({row.get("prediction_text") for row in values}) == 1
        complete += int(is_complete)
        consistent += int(is_complete and is_consistent)
        per_group["|".join(str(value) for value in key)] = {
            "views": len(values),
            "complete_reverse_cyclic4": is_complete,
            "predicted_text_consistent": is_consistent,
            **_accuracy_summary(values),
        }
    return {
        "query_groups": len(grouped),
        "complete_groups": complete,
        "consistent_complete_groups": consistent,
        "agreement_rate": consistent / complete if complete else None,
        "groups": per_group,
    }


def _noop_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    standard = [row for row in rows if row.get("condition", "standard") == "standard"]
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in standard:
        pair_id = row.get("distractor_pair_id")
        variant = row.get("distractor_variant")
        if pair_id is None or variant not in {"clean", "distractor"}:
            continue
        key = (
            pair_id,
            row.get("query_comparison_id"),
            row.get("query_ordinal"),
            row.get("probe_role", "delayed"),
            row.get("choice_view_index"),
        )
        if variant in grouped[key]:
            raise ValueError(f"Duplicate clean/noop member: {key}, {variant}")
        grouped[key][str(variant)] = row
    pairs = [pair for pair in grouped.values() if set(pair) == {"clean", "distractor"}]
    if not pairs:
        return None
    agreements = 0
    damages: list[float] = []
    for pair in pairs:
        clean = pair["clean"]
        distractor = pair["distractor"]
        if clean.get("target_text") != distractor.get("target_text"):
            raise ValueError("Clean/noop pair changed semantic target")
        agreements += int(clean.get("prediction_text") == distractor.get("prediction_text"))
        damages.append(float(_correct(clean) - _correct(distractor)))
    return {
        "n_pairs": len(pairs),
        "predicted_text_agreements": agreements,
        "predicted_text_agreement_rate": agreements / len(pairs),
        "accuracy_damage": statistics.fmean(damages),
    }


def descriptive_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    standard = [row for row in rows if row.get("condition", "standard") == "standard"]
    if not standard:
        raise ValueError("Baseline predictions contain no standard condition")
    margins = []
    predicted_margins = []
    for row in standard:
        scores = [float(value) for value in row["choice_mean_nll"]]
        target = int(row["target_index"])
        prediction = int(row["prediction_index"])
        margins.append(min(score for index, score in enumerate(scores) if index != target) - scores[target])
        predicted_margins.append(
            min(score for index, score in enumerate(scores) if index != prediction) - scores[prediction]
        )
    stale = [
        int(row["prediction_text"] == row["stale_target_text"] and row["prediction_text"] != row["target_text"])
        for row in standard
        if isinstance(row.get("stale_target_text"), str)
    ]
    donor_rows = [
        row
        for row in rows
        if row.get("condition") == "state_swap" and isinstance(row.get("donor_target_index"), int)
    ]
    # A matched pair can share the same target at an early query boundary. Such a
    # row is valid for the paired accuracy intervention, but it cannot distinguish
    # following the donor state from retaining the recipient answer. Keep the
    # all-row diagnostic, while defining the primary donor-answer rate only on
    # counterfactual rows whose donor and recipient targets differ.
    informative_donor_rows = [
        row for row in donor_rows if int(row["donor_target_index"]) != int(row["target_index"])
    ]
    donor = [
        int(row["prediction_index"] == row["donor_target_index"])
        for row in informative_donor_rows
    ]
    donor_all = [
        int(row["prediction_index"] == row["donor_target_index"])
        for row in donor_rows
    ]
    latency = [float(row["latency_seconds"]) for row in standard if row.get("latency_seconds") is not None]
    tokens = [float(row["history_token_count"]) for row in standard]
    history_bytes = [float(row["history_utf8_bytes"]) for row in standard]
    by_topic_subtype: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in standard:
        if row.get("topic") is not None and row.get("subtype") is not None:
            by_topic_subtype[(str(row["topic"]), str(row["subtype"]))].append(row)
    macro_values = [_accuracy_summary(values)["accuracy"] for values in by_topic_subtype.values()]
    return {
        "standard": _accuracy_summary(standard),
        "macro_topic_subtype_accuracy": statistics.fmean(macro_values) if macro_values else None,
        "macro_topic_subtype_cells": len(macro_values),
        "by_condition": _group_accuracy(rows, "condition"),
        "by_event_kind": _group_accuracy(standard, "subtype"),
        "by_ood_group": _group_accuracy(standard, "ood_group"),
        "by_split": _group_accuracy(standard, "split"),
        "by_target_position": _group_accuracy(standard, "target_index"),
        "by_probe_role": _group_accuracy(standard, "probe_role"),
        "rotation": _rotation_summary(rows),
        "paired_conditions": _condition_pairs(
            rows,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
        "stale_answer_error": {
            "n": len(stale),
            "count": sum(stale),
            "rate": statistics.fmean(stale) if stale else None,
        },
        "clean_noop": _noop_summary(rows),
        "state_swap_donor_answer": {
            "n": len(donor),
            "count": sum(donor),
            "rate": statistics.fmean(donor) if donor else None,
            "eligibility": "donor_target_index != target_index",
            "excluded_equal_target": len(donor_rows) - len(informative_donor_rows),
            "all_mapped_rows": {
                "n": len(donor_all),
                "count": sum(donor_all),
                "rate": statistics.fmean(donor_all) if donor_all else None,
            },
        },
        "nll_margin": {
            "target_vs_best_wrong": _numeric_summary(margins),
            "predicted_vs_runner_up": _numeric_summary(predicted_margins),
            "positive_target_margin_rate": statistics.fmean(value > 0 for value in margins),
        },
        "efficiency": {
            "condition_scope": "standard",
            "latency_seconds": _numeric_summary(latency),
            "history_token_count": _numeric_summary(tokens),
            "history_utf8_bytes": _numeric_summary(history_bytes),
            "peak_reader_vram_gib": max(
                (float(row.get("peak_reader_vram_gib", 0.0)) for row in standard),
                default=0.0,
            ),
            "peak_vram_gib": max(
                (float(row.get("peak_vram_gib", 0.0)) for row in standard),
                default=0.0,
            ),
            "constant_visual_input_bytes": sorted(
                {
                    int(row["constant_visual_input_bytes"])
                    for row in standard
                    if row.get("constant_visual_input_bytes") is not None
                }
            ),
        },
    }


def replication_report(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload_a = scientific_prediction_payload(rows_a)
    payload_b = scientific_prediction_payload(rows_b)
    identities_a = {prediction_identity(row) for row in rows_a}
    identities_b = {prediction_identity(row) for row in rows_b}
    return {
        "schema": REPLICATION_SCHEMA,
        "replica_a": payload_a,
        "replica_b": payload_b,
        "identity_sets_match": identities_a == identities_b,
        "bitwise_scientific_payload_match": payload_a == payload_b,
        "passed": identities_a == identities_b and payload_a == payload_b,
    }


def text_only_sensitivity_report(
    blank_rows: Sequence[Mapping[str, Any]],
    text_rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    blank_standard = {
        pair_identity(row): row for row in blank_rows if row.get("condition", "standard") == "standard"
    }
    text_standard = {
        pair_identity(row): row for row in text_rows if row.get("condition", "standard") == "standard"
    }
    if set(blank_standard) != set(text_standard):
        raise ValueError("Blank-image and text-only standard sensitivity rows are not exactly paired")
    differences: list[tuple[str, float]] = []
    prediction_agreements = 0
    for key in sorted(blank_standard, key=repr):
        blank = blank_standard[key]
        text = text_standard[key]
        group = blank.get("semantic_group_id") or blank.get("episode_id")
        differences.append((str(group), float(_correct(text) - _correct(blank))))
        prediction_agreements += int(text.get("prediction_text") == blank.get("prediction_text"))
    blank_gate = score_r3_micro(list(blank_rows), suite)
    text_gate = score_r3_micro(list(text_rows), suite)
    return {
        "schema": "vlm.qwen-history-baseline-text-only-micro-sensitivity.v1",
        "role": "micro_sensitivity_not_formal_baseline",
        "suite": suite,
        "blank_image": _accuracy_summary(list(blank_standard.values())),
        "text_only": _accuracy_summary(list(text_standard.values())),
        "text_only_minus_blank_accuracy": statistics.fmean(value for _, value in differences),
        "prediction_text_agreements": prediction_agreements,
        "prediction_text_agreement_rate": prediction_agreements / len(differences),
        "paired_bootstrap": _stratified_paired_bootstrap(
            differences,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "blank_micro_gate": blank_gate,
        "text_only_micro_gate": text_gate,
        "text_only_scientific_prediction_payload": scientific_prediction_payload(text_rows),
    }


def score_baseline(
    *,
    predictions: Path,
    prediction_report: Path,
    suite: str,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 2026,
    replica_b_predictions: Path | None = None,
    replica_b_report: Path | None = None,
    text_only_predictions: Path | None = None,
    text_only_report: Path | None = None,
) -> dict[str, Any]:
    rows = load_jsonl(predictions)
    validate_rows(rows)
    companion = validate_prediction_report(
        predictions=predictions,
        rows=rows,
        report_path=prediction_report,
    )
    metrics = descriptive_metrics(
        rows,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "suite": suite,
        "method": METHOD,
        "predictions": str(predictions.resolve()),
        "predictions_sha256": sha256_file(predictions),
        "prediction_report": str(prediction_report.resolve()),
        "prediction_report_sha256": sha256_file(prediction_report),
        "episodes_sha256": companion["episodes_sha256"],
        "reader_revision": companion["reader_revision"],
        "record_count": len(rows),
        "scientific_prediction_payload": scientific_prediction_payload(rows),
        "descriptive_metrics": metrics,
        "bootstrap": {"iterations": bootstrap_iterations, "seed": bootstrap_seed},
    }
    if suite in {"set8", "transition16"}:
        payload["micro_gate"] = score_r3_micro(list(rows), suite)
        payload["passed"] = bool(payload["micro_gate"]["passed"])
    elif suite == "formal":
        payload["passed"] = True
    else:
        raise ValueError("suite must be set8, transition16, or formal")
    if (replica_b_predictions is None) != (replica_b_report is None):
        raise ValueError("Replica B predictions and companion report must be supplied together")
    if replica_b_predictions is not None and replica_b_report is not None:
        rows_b = load_jsonl(replica_b_predictions)
        validate_rows(rows_b)
        companion_b = validate_prediction_report(
            predictions=replica_b_predictions,
            rows=rows_b,
            report_path=replica_b_report,
        )
        if companion_b["episodes_sha256"] != companion["episodes_sha256"]:
            raise ValueError("A/B replicas evaluated different episode artifacts")
        if companion_b["reader_revision"] != companion["reader_revision"]:
            raise ValueError("A/B replicas used different Reader revisions")
        payload["replica_b"] = {
            "predictions": str(replica_b_predictions.resolve()),
            "predictions_sha256": sha256_file(replica_b_predictions),
            "prediction_report": str(replica_b_report.resolve()),
            "prediction_report_sha256": sha256_file(replica_b_report),
        }
        payload["replication"] = replication_report(rows, rows_b)
        payload["passed"] = bool(payload["passed"] and payload["replication"]["passed"])
    if (text_only_predictions is None) != (text_only_report is None):
        raise ValueError("Text-only predictions and companion report must be supplied together")
    if text_only_predictions is not None and text_only_report is not None:
        if suite == "formal":
            raise ValueError("text_only is a micro sensitivity analysis and is prohibited in formal scoring")
        text_rows = load_jsonl(text_only_predictions)
        validate_rows(
            text_rows,
            expected_method=TEXT_ONLY_METHOD,
            expected_input_mode="text_only",
            expected_micro_sensitivity=True,
        )
        text_companion = validate_prediction_report(
            predictions=text_only_predictions,
            rows=text_rows,
            report_path=text_only_report,
            expected_method=TEXT_ONLY_METHOD,
            expected_input_mode="text_only",
            expected_micro_sensitivity=True,
        )
        if text_companion["episodes_sha256"] != companion["episodes_sha256"]:
            raise ValueError("Blank-image and text-only sensitivity runs used different episode artifacts")
        if text_companion["reader_revision"] != companion["reader_revision"]:
            raise ValueError("Blank-image and text-only sensitivity runs used different Reader revisions")
        payload["text_only_sensitivity"] = text_only_sensitivity_report(
            rows,
            text_rows,
            suite=suite,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
        payload["text_only_input"] = {
            "predictions": str(text_only_predictions.resolve()),
            "predictions_sha256": sha256_file(text_only_predictions),
            "prediction_report": str(text_only_report.resolve()),
            "prediction_report_sha256": sha256_file(text_only_report),
        }
    scientific = {
        "suite": payload["suite"],
        "method": payload["method"],
        "scientific_prediction_payload": payload["scientific_prediction_payload"],
        "micro_gate": payload.get("micro_gate"),
        "replication": payload.get("replication"),
    }
    payload["scientific_payload_sha256"] = canonical_sha256(scientific)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the locked Qwen full-event-history baseline")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path, required=True)
    parser.add_argument("--suite", choices=("set8", "transition16", "formal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replica-b-predictions", type=Path)
    parser.add_argument("--replica-b-report", type=Path)
    parser.add_argument("--text-only-predictions", type=Path)
    parser.add_argument("--text-only-report", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--fail-on-gate", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing to overwrite existing score report: {args.output}")
    report = score_baseline(
        predictions=args.predictions,
        prediction_report=args.prediction_report,
        suite=args.suite,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        replica_b_predictions=args.replica_b_predictions,
        replica_b_report=args.replica_b_report,
        text_only_predictions=args.text_only_predictions,
        text_only_report=args.text_only_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] or not args.fail_on_gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
