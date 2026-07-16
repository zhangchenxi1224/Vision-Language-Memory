from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train.lightweight_episode import (  # noqa: E402
    episode_value,
    event_payload,
    query_payload,
    training_subset_audit,
    turn_kind,
    validate_overfit_gate_episodes,
)
from vision_memory.data import read_jsonl  # noqa: E402
from vision_memory.dreamlite import assert_no_frozen_parameter_grads, freeze_module  # noqa: E402
from vision_memory.lightweight import LightweightVisualUpdater  # noqa: E402
from vision_memory.reader import (  # noqa: E402
    qwen3vl_choice_nll,
    qwen3vl_listwise_choice_ce,
    qwen3vl_target_only_ce,
)
from vision_memory.repro import (  # noqa: E402
    canonical_object_sha256,
    canonical_tensor_sha256,
    configure_strict_cuda_determinism,
    model_optimizer_rng_manifest,
    named_tensors_manifest,
)
from vision_memory.training import format_mcq_query, run_episode  # noqa: E402


EPISODE_COUNT = 64
SEED = 0
STATE_CHANNELS = 64
STATE_SIZE = 64
PRODUCTION_OUTPUT_SIZE = 256
DETERMINISTIC_READER_SIZE = 256
ALLOWED_STEP_COUNTS = (1, 100, 2000)
REACHABILITY_STEP_BUDGET = 2000
REACHABILITY_PREDICTION_COUNT = 128
REACHABILITY_MINIMUM_CORRECT = 116
REACHABILITY_MILESTONE_STEPS = (1, 2, 10, 100, 500, 1000, 1500, 1900, 2000)
READER_LOSS_MODES = ("target-only", "listwise-choice")
R2_CANONICAL_MINIMUM_CORRECT = 116
R2_ROTATED_MINIMUM_CORRECT = 116
R2_TARGET_POSITION_MINIMUM_CORRECT = 28
R2_TARGET_POSITION_EXPECTED_COUNT = 32
R2_MIXED_MINIMUM_CORRECT = 20
R2_MIXED_EXPECTED_COUNT = 24
R2_DISTRACTOR_PAIR_EXPECTED_COUNT = 64
R2_DISTRACTOR_AGREEMENT_MINIMUM = 60
EXPECTED_QWEN_IMAGE_GRID = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
}
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 5.0
GRADIENT_ACCUMULATION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bitwise reproducibility probe for the exact-64 lightweight updater")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=ALLOWED_STEP_COUNTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--reader-loss-mode",
        choices=READER_LOSS_MODES,
        default="target-only",
        help="Training objective; target-only preserves the historical R1/D2R protocol.",
    )
    return parser.parse_args()


def reader_objective_contract(reader_loss_mode: str) -> dict[str, Any]:
    """Return the fixed, canonical objective semantics for one protocol mode."""

    if reader_loss_mode == "target-only":
        return {
            "mode": "target-only",
            "historical_scope": "R1/D2R-only",
            "token_ce": "fp32-logsumexp-minus-target-token-score",
            "choice_ce": None,
            "query_loss": "mean-target-token-nll",
            "query_aggregation": "unweighted-mean-within-episode",
            "choice_training_scores": None,
            "choice_logit_temperature_float_hex": None,
            "evaluation_views": ["canonical"],
        }
    if reader_loss_mode == "listwise-choice":
        return {
            "mode": "listwise-choice",
            "historical_scope": "prospective-R2/D2L-only",
            "token_ce": "fp32-logsumexp-minus-target-token-score",
            "choice_ce": "fp32-logsumexp-minus-target-choice-logit",
            "query_loss": "fp32-logsumexp-minus-target-choice-logit",
            "query_aggregation": "unweighted-mean-within-episode",
            "choice_count": 4,
            "choice_forward_order": "stored-choice-order",
            "choice_token_reduction": "mean",
            "choice_training_scores": "negative-mean-token-nll",
            "choice_logit_temperature_float_hex": float(1.0).hex(),
            "evaluation_views": ["canonical", "left-rotate-one"],
        }
    raise ValueError(f"Unsupported reader_loss_mode: {reader_loss_mode!r}")


def evaluation_views_for_reader_loss_mode(reader_loss_mode: str) -> tuple[str, ...]:
    reader_objective_contract(reader_loss_mode)
    return ("canonical", "left-rotate-one") if reader_loss_mode == "listwise-choice" else ("canonical",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    if not marker.is_file() or not marker.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Reader has no non-empty revision lock: {marker}")
    return marker.read_text(encoding="utf-8").strip()


def validate_qwen_image_grid_contract(image_processor: Any, *, image_size: int) -> dict[str, int]:
    actual: dict[str, int] = {}
    for field, expected in EXPECTED_QWEN_IMAGE_GRID.items():
        value = getattr(image_processor, field, None)
        if value is None:
            raise RuntimeError(f"Qwen image processor does not expose required {field}.")
        actual[field] = int(value)
        if actual[field] != expected:
            raise RuntimeError(f"Qwen image processor {field} drifted: expected {expected}, got {actual[field]}.")
    spatial_factor = actual["patch_size"] * actual["merge_size"]
    if image_size % spatial_factor:
        raise RuntimeError(
            "Deterministic reader size must be divisible by patch_size * merge_size; "
            f"got image_size={image_size}, spatial_factor={spatial_factor}."
        )
    return {**actual, "spatial_factor": spatial_factor}


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def float_hex(value: float | torch.Tensor) -> str:
    number = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Canonical trace encountered a non-finite scalar: {number}")
    return number.hex()


def episode_schedule(count: int, steps: int) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    epoch = 0
    while len(schedule) < steps:
        order = list(range(count))
        random.Random((SEED << 16) ^ epoch).shuffle(order)
        schedule.extend((epoch, index) for index in order)
        epoch += 1
    return schedule[:steps]


def normalized_categorical_label(value: Any) -> str | None:
    value = getattr(value, "value", value)
    return None if value is None else str(value)


def grouped_prediction_summary(
    predictions: list[dict[str, Any]],
    *,
    event_kind_field: str = "event_kind",
) -> dict[str, dict[str, dict[str, Any]]]:
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for field in ("target_index", event_kind_field, "distractor_variant", "turn_type", "topic"):
        groups: dict[str, dict[str, int]] = {}
        for prediction in predictions:
            key = str(prediction[field])
            group = groups.setdefault(key, {"count": 0, "correct": 0})
            group["count"] += 1
            group["correct"] += int(prediction["correct"])
        summary[field] = {
            key: {
                **groups[key],
                "accuracy_float_hex": (groups[key]["correct"] / groups[key]["count"]).hex(),
            }
            for key in sorted(groups)
        }
    return summary


def reachability_gate_summary(
    *,
    steps: int,
    optimizer_steps_completed: int,
    predictions: list[dict[str, Any]],
    positive_gradient_steps: int,
    clipped_steps: int,
    reader_loss_mode: str = "target-only",
) -> dict[str, Any]:
    if reader_loss_mode not in READER_LOSS_MODES:
        raise ValueError(f"Unsupported reader_loss_mode: {reader_loss_mode!r}")
    applicable = reader_loss_mode == "target-only" and steps == REACHABILITY_STEP_BUDGET
    final_prediction_count = len(predictions)
    if final_prediction_count == 0:
        raise RuntimeError("Reachability evaluation produced no predictions.")
    final_correct = sum(int(record["correct"]) for record in predictions)
    reached_step_budget = optimizer_steps_completed == steps
    prediction_count_matches = final_prediction_count == REACHABILITY_PREDICTION_COUNT
    gradient_chain_valid = positive_gradient_steps == optimizer_steps_completed == steps
    threshold_reached = (
        prediction_count_matches
        and final_correct >= REACHABILITY_MINIMUM_CORRECT
        and final_correct * 10 >= final_prediction_count * 9
    )
    passed = (
        reached_step_budget and prediction_count_matches and gradient_chain_valid and threshold_reached
        if applicable
        else None
    )
    return {
        "applicable": applicable,
        "passed": passed,
        "reader_loss_mode": reader_loss_mode,
        "historical_scope": "R1/D2R target-only only",
        "step_budget": REACHABILITY_STEP_BUDGET,
        "optimizer_steps_completed": optimizer_steps_completed,
        "reached_step_budget": reached_step_budget,
        "expected_prediction_count": REACHABILITY_PREDICTION_COUNT,
        "final_prediction_count": final_prediction_count,
        "prediction_count_matches": prediction_count_matches,
        "minimum_correct": REACHABILITY_MINIMUM_CORRECT,
        "final_correct": final_correct,
        "final_accuracy_float_hex": (final_correct / final_prediction_count).hex(),
        "threshold_fraction": {"numerator": 9, "denominator": 10},
        "threshold_reached": threshold_reached,
        "positive_gradient_steps": positive_gradient_steps,
        "gradient_chain_valid": gradient_chain_valid,
        "clipped_steps": clipped_steps,
        "trace_values_finite": True,
        "reader_frozen_parameter_gradients": 0,
        "grouped_predictions": grouped_prediction_summary(predictions),
    }


def rotate_choices_left_one(
    choices: tuple[str, ...],
    target_index: int,
) -> tuple[tuple[str, ...], int]:
    if len(choices) != 4:
        raise ValueError("The left-rotate-one R2 view requires exactly four choices.")
    if isinstance(target_index, bool) or not isinstance(target_index, int) or not 0 <= target_index < 4:
        raise ValueError("target_index must be an integer in [0, 3].")
    return choices[1:] + choices[:1], (target_index - 1) % 4


def _prediction_view_identity(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("episode_id"),
        record.get("turn_id"),
        record.get("query_ordinal"),
        record.get("comparison_id"),
    )


def _prediction_view_alignment_summary(
    canonical_predictions: list[dict[str, Any]],
    rotated_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    def indexed(records: list[dict[str, Any]]) -> tuple[dict[tuple[Any, ...], dict[str, Any]], list[str]]:
        result: dict[tuple[Any, ...], dict[str, Any]] = {}
        duplicates: list[str] = []
        for record in records:
            identity = _prediction_view_identity(record)
            if identity in result:
                duplicates.append(repr(identity))
            else:
                result[identity] = record
        return result, sorted(duplicates)

    canonical_by_identity, canonical_duplicates = indexed(canonical_predictions)
    rotated_by_identity, rotated_duplicates = indexed(rotated_predictions)
    canonical_identities = set(canonical_by_identity)
    rotated_identities = set(rotated_by_identity)
    missing_rotated = sorted(repr(value) for value in canonical_identities - rotated_identities)
    unexpected_rotated = sorted(repr(value) for value in rotated_identities - canonical_identities)
    invalid: list[dict[str, Any]] = []
    matched_count = 0
    for identity in sorted(canonical_identities & rotated_identities, key=repr):
        canonical = canonical_by_identity[identity]
        rotated = rotated_by_identity[identity]
        reasons: list[str] = []
        canonical_choices = canonical.get("choices")
        rotated_choices = rotated.get("choices")
        canonical_target = canonical.get("target_index")
        rotated_target = rotated.get("target_index")
        canonical_predicted = canonical.get("predicted_index")
        rotated_predicted = rotated.get("predicted_index")
        if canonical.get("view") != "canonical":
            reasons.append("canonical-view-label")
        if rotated.get("view") != "left-rotate-one":
            reasons.append("rotated-view-label")
        if not isinstance(canonical_choices, list) or len(canonical_choices) != 4:
            reasons.append("canonical-choices")
        if not isinstance(rotated_choices, list) or len(rotated_choices) != 4:
            reasons.append("rotated-choices")
        if (
            isinstance(canonical_choices, list)
            and len(canonical_choices) == 4
            and isinstance(rotated_choices, list)
            and rotated_choices != canonical_choices[1:] + canonical_choices[:1]
        ):
            reasons.append("choices-not-left-rotated")
        if isinstance(canonical_target, bool) or not isinstance(canonical_target, int) or not 0 <= canonical_target < 4:
            reasons.append("canonical-target-index")
        elif rotated_target != (canonical_target - 1) % 4:
            reasons.append("target-index-not-synchronized")
        if canonical.get("target_text") != rotated.get("target_text"):
            reasons.append("target-text-changed")
        if isinstance(canonical_choices, list) and isinstance(canonical_target, int) and 0 <= canonical_target < 4:
            if canonical.get("target_text") != canonical_choices[canonical_target]:
                reasons.append("canonical-target-text-choice-mismatch")
        if isinstance(rotated_choices, list) and isinstance(rotated_target, int) and 0 <= rotated_target < 4:
            if rotated.get("target_text") != rotated_choices[rotated_target]:
                reasons.append("rotated-target-text-choice-mismatch")
        for label, record, predicted, choices in (
            ("canonical", canonical, canonical_predicted, canonical_choices),
            ("rotated", rotated, rotated_predicted, rotated_choices),
        ):
            if isinstance(predicted, bool) or not isinstance(predicted, int) or not 0 <= predicted < 4:
                reasons.append(f"{label}-predicted-index")
            elif isinstance(choices, list) and len(choices) == 4:
                if record.get("predicted_text") != choices[predicted]:
                    reasons.append(f"{label}-predicted-text-choice-mismatch")
            target = record.get("target_index")
            if isinstance(predicted, int) and isinstance(target, int):
                if record.get("correct") is not (predicted == target):
                    reasons.append(f"{label}-correct-flag-mismatch")
        for field in ("distractor_variant", "turn_type", "topic", "state_event_kind"):
            if canonical.get(field) != rotated.get(field):
                reasons.append(f"{field}-changed")
        if reasons:
            invalid.append({"identity": repr(identity), "reasons": sorted(set(reasons))})
        else:
            matched_count += 1
    expected_count = REACHABILITY_PREDICTION_COUNT
    passed = (
        len(canonical_predictions) == expected_count
        and len(rotated_predictions) == expected_count
        and not canonical_duplicates
        and not rotated_duplicates
        and not missing_rotated
        and not unexpected_rotated
        and not invalid
        and matched_count == expected_count
    )
    return {
        "expected_prediction_count": expected_count,
        "canonical_prediction_count": len(canonical_predictions),
        "rotated_prediction_count": len(rotated_predictions),
        "matched_prediction_count": matched_count,
        "canonical_duplicate_identities": canonical_duplicates,
        "rotated_duplicate_identities": rotated_duplicates,
        "missing_rotated_identities": missing_rotated,
        "unexpected_rotated_identities": unexpected_rotated,
        "invalid_pair_count": len(invalid),
        "invalid_pairs": invalid,
        "passed": passed,
    }


def _position_gate_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    groups = grouped_prediction_summary(predictions)["target_index"]
    positions: dict[str, dict[str, Any]] = {}
    for index in range(4):
        value = groups.get(str(index), {"count": 0, "correct": 0})
        count = int(value["count"])
        correct = int(value["correct"])
        positions[str(index)] = {
            "count": count,
            "correct": correct,
            "expected_count": R2_TARGET_POSITION_EXPECTED_COUNT,
            "minimum_correct": R2_TARGET_POSITION_MINIMUM_CORRECT,
            "passed": count == R2_TARGET_POSITION_EXPECTED_COUNT and correct >= R2_TARGET_POSITION_MINIMUM_CORRECT,
        }
    return {
        "positions": positions,
        "passed": all(value["passed"] for value in positions.values()),
    }


def _distractor_prediction_agreement(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    missing_comparison_ids = 0
    for prediction in predictions:
        comparison_id = prediction.get("comparison_id")
        if not isinstance(comparison_id, str) or not comparison_id:
            missing_comparison_ids += 1
            continue
        groups.setdefault(comparison_id, []).append(prediction)

    valid_pairs = 0
    predicted_text_agreements = 0
    invalid_pair_ids: list[str] = []
    for comparison_id in sorted(groups):
        members = groups[comparison_id]
        variants = {member.get("distractor_variant") for member in members}
        target_texts = {member.get("target_text") for member in members}
        choice_orders = {
            tuple(member["choices"])
            if isinstance(member.get("choices"), list) and len(member["choices"]) == 4
            else None
            for member in members
        }
        if (
            len(members) != 2
            or variants != {"clean", "distractor"}
            or len(target_texts) != 1
            or None in choice_orders
            or len(choice_orders) != 1
        ):
            invalid_pair_ids.append(comparison_id)
            continue
        valid_pairs += 1
        predicted_text_agreements += int(members[0].get("predicted_text") == members[1].get("predicted_text"))
    return {
        "expected_pair_count": R2_DISTRACTOR_PAIR_EXPECTED_COUNT,
        "valid_pair_count": valid_pairs,
        "invalid_pair_count": len(invalid_pair_ids),
        "invalid_pair_ids": invalid_pair_ids,
        "missing_comparison_id_count": missing_comparison_ids,
        "minimum_predicted_text_agreements": R2_DISTRACTOR_AGREEMENT_MINIMUM,
        "predicted_text_agreements": predicted_text_agreements,
        "passed": valid_pairs == R2_DISTRACTOR_PAIR_EXPECTED_COUNT
        and not invalid_pair_ids
        and missing_comparison_ids == 0
        and predicted_text_agreements >= R2_DISTRACTOR_AGREEMENT_MINIMUM,
    }


def listwise_gradient_trace_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    query_record_count = 0
    finite_query_record_count = 0
    positive_image_gradient_query_count = 0
    steps_with_positive_image_gradient = 0
    steps_with_query_records = 0
    positive_updater_gradient_steps = 0
    for step in trace:
        updater_grad_norm = step.get("gradient_norm_before_clip_float_hex")
        try:
            updater_grad_value = float.fromhex(updater_grad_norm) if isinstance(updater_grad_norm, str) else math.nan
        except ValueError:
            updater_grad_value = math.nan
        positive_updater_gradient_steps += int(math.isfinite(updater_grad_value) and updater_grad_value > 0.0)
        records = step.get("listwise_queries", [])
        if not isinstance(records, list):
            records = []
        steps_with_query_records += int(bool(records))
        step_has_positive_image_gradient = False
        for record in records:
            query_record_count += 1
            finite = bool(record.get("all_values_finite"))
            finite_query_record_count += int(finite)
            image_grad_norm = record.get("image_gradient_norm_float_hex")
            try:
                image_grad_value = float.fromhex(image_grad_norm) if isinstance(image_grad_norm, str) else math.nan
            except ValueError:
                image_grad_value = math.nan
            positive = math.isfinite(image_grad_value) and image_grad_value > 0.0
            positive_image_gradient_query_count += int(positive)
            step_has_positive_image_gradient = step_has_positive_image_gradient or positive
        steps_with_positive_image_gradient += int(step_has_positive_image_gradient)
    all_records_finite = query_record_count > 0 and finite_query_record_count == query_record_count
    return {
        "optimizer_step_count": len(trace),
        "steps_with_query_records": steps_with_query_records,
        "query_record_count": query_record_count,
        "finite_query_record_count": finite_query_record_count,
        "all_records_finite": all_records_finite,
        "positive_image_gradient_query_count": positive_image_gradient_query_count,
        "all_query_image_gradients_positive": query_record_count > 0
        and positive_image_gradient_query_count == query_record_count,
        "steps_with_positive_image_gradient": steps_with_positive_image_gradient,
        "every_step_has_positive_image_gradient": bool(trace) and steps_with_positive_image_gradient == len(trace),
        "positive_updater_gradient_steps": positive_updater_gradient_steps,
        "every_step_has_positive_updater_gradient": bool(trace)
        and positive_updater_gradient_steps == len(trace),
    }


def r2a_autograd_diagnostic(reader_loss_mode: str, trace: list[dict[str, Any]]) -> dict[str, Any]:
    if reader_loss_mode not in READER_LOSS_MODES:
        raise ValueError(f"Unsupported reader_loss_mode: {reader_loss_mode!r}")
    if reader_loss_mode == "target-only":
        return {
            "applicable": False,
            "passed": None,
            "scope": "R2a is prospective listwise-choice only; historical target-only is unchanged.",
        }
    summary = listwise_gradient_trace_summary(trace)
    passed = (
        summary["all_records_finite"]
        and summary["steps_with_query_records"] == len(trace)
        and summary["all_query_image_gradients_positive"]
        and summary["every_step_has_positive_image_gradient"]
        and summary["every_step_has_positive_updater_gradient"]
    )
    return {
        "applicable": True,
        "passed": passed,
        "scope": "listwise per-query image and updater autograd diagnostic; not a 1/100-step scientific gate",
        **summary,
    }


def r2_gate_summary(
    *,
    reader_loss_mode: str,
    steps: int,
    optimizer_steps_completed: int,
    canonical_predictions: list[dict[str, Any]],
    rotated_predictions: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    positive_gradient_steps: int,
    clipped_steps: int,
) -> dict[str, Any]:
    if reader_loss_mode not in READER_LOSS_MODES:
        raise ValueError(f"Unsupported reader_loss_mode: {reader_loss_mode!r}")
    if reader_loss_mode == "target-only":
        return {
            "applicable": False,
            "passed": None,
            "reader_loss_mode": reader_loss_mode,
            "historical_scope": "R2/D2L is prospective listwise-choice only",
            "step_budget": REACHABILITY_STEP_BUDGET,
        }
    applicable = reader_loss_mode == "listwise-choice" and steps == REACHABILITY_STEP_BUDGET
    canonical_correct = sum(int(record["correct"]) for record in canonical_predictions)
    rotated_correct = sum(int(record["correct"]) for record in rotated_predictions)
    canonical_count = len(canonical_predictions)
    rotated_count = len(rotated_predictions)
    canonical_threshold_reached = (
        canonical_count == REACHABILITY_PREDICTION_COUNT and canonical_correct >= R2_CANONICAL_MINIMUM_CORRECT
    )
    rotated_threshold_reached = (
        rotated_count == REACHABILITY_PREDICTION_COUNT and rotated_correct >= R2_ROTATED_MINIMUM_CORRECT
    )
    canonical_positions = _position_gate_summary(canonical_predictions)
    rotated_positions = _position_gate_summary(rotated_predictions)
    view_alignment = _prediction_view_alignment_summary(canonical_predictions, rotated_predictions)
    canonical_mixed = [record for record in canonical_predictions if record.get("turn_type") == "mixed"]
    canonical_mixed_correct = sum(int(record["correct"]) for record in canonical_mixed)
    canonical_mixed_passed = (
        len(canonical_mixed) == R2_MIXED_EXPECTED_COUNT and canonical_mixed_correct >= R2_MIXED_MINIMUM_CORRECT
    )
    distractor_agreement = _distractor_prediction_agreement(canonical_predictions)
    listwise_gradients = listwise_gradient_trace_summary(trace)
    reached_step_budget = optimizer_steps_completed == steps == REACHABILITY_STEP_BUDGET
    updater_gradient_chain_valid = positive_gradient_steps == optimizer_steps_completed == steps
    listwise_gradient_evidence_valid = (
        listwise_gradients["all_records_finite"]
        and listwise_gradients["steps_with_query_records"] == optimizer_steps_completed
        and listwise_gradients["every_step_has_positive_image_gradient"]
        and listwise_gradients["every_step_has_positive_updater_gradient"]
    )
    passed = (
        reached_step_budget
        and updater_gradient_chain_valid
        and listwise_gradient_evidence_valid
        and canonical_threshold_reached
        and rotated_threshold_reached
        and canonical_positions["passed"]
        and rotated_positions["passed"]
        and view_alignment["passed"]
        and canonical_mixed_passed
        and distractor_agreement["passed"]
        if applicable
        else None
    )
    return {
        "applicable": applicable,
        "passed": passed,
        "reader_loss_mode": reader_loss_mode,
        "historical_scope": "prospective R2/D2L listwise-choice only",
        "step_budget": REACHABILITY_STEP_BUDGET,
        "optimizer_steps_completed": optimizer_steps_completed,
        "reached_step_budget": reached_step_budget,
        "positive_gradient_steps": positive_gradient_steps,
        "updater_gradient_chain_valid": updater_gradient_chain_valid,
        "clipped_steps": clipped_steps,
        "view_alignment": view_alignment,
        "canonical": {
            "prediction_count": canonical_count,
            "correct": canonical_correct,
            "minimum_correct": R2_CANONICAL_MINIMUM_CORRECT,
            "threshold_reached": canonical_threshold_reached,
            "target_positions": canonical_positions,
            "mixed": {
                "count": len(canonical_mixed),
                "correct": canonical_mixed_correct,
                "expected_count": R2_MIXED_EXPECTED_COUNT,
                "minimum_correct": R2_MIXED_MINIMUM_CORRECT,
                "passed": canonical_mixed_passed,
            },
            "distractor_prediction_agreement": distractor_agreement,
            "grouped_predictions": grouped_prediction_summary(
                canonical_predictions,
                event_kind_field="state_event_kind",
            ),
        },
        "left_rotate_one": {
            "prediction_count": rotated_count,
            "correct": rotated_correct,
            "minimum_correct": R2_ROTATED_MINIMUM_CORRECT,
            "threshold_reached": rotated_threshold_reached,
            "target_positions": rotated_positions,
            "grouped_predictions": grouped_prediction_summary(
                rotated_predictions,
                event_kind_field="state_event_kind",
            ),
        },
        "listwise_gradient_evidence": listwise_gradients,
        "listwise_gradient_evidence_valid": listwise_gradient_evidence_valid,
        "trace_values_finite": listwise_gradients["all_records_finite"],
        "reader_frozen_parameter_gradients": 0,
    }


def gradient_manifest(model: torch.nn.Module) -> dict[str, Any]:
    missing = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(f"Trainable tensors have no gradient: {missing[:8]}")
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite:
        raise RuntimeError(f"Trainable tensors have non-finite gradients: {nonfinite[:8]}")
    return named_tensors_manifest(
        (name, parameter.grad) for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def module_gradient_norms(model: LightweightVisualUpdater) -> dict[str, str]:
    modules = {
        "event_encoder": model.event_encoder,
        "event_projection": model.event_projection,
        "event_spatial_projection": model.event_spatial_projection,
        "film": model.film,
        "cell": model.cell,
        "rgb_head": model.rgb_head,
    }
    result: dict[str, str] = {}
    for name, module in modules.items():
        gradients = [
            parameter.grad.detach().float()
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError(f"Module {name!r} has no gradients.")
        norm = torch.sqrt(torch.stack([gradient.square().sum() for gradient in gradients]).sum())
        result[name] = float_hex(norm)
    return result


def listwise_query_trace_records(captures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for ordinal, capture in enumerate(captures):
        output = capture["output"]
        image = capture["image"]
        target_index = int(capture["target_index"])
        choices = tuple(capture["choices"])
        if image.grad is None:
            raise RuntimeError(f"Listwise query {ordinal} produced no retained updater-image gradient.")
        choice_mean_nll = output.choice_mean_nll.detach().float()
        choice_logits = output.choice_logits.detach().float()
        if choice_mean_nll.shape != (4,) or choice_logits.shape != (4,):
            raise RuntimeError(
                "Listwise trace requires four scalar choice scores; "
                f"got nll={tuple(choice_mean_nll.shape)}, logits={tuple(choice_logits.shape)}."
            )
        log_probabilities = torch.log_softmax(choice_logits, dim=0)
        probabilities = torch.softmax(choice_logits, dim=0)
        entropy = -(probabilities * log_probabilities).sum()
        image_gradient_norm = image.grad.detach().float().norm()
        target_nll = choice_mean_nll[target_index]
        wrong_mask = torch.arange(4, device=choice_mean_nll.device) != target_index
        best_wrong_nll = choice_mean_nll[wrong_mask].min()
        margin = best_wrong_nll - target_nll
        ranking = sorted(
            range(4),
            key=lambda index: (float(choice_mean_nll[index].item()), index),
        )
        finite_values = torch.cat(
            [
                choice_mean_nll.reshape(-1),
                choice_logits.reshape(-1),
                output.loss.detach().float().reshape(-1),
                entropy.reshape(-1),
                image_gradient_norm.reshape(-1),
            ]
        )
        all_values_finite = bool(torch.isfinite(finite_values).all().item())
        if not all_values_finite:
            raise RuntimeError(f"Listwise query {ordinal} produced a non-finite trace value.")
        records.append(
            {
                "query_ordinal": ordinal,
                "choices": list(choices),
                "target_index": target_index,
                "target_text": choices[target_index],
                "choice_token_counts": list(output.choice_token_counts),
                "choice_mean_nll_tensor_sha256": canonical_tensor_sha256(choice_mean_nll),
                "choice_mean_nll_float_hex": [float_hex(value) for value in choice_mean_nll],
                "choice_logits_float_hex": [float_hex(value) for value in choice_logits],
                "listwise_loss_float_hex": float_hex(output.loss),
                "target_mean_nll_float_hex": float_hex(target_nll),
                "best_wrong_mean_nll_float_hex": float_hex(best_wrong_nll),
                "margin_float_hex": float_hex(margin),
                "target_rank": ranking.index(target_index) + 1,
                "choice_entropy_float_hex": float_hex(entropy),
                "image_gradient_norm_float_hex": float_hex(image_gradient_norm),
                "image_gradient_positive": bool(image_gradient_norm.item() > 0.0),
                "all_values_finite": all_values_finite,
            }
        )
    return records


def runtime_metadata(device: torch.device, determinism: dict[str, Any]) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "-L"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    nvidia_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "pid": os.getpid(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": importlib.metadata.version("transformers"),
        "gpu": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "uuid": str(getattr(properties, "uuid", "")) or None,
        },
        "nvidia_smi_L": nvidia_smi.stdout.strip() if nvidia_smi.returncode == 0 else None,
        "nvidia_smi_inventory": nvidia_query.stdout.strip() if nvidia_query.returncode == 0 else None,
        "determinism": determinism,
    }


def evaluate_canonical_predictions(
    *,
    episodes: list[Any],
    updater: LightweightVisualUpdater,
    reader: Any,
    processor: Any,
    device: torch.device,
    view: str = "canonical",
    include_r2_fields: bool = False,
) -> list[dict[str, Any]]:
    if view not in {"canonical", "left-rotate-one"}:
        raise ValueError("Prediction view must be 'canonical' or 'left-rotate-one'.")
    updater.eval()
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for episode in episodes:
            state = updater.initial_state(batch_size=1, device=device, dtype=torch.float32)
            query_ordinal = 0
            last_event_kind: str | None = None
            last_state_event_kind: str | None = None
            for turn_id, turn in enumerate(episode_value(episode, "turns")):
                kind = turn_kind(turn)
                if kind in {"event", "mixed"}:
                    event_text, last_event_kind = event_payload(turn)
                    if last_event_kind in {"set", "overwrite", "clear"}:
                        last_state_event_kind = last_event_kind
                    state = updater.update(state, event_text)
                if kind in {"query", "mixed"}:
                    query, choices, target_index, comparison_id = query_payload(turn)
                    if view == "left-rotate-one":
                        choices, target_index = rotate_choices_left_one(choices, target_index)
                    image = updater.render_deterministic_repro(state, target_size=DETERMINISTIC_READER_SIZE)[0]
                    score = qwen3vl_choice_nll(
                        model=reader,
                        processor=processor,
                        image=image,
                        query=format_mcq_query(query, choices),
                        choices=choices,
                        device=device,
                        do_resize=False,
                        deterministic_ce=True,
                    )
                    choice_nll = tuple(float(value) for value in score.mean_nll)
                    target_nll = choice_nll[target_index]
                    best_wrong_nll = min(value for index, value in enumerate(choice_nll) if index != target_index)
                    margin = best_wrong_nll - target_nll
                    ranking = sorted(range(len(choice_nll)), key=lambda index: (choice_nll[index], index))
                    predicted_index = score.predicted_index
                    record = {
                        "episode_id": str(episode_value(episode, "episode_id")),
                        "event_kind": last_event_kind,
                        "distractor_variant": normalized_categorical_label(
                            episode_value(episode, "distractor_variant")
                        ),
                        "topic": normalized_categorical_label(episode_value(episode, "topic")),
                        "turn_id": turn_id,
                        "turn_type": kind,
                        "query_ordinal": query_ordinal,
                        "comparison_id": comparison_id,
                        "target_index": target_index,
                        "predicted_index": predicted_index,
                        "correct": predicted_index == target_index,
                        "choice_mean_nll_float_hex": [float_hex(value) for value in choice_nll],
                    }
                    if include_r2_fields:
                        record.update(
                            {
                                "view": view,
                                "state_event_kind": last_state_event_kind,
                                "choices": list(choices),
                                "target_text": choices[target_index],
                                "predicted_text": choices[predicted_index],
                                "target_rank": ranking.index(target_index) + 1,
                                "target_mean_nll_float_hex": float_hex(target_nll),
                                "best_wrong_mean_nll_float_hex": float_hex(best_wrong_nll),
                                "margin_float_hex": float_hex(margin),
                            }
                        )
                    predictions.append(record)
                    query_ordinal += 1
    updater.train()
    return predictions


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    reader_loss_mode = str(args.reader_loss_mode)
    objective_contract = reader_objective_contract(reader_loss_mode)
    if not torch.cuda.is_available():
        raise RuntimeError("The bitwise lightweight reproducibility probe requires CUDA.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("--device must select CUDA.")
    git_commit = git_value("rev-parse", "HEAD")
    git_status = git_value("status", "--porcelain=v1", "--untracked-files=all")
    if git_commit is None or git_status is None:
        raise RuntimeError("The reproducibility probe requires an inspectable Git worktree.")
    if git_status:
        raise RuntimeError("The reproducibility probe refuses a dirty Git worktree.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("The reproducibility probe refuses a non-empty --output-dir.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    determinism = configure_strict_cuda_determinism(SEED)
    episodes = list(read_jsonl(args.train))[:EPISODE_COUNT]
    validate_overfit_gate_episodes(episodes)
    subset = training_subset_audit(episodes)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=DETERMINISTIC_READER_SIZE * DETERMINISTIC_READER_SIZE,
        max_pixels=DETERMINISTIC_READER_SIZE * DETERMINISTIC_READER_SIZE,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("The deterministic probe requires a tensor-native fast Qwen processor.")
    qwen_image_grid = validate_qwen_image_grid_contract(
        processor.image_processor,
        image_size=DETERMINISTIC_READER_SIZE,
    )
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False

    updater = LightweightVisualUpdater(
        state_channels=STATE_CHANNELS,
        state_size=STATE_SIZE,
        output_size=PRODUCTION_OUTPUT_SIZE,
        learned_initial_state=False,
    ).to(device=device, dtype=torch.float32)
    trainable = [parameter for parameter in updater.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )
    initial = model_optimizer_rng_manifest(updater, optimizer)

    def update_fn(state: torch.Tensor, event_text: str, _episode_id: str, _turn_id: str | int) -> torch.Tensor:
        return updater.update(state, event_text)

    def reader_loss(image: torch.Tensor, query: str, target: str):
        return qwen3vl_target_only_ce(
            model=reader,
            processor=processor,
            image=image[0],
            query=query,
            target=target,
            device=device,
            require_image_grad=True,
            do_resize=False,
            deterministic_ce=True,
        )

    listwise_captures: list[dict[str, Any]] = []

    def choice_reader_loss(
        image: torch.Tensor,
        query: str,
        choices: tuple[str, ...],
        target_index: int,
    ):
        if not image.requires_grad or image.grad_fn is None:
            raise RuntimeError("Listwise Reader received an updater image without an autograd path.")
        image.retain_grad()
        output = qwen3vl_listwise_choice_ce(
            model=reader,
            processor=processor,
            image=image[0],
            query=query,
            choices=choices,
            target_index=target_index,
            device=device,
            require_image_grad=True,
            do_resize=False,
            deterministic_ce=True,
        )
        listwise_captures.append(
            {
                "output": output,
                "image": image,
                "choices": choices,
                "target_index": target_index,
            }
        )
        return output

    trace: list[dict[str, Any]] = []
    milestones: dict[str, Any] = {}
    step_one_gradients: dict[str, Any] | None = None
    milestone_steps = {step for step in REACHABILITY_MILESTONE_STEPS if step <= args.steps} | {args.steps}
    positive_gradient_steps = 0
    clipped_steps = 0
    schedule = episode_schedule(len(episodes), args.steps)
    schedule_records = [
        {
            "optimizer_step": optimizer_step,
            "epoch": epoch,
            "episode_index": episode_index,
            "episode_id": str(episode_value(episodes[episode_index], "episode_id")),
        }
        for optimizer_step, (epoch, episode_index) in enumerate(schedule, start=1)
    ]
    torch.cuda.reset_peak_memory_stats(device)
    for optimizer_step, (epoch, episode_index) in enumerate(
        schedule,
        start=1,
    ):
        episode = episodes[episode_index]
        optimizer.zero_grad(set_to_none=True)
        listwise_captures.clear()
        episode_kwargs: dict[str, Any] = {
            "episode": episode,
            "initial_state": updater.initial_state(batch_size=1, device=device, dtype=torch.float32),
            "update_fn": update_fn,
            "decode_fn": lambda state: updater.render_deterministic_repro(state, target_size=DETERMINISTIC_READER_SIZE),
            "reader_loss_mode": reader_loss_mode,
            "noop_policy": "update",
            "collect_states": False,
        }
        if reader_loss_mode == "target-only":
            episode_kwargs["reader_loss_fn"] = reader_loss
        else:
            episode_kwargs["choice_reader_loss_fn"] = choice_reader_loss
        result = run_episode(
            **episode_kwargs,
        )
        if not torch.isfinite(result.loss):
            raise RuntimeError(f"Non-finite loss at optimizer step {optimizer_step}.")
        result.loss.backward()
        assert_no_frozen_parameter_grads(reader, "Qwen Reader")
        listwise_records = (
            listwise_query_trace_records(listwise_captures) if reader_loss_mode == "listwise-choice" else []
        )
        if reader_loss_mode == "listwise-choice" and len(listwise_records) != result.query_count:
            raise RuntimeError(
                "Listwise closure/query count mismatch: "
                f"captured {len(listwise_records)}, episode executed {result.query_count}."
            )
        listwise_captures.clear()
        raw_gradients = gradient_manifest(updater)
        raw_module_gradient_norms = module_gradient_norms(updater)
        norm_before_clip = torch.nn.utils.clip_grad_norm_(
            trainable,
            GRADIENT_CLIP,
            error_if_nonfinite=True,
            foreach=False,
        )
        clipped_gradients = gradient_manifest(updater)
        clipped_module_gradient_norms = module_gradient_norms(updater)
        norm_before_clip_value = float(norm_before_clip.item())
        clipping_factor = min(1.0, GRADIENT_CLIP / (norm_before_clip_value + 1e-6))
        positive_gradient_steps += int(norm_before_clip_value > 0.0)
        clipped_steps += int(clipping_factor < 1.0)
        if optimizer_step == 1:
            step_one_gradients = {
                "raw": raw_gradients,
                "clipped": clipped_gradients,
            }
        optimizer.step()
        trace_record = {
            "optimizer_step": optimizer_step,
            "epoch": epoch,
            "episode_index": episode_index,
            "episode_id": str(episode_value(episode, "episode_id")),
            "loss_tensor_sha256": canonical_tensor_sha256(result.loss.detach()),
            "loss_float_hex": float_hex(result.loss),
            "gradient_norm_before_clip_float_hex": float_hex(norm_before_clip),
            "gradient_clipping_factor_float_hex": clipping_factor.hex(),
            "raw_gradient_bundle_sha256": raw_gradients["bundle_sha256"],
            "clipped_gradient_bundle_sha256": clipped_gradients["bundle_sha256"],
            "raw_module_gradient_norms_float_hex": raw_module_gradient_norms,
            "clipped_module_gradient_norms_float_hex": clipped_module_gradient_norms,
        }
        if reader_loss_mode == "listwise-choice":
            trace_record["reader_loss_mode"] = reader_loss_mode
            trace_record["listwise_queries"] = listwise_records
        trace.append(trace_record)
        if optimizer_step in milestone_steps:
            milestones[str(optimizer_step)] = model_optimizer_rng_manifest(updater, optimizer)

    if step_one_gradients is None:
        raise RuntimeError("The probe did not execute optimizer step 1.")
    if len(trace) != args.steps:
        raise RuntimeError(f"Expected {args.steps} trace records, got {len(trace)}.")

    predictions = evaluate_canonical_predictions(
        episodes=episodes,
        updater=updater,
        reader=reader,
        processor=processor,
        device=device,
        view="canonical",
        include_r2_fields=reader_loss_mode == "listwise-choice",
    )
    rotated_predictions = (
        evaluate_canonical_predictions(
            episodes=episodes,
            updater=updater,
            reader=reader,
            processor=processor,
            device=device,
            view="left-rotate-one",
            include_r2_fields=True,
        )
        if "left-rotate-one" in evaluation_views_for_reader_loss_mode(reader_loss_mode)
        else []
    )
    assert_no_frozen_parameter_grads(reader, "Qwen Reader")
    trace_path = args.output_dir / "canonical_trace.jsonl"
    predictions_path = args.output_dir / "canonical_predictions.jsonl"
    write_jsonl(trace_path, trace)
    write_jsonl(predictions_path, predictions)
    rotated_predictions_path: Path | None = None
    if rotated_predictions:
        rotated_predictions_path = args.output_dir / "left_rotate_one_predictions.jsonl"
        write_jsonl(rotated_predictions_path, rotated_predictions)
    runtime = runtime_metadata(device, determinism)
    reachability_gate = reachability_gate_summary(
        steps=args.steps,
        optimizer_steps_completed=len(trace),
        predictions=predictions,
        positive_gradient_steps=positive_gradient_steps,
        clipped_steps=clipped_steps,
        reader_loss_mode=reader_loss_mode,
    )
    r2a_diagnostic = r2a_autograd_diagnostic(reader_loss_mode, trace)
    r2_gate = r2_gate_summary(
        reader_loss_mode=reader_loss_mode,
        steps=args.steps,
        optimizer_steps_completed=len(trace),
        canonical_predictions=predictions,
        rotated_predictions=rotated_predictions,
        trace=trace,
        positive_gradient_steps=positive_gradient_steps,
        clipped_steps=clipped_steps,
    )

    if reachability_gate["applicable"]:
        run_purpose = "strict-deterministic-exact64-reachability"
    elif r2_gate["applicable"]:
        run_purpose = "strict-deterministic-exact64-listwise-r2"
    else:
        run_purpose = "bitwise-reproducibility-audit"
    protocol = {
        "schema_version": "vision_memory.lightweight_determinism_protocol.v4",
        "run_purpose": run_purpose,
        "reader_loss_mode": reader_loss_mode,
        "reader_objective": objective_contract,
        "episode_count": EPISODE_COUNT,
        "seed": SEED,
        "steps": args.steps,
        "state_channels": STATE_CHANNELS,
        "state_size": STATE_SIZE,
        "production_output_size": PRODUCTION_OUTPUT_SIZE,
        "deterministic_reader_size": DETERMINISTIC_READER_SIZE,
        "renderer": "integer-repeat-without-crop",
        "qwen_do_resize": False,
        "qwen_image_grid": qwen_image_grid,
        "reader_ce": (
            "fp32-logsumexp-minus-target-score"
            if reader_loss_mode == "target-only"
            else "four-choice-mean-token-nll-plus-fp32-listwise-logsumexp"
        ),
        "attention": "sdpa-math-only",
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate_float_hex": LEARNING_RATE.hex(),
        "weight_decay_float_hex": WEIGHT_DECAY.hex(),
        "gradient_clip_float_hex": GRADIENT_CLIP.hex(),
        "optimizer": "AdamW(foreach=False)",
        "dtype": str(dtype).removeprefix("torch."),
        "determinism": determinism,
        "milestone_steps": sorted(milestone_steps),
    }
    prediction_views: dict[str, Any] = {
        "canonical": {
            "schema": "stored-choice-order",
            "predictions_sha256": canonical_object_sha256(predictions),
            "predictions_file_sha256": sha256_file(predictions_path),
            "prediction_count": len(predictions),
            "correct": sum(int(record["correct"]) for record in predictions),
        }
    }
    if rotated_predictions_path is not None:
        prediction_views["left_rotate_one"] = {
            "schema": "choices-left-rotated-one-and-target-index-synchronized",
            "predictions_sha256": canonical_object_sha256(rotated_predictions),
            "predictions_file_sha256": sha256_file(rotated_predictions_path),
            "prediction_count": len(rotated_predictions),
            "correct": sum(int(record["correct"]) for record in rotated_predictions),
        }
    comparison_payload = {
        "protocol": protocol,
        "reader_loss_mode": reader_loss_mode,
        "reader_objective": objective_contract,
        "git": {"commit": git_commit, "clean": True},
        "runtime_fingerprint": {
            key: runtime[key]
            for key in (
                "hostname",
                "slurm_job_id",
                "cuda_visible_devices",
                "python",
                "torch",
                "cuda_runtime",
                "cudnn",
                "transformers",
                "gpu",
                "nvidia_smi_L",
                "nvidia_smi_inventory",
            )
        },
        "train_sha256": sha256_file(args.train),
        "train_subset": subset,
        "reader_revision": locked_revision(args.reader),
        "training_schedule_count": len(schedule_records),
        "training_schedule_sha256": canonical_object_sha256(schedule_records),
        "initial": initial,
        "step_one_gradients": step_one_gradients,
        "trace": trace,
        "trace_sha256": canonical_object_sha256(trace),
        "trace_file_sha256": sha256_file(trace_path),
        "milestones": milestones,
        "final_predictions_sha256": canonical_object_sha256(predictions),
        "final_predictions_file_sha256": sha256_file(predictions_path),
        "final_prediction_count": len(predictions),
        "final_correct": sum(int(record["correct"]) for record in predictions),
        "prediction_views": prediction_views,
        "r2a_autograd_diagnostic": r2a_diagnostic,
        "reachability_gate": reachability_gate,
        "r2_gate": r2_gate,
    }
    report = {
        "schema_version": "vision_memory.lightweight_determinism_report.v3",
        "status": "complete",
        "reader_loss_mode": reader_loss_mode,
        "reader_objective": objective_contract,
        "r2a_autograd_diagnostic": r2a_diagnostic,
        "reachability_gate": reachability_gate,
        "r2_gate": r2_gate,
        "comparison_payload_sha256": canonical_object_sha256(comparison_payload),
        "comparison_payload": comparison_payload,
        "runtime": runtime,
        "provenance": {
            "git_commit": git_commit,
            "git_clean": True,
            "train": str(args.train.resolve()),
            "reader": str(args.reader.resolve()),
            "output_dir": str(args.output_dir.resolve()),
        },
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    return report


def main() -> int:
    args = parse_args()
    refused_preexisting_output = args.output_dir.exists() and any(args.output_dir.iterdir())
    try:
        report = run_probe(args)
    except Exception as error:
        if not refused_preexisting_output:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": "vision_memory.lightweight_determinism_report.v3",
                "status": "failed",
                "reader_loss_mode": getattr(args, "reader_loss_mode", None),
                "error_type": type(error).__name__,
                "error": str(error),
            }
            write_json(args.output_dir / "report.json", failure)
        raise
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
