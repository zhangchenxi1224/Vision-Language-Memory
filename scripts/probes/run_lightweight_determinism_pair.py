from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.repro import (  # noqa: E402
    assert_determinism_environment,
    canonical_object_sha256,
    compare_bitwise_repro_reports,
)
from scripts.probes.lightweight_determinism import (  # noqa: E402
    READER_LOSS_MODES,
    reader_objective_contract,
)


PROBE = ROOT / "scripts" / "probes" / "lightweight_determinism.py"
ALLOWED_STEP_COUNTS = (1, 100, 2000)
REACHABILITY_STEP_BUDGET = 2000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two fresh lightweight-determinism processes serially in one Slurm allocation"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=ALLOWED_STEP_COUNTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reader-loss-mode", choices=READER_LOSS_MODES, default="target-only")
    return parser.parse_args()


def read_report(path: Path, *, returncode: int) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "failed",
            "error": "child produced no report.json",
            "returncode": returncode,
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "failed",
            "error": f"child report is unreadable: {error}",
            "returncode": returncode,
        }
    report["wrapper_observed_returncode"] = returncode
    if returncode != 0 and report.get("status") == "complete":
        report["status"] = "failed"
        report["error"] = "child returned non-zero despite a complete report"
    return report


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_reachability_gate_contract(
    gate: Any,
    *,
    reader_loss_mode: str,
    steps: int,
) -> dict[str, Any]:
    expected_applicable = reader_loss_mode == "target-only" and steps == REACHABILITY_STEP_BUDGET
    checks: dict[str, bool] = {
        "is_mapping": isinstance(gate, dict),
    }
    if not isinstance(gate, dict):
        return {"valid": False, "checks": checks}
    checks.update(
        {
            "mode_matches": gate.get("reader_loss_mode") == reader_loss_mode,
            "applicability_matches": gate.get("applicable") is expected_applicable,
            "step_budget_matches": gate.get("step_budget") == REACHABILITY_STEP_BUDGET,
            "passed_is_none_when_inapplicable": expected_applicable or gate.get("passed") is None,
        }
    )
    if not expected_applicable:
        return {"valid": all(checks.values()), "checks": checks}

    optimizer_steps = gate.get("optimizer_steps_completed")
    prediction_count = gate.get("final_prediction_count")
    final_correct = gate.get("final_correct")
    positive_gradient_steps = gate.get("positive_gradient_steps")
    clipped_steps = gate.get("clipped_steps")
    reached_step_budget = optimizer_steps == steps == REACHABILITY_STEP_BUDGET
    prediction_count_matches = prediction_count == 128
    threshold_reached = (
        prediction_count_matches
        and _integer(final_correct)
        and final_correct >= 116
        and final_correct * 10 >= prediction_count * 9
    )
    positive_gradient_steps_in_range = (
        _integer(positive_gradient_steps)
        and _integer(optimizer_steps)
        and 0 <= positive_gradient_steps <= optimizer_steps
    )
    gradient_chain_valid = positive_gradient_steps == optimizer_steps == steps
    recomputed_passed = reached_step_budget and prediction_count_matches and gradient_chain_valid and threshold_reached
    checks.update(
        {
            "optimizer_steps_valid": optimizer_steps == steps,
            "reached_step_budget_valid": gate.get("reached_step_budget") is reached_step_budget,
            "expected_prediction_count_valid": gate.get("expected_prediction_count") == 128,
            "prediction_count_valid": prediction_count_matches,
            "prediction_count_flag_valid": gate.get("prediction_count_matches") is prediction_count_matches,
            "minimum_correct_valid": gate.get("minimum_correct") == 116,
            "final_correct_valid": _integer(final_correct)
            and _integer(prediction_count)
            and 0 <= final_correct <= prediction_count,
            "threshold_fraction_valid": gate.get("threshold_fraction") == {"numerator": 9, "denominator": 10},
            "threshold_flag_valid": gate.get("threshold_reached") is threshold_reached,
            "positive_gradient_steps_valid": positive_gradient_steps_in_range,
            "gradient_chain_flag_valid": gate.get("gradient_chain_valid") is gradient_chain_valid,
            "clipped_steps_valid": _integer(clipped_steps) and 0 <= clipped_steps <= steps,
            "trace_values_finite": gate.get("trace_values_finite") is True,
            "reader_frozen": gate.get("reader_frozen_parameter_gradients") == 0,
            "grouped_predictions_present": isinstance(gate.get("grouped_predictions"), dict),
            "passed_recomputes": gate.get("passed") is recomputed_passed,
        }
    )
    return {"valid": all(checks.values()), "checks": checks, "recomputed_passed": recomputed_passed}


def _validate_position_summary(value: Any) -> tuple[bool, bool]:
    if not isinstance(value, dict) or not isinstance(value.get("positions"), dict):
        return False, False
    positions = value["positions"]
    if set(positions) != {"0", "1", "2", "3"}:
        return False, False
    position_passes: list[bool] = []
    structurally_valid = True
    for index in range(4):
        item = positions[str(index)]
        if not isinstance(item, dict):
            return False, False
        count = item.get("count")
        correct = item.get("correct")
        passed = count == 32 and _integer(correct) and correct >= 28
        structurally_valid = structurally_valid and (
            count == 32
            and _integer(correct)
            and 0 <= correct <= count
            and item.get("expected_count") == 32
            and item.get("minimum_correct") == 28
            and item.get("passed") is passed
        )
        position_passes.append(passed)
    recomputed = all(position_passes)
    structurally_valid = structurally_valid and value.get("passed") is recomputed
    return structurally_valid, recomputed


def _validate_view_alignment_summary(value: Any) -> tuple[bool, bool]:
    if not isinstance(value, dict):
        return False, False
    canonical_duplicates = value.get("canonical_duplicate_identities")
    rotated_duplicates = value.get("rotated_duplicate_identities")
    missing_rotated = value.get("missing_rotated_identities")
    unexpected_rotated = value.get("unexpected_rotated_identities")
    invalid_pairs = value.get("invalid_pairs")
    invalid_pair_count = value.get("invalid_pair_count")
    lists_valid = all(
        isinstance(item, list)
        for item in (
            canonical_duplicates,
            rotated_duplicates,
            missing_rotated,
            unexpected_rotated,
            invalid_pairs,
        )
    )
    passed = bool(
        value.get("expected_prediction_count") == 128
        and value.get("canonical_prediction_count") == 128
        and value.get("rotated_prediction_count") == 128
        and value.get("matched_prediction_count") == 128
        and lists_valid
        and not canonical_duplicates
        and not rotated_duplicates
        and not missing_rotated
        and not unexpected_rotated
        and invalid_pair_count == 0
        and not invalid_pairs
    )
    structurally_valid = bool(
        lists_valid
        and _integer(invalid_pair_count)
        and invalid_pair_count == len(invalid_pairs)
        and value.get("passed") is passed
    )
    return structurally_valid, passed


def validate_r2_gate_contract(
    gate: Any,
    *,
    reader_loss_mode: str,
    steps: int,
) -> dict[str, Any]:
    expected_applicable = reader_loss_mode == "listwise-choice" and steps == REACHABILITY_STEP_BUDGET
    checks: dict[str, bool] = {"is_mapping": isinstance(gate, dict)}
    if not isinstance(gate, dict):
        return {"valid": False, "checks": checks}
    checks.update(
        {
            "mode_matches": gate.get("reader_loss_mode") == reader_loss_mode,
            "applicability_matches": gate.get("applicable") is expected_applicable,
            "step_budget_matches": gate.get("step_budget") == REACHABILITY_STEP_BUDGET,
            "passed_is_none_when_inapplicable": expected_applicable or gate.get("passed") is None,
        }
    )
    if not expected_applicable:
        return {"valid": all(checks.values()), "checks": checks}

    canonical = gate.get("canonical")
    rotated = gate.get("left_rotate_one")
    gradients = gate.get("listwise_gradient_evidence")
    if not isinstance(canonical, dict) or not isinstance(rotated, dict) or not isinstance(gradients, dict):
        checks["required_nested_summaries_present"] = False
        return {"valid": False, "checks": checks}

    canonical_position_valid, canonical_position_passed = _validate_position_summary(canonical.get("target_positions"))
    rotated_position_valid, rotated_position_passed = _validate_position_summary(rotated.get("target_positions"))
    view_alignment_valid, view_alignment_passed = _validate_view_alignment_summary(gate.get("view_alignment"))
    canonical_count = canonical.get("prediction_count")
    canonical_correct = canonical.get("correct")
    rotated_count = rotated.get("prediction_count")
    rotated_correct = rotated.get("correct")
    canonical_threshold = canonical_count == 128 and _integer(canonical_correct) and canonical_correct >= 116
    rotated_threshold = rotated_count == 128 and _integer(rotated_correct) and rotated_correct >= 116

    mixed = canonical.get("mixed")
    agreement = canonical.get("distractor_prediction_agreement")
    if not isinstance(mixed, dict) or not isinstance(agreement, dict):
        checks["required_canonical_safeguards_present"] = False
        return {"valid": False, "checks": checks}
    mixed_passed = mixed.get("count") == 24 and _integer(mixed.get("correct")) and mixed["correct"] >= 20
    agreement_passed = (
        agreement.get("valid_pair_count") == 64
        and agreement.get("invalid_pair_count") == 0
        and agreement.get("missing_comparison_id_count") == 0
        and _integer(agreement.get("predicted_text_agreements"))
        and agreement["predicted_text_agreements"] >= 60
    )
    optimizer_steps = gate.get("optimizer_steps_completed")
    positive_gradient_steps = gate.get("positive_gradient_steps")
    reached_step_budget = optimizer_steps == steps == REACHABILITY_STEP_BUDGET
    positive_gradient_steps_in_range = (
        _integer(positive_gradient_steps)
        and _integer(optimizer_steps)
        and 0 <= positive_gradient_steps <= optimizer_steps
    )
    updater_gradient_chain_valid = positive_gradient_steps == optimizer_steps == steps
    query_count = gradients.get("query_record_count")
    finite_query_count = gradients.get("finite_query_record_count")
    positive_query_count = gradients.get("positive_image_gradient_query_count")
    steps_with_queries = gradients.get("steps_with_query_records")
    steps_with_positive = gradients.get("steps_with_positive_image_gradient")
    positive_updater_steps = gradients.get("positive_updater_gradient_steps")
    all_records_finite = _integer(query_count) and query_count > 0 and finite_query_count == query_count
    all_query_gradients_positive = _integer(query_count) and query_count > 0 and positive_query_count == query_count
    every_step_has_positive = steps_with_positive == steps
    every_step_has_positive_updater = positive_updater_steps == steps
    listwise_gradient_evidence_valid = (
        gradients.get("optimizer_step_count") == steps
        and steps_with_queries == steps
        and all_records_finite
        and every_step_has_positive
        and every_step_has_positive_updater
    )
    recomputed_passed = (
        reached_step_budget
        and updater_gradient_chain_valid
        and listwise_gradient_evidence_valid
        and canonical_threshold
        and rotated_threshold
        and canonical_position_passed
        and rotated_position_passed
        and view_alignment_passed
        and mixed_passed
        and agreement_passed
    )
    clipped_steps = gate.get("clipped_steps")
    checks.update(
        {
            "required_nested_summaries_present": True,
            "optimizer_steps_valid": optimizer_steps == steps,
            "reached_step_budget_valid": gate.get("reached_step_budget") is reached_step_budget,
            "positive_gradient_steps_valid": positive_gradient_steps_in_range,
            "updater_gradient_flag_valid": gate.get("updater_gradient_chain_valid") is updater_gradient_chain_valid,
            "clipped_steps_valid": _integer(clipped_steps) and 0 <= clipped_steps <= steps,
            "canonical_count_and_correct_valid": canonical_count == 128
            and _integer(canonical_correct)
            and 0 <= canonical_correct <= canonical_count,
            "canonical_threshold_fields_valid": canonical.get("minimum_correct") == 116
            and canonical.get("threshold_reached") is canonical_threshold,
            "rotated_count_and_correct_valid": rotated_count == 128
            and _integer(rotated_correct)
            and 0 <= rotated_correct <= rotated_count,
            "rotated_threshold_fields_valid": rotated.get("minimum_correct") == 116
            and rotated.get("threshold_reached") is rotated_threshold,
            "canonical_position_summary_valid": canonical_position_valid,
            "rotated_position_summary_valid": rotated_position_valid,
            "view_alignment_summary_valid": view_alignment_valid,
            "mixed_fields_valid": mixed.get("expected_count") == 24
            and mixed.get("minimum_correct") == 20
            and mixed.get("count") == 24
            and _integer(mixed.get("correct"))
            and 0 <= mixed["correct"] <= mixed["count"]
            and mixed.get("passed") is mixed_passed,
            "agreement_fields_valid": agreement.get("expected_pair_count") == 64
            and agreement.get("minimum_predicted_text_agreements") == 60
            and _integer(agreement.get("valid_pair_count"))
            and 0 <= agreement["valid_pair_count"] <= 64
            and _integer(agreement.get("invalid_pair_count"))
            and agreement["invalid_pair_count"] >= 0
            and _integer(agreement.get("missing_comparison_id_count"))
            and agreement["missing_comparison_id_count"] >= 0
            and _integer(agreement.get("predicted_text_agreements"))
            and 0 <= agreement["predicted_text_agreements"] <= agreement["valid_pair_count"]
            and isinstance(agreement.get("invalid_pair_ids"), list)
            and len(agreement["invalid_pair_ids"]) == agreement.get("invalid_pair_count")
            and agreement.get("passed") is agreement_passed,
            "grouped_predictions_present": isinstance(canonical.get("grouped_predictions"), dict)
            and isinstance(rotated.get("grouped_predictions"), dict),
            "gradient_evidence_fields_valid": gradients.get("optimizer_step_count") == steps
            and _integer(steps_with_queries)
            and 0 <= steps_with_queries <= steps
            and _integer(query_count)
            and query_count > 0
            and _integer(finite_query_count)
            and 0 <= finite_query_count <= query_count
            and _integer(positive_query_count)
            and 0 <= positive_query_count <= query_count
            and _integer(steps_with_positive)
            and 0 <= steps_with_positive <= steps
            and _integer(positive_updater_steps)
            and 0 <= positive_updater_steps <= steps
            and gradients.get("all_records_finite") is all_records_finite
            and gradients.get("all_query_image_gradients_positive") is all_query_gradients_positive
            and gradients.get("every_step_has_positive_image_gradient") is every_step_has_positive
            and gradients.get("every_step_has_positive_updater_gradient") is every_step_has_positive_updater
            and gate.get("listwise_gradient_evidence_valid") is listwise_gradient_evidence_valid,
            "trace_values_finite": gate.get("trace_values_finite") is True,
            "reader_frozen": gate.get("reader_frozen_parameter_gradients") == 0,
            "passed_recomputes": gate.get("passed") is recomputed_passed,
        }
    )
    return {"valid": all(checks.values()), "checks": checks, "recomputed_passed": recomputed_passed}


def validate_r2a_contract(
    diagnostic: Any,
    *,
    reader_loss_mode: str,
    steps: int,
) -> dict[str, Any]:
    expected_applicable = reader_loss_mode == "listwise-choice"
    checks: dict[str, bool] = {"is_mapping": isinstance(diagnostic, dict)}
    if not isinstance(diagnostic, dict):
        return {"valid": False, "checks": checks}
    checks.update(
        {
            "applicability_matches": diagnostic.get("applicable") is expected_applicable,
            "passed_is_none_when_inapplicable": expected_applicable or diagnostic.get("passed") is None,
        }
    )
    if not expected_applicable:
        return {"valid": all(checks.values()), "checks": checks, "recomputed_passed": None}

    optimizer_steps = diagnostic.get("optimizer_step_count")
    steps_with_queries = diagnostic.get("steps_with_query_records")
    query_count = diagnostic.get("query_record_count")
    finite_count = diagnostic.get("finite_query_record_count")
    positive_query_count = diagnostic.get("positive_image_gradient_query_count")
    steps_with_positive = diagnostic.get("steps_with_positive_image_gradient")
    positive_updater_steps = diagnostic.get("positive_updater_gradient_steps")
    all_records_finite = _integer(query_count) and query_count > 0 and finite_count == query_count
    all_query_gradients_positive = _integer(query_count) and query_count > 0 and positive_query_count == query_count
    every_step_has_positive = (
        _integer(optimizer_steps) and optimizer_steps > 0 and steps_with_positive == optimizer_steps
    )
    every_step_has_positive_updater = (
        _integer(optimizer_steps) and optimizer_steps > 0 and positive_updater_steps == optimizer_steps
    )
    recomputed_passed = (
        all_records_finite
        and steps_with_queries == optimizer_steps == steps
        and all_query_gradients_positive
        and every_step_has_positive
        and every_step_has_positive_updater
    )
    checks.update(
        {
            "optimizer_steps_valid": optimizer_steps == steps,
            "steps_with_queries_valid": _integer(steps_with_queries) and 0 <= steps_with_queries <= steps,
            "query_count_valid": _integer(query_count) and query_count > 0,
            "finite_count_valid": _integer(query_count) and _integer(finite_count) and 0 <= finite_count <= query_count,
            "positive_query_count_valid": _integer(positive_query_count)
            and _integer(query_count)
            and 0 <= positive_query_count <= query_count,
            "positive_steps_valid": _integer(steps_with_positive) and 0 <= steps_with_positive <= steps,
            "positive_updater_steps_valid": _integer(positive_updater_steps)
            and 0 <= positive_updater_steps <= steps,
            "all_records_finite_recomputes": diagnostic.get("all_records_finite") is all_records_finite,
            "all_query_gradients_positive_recomputes": diagnostic.get("all_query_image_gradients_positive")
            is all_query_gradients_positive,
            "every_step_positive_recomputes": diagnostic.get("every_step_has_positive_image_gradient")
            is every_step_has_positive,
            "every_step_updater_positive_recomputes": diagnostic.get("every_step_has_positive_updater_gradient")
            is every_step_has_positive_updater,
            "passed_recomputes": diagnostic.get("passed") is recomputed_passed,
        }
    )
    return {"valid": all(checks.values()), "checks": checks, "recomputed_passed": recomputed_passed}


def child_canonical_contract(
    *,
    reader_loss_mode: str,
    steps: int,
    child_result: dict[str, Any],
) -> dict[str, Any]:
    report = child_result.get("report", {})
    payload = report.get("comparison_payload", {}) if isinstance(report, dict) else {}
    protocol = payload.get("protocol", {}) if isinstance(payload, dict) else {}
    expected_objective = reader_objective_contract(reader_loss_mode)
    expected_r1_applicable = reader_loss_mode == "target-only" and steps == REACHABILITY_STEP_BUDGET
    expected_r2_applicable = reader_loss_mode == "listwise-choice" and steps == REACHABILITY_STEP_BUDGET
    report_r1 = report.get("reachability_gate") if isinstance(report, dict) else None
    payload_r1 = payload.get("reachability_gate") if isinstance(payload, dict) else None
    report_r2 = report.get("r2_gate") if isinstance(report, dict) else None
    payload_r2 = payload.get("r2_gate") if isinstance(payload, dict) else None
    report_r2a = report.get("r2a_autograd_diagnostic") if isinstance(report, dict) else None
    payload_r2a = payload.get("r2a_autograd_diagnostic") if isinstance(payload, dict) else None
    r1_contract = validate_reachability_gate_contract(
        report_r1,
        reader_loss_mode=reader_loss_mode,
        steps=steps,
    )
    r2_contract = validate_r2_gate_contract(
        report_r2,
        reader_loss_mode=reader_loss_mode,
        steps=steps,
    )
    r2a_contract = validate_r2a_contract(
        report_r2a,
        reader_loss_mode=reader_loss_mode,
        steps=steps,
    )
    checks = {
        "report_schema_v3": report.get("schema_version") == "vision_memory.lightweight_determinism_report.v3",
        "protocol_schema_v4": protocol.get("schema_version") == "vision_memory.lightweight_determinism_protocol.v4",
        "report_mode_matches": report.get("reader_loss_mode") == reader_loss_mode,
        "payload_mode_matches": payload.get("reader_loss_mode") == reader_loss_mode,
        "protocol_mode_matches": protocol.get("reader_loss_mode") == reader_loss_mode,
        "report_objective_matches": report.get("reader_objective") == expected_objective,
        "payload_objective_matches": payload.get("reader_objective") == expected_objective,
        "protocol_objective_matches": protocol.get("reader_objective") == expected_objective,
        "comparison_payload_sha_matches": isinstance(payload, dict)
        and report.get("comparison_payload_sha256") == canonical_object_sha256(payload),
        "r1_top_payload_consistent": isinstance(report_r1, dict) and report_r1 == payload_r1,
        "r2_top_payload_consistent": isinstance(report_r2, dict) and report_r2 == payload_r2,
        "r2a_top_payload_consistent": isinstance(report_r2a, dict) and report_r2a == payload_r2a,
        "r1_mode_matches": isinstance(report_r1, dict) and report_r1.get("reader_loss_mode") == reader_loss_mode,
        "r2_mode_matches": isinstance(report_r2, dict) and report_r2.get("reader_loss_mode") == reader_loss_mode,
        "r1_applicability_matches": isinstance(report_r1, dict)
        and report_r1.get("applicable") is expected_r1_applicable,
        "r2_applicability_matches": isinstance(report_r2, dict)
        and report_r2.get("applicable") is expected_r2_applicable,
        "r1_gate_semantics_valid": r1_contract["valid"],
        "r2_gate_semantics_valid": r2_contract["valid"],
        "r2a_semantics_valid": r2a_contract["valid"],
        "r2a_passed_for_listwise": reader_loss_mode != "listwise-choice"
        or r2a_contract.get("recomputed_passed") is True,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "expected_reader_objective": expected_objective,
        "expected_r1_applicable": expected_r1_applicable,
        "expected_r2_applicable": expected_r2_applicable,
        "r1_contract": r1_contract,
        "r2_contract": r2_contract,
        "r2a_contract": r2a_contract,
    }


def _pair_child_gate(
    *,
    gate_name: str,
    applicable: bool,
    reader_loss_mode: str,
    steps: int,
    child_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    child_gates = {replica: child_results[replica]["report"].get(gate_name) for replica in ("a", "b")}
    child_payload_gates = {
        replica: child_results[replica]["report"].get("comparison_payload", {}).get(gate_name) for replica in ("a", "b")
    }
    child_gate_consistent = {replica: child_gates[replica] == child_payload_gates[replica] for replica in ("a", "b")}
    child_contracts = {
        replica: child_canonical_contract(
            reader_loss_mode=reader_loss_mode,
            steps=steps,
            child_result=child_results[replica],
        )
        for replica in ("a", "b")
    }
    if not applicable:
        return {
            "applicable": False,
            "passed": None,
            "step_budget": REACHABILITY_STEP_BUDGET,
            "reader_loss_mode": reader_loss_mode,
            "gate_name": gate_name,
            "children": child_gates,
            "children_payload_consistent": child_gate_consistent,
            "children_contracts": child_contracts,
        }
    children_passed = {
        replica: bool(
            child_results[replica]["returncode"] == 0
            and child_results[replica]["report"].get("status") == "complete"
            and isinstance(child_gates[replica], dict)
            and child_gate_consistent[replica]
            and child_contracts[replica]["valid"]
            and child_gates[replica].get("applicable") is True
            and child_gates[replica].get("passed") is True
        )
        for replica in ("a", "b")
    }
    return {
        "applicable": True,
        "passed": all(children_passed.values()),
        "step_budget": REACHABILITY_STEP_BUDGET,
        "reader_loss_mode": reader_loss_mode,
        "gate_name": gate_name,
        "children_passed": children_passed,
        "children": child_gates,
        "children_payload_consistent": child_gate_consistent,
        "children_contracts": child_contracts,
    }


def pair_reachability_gate(
    *,
    steps: int,
    child_results: dict[str, dict[str, Any]],
    reader_loss_mode: str = "target-only",
) -> dict[str, Any]:
    return _pair_child_gate(
        gate_name="reachability_gate",
        applicable=reader_loss_mode == "target-only" and steps == REACHABILITY_STEP_BUDGET,
        reader_loss_mode=reader_loss_mode,
        steps=steps,
        child_results=child_results,
    )


def pair_r2_gate(
    *,
    steps: int,
    child_results: dict[str, dict[str, Any]],
    reader_loss_mode: str = "listwise-choice",
) -> dict[str, Any]:
    return _pair_child_gate(
        gate_name="r2_gate",
        applicable=reader_loss_mode == "listwise-choice" and steps == REACHABILITY_STEP_BUDGET,
        reader_loss_mode=reader_loss_mode,
        steps=steps,
        child_results=child_results,
    )


def pair_r2a_autograd_diagnostic(
    *,
    steps: int,
    child_results: dict[str, dict[str, Any]],
    reader_loss_mode: str,
) -> dict[str, Any]:
    applicable = reader_loss_mode == "listwise-choice"
    child_diagnostics = {
        replica: child_results[replica]["report"].get("r2a_autograd_diagnostic") for replica in ("a", "b")
    }
    child_payload_diagnostics = {
        replica: child_results[replica]["report"].get("comparison_payload", {}).get("r2a_autograd_diagnostic")
        for replica in ("a", "b")
    }
    child_contracts = {
        replica: child_canonical_contract(
            reader_loss_mode=reader_loss_mode,
            steps=steps,
            child_result=child_results[replica],
        )
        for replica in ("a", "b")
    }
    child_passed = {
        replica: bool(
            applicable
            and child_results[replica]["returncode"] == 0
            and child_results[replica]["report"].get("status") == "complete"
            and isinstance(child_diagnostics[replica], dict)
            and child_diagnostics[replica] == child_payload_diagnostics[replica]
            and child_diagnostics[replica].get("applicable") is True
            and child_diagnostics[replica].get("passed") is True
            and child_contracts[replica]["valid"]
        )
        for replica in ("a", "b")
    }
    return {
        "applicable": applicable,
        "passed": all(child_passed.values()) if applicable else None,
        "scope": "paired R2a autograd diagnostic; not the 1/100-step R2 scientific gate",
        "children_passed": child_passed,
        "children": child_diagnostics,
        "children_payload_consistent": {
            replica: child_diagnostics[replica] == child_payload_diagnostics[replica] for replica in ("a", "b")
        },
    }


def main() -> int:
    args = parse_args()
    assert_determinism_environment()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("The paired reproducibility wrapper must run inside one Slurm allocation.")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("CUDA_VISIBLE_DEVICES must identify the allocation's physical GPU.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("The paired reproducibility wrapper refuses a non-empty --output-dir.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    child_results: dict[str, dict[str, Any]] = {}
    for replica in ("a", "b"):
        replica_dir = args.output_dir / replica
        command = [
            sys.executable,
            str(PROBE),
            "--train",
            str(args.train),
            "--reader",
            str(args.reader),
            "--output-dir",
            str(replica_dir),
            "--steps",
            str(args.steps),
            "--device",
            args.device,
            "--reader-loss-mode",
            args.reader_loss_mode,
        ]
        stdout_path = args.output_dir / f"replica_{replica}.stdout.log"
        stderr_path = args.output_dir / f"replica_{replica}.stderr.log"
        with (
            stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_handle,
            stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_handle,
        ):
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=os.environ.copy(),
            )
        child_results[replica] = {
            "returncode": completed.returncode,
            "report": read_report(replica_dir / "report.json", returncode=completed.returncode),
            "stdout": str(stdout_path.resolve()),
            "stderr": str(stderr_path.resolve()),
        }

    comparison = compare_bitwise_repro_reports(
        child_results["a"]["report"],
        child_results["b"]["report"],
    )
    child_contracts = {
        replica: child_canonical_contract(
            reader_loss_mode=args.reader_loss_mode,
            steps=args.steps,
            child_result=child_results[replica],
        )
        for replica in ("a", "b")
    }
    reproducibility_valid = (
        comparison["valid"]
        and child_results["a"]["returncode"] == 0
        and child_results["b"]["returncode"] == 0
        and all(contract["valid"] for contract in child_contracts.values())
    )
    reachability_gate = pair_reachability_gate(
        steps=args.steps,
        child_results=child_results,
        reader_loss_mode=args.reader_loss_mode,
    )
    r2_gate = pair_r2_gate(
        steps=args.steps,
        child_results=child_results,
        reader_loss_mode=args.reader_loss_mode,
    )
    r2a_diagnostic = pair_r2a_autograd_diagnostic(
        steps=args.steps,
        child_results=child_results,
        reader_loss_mode=args.reader_loss_mode,
    )
    if reachability_gate["applicable"]:
        scientific_gate_name = "R1/D2R-target-only"
        scientific_gate_passed = reachability_gate["passed"] is True
    elif r2_gate["applicable"]:
        scientific_gate_name = "R2/D2L-listwise-choice"
        scientific_gate_passed = r2_gate["passed"] is True
    else:
        scientific_gate_name = None
        scientific_gate_passed = None
    overall_passed = reproducibility_valid and (
        scientific_gate_passed is True if scientific_gate_name is not None else True
    )
    objective_contract = reader_objective_contract(args.reader_loss_mode)
    pair_report = {
        "schema_version": "vision_memory.lightweight_determinism_pair.v3",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "steps": args.steps,
        "reader_loss_mode": args.reader_loss_mode,
        "reader_objective": objective_contract,
        "children": child_results,
        "comparison": comparison,
        "children_canonical_contracts": child_contracts,
        "reproducibility_valid": reproducibility_valid,
        "reachability_gate": reachability_gate,
        "reachability_gate_passed": reachability_gate["passed"],
        "r2_gate": r2_gate,
        "r2_gate_passed": r2_gate["passed"],
        "r2a_autograd_diagnostic": r2a_diagnostic,
        "scientific_gate_name": scientific_gate_name,
        "scientific_gate_passed": scientific_gate_passed,
        "valid": reproducibility_valid,
        "overall_passed": overall_passed,
    }
    pair_path = args.output_dir / "pair_report.json"
    pair_path.write_text(
        json.dumps(pair_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(pair_report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if pair_report["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
