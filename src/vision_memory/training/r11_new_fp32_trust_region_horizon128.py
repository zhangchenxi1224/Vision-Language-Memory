"""Locked horizon-128 facade for the delivered FP32 trust-region algorithm."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vision_memory.training import r11_new_fp32_trust_region as base


PROTOCOL = base.PROTOCOL
PREFIX = base.PREFIX
RADII = base.RADII
CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs/experiments/r11_new_fp32_trust_region_horizon128_target01.json"
)
CONFIG_BYTES_SHA256 = "d687c159644551ea06cb0597ac6b46efbcd1761367e249b30251a6de06931c78"
CONFIG_CANONICAL_SHA256 = "542ae95692c3e97c580878cddbd200669d846275ebc09abc7c24762e20835992"

require = base.require
sha256_file = base.sha256_file
canonical_sha = base.canonical_sha
unit_vector = base.unit_vector
cosine = base.cosine
iteration_id = base.iteration_id
candidate_id = base.candidate_id
validate_candidate_row = base.validate_candidate_row
validate_iteration_row = base.validate_iteration_row
summarize_trace = base.summarize_trace
classify_outcome = base.classify_outcome
validate_inventory = base.validate_inventory
audit_delivery = base.audit_delivery


def _without(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def load_config() -> dict[str, Any]:
    """Load the preregistered extension and prove that only horizon changes."""
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "Horizon128 config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(canonical_sha(value) == CONFIG_CANONICAL_SHA256, "Horizon128 config canonical hash drift.")
    require(
        value.get("schema") == f"{PREFIX}-config.v1"
        and value.get("protocol") == PROTOCOL
        and value.get("experiment_variant") == "horizon128",
        "Horizon128 config identity drift.",
    )
    parent = base.load_config()
    algorithm = value["algorithm"]
    expected_algorithm = {**parent["algorithm"], "maximum_formal_iterations": 128}
    require(algorithm == expected_algorithm, "Horizon128 changed a parent algorithm field other than budget.")
    require(
        value["base_precision_contract"] == parent["base_precision_contract"]
        and value["fixed_fp32_anchor"] == parent["fixed_fp32_anchor"]
        and value["parent_precision_result"] == parent["parent_precision_result"],
        "Horizon128 changed a fixed target, precision, or parent binding.",
    )
    require(
        value["technical_preflight_gate"] == parent["technical_preflight_gate"],
        "Horizon128 changed the technical preflight gate.",
    )
    expected_formal = {
        **parent["formal_gate"],
        "maximum_full_chain_forward_calls": 771,
        "maximum_backward_calls": 128,
        "maximum_parameter_updates": 128,
    }
    require(value["formal_gate"] == expected_formal, "Horizon128 formal gate drift.")
    parent_result = value["parent_trust_result"]
    require(
        parent_result["delivery_commit"] == "2c0849235844895428fe65cb77015f1a58272aad"
        and parent_result["classification"] == "monotone_progress_above_capture"
        and parent_result["iteration_count"] == 32
        and parent_result["accepted_update_count"] == 32
        and parent_result["stop_reason"] == "maximum_iterations"
        and parent_result["formal_success"] is False
        and parent_result["phase2_allowed"] is False,
        "Horizon128 parent result identity drift.",
    )
    prefix_gate = value["parent_prefix_reproduction_gate"]
    require(
        prefix_gate["external_independent_audit_required_before_accepting_classification"] is True
        and prefix_gate["iteration_rows_exact_prefix_count"] == 32
        and prefix_gate["candidate_rows_exact_prefix_count"] == 160
        and prefix_gate["all_parent_metric_rows_byte_equal"] is True,
        "Horizon128 parent-prefix gate drift.",
    )
    require(
        value["fixed_target_success_gate"] == parent["fixed_target_success_gate"]
        and value["interpretation_boundaries"]["single_changed_scientific_factor"]
        == "maximum_formal_iterations: 32 -> 128"
        and value["interpretation_boundaries"]["formal_picture_memory_success_always_false"] is True
        and value["interpretation_boundaries"]["phase2_always_false"] is True,
        "Horizon128 success or interpretation boundary drift.",
    )
    return value
