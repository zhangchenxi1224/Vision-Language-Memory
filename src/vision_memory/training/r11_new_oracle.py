"""Fail-closed, model-free contracts for R11_new Phase 1A.

R11_new is deliberately different from canonical R11: the sole trainable
object is the initial DreamLite noise tensor ``x_T`` and every objective
evaluation executes the complete frozen, source-anchored DreamLite path.
This module contains no model loading or training code; it freezes the
preregistration, schedule, information boundary, and recomputable gates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vision_memory.data import CYCLIC4, REVERSE_CYCLIC4
from vision_memory.event_noise import event_seed
from vision_memory.training.r10_alignment import (
    R10_SELECTED_SEGMENTS_SHA256,
    R10_TARGET_IDS,
    target_gate,
    target_statistics,
)


R11_NEW_CONFIG_SCHEMA = "vision_memory.r11-new-frozen-dreamlite-oracle-phase1a-config.v1"
R11_NEW_PROTOCOL = "R11-New-Frozen-DreamLite-Oracle-Phase1A"
R11_NEW_STATUS = "preregistered_before_any_r11_new_model_outcome"
R11_NEW_PARENT_R11_COMPARISON_SHA256 = "f8b048f9cbe9fd4df9460043297904b5c9d476f386d6844d12fd4a5f8f636bb5"
R11_NEW_TARGET_IDS = R10_TARGET_IDS
R11_NEW_TARGETS_PAYLOAD_SHA256 = R10_SELECTED_SEGMENTS_SHA256
R11_NEW_TRAIN_SHA256 = "24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184"
R11_NEW_DEV_SHA256 = "8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303"

R11_NEW_SEED = 0
R11_NEW_OPTIMIZER_STEPS = 256
R11_NEW_CHECKPOINT_STEPS = (0, 64, 128, 192, 256)
R11_NEW_LEARNING_RATE = 0.05
R11_NEW_WEIGHT_DECAY = 0.0
R11_NEW_DIFFUSION_STEPS = 4
R11_NEW_EFFECTIVE_START_SIGMA = 0.5
R11_NEW_EFFECTIVE_SIGMA_SCHEDULE = (0.5, 0.375, 0.25, 0.125)
R11_NEW_PRIMARY_ENDPOINT = "raw_x_T_step256"
R11_NEW_TRAINABLE_PARAMETER = "x_T_fp32"
R11_NEW_NOISE_KEY = ("global_seed", "source_episode_id", "source_turn_id")
R11_NEW_DREAMLITE_INPUTS = ("source_latent", "event_text", "x_T")
R11_NEW_READER_LOSS_INPUTS = ("decoded_endpoint", "query_text", "answer_choices", "target_index")
R11_NEW_TARGET_INSTANCE = "vlm-r3-h200x2-live-20260717"
R11_NEW_MINIMUM_FREE_BYTES = 50 * 1024**3
R11_NEW_FORBIDDEN_DREAMLITE_INPUTS = (
    "query_text",
    "answer_choices",
    "answer_text",
    "target_index",
    "segment_id",
    "item_id",
    "per_item_embedding",
)


@dataclass(frozen=True)
class Phase1AScheduleStep:
    optimizer_step: int
    target_index: int
    target_segment_id: str
    forward_cyclic_training_view: int
    permutation: tuple[int, int, int, int]


def _target_phase(segment_id: str) -> int:
    return int.from_bytes(hashlib.sha256(segment_id.encode()).digest()[:2], "big") % 4


def build_phase1a_schedule(target_index: int) -> tuple[Phase1AScheduleStep, ...]:
    """Return the locked 256-step, four-view schedule for one F1 target."""

    if isinstance(target_index, bool) or not isinstance(target_index, int):
        raise TypeError("R11_new target_index must be an integer, not bool.")
    if not 0 <= target_index < len(R11_NEW_TARGET_IDS):
        raise ValueError("R11_new target_index must be in [0, 7].")
    segment_id = R11_NEW_TARGET_IDS[target_index]
    phase = _target_phase(segment_id)
    values = tuple(
        Phase1AScheduleStep(
            optimizer_step=step_zero + 1,
            target_index=target_index,
            target_segment_id=segment_id,
            forward_cyclic_training_view=(view := (step_zero + phase) % 4),
            permutation=CYCLIC4[view],
        )
        for step_zero in range(R11_NEW_OPTIMIZER_STEPS)
    )
    counts = Counter(value.forward_cyclic_training_view for value in values)
    if counts != Counter({0: 64, 1: 64, 2: 64, 3: 64}):
        raise RuntimeError(f"R11_new choice-view schedule drifted: {dict(counts)}")
    return values


def locked_event_noise_seed(source_episode_id: str, source_turn_id: str | int) -> int:
    """Derive the exact inherited seed without query/answer identifiers."""

    if not isinstance(source_episode_id, str) or not source_episode_id:
        raise ValueError("source_episode_id must be a non-empty string.")
    if isinstance(source_turn_id, bool) or not isinstance(source_turn_id, (str, int)):
        raise TypeError("source_turn_id must be a string or integer, not bool.")
    return event_seed(R11_NEW_SEED, source_episode_id, source_turn_id)


def validate_information_boundary(
    *,
    dreamlite_inputs: Sequence[str],
    noise_key: Sequence[str],
    reader_loss_inputs: Sequence[str],
) -> dict[str, Any]:
    """Fail closed unless query supervision is downstream of DreamLite only."""

    observed_dreamlite = tuple(dreamlite_inputs)
    observed_noise = tuple(noise_key)
    observed_reader = tuple(reader_loss_inputs)
    forbidden = set(R11_NEW_FORBIDDEN_DREAMLITE_INPUTS)
    leaked = sorted((set(observed_dreamlite) | set(observed_noise)) & forbidden)
    if observed_dreamlite != R11_NEW_DREAMLITE_INPUTS:
        raise ValueError(
            "R11_new DreamLite inputs drifted or contain supervision: "
            f"expected {R11_NEW_DREAMLITE_INPUTS}, observed {observed_dreamlite}."
        )
    if observed_noise != R11_NEW_NOISE_KEY:
        raise ValueError(f"R11_new noise key drifted: expected {R11_NEW_NOISE_KEY}, observed {observed_noise}.")
    if observed_reader != R11_NEW_READER_LOSS_INPUTS:
        raise ValueError(
            f"R11_new Reader-loss inputs drifted: expected {R11_NEW_READER_LOSS_INPUTS}, observed {observed_reader}."
        )
    if leaked:
        raise ValueError(f"R11_new supervision leaked into DreamLite/noise inputs: {leaked}.")
    return {
        "passed": True,
        "dreamlite_inputs": list(observed_dreamlite),
        "noise_key": list(observed_noise),
        "reader_loss_inputs": list(observed_reader),
        "leaked_fields": [],
    }


def expected_phase1a_config() -> dict[str, Any]:
    """Return the complete immutable machine-readable Phase 1A contract."""

    return {
        "schema": R11_NEW_CONFIG_SCHEMA,
        "protocol": R11_NEW_PROTOCOL,
        "status": R11_NEW_STATUS,
        "phase": "phase1a_query_level_frozen_dreamlite_bridge_oracle",
        "parent_evidence": {
            "canonical_r11_comparison_sha256": R11_NEW_PARENT_R11_COMPARISON_SHA256,
            "canonical_r11_target_passes": 8,
            "canonical_r11_total_targets": 8,
            "canonical_r11_is_not_phase1a": True,
        },
        "deployment": {
            "target_instance": R11_NEW_TARGET_INSTANCE,
            "result_storage": "inspire_ssd",
            "output_root_prefix": "/inspire/ssd/",
            "minimum_free_bytes": R11_NEW_MINIMUM_FREE_BYTES,
            "fresh_output_root_required": True,
            "duplicate_process_lock_required": True,
        },
        "fixed_data": {
            "train_sha256": R11_NEW_TRAIN_SHA256,
            "dev_sha256": R11_NEW_DEV_SHA256,
        },
        "target_selection": {
            "family": "F1",
            "independent_run_per_target": True,
            "target_count": 8,
            "target_ids": list(R11_NEW_TARGET_IDS),
            "selected_segment_payload_sha256": R11_NEW_TARGETS_PAYLOAD_SHA256,
        },
        "source_state": {
            "rgb_numerator": 127,
            "rgb_denominator": 255,
            "encoding": "frozen VAE posterior mean in DreamLite model-latent scale/shift",
            "persistent_state": "latent",
            "reset": "decode the unchanged blank source latent",
        },
        "oracle_variable": {
            "name": R11_NEW_TRAINABLE_PARAMETER,
            "meaning": "DreamLite initial noise tensor x_T",
            "count": 1,
            "storage_dtype": "torch.float32",
            "forward_dtype": "DreamLite compute dtype",
            "initialization": "Gaussian from make_event_generator",
            "seed": R11_NEW_SEED,
            "noise_key": list(R11_NEW_NOISE_KEY),
            "unconstrained": True,
        },
        "dreamlite_path": {
            "execution": "complete frozen base DreamLite source-anchored edit path",
            "dreamlite_inputs": list(R11_NEW_DREAMLITE_INPUTS),
            "unet_executed_and_frozen": True,
            "condition_encoder_executed_and_frozen": True,
            "scheduler_executed_and_fixed": True,
            "vae_decoder_executed_and_frozen": True,
            "reader_executed_and_frozen": True,
            "lora_trainable_parameters": 0,
            "diffusion_steps": R11_NEW_DIFFUSION_STEPS,
            "effective_start_sigma": R11_NEW_EFFECTIVE_START_SIGMA,
            "effective_sigma_schedule": list(R11_NEW_EFFECTIVE_SIGMA_SCHEDULE),
            "initial_flow_state": "0.5*source_latent + 0.5*x_T",
        },
        "optimization": {
            "optimizer": "Adam",
            "learning_rate": R11_NEW_LEARNING_RATE,
            "weight_decay": R11_NEW_WEIGHT_DECAY,
            "schedule": "constant",
            "optimizer_steps": R11_NEW_OPTIMIZER_STEPS,
            "gradient_clipping": None,
            "checkpoint_steps": list(R11_NEW_CHECKPOINT_STEPS),
            "primary_endpoint": R11_NEW_PRIMARY_ENDPOINT,
            "best_checkpoint_selection_forbidden": True,
            "strict_determinism": True,
        },
        "choice_views": {
            "training_family": [list(value) for value in CYCLIC4],
            "training_exposures_per_view": 64,
            "endpoint_family": [list(value) for value in REVERSE_CYCLIC4],
            "endpoint_views": 4,
        },
        "information_boundary": {
            "dreamlite_inputs": list(R11_NEW_DREAMLITE_INPUTS),
            "noise_key": list(R11_NEW_NOISE_KEY),
            "reader_loss_inputs": list(R11_NEW_READER_LOSS_INPUTS),
            "forbidden_dreamlite_or_noise_inputs": list(R11_NEW_FORBIDDEN_DREAMLITE_INPUTS),
        },
        "evaluation": {
            "m0": "full frozen DreamLite endpoint before optimizing x_T",
            "endpoint": R11_NEW_PRIMARY_ENDPOINT,
            "conditions": ["normal", "reset"],
            "reset": "blank source decode",
            "raw_endpoint_only": True,
        },
        "per_target_gate": {
            "technical_gate": True,
            "relative_normal_ce_change_at_endpoint_lte": -0.2,
            "improved_fixed_reverse_choice_views": 4,
            "accuracy_delta_gte": 0.25,
            "normal_reset_difference_in_differences_lt": 0.0,
        },
        "arm_gate": {
            "required_target_passes": 8,
            "total_targets": 8,
            "partial_pass_counts_are_diagnostic_only": True,
        },
        "mvp_route": {
            "sequence": ["phase1a", "phase2", "phase3a", "phase3b"],
            "phase1b_required_before_phase2": False,
            "phase1b_status": "deferred_future_state_level_confirmation",
            "phase2_uses_existing_train_split_only": True,
            "phase2_full_train_coverage_required": False,
            "phase2_supervision_scope": "query_level",
            "phase2_candidate_filter": "locked_train_sha_build_r5_family_pools_pairing_seed0_f1_single_event_unique_segment_id",
            "phase2_selection_protocol": "reuse_r10_f1_hash_order_v1",
            "phase2_selection_seed": 20260831,
            "phase2_selection_key": "sha256('R10-VisualAlignment-LowerBound' + unit-separator + '20260831' + unit-separator + 'F1' + unit-separator + segment_id), then segment_id",
            "phase2_phase1a_first8_reused": True,
            "phase2_membership_locked_before_any_phase1a_outcome": True,
            "phase2_candidate_count": 7504,
            "phase2_bank8_ids_sha256": "08c25bbb753e7ffb3a0fd760d0bbf079b113f1db12be9eba4af1505ad57e86ff",
            "phase2_bank8_payload_sha256": R11_NEW_TARGETS_PAYLOAD_SHA256,
            "phase2_bank64_ids_sha256": "5cbd99fdc537f67cba311ed39144516735d0da149e87095565118b162a872fcc",
            "phase2_bank64_payload_sha256": "d7d3a3d12182fd3169c5b9b5127617f9c1c5b81462a94c2d8afccb256973d98a",
            "phase2_bank128_ids_sha256": "c762b52c2b71ba8b977b6bec339a9586ef000440021c8bdf38bef28006d99f37",
            "phase2_bank128_payload_sha256": "6b817ffdc488df1925294aa6169e8d33cb738877432fb9da46b2844aec6a3665",
            "phase2_required_technical_passes": "all_selected_items",
            "phase2_required_oracle_passes": "all_selected_items",
            "phase2_failed_item_replacement_forbidden": True,
            "phase2_technical_retry_same_id_only": True,
            "phase3_requires_complete_selected_bank": True,
            "phase2_minimum_bank_size": 64,
            "phase2_optional_bank_size": 128,
            "phase2_calibration_items": 8,
            "phase2_bank64_is_bank128_prefix": True,
            "phase2_default_size_if_clock_unreliable": 64,
            "program_planning_window_hours": 30.0,
            "program_clock_start_event": "phase1a_technical_preflight_controller_launch",
            "phase3_and_reporting_reserve_hours": 9.0,
            "phase2_projection_safety_factor": 1.15,
            "phase2_projection_statistic": "p90_wall_clock_seconds_first_8",
            "phase2_projection_p90_method": "nearest_rank_ceil_0p90_n8",
            "phase2_projection_remaining_items_formula": "N-8",
            "phase2_projected_remaining_items_for_128": 120,
            "phase2_expand_to_128_if_projected_total_hours_lte": 30.0,
            "phase2_minimum64_even_if_window_infeasible": True,
            "phase2_size_decision_before_item_9": True,
            "phase3_initial_claim": "locked_training_subset_learnability_diagnostic",
        },
        "success_boundary": {
            "query_level_diagnostic_only": True,
            "full_chain_reachability_only": True,
            "scientific_success_gate": False,
            "cannot_substitute_canonical_r11_results": True,
            "cannot_establish_state_level_or_shared_writer_success": True,
        },
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def validate_phase1a_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require semantic equality with the complete preregistered contract."""

    observed = dict(config)
    expected = expected_phase1a_config()
    if observed != expected:
        observed_digest = hashlib.sha256(_canonical_json(observed)).hexdigest()
        expected_digest = hashlib.sha256(_canonical_json(expected)).hexdigest()
        raise ValueError(
            "R11_new Phase 1A config drifted from the immutable contract: "
            f"expected_sha256={expected_digest}, observed_sha256={observed_digest}."
        )
    boundary = observed["information_boundary"]
    validate_information_boundary(
        dreamlite_inputs=boundary["dreamlite_inputs"],
        noise_key=boundary["noise_key"],
        reader_loss_inputs=boundary["reader_loss_inputs"],
    )
    return {
        "passed": True,
        "schema": R11_NEW_CONFIG_SCHEMA,
        "canonical_sha256": hashlib.sha256(_canonical_json(observed)).hexdigest(),
    }


def phase1a_effective_sigmas_match(observed: object) -> bool:
    """Validate raw scheduler observations using the existing forward tolerance.

    The nominal config remains exact.  Scheduler shift inversion round-trips
    through FP32, so observations need the same numeric comparison already
    used by the frozen forward path; never round or replace the raw values.
    """

    if not isinstance(observed, (list, tuple)) or len(observed) != R11_NEW_DIFFUSION_STEPS:
        return False
    return all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and math.isclose(value, expected, rel_tol=2e-6, abs_tol=2e-6)
        for value, expected in zip(observed, R11_NEW_EFFECTIVE_SIGMA_SCHEDULE, strict=True)
    )


def phase1a_technical_gate(
    receipts: Sequence[Mapping[str, Any]],
    *,
    target_segment_id: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the exact Phase 1A engineering gate from raw receipts."""

    if target_segment_id not in R11_NEW_TARGET_IDS:
        raise ValueError(f"R11_new received an unlocked target: {target_segment_id!r}.")
    target_index = R11_NEW_TARGET_IDS.index(target_segment_id)
    schedule = build_phase1a_schedule(target_index)
    expected_by_step = {row.optimizer_step: row for row in schedule}
    view_counts: Counter[int] = Counter()
    finite_gradient_norms: list[float] = []
    rows_valid = len(receipts) == R11_NEW_OPTIMIZER_STEPS
    for position, row in enumerate(receipts, start=1):
        try:
            expected = expected_by_step[position]
            gradient_norm = float(row["gradient_norm"])
            nonzero_fraction = float(row["gradient_nonzero_fraction"])
            loss = float(row["loss_before_step"])
            view = int(row["forward_cyclic_training_view"])
            row_valid = bool(
                row.get("kind") == "optimizer_step"
                and int(row["optimizer_step"]) == position
                and row.get("target_segment_id") == target_segment_id
                and view == expected.forward_cyclic_training_view
                and tuple(row["permutation"]) == expected.permutation
                and math.isfinite(loss)
                and math.isfinite(gradient_norm)
                and gradient_norm > 0.0
                and math.isfinite(nonzero_fraction)
                and nonzero_fraction > 0.0
                and bool(row["full_dreamlite_forward_executed"])
                and int(row["denoiser_steps_executed"]) == R11_NEW_DIFFUSION_STEPS
                and phase1a_effective_sigmas_match(row["effective_sigma_schedule"])
            )
        except (KeyError, TypeError, ValueError):
            row_valid = False
            gradient_norm = float("nan")
            view = -1
        rows_valid = rows_valid and row_valid
        if row_valid:
            view_counts[view] += 1
            finite_gradient_norms.append(gradient_norm)

    expected_audit = {
        "checkpoint_steps_observed": list(R11_NEW_CHECKPOINT_STEPS),
        "latent_checkpoint_steps_observed": list(R11_NEW_CHECKPOINT_STEPS),
        "image_checkpoint_steps_observed": list(R11_NEW_CHECKPOINT_STEPS),
        "trainable_parameter_names": [R11_NEW_TRAINABLE_PARAMETER],
        "trainable_parameter_dtypes": {R11_NEW_TRAINABLE_PARAMETER: "torch.float32"},
        "frozen_components": {
            "dreamlite_unet": True,
            "condition_encoder": True,
            "vae": True,
            "reader": True,
        },
        "frozen_gradients_absent": True,
        "full_model_snapshots_unchanged": True,
        "information_boundary_passed": True,
        "source_contract_verified": True,
        "event_noise_contract_verified": True,
        "optimizer": "Adam",
        "learning_rate": R11_NEW_LEARNING_RATE,
        "weight_decay": R11_NEW_WEIGHT_DECAY,
        "gradient_clip": None,
        "primary_endpoint": R11_NEW_PRIMARY_ENDPOINT,
    }
    audit_valid = all(audit.get(key) == value for key, value in expected_audit.items())
    expected_counts = {0: 64, 1: 64, 2: 64, 3: 64}
    counts = dict(sorted(view_counts.items()))
    passed = bool(rows_valid and counts == expected_counts and audit_valid)
    return {
        "schema": "vision_memory.r11-new-phase1a-technical-gate.v1",
        "passed": passed,
        "optimizer_step_records": len(receipts),
        "receipts_valid": rows_valid,
        "training_view_counts": counts,
        "expected_training_view_counts": expected_counts,
        "audit_valid": audit_valid,
        "minimum_gradient_norm": min(finite_gradient_norms) if finite_gradient_norms else None,
        "query_level_diagnostic_only": True,
        "scientific_success_gate": False,
    }


def phase1a_target_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    target_segment_id: str,
    endpoint: str = R11_NEW_PRIMARY_ENDPOINT,
) -> dict[str, Any]:
    if endpoint != R11_NEW_PRIMARY_ENDPOINT:
        raise ValueError("R11_new forbids best/intermediate endpoint selection.")
    if target_segment_id not in R11_NEW_TARGET_IDS:
        raise ValueError("R11_new target statistics received an unlocked target.")
    return target_statistics(
        rows,
        suite=suite,
        target_segment_id=target_segment_id,
        endpoint=endpoint,
    )


def phase1a_target_gate(statistics: Mapping[str, Any], *, technical_gate: bool) -> bool:
    """Apply the unchanged canonical R11/R10 diagnostic threshold."""

    return target_gate(statistics, technical_gate=technical_gate)


def phase1a_arm_gate(target_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require 8/8 independently valid targets without upgrading the claim."""

    by_id = {str(row.get("target_segment_id")): row for row in target_results}
    coverage_valid = len(target_results) == 8 and set(by_id) == set(R11_NEW_TARGET_IDS)
    passed_ids = sorted(
        target_id
        for target_id, row in by_id.items()
        if bool(row.get("technical_gate")) and bool(row.get("target_reachability_gate"))
    )
    passed = coverage_valid and len(passed_ids) == 8
    return {
        "passed": passed,
        "coverage_valid": coverage_valid,
        "target_pass_count": len(passed_ids),
        "required_target_pass_count": 8,
        "passed_target_ids": passed_ids,
        "query_level_diagnostic_only": True,
        "scientific_success_gate": False,
    }


__all__ = [
    "Phase1AScheduleStep",
    "R11_NEW_CHECKPOINT_STEPS",
    "R11_NEW_CONFIG_SCHEMA",
    "R11_NEW_DEV_SHA256",
    "R11_NEW_DIFFUSION_STEPS",
    "R11_NEW_DREAMLITE_INPUTS",
    "R11_NEW_EFFECTIVE_SIGMA_SCHEDULE",
    "R11_NEW_EFFECTIVE_START_SIGMA",
    "R11_NEW_LEARNING_RATE",
    "R11_NEW_MINIMUM_FREE_BYTES",
    "R11_NEW_NOISE_KEY",
    "R11_NEW_OPTIMIZER_STEPS",
    "R11_NEW_PARENT_R11_COMPARISON_SHA256",
    "R11_NEW_PRIMARY_ENDPOINT",
    "R11_NEW_PROTOCOL",
    "R11_NEW_READER_LOSS_INPUTS",
    "R11_NEW_SEED",
    "R11_NEW_TARGETS_PAYLOAD_SHA256",
    "R11_NEW_TARGET_IDS",
    "R11_NEW_TARGET_INSTANCE",
    "R11_NEW_TRAINABLE_PARAMETER",
    "R11_NEW_TRAIN_SHA256",
    "R11_NEW_WEIGHT_DECAY",
    "build_phase1a_schedule",
    "expected_phase1a_config",
    "locked_event_noise_seed",
    "phase1a_arm_gate",
    "phase1a_effective_sigmas_match",
    "phase1a_target_gate",
    "phase1a_target_statistics",
    "phase1a_technical_gate",
    "validate_information_boundary",
    "validate_phase1a_config",
]
