"""Immutable source-only initialization contracts; old experiment stays untouched."""

from __future__ import annotations

from typing import Any, Mapping

from vision_memory.training.r11_new_bridge import (
    BRIDGE_BASE_LEARNING_RATE,
    BRIDGE_CHECKPOINT_STEPS,
    BRIDGE_L2_RATIO_MAX,
    BRIDGE_LR_COSINE_DENOMINATOR,
    BRIDGE_LR_INTERVENTION_FIRST_UPDATE,
    BRIDGE_MODEL_SNAPSHOT_BINDINGS,
    BRIDGE_MSE_RATIO_MAX,
    BRIDGE_OPTIMIZER_STEPS,
    BRIDGE_PARENT_ENDPOINT_MSE,
    BRIDGE_PARENT_ENDPOINT_READER_CE,
    BRIDGE_PARENT_TARGET_MANIFEST_SHA256,
    BRIDGE_PHASE1A_SOURCE_ROOT,
    BRIDGE_PRIMARY_ENDPOINT,
    BRIDGE_SOURCE_LATENTS_SHA256,
    BRIDGE_TARGET_INDEX,
    BRIDGE_TARGET_SEGMENT_ID,
    BRIDGE_TEACHER_FILE_SHA256,
    BRIDGE_TEACHER_NRMSE_MAX,
    BRIDGE_TEACHER_REPLAY_MEAN_CE_MAX,
    BRIDGE_TEACHER_STD,
    BRIDGE_TEACHER_TENSOR_SHA256,
    BRIDGE_UNCHANGED_PARITY_BINDINGS,
    bridge_distance_gate,
    bridge_distance_statistics,
    bridge_initialization_hypothesis_audit,
    bridge_optimizer_learning_rate,
    canonical_json_sha256,
    endpoint_reader_transfer_gate,
    reader_checkpoint_statistics,
    teacher_replay_gate,
)


BRIDGE_PROTOCOL = "R11-New-Canonical-Latent-Bridge-Source-Only-Init-Target01"
BRIDGE_CONFIG_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-source-only-init-config.v1"
BRIDGE_CONFIG_FILE_SHA256 = "2bdf0e0b45e2c0b22d4c04711c0be11c5e9b94fc30d7ee70ed1aa3cb219b1702"
BRIDGE_CONFIG_CANONICAL_SHA256 = "cabaed92d8f80871f8a83511dd107b81672ba1b7fb33844a4241d2c0350ed1fe"
BRIDGE_INITIALIZATION_SCHEMA = "vision_memory.r11-new-source-only-initialization.v1"
BRIDGE_INITIALIZATION_FORMULA = "x_T_init_fp32=source_latents_fp32.clone()"
BRIDGE_INITIALIZATION_FILENAME = "source_only_initialization.pt"
BRIDGE_TEACHER_REFERENCE_COMPARISON_SHA256 = "f66650cd6f66cbdf54c0694e8c55aa18c6ad3497ad487f8f7cb620f107d1d8ed"
BRIDGE_TEACHER_REFERENCE_RAW_SHA256 = "2af2d9a0b6124e450a7b3658352bd1c6e58d7cede2ddcec384064e1dae9d909a"
BRIDGE_TEACHER_REFERENCE_ENDPOINT_MSE = 0.06570044904947281
BRIDGE_TEACHER_REFERENCE_ENDPOINT_READER_CE = 10.97591445709591

BRIDGE_REQUIRED_TECHNICAL_FIELDS = (
    "receipts_exact",
    "steps_contiguous",
    "four_step_schedule_exact",
    "finite_metrics",
    "finite_nonzero_gradient_every_step",
    "only_x_T_fp32_trainable",
    "frozen_gradients_absent",
    "snapshots_unchanged",
    "optimizer_contract_valid",
    "optimizer_lr_schedule_exact",
    "source_only_initialization_artifact_valid",
    "trajectory_point0_binding_valid_every_checkpoint",
    "gradient_clipping_absent",
    "checkpoint_hashes_valid",
    "condition_artifact_valid",
    "step0_parity_valid",
    "teacher_binding_valid",
    "artifact_contract_valid",
)

# Explicit re-exports preserve shared mathematics, not old experiment identities.
__all__ = [
    "BRIDGE_BASE_LEARNING_RATE", "BRIDGE_CHECKPOINT_STEPS", "BRIDGE_CONFIG_CANONICAL_SHA256",
    "BRIDGE_CONFIG_FILE_SHA256", "BRIDGE_CONFIG_SCHEMA", "BRIDGE_INITIALIZATION_FILENAME",
    "BRIDGE_INITIALIZATION_FORMULA", "BRIDGE_INITIALIZATION_SCHEMA", "BRIDGE_L2_RATIO_MAX",
    "BRIDGE_LR_COSINE_DENOMINATOR", "BRIDGE_LR_INTERVENTION_FIRST_UPDATE",
    "BRIDGE_MODEL_SNAPSHOT_BINDINGS", "BRIDGE_MSE_RATIO_MAX", "BRIDGE_OPTIMIZER_STEPS",
    "BRIDGE_PARENT_ENDPOINT_MSE", "BRIDGE_PARENT_ENDPOINT_READER_CE",
    "BRIDGE_PARENT_TARGET_MANIFEST_SHA256", "BRIDGE_PHASE1A_SOURCE_ROOT", "BRIDGE_PRIMARY_ENDPOINT",
    "BRIDGE_PROTOCOL", "BRIDGE_REQUIRED_TECHNICAL_FIELDS", "BRIDGE_SOURCE_LATENTS_SHA256",
    "BRIDGE_TARGET_INDEX", "BRIDGE_TARGET_SEGMENT_ID", "BRIDGE_TEACHER_FILE_SHA256",
    "BRIDGE_TEACHER_NRMSE_MAX", "BRIDGE_TEACHER_REFERENCE_COMPARISON_SHA256",
    "BRIDGE_TEACHER_REFERENCE_ENDPOINT_MSE", "BRIDGE_TEACHER_REFERENCE_ENDPOINT_READER_CE",
    "BRIDGE_TEACHER_REFERENCE_RAW_SHA256", "BRIDGE_TEACHER_REPLAY_MEAN_CE_MAX",
    "BRIDGE_TEACHER_STD", "BRIDGE_TEACHER_TENSOR_SHA256", "BRIDGE_UNCHANGED_PARITY_BINDINGS",
    "bridge_decision", "bridge_distance_gate", "bridge_distance_statistics",
    "bridge_information_boundary", "bridge_initialization_hypothesis_audit",
    "bridge_initialization_hypothesis_decision", "bridge_optimizer_learning_rate",
    "bridge_technical_gate", "canonical_json_sha256", "endpoint_reader_transfer_gate",
    "reader_checkpoint_statistics", "teacher_replay_gate", "validate_bridge_config",
    "validate_bridge_information_boundary",
]


def validate_bridge_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the full preregistered JSON and independently bind implementation constants."""

    if config.get("schema") != BRIDGE_CONFIG_SCHEMA:
        raise ValueError("R11_new source-only config schema drifted.")
    if canonical_json_sha256(config) != BRIDGE_CONFIG_CANONICAL_SHA256:
        raise ValueError("R11_new source-only config canonical JSON drifted.")
    expected = {
        "target_selection": {"target_index": BRIDGE_TARGET_INDEX, "target_segment_id": BRIDGE_TARGET_SEGMENT_ID},
        "canonical_teacher": {
            "file_sha256": BRIDGE_TEACHER_FILE_SHA256,
            "tensor_sha256": BRIDGE_TEACHER_TENSOR_SHA256,
            "population_std": BRIDGE_TEACHER_STD,
            "shape": [1, 4, 128, 128], "dtype": "torch.float32",
        },
        "single_changed_solver_factor": {
            "factor": "x_T_initialization", "new_value": "source-only deterministic initialization",
            "formula": BRIDGE_INITIALIZATION_FORMULA,
            "initializer_allowed_inputs": ["source_latents"],
            "initializer_forbidden_inputs": ["teacher", "query", "answer", "choices", "target_index", "sample_id"],
            "initialization_teacher_assisted": False, "optimization_teacher_supervised": True,
            "answer_independent_writer_usable": False,
        },
        "initialization_binding": {
            "artifact_schema": BRIDGE_INITIALIZATION_SCHEMA,
            "artifact_filename": BRIDGE_INITIALIZATION_FILENAME,
            "source_latents_fp32_sha256": BRIDGE_SOURCE_LATENTS_SHA256,
            "nominal_effective_sigma_schedule": [0.5, 0.375, 0.25, 0.125],
            "actual_sigma_must_come_from_scheduler_setup": True,
            "artifact_required_before_first_forward": True,
            "artifact_must_store": ["source_latents_fp32", "x_T_init_fp32", "reconstructed_start_state_compute"],
            "teacher_tensor_forbidden_in_initialization_artifact": True,
            "independent_recomputation_required": True, "initial_x_t_equals_source": True,
            "trajectory_point0_must_match_artifact": True,
            "trajectory_point0_binding_valid_every_checkpoint": True,
            "compute_device_type": "cuda", "compute_dtype": "torch.bfloat16",
            "no_cpu_fallback_for_cuda_recomputation": True, "no_teacher_neighborhood_threshold": True,
        },
        "unchanged_contract": {
            "only_trainable": "x_T_fp32", "diffusion_steps": 4,
            "effective_sigma_schedule": [0.5, 0.375, 0.25, 0.125], "optimizer": "Adam",
            "base_learning_rate": BRIDGE_BASE_LEARNING_RATE, "weight_decay": 0.0,
            "optimizer_steps": BRIDGE_OPTIMIZER_STEPS, "gradient_clipping": None,
            "checkpoint_steps": list(BRIDGE_CHECKPOINT_STEPS), "primary_endpoint": BRIDGE_PRIMARY_ENDPOINT,
            "best_checkpoint_selection_forbidden": True, "global_determinism_seed": 0,
            "strict_determinism": True, "dreamlite_device": "cuda:0", "reader_device": "cuda:1",
        },
        "preflight_gate": {
            "optimizer_steps": 0, "full_forward_calls": 1, "backward_calls": 1,
            "require_initial_x_t_equals_source": True,
            "require_native_mul_add_trajectory_point0_binding": True,
            "require_initializer_excludes_teacher_query_answer_choices": True,
            "bridge_result_evaluated": False,
        },
        "formal_technical_gate": {
            "optimizer_receipts_exact": BRIDGE_OPTIMIZER_STEPS,
            "optimizer_lr_schedule_exact_per_receipt": True,
            "source_only_initialization_artifact_valid": True,
            "trajectory_point0_binding_valid_every_checkpoint": True,
            "exact_checkpoint_hash_triplets": list(BRIDGE_CHECKPOINT_STEPS),
        },
        "primary_bridge_gate": {
            "endpoint": "raw_step_256_only", "technical_gate": True, "teacher_replay_gate": True,
            "mse_ratio_to_m0_lte": BRIDGE_MSE_RATIO_MAX,
            "l2_distance_ratio_to_m0_lte": BRIDGE_L2_RATIO_MAX,
            "teacher_normalized_rmse_lte": BRIDGE_TEACHER_NRMSE_MAX,
            "endpoint_reverse_cyclic_accuracy_eq": 1.0,
        },
        "parent_bridge": {
            "endpoint_mse": BRIDGE_PARENT_ENDPOINT_MSE,
            "endpoint_reader_mean_ce": BRIDGE_PARENT_ENDPOINT_READER_CE,
        },
        "teacher_matched_reference": {
            "role": "descriptive_only_not_the_secondary_comparator",
            "comparison_sha256": BRIDGE_TEACHER_REFERENCE_COMPARISON_SHA256,
            "raw_artifacts_sha256": BRIDGE_TEACHER_REFERENCE_RAW_SHA256,
            "endpoint_mse": BRIDGE_TEACHER_REFERENCE_ENDPOINT_MSE,
            "endpoint_reader_mean_ce": BRIDGE_TEACHER_REFERENCE_ENDPOINT_READER_CE,
        },
        "secondary_initialization_hypothesis_audit": {"scientific_success_gate": False},
        "interpretation_boundaries": {
            "diagnostic_only": True, "initialization_teacher_assisted": False,
            "optimization_teacher_supervised": True, "answer_independent_writer_usable": False,
            "formal_success_always_false": True, "phase2_remains_blocked": True,
            "no_shared_writer_claim": True, "no_id_ood_claim": True,
            "no_reachability_theorem_from_one_failure": True,
            "no_dominant_bottleneck_claim_from_one_success": True,
            "no_teacher_content_contribution_identification_from_source_only_comparison": True,
            "teacher_reference_descriptive_only": True,
            "no_best_checkpoint_rescue": True, "no_post_result_threshold_or_initialization_change": True,
        },
    }
    if config.get("protocol") != BRIDGE_PROTOCOL:
        raise ValueError("R11_new source-only config protocol drifted.")
    for section, values in expected.items():
        for key, value in values.items():
            observed = config[section].get(key)
            if type(observed) is not type(value) or observed != value:
                raise ValueError(f"R11_new source-only config constant mismatch: {section}.{key}.")
    if config["preflight_gate"]["teacher_replay"] != {
        "fixed_reverse_cyclic_permutations": 4, "all_four_correct": True,
        "mean_ce_lte": BRIDGE_TEACHER_REPLAY_MEAN_CE_MAX,
    }:
        raise ValueError("R11_new source-only teacher replay contract drifted.")
    return dict(config)


def bridge_information_boundary() -> dict[str, bool]:
    """Initializer independence does not remove dense oracle supervision."""

    return {
        "initialization_teacher_assisted": False,
        "optimization_teacher_supervised": True,
        "answer_independent_writer_usable": False,
        "formal_success": False,
        "phase2_allowed": False,
    }


def validate_bridge_information_boundary(boundary: Mapping[str, Any]) -> bool:
    return all(boundary.get(name) is value for name, value in bridge_information_boundary().items())


def bridge_technical_gate(audit: Mapping[str, Any]) -> bool:
    return all(audit.get(name) is True for name in BRIDGE_REQUIRED_TECHNICAL_FIELDS)


def bridge_decision(*, distance_pass: bool, reader_transfer_pass: bool) -> str:
    if distance_pass and reader_transfer_pass:
        return "distance_pass_reader_pass_revalidate_original_qa_solver"
    if distance_pass:
        return "distance_pass_reader_fail_test_teacher_neighborhood"
    if reader_transfer_pass:
        return "distance_fail_reader_pass_prioritize_qa_objective"
    return "distance_fail_reader_fail_change_one_solver_factor"


def bridge_initialization_hypothesis_decision(
    *, distance_pass: bool, reader_transfer_pass: bool, audit: Mapping[str, Any],
) -> str:
    primary = bridge_decision(distance_pass=distance_pass, reader_transfer_pass=reader_transfer_pass)
    if distance_pass or reader_transfer_pass:
        return f"primary_branch_{primary}"
    if audit.get("eligible") is True and audit.get("passed") is True:
        return "distance_fail_reader_fail_secondary_init_improves"
    return "distance_fail_reader_fail_secondary_init_not_improve"
