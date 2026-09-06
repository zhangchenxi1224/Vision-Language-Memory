"""Pure contracts for the teacher-matched-initialization bridge diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from vision_memory.data import REVERSE_CYCLIC4


BRIDGE_PROTOCOL = "R11-New-Canonical-Latent-Bridge-Teacher-Matched-Init-Target01"
BRIDGE_CONFIG_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-teacher-matched-init-config.v1"
BRIDGE_CONFIG_FILE_SHA256 = "3168134145c8e715c9d134d00f25e2152c589d75730474c323dee1be2d13cf2f"
BRIDGE_CONFIG_CANONICAL_SHA256 = "ecd84a874f282da48db5498e2db663d7b4f37b0bb4dd2f4f1500d1a4924ca3a7"
BRIDGE_TARGET_INDEX = 1
BRIDGE_TARGET_SEGMENT_ID = "r5-f1-392d41fd097d069c42218e0a"
BRIDGE_TEACHER_FILE_SHA256 = "d359291de63bb5232325b2e7a9294ff3d861287c06e63da2ab6ebe42eab036b9"
BRIDGE_TEACHER_TENSOR_SHA256 = "6857afeffd37124bb196ab7c6607580c57c950d72d760ca6b49f8cc00bdef3f1"
BRIDGE_TEACHER_STD = 0.6546660661697388
BRIDGE_SOURCE_LATENTS_SHA256 = "719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa"
BRIDGE_PARENT_TARGET_MANIFEST_SHA256 = "cd5b740f1f60b32bfb3b8ccf8ba2cfe84bb4ea9c650649810c07d9d9b3972184"
BRIDGE_PHASE1A_SOURCE_ROOT = (
    "/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-new/"
    "r11-new-phase1a-2cde77e-20260905-round02/target-01-retry01"
)
# The intervention replaces x_T / endpoint / optimizer-prefix parity only.
# Keep these independently locked, not merely observed == manifest-declared.
BRIDGE_UNCHANGED_PARITY_BINDINGS = {
    "target_index": BRIDGE_TARGET_INDEX,
    "target_segment_id": BRIDGE_TARGET_SEGMENT_ID,
    "blank_source_rgb_sha256": "a3b784da71eaa113fb4d9d71502a7a3526ba0d41e2d42ed96fe79111ca3dba65",
    "source_latents_fp32_sha256": BRIDGE_SOURCE_LATENTS_SHA256,
    "event_text_sha256": "f170a7e2dfe0070fbd160c09d29dbcf897ddbf5f75929a3ee4af84cf627965bb",
    "condition_prompt_embeds_sha256": "473bd457d6fff070a71b119a19d950b8d094cfaf6f126ceb817330eb01263a60",
    "condition_attention_mask_sha256": "4f941a468150ea22f64ac4f7304e9a94a3dd1c721d07dd7f8ebd10185fbe2ea9",
}
BRIDGE_MODEL_SNAPSHOT_BINDINGS = {
    "dreamlite_snapshot_manifest_sha256": "1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159",
    "reader_snapshot_manifest_sha256": "159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c",
}
BRIDGE_OPTIMIZER_STEPS = 256
BRIDGE_CHECKPOINT_STEPS = (0, 64, 128, 192, 256)
BRIDGE_MSE_RATIO_MAX = 0.01
BRIDGE_L2_RATIO_MAX = 0.1
BRIDGE_TEACHER_NRMSE_MAX = 0.1
BRIDGE_TEACHER_REPLAY_MEAN_CE_MAX = 0.001
BRIDGE_PRIMARY_ENDPOINT = "raw_step_256"
BRIDGE_BASE_LEARNING_RATE = 0.05
BRIDGE_LR_INTERVENTION_FIRST_UPDATE = 129
BRIDGE_LR_COSINE_DENOMINATOR = 128
BRIDGE_TEACHER_MATCHED_SIGMA = 0.5
BRIDGE_START_TEACHER_NRMSE_MAX = 0.01
BRIDGE_PARENT_ENDPOINT_MSE = 0.09553645551204681
BRIDGE_PARENT_ENDPOINT_READER_CE = 25.536474171257463


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_bridge_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on any change to the result-before preregistration."""

    if config.get("schema") != BRIDGE_CONFIG_SCHEMA:
        raise ValueError("R11_new bridge teacher-matched config schema drifted.")
    observed = canonical_json_sha256(config)
    if observed != BRIDGE_CONFIG_CANONICAL_SHA256:
        raise ValueError(
            "R11_new bridge teacher-matched config differs from the "
            f"preregistered canonical JSON: {observed}"
        )
    if config["target_selection"]["target_index"] != BRIDGE_TARGET_INDEX:
        raise ValueError("R11_new bridge target index drifted.")
    if config["target_selection"]["target_segment_id"] != BRIDGE_TARGET_SEGMENT_ID:
        raise ValueError("R11_new bridge target segment drifted.")
    changed = config.get("single_changed_solver_factor", {})
    if (
        changed.get("factor") != "x_T_initialization"
        or changed.get("new_value") != "teacher-state-matched deterministic initialization"
        or changed.get("sigma_start") != BRIDGE_TEACHER_MATCHED_SIGMA
        or changed.get("teacher_assisted") is not True
        or changed.get("answer_independent_writer_usable") is not False
    ):
        raise ValueError("R11_new bridge teacher-matched initialization drifted.")
    # The digest locks the entire JSON; these checks additionally fail closed if
    # an implementation constant drifts while the preregistration stays fixed.
    expected_sections = {
        "canonical_teacher": {
            "file_sha256": BRIDGE_TEACHER_FILE_SHA256,
            "tensor_sha256": BRIDGE_TEACHER_TENSOR_SHA256,
            "population_std": BRIDGE_TEACHER_STD,
            "shape": [1, 4, 128, 128],
            "dtype": "torch.float32",
        },
        "initialization_binding": {
            "source_latents_fp32_sha256": BRIDGE_SOURCE_LATENTS_SHA256,
            "teacher_fp32_sha256": BRIDGE_TEACHER_TENSOR_SHA256,
            "nominal_effective_sigma_schedule": [0.5, 0.375, 0.25, 0.125],
            "actual_sigma_must_come_from_scheduler_setup": True,
            "artifact_required_before_first_forward": True,
            "independent_recomputation_required": True,
            "trajectory_point0_must_match_artifact": True,
            "trajectory_point0_teacher_normalized_rmse_lte": BRIDGE_START_TEACHER_NRMSE_MAX,
        },
        "unchanged_contract": {
            "only_trainable": "x_T_fp32",
            "diffusion_steps": 4,
            "effective_sigma_schedule": [0.5, 0.375, 0.25, 0.125],
            "optimizer": "Adam",
            "base_learning_rate": BRIDGE_BASE_LEARNING_RATE,
            "weight_decay": 0.0,
            "optimizer_steps": BRIDGE_OPTIMIZER_STEPS,
            "gradient_clipping": None,
            "checkpoint_steps": list(BRIDGE_CHECKPOINT_STEPS),
            "primary_endpoint": BRIDGE_PRIMARY_ENDPOINT,
            "best_checkpoint_selection_forbidden": True,
            "global_determinism_seed": 0,
            "strict_determinism": True,
        },
        "preflight_gate": {
            "optimizer_steps": 0,
            "full_forward_calls": 1,
            "backward_calls": 1,
            "require_trajectory_point0_teacher_normalized_rmse_lte": BRIDGE_START_TEACHER_NRMSE_MAX,
            "bridge_result_evaluated": False,
        },
        "formal_technical_gate": {
            "optimizer_receipts_exact": BRIDGE_OPTIMIZER_STEPS,
            "optimizer_lr_schedule_exact_per_receipt": True,
            "teacher_matched_initialization_artifact_valid": True,
            "trajectory_point0_binding_valid_every_checkpoint": True,
            "exact_checkpoint_hash_triplets": list(BRIDGE_CHECKPOINT_STEPS),
        },
        "primary_bridge_gate": {
            "endpoint": "raw_step_256_only",
            "technical_gate": True,
            "teacher_replay_gate": True,
            "mse_ratio_to_m0_lte": BRIDGE_MSE_RATIO_MAX,
            "l2_distance_ratio_to_m0_lte": BRIDGE_L2_RATIO_MAX,
            "teacher_normalized_rmse_lte": BRIDGE_TEACHER_NRMSE_MAX,
            "endpoint_reverse_cyclic_accuracy_eq": 1.0,
        },
        "parent_bridge": {
            "endpoint_mse": BRIDGE_PARENT_ENDPOINT_MSE,
            "endpoint_reader_mean_ce": BRIDGE_PARENT_ENDPOINT_READER_CE,
        },
        "secondary_initialization_hypothesis_audit": {"scientific_success_gate": False},
        "interpretation_boundaries": {
            "diagnostic_only": True,
            "teacher_assisted_initialization": True,
            "formal_success_always_false": True,
            "phase2_remains_blocked": True,
            "no_shared_writer_claim": True,
            "no_id_ood_claim": True,
            "no_reachability_theorem_from_one_failure": True,
            "no_dominant_bottleneck_claim_from_one_success": True,
            "no_best_checkpoint_rescue": True,
            "no_post_result_threshold_or_initialization_change": True,
        },
    }
    if config.get("protocol") != BRIDGE_PROTOCOL:
        raise ValueError("R11_new bridge config protocol drifted.")
    for section, expected in expected_sections.items():
        for key, value in expected.items():
            observed_value = config[section].get(key)
            if type(observed_value) is not type(value) or observed_value != value:
                raise ValueError(f"R11_new bridge config constant mismatch: {section}.{key}.")
    if config["preflight_gate"]["teacher_replay"] != {
        "fixed_reverse_cyclic_permutations": 4,
        "all_four_correct": True,
        "mean_ce_lte": BRIDGE_TEACHER_REPLAY_MEAN_CE_MAX,
    }:
        raise ValueError("R11_new bridge config teacher replay contract drifted.")
    return dict(config)


def bridge_optimizer_learning_rate(update_index: int) -> float:
    """Return the preregistered optimizer LR for one 1-indexed update."""

    if isinstance(update_index, bool) or not isinstance(update_index, int):
        raise TypeError("R11_new bridge update index must be an integer.")
    if update_index < 1 or update_index > BRIDGE_OPTIMIZER_STEPS:
        raise ValueError("R11_new bridge update index is outside 1..256.")
    if update_index < BRIDGE_LR_INTERVENTION_FIRST_UPDATE:
        return BRIDGE_BASE_LEARNING_RATE
    progress = (update_index - 128) / BRIDGE_LR_COSINE_DENOMINATOR
    return 0.5 * BRIDGE_BASE_LEARNING_RATE * (1.0 + math.cos(math.pi * progress))


def bridge_initialization_hypothesis_audit(
    *,
    endpoint_mse: float,
    endpoint_reader_mean_ce: float,
    technical_gate: bool,
    teacher_replay_gate: bool,
    distance_pass: bool,
    reader_transfer_pass: bool,
) -> dict[str, bool]:
    """Evaluate the secondary, non-rescuing initialization hypothesis audit."""

    values = (endpoint_mse, endpoint_reader_mean_ce)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
        for value in values
    ):
        raise ValueError("R11_new bridge initialization audit received an invalid metric.")
    eligible = bool(
        technical_gate is True
        and teacher_replay_gate is True
        and distance_pass is False
        and reader_transfer_pass is False
    )
    mse_improves = endpoint_mse < BRIDGE_PARENT_ENDPOINT_MSE
    reader_ce_improves = endpoint_reader_mean_ce < BRIDGE_PARENT_ENDPOINT_READER_CE
    return {
        "eligible": eligible,
        "absolute_endpoint_mse_improves_parent": mse_improves,
        "endpoint_reader_ce_improves_parent": reader_ce_improves,
        "passed": bool(eligible and mse_improves and reader_ce_improves),
    }


def bridge_initialization_hypothesis_decision(
    *,
    distance_pass: bool,
    reader_transfer_pass: bool,
    audit: Mapping[str, Any],
) -> str:
    primary = bridge_decision(
        distance_pass=distance_pass,
        reader_transfer_pass=reader_transfer_pass,
    )
    if distance_pass or reader_transfer_pass:
        return f"primary_branch_{primary}"
    if audit.get("passed") is True:
        return "distance_fail_reader_fail_secondary_init_improves"
    return "distance_fail_reader_fail_secondary_init_not_improve"


def bridge_distance_statistics(
    *,
    mse: float,
    m0_mse: float,
    tensor_numel: int,
    teacher_population_std: float = BRIDGE_TEACHER_STD,
) -> dict[str, float | int]:
    values = (mse, m0_mse, teacher_population_std)
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("R11_new bridge distance input is non-finite.")
    if mse < 0.0 or m0_mse <= 0.0 or teacher_population_std <= 0.0:
        raise ValueError("R11_new bridge distance input has an invalid sign.")
    if isinstance(tensor_numel, bool) or not isinstance(tensor_numel, int) or tensor_numel <= 0:
        raise ValueError("R11_new bridge tensor_numel must be a positive integer.")
    rmse = math.sqrt(mse)
    m0_rmse = math.sqrt(m0_mse)
    mse_ratio = mse / m0_mse
    l2_ratio = rmse / m0_rmse
    return {
        "mse": mse,
        "rmse": rmse,
        "l2_distance": rmse * math.sqrt(tensor_numel),
        "m0_mse": m0_mse,
        "m0_rmse": m0_rmse,
        "mse_ratio_to_m0": mse_ratio,
        "l2_distance_ratio_to_m0": l2_ratio,
        "relative_mse_change": mse_ratio - 1.0,
        "teacher_population_std": teacher_population_std,
        "teacher_normalized_rmse": rmse / teacher_population_std,
        "tensor_numel": tensor_numel,
    }


def bridge_distance_gate(statistics: Mapping[str, Any], *, technical_gate: bool) -> bool:
    if not technical_gate:
        return False
    required = (
        ("mse_ratio_to_m0", BRIDGE_MSE_RATIO_MAX),
        ("l2_distance_ratio_to_m0", BRIDGE_L2_RATIO_MAX),
        ("teacher_normalized_rmse", BRIDGE_TEACHER_NRMSE_MAX),
    )
    return all(
        isinstance(statistics.get(name), (int, float))
        and not isinstance(statistics.get(name), bool)
        and math.isfinite(float(statistics[name]))
        and float(statistics[name]) <= maximum
        for name, maximum in required
    )


def _row_ce_from_logits(row: Mapping[str, Any]) -> float:
    permutation = tuple(row.get("permutation", ()))
    if permutation not in REVERSE_CYCLIC4:
        raise ValueError("R11_new bridge row uses an unlocked permutation.")
    logits = row.get("choice_logits_ordered")
    if not isinstance(logits, Sequence) or isinstance(logits, (str, bytes)) or len(logits) != 4:
        raise ValueError("R11_new bridge row must expose four ordered logits.")
    values = [float(value) for value in logits]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("R11_new bridge row contains non-finite logits.")
    original_target = row.get("target_index")
    if isinstance(original_target, bool) or not isinstance(original_target, int):
        raise ValueError("R11_new bridge row target_index is malformed.")
    ordered_target = permutation.index(original_target)
    maximum = max(values)
    logsumexp = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    return logsumexp - values[ordered_target]


def reader_checkpoint_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str,
    condition: str,
) -> dict[str, Any]:
    selected = [row for row in rows if row.get("checkpoint") == checkpoint and row.get("condition") == condition]
    if len(selected) != 4:
        raise ValueError(f"R11_new bridge expected four rows for {checkpoint}/{condition}, got {len(selected)}.")
    by_view = {row.get("view_index"): row for row in selected}
    if set(by_view) != set(range(4)):
        raise ValueError("R11_new bridge rows lack exact view indices 0..3.")
    observed_permutations = tuple(tuple(by_view[index].get("permutation", ())) for index in range(4))
    if observed_permutations != tuple(REVERSE_CYCLIC4):
        raise ValueError("R11_new bridge reverse-cyclic order drifted.")
    ces: list[float] = []
    correct: list[bool] = []
    for index in range(4):
        row = by_view[index]
        recomputed_ce = _row_ce_from_logits(row)
        declared_ce = float(row.get("ce"))
        if not math.isclose(recomputed_ce, declared_ce, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("R11_new bridge row CE does not match its raw logits.")
        permutation = tuple(row["permutation"])
        logits = [float(value) for value in row["choice_logits_ordered"]]
        predicted_original = permutation[max(range(4), key=logits.__getitem__)]
        recomputed_correct = predicted_original == int(row["target_index"])
        if bool(row.get("correct")) != recomputed_correct:
            raise ValueError("R11_new bridge row correctness does not match its raw logits.")
        ces.append(recomputed_ce)
        correct.append(recomputed_correct)
    return {
        "checkpoint": checkpoint,
        "condition": condition,
        "row_count": 4,
        "permutations_are_independent_statistical_units": False,
        "mean_ce": sum(ces) / 4.0,
        "accuracy": sum(correct) / 4.0,
        "all_four_correct": all(correct),
        "per_view_ce": ces,
    }


def teacher_replay_gate(statistics: Mapping[str, Any]) -> bool:
    return bool(
        statistics.get("row_count") == 4
        and statistics.get("all_four_correct") is True
        and isinstance(statistics.get("mean_ce"), (int, float))
        and not isinstance(statistics.get("mean_ce"), bool)
        and math.isfinite(float(statistics["mean_ce"]))
        and float(statistics["mean_ce"]) <= BRIDGE_TEACHER_REPLAY_MEAN_CE_MAX
    )


def endpoint_reader_transfer_gate(statistics: Mapping[str, Any]) -> bool:
    return bool(
        statistics.get("row_count") == 4
        and statistics.get("all_four_correct") is True
        and float(statistics.get("accuracy", -1.0)) == 1.0
    )


def bridge_decision(*, distance_pass: bool, reader_transfer_pass: bool) -> str:
    if distance_pass and reader_transfer_pass:
        return "distance_pass_reader_pass_design_answer_independent_initializer"
    if distance_pass:
        return "distance_pass_reader_fail_test_teacher_neighborhood"
    if reader_transfer_pass:
        return "distance_fail_reader_pass_prioritize_qa_objective"
    return "distance_fail_reader_fail_change_one_solver_factor"


def bridge_technical_gate(audit: Mapping[str, Any]) -> bool:
    required_true = (
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
        "teacher_matched_initialization_artifact_valid",
        "trajectory_point0_binding_valid_every_checkpoint",
        "gradient_clipping_absent",
        "checkpoint_hashes_valid",
        "condition_artifact_valid",
        "step0_parity_valid",
        "teacher_binding_valid",
        "artifact_contract_valid",
    )
    return all(audit.get(name) is True for name in required_true)
