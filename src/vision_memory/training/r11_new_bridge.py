"""Pure contracts for the post-step-128 LR-schedule bridge diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from vision_memory.data import REVERSE_CYCLIC4


BRIDGE_PROTOCOL = "R11-New-Canonical-Latent-Bridge-Post128-Cosine-Target01"
BRIDGE_CONFIG_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-lr-schedule-config.v1"
BRIDGE_CONFIG_FILE_SHA256 = "c9794f5197f6c62f2f84af3cf0db9aee0ff225b4649004967f52d1424a247dc5"
BRIDGE_CONFIG_CANONICAL_SHA256 = "5e6110d8bc02d3495f4ce621ed01dacd8fe9922b4b4a23cfd8c7f2dc7b51985d"
BRIDGE_TARGET_INDEX = 1
BRIDGE_TARGET_SEGMENT_ID = "r5-f1-392d41fd097d069c42218e0a"
BRIDGE_TEACHER_FILE_SHA256 = "d359291de63bb5232325b2e7a9294ff3d861287c06e63da2ab6ebe42eab036b9"
BRIDGE_TEACHER_TENSOR_SHA256 = "6857afeffd37124bb196ab7c6607580c57c950d72d760ca6b49f8cc00bdef3f1"
BRIDGE_TEACHER_STD = 0.6546660661697388
BRIDGE_INITIAL_X_T_SHA256 = "c970092e2afca24ededea1aec2892bd6bd54ba0dd2193522dab22af10ac1d991"
BRIDGE_INITIAL_Z_T_SHA256 = "11c7216fe2a70f0caa314d182b2c176b4f50c78f2d081f5aa0271184e5c8e659"
BRIDGE_PARENT_TARGET_MANIFEST_SHA256 = "cd5b740f1f60b32bfb3b8ccf8ba2cfe84bb4ea9c650649810c07d9d9b3972184"
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
BRIDGE_PARENT_ENDPOINT_MSE_RATIO = 0.8105615501236992
BRIDGE_PARENT_STEP128_PNG_SHA256 = "ccfff48bf0bc8cb2ff38cfeb8cbefb845a90ac80deeab3de26326f7b0faf7bd0"
BRIDGE_PARENT_STEP128_OPTIMIZER_SHA256 = "b26457294d573890ba6d7ebf220eb0b4e7dec844338be48186893a6b17ad8004"
BRIDGE_PARENT_STEP128_TENSOR_SHA256 = {
    "x_T_fp32": "482724b2a7ac88624c054a543c8267ae0a318f67177206c1a7d9a2c3ac364ddb",
    "z_t_fp32": "34125c8426d0eccf9eca70578c164673d263f592e05bbd2a8b4161be239a5f93",
    "trajectory_fp32": [
        "7e7a69ae62ab788b35cb7e0d984bd5ccb992f406074b84f3c02e0db3ef4c249d",
        "5b366db05dccd59a605e1d07d4e74b51dd0dce4740f743602e8c5fa88c8dc08a",
        "6c76df63f60f19a36d64d309877e9e3a5d49c51ee9608de65f911dd369b7f900",
        "5a6a3ec5fdafca390c8571addca3dab1933dfe00667f313dcfff67193888f242",
        "34125c8426d0eccf9eca70578c164673d263f592e05bbd2a8b4161be239a5f93",
    ],
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_bridge_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on any change to the result-before preregistration."""

    if config.get("schema") != BRIDGE_CONFIG_SCHEMA:
        raise ValueError("R11_new bridge config schema drifted.")
    observed = canonical_json_sha256(config)
    if observed != BRIDGE_CONFIG_CANONICAL_SHA256:
        raise ValueError(f"R11_new bridge config differs from the preregistered canonical JSON: {observed}")
    if config["target_selection"]["target_index"] != BRIDGE_TARGET_INDEX:
        raise ValueError("R11_new bridge target index drifted.")
    if config["target_selection"]["target_segment_id"] != BRIDGE_TARGET_SEGMENT_ID:
        raise ValueError("R11_new bridge target segment drifted.")
    changed = config.get("single_changed_solver_factor", {})
    if (
        changed.get("factor") != "optimizer_learning_rate_schedule"
        or changed.get("explicitly_not_changed")
        != "DreamLite diffusion scheduler, sigma schedule, or denoising step count"
        or changed.get("intervention_first_update") != BRIDGE_LR_INTERVENTION_FIRST_UPDATE
    ):
        raise ValueError("R11_new bridge LR-schedule intervention drifted.")
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


def bridge_schedule_hypothesis_audit(
    *,
    step128_mse_ratio: float,
    endpoint_mse_ratio: float,
    technical_gate: bool,
    teacher_replay_gate: bool,
    pre_intervention_parity: bool,
    distance_pass: bool,
    reader_transfer_pass: bool,
) -> dict[str, bool]:
    """Evaluate the secondary, non-rescuing LR-schedule hypothesis audit."""

    values = (step128_mse_ratio, endpoint_mse_ratio)
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
        raise ValueError("R11_new bridge schedule audit received an invalid MSE ratio.")
    eligible = bool(
        technical_gate
        and teacher_replay_gate
        and pre_intervention_parity
        and not distance_pass
        and not reader_transfer_pass
    )
    non_rebound = endpoint_mse_ratio <= step128_mse_ratio
    beats_parent = endpoint_mse_ratio < BRIDGE_PARENT_ENDPOINT_MSE_RATIO
    return {
        "eligible": eligible,
        "post128_non_rebound": non_rebound,
        "beats_parent_endpoint": beats_parent,
        "passed": bool(eligible and non_rebound and beats_parent),
    }


def bridge_schedule_hypothesis_decision(
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
        return "distance_fail_reader_fail_secondary_schedule_pass"
    return "distance_fail_reader_fail_secondary_schedule_fail"


def bridge_pre_intervention_parity(record: Mapping[str, Any]) -> bool:
    """Check the exact semantic state before the first changed update."""

    return bool(
        record.get("optimizer_step") == 128
        and record.get("tensor_sha256") == BRIDGE_PARENT_STEP128_TENSOR_SHA256
        and record.get("optimizer_state_sha256")
        == BRIDGE_PARENT_STEP128_OPTIMIZER_SHA256
        and record.get("png_sha256") == BRIDGE_PARENT_STEP128_PNG_SHA256
    )


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
        return "distance_pass_reader_pass_prioritize_qa_objective"
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
        "pre_intervention_step128_parity_valid",
        "gradient_clipping_absent",
        "checkpoint_hashes_valid",
        "condition_artifact_valid",
        "step0_parity_valid",
        "teacher_binding_valid",
        "artifact_contract_valid",
    )
    return all(audit.get(name) is True for name in required_true)
