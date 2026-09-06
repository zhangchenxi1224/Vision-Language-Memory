"""Pure contracts for the preregistered R11_new canonical-latent bridge diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from vision_memory.data import REVERSE_CYCLIC4


BRIDGE_PROTOCOL = "R11-New-Canonical-Latent-Bridge-Distance-Target01"
BRIDGE_CONFIG_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-config.v1"
BRIDGE_CONFIG_FILE_SHA256 = "ab00453511cb43265a3e3d2af6aa11c8d0aa2cf6e9ca5baf44c8695d4a8bcde0"
BRIDGE_CONFIG_CANONICAL_SHA256 = "08996fd26b5661d4a2314c2340cedf5f4b3a57a89f2d8dacaaa8c1770d8dbd68"
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
    return dict(config)


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
        "gradient_clipping_absent",
        "checkpoint_hashes_valid",
        "condition_artifact_valid",
        "step0_parity_valid",
        "teacher_binding_valid",
        "artifact_contract_valid",
    )
    return all(audit.get(name) is True for name in required_true)
