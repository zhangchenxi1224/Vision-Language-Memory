"""Contracts and independent audit for the FP32 L-BFGS trust-region control."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vision_memory.repro import canonical_tensor_sha256
from vision_memory.training import r11_new_fp32_prpplus_trust_region as prp_core
from vision_memory.training import r11_new_fp32_trust_region as trust_core
from vision_memory.training import r11_new_oracle_terminal_capture as terminal_core


PROTOCOL = "R11-New-FP32-LBFGS-Trust-Region-Target01"
PREFIX = "vision_memory.r11-new-fp32-lbfgs-trust-region"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/experiments/r11_new_fp32_lbfgs_trust_region_target01.json"
CONFIG_BYTES_SHA256 = "2fb70c6474812b3fac42b740cb0d50762fa9314f79de7d7ef84b0f827769abab"
CONFIG_CANONICAL_SHA256 = "35bedc5c5e99abb3b37324097226712915b7681bb279721de00ac3a4ba287542"
RADII = trust_core.RADII
HISTORY_SIZE = 10
CURVATURE_RELATIVE_THRESHOLD = 1e-8
RESTART_REASONS = (
    "initial",
    "none",
    "no_usable_history",
    "nonfinite_history_cleared",
    "non_descent_history_cleared",
)

require = trust_core.require
sha256_file = trust_core.sha256_file
canonical_sha = trust_core.canonical_sha
unit_vector = trust_core.unit_vector
cosine = trust_core.cosine


def load_config() -> dict[str, Any]:
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "L-BFGS config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(canonical_sha(value) == CONFIG_CANONICAL_SHA256, "L-BFGS config canonical hash drift.")
    require(
        value.get("schema") == f"{PREFIX}-config.v1"
        and value.get("protocol") == PROTOCOL
        and value.get("experiment_variant") == "lbfgs_matched_horizon128",
        "L-BFGS config identity drift.",
    )
    parent = prp_core.load_config()
    require(
        value["base_precision_contract"] == parent["base_precision_contract"]
        and value["parent_precision_result"] == parent["parent_precision_result"]
        and value["fixed_fp32_anchor"] == parent["fixed_fp32_anchor"]
        and value["parent_first_step_control"] == parent["parent_first_step_control"],
        "L-BFGS changed the fixed precision, target, plateau, or first-step contract.",
    )
    algorithm = value["algorithm"]
    require(
        tuple(algorithm["candidate_radii_l2_in_fixed_order"]) == RADII
        and algorithm["maximum_formal_iterations"] == 128
        and algorithm["technical_preflight_iterations"] == 1
        and algorithm["stop_after_loss_ratio_lte"] == 0.01
        and algorithm["minimum_relative_improvement_to_accept"]
        == parent["algorithm"]["minimum_relative_improvement_to_accept"]
        and algorithm["no_torch_optimizer"] is True
        and algorithm["no_gradient_clipping"] is True
        and algorithm["direction_method"] == "explicit limited-memory BFGS inverse-curvature two-loop recursion"
        and algorithm["history_size"] == HISTORY_SIZE
        and algorithm["curvature_relative_threshold"] == CURVATURE_RELATIVE_THRESHOLD
        and algorithm["two_loop_vector_arithmetic"]
        == "q and r use CPU float64; raw direction is cast once to float32 before unit-L2 normalization",
        "L-BFGS algorithm boundary drift.",
    )
    fixed_parent_fields = (
        "parameterization",
        "initial_residual_delta",
        "trainable_values",
        "frozen_values",
        "loss",
        "gradient_nonzero_fraction_gte",
        "candidate_radii_l2_in_fixed_order",
        "candidate_formula",
        "evaluate_all_candidates_every_iteration",
        "selection",
        "minimum_relative_improvement_to_accept",
        "maximum_formal_iterations",
        "technical_preflight_iterations",
        "stop_after_loss_ratio_lte",
        "stop_if_no_candidate_is_acceptable",
        "no_torch_optimizer",
        "no_gradient_clipping",
        "teacher_x_T_forbidden_from_update_direction_or_candidate_selection",
        "checkpoint_every_accepted_update",
        "final_independent_replay_required",
    )
    require(
        all(algorithm[name] == parent["algorithm"][name] for name in fixed_parent_fields),
        "L-BFGS changed a non-direction algorithm field.",
    )
    parent_result = value["parent_prpplus_result"]
    require(
        parent_result["training_commit"] == "de26e3a918861ab3d3214e229469ee246b395876"
        and parent_result["delivery_commit"] == "efc2a9c626cf5846d9d7970cc15d45ea85c8f9ae"
        and parent_result["classification"] == "prpplus_material_improvement_only"
        and parent_result["iteration_count"] == 65
        and parent_result["accepted_update_count"] == 64
        and parent_result["final_loss_ratio_to_plateau"] == 0.02859118701878591
        and parent_result["stop_reason"] == "no_acceptable_candidate"
        and parent_result["formal_success"] is False
        and parent_result["phase2_allowed"] is False,
        "L-BFGS parent PRP+ result identity drift.",
    )
    require(
        value["technical_preflight_gate"]["full_chain_forward_calls"] == 9
        and value["formal_gate"]["maximum_full_chain_forward_calls"] == 771
        and value["formal_gate"]["all_curvature_pairs_and_two_loop_directions_independently_reconstructed"] is True
        and value["fixed_target_success_gate"]["classification"] == "fixed_target_trust_region_success"
        and value["matched_budget_gate"]["parent_final_loss_ratio"] == parent_result["final_loss_ratio_to_plateau"]
        and value["matched_budget_gate"]["material_final_loss_ratio_lte"] == 0.027161627667846616
        and value["interpretation_boundaries"]["all_non_direction_algorithm_fields_fixed"] is True
        and value["interpretation_boundaries"]["matched_update_and_forward_budget"] is True
        and value["interpretation_boundaries"]["formal_picture_memory_success_always_false"] is True
        and value["interpretation_boundaries"]["phase2_always_false"] is True,
        "L-BFGS gate or interpretation boundary drift.",
    )
    return value


def iteration_id(iteration: int) -> str:
    require(type(iteration) is int and iteration >= 0, "Invalid L-BFGS iteration.")
    return f"iteration-{iteration:02d}"


def candidate_id(iteration: int, radius_index: int) -> str:
    require(
        type(iteration) is int and iteration >= 0 and type(radius_index) is int and 0 <= radius_index < len(RADII),
        "Invalid L-BFGS candidate identity.",
    )
    return f"iteration-{iteration:02d}__radius-{radius_index:02d}"


def _finite_number(value: Any, *, nonnegative: bool = False) -> bool:
    return bool(type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0.0))


def _valid_search_tensor(value: Any, *, shape: torch.Size | None = None) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        and value.numel() > 0
        and (shape is None or value.shape == shape)
        and torch.isfinite(value).all()
    )


def form_curvature_pair(
    *,
    iteration: int,
    current_x: torch.Tensor,
    previous_x: torch.Tensor | None,
    gradient: torch.Tensor,
    previous_gradient: torch.Tensor | None,
    relative_threshold: float = CURVATURE_RELATIVE_THRESHOLD,
) -> dict[str, Any]:
    """Form the preregistered consecutive-state curvature pair without mutating history."""
    require(
        type(iteration) is int
        and iteration >= 0
        and _valid_search_tensor(current_x)
        and _valid_search_tensor(gradient, shape=current_x.shape)
        and type(relative_threshold) is float
        and math.isfinite(relative_threshold)
        and relative_threshold >= 0.0,
        "Invalid L-BFGS curvature input.",
    )
    if iteration == 0:
        require(previous_x is None and previous_gradient is None, "Initial L-BFGS curvature state is not empty.")
        return {
            "pair": None,
            "s": None,
            "y": None,
            "pair_iteration": None,
            "accepted": None,
            "reason": "initial",
            "s_dot_y": None,
            "s_norm": None,
            "y_norm": None,
            "cosine": None,
            "threshold_value": None,
        }
    require(
        _valid_search_tensor(previous_x, shape=current_x.shape)
        and _valid_search_tensor(previous_gradient, shape=gradient.shape),
        "Missing previous accepted L-BFGS state.",
    )
    s = (current_x - previous_x).detach().to(dtype=torch.float32, device="cpu").clone()
    y = (gradient - previous_gradient).detach().to(dtype=torch.float32, device="cpu").clone()
    s_double = s.double().flatten()
    y_double = y.double().flatten()
    s_dot_y = float(torch.dot(s_double, y_double))
    s_norm = float(s_double.norm())
    y_norm = float(y_double.norm())
    require(s_norm > 0.0 and math.isfinite(s_norm) and math.isfinite(y_norm), "Invalid L-BFGS curvature norm.")
    denominator = s_norm * y_norm
    curvature_cosine = s_dot_y / denominator if denominator > 0.0 else 0.0
    threshold_value = relative_threshold * denominator
    accepted = bool(s_dot_y > threshold_value)
    pair = None
    if accepted:
        pair = {
            "iteration": iteration,
            "s": s,
            "y": y,
            "s_dot_y": s_dot_y,
            "y_dot_y": float(torch.dot(y_double, y_double)),
            "rho": 1.0 / s_dot_y,
        }
    return {
        "pair": pair,
        "s": s,
        "y": y,
        "pair_iteration": iteration,
        "accepted": accepted,
        "reason": "accepted" if accepted else "curvature_gate_rejected",
        "s_dot_y": s_dot_y,
        "s_norm": s_norm,
        "y_norm": y_norm,
        "cosine": curvature_cosine,
        "threshold_value": threshold_value,
    }


def curvature_row_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    s = state["s"]
    y = state["y"]
    return {
        "curvature_pair_iteration": state["pair_iteration"],
        "curvature_pair_accepted": state["accepted"],
        "curvature_pair_reason": state["reason"],
        "curvature_s_fp32_sha256": canonical_tensor_sha256(s) if s is not None else None,
        "curvature_y_fp32_sha256": canonical_tensor_sha256(y) if y is not None else None,
        "curvature_s_dot_y": state["s_dot_y"],
        "curvature_s_norm": state["s_norm"],
        "curvature_y_norm": state["y_norm"],
        "curvature_cosine": state["cosine"],
        "curvature_threshold_value": state["threshold_value"],
    }


def _validated_history_pair(pair: Mapping[str, Any], shape: torch.Size) -> tuple[int, torch.Tensor, torch.Tensor]:
    pair_iteration = pair.get("iteration")
    s = pair.get("s")
    y = pair.get("y")
    require(
        type(pair_iteration) is int
        and pair_iteration >= 1
        and _valid_search_tensor(s, shape=shape)
        and _valid_search_tensor(y, shape=shape),
        "Invalid L-BFGS history pair.",
    )
    s_double = s.double().flatten()
    y_double = y.double().flatten()
    s_dot_y = float(torch.dot(s_double, y_double))
    y_dot_y = float(torch.dot(y_double, y_double))
    require(
        s_dot_y > 0.0
        and y_dot_y > 0.0
        and pair.get("s_dot_y") == s_dot_y
        and pair.get("y_dot_y") == y_dot_y
        and pair.get("rho") == 1.0 / s_dot_y,
        "Invalid L-BFGS positive-curvature history.",
    )
    return pair_iteration, s, y


def lbfgs_direction(
    gradient: torch.Tensor,
    history: Sequence[Mapping[str, Any]],
    *,
    iteration: int,
) -> dict[str, Any]:
    """Construct a deterministic CPU/float64 two-loop inverse-curvature direction."""
    require(
        type(iteration) is int
        and iteration >= 0
        and _valid_search_tensor(gradient)
        and float(gradient.double().norm()) > 0.0
        and len(history) <= HISTORY_SIZE,
        "Invalid L-BFGS direction input.",
    )
    checked = [_validated_history_pair(pair, gradient.shape) for pair in history]
    pair_iterations = [item[0] for item in checked]
    require(
        pair_iterations == sorted(set(pair_iterations)) and all(index <= iteration for index in pair_iterations),
        "Invalid L-BFGS history order.",
    )
    q = gradient.detach().to(dtype=torch.float64, device="cpu").clone()
    alpha_by_iteration: dict[int, float] = {}
    rho_by_iteration: dict[int, float] = {}
    for pair, (pair_iteration, s, y) in zip(reversed(history), reversed(checked), strict=True):
        rho = float(pair["rho"])
        alpha = rho * float(torch.dot(s.double().flatten(), q.flatten()))
        q = q - alpha * y.double()
        rho_by_iteration[pair_iteration] = rho
        alpha_by_iteration[pair_iteration] = alpha
    if checked:
        last_pair = history[-1]
        gamma = float(last_pair["s_dot_y"]) / float(last_pair["y_dot_y"])
    else:
        gamma = 1.0
    require(math.isfinite(gamma) and gamma > 0.0, "Invalid L-BFGS inverse-Hessian scale.")
    r = gamma * q
    coefficients: list[dict[str, Any]] = []
    for pair, (pair_iteration, s, y) in zip(history, checked, strict=True):
        rho = rho_by_iteration[pair_iteration]
        alpha = alpha_by_iteration[pair_iteration]
        beta = rho * float(torch.dot(y.double().flatten(), r.flatten()))
        r = r + s.double() * (alpha - beta)
        coefficients.append(
            {
                "pair_iteration": pair_iteration,
                "rho": rho,
                "alpha": alpha,
                "beta": beta,
            }
        )
    raw_double = -r
    finite_raw = bool(torch.isfinite(raw_double).all())
    raw_direction = raw_double.float() if finite_raw else -gradient.detach().cpu().clone()
    derivative = (
        float(torch.dot(gradient.double().flatten(), raw_direction.double().flatten())) if finite_raw else math.nan
    )
    history_cleared = False
    if not finite_raw or not math.isfinite(derivative):
        raw_direction = -gradient.detach().cpu().clone()
        history_cleared = bool(history)
        restart_reason = "nonfinite_history_cleared"
    elif derivative >= 0.0:
        raw_direction = -gradient.detach().cpu().clone()
        history_cleared = bool(history)
        restart_reason = "non_descent_history_cleared"
    elif not history:
        restart_reason = "initial" if iteration == 0 else "no_usable_history"
    else:
        restart_reason = "none"
    raw_direction = raw_direction.detach().to(dtype=torch.float32, device="cpu").clone()
    direction = unit_vector(raw_direction)
    directional_derivative = float(torch.dot(gradient.double().flatten(), direction.double().flatten()))
    negative_gradient_cosine = cosine(direction, -gradient)
    require(
        bool(torch.isfinite(direction).all())
        and directional_derivative < 0.0
        and negative_gradient_cosine > 0.0
        and abs(float(direction.double().norm()) - 1.0) <= 1e-6,
        "L-BFGS failed to produce a unit descent direction.",
    )
    history_after = [] if history_cleared else pair_iterations
    return {
        "raw_direction": raw_direction,
        "direction": direction,
        "restart_reason": restart_reason,
        "history_cleared": history_cleared,
        "history_pair_iterations_before_direction": pair_iterations,
        "history_pair_iterations_used": pair_iterations,
        "history_pair_iterations_after_direction": history_after,
        "initial_inverse_hessian_scale": gamma,
        "two_loop_coefficients": coefficients,
        "directional_derivative": directional_derivative,
        "negative_gradient_cosine": negative_gradient_cosine,
    }


def direction_row_metadata(state: Mapping[str, Any]) -> dict[str, Any]:
    before = list(state["history_pair_iterations_before_direction"])
    used = list(state["history_pair_iterations_used"])
    after = list(state["history_pair_iterations_after_direction"])
    return {
        "restart_reason": state["restart_reason"],
        "history_cleared": state["history_cleared"],
        "history_pair_iterations_before_direction": before,
        "history_pair_iterations_used": used,
        "history_pair_iterations_after_direction": after,
        "history_size_before_direction": len(before),
        "history_size_used": len(used),
        "history_size_after_direction": len(after),
        "initial_inverse_hessian_scale": state["initial_inverse_hessian_scale"],
        "two_loop_coefficients": list(state["two_loop_coefficients"]),
        "negative_gradient_cosine": state["negative_gradient_cosine"],
        "analytic_directional_derivative": state["directional_derivative"],
    }


def validate_candidate_row(row: Mapping[str, Any]) -> tuple[int, int]:
    iteration = row.get("iteration")
    radius_index = row.get("radius_index")
    require(
        row.get("schema") == f"{PREFIX}-candidate-row.v1"
        and row.get("protocol") == PROTOCOL
        and type(iteration) is int
        and iteration >= 0
        and type(radius_index) is int
        and 0 <= radius_index < len(RADII)
        and row.get("radius_l2") == RADII[radius_index]
        and row.get("candidate_id") == candidate_id(iteration, radius_index),
        "Invalid L-BFGS candidate row.",
    )
    require(
        all(
            _finite_number(row.get(name), nonnegative=True)
            for name in ("loss", "loss_ratio_to_plateau", "loss_ratio_to_current")
        )
        and all(
            isinstance(row.get(name), str) and len(row[name]) == 64
            for name in ("candidate_x_T_fp32_sha256", "endpoint_fp32_sha256")
        ),
        "Invalid L-BFGS candidate value.",
    )
    return iteration, radius_index


def _validate_history_metadata(row: Mapping[str, Any], iteration: int) -> None:
    before = row.get("history_pair_iterations_before_direction")
    used = row.get("history_pair_iterations_used")
    after = row.get("history_pair_iterations_after_direction")
    require(
        all(isinstance(value, list) for value in (before, used, after))
        and before == used
        and before == sorted(set(before))
        and all(type(value) is int and 1 <= value <= iteration for value in before)
        and len(before) <= HISTORY_SIZE
        and row.get("history_size_before_direction") == len(before)
        and row.get("history_size_used") == len(used)
        and row.get("history_size_after_direction") == len(after)
        and isinstance(row.get("history_cleared"), bool)
        and after == ([] if row["history_cleared"] else before),
        "Invalid L-BFGS history metadata.",
    )
    coefficients = row.get("two_loop_coefficients")
    require(
        isinstance(coefficients, list)
        and [item.get("pair_iteration") for item in coefficients if isinstance(item, Mapping)] == used
        and all(
            isinstance(item, Mapping)
            and _finite_number(item.get("rho"))
            and item["rho"] > 0.0
            and _finite_number(item.get("alpha"))
            and _finite_number(item.get("beta"))
            for item in coefficients
        )
        and _finite_number(row.get("initial_inverse_hessian_scale"))
        and row["initial_inverse_hessian_scale"] > 0.0,
        "Invalid L-BFGS two-loop metadata.",
    )


def _validate_curvature_metadata(row: Mapping[str, Any], iteration: int) -> None:
    names = (
        "curvature_s_dot_y",
        "curvature_s_norm",
        "curvature_y_norm",
        "curvature_cosine",
        "curvature_threshold_value",
    )
    hashes = (row.get("curvature_s_fp32_sha256"), row.get("curvature_y_fp32_sha256"))
    if iteration == 0:
        require(
            row.get("curvature_pair_iteration") is None
            and row.get("curvature_pair_accepted") is None
            and row.get("curvature_pair_reason") == "initial"
            and hashes == (None, None)
            and all(row.get(name) is None for name in names),
            "Invalid initial L-BFGS curvature metadata.",
        )
        return
    require(
        row.get("curvature_pair_iteration") == iteration
        and isinstance(row.get("curvature_pair_accepted"), bool)
        and row.get("curvature_pair_reason") in ("accepted", "curvature_gate_rejected")
        and all(isinstance(value, str) and len(value) == 64 for value in hashes)
        and all(_finite_number(row.get(name)) for name in names)
        and row["curvature_s_norm"] > 0.0
        and row["curvature_y_norm"] >= 0.0
        and row["curvature_threshold_value"] >= 0.0
        and row["curvature_pair_accepted"] is (row["curvature_s_dot_y"] > row["curvature_threshold_value"])
        and row["curvature_pair_reason"]
        == ("accepted" if row["curvature_pair_accepted"] else "curvature_gate_rejected"),
        "Invalid L-BFGS curvature metadata.",
    )


def validate_iteration_row(row: Mapping[str, Any]) -> int:
    iteration = row.get("iteration")
    require(
        row.get("schema") == f"{PREFIX}-iteration-row.v1"
        and row.get("protocol") == PROTOCOL
        and type(iteration) is int
        and iteration >= 0
        and row.get("iteration_id") == iteration_id(iteration)
        and isinstance(row.get("accepted"), bool)
        and type(row.get("selected_radius_index")) is int
        and 0 <= row["selected_radius_index"] < len(RADII)
        and row.get("selected_radius_l2") == RADII[row["selected_radius_index"]]
        and row.get("selected_candidate_id") == candidate_id(iteration, row["selected_radius_index"])
        and row.get("restart_reason") in RESTART_REASONS,
        "Invalid L-BFGS iteration row.",
    )
    require(
        all(
            _finite_number(row.get(name), nonnegative=True)
            for name in (
                "current_loss",
                "current_loss_ratio_to_plateau",
                "gradient_norm",
                "gradient_nonzero_fraction",
                "negative_gradient_cosine",
                "selected_loss",
                "selected_loss_ratio_to_plateau",
                "selected_candidate_residual_l2",
            )
        )
        and _finite_number(row.get("relative_improvement"))
        and _finite_number(row.get("analytic_directional_derivative"))
        and row["analytic_directional_derivative"] < 0.0
        and row["negative_gradient_cosine"] > 0.0
        and all(
            isinstance(row.get(name), str) and len(row[name]) == 64
            for name in (
                "current_x_T_fp32_sha256",
                "current_endpoint_fp32_sha256",
                "gradient_fp32_sha256",
                "raw_direction_fp32_sha256",
                "direction_fp32_sha256",
                "selected_x_T_fp32_sha256",
                "selected_endpoint_fp32_sha256",
            )
        ),
        "Invalid L-BFGS iteration value.",
    )
    _validate_curvature_metadata(row, iteration)
    _validate_history_metadata(row, iteration)
    if iteration == 0:
        require(
            row["restart_reason"] == "initial" and row["history_size_before_direction"] == 0,
            "Invalid initial L-BFGS direction metadata.",
        )
    elif row["history_size_before_direction"] == 0:
        require(row["restart_reason"] == "no_usable_history", "Missing no-history L-BFGS restart marker.")
    elif row["history_cleared"]:
        require(
            row["restart_reason"] in ("nonfinite_history_cleared", "non_descent_history_cleared"),
            "Invalid L-BFGS history-clear marker.",
        )
    else:
        require(row["restart_reason"] == "none", "Unexpected L-BFGS restart marker.")
    return iteration


def summarize_trace(
    iteration_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    require(iteration_rows, "L-BFGS trace is empty.")
    require(len(candidate_rows) == len(iteration_rows) * len(RADII), "L-BFGS candidate count drift.")
    iterations: dict[int, Mapping[str, Any]] = {}
    for row in iteration_rows:
        index = validate_iteration_row(row)
        require(index not in iterations, "Duplicate L-BFGS iteration row.")
        iterations[index] = row
    require(set(iterations) == set(range(len(iteration_rows))), "L-BFGS iteration coverage drift.")
    candidates: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in candidate_rows:
        key = validate_candidate_row(row)
        require(key not in candidates, "Duplicate L-BFGS candidate row.")
        candidates[key] = row
    require(
        set(candidates)
        == {
            (iteration, radius_index) for iteration in range(len(iteration_rows)) for radius_index in range(len(RADII))
        },
        "L-BFGS candidate coverage drift.",
    )
    accepted_losses: list[float] = []
    previous_selected_endpoint_hash = None
    previous_selected_x_hash = None
    accepted_count = 0
    stop_reason = None
    minimum_improvement = float(config["algorithm"]["minimum_relative_improvement_to_accept"])
    for index in range(len(iteration_rows)):
        row = iterations[index]
        local = [candidates[(index, radius_index)] for radius_index in range(len(RADII))]
        best = min(local, key=lambda item: (item["loss"], item["radius_l2"]))
        require(
            row["selected_radius_index"] == best["radius_index"]
            and row["selected_loss"] == best["loss"]
            and row["selected_loss_ratio_to_plateau"] == best["loss_ratio_to_plateau"]
            and row["selected_x_T_fp32_sha256"] == best["candidate_x_T_fp32_sha256"]
            and row["selected_endpoint_fp32_sha256"] == best["endpoint_fp32_sha256"],
            "L-BFGS candidate selection drift.",
        )
        relative_improvement = (row["current_loss"] - best["loss"]) / row["current_loss"]
        accepted = relative_improvement >= minimum_improvement
        require(
            row["accepted"] == accepted
            and math.isclose(row["relative_improvement"], relative_improvement, rel_tol=1e-9, abs_tol=1e-12),
            "L-BFGS acceptance drift.",
        )
        if previous_selected_endpoint_hash is not None:
            require(
                row["current_endpoint_fp32_sha256"] == previous_selected_endpoint_hash
                and row["current_x_T_fp32_sha256"] == previous_selected_x_hash,
                "L-BFGS accepted-state chain drift.",
            )
        if accepted:
            accepted_count += 1
            accepted_losses.append(float(best["loss"]))
            previous_selected_endpoint_hash = best["endpoint_fp32_sha256"]
            previous_selected_x_hash = best["candidate_x_T_fp32_sha256"]
        else:
            require(index == len(iteration_rows) - 1, "L-BFGS continued after rejection.")
            stop_reason = "no_acceptable_candidate"
    require(
        all(right < left for left, right in zip(accepted_losses, accepted_losses[1:])),
        "L-BFGS accepted losses are not strictly monotone.",
    )
    final = iterations[len(iteration_rows) - 1]
    if final["accepted"]:
        final_loss = float(final["selected_loss"])
        final_ratio = float(final["selected_loss_ratio_to_plateau"])
        final_x_hash = final["selected_x_T_fp32_sha256"]
        final_endpoint_hash = final["selected_endpoint_fp32_sha256"]
    else:
        final_loss = float(final["current_loss"])
        final_ratio = float(final["current_loss_ratio_to_plateau"])
        final_x_hash = final["current_x_T_fp32_sha256"]
        final_endpoint_hash = final["current_endpoint_fp32_sha256"]
    if stop_reason is None:
        stop_reason = (
            "success_threshold"
            if final_ratio <= config["algorithm"]["stop_after_loss_ratio_lte"]
            else "maximum_iterations"
        )
    curvature_counts = Counter(
        row["curvature_pair_reason"] for row in iteration_rows if row["curvature_pair_reason"] != "initial"
    )
    return {
        "iteration_count": len(iteration_rows),
        "accepted_update_count": accepted_count,
        "all_accepted_losses_strictly_monotone": True,
        "initial_loss": float(iterations[0]["current_loss"]),
        "initial_loss_ratio_to_plateau": float(iterations[0]["current_loss_ratio_to_plateau"]),
        "final_loss": final_loss,
        "final_loss_ratio_to_plateau": final_ratio,
        "final_x_T_fp32_sha256": final_x_hash,
        "final_endpoint_fp32_sha256": final_endpoint_hash,
        "stop_reason": stop_reason,
        "accepted_loss_sequence": accepted_losses,
        "restart_counts": dict(sorted(Counter(row["restart_reason"] for row in iteration_rows).items())),
        "curvature_pair_counts": dict(sorted(curvature_counts.items())),
        "maximum_history_size_used": max(row["history_size_used"] for row in iteration_rows),
    }


def classify_outcome(summary: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    ratio = float(summary["final_loss_ratio_to_plateau"])
    if (
        ratio <= config["fixed_target_success_gate"]["final_loss_ratio_to_fp32_plateau_lte"]
        and summary["accepted_update_count"] >= config["fixed_target_success_gate"]["accepted_update_count_gte"]
        and summary["all_accepted_losses_strictly_monotone"]
    ):
        return "fixed_target_trust_region_success"
    if ratio <= config["matched_budget_gate"]["material_final_loss_ratio_lte"]:
        return "lbfgs_material_improvement_only"
    if ratio < 1.0 and summary["accepted_update_count"] > 0:
        return "lbfgs_no_matched_budget_advantage"
    return "lbfgs_stalled"


def validate_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory = json.loads((root / "artifact_inventory.json").read_text(encoding="utf-8"))
    require(inventory.get("schema") == f"{PREFIX}-inventory.v1", "L-BFGS inventory schema drift.")
    listed: set[str] = set()
    for item in inventory.get("artifacts", []):
        relative = item.get("path")
        require(
            isinstance(relative, str)
            and relative not in listed
            and "\\" not in relative
            and not Path(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            "Unsafe L-BFGS inventory path.",
        )
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(), "Missing L-BFGS artifact.")
        require(
            path.stat().st_size == item.get("bytes") and sha256_file(path) == item.get("sha256"),
            "L-BFGS artifact bytes/hash mismatch.",
        )
        listed.add(relative)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    require(observed == listed and inventory.get("artifact_count") == len(listed), "Incomplete L-BFGS inventory.")
    return {"artifact_count": len(listed), "listed": listed}


def _valid_x(value: Any) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        and tuple(value.shape) == (1, 4, 128, 128)
        and torch.isfinite(value).all()
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    require(text.endswith("\n"), f"Missing terminal newline: {path}")
    return [json.loads(line) for line in text.splitlines()]


def _audit_parent_first_step(
    iteration: Mapping[str, Any], candidates: Mapping[tuple[int, int], Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    control = config["parent_first_step_control"]
    scalar_names = ("current_loss", "selected_loss", "selected_loss_ratio_to_plateau")
    hash_names = (
        "current_x_T_fp32_sha256",
        "current_endpoint_fp32_sha256",
        "gradient_fp32_sha256",
        "direction_fp32_sha256",
        "selected_x_T_fp32_sha256",
        "selected_endpoint_fp32_sha256",
    )
    require(
        all(iteration[name] == control[name] for name in scalar_names + hash_names)
        and iteration["selected_radius_index"] == control["selected_radius_index"]
        and iteration["selected_radius_l2"] == control["selected_radius_l2"]
        and iteration["restart_reason"] == "initial"
        and iteration["curvature_pair_reason"] == "initial"
        and iteration["history_size_used"] == 0
        and iteration["initial_inverse_hessian_scale"] == 1.0,
        "L-BFGS first iteration does not reproduce the parent.",
    )
    for radius_index, expected in enumerate(control["candidates"]):
        observed = candidates[(0, radius_index)]
        require(
            observed["radius_l2"] == expected["radius_l2"]
            and observed["candidate_x_T_fp32_sha256"] == expected["candidate_x_T_fp32_sha256"]
            and observed["endpoint_fp32_sha256"] == expected["endpoint_fp32_sha256"]
            and observed["loss"] == expected["loss"]
            and observed["loss_ratio_to_plateau"] == expected["loss_ratio_to_plateau"],
            "L-BFGS first candidate does not reproduce the parent.",
        )


def _primitive_metadata_matches(row: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    return all(row.get(name) == value for name, value in expected.items())


def audit_delivery(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    inventory = validate_inventory(root)
    required = {
        "config.json",
        "manifest.json",
        "result.json",
        "terminal.json",
        "runtime.json",
        "target/fixed_parent_target.pt",
        "parent/lr001-step256.pt",
        "trust_region_tensors.pt",
        "iteration_metrics.jsonl",
        "candidate_metrics.jsonl",
        "model_snapshot_verification_start.json",
        "model_snapshot_verification_end.json",
    }
    require(required.issubset(inventory["listed"]), "Required L-BFGS artifacts missing.")
    saved_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    require(saved_config == config and manifest.get("config_sha256") == canonical_sha(config), "L-BFGS config drift.")
    require(
        manifest.get("protocol") == result.get("protocol") == terminal.get("protocol") == PROTOCOL
        and manifest.get("git_commit") == result.get("git_commit") == terminal.get("git_commit"),
        "L-BFGS protocol/Git drift.",
    )
    information_boundary = manifest.get("optimization_information_boundary", {})
    require(
        information_boundary.get("direction_method") == config["algorithm"]["direction_method"]
        and information_boundary.get("direction_uses_only_current_gradient_and_accepted_history") is True
        and information_boundary.get("curvature_pairs_use_only_consecutive_accepted_x_and_gradients") is True
        and information_boundary.get("history_size") == HISTORY_SIZE
        and information_boundary.get("curvature_relative_threshold") == CURVATURE_RELATIVE_THRESHOLD
        and information_boundary.get("two_loop_cpu_float64") is True
        and information_boundary.get("matched_parent_prpplus_updates")
        == config["algorithm"]["maximum_formal_iterations"],
        "L-BFGS optimization information boundary drift.",
    )
    require(
        terminal.get("status") == "technical_completed"
        and terminal.get("engineering_gate") is True
        and terminal.get("exit_code") == 0
        and terminal.get("formal_success") is False
        and terminal.get("phase2_allowed") is False
        and result.get("engineering_gate") is True
        and result.get("formal_success") is False
        and result.get("phase2_allowed") is False,
        "L-BFGS result boundary drift.",
    )
    start_snapshot = json.loads((root / "model_snapshot_verification_start.json").read_text(encoding="utf-8"))
    end_snapshot = json.loads((root / "model_snapshot_verification_end.json").read_text(encoding="utf-8"))
    require(
        start_snapshot == end_snapshot
        and result.get("snapshots_unchanged") is True
        and result.get("models_frozen") is True
        and result.get("only_student_x_T_trainable") is True,
        "L-BFGS frozen/snapshot evidence drift.",
    )
    fixed = terminal_core.load_config()["fixed_parent_artifacts"]
    target_path = root / "target/fixed_parent_target.pt"
    plateau_path = root / "parent/lr001-step256.pt"
    require(
        sha256_file(target_path) == fixed["source_target_artifact_sha256"]
        and sha256_file(plateau_path) == fixed["source_plateau_checkpoint_sha256"],
        "Copied L-BFGS parent bytes drift.",
    )
    target = torch.load(target_path, map_location="cpu", weights_only=True)
    plateau = torch.load(plateau_path, map_location="cpu", weights_only=True)
    bundle_path = root / "trust_region_tensors.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    require(
        bundle.get("schema") == f"{PREFIX}-tensor-bundle.v1"
        and bundle.get("protocol") == PROTOCOL
        and result.get("tensor_bundle_sha256") == sha256_file(bundle_path),
        "L-BFGS tensor-bundle identity drift.",
    )
    anchor = config["fixed_fp32_anchor"]
    plateau_x = bundle.get("plateau_x_T_fp32")
    teacher_endpoint = bundle.get("teacher_endpoint_fp32")
    plateau_endpoint = bundle.get("plateau_endpoint_fp32")
    bf16_teacher = bundle.get("bf16_teacher_replay_endpoint_fp32")
    require(
        all(_valid_x(value) for value in (plateau_x, teacher_endpoint, plateau_endpoint, bf16_teacher))
        and torch.equal(plateau_x, plateau["student_x_T_fp32"])
        and torch.equal(bf16_teacher, target["teacher_endpoint_fp32"])
        and canonical_tensor_sha256(teacher_endpoint) == anchor["teacher_endpoint_fp32_sha256"]
        and canonical_tensor_sha256(plateau_endpoint) == anchor["plateau_endpoint_fp32_sha256"],
        "L-BFGS fixed FP32 anchor drift.",
    )
    baseline = float((plateau_endpoint - teacher_endpoint).square().mean())
    require(
        math.isclose(baseline, anchor["plateau_mse"], rel_tol=1e-6, abs_tol=1e-12)
        and result.get("fp32_lift_audit", {}).get("passed") is True,
        "L-BFGS baseline/precision lift drift.",
    )
    iteration_rows = _read_rows(root / "iteration_metrics.jsonl")
    candidate_rows = _read_rows(root / "candidate_metrics.jsonl")
    summary = summarize_trace(iteration_rows, candidate_rows, config)
    tensor_iterations = bundle.get("iterations", {})
    require(set(tensor_iterations) == set(range(len(iteration_rows))), "L-BFGS tensor coverage drift.")
    candidate_by_key = {validate_candidate_row(row): row for row in candidate_rows}
    iteration_by_index = {validate_iteration_row(row): row for row in iteration_rows}
    previous_x = None
    previous_gradient = None
    history: list[dict[str, Any]] = []
    for index in range(len(iteration_rows)):
        row = iteration_by_index[index]
        record = tensor_iterations[index]
        current_x = record.get("current_x_T_fp32")
        current_endpoint = record.get("current_endpoint_fp32")
        gradient = record.get("gradient_fp32")
        raw_direction = record.get("inverse_hessian_raw_direction_fp32")
        direction = record.get("negative_gradient_direction_fp32")
        require(
            all(_valid_x(value) for value in (current_x, current_endpoint, gradient, raw_direction, direction)),
            "Invalid L-BFGS current tensors.",
        )
        pair_state = form_curvature_pair(
            iteration=index,
            current_x=current_x,
            previous_x=previous_x,
            gradient=gradient,
            previous_gradient=previous_gradient,
        )
        saved_s = record.get("curvature_s_fp32")
        saved_y = record.get("curvature_y_fp32")
        require(
            (saved_s is None and pair_state["s"] is None or torch.equal(saved_s, pair_state["s"]))
            and (saved_y is None and pair_state["y"] is None or torch.equal(saved_y, pair_state["y"]))
            and _primitive_metadata_matches(row, curvature_row_metadata(pair_state)),
            "L-BFGS curvature reconstruction drift.",
        )
        if pair_state["accepted"]:
            history.append(pair_state["pair"])
            history = history[-HISTORY_SIZE:]
        expected = lbfgs_direction(gradient, history, iteration=index)
        expected_direction_metadata = direction_row_metadata(expected)
        require(
            torch.equal(raw_direction, expected["raw_direction"])
            and torch.equal(direction, expected["direction"])
            and canonical_tensor_sha256(current_x) == row["current_x_T_fp32_sha256"]
            and canonical_tensor_sha256(current_endpoint) == row["current_endpoint_fp32_sha256"]
            and canonical_tensor_sha256(gradient) == row["gradient_fp32_sha256"]
            and canonical_tensor_sha256(raw_direction) == row["raw_direction_fp32_sha256"]
            and canonical_tensor_sha256(direction) == row["direction_fp32_sha256"]
            and _primitive_metadata_matches(row, expected_direction_metadata),
            "L-BFGS two-loop recurrence or tensor hash drift.",
        )
        if expected["history_cleared"]:
            history = []
        require(
            record.get("history_pair_iterations_after_direction")
            == expected["history_pair_iterations_after_direction"],
            "L-BFGS saved history-state drift.",
        )
        current_loss = float((current_endpoint - teacher_endpoint).square().mean())
        selected_candidate = record["candidates"][row["selected_radius_index"]]
        selected_residual_l2 = float((selected_candidate["x_T_fp32"] - plateau_x).double().norm())
        require(
            math.isclose(current_loss, row["current_loss"], rel_tol=1e-6, abs_tol=1e-12)
            and math.isclose(
                current_loss / baseline,
                row["current_loss_ratio_to_plateau"],
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
            and math.isclose(float(gradient.double().norm()), row["gradient_norm"], rel_tol=1e-6, abs_tol=1e-12)
            and math.isclose(
                float((gradient != 0).double().mean()), row["gradient_nonzero_fraction"], rel_tol=0.0, abs_tol=0.0
            )
            and row["gradient_nonzero_fraction"] >= config["algorithm"]["gradient_nonzero_fraction_gte"]
            and math.isclose(selected_residual_l2, row["selected_candidate_residual_l2"], rel_tol=1e-6, abs_tol=1e-12),
            "L-BFGS current metric drift.",
        )
        candidates = record.get("candidates", {})
        require(set(candidates) == set(range(len(RADII))), "L-BFGS candidate tensor coverage drift.")
        for radius_index, radius in enumerate(RADII):
            candidate = candidates[radius_index]
            candidate_x = candidate.get("x_T_fp32")
            endpoint = candidate.get("endpoint_fp32")
            metric = candidate_by_key[(index, radius_index)]
            expected_x = current_x + radius * direction
            loss = float((endpoint - teacher_endpoint).square().mean())
            require(
                _valid_x(candidate_x)
                and _valid_x(endpoint)
                and torch.equal(candidate_x, expected_x)
                and canonical_tensor_sha256(candidate_x) == metric["candidate_x_T_fp32_sha256"]
                and canonical_tensor_sha256(endpoint) == metric["endpoint_fp32_sha256"]
                and math.isclose(loss, metric["loss"], rel_tol=1e-6, abs_tol=1e-12)
                and math.isclose(loss / baseline, metric["loss_ratio_to_plateau"], rel_tol=1e-6, abs_tol=1e-12)
                and math.isclose(loss / current_loss, metric["loss_ratio_to_current"], rel_tol=1e-6, abs_tol=1e-12),
                "L-BFGS candidate tensor/metric drift.",
            )
        previous_x = current_x.clone()
        previous_gradient = gradient.clone()
    _audit_parent_first_step(iteration_by_index[0], candidate_by_key, config)
    final_x = bundle.get("final_x_T_fp32")
    final_endpoint = bundle.get("final_endpoint_fp32")
    final_replay = bundle.get("final_replay_endpoint_fp32")
    require(
        all(_valid_x(value) for value in (final_x, final_endpoint, final_replay))
        and torch.equal(final_endpoint, final_replay)
        and canonical_tensor_sha256(final_x) == summary["final_x_T_fp32_sha256"]
        and canonical_tensor_sha256(final_endpoint) == summary["final_endpoint_fp32_sha256"]
        and result.get("final_replay_bitwise_equal") is True,
        "L-BFGS final replay drift.",
    )
    mode = result.get("mode")
    require(mode in ("technical-preflight", "formal") and terminal.get("mode") == mode, "L-BFGS mode drift.")
    expected_counters = {
        "full_chain_forward_calls": 3 + 6 * len(iteration_rows),
        "reader_forward_calls": 0,
        "backward_calls": len(iteration_rows),
        "optimizer_steps": 0,
        "parameter_updates": summary["accepted_update_count"],
    }
    require(result.get("counters") == expected_counters, "L-BFGS execution counter drift.")
    classification = classify_outcome(summary, config) if mode == "formal" else None
    require(
        result.get("trace_summary") == summary
        and result.get("classification") == classification
        and result.get("fixed_target_optimization_success") is (classification == "fixed_target_trust_region_success"),
        "L-BFGS summary/classification drift.",
    )
    if mode == "technical-preflight":
        require(
            len(iteration_rows) == 1 and summary["accepted_update_count"] == 1 and result.get("preflight_gate") is True,
            "L-BFGS preflight execution drift.",
        )
    else:
        require(
            len(iteration_rows) <= config["algorithm"]["maximum_formal_iterations"],
            "L-BFGS formal iteration budget exceeded.",
        )
        prerequisite = manifest.get("validation", {}).get("preflight_prerequisite", {})
        require(
            prerequisite.get("passed") is True
            and prerequisite.get("mode") == "technical-preflight"
            and prerequisite.get("git_commit") == result["git_commit"],
            "L-BFGS preflight chain drift.",
        )
    accepted_indices = [row["iteration"] for row in iteration_rows if row["accepted"]]
    checkpoint_files = {f"checkpoints/{iteration_id(index)}.pt" for index in accepted_indices}
    require(checkpoint_files.issubset(inventory["listed"]), "L-BFGS accepted checkpoints missing.")
    metadata_names = (
        "curvature_pair_iteration",
        "curvature_pair_accepted",
        "curvature_pair_reason",
        "curvature_s_fp32_sha256",
        "curvature_y_fp32_sha256",
        "curvature_s_dot_y",
        "curvature_s_norm",
        "curvature_y_norm",
        "curvature_cosine",
        "curvature_threshold_value",
        "restart_reason",
        "history_cleared",
        "history_pair_iterations_before_direction",
        "history_pair_iterations_used",
        "history_pair_iterations_after_direction",
        "history_size_before_direction",
        "history_size_used",
        "history_size_after_direction",
        "initial_inverse_hessian_scale",
        "two_loop_coefficients",
        "negative_gradient_cosine",
        "analytic_directional_derivative",
    )
    for index in accepted_indices:
        checkpoint = torch.load(root / f"checkpoints/{iteration_id(index)}.pt", map_location="cpu", weights_only=True)
        row = iteration_by_index[index]
        record = tensor_iterations[index]
        require(
            checkpoint.get("schema") == f"{PREFIX}-checkpoint.v1"
            and checkpoint.get("protocol") == PROTOCOL
            and checkpoint.get("iteration") == index
            and canonical_tensor_sha256(checkpoint["student_x_T_fp32"]) == row["selected_x_T_fp32_sha256"]
            and canonical_tensor_sha256(checkpoint["endpoint_fp32"]) == row["selected_endpoint_fp32_sha256"]
            and canonical_tensor_sha256(checkpoint["gradient_fp32"]) == row["gradient_fp32_sha256"]
            and canonical_tensor_sha256(checkpoint["inverse_hessian_raw_direction_fp32"])
            == row["raw_direction_fp32_sha256"]
            and canonical_tensor_sha256(checkpoint["negative_gradient_direction_fp32"]) == row["direction_fp32_sha256"]
            and (
                checkpoint.get("curvature_s_fp32") is None
                and record.get("curvature_s_fp32") is None
                or torch.equal(checkpoint["curvature_s_fp32"], record["curvature_s_fp32"])
            )
            and (
                checkpoint.get("curvature_y_fp32") is None
                and record.get("curvature_y_fp32") is None
                or torch.equal(checkpoint["curvature_y_fp32"], record["curvature_y_fp32"])
            )
            and all(checkpoint.get(name) == row[name] for name in metadata_names),
            "L-BFGS checkpoint drift.",
        )
    return {
        "passed": True,
        "mode": mode,
        "artifact_count": inventory["artifact_count"],
        "git_commit": result["git_commit"],
        "classification": classification,
        "trace_summary": summary,
        "fixed_target_optimization_success": bool(classification == "fixed_target_trust_region_success"),
        "formal_success": False,
        "phase2_allowed": False,
    }
