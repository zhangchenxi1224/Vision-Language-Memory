"""Contracts and independent audit for the FP32 PRP+ trust-region control."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vision_memory.repro import canonical_tensor_sha256
from vision_memory.training import r11_new_fp32_trust_region as trust_core
from vision_memory.training import r11_new_fp32_trust_region_horizon128 as horizon_core
from vision_memory.training import r11_new_oracle_terminal_capture as terminal_core


PROTOCOL = "R11-New-FP32-PRPPlus-Trust-Region-Target01"
PREFIX = "vision_memory.r11-new-fp32-prpplus-trust-region"
CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs/experiments/r11_new_fp32_prpplus_trust_region_target01.json"
)
CONFIG_BYTES_SHA256 = "7786f9d9175e9c087bba0d2b12bcd08c42acad43356cf1158b91c4fe5aae7312"
CONFIG_CANONICAL_SHA256 = "6a3443aadc96e342631559884a988f18d18568bbb8f42c0d4b43c5213f742c35"
RADII = trust_core.RADII
RESTART_REASONS = ("initial", "none", "negative_beta_clamped", "non_descent")

require = trust_core.require
sha256_file = trust_core.sha256_file
canonical_sha = trust_core.canonical_sha
unit_vector = trust_core.unit_vector
cosine = trust_core.cosine


def load_config() -> dict[str, Any]:
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "PRP+ config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(canonical_sha(value) == CONFIG_CANONICAL_SHA256, "PRP+ config canonical hash drift.")
    require(
        value.get("schema") == f"{PREFIX}-config.v1"
        and value.get("protocol") == PROTOCOL
        and value.get("experiment_variant") == "prpplus_matched_horizon128",
        "PRP+ config identity drift.",
    )
    parent = horizon_core.load_config()
    require(
        value["base_precision_contract"] == parent["base_precision_contract"]
        and value["parent_precision_result"] == parent["parent_precision_result"]
        and value["fixed_fp32_anchor"] == parent["fixed_fp32_anchor"],
        "PRP+ changed the fixed precision, target, or plateau contract.",
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
        and algorithm["direction_method"] == "Polak-Ribiere-Polyak plus nonlinear conjugate gradient",
        "PRP+ algorithm boundary drift.",
    )
    fixed_parent_fields = (
        "parameterization",
        "initial_residual_delta",
        "trainable_values",
        "frozen_values",
        "loss",
        "gradient_nonzero_fraction_gte",
        "candidate_radii_l2_in_fixed_order",
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
        "PRP+ changed a non-direction horizon-128 algorithm field.",
    )
    parent_result = value["parent_horizon_result"]
    require(
        parent_result["delivery_commit"] == "88dddced0656cdc57d29bb9d10907de708cad76c"
        and parent_result["classification"] == "strong_capture_reached_only"
        and parent_result["iteration_count"] == 128
        and parent_result["accepted_update_count"] == 128
        and parent_result["stop_reason"] == "maximum_iterations"
        and parent_result["formal_success"] is False
        and parent_result["phase2_allowed"] is False,
        "PRP+ parent horizon result identity drift.",
    )
    require(
        value["technical_preflight_gate"]["full_chain_forward_calls"] == 9
        and value["formal_gate"]["maximum_full_chain_forward_calls"] == 771
        and value["fixed_target_success_gate"]["classification"] == "fixed_target_trust_region_success"
        and value["matched_budget_gate"]["parent_final_loss_ratio"] == parent_result["final_loss_ratio_to_plateau"]
        and value["interpretation_boundaries"]["matched_update_and_forward_budget"] is True
        and value["interpretation_boundaries"]["formal_picture_memory_success_always_false"] is True
        and value["interpretation_boundaries"]["phase2_always_false"] is True,
        "PRP+ gate or interpretation boundary drift.",
    )
    return value


def iteration_id(iteration: int) -> str:
    require(type(iteration) is int and iteration >= 0, "Invalid PRP+ iteration.")
    return f"iteration-{iteration:02d}"


def candidate_id(iteration: int, radius_index: int) -> str:
    require(
        type(iteration) is int and iteration >= 0 and type(radius_index) is int and 0 <= radius_index < len(RADII),
        "Invalid PRP+ candidate identity.",
    )
    return f"iteration-{iteration:02d}__radius-{radius_index:02d}"


def _finite_number(value: Any, *, nonnegative: bool = False) -> bool:
    return bool(type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0.0))


def prp_plus_direction(
    gradient: torch.Tensor,
    previous_gradient: torch.Tensor | None,
    previous_raw_direction: torch.Tensor | None,
) -> dict[str, Any]:
    """Construct the preregistered deterministic PRP+ descent direction."""
    require(
        isinstance(gradient, torch.Tensor)
        and gradient.dtype == torch.float32
        and bool(torch.isfinite(gradient).all())
        and float(gradient.double().norm()) > 0.0,
        "Invalid PRP+ gradient.",
    )
    if previous_gradient is None or previous_raw_direction is None:
        require(previous_gradient is None and previous_raw_direction is None, "Incomplete initial PRP+ state.")
        beta_raw = 0.0
        beta = 0.0
        restart_reason = "initial"
        raw_direction = -gradient
    else:
        require(
            previous_gradient.dtype == torch.float32
            and previous_raw_direction.dtype == torch.float32
            and previous_gradient.shape == gradient.shape
            and previous_raw_direction.shape == gradient.shape
            and bool(torch.isfinite(previous_gradient).all())
            and bool(torch.isfinite(previous_raw_direction).all()),
            "Invalid previous PRP+ state.",
        )
        denominator = float((previous_gradient.double() * previous_gradient.double()).sum())
        require(denominator > 0.0 and math.isfinite(denominator), "Invalid PRP+ denominator.")
        gradient_difference = gradient.double() - previous_gradient.double()
        beta_raw = float((gradient.double() * gradient_difference).sum()) / denominator
        require(math.isfinite(beta_raw), "Nonfinite PRP+ beta.")
        beta = max(0.0, beta_raw)
        restart_reason = "negative_beta_clamped" if beta_raw < 0.0 else "none"
        raw_direction = -gradient + beta * previous_raw_direction
        derivative = float((gradient.double() * raw_direction.double()).sum())
        if not bool(torch.isfinite(raw_direction).all()) or not math.isfinite(derivative) or derivative >= 0.0:
            beta = 0.0
            restart_reason = "non_descent"
            raw_direction = -gradient
    raw_direction = raw_direction.detach().to(dtype=torch.float32).clone()
    direction = unit_vector(raw_direction)
    directional_derivative = float((gradient.double() * direction.double()).sum())
    negative_gradient_cosine = cosine(direction, -gradient)
    require(
        bool(torch.isfinite(direction).all())
        and directional_derivative < 0.0
        and negative_gradient_cosine > 0.0
        and abs(float(direction.double().norm()) - 1.0) <= 1e-6,
        "PRP+ failed to produce a unit descent direction.",
    )
    return {
        "raw_direction": raw_direction,
        "direction": direction,
        "beta_raw": beta_raw,
        "beta": beta,
        "restart_reason": restart_reason,
        "directional_derivative": directional_derivative,
        "negative_gradient_cosine": negative_gradient_cosine,
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
        "Invalid PRP+ candidate row.",
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
        "Invalid PRP+ candidate value.",
    )
    return iteration, radius_index


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
        "Invalid PRP+ iteration row.",
    )
    require(
        all(
            _finite_number(row.get(name), nonnegative=True)
            for name in (
                "current_loss",
                "current_loss_ratio_to_plateau",
                "gradient_norm",
                "gradient_nonzero_fraction",
                "beta",
                "negative_gradient_cosine",
                "selected_loss",
                "selected_loss_ratio_to_plateau",
                "selected_candidate_residual_l2",
            )
        )
        and _finite_number(row.get("beta_raw"))
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
        "Invalid PRP+ iteration value.",
    )
    previous_hashes = (row.get("previous_gradient_fp32_sha256"), row.get("previous_raw_direction_fp32_sha256"))
    if iteration == 0:
        require(previous_hashes == (None, None) and row["restart_reason"] == "initial", "Invalid initial PRP+ row.")
    else:
        require(
            all(isinstance(value, str) and len(value) == 64 for value in previous_hashes),
            "Missing previous PRP+ hashes.",
        )
    return iteration


def summarize_trace(
    iteration_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    require(iteration_rows, "PRP+ trace is empty.")
    require(len(candidate_rows) == len(iteration_rows) * len(RADII), "PRP+ candidate count drift.")
    iterations: dict[int, Mapping[str, Any]] = {}
    for row in iteration_rows:
        index = validate_iteration_row(row)
        require(index not in iterations, "Duplicate PRP+ iteration row.")
        iterations[index] = row
    require(set(iterations) == set(range(len(iteration_rows))), "PRP+ iteration coverage drift.")
    candidates: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in candidate_rows:
        key = validate_candidate_row(row)
        require(key not in candidates, "Duplicate PRP+ candidate row.")
        candidates[key] = row
    require(
        set(candidates)
        == {
            (iteration, radius_index) for iteration in range(len(iteration_rows)) for radius_index in range(len(RADII))
        },
        "PRP+ candidate coverage drift.",
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
            "PRP+ candidate selection drift.",
        )
        relative_improvement = (row["current_loss"] - best["loss"]) / row["current_loss"]
        accepted = relative_improvement >= minimum_improvement
        require(
            row["accepted"] == accepted
            and math.isclose(row["relative_improvement"], relative_improvement, rel_tol=1e-9, abs_tol=1e-12),
            "PRP+ acceptance drift.",
        )
        if previous_selected_endpoint_hash is not None:
            require(
                row["current_endpoint_fp32_sha256"] == previous_selected_endpoint_hash
                and row["current_x_T_fp32_sha256"] == previous_selected_x_hash,
                "PRP+ accepted-state chain drift.",
            )
        if accepted:
            accepted_count += 1
            accepted_losses.append(float(best["loss"]))
            previous_selected_endpoint_hash = best["endpoint_fp32_sha256"]
            previous_selected_x_hash = best["candidate_x_T_fp32_sha256"]
        else:
            require(index == len(iteration_rows) - 1, "PRP+ continued after rejection.")
            stop_reason = "no_acceptable_candidate"
    require(
        all(right < left for left, right in zip(accepted_losses, accepted_losses[1:])),
        "PRP+ accepted losses are not strictly monotone.",
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
        return "prpplus_material_improvement_only"
    if ratio < 1.0 and summary["accepted_update_count"] > 0:
        return "prpplus_no_matched_budget_advantage"
    return "prpplus_stalled"


def validate_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory = json.loads((root / "artifact_inventory.json").read_text(encoding="utf-8"))
    require(inventory.get("schema") == f"{PREFIX}-inventory.v1", "PRP+ inventory schema drift.")
    listed: set[str] = set()
    for item in inventory.get("artifacts", []):
        relative = item.get("path")
        require(
            isinstance(relative, str)
            and relative not in listed
            and "\\" not in relative
            and not Path(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            "Unsafe PRP+ inventory path.",
        )
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(), "Missing PRP+ artifact.")
        require(
            path.stat().st_size == item.get("bytes") and sha256_file(path) == item.get("sha256"),
            "PRP+ artifact bytes/hash mismatch.",
        )
        listed.add(relative)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    require(observed == listed and inventory.get("artifact_count") == len(listed), "Incomplete PRP+ inventory.")
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
        and iteration["beta"] == 0.0
        and iteration["beta_raw"] == 0.0,
        "PRP+ first iteration does not reproduce the parent.",
    )
    for radius_index, expected in enumerate(control["candidates"]):
        observed = candidates[(0, radius_index)]
        require(
            observed["radius_l2"] == expected["radius_l2"]
            and observed["candidate_x_T_fp32_sha256"] == expected["candidate_x_T_fp32_sha256"]
            and observed["endpoint_fp32_sha256"] == expected["endpoint_fp32_sha256"]
            and observed["loss"] == expected["loss"]
            and observed["loss_ratio_to_plateau"] == expected["loss_ratio_to_plateau"],
            "PRP+ first candidate does not reproduce the parent.",
        )


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
    require(required.issubset(inventory["listed"]), "Required PRP+ artifacts missing.")
    saved_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    require(saved_config == config and manifest.get("config_sha256") == canonical_sha(config), "PRP+ config drift.")
    require(
        manifest.get("protocol") == result.get("protocol") == terminal.get("protocol") == PROTOCOL
        and manifest.get("git_commit") == result.get("git_commit") == terminal.get("git_commit"),
        "PRP+ protocol/Git drift.",
    )
    information_boundary = manifest.get("optimization_information_boundary", {})
    require(
        information_boundary.get("direction_method") == config["algorithm"]["direction_method"]
        and information_boundary.get(
            "direction_uses_only_current_gradient_previous_gradient_and_previous_raw_direction"
        )
        is True
        and information_boundary.get("matched_parent_horizon_updates")
        == config["algorithm"]["maximum_formal_iterations"],
        "PRP+ optimization information boundary drift.",
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
        "PRP+ result boundary drift.",
    )
    start_snapshot = json.loads((root / "model_snapshot_verification_start.json").read_text(encoding="utf-8"))
    end_snapshot = json.loads((root / "model_snapshot_verification_end.json").read_text(encoding="utf-8"))
    require(
        start_snapshot == end_snapshot
        and result.get("snapshots_unchanged") is True
        and result.get("models_frozen") is True
        and result.get("only_student_x_T_trainable") is True,
        "PRP+ frozen/snapshot evidence drift.",
    )
    fixed = terminal_core.load_config()["fixed_parent_artifacts"]
    target_path = root / "target/fixed_parent_target.pt"
    plateau_path = root / "parent/lr001-step256.pt"
    require(
        sha256_file(target_path) == fixed["source_target_artifact_sha256"]
        and sha256_file(plateau_path) == fixed["source_plateau_checkpoint_sha256"],
        "Copied PRP+ parent bytes drift.",
    )
    target = torch.load(target_path, map_location="cpu", weights_only=True)
    plateau = torch.load(plateau_path, map_location="cpu", weights_only=True)
    bundle_path = root / "trust_region_tensors.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    require(
        bundle.get("schema") == f"{PREFIX}-tensor-bundle.v1"
        and bundle.get("protocol") == PROTOCOL
        and result.get("tensor_bundle_sha256") == sha256_file(bundle_path),
        "PRP+ tensor-bundle identity drift.",
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
        "PRP+ fixed FP32 anchor drift.",
    )
    baseline = float((plateau_endpoint - teacher_endpoint).square().mean())
    require(
        math.isclose(baseline, anchor["plateau_mse"], rel_tol=1e-6, abs_tol=1e-12)
        and result.get("fp32_lift_audit", {}).get("passed") is True,
        "PRP+ baseline/precision lift drift.",
    )
    iteration_rows = _read_rows(root / "iteration_metrics.jsonl")
    candidate_rows = _read_rows(root / "candidate_metrics.jsonl")
    summary = summarize_trace(iteration_rows, candidate_rows, config)
    tensor_iterations = bundle.get("iterations", {})
    require(set(tensor_iterations) == set(range(len(iteration_rows))), "PRP+ tensor coverage drift.")
    candidate_by_key = {validate_candidate_row(row): row for row in candidate_rows}
    iteration_by_index = {validate_iteration_row(row): row for row in iteration_rows}
    previous_gradient = None
    previous_raw_direction = None
    for index in range(len(iteration_rows)):
        row = iteration_by_index[index]
        record = tensor_iterations[index]
        current_x = record.get("current_x_T_fp32")
        current_endpoint = record.get("current_endpoint_fp32")
        gradient = record.get("gradient_fp32")
        raw_direction = record.get("conjugate_raw_direction_fp32")
        direction = record.get("negative_gradient_direction_fp32")
        require(
            all(_valid_x(value) for value in (current_x, current_endpoint, gradient, raw_direction, direction)),
            "Invalid PRP+ current tensors.",
        )
        expected = prp_plus_direction(gradient, previous_gradient, previous_raw_direction)
        require(
            torch.equal(raw_direction, expected["raw_direction"])
            and torch.equal(direction, expected["direction"])
            and canonical_tensor_sha256(current_x) == row["current_x_T_fp32_sha256"]
            and canonical_tensor_sha256(current_endpoint) == row["current_endpoint_fp32_sha256"]
            and canonical_tensor_sha256(gradient) == row["gradient_fp32_sha256"]
            and canonical_tensor_sha256(raw_direction) == row["raw_direction_fp32_sha256"]
            and canonical_tensor_sha256(direction) == row["direction_fp32_sha256"]
            and row["beta_raw"] == expected["beta_raw"]
            and row["beta"] == expected["beta"]
            and row["restart_reason"] == expected["restart_reason"]
            and math.isclose(
                row["analytic_directional_derivative"],
                expected["directional_derivative"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            and math.isclose(
                row["negative_gradient_cosine"],
                expected["negative_gradient_cosine"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            ),
            "PRP+ recurrence or tensor hash drift.",
        )
        if index == 0:
            require(
                record.get("previous_gradient_fp32") is None
                and record.get("previous_raw_direction_fp32") is None
                and row["previous_gradient_fp32_sha256"] is None
                and row["previous_raw_direction_fp32_sha256"] is None,
                "Unexpected initial PRP+ history.",
            )
        else:
            require(
                torch.equal(record.get("previous_gradient_fp32"), previous_gradient)
                and torch.equal(record.get("previous_raw_direction_fp32"), previous_raw_direction)
                and row["previous_gradient_fp32_sha256"] == canonical_tensor_sha256(previous_gradient)
                and row["previous_raw_direction_fp32_sha256"] == canonical_tensor_sha256(previous_raw_direction),
                "PRP+ saved history drift.",
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
            "PRP+ current metric drift.",
        )
        candidates = record.get("candidates", {})
        require(set(candidates) == set(range(len(RADII))), "PRP+ candidate tensor coverage drift.")
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
                "PRP+ candidate tensor/metric drift.",
            )
        previous_gradient = gradient.clone()
        previous_raw_direction = raw_direction.clone()
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
        "PRP+ final replay drift.",
    )
    mode = result.get("mode")
    require(mode in ("technical-preflight", "formal") and terminal.get("mode") == mode, "PRP+ mode drift.")
    expected_counters = {
        "full_chain_forward_calls": 3 + 6 * len(iteration_rows),
        "reader_forward_calls": 0,
        "backward_calls": len(iteration_rows),
        "optimizer_steps": 0,
        "parameter_updates": summary["accepted_update_count"],
    }
    require(result.get("counters") == expected_counters, "PRP+ execution counter drift.")
    classification = classify_outcome(summary, config) if mode == "formal" else None
    require(
        result.get("trace_summary") == summary
        and result.get("classification") == classification
        and result.get("fixed_target_optimization_success") is (classification == "fixed_target_trust_region_success"),
        "PRP+ summary/classification drift.",
    )
    if mode == "technical-preflight":
        require(
            len(iteration_rows) == 1 and summary["accepted_update_count"] == 1 and result.get("preflight_gate") is True,
            "PRP+ preflight execution drift.",
        )
    else:
        require(
            len(iteration_rows) <= config["algorithm"]["maximum_formal_iterations"],
            "PRP+ formal iteration budget exceeded.",
        )
        prerequisite = manifest.get("validation", {}).get("preflight_prerequisite", {})
        require(
            prerequisite.get("passed") is True
            and prerequisite.get("mode") == "technical-preflight"
            and prerequisite.get("git_commit") == result["git_commit"],
            "PRP+ preflight chain drift.",
        )
    accepted_indices = [row["iteration"] for row in iteration_rows if row["accepted"]]
    checkpoint_files = {f"checkpoints/{iteration_id(index)}.pt" for index in accepted_indices}
    require(checkpoint_files.issubset(inventory["listed"]), "PRP+ accepted checkpoints missing.")
    for index in accepted_indices:
        checkpoint = torch.load(root / f"checkpoints/{iteration_id(index)}.pt", map_location="cpu", weights_only=True)
        row = iteration_by_index[index]
        require(
            checkpoint.get("schema") == f"{PREFIX}-checkpoint.v1"
            and checkpoint.get("protocol") == PROTOCOL
            and checkpoint.get("iteration") == index
            and canonical_tensor_sha256(checkpoint["student_x_T_fp32"]) == row["selected_x_T_fp32_sha256"]
            and canonical_tensor_sha256(checkpoint["endpoint_fp32"]) == row["selected_endpoint_fp32_sha256"]
            and canonical_tensor_sha256(checkpoint["gradient_fp32"]) == row["gradient_fp32_sha256"]
            and canonical_tensor_sha256(checkpoint["conjugate_raw_direction_fp32"]) == row["raw_direction_fp32_sha256"]
            and canonical_tensor_sha256(checkpoint["negative_gradient_direction_fp32"]) == row["direction_fp32_sha256"]
            and checkpoint["beta_raw"] == row["beta_raw"]
            and checkpoint["beta"] == row["beta"]
            and checkpoint["restart_reason"] == row["restart_reason"],
            "PRP+ checkpoint drift.",
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
