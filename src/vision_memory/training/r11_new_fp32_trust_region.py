"""Contracts and independent audit for the R11_new FP32 trust-region control."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vision_memory.repro import canonical_tensor_sha256
from vision_memory.training import r11_new_activation_precision_control as precision_core
from vision_memory.training import r11_new_oracle_terminal_capture as terminal_core


PROTOCOL = "R11-New-FP32-Trust-Region-Target01"
PREFIX = "vision_memory.r11-new-fp32-trust-region"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/experiments/r11_new_fp32_trust_region_target01.json"
CONFIG_BYTES_SHA256 = "f533d376aa9d377ec1a0ed04ad6cffc6977f3e05b9cfa00b6df32dd8502d5d01"
CONFIG_CANONICAL_SHA256 = "8d1e7ed000e4016c7cd15c3ca0bc507c144ebfba61c8510a55b23ae2a46e6c11"
RADII = (0.1, 0.03, 0.01, 0.003, 0.001)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_config() -> dict[str, Any]:
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "Trust config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(
        canonical_sha(value) == CONFIG_CANONICAL_SHA256,
        "Trust config canonical hash drift.",
    )
    require(
        value.get("schema") == f"{PREFIX}-config.v1" and value.get("protocol") == PROTOCOL,
        "Trust config identity drift.",
    )
    algorithm = value["algorithm"]
    require(
        tuple(algorithm["candidate_radii_l2_in_fixed_order"]) == RADII
        and algorithm["maximum_formal_iterations"] == 32
        and algorithm["technical_preflight_iterations"] == 1
        and algorithm["stop_after_loss_ratio_lte"] == 0.01
        and algorithm["no_torch_optimizer"] is True
        and algorithm["no_gradient_clipping"] is True
        and value["interpretation_boundaries"]["formal_picture_memory_success_always_false"] is True
        and value["interpretation_boundaries"]["phase2_always_false"] is True,
        "Trust algorithm/boundary drift.",
    )
    return value


def unit_vector(value: torch.Tensor) -> torch.Tensor:
    return precision_core.unit_vector(value)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return precision_core.cosine(left, right)


def iteration_id(iteration: int) -> str:
    require(type(iteration) is int and iteration >= 0, "Invalid trust iteration.")
    return f"iteration-{iteration:02d}"


def candidate_id(iteration: int, radius_index: int) -> str:
    require(
        type(iteration) is int and iteration >= 0 and type(radius_index) is int and 0 <= radius_index < len(RADII),
        "Invalid trust candidate identity.",
    )
    return f"iteration-{iteration:02d}__radius-{radius_index:02d}"


def _finite_number(value: Any, *, nonnegative: bool = False) -> bool:
    return bool(type(value) in (int, float) and math.isfinite(value) and (not nonnegative or value >= 0.0))


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
        "Invalid trust candidate row.",
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
        "Invalid trust candidate value.",
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
        and row.get("selected_candidate_id") == candidate_id(iteration, row["selected_radius_index"]),
        "Invalid trust iteration row.",
    )
    require(
        all(
            _finite_number(row.get(name), nonnegative=True)
            for name in (
                "current_loss",
                "current_loss_ratio_to_plateau",
                "gradient_norm",
                "gradient_nonzero_fraction",
                "selected_loss",
                "selected_loss_ratio_to_plateau",
                "selected_candidate_residual_l2",
            )
        )
        and _finite_number(row.get("relative_improvement"))
        and _finite_number(row.get("analytic_directional_derivative"))
        and row["analytic_directional_derivative"] < 0.0
        and all(
            isinstance(row.get(name), str) and len(row[name]) == 64
            for name in (
                "current_x_T_fp32_sha256",
                "current_endpoint_fp32_sha256",
                "gradient_fp32_sha256",
                "direction_fp32_sha256",
                "selected_x_T_fp32_sha256",
                "selected_endpoint_fp32_sha256",
            )
        ),
        "Invalid trust iteration value.",
    )
    return iteration


def summarize_trace(
    iteration_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    require(iteration_rows, "Trust trace is empty.")
    require(
        len(candidate_rows) == len(iteration_rows) * len(RADII),
        "Trust candidate count drift.",
    )
    iterations: dict[int, Mapping[str, Any]] = {}
    for row in iteration_rows:
        index = validate_iteration_row(row)
        require(index not in iterations, "Duplicate trust iteration row.")
        iterations[index] = row
    require(set(iterations) == set(range(len(iteration_rows))), "Trust iteration coverage drift.")
    candidates: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in candidate_rows:
        key = validate_candidate_row(row)
        require(key not in candidates, "Duplicate trust candidate row.")
        candidates[key] = row
    require(
        set(candidates)
        == {
            (iteration, radius_index) for iteration in range(len(iteration_rows)) for radius_index in range(len(RADII))
        },
        "Trust candidate coverage drift.",
    )
    accepted_losses = []
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
            "Trust candidate selection drift.",
        )
        relative_improvement = (row["current_loss"] - best["loss"]) / row["current_loss"]
        accepted = relative_improvement >= minimum_improvement
        require(
            row["accepted"] == accepted
            and math.isclose(row["relative_improvement"], relative_improvement, rel_tol=1e-9, abs_tol=1e-12),
            "Trust acceptance drift.",
        )
        if previous_selected_endpoint_hash is not None:
            require(
                row["current_endpoint_fp32_sha256"] == previous_selected_endpoint_hash
                and row["current_x_T_fp32_sha256"] == previous_selected_x_hash,
                "Trust accepted-state chain drift.",
            )
        if accepted:
            accepted_count += 1
            accepted_losses.append(float(best["loss"]))
            previous_selected_endpoint_hash = best["endpoint_fp32_sha256"]
            previous_selected_x_hash = best["candidate_x_T_fp32_sha256"]
        else:
            require(index == len(iteration_rows) - 1, "Trust continued after rejection.")
            stop_reason = "no_acceptable_candidate"
    require(
        all(right < left for left, right in zip(accepted_losses, accepted_losses[1:])),
        "Trust accepted losses are not strictly monotone.",
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
    }


def classify_outcome(summary: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    ratio = float(summary["final_loss_ratio_to_plateau"])
    if (
        ratio <= config["fixed_target_success_gate"]["final_loss_ratio_to_fp32_plateau_lte"]
        and summary["accepted_update_count"] >= config["fixed_target_success_gate"]["accepted_update_count_gte"]
        and summary["all_accepted_losses_strictly_monotone"]
    ):
        return "fixed_target_trust_region_success"
    if ratio <= 0.1:
        return "strong_capture_reached_only"
    if ratio < 1.0 and summary["accepted_update_count"] > 0:
        return "monotone_progress_above_capture"
    return "trust_region_stalled"


def validate_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory = json.loads((root / "artifact_inventory.json").read_text(encoding="utf-8"))
    require(inventory.get("schema") == f"{PREFIX}-inventory.v1", "Trust inventory schema drift.")
    listed: set[str] = set()
    for item in inventory.get("artifacts", []):
        relative = item.get("path")
        require(
            isinstance(relative, str)
            and relative not in listed
            and "\\" not in relative
            and not Path(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            "Unsafe trust inventory path.",
        )
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root) and path.is_file() and not path.is_symlink(),
            "Escaping/missing trust artifact.",
        )
        require(
            path.stat().st_size == item.get("bytes") and sha256_file(path) == item.get("sha256"),
            "Trust artifact bytes/hash mismatch.",
        )
        listed.add(relative)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    require(
        observed == listed and inventory.get("artifact_count") == len(listed),
        "Incomplete trust inventory.",
    )
    return {"artifact_count": len(listed), "listed": listed}


def _valid_x(value: Any) -> bool:
    return bool(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        and tuple(value.shape) == (1, 4, 128, 128)
        and torch.isfinite(value).all()
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
    require(required.issubset(inventory["listed"]), "Required trust artifacts missing.")
    saved_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    require(saved_config == config and manifest.get("config_sha256") == canonical_sha(config), "Config drift.")
    require(
        manifest.get("protocol") == result.get("protocol") == terminal.get("protocol") == PROTOCOL
        and manifest.get("git_commit") == result.get("git_commit") == terminal.get("git_commit"),
        "Trust protocol/Git drift.",
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
        "Trust result boundary drift.",
    )
    start_snapshot = json.loads((root / "model_snapshot_verification_start.json").read_text(encoding="utf-8"))
    end_snapshot = json.loads((root / "model_snapshot_verification_end.json").read_text(encoding="utf-8"))
    require(
        start_snapshot == end_snapshot
        and result.get("snapshots_unchanged") is True
        and result.get("models_frozen") is True
        and result.get("only_student_x_T_trainable") is True,
        "Trust frozen/snapshot evidence drift.",
    )
    terminal_config = terminal_core.load_config()
    fixed = terminal_config["fixed_parent_artifacts"]
    target_path = root / "target/fixed_parent_target.pt"
    plateau_path = root / "parent/lr001-step256.pt"
    require(
        sha256_file(target_path) == fixed["source_target_artifact_sha256"]
        and sha256_file(plateau_path) == fixed["source_plateau_checkpoint_sha256"],
        "Copied trust parent bytes drift.",
    )
    target = torch.load(target_path, map_location="cpu", weights_only=True)
    plateau = torch.load(plateau_path, map_location="cpu", weights_only=True)
    bundle_path = root / "trust_region_tensors.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    require(
        bundle.get("schema") == f"{PREFIX}-tensor-bundle.v1"
        and bundle.get("protocol") == PROTOCOL
        and result.get("tensor_bundle_sha256") == sha256_file(bundle_path),
        "Trust tensor-bundle identity drift.",
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
        "Trust fixed FP32 anchor drift.",
    )
    baseline = float((plateau_endpoint - teacher_endpoint).square().mean())
    require(
        math.isclose(baseline, anchor["plateau_mse"], rel_tol=1e-6, abs_tol=1e-12)
        and result.get("fp32_lift_audit", {}).get("passed") is True,
        "Trust baseline/precision lift drift.",
    )
    iteration_rows = [
        json.loads(line) for line in (root / "iteration_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    candidate_rows = [
        json.loads(line) for line in (root / "candidate_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary = summarize_trace(iteration_rows, candidate_rows, config)
    tensor_iterations = bundle.get("iterations", {})
    require(
        set(tensor_iterations) == set(range(len(iteration_rows))),
        "Trust iteration tensor coverage drift.",
    )
    candidate_by_key = {validate_candidate_row(row): row for row in candidate_rows}
    iteration_by_index = {validate_iteration_row(row): row for row in iteration_rows}
    for index in range(len(iteration_rows)):
        row = iteration_by_index[index]
        record = tensor_iterations[index]
        current_x = record.get("current_x_T_fp32")
        current_endpoint = record.get("current_endpoint_fp32")
        gradient = record.get("gradient_fp32")
        direction = record.get("negative_gradient_direction_fp32")
        require(
            all(_valid_x(value) for value in (current_x, current_endpoint, gradient, direction))
            and canonical_tensor_sha256(current_x) == row["current_x_T_fp32_sha256"]
            and canonical_tensor_sha256(current_endpoint) == row["current_endpoint_fp32_sha256"]
            and canonical_tensor_sha256(gradient) == row["gradient_fp32_sha256"]
            and canonical_tensor_sha256(direction) == row["direction_fp32_sha256"]
            and cosine(direction, -gradient) >= 1.0 - 1e-6
            and abs(float(direction.double().norm()) - 1.0) <= 1e-6,
            "Trust current/gradient tensor drift.",
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
                float((gradient != 0).double().mean()),
                row["gradient_nonzero_fraction"],
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and row["gradient_nonzero_fraction"] >= config["algorithm"]["gradient_nonzero_fraction_gte"]
            and math.isclose(
                selected_residual_l2,
                row["selected_candidate_residual_l2"],
                rel_tol=1e-6,
                abs_tol=1e-12,
            ),
            "Trust current metric drift.",
        )
        candidates = record.get("candidates", {})
        require(set(candidates) == set(range(len(RADII))), "Trust candidate tensor coverage drift.")
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
                and math.isclose(
                    loss / baseline,
                    metric["loss_ratio_to_plateau"],
                    rel_tol=1e-6,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    loss / current_loss,
                    metric["loss_ratio_to_current"],
                    rel_tol=1e-6,
                    abs_tol=1e-12,
                ),
                "Trust candidate tensor/metric drift.",
            )
    first = tensor_iterations[0]
    require(
        canonical_tensor_sha256(first["gradient_fp32"]) == anchor["gradient_fp32_sha256"]
        and canonical_tensor_sha256(first["negative_gradient_direction_fp32"])
        == anchor["negative_gradient_direction_fp32_sha256"],
        "Trust first gradient does not reproduce precision parent.",
    )
    controls = {row["radius_l2"]: row for row in anchor["overlap_candidate_controls"]}
    for radius_index, radius in enumerate(RADII):
        if radius not in controls:
            continue
        expected = controls[radius]
        observed = candidate_by_key[(0, radius_index)]
        require(
            observed["candidate_x_T_fp32_sha256"] == expected["x_T_fp32_sha256"]
            and observed["endpoint_fp32_sha256"] == expected["endpoint_fp32_sha256"]
            and math.isclose(
                observed["loss_ratio_to_plateau"],
                expected["loss_ratio_to_plateau"],
                rel_tol=1e-6,
                abs_tol=1e-12,
            ),
            "Trust overlap candidate does not reproduce precision parent.",
        )
    final_x = bundle.get("final_x_T_fp32")
    final_endpoint = bundle.get("final_endpoint_fp32")
    final_replay = bundle.get("final_replay_endpoint_fp32")
    require(
        all(_valid_x(value) for value in (final_x, final_endpoint, final_replay))
        and torch.equal(final_endpoint, final_replay)
        and canonical_tensor_sha256(final_x) == summary["final_x_T_fp32_sha256"]
        and canonical_tensor_sha256(final_endpoint) == summary["final_endpoint_fp32_sha256"]
        and result.get("final_replay_bitwise_equal") is True,
        "Trust final replay drift.",
    )
    mode = result.get("mode")
    require(mode in ("technical-preflight", "formal") and terminal.get("mode") == mode, "Mode drift.")
    expected_counters = {
        "full_chain_forward_calls": 3 + 6 * len(iteration_rows),
        "reader_forward_calls": 0,
        "backward_calls": len(iteration_rows),
        "optimizer_steps": 0,
        "parameter_updates": summary["accepted_update_count"],
    }
    require(result.get("counters") == expected_counters, "Trust execution counter drift.")
    classification = classify_outcome(summary, config) if mode == "formal" else None
    require(
        result.get("trace_summary") == summary
        and result.get("classification") == classification
        and result.get("fixed_target_optimization_success") is (classification == "fixed_target_trust_region_success"),
        "Trust summary/classification drift.",
    )
    if mode == "technical-preflight":
        require(
            len(iteration_rows) == 1 and summary["accepted_update_count"] == 1 and result.get("preflight_gate") is True,
            "Trust preflight execution drift.",
        )
    else:
        require(
            len(iteration_rows) <= config["algorithm"]["maximum_formal_iterations"],
            "Trust formal iteration budget exceeded.",
        )
        prerequisite = manifest.get("validation", {}).get("preflight_prerequisite", {})
        require(
            prerequisite.get("passed") is True
            and prerequisite.get("mode") == "technical-preflight"
            and prerequisite.get("git_commit") == result["git_commit"],
            "Trust preflight chain drift.",
        )
    accepted_indices = [row["iteration"] for row in iteration_rows if row["accepted"]]
    checkpoint_files = {f"checkpoints/{iteration_id(index)}.pt" for index in accepted_indices}
    require(checkpoint_files.issubset(inventory["listed"]), "Trust accepted checkpoints missing.")
    for index in accepted_indices:
        checkpoint = torch.load(root / f"checkpoints/{iteration_id(index)}.pt", map_location="cpu", weights_only=True)
        row = iteration_by_index[index]
        require(
            checkpoint.get("schema") == f"{PREFIX}-checkpoint.v1"
            and checkpoint.get("iteration") == index
            and canonical_tensor_sha256(checkpoint["student_x_T_fp32"]) == row["selected_x_T_fp32_sha256"]
            and canonical_tensor_sha256(checkpoint["endpoint_fp32"]) == row["selected_endpoint_fp32_sha256"],
            "Trust checkpoint drift.",
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
