"""Run the preregistered matched-budget FP32 PRP+ trust-region control."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import run_r11_new_fp32_trust_region as parent_runner  # noqa: E402
from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402
from vision_memory.training import r11_new_fp32_prpplus_trust_region as core  # noqa: E402
from vision_memory.training import r11_new_fp32_trust_region_horizon128 as horizon_core  # noqa: E402


_BASE_VALIDATE_ENVIRONMENT = parent_runner.validate_environment
_BASE_RUN_EXPERIMENT = parent_runner.run_experiment


def validate_environment(args: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    validation = _BASE_VALIDATE_ENVIRONMENT(args, config)
    contract = config["parent_horizon_result"]
    parent_root = Path(contract["source_root"])
    external_audit = parent_root.parent / "r11-new-trust-h128-8efeabc-20260907-parent-prefix-audit.json"
    bindings = {
        "result": (parent_root / "result.json", contract["source_result_sha256"]),
        "manifest": (parent_root / "manifest.json", contract["source_manifest_sha256"]),
        "terminal": (parent_root / "terminal.json", contract["source_terminal_sha256"]),
        "inventory": (parent_root / "artifact_inventory.json", contract["source_inventory_sha256"]),
        "iterations": (parent_root / "iteration_metrics.jsonl", contract["source_iteration_metrics_sha256"]),
        "candidates": (parent_root / "candidate_metrics.jsonl", contract["source_candidate_metrics_sha256"]),
        "tensor_bundle": (parent_root / "trust_region_tensors.pt", contract["source_tensor_bundle_sha256"]),
        "final_checkpoint": (
            parent_root / "checkpoints/iteration-127.pt",
            contract["source_final_checkpoint_sha256"],
        ),
        "archive": (Path(contract["source_archive_path"]), contract["source_archive_sha256"]),
        "external_prefix_audit": (external_audit, contract["source_external_prefix_audit_sha256"]),
    }
    for name, (path, expected) in bindings.items():
        core.require(path.is_file() and core.sha256_file(path) == expected, f"PRP+ parent {name} binding drift.")
    parent_config = horizon_core.load_config()
    parent_audit = horizon_core.audit_delivery(parent_root, parent_config)
    core.require(
        parent_audit["mode"] == "formal"
        and parent_audit["classification"] == contract["classification"]
        and parent_audit["trace_summary"]["iteration_count"] == contract["iteration_count"]
        and parent_audit["trace_summary"]["final_x_T_fp32_sha256"] == contract["final_x_T_fp32_sha256"]
        and parent_audit["formal_success"] is False
        and parent_audit["phase2_allowed"] is False,
        "Wrong PRP+ parent horizon result.",
    )
    return {
        **validation,
        "parent_horizon_audit": parent_audit,
        "parent_horizon_bindings": {
            name: {"path": str(path), "sha256": expected} for name, (path, expected) in bindings.items()
        },
    }


def run_search(
    *,
    args: Any,
    config: Mapping[str, Any],
    oracle: Any,
    plateau_x: torch.Tensor,
    teacher_endpoint: torch.Tensor,
    expected_plateau_endpoint: torch.Tensor | None,
    counters: dict[str, int],
    forward: Any,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, dict[str, Any]],
    torch.Tensor,
    torch.Tensor,
]:
    maximum_iterations = (
        config["algorithm"]["technical_preflight_iterations"]
        if args.mode == "technical-preflight"
        else config["algorithm"]["maximum_formal_iterations"]
    )
    baseline_mse = config["fixed_fp32_anchor"]["plateau_mse"]
    minimum_improvement = config["algorithm"]["minimum_relative_improvement_to_accept"]
    success_ratio = config["algorithm"]["stop_after_loss_ratio_lte"]
    current_x = plateau_x.clone()
    expected_current_endpoint = expected_plateau_endpoint.clone() if expected_plateau_endpoint is not None else None
    previous_gradient: torch.Tensor | None = None
    previous_raw_direction: torch.Tensor | None = None
    iteration_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    tensor_iterations: dict[int, dict[str, Any]] = {}
    with (
        (args.output_root / "iteration_metrics.jsonl").open("x", encoding="utf-8", newline="\n") as iteration_metrics,
        (args.output_root / "candidate_metrics.jsonl").open("x", encoding="utf-8", newline="\n") as candidate_metrics,
    ):
        for iteration in range(maximum_iterations):
            parent_runner._set_student(oracle, current_x)
            oracle.x_T_fp32.grad = None
            output = forward(gradient=True)
            current_endpoint = output.z_t.detach().float().cpu()
            if expected_current_endpoint is None:
                expected_current_endpoint = current_endpoint.clone()
            else:
                core.require(
                    torch.equal(current_endpoint, expected_current_endpoint), "PRP+ current endpoint replay drift."
                )
            target = teacher_endpoint.to(output.z_t.device, dtype=torch.float32)
            loss = (output.z_t.float() - target).square().mean()
            core.require(loss.numel() == 1 and bool(torch.isfinite(loss)), "Invalid PRP+ loss.")
            loss.backward()
            counters["backward_calls"] += 1
            gradient = oracle.x_T_fp32.grad
            core.require(
                gradient is not None and gradient.dtype == torch.float32 and bool(torch.isfinite(gradient).all()),
                "Invalid PRP+ gradient.",
            )
            gradient_cpu = gradient.detach().cpu().clone()
            nonzero_fraction = float((gradient_cpu != 0).double().mean())
            core.require(
                float(gradient_cpu.double().norm()) > 0.0
                and nonzero_fraction >= config["algorithm"]["gradient_nonzero_fraction_gte"],
                "PRP+ gradient density gate failed.",
            )
            direction_state = core.prp_plus_direction(gradient_cpu, previous_gradient, previous_raw_direction)
            raw_direction = direction_state["raw_direction"]
            direction = direction_state["direction"]
            oracle.x_T_fp32.grad = None
            current_loss = float(loss.detach())
            local_records: list[dict[str, Any]] = []
            local_tensors: dict[int, dict[str, torch.Tensor]] = {}
            best_image = None
            best_key = None
            for radius_index, radius in enumerate(core.RADII):
                candidate_x = current_x + radius * direction
                parent_runner._set_student(oracle, candidate_x)
                candidate_output = forward(gradient=False)
                endpoint = candidate_output.z_t.detach().float().cpu()
                record = parent_runner._candidate_record(
                    iteration=iteration,
                    radius_index=radius_index,
                    candidate_x=candidate_x,
                    endpoint=endpoint,
                    teacher_endpoint=teacher_endpoint,
                    baseline_mse=baseline_mse,
                    current_loss=current_loss,
                )
                local_records.append(record)
                candidate_rows.append(record)
                local_tensors[radius_index] = {"x_T_fp32": candidate_x.clone(), "endpoint_fp32": endpoint}
                candidate_metrics.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                candidate_metrics.flush()
                os.fsync(candidate_metrics.fileno())
                key = (record["loss"], record["radius_l2"])
                if best_key is None or key < best_key:
                    best_key = key
                    best_image = candidate_output.image.detach().float().cpu()
                print(
                    f"prpplus iteration={iteration} candidate={radius_index} "
                    f"radius={radius:.6g} ratio={record['loss_ratio_to_plateau']:.10g}",
                    flush=True,
                )
            best = min(local_records, key=lambda row: (row["loss"], row["radius_l2"]))
            best_tensors = local_tensors[best["radius_index"]]
            relative_improvement = (current_loss - best["loss"]) / current_loss
            accepted = relative_improvement >= minimum_improvement
            selected_residual_l2 = float((best_tensors["x_T_fp32"] - plateau_x).double().norm())
            iteration_row = {
                "schema": f"{core.PREFIX}-iteration-row.v1",
                "protocol": core.PROTOCOL,
                "iteration_id": core.iteration_id(iteration),
                "iteration": iteration,
                "current_loss": current_loss,
                "current_loss_ratio_to_plateau": current_loss / baseline_mse,
                "current_x_T_fp32_sha256": canonical_tensor_sha256(current_x),
                "current_endpoint_fp32_sha256": canonical_tensor_sha256(current_endpoint),
                "gradient_fp32_sha256": canonical_tensor_sha256(gradient_cpu),
                "previous_gradient_fp32_sha256": (
                    canonical_tensor_sha256(previous_gradient) if previous_gradient is not None else None
                ),
                "previous_raw_direction_fp32_sha256": (
                    canonical_tensor_sha256(previous_raw_direction) if previous_raw_direction is not None else None
                ),
                "raw_direction_fp32_sha256": canonical_tensor_sha256(raw_direction),
                "direction_fp32_sha256": canonical_tensor_sha256(direction),
                "gradient_norm": float(gradient_cpu.double().norm()),
                "gradient_nonzero_fraction": nonzero_fraction,
                "beta_raw": direction_state["beta_raw"],
                "beta": direction_state["beta"],
                "restart_reason": direction_state["restart_reason"],
                "negative_gradient_cosine": direction_state["negative_gradient_cosine"],
                "analytic_directional_derivative": direction_state["directional_derivative"],
                "selected_candidate_id": best["candidate_id"],
                "selected_radius_index": best["radius_index"],
                "selected_radius_l2": best["radius_l2"],
                "selected_loss": best["loss"],
                "selected_loss_ratio_to_plateau": best["loss_ratio_to_plateau"],
                "selected_x_T_fp32_sha256": best["candidate_x_T_fp32_sha256"],
                "selected_endpoint_fp32_sha256": best["endpoint_fp32_sha256"],
                "selected_candidate_residual_l2": selected_residual_l2,
                "relative_improvement": relative_improvement,
                "accepted": accepted,
            }
            iteration_rows.append(iteration_row)
            iteration_metrics.write(json.dumps(iteration_row, sort_keys=True, allow_nan=False) + "\n")
            iteration_metrics.flush()
            os.fsync(iteration_metrics.fileno())
            tensor_iterations[iteration] = {
                "current_x_T_fp32": current_x.clone(),
                "current_endpoint_fp32": current_endpoint,
                "gradient_fp32": gradient_cpu,
                "previous_gradient_fp32": previous_gradient.clone() if previous_gradient is not None else None,
                "previous_raw_direction_fp32": (
                    previous_raw_direction.clone() if previous_raw_direction is not None else None
                ),
                "conjugate_raw_direction_fp32": raw_direction,
                "negative_gradient_direction_fp32": direction,
                "candidates": local_tensors,
            }
            if accepted:
                counters["parameter_updates"] += 1
                current_x = best_tensors["x_T_fp32"].clone()
                expected_current_endpoint = best_tensors["endpoint_fp32"].clone()
                checkpoint = {
                    "schema": f"{core.PREFIX}-checkpoint.v1",
                    "protocol": core.PROTOCOL,
                    "iteration": iteration,
                    "student_x_T_fp32": current_x,
                    "residual_delta_fp32": current_x - plateau_x,
                    "endpoint_fp32": expected_current_endpoint,
                    "gradient_fp32": gradient_cpu,
                    "conjugate_raw_direction_fp32": raw_direction,
                    "negative_gradient_direction_fp32": direction,
                    "beta_raw": direction_state["beta_raw"],
                    "beta": direction_state["beta"],
                    "restart_reason": direction_state["restart_reason"],
                }
                phase1a._atomic_torch_save(
                    args.output_root / f"checkpoints/{core.iteration_id(iteration)}.pt",
                    checkpoint,
                )
                core.require(best_image is not None, "Missing accepted PRP+ image.")
                phase1a._save_image(
                    args.output_root / f"accepted_images/{core.iteration_id(iteration)}.png",
                    best_image,
                )
            previous_gradient = gradient_cpu.clone()
            previous_raw_direction = raw_direction.clone()
            print(
                f"prpplus selected iteration={iteration} beta={direction_state['beta']:.10g} "
                f"restart={direction_state['restart_reason']} radius={best['radius_l2']:.6g} "
                f"ratio={best['loss_ratio_to_plateau']:.10g} accepted={accepted}",
                flush=True,
            )
            if args.mode == "technical-preflight":
                break
            if not accepted or best["loss_ratio_to_plateau"] <= success_ratio:
                break
    return iteration_rows, candidate_rows, tensor_iterations, current_x, expected_current_endpoint


def write_report(root: Path, result: Mapping[str, Any]) -> None:
    summary = result["trace_summary"]
    lines = [
        "# R11_new FP32 PRP+ nonlinear conjugate trust-region control",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Engineering gate: `{result['engineering_gate']}`",
        f"- Counters: `{json.dumps(result['counters'], sort_keys=True)}`",
        f"- Iterations: `{summary['iteration_count']}`",
        f"- Accepted updates: `{summary['accepted_update_count']}`",
        f"- Final loss ratio: `{summary['final_loss_ratio_to_plateau']}`",
        f"- Restart counts: `{json.dumps(summary['restart_counts'], sort_keys=True)}`",
    ]
    if result["mode"] == "formal":
        lines += [
            f"- Classification: `{result['classification']}`",
            f"- Fixed-target optimization success: `{result['fixed_target_optimization_success']}`",
        ]
    lines += [
        "",
        "This is a matched-budget single-target oracle optimization control, not shared Picture Memory training.",
        "Formal Picture Memory success and Phase 2 remain false.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_experiment(args: Any, config: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    result = _BASE_RUN_EXPERIMENT(args, config, validation)
    manifest_path = args.output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["optimization_information_boundary"].update(
        {
            "direction_method": config["algorithm"]["direction_method"],
            "direction_uses_only_current_gradient_previous_gradient_and_previous_raw_direction": True,
            "matched_parent_horizon_updates": config["algorithm"]["maximum_formal_iterations"],
        }
    )
    phase1a._atomic_json(manifest_path, manifest)
    return result


def main(argv: list[str] | None = None) -> int:
    original = {
        "core": parent_runner.core,
        "validate_environment": parent_runner.validate_environment,
        "run_search": parent_runner.run_search,
        "write_report": parent_runner.write_report,
        "run_experiment": parent_runner.run_experiment,
    }
    parent_runner.core = core
    parent_runner.validate_environment = validate_environment
    parent_runner.run_search = run_search
    parent_runner.write_report = write_report
    parent_runner.run_experiment = run_experiment
    try:
        return parent_runner.main(argv)
    finally:
        parent_runner.core = original["core"]
        parent_runner.validate_environment = original["validate_environment"]
        parent_runner.run_search = original["run_search"]
        parent_runner.write_report = original["write_report"]
        parent_runner.run_experiment = original["run_experiment"]


if __name__ == "__main__":
    raise SystemExit(main())
