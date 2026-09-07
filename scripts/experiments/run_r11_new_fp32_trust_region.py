"""Run or independently audit the preregistered R11_new FP32 trust-region control."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import run_r11_new_activation_precision_control as precision_runner  # noqa: E402
from scripts.experiments import run_r11_new_reachable_lr_control as low_lr_runner  # noqa: E402
from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import r11_new_activation_precision_control as precision_core  # noqa: E402
from vision_memory.training import r11_new_direction_fidelity as direction_core  # noqa: E402
from vision_memory.training import r11_new_fp32_trust_region as core  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("technical-preflight", "formal", "audit"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--preflight-root", type=Path)
    return parser.parse_args(argv)


def validate_environment(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    precision_config = precision_core.load_config()
    contract = config["base_precision_contract"]
    core.require(
        precision_core.sha256_file(precision_core.CONFIG_PATH) == contract["config_byte_sha256"]
        and precision_core.canonical_sha(precision_config) == contract["config_canonical_sha256"],
        "Base precision-control contract drift.",
    )
    base_args = argparse.Namespace(**vars(args))
    base_args.mode = "technical-preflight"
    base_args.preflight_root = None
    base_validation = precision_runner.validate_environment(base_args, precision_config)

    parent = config["parent_precision_result"]
    parent_root = Path(parent["source_root"])
    bindings = {
        "result": (parent_root / "result.json", parent["source_result_sha256"]),
        "manifest": (parent_root / "manifest.json", parent["source_manifest_sha256"]),
        "terminal": (parent_root / "terminal.json", parent["source_terminal_sha256"]),
        "inventory": (
            parent_root / "artifact_inventory.json",
            parent["source_inventory_sha256"],
        ),
        "gradient_metrics": (
            parent_root / "gradient_metrics.jsonl",
            parent["source_gradient_metrics_sha256"],
        ),
        "archive": (Path(parent["source_archive_path"]), parent["source_archive_sha256"]),
    }
    for name, (path, expected_hash) in bindings.items():
        core.require(
            path.is_file() and core.sha256_file(path) == expected_hash,
            f"Trust parent {name} binding drift.",
        )
    parent_audit = precision_core.audit_delivery(parent_root, precision_config)
    core.require(
        parent_audit["mode"] == "formal"
        and parent_audit["classification"] == parent["classification"]
        and parent_audit["formal_success"] is False,
        "Wrong parent precision-control result.",
    )
    prerequisite = None
    if args.mode == "technical-preflight":
        core.require(args.preflight_root is None, "Preflight cannot consume another preflight.")
    else:
        core.require(
            args.preflight_root is not None and args.preflight_root.resolve() != args.output_root.resolve(),
            "Formal requires a distinct trust preflight root.",
        )
        prerequisite = core.audit_delivery(args.preflight_root, config)
        core.require(
            prerequisite["mode"] == "technical-preflight" and prerequisite["git_commit"] == args.expected_commit,
            "Wrong trust preflight prerequisite.",
        )
        prerequisite = {
            **prerequisite,
            "root": str(args.preflight_root.resolve()),
            "terminal_sha256": core.sha256_file(args.preflight_root / "terminal.json"),
            "inventory_sha256": core.sha256_file(args.preflight_root / "artifact_inventory.json"),
            "tensor_bundle_sha256": core.sha256_file(args.preflight_root / "trust_region_tensors.pt"),
        }
    return {
        **base_validation,
        "base_precision_config": {
            "path": str(precision_core.CONFIG_PATH),
            "byte_sha256": contract["config_byte_sha256"],
            "canonical_sha256": contract["config_canonical_sha256"],
        },
        "parent_precision_audit": parent_audit,
        "parent_bindings": {
            name: {"path": str(path), "sha256": expected_hash} for name, (path, expected_hash) in bindings.items()
        },
        "preflight_prerequisite": prerequisite,
    }


def load_runtime(args: argparse.Namespace, counters: dict[str, int]):
    return precision_runner.load_runtime(args, counters)


def _set_student(oracle: phase1a.FrozenDreamLiteOracle, value: torch.Tensor) -> None:
    precision_runner.set_student(oracle, value)


def _candidate_record(
    *,
    iteration: int,
    radius_index: int,
    candidate_x: torch.Tensor,
    endpoint: torch.Tensor,
    teacher_endpoint: torch.Tensor,
    baseline_mse: float,
    current_loss: float,
) -> dict[str, Any]:
    loss = float((endpoint - teacher_endpoint).square().mean())
    return {
        "schema": f"{core.PREFIX}-candidate-row.v1",
        "protocol": core.PROTOCOL,
        "candidate_id": core.candidate_id(iteration, radius_index),
        "iteration": iteration,
        "radius_index": radius_index,
        "radius_l2": core.RADII[radius_index],
        "candidate_x_T_fp32_sha256": canonical_tensor_sha256(candidate_x),
        "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
        "loss": loss,
        "loss_ratio_to_plateau": loss / baseline_mse,
        "loss_ratio_to_current": loss / current_loss,
    }


def run_search(
    *,
    args: argparse.Namespace,
    config: Mapping[str, Any],
    oracle: phase1a.FrozenDreamLiteOracle,
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
    iteration_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    tensor_iterations: dict[int, dict[str, Any]] = {}
    with (
        (args.output_root / "iteration_metrics.jsonl").open("x", encoding="utf-8", newline="\n") as iteration_metrics,
        (args.output_root / "candidate_metrics.jsonl").open("x", encoding="utf-8", newline="\n") as candidate_metrics,
    ):
        for iteration in range(maximum_iterations):
            _set_student(oracle, current_x)
            oracle.x_T_fp32.grad = None
            output = forward(gradient=True)
            current_endpoint = output.z_t.detach().float().cpu()
            if expected_current_endpoint is None:
                expected_current_endpoint = current_endpoint.clone()
            else:
                core.require(
                    torch.equal(current_endpoint, expected_current_endpoint),
                    "Trust current endpoint replay drift.",
                )
            target = teacher_endpoint.to(output.z_t.device, dtype=torch.float32)
            loss = (output.z_t.float() - target).square().mean()
            core.require(loss.numel() == 1 and bool(torch.isfinite(loss)), "Invalid trust loss.")
            loss.backward()
            counters["backward_calls"] += 1
            gradient = oracle.x_T_fp32.grad
            core.require(
                gradient is not None and gradient.dtype == torch.float32 and bool(torch.isfinite(gradient).all()),
                "Invalid trust gradient.",
            )
            gradient_cpu = gradient.detach().cpu().clone()
            nonzero_fraction = float((gradient_cpu != 0).double().mean())
            core.require(
                float(gradient_cpu.double().norm()) > 0.0
                and nonzero_fraction >= config["algorithm"]["gradient_nonzero_fraction_gte"],
                "Trust gradient density gate failed.",
            )
            direction = core.unit_vector(-gradient_cpu)
            oracle.x_T_fp32.grad = None
            current_loss = float(loss.detach())
            local_records: list[dict[str, Any]] = []
            local_tensors: dict[int, dict[str, torch.Tensor]] = {}
            best_image = None
            best_key = None
            for radius_index, radius in enumerate(core.RADII):
                candidate_x = current_x + radius * direction
                _set_student(oracle, candidate_x)
                candidate_output = forward(gradient=False)
                endpoint = candidate_output.z_t.detach().float().cpu()
                record = _candidate_record(
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
                local_tensors[radius_index] = {
                    "x_T_fp32": candidate_x.clone(),
                    "endpoint_fp32": endpoint,
                }
                candidate_metrics.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                candidate_metrics.flush()
                os.fsync(candidate_metrics.fileno())
                key = (record["loss"], record["radius_l2"])
                if best_key is None or key < best_key:
                    best_key = key
                    best_image = candidate_output.image.detach().float().cpu()
                print(
                    f"trust iteration={iteration} candidate={radius_index} "
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
                "direction_fp32_sha256": canonical_tensor_sha256(direction),
                "gradient_norm": float(gradient_cpu.double().norm()),
                "gradient_nonzero_fraction": nonzero_fraction,
                "analytic_directional_derivative": float((gradient_cpu.double() * direction.double()).sum()),
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
                    "negative_gradient_direction_fp32": direction,
                }
                phase1a._atomic_torch_save(
                    args.output_root / f"checkpoints/{core.iteration_id(iteration)}.pt",
                    checkpoint,
                )
                core.require(best_image is not None, "Missing accepted trust image.")
                phase1a._save_image(
                    args.output_root / f"accepted_images/{core.iteration_id(iteration)}.png",
                    best_image,
                )
            print(
                f"trust selected iteration={iteration} radius={best['radius_l2']:.6g} "
                f"ratio={best['loss_ratio_to_plateau']:.10g} accepted={accepted}",
                flush=True,
            )
            if args.mode == "technical-preflight":
                break
            if not accepted or best["loss_ratio_to_plateau"] <= success_ratio:
                break
    return (
        iteration_rows,
        candidate_rows,
        tensor_iterations,
        current_x,
        expected_current_endpoint,
    )


def write_report(root: Path, result: Mapping[str, Any]) -> None:
    summary = result["trace_summary"]
    lines = [
        "# R11_new FP32 residual-centered trust-region control",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Engineering gate: `{result['engineering_gate']}`",
        f"- Counters: `{json.dumps(result['counters'], sort_keys=True)}`",
        f"- Iterations: `{summary['iteration_count']}`",
        f"- Accepted updates: `{summary['accepted_update_count']}`",
        f"- Final loss ratio: `{summary['final_loss_ratio_to_plateau']}`",
    ]
    if result["mode"] == "formal":
        lines += [
            f"- Classification: `{result['classification']}`",
            f"- Fixed-target optimization success: `{result['fixed_target_optimization_success']}`",
        ]
    lines += [
        "",
        "This is single-target oracle endpoint optimization, not shared Picture Memory training.",
        "Formal Picture Memory success and Phase 2 remain false.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_experiment(
    args: argparse.Namespace, config: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    phase1a._atomic_json(args.output_root / "config.json", config)
    phase1a._write_environment(args.output_root / "environment.txt")
    phase1a._atomic_json(
        args.output_root / "runtime.json",
        {**phase1a._runtime_versions(), "python_executable": str(Path(sys.executable).resolve())},
    )
    phase1a._atomic_json(args.output_root / "determinism.json", configure_strict_cuda_determinism(0))
    direction_config = direction_core.load_config()
    snapshots_start = {
        name: phase1a.verify_snapshot_binding(value) for name, value in direction_config["model_snapshots"].items()
    }
    phase1a._atomic_json(args.output_root / "model_snapshot_verification_start.json", snapshots_start)
    counters = {
        "full_chain_forward_calls": 0,
        "reader_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
    }
    started = time.monotonic()
    (
        reader,
        oracle,
        source_latents,
        context,
        target,
        _alpha_checkpoint,
        plateau_checkpoint,
        target_record,
        initialization,
        forward,
    ) = load_runtime(args, counters)

    def forbidden_reader(_module: Any, _inputs: Any) -> None:
        counters["reader_forward_calls"] += 1
        raise RuntimeError("Reader is forbidden in the FP32 trust-region control.")

    reader_hook = reader.model.register_forward_pre_hook(forbidden_reader)
    try:
        core.require(
            low_lr_runner.models_frozen(oracle, reader) and low_lr_runner.only_student_trainable(oracle, reader),
            "Initial trust frozen/trainable contract failed.",
        )
        baseline_state, baseline_summary = precision_runner.capture_precision_state(oracle)
        teacher_x_t = target["teacher_x_T_fp32"].detach().float().cpu()
        plateau_x = plateau_checkpoint["student_x_T_fp32"].detach().float().cpu()
        bf16_teacher = target["teacher_endpoint_fp32"].detach().float().cpu()
        _set_student(oracle, teacher_x_t)
        fp32_lift_audit = precision_runner.lift_oracle_to_fp32(oracle, baseline_state, baseline_summary)
        fp32_teacher_output = forward(gradient=False)
        teacher_endpoint = fp32_teacher_output.z_t.detach().float().cpu()
        phase1a._save_image(args.output_root / "target/fp32_teacher.png", fp32_teacher_output.image)
        anchor = config["fixed_fp32_anchor"]
        core.require(
            canonical_tensor_sha256(teacher_endpoint) == anchor["teacher_endpoint_fp32_sha256"],
            "Trust FP32 teacher endpoint drift.",
        )
        manifest = {
            "schema": f"{core.PREFIX}-manifest.v1",
            "protocol": core.PROTOCOL,
            "mode": args.mode,
            "git_commit": args.expected_commit,
            "config_sha256": core.canonical_sha(config),
            "validation": dict(validation),
            "target": target_record,
            "initialization": initialization,
            "target_segment_id": context["target"].segment_id,
            "optimization_information_boundary": {
                "teacher_endpoint_supervision": True,
                "teacher_x_T_used_to_generate_target_only": True,
                "teacher_x_T_used_by_update_rule": False,
                "residual_center": "fixed parent plateau xT",
                "fixed_candidate_radii": list(core.RADII),
                "torch_optimizer_used": False,
                "gradient_clipping_used": False,
                "reader_used": False,
            },
            "formal_success": False,
            "phase2_allowed": False,
        }
        phase1a._atomic_json(args.output_root / "manifest.json", manifest)
        (
            iteration_rows,
            candidate_rows,
            tensor_iterations,
            final_x,
            final_endpoint,
        ) = run_search(
            args=args,
            config=config,
            oracle=oracle,
            plateau_x=plateau_x,
            teacher_endpoint=teacher_endpoint,
            expected_plateau_endpoint=None,
            counters=counters,
            forward=forward,
        )
        # The first gradient forward generated the FP32 plateau endpoint. It must be
        # reconstructed from the saved trace before any interpretation.
        first_plateau_endpoint = tensor_iterations[0]["current_endpoint_fp32"]
        core.require(
            canonical_tensor_sha256(first_plateau_endpoint) == anchor["plateau_endpoint_fp32_sha256"],
            "Trust FP32 plateau endpoint drift.",
        )
        _set_student(oracle, final_x)
        final_output = forward(gradient=False)
        final_replay = final_output.z_t.detach().float().cpu()
        core.require(torch.equal(final_replay, final_endpoint), "Trust final replay drift.")
        phase1a._save_image(args.output_root / "final/final.png", final_output.image)
        tensor_bundle = {
            "schema": f"{core.PREFIX}-tensor-bundle.v1",
            "protocol": core.PROTOCOL,
            "bf16_teacher_replay_endpoint_fp32": bf16_teacher,
            "plateau_x_T_fp32": plateau_x,
            "teacher_endpoint_fp32": teacher_endpoint,
            "plateau_endpoint_fp32": first_plateau_endpoint,
            "iterations": tensor_iterations,
            "final_x_T_fp32": final_x,
            "final_endpoint_fp32": final_endpoint,
            "final_replay_endpoint_fp32": final_replay,
        }
        phase1a._atomic_torch_save(args.output_root / "trust_region_tensors.pt", tensor_bundle)
        snapshots_end = {
            name: phase1a.verify_snapshot_binding(value) for name, value in direction_config["model_snapshots"].items()
        }
        phase1a._atomic_json(args.output_root / "model_snapshot_verification_end.json", snapshots_end)
        trace_summary = core.summarize_trace(iteration_rows, candidate_rows, config)
        classification = core.classify_outcome(trace_summary, config) if args.mode == "formal" else None
        fixed_target_success = classification == "fixed_target_trust_region_success"
        frozen_gate = low_lr_runner.models_frozen(oracle, reader) and low_lr_runner.only_student_trainable(
            oracle, reader
        )
        expected_counters = {
            "full_chain_forward_calls": 3 + 6 * len(iteration_rows),
            "reader_forward_calls": 0,
            "backward_calls": len(iteration_rows),
            "optimizer_steps": 0,
            "parameter_updates": trace_summary["accepted_update_count"],
        }
        first = iteration_rows[0]
        controls = {row["radius_l2"]: row for row in anchor["overlap_candidate_controls"]}
        first_candidates = {row["radius_l2"]: row for row in candidate_rows if row["iteration"] == 0}
        overlap_gate = all(
            first_candidates[radius]["candidate_x_T_fp32_sha256"] == expected["x_T_fp32_sha256"]
            and first_candidates[radius]["endpoint_fp32_sha256"] == expected["endpoint_fp32_sha256"]
            and math.isclose(
                first_candidates[radius]["loss_ratio_to_plateau"],
                expected["loss_ratio_to_plateau"],
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
            for radius, expected in controls.items()
        )
        common_gate = bool(
            counters == expected_counters
            and canonical_tensor_sha256(tensor_iterations[0]["gradient_fp32"]) == anchor["gradient_fp32_sha256"]
            and canonical_tensor_sha256(tensor_iterations[0]["negative_gradient_direction_fp32"])
            == anchor["negative_gradient_direction_fp32_sha256"]
            and overlap_gate
            and fp32_lift_audit["passed"]
            and torch.equal(final_endpoint, final_replay)
            and snapshots_end == snapshots_start
            and frozen_gate
        )
        if args.mode == "technical-preflight":
            engineering_gate = bool(
                common_gate
                and counters["full_chain_forward_calls"] == 9
                and counters["backward_calls"] == 1
                and counters["parameter_updates"] == 1
                and first["accepted"]
            )
            core.require(engineering_gate, "Trust technical preflight gate failed.")
        else:
            engineering_gate = bool(
                common_gate
                and len(iteration_rows) <= config["algorithm"]["maximum_formal_iterations"]
                and counters["full_chain_forward_calls"] <= config["formal_gate"]["maximum_full_chain_forward_calls"]
                and trace_summary["all_accepted_losses_strictly_monotone"]
            )
            core.require(engineering_gate, "Trust formal technical gate failed.")
        result = {
            "schema": f"{core.PREFIX}-result.v1",
            "protocol": core.PROTOCOL,
            "mode": args.mode,
            "git_commit": args.expected_commit,
            "engineering_gate": engineering_gate,
            "preflight_gate": engineering_gate if args.mode == "technical-preflight" else None,
            "fp32_lift_audit": fp32_lift_audit,
            "trace_summary": trace_summary,
            "classification": classification,
            "fixed_target_optimization_success": fixed_target_success,
            "final_replay_bitwise_equal": True,
            "counters": counters,
            "target_record": target_record,
            "tensor_bundle_sha256": core.sha256_file(args.output_root / "trust_region_tensors.pt"),
            "models_frozen": True,
            "only_student_x_T_trainable": True,
            "snapshots_unchanged": snapshots_end == snapshots_start,
            "formal_success": False,
            "phase2_allowed": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        phase1a._atomic_json(args.output_root / "result.json", result)
        write_report(args.output_root, result)
        return result
    finally:
        reader_hook.remove()
        del source_latents


def artifact_inventory(root: Path) -> dict[str, Any]:
    artifacts = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": core.sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    ]
    return {
        "schema": f"{core.PREFIX}-inventory.v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = core.load_config()
    if args.mode == "audit":
        print(
            json.dumps(
                core.audit_delivery(args.output_root, config),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    from scripts.inspire import run_r11_new_identity_condition_bridge as safety

    validation = validate_environment(args, config)
    lock_path = Path("/tmp/vision-memory-r11-new-fp32-trust-region.lock")
    lock_path.mkdir(exist_ok=False)
    owner = {
        "owner_token": uuid.uuid4().hex,
        "pid": os.getpid(),
        "mode": args.mode,
        "git_commit": args.expected_commit,
        "output_root": str(args.output_root),
        "started_at_utc": safety._utc_now(),
    }
    phase1a._atomic_json(lock_path / "owner.json", owner)
    lock = {
        "path": str(lock_path),
        "owner_path": str(lock_path / "owner.json"),
        "owner": owner,
        "owner_sha256": core.sha256_file(lock_path / "owner.json"),
    }
    claimed = False
    terminal = {
        "schema": f"{core.PREFIX}-terminal.v1",
        "protocol": core.PROTOCOL,
        "mode": args.mode,
        "status": "technical_failed",
        "engineering_gate": False,
        "exit_code": 2,
        "git_commit": args.expected_commit,
        "formal_success": False,
        "phase2_allowed": False,
        "started_at_utc": safety._utc_now(),
    }
    try:
        args.output_root.mkdir(parents=True, exist_ok=False)
        claimed = True
        phase1a._atomic_json(args.output_root / "launch.json", {"owner": owner, "validation": validation})
        with (
            (args.output_root / "stdout.log").open("x", encoding="utf-8") as stdout,
            (args.output_root / "stderr.log").open("x", encoding="utf-8") as stderr,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                result = run_experiment(args, config, validation)
            except Exception:
                traceback.print_exc()
                raise
        terminal.update(
            status="technical_completed",
            engineering_gate=True,
            exit_code=0,
            result_sha256=core.sha256_file(args.output_root / "result.json"),
            classification=result.get("classification"),
            fixed_target_optimization_success=result.get("fixed_target_optimization_success", False),
        )
    except Exception as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            terminal["lock_release"] = safety._release_lock(lock)
        except Exception as error:
            terminal.update(
                status="technical_failed",
                engineering_gate=False,
                exit_code=2,
                lock_release_error=f"{type(error).__name__}: {error}",
            )
        if claimed:
            terminal["completed_at_utc"] = safety._utc_now()
            phase1a._atomic_json(args.output_root / "terminal.json", terminal)
            phase1a._atomic_json(args.output_root / "artifact_inventory.json", artifact_inventory(args.output_root))
            if terminal["exit_code"] == 0:
                core.audit_delivery(args.output_root, config)
    print(json.dumps(terminal, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return int(terminal["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
