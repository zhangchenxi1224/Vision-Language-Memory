"""Run or independently audit the preregistered activation-precision control."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import run_r11_new_direction_fidelity as direction_runner  # noqa: E402
from scripts.experiments import run_r11_new_reachable_lr_control as low_lr_runner  # noqa: E402
from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import r11_new_activation_precision_control as core  # noqa: E402
from vision_memory.training import r11_new_direction_fidelity as direction_core  # noqa: E402
from vision_memory.training import r11_new_oracle_terminal_capture as terminal_core  # noqa: E402
from vision_memory.training import r11_new_reachable_lr_control as low_lr_core  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("technical-preflight", "formal", "audit"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--preflight-root", type=Path)
    return parser.parse_args(argv)


def validate_environment(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    terminal_config = terminal_core.load_config()
    contract = config["base_terminal_contract"]
    core.require(
        terminal_core.sha256_file(terminal_core.CONFIG_PATH) == contract["config_byte_sha256"]
        and terminal_core.canonical_sha(terminal_config) == contract["config_canonical_sha256"],
        "Base terminal-capture contract drift.",
    )
    direction_config = direction_core.load_config()
    low_lr_config = low_lr_core.load_config()
    for key in terminal_config["base_runtime_contract"]["reuse_exact_sections"]:
        core.require(
            direction_config[key] == low_lr_config[key],
            f"Reused precision runtime section drift: {key}.",
        )
    base_args = argparse.Namespace(**vars(args))
    base_args.mode = "technical-preflight"
    base_args.preflight_root = None
    base_validation = low_lr_runner.validate_environment(base_args, low_lr_config)

    parent = config["parent_terminal_failure"]
    parent_root = Path(parent["source_root"])
    fixed = terminal_config["fixed_parent_artifacts"]
    bindings = {
        "result": (parent_root / "result.json", parent["source_result_sha256"]),
        "manifest": (parent_root / "manifest.json", parent["source_manifest_sha256"]),
        "terminal": (parent_root / "terminal.json", parent["source_terminal_sha256"]),
        "inventory": (
            parent_root / "artifact_inventory.json",
            parent["source_inventory_sha256"],
        ),
        "archive": (Path(parent["source_archive_path"]), parent["source_archive_sha256"]),
        "target": (
            Path(fixed["source_target_path"]),
            fixed["source_target_artifact_sha256"],
        ),
        "plateau": (
            Path(fixed["source_plateau_checkpoint_path"]),
            fixed["source_plateau_checkpoint_sha256"],
        ),
    }
    for name, (path, expected_hash) in bindings.items():
        core.require(
            path.is_file() and core.sha256_file(path) == expected_hash,
            f"Precision parent {name} binding drift.",
        )
    parent_audit = terminal_core.audit_delivery(parent_root, terminal_config)
    core.require(
        parent_audit["mode"] == "formal"
        and parent_audit["classification"] == parent["classification"]
        and parent_audit["formal_success"] is False,
        "Wrong parent terminal-capture result.",
    )

    prerequisite = None
    if args.mode == "technical-preflight":
        core.require(args.preflight_root is None, "Preflight cannot consume another preflight.")
    else:
        core.require(
            args.preflight_root is not None and args.preflight_root.resolve() != args.output_root.resolve(),
            "Formal requires a distinct precision preflight root.",
        )
        prerequisite = core.audit_delivery(args.preflight_root, config)
        core.require(
            prerequisite["mode"] == "technical-preflight" and prerequisite["git_commit"] == args.expected_commit,
            "Wrong precision preflight prerequisite.",
        )
        prerequisite = {
            **prerequisite,
            "root": str(args.preflight_root.resolve()),
            "terminal_sha256": core.sha256_file(args.preflight_root / "terminal.json"),
            "inventory_sha256": core.sha256_file(args.preflight_root / "artifact_inventory.json"),
            "tensor_bundle_sha256": core.sha256_file(args.preflight_root / "precision_tensors.pt"),
        }
    return {
        **base_validation,
        "base_terminal_config": {
            "path": str(terminal_core.CONFIG_PATH),
            "byte_sha256": contract["config_byte_sha256"],
            "canonical_sha256": contract["config_canonical_sha256"],
        },
        "parent_terminal_audit": parent_audit,
        "parent_bindings": {
            name: {"path": str(path), "sha256": expected_hash} for name, (path, expected_hash) in bindings.items()
        },
        "preflight_prerequisite": prerequisite,
    }


def load_runtime(args: argparse.Namespace, counters: dict[str, int]):
    return direction_runner.load_runtime(args, direction_core.load_config(), counters)


def set_student(oracle: phase1a.FrozenDreamLiteOracle, value: torch.Tensor) -> None:
    with torch.no_grad():
        oracle.x_T_fp32.copy_(value.to(oracle.x_T_fp32.device, dtype=torch.float32))
        oracle.initial_x_T_fp32.copy_(value.to(oracle.initial_x_T_fp32.device, dtype=torch.float32))


def _named_precision_tensors(
    oracle: phase1a.FrozenDreamLiteOracle,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for scope, module in (("unet", oracle.unet), ("vae", oracle.vae)):
        for kind, values in (
            ("parameter", module.named_parameters()),
            ("buffer", module.named_buffers()),
        ):
            for name, value in values:
                key = f"{scope}.{kind}.{name}"
                core.require(key not in tensors, "Duplicate precision tensor name.")
                tensors[key] = value
    return tensors


def _sample(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().reshape(-1)
    if flat.numel() == 0:
        return flat.cpu().clone()
    indices = sorted({0, flat.numel() // 4, flat.numel() // 2, 3 * flat.numel() // 4, flat.numel() - 1})
    return flat[torch.tensor(indices, device=flat.device)].cpu().clone()


def capture_precision_state(
    oracle: phase1a.FrozenDreamLiteOracle,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    state: dict[str, dict[str, Any]] = {}
    dtype_counts: Counter[str] = Counter()
    dtype_numel: Counter[str] = Counter()
    floating_tensor_count = 0
    floating_numel = 0
    for name, value in _named_precision_tensors(oracle).items():
        dtype = str(value.dtype)
        dtype_counts[dtype] += 1
        dtype_numel[dtype] += value.numel()
        floating = bool(value.is_floating_point())
        floating_tensor_count += int(floating)
        floating_numel += value.numel() if floating else 0
        state[name] = {
            "dtype": dtype,
            "numel": value.numel(),
            "floating": floating,
            "sample": _sample(value),
        }
    summary = {
        "tensor_count": len(state),
        "floating_tensor_count": floating_tensor_count,
        "floating_numel": floating_numel,
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "dtype_numel": dict(sorted(dtype_numel.items())),
        "compute_dtype": str(oracle.compute_dtype),
        "source_latents_dtype": str(oracle.source_latents.dtype),
        "prompt_embeds_dtype": str(oracle.prompt_embeds.dtype),
        "attention_mask_dtype": str(oracle.prompt_attention_mask.dtype),
    }
    return state, summary


def compare_precision_states(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    lift_floating: bool,
) -> dict[str, Any]:
    core.require(set(before) == set(after), "Precision tensor topology changed.")
    sampled_values = 0
    baseline_dtype_counts: Counter[str] = Counter()
    for name in sorted(before):
        left = before[name]
        right = after[name]
        core.require(left["numel"] == right["numel"], "Precision tensor size changed.")
        baseline_dtype_counts[left["dtype"]] += 1
        expected_dtype = "torch.float32" if lift_floating and left["floating"] else left["dtype"]
        core.require(right["dtype"] == expected_dtype, f"Unexpected lifted dtype: {name}.")
        expected_sample = left["sample"].float() if left["floating"] and lift_floating else left["sample"]
        core.require(
            torch.equal(right["sample"], expected_sample),
            f"Sampled value changed during precision lift: {name}.",
        )
        sampled_values += int(expected_sample.numel())
    return {
        "tensor_topology_unchanged": True,
        "all_expected_dtypes": True,
        "all_sampled_values_bitwise_exact": True,
        "sampled_value_count": sampled_values,
        "baseline_dtype_tensor_counts": dict(sorted(baseline_dtype_counts.items())),
    }


def lift_oracle_to_fp32(
    oracle: phase1a.FrozenDreamLiteOracle,
    baseline_state: Mapping[str, Mapping[str, Any]],
    baseline_summary: Mapping[str, Any],
) -> dict[str, Any]:
    source_before = oracle.source_latents.detach().cpu()
    prompt_before = oracle.prompt_embeds.detach().cpu()
    attention_before = oracle.prompt_attention_mask.detach().cpu()
    student_before = oracle.x_T_fp32.detach().cpu()
    text_before = {
        name: {
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "floating": bool(value.is_floating_point()),
            "sample": _sample(value),
        }
        for name, value in oracle.text_encoder.state_dict().items()
    }
    oracle.unet.to(dtype=torch.float32)
    oracle.vae.to(dtype=torch.float32)
    oracle.source_latents = oracle.source_latents.float()
    oracle.prompt_embeds = oracle.prompt_embeds.float()
    oracle.compute_dtype = torch.float32
    after_state, after_summary = capture_precision_state(oracle)
    value_audit = compare_precision_states(baseline_state, after_state, lift_floating=True)
    text_after = {
        name: {
            "dtype": str(value.dtype),
            "numel": value.numel(),
            "floating": bool(value.is_floating_point()),
            "sample": _sample(value),
        }
        for name, value in oracle.text_encoder.state_dict().items()
    }
    text_audit = compare_precision_states(text_before, text_after, lift_floating=False)
    source_after = oracle.source_latents.detach().cpu()
    prompt_after = oracle.prompt_embeds.detach().cpu()
    core.require(
        oracle.compute_dtype == torch.float32
        and source_after.dtype == torch.float32
        and prompt_after.dtype == torch.float32
        and torch.equal(source_after, source_before.float())
        and torch.equal(prompt_after, prompt_before.float())
        and torch.equal(oracle.prompt_attention_mask.detach().cpu(), attention_before)
        and torch.equal(oracle.x_T_fp32.detach().cpu(), student_before)
        and after_summary["floating_tensor_count"] == after_summary["dtype_tensor_counts"].get("torch.float32", 0),
        "FP32 activation precision lift failed.",
    )
    return {
        "passed": True,
        "contract": "same in-memory BF16-valued weights lifted exactly into FP32 arithmetic",
        "baseline_layout": dict(baseline_summary),
        "lifted_layout": after_summary,
        "model_value_audit": value_audit,
        "text_encoder_unchanged_audit": text_audit,
        "source_latents_bf16_sha256_before": canonical_tensor_sha256(source_before),
        "source_latents_bf16_roundtrip_sha256_after": canonical_tensor_sha256(source_after.bfloat16()),
        "prompt_embeds_bf16_sha256_before": canonical_tensor_sha256(prompt_before),
        "prompt_embeds_bf16_roundtrip_sha256_after": canonical_tensor_sha256(prompt_after.bfloat16()),
        "attention_mask_sha256": canonical_tensor_sha256(attention_before),
        "student_x_T_fp32_sha256": canonical_tensor_sha256(student_before),
        "text_encoder_executed": False,
        "model_reloaded": False,
        "conditioning_regenerated": False,
    }


def prepare_condition_gradient(
    *,
    condition: str,
    oracle: phase1a.FrozenDreamLiteOracle,
    teacher_endpoint: torch.Tensor,
    plateau_x_t: torch.Tensor,
    counters: dict[str, int],
    forward: Any,
    output_root: Path,
    precision_audit: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    set_student(oracle, plateau_x_t)
    oracle.x_T_fp32.grad = None
    output = forward(gradient=True)
    plateau_endpoint = output.z_t.detach().float().cpu()
    target = teacher_endpoint.to(output.z_t.device, dtype=torch.float32)
    loss = (output.z_t.float() - target).square().mean()
    core.require(loss.numel() == 1 and bool(torch.isfinite(loss)), "Invalid precision loss.")
    loss.backward()
    counters["backward_calls"] += 1
    gradient = oracle.x_T_fp32.grad
    core.require(
        gradient is not None and gradient.dtype == torch.float32 and bool(torch.isfinite(gradient).all()),
        "Invalid precision gradient.",
    )
    gradient_cpu = gradient.detach().cpu().clone()
    nonzero_fraction = float((gradient_cpu != 0).double().mean())
    core.require(
        float(gradient_cpu.double().norm()) > 0.0
        and nonzero_fraction >= config["gradient_control"]["gradient_nonzero_fraction_gte"],
        "Precision gradient density gate failed.",
    )
    direction = core.unit_vector(-gradient_cpu)
    oracle.x_T_fp32.grad = None
    phase1a._save_image(output_root / f"conditions/{condition}/plateau.png", output.image)
    plateau_mse = float((plateau_endpoint - teacher_endpoint).square().mean())
    core.require(plateau_mse > 0.0 and math.isfinite(plateau_mse), "Invalid plateau MSE.")
    record = {
        "compute_dtype": str(oracle.compute_dtype),
        "plateau_mse": plateau_mse,
        "teacher_endpoint_fp32_sha256": canonical_tensor_sha256(teacher_endpoint),
        "plateau_endpoint_fp32_sha256": canonical_tensor_sha256(plateau_endpoint),
        "gradient_fp32_sha256": canonical_tensor_sha256(gradient_cpu),
        "direction_fp32_sha256": canonical_tensor_sha256(direction),
        "gradient_norm": float(gradient_cpu.double().norm()),
        "gradient_nonzero_fraction": nonzero_fraction,
        "analytic_negative_gradient_derivative": float((gradient_cpu.double() * direction.double()).sum()),
        "precision_audit": dict(precision_audit),
    }
    tensors = {
        "teacher_endpoint": teacher_endpoint.detach().float().cpu().clone(),
        "plateau_endpoint": plateau_endpoint,
        "gradient": gradient_cpu,
        "negative_gradient_direction": direction,
        "path_endpoints": {},
        "gradient_scan_endpoints": {},
    }
    return record, tensors


def path_record(
    *,
    condition: str,
    pass_name: str,
    point_index: int,
    point: torch.Tensor,
    endpoint: torch.Tensor,
    teacher_endpoint: torch.Tensor,
    plateau_mse: float,
    terminal_config: Mapping[str, Any],
) -> dict[str, Any]:
    binding = terminal_config["terminal_path_contract"]["point_bindings"][point_index]
    core.require(
        canonical_tensor_sha256(point) == binding["x_T_fp32_sha256"],
        "Precision path candidate drift.",
    )
    loss = float((endpoint - teacher_endpoint).square().mean())
    return {
        "schema": f"{core.PREFIX}-path-row.v1",
        "protocol": core.PROTOCOL,
        "row_id": core.path_row_id(condition, pass_name, point_index),
        "condition": condition,
        "pass": pass_name,
        "source_point_index": point_index,
        "requested_remaining_l2": binding["requested_remaining_l2"],
        "x_T_fp32_sha256": binding["x_T_fp32_sha256"],
        "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
        "endpoint_bitwise_equal_to_condition_teacher": torch.equal(endpoint, teacher_endpoint),
        "loss": loss,
        "loss_ratio_to_plateau": loss / plateau_mse,
    }


def execute_path_scan(
    *,
    condition: str,
    oracle: phase1a.FrozenDreamLiteOracle,
    points: Mapping[int, torch.Tensor],
    teacher_endpoint: torch.Tensor,
    plateau_mse: float,
    teacher_image_path: Path,
    terminal_config: Mapping[str, Any],
    forward: Any,
    output_root: Path,
    metrics: Any,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    endpoints: dict[str, torch.Tensor] = {}
    orders = {
        "teacher-outward": core.PATH_POINT_INDICES,
        "plateau-inward": tuple(reversed(core.PATH_POINT_INDICES)),
    }
    for pass_name in core.PASSES:
        for point_index in orders[pass_name]:
            point = points[point_index]
            reused_teacher = pass_name == "teacher-outward" and point_index == 0
            output = None
            if reused_teacher:
                endpoint = teacher_endpoint.clone()
            else:
                set_student(oracle, point)
                output = forward(gradient=False)
                endpoint = output.z_t.detach().float().cpu()
            row = path_record(
                condition=condition,
                pass_name=pass_name,
                point_index=point_index,
                point=point,
                endpoint=endpoint,
                teacher_endpoint=teacher_endpoint,
                plateau_mse=plateau_mse,
                terminal_config=terminal_config,
            )
            rows.append(row)
            endpoints[row["row_id"]] = endpoint
            metrics.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            metrics.flush()
            os.fsync(metrics.fileno())
            image_path = output_root / f"path_images/{row['row_id']}.png"
            if output is None:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(teacher_image_path, image_path)
            else:
                phase1a._save_image(image_path, output.image)
            print(
                f"precision path {condition} {len(rows)}/14 {row['row_id']} ratio={row['loss_ratio_to_plateau']:.10g}",
                flush=True,
            )
    return rows, endpoints


def execute_gradient_scan(
    *,
    condition: str,
    oracle: phase1a.FrozenDreamLiteOracle,
    plateau_x_t: torch.Tensor,
    teacher_endpoint: torch.Tensor,
    plateau_mse: float,
    gradient: torch.Tensor,
    direction: torch.Tensor,
    forward: Any,
    output_root: Path,
    metrics: Any,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    endpoints: dict[str, torch.Tensor] = {}
    analytic = float((gradient.double() * direction.double()).sum())
    core.require(analytic < 0.0, "Negative-gradient analytic derivative is not negative.")
    for radius_index, radius in enumerate(core.GRADIENT_RADII):
        for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
            scanned = plateau_x_t + sign * radius * direction
            set_student(oracle, scanned)
            output = forward(gradient=False)
            endpoint = output.z_t.detach().float().cpu()
            loss = float((endpoint - teacher_endpoint).square().mean())
            row = {
                "schema": f"{core.PREFIX}-gradient-row.v1",
                "protocol": core.PROTOCOL,
                "row_id": core.gradient_row_id(condition, radius_index, sign_name),
                "condition": condition,
                "radius_index": radius_index,
                "radius_l2": radius,
                "sign": sign_name,
                "loss": loss,
                "loss_ratio_to_plateau": loss / plateau_mse,
                "analytic_directional_derivative": analytic,
                "scanned_x_T_fp32_sha256": canonical_tensor_sha256(scanned),
                "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
            }
            rows.append(row)
            endpoints[row["row_id"]] = endpoint
            metrics.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            metrics.flush()
            os.fsync(metrics.fileno())
            phase1a._save_image(output_root / f"gradient_images/{row['row_id']}.png", output.image)
            print(
                f"precision gradient {condition} {len(rows)}/12 {row['row_id']} "
                f"ratio={row['loss_ratio_to_plateau']:.10g}",
                flush=True,
            )
    return rows, endpoints


def write_report(root: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# R11_new activation-precision control",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Engineering gate: `{result['engineering_gate']}`",
        f"- Counters: `{json.dumps(result['counters'], sort_keys=True)}`",
    ]
    if result["mode"] == "formal":
        summary = result["scan_summary"]
        lines += [
            f"- Classification: `{result['classification']}`",
            f"- BF16 capture width: `{summary['bf16_capture_width']}`",
            f"- FP32 capture width: `{summary['fp32_capture_width']}`",
            f"- FP32/BF16 widening: `{summary['fp32_to_bf16_capture_widening_factor']}`",
            "- FP32 best negative-gradient ratio: "
            f"`{summary['conditions']['fp32-lifted']['gradient']['best_positive_loss_ratio']}`",
        ]
    lines += [
        "",
        "This is a fixed-target numerical precision diagnostic using oracle teacher xT.",
        "It does not establish Picture Memory training success, Reader success, ID/OOD success, or Phase 2 eligibility.",
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
    terminal_config = terminal_core.load_config()
    snapshots_start = {
        name: phase1a.verify_snapshot_binding(value) for name, value in direction_config["model_snapshots"].items()
    }
    phase1a._atomic_json(args.output_root / "model_snapshot_verification_start.json", snapshots_start)
    counters = {
        "full_chain_forward_calls": 0,
        "reader_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
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
        raise RuntimeError("Reader is forbidden in the activation-precision control.")

    reader_hook = reader.model.register_forward_pre_hook(forbidden_reader)
    try:
        core.require(
            low_lr_runner.models_frozen(oracle, reader) and low_lr_runner.only_student_trainable(oracle, reader),
            "Initial precision frozen/trainable contract failed.",
        )
        fixed = terminal_config["fixed_parent_artifacts"]
        core.require(
            core.sha256_file(args.output_root / "target/fixed_parent_target.pt")
            == fixed["source_target_artifact_sha256"]
            and core.sha256_file(args.output_root / "parent/lr001-step256.pt")
            == fixed["source_plateau_checkpoint_sha256"],
            "Copied precision parent bytes drift.",
        )
        all_points = terminal_core.construct_points(
            target["teacher_x_T_fp32"], plateau_checkpoint["student_x_T_fp32"], terminal_config
        )
        points = {index: all_points[index] for index in core.PATH_POINT_INDICES}
        teacher_x_t = target["teacher_x_T_fp32"].detach().float().cpu()
        plateau_x_t = plateau_checkpoint["student_x_T_fp32"].detach().float().cpu()
        bf16_teacher = target["teacher_endpoint_fp32"].detach().float().cpu()
        bf16_teacher_image = args.output_root / "conditions/bf16-baseline/teacher.png"
        bf16_teacher_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.output_root / "target/fixed_parent_target.png", bf16_teacher_image)

        baseline_state, baseline_summary = capture_precision_state(oracle)
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
            "precision_information_boundary": {
                "same_loaded_bf16_parameter_values": True,
                "condition_order": list(core.CONDITIONS),
                "teacher_x_T_exposed": True,
                "condition_specific_teacher_endpoints": True,
                "condition_specific_plateau_denominators": True,
                "text_encoder_reexecuted": False,
                "reader_used": False,
                "optimizer_steps": 0,
            },
            "formal_success": False,
            "phase2_allowed": False,
        }
        phase1a._atomic_json(args.output_root / "manifest.json", manifest)

        condition_records: dict[str, dict[str, Any]] = {}
        condition_tensors: dict[str, dict[str, Any]] = {}
        baseline_precision_audit = {
            "passed": True,
            "condition": "bf16-baseline",
            "layout": baseline_summary,
            "model_reloaded": False,
            "conditioning_regenerated": False,
        }
        condition_records["bf16-baseline"], condition_tensors["bf16-baseline"] = prepare_condition_gradient(
            condition="bf16-baseline",
            oracle=oracle,
            teacher_endpoint=bf16_teacher,
            plateau_x_t=plateau_x_t,
            counters=counters,
            forward=forward,
            output_root=args.output_root,
            precision_audit=baseline_precision_audit,
            config=config,
        )
        core.require(
            torch.equal(
                condition_tensors["bf16-baseline"]["plateau_endpoint"],
                plateau_checkpoint["endpoint_fp32"],
            ),
            "BF16 plateau gradient replay differs from parent.",
        )

        all_path_rows: list[dict[str, Any]] = []
        all_gradient_rows: list[dict[str, Any]] = []
        path_metrics = None
        gradient_metrics = None
        if args.mode == "formal":
            path_metrics = (args.output_root / "path_metrics.jsonl").open("x", encoding="utf-8", newline="\n")
            gradient_metrics = (args.output_root / "gradient_metrics.jsonl").open("x", encoding="utf-8", newline="\n")
            rows, endpoints = execute_path_scan(
                condition="bf16-baseline",
                oracle=oracle,
                points=points,
                teacher_endpoint=bf16_teacher,
                plateau_mse=condition_records["bf16-baseline"]["plateau_mse"],
                teacher_image_path=bf16_teacher_image,
                terminal_config=terminal_config,
                forward=forward,
                output_root=args.output_root,
                metrics=path_metrics,
            )
            all_path_rows.extend(rows)
            condition_tensors["bf16-baseline"]["path_endpoints"] = endpoints
            rows, endpoints = execute_gradient_scan(
                condition="bf16-baseline",
                oracle=oracle,
                plateau_x_t=plateau_x_t,
                teacher_endpoint=bf16_teacher,
                plateau_mse=condition_records["bf16-baseline"]["plateau_mse"],
                gradient=condition_tensors["bf16-baseline"]["gradient"],
                direction=condition_tensors["bf16-baseline"]["negative_gradient_direction"],
                forward=forward,
                output_root=args.output_root,
                metrics=gradient_metrics,
            )
            all_gradient_rows.extend(rows)
            condition_tensors["bf16-baseline"]["gradient_scan_endpoints"] = endpoints

        set_student(oracle, teacher_x_t)
        fp32_lift_audit = lift_oracle_to_fp32(oracle, baseline_state, baseline_summary)
        with torch.no_grad():
            fp32_teacher_output = forward(gradient=False)
        fp32_teacher = fp32_teacher_output.z_t.detach().float().cpu()
        fp32_teacher_image = args.output_root / "conditions/fp32-lifted/teacher.png"
        phase1a._save_image(fp32_teacher_image, fp32_teacher_output.image)
        condition_records["fp32-lifted"], condition_tensors["fp32-lifted"] = prepare_condition_gradient(
            condition="fp32-lifted",
            oracle=oracle,
            teacher_endpoint=fp32_teacher,
            plateau_x_t=plateau_x_t,
            counters=counters,
            forward=forward,
            output_root=args.output_root,
            precision_audit=fp32_lift_audit,
            config=config,
        )
        if args.mode == "formal":
            core.require(path_metrics is not None and gradient_metrics is not None, "Metrics missing.")
            rows, endpoints = execute_path_scan(
                condition="fp32-lifted",
                oracle=oracle,
                points=points,
                teacher_endpoint=fp32_teacher,
                plateau_mse=condition_records["fp32-lifted"]["plateau_mse"],
                teacher_image_path=fp32_teacher_image,
                terminal_config=terminal_config,
                forward=forward,
                output_root=args.output_root,
                metrics=path_metrics,
            )
            all_path_rows.extend(rows)
            condition_tensors["fp32-lifted"]["path_endpoints"] = endpoints
            rows, endpoints = execute_gradient_scan(
                condition="fp32-lifted",
                oracle=oracle,
                plateau_x_t=plateau_x_t,
                teacher_endpoint=fp32_teacher,
                plateau_mse=condition_records["fp32-lifted"]["plateau_mse"],
                gradient=condition_tensors["fp32-lifted"]["gradient"],
                direction=condition_tensors["fp32-lifted"]["negative_gradient_direction"],
                forward=forward,
                output_root=args.output_root,
                metrics=gradient_metrics,
            )
            all_gradient_rows.extend(rows)
            condition_tensors["fp32-lifted"]["gradient_scan_endpoints"] = endpoints
            path_metrics.close()
            gradient_metrics.close()

        tensor_bundle = {
            "schema": f"{core.PREFIX}-tensor-bundle.v1",
            "protocol": core.PROTOCOL,
            "candidates": points,
            "conditions": condition_tensors,
        }
        phase1a._atomic_torch_save(args.output_root / "precision_tensors.pt", tensor_bundle)
        snapshots_end = {
            name: phase1a.verify_snapshot_binding(value) for name, value in direction_config["model_snapshots"].items()
        }
        phase1a._atomic_json(args.output_root / "model_snapshot_verification_end.json", snapshots_end)
        frozen_gate = low_lr_runner.models_frozen(oracle, reader) and low_lr_runner.only_student_trainable(
            oracle, reader
        )
        if args.mode == "technical-preflight":
            expected_counters = {
                "full_chain_forward_calls": 4,
                "reader_forward_calls": 0,
                "backward_calls": 2,
                "optimizer_steps": 0,
            }
            engineering_gate = bool(
                counters == expected_counters
                and torch.equal(
                    condition_tensors["bf16-baseline"]["teacher_endpoint"],
                    target["teacher_endpoint_fp32"],
                )
                and torch.equal(
                    condition_tensors["bf16-baseline"]["plateau_endpoint"],
                    plateau_checkpoint["endpoint_fp32"],
                )
                and all(
                    bool(torch.isfinite(condition_tensors[condition][name]).all())
                    for condition in core.CONDITIONS
                    for name in ("teacher_endpoint", "plateau_endpoint", "gradient")
                )
                and fp32_lift_audit["passed"]
                and snapshots_end == snapshots_start
                and frozen_gate
            )
            core.require(engineering_gate, "Precision technical preflight gate failed.")
            scan_summary = None
            classification = None
        else:
            expected_counters = {
                "full_chain_forward_calls": 54,
                "reader_forward_calls": 0,
                "backward_calls": 2,
                "optimizer_steps": 0,
            }
            scan_summary = core.summarize_experiment(all_path_rows, all_gradient_rows, config)
            classification = core.classify_outcome(scan_summary)
            engineering_gate = bool(
                counters == expected_counters
                and len(all_path_rows) == 28
                and len(all_gradient_rows) == 24
                and all(
                    len(condition_tensors[condition]["path_endpoints"]) == 14
                    and len(condition_tensors[condition]["gradient_scan_endpoints"]) == 12
                    for condition in core.CONDITIONS
                )
                and scan_summary["bf16_parent_reproduced"]
                and fp32_lift_audit["passed"]
                and snapshots_end == snapshots_start
                and frozen_gate
            )
            core.require(engineering_gate, "Precision formal technical gate failed.")
        result = {
            "schema": f"{core.PREFIX}-result.v1",
            "protocol": core.PROTOCOL,
            "mode": args.mode,
            "git_commit": args.expected_commit,
            "engineering_gate": engineering_gate,
            "preflight_gate": engineering_gate if args.mode == "technical-preflight" else None,
            "conditions": condition_records,
            "scan_summary": scan_summary,
            "classification": classification,
            "counters": counters,
            "target_record": target_record,
            "tensor_bundle_sha256": core.sha256_file(args.output_root / "precision_tensors.pt"),
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
    lock_path = Path("/tmp/vision-memory-r11-new-activation-precision-control.lock")
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
