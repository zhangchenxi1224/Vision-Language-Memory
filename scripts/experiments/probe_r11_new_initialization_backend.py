"""Reproduce initialization backend arithmetic with tensors only, never models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Sequence

import torch
from torch import Tensor


SCHEMA = "vision_memory.r11-new-initialization-backend-probe.v1"
SEED = 0
SHAPE = (1, 4, 128, 128)
SIGMA = 0.4999999701976776


def _inverse(source: Tensor, teacher: Tensor) -> Tensor:
    return teacher.sub(source.mul(1 - SIGMA)).div(SIGMA)


def _start(source: Tensor, x_t: Tensor) -> Tensor:
    return source.to(torch.bfloat16).mul(1 - SIGMA).add(
        x_t.to(torch.bfloat16), alpha=SIGMA,
    )


def _tensor_record(value: Tensor) -> dict[str, Any]:
    cpu = value.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "raw_bytes_sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def _compare(left: Tensor, right: Tensor) -> dict[str, Any]:
    left_cpu = left.detach().cpu().contiguous()
    right_cpu = right.detach().cpu().contiguous()
    if left_cpu.shape != right_cpu.shape or left_cpu.dtype != right_cpu.dtype:
        raise ValueError("Backend comparison shape/dtype mismatch.")
    if not torch.isfinite(left_cpu).all() or not torch.isfinite(right_cpu).all():
        raise ValueError("Backend comparison produced non-finite tensors.")
    left_record, right_record = _tensor_record(left_cpu), _tensor_record(right_cpu)
    return {
        "max_abs_difference": float((left_cpu.float() - right_cpu.float()).abs().max()),
        "different_element_count": int(torch.count_nonzero(left_cpu != right_cpu)),
        "element_count": left_cpu.numel(),
        "torch_equal": torch.equal(left_cpu, right_cpu),
        "raw_bytes_equal": left_record["raw_bytes_sha256"] == right_record["raw_bytes_sha256"],
        "left": left_record,
        "right": right_record,
    }


@torch.no_grad()
def run_probe(device_name: str = "cuda:0") -> dict[str, Any]:
    device = torch.device(device_name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU or CUDA probe backends are supported.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is forbidden.")

    # Generate once on CPU, then copy the same inputs to the requested backend.
    torch.manual_seed(SEED)
    source_cpu = torch.randn(SHAPE, dtype=torch.bfloat16).float()
    teacher_cpu = torch.randn_like(source_cpu)
    source_device = source_cpu.to(device)
    teacher_device = teacher_cpu.to(device)

    x_t_cpu = _inverse(source_cpu, teacher_cpu)
    x_t_device = _inverse(source_device, teacher_device)
    x_t_device_repeat = _inverse(source_device, teacher_device)

    # Isolate h0 arithmetic: both backends receive the SAME device-produced xT.
    shared_x_t_cpu = x_t_device.cpu()
    start_cpu_shared_x_t = _start(source_cpu, shared_x_t_cpu)
    start_device_shared_x_t = _start(source_device, x_t_device)
    start_device_repeat = _start(source_device, x_t_device)

    return {
        "schema": SCHEMA,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "fresh_execution_of_this_tensor_only_probe",
        "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "platform_system": platform.system(),
            "requested_device": device_name,
            "actual_compute_device_type": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "contract": {
            "seed": SEED,
            "shape": list(SHAPE),
            "sigma": SIGMA,
            "input_generation_device": "cpu",
            "source_expression": "s=torch.randn(1,4,128,128,dtype=torch.bfloat16).float()",
            "teacher_expression": "z=torch.randn_like(s)",
            "inverse_expression": "z.sub(s.mul(1-sigma)).div(sigma)",
            "start_expression": "s.to(torch.bfloat16).mul(1-sigma).add(x.to(torch.bfloat16),alpha=sigma)",
            "h0_comparison_uses_same_device_produced_x_T": True,
            "model_forward_calls": 0,
            "training_steps": 0,
            "scientific_parameters_changed": False,
        },
        "inputs": {"source_fp32": _tensor_record(source_cpu), "teacher_fp32": _tensor_record(teacher_cpu)},
        "comparisons": {
            "cpu_vs_device_x_T": _compare(x_t_cpu, x_t_device),
            "cpu_vs_device_h0_with_shared_device_x_T": _compare(start_cpu_shared_x_t, start_device_shared_x_t),
            "device_repeat_x_T": _compare(x_t_device, x_t_device_repeat),
            "device_repeat_h0": _compare(start_device_shared_x_t, start_device_repeat),
        },
        "interpretation": {
            "cpu_mode_is_cpu_only_smoke_test": device.type == "cpu",
            "cross_backend_difference_required_for_success": False,
            "model_or_scientific_result": False,
            "conclusion_boundary": (
                "This only measures backend arithmetic for fixed toy tensors. "
                "It neither evaluates a model nor changes scientific gates; "
                "CUDA artifact verification must recompute on its bound CUDA backend."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0", help="Explicit backend; default cuda:0, no CPU fallback.")
    parser.add_argument("--output", type=Path, help="Optional JSON artifact path; existing paths are rejected.")
    args = parser.parse_args(argv)
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    report = run_probe(args.device)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        # Exclusive creation also closes the race after the early existence check.
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
