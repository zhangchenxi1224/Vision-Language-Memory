from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_functional


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_GRID_THW,
    R3_QWEN_READER_INPUT_HW,
    R3_QWEN_READER_PIXEL_VALUES_SHAPE,
    R3_QWEN_READER_RESIZE_CONTRACT,
    deterministic_qwen_reader_resize,
)
from vision_memory.repro import (  # noqa: E402
    canonical_tensor_sha256,
    configure_strict_cuda_determinism,
    emit_json_report,
    probe_provenance,
)


EXPECTED_MIN_MAX_PIXELS = 256 * 256
EXPECTED_PATCH_SIZE = 16
EXPECTED_TEMPORAL_PATCH_SIZE = 2
EXPECTED_MERGE_SIZE = 2
EXPECTED_BICUBIC_RESAMPLE_VALUE = 3
LEGACY_NATIVE_THRESHOLDS = {
    "float32": {"candidate_relative_l2_max": 1e-5, "candidate_cosine_min": 0.999999},
    "bfloat16": {"candidate_relative_l2_max": 1e-2, "candidate_cosine_min": 0.9999},
}
LEGACY_NATIVE_REPLICAS = 3
REQUIRED_STAGE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that the deterministic R3 Qwen resize preserves the frozen fast-processor "
            "forward exactly and has a bitwise-repeatable strict backward."
        )
    )
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _size_value(size: Any, key: str) -> int | None:
    if isinstance(size, Mapping):
        value = size.get(key)
    else:
        value = getattr(size, key, None)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return int(value)


def _resample_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_fast_processor(image_processor: Any) -> dict[str, Any]:
    """Fail-closed audit of the real Qwen fast image-processor geometry."""

    name = type(image_processor).__name__
    size = getattr(image_processor, "size", None)
    observed = {
        "class": name,
        "do_resize": getattr(image_processor, "do_resize", None),
        "min_pixels": getattr(image_processor, "min_pixels", None),
        "max_pixels": getattr(image_processor, "max_pixels", None),
        "shortest_edge": _size_value(size, "shortest_edge"),
        "longest_edge": _size_value(size, "longest_edge"),
        "patch_size": getattr(image_processor, "patch_size", None),
        "temporal_patch_size": getattr(image_processor, "temporal_patch_size", None),
        "merge_size": getattr(image_processor, "merge_size", None),
        "resample": str(getattr(image_processor, "resample", None)),
        "resample_value": _resample_value(getattr(image_processor, "resample", None)),
    }
    checks = {
        "fast_tensor_processor": "Fast" in name,
        "resize_enabled_by_default": observed["do_resize"] is True,
        "min_pixels_locked": observed["min_pixels"] == EXPECTED_MIN_MAX_PIXELS,
        "max_pixels_locked": observed["max_pixels"] == EXPECTED_MIN_MAX_PIXELS,
        "patch_size_locked": observed["patch_size"] == EXPECTED_PATCH_SIZE,
        "temporal_patch_size_locked": observed["temporal_patch_size"] == EXPECTED_TEMPORAL_PATCH_SIZE,
        "merge_size_locked": observed["merge_size"] == EXPECTED_MERGE_SIZE,
        "bicubic_resample_locked": observed["resample_value"] == EXPECTED_BICUBIC_RESAMPLE_VALUE,
        "callable": callable(image_processor),
    }
    return {"passed": all(checks.values()), "observed": observed, "checks": checks}


def _processor_output(image_processor: Any, image: Tensor, *, do_resize: bool | None) -> tuple[Tensor, Tensor]:
    kwargs: dict[str, Any] = {
        "images": [image],
        "return_tensors": "pt",
        "do_rescale": False,
    }
    if do_resize is not None:
        kwargs["do_resize"] = do_resize
    batch = image_processor(**kwargs)
    pixel_values = batch.get("pixel_values") if isinstance(batch, Mapping) else getattr(batch, "pixel_values", None)
    image_grid_thw = (
        batch.get("image_grid_thw") if isinstance(batch, Mapping) else getattr(batch, "image_grid_thw", None)
    )
    if not isinstance(pixel_values, Tensor) or not isinstance(image_grid_thw, Tensor):
        raise TypeError("Qwen fast processor must return tensor pixel_values and image_grid_thw.")
    return pixel_values, image_grid_thw


def _grid_values(grid: Tensor) -> tuple[int, ...]:
    if grid.numel() != 3:
        return ()
    return tuple(int(value) for value in grid.detach().cpu().reshape(-1).tolist())


def _tensor_summary(tensor: Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "sha256": canonical_tensor_sha256(tensor),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_inspire_execution_binding(environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Bind formal R3-R0 evidence to one launcher worker input and preflight."""

    observed = os.environ if environment is None else environment
    required = (
        "VLM_STAGE_WORKER_INPUT",
        "VLM_STAGE_CONFIGURATION_SHA256",
        "VLM_STAGE_PREFLIGHT",
        "VLM_STAGE_PREFLIGHT_SHA256",
    )
    missing = [name for name in required if not observed.get(name)]
    if missing:
        raise RuntimeError(f"R3-R0 must run through the Inspire formal launcher; missing {missing}.")
    worker_input = Path(observed["VLM_STAGE_WORKER_INPUT"]).resolve()
    preflight = Path(observed["VLM_STAGE_PREFLIGHT"]).resolve()
    configuration_sha256 = observed["VLM_STAGE_CONFIGURATION_SHA256"]
    preflight_sha256 = observed["VLM_STAGE_PREFLIGHT_SHA256"]
    if len(configuration_sha256) != 64 or len(preflight_sha256) != 64:
        raise ValueError("Inspire execution binding contains an invalid SHA256 digest.")
    if not worker_input.is_file() or _sha256_file(worker_input) != configuration_sha256:
        raise ValueError("Inspire worker_input.json is missing or its SHA256 does not match.")
    configuration = json.loads(worker_input.read_text(encoding="utf-8"))
    if not isinstance(configuration, dict):
        raise ValueError("Inspire worker_input.json must contain one object.")
    if configuration.get("stage") != "r3-r0" or configuration.get("infrastructure_stage") is not False:
        raise ValueError("R3-R0 requires a non-infrastructure Inspire stage named r3-r0.")
    if Path(str(configuration.get("preflight"))).resolve() != preflight:
        raise ValueError("Inspire worker input and stage environment reference different preflights.")
    if configuration.get("preflight_sha256") != preflight_sha256:
        raise ValueError("Inspire worker input and stage environment disagree on preflight SHA256.")
    if configuration.get("strict_environment") != REQUIRED_STAGE_ENVIRONMENT:
        raise ValueError("R3-R0 worker input does not lock the required stage environment.")
    expected_commit = configuration.get("expected_commit")
    if (
        not isinstance(expected_commit, str)
        or len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
    ):
        raise ValueError("R3-R0 worker input does not bind a full lowercase Git commit.")
    if not preflight.is_file() or _sha256_file(preflight) != preflight_sha256:
        raise ValueError("R3-R0 formal preflight is missing or its SHA256 does not match.")
    sidecar = preflight.with_suffix(preflight.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[0] != preflight_sha256:
        raise ValueError("R3-R0 formal preflight SHA sidecar is missing or disagrees.")
    preflight_report = json.loads(preflight.read_text(encoding="utf-8"))
    if (
        not isinstance(preflight_report, dict)
        or preflight_report.get("passed") is not True
        or preflight_report.get("formal_ready") is not True
        or preflight_report.get("git", {}).get("commit") != expected_commit
    ):
        raise ValueError("R3-R0 preflight is not formal-ready for the bound Git commit.")
    return {
        "passed": True,
        "stage": "r3-r0",
        "infrastructure_stage": False,
        "git_commit": expected_commit,
        "worker_input_path": str(worker_input),
        "worker_input_sha256": configuration_sha256,
        "formal_preflight_path": str(preflight),
        "formal_preflight_sha256": preflight_sha256,
    }


def compare_forward_paths(*, image_processor: Any, image: Tensor) -> dict[str, Any]:
    """Compare old processor-default resize against helper + processor no-resize."""

    if tuple(image.shape) != (3, *R3_QWEN_READER_INPUT_HW):
        raise ValueError("Forward comparison requires one locked 1024x1024 RGB image.")
    with torch.no_grad():
        legacy_pixels, legacy_grid = _processor_output(image_processor, image, do_resize=None)
        resized = deterministic_qwen_reader_resize(image, contract=R3_QWEN_READER_RESIZE_CONTRACT)
        candidate_pixels, candidate_grid = _processor_output(image_processor, resized, do_resize=False)

    same_pixel_shape = tuple(legacy_pixels.shape) == tuple(candidate_pixels.shape) == tuple(
        R3_QWEN_READER_PIXEL_VALUES_SHAPE
    )
    same_grid_shape = tuple(legacy_grid.shape) == tuple(candidate_grid.shape) == (1, 3)
    legacy_grid_values = _grid_values(legacy_grid)
    candidate_grid_values = _grid_values(candidate_grid)
    grid_exact = bool(
        same_grid_shape
        and legacy_grid_values == candidate_grid_values == tuple(R3_QWEN_READER_GRID_THW)
        and torch.equal(legacy_grid.detach().cpu(), candidate_grid.detach().cpu())
    )
    pixels_exact = bool(same_pixel_shape and torch.equal(legacy_pixels, candidate_pixels))
    finite = bool(torch.isfinite(legacy_pixels).all().item() and torch.isfinite(candidate_pixels).all().item())
    max_absolute_difference = (
        float((legacy_pixels.float() - candidate_pixels.float()).abs().max().item())
        if tuple(legacy_pixels.shape) == tuple(candidate_pixels.shape) and legacy_pixels.numel()
        else None
    )
    passed = bool(pixels_exact and grid_exact and finite)
    return {
        "passed": passed,
        "input": _tensor_summary(image),
        "resized": _tensor_summary(resized),
        "legacy_pixel_values": _tensor_summary(legacy_pixels),
        "candidate_pixel_values": _tensor_summary(candidate_pixels),
        "pixel_values_torch_equal": pixels_exact,
        "pixel_values_max_absolute_difference": max_absolute_difference,
        "expected_pixel_values_shape": list(R3_QWEN_READER_PIXEL_VALUES_SHAPE),
        "pixel_values_shape_locked": same_pixel_shape,
        "legacy_grid_thw": list(legacy_grid_values),
        "candidate_grid_thw": list(candidate_grid_values),
        "expected_grid_thw": list(R3_QWEN_READER_GRID_THW),
        "grid_torch_equal_and_locked": grid_exact,
    }


def _one_strict_backward(*, image_processor: Any, source: Tensor) -> dict[str, Any]:
    image = source.detach().clone().requires_grad_(True)
    resized = deterministic_qwen_reader_resize(image, contract=R3_QWEN_READER_RESIZE_CONTRACT)
    pixel_values, image_grid_thw = _processor_output(image_processor, resized, do_resize=False)
    if tuple(pixel_values.shape) != tuple(R3_QWEN_READER_PIXEL_VALUES_SHAPE):
        raise RuntimeError("Strict backward received an unexpected Qwen pixel_values shape.")
    if _grid_values(image_grid_thw) != tuple(R3_QWEN_READER_GRID_THW):
        raise RuntimeError("Strict backward received an unexpected Qwen image grid.")
    # The FP32 reduction is deterministic under the locked CUDA protocol and
    # exercises normalization, patchification, the custom resize adjoint, and
    # the device/dtype return path without requiring the 4B Reader weights.
    loss = pixel_values.float().square().mean()
    loss.backward()
    gradient = image.grad
    if gradient is None:
        raise RuntimeError("The deterministic Qwen resize detached the source image.")
    return {
        "loss": float(loss.detach().item()),
        "loss_float_hex": float(loss.detach().item()).hex(),
        "pixel_values_sha256": canonical_tensor_sha256(pixel_values),
        "gradient": gradient.detach().clone(),
    }


def compare_strict_backwards(*, image_processor: Any, image: Tensor) -> dict[str, Any]:
    """Run the locked helper backward twice and require bitwise-identical gradients."""

    first = _one_strict_backward(image_processor=image_processor, source=image)
    second = _one_strict_backward(image_processor=image_processor, source=image)
    first_gradient = first.pop("gradient")
    second_gradient = second.pop("gradient")
    finite = bool(
        torch.isfinite(first_gradient).all().item() and torch.isfinite(second_gradient).all().item()
    )
    first_norm = float(first_gradient.float().norm().item())
    second_norm = float(second_gradient.float().norm().item())
    nonzero = bool(math.isfinite(first_norm) and math.isfinite(second_norm) and first_norm > 0 and second_norm > 0)
    exact = bool(torch.equal(first_gradient, second_gradient))
    loss_exact = bool(first["loss_float_hex"] == second["loss_float_hex"])
    gradient_max_absolute_difference = float(
        (first_gradient.float() - second_gradient.float()).abs().max().item()
    )
    passed = bool(finite and nonzero and exact and loss_exact)
    return {
        "passed": passed,
        "first": {
            **first,
            "gradient": _tensor_summary(first_gradient),
            "gradient_norm": first_norm,
        },
        "second": {
            **second,
            "gradient": _tensor_summary(second_gradient),
            "gradient_norm": second_norm,
        },
        "gradient_finite": finite,
        "gradient_nonzero": nonzero,
        "gradient_torch_equal": exact,
        "gradient_max_absolute_difference": gradient_max_absolute_difference,
        "loss_bitwise_equal": loss_exact,
    }


def compare_cpu_adjoint_reference(*, image: Tensor) -> dict[str, Any]:
    """Require the custom CUDA-facing backward to equal native CPU autograd exactly.

    This is deliberately separate from the helper implementation: the reference
    invokes torchvision's public resize directly on an FP32 CPU source.  For BF16,
    both the output cotangent and final source gradient follow the locked mixed-
    precision cast contract.
    """

    if tuple(image.shape) != (3, *R3_QWEN_READER_INPUT_HW):
        raise ValueError("CPU adjoint comparison requires one locked 1024x1024 RGB image.")
    output_gradient_fp32 = torch.linspace(
        -1.0,
        1.0,
        steps=3 * 256 * 256,
        device="cpu",
        dtype=torch.float32,
    ).reshape(3, 256, 256)
    output_gradient = output_gradient_fp32.to(device=image.device, dtype=image.dtype)

    candidate_source = image.detach().clone().requires_grad_(True)
    candidate_output = deterministic_qwen_reader_resize(
        candidate_source,
        contract=R3_QWEN_READER_RESIZE_CONTRACT,
    )
    candidate_output.backward(output_gradient)
    candidate_gradient = candidate_source.grad
    if candidate_gradient is None:
        raise RuntimeError("Candidate resize did not return a source gradient.")

    with torch.enable_grad():
        reference_source = torch.zeros(
            (1, 3, *R3_QWEN_READER_INPUT_HW),
            device="cpu",
            dtype=torch.float32,
            requires_grad=True,
        )
        reference_output = tv_functional.resize(
            reference_source,
            [256, 256],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )[0]
        (reference_gradient_fp32,) = torch.autograd.grad(
            outputs=reference_output,
            inputs=reference_source,
            grad_outputs=output_gradient.detach().to(device="cpu", dtype=torch.float32),
            create_graph=False,
            retain_graph=False,
        )
    reference_gradient = reference_gradient_fp32[0].to(device=image.device, dtype=image.dtype)

    finite = bool(
        torch.isfinite(candidate_gradient).all().item()
        and torch.isfinite(reference_gradient).all().item()
    )
    candidate_norm = float(candidate_gradient.float().norm().item())
    reference_norm = float(reference_gradient.float().norm().item())
    nonzero = bool(
        math.isfinite(candidate_norm)
        and math.isfinite(reference_norm)
        and candidate_norm > 0
        and reference_norm > 0
    )
    exact = bool(torch.equal(candidate_gradient, reference_gradient))
    maximum_difference = float(
        (candidate_gradient.float() - reference_gradient.float()).abs().max().item()
    )
    return {
        "passed": bool(finite and nonzero and exact and maximum_difference == 0.0),
        "reference": "native-torchvision-cpu-fp32-autograd",
        "output_gradient": _tensor_summary(output_gradient),
        "candidate_gradient": _tensor_summary(candidate_gradient),
        "reference_gradient": _tensor_summary(reference_gradient),
        "gradient_finite": finite,
        "gradient_nonzero": nonzero,
        "gradient_torch_equal": exact,
        "gradient_max_absolute_difference": maximum_difference,
        "candidate_gradient_norm": candidate_norm,
        "reference_gradient_norm": reference_norm,
    }


def _gradient_distance(candidate: Tensor, reference: Tensor) -> tuple[float, float]:
    candidate_cpu = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    reference_cpu = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    candidate_norm = torch.linalg.vector_norm(candidate_cpu)
    reference_norm = torch.linalg.vector_norm(reference_cpu)
    if float(candidate_norm) == 0.0 or float(reference_norm) == 0.0:
        return math.inf, -1.0
    relative_l2 = float(torch.linalg.vector_norm(candidate_cpu - reference_cpu) / reference_norm)
    cosine = float(torch.dot(candidate_cpu, reference_cpu) / (candidate_norm * reference_norm))
    return relative_l2, max(-1.0, min(1.0, cosine))


def _one_legacy_native_backward(*, image_processor: Any, source: Tensor) -> dict[str, Any]:
    """Run the legacy native backward only as an isolated numerical reference."""

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False, warn_only=False)
        image = source.detach().clone().requires_grad_(True)
        pixel_values, image_grid_thw = _processor_output(image_processor, image, do_resize=None)
        if tuple(pixel_values.shape) != tuple(R3_QWEN_READER_PIXEL_VALUES_SHAPE):
            raise RuntimeError("Legacy native reference received an unexpected Qwen pixel_values shape.")
        if _grid_values(image_grid_thw) != tuple(R3_QWEN_READER_GRID_THW):
            raise RuntimeError("Legacy native reference received an unexpected Qwen image grid.")
        loss = pixel_values.float().square().mean()
        loss.backward()
        gradient = image.grad
        if gradient is None:
            raise RuntimeError("Legacy native reference did not return a source gradient.")
        result = {
            "loss": float(loss.detach().item()),
            "loss_float_hex": float(loss.detach().item()).hex(),
            "pixel_values_sha256": canonical_tensor_sha256(pixel_values),
            "gradient": gradient.detach().clone(),
        }
    finally:
        torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)
    result["determinism_restored"] = bool(
        torch.are_deterministic_algorithms_enabled() == previous_enabled
        and torch.is_deterministic_algorithms_warn_only_enabled() == previous_warn_only
    )
    return result


def compare_legacy_native_cuda_reference(*, image_processor: Any, image: Tensor) -> dict[str, Any]:
    """Numerically compare strict candidate gradients with legacy CUDA reductions.

    The legacy branch is reference-only: it has no model weights, optimizer, or
    scientific metric, and strict determinism is restored before returning.
    Native run-to-run variation is recorded but never used to relax thresholds.
    """

    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise RuntimeError("The candidate path must enter the legacy comparison under strict determinism.")
    label = "bfloat16" if image.dtype == torch.bfloat16 else "float32"
    if label not in LEGACY_NATIVE_THRESHOLDS or image.dtype not in {torch.float32, torch.bfloat16}:
        raise TypeError("Legacy native reference supports only the locked FP32/BF16 R3 dtypes.")
    thresholds = LEGACY_NATIVE_THRESHOLDS[label]
    candidate = _one_strict_backward(image_processor=image_processor, source=image)
    candidate_gradient = candidate.pop("gradient")
    native = [
        _one_legacy_native_backward(image_processor=image_processor, source=image)
        for _ in range(LEGACY_NATIVE_REPLICAS)
    ]
    native_gradients = [run.pop("gradient") for run in native]

    candidate_distances = [
        _gradient_distance(candidate_gradient, native_gradient)
        for native_gradient in native_gradients
    ]
    native_repeat_distances = [
        _gradient_distance(native_gradients[left], native_gradients[right])[0]
        for left in range(len(native_gradients))
        for right in range(left + 1, len(native_gradients))
    ]
    relative_l2_max = max(distance[0] for distance in candidate_distances)
    cosine_min = min(distance[1] for distance in candidate_distances)
    native_repeat_relative_l2_max = max(native_repeat_distances, default=0.0)
    all_gradients = [candidate_gradient, *native_gradients]
    finite_nonzero = all(
        bool(torch.isfinite(gradient).all().item())
        and math.isfinite(float(gradient.float().norm().item()))
        and float(gradient.float().norm().item()) > 0.0
        for gradient in all_gradients
    )
    determinism_restored = all(bool(run["determinism_restored"]) for run in native)
    passed = bool(
        finite_nonzero
        and determinism_restored
        and relative_l2_max <= thresholds["candidate_relative_l2_max"]
        and cosine_min >= thresholds["candidate_cosine_min"]
    )
    return {
        "passed": passed,
        "replicas": LEGACY_NATIVE_REPLICAS,
        "reference_only": True,
        "no_optimizer": True,
        "no_scientific_metric": True,
        "candidate_strict_determinism": True,
        "native_reference_determinism_disabled": True,
        "determinism_restored": determinism_restored,
        "thresholds": dict(thresholds),
        "all_gradients_finite_nonzero": finite_nonzero,
        "candidate_relative_l2_max": relative_l2_max,
        "candidate_cosine_min": cosine_min,
        "native_repeat_relative_l2_max": native_repeat_relative_l2_max,
        "candidate": {
            **candidate,
            "gradient": _tensor_summary(candidate_gradient),
            "gradient_norm": float(candidate_gradient.float().norm().item()),
        },
        "native_runs": [
            {
                **run,
                "gradient": _tensor_summary(gradient),
                "gradient_norm": float(gradient.float().norm().item()),
                "candidate_relative_l2": distance[0],
                "candidate_cosine": distance[1],
            }
            for run, gradient, distance in zip(native, native_gradients, candidate_distances, strict=True)
        ],
    }


def run_contract(*, image_processor: Any, device: torch.device, seed: int) -> dict[str, Any]:
    processor_audit = audit_fast_processor(image_processor)
    if not processor_audit["passed"]:
        raise RuntimeError(f"Qwen fast processor contract drifted: {processor_audit['checks']}")
    generator = torch.Generator(device=device).manual_seed(seed)
    base = torch.rand(
        (3, *R3_QWEN_READER_INPUT_HW),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    dtype_reports: dict[str, Any] = {}
    for label, image in (("float32", base), ("bfloat16", base.to(torch.bfloat16))):
        forward = compare_forward_paths(image_processor=image_processor, image=image)
        backward = compare_strict_backwards(image_processor=image_processor, image=image)
        adjoint = compare_cpu_adjoint_reference(image=image)
        legacy_native = compare_legacy_native_cuda_reference(
            image_processor=image_processor,
            image=image,
        )
        dtype_reports[label] = {
            "passed": bool(
                forward["passed"]
                and backward["passed"]
                and adjoint["passed"]
                and legacy_native["passed"]
            ),
            "forward_equivalence": forward,
            "strict_backward_repeat": backward,
            "cpu_adjoint_reference": adjoint,
            "legacy_native_cuda_reference": legacy_native,
        }
    passed = bool(processor_audit["passed"] and all(item["passed"] for item in dtype_reports.values()))
    return {
        "schema_version": 2,
        "probe": "r3_qwen_resize_forward_backward_contract",
        "passed": passed,
        "resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "seed": seed,
        "device": str(device),
        "processor": processor_audit,
        "dtypes": dtype_reports,
    }


def _runtime_report(device: torch.device) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in ("torchvision", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "packages": packages,
        "device_name": properties.name,
        "device_total_memory_bytes": int(properties.total_memory),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_report: dict[str, Any] = {
        "schema_version": 2,
        "probe": "r3_qwen_resize_forward_backward_contract",
        "passed": False,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("The real Qwen resize contract probe requires CUDA.")
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("--device must select a CUDA device.")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The locked R3 probe requires CUDA BF16 support.")
        determinism = configure_strict_cuda_determinism(args.seed)
        execution_binding = audit_inspire_execution_binding()

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            args.reader,
            local_files_only=True,
            use_fast=True,
            min_pixels=EXPECTED_MIN_MAX_PIXELS,
            max_pixels=EXPECTED_MIN_MAX_PIXELS,
        )
        report = run_contract(
            image_processor=processor.image_processor,
            device=device,
            seed=args.seed,
        )
        report.update(
            {
                "strict_determinism": determinism,
                "execution_binding": execution_binding,
                "runtime": _runtime_report(device),
                "provenance": probe_provenance(
                    root=ROOT,
                    arguments=args,
                    models={"reader": args.reader},
                ),
            }
        )
    except Exception as error:  # noqa: BLE001 - every contract failure must emit an audit JSON
        report = {
            **base_report,
            "error": {"type": type(error).__name__, "message": str(error)},
            "provenance": probe_provenance(
                root=ROOT,
                arguments=args,
                models={"reader": args.reader},
            ),
        }
    emit_json_report(report, args.output_json)
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
