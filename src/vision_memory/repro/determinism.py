"""Fail-closed helpers for bitwise CUDA reproducibility diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import random
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


REQUIRED_DETERMINISM_ENV = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def assert_determinism_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Require process-level settings that must be present before Python/CUDA starts."""

    observed = os.environ if environment is None else environment
    mismatches = {
        key: {"expected": expected, "observed": observed.get(key)}
        for key, expected in REQUIRED_DETERMINISM_ENV.items()
        if observed.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Strict determinism environment mismatch: {mismatches}")
    return {key: observed[key] for key in sorted(REQUIRED_DETERMINISM_ENV)}


def configure_strict_cuda_determinism(seed: int = 0) -> dict[str, Any]:
    """Enable the fixed math-only CUDA protocol used by the diagnostic probe."""

    environment = assert_determinism_environment()
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    required_backend_controls = (
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
        "enable_cudnn_sdp",
        "enable_math_sdp",
    )
    missing = [name for name in required_backend_controls if not hasattr(torch.backends.cuda, name)]
    if missing:
        raise RuntimeError(f"Pinned Torch lacks required SDPA backend controls: {missing}")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    def backend_enabled(name: str) -> bool | None:
        function = getattr(torch.backends.cuda, name, None)
        return None if function is None else bool(function())

    report = {
        "seed": seed,
        "environment": environment,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "sdpa": {
            "flash": backend_enabled("flash_sdp_enabled"),
            "memory_efficient": backend_enabled("mem_efficient_sdp_enabled"),
            "cudnn": backend_enabled("cudnn_sdp_enabled"),
            "math": backend_enabled("math_sdp_enabled"),
        },
    }
    expected_sdpa = {"flash": False, "memory_efficient": False, "cudnn": False, "math": True}
    if report["sdpa"] != expected_sdpa:
        raise RuntimeError(f"Strict math-only SDPA configuration failed: {report['sdpa']}")
    if not report["deterministic_algorithms"] or report["deterministic_warn_only"]:
        raise RuntimeError("Strict deterministic algorithms were not enabled fail-closed.")
    return report


def _tensor_bytes(tensor: Tensor) -> bytes:
    if tensor.layout != torch.strided:
        raise TypeError(f"Canonical tensor hashing requires strided tensors; got {tensor.layout}.")
    value = tensor.detach().cpu().contiguous()
    if value.is_quantized:
        raise TypeError("Canonical tensor hashing does not accept quantized tensors.")
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def canonical_tensor_manifest(tensor: Tensor) -> dict[str, Any]:
    """Hash logical tensor contents together with dtype and shape."""

    metadata = {
        "schema_version": "vision_memory.canonical_tensor.v1",
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
    }
    digest = hashlib.sha256()
    digest.update(b"vision_memory.canonical_tensor.v1\0")
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0")
    digest.update(_tensor_bytes(tensor))
    return {**metadata, "sha256": digest.hexdigest()}


def canonical_tensor_sha256(tensor: Tensor) -> str:
    return str(canonical_tensor_manifest(tensor)["sha256"])


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return {"__float64_be__": struct.pack(">d", value).hex()}
    if isinstance(value, Tensor):
        return {"__tensor__": canonical_tensor_manifest(value)}
    if isinstance(value, bytes):
        return {
            "__bytes__": {
                "length": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        }
    if isinstance(value, Path):
        return {"__path__": str(value)}
    if isinstance(value, torch.dtype):
        return {"__torch_dtype__": str(value)}
    if isinstance(value, torch.device):
        return {"__torch_device__": str(value)}
    if isinstance(value, Mapping):
        pairs = [(_canonical_value(key), _canonical_value(item)) for key, item in value.items()]
        pairs.sort(key=lambda pair: json.dumps(pair[0], sort_keys=True, separators=(",", ":")))
        return {"__mapping__": pairs}
    if isinstance(value, tuple):
        return {"__tuple__": [_canonical_value(item) for item in value]}
    if isinstance(value, list):
        return {"__list__": [_canonical_value(item) for item in value]}
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        items.sort(key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        return {"__set__": items}
    raise TypeError(f"Unsupported canonical object type: {type(value)!r}")


def canonical_object_sha256(value: Any) -> str:
    """Return a type-preserving, order-stable object digest."""

    payload = json.dumps(_canonical_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def named_tensors_manifest(
    values: Mapping[str, Tensor | None] | Iterable[tuple[str, Tensor | None]],
) -> dict[str, Any]:
    """Return both a bundle digest and per-name tensor digests."""

    items = values.items() if isinstance(values, Mapping) else values
    tensors = {
        str(name): None if tensor is None else canonical_tensor_manifest(tensor)
        for name, tensor in sorted(items, key=lambda item: str(item[0]))
    }
    return {
        "bundle_sha256": canonical_object_sha256(tensors),
        "tensors": tensors,
    }


def rng_manifest() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and every visible CUDA RNG state."""

    import numpy as np

    numpy_state = np.random.get_state()
    numpy_payload = {
        "bit_generator": numpy_state[0],
        "keys": numpy_state[1].tolist(),
        "position": int(numpy_state[2]),
        "has_gauss": int(numpy_state[3]),
        "cached_gaussian": float(numpy_state[4]),
    }
    components = {
        "python_sha256": canonical_object_sha256(random.getstate()),
        "numpy_sha256": canonical_object_sha256(numpy_payload),
        "torch_cpu_sha256": canonical_tensor_sha256(torch.get_rng_state()),
        "torch_cuda_sha256": [canonical_tensor_sha256(state) for state in torch.cuda.get_rng_state_all()],
    }
    return {**components, "bundle_sha256": canonical_object_sha256(components)}


def model_optimizer_rng_manifest(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    # ``state_dict`` intentionally omits non-persistent buffers.  Reproducibility
    # fingerprints must include them: the lightweight updater's fixed spatial basis
    # and zero initial state both affect every forward despite being non-persistent.
    model_tensors = [
        *((f"parameter:{name}", parameter) for name, parameter in model.named_parameters()),
        *((f"buffer:{name}", buffer) for name, buffer in model.named_buffers()),
    ]
    model_manifest = named_tensors_manifest(model_tensors)
    optimizer_sha256 = canonical_object_sha256(optimizer.state_dict())
    rng = rng_manifest()
    bundle = {
        "model_sha256": model_manifest["bundle_sha256"],
        "optimizer_sha256": optimizer_sha256,
        "rng_sha256": rng["bundle_sha256"],
    }
    return {
        "bundle_sha256": canonical_object_sha256(bundle),
        "model": model_manifest,
        "optimizer_sha256": optimizer_sha256,
        "rng": rng,
    }


def _deep_differences(left: Any, right: Any, *, path: str = "$", limit: int = 100) -> list[str]:
    differences: list[str] = []

    def visit(first: Any, second: Any, current: str) -> None:
        if len(differences) >= limit:
            return
        if type(first) is not type(second):
            differences.append(f"{current}: type {type(first).__name__} != {type(second).__name__}")
            return
        if isinstance(first, Mapping):
            first_keys = set(first)
            second_keys = set(second)
            for key in sorted(first_keys - second_keys, key=str):
                differences.append(f"{current}.{key}: missing on right")
            for key in sorted(second_keys - first_keys, key=str):
                differences.append(f"{current}.{key}: missing on left")
            for key in sorted(first_keys & second_keys, key=str):
                visit(first[key], second[key], f"{current}.{key}")
            return
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
            if len(first) != len(second):
                differences.append(f"{current}: length {len(first)} != {len(second)}")
                return
            for index, (first_item, second_item) in enumerate(zip(first, second, strict=True)):
                visit(first_item, second_item, f"{current}[{index}]")
            return
        if first != second:
            differences.append(f"{current}: {first!r} != {second!r}")

    visit(left, right, path)
    return differences


def compare_bitwise_repro_reports(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Compare only the reports' explicitly canonical, runtime-free payloads."""

    statuses = [first.get("status"), second.get("status")]
    if statuses != ["complete", "complete"]:
        return {
            "valid": False,
            "reason": "both replicas must complete",
            "statuses": statuses,
            "mismatches": [],
        }
    if "comparison_payload" not in first or "comparison_payload" not in second:
        raise ValueError("Both reports must contain comparison_payload.")
    first_payload = first["comparison_payload"]
    second_payload = second["comparison_payload"]
    first_sha = canonical_object_sha256(first_payload)
    second_sha = canonical_object_sha256(second_payload)
    differences = _deep_differences(first_payload, second_payload)
    valid = first_sha == second_sha and not differences
    return {
        "valid": valid,
        "reason": "bitwise canonical payloads match" if valid else "canonical payload mismatch",
        "first_payload_sha256": first_sha,
        "second_payload_sha256": second_sha,
        "mismatches": differences,
    }
