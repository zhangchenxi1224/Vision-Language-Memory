"""Small, dependency-light contracts for reproducible gradient probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torch import nn


DETERMINISTIC_FIXTURE_ID = "vision-memory-rgb-blocks-v1"
DETERMINISTIC_FIXTURE_RGB_SHA256_1024 = "c44093f3ad73d6a3d62b5bf9b8ad226f65e65afd7841d5ef3ed80bc7d14a841a"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return {key: _jsonable(item) for key, item in sorted(vars(value).items())}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rgb_sha256(image: Image.Image) -> str:
    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(b"vision-memory-canonical-rgb-v1\0")
    digest.update(f"{rgb.width}x{rgb.height}\0".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def _deterministic_fixture(resolution: int) -> Image.Image:
    if resolution <= 0:
        raise ValueError("Fixture resolution must be positive.")
    image = Image.new("RGB", (resolution, resolution), color=(19, 29, 43))
    draw = ImageDraw.Draw(image)
    tile = max(1, resolution // 16)
    for y_index, top in enumerate(range(0, resolution, tile)):
        for x_index, left in enumerate(range(0, resolution, tile)):
            color = (
                (31 * x_index + 17 * y_index + 23) % 256,
                (13 * x_index + 47 * y_index + 71) % 256,
                (59 * x_index + 7 * y_index + 101) % 256,
            )
            draw.rectangle(
                (left, top, min(left + tile - 1, resolution - 1), min(top + tile - 1, resolution - 1)),
                fill=color,
            )
    inset = resolution // 4
    draw.rectangle((inset, inset, resolution - inset - 1, resolution - inset - 1), fill=(196, 48, 52))
    stripe = max(1, resolution // 32)
    draw.rectangle((0, resolution // 2 - stripe, resolution - 1, resolution // 2 + stripe), fill=(235, 221, 84))
    return image


def load_source_image(path: Path | None, *, resolution: int = 1024) -> tuple[Image.Image, dict[str, Any]]:
    """Load a user image or return the versioned deterministic RGB fixture.

    The reported RGB digest hashes dimensions plus decoded RGB pixels, so it is stable
    across PNG encoder settings and is suitable for positive/negative-control pairing.
    """

    if path is None:
        image = _deterministic_fixture(resolution)
        metadata = {
            "origin": "deterministic_fixture",
            "fixture_id": f"{DETERMINISTIC_FIXTURE_ID}-{resolution}",
            "path": None,
            "file_sha256": None,
        }
    else:
        resolved = path.expanduser().resolve(strict=True)
        with Image.open(resolved) as opened:
            image = opened.convert("RGB").copy()
        metadata = {
            "origin": "file",
            "fixture_id": None,
            "path": str(resolved),
            "file_sha256": _file_sha256(resolved),
        }
    metadata.update(
        {
            "mode": "RGB",
            "size": [image.width, image.height],
            "rgb_sha256": _rgb_sha256(image),
        }
    )
    if path is None and resolution == 1024 and metadata["rgb_sha256"] != DETERMINISTIC_FIXTURE_RGB_SHA256_1024:
        raise RuntimeError("The deterministic 1024 RGB fixture no longer matches its locked SHA256.")
    return image, metadata


def _run_git(root: Path, *arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_metadata(root: Path) -> dict[str, Any]:
    status = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": _run_git(root, "rev-parse", "HEAD"),
        "clean": status == "" if status is not None else None,
        "status_sha256": None if status is None else hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _locked_models(root: Path) -> dict[str, dict[str, Any]]:
    lock_path = root / "models.lock.json"
    if not lock_path.is_file():
        return {}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return dict(lock.get("models", {}))


def _model_metadata(root: Path, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    marker = resolved / ".locked_revision"
    marker_revision = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    expected = None
    repo_id = None
    for specification in _locked_models(root).values():
        local_dir = Path(str(specification.get("local_dir", "")))
        if local_dir.name == resolved.name:
            expected = specification.get("revision")
            repo_id = specification.get("repo_id")
            break
    snapshot_revision = None
    if resolved.parent.name == "snapshots" and len(resolved.name) >= 7:
        snapshot_revision = resolved.name
    observed = marker_revision or snapshot_revision
    return {
        "path": str(resolved),
        "repo_id": repo_id,
        "expected_revision": expected,
        "observed_revision": observed,
        "revision_matches_lock": None if observed is None or expected is None else observed == expected,
    }


def probe_provenance(
    *,
    root: Path,
    arguments: argparse.Namespace | Mapping[str, Any],
    models: Mapping[str, Path],
    source_image: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "git": _git_metadata(root),
        "arguments": _jsonable(arguments),
        "models": {name: _model_metadata(root, path) for name, path in sorted(models.items())},
        "source_image": None if source_image is None else _jsonable(source_image),
        "runtime": {
            "python_pid": os.getpid(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }


def _unique_cuda_devices(devices: Iterable[torch.device | str]) -> list[torch.device]:
    unique: dict[int, torch.device] = {}
    for value in devices:
        device = torch.device(value)
        if device.type != "cuda":
            continue
        index = torch.cuda.current_device() if device.index is None else device.index
        unique[index] = torch.device("cuda", index)
    return [unique[index] for index in sorted(unique)]


def reset_cuda_peak_memory(devices: Iterable[torch.device | str]) -> None:
    for device in _unique_cuda_devices(devices):
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_memory_report(devices: Iterable[torch.device | str]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for device in _unique_cuda_devices(devices):
        torch.cuda.synchronize(device)
        report[str(device)] = {
            "name": torch.cuda.get_device_name(device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
    return report


def seed_adapter_initialization(seed: int) -> None:
    """Seed adapter creation separately from per-event diffusion noise."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lora_trainable_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Fail closed unless every trainable tensor is a PEFT LoRA A/B weight."""

    named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    unexpected = [name for name, _parameter in named if "lora_A." not in name and "lora_B." not in name]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable non-LoRA parameters: {unexpected[:8]}")
    if not named:
        raise RuntimeError("LoRA injection produced no trainable A/B parameters.")
    return [parameter for _name, parameter in named]


def assert_no_frozen_parameter_grads(
    modules: Mapping[str, nn.Module],
    *,
    fully_frozen: Iterable[str] = (),
) -> dict[str, dict[str, int]]:
    """Assert frozen weights have no gradient and fully frozen modules stay frozen."""

    fully_frozen_names = set(fully_frozen)
    report: dict[str, dict[str, int]] = {}
    for name, module in modules.items():
        parameters = list(module.parameters())
        trainable = [parameter for parameter in parameters if parameter.requires_grad]
        if name in fully_frozen_names and trainable:
            raise RuntimeError(f"{name} unexpectedly has {len(trainable)} trainable parameter tensors.")
        frozen = parameters if name in fully_frozen_names else [parameter for parameter in parameters if not parameter.requires_grad]
        with_grad = [parameter for parameter in frozen if parameter.grad is not None]
        nonfinite = sum(
            int((~torch.isfinite(parameter.grad.detach())).sum().item())
            for parameter in with_grad
        )
        report[name] = {
            "parameter_tensors": len(parameters),
            "trainable_parameter_tensors": len(trainable),
            "frozen_parameter_tensors": len(frozen),
            "frozen_tensors_with_grad": len(with_grad),
            "frozen_nonfinite_grad_elements": nonfinite,
        }
        if with_grad:
            raise RuntimeError(f"{name} accumulated gradients for {len(with_grad)} frozen parameter tensors.")
    return report


def emit_json_report(report: Mapping[str, Any], output_path: Path | None = None) -> None:
    payload = json.dumps(_jsonable(report), indent=2, ensure_ascii=False, sort_keys=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def validate_e2e_pair_reports(
    positive: Mapping[str, Any],
    detached: Mapping[str, Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Validate that two-event positive and detach runs form a strict pair."""

    if positive.get("events") != 2 or detached.get("events") != 2:
        raise ValueError("Both E2E pair members must contain exactly two events.")
    if positive.get("detach_between_events") is not False:
        raise ValueError("The positive report must have detach_between_events=false.")
    if detached.get("detach_between_events") is not True:
        raise ValueError("The negative-control report must have detach_between_events=true.")
    if positive.get("pair_id") != detached.get("pair_id"):
        raise ValueError("E2E reports have different pair_id values.")
    if positive.get("pair_metadata") != detached.get("pair_metadata"):
        raise ValueError("E2E reports differ in metadata other than the detach intervention.")

    positive_loss = float(positive["loss"])
    detached_loss = float(detached["loss"])
    loss_close = math.isclose(positive_loss, detached_loss, abs_tol=atol, rel_tol=rtol)
    if not loss_close:
        raise ValueError(
            f"Forward losses differ: positive={positive_loss}, detached={detached_loss}, atol={atol}, rtol={rtol}."
        )

    positive_intermediate = positive.get("intermediate_gradients", [])
    detached_intermediate = detached.get("intermediate_gradients", [])
    if len(positive_intermediate) != 1 or len(detached_intermediate) != 1:
        raise ValueError("A two-event pair must report exactly one intermediate state gradient.")
    positive_norm = positive_intermediate[0].get("norm")
    if positive_norm is None or not math.isfinite(float(positive_norm)) or float(positive_norm) <= 0:
        raise ValueError("The positive run lacks a finite, non-zero intermediate gradient.")
    if detached_intermediate[0].get("norm") is not None:
        raise ValueError("The detach control unexpectedly has an intermediate gradient.")

    for label, report in (("positive", positive), ("detached", detached)):
        for key in ("lora_grad_norm", "unclamped_image_grad_norm"):
            value = float(report[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"The {label} report has invalid {key}={value}.")

    return {
        "valid": True,
        "pair_id": positive.get("pair_id"),
        "positive_loss": positive_loss,
        "detached_loss": detached_loss,
        "absolute_loss_difference": abs(positive_loss - detached_loss),
        "atol": atol,
        "rtol": rtol,
        "positive_intermediate_grad_norm": float(positive_norm),
        "detached_intermediate_grad_norm": None,
    }
