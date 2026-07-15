"""Atomic LoRA-only training checkpoints with exact RNG/cursor recovery."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _trainable_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def save_training_checkpoint(
    path: str | Path,
    *,
    trainable_module: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    episode_cursor: int,
    optimizer_step: int,
    manifest: Mapping[str, Any],
    trainer_state: Mapping[str, Any] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "trainable_state": _trainable_state(trainable_module),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "episode_cursor": int(episode_cursor),
        "optimizer_step": int(optimizer_step),
        "rng_state": _rng_state(),
        "manifest": dict(manifest),
        "trainer_state": dict(trainer_state or {}),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    trainable_module: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported training checkpoint schema.")
    if expected_manifest is not None and payload.get("manifest") != dict(expected_manifest):
        raise ValueError("Checkpoint manifest does not match the current run.")

    trainable = {name: parameter for name, parameter in trainable_module.named_parameters() if parameter.requires_grad}
    saved = payload["trainable_state"]
    if set(saved) != set(trainable):
        missing = sorted(set(trainable) - set(saved))
        unexpected = sorted(set(saved) - set(trainable))
        raise ValueError(f"Trainable parameter mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(payload["optimizer"])
    _restore_rng_state(payload["rng_state"])
    return payload


def load_trainable_weights(path: str | Path, *, trainable_module: nn.Module) -> dict[str, Any]:
    """Load only the trainable tensors for inference/evaluation, without optimizer or RNG mutation."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported training checkpoint schema.")
    trainable = {name: parameter for name, parameter in trainable_module.named_parameters() if parameter.requires_grad}
    saved = payload["trainable_state"]
    if set(saved) != set(trainable):
        missing = sorted(set(trainable) - set(saved))
        unexpected = sorted(set(saved) - set(trainable))
        raise ValueError(f"Trainable parameter mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, parameter in trainable.items():
            parameter.copy_(saved[name].to(device=parameter.device, dtype=parameter.dtype))
    return payload
