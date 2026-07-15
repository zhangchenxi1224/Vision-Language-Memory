"""Stable seed derivation shared by probes and episode training."""

from __future__ import annotations

import hashlib

import torch


def event_seed(global_seed: int, episode_id: str, turn_id: str | int) -> int:
    payload = f"vlm-event-noise-v1\0{int(global_seed)}\0{episode_id}\0{turn_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def make_event_generator(
    *,
    device: torch.device | str,
    global_seed: int,
    episode_id: str,
    turn_id: str | int,
) -> torch.Generator:
    generator = torch.Generator(device=torch.device(device))
    generator.manual_seed(event_seed(global_seed, episode_id, turn_id))
    return generator
