"""Frozen Reader helpers that preserve gradients with respect to image inputs."""

from .qwen3vl import ReaderLossOutput, qwen3vl_target_only_ce

__all__ = ["ReaderLossOutput", "qwen3vl_target_only_ce"]

