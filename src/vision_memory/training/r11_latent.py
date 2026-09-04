"""Model-free constants for the R11 VAE-latent reachability oracle."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


R11_PROTOCOL = "R11-VAE-Latent-Reachability"
R11_OPTIMIZER_STEPS = 256
R11_CHECKPOINT_STEPS = (0, 64, 128, 192, 256)
R11_LEARNING_RATE = 0.05


def training_view_counts(rows: Sequence[Mapping[str, object]]) -> dict[int, int]:
    values = [row for row in rows if row.get("kind") == "optimizer_step"]
    return dict(
        sorted(Counter(int(row["forward_cyclic_training_view"]) for row in values).items())
    )


__all__ = [
    "R11_CHECKPOINT_STEPS",
    "R11_LEARNING_RATE",
    "R11_OPTIMIZER_STEPS",
    "R11_PROTOCOL",
    "training_view_counts",
]
