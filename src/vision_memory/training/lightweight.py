"""Trainable static-image control used beside the recurrent lightweight updater."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StaticLearnedInitialImage(nn.Module):
    """One global learned RGB image; all event updates are intentionally identity maps."""

    def __init__(self, *, output_size: int = 256, initial_value: float = 0.5) -> None:
        super().__init__()
        if output_size < 1:
            raise ValueError("output_size must be positive.")
        if not 0.0 < initial_value < 1.0:
            raise ValueError("initial_value must be strictly between zero and one.")
        self.output_size = output_size
        logit = torch.logit(torch.tensor(initial_value, dtype=torch.float32))
        self.image_logits = nn.Parameter(torch.full((1, 3, output_size, output_size), float(logit)))

    def initial_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if batch_size != 1:
            raise ValueError("StaticLearnedInitialImage currently supports batch_size=1.")
        return self.image_logits.to(device=device, dtype=dtype)

    def update(self, state: Tensor, event_text: str) -> Tensor:
        if not isinstance(event_text, str) or not event_text.strip():
            raise ValueError("Static baseline still requires a valid routed event string for auditability.")
        return state

    def render(self, state: Tensor) -> Tensor:
        if state.shape != (1, 3, self.output_size, self.output_size):
            raise ValueError(
                f"Expected static image logits [1, 3, {self.output_size}, {self.output_size}], got {tuple(state.shape)}"
            )
        return torch.sigmoid(state)


__all__ = ["StaticLearnedInitialImage"]
