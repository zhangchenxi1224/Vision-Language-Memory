"""Small fixed differentiable Reader for local episode/BPTT smoke tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class HashChoiceReader(nn.Module):
    """Score choices against low-resolution RGB codes without trainable parameters.

    This is an interface/optimization surrogate, not a scientific baseline and not a
    substitute for frozen Qwen. It deliberately never receives the target index.
    """

    def __init__(self, *, feature_size: int = 8, logit_scale: float = 8.0) -> None:
        super().__init__()
        if feature_size < 1:
            raise ValueError("feature_size must be positive")
        self.feature_size = feature_size
        self.logit_scale = logit_scale

    def _choice_code(self, choice: str, dimension: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        raw = bytearray()
        block = 0
        while len(raw) * 8 < dimension:
            raw.extend(
                hashlib.blake2b(
                    choice.lower().encode("utf-8") + block.to_bytes(4, "little"),
                    digest_size=32,
                    person=b"vlm-choice",
                ).digest()
            )
            block += 1
        values = [1.0 if raw[index // 8] & (1 << (index % 8)) else -1.0 for index in range(dimension)]
        return torch.tensor(values, device=device, dtype=dtype)

    def forward(self, *, image: Tensor, query: str, choices: Sequence[str]) -> Tensor:
        del query  # Qwen replacements may use it; this fixed surrogate only tests state coding.
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError(f"Expected image [1, 3, H, W], got {tuple(image.shape)}")
        if len(choices) != 4:
            raise ValueError("Exactly four choices are required")
        features = F.adaptive_avg_pool2d(image, (self.feature_size, self.feature_size)).flatten(1)
        features = F.normalize(features - 0.5, dim=-1)
        codes = torch.stack(
            [
                self._choice_code(
                    choice,
                    features.shape[-1],
                    device=features.device,
                    dtype=features.dtype,
                )
                for choice in choices
            ]
        )
        codes = F.normalize(codes, dim=-1)
        return self.logit_scale * features @ codes.transpose(0, 1)
