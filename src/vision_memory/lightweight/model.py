"""Hashed text encoder plus FiLM-ConvGRU visual memory."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


_TOKEN_PATTERN = re.compile(r"[\w']+|[^\w\s]", flags=re.UNICODE)
_LOW_FREQUENCY_DCT_MODES = (
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 2),
    (2, 2),
    (3, 0),
    (0, 3),
    (3, 1),
    (1, 3),
    (3, 2),
    (2, 3),
    (4, 0),
    (0, 4),
)


def _fixed_spatial_basis(state_size: int, basis_count: int) -> Tensor:
    """Return deterministic, zero-mean, unit-RMS low-frequency DCT-II maps."""

    if state_size < 5:
        raise ValueError("state_size must be at least 5 for the fixed spatial basis")
    if basis_count < 1 or basis_count > len(_LOW_FREQUENCY_DCT_MODES):
        raise ValueError(f"basis_count must be in [1, {len(_LOW_FREQUENCY_DCT_MODES)}]")

    coordinates = torch.arange(state_size, dtype=torch.float32) + 0.5
    one_dimensional = torch.stack(
        [torch.cos(torch.pi * frequency * coordinates / state_size) for frequency in range(state_size)]
    )
    frequency_pairs = _LOW_FREQUENCY_DCT_MODES[:basis_count]
    basis = torch.stack(
        [
            one_dimensional[vertical].unsqueeze(1) * one_dimensional[horizontal].unsqueeze(0)
            for vertical, horizontal in frequency_pairs
        ]
    )
    basis = basis - basis.mean(dim=(-2, -1), keepdim=True)
    root_mean_square = basis.square().mean(dim=(-2, -1), keepdim=True).sqrt()
    if not torch.isfinite(root_mean_square).all() or (root_mean_square <= 1e-6).any():
        raise RuntimeError("Fixed coordinate/Fourier basis is degenerate")
    return basis / root_mean_square


def _stable_token_id(token: str, vocabulary_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8, person=b"vlm-event").digest()
    return 2 + int.from_bytes(digest, "little") % (vocabulary_size - 2)


class HashedBiGRUEncoder(nn.Module):
    """Dependency-free deterministic tokenization with a trainable BiGRU."""

    def __init__(
        self,
        *,
        vocabulary_size: int = 8_192,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        max_tokens: int = 64,
    ) -> None:
        super().__init__()
        if vocabulary_size < 4:
            raise ValueError("vocabulary_size must be at least 4")
        self.vocabulary_size = vocabulary_size
        self.max_tokens = max_tokens
        self.output_dim = hidden_dim * 2
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

    def tokenize(self, texts: Sequence[str], *, device: torch.device) -> tuple[Tensor, Tensor]:
        token_rows: list[list[int]] = []
        for text in texts:
            tokens = _TOKEN_PATTERN.findall(text.lower())[: self.max_tokens]
            ids = [_stable_token_id(token, self.vocabulary_size) for token in tokens] or [1]
            token_rows.append(ids)
        lengths = torch.tensor([len(row) for row in token_rows], dtype=torch.long)
        width = int(lengths.max().item())
        input_ids = torch.zeros(len(token_rows), width, dtype=torch.long, device=device)
        for index, row in enumerate(token_rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        return input_ids, lengths

    def forward(self, texts: str | Sequence[str]) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("texts must be non-empty")
        device = self.embedding.weight.device
        input_ids, lengths = self.tokenize(texts, device=device)
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return torch.cat([hidden[0], hidden[1]], dim=-1)


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, *, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        combined_channels = input_channels + hidden_channels
        self.gates = nn.Conv2d(combined_channels, hidden_channels * 2, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(combined_channels, hidden_channels, kernel_size, padding=padding)
        # Retain most of the previous state while leaving enough initial write
        # capacity for overwrite and mixed-event transitions.
        if self.gates.bias is not None:
            with torch.no_grad():
                self.gates.bias[hidden_channels:].fill_(-1.0)

    def forward(self, inputs: Tensor, hidden: Tensor) -> Tensor:
        if inputs.ndim != 4 or hidden.ndim != 4:
            raise ValueError("ConvGRU inputs and hidden state must be BCHW tensors")
        if inputs.shape[0] != hidden.shape[0] or inputs.shape[-2:] != hidden.shape[-2:]:
            raise ValueError("ConvGRU input/state batch and spatial dimensions must match")
        reset, update = self.gates(torch.cat([inputs, hidden], dim=1)).chunk(2, dim=1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update)
        candidate = torch.tanh(self.candidate(torch.cat([inputs, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class LightweightVisualUpdater(nn.Module):
    """64-channel 64x64 recurrent state rendered as a differentiable RGB image."""

    _SPATIAL_BASIS_COUNT = 16

    def __init__(
        self,
        *,
        state_channels: int = 64,
        state_size: int = 64,
        output_size: int = 1_024,
        vocabulary_size: int = 8_192,
        embedding_dim: int = 128,
        text_hidden_dim: int = 128,
        learned_initial_state: bool = False,
    ) -> None:
        super().__init__()
        if state_channels < 1 or output_size < 1:
            raise ValueError("state_channels and output_size must be positive")
        self.state_channels = state_channels
        self.state_size = state_size
        self.output_size = output_size
        self.event_encoder = HashedBiGRUEncoder(
            vocabulary_size=vocabulary_size,
            embedding_dim=embedding_dim,
            hidden_dim=text_hidden_dim,
        )
        event_dim = self.event_encoder.output_dim
        self.event_projection = nn.Linear(event_dim, state_channels)
        self.event_spatial_projection = nn.Linear(
            event_dim,
            state_channels * self._SPATIAL_BASIS_COUNT,
            bias=False,
        )
        nn.init.xavier_uniform_(self.event_spatial_projection.weight, gain=1.0)
        spatial_basis = _fixed_spatial_basis(state_size, self._SPATIAL_BASIS_COUNT)
        self.register_buffer("spatial_basis", spatial_basis, persistent=False)
        self.film = nn.Linear(event_dim, state_channels * 2)
        self.cell = ConvGRUCell(state_channels, state_channels)
        head_channels = max(16, state_channels // 2)
        self.rgb_head = nn.Sequential(
            nn.Conv2d(state_channels, head_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(head_channels, 3, 1),
            nn.Sigmoid(),
        )
        initial = torch.zeros(1, state_channels, state_size, state_size)
        if learned_initial_state:
            self.initial_hidden = nn.Parameter(initial)
        else:
            self.register_buffer("initial_hidden", initial, persistent=False)

    def initial_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        initial_hidden = (
            torch.tanh(self.initial_hidden) if isinstance(self.initial_hidden, nn.Parameter) else self.initial_hidden
        )
        return initial_hidden.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1).clone()

    def update(self, state: Tensor, event_text: str | Sequence[str]) -> Tensor:
        if state.ndim != 4 or state.shape[1:] != (self.state_channels, self.state_size, self.state_size):
            raise ValueError(
                f"Expected state [B, {self.state_channels}, {self.state_size}, {self.state_size}], "
                f"got {tuple(state.shape)}"
            )
        texts = [event_text] if isinstance(event_text, str) else list(event_text)
        if len(texts) != state.shape[0]:
            raise ValueError("One event string is required for every state in the batch")
        expected_basis_shape = (self._SPATIAL_BASIS_COUNT, self.state_size, self.state_size)
        if tuple(self.spatial_basis.shape) != expected_basis_shape:
            raise RuntimeError(
                f"Expected fixed spatial_basis shape {expected_basis_shape}, got {tuple(self.spatial_basis.shape)}"
            )
        features = self.event_encoder(texts).to(device=state.device, dtype=state.dtype)
        event_map = self.event_projection(features).unsqueeze(-1).unsqueeze(-1)
        spatial_coefficients = self.event_spatial_projection(features).reshape(
            state.shape[0], self.state_channels, self._SPATIAL_BASIS_COUNT
        )
        spatial_basis = self.spatial_basis.to(device=state.device, dtype=state.dtype)
        spatial_event_map = torch.einsum("bck,khw->bchw", spatial_coefficients, spatial_basis)
        event_map = torch.tanh(event_map + spatial_event_map / math.sqrt(self._SPATIAL_BASIS_COUNT))
        gamma, beta = self.film(features).chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1).unsqueeze(-1)
        # FiLM acts before the GRU and is bounded with the spatial writer. The cell
        # can therefore preserve its convex-combination hidden-state invariant.
        conditioned_event_map = torch.tanh((1.0 + 0.1 * torch.tanh(gamma)) * event_map + 0.1 * beta)
        return self.cell(conditioned_event_map, state)

    def render(self, state: Tensor) -> Tensor:
        image = self.rgb_head(state)
        if image.shape[-2:] != (self.output_size, self.output_size):
            image = F.interpolate(
                image,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return image

    def render_deterministic_repro(self, state: Tensor, *, target_size: int = 256) -> Tensor:
        """Render through a strict-determinism diagnostic path.

        The production renderer above deliberately remains unchanged.  Its CUDA
        bilinear backward is not supported by ``torch.use_deterministic_algorithms``.
        This diagnostic-only path instead expands each RGB-head pixel by an integer
        repeat and optionally takes a centered crop.  The reproducibility probe uses
        the production 64x64 state and 256x256 output settings without a crop.  Its
        256x256 result lies on the locked Qwen3-VL processor's 32-pixel spatial grid, so
        Qwen resizing can be disabled without changing the updater or RGB-head parameters.
        """

        if target_size <= 0:
            raise ValueError("target_size must be positive")
        image = self.rgb_head(state)
        height, width = image.shape[-2:]
        if height != width:
            raise ValueError("Deterministic reproducibility rendering requires a square RGB-head output.")
        if self.output_size % height != 0:
            raise ValueError(
                "Deterministic reproducibility rendering requires output_size to be an integer "
                f"multiple of the RGB-head size; got output_size={self.output_size}, head_size={height}."
            )
        repeat_factor = self.output_size // height
        expanded = image.repeat_interleave(repeat_factor, dim=-2).repeat_interleave(repeat_factor, dim=-1)
        crop = self.output_size - target_size
        if crop < 0 or crop % 2:
            raise ValueError(
                "target_size must not exceed output_size and must leave an even centered crop; "
                f"got target_size={target_size}, output_size={self.output_size}."
            )
        offset = crop // 2
        return expanded[..., offset : offset + target_size, offset : offset + target_size]

    def forward(self, state: Tensor, event_text: str | Sequence[str]) -> Tensor:
        return self.update(state, event_text)
