"""Oracle-routed episode execution independent of any concrete Reader model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .schema import Episode


class EpisodeUpdater(Protocol):
    def initial_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor: ...

    def update(self, state: Tensor, event_text: str) -> Tensor: ...

    def render(self, state: Tensor) -> Tensor: ...


class ChoiceReader(Protocol):
    def __call__(self, *, image: Tensor, query: str, choices: Sequence[str]) -> Tensor:
        """Return four differentiable choice logits without receiving the target label."""
        ...


@dataclass(frozen=True)
class ReaderOutput:
    logits: Tensor
    target_index: int
    target_token_count: int
    loss: Tensor


@dataclass(frozen=True)
class EpisodeRunOutput:
    loss: Tensor
    final_state: Tensor
    final_image: Tensor
    reader_outputs: tuple[ReaderOutput, ...]
    update_count: int
    query_count: int


def _apply_recurrence(updater: EpisodeUpdater, state: Tensor, mode: str) -> Tensor:
    if mode == "direct":
        return state
    if mode == "decode_reencode":
        transform = getattr(updater, "decode_reencode", None)
        if transform is None:
            raise ValueError("recurrence_mode='decode_reencode' requires updater.decode_reencode(state)")
        transformed = transform(state)
        if not isinstance(transformed, Tensor):
            raise TypeError("updater.decode_reencode must return a Tensor")
        return transformed
    raise ValueError("recurrence_mode must be 'direct' or 'decode_reencode'")


def run_episode(
    episode: Episode,
    *,
    updater: EpisodeUpdater,
    reader: ChoiceReader,
    recurrence_mode: str = "direct",
    detach_between_events: bool = False,
    initial_state: Tensor | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EpisodeRunOutput:
    """Execute event/query/mixed turns with an oracle router.

    Event labels and target indices are supervision metadata: only event_text reaches the
    updater, and only query text/choices reach the Reader. In a mixed turn, update happens
    before rendering and reading. Pure query turns never invoke the updater.
    """

    device = torch.device(device)
    state = initial_state
    if state is None:
        state = updater.initial_state(batch_size=1, device=device, dtype=dtype)
    if not isinstance(state, Tensor):
        raise TypeError("initial_state must be a Tensor")

    reader_outputs: list[ReaderOutput] = []
    update_count = 0
    for turn in episode.turns:
        if turn.calls_updater:
            if turn.event_text is None:
                raise RuntimeError("Schema invariant violated: updater turn has no event_text")
            if update_count:
                state = _apply_recurrence(updater, state, recurrence_mode)
                if detach_between_events:
                    state = state.detach()
            state = updater.update(state, turn.event_text)
            update_count += 1

        if turn.calls_reader:
            if turn.query is None:
                raise RuntimeError("Schema invariant violated: reader turn has no query")
            image = updater.render(state)
            logits = reader(image=image, query=turn.query.text, choices=turn.query.choices)
            if not isinstance(logits, Tensor):
                raise TypeError("Reader must return a Tensor")
            if logits.ndim == 1:
                logits = logits.unsqueeze(0)
            if logits.shape != (1, 4):
                raise ValueError(f"Reader must return logits with shape [1, 4], got {tuple(logits.shape)}")
            target = torch.tensor([turn.query.target_index], device=logits.device, dtype=torch.long)
            token_normalized_loss = F.cross_entropy(logits.float(), target) / turn.query.target_token_count
            reader_outputs.append(
                ReaderOutput(
                    logits=logits,
                    target_index=turn.query.target_index,
                    target_token_count=turn.query.target_token_count,
                    loss=token_normalized_loss,
                )
            )

    if not reader_outputs:
        raise ValueError(f"Episode {episode.episode_id} contains no query")
    # Each query is first normalized by its own target-token count, then all probes are
    # weighted equally. This avoids longer target strings or query-rich episodes dominating.
    loss = torch.stack([output.loss for output in reader_outputs]).mean()
    final_image = updater.render(state)
    return EpisodeRunOutput(
        loss=loss,
        final_state=state,
        final_image=final_image,
        reader_outputs=tuple(reader_outputs),
        update_count=update_count,
        query_count=len(reader_outputs),
    )
