"""Oracle-routed episode execution without exposing hidden ledgers to models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor


UpdateFn = Callable[[Tensor, str, str, str | int], Tensor]
DecodeFn = Callable[[Tensor], Tensor]
EncodeFn = Callable[[Tensor], Tensor]
ReaderLossFn = Callable[[Tensor, str, str], Any]


@dataclass(frozen=True)
class EpisodeLossOutput:
    loss: Tensor
    final_state: Tensor
    states: tuple[Tensor, ...]
    query_count: int
    target_token_count: int
    route_trace: tuple[str, ...]


def format_mcq_query(query: str, choices: Sequence[str] | None) -> str:
    if not choices:
        return query
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(choices) > len(labels):
        raise ValueError("MCQ formatter supports at most 26 choices.")
    rendered = "\n".join(f"{labels[index]}. {choice}" for index, choice in enumerate(choices))
    return f"{query}\n{rendered}\nAnswer with the option text only."


def _required_text(turn: Mapping[str, Any] | Any, key: str) -> str:
    value = turn.get(key) if isinstance(turn, Mapping) else getattr(turn, key, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Turn requires a non-empty {key!r} string.")
    return value.strip()


def _query_payload(turn: Mapping[str, Any] | Any) -> tuple[str, str]:
    nested = turn.get("query") if isinstance(turn, Mapping) else getattr(turn, "query", None)
    if nested is not None:
        query = _required_text(nested, "text")
        raw_choices = nested.get("choices") if isinstance(nested, Mapping) else getattr(nested, "choices", None)
        raw_target_index = (
            nested.get("target_index") if isinstance(nested, Mapping) else getattr(nested, "target_index", None)
        )
        raw_target_text = nested.get("target_text") if isinstance(nested, Mapping) else None
    else:
        query = _required_text(turn, "query_text")
        raw_choices = turn.get("choices") if isinstance(turn, Mapping) else getattr(turn, "choices", None)
        raw_target_index = (
            turn.get("target_index") if isinstance(turn, Mapping) else getattr(turn, "target_index", None)
        )
        raw_target_text = turn.get("target_text") if isinstance(turn, Mapping) else None
    choices: list[str] | None = None
    if raw_choices is not None:
        if not isinstance(raw_choices, Sequence) or isinstance(raw_choices, (str, bytes)):
            raise ValueError("choices must be a sequence of strings.")
        choices = [str(item) for item in raw_choices]

    if raw_target_text is not None:
        target = str(raw_target_text).strip()
    else:
        target_index = raw_target_index
        if choices is None or not isinstance(target_index, int) or not 0 <= target_index < len(choices):
            raise ValueError("Query requires target_text or a valid target_index into choices.")
        target = choices[target_index]
    return format_mcq_query(query, choices), target


def _loss_and_tokens(result: Any) -> tuple[Tensor, int]:
    if isinstance(result, Tensor):
        return result, 1
    loss = getattr(result, "loss", None)
    if not isinstance(loss, Tensor):
        raise TypeError("Reader loss callable must return a Tensor or an object with a Tensor .loss field.")
    target_ids = getattr(result, "target_ids", None)
    token_count = int(target_ids.numel()) if isinstance(target_ids, Tensor) else 1
    if token_count <= 0:
        raise ValueError("Reader returned an empty target token sequence.")
    return loss, token_count


def run_episode(
    *,
    episode: Mapping[str, Any] | Any,
    initial_state: Tensor,
    update_fn: UpdateFn,
    decode_fn: DecodeFn,
    reader_loss_fn: ReaderLossFn,
    recurrence_mode: str = "direct_latent",
    reencode_fn: EncodeFn | None = None,
    reencode_decode_fn: DecodeFn | None = None,
    detach_between_events: bool = False,
    collect_states: bool = True,
) -> EpisodeLossOutput:
    """Execute one oracle-routed episode and return token-normalized delayed query CE.

    Only explicitly selected event/query fields cross the model boundary. Metadata and the
    hidden ledger, if present in the record for auditing, are never forwarded.
    """

    if recurrence_mode not in {"direct_latent", "decode_reencode"}:
        raise ValueError(f"Unsupported recurrence_mode: {recurrence_mode}")
    if recurrence_mode == "decode_reencode" and reencode_fn is None:
        raise ValueError("decode_reencode recurrence requires reencode_fn.")

    episode_id = str(episode.get("episode_id", "") if isinstance(episode, Mapping) else getattr(episode, "episode_id", ""))
    if not episode_id:
        raise ValueError("episode_id is required.")
    turns = episode.get("turns") if isinstance(episode, Mapping) else getattr(episode, "turns", None)
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)) or not turns:
        raise ValueError("Episode requires a non-empty turns sequence.")

    state = initial_state
    states: list[Tensor] = []
    route_trace: list[str] = []
    weighted_losses: list[Tensor] = []
    target_token_count = 0
    query_count = 0
    event_count = 0

    for index, raw_turn in enumerate(turns):
        if isinstance(raw_turn, Mapping):
            kind = raw_turn.get("kind", raw_turn.get("type"))
            turn_id = raw_turn.get("turn_id", index)
        else:
            kind_value = getattr(raw_turn, "type", None)
            kind = getattr(kind_value, "value", kind_value)
            turn_id = getattr(raw_turn, "turn_id", index)
        if kind not in {"event", "query", "mixed"}:
            raise ValueError(f"Turn {index} has unsupported kind {kind!r}.")

        if kind in {"event", "mixed"}:
            event_text = _required_text(raw_turn, "event_text")
            source = state.detach() if detach_between_events and event_count > 0 else state
            state = update_fn(source, event_text, episode_id, turn_id)
            if recurrence_mode == "decode_reencode":
                assert reencode_fn is not None
                bottleneck_decode = reencode_decode_fn or decode_fn
                state = reencode_fn(bottleneck_decode(state))
            if collect_states:
                states.append(state)
            route_trace.append(f"{turn_id}:update")
            event_count += 1

        if kind in {"query", "mixed"}:
            query, target = _query_payload(raw_turn)
            image = decode_fn(state)
            result = reader_loss_fn(image, query, target)
            loss, token_count = _loss_and_tokens(result)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite reader loss at turn {turn_id}.")
            weighted_losses.append(loss * token_count)
            target_token_count += token_count
            query_count += 1
            route_trace.append(f"{turn_id}:read")

    if not weighted_losses or target_token_count == 0:
        raise ValueError("Episode contains no supervised query.")
    total_loss = torch.stack(weighted_losses).sum() / target_token_count
    return EpisodeLossOutput(
        loss=total_loss,
        final_state=state,
        states=tuple(states),
        query_count=query_count,
        target_token_count=target_token_count,
        route_trace=tuple(route_trace),
    )
