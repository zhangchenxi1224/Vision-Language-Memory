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
ChoiceReaderLossFn = Callable[[Tensor, str, tuple[str, ...], int], Any]


@dataclass(frozen=True)
class _QueryPayload:
    formatted_query: str
    target_text: str
    choices: tuple[str, ...] | None
    target_index: int | None


@dataclass(frozen=True)
class EpisodeLossOutput:
    loss: Tensor
    final_state: Tensor
    states: tuple[Tensor, ...]
    query_count: int
    target_token_count: int
    route_trace: tuple[str, ...]
    updater_trace: tuple[str, ...]


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


def _event_kind(turn: Mapping[str, Any] | Any) -> str | None:
    if isinstance(turn, Mapping):
        value = turn.get("event_kind", turn.get("transition"))
    else:
        value = getattr(turn, "event_kind", None)
    value = getattr(value, "value", value)
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _query_payload(turn: Mapping[str, Any] | Any) -> _QueryPayload:
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
    choices: tuple[str, ...] | None = None
    if raw_choices is not None:
        if not isinstance(raw_choices, Sequence) or isinstance(raw_choices, (str, bytes)):
            raise ValueError("choices must be a sequence of strings.")
        choices = tuple(str(item) for item in raw_choices)

    target_index = (
        raw_target_index if isinstance(raw_target_index, int) and not isinstance(raw_target_index, bool) else None
    )
    if raw_target_text is not None:
        target = str(raw_target_text).strip()
    else:
        if choices is None or not isinstance(target_index, int) or not 0 <= target_index < len(choices):
            raise ValueError("Query requires target_text or a valid target_index into choices.")
        target = choices[target_index]
    return _QueryPayload(
        formatted_query=format_mcq_query(query, choices),
        target_text=target,
        choices=choices,
        target_index=target_index,
    )


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
    reader_loss_fn: ReaderLossFn | None = None,
    reader_loss_mode: str = "target-only",
    choice_reader_loss_fn: ChoiceReaderLossFn | None = None,
    recurrence_mode: str = "direct_latent",
    reencode_fn: EncodeFn | None = None,
    reencode_decode_fn: DecodeFn | None = None,
    detach_between_events: bool = False,
    collect_states: bool = True,
    noop_policy: str = "update",
) -> EpisodeLossOutput:
    """Execute one oracle-routed episode and return token-normalized delayed query CE.

    Only explicitly selected event/query fields cross the model boundary. Metadata and the
    hidden ledger, if present in the record for auditing, are never forwarded.
    """

    if reader_loss_mode not in {"target-only", "listwise-choice"}:
        raise ValueError("reader_loss_mode must be 'target-only' or 'listwise-choice'.")
    if reader_loss_mode == "target-only":
        if reader_loss_fn is None:
            raise ValueError("target-only reader loss mode requires reader_loss_fn.")
        if choice_reader_loss_fn is not None:
            raise ValueError("target-only reader loss mode does not accept choice_reader_loss_fn.")
    else:
        if choice_reader_loss_fn is None:
            raise ValueError("listwise-choice reader loss mode requires choice_reader_loss_fn.")
        if reader_loss_fn is not None:
            raise ValueError("listwise-choice reader loss mode does not accept reader_loss_fn.")
    if recurrence_mode not in {"direct_latent", "decode_reencode"}:
        raise ValueError(f"Unsupported recurrence_mode: {recurrence_mode}")
    if noop_policy not in {"update", "skip"}:
        raise ValueError("noop_policy must be 'update' or 'skip'.")
    if recurrence_mode == "decode_reencode" and reencode_fn is None:
        raise ValueError("decode_reencode recurrence requires reencode_fn.")

    episode_id = str(
        episode.get("episode_id", "") if isinstance(episode, Mapping) else getattr(episode, "episode_id", "")
    )
    if not episode_id:
        raise ValueError("episode_id is required.")
    turns = episode.get("turns") if isinstance(episode, Mapping) else getattr(episode, "turns", None)
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)) or not turns:
        raise ValueError("Episode requires a non-empty turns sequence.")

    state = initial_state
    states: list[Tensor] = []
    route_trace: list[str] = []
    updater_trace: list[str] = []
    query_losses: list[Tensor] = []
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
            event_kind = _event_kind(raw_turn)
            if noop_policy == "skip" and event_kind is None:
                raise ValueError(
                    f"Turn {turn_id} is missing event_kind/transition; skip-noop routing must fail closed."
                )
            if noop_policy == "skip" and event_kind == "noop":
                route_trace.append(f"{turn_id}:skip-noop")
                updater_trace.append(f"{turn_id}:noop:skip")
            else:
                source = state.detach() if detach_between_events and event_count > 0 else state
                state = update_fn(source, event_text, episode_id, turn_id)
                if recurrence_mode == "decode_reencode":
                    assert reencode_fn is not None
                    bottleneck_decode = reencode_decode_fn or decode_fn
                    state = reencode_fn(bottleneck_decode(state))
                if collect_states:
                    states.append(state)
                route_trace.append(f"{turn_id}:update")
                updater_trace.append(f"{turn_id}:{event_kind or 'unknown'}:update")
                event_count += 1

        if kind in {"query", "mixed"}:
            query_payload = _query_payload(raw_turn)
            image = decode_fn(state)
            if reader_loss_mode == "target-only":
                assert reader_loss_fn is not None
                result = reader_loss_fn(image, query_payload.formatted_query, query_payload.target_text)
            else:
                assert choice_reader_loss_fn is not None
                if query_payload.choices is None or query_payload.target_index is None:
                    raise ValueError(
                        "listwise-choice reader loss requires ordered choices and an explicit target_index."
                    )
                if not 0 <= query_payload.target_index < len(query_payload.choices):
                    raise ValueError("listwise-choice target_index is outside the ordered choices.")
                if query_payload.target_text != query_payload.choices[query_payload.target_index]:
                    raise ValueError("listwise-choice target_text and target_index are inconsistent.")
                result = choice_reader_loss_fn(
                    image,
                    query_payload.formatted_query,
                    query_payload.choices,
                    query_payload.target_index,
                )
            loss, token_count = _loss_and_tokens(result)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite reader loss at turn {turn_id}.")
            # qwen3vl_target_only_ce already averages CE over this query's target
            # tokens. Average those per-query means here so episodes with several
            # reads do not overweight either long answer strings or extra queries.
            query_losses.append(loss)
            target_token_count += token_count
            query_count += 1
            route_trace.append(f"{turn_id}:read")

    if not query_losses or target_token_count == 0:
        raise ValueError("Episode contains no supervised query.")
    total_loss = torch.stack(query_losses).mean()
    return EpisodeLossOutput(
        loss=total_loss,
        final_state=state,
        states=tuple(states),
        query_count=query_count,
        target_token_count=target_token_count,
        route_trace=tuple(route_trace),
        updater_trace=tuple(updater_trace),
    )
