"""Oracle-routed episode execution without exposing hidden ledgers to models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from vision_memory.data.schema import reject_hidden_ledger


UpdateFn = Callable[[Tensor, str, str, str | int], Tensor]
DecodeFn = Callable[[Tensor], Tensor]
EncodeFn = Callable[[Tensor], Tensor]
ReaderLossFn = Callable[[Tensor, str, str], Any]
ChoiceReaderLossFn = Callable[[Tensor, str, tuple[str, ...], int], Any]
ChoiceViewFn = Callable[[str, str | int, tuple[str, ...], int], tuple[Sequence[str], int]]
StateSupervisionFn = Callable[[Tensor, str, str | int], Any]


@dataclass(frozen=True)
class _QueryPayload:
    query_text: str
    formatted_query: str
    target_text: str
    choices: tuple[str, ...] | None
    target_index: int | None


@dataclass(frozen=True)
class EpisodeLossOutput:
    loss: Tensor
    qa_loss: Tensor | None
    state_supervision_loss: Tensor | None
    latent_distill_loss: Tensor | None
    image_distill_loss: Tensor | None
    visual_feature_distill_loss: Tensor | None
    training_regime: str
    objective_stage: str
    final_state: Tensor
    states: tuple[Tensor, ...]
    state_ids: tuple[str, ...]
    query_count: int
    target_token_count: int
    route_trace: tuple[str, ...]
    updater_trace: tuple[str, ...]
    gradient_audit_states: tuple[Tensor, ...]
    gradient_audit_images: tuple[Tensor, ...]
    gradient_audit_features: tuple[Tensor, ...]

    @property
    def distill_loss(self) -> Tensor | None:
        return self.state_supervision_loss

    @property
    def total_loss(self) -> Tensor:
        return self.loss


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
        query_text=query,
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


def _loss_tensor(result: Any, *, source: str) -> Tensor:
    if isinstance(result, Tensor):
        return result
    loss = getattr(result, "loss", None)
    if not isinstance(loss, Tensor):
        raise TypeError(f"{source} must return a Tensor or an object with a Tensor .loss field.")
    return loss


def _optional_loss_component(result: Any, name: str) -> Tensor | None:
    value = getattr(result, name, None)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise TypeError(f"state_supervision_fn component {name!r} must be a Tensor when provided.")
    if value.numel() != 1:
        raise ValueError(f"state_supervision_fn component {name!r} must be scalar.")
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite state supervision component {name!r}.")
    return value


def _apply_choice_view(
    payload: _QueryPayload,
    *,
    choice_view_fn: ChoiceViewFn | None,
    episode_id: str,
    turn_id: str | int,
) -> _QueryPayload:
    if choice_view_fn is None:
        return payload
    if payload.choices is None or payload.target_index is None:
        raise ValueError("choice_view_fn requires ordered choices and a valid target_index.")
    transformed = choice_view_fn(episode_id, turn_id, payload.choices, payload.target_index)
    if not isinstance(transformed, tuple) or len(transformed) != 2:
        raise TypeError("choice_view_fn must return (choices, target_index).")
    raw_choices, target_index = transformed
    if not isinstance(raw_choices, Sequence) or isinstance(raw_choices, (str, bytes)):
        raise TypeError("choice_view_fn choices must be a sequence of strings.")
    choices = tuple(str(choice) for choice in raw_choices)
    if len(choices) != len(payload.choices) or Counter(choices) != Counter(payload.choices):
        raise ValueError("choice_view_fn must return a permutation of the original choices.")
    if isinstance(target_index, bool) or not isinstance(target_index, int) or not 0 <= target_index < len(choices):
        raise ValueError("choice_view_fn returned an invalid target_index.")
    if choices[target_index] != payload.target_text:
        raise ValueError("choice_view_fn changed the semantic target or failed to synchronize target_index.")
    return _QueryPayload(
        query_text=payload.query_text,
        formatted_query=format_mcq_query(payload.query_text, choices),
        target_text=payload.target_text,
        choices=choices,
        target_index=target_index,
    )


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
    choice_view_fn: ChoiceViewFn | None = None,
    training_regime: str = "qa_only",
    state_supervision_fn: StateSupervisionFn | None = None,
    objective_stage: str = "qa",
    audit_gradients: bool = False,
    require_mixed_delayed_probe: bool = False,
) -> EpisodeLossOutput:
    """Execute one oracle-routed episode and return token-normalized delayed query CE.

    Only explicitly selected event/query fields cross the model boundary. Metadata and the
    hidden ledger, if present in the record for auditing, are never forwarded.
    """

    if objective_stage not in {"qa", "distill"}:
        raise ValueError("R3 objective_stage must be exactly 'qa' or 'distill'; mixed objectives are forbidden.")
    if isinstance(episode, Mapping):
        reject_hidden_ledger(episode)
    if reader_loss_mode not in {"target-only", "listwise-choice"}:
        raise ValueError("reader_loss_mode must be 'target-only' or 'listwise-choice'.")
    if objective_stage == "distill":
        if reader_loss_fn is not None or choice_reader_loss_fn is not None or choice_view_fn is not None:
            raise ValueError("distill objective forbids Reader loss functions and choice views.")
    elif reader_loss_mode == "target-only":
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
    if training_regime not in {"qa_only", "teacher_assisted"}:
        raise ValueError("training_regime must be 'qa_only' or 'teacher_assisted'.")
    if training_regime == "qa_only" and state_supervision_fn is not None:
        raise ValueError("qa_only training forbids state_supervision_fn and teacher-derived losses.")
    if training_regime == "teacher_assisted" and objective_stage == "distill" and state_supervision_fn is None:
        raise ValueError("teacher_assisted distill objective requires state_supervision_fn.")
    if objective_stage == "qa" and state_supervision_fn is not None:
        raise ValueError("qa objective must unload teacher supervision before Reader fine-tuning.")
    if objective_stage == "distill" and training_regime != "teacher_assisted":
        raise ValueError("distill objective is restricted to the teacher_assisted lineage.")
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
    turn_kinds = [
        (
            turn.get("kind", turn.get("type"))
            if isinstance(turn, Mapping)
            else getattr(getattr(turn, "type", None), "value", getattr(turn, "type", None))
        )
        for turn in turns
    ]
    if require_mixed_delayed_probe:
        for index, kind in enumerate(turn_kinds):
            if kind != "mixed":
                continue
            mixed_payload = _query_payload(turns[index])
            found_matching_probe = False
            for later_index in range(index + 1, len(turns)):
                later_kind = turn_kinds[later_index]
                if later_kind in {"event", "mixed"}:
                    break
                if later_kind != "query":
                    continue
                candidate = _query_payload(turns[later_index])
                if (
                    candidate.target_text == mixed_payload.target_text
                    and candidate.choices is not None
                    and mixed_payload.choices is not None
                    and Counter(candidate.choices) == Counter(mixed_payload.choices)
                ):
                    found_matching_probe = True
                    break
            if not found_matching_probe:
                raise ValueError(
                    "formal R3 requires every mixed turn to have a same-target, same-choice-multiset "
                    "pure-query probe before the next updater"
                )

    state = initial_state
    states: list[Tensor] = []
    route_trace: list[str] = []
    updater_trace: list[str] = []
    query_losses: list[Tensor] = []
    state_supervision_losses: list[Tensor] = []
    latent_distill_losses: list[Tensor] = []
    image_distill_losses: list[Tensor] = []
    visual_feature_distill_losses: list[Tensor] = []
    state_ids: list[str] = []
    gradient_audit_states: list[Tensor] = []
    gradient_audit_images: list[Tensor] = []
    gradient_audit_features: list[Tensor] = []
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
                if audit_gradients:
                    if not state.requires_grad:
                        raise RuntimeError("Updated state does not require gradients under the R3 audit contract.")
                    state.retain_grad()
                    gradient_audit_states.append(state)
                route_trace.append(f"{turn_id}:update")
                updater_trace.append(f"{turn_id}:{event_kind or 'unknown'}:update")
                event_count += 1
                if state_supervision_fn is not None:
                    state_result = state_supervision_fn(state, episode_id, turn_id)
                    state_loss = _loss_tensor(state_result, source="state_supervision_fn")
                    if not torch.isfinite(state_loss):
                        raise RuntimeError(f"Non-finite state supervision loss at turn {turn_id}.")
                    state_supervision_losses.append(state_loss)
                    teacher_state_id = getattr(state_result, "teacher_state_id", None)
                    if teacher_state_id is not None:
                        if not isinstance(teacher_state_id, str) or len(teacher_state_id) != 64:
                            raise ValueError("state_supervision_fn teacher_state_id must be a SHA256 string.")
                        state_ids.append(teacher_state_id)
                    for name, destination in (
                        ("latent_raw", latent_distill_losses),
                        ("image_raw", image_distill_losses),
                        ("feature_raw", visual_feature_distill_losses),
                    ):
                        component = _optional_loss_component(state_result, name)
                        if component is not None:
                            destination.append(component)
                    if audit_gradients:
                        for name, destination in (
                            ("student_image", gradient_audit_images),
                            ("student_feature", gradient_audit_features),
                        ):
                            tensor = getattr(state_result, name, None)
                            if not isinstance(tensor, Tensor) or not tensor.requires_grad:
                                raise RuntimeError(
                                    f"state_supervision_fn must expose gradient-bearing {name} under audit."
                                )
                            tensor.retain_grad()
                            destination.append(tensor)

        if kind in {"query", "mixed"}:
            if objective_stage == "distill":
                route_trace.append(f"{turn_id}:read-skipped-distill")
                continue
            query_payload = _apply_choice_view(
                _query_payload(raw_turn),
                choice_view_fn=choice_view_fn,
                episode_id=episode_id,
                turn_id=turn_id,
            )
            image = decode_fn(state)
            if audit_gradients:
                if not image.requires_grad:
                    raise RuntimeError("Reader image does not require gradients under the R3 audit contract.")
                image.retain_grad()
                gradient_audit_images.append(image)
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

    qa_loss = torch.stack(query_losses).mean() if query_losses else None
    state_supervision_loss = torch.stack(state_supervision_losses).mean() if state_supervision_losses else None
    if objective_stage == "qa" and (qa_loss is None or target_token_count == 0):
        raise ValueError("QA episode contains no supervised query.")
    if objective_stage == "distill" and training_regime == "teacher_assisted" and state_supervision_loss is None:
        raise ValueError("Teacher distill episode contains no supervised updater state.")
    if objective_stage == "qa":
        assert qa_loss is not None
        total_loss = qa_loss
    elif objective_stage == "distill":
        assert state_supervision_loss is not None
        total_loss = state_supervision_loss

    def component_mean(values: list[Tensor]) -> Tensor | None:
        return torch.stack(values).mean() if values else None

    return EpisodeLossOutput(
        loss=total_loss,
        qa_loss=qa_loss,
        state_supervision_loss=state_supervision_loss,
        latent_distill_loss=component_mean(latent_distill_losses),
        image_distill_loss=component_mean(image_distill_losses),
        visual_feature_distill_loss=component_mean(visual_feature_distill_losses),
        training_regime=training_regime,
        objective_stage=objective_stage,
        final_state=state,
        states=tuple(states),
        state_ids=tuple(state_ids),
        query_count=query_count,
        target_token_count=target_token_count,
        route_trace=tuple(route_trace),
        updater_trace=tuple(updater_trace),
        gradient_audit_states=tuple(gradient_audit_states),
        gradient_audit_images=tuple(gradient_audit_images),
        gradient_audit_features=tuple(gradient_audit_features),
    )
