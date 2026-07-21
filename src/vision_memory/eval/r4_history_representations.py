"""Query-free history representations for the prospective R4 Qwen baselines.

The public rendering boundary intentionally accepts only a sequence of
``VisibleEvent`` objects.  Query text, choices, labels, episode identifiers,
and privileged state are therefore not representable inputs to this module.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from vision_memory.data.schema import EventKind, Turn


R4_HISTORY_REPRESENTATION_SCHEMA = "vlm.r4.history-representation.v1"
QWEN_R4_RAW_HISTORY = "qwen_r4_raw_history"
QWEN_R4_OPERATION_TAGGED_HISTORY = "qwen_r4_operation_tagged_history"
QWEN_R4_LAST_EFFECTIVE_EVENT = "qwen_r4_last_effective_event"
R4_HISTORY_METHODS = (
    QWEN_R4_RAW_HISTORY,
    QWEN_R4_OPERATION_TAGGED_HISTORY,
    QWEN_R4_LAST_EFFECTIVE_EVENT,
)
R4_HISTORY_TASK_INSTRUCTION = (
    "Apply the visible memory events in chronological order. A later effective update replaces "
    "the earlier preference; clearing removes the current preference; an unrelated or no-op "
    "event leaves it unchanged. Answer the current question from the final state, using only the "
    "memory below."
)
R4_EMPTY_MEMORY = "<unset>"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class VisibleEvent:
    """The complete and only input atom accepted by an R4 representation."""

    kind: EventKind
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EventKind):
            object.__setattr__(self, "kind", EventKind(str(self.kind)))
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("VisibleEvent.text must be a non-empty string.")
        if self.text != self.text.strip():
            raise ValueError("VisibleEvent.text must not contain boundary whitespace.")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "text": self.text}


@dataclass(frozen=True)
class HistoryRepresentation:
    """Rendered memory plus the audit fields required in prediction rows."""

    method: str
    memory_text: str
    representation_contract_sha256: str
    source_event_stream_sha256: str
    memory_text_sha256: str
    source_event_count: int
    retained_event_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_representation": self.method,
            "memory_text": self.memory_text,
            "representation_contract_sha256": self.representation_contract_sha256,
            "source_event_stream_sha256": self.source_event_stream_sha256,
            "memory_text_sha256": self.memory_text_sha256,
            "source_event_count": self.source_event_count,
            "retained_event_count": self.retained_event_count,
            "memory_utf8_bytes": len(self.memory_text.encode("utf-8")),
        }


def _checked_events(events: Sequence[VisibleEvent]) -> tuple[VisibleEvent, ...]:
    if isinstance(events, (str, bytes)):
        raise TypeError("events must be a sequence of VisibleEvent objects.")
    checked = tuple(events)
    if any(not isinstance(event, VisibleEvent) for event in checked):
        raise TypeError("Representations accept only VisibleEvent objects.")
    return checked


def source_event_stream_sha256(events: Sequence[VisibleEvent]) -> str:
    """Hash only visible event kind/text pairs, independent of episode paths."""

    checked = _checked_events(events)
    envelope = {
        "schema": "vlm.r4.visible-event-stream.v1",
        "events": [event.to_dict() for event in checked],
    }
    return _sha256_bytes(_canonical_bytes(envelope))


def _contract(method: str) -> dict[str, object]:
    if method not in R4_HISTORY_METHODS:
        raise ValueError(f"Unsupported R4 history method: {method!r}.")
    common: dict[str, object] = {
        "schema": R4_HISTORY_REPRESENTATION_SCHEMA,
        "method": method,
        "task_instruction": R4_HISTORY_TASK_INSTRUCTION,
        "empty_memory": R4_EMPTY_MEMORY,
        "audit_input_fields": ["events[].kind", "events[].text"],
        "forbidden_inputs": [
            "query",
            "choices",
            "target",
            "target_index",
            "episode_id",
            "entity_id",
            "template_id",
            "ledger",
            "teacher",
            "future_events",
        ],
    }
    if method == QWEN_R4_RAW_HISTORY:
        common["rendering"] = "numbered original event text"
        common["model_visible_input_fields"] = ["events[].text"]
        common["router_metadata_used_for_rendering"] = False
    elif method == QWEN_R4_OPERATION_TAGGED_HISTORY:
        common["rendering"] = "numbered [KIND] plus original event text"
        common["model_visible_input_fields"] = ["events[].kind", "events[].text"]
        common["router_metadata_used_for_rendering"] = True
    else:
        common["rendering"] = "last non-noop [KIND] plus original event text"
        common["reducer"] = "ignore noop; retain only final set/overwrite/clear; preserve order"
        common["scope"] = "one entity-slot scope per episode"
        common["model_visible_input_fields"] = ["last_effective.kind", "last_effective.text"]
        common["router_metadata_used_for_rendering"] = True
    return common


def representation_contract_sha256(method: str) -> str:
    return _sha256_bytes(_canonical_bytes(_contract(method)))


def last_effective_event(events: Sequence[VisibleEvent]) -> VisibleEvent | None:
    """Return the final state-changing event; no-op events are truly ignored."""

    checked = _checked_events(events)
    for event in reversed(checked):
        if event.kind is not EventKind.NOOP:
            return event
    return None


def _memory_body(method: str, events: tuple[VisibleEvent, ...]) -> tuple[str, int]:
    if method == QWEN_R4_RAW_HISTORY:
        lines = [f"{index}. {event.text}" for index, event in enumerate(events, start=1)]
        return ("\n".join(lines) if lines else R4_EMPTY_MEMORY, len(events))
    if method == QWEN_R4_OPERATION_TAGGED_HISTORY:
        lines = [
            f"{index}. [{event.kind.value.upper()}] {event.text}"
            for index, event in enumerate(events, start=1)
        ]
        return ("\n".join(lines) if lines else R4_EMPTY_MEMORY, len(events))
    if method == QWEN_R4_LAST_EFFECTIVE_EVENT:
        retained = last_effective_event(events)
        if retained is None:
            return R4_EMPTY_MEMORY, 0
        return f"[{retained.kind.value.upper()}] {retained.text}", 1
    raise ValueError(f"Unsupported R4 history method: {method!r}.")


def render_history_representation(
    method: str,
    events: Sequence[VisibleEvent],
) -> HistoryRepresentation:
    """Render one method from events only, with no answer-bearing inputs."""

    checked = _checked_events(events)
    body, retained_count = _memory_body(method, checked)
    memory_text = f"{R4_HISTORY_TASK_INSTRUCTION}\n\nVisible event memory:\n{body}"
    return HistoryRepresentation(
        method=method,
        memory_text=memory_text,
        representation_contract_sha256=representation_contract_sha256(method),
        source_event_stream_sha256=source_event_stream_sha256(checked),
        memory_text_sha256=_sha256_bytes(memory_text.encode("utf-8")),
        source_event_count=len(checked),
        retained_event_count=retained_count,
    )


def visible_event_streams_at_queries(turns: Sequence[Turn]) -> tuple[tuple[VisibleEvent, ...], ...]:
    """Extract causal event prefixes; a mixed event is written before its query."""

    visible: list[VisibleEvent] = []
    prefixes: list[tuple[VisibleEvent, ...]] = []
    for turn in turns:
        if turn.calls_updater:
            if turn.event_kind is None or turn.event_text is None:  # schema should make this impossible
                raise ValueError("Updater turn is missing its visible event fields.")
            visible.append(VisibleEvent(turn.event_kind, turn.event_text))
        if turn.calls_reader:
            prefixes.append(tuple(visible))
    return tuple(prefixes)


def reset_event_stream(events: Sequence[VisibleEvent]) -> tuple[VisibleEvent, ...]:
    _checked_events(events)
    return ()


def shuffle_event_stream(
    events: Sequence[VisibleEvent],
    permutation: Sequence[int],
) -> tuple[VisibleEvent, ...]:
    """Apply an explicit, auditable permutation without consulting metadata."""

    checked = _checked_events(events)
    order = tuple(int(index) for index in permutation)
    if sorted(order) != list(range(len(checked))):
        raise ValueError("permutation must contain every event index exactly once.")
    return tuple(checked[index] for index in order)


def state_swap_event_stream(
    events: Sequence[VisibleEvent],
    donor_events: Sequence[VisibleEvent],
) -> tuple[VisibleEvent, ...]:
    """Return the donor stream after validating both intervention operands."""

    _checked_events(events)
    return _checked_events(donor_events)


def visible_events(items: Iterable[tuple[EventKind, str]]) -> tuple[VisibleEvent, ...]:
    """Small adapter for callers that already hold query-free kind/text pairs."""

    return tuple(VisibleEvent(kind, text) for kind, text in items)
