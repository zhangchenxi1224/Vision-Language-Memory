"""Auditable whole-episode curriculum selection for state-transition training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CurriculumSelection:
    curriculum: str
    input_count: int
    selected_count: int
    excluded_count: int
    selected_episode_ids_sha256: str
    excluded_by_reason: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum": self.curriculum,
            "input_count": self.input_count,
            "selected_count": self.selected_count,
            "excluded_count": self.excluded_count,
            "selected_episode_ids_sha256": self.selected_episode_ids_sha256,
            "excluded_by_reason": dict(self.excluded_by_reason),
        }


def _episode_id(episode: Mapping[str, Any] | Any) -> str:
    value = episode.get("episode_id") if isinstance(episode, Mapping) else getattr(episode, "episode_id", None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every curriculum candidate requires a non-empty episode_id.")
    return value


def _event_kinds(episode: Mapping[str, Any] | Any) -> tuple[str, ...]:
    turns = episode.get("turns") if isinstance(episode, Mapping) else getattr(episode, "turns", None)
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise ValueError(f"Episode {_episode_id(episode)!r} has no valid turns sequence.")
    result: list[str] = []
    for index, turn in enumerate(turns):
        if isinstance(turn, Mapping):
            kind = turn.get("kind", turn.get("type"))
            event_kind = turn.get("event_kind", turn.get("transition"))
        else:
            kind = getattr(getattr(turn, "type", None), "value", getattr(turn, "type", None))
            event_kind = getattr(turn, "event_kind", None)
        kind = getattr(kind, "value", kind)
        if kind not in {"event", "mixed"}:
            continue
        event_kind = getattr(event_kind, "value", event_kind)
        if event_kind is None or not str(event_kind).strip():
            raise ValueError(
                f"Episode {_episode_id(episode)!r} turn {index} lacks an event kind; "
                "curriculum selection must not infer a label."
            )
        normalized = str(event_kind).strip().lower()
        if normalized not in {"set", "overwrite", "clear", "noop"}:
            raise ValueError(f"Unsupported event kind {normalized!r} in episode {_episode_id(episode)!r}.")
        result.append(normalized)
    return tuple(result)


def select_curriculum_episodes(
    episodes: Sequence[Mapping[str, Any] | Any],
    *,
    curriculum: str,
) -> tuple[list[Mapping[str, Any] | Any], CurriculumSelection]:
    """Select whole episodes without deleting or relabeling individual turns.

    ``set-only`` admits episodes whose state-changing events are all ``set``. ``noop``
    distractors remain eligible and retain their original label, so no target ledger can
    become inconsistent through turn-level surgery.
    """

    if curriculum not in {"full", "set-only"}:
        raise ValueError("curriculum must be 'full' or 'set-only'.")
    selected: list[Mapping[str, Any] | Any] = []
    excluded: dict[str, int] = {}
    for episode in episodes:
        event_kinds = _event_kinds(episode)
        reason: str | None = None
        if curriculum == "set-only":
            state_changes = {kind for kind in event_kinds if kind != "noop"}
            if not state_changes:
                reason = "no_set_event"
            elif state_changes != {"set"}:
                reason = "contains_non_set_transition"
        if reason is None:
            selected.append(episode)
        else:
            excluded[reason] = excluded.get(reason, 0) + 1

    ids = "\n".join(_episode_id(episode) for episode in selected).encode("utf-8")
    audit = CurriculumSelection(
        curriculum=curriculum,
        input_count=len(episodes),
        selected_count=len(selected),
        excluded_count=len(episodes) - len(selected),
        selected_episode_ids_sha256=hashlib.sha256(ids).hexdigest(),
        excluded_by_reason=excluded,
    )
    return selected, audit
