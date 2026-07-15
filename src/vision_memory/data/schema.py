"""Strict JSONL schema for oracle-routed stateful-memory episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class TurnType(str, Enum):
    EVENT = "event"
    QUERY = "query"
    MIXED = "mixed"


class EventKind(str, Enum):
    SET = "set"
    OVERWRITE = "overwrite"
    CLEAR = "clear"
    NOOP = "noop"


FORBIDDEN_LEDGER_KEYS = frozenset(
    {
        "hidden_ledger",
        "preference_ledger",
        "oracle_preference",
        "oracle_ledger",
        "ledger",
    }
)


def _reject_hidden_ledger(value: Any, *, path: str = "episode") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_LEDGER_KEYS or normalized.startswith("hidden_ledger"):
                raise ValueError(f"Hidden ledger field is forbidden at {path}.{key}")
            _reject_hidden_ledger(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_hidden_ledger(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class QuerySpec:
    text: str
    choices: tuple[str, str, str, str]
    target_index: int
    target_token_count: int = 1

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("query.text must be non-empty")
        if len(self.choices) != 4 or any(not choice.strip() for choice in self.choices):
            raise ValueError("query.choices must contain exactly four non-empty strings")
        if len(set(self.choices)) != 4:
            raise ValueError("query.choices must be distinct")
        if not 0 <= self.target_index < 4:
            raise ValueError("query.target_index must be in [0, 3]")
        if self.target_token_count < 1:
            raise ValueError("query.target_token_count must be positive")

    @property
    def target(self) -> str:
        return self.choices[self.target_index]

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "choices": list(self.choices),
            "target_index": self.target_index,
            "target_token_count": self.target_token_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "QuerySpec":
        allowed = {"text", "choices", "target_index", "target_token_count"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown query fields: {sorted(unknown)}")
        required = {"text", "choices", "target_index"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"Missing query fields: {sorted(missing)}")
        choices = tuple(str(item) for item in value["choices"])
        if len(choices) != 4:
            raise ValueError("query.choices must contain exactly four entries")
        return cls(
            text=str(value["text"]),
            choices=choices,  # type: ignore[arg-type]
            target_index=int(value["target_index"]),
            target_token_count=int(value.get("target_token_count", 1)),
        )


@dataclass(frozen=True)
class Turn:
    type: TurnType
    event_kind: EventKind | None = None
    event_text: str | None = None
    query: QuerySpec | None = None

    def __post_init__(self) -> None:
        has_event = self.event_kind is not None or self.event_text is not None
        if self.type is TurnType.EVENT:
            if self.event_kind is None or not (self.event_text or "").strip() or self.query is not None:
                raise ValueError("event turn requires event_kind/event_text and forbids query")
        elif self.type is TurnType.QUERY:
            if has_event or self.query is None:
                raise ValueError("query turn requires query and forbids all event fields")
        elif self.type is TurnType.MIXED:
            if self.event_kind is None or not (self.event_text or "").strip() or self.query is None:
                raise ValueError("mixed turn requires event_kind, event_text, and query")
        else:  # pragma: no cover - Enum construction prevents this
            raise ValueError(f"Unsupported turn type: {self.type}")

    @property
    def calls_updater(self) -> bool:
        return self.type in {TurnType.EVENT, TurnType.MIXED}

    @property
    def calls_reader(self) -> bool:
        return self.type in {TurnType.QUERY, TurnType.MIXED}

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"type": self.type.value}
        if self.calls_updater:
            value["event_kind"] = self.event_kind.value if self.event_kind else None
            value["event_text"] = self.event_text
        if self.calls_reader:
            value["query"] = self.query.to_dict() if self.query else None
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Turn":
        allowed = {"type", "event_kind", "event_text", "query"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown turn fields: {sorted(unknown)}")
        if "type" not in value:
            raise ValueError("Turn is missing type")
        turn_type = TurnType(str(value["type"]))
        event_kind = EventKind(str(value["event_kind"])) if value.get("event_kind") is not None else None
        query = QuerySpec.from_dict(value["query"]) if value.get("query") is not None else None
        return cls(
            type=turn_type,
            event_kind=event_kind,
            event_text=str(value["event_text"]) if value.get("event_text") is not None else None,
            query=query,
        )


@dataclass(frozen=True)
class Episode:
    episode_id: str
    split: str
    seed: int
    entity_id: str
    template_id: str
    turns: tuple[Turn, ...]
    pair_id: str
    counterfactual_episode_id: str
    topic: str
    ood_group: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "episode_id",
            "split",
            "entity_id",
            "template_id",
            "pair_id",
            "counterfactual_episode_id",
            "topic",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.counterfactual_episode_id == self.episode_id:
            raise ValueError("counterfactual_episode_id must refer to another episode")
        if not 4 <= len(self.turns) <= 16:
            raise ValueError("episodes must contain between 4 and 16 turns")
        if not any(turn.calls_reader for turn in self.turns):
            raise ValueError("episode must contain at least one query")

    @property
    def update_count(self) -> int:
        return sum(turn.calls_updater for turn in self.turns)

    @property
    def query_count(self) -> int:
        return sum(turn.calls_reader for turn in self.turns)

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "episode_id": self.episode_id,
            "split": self.split,
            "seed": self.seed,
            "entity_id": self.entity_id,
            "template_id": self.template_id,
            "pair_id": self.pair_id,
            "counterfactual_episode_id": self.counterfactual_episode_id,
            "topic": self.topic,
            "turns": [turn.to_dict() for turn in self.turns],
        }
        if self.ood_group is not None:
            value["ood_group"] = self.ood_group
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Episode":
        _reject_hidden_ledger(value)
        allowed = {
            "episode_id",
            "split",
            "seed",
            "entity_id",
            "template_id",
            "pair_id",
            "counterfactual_episode_id",
            "topic",
            "turns",
            "ood_group",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Unknown episode fields: {sorted(unknown)}")
        required = allowed - {"ood_group"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"Missing episode fields: {sorted(missing)}")
        return cls(
            episode_id=str(value["episode_id"]),
            split=str(value["split"]),
            seed=int(value["seed"]),
            entity_id=str(value["entity_id"]),
            template_id=str(value["template_id"]),
            pair_id=str(value["pair_id"]),
            counterfactual_episode_id=str(value["counterfactual_episode_id"]),
            topic=str(value["topic"]),
            turns=tuple(Turn.from_dict(item) for item in value["turns"]),
            ood_group=str(value["ood_group"]) if value.get("ood_group") is not None else None,
        )


def read_jsonl(path: Path) -> list[Episode]:
    episodes: list[Episode] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                episodes.append(Episode.from_dict(value))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return episodes


def iter_jsonl(path: Path) -> Iterator[Episode]:
    yield from read_jsonl(path)


def write_jsonl(path: Path, episodes: Iterable[Episode]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
