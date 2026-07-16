"""Canonical, query-independent semantic-state contracts for visual teachers."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


SEMANTIC_STATE_SCHEMA = "vlm.semantic_state.v1"
_STATE_ID_DOMAIN = b"vlm.semantic_state.v1\0"
_STATUSES = frozenset({"active", "cleared", "unset"})
_FORBIDDEN_SUPERVISION_KEYS = frozenset(
    {
        "answer",
        "answer_index",
        "choices",
        "correct_answer",
        "correct_index",
        "label",
        "options",
        "query",
        "query_text",
        "target",
        "target_choice",
        "target_index",
        "target_text",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value with the repository's locked rules."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA256 digest.")
    return value


def reject_supervision_keys(value: Any, *, path: str = "value") -> None:
    """Reject query/answer supervision recursively at a teacher boundary."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in _FORBIDDEN_SUPERVISION_KEYS:
                raise ValueError(f"Supervision key {key!r} is forbidden at {path}.")
            reject_supervision_keys(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            reject_supervision_keys(child, path=f"{path}[{index}]")


def _normalized_text(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or normalized != normalized.strip():
        raise ValueError(f"{field} must be non-empty and have no surrounding whitespace.")
    if any(character.isspace() and character != " " for character in normalized) or "  " in normalized:
        raise ValueError(f"{field} must use single ASCII spaces as internal separators.")
    if any(unicodedata.category(character) in {"Cc", "Cs"} for character in normalized):
        raise ValueError(f"{field} contains a control or surrogate character.")
    return normalized


@dataclass(frozen=True)
class SemanticStateEntry:
    """One current entity-slot value; no event history or query is represented."""

    entity_id: str
    entity_text: str
    slot_id: str
    slot_text: str
    status: str
    value_id: str | None = None
    value_text: str | None = None

    def __post_init__(self) -> None:
        for field in ("entity_id", "entity_text", "slot_id", "slot_text"):
            object.__setattr__(self, field, _normalized_text(getattr(self, field), field=field))
        normalized_status = _normalized_text(self.status, field="status").casefold()
        if normalized_status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)}.")
        object.__setattr__(self, "status", normalized_status)

        if normalized_status == "active":
            if self.value_id is None or self.value_text is None:
                raise ValueError("An active semantic-state entry requires value_id and value_text.")
            object.__setattr__(self, "value_id", _normalized_text(self.value_id, field="value_id"))
            object.__setattr__(self, "value_text", _normalized_text(self.value_text, field="value_text"))
        elif self.value_id is not None or self.value_text is not None:
            raise ValueError("A cleared or unset semantic-state entry cannot contain a value.")

    @property
    def key(self) -> tuple[str, str]:
        return self.entity_id, self.slot_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_text": self.entity_text,
            "slot_id": self.slot_id,
            "slot_text": self.slot_text,
            "status": self.status,
            "value_id": self.value_id,
            "value_text": self.value_text,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticStateEntry":
        reject_supervision_keys(value, path="semantic_state.entries[]")
        allowed = {"entity_id", "entity_text", "slot_id", "slot_text", "status", "value_id", "value_text"}
        unknown = set(value) - allowed
        missing = {"entity_id", "entity_text", "slot_id", "slot_text", "status"} - set(value)
        if unknown:
            raise ValueError(f"Unknown semantic-state entry fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"Missing semantic-state entry fields: {sorted(missing)}")
        return cls(
            entity_id=value["entity_id"],
            entity_text=value["entity_text"],
            slot_id=value["slot_id"],
            slot_text=value["slot_text"],
            status=value["status"],
            value_id=value.get("value_id"),
            value_text=value.get("value_text"),
        )


@dataclass(frozen=True)
class SemanticState:
    """A complete current ledger with path- and query-independent identity."""

    entries: tuple[SemanticStateEntry, ...]
    schema: str = SEMANTIC_STATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SEMANTIC_STATE_SCHEMA:
            raise ValueError(f"Unsupported semantic-state schema: {self.schema!r}.")
        entries = tuple(self.entries)
        if any(not isinstance(entry, SemanticStateEntry) for entry in entries):
            raise TypeError("entries must contain only SemanticStateEntry values.")
        keys = [entry.key for entry in entries]
        if len(keys) != len(set(keys)):
            raise ValueError("A semantic state cannot contain duplicate entity_id/slot_id entries.")
        surface_keys = [(entry.entity_text, entry.slot_text) for entry in entries]
        if len(surface_keys) != len(set(surface_keys)):
            raise ValueError("A semantic state cannot render duplicate entity_text/slot_text entries.")
        object.__setattr__(self, "entries", entries)

    @property
    def sorted_entries(self) -> tuple[SemanticStateEntry, ...]:
        return tuple(sorted(self.entries, key=lambda entry: entry.key))

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "entries": [entry.to_dict() for entry in self.sorted_entries]}

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def canonical_sha256(self) -> str:
        return sha256_bytes(self.canonical_bytes)

    @property
    def state_id(self) -> str:
        return sha256_bytes(_STATE_ID_DOMAIN + self.canonical_bytes)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticState":
        reject_supervision_keys(value, path="semantic_state")
        if set(value) != {"schema", "entries"}:
            unknown = set(value) - {"schema", "entries"}
            missing = {"schema", "entries"} - set(value)
            raise ValueError(f"Semantic-state fields differ from the contract; unknown={sorted(unknown)}, missing={sorted(missing)}")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes, bytearray)):
            raise TypeError("semantic_state.entries must be a sequence.")
        return cls(
            schema=str(value["schema"]),
            entries=tuple(SemanticStateEntry.from_dict(entry) for entry in raw_entries),
        )


__all__ = [
    "SEMANTIC_STATE_SCHEMA",
    "SemanticState",
    "SemanticStateEntry",
    "canonical_json_bytes",
    "reject_supervision_keys",
    "require_sha256",
    "sha256_bytes",
]
