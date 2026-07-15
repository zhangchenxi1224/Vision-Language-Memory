"""Fail-closed PrefEval export reader with a narrow supervised training boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_LABEL_KEYS = frozenset({"label", "target", "target_index", "target_choice", "aligned_op", "explanation"})
_MODEL_INPUT_FIELDS = frozenset(
    {
        "schema_version",
        "sample_id",
        "base_pair_id",
        "topic",
        "form",
        "split",
        "protocol",
        "forced_write_k",
        "turns",
    }
)


def _reject_supervision(value: Any, *, path: str = "model_input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _LABEL_KEYS:
                raise ValueError(f"Supervision key {key!r} is forbidden inside {path}.")
            _reject_supervision(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_supervision(child, path=f"{path}[{index}]")


def _non_empty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True)
class AdaptedPrefEvalRecord:
    """Validated PrefEval record whose label remains outside ``model_input``."""

    model_input: Mapping[str, Any]
    target_index: int
    target_choice: str
    audit: Mapping[str, Any]

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "AdaptedPrefEvalRecord":
        allowed = {"schema_version", "model_input", "label", "audit"}
        if set(record) != allowed:
            raise ValueError(f"PrefEval record fields must be exactly {sorted(allowed)}.")
        if record.get("schema_version") != "vision_memory.prefeval.episode.v1":
            raise ValueError("Unsupported PrefEval export schema_version.")
        model_input = record.get("model_input")
        label = record.get("label")
        audit = record.get("audit")
        if not isinstance(model_input, Mapping) or set(model_input) != _MODEL_INPUT_FIELDS:
            raise ValueError("PrefEval model_input has missing or unknown fields.")
        if not isinstance(label, Mapping) or set(label) != {"target_index", "target_choice"}:
            raise ValueError("PrefEval label must contain only target_index and target_choice.")
        if not isinstance(audit, Mapping):
            raise ValueError("PrefEval audit must be an object.")
        _reject_supervision(model_input)

        if model_input.get("schema_version") != "vision_memory.prefeval.model-input.v1":
            raise ValueError("Unsupported PrefEval model_input schema_version.")
        _non_empty_string(model_input.get("sample_id"), path="model_input.sample_id")
        _non_empty_string(model_input.get("base_pair_id"), path="model_input.base_pair_id")
        _non_empty_string(model_input.get("topic"), path="model_input.topic")
        form = _non_empty_string(model_input.get("form"), path="model_input.form")
        if form not in {"explicit", "implicit_choice", "implicit_persona"}:
            raise ValueError(f"Unsupported PrefEval form: {form!r}.")
        split = _non_empty_string(model_input.get("split"), path="model_input.split")
        if split not in {"adapt_train", "adapt_dev", "adapt_ood"}:
            raise ValueError(f"Unsupported adapted PrefEval split: {split!r}.")
        protocol = model_input.get("protocol")
        forced_write_k = model_input.get("forced_write_k")
        if protocol == "oracle-sparse" and forced_write_k != 0:
            raise ValueError("oracle-sparse requires forced_write_k=0.")
        if protocol == "forced-write" and forced_write_k not in {0, 2, 5, 10}:
            raise ValueError("forced-write requires k in {0,2,5,10}.")
        if protocol not in {"oracle-sparse", "forced-write"}:
            raise ValueError(f"Unsupported PrefEval protocol: {protocol!r}.")
        turns = model_input.get("turns")
        if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)) or not turns:
            raise ValueError("model_input.turns must be a non-empty sequence.")
        query_count = 0
        for index, turn in enumerate(turns):
            if not isinstance(turn, Mapping):
                raise ValueError(f"model_input.turns[{index}] must be an object.")
            turn_type = turn.get("type")
            if turn_type == "event":
                allowed_turn = {"type", "text", "event_type", "evidence_source"}
                if set(turn) - allowed_turn:
                    raise ValueError(f"Unknown event fields at model_input.turns[{index}].")
                _non_empty_string(turn.get("text"), path=f"model_input.turns[{index}].text")
                if turn.get("event_type") not in {"set", "noop"}:
                    raise ValueError(f"Unsupported event_type at model_input.turns[{index}].")
            elif turn_type == "query":
                if set(turn) != {"type", "text", "options"}:
                    raise ValueError(f"Query fields must be type/text/options at model_input.turns[{index}].")
                _non_empty_string(turn.get("text"), path=f"model_input.turns[{index}].text")
                options = turn.get("options")
                if (
                    not isinstance(options, Sequence)
                    or isinstance(options, (str, bytes))
                    or len(options) != 4
                    or any(not isinstance(option, str) or not option.strip() for option in options)
                ):
                    raise ValueError(f"model_input.turns[{index}].options must contain four strings.")
                query_count += 1
            else:
                raise ValueError(f"Unsupported turn type at model_input.turns[{index}].")
        if query_count != 1:
            raise ValueError("Each PrefEval adapted record must contain exactly one query.")
        if turns[-1].get("type") != "query":
            raise ValueError("The PrefEval query must be the final routed turn.")

        target_index = label.get("target_index")
        target_choice = label.get("target_choice")
        if not isinstance(target_index, int) or not 0 <= target_index < 4:
            raise ValueError("label.target_index must be in [0, 3].")
        if target_choice != "ABCD"[target_index]:
            raise ValueError("label.target_choice is inconsistent with label.target_index.")
        return cls(
            model_input=dict(model_input),
            target_index=target_index,
            target_choice=target_choice,
            audit=dict(audit),
        )

    def supervised_episode(self) -> dict[str, Any]:
        """Create the runner record; target is used only by the loss boundary.

        Updater inputs are copied solely from event ``text`` and Reader inputs solely from
        query ``text/options``. The audit object and label name/value never enter either
        callable.
        """

        turns: list[dict[str, Any]] = []
        for turn_id, turn in enumerate(self.model_input["turns"]):
            if turn["type"] == "event":
                turns.append(
                    {
                        "turn_id": turn_id,
                        "kind": "event",
                        "event_kind": turn["event_type"],
                        "event_text": turn["text"],
                    }
                )
            else:
                turns.append(
                    {
                        "turn_id": turn_id,
                        "kind": "query",
                        "query_text": turn["text"],
                        "choices": list(turn["options"]),
                        "target_index": self.target_index,
                    }
                )
        return {
            "episode_id": self.model_input["sample_id"],
            "split": self.model_input["split"],
            "topic": self.model_input["topic"],
            "form": self.model_input["form"],
            "base_pair_id": self.model_input["base_pair_id"],
            "protocol": self.model_input["protocol"],
            "forced_write_k": self.model_input["forced_write_k"],
            "turns": turns,
        }


def read_prefeval_adapted_jsonl(
    path: Path,
    *,
    allowed_splits: set[str] | None = None,
) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, Mapping):
                    raise ValueError("record must be an object")
                adapted = AdaptedPrefEvalRecord.from_record(raw)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            split = str(adapted.model_input["split"])
            if allowed_splits is None or split in allowed_splits:
                episodes.append(adapted.supervised_episode())
    if not episodes:
        raise ValueError(f"No eligible adapted PrefEval records found in {path}.")
    return episodes


def read_prefeval_supervised_jsonl(
    path: Path,
    *,
    allowed_splits: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Read the exact boundary format emitted by ``convert_prefeval_adapted.py``."""

    expected_fields = {
        "episode_id",
        "split",
        "topic",
        "form",
        "base_pair_id",
        "protocol",
        "forced_write_k",
        "turns",
    }
    episodes: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, Mapping) or set(record) != expected_fields:
                    raise ValueError("supervised boundary record has missing or unknown fields")
                _non_empty_string(record.get("episode_id"), path="episode_id")
                split = _non_empty_string(record.get("split"), path="split")
                if split not in {"adapt_train", "adapt_dev", "adapt_ood"}:
                    raise ValueError(f"unsupported split {split!r}")
                turns = record.get("turns")
                if not isinstance(turns, list) or not turns:
                    raise ValueError("turns must be a non-empty list")
                query_count = 0
                for index, turn in enumerate(turns):
                    if not isinstance(turn, Mapping):
                        raise ValueError(f"turn {index} must be an object")
                    if turn.get("kind") == "event":
                        if set(turn) != {"turn_id", "kind", "event_kind", "event_text"}:
                            raise ValueError(f"event turn {index} has missing or unknown fields")
                        if turn.get("event_kind") not in {"set", "noop"}:
                            raise ValueError(f"event turn {index} has an invalid event_kind")
                        if turn.get("turn_id") != index:
                            raise ValueError(f"event turn {index} has a non-canonical turn_id")
                        _non_empty_string(turn.get("event_text"), path=f"turns[{index}].event_text")
                    elif turn.get("kind") == "query":
                        if set(turn) != {"turn_id", "kind", "query_text", "choices", "target_index"}:
                            raise ValueError(f"query turn {index} has missing or unknown fields")
                        choices = turn.get("choices")
                        target_index = turn.get("target_index")
                        if turn.get("turn_id") != index:
                            raise ValueError(f"query turn {index} has a non-canonical turn_id")
                        _non_empty_string(turn.get("query_text"), path=f"turns[{index}].query_text")
                        if (
                            not isinstance(choices, list)
                            or len(choices) != 4
                            or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
                            or len(set(choices)) != 4
                        ):
                            raise ValueError(f"query turn {index} must have four choices")
                        if not isinstance(target_index, int) or not 0 <= target_index < 4:
                            raise ValueError(f"query turn {index} has an invalid target_index")
                        query_count += 1
                    else:
                        raise ValueError(f"turn {index} has an invalid kind")
                if query_count != 1:
                    raise ValueError("each PrefEval boundary episode must have one query")
                if turns[-1].get("kind") != "query":
                    raise ValueError("the PrefEval query must be the final routed turn")
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if allowed_splits is None or split in allowed_splits:
                episodes.append(record)
    if not episodes:
        raise ValueError(f"No eligible converted PrefEval records found in {path}.")
    return episodes


__all__ = [
    "AdaptedPrefEvalRecord",
    "read_prefeval_adapted_jsonl",
    "read_prefeval_supervised_jsonl",
]
