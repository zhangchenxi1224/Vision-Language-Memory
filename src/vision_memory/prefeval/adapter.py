"""Leakage-aware adapter from an independent PrefEval checkout to state episodes."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

from .manifest import (
    ADAPTATION_SEED,
    CHOICES,
    FORCED_WRITE_COUNTS,
    FORMS,
    OPTION_SHUFFLE_SEED,
    TOPICS,
    assign_base_pair_splits,
)


FORBIDDEN_MODEL_KEYS = frozenset({"preference", "explanation", "aligned_op"})
Protocol = Literal["oracle-sparse", "forced-write"]


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: Any, *, source: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected non-empty text at {source}")
    return value


def _conversation_items(conversation: Any, *, source: str) -> list[tuple[str, Mapping[str, Any]]]:
    if isinstance(conversation, Mapping):
        def sort_key(item: tuple[Any, Any]) -> tuple[int, str]:
            raw = str(item[0])
            return (int(raw), "") if raw.isdigit() else (2**31 - 1, raw)

        raw_items = sorted(conversation.items(), key=sort_key)
    elif isinstance(conversation, list):
        raw_items = list(enumerate(conversation))
    else:
        raise ValueError(f"Expected a conversation object/list at {source}")
    result: list[tuple[str, Mapping[str, Any]]] = []
    for key, turn in raw_items:
        if not isinstance(turn, Mapping):
            raise ValueError(f"Expected a conversation turn object at {source}.{key}")
        result.append((str(key), turn))
    return result


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _oracle_persona_user_turn(hidden_reference: str, conversation: Any, *, source: str) -> tuple[str, str]:
    """Use the hidden reference only to locate a raw user turn; never return the field itself."""

    reference_tokens = set(_TOKEN_PATTERN.findall(hidden_reference.lower()))
    reference_normalized = " ".join(_TOKEN_PATTERN.findall(hidden_reference.lower()))
    candidates: list[tuple[float, int, str, str]] = []
    for position, (turn_id, turn) in enumerate(_conversation_items(conversation, source=source)):
        user = turn.get("user")
        if not isinstance(user, str) or not user.strip():
            continue
        normalized = " ".join(_TOKEN_PATTERN.findall(user.lower()))
        tokens = set(normalized.split())
        overlap = len(reference_tokens & tokens)
        token_f1 = 0.0
        if reference_tokens and tokens:
            token_f1 = 2.0 * overlap / (len(reference_tokens) + len(tokens))
        sequence_score = SequenceMatcher(None, reference_normalized, normalized).ratio()
        candidates.append((token_f1 + sequence_score, -position, turn_id, user))
    if not candidates:
        raise ValueError(f"Persona conversation has no non-empty user turn at {source}")
    _score, _negative_position, turn_id, user_text = max(candidates)
    return turn_id, user_text


def _choice_evidence(row: Mapping[str, Any], *, source: str) -> str:
    conversation = row.get("conversation")
    if not isinstance(conversation, Mapping):
        raise ValueError(f"Expected a choice conversation object at {source}.conversation")
    query = _required_text(conversation.get("query"), source=f"{source}.conversation.query")
    assistant = _required_text(
        conversation.get("assistant_options"), source=f"{source}.conversation.assistant_options"
    )
    selection = _required_text(
        conversation.get("user_selection"), source=f"{source}.conversation.user_selection"
    )
    return f"User: {query}\nAssistant: {assistant}\nUser: {selection}"


def _flatten_distractor_exchanges(raw: Any, *, source: str) -> list[str]:
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of distractor conversations at {source}")
    exchanges: list[str] = []
    for conversation_index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"Expected an object at {source}[{conversation_index}]")
        messages = item.get("conversation")
        if not isinstance(messages, list):
            raise ValueError(f"Expected a message list at {source}[{conversation_index}].conversation")
        for message_index in range(len(messages) - 1):
            user = messages[message_index]
            assistant = messages[message_index + 1]
            if not isinstance(user, Mapping) or not isinstance(assistant, Mapping):
                continue
            if user.get("role") != "user" or assistant.get("role") != "assistant":
                continue
            user_text = user.get("content")
            assistant_text = assistant.get("content")
            if isinstance(user_text, str) and user_text.strip() and isinstance(assistant_text, str) and assistant_text.strip():
                exchanges.append(f"User: {user_text}\nAssistant: {assistant_text}")
    if not exchanges:
        raise ValueError(f"No user/assistant distractor exchanges found at {source}")
    return exchanges


def _stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}\x1f{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _validate_model_keys(value: Any, *, path: str = "model_input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_MODEL_KEYS:
                raise ValueError(f"Forbidden hidden-data key {key!r} at {path}")
            _validate_model_keys(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_model_keys(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class PrefEvalTurn:
    type: Literal["event", "query"]
    text: str
    event_type: Literal["set", "noop"] | None = None
    options: tuple[str, ...] = ()
    evidence_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.type, "text": self.text}
        if self.event_type is not None:
            result["event_type"] = self.event_type
        if self.options:
            result["options"] = list(self.options)
        if self.evidence_source is not None:
            result["evidence_source"] = self.evidence_source
        return result


@dataclass(frozen=True)
class PrefEvalEpisode:
    sample_id: str
    base_pair_id: str
    topic: str
    row_index: int
    form: str
    split: str
    protocol: Protocol
    forced_write_k: int
    turns: tuple[PrefEvalTurn, ...]
    target_index: int
    target_choice: str
    option_permutation: tuple[int, ...]
    source_sha256: Mapping[str, str]

    def model_input(self) -> dict[str, Any]:
        """Return only the fields that an updater/reader is allowed to consume."""

        result = {
            "schema_version": "vision_memory.prefeval.model-input.v1",
            "sample_id": self.sample_id,
            "base_pair_id": self.base_pair_id,
            "topic": self.topic,
            "form": self.form,
            "split": self.split,
            "protocol": self.protocol,
            "forced_write_k": self.forced_write_k,
            "turns": [turn.to_dict() for turn in self.turns],
        }
        _validate_model_keys(result)
        return result

    def to_record(self) -> dict[str, Any]:
        """Return a training/evaluation record with labels outside the model-input view."""

        return {
            "schema_version": "vision_memory.prefeval.episode.v1",
            "model_input": self.model_input(),
            "label": {"target_index": self.target_index, "target_choice": self.target_choice},
            "audit": {
                "row_index": self.row_index,
                "option_permutation": list(self.option_permutation),
                "source_sha256": dict(self.source_sha256),
            },
        }


class PrefEvalAdapter:
    """Load canonical PrefEval files without importing or modifying their repository."""

    def __init__(
        self,
        prefeval_root: str | Path,
        *,
        adaptation_seed: int = ADAPTATION_SEED,
        option_seed: int = OPTION_SHUFFLE_SEED,
        distractor_seed: int = ADAPTATION_SEED,
    ) -> None:
        root = Path(prefeval_root).expanduser().resolve()
        if (root / "benchmark_dataset").is_dir():
            data_root = root / "benchmark_dataset"
        elif (root / "mcq_options").is_dir() and (root / "explicit_preference").is_dir():
            data_root = root
        else:
            raise FileNotFoundError(
                f"--prefeval-root must be a PrefEval checkout or benchmark_dataset directory: {root}"
            )
        self.prefeval_root = root
        self.data_root = data_root
        self.adaptation_seed = adaptation_seed
        self.option_seed = option_seed
        self.distractor_seed = distractor_seed
        self._topic_rows: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
        self._source_hashes: dict[str, str] = {}
        self._load_and_validate_topics()

        distractor_path = self.data_root / "filtered_inter_turns.json"
        distractor_raw = _read_json(distractor_path)
        self._distractors = _flatten_distractor_exchanges(distractor_raw, source=str(distractor_path))
        self._source_hashes[str(distractor_path.relative_to(self.data_root))] = _sha256_file(distractor_path)

        ids_by_topic = {
            topic: [f"{topic}:{index:04d}" for index in range(len(self._topic_rows[topic]["mcq"]))]
            for topic in TOPICS
        }
        self._split_by_pair = assign_base_pair_splits(ids_by_topic, seed=adaptation_seed)

    def _topic_paths(self, topic: str) -> dict[str, Path]:
        return {
            "mcq": self.data_root / "mcq_options" / f"{topic}.json",
            "explicit": self.data_root / "explicit_preference" / f"{topic}.json",
            "implicit_choice": self.data_root / "implicit_preference" / "choice-based" / f"{topic}.json",
            "implicit_persona": self.data_root / "implicit_preference" / "persona-driven" / f"{topic}.json",
        }

    def _load_and_validate_topics(self) -> None:
        for topic in TOPICS:
            paths = self._topic_paths(topic)
            rows_by_form: dict[str, list[Mapping[str, Any]]] = {}
            for form, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(f"Missing canonical PrefEval file: {path}")
                raw = _read_json(path)
                if not isinstance(raw, list) or not raw:
                    raise ValueError(f"Expected a non-empty JSON list in {path}")
                if not all(isinstance(row, Mapping) for row in raw):
                    raise ValueError(f"Expected only JSON objects in {path}")
                rows_by_form[form] = list(raw)
                self._source_hashes[str(path.relative_to(self.data_root))] = _sha256_file(path)

            counts = {form: len(rows) for form, rows in rows_by_form.items()}
            if len(set(counts.values())) != 1:
                raise ValueError(f"The three forms and MCQs are not aligned for {topic}: {counts}")
            for row_index in range(counts["mcq"]):
                questions = {
                    form: _required_text(rows[row_index].get("question"), source=f"{paths[form]}[{row_index}].question")
                    for form, rows in rows_by_form.items()
                }
                if len(set(questions.values())) != 1:
                    raise ValueError(f"Question mismatch across forms for {topic}:{row_index:04d}")
                options = rows_by_form["mcq"][row_index].get("classification_task_options")
                if (
                    not isinstance(options, list)
                    or len(options) != 4
                    or not all(isinstance(option, str) and option.strip() for option in options)
                    or len(set(options)) != 4
                ):
                    raise ValueError(f"Expected four unique MCQ options for {topic}:{row_index:04d}")
            self._topic_rows[topic] = rows_by_form

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "vision_memory.prefeval.manifest.v1",
            "data_root": str(self.data_root),
            "topics": list(TOPICS),
            "forms": list(FORMS),
            "base_pair_count": sum(len(self._topic_rows[topic]["mcq"]) for topic in TOPICS),
            "form_bound_sample_count": sum(len(self._topic_rows[topic]["mcq"]) for topic in TOPICS) * len(FORMS),
            "adaptation_seed": self.adaptation_seed,
            "option_shuffle_seed": self.option_seed,
            "distractor_seed": self.distractor_seed,
            "source_sha256": dict(sorted(self._source_hashes.items())),
        }

    def _permutations_for_topic(self, topic: str) -> list[tuple[int, ...]]:
        rng = random.Random(self.option_seed)
        return [tuple(rng.sample(range(4), 4)) for _ in self._topic_rows[topic]["mcq"]]

    def _evidence(self, form: str, row: Mapping[str, Any], *, source: str) -> tuple[str, str]:
        if form == "explicit":
            # The explicit disclosure is legitimate task input. It is exposed only as
            # event text, never as a privileged metadata field.
            return _required_text(row.get("preference"), source=f"{source}.preference"), "explicit_disclosure"
        if form == "implicit_choice":
            return _choice_evidence(row, source=source), "raw_choice_exchange"
        if form == "implicit_persona":
            hidden_reference = _required_text(row.get("preference"), source=f"{source}.preference")
            turn_id, user_text = _oracle_persona_user_turn(
                hidden_reference, row.get("conversation"), source=f"{source}.conversation"
            )
            return user_text, f"raw_persona_user_turn:{turn_id}"
        raise ValueError(f"Unsupported PrefEval form: {form}")

    def _sample_distractors(self, base_pair_id: str, form: str, count: int) -> list[str]:
        if count > len(self._distractors):
            raise ValueError(f"Requested {count} distractors but only {len(self._distractors)} are available")
        rng = random.Random(_stable_seed(self.distractor_seed, f"{base_pair_id}:{form}"))
        return rng.sample(self._distractors, count)

    def iter_episodes(
        self,
        *,
        forms: Sequence[str] = FORMS,
        protocol: Protocol = "oracle-sparse",
        forced_write_k: int = 0,
        splits: Iterable[str] | None = None,
    ) -> Iterator[PrefEvalEpisode]:
        requested_forms = tuple(forms)
        unknown_forms = sorted(set(requested_forms) - set(FORMS))
        if unknown_forms:
            raise ValueError(f"Unknown PrefEval forms: {', '.join(unknown_forms)}")
        if protocol not in ("oracle-sparse", "forced-write"):
            raise ValueError(f"Unknown PrefEval protocol: {protocol}")
        if protocol == "oracle-sparse" and forced_write_k != 0:
            raise ValueError("oracle-sparse requires forced_write_k=0")
        if protocol == "forced-write" and forced_write_k not in FORCED_WRITE_COUNTS:
            raise ValueError(f"forced-write count must be one of {FORCED_WRITE_COUNTS}")
        selected_splits = None if splits is None else set(splits)

        for topic in TOPICS:
            rows = self._topic_rows[topic]
            permutations = self._permutations_for_topic(topic)
            paths = self._topic_paths(topic)
            for row_index, mcq_row in enumerate(rows["mcq"]):
                base_pair_id = f"{topic}:{row_index:04d}"
                split = self._split_by_pair[base_pair_id]
                if selected_splits is not None and split not in selected_splits:
                    continue
                original_options = tuple(mcq_row["classification_task_options"])
                permutation = permutations[row_index]
                options = tuple(original_options[index] for index in permutation)
                target_index = permutation.index(0)
                question = _required_text(mcq_row.get("question"), source=f"{paths['mcq']}[{row_index}].question")

                for form in requested_forms:
                    row = rows[form][row_index]
                    source = f"{paths[form]}[{row_index}]"
                    evidence, evidence_source = self._evidence(form, row, source=source)
                    turns = [
                        PrefEvalTurn(
                            type="event",
                            event_type="set",
                            text=evidence,
                            evidence_source=evidence_source,
                        )
                    ]
                    if protocol == "forced-write":
                        turns.extend(
                            PrefEvalTurn(type="event", event_type="noop", text=text, evidence_source="distractor")
                            for text in self._sample_distractors(base_pair_id, form, forced_write_k)
                        )
                    turns.append(PrefEvalTurn(type="query", text=question, options=options))
                    sample_id = f"{base_pair_id}:{form}:{protocol}:k{forced_write_k}"
                    relevant_hashes = {
                        str(paths["mcq"].relative_to(self.data_root)): self._source_hashes[
                            str(paths["mcq"].relative_to(self.data_root))
                        ],
                        str(paths[form].relative_to(self.data_root)): self._source_hashes[
                            str(paths[form].relative_to(self.data_root))
                        ],
                    }
                    if protocol == "forced-write":
                        distractor_key = "filtered_inter_turns.json"
                        relevant_hashes[distractor_key] = self._source_hashes[distractor_key]
                    yield PrefEvalEpisode(
                        sample_id=sample_id,
                        base_pair_id=base_pair_id,
                        topic=topic,
                        row_index=row_index,
                        form=form,
                        split=split,
                        protocol=protocol,
                        forced_write_k=forced_write_k,
                        turns=tuple(turns),
                        target_index=target_index,
                        target_choice=CHOICES[target_index],
                        option_permutation=permutation,
                        source_sha256=relevant_hashes,
                    )


__all__ = [
    "FORBIDDEN_MODEL_KEYS",
    "PrefEvalAdapter",
    "PrefEvalEpisode",
    "PrefEvalTurn",
    "Protocol",
]
