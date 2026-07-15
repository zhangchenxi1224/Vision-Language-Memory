"""Corpus-level validation for generated episode JSONL files."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .generator import OOD_GROUPS
from .schema import EventKind, Episode, read_jsonl


class DatasetValidationError(ValueError):
    """Raised when a dataset violates a preregistered structural invariant."""


@dataclass(frozen=True)
class SplitValidation:
    episodes: int
    queries: int
    updates: int
    target_position_counts: dict[int, int]
    max_target_position_deviation: float
    event_kind_counts: dict[str, int]


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    splits: dict[str, SplitValidation]
    total_episodes: int
    total_queries: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fail(message: str) -> None:
    raise DatasetValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(root: Path) -> Mapping[str, Any] | None:
    path = root / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Invalid manifest {path}: {exc}")
    return None


def _validate_split(split: str, episodes: list[Episode], *, balance_tolerance: float) -> SplitValidation:
    if not episodes:
        _fail(f"{split} is empty")
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        _fail(f"{split} contains duplicate episode_id values")
    by_id = {episode.episode_id: episode for episode in episodes}
    position_counts: Counter[int] = Counter()
    event_counts: Counter[str] = Counter()
    queries = 0
    updates = 0

    for episode in episodes:
        if episode.split != split:
            _fail(f"{episode.episode_id} declares split={episode.split!r}, expected {split!r}")
        if episode.counterfactual_episode_id not in by_id:
            _fail(f"{episode.episode_id} points outside its split for counterfactual")
        mate = by_id[episode.counterfactual_episode_id]
        if mate.counterfactual_episode_id != episode.episode_id or mate.pair_id != episode.pair_id:
            _fail(f"Counterfactual link is not reciprocal for {episode.episode_id}")
        if (mate.entity_id, mate.template_id, mate.topic) != (
            episode.entity_id,
            episode.template_id,
            episode.topic,
        ):
            _fail(f"Matched pair metadata differs for {episode.pair_id}")
        targets = [turn.query.target for turn in episode.turns if turn.query is not None]
        mate_targets = [turn.query.target for turn in mate.turns if turn.query is not None]
        if targets[-1] == mate_targets[-1]:
            _fail(f"Matched pair {episode.pair_id} has the same final target")
        final_query = next(turn.query for turn in reversed(episode.turns) if turn.query is not None)
        mate_final_query = next(turn.query for turn in reversed(mate.turns) if turn.query is not None)
        if set(final_query.choices) != set(mate_final_query.choices):
            _fail(f"Matched pair {episode.pair_id} does not share the same final choice set")

        if episode.ood_group == "heldout_length":
            if not 9 <= len(episode.turns) <= 16:
                _fail(f"Length-OOD episode {episode.episode_id} is outside 9-16 turns")
        else:
            if not 4 <= len(episode.turns) <= 8:
                _fail(f"ID-style episode {episode.episode_id} is outside 4-8 turns")
            if not 2 <= episode.update_count <= 5:
                _fail(f"{episode.episode_id} has {episode.update_count} updates, expected 2-5")
        if not 1 <= episode.query_count <= 3:
            _fail(f"{episode.episode_id} has {episode.query_count} queries, expected 1-3")

        updates += episode.update_count
        queries += episode.query_count
        for turn in episode.turns:
            if turn.event_kind is not None:
                event_counts[turn.event_kind.value] += 1
            if turn.query is not None:
                position_counts[turn.query.target_index] += 1

    expected_fraction = 0.25
    deviations = [abs(position_counts[index] / queries - expected_fraction) for index in range(4)]
    max_deviation = max(deviations)
    if max_deviation > balance_tolerance:
        _fail(
            f"{split} target positions exceed tolerance: counts={dict(position_counts)}, "
            f"max_deviation={max_deviation:.6f}, tolerance={balance_tolerance:.6f}"
        )
    missing_kinds = {kind.value for kind in EventKind} - set(event_counts)
    if missing_kinds:
        _fail(f"{split} is missing event kinds: {sorted(missing_kinds)}")

    return SplitValidation(
        episodes=len(episodes),
        queries=queries,
        updates=updates,
        target_position_counts={index: position_counts[index] for index in range(4)},
        max_target_position_deviation=max_deviation,
        event_kind_counts=dict(sorted(event_counts.items())),
    )


def validate_dataset(
    root: Path,
    *,
    expected_sizes: Mapping[str, int] | None = None,
    balance_tolerance: float = 0.02,
    verify_manifest_hashes: bool = True,
) -> ValidationReport:
    """Validate schema, leakage, grouping, pairing, and target balance."""

    if balance_tolerance < 0:
        raise ValueError("balance_tolerance must be non-negative")
    split_names = ("train", "dev", "test_id", "test_ood")
    manifest = _load_manifest(root)
    split_episodes: dict[str, list[Episode]] = {}
    for split in split_names:
        path = root / f"{split}.jsonl"
        if not path.exists():
            _fail(f"Missing split file: {path}")
        try:
            episodes = read_jsonl(path)
        except ValueError as exc:
            _fail(str(exc))
        split_episodes[split] = episodes
        if expected_sizes is not None and len(episodes) != int(expected_sizes[split]):
            _fail(f"{split} contains {len(episodes)} episodes, expected {expected_sizes[split]}")
        if manifest is not None:
            declared = manifest.get("splits", {}).get(split, {})
            if int(declared.get("episodes", -1)) != len(episodes):
                _fail(f"Manifest count mismatch for {split}")
            if verify_manifest_hashes and declared.get("sha256") != _sha256(path):
                _fail(f"Manifest SHA256 mismatch for {split}")

    # Entity/template grouping is intentionally split-disjoint. Counterfactual mates are
    # the only episodes allowed to share a controlled entity within a split.
    for index, left in enumerate(split_names):
        left_entities = {episode.entity_id for episode in split_episodes[left]}
        left_templates = {episode.template_id for episode in split_episodes[left]}
        for right in split_names[index + 1 :]:
            entity_overlap = left_entities & {episode.entity_id for episode in split_episodes[right]}
            template_overlap = left_templates & {episode.template_id for episode in split_episodes[right]}
            if entity_overlap:
                _fail(f"Entity leakage between {left} and {right}: {sorted(entity_overlap)[:3]}")
            if template_overlap:
                _fail(f"Template leakage between {left} and {right}: {sorted(template_overlap)[:3]}")

    ood_counts = Counter(episode.ood_group for episode in split_episodes["test_ood"])
    if set(ood_counts) != set(OOD_GROUPS) or len(set(ood_counts.values())) != 1:
        _fail(f"test_ood must be evenly divided among {list(OOD_GROUPS)}, got {dict(ood_counts)}")
    for split in ("train", "dev", "test_id"):
        if any(episode.ood_group is not None for episode in split_episodes[split]):
            _fail(f"{split} unexpectedly contains ood_group labels")

    reports = {
        split: _validate_split(split, episodes, balance_tolerance=balance_tolerance)
        for split, episodes in split_episodes.items()
    }
    return ValidationReport(
        valid=True,
        splits=reports,
        total_episodes=sum(report.episodes for report in reports.values()),
        total_queries=sum(report.queries for report in reports.values()),
    )
