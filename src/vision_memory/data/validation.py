"""Corpus-level validation for generated episode JSONL files."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .generator import OOD_GROUPS
from .schema import (
    DistractorVariant,
    Episode,
    EventKind,
    read_jsonl,
    surface_template_signatures,
)


class DatasetValidationError(ValueError):
    """Raised when a dataset violates a preregistered structural invariant."""


@dataclass(frozen=True)
class SplitValidation:
    episodes: int
    queries: int
    mixed_queries: int
    updates: int
    target_position_counts: dict[int, int]
    max_target_position_deviation: float
    event_kind_counts: dict[str, int]
    distractor_variant_counts: dict[str, int]
    matched_distractor_pairs: int
    entity_surface_count: int
    template_family_count: int
    surface_template_signature_count: int
    canonical_payloads: int
    canonical_target_share_reference: float
    canonical_balance_enforced: bool
    max_canonical_target_share_deviation: float
    max_canonical_order_variants: int


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


def _set_sha256(values: set[str]) -> str:
    payload = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_manifest(root: Path) -> Mapping[str, Any] | None:
    path = root / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Invalid manifest {path}: {exc}")
    return None


def _queries(episode: Episode):
    return [turn.query for turn in episode.turns if turn.query is not None]


def _validate_visible_metadata(episode: Episode) -> None:
    if episode.entity_surface is None:
        _fail(f"{episode.episode_id} is missing entity_surface")
    if episode.template_family is None:
        _fail(f"{episode.episode_id} is missing template_family")
    if episode.distractor_variant is None:
        _fail(f"{episode.episode_id} is missing distractor_variant")
    entity = episode.entity_surface.casefold()
    family = episode.template_family.casefold()
    for turn_index, turn in enumerate(episode.turns):
        if turn.calls_updater:
            text = (turn.event_text or "").casefold()
            if family not in text:
                _fail(
                    f"{episode.episode_id} turn {turn_index} does not expose its actual template_family in event_text"
                )
            if turn.event_kind is not EventKind.NOOP and entity not in text:
                _fail(
                    f"{episode.episode_id} turn {turn_index} does not contain entity_surface "
                    "in its state-changing event_text"
                )
        if turn.calls_reader:
            if turn.query is None:  # pragma: no cover - schema already enforces this
                _fail(f"{episode.episode_id} turn {turn_index} has no query")
            text = turn.query.text.casefold()
            if family not in text:
                _fail(
                    f"{episode.episode_id} turn {turn_index} does not expose its actual template_family in query text"
                )
            if entity not in text:
                _fail(f"{episode.episode_id} turn {turn_index} does not contain entity_surface in query text")
            if turn.query.comparison_id is None:
                _fail(f"{episode.episode_id} turn {turn_index} is missing query.comparison_id")


def _validate_counterfactual_pair(episode: Episode, mate: Episode) -> None:
    if mate.counterfactual_episode_id != episode.episode_id or mate.pair_id != episode.pair_id:
        _fail(f"Counterfactual link is not reciprocal for {episode.episode_id}")
    metadata = (
        "entity_id",
        "entity_surface",
        "template_id",
        "template_family",
        "topic",
        "ood_group",
        "distractor_variant",
    )
    if any(getattr(mate, field) != getattr(episode, field) for field in metadata):
        _fail(f"Matched counterfactual metadata differs for {episode.pair_id}")
    targets = [query.target for query in _queries(episode)]
    mate_targets = [query.target for query in _queries(mate)]
    if targets[-1] == mate_targets[-1]:
        _fail(f"Matched counterfactual pair {episode.pair_id} has the same final target")
    final_query = _queries(episode)[-1]
    mate_final_query = _queries(mate)[-1]
    if set(final_query.choices) != set(mate_final_query.choices):
        _fail(f"Matched counterfactual pair {episode.pair_id} has different final choices")


def _validate_distractor_pair(episode: Episode, by_id: Mapping[str, Episode]) -> None:
    variant = episode.distractor_variant
    if variant is DistractorVariant.UNPAIRED:
        if episode.distractor_pair_id is not None or episode.distractor_episode_id is not None:
            _fail(f"Unpaired episode {episode.episode_id} contains distractor links")
        return
    if variant not in {DistractorVariant.CLEAN, DistractorVariant.DISTRACTOR}:
        _fail(f"{episode.episode_id} has unsupported distractor_variant={variant}")
    if episode.distractor_episode_id not in by_id:
        _fail(f"{episode.episode_id} points outside its split for distractor mate")
    mate = by_id[episode.distractor_episode_id]
    expected = DistractorVariant.DISTRACTOR if variant is DistractorVariant.CLEAN else DistractorVariant.CLEAN
    if (
        mate.distractor_variant is not expected
        or mate.distractor_episode_id != episode.episode_id
        or mate.distractor_pair_id != episode.distractor_pair_id
    ):
        _fail(f"Clean/distractor link is not reciprocal for {episode.episode_id}")
    metadata = (
        "entity_id",
        "entity_surface",
        "template_id",
        "template_family",
        "topic",
        "ood_group",
    )
    if any(getattr(mate, field) != getattr(episode, field) for field in metadata):
        _fail(f"Clean/distractor metadata differs for {episode.distractor_pair_id}")
    episode_queries = _queries(episode)
    mate_queries = _queries(mate)
    if len(episode_queries) != len(mate_queries):
        _fail(f"Clean/distractor query counts differ for {episode.distractor_pair_id}")
    for query, mate_query in zip(episode_queries, mate_queries, strict=True):
        if (
            query.comparison_id != mate_query.comparison_id
            or query.text != mate_query.text
            or query.choices != mate_query.choices
            or query.target_index != mate_query.target_index
        ):
            _fail(f"Clean/distractor query payload differs for {episode.distractor_pair_id}")
    clean = episode if variant is DistractorVariant.CLEAN else mate
    distractor = mate if variant is DistractorVariant.CLEAN else episode
    if clean.distractor_turn_indices:
        _fail(f"Clean episode {clean.episode_id} contains no-op distractors")
    if not distractor.distractor_turn_indices:
        _fail(f"Distractor episode {distractor.episode_id} contains no no-op writes")


def _validate_split(
    split: str,
    episodes: list[Episode],
    *,
    balance_tolerance: float,
    expected_event_kinds: frozenset[EventKind],
    enforce_canonical_payload_balance: bool,
) -> SplitValidation:
    if not episodes:
        _fail(f"{split} is empty")
    ids = [episode.episode_id for episode in episodes]
    if len(ids) != len(set(ids)):
        _fail(f"{split} contains duplicate episode_id values")
    by_id = {episode.episode_id: episode for episode in episodes}
    position_counts: Counter[int] = Counter()
    event_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    comparison_members: dict[str, list[Episode]] = defaultdict(list)
    canonical_targets: dict[tuple[str, tuple[str, ...]], Counter[str]] = defaultdict(Counter)
    canonical_orders: dict[tuple[str, tuple[str, ...]], set[tuple[str, str, str, str]]] = defaultdict(set)
    entity_id_to_surface: dict[str, str] = {}
    surface_to_entity_id: dict[str, str] = {}
    template_id_to_family: dict[str, str] = {}
    queries = 0
    mixed_queries = 0
    updates = 0

    for episode in episodes:
        if episode.split != split:
            _fail(f"{episode.episode_id} declares split={episode.split!r}, expected {split!r}")
        _validate_visible_metadata(episode)
        entity_surface = episode.entity_surface or ""
        template_family = episode.template_family or ""
        previous_surface = entity_id_to_surface.setdefault(episode.entity_id, entity_surface)
        if previous_surface != entity_surface:
            _fail(f"entity_id {episode.entity_id} maps to multiple visible entity strings")
        previous_id = surface_to_entity_id.setdefault(entity_surface, episode.entity_id)
        if previous_id != episode.entity_id:
            _fail(f"entity_surface {entity_surface!r} maps to multiple entity IDs")
        previous_family = template_id_to_family.setdefault(episode.template_id, template_family)
        if previous_family != template_family:
            _fail(f"template_id {episode.template_id} maps to multiple visible template families")

        if episode.counterfactual_episode_id not in by_id:
            _fail(f"{episode.episode_id} points outside its split for counterfactual")
        _validate_counterfactual_pair(episode, by_id[episode.counterfactual_episode_id])
        _validate_distractor_pair(episode, by_id)
        variant_counts[episode.distractor_variant.value] += 1

        for query in _queries(episode):
            comparison_members[query.comparison_id].append(episode)

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
        mixed_queries += episode.mixed_query_count
        for turn in episode.turns:
            if turn.event_kind is not None:
                event_counts[turn.event_kind.value] += 1
            if turn.query is not None:
                position_counts[turn.query.target_index] += 1
                canonical_key = (turn.query.text, tuple(sorted(turn.query.choices)))
                canonical_targets[canonical_key][turn.query.target] += 1
                canonical_orders[canonical_key].add(turn.query.choices)

    for comparison_id, members in comparison_members.items():
        variants = {episode.distractor_variant for episode in members}
        if variants == {DistractorVariant.CLEAN, DistractorVariant.DISTRACTOR}:
            if len(members) != 2:
                _fail(f"comparison_id {comparison_id} does not identify exactly one matched pair")
        elif variants != {DistractorVariant.UNPAIRED} or len(members) != 1:
            _fail(f"comparison_id {comparison_id} has invalid distractor membership")

    expected_fraction = 0.25
    canonical_deviations: list[float] = []
    for canonical_key, target_counts in canonical_targets.items():
        _, candidates = canonical_key
        member_count = sum(target_counts.values())
        shares = {candidate: target_counts[candidate] / member_count for candidate in candidates}
        canonical_deviations.extend(abs(share - expected_fraction) for share in shares.values())
        if enforce_canonical_payload_balance and any(
            target_counts[candidate] * 4 != member_count for candidate in candidates
        ):
            _fail(
                "Full-profile canonical payload target shares must each equal 0.25: "
                f"text={canonical_key[0]!r}, choices={candidates!r}, "
                f"counts={dict(target_counts)}"
            )
        order_variants = len(canonical_orders[canonical_key])
        if enforce_canonical_payload_balance and order_variants != 1:
            _fail(
                "Full-profile canonical payload must have exactly one ordered choices tuple: "
                f"text={canonical_key[0]!r}, choices={candidates!r}, "
                f"ordered_variants={order_variants}"
            )

    deviations = [abs(position_counts[index] / queries - expected_fraction) for index in range(4)]
    max_deviation = max(deviations)
    if max_deviation > balance_tolerance:
        _fail(
            f"{split} target positions exceed tolerance: counts={dict(position_counts)}, "
            f"max_deviation={max_deviation:.6f}, tolerance={balance_tolerance:.6f}"
        )
    expected_kind_values = {kind.value for kind in expected_event_kinds}
    actual_kind_values = set(event_counts)
    missing_kinds = expected_kind_values - actual_kind_values
    if missing_kinds:
        _fail(f"{split} is missing event kinds: {sorted(missing_kinds)}")
    unexpected_kinds = actual_kind_values - expected_kind_values
    if unexpected_kinds:
        _fail(f"{split} contains event kinds excluded by its transition profile: {sorted(unexpected_kinds)}")

    entities = {episode.entity_surface for episode in episodes if episode.entity_surface is not None}
    families = {episode.template_family for episode in episodes if episode.template_family is not None}
    signatures = {signature for episode in episodes for signature in surface_template_signatures(episode)}
    matched_pairs = sum(episode.distractor_variant is DistractorVariant.CLEAN for episode in episodes)
    return SplitValidation(
        episodes=len(episodes),
        queries=queries,
        mixed_queries=mixed_queries,
        updates=updates,
        target_position_counts={index: position_counts[index] for index in range(4)},
        max_target_position_deviation=max_deviation,
        event_kind_counts=dict(sorted(event_counts.items())),
        distractor_variant_counts=dict(sorted(variant_counts.items())),
        matched_distractor_pairs=matched_pairs,
        entity_surface_count=len(entities),
        template_family_count=len(families),
        surface_template_signature_count=len(signatures),
        canonical_payloads=len(canonical_targets),
        canonical_target_share_reference=expected_fraction,
        canonical_balance_enforced=enforce_canonical_payload_balance,
        max_canonical_target_share_deviation=max(canonical_deviations, default=0.0),
        max_canonical_order_variants=max(
            (len(orders) for orders in canonical_orders.values()),
            default=0,
        ),
    )


def _validate_surface_partitions(
    split_episodes: Mapping[str, list[Episode]],
    manifest: Mapping[str, Any] | None,
) -> None:
    split_names = tuple(split_episodes)
    surface_sets: dict[str, dict[str, set[str]]] = {}
    for split, episodes in split_episodes.items():
        surface_sets[split] = {
            "entity_surface": {episode.entity_surface for episode in episodes if episode.entity_surface is not None},
            "template_family": {episode.template_family for episode in episodes if episode.template_family is not None},
            "surface_template_signature": {
                signature for episode in episodes for signature in surface_template_signatures(episode)
            },
        }

    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            for label, left_values in surface_sets[left].items():
                overlap = left_values & surface_sets[right][label]
                if overlap:
                    _fail(f"Model-visible {label} leakage between {left} and {right}: {sorted(overlap)[:3]}")

    if manifest is None:
        return
    declared_partitions = manifest.get("surface_partitions")
    if not isinstance(declared_partitions, Mapping):
        _fail("Manifest is missing surface_partitions")
    for split in split_names:
        declared = declared_partitions.get(split)
        if not isinstance(declared, Mapping):
            _fail(f"Manifest is missing surface partition for {split}")
        entities = surface_sets[split]["entity_surface"]
        families = surface_sets[split]["template_family"]
        signatures = surface_sets[split]["surface_template_signature"]
        expected = {
            "entity_surface_count": len(entities),
            "entity_surface_sha256": _set_sha256(entities),
            "template_family_count": len(families),
            "template_families": sorted(families),
            "template_family_sha256": _set_sha256(families),
            "surface_template_signature_count": len(signatures),
            "surface_template_signature_sha256": _set_sha256(signatures),
        }
        for field, value in expected.items():
            if declared.get(field) != value:
                _fail(f"Manifest surface partition mismatch for {split}.{field}")


def validate_dataset(
    root: Path,
    *,
    expected_sizes: Mapping[str, int] | None = None,
    balance_tolerance: float = 0.02,
    verify_manifest_hashes: bool = True,
) -> ValidationReport:
    """Validate schema, real visible surfaces, grouping, pairing, and balance."""

    if balance_tolerance < 0:
        raise ValueError("balance_tolerance must be non-negative")
    split_names = ("train", "dev", "test_id", "test_ood")
    manifest = _load_manifest(root)
    transition_profile = "full" if manifest is None else manifest.get("transition_profile", "full")
    if transition_profile == "full":
        expected_event_kinds = frozenset(EventKind)
    elif transition_profile == "set_only":
        expected_event_kinds = frozenset({EventKind.SET, EventKind.NOOP})
    else:
        _fail(f"Manifest has unsupported transition_profile={transition_profile!r}")
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

    # Retain ID-level checks, but do not mistake split-prefixed IDs for proof of
    # leakage control: _validate_surface_partitions checks actual model-visible text.
    for index, left in enumerate(split_names):
        left_entities = {episode.entity_id for episode in split_episodes[left]}
        left_templates = {episode.template_id for episode in split_episodes[left]}
        for right in split_names[index + 1 :]:
            entity_overlap = left_entities & {episode.entity_id for episode in split_episodes[right]}
            template_overlap = left_templates & {episode.template_id for episode in split_episodes[right]}
            if entity_overlap:
                _fail(f"Entity ID leakage between {left} and {right}: {sorted(entity_overlap)[:3]}")
            if template_overlap:
                _fail(f"Template ID leakage between {left} and {right}: {sorted(template_overlap)[:3]}")
    _validate_surface_partitions(split_episodes, manifest)

    ood_counts = Counter(episode.ood_group for episode in split_episodes["test_ood"])
    if set(ood_counts) != set(OOD_GROUPS) or len(set(ood_counts.values())) != 1:
        _fail(f"test_ood must be evenly divided among {list(OOD_GROUPS)}, got {dict(ood_counts)}")
    for split in ("train", "dev", "test_id"):
        if any(episode.ood_group is not None for episode in split_episodes[split]):
            _fail(f"{split} unexpectedly contains ood_group labels")

    reports = {
        split: _validate_split(
            split,
            episodes,
            balance_tolerance=balance_tolerance,
            expected_event_kinds=expected_event_kinds,
            enforce_canonical_payload_balance=transition_profile == "full",
        )
        for split, episodes in split_episodes.items()
    }
    return ValidationReport(
        valid=True,
        splits=reports,
        total_episodes=sum(report.episodes for report in reports.values()),
        total_queries=sum(report.queries for report in reports.values()),
    )
