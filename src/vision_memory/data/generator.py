"""Deterministic programmatic episodes for recurrent visual-memory experiments."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .schema import (
    DistractorVariant,
    Episode,
    EventKind,
    QuerySpec,
    Turn,
    TurnType,
    surface_template_signatures,
    write_jsonl,
)


NO_ACTIVE_PREFERENCE = "no active preference"
OOD_GROUPS = ("heldout_entity", "heldout_topic", "heldout_paraphrase", "heldout_length")
TRANSITION_PROFILES = ("full", "set_only")

TOPICS: dict[str, tuple[str, ...]] = {
    "color": ("red", "blue", "green", "yellow", "purple", "orange", "white", "black"),
    "material": ("wood", "glass", "steel", "ceramic", "linen", "leather", "stone", "paper"),
    "drink": ("tea", "coffee", "water", "juice", "cocoa", "milk", "lemonade", "soda"),
    "style": ("minimal", "vintage", "modern", "rustic", "formal", "playful", "sporty", "classic"),
    "meal": ("pasta", "salad", "curry", "soup", "tacos", "rice", "pizza", "sandwich"),
    "music": ("jazz", "classical", "folk", "rock", "ambient", "blues", "pop", "electronic"),
}

HELDOUT_TOPICS: dict[str, tuple[str, ...]] = {
    "fragrance": ("citrus", "cedar", "vanilla", "mint", "lavender", "rose", "amber", "pine"),
    "lighting": ("warm", "cool", "dim", "bright", "soft", "focused", "natural", "colored"),
}

BASE_ENTITY_FAMILIES = ("mug", "lamp", "chair", "backpack", "notebook", "desk", "room", "device")
HELDOUT_ENTITY_FAMILIES = ("telescope", "violin", "greenhouse", "statue", "drone", "kayak")

# These are literal model-visible markers, not renamed metadata IDs.  Every event
# and query is prefixed by its marker and each split owns a disjoint family bank.
SPLIT_TEMPLATE_FAMILIES: dict[str, tuple[str, ...]] = {
    "train": ("memory memo", "choice profile", "preference register", "selection note"),
    "dev": ("user brief", "option record", "taste card", "choice docket"),
    "test_id": ("preference snapshot", "decision file", "selection profile", "saved choice note"),
    "test_ood": ("choice bulletin", "preference folio", "selection ledger", "user option sheet"),
}
PARAPHRASE_TEMPLATE_FAMILIES = (
    "inclination capsule",
    "favoring digest",
    "selection abstract",
    "choice précis",
)


@dataclass(frozen=True)
class DatasetSizes:
    train: int = 5_000
    dev: int = 500
    test_id: int = 1_000
    test_ood: int = 1_000

    def as_dict(self) -> dict[str, int]:
        return {
            "train": self.train,
            "dev": self.dev,
            "test_id": self.test_id,
            "test_ood": self.test_ood,
        }

    def validate(self) -> None:
        for split, size in self.as_dict().items():
            if size <= 0 or size % 2:
                raise ValueError(f"{split} size must be a positive even number, got {size}")
        if self.test_ood % (2 * len(OOD_GROUPS)):
            raise ValueError("test_ood size must be divisible by 8 so every OOD group contains matched pairs")


def _surface_marker(template_family: str) -> str:
    return f"{template_family.title()}:"


def _event_text(
    kind: EventKind,
    entity: str,
    topic: str,
    value: str | None,
    style: int,
    template_family: str,
    *,
    paraphrase: bool = False,
) -> str:
    marker = _surface_marker(template_family)
    if kind is EventKind.NOOP:
        distractors = (
            "the hallway clock was repaired yesterday.",
            "a delivery truck stopped outside at noon.",
            "the meeting agenda contains three unrelated items.",
            "rain is expected in another city this weekend.",
        )
        return f"{marker} Unrelated update: {distractors[style % len(distractors)]}"
    if kind is EventKind.CLEAR:
        forms = (
            (
                f"Treat {entity} as having no favored option in the {topic} category.",
                f"The earlier {topic} inclination associated with {entity} no longer applies.",
            )
            if paraphrase
            else (
                f"Forget any saved {topic} preference for {entity}.",
                f"Clear the remembered {topic} choice for {entity}.",
            )
        )
        return f"{marker} {forms[style % len(forms)]}"
    if value is None:
        raise ValueError(f"{kind.value} requires a value")
    forms = (
        (
            f"When deciding the {topic} category for {entity}, favor {value} over the alternatives.",
            f"Associate {entity}'s most suitable {topic} selection with {value} from this point onward.",
            f"If {entity} comes up later, resolve its {topic} inclination in favor of {value}.",
        )
        if paraphrase
        else (
            f"For {entity}, the preferred {topic} is {value}.",
            f"Remember that {entity} should use {value} as its {topic} choice.",
            f"The current {topic} preference for {entity} is now {value}.",
        )
    )
    return f"{marker} {forms[style % len(forms)]}"


def _query(
    entity: str,
    topic: str,
    target: str,
    rng: random.Random,
    *,
    matched_values: tuple[str, str, str, str],
    template_family: str,
    comparison_id: str,
) -> QuerySpec:
    if target == NO_ACTIVE_PREFERENCE:
        choices = [NO_ACTIVE_PREFERENCE, *matched_values[:3]]
    else:
        if target not in matched_values:
            raise ValueError("Counterfactual target must be present in the pair-matched choices")
        choices = list(matched_values)
    rng.shuffle(choices)
    marker = _surface_marker(template_family)
    return QuerySpec(
        text=f"{marker} What is the current {topic} preference for {entity}? Choose exactly one option.",
        choices=tuple(choices),  # type: ignore[arg-type]
        target_index=choices.index(target),
        comparison_id=comparison_id,
    )


def _build_turns(
    *,
    entity: str,
    topic: str,
    initial: str,
    final: str,
    pattern: int,
    style: int,
    rng: random.Random,
    long_length: int | None,
    matched_values: tuple[str, str, str, str],
    paraphrase: bool,
    template_family: str,
    comparison_prefix: str,
    include_distractors: bool,
    transition_profile: str,
) -> tuple[Turn, ...]:
    def event(kind: EventKind, value: str | None, offset: int = 0) -> Turn:
        return Turn(
            TurnType.EVENT,
            kind,
            _event_text(
                kind,
                entity,
                topic,
                value,
                style + offset,
                template_family,
                paraphrase=paraphrase,
            ),
        )

    set_initial = event(EventKind.SET, initial)
    overwrite = event(EventKind.OVERWRITE, final, 1)
    noop = event(EventKind.NOOP, None)
    query_initial = Turn(
        TurnType.QUERY,
        query=_query(
            entity,
            topic,
            initial,
            rng,
            matched_values=matched_values,
            template_family=template_family,
            comparison_id=f"{comparison_prefix}:q0",
        ),
    )
    query_final = Turn(
        TurnType.QUERY,
        query=_query(
            entity,
            topic,
            initial if transition_profile == "set_only" else final,
            rng,
            matched_values=matched_values,
            template_family=template_family,
            comparison_id=f"{comparison_prefix}:q1",
        ),
    )

    if transition_profile == "set_only":
        # This is a separately generated curriculum, not a turn-filtered view of
        # the full transition corpus.  Every state-changing event reaffirms the
        # same preference, so both reads remain semantically valid.  Clean and
        # distractor mates retain byte-identical query payloads.
        reaffirm = event(EventKind.SET, initial, 1)
        if pattern % 2 == 0:
            if include_distractors:
                mixed = Turn(
                    TurnType.MIXED,
                    EventKind.NOOP,
                    _event_text(
                        EventKind.NOOP,
                        entity,
                        topic,
                        None,
                        style + 1,
                        template_family,
                    ),
                    query_final.query,
                )
                turns = [set_initial, noop, query_initial, mixed]
            else:
                mixed = Turn(
                    TurnType.MIXED,
                    EventKind.SET,
                    _event_text(
                        EventKind.SET,
                        entity,
                        topic,
                        initial,
                        style + 2,
                        template_family,
                        paraphrase=paraphrase,
                    ),
                    query_final.query,
                )
                turns = [set_initial, reaffirm, query_initial, mixed]
        elif include_distractors:
            turns = [set_initial, query_initial, noop, query_final]
        else:
            turns = [set_initial, query_initial, reaffirm, query_final]
    elif transition_profile != "full":
        raise ValueError(
            f"transition_profile must be one of {TRANSITION_PROFILES}, got {transition_profile!r}"
        )

    elif pattern == 0:
        if include_distractors:
            mixed = Turn(
                TurnType.MIXED,
                EventKind.NOOP,
                _event_text(
                    EventKind.NOOP,
                    entity,
                    topic,
                    None,
                    style + 1,
                    template_family,
                ),
                query_final.query,
            )
            turns = [set_initial, noop, query_initial, overwrite, mixed]
        else:
            turns = [set_initial, query_initial, overwrite, query_final]
    elif pattern == 1:
        turns = [set_initial, query_initial]
        if include_distractors:
            turns.append(noop)
        turns.extend((overwrite, query_final))
    elif pattern == 2:
        clear = event(EventKind.CLEAR, None)
        query_clear = Turn(
            TurnType.QUERY,
            query=_query(
                entity,
                topic,
                NO_ACTIVE_PREFERENCE,
                rng,
                matched_values=matched_values,
                template_family=template_family,
                comparison_id=f"{comparison_prefix}:q0",
            ),
        )
        set_final_mixed = Turn(
            TurnType.MIXED,
            EventKind.SET,
            _event_text(
                EventKind.SET,
                entity,
                topic,
                final,
                style + 1,
                template_family,
                paraphrase=paraphrase,
            ),
            _query(
                entity,
                topic,
                final,
                rng,
                matched_values=matched_values,
                template_family=template_family,
                comparison_id=f"{comparison_prefix}:q1",
            ),
        )
        turns = [set_initial]
        if include_distractors:
            turns.append(noop)
        turns.extend((clear, query_clear, set_final_mixed))
    else:
        turns = [set_initial, query_initial, overwrite]
        if include_distractors:
            turns.append(noop)
        turns.append(query_final)

    if long_length is not None:
        if not 9 <= long_length <= 16:
            raise ValueError("long_length must be in [9, 16]")
        # Length-OOD keeps clean/distractor members at the same turn count.  The
        # clean stream receives target-consistent reaffirmations, while its mate
        # receives irrelevant no-op writes.  This is recorded in the manifest so
        # it is not confused with a call-count-matched causal contrast.
        while len(turns) < long_length:
            insertion = max(1, len(turns) - 1)
            if include_distractors:
                filler = event(EventKind.NOOP, None, len(turns))
            elif transition_profile == "set_only":
                filler = event(EventKind.SET, initial, len(turns))
            else:
                filler = event(EventKind.OVERWRITE, final, len(turns))
            turns.insert(insertion, filler)
    return tuple(turns)


def _rebalance_target_positions(episodes: list[Episode], *, seed: int) -> list[Episode]:
    """Balance labels while keeping clean/distractor query choices identical."""

    references: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for episode_index, episode in enumerate(episodes):
        for turn_index, turn in enumerate(episode.turns):
            if turn.query is None:
                continue
            if turn.query.comparison_id is None:
                raise ValueError("Generated queries must contain comparison_id")
            references[turn.query.comparison_id].append((episode_index, turn_index))

    query_count = sum(len(items) for items in references.values())
    if query_count % 4:
        raise ValueError("Generated query count must be divisible by four for exact label balance")
    target_per_position = query_count // 4
    rng = random.Random(seed)
    groups = list(references.items())
    rng.shuffle(groups)
    groups.sort(key=lambda item: len(item[1]), reverse=True)
    position_counts = [0, 0, 0, 0]
    desired_by_group: dict[str, int] = {}
    tie_order = list(range(4))
    rng.shuffle(tie_order)
    tie_rank = {value: index for index, value in enumerate(tie_order)}
    for comparison_id, items in groups:
        weight = len(items)
        candidates = [
            position
            for position in range(4)
            if position_counts[position] + weight <= target_per_position
        ]
        if not candidates:
            raise ValueError("Unable to balance paired target positions without breaking pair alignment")
        desired = min(candidates, key=lambda position: (position_counts[position], tie_rank[position]))
        desired_by_group[comparison_id] = desired
        position_counts[desired] += weight
    if position_counts != [target_per_position] * 4:
        raise ValueError(f"Internal target balancing failure: {position_counts}")

    balanced: list[Episode] = []
    for episode in episodes:
        turns: list[Turn] = []
        for turn in episode.turns:
            if turn.query is None:
                turns.append(turn)
                continue
            query = turn.query
            desired_index = desired_by_group[query.comparison_id]
            target = query.target
            remaining = [choice for choice in query.choices if choice != target]
            choices = remaining.copy()
            choices.insert(desired_index, target)
            updated_query = replace(query, choices=tuple(choices), target_index=desired_index)
            turns.append(replace(turn, query=updated_query))
        balanced.append(replace(episode, turns=tuple(turns)))
    return balanced


def _entity_surface(seed: int, split: str, group_number: int, entity_family: str) -> str:
    token = hashlib.sha256(f"{seed}:{split}:{group_number}".encode()).hexdigest()[:10]
    return f"{entity_family} {token}"


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _split_episodes(
    split: str,
    size: int,
    global_seed: int,
    *,
    transition_profile: str,
) -> list[Episode]:
    segment_size = size // len(OOD_GROUPS) if split == "test_ood" else size
    segments = [(group, segment_size) for group in OOD_GROUPS] if split == "test_ood" else [(None, size)]
    episodes: list[Episode] = []
    episode_number = 0
    group_number = 0

    for ood_group, count in segments:
        remaining = count
        while remaining:
            chunk_size = 4 if remaining >= 4 else 2
            complete_distractor_grid = chunk_size == 4
            pattern = group_number % 4
            style = group_number % 12
            entity_families = (
                HELDOUT_ENTITY_FAMILIES if ood_group == "heldout_entity" else BASE_ENTITY_FAMILIES
            )
            entity_family = entity_families[group_number % len(entity_families)]
            entity = _entity_surface(global_seed, split, group_number, entity_family)
            entity_id = f"{split}-{ood_group or 'id'}-entity-{group_number:06d}"
            family_bank = (
                PARAPHRASE_TEMPLATE_FAMILIES
                if ood_group == "heldout_paraphrase"
                else SPLIT_TEMPLATE_FAMILIES[split]
            )
            template_family = family_bank[group_number % len(family_bank)]
            template_id = (
                f"{split}-{ood_group or 'id'}-{_slug(template_family)}-"
                f"pattern-{pattern}-style-{style % 3}"
            )
            topic_bank = HELDOUT_TOPICS if ood_group == "heldout_topic" else TOPICS
            topic_names = tuple(topic_bank)
            topic = topic_names[group_number % len(topic_names)]
            values = topic_bank[topic]
            pair_rng = random.Random(
                global_seed * 1_000_003 + group_number + sum(map(ord, split))
            )
            selected = pair_rng.sample(values, 4)
            long_length = 9 + (group_number % 8) if ood_group == "heldout_length" else None
            stream_variants = (
                (DistractorVariant.CLEAN, DistractorVariant.DISTRACTOR)
                if complete_distractor_grid
                else (DistractorVariant.UNPAIRED,)
            )
            ids: dict[tuple[int, DistractorVariant], str] = {}
            for semantic_variant in range(2):
                for stream_variant in stream_variants:
                    ids[(semantic_variant, stream_variant)] = f"{split}-{episode_number:07d}"
                    episode_number += 1

            for semantic_variant in range(2):
                for stream_variant in stream_variants:
                    episode_id = ids[(semantic_variant, stream_variant)]
                    counterfactual_id = ids[(1 - semantic_variant, stream_variant)]
                    include_distractors = stream_variant is not DistractorVariant.CLEAN
                    semantic_seed = (
                        global_seed * 10_000_019 + group_number * 2 + semantic_variant
                    )
                    comparison_prefix = (
                        f"{split}-stream-{group_number:06d}-semantic-{semantic_variant}"
                    )
                    turns = _build_turns(
                        entity=entity,
                        topic=topic,
                        initial=selected[semantic_variant],
                        final=selected[semantic_variant + 2],
                        pattern=pattern,
                        style=style + (2 if ood_group == "heldout_paraphrase" else 0),
                        rng=random.Random(semantic_seed),
                        long_length=long_length,
                        matched_values=tuple(selected),  # type: ignore[arg-type]
                        paraphrase=ood_group == "heldout_paraphrase",
                        template_family=template_family,
                        comparison_prefix=comparison_prefix,
                        include_distractors=include_distractors,
                        transition_profile=transition_profile,
                    )
                    if complete_distractor_grid:
                        opposite = (
                            DistractorVariant.DISTRACTOR
                            if stream_variant is DistractorVariant.CLEAN
                            else DistractorVariant.CLEAN
                        )
                        distractor_episode_id = ids[(semantic_variant, opposite)]
                        distractor_pair_id = (
                            f"{split}-stream-{group_number:06d}-semantic-{semantic_variant}"
                        )
                    else:
                        distractor_episode_id = None
                        distractor_pair_id = None
                    episodes.append(
                        Episode(
                            episode_id=episode_id,
                            split=split,
                            seed=(
                                global_seed * 100_000_007
                                + group_number * 10
                                + semantic_variant * 2
                                + int(stream_variant is DistractorVariant.DISTRACTOR)
                            ),
                            entity_id=entity_id,
                            entity_surface=entity,
                            template_id=template_id,
                            template_family=template_family,
                            turns=turns,
                            pair_id=(
                                f"{split}-counterfactual-{group_number:06d}-"
                                f"{stream_variant.value}"
                            ),
                            counterfactual_episode_id=counterfactual_id,
                            distractor_variant=stream_variant,
                            distractor_pair_id=distractor_pair_id,
                            distractor_episode_id=distractor_episode_id,
                            topic=topic,
                            ood_group=ood_group,
                        )
                    )
            group_number += 1
            remaining -= chunk_size

    if len(episodes) != size:
        raise RuntimeError(f"Internal generator error: {split} produced {len(episodes)}, expected {size}")
    return _rebalance_target_positions(episodes, seed=global_seed + sum(map(ord, split)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_sha256(values: set[str]) -> str:
    payload = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _surface_partition(episodes: list[Episode]) -> dict[str, Any]:
    entities = {episode.entity_surface for episode in episodes if episode.entity_surface is not None}
    families = {episode.template_family for episode in episodes if episode.template_family is not None}
    signatures = {
        signature for episode in episodes for signature in surface_template_signatures(episode)
    }
    variants = Counter(
        episode.distractor_variant.value
        for episode in episodes
        if episode.distractor_variant is not None
    )
    return {
        "entity_surface_count": len(entities),
        "entity_surface_sha256": _set_sha256(entities),
        "template_family_count": len(families),
        "template_families": sorted(families),
        "template_family_sha256": _set_sha256(families),
        "surface_template_signature_count": len(signatures),
        "surface_template_signature_sha256": _set_sha256(signatures),
        "mixed_queries": sum(episode.mixed_query_count for episode in episodes),
        "distractor_variants": dict(sorted(variants.items())),
    }


def generate_dataset(
    output_dir: Path,
    *,
    sizes: DatasetSizes = DatasetSizes(),
    seed: int = 2026,
    transition_profile: str = "full",
) -> dict[str, Any]:
    """Generate four fixed JSONL splits plus a content-addressed manifest."""

    sizes.validate()
    if transition_profile not in TRANSITION_PROFILES:
        raise ValueError(
            f"transition_profile must be one of {TRANSITION_PROFILES}, got {transition_profile!r}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict[str, Any]] = {}
    surface_partitions: dict[str, dict[str, Any]] = {}
    for split, size in sizes.as_dict().items():
        episodes = _split_episodes(
            split,
            size,
            seed,
            transition_profile=transition_profile,
        )
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, episodes)
        split_files[split] = {
            "path": path.name,
            "episodes": len(episodes),
            "queries": sum(episode.query_count for episode in episodes),
            "sha256": _sha256(path),
        }
        surface_partitions[split] = _surface_partition(episodes)

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "generator": "vision_memory.data.generator",
        "seed": seed,
        "transition_profile": transition_profile,
        "sizes": sizes.as_dict(),
        "splits": split_files,
        "surface_partitions": surface_partitions,
        "ood_groups": list(OOD_GROUPS),
        "base_entity_families": list(BASE_ENTITY_FAMILIES),
        "heldout_entity_families": list(HELDOUT_ENTITY_FAMILIES),
        "heldout_topics": list(HELDOUT_TOPICS),
        "hidden_ledger_serialized": False,
        "pairing": {
            "counterfactual": "same stream, entity, template, topic, and choices; different final target",
            "distractor": "same semantic stream and queries; clean has no no-op writes",
            "length_ood_clean_filler": (
                "target-consistent set reaffirmation"
                if transition_profile == "set_only"
                else "target-consistent overwrite reaffirmation"
            ),
            "unpaired_policy": "two-episode stratum residue excluded from matched distractor metrics",
        },
        "model_visible_fields": ["turns[].event_text", "turns[].query.text", "turns[].query.choices"],
        "analysis_only_fields": [
            "entity_id",
            "template_id",
            "entity_surface",
            "template_family",
            "pair_id",
            "counterfactual_episode_id",
            "distractor_variant",
            "distractor_pair_id",
            "distractor_episode_id",
            "turns[].query.comparison_id",
            "turns[].query.target_index",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
