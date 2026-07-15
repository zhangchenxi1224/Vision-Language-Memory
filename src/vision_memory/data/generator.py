"""Deterministic programmatic episodes for recurrent visual-memory experiments."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .schema import EventKind, Episode, QuerySpec, Turn, TurnType, write_jsonl


NO_ACTIVE_PREFERENCE = "no active preference"
OOD_GROUPS = ("heldout_entity", "heldout_topic", "heldout_paraphrase", "heldout_length")

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


def _event_text(
    kind: EventKind,
    entity: str,
    topic: str,
    value: str | None,
    style: int,
    *,
    paraphrase: bool = False,
) -> str:
    if kind is EventKind.NOOP:
        distractors = (
            "The hallway clock was repaired yesterday.",
            "A delivery truck stopped outside at noon.",
            "The meeting agenda contains three unrelated items.",
            "Rain is expected in another city this weekend.",
        )
        return distractors[style % len(distractors)]
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
        return forms[style % len(forms)]
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
    return forms[style % len(forms)]


def _query(
    entity: str,
    topic: str,
    target: str,
    rng: random.Random,
    *,
    matched_values: tuple[str, str, str, str],
) -> QuerySpec:
    if target == NO_ACTIVE_PREFERENCE:
        choices = [NO_ACTIVE_PREFERENCE, *matched_values[:3]]
    else:
        if target not in matched_values:
            raise ValueError("Counterfactual target must be present in the pair-matched choices")
        choices = list(matched_values)
    rng.shuffle(choices)
    return QuerySpec(
        text=f"What is the current {topic} preference for {entity}? Choose exactly one option.",
        choices=tuple(choices),  # type: ignore[arg-type]
        target_index=choices.index(target),
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
) -> tuple[Turn, ...]:
    set_initial = Turn(
        TurnType.EVENT,
        EventKind.SET,
        _event_text(EventKind.SET, entity, topic, initial, style, paraphrase=paraphrase),
    )
    overwrite = Turn(
        TurnType.EVENT,
        EventKind.OVERWRITE,
        _event_text(EventKind.OVERWRITE, entity, topic, final, style + 1, paraphrase=paraphrase),
    )
    noop = Turn(
        TurnType.EVENT,
        EventKind.NOOP,
        _event_text(EventKind.NOOP, entity, topic, None, style),
    )
    query_initial = Turn(
        TurnType.QUERY,
        query=_query(entity, topic, initial, rng, matched_values=matched_values),
    )
    query_final = Turn(
        TurnType.QUERY,
        query=_query(entity, topic, final, rng, matched_values=matched_values),
    )

    if pattern == 0:
        mixed = Turn(
            TurnType.MIXED,
            EventKind.NOOP,
            _event_text(EventKind.NOOP, entity, topic, None, style + 1),
            _query(entity, topic, final, rng, matched_values=matched_values),
        )
        turns = [set_initial, noop, query_initial, overwrite, mixed]
    elif pattern == 1:
        turns = [set_initial, query_initial, noop, overwrite, query_final]
    elif pattern == 2:
        clear = Turn(
            TurnType.EVENT,
            EventKind.CLEAR,
            _event_text(EventKind.CLEAR, entity, topic, None, style, paraphrase=paraphrase),
        )
        query_clear = Turn(
            TurnType.QUERY,
            query=_query(
                entity,
                topic,
                NO_ACTIVE_PREFERENCE,
                rng,
                matched_values=matched_values,
            ),
        )
        set_final_mixed = Turn(
            TurnType.MIXED,
            EventKind.SET,
            _event_text(EventKind.SET, entity, topic, final, style + 1, paraphrase=paraphrase),
            _query(entity, topic, final, rng, matched_values=matched_values),
        )
        turns = [set_initial, clear, query_clear, set_final_mixed]
    else:
        turns = [set_initial, overwrite, noop, query_final]

    if long_length is not None:
        if not 9 <= long_length <= 16:
            raise ValueError("long_length must be in [9, 16]")
        # Insert forced no-op updater calls before the final read. The final read remains
        # semantically unchanged, which gives a controlled length/distractor challenge.
        while len(turns) < long_length:
            insertion = max(1, len(turns) - 1)
            turns.insert(
                insertion,
                Turn(
                    TurnType.EVENT,
                    EventKind.NOOP,
                    _event_text(EventKind.NOOP, entity, topic, None, style + len(turns)),
                ),
            )
    return tuple(turns)


def _rebalance_target_positions(episodes: list[Episode], *, seed: int) -> list[Episode]:
    """Place correct choices round-robin without changing their semantic target."""

    query_count = sum(episode.query_count for episode in episodes)
    target_positions = [index % 4 for index in range(query_count)]
    random.Random(seed).shuffle(target_positions)
    query_number = 0
    balanced: list[Episode] = []
    for episode in episodes:
        turns: list[Turn] = []
        for turn in episode.turns:
            if turn.query is None:
                turns.append(turn)
                continue
            query = turn.query
            target = query.target
            remaining = [choice for choice in query.choices if choice != target]
            desired_index = target_positions[query_number]
            choices = remaining.copy()
            choices.insert(desired_index, target)
            updated_query = replace(query, choices=tuple(choices), target_index=desired_index)
            turns.append(replace(turn, query=updated_query))
            query_number += 1
        balanced.append(replace(episode, turns=tuple(turns)))
    return balanced


def _split_episodes(split: str, size: int, global_seed: int) -> list[Episode]:
    group_size = size // len(OOD_GROUPS) if split == "test_ood" else None
    episodes: list[Episode] = []
    for episode_index in range(0, size, 2):
        if split == "test_ood":
            group = OOD_GROUPS[min(episode_index // int(group_size), len(OOD_GROUPS) - 1)]
        else:
            group = None
        pair_number = episode_index // 2
        pair_id = f"{split}-pair-{pair_number:06d}"
        entity_id = f"{split}-{group or 'id'}-entity-{pair_number:06d}"
        entity_families = HELDOUT_ENTITY_FAMILIES if group == "heldout_entity" else BASE_ENTITY_FAMILIES
        entity_family = entity_families[pair_number % len(entity_families)]
        entity = f"{entity_family} {pair_number}"
        style = pair_number % 6
        pattern = pair_number % 4
        template_tag = "paraphrase" if group == "heldout_paraphrase" else "base"
        template_id = f"{split}-{group or 'id'}-{template_tag}-template-{style:02d}-{pattern}"
        topic_bank = HELDOUT_TOPICS if group == "heldout_topic" else TOPICS
        topic_names = tuple(topic_bank)
        topic = topic_names[pair_number % len(topic_names)]
        values = topic_bank[topic]
        pair_rng = random.Random(global_seed * 1_000_003 + pair_number + sum(map(ord, split)))
        selected = pair_rng.sample(values, 4)
        long_length = 9 + (pair_number % 8) if group == "heldout_length" else None

        ids = (f"{split}-{episode_index:07d}", f"{split}-{episode_index + 1:07d}")
        for variant in range(2):
            episode_id = ids[variant]
            episode_seed = global_seed * 10_000_019 + episode_index + variant
            rng = random.Random(episode_seed)
            turns = _build_turns(
                entity=entity,
                topic=topic,
                initial=selected[variant],
                final=selected[variant + 2],
                pattern=pattern,
                style=style + (2 if group == "heldout_paraphrase" else 0),
                rng=rng,
                long_length=long_length,
                matched_values=tuple(selected),  # type: ignore[arg-type]
                paraphrase=group == "heldout_paraphrase",
            )
            episodes.append(
                Episode(
                    episode_id=episode_id,
                    split=split,
                    seed=episode_seed,
                    entity_id=entity_id,
                    template_id=template_id,
                    turns=turns,
                    pair_id=pair_id,
                    counterfactual_episode_id=ids[1 - variant],
                    topic=topic,
                    ood_group=group,
                )
            )
    return _rebalance_target_positions(episodes, seed=global_seed + sum(map(ord, split)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_dataset(output_dir: Path, *, sizes: DatasetSizes = DatasetSizes(), seed: int = 2026) -> dict[str, Any]:
    """Generate the four fixed JSONL splits and a content-addressed manifest."""

    sizes.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict[str, Any]] = {}
    for split, size in sizes.as_dict().items():
        episodes = _split_episodes(split, size, seed)
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, episodes)
        split_files[split] = {
            "path": path.name,
            "episodes": len(episodes),
            "queries": sum(episode.query_count for episode in episodes),
            "sha256": _sha256(path),
        }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "generator": "vision_memory.data.generator",
        "seed": seed,
        "sizes": sizes.as_dict(),
        "splits": split_files,
        "ood_groups": list(OOD_GROUPS),
        "base_entity_families": list(BASE_ENTITY_FAMILIES),
        "heldout_entity_families": list(HELDOUT_ENTITY_FAMILIES),
        "heldout_topics": list(HELDOUT_TOPICS),
        "hidden_ledger_serialized": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
