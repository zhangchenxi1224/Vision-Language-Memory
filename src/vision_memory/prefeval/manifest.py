"""Canonical PrefEval inventory and deterministic adaptation split policy."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass


TOPICS = (
    "education_learning_styles",
    "education_resources",
    "entertain_games",
    "entertain_music_book",
    "entertain_shows",
    "entertain_sports",
    "lifestyle_beauty",
    "lifestyle_dietary",
    "lifestyle_fit",
    "lifestyle_health",
    "pet_ownership",
    "professional_work_location_style",
    "shop_fashion",
    "shop_home",
    "shop_motors",
    "shop_technology",
    "travel_activities",
    "travel_hotel",
    "travel_restaurant",
    "travel_transportation",
)

FORMS = ("explicit", "implicit_choice", "implicit_persona")
FORCED_WRITE_COUNTS = (0, 2, 5, 10)
ADAPTATION_SEED = 2026
OPTION_SHUFFLE_SEED = 41
CHOICES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class TopicSplit:
    """The preregistered 16-topic adaptation / four-topic OOD split."""

    adaptation_topics: tuple[str, ...]
    ood_topics: tuple[str, ...]
    seed: int = ADAPTATION_SEED


def adaptation_topic_split(seed: int = ADAPTATION_SEED) -> TopicSplit:
    """Shuffle the fixed manifest once and reserve the first four topics for OOD."""

    shuffled = list(TOPICS)
    random.Random(seed).shuffle(shuffled)
    ood = tuple(shuffled[:4])
    ood_set = set(ood)
    adaptation = tuple(topic for topic in TOPICS if topic not in ood_set)
    return TopicSplit(adaptation_topics=adaptation, ood_topics=ood, seed=seed)


def _stable_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{seed}\x1f{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def assign_base_pair_splits(
    base_pair_ids_by_topic: dict[str, Iterable[str]],
    *,
    seed: int = ADAPTATION_SEED,
    dev_fraction: float = 0.1,
) -> dict[str, str]:
    """Assign whole base pairs, and therefore all three forms, to train/dev/OOD."""

    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be strictly between zero and one")
    topic_split = adaptation_topic_split(seed)
    ood_topics = set(topic_split.ood_topics)
    unknown = sorted(set(base_pair_ids_by_topic) - set(TOPICS))
    if unknown:
        raise ValueError(f"Unknown PrefEval topics: {', '.join(unknown)}")

    result: dict[str, str] = {}
    for topic in TOPICS:
        pair_ids = sorted(set(base_pair_ids_by_topic.get(topic, ())))
        if not pair_ids:
            continue
        if topic in ood_topics:
            for pair_id in pair_ids:
                result[pair_id] = "adapt_ood"
            continue
        shuffled = pair_ids[:]
        random.Random(_stable_seed(seed, topic)).shuffle(shuffled)
        dev_count = max(1, round(len(shuffled) * dev_fraction))
        dev_ids = set(shuffled[:dev_count])
        for pair_id in pair_ids:
            result[pair_id] = "adapt_dev" if pair_id in dev_ids else "adapt_train"
    return result


__all__ = [
    "ADAPTATION_SEED",
    "CHOICES",
    "FORCED_WRITE_COUNTS",
    "FORMS",
    "OPTION_SHUFFLE_SEED",
    "TOPICS",
    "TopicSplit",
    "adaptation_topic_split",
    "assign_base_pair_splits",
]
