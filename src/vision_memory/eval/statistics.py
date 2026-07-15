"""Paired hierarchical bootstrap and family-wise Holm correction."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .metrics import correctness


DEFAULT_PAIR_FIELDS = (
    "base_pair_id",
    "form",
    "condition",
    "protocol",
    "forced_write_k",
    "seed",
)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_hierarchical_bootstrap(
    records: Iterable[Mapping[str, Any]],
    *,
    method_a: str,
    method_b: str,
    iterations: int = 10_000,
    seed: int = 2026,
    method_field: str = "method",
    subtype_field: str = "subtype",
    pair_fields: Sequence[str] = DEFAULT_PAIR_FIELDS,
    confidence: float = 0.95,
    strict_pairs: bool = True,
) -> dict[str, Any]:
    """Bootstrap paired correctness deltas, clustering by topic and subtype.

    Topics are sampled with replacement. Within every sampled topic/subtype cell,
    paired base units are sampled with replacement. Cell means then receive equal
    weight, matching a topic-by-subtype macro estimand.
    """

    if method_a == method_b:
        raise ValueError("method_a and method_b must differ")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")

    method_maps: dict[str, dict[tuple[Any, ...], tuple[str, str, float]]] = {
        method_a: {},
        method_b: {},
    }
    for row in records:
        method = row.get(method_field)
        if method not in method_maps:
            continue
        topic = row.get("topic")
        if not isinstance(topic, str) or not topic:
            raise ValueError("Every bootstrap record must have a non-empty topic")
        subtype_value = row.get(subtype_field, row.get("form"))
        if subtype_value is None:
            raise ValueError(f"Every bootstrap record needs {subtype_field!r} or 'form'")
        subtype = str(subtype_value)
        key = tuple(row.get(field, "standard" if field == "condition" else None) for field in pair_fields)
        if key in method_maps[str(method)]:
            raise ValueError(f"Duplicate {method!r} bootstrap record for pair key {key}")
        method_maps[str(method)][key] = (topic, subtype, correctness(row))

    keys_a = set(method_maps[method_a])
    keys_b = set(method_maps[method_b])
    if strict_pairs and keys_a != keys_b:
        raise ValueError(
            f"Unpaired bootstrap inputs: {len(keys_a - keys_b)} only in {method_a}, "
            f"{len(keys_b - keys_a)} only in {method_b}"
        )
    paired_keys = sorted(keys_a & keys_b, key=repr)
    if not paired_keys:
        raise ValueError(f"No paired records found for {method_a!r} versus {method_b!r}")

    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key in paired_keys:
        topic_a, subtype_a, score_a = method_maps[method_a][key]
        topic_b, subtype_b, score_b = method_maps[method_b][key]
        if (topic_a, subtype_a) != (topic_b, subtype_b):
            raise ValueError(f"Pair {key} disagrees on topic/subtype across methods")
        strata[topic_a][subtype_a].append(score_a - score_b)

    observed_cells = [
        sum(values) / len(values)
        for subtype_values in strata.values()
        for values in subtype_values.values()
    ]
    observed = sum(observed_cells) / len(observed_cells)

    rng = random.Random(seed)
    topics = sorted(strata)
    draws: list[float] = []
    for _ in range(iterations):
        sampled_cell_means: list[float] = []
        for topic in rng.choices(topics, k=len(topics)):
            for subtype in sorted(strata[topic]):
                values = strata[topic][subtype]
                sampled = rng.choices(values, k=len(values))
                sampled_cell_means.append(sum(sampled) / len(sampled))
        draws.append(sum(sampled_cell_means) / len(sampled_cell_means))
    draws.sort()
    tail = (1.0 - confidence) / 2.0
    lower = _percentile(draws, tail)
    upper = _percentile(draws, 1.0 - tail)
    less_or_equal_zero = sum(value <= 0.0 for value in draws)
    greater_or_equal_zero = sum(value >= 0.0 for value in draws)
    p_value = min(
        1.0,
        2.0
        * min(
            (less_or_equal_zero + 1) / (iterations + 1),
            (greater_or_equal_zero + 1) / (iterations + 1),
        ),
    )
    return {
        "method_a": method_a,
        "method_b": method_b,
        "estimand": "topic_subtype_macro_accuracy_delta",
        "observed_delta": observed,
        "confidence": confidence,
        "ci_lower": lower,
        "ci_upper": upper,
        "two_sided_p_value": p_value,
        "iterations": iterations,
        "seed": seed,
        "n_pairs": len(paired_keys),
        "n_topics": len(strata),
        "n_topic_subtype_cells": len(observed_cells),
        "unpaired_a": len(keys_a - keys_b),
        "unpaired_b": len(keys_b - keys_a),
    }


def holm_correction(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, dict[str, Any]]:
    """Return monotone Holm-adjusted p-values and step-down decisions."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if not p_values:
        return {}
    for name, value in p_values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid p-value for {name!r}: {value!r}")
    ordered = sorted(((name, float(value)) for name, value in p_values.items()), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted_by_name: dict[str, float] = {}
    running_adjusted = 0.0
    still_rejecting = True
    rejected_by_name: dict[str, bool] = {}
    for rank, (name, value) in enumerate(ordered, start=1):
        multiplier = total - rank + 1
        running_adjusted = max(running_adjusted, min(1.0, multiplier * value))
        adjusted_by_name[name] = running_adjusted
        threshold = alpha / multiplier
        rejected = still_rejecting and value <= threshold
        rejected_by_name[name] = rejected
        if not rejected:
            still_rejecting = False
    return {
        name: {
            "raw_p_value": float(p_values[name]),
            "adjusted_p_value": adjusted_by_name[name],
            "rejected": rejected_by_name[name],
            "alpha": alpha,
        }
        for name in p_values
    }


__all__ = ["DEFAULT_PAIR_FIELDS", "holm_correction", "paired_hierarchical_bootstrap"]
