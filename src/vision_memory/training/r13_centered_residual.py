"""Model-free contracts for the R13 mean-centered residual-writer experiment.

R13 is a surgical follow-up to R12.  It preserves the one-SET F1 data,
schedule, Reader metrics, reset control, and wrong-event donor control while
removing the shortcut observed in R12: a trainable event-independent visual
code.  The trainable branch is constrained to have exactly zero mean over the
fixed 144-event training set and is added to a frozen R12 common visual base.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Mapping, Sequence

from .r5_compose import R5Segment, canonical_sha256
from .r12_shared_writer import (
    R12_DEV_VALUE_COUNT,
    conditioned_target_gate,
    conditioned_target_statistics,
    target_value,
)


R13_PROTOCOL = "R13-Mean-Centered-Conditional-Residual-Writer"
R13_FRESH_FINAL_SELECTION_SEED = 20260904
R13_FRESH_DEV_FINAL_COUNT = R12_DEV_VALUE_COUNT

# Locked by the deterministic, model-free selection audit before any R13 model
# execution.  Tests can explicitly request an unlocked synthetic selection.
R13_FRESH_DEV_FINAL_SHA256: str | None = "87c728758aa0d8eec0544f24e85561c2ec7dd608aba770962db55ba72e3e07c6"


def _digest(seed: int, namespace: str, *parts: object) -> bytes:
    payload = "\x1f".join((R13_PROTOCOL, str(seed), namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode()).digest()


def _validate_pool(pool: Sequence[R5Segment]) -> tuple[R5Segment, ...]:
    values = tuple(pool)
    if not values:
        raise ValueError("R13 received an empty F1 pool.")
    if len({segment.segment_id for segment in values}) != len(values):
        raise ValueError("R13 F1 pool contains duplicate segment IDs.")
    for segment in values:
        target_value(segment)
    return values


def select_fresh_dev_final_f1(
    pool: Sequence[R5Segment],
    *,
    excluded_entities: Sequence[str] | set[str],
    seed: int = R13_FRESH_FINAL_SELECTION_SEED,
    expected_value_count: int = R13_FRESH_DEV_FINAL_COUNT,
    enforce_locked: bool = True,
) -> tuple[R5Segment, ...]:
    """Select a fresh entity-disjoint, value-complete, position-balanced final set.

    Selection uses IDs and structural labels only; no model output participates.
    Exactly one item is selected for every dev target value, with 6/6/6/6
    answer positions for the locked 24-value F1 dev vocabulary.
    """

    values = _validate_pool(pool)
    excluded = set(excluded_entities)
    target_values = sorted(
        {target_value(segment) for segment in values},
        key=lambda value: (_digest(seed, "value-order", value), value),
    )
    if len(target_values) != expected_value_count:
        raise ValueError(f"R13 fresh final requires {expected_value_count} target values, got {len(target_values)}.")
    if expected_value_count % 4:
        raise ValueError("R13 fresh-final value count must be divisible by four.")

    candidates: dict[str, tuple[R5Segment, ...]] = {}
    for value in target_values:
        ordered = sorted(
            (
                segment
                for segment in values
                if target_value(segment) == value and segment.query_entity_id not in excluded
            ),
            key=lambda segment: (
                _digest(
                    seed,
                    "candidate",
                    value,
                    segment.query.target_index,
                    segment.query_entity_id,
                    segment.segment_id,
                ),
                segment.segment_id,
            ),
        )
        unique: dict[tuple[str, int], R5Segment] = {}
        for segment in ordered:
            unique.setdefault((segment.query_entity_id, segment.query.target_index), segment)
        candidates[value] = tuple(unique.values())
        if not candidates[value]:
            raise ValueError(f"R13 has no fresh candidate for dev value {value!r}.")

    remaining = {position: expected_value_count // 4 for position in range(4)}
    used_entities = set(excluded)
    assignment: dict[str, R5Segment] = {}

    def options(value: str) -> list[R5Segment]:
        return [
            segment
            for segment in candidates[value]
            if segment.query_entity_id not in used_entities and remaining[segment.query.target_index] > 0
        ]

    def feasible(unassigned: Sequence[str]) -> bool:
        if any(not options(value) for value in unassigned):
            return False
        for position in range(4):
            eligible = sum(
                any(
                    segment.query_entity_id not in used_entities and segment.query.target_index == position
                    for segment in candidates[value]
                )
                for value in unassigned
            )
            if eligible < remaining[position]:
                return False
        return True

    def solve() -> bool:
        if len(assignment) == len(target_values):
            return all(value == 0 for value in remaining.values())
        unassigned = [value for value in target_values if value not in assignment]
        value = min(
            unassigned,
            key=lambda item: (len(options(item)), _digest(seed, "search-value", item), item),
        )
        ordered = sorted(
            options(value),
            key=lambda segment: (
                -remaining[segment.query.target_index],
                _digest(seed, "search-candidate", value, segment.segment_id),
                segment.segment_id,
            ),
        )
        for segment in ordered:
            position = segment.query.target_index
            assignment[value] = segment
            used_entities.add(segment.query_entity_id)
            remaining[position] -= 1
            rest = [item for item in unassigned if item != value]
            if feasible(rest) and solve():
                return True
            remaining[position] += 1
            used_entities.remove(segment.query_entity_id)
            del assignment[value]
        return False

    if not solve():
        raise ValueError("R13 cannot satisfy the fresh-final entity/position constraints.")
    selected = tuple(assignment[value] for value in target_values)
    expected_positions = Counter({position: expected_value_count // 4 for position in range(4)})
    if Counter(segment.query.target_index for segment in selected) != expected_positions:
        raise RuntimeError("R13 fresh final lost exact choice-position balance.")
    if len({segment.query_entity_id for segment in selected}) != len(selected):
        raise RuntimeError("R13 fresh final contains duplicate entities.")
    if {segment.query_entity_id for segment in selected} & excluded:
        raise RuntimeError("R13 fresh final overlaps a previously exposed entity.")
    if enforce_locked:
        if seed != R13_FRESH_FINAL_SELECTION_SEED or expected_value_count != R13_FRESH_DEV_FINAL_COUNT:
            raise ValueError("Locked R13 fresh-final selection forbids seed/count drift.")
        observed = canonical_sha256([segment.to_dict() for segment in selected])
        if R13_FRESH_DEV_FINAL_SHA256 is None:
            raise RuntimeError(f"R13 fresh-final hash has not been preregistered; observed {observed}.")
        if observed != R13_FRESH_DEV_FINAL_SHA256:
            raise RuntimeError(
                f"R13 fresh-final selection drifted: expected {R13_FRESH_DEV_FINAL_SHA256}, observed {observed}."
            )
    return selected


def centered_target_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    target_segment_id: str,
    endpoint: str,
) -> dict[str, Any]:
    """Apply the unchanged R12 gate and add attribution to the frozen base."""

    result = conditioned_target_statistics(
        rows,
        suite=suite,
        target_segment_id=target_segment_id,
        endpoint=endpoint,
    )

    def cell(checkpoint: str, condition: str) -> list[Mapping[str, Any]]:
        selected = [
            row
            for row in rows
            if row.get("suite") == suite
            and row.get("pair_unit") == target_segment_id
            and row.get("checkpoint") == checkpoint
            and row.get("condition") == condition
        ]
        if len(selected) != 4:
            raise ValueError(
                f"R13 expected four views for {target_segment_id}/{checkpoint}/{condition}, got {len(selected)}."
            )
        return sorted(selected, key=lambda row: int(row["view_index"]))

    normal_m0 = cell("m0", "normal")
    base_m0 = cell("m0", "base")
    normal_endpoint = cell(endpoint, "normal")
    base_endpoint = cell(endpoint, "base")

    def mean(name: str, values: Sequence[Mapping[str, Any]]) -> float:
        return sum(float(row[name]) for row in values) / len(values)

    normal_ce = mean("ce", normal_endpoint)
    base_ce = mean("ce", base_endpoint)
    normal_accuracy = mean("correct", normal_endpoint)
    base_accuracy = mean("correct", base_endpoint)
    m0_difference = mean("ce", normal_m0) - mean("ce", base_m0)
    endpoint_difference = normal_ce - base_ce
    result.update(
        {
            "base_ce": base_ce,
            "base_accuracy": base_accuracy,
            "relative_normal_ce_vs_base": (normal_ce - base_ce) / max(base_ce, 1e-12),
            "normal_accuracy_delta_vs_base": normal_accuracy - base_accuracy,
            "normal_better_than_base_views": sum(
                float(normal["ce"]) < float(base["ce"]) for normal, base in zip(normal_endpoint, base_endpoint)
            ),
            "normal_base_difference_in_differences": endpoint_difference - m0_difference,
        }
    )
    return result


def centered_target_gate(statistics: Mapping[str, Any], *, technical_gate: bool) -> bool:
    """Require R12 normal/reset/donor gates plus a same-strength base gate."""

    return bool(
        conditioned_target_gate(statistics, technical_gate=technical_gate)
        and float(statistics["relative_normal_ce_vs_base"]) <= -0.20
        and int(statistics["normal_better_than_base_views"]) == 4
        and float(statistics["normal_accuracy_delta_vs_base"]) >= 0.25
        and float(statistics["normal_base_difference_in_differences"]) < 0.0
    )
