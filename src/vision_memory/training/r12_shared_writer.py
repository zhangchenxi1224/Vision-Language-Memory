"""Model-free contracts for the R12 shared event-to-latent writer experiment.

R12 is the first experiment after the R11 per-target latent oracle.  It keeps the
same one-SET F1 boundary, but replaces one trainable latent per item with one
shared event-conditioned writer.  Selection is deterministic, value/choice-position
balanced, and entity-disjoint across the two development subsets.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .r5_compose import R5Segment, canonical_sha256
from .r10_alignment import target_gate, target_statistics


R12_PROTOCOL = "R12-Shared-Event-Latent-Writer"
R12_SELECTION_SEED = 20260831
R12_TRAIN_VALUE_COUNT = 36
R12_DEV_VALUE_COUNT = 24
R12_TRAIN_SEGMENT_COUNT = R12_TRAIN_VALUE_COUNT * 4
R12_TRAIN_AUDIT_COUNT = R12_TRAIN_VALUE_COUNT
R12_DEV_SELECT_COUNT = R12_DEV_VALUE_COUNT
R12_DEV_FINAL_COUNT = R12_DEV_VALUE_COUNT
R12_EPOCHS = 32
R12_GRADIENT_ACCUMULATION = 4
R12_MICRO_STEPS = R12_TRAIN_SEGMENT_COUNT * R12_EPOCHS
R12_OPTIMIZER_STEPS = R12_MICRO_STEPS // R12_GRADIENT_ACCUMULATION
R12_CHECKPOINT_EPOCHS = (0, 8, 16, 24, 32)
R12_CHECKPOINT_STEPS = tuple(
    epoch * R12_TRAIN_SEGMENT_COUNT // R12_GRADIENT_ACCUMULATION
    for epoch in R12_CHECKPOINT_EPOCHS
)
R12_BASIS_COUNT = 48
R12_BASIS_OUTPUT_NORM = 80.0
R12_WRITER_LEARNING_RATE = 1e-3
R12_WEIGHT_DECAY = 1e-4
R12_LATENT_RMS_SOFT_LIMIT = 0.50

# These are filled by the model-free selection lock step before any R12 model
# outcome is observed.  A training entrypoint must fail closed while any value is
# unset; tests can explicitly request an unlocked synthetic selection.
R12_TRAIN_SELECTION_SHA256: str | None = None
R12_TRAIN_AUDIT_SHA256: str | None = None
R12_DEV_SELECT_SHA256: str | None = None
R12_DEV_FINAL_SHA256: str | None = None


def target_value(segment: R5Segment) -> str:
    """Return the visible MCQ target text after validating the F1 boundary."""

    if segment.family != "F1" or len(segment.events) != 1:
        raise ValueError("R12 accepts one-event F1 segments only.")
    index = segment.query.target_index
    if not 0 <= index < len(segment.query.choices):
        raise ValueError("R12 segment has an invalid target index.")
    value = segment.query.choices[index].strip()
    if not value:
        raise ValueError("R12 target value must be non-empty.")
    return value


def _digest(seed: int, namespace: str, *parts: object) -> bytes:
    payload = "\x1f".join((R12_PROTOCOL, str(seed), namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode()).digest()


def _validate_pool(pool: Sequence[R5Segment]) -> tuple[R5Segment, ...]:
    values = tuple(pool)
    if not values:
        raise ValueError("R12 received an empty F1 pool.")
    if len({segment.segment_id for segment in values}) != len(values):
        raise ValueError("R12 F1 pool contains duplicate segment IDs.")
    for segment in values:
        target_value(segment)
    return values


def _payload_sha(segments: Sequence[R5Segment]) -> str:
    return canonical_sha256([segment.to_dict() for segment in segments])


def _assert_hash(name: str, segments: Sequence[R5Segment], expected: str | None) -> None:
    observed = _payload_sha(segments)
    if expected is None:
        raise RuntimeError(f"R12 {name} hash has not been preregistered; observed {observed}.")
    if observed != expected:
        raise RuntimeError(f"R12 {name} selection drifted: expected {expected}, observed {observed}.")


def select_balanced_train_f1(
    pool: Sequence[R5Segment],
    *,
    seed: int = R12_SELECTION_SEED,
    expected_value_count: int = R12_TRAIN_VALUE_COUNT,
    enforce_locked: bool = True,
) -> tuple[tuple[R5Segment, ...], tuple[R5Segment, ...]]:
    """Select one unique-entity segment for every target-value/choice-position cell.

    The returned training audit is a strict subset containing one segment per
    target value and exactly balanced choice positions.
    """

    values = _validate_pool(pool)
    target_values = sorted(
        {target_value(segment) for segment in values},
        key=lambda value: (_digest(seed, "train-value-order", value), value),
    )
    if len(target_values) != expected_value_count:
        raise ValueError(
            f"R12 train pool requires {expected_value_count} target values, got {len(target_values)}."
        )
    by_cell: dict[tuple[str, int], list[R5Segment]] = {}
    for value in target_values:
        for position in range(4):
            candidates = [
                segment
                for segment in values
                if target_value(segment) == value and segment.query.target_index == position
            ]
            by_cell[(value, position)] = sorted(
                candidates,
                key=lambda segment: (
                    _digest(seed, "train-cell", value, position, segment.segment_id),
                    segment.segment_id,
                ),
            )

    used_entities: set[str] = set()
    selected: list[R5Segment] = []
    selected_by_cell: dict[tuple[str, int], R5Segment] = {}
    cells = [(value, position) for value in target_values for position in range(4)]
    cells.sort(key=lambda cell: (_digest(seed, "train-cell-order", *cell), cell))
    for cell in cells:
        candidate = next(
            (
                segment
                for segment in by_cell[cell]
                if segment.query_entity_id not in used_entities
            ),
            None,
        )
        if candidate is None:
            raise ValueError(f"R12 cannot satisfy unique-entity train cell {cell}.")
        used_entities.add(candidate.query_entity_id)
        selected.append(candidate)
        selected_by_cell[cell] = candidate

    selected.sort(key=lambda segment: (_digest(seed, "train-output", segment.segment_id), segment.segment_id))
    audit = tuple(
        selected_by_cell[(value, rank % 4)]
        for rank, value in enumerate(target_values)
    )
    selected_tuple = tuple(selected)
    if len(selected_tuple) != expected_value_count * 4 or len(audit) != expected_value_count:
        raise RuntimeError("R12 balanced train selection has an invalid size.")
    if Counter(segment.query.target_index for segment in selected_tuple) != Counter({0: expected_value_count, 1: expected_value_count, 2: expected_value_count, 3: expected_value_count}):
        raise RuntimeError("R12 train selection lost choice-position balance.")
    if expected_value_count % 4 == 0 and Counter(segment.query.target_index for segment in audit) != Counter({0: expected_value_count // 4, 1: expected_value_count // 4, 2: expected_value_count // 4, 3: expected_value_count // 4}):
        raise RuntimeError("R12 train audit lost choice-position balance.")
    if enforce_locked:
        if seed != R12_SELECTION_SEED or expected_value_count != R12_TRAIN_VALUE_COUNT:
            raise ValueError("Locked R12 train selection forbids seed/count drift.")
        _assert_hash("train", selected_tuple, R12_TRAIN_SELECTION_SHA256)
        _assert_hash("train-audit", audit, R12_TRAIN_AUDIT_SHA256)
    return selected_tuple, audit


def select_entity_disjoint_dev_f1(
    pool: Sequence[R5Segment],
    *,
    seed: int = R12_SELECTION_SEED,
    expected_value_count: int = R12_DEV_VALUE_COUNT,
    enforce_locked: bool = True,
) -> tuple[tuple[R5Segment, ...], tuple[R5Segment, ...]]:
    """Select one dev-select and one sealed dev-final item per target value.

    All selected entities are globally unique.  Choice positions are exactly
    balanced within each split and shifted by one between select and final.
    """

    values = _validate_pool(pool)
    target_values = sorted(
        {target_value(segment) for segment in values},
        key=lambda value: (_digest(seed, "dev-value-order", value), value),
    )
    if len(target_values) != expected_value_count:
        raise ValueError(
            f"R12 dev pool requires {expected_value_count} target values, got {len(target_values)}."
        )

    used_entities: set[str] = set()
    outputs: dict[str, list[R5Segment]] = {"select": [], "final": []}
    for split, offset in (("select", 0), ("final", 1)):
        for rank, value in enumerate(target_values):
            position = (rank + offset) % 4
            candidates = sorted(
                (
                    segment
                    for segment in values
                    if target_value(segment) == value and segment.query.target_index == position
                ),
                key=lambda segment: (
                    _digest(seed, f"dev-{split}", value, position, segment.segment_id),
                    segment.segment_id,
                ),
            )
            candidate = next(
                (
                    segment
                    for segment in candidates
                    if segment.query_entity_id not in used_entities
                ),
                None,
            )
            if candidate is None:
                raise ValueError(
                    f"R12 cannot satisfy entity-disjoint dev {split} cell {(value, position)}."
                )
            used_entities.add(candidate.query_entity_id)
            outputs[split].append(candidate)

    select = tuple(outputs["select"])
    final = tuple(outputs["final"])
    expected_positions = Counter({0: expected_value_count // 4, 1: expected_value_count // 4, 2: expected_value_count // 4, 3: expected_value_count // 4})
    if expected_value_count % 4 or Counter(segment.query.target_index for segment in select) != expected_positions or Counter(segment.query.target_index for segment in final) != expected_positions:
        raise RuntimeError("R12 dev selection lost exact choice-position balance.")
    if {segment.query_entity_id for segment in select} & {segment.query_entity_id for segment in final}:
        raise RuntimeError("R12 dev select/final entities overlap.")
    if enforce_locked:
        if seed != R12_SELECTION_SEED or expected_value_count != R12_DEV_VALUE_COUNT:
            raise ValueError("Locked R12 dev selection forbids seed/count drift.")
        _assert_hash("dev-select", select, R12_DEV_SELECT_SHA256)
        _assert_hash("dev-final", final, R12_DEV_FINAL_SHA256)
    return select, final


@dataclass(frozen=True)
class R12TrainingUnit:
    global_micro_index: int
    epoch_zero: int
    micro_in_epoch: int
    optimizer_step_zero: int
    forward_cyclic_training_view: int
    segment: R5Segment

    def receipt(self) -> dict[str, Any]:
        return {
            "global_micro_index": self.global_micro_index,
            "epoch_zero": self.epoch_zero,
            "micro_in_epoch": self.micro_in_epoch,
            "optimizer_step_zero": self.optimizer_step_zero,
            "forward_cyclic_training_view": self.forward_cyclic_training_view,
            "segment_id": self.segment.segment_id,
        }


def build_training_schedule(
    selected: Sequence[R5Segment],
    *,
    seed: int = R12_SELECTION_SEED,
    epochs: int = R12_EPOCHS,
    gradient_accumulation: int = R12_GRADIENT_ACCUMULATION,
    enforce_locked: bool = True,
) -> tuple[R12TrainingUnit, ...]:
    values = _validate_pool(selected)
    if epochs <= 0 or epochs % 4:
        raise ValueError("R12 epochs must be a positive multiple of four.")
    if gradient_accumulation <= 0 or len(values) % gradient_accumulation:
        raise ValueError("R12 gradient accumulation must divide the selected train size.")
    units: list[R12TrainingUnit] = []
    for epoch in range(epochs):
        ordered = sorted(
            values,
            key=lambda segment: (
                _digest(seed, "train-epoch", epoch, segment.segment_id),
                segment.segment_id,
            ),
        )
        for micro_in_epoch, segment in enumerate(ordered):
            global_micro = epoch * len(values) + micro_in_epoch
            phase = int.from_bytes(hashlib.sha256(segment.segment_id.encode()).digest()[:2], "big") % 4
            units.append(
                R12TrainingUnit(
                    global_micro_index=global_micro,
                    epoch_zero=epoch,
                    micro_in_epoch=micro_in_epoch,
                    optimizer_step_zero=global_micro // gradient_accumulation,
                    forward_cyclic_training_view=(epoch + phase) % 4,
                    segment=segment,
                )
            )
    counts: Counter[tuple[str, int]] = Counter(
        (unit.segment.segment_id, unit.forward_cyclic_training_view) for unit in units
    )
    expected_per_view = epochs // 4
    if any(counts[(segment.segment_id, view)] != expected_per_view for segment in values for view in range(4)):
        raise RuntimeError("R12 schedule lost exact per-segment choice-view balance.")
    if enforce_locked:
        if (
            seed != R12_SELECTION_SEED
            or epochs != R12_EPOCHS
            or gradient_accumulation != R12_GRADIENT_ACCUMULATION
            or len(values) != R12_TRAIN_SEGMENT_COUNT
        ):
            raise ValueError("Locked R12 schedule forbids seed/epoch/batch/size drift.")
        if len(units) != R12_MICRO_STEPS:
            raise RuntimeError("R12 schedule micro-step count drifted.")
    return tuple(units)


def donor_derangement(
    segments: Sequence[R5Segment], *, seed: int = R12_SELECTION_SEED
) -> dict[str, str]:
    """Map every item to a different-value donor in a deterministic cycle."""

    values = _validate_pool(segments)
    if len({target_value(segment) for segment in values}) != len(values):
        raise ValueError("R12 donor derangement requires one segment per target value.")
    ordered = sorted(
        values,
        key=lambda segment: (_digest(seed, "donor-order", segment.segment_id), segment.segment_id),
    )
    if len(ordered) < 2:
        raise ValueError("R12 donor derangement requires at least two items.")
    mapping = {
        segment.segment_id: ordered[(index + 1) % len(ordered)].segment_id
        for index, segment in enumerate(ordered)
    }
    if any(
        target_value(next(value for value in ordered if value.segment_id == donor_id))
        == target_value(next(value for value in ordered if value.segment_id == source_id))
        for source_id, donor_id in mapping.items()
    ):
        raise RuntimeError("R12 donor derangement retained a target value.")
    return mapping


def conditioned_target_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    target_segment_id: str,
    endpoint: str,
) -> dict[str, Any]:
    """Extend the unchanged R10 normal/reset gate with an event-donor control."""

    base = target_statistics(
        rows,
        suite=suite,
        target_segment_id=target_segment_id,
        endpoint=endpoint,
    )

    def cell(checkpoint: str, condition: str) -> list[Mapping[str, Any]]:
        result = [
            row
            for row in rows
            if row.get("suite") == suite
            and row.get("checkpoint") == checkpoint
            and row.get("condition") == condition
            and row.get("pair_unit") == target_segment_id
        ]
        if len(result) != 4 or {int(row["view_index"]) for row in result} != {0, 1, 2, 3}:
            raise ValueError(
                f"R12 target donor cell is incomplete: {suite}:{target_segment_id}:{checkpoint}:{condition}"
            )
        return result

    cells = {
        (checkpoint, condition): cell(checkpoint, condition)
        for checkpoint in ("m0", endpoint)
        for condition in ("normal", "donor")
    }

    def mean(checkpoint: str, condition: str, field: str) -> float:
        result = sum(float(row[field]) for row in cells[(checkpoint, condition)]) / 4.0
        if not math.isfinite(result):
            raise ValueError("R12 target donor statistics contain a non-finite value.")
        return result

    endpoint_normal = mean(endpoint, "normal", "ce")
    endpoint_donor = mean(endpoint, "donor", "ce")
    normal_by_view = {int(row["view_index"]): float(row["ce"]) for row in cells[(endpoint, "normal")]}
    donor_by_view = {int(row["view_index"]): float(row["ce"]) for row in cells[(endpoint, "donor")]}
    per_view = {view: normal_by_view[view] - donor_by_view[view] for view in range(4)}
    donor_did = (
        endpoint_normal - endpoint_donor
    ) - (
        mean("m0", "normal", "ce") - mean("m0", "donor", "ce")
    )
    base.update(
        {
            "endpoint_donor_mean_ce": endpoint_donor,
            "normal_vs_donor_relative_change": endpoint_normal / endpoint_donor - 1.0,
            "normal_vs_donor_per_view_delta_ce": dict(sorted(per_view.items())),
            "normal_better_than_donor_views": sum(value < 0.0 for value in per_view.values()),
            "endpoint_donor_accuracy": mean(endpoint, "donor", "correct"),
            "normal_vs_donor_accuracy_delta": (
                mean(endpoint, "normal", "correct") - mean(endpoint, "donor", "correct")
            ),
            "normal_donor_difference_in_differences": donor_did,
        }
    )
    return base


def conditioned_target_gate(statistics: Mapping[str, Any], *, technical_gate: bool) -> bool:
    return bool(
        target_gate(statistics, technical_gate=technical_gate)
        and float(statistics["normal_vs_donor_relative_change"]) <= -0.20
        and int(statistics["normal_better_than_donor_views"]) == 4
        and float(statistics["normal_vs_donor_accuracy_delta"]) >= 0.25
        and float(statistics["normal_donor_difference_in_differences"]) < 0.0
    )


def selection_audit(segments: Sequence[R5Segment]) -> dict[str, Any]:
    values = _validate_pool(segments)
    return {
        "count": len(values),
        "segment_ids": [segment.segment_id for segment in values],
        "payload_sha256": _payload_sha(values),
        "unique_entities": len({segment.query_entity_id for segment in values}),
        "unique_target_values": len({target_value(segment) for segment in values}),
        "target_index_counts": dict(sorted(Counter(segment.query.target_index for segment in values).items())),
        "target_values": sorted({target_value(segment) for segment in values}),
    }


__all__ = [
    "R12_BASIS_COUNT",
    "R12_BASIS_OUTPUT_NORM",
    "R12_CHECKPOINT_EPOCHS",
    "R12_CHECKPOINT_STEPS",
    "R12_DEV_FINAL_COUNT",
    "R12_DEV_FINAL_SHA256",
    "R12_DEV_SELECT_COUNT",
    "R12_DEV_SELECT_SHA256",
    "R12_EPOCHS",
    "R12_GRADIENT_ACCUMULATION",
    "R12_LATENT_RMS_SOFT_LIMIT",
    "R12_MICRO_STEPS",
    "R12_OPTIMIZER_STEPS",
    "R12_PROTOCOL",
    "R12_SELECTION_SEED",
    "R12_TRAIN_AUDIT_COUNT",
    "R12_TRAIN_AUDIT_SHA256",
    "R12_TRAIN_SEGMENT_COUNT",
    "R12_TRAIN_SELECTION_SHA256",
    "R12_WEIGHT_DECAY",
    "R12_WRITER_LEARNING_RATE",
    "R12TrainingUnit",
    "build_training_schedule",
    "conditioned_target_gate",
    "conditioned_target_statistics",
    "donor_derangement",
    "select_balanced_train_f1",
    "select_entity_disjoint_dev_f1",
    "selection_audit",
    "target_value",
]
