"""Model-free contracts for R15 synchronous round-robin donor training.

R15 keeps the R14 writer and causal evaluation fixed.  It changes the
training contrast estimator in exactly two measured ways: both directions of
an event pair share one parameter snapshot, and every event sees every wrong
target value through a deterministic complete round-robin schedule.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from vision_memory.training.r5_compose import R5Segment, canonical_sha256
from vision_memory.training.r12_shared_writer import target_value
from vision_memory.training.r14_symmetric_donor import (
    R14_FRESH_DEV_FINAL_COUNT,
    R14_FRESH_DEV_FINAL_SHA256,
    choice_target_margin,
    symmetric_ranking_loss,
)


R15_PROTOCOL = "R15-Synchronous-Round-Robin-Centered-Residual-Writer"
R15_PAIR_SEED = 20260905
R15_POOL_PAIRING_SEED = 20260904
R15_TARGET_VALUE_COUNT = 36
R15_MEMBERS_PER_VALUE = 4
R15_TRAIN_SEGMENT_COUNT = R15_TARGET_VALUE_COUNT * R15_MEMBERS_PER_VALUE
R15_EPOCHS = 36
R15_VALUE_PAIRS_PER_ROUND = R15_TARGET_VALUE_COUNT // 2
R15_EVENT_PAIRS_PER_ROUND = R15_TRAIN_SEGMENT_COUNT // 2
R15_PAIR_MICRO_STEPS = R15_EVENT_PAIRS_PER_ROUND * R15_EPOCHS
R15_DIRECTIONAL_EXAMPLES = R15_PAIR_MICRO_STEPS * 2
R15_READER_CALLS_PER_PAIR = 4
R15_READER_CALLS = R15_PAIR_MICRO_STEPS * R15_READER_CALLS_PER_PAIR
R15_PAIR_GRADIENT_ACCUMULATION = 2
R15_BACKWARD_LOSS_DIVISOR = 4.0
R15_OPTIMIZER_STEPS = R15_PAIR_MICRO_STEPS // R15_PAIR_GRADIENT_ACCUMULATION
R15_CHECKPOINT_EPOCHS = (0, 9, 18, 27, 36)
R15_CHECKPOINT_STEPS = tuple(
    epoch * R15_EVENT_PAIRS_PER_ROUND // R15_PAIR_GRADIENT_ACCUMULATION for epoch in R15_CHECKPOINT_EPOCHS
)
R15_RANKING_MARGIN = math.log(4.0)
R15_RANKING_WEIGHT = 1.0
R15_FRESH_DEV_FINAL_COUNT = R14_FRESH_DEV_FINAL_COUNT
R15_FRESH_DEV_FINAL_SHA256 = R14_FRESH_DEV_FINAL_SHA256


def _digest(seed: int, namespace: str, *parts: object) -> bytes:
    payload = "\x1f".join((R15_PROTOCOL, str(seed), namespace, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode()).digest()


def _segment_phase(segment_id: str) -> int:
    return int.from_bytes(hashlib.sha256(segment_id.encode()).digest()[:2], "big") % 4


def _validated_groups(
    segments: Sequence[R5Segment],
    *,
    seed: int,
) -> tuple[tuple[str, ...], dict[str, tuple[R5Segment, ...]], int]:
    values = tuple(segments)
    if not values or len({segment.segment_id for segment in values}) != len(values):
        raise ValueError("R15 requires a non-empty sequence of unique training segments.")
    groups: dict[str, list[R5Segment]] = defaultdict(list)
    for segment in values:
        groups[target_value(segment)].append(segment)
    sizes = {len(group) for group in groups.values()}
    if len(sizes) != 1:
        raise ValueError("R15 requires equal members per target value.")
    member_count = next(iter(sizes))
    if member_count < 2:
        raise ValueError("R15 requires at least two members per target value.")
    ordered_targets = tuple(sorted(groups, key=lambda value: (_digest(seed, "target-order", value), value)))
    if len(ordered_targets) < 2 or len(ordered_targets) % 2:
        raise ValueError("R15 requires a positive even number of target values.")
    ordered_groups = {
        value: tuple(
            sorted(
                groups[value],
                key=lambda segment: (
                    _digest(seed, "member-order", segment.segment_id),
                    segment.segment_id,
                ),
            )
        )
        for value in ordered_targets
    }
    return ordered_targets, ordered_groups, member_count


def round_robin_target_rounds(
    target_values: Sequence[str],
    *,
    seed: int = R15_PAIR_SEED,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Return the ``n-1`` one-factorization rounds of complete graph K_n."""

    values = tuple(target_values)
    if len(values) < 2 or len(values) % 2 or len(set(values)) != len(values):
        raise ValueError("R15 round robin requires distinct positive-even target values.")
    rotation = list(sorted(values, key=lambda value: (_digest(seed, "target-order", value), value)))
    rounds: list[tuple[tuple[str, str], ...]] = []
    for round_zero in range(len(rotation) - 1):
        pairs = [(rotation[index], rotation[-1 - index]) for index in range(len(rotation) // 2)]
        pairs.sort(
            key=lambda pair: (
                _digest(seed, "value-pair-order", round_zero, *sorted(pair)),
                tuple(sorted(pair)),
            )
        )
        rounds.append(tuple(pairs))
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    coverage = Counter(frozenset(pair) for round_pairs in rounds for pair in round_pairs)
    if len(coverage) != len(values) * (len(values) - 1) // 2 or set(coverage.values()) != {1}:
        raise RuntimeError("R15 target one-factorization lost complete unique pair coverage.")
    return tuple(rounds)


def _balanced_member_shifts(*, epochs: int, member_count: int) -> tuple[int, ...]:
    """Balance offsets and make the repeated final round differ from round zero."""

    if member_count < 2 or epochs < member_count or epochs % member_count:
        raise ValueError("R15 epochs must be a positive multiple of the member count.")
    prefix = tuple(index % member_count for index in range(epochs - member_count))
    tail = (0, *range(2, member_count), 1)
    shifts = prefix + tail
    expected = epochs // member_count
    if (
        len(shifts) != epochs
        or Counter(shifts) != Counter({offset: expected for offset in range(member_count)})
        or shifts[0] != 0
        or shifts[-1] != 1
    ):
        raise RuntimeError("R15 member-shift construction lost its locked balance.")
    return shifts


@dataclass(frozen=True)
class R15PairTrainingUnit:
    global_pair_index: int
    epoch_zero: int
    pair_in_epoch: int
    optimizer_step_zero: int
    pair_in_optimizer_step: int
    value_pair_index: int
    member_pair_rank: int
    member_shift: int
    left_member_rank: int
    right_member_rank: int
    left_forward_cyclic_training_view: int
    right_forward_cyclic_training_view: int
    left_segment: R5Segment
    right_segment: R5Segment

    def receipt(self) -> dict[str, Any]:
        return {
            "global_pair_index": self.global_pair_index,
            "epoch_zero": self.epoch_zero,
            "pair_in_epoch": self.pair_in_epoch,
            "optimizer_step_zero": self.optimizer_step_zero,
            "pair_in_optimizer_step": self.pair_in_optimizer_step,
            "value_pair_index": self.value_pair_index,
            "member_pair_rank": self.member_pair_rank,
            "member_shift": self.member_shift,
            "left_member_rank": self.left_member_rank,
            "right_member_rank": self.right_member_rank,
            "left_forward_cyclic_training_view": self.left_forward_cyclic_training_view,
            "right_forward_cyclic_training_view": self.right_forward_cyclic_training_view,
            "left_segment_id": self.left_segment.segment_id,
            "left_target_value": target_value(self.left_segment),
            "right_segment_id": self.right_segment.segment_id,
            "right_target_value": target_value(self.right_segment),
        }


def build_synchronous_round_robin_schedule(
    segments: Sequence[R5Segment],
    *,
    seed: int = R15_PAIR_SEED,
    epochs: int = R15_EPOCHS,
    pair_gradient_accumulation: int = R15_PAIR_GRADIENT_ACCUMULATION,
    enforce_locked: bool = True,
) -> tuple[R15PairTrainingUnit, ...]:
    """Build atomic bidirectional event pairs with complete wrong-value coverage."""

    targets, groups, member_count = _validated_groups(segments, seed=seed)
    if epochs < len(targets) or pair_gradient_accumulation != 2:
        raise ValueError("R15 requires at least n rounds and exactly two complete pairs per update.")
    if epochs % 4:
        raise ValueError("R15 epochs must preserve exact four-view balance.")
    base_rounds = round_robin_target_rounds(targets, seed=seed)
    shifts = _balanced_member_shifts(epochs=epochs, member_count=member_count)
    units: list[R15PairTrainingUnit] = []
    for epoch_zero in range(epochs):
        round_pairs = base_rounds[epoch_zero % len(base_rounds)]
        shift = shifts[epoch_zero]
        for member_rank in range(member_count):
            for value_pair_index, (left_target, right_target) in enumerate(round_pairs):
                left = groups[left_target][member_rank]
                right_rank = (member_rank + shift) % member_count
                right = groups[right_target][right_rank]
                global_pair_index = len(units)
                units.append(
                    R15PairTrainingUnit(
                        global_pair_index=global_pair_index,
                        epoch_zero=epoch_zero,
                        pair_in_epoch=global_pair_index % (len(segments) // 2),
                        optimizer_step_zero=global_pair_index // pair_gradient_accumulation,
                        pair_in_optimizer_step=global_pair_index % pair_gradient_accumulation,
                        value_pair_index=value_pair_index,
                        member_pair_rank=member_rank,
                        member_shift=shift,
                        left_member_rank=member_rank,
                        right_member_rank=right_rank,
                        left_forward_cyclic_training_view=(epoch_zero + _segment_phase(left.segment_id)) % 4,
                        right_forward_cyclic_training_view=(epoch_zero + _segment_phase(right.segment_id)) % 4,
                        left_segment=left,
                        right_segment=right,
                    )
                )
    audit = synchronous_schedule_audit(
        segments,
        units,
        seed=seed,
        epochs=epochs,
        pair_gradient_accumulation=pair_gradient_accumulation,
    )
    if not audit["passed"]:
        raise RuntimeError(f"R15 schedule failed its construction audit: {audit['checks']}")
    if enforce_locked and (
        seed != R15_PAIR_SEED
        or epochs != R15_EPOCHS
        or pair_gradient_accumulation != R15_PAIR_GRADIENT_ACCUMULATION
        or len(targets) != R15_TARGET_VALUE_COUNT
        or member_count != R15_MEMBERS_PER_VALUE
        or len(segments) != R15_TRAIN_SEGMENT_COUNT
        or len(units) != R15_PAIR_MICRO_STEPS
    ):
        raise ValueError("Locked R15 schedule forbids seed/epoch/batch/data-shape drift.")
    return tuple(units)


def synchronous_schedule_audit(
    segments: Sequence[R5Segment],
    schedule: Sequence[R15PairTrainingUnit],
    *,
    seed: int = R15_PAIR_SEED,
    epochs: int = R15_EPOCHS,
    pair_gradient_accumulation: int = R15_PAIR_GRADIENT_ACCUMULATION,
) -> dict[str, Any]:
    """Audit all coverage, balance, and optimizer-boundary invariants."""

    targets, groups, member_count = _validated_groups(segments, seed=seed)
    units = tuple(schedule)
    receipts = [unit.receipt() for unit in units]
    segment_ids = {segment.segment_id for segment in segments}
    expected_pairs_per_round = len(segments) // 2
    expected_pairs = epochs * expected_pairs_per_round
    expected_updates = expected_pairs // pair_gradient_accumulation
    appearances: Counter[tuple[str, int]] = Counter()
    views: Counter[tuple[str, int]] = Counter()
    donor_targets: dict[str, set[str]] = defaultdict(set)
    donor_ids: dict[str, set[str]] = defaultdict(set)
    donor_member_ranks: Counter[tuple[str, int]] = Counter()
    value_pairs_by_round: dict[int, list[frozenset[str]]] = defaultdict(list)
    event_pairs_by_round: dict[int, set[frozenset[str]]] = defaultdict(set)
    shifts_by_round: dict[int, set[int]] = defaultdict(set)
    updates: dict[int, list[R15PairTrainingUnit]] = defaultdict(list)
    cross_value = True
    known_segments = True
    for unit in units:
        left_id = unit.left_segment.segment_id
        right_id = unit.right_segment.segment_id
        left_target = target_value(unit.left_segment)
        right_target = target_value(unit.right_segment)
        known_segments &= left_id in segment_ids and right_id in segment_ids
        cross_value &= left_target != right_target and left_id != right_id
        appearances[(left_id, unit.epoch_zero)] += 1
        appearances[(right_id, unit.epoch_zero)] += 1
        views[(left_id, unit.left_forward_cyclic_training_view)] += 1
        views[(right_id, unit.right_forward_cyclic_training_view)] += 1
        donor_targets[left_id].add(right_target)
        donor_targets[right_id].add(left_target)
        donor_ids[left_id].add(right_id)
        donor_ids[right_id].add(left_id)
        donor_member_ranks[(left_id, unit.right_member_rank)] += 1
        donor_member_ranks[(right_id, unit.left_member_rank)] += 1
        value_pairs_by_round[unit.epoch_zero].append(frozenset((left_target, right_target)))
        event_pairs_by_round[unit.epoch_zero].add(frozenset((left_id, right_id)))
        shifts_by_round[unit.epoch_zero].add(unit.member_shift)
        updates[unit.optimizer_step_zero].append(unit)

    complete_rounds = len(targets) - 1
    first_factorization = Counter(
        pair for round_zero in range(min(complete_rounds, epochs)) for pair in value_pairs_by_round[round_zero]
    )
    expected_value_pairs = len(targets) * (len(targets) - 1) // 2
    view_expected = epochs // 4 if epochs % 4 == 0 else -1
    donor_rank_expected = epochs // member_count if epochs % member_count == 0 else -1
    optimizer_atomic = all(
        len(batch) == pair_gradient_accumulation
        and {unit.pair_in_optimizer_step for unit in batch} == set(range(pair_gradient_accumulation))
        and len({unit.epoch_zero for unit in batch}) == 1
        and len({frozenset((target_value(unit.left_segment), target_value(unit.right_segment))) for unit in batch})
        == pair_gradient_accumulation
        and len({target_value(segment) for unit in batch for segment in (unit.left_segment, unit.right_segment)})
        == 2 * pair_gradient_accumulation
        and len({segment.segment_id for unit in batch for segment in (unit.left_segment, unit.right_segment)})
        == 2 * pair_gradient_accumulation
        for batch in updates.values()
    )
    repeat_round = complete_rounds
    has_locked_repeat = epochs > repeat_round
    checks = {
        "pair_count": len(units) == expected_pairs,
        "global_pair_indices": [unit.global_pair_index for unit in units] == list(range(expected_pairs)),
        "pair_in_epoch_indices": all(
            unit.pair_in_epoch == unit.global_pair_index % expected_pairs_per_round for unit in units
        ),
        "known_unique_segments": known_segments and len(segment_ids) == len(segments),
        "cross_target_value_pairs": cross_value,
        "one_event_appearance_per_round": all(
            appearances[(segment_id, round_zero)] == 1 for segment_id in segment_ids for round_zero in range(epochs)
        ),
        "complete_value_pair_factorization": (
            len(first_factorization) == expected_value_pairs and set(first_factorization.values()) == {member_count}
        ),
        "every_wrong_target_seen": all(
            len(donor_targets[segment_id]) == len(targets) - 1 for segment_id in segment_ids
        ),
        "unique_donor_identity_each_round": all(len(donor_ids[segment_id]) == epochs for segment_id in segment_ids),
        "balanced_donor_member_ranks": donor_rank_expected >= 0
        and all(
            donor_member_ranks[(segment_id, rank)] == donor_rank_expected
            for segment_id in segment_ids
            for rank in range(member_count)
        ),
        "balanced_choice_views": view_expected >= 0
        and all(views[(segment_id, view)] == view_expected for segment_id in segment_ids for view in range(4)),
        "single_shift_per_round": all(len(shifts_by_round[index]) == 1 for index in range(epochs)),
        "repeat_value_round_zero": not has_locked_repeat
        or Counter(value_pairs_by_round[0]) == Counter(value_pairs_by_round[repeat_round]),
        "repeat_event_pairs_disjoint": not has_locked_repeat
        or event_pairs_by_round[0].isdisjoint(event_pairs_by_round[repeat_round]),
        "repeat_member_shift_plus_one": not has_locked_repeat
        or next(iter(shifts_by_round[repeat_round])) == (next(iter(shifts_by_round[0])) + 1) % member_count,
        "optimizer_step_count": len(updates) == expected_updates,
        "optimizer_steps_contiguous": set(updates) == set(range(expected_updates)),
        "optimizer_atomic_distinct_pairs": optimizer_atomic,
        "no_cross_round_accumulation": all(len({unit.epoch_zero for unit in batch}) == 1 for batch in updates.values()),
    }
    return {
        "schema": "vision_memory.r15-synchronous-round-robin-schedule-audit.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "seed": seed,
        "epochs": epochs,
        "target_value_count": len(targets),
        "members_per_target_value": member_count,
        "segment_count": len(segments),
        "value_pairs_in_complete_factorization": len(first_factorization),
        "pair_micro_steps": len(units),
        "directional_examples": len(units) * 2,
        "reader_calls": len(units) * R15_READER_CALLS_PER_PAIR,
        "optimizer_steps": len(updates),
        "pair_gradient_accumulation": pair_gradient_accumulation,
        "backward_loss_divisor": R15_BACKWARD_LOSS_DIVISOR,
        "member_shifts": [next(iter(shifts_by_round[index]), None) for index in range(epochs)],
        "schedule_sha256": canonical_sha256(receipts),
        "receipts": receipts,
    }


@dataclass(frozen=True)
class R15PairObjective:
    pair_sum: Tensor
    backward_loss: Tensor
    left_ranking_loss: Tensor
    right_ranking_loss: Tensor


def synchronous_pair_objective(
    left_own_ce: Tensor,
    left_donor_ce: Tensor,
    right_own_ce: Tensor,
    right_donor_ce: Tensor,
    left_regularizer: Tensor,
    right_regularizer: Tensor,
    *,
    margin: float = R15_RANKING_MARGIN,
    ranking_weight: float = R15_RANKING_WEIGHT,
) -> R15PairObjective:
    """Return the only allowed R15 pair sum and its once-scaled backward loss."""

    tensors = (
        left_own_ce,
        left_donor_ce,
        right_own_ce,
        right_donor_ce,
        left_regularizer,
        right_regularizer,
    )
    if any(value.numel() != 1 or not torch.isfinite(value) for value in tensors):
        raise ValueError("R15 pair objective requires finite scalar tensors.")
    if not math.isfinite(ranking_weight) or ranking_weight < 0.0:
        raise ValueError("R15 ranking weight must be finite and non-negative.")
    left_ranking = symmetric_ranking_loss(left_own_ce, left_donor_ce, margin=margin)
    right_ranking = symmetric_ranking_loss(right_own_ce, right_donor_ce, margin=margin)
    pair_sum = (
        left_own_ce
        + right_own_ce
        + ranking_weight * (left_ranking + right_ranking)
        + left_regularizer.to(device=left_own_ce.device)
        + right_regularizer.to(device=left_own_ce.device)
    )
    if not torch.isfinite(pair_sum):
        raise RuntimeError("R15 pair objective became non-finite.")
    return R15PairObjective(
        pair_sum=pair_sum,
        backward_loss=pair_sum / R15_BACKWARD_LOSS_DIVISOR,
        left_ranking_loss=left_ranking,
        right_ranking_loss=right_ranking,
    )


__all__ = [
    "R15_BACKWARD_LOSS_DIVISOR",
    "R15_CHECKPOINT_EPOCHS",
    "R15_CHECKPOINT_STEPS",
    "R15_DIRECTIONAL_EXAMPLES",
    "R15_EPOCHS",
    "R15_EVENT_PAIRS_PER_ROUND",
    "R15_FRESH_DEV_FINAL_COUNT",
    "R15_FRESH_DEV_FINAL_SHA256",
    "R15_MEMBERS_PER_VALUE",
    "R15_OPTIMIZER_STEPS",
    "R15_PAIR_GRADIENT_ACCUMULATION",
    "R15_PAIR_MICRO_STEPS",
    "R15_PAIR_SEED",
    "R15_POOL_PAIRING_SEED",
    "R15_PROTOCOL",
    "R15_RANKING_MARGIN",
    "R15_RANKING_WEIGHT",
    "R15_READER_CALLS",
    "R15_READER_CALLS_PER_PAIR",
    "R15_TARGET_VALUE_COUNT",
    "R15_TRAIN_SEGMENT_COUNT",
    "R15_VALUE_PAIRS_PER_ROUND",
    "R15PairObjective",
    "R15PairTrainingUnit",
    "build_synchronous_round_robin_schedule",
    "choice_target_margin",
    "round_robin_target_rounds",
    "synchronous_pair_objective",
    "synchronous_schedule_audit",
]
