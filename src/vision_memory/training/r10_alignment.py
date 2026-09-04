"""Model-free contracts for the R10 visual-alignment lower-bound diagnostic."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from .r5_compose import R5Segment, ScheduledR5Segment, canonical_sha256


R10_PROTOCOL = "R10-VisualAlignment-LowerBound"
R10_SELECTION_SEED = 20260831
R10_TARGET_COUNT = 8
R10_OPTIMIZER_STEPS = 128
R10_TARGET_IDS = (
    "r5-f1-8015bf53a4067aaa7e882288",
    "r5-f1-392d41fd097d069c42218e0a",
    "r5-f1-1aee01c0f3e7684c05c9122c",
    "r5-f1-807090710dd4c077a97348ba",
    "r5-f1-02c8aa9dc3523351c4d5f9c7",
    "r5-f1-11dddf29efebc033a995ab92",
    "r5-f1-de4899e05b0aa7f3d8373171",
    "r5-f1-850189e424efde62468b2ef9",
)
R10_SELECTED_SEGMENTS_SHA256 = "6198beb3a3758fd7df912c6956bc05eac0ace8603708f37147826c65a4d61845"


def _selection_digest(segment_id: str, *, seed: int) -> bytes:
    return hashlib.sha256(
        f"{R10_PROTOCOL}\x1f{seed}\x1fF1\x1f{segment_id}".encode()
    ).digest()


def select_f1_targets(
    pools: Mapping[str, Sequence[R5Segment]],
    *,
    seed: int = R10_SELECTION_SEED,
    count: int = R10_TARGET_COUNT,
    enforce_locked: bool = True,
) -> tuple[R5Segment, ...]:
    """Select the preregistered F1 targets without consulting model outcomes."""

    if set(pools) and "F1" not in pools:
        raise ValueError("R10 target selection requires an F1 pool.")
    values = tuple(pools.get("F1", ()))
    if len(values) < count or count <= 0:
        raise ValueError(f"R10 requires at least {count} F1 segments, got {len(values)}.")
    if len({segment.segment_id for segment in values}) != len(values):
        raise ValueError("R10 F1 pool contains duplicate segment IDs.")
    ordered = sorted(
        values,
        key=lambda segment: (
            _selection_digest(segment.segment_id, seed=seed),
            segment.segment_id,
        ),
    )
    selected = tuple(ordered[:count])
    if enforce_locked:
        observed_ids = tuple(segment.segment_id for segment in selected)
        observed_sha = canonical_sha256([segment.to_dict() for segment in selected])
        if seed != R10_SELECTION_SEED or count != R10_TARGET_COUNT:
            raise ValueError("Locked R10 target selection forbids seed/count drift.")
        if observed_ids != R10_TARGET_IDS or observed_sha != R10_SELECTED_SEGMENTS_SHA256:
            raise RuntimeError(
                "R10 F1 target selection drifted: "
                f"ids={observed_ids}, payload_sha256={observed_sha}"
            )
    return selected


def build_single_target_schedule(
    selected: Sequence[R5Segment],
    *,
    target_index: int,
    optimizer_steps: int = R10_OPTIMIZER_STEPS,
    schedule_seed: int = R10_SELECTION_SEED,
) -> tuple[ScheduledR5Segment, ...]:
    """Build eight-item blocks while exposing only the target to optimization.

    The target occupies positions 0,1,2,3 in a strict cycle. Since R5's forward
    choice rotation is ``(global_micro_index + target_phase) % 4`` and every
    block has eight entries, this gives exactly 32 exposures to each forward
    cyclic view over 128 steps. The seven unoptimized items make the schedule
    and cursor shape explicit without contributing gradients.
    """

    segments = tuple(selected)
    if len(segments) != R10_TARGET_COUNT or len({value.segment_id for value in segments}) != len(segments):
        raise ValueError("R10 schedule requires exactly eight unique selected F1 segments.")
    if not 0 <= target_index < len(segments):
        raise ValueError("R10 target_index is outside the selected target list.")
    if optimizer_steps != R10_OPTIMIZER_STEPS:
        raise ValueError("R10 schedule fixes 128 optimizer steps.")
    if any(segment.family != "F1" or len(segment.events) != 1 for segment in segments):
        raise ValueError("R10 schedule accepts one-event F1 segments only.")

    target = segments[target_index]
    others = tuple(segment for index, segment in enumerate(segments) if index != target_index)
    draws: defaultdict[str, int] = defaultdict(int)
    units: list[ScheduledR5Segment] = []
    for step_zero in range(optimizer_steps):
        ordered_others = sorted(
            others,
            key=lambda segment: (
                hashlib.sha256(
                    f"{R10_PROTOCOL}\x1f{schedule_seed}\x1f{step_zero}\x1f{segment.segment_id}".encode()
                ).digest(),
                segment.segment_id,
            ),
        )
        target_position = step_zero % 4
        order = list(ordered_others)
        order.insert(target_position, target)
        for micro_in_step, segment in enumerate(order):
            units.append(
                ScheduledR5Segment(
                    global_micro_index=step_zero * R10_TARGET_COUNT + micro_in_step,
                    optimizer_step_zero=step_zero,
                    micro_in_step=micro_in_step,
                    phase="r10_single_target_alignment",
                    family_draw_index=draws[segment.family],
                    segment=segment,
                    selected_step_indices=None,
                )
            )
            draws[segment.family] += 1
    audit = target_training_view_counts(units, target_segment_id=target.segment_id)
    if audit != {0: 32, 1: 32, 2: 32, 3: 32}:
        raise RuntimeError(f"R10 target choice-view schedule drifted: {audit}")
    return tuple(units)


def target_training_view_counts(
    schedule: Sequence[ScheduledR5Segment],
    *,
    target_segment_id: str,
) -> dict[int, int]:
    phase = int.from_bytes(hashlib.sha256(target_segment_id.encode()).digest()[:2], "big") % 4
    counts: Counter[int] = Counter()
    for unit in schedule:
        if unit.segment.segment_id == target_segment_id:
            counts[(unit.global_micro_index + phase) % 4] += 1
    return dict(sorted(counts.items()))


def _cell(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    checkpoint: str,
    condition: str,
    target_segment_id: str,
) -> list[Mapping[str, Any]]:
    values = [
        row
        for row in rows
        if row.get("suite") == suite
        and row.get("checkpoint") == checkpoint
        and row.get("condition") == condition
        and row.get("pair_unit") == target_segment_id
    ]
    if len(values) != 4 or {int(row["view_index"]) for row in values} != {0, 1, 2, 3}:
        raise ValueError(
            "R10 target cell is incomplete: "
            f"{suite}:{target_segment_id}:{checkpoint}:{condition}"
        )
    return values


def target_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    target_segment_id: str,
    endpoint: str,
) -> dict[str, Any]:
    cells = {
        (checkpoint, condition): _cell(
            rows,
            suite=suite,
            checkpoint=checkpoint,
            condition=condition,
            target_segment_id=target_segment_id,
        )
        for checkpoint in ("m0", endpoint)
        for condition in ("normal", "reset")
    }

    def mean(checkpoint: str, condition: str, field: str) -> float:
        values = [float(row[field]) for row in cells[(checkpoint, condition)]]
        result = sum(values) / len(values)
        if not math.isfinite(result):
            raise ValueError("R10 target statistics contain a non-finite value.")
        return result

    m0_normal = mean("m0", "normal", "ce")
    endpoint_normal = mean(endpoint, "normal", "ce")
    m0_by_view = {int(row["view_index"]): float(row["ce"]) for row in cells[("m0", "normal")]}
    endpoint_by_view = {
        int(row["view_index"]): float(row["ce"]) for row in cells[(endpoint, "normal")]
    }
    per_view = {view: endpoint_by_view[view] - m0_by_view[view] for view in range(4)}
    m0_accuracy = mean("m0", "normal", "correct")
    endpoint_accuracy = mean(endpoint, "normal", "correct")
    did = (
        endpoint_normal - mean(endpoint, "reset", "ce")
    ) - (
        m0_normal - mean("m0", "reset", "ce")
    )
    return {
        "target_segment_id": target_segment_id,
        "suite": suite,
        "endpoint": endpoint,
        "m0_normal_mean_ce": m0_normal,
        "endpoint_normal_mean_ce": endpoint_normal,
        "delta_ce": endpoint_normal - m0_normal,
        "relative_change": endpoint_normal / m0_normal - 1.0,
        "per_view_delta_ce": dict(sorted(per_view.items())),
        "improved_choice_views": sum(value < 0.0 for value in per_view.values()),
        "m0_normal_accuracy": m0_accuracy,
        "endpoint_normal_accuracy": endpoint_accuracy,
        "accuracy_delta": endpoint_accuracy - m0_accuracy,
        "normal_reset_difference_in_differences": did,
        "condition_mean_ce": {
            checkpoint: {
                condition: mean(checkpoint, condition, "ce")
                for condition in ("normal", "reset")
            }
            for checkpoint in ("m0", endpoint)
        },
        "bootstrap_ci_used": False,
        "bootstrap_ci_reason": "one independent target; choice views are deterministic, not independent units",
    }


def target_gate(statistics: Mapping[str, Any], *, technical_gate: bool) -> bool:
    return bool(
        technical_gate
        and float(statistics["relative_change"]) <= -0.20
        and int(statistics["improved_choice_views"]) == 4
        and float(statistics["accuracy_delta"]) >= 0.25
        and float(statistics["normal_reset_difference_in_differences"]) < 0.0
    )


__all__ = [
    "R10_OPTIMIZER_STEPS",
    "R10_PROTOCOL",
    "R10_SELECTED_SEGMENTS_SHA256",
    "R10_SELECTION_SEED",
    "R10_TARGET_COUNT",
    "R10_TARGET_IDS",
    "build_single_target_schedule",
    "select_f1_targets",
    "target_gate",
    "target_statistics",
    "target_training_view_counts",
]
