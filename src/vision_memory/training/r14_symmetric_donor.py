"""Model-free contracts for R14 symmetric wrong-donor ranking.

R14 preserves the R13 writer, data, schedule, and causal evaluation.  Its only
scientific intervention is training-time credit assignment: every train event
is paired bidirectionally with one different-value event, and the own image must
beat that wrong-donor image by a bounded four-choice CE margin.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from vision_memory.training.r5_compose import R5Segment
from vision_memory.training.r12_shared_writer import target_value
from vision_memory.training.r13_centered_residual import (
    R13_FRESH_DEV_FINAL_COUNT,
    R13_FRESH_DEV_FINAL_SHA256,
    centered_target_gate,
    centered_target_statistics,
    select_fresh_dev_final_f1,
)


R14_PROTOCOL = "R14-Symmetric-Donor-Ranking-Centered-Residual-Writer"
R14_PAIR_SEED = 20260904
R14_RANKING_MARGIN = math.log(4.0)
R14_RANKING_WEIGHT = 1.0
R14_FRESH_DEV_FINAL_COUNT = R13_FRESH_DEV_FINAL_COUNT
R14_FRESH_DEV_FINAL_SHA256 = R13_FRESH_DEV_FINAL_SHA256


def _digest(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\x1f{namespace}\x1f{value}".encode()).hexdigest()


def symmetric_donor_mapping(segments: Sequence[R5Segment], *, seed: int = R14_PAIR_SEED) -> dict[str, str]:
    """Return a deterministic involution pairing equal-size, different-value groups."""

    values = tuple(segments)
    if not values or len({segment.segment_id for segment in values}) != len(values):
        raise ValueError("R14 pairing requires a non-empty unique segment sequence.")
    groups: dict[str, list[R5Segment]] = defaultdict(list)
    for segment in values:
        groups[target_value(segment)].append(segment)
    group_sizes = {len(group) for group in groups.values()}
    if len(group_sizes) != 1:
        raise ValueError("R14 pairing requires equal examples per target value.")
    ordered_targets = sorted(groups, key=lambda value: (_digest(seed, "target-order", value), value))
    if len(ordered_targets) < 2 or len(ordered_targets) % 2:
        raise ValueError("R14 pairing requires a positive even number of target values.")
    half = len(ordered_targets) // 2
    mapping: dict[str, str] = {}
    for left_target, right_target in zip(ordered_targets[:half], ordered_targets[half:], strict=True):
        left = sorted(
            groups[left_target],
            key=lambda segment: (_digest(seed, "member-order", segment.segment_id), segment.segment_id),
        )
        right = sorted(
            groups[right_target],
            key=lambda segment: (_digest(seed, "member-order", segment.segment_id), segment.segment_id),
        )
        for left_segment, right_segment in zip(left, right, strict=True):
            mapping[left_segment.segment_id] = right_segment.segment_id
            mapping[right_segment.segment_id] = left_segment.segment_id
    by_id = {segment.segment_id: segment for segment in values}
    if set(mapping) != set(by_id):
        raise RuntimeError("R14 pairing did not cover every training segment exactly once.")
    if any(mapping.get(donor) != source or source == donor for source, donor in mapping.items()):
        raise RuntimeError("R14 pairing is not a fixed-point-free involution.")
    if any(target_value(by_id[source]) == target_value(by_id[donor]) for source, donor in mapping.items()):
        raise RuntimeError("R14 pairing retained a target value.")
    return mapping


def symmetric_pairing_audit(
    segments: Sequence[R5Segment],
    mapping: Mapping[str, str],
    *,
    seed: int = R14_PAIR_SEED,
) -> dict[str, Any]:
    """Describe and hash each undirected training pair once."""

    values = tuple(segments)
    by_id = {segment.segment_id: segment for segment in values}
    if set(mapping) != set(by_id):
        raise ValueError("R14 pairing audit received incomplete mapping keys.")
    rows = []
    seen: set[str] = set()
    for source in sorted(mapping):
        donor = mapping[source]
        if donor not in by_id or mapping.get(donor) != source or source == donor:
            raise ValueError("R14 pairing audit received a non-involutive mapping.")
        if source in seen:
            continue
        left, right = sorted((source, donor))
        rows.append(
            {
                "left_segment_id": left,
                "left_target_value": target_value(by_id[left]),
                "right_segment_id": right,
                "right_target_value": target_value(by_id[right]),
            }
        )
        seen.update((source, donor))
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "vision_memory.r14-symmetric-donor-pairing-audit.v1",
        "seed": seed,
        "segment_count": len(values),
        "pair_count": len(rows),
        "different_target_value": all(row["left_target_value"] != row["right_target_value"] for row in rows),
        "involution": len(seen) == len(values),
        "pairs_sha256": hashlib.sha256(canonical).hexdigest(),
        "pairs": rows,
    }


def symmetric_ranking_loss(
    own_ce: Tensor,
    donor_ce: Tensor,
    *,
    margin: float = R14_RANKING_MARGIN,
) -> Tensor:
    """Boundedly require wrong-donor CE to exceed own CE by the fixed margin."""

    if own_ce.numel() != 1 or donor_ce.numel() != 1:
        raise ValueError("R14 ranking requires scalar own and donor CE tensors.")
    if not torch.isfinite(own_ce) or not torch.isfinite(donor_ce):
        raise ValueError("R14 ranking received non-finite CE.")
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("R14 ranking margin must be finite and positive.")
    return torch.relu(own_ce.new_tensor(margin) + own_ce - donor_ce)


def choice_target_margin(choice_logits: Tensor, target_index: int) -> Tensor:
    """Return target logit minus the strongest alternative without detaching."""

    logits = choice_logits.reshape(-1)
    if logits.numel() != 4 or not 0 <= target_index < 4:
        raise ValueError("R14 target margin requires four logits and target_index in [0,3].")
    alternatives = torch.cat((logits[:target_index], logits[target_index + 1 :]))
    return logits[target_index] - alternatives.max()


__all__ = [
    "R14_FRESH_DEV_FINAL_COUNT",
    "R14_FRESH_DEV_FINAL_SHA256",
    "R14_PAIR_SEED",
    "R14_PROTOCOL",
    "R14_RANKING_MARGIN",
    "R14_RANKING_WEIGHT",
    "centered_target_gate",
    "centered_target_statistics",
    "choice_target_margin",
    "select_fresh_dev_final_f1",
    "symmetric_donor_mapping",
    "symmetric_pairing_audit",
    "symmetric_ranking_loss",
]
