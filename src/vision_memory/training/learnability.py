"""Deterministic curricula and explicit functional/coordinate gates."""
from __future__ import annotations

import math
from .latent_bank_unet import stable_seed


def teacher_order(bank, anchor_run="direct-gaussian-s00-a1"):
    records = bank["teachers"]
    anchor = [r["teacher_id"] for r in records if any(s["run_id"] == anchor_run for s in r["source_runs"])]
    if len(anchor) != 1:
        raise ValueError("Require exactly one preregistered anchor run")
    ids = sorted(r["teacher_id"] for r in records)
    return anchor + [tid for tid in ids if tid != anchor[0]]


def training_pair(stage, step, ids, seed):
    if stage not in ("single", "noise", "set") or step < 0 or not ids:
        raise ValueError("Invalid curriculum stage, step or teachers")
    index = 0 if stage == "single" else step % len(ids)
    return stable_seed(seed, "learnability-training", index), ids[index if stage == "set" else 0]


def evaluation_seed(seed, split, index):
    if split not in ("validation", "test"):
        raise ValueError("Invalid held-out noise split")
    return stable_seed(seed, "learnability-" + split, index)


def gate(rows, stage, relative_rms_limit=.1):
    """Only original prompt and train/validation noises influence advancement.

    Final test noises and paraphrases never choose rank, budget or checkpoints.
    A 90% RMS reduction is an operational coordinate gate, not a theorem about
    readable-region size. Absolute RMS and raw QA are always reported too.
    """
    selected = [r for r in rows if r["prompt_id"] == "original_open" and
                r["split"] in (("train",) if stage == "single" else ("train", "validation"))]
    if not selected:
        return {"passed": False, "reason": "missing_gate_rows"}
    functional = all(r["scorer"]["strict_correct"] and r["scorer"]["answer_followed_immediately_by_eos"] for r in selected)
    # Set validation may enter any readable region: no arbitrary unseen-noise
    # teacher is fabricated to define a regression label.
    coordinate_rows = [r for r in selected if stage != "set" or r["split"] == "train"]
    coordinate = all(math.isfinite(r["relative_rms"]) and r["relative_rms"] <= relative_rms_limit for r in coordinate_rows)
    return {"passed": functional and coordinate, "functional_pass": functional,
            "coordinate_pass": coordinate, "functional_rows": len(selected),
            "coordinate_rows": len(coordinate_rows), "relative_rms_limit": relative_rms_limit}
