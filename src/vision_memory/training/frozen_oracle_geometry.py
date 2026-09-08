"""Prospective Frozen DreamLite geometry protocol; CPU-only mathematics.

Population moment matching deliberately preserves Gaussian radial variation.
No Reader outcomes are used to choose seeds, tasks, or the primary checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Sequence

import numpy as np

SCHEMA = "vision_memory.frozen-oracle-geometry.v1"
DISTRIBUTIONS = ("gaussian", "uniform", "rademacher", "sphere", "heavy_tail")
CHECKPOINT_STEPS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def make_initial_array(shape: Sequence[int], distribution: str, seed: int,
                       scale: float = 1.0) -> np.ndarray:
    """CPU PCG64, float64 draw, one final float32 cast; no task/repeat key."""
    if not shape or any(isinstance(n, bool) or int(n) != n or n <= 0 for n in shape):
        raise ValueError("shape must contain positive integers")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive and finite")
    rng = np.random.Generator(np.random.PCG64(seed))
    shape = tuple(int(n) for n in shape)
    if distribution == "gaussian":
        x = rng.standard_normal(shape)
    elif distribution == "uniform":
        x = rng.uniform(-math.sqrt(3), math.sqrt(3), shape)
    elif distribution == "rademacher":
        x = rng.integers(0, 2, size=shape).astype(np.float64) * 2 - 1
    elif distribution == "sphere":
        x = rng.standard_normal(shape)
        x /= np.sqrt(np.mean(x * x))
    elif distribution in ("heavy_tail", "heavy-tail", "student_t5"):
        x = rng.standard_t(5, size=shape) * math.sqrt(3 / 5)
    else:
        raise ValueError(f"unknown initialization distribution: {distribution}")
    return np.asarray(scale * x, dtype=np.float32)


def array_stats(value: np.ndarray) -> dict[str, Any]:
    x = np.asarray(value, dtype=np.float64)
    centered = x - x.mean()
    var = float(np.mean(centered ** 2))
    return {"shape": list(x.shape), "mean": float(x.mean()),
            "rms": float(np.sqrt(np.mean(x * x))), "norm": float(np.linalg.norm(x)),
            "variance": var, "kurtosis": float(np.mean(centered ** 4) / var ** 2) if var else None,
            "max_abs": float(np.max(np.abs(x)))}


def build_manifest() -> dict[str, Any]:
    """158 independent optimization runs; shared conditions counted once."""
    runs: list[dict[str, Any]] = []
    memberships: dict[str, list[str]] = {x: [] for x in ("A1", "A2", "A3", "A4", "A9")}

    def add(stage: str, target: int, seed: int, distribution="gaussian", scale=1.0, repeat=None):
        rid = f"{stage}-t{target:02d}-{distribution}-s{seed:02d}-a{scale:g}"
        if repeat is not None:
            rid += f"-r{repeat}"
        runs.append({"run_id": rid, "stage": stage, "target_index": target,
                     "seed": seed, "distribution": distribution, "scale": scale,
                     "repeat": repeat, "mode": "optimize"})
        memberships[stage].append(rid)
        return rid

    for target in (0, 1):
        for repeat in range(3):
            add("A1", target, 0, repeat=repeat)
    anchor_runs = [add("A2", 1, seed) for seed in range(32)]
    memberships["A3"].extend(anchor_runs[:8])
    for distribution in DISTRIBUTIONS[1:]:
        for seed in range(8):
            add("A3", 1, seed, distribution)
    memberships["A4"].extend(anchor_runs[:8])
    for scale in (0.25, 0.5, 2.0, 4.0):
        for seed in range(8):
            add("A4", 1, seed, scale=scale)
    memberships["A9"].extend(anchor_runs[:8])
    for target in range(8):
        if target != 1:
            for seed in range(8):
                add("A9", target, seed)
    return {"schema": SCHEMA, "anchor_target_index": 1, "runs": runs,
            "study_memberships": memberships,
            "counts": dict(Counter(run["stage"] for run in runs)),
            "unique_optimization_runs": len(runs),
            "reuse_rule": "same target/seed/distribution/scale reused, never independent replicas",
            "local_basin": {"rhos": [0.01, 0.05, 0.1, 0.25], "directions": 8,
                            "rho_unit": "additive coordinate RMS", "seed_start": 10000,
                            "selection": "lexicographically first successful A2 run",
                            "direct_evaluation_separate_from_reoptimization": True},
            "interpolation": {"spaces": ["control", "endpoint"],
                              "lambdas": [i / 10 for i in range(11)],
                              "pair_selection": "all pairs among first 8 successful A2 run IDs",
                              "claim_limit": "finite grid straight-line connectivity only"}}


def wilson(successes: int, total: int) -> list[float] | None:
    if not 0 <= successes <= total:
        raise ValueError("invalid counts")
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    den = 1 + z * z / total
    mid = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total ** 2)) / den
    return [max(0., mid - half), min(1., mid + half)]


def pairwise_rmse(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    norms = np.einsum("ij,ij->i", x, x)
    d2 = np.maximum(norms[:, None] + norms[None, :] - 2 * x @ x.T, 0) / x.shape[1]
    np.fill_diagonal(d2, 0.)
    return np.sqrt(d2)


def geometry_statistics(values: np.ndarray, task_ids: Sequence[Any], seeds: Sequence[int]) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64).reshape(len(values), -1)
    if len(x) < 2 or len(task_ids) != len(x) or len(seeds) != len(x):
        raise ValueError("at least two aligned observations are required")
    distances = pairwise_rmse(x)
    same = np.asarray(task_ids)[:, None] == np.asarray(task_ids)[None, :]
    seed_match = np.asarray(seeds)[:, None] == np.asarray(seeds)[None, :]
    upper = np.triu(np.ones_like(same, dtype=bool), 1)

    def mean(mask):
        return float(distances[mask].mean()) if mask.any() else None

    within, between = mean(same & upper), mean(~same & upper)
    x -= x.mean(axis=0)
    _, singular, _ = np.linalg.svd(x, full_matrices=False)
    variance = singular ** 2
    fractions = variance / variance.sum() if variance.sum() else np.zeros_like(variance)
    ranks = {f"r{p}": int(np.searchsorted(np.cumsum(fractions), p / 100) + 1)
             if variance.sum() else 0 for p in (90, 95, 99)}
    rank_limit = min(len(x) - 1, x.shape[1])
    null = np.random.default_rng(20260908).standard_normal(x.shape)
    null -= null.mean(axis=0)
    ns = np.linalg.svd(null, compute_uv=False) ** 2
    null_r95 = int(np.searchsorted(np.cumsum(ns / ns.sum()), .95) + 1)
    return {"n": len(x), "dimensions": x.shape[1], "within_rmse": within,
            "between_rmse": between, "between_same_seed_rmse": mean(~same & seed_match & upper),
            "within_between_ratio": within / between if within is not None and between else None,
            "pca": {**ranks, "sample_rank_limit": rank_limit,
                    "r95_fraction_of_sample_rank": ranks["r95"] / rank_limit,
                    "random_null_r95": null_r95, "explained_variance": fractions.tolist(),
                    "claim_limit": "sample rank alone is not evidence of low dimensionality"}}


def determinism_gate(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups = {target: [s for s in summaries if s.get("target_index") == target] for target in (0, 1)}
    errors = []
    for target, rows in groups.items():
        if len(rows) != 3:
            errors.append(f"target {target}: expected 3 repeats, received {len(rows)}")
            continue
        for key in ("initial_xT_sha256", "optimized_xT_sha256", "endpoint_z_sha256",
                    "loss_trajectory_sha256", "gradient_trajectory_sha256"):
            observed = [row.get(key) for row in rows]
            if any(not value for value in observed) or len(set(observed)) != 1:
                errors.append(f"target {target}: missing or different {key}")
        if any(not row.get("technical_pass") for row in rows):
            errors.append(f"target {target}: technical gate failed")
    return {"passed": not errors, "errors": errors,
            "requires_bitwise_tensor_and_trajectory_equality": True}
