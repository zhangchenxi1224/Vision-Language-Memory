"""Metrics for the programmatic recurrent-memory benchmark."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .metrics import correctness


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _group_accuracy(rows: list[Mapping[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            groups[str(value)].append(correctness(row))
    return {
        key: {"n": len(values), "accuracy": _mean(values)}
        for key, values in sorted(groups.items())
    }


def _condition_drop(rows: list[Mapping[str, Any]], condition: str) -> dict[str, Any] | None:
    baseline: dict[tuple[Any, ...], float] = {}
    degraded: dict[tuple[Any, ...], float] = {}
    for row in rows:
        key = (
            row.get("episode_id"),
            row.get("query_id"),
            row.get("method"),
            row.get("seed"),
            row.get("counterfactual_variant"),
        )
        bucket = baseline if row.get("condition", "standard") == "standard" else degraded if row.get("condition") == condition else None
        if bucket is not None:
            if key in bucket:
                raise ValueError(f"Duplicate {condition} diagnostic key: {key}")
            bucket[key] = correctness(row)
    common = sorted(set(baseline) & set(degraded), key=repr)
    if not common:
        return None
    values = [baseline[key] - degraded[key] for key in common]
    return {"n_pairs": len(common), "accuracy_drop": _mean(values)}


def _matched_distractor_damage(rows: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    clean: dict[tuple[Any, ...], float] = {}
    distractor: dict[tuple[Any, ...], float] = {}
    for row in rows:
        pair_id = row.get("counterfactual_pair_id")
        variant = row.get("counterfactual_variant")
        if pair_id is None or variant not in {"clean", "distractor"}:
            continue
        key = (pair_id, row.get("query_id"), row.get("method"), row.get("seed"), row.get("condition", "standard"))
        target = clean if variant == "clean" else distractor
        if key in target:
            raise ValueError(f"Duplicate matched-counterfactual key: {key}")
        target[key] = correctness(row)
    common = sorted(set(clean) & set(distractor), key=repr)
    if not common:
        return None
    values = [clean[key] - distractor[key] for key in common]
    return {"n_pairs": len(common), "accuracy_damage": _mean(values)}


def _noise_robustness(rows: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("diffusion_seed") is None:
            continue
        key = (row.get("episode_id"), row.get("query_id"), row.get("method"))
        groups[key].append(correctness(row))
    if not groups:
        return None
    means = [sum(values) / len(values) for values in groups.values()]
    invariant = [float(all(value == values[0] for value in values)) for values in groups.values()]
    return {
        "n_episode_queries": len(groups),
        "mean_accuracy_across_noise": _mean(means),
        "prediction_correctness_invariance_rate": _mean(invariant),
    }


def compute_synthetic_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("No synthetic prediction records were provided.")
    scores = [correctness(row) for row in rows]
    subtype = _group_accuracy(rows, "subtype")
    stale_values: list[float] = []
    donor_values: list[float] = []
    for row in rows:
        prediction = row.get("prediction_index")
        target = row.get("target_index")
        stale = row.get("stale_target_index")
        if isinstance(stale, int):
            stale_values.append(float(prediction == stale and prediction != target))
        if row.get("condition") == "state_swap" and isinstance(row.get("donor_target_index"), int):
            donor_values.append(float(prediction == row["donor_target_index"]))

    latency = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    return {
        "n": len(rows),
        "micro_accuracy": _mean(scores),
        "subtype_macro_accuracy": _mean([float(value["accuracy"]) for value in subtype.values()]) if subtype else None,
        "by_subtype": subtype,
        "by_split": _group_accuracy(rows, "split"),
        "by_ood_group": _group_accuracy(rows, "ood_group"),
        "stale_answer_error": {"n": len(stale_values), "rate": _mean(stale_values)} if stale_values else None,
        "matched_distractor_damage": _matched_distractor_damage(rows),
        "reset": _condition_drop(rows, "reset"),
        "shuffle": _condition_drop(rows, "shuffle"),
        "state_swap": _condition_drop(rows, "state_swap"),
        "state_swap_donor_answer": {"n": len(donor_values), "rate": _mean(donor_values)} if donor_values else None,
        "noise_robustness": _noise_robustness(rows),
        "efficiency": {
            "n_latency_records": len(latency),
            "mean_latency_seconds": _mean(latency),
            "peak_vram_gib": max((float(row.get("peak_vram_gib", 0.0)) for row in rows), default=0.0),
            "gpu_hours": sum(float(row.get("gpu_hours", 0.0)) for row in rows),
            "failures": sum(int(row.get("failed", False)) for row in rows),
        },
    }


__all__ = ["compute_synthetic_metrics"]
