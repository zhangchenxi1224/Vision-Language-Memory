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


def _macro_accuracy(rows: list[Mapping[str, Any]], fields: tuple[str, ...]) -> tuple[float | None, int]:
    cells: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        values = tuple(str(row[field]) for field in fields if row.get(field) is not None)
        if len(values) == len(fields):
            cells[values].append(correctness(row))
    means = [sum(values) / len(values) for values in cells.values()]
    return _mean(means), len(means)


def _condition_drop(rows: list[Mapping[str, Any]], condition: str) -> dict[str, Any] | None:
    baseline: dict[tuple[Any, ...], float] = {}
    degraded: dict[tuple[Any, ...], float] = {}
    for row in rows:
        key = (
            row.get("episode_id"),
            row.get("query_id"),
            row.get("method"),
            row.get("seed"),
            row.get("diffusion_seed"),
            row.get("split"),
            row.get("protocol"),
            row.get("form"),
            row.get("recurrence_mode"),
            row.get("distractor_variant"),
            row.get("noop_policy"),
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
        pair_id = row.get("distractor_pair_id")
        comparison_id = row.get("query_comparison_id")
        variant = row.get("distractor_variant")
        if pair_id is None or comparison_id is None or variant not in {"clean", "distractor"}:
            continue
        key = (
            pair_id,
            comparison_id,
            row.get("method"),
            row.get("seed"),
            row.get("diffusion_seed"),
            row.get("split"),
            row.get("protocol"),
            row.get("form"),
            row.get("recurrence_mode"),
            row.get("condition", "standard"),
            row.get("noop_policy"),
        )
        target = clean if variant == "clean" else distractor
        if key in target:
            raise ValueError(f"Duplicate matched-counterfactual key: {key}")
        target[key] = correctness(row)
    common = sorted(set(clean) & set(distractor), key=repr)
    if not common:
        return None
    values = [clean[key] - distractor[key] for key in common]
    return {"n_pairs": len(common), "accuracy_damage": _mean(values)}


def _noop_filter_effect(rows: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Compare the same stream with no-op writes kept versus prefiltered.

    This is intentionally separate from clean/distractor dataset pairing: both
    interventions share one episode and therefore one semantic target.
    """

    keep: dict[tuple[Any, ...], float] = {}
    skip: dict[tuple[Any, ...], float] = {}
    for row in rows:
        policy = row.get("noop_policy")
        intervention_pair_id = row.get("noop_intervention_pair_id")
        if policy not in {"keep", "skip"} or intervention_pair_id is None:
            continue
        key = (
            intervention_pair_id,
            row.get("method"),
            row.get("seed"),
            row.get("diffusion_seed"),
            row.get("split"),
            row.get("protocol"),
            row.get("forced_write_k"),
            row.get("form"),
            row.get("recurrence_mode"),
            row.get("condition", "standard"),
            row.get("distractor_variant"),
        )
        target = keep if policy == "keep" else skip
        if key in target:
            raise ValueError(f"Duplicate no-op {policy} intervention key: {key}")
        target[key] = correctness(row)
    common = sorted(set(keep) & set(skip), key=repr)
    if not common:
        return None
    keep_scores = [keep[key] for key in common]
    skip_scores = [skip[key] for key in common]
    return {
        "n_pairs": len(common),
        "keep_accuracy": _mean(keep_scores),
        "skip_accuracy": _mean(skip_scores),
        "skip_minus_keep_accuracy": _mean(
            [skip_score - keep_score for keep_score, skip_score in zip(keep_scores, skip_scores, strict=True)]
        ),
    }


def _noise_robustness(rows: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    groups: dict[tuple[Any, ...], dict[Any, float]] = defaultdict(dict)
    for row in rows:
        if row.get("condition", "standard") != "standard":
            continue
        if row.get("diffusion_seed") is None:
            continue
        key = (
            row.get("episode_id"),
            row.get("query_id"),
            row.get("method"),
            row.get("seed"),
            row.get("split"),
            row.get("protocol"),
            row.get("form"),
            row.get("recurrence_mode"),
            row.get("distractor_variant"),
            row.get("noop_policy"),
        )
        diffusion_seed = row.get("diffusion_seed")
        if diffusion_seed in groups[key]:
            raise ValueError(f"Duplicate diffusion-seed record for noise robustness: {key}, {diffusion_seed}")
        groups[key][diffusion_seed] = correctness(row)
    repeated_groups = {key: values for key, values in groups.items() if len(values) > 1}
    if not repeated_groups:
        return None
    score_lists = [list(values.values()) for values in repeated_groups.values()]
    means = [sum(values) / len(values) for values in score_lists]
    invariant = [float(all(value == values[0] for value in values)) for values in score_lists]
    return {
        "n_episode_queries": len(repeated_groups),
        "diffusion_seed_counts": sorted({len(values) for values in repeated_groups.values()}),
        "mean_accuracy_across_noise": _mean(means),
        "prediction_correctness_invariance_rate": _mean(invariant),
    }


def compute_synthetic_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("No synthetic prediction records were provided.")
    scores = [correctness(row) for row in rows]
    subtype = _group_accuracy(rows, "subtype")
    topic_subtype_macro, topic_subtype_cells = _macro_accuracy(rows, ("topic", "subtype"))
    stale_values: list[float] = []
    donor_values: list[float] = []
    for row in rows:
        prediction = row.get("prediction_index")
        target = row.get("target_index")
        stale = row.get("stale_target_index")
        if row.get("condition", "standard") == "standard" and isinstance(stale, int):
            stale_values.append(float(prediction == stale and prediction != target))
        if row.get("condition") == "state_swap" and isinstance(row.get("donor_target_index"), int):
            donor_values.append(float(prediction == row["donor_target_index"]))

    latency = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    event_latency = [
        float(row["event_latency_seconds"])
        for row in rows
        if row.get("event_latency_seconds") is not None
    ]
    query_latency = [
        float(row["query_latency_seconds"])
        for row in rows
        if row.get("query_latency_seconds") is not None
    ]
    state_bytes = [int(row["state_bytes"]) for row in rows if row.get("state_bytes") is not None]
    return {
        "n": len(rows),
        "micro_accuracy": _mean(scores),
        "macro_mcq_accuracy": topic_subtype_macro,
        "macro_mcq_cells": topic_subtype_cells,
        "macro_mcq_definition": "equal-weight mean over observed topic x transition-subtype cells",
        "subtype_macro_accuracy": _mean([float(value["accuracy"]) for value in subtype.values()]) if subtype else None,
        "by_subtype": subtype,
        "by_split": _group_accuracy(rows, "split"),
        "by_ood_group": _group_accuracy(rows, "ood_group"),
        "stale_answer_error": {"n": len(stale_values), "rate": _mean(stale_values)} if stale_values else None,
        "matched_distractor_damage": _matched_distractor_damage(rows),
        "noop_filter_effect": _noop_filter_effect(rows),
        "reset": _condition_drop(rows, "reset"),
        "shuffle": _condition_drop(rows, "shuffle"),
        "state_swap": _condition_drop(rows, "state_swap"),
        "state_swap_donor_answer": {"n": len(donor_values), "rate": _mean(donor_values)} if donor_values else None,
        "noise_robustness": _noise_robustness(rows),
        "efficiency": {
            "n_latency_records": len(latency),
            "mean_latency_seconds": _mean(latency),
            "mean_event_latency_seconds": _mean(event_latency),
            "mean_query_latency_seconds": _mean(query_latency),
            "state_bytes": max(state_bytes) if state_bytes else None,
            "peak_vram_gib": max((float(row.get("peak_vram_gib", 0.0)) for row in rows), default=0.0),
            "peak_updater_vram_gib": max(
                (float(row.get("peak_updater_vram_gib", 0.0)) for row in rows),
                default=0.0,
            ),
            "peak_reader_vram_gib": max(
                (float(row.get("peak_reader_vram_gib", 0.0)) for row in rows),
                default=0.0,
            ),
            "gpu_hours": sum(float(row.get("gpu_hours", 0.0)) for row in rows),
            "failures": sum(int(row.get("failed", False)) for row in rows),
        },
    }


__all__ = ["compute_synthetic_metrics"]
