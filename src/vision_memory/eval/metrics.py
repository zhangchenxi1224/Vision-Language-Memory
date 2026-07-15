"""PrefEval classification metrics and paired state-diagnostic summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


CHOICE_TO_INDEX = {choice: index for index, choice in enumerate("ABCD")}


def _index(record: Mapping[str, Any], stem: str) -> int:
    index_key = f"{stem}_index"
    choice_key = f"{stem}_choice"
    value = record.get(index_key, record.get(stem, record.get(choice_key)))
    if isinstance(value, bool):
        raise ValueError(f"{index_key} cannot be boolean")
    if isinstance(value, int) and 0 <= value < 4:
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in CHOICE_TO_INDEX:
            return CHOICE_TO_INDEX[normalized]
        if normalized.isdigit() and 0 <= int(normalized) < 4:
            return int(normalized)
    raise ValueError(f"Missing or invalid {index_key}/{choice_key}: {value!r}")


def correctness(record: Mapping[str, Any]) -> float:
    return float(_index(record, "prediction") == _index(record, "target"))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return sum(values) / len(values)


def topic_form_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the headline equal-weight topic x form macro accuracy."""

    rows = list(records)
    if not rows:
        raise ValueError("No evaluation records were provided")
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        topic = row.get("topic")
        form = row.get("form")
        if not isinstance(topic, str) or not topic:
            raise ValueError("Every record must have a non-empty topic")
        if not isinstance(form, str) or not form:
            raise ValueError("Every record must have a non-empty form")
        cells[(topic, form)].append(correctness(row))

    cell_rows = [
        {"topic": topic, "form": form, "n": len(scores), "accuracy": _mean(scores)}
        for (topic, form), scores in sorted(cells.items())
    ]
    by_topic: dict[str, list[float]] = defaultdict(list)
    by_form: dict[str, list[float]] = defaultdict(list)
    for cell in cell_rows:
        by_topic[cell["topic"]].append(cell["accuracy"])
        by_form[cell["form"]].append(cell["accuracy"])
    return {
        "n": len(rows),
        "micro_accuracy": _mean([correctness(row) for row in rows]),
        "topic_form_macro_accuracy": _mean([cell["accuracy"] for cell in cell_rows]),
        "cells": cell_rows,
        "by_topic_macro_accuracy": {key: _mean(values) for key, values in sorted(by_topic.items())},
        "by_form_macro_accuracy": {key: _mean(values) for key, values in sorted(by_form.items())},
    }


def _pair_key(row: Mapping[str, Any], *, omit: set[str]) -> tuple[Any, ...]:
    fields = (
        "base_pair_id",
        "query_id",
        "form",
        "split",
        "method",
        "seed",
        "diffusion_seed",
        "protocol",
        "forced_write_k",
        "recurrence_mode",
        "counterfactual_variant",
        "noop_policy",
    )
    return tuple(row.get(field) for field in fields if field not in omit)


def _paired_condition_drop(
    rows: Sequence[Mapping[str, Any]],
    degraded_condition: str,
    *,
    baseline_condition: str = "standard",
) -> dict[str, Any] | None:
    by_condition: dict[str, dict[tuple[Any, ...], float]] = defaultdict(dict)
    for row in rows:
        condition = str(row.get("condition", "standard"))
        if condition not in (baseline_condition, degraded_condition):
            continue
        key = _pair_key(row, omit=set())
        if key in by_condition[condition]:
            raise ValueError(f"Duplicate {condition!r} diagnostic record for key {key}")
        by_condition[condition][key] = correctness(row)
    common = sorted(
        set(by_condition.get(baseline_condition, ())) & set(by_condition.get(degraded_condition, ())),
        key=repr,
    )
    if not common:
        return None
    differences = [
        by_condition[baseline_condition][key] - by_condition[degraded_condition][key] for key in common
    ]
    return {"n_pairs": len(common), "accuracy_drop": _mean(differences)}


def _distractor_damage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    baseline: dict[tuple[Any, ...], float] = {}
    forced: dict[int, dict[tuple[Any, ...], float]] = defaultdict(dict)
    for row in rows:
        if str(row.get("condition", "standard")) != "standard":
            continue
        protocol = row.get("protocol")
        # PrefEval sample/query IDs intentionally encode protocol and k, so they
        # cannot pair oracle-sparse with forced-write. Match the same underlying
        # base pair/form and experimental seed while excluding protocol-specific IDs.
        key = (
            row.get("base_pair_id"),
            row.get("form"),
            row.get("split"),
            row.get("method"),
            row.get("seed"),
            row.get("diffusion_seed"),
            row.get("recurrence_mode"),
            row.get("counterfactual_variant"),
            row.get("noop_policy"),
            row.get("condition", "standard"),
        )
        if protocol == "oracle-sparse":
            if key in baseline:
                raise ValueError(f"Duplicate oracle-sparse record for key {key}")
            baseline[key] = correctness(row)
        elif protocol == "forced-write":
            count = row.get("forced_write_k")
            if not isinstance(count, int) or count < 0:
                raise ValueError("forced-write records require a non-negative forced_write_k")
            if key in forced[count]:
                raise ValueError(f"Duplicate forced-write k={count} record for key {key}")
            forced[count][key] = correctness(row)

    result: dict[str, Any] = {}
    for count, values in sorted(forced.items()):
        common = set(baseline) & set(values)
        if common:
            result[str(count)] = {
                "n_pairs": len(common),
                "accuracy_damage": _mean([baseline[key] - values[key] for key in common]),
            }
    return result


def _noop_filter_effect(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    by_policy: dict[str, dict[tuple[Any, ...], float]] = {"keep": {}, "skip": {}}
    for row in rows:
        policy = row.get("noop_policy")
        intervention_pair_id = row.get("noop_intervention_pair_id")
        if policy not in by_policy or intervention_pair_id is None:
            continue
        key = (intervention_pair_id, *_pair_key(row, omit={"noop_policy"}))
        if key in by_policy[policy]:
            raise ValueError(f"Duplicate no-op {policy!r} intervention record for key {key}")
        by_policy[policy][key] = correctness(row)
    common = sorted(set(by_policy["keep"]) & set(by_policy["skip"]), key=repr)
    if not common:
        return None
    keep = [by_policy["keep"][key] for key in common]
    skip = [by_policy["skip"][key] for key in common]
    return {
        "n_pairs": len(common),
        "keep_accuracy": _mean(keep),
        "skip_accuracy": _mean(skip),
        "skip_minus_keep_accuracy": _mean(
            [skip_score - keep_score for keep_score, skip_score in zip(keep, skip, strict=True)]
        ),
    }


def diagnostic_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    extra_binary_fields: Sequence[str] = (),
) -> dict[str, Any]:
    """Summarize optional stale/distractor/reset/shuffle/swap diagnostics.

    The record schema is intentionally additive: callers may omit every diagnostic
    field for the main table and add them only for relevant controlled conditions.
    """

    rows = list(records)
    result: dict[str, Any] = {
        "reset": _paired_condition_drop(rows, "reset"),
        "shuffle": _paired_condition_drop(rows, "shuffle"),
        "state_swap": _paired_condition_drop(rows, "state_swap"),
        "distractor_damage_by_k": _distractor_damage(rows),
        "noop_filter_effect": _noop_filter_effect(rows),
    }

    stale_values: list[float] = []
    donor_values: list[float] = []
    for row in rows:
        if str(row.get("condition", "standard")) == "standard" and (
            row.get("stale_target_index") is not None or row.get("stale_target_choice") is not None
        ):
            stale = _index(row, "stale_target")
            prediction = _index(row, "prediction")
            target = _index(row, "target")
            stale_values.append(float(prediction == stale and prediction != target))
        if str(row.get("condition", "standard")) == "state_swap" and (
            row.get("donor_target_index") is not None or row.get("donor_target_choice") is not None
        ):
            donor_values.append(float(_index(row, "prediction") == _index(row, "donor_target")))
    result["stale_answer_error"] = (
        {"n": len(stale_values), "rate": _mean(stale_values)} if stale_values else None
    )
    result["state_swap_donor_answer"] = (
        {"n": len(donor_values), "rate": _mean(donor_values)} if donor_values else None
    )

    extras: dict[str, Any] = {}
    for field in extra_binary_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if any(value not in (0.0, 1.0) for value in values):
            raise ValueError(f"Extra metric {field!r} must contain only booleans/0/1")
        extras[field] = {"n": len(values), "rate": _mean(values)} if values else None
    result["extra_binary_metrics"] = extras
    return result


def compute_prefeval_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    extra_binary_fields: Sequence[str] = (),
) -> dict[str, Any]:
    rows = list(records)
    return {
        "headline": topic_form_metrics(rows),
        "diagnostics": diagnostic_metrics(rows, extra_binary_fields=extra_binary_fields),
    }


__all__ = [
    "compute_prefeval_metrics",
    "correctness",
    "diagnostic_metrics",
    "topic_form_metrics",
]
