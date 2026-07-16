"""Fail-closed scientific scoring for R3 Set8 and Transition16 predictions."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_prediction_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def _correct(row: Mapping[str, Any]) -> bool:
    return int(row["prediction_index"]) == int(row["target_index"])


def _episode_kind(episode_id: str) -> str:
    prefix = "r3-transition-"
    if not episode_id.startswith(prefix):
        raise ValueError(f"Not a Transition16 episode id: {episode_id}")
    return episode_id[len(prefix) :].split("-", 1)[0]


def _read_form(episode_id: str) -> str:
    if "-mixed-" in episode_id:
        return "mixed"
    if "-separate-" in episode_id:
        return "separate"
    raise ValueError(f"Transition16 episode id has no read form: {episode_id}")


def _standard_delayed(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("condition") == "standard"
        and row.get("probe_role", "delayed") == "delayed"
        and row.get("choice_view_family") == "reverse-cyclic4"
    ]
    identities = [
        (str(row.get("episode_id")), int(row.get("query_ordinal", 0)), int(row.get("choice_view_index", -1)))
        for row in selected
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate standard delayed prediction identities")
    return selected


def _intervention_correct(rows: Iterable[dict[str, Any]], condition: str) -> tuple[int, int]:
    selected = [
        row
        for row in rows
        if row.get("condition") == condition
        and row.get("probe_role", "delayed") == "delayed"
        and row.get("choice_view_family") == "reverse-cyclic4"
    ]
    return sum(_correct(row) for row in selected), len(selected)


def _position_summary(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[int, list[bool]] = defaultdict(list)
    for row in rows:
        counts[int(row["target_index"])].append(_correct(row))
    return {
        str(position): {"correct": sum(values), "count": len(values)}
        for position, values in sorted(counts.items())
    }


def _state_rotation_summary(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_id"])].append(row)
    summary: dict[str, dict[str, Any]] = {}
    consistent = 0
    for episode_id, values in sorted(grouped.items()):
        predictions = [str(row["prediction_text"]) for row in values]
        is_consistent = len(set(predictions)) == 1
        consistent += int(is_consistent)
        summary[episode_id] = {
            "correct": sum(_correct(row) for row in values),
            "count": len(values),
            "predicted_text_consistent": is_consistent,
        }
    return summary, consistent


def score_set8(rows: list[dict[str, Any]]) -> dict[str, Any]:
    standard = _standard_delayed(rows)
    if len(standard) != 32:
        raise ValueError(f"Set8 requires exactly 32 standard delayed reverse views, got {len(standard)}")
    if any(not str(row.get("episode_id", "")).startswith("r3-set8-") for row in standard):
        raise ValueError("Set8 predictions contain a non-Set8 episode")
    positions = _position_summary(standard)
    if set(positions) != {"0", "1", "2", "3"} or any(item["count"] != 8 for item in positions.values()):
        raise ValueError(f"Set8 target positions are not 8/8/8/8: {positions}")
    states, consistent_count = _state_rotation_summary(standard)
    if len(states) != 8 or any(item["count"] != 4 for item in states.values()):
        raise ValueError("Set8 requires 8 states with four views each")
    correct = sum(_correct(row) for row in standard)
    reset_correct, reset_count = _intervention_correct(rows, "reset")
    shuffle_correct, shuffle_count = _intervention_correct(rows, "shuffle")
    if reset_count != 32 or shuffle_count != 32:
        raise ValueError("Set8 requires 32 reset and 32 shuffle delayed views")
    checks = {
        "accuracy": correct >= 30,
        "positions": all(item["correct"] >= 7 for item in positions.values()),
        "per_state": all(item["correct"] >= 3 for item in states.values()),
        "rotation_consistency": consistent_count >= 7,
        "reset_drop": correct - reset_correct >= 8,
        "shuffle_drop": correct - shuffle_correct >= 8,
    }
    payload = {
        "schema_version": "vlm.r3.set8_gate.v1",
        "suite": "set8",
        "correct": correct,
        "count": len(standard),
        "positions": positions,
        "states": states,
        "consistent_state_count": consistent_count,
        "interventions": {
            "reset": {"correct": reset_correct, "count": reset_count, "drop": correct - reset_correct},
            "shuffle": {
                "correct": shuffle_correct,
                "count": shuffle_count,
                "drop": correct - shuffle_correct,
            },
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    return {**payload, "scientific_payload_sha256": _canonical_sha256(payload)}


def _clean_noop_agreement(rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        pair_id = row.get("distractor_pair_id")
        variant = row.get("distractor_variant")
        if pair_id is None or variant not in {"clean", "distractor"}:
            continue
        grouped[(str(pair_id), int(row["choice_view_index"]))][str(variant)] = row
    agreements = 0
    valid = 0
    for pair in grouped.values():
        if set(pair) != {"clean", "distractor"}:
            continue
        clean = pair["clean"]
        distractor = pair["distractor"]
        if clean["choices"] != distractor["choices"] or clean["target_text"] != distractor["target_text"]:
            raise ValueError("A clean/noop pair changed choices or semantic target")
        valid += 1
        agreements += int(clean["prediction_text"] == distractor["prediction_text"])
    return agreements, valid


def score_transition16(rows: list[dict[str, Any]]) -> dict[str, Any]:
    standard = _standard_delayed(rows)
    if len(standard) != 64:
        raise ValueError(f"Transition16 requires 64 standard delayed reverse views, got {len(standard)}")
    positions = _position_summary(standard)
    if set(positions) != {"0", "1", "2", "3"} or any(item["count"] != 16 for item in positions.values()):
        raise ValueError(f"Transition16 target positions are not 16 each: {positions}")
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mixed: list[dict[str, Any]] = []
    for row in standard:
        episode_id = str(row["episode_id"])
        kind = _episode_kind(episode_id)
        read_form = _read_form(episode_id)
        by_kind[kind].append(row)
        by_cell[f"{kind}:{read_form}"].append(row)
        if read_form == "mixed":
            mixed.append(row)
    expected_kinds = {"set", "overwrite", "clear", "noop"}
    if set(by_kind) != expected_kinds or any(len(values) != 16 for values in by_kind.values()):
        raise ValueError("Transition16 terminal kinds must each contain 16 views")
    if len(by_cell) != 8 or any(len(values) != 8 for values in by_cell.values()):
        raise ValueError("Transition16 kind/read-form cells must each contain 8 views")
    if len(mixed) != 32:
        raise ValueError("Transition16 requires 32 mixed delayed views")
    agreement, pair_count = _clean_noop_agreement(standard)
    if pair_count != 16:
        raise ValueError(f"Transition16 requires 16 clean/noop view pairs, got {pair_count}")
    correct = sum(_correct(row) for row in standard)
    reset_correct, reset_count = _intervention_correct(rows, "reset")
    shuffle_correct, shuffle_count = _intervention_correct(rows, "shuffle")
    if reset_count != 64 or shuffle_count != 64:
        raise ValueError("Transition16 requires 64 reset and 64 shuffle delayed views")
    swap_probe = [
        row
        for row in rows
        if row.get("condition") == "state_swap"
        and row.get("probe_role", "delayed") == "delayed"
        and row.get("choice_view_family") == "reverse-cyclic4"
        and _episode_kind(str(row["episode_id"])) in {"set", "clear"}
        and int(row.get("choice_view_index", -1)) < 2
    ]
    if len(swap_probe) != 16 or any(row.get("donor_target_index") is None for row in swap_probe):
        raise ValueError("Transition16 requires the locked 16-record state-swap donor probe")
    donor_correct = sum(int(row["prediction_index"]) == int(row["donor_target_index"]) for row in swap_probe)
    kind_summary = {
        kind: {"correct": sum(_correct(row) for row in values), "count": len(values)}
        for kind, values in sorted(by_kind.items())
    }
    cell_summary = {
        cell: {"correct": sum(_correct(row) for row in values), "count": len(values)}
        for cell, values in sorted(by_cell.items())
    }
    mixed_correct = sum(_correct(row) for row in mixed)
    checks = {
        "accuracy": correct >= 58,
        "positions": all(item["correct"] >= 14 for item in positions.values()),
        "terminal_kinds": all(item["correct"] >= 14 for item in kind_summary.values()),
        "mixed": mixed_correct >= 28,
        "cells": all(item["correct"] >= 7 for item in cell_summary.values()),
        "clean_noop_agreement": agreement >= 15,
        "reset_drop": correct - reset_correct >= 16,
        "shuffle_drop": correct - shuffle_correct >= 16,
        "state_swap": donor_correct >= 12,
    }
    payload = {
        "schema_version": "vlm.r3.transition16_gate.v1",
        "suite": "transition16",
        "correct": correct,
        "count": len(standard),
        "positions": positions,
        "terminal_kinds": kind_summary,
        "cells": cell_summary,
        "mixed": {"correct": mixed_correct, "count": len(mixed)},
        "clean_noop_agreement": {"agreements": agreement, "pair_count": pair_count},
        "interventions": {
            "reset": {"correct": reset_correct, "count": reset_count, "drop": correct - reset_correct},
            "shuffle": {
                "correct": shuffle_correct,
                "count": shuffle_count,
                "drop": correct - shuffle_correct,
            },
            "state_swap": {"donor_answers": donor_correct, "count": len(swap_probe)},
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    return {**payload, "scientific_payload_sha256": _canonical_sha256(payload)}


def score_r3_micro(rows: list[dict[str, Any]], suite: str) -> dict[str, Any]:
    if suite == "set8":
        return score_set8(rows)
    if suite == "transition16":
        return score_transition16(rows)
    raise ValueError("suite must be 'set8' or 'transition16'")
