"""Apply the preregistered lexicographic R5 pilot selection rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "vision_memory.r5-compose-pilot-selection.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _longest_gap_margin(evaluation: Mapping[str, Any]) -> tuple[int | None, float | None]:
    breakdown = evaluation.get("breakdowns", {}).get("query_gap", {})
    values: list[tuple[int, float]] = []
    for key, metric in breakdown.items():
        parts = str(key).split("|")
        if len(parts) != 4 or parts[:3] != ["ema_step128", "mechanism_select_32", "normal"]:
            continue
        try:
            gap = int(parts[3])
            margin = float(metric["mean_margin"])
        except (KeyError, TypeError, ValueError):
            continue
        values.append((gap, margin))
    return max(values, default=(None, None), key=lambda item: -1 if item[0] is None else item[0])


def candidate(directory: Path) -> dict[str, Any]:
    summary = _load(directory / "summary.json")
    selection = summary.get("pilot_selection")
    evaluation = summary.get("pilot_endpoint_evaluation")
    if not isinstance(selection, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError(f"R5 pilot is missing fixed endpoint selection evidence: {directory}")
    gap, margin = _longest_gap_margin(evaluation)
    return {
        "name": directory.name,
        "directory": str(directory.resolve()),
        "persistent_state": summary["persistent_state"],
        "tbptt_horizon": int(summary["tbptt_horizon"]),
        "gradient_mode": summary["gradient_mode"],
        "selected_step_count": int(summary["selected_step_count"]),
        "technical_gate_passed": bool(selection["technical_gate_passed"]),
        "mechanism_gate_passed": bool(selection["mechanism_gate_passed"]),
        "eligible": bool(selection["eligible_for_selection"]),
        "delayed_ce": float(selection["delayed_mechanism_ce"]["endpoint"]),
        "delayed_ce_delta": float(selection["delayed_mechanism_ce"]["delta"]),
        "formal_ce": float(selection["formal_select_ce"]["endpoint"]),
        "formal_ce_delta": float(selection["formal_select_ce"]["delta"]),
        "longest_observed_gap": gap,
        "longest_gap_margin": margin,
        "elapsed_seconds": float(summary["elapsed_seconds"]),
    }


def select(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("R5 pilot selection requires at least one candidate.")
    eligible = [value for value in candidates if value["eligible"]]
    if not eligible:
        return {
            "schema": SCHEMA,
            "decision": "no_eligible_pilot",
            "winner": None,
            "candidates": candidates,
            "reason": "No pilot passed both the technical and mechanism gates.",
        }

    def ranking(value: Mapping[str, Any]) -> tuple[Any, ...]:
        margin = value["longest_gap_margin"]
        return (
            value["delayed_ce"],
            value["formal_ce"],
            -(float(margin) if margin is not None else float("-inf")),
            value["elapsed_seconds"],
            value["name"],
        )

    winner = min(eligible, key=ranking)
    return {
        "schema": SCHEMA,
        "decision": "winner_selected",
        "winner": winner,
        "candidates": candidates,
        "ranking_rule": [
            "lowest_delayed_dev_ce",
            "lowest_formal_dev_ce",
            "highest_longest_observed_gap_margin",
            "shortest_wall_clock",
        ],
        "gap4_note": (
            "The fixed F1-F6 curriculum contains at most three updater calls (gap=3); "
            "the preregistered gap-4 tie-break therefore falls back to the longest observed gap."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select([candidate(path) for path in args.pilot_dir])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "winner": result["winner"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
