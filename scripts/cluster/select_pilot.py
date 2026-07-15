from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from pathlib import Path
from typing import Any, Iterable


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    if not records:
        raise ValueError(f"No prediction records found: {path}")
    return records


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def pair_key(record: dict[str, Any]) -> tuple[str, str]:
    episode_id = record.get("episode_id")
    query_id = record.get("query_id")
    if not isinstance(episode_id, str) or not isinstance(query_id, str):
        raise ValueError("Every pilot prediction needs string episode_id and query_id fields.")
    return episode_id, query_id


def correct(record: dict[str, Any]) -> float:
    prediction = record.get("prediction_index")
    target = record.get("target_index")
    if not isinstance(prediction, int) or not isinstance(target, int):
        raise ValueError("Every pilot prediction needs integer prediction_index and target_index fields.")
    return float(prediction == target)


def condition_scores(
    records: Iterable[dict[str, Any]],
    *,
    condition: str,
    method: str | None = None,
) -> dict[tuple[str, str], float]:
    selected: dict[tuple[str, str], float] = {}
    for record in records:
        if record.get("condition", "standard") != condition:
            continue
        if method is not None and record.get("method") != method:
            continue
        if record.get("noop_policy", "keep") != "keep":
            continue
        key = pair_key(record)
        if key in selected:
            raise ValueError(f"Duplicate {condition!r} prediction for {key!r}")
        selected[key] = correct(record)
    if not selected:
        suffix = "" if method is None else f" for method {method!r}"
        raise ValueError(f"No {condition!r} prediction records{suffix}.")
    return selected


def accuracy(scores: dict[tuple[str, str], float]) -> float:
    return sum(scores.values()) / len(scores)


def require_same_keys(reference: dict, candidate: dict, *, label: str) -> None:
    if set(reference) != set(candidate):
        only_reference = len(set(reference) - set(candidate))
        only_candidate = len(set(candidate) - set(reference))
        raise ValueError(
            f"Unpaired pilot predictions for {label}: "
            f"{only_reference} only in reference, {only_candidate} only in candidate"
        )


def evaluate_selection(
    specification: dict[str, Any],
    *,
    minimum_gain: float,
    minimum_intervention_drop: float,
) -> dict[str, Any]:
    if specification.get("selection_split") != "dev":
        raise ValueError("Pilot selection must be locked to dev; test data is forbidden.")
    candidates = specification.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("Pilot specification must contain exactly three candidates.")

    blank_path = Path(str(specification["blank_predictions"]))
    frozen_path = Path(str(specification["frozen_predictions"]))
    blank = condition_scores(read_jsonl(blank_path), condition="standard", method="query_only")
    frozen = condition_scores(read_jsonl(frozen_path), condition="standard", method="frozen_dreamlite")
    require_same_keys(blank, frozen, label="blank versus frozen")

    candidate_reports: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Every pilot candidate must be a JSON object.")
        learning_rate = float(candidate["learning_rate"])
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError(f"Invalid pilot learning rate: {learning_rate}")
        summary_path = Path(str(candidate["summary"]))
        predictions_path = Path(str(candidate["predictions"]))
        checkpoint = Path(str(candidate["checkpoint"]))
        resume_checkpoint = Path(str(candidate["resume_checkpoint"]))
        for required in (summary_path, predictions_path, checkpoint, resume_checkpoint):
            if not required.is_file() or required.stat().st_size <= 0:
                raise ValueError(f"Pilot artifact is missing or empty: {required}")

        summary = read_json(summary_path)
        dev_loss = summary.get("best_dev_loss")
        optimizer_steps = summary.get("optimizer_steps")
        if not isinstance(dev_loss, (int, float)) or not math.isfinite(float(dev_loss)):
            raise ValueError(f"Pilot {learning_rate} has no finite best_dev_loss.")
        if not isinstance(optimizer_steps, int) or optimizer_steps <= 0:
            raise ValueError(f"Pilot {learning_rate} has no positive optimizer_steps.")

        records = read_jsonl(predictions_path)
        standard = condition_scores(records, condition="standard")
        reset = condition_scores(records, condition="reset")
        shuffle = condition_scores(records, condition="shuffle")
        require_same_keys(blank, standard, label=f"blank versus lr={learning_rate}")
        require_same_keys(frozen, standard, label=f"frozen versus lr={learning_rate}")
        require_same_keys(standard, reset, label=f"standard versus reset at lr={learning_rate}")
        require_same_keys(standard, shuffle, label=f"standard versus shuffle at lr={learning_rate}")

        standard_accuracy = accuracy(standard)
        candidate_reports.append(
            {
                "learning_rate": learning_rate,
                "candidate_dir": str(candidate["candidate_dir"]),
                "checkpoint": str(checkpoint),
                "resume_checkpoint": str(resume_checkpoint),
                "best_dev_loss": float(dev_loss),
                "optimizer_steps": optimizer_steps,
                "queries": len(standard),
                "accuracy": standard_accuracy,
                "blank_accuracy": accuracy(blank),
                "frozen_accuracy": accuracy(frozen),
                "gain_over_blank": standard_accuracy - accuracy(blank),
                "gain_over_frozen": standard_accuracy - accuracy(frozen),
                "reset_drop": standard_accuracy - accuracy(reset),
                "shuffle_drop": standard_accuracy - accuracy(shuffle),
            }
        )

    selected = min(candidate_reports, key=lambda item: (item["best_dev_loss"], item["learning_rate"]))
    checks = {
        "gain_over_blank": selected["gain_over_blank"] >= minimum_gain,
        "gain_over_frozen": selected["gain_over_frozen"] >= minimum_gain,
        "reset_or_shuffle_drop": max(selected["reset_drop"], selected["shuffle_drop"])
        >= minimum_intervention_drop,
        "finite_dev_loss": math.isfinite(selected["best_dev_loss"]),
        "checkpoint_present": Path(selected["checkpoint"]).is_file(),
        "resume_checkpoint_present": Path(selected["resume_checkpoint"]).is_file(),
    }
    return {
        "schema_version": 1,
        "selection_split": "dev",
        "selection_rule": "minimum best_dev_loss; ties resolved by lower learning_rate",
        "minimum_gain": minimum_gain,
        "minimum_intervention_drop": minimum_intervention_drop,
        "candidates": candidate_reports,
        "selected": selected,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select and gate the three DreamLite learning-rate pilots")
    parser.add_argument("--specification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path, required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.10)
    parser.add_argument("--minimum-intervention-drop", type=float, default=0.10)
    args = parser.parse_args()
    if not 0 <= args.minimum_gain <= 1 or not 0 <= args.minimum_intervention_drop <= 1:
        raise SystemExit("Pilot thresholds must lie in [0, 1].")

    report = evaluate_selection(
        read_json(args.specification),
        minimum_gain=args.minimum_gain,
        minimum_intervention_drop=args.minimum_intervention_drop,
    )
    write_atomic(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    selected = report["selected"]
    env_lines = {
        "VLM_SELECTED_LR": str(selected["learning_rate"]),
        "VLM_SELECTED_DIR": str(selected["candidate_dir"]),
        "VLM_SELECTED_CHECKPOINT": str(selected["checkpoint"]),
        "VLM_SELECTED_RESUME_CHECKPOINT": str(selected["resume_checkpoint"]),
    }
    write_atomic(
        args.env_output,
        "".join(f"export {name}={shlex.quote(value)}\n" for name, value in env_lines.items()),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
