from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a labeled, non-comparative prediction summary")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{args.predictions}:{line_number} is not a JSON object")
            if not isinstance(record.get("prediction_index"), int) or not isinstance(record.get("target_index"), int):
                raise ValueError(f"{args.predictions}:{line_number} has no integer prediction/target index")
            records.append(record)
    if not records:
        raise ValueError("No prediction records were provided.")

    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    topic_form: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for record in records:
        correct = float(record["prediction_index"] == record["target_index"])
        key = tuple(
            str(record.get(field, "unknown"))
            for field in (
                "method",
                "seed",
                "split",
                "condition",
                "protocol",
                "form",
                "noop_policy",
                "recurrence_mode",
                "diffusion_seed",
            )
        )
        grouped[key].append(correct)
        topic_form[
            (
                str(record.get("method", "unknown")),
                str(record.get("seed", "unknown")),
                str(record.get("topic", "unknown")),
                str(record.get("form", "unknown")),
            )
        ].append(correct)

    groups = [
        {
            "method": key[0],
            "seed": key[1],
            "split": key[2],
            "condition": key[3],
            "protocol": key[4],
            "form": key[5],
            "noop_policy": key[6],
            "recurrence_mode": key[7],
            "diffusion_seed": key[8],
            "n": len(values),
            "accuracy": statistics.fmean(values),
        }
        for key, values in sorted(grouped.items())
    ]
    topic_form_groups = [
        {
            "method": key[0],
            "seed": key[1],
            "topic": key[2],
            "form": key[3],
            "n": len(values),
            "accuracy": statistics.fmean(values),
        }
        for key, values in sorted(topic_form.items())
    ]
    report = {
        "schema_version": 1,
        "label": args.label,
        "predictions": str(args.predictions.resolve()),
        "records": len(records),
        "micro_accuracy": statistics.fmean(
            float(record["prediction_index"] == record["target_index"]) for record in records
        ),
        "groups": groups,
        "topic_form_groups": topic_form_groups,
        "topic_form_macro_accuracy": statistics.fmean(item["accuracy"] for item in topic_form_groups),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "label": args.label, "records": len(records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
