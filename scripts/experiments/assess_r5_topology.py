"""Compare same-H200 and split-H200 R5 topology smokes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "vision_memory.r5-compose-topology-decision.v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _optimizer_losses(directory: Path) -> list[float]:
    values: list[float] = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "optimizer_step":
            values.append(float(row["loss_mean"]))
    return values


def assess(same_dir: Path, split_dir: Path) -> dict[str, Any]:
    same = _load_json(same_dir / "summary.json")
    split = _load_json(split_dir / "summary.json")
    if not same.get("same_device") or split.get("same_device"):
        raise ValueError("R5 topology inputs are not one same-device and one split-device run.")
    same_losses = _optimizer_losses(same_dir)
    split_losses = _optimizer_losses(split_dir)
    if len(same_losses) != len(split_losses) or not same_losses:
        raise ValueError("R5 topology runs have unmatched optimizer-step losses.")
    absolute = [abs(left - right) for left, right in zip(same_losses, split_losses, strict=True)]
    relative = [difference / max(abs(reference), 1e-12) for difference, reference in zip(absolute, split_losses, strict=True)]
    throughput_ratio = float(same["micro_segments_per_second"]) / float(split["micro_segments_per_second"])
    peak = max(float(same["updater_peak_memory_gib"]), float(same["reader_peak_memory_gib"]))
    checks = {
        "same_device_technical_gate": bool(same["technical_gate"]["passed"]),
        "split_device_technical_gate": bool(split["technical_gate"]["passed"]),
        "same_device_peak_below_110_gib": peak < 110.0,
        "same_device_throughput_at_least_55_percent": throughput_ratio >= 0.55,
        "loss_parity_absolute_1e_4_relative_1e_3": max(absolute) <= 1e-4 and max(relative) <= 1e-3,
        "all_losses_finite": all(math.isfinite(value) for value in same_losses + split_losses),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "decision": "single_h200_parallel_arms" if passed else "dual_h200_serial_latent_only",
        "passed": passed,
        "checks": checks,
        "same_device_peak_memory_gib": peak,
        "same_device_to_split_throughput_ratio": throughput_ratio,
        "loss_parity": {
            "steps": len(same_losses),
            "maximum_absolute_difference": max(absolute),
            "maximum_relative_difference": max(relative),
        },
        "same_summary": same,
        "split_summary": split,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--same-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.same_dir, args.split_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "checks": result["checks"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
