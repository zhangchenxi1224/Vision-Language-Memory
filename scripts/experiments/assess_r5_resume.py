"""Audit deterministic R5 checkpoint resume against an uninterrupted smoke run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


SCHEMA = "vision_memory.r5-compose-resume-audit.v1"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"Invalid R5 checkpoint: {path}")
    return value


def _tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return (
            isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return type(left) is type(right) and left == right


def _step_loss(directory: Path, step: int) -> float:
    matches: list[float] = []
    for line in (directory / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") == "optimizer_step" and row.get("optimizer_step") == step:
            matches.append(float(row["loss_mean"]))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one optimizer-step {step} loss in {directory}; got {len(matches)}.")
    return matches[0]


def assess(direct_dir: Path, resumed_dir: Path) -> dict[str, Any]:
    direct = _load_checkpoint(direct_dir / "endpoint_raw.pt")
    resumed = _load_checkpoint(resumed_dir / "endpoint_raw.pt")
    direct_trainable = direct.get("trainable_state")
    resumed_trainable = resumed.get("trainable_state")
    if not isinstance(direct_trainable, Mapping) or not isinstance(resumed_trainable, Mapping):
        raise ValueError("R5 resume audit checkpoints lack trainable_state mappings.")
    common = sorted(set(direct_trainable) & set(resumed_trainable))
    maximum_difference = max(
        (
            float((direct_trainable[name].float() - resumed_trainable[name].float()).abs().max())
            for name in common
        ),
        default=float("inf"),
    )
    direct_ema = direct.get("trainer_state", {}).get("ema_state")
    resumed_ema = resumed.get("trainer_state", {}).get("ema_state")
    direct_loss = _step_loss(direct_dir, 2)
    resumed_loss = _step_loss(resumed_dir, 2)
    checks = {
        "both_reach_step_2": direct.get("optimizer_step") == resumed.get("optimizer_step") == 2,
        "trainable_key_sets_match": set(direct_trainable) == set(resumed_trainable) and bool(common),
        "trainable_tensors_bitwise_equal": _tree_equal(direct_trainable, resumed_trainable),
        "ema_tensors_bitwise_equal": _tree_equal(direct_ema, resumed_ema),
        "optimizer_state_bitwise_equal": _tree_equal(direct.get("optimizer"), resumed.get("optimizer")),
        "step_2_loss_bitwise_equal": direct_loss == resumed_loss,
    }
    return {
        "schema": SCHEMA,
        "passed": all(checks.values()),
        "checks": checks,
        "direct_dir": str(direct_dir.resolve()),
        "resumed_dir": str(resumed_dir.resolve()),
        "maximum_trainable_absolute_difference": maximum_difference,
        "step_2_loss": {"direct": direct_loss, "resumed": resumed_loss},
        "resume_source": str((direct_dir / "checkpoints" / "step-000001.pt").resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--resumed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.direct_dir, args.resumed_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "checks": result["checks"]}, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
