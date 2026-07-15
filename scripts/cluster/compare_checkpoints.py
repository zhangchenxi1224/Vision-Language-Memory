from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def compare_values(reference: Any, resumed: Any, *, atol: float, rtol: float, path: str) -> list[str]:
    if isinstance(reference, torch.Tensor):
        if not isinstance(resumed, torch.Tensor):
            return [f"{path}: resumed value is not a tensor"]
        if reference.shape != resumed.shape or reference.dtype != resumed.dtype:
            return [
                f"{path}: tensor metadata differs "
                f"({tuple(reference.shape)}, {reference.dtype}) != ({tuple(resumed.shape)}, {resumed.dtype})"
            ]
        if not torch.allclose(reference, resumed, atol=atol, rtol=rtol, equal_nan=False):
            difference = (reference.to(torch.float64) - resumed.to(torch.float64)).abs()
            return [f"{path}: tensor max_abs_difference={float(difference.max().item())}"]
        return []
    if isinstance(reference, np.ndarray):
        if not isinstance(resumed, np.ndarray):
            return [f"{path}: resumed value is not a NumPy array"]
        if reference.shape != resumed.shape or reference.dtype != resumed.dtype:
            return [
                f"{path}: array metadata differs "
                f"({reference.shape}, {reference.dtype}) != ({resumed.shape}, {resumed.dtype})"
            ]
        if not np.allclose(reference, resumed, atol=atol, rtol=rtol, equal_nan=False):
            difference = np.abs(reference.astype(np.float64) - resumed.astype(np.float64))
            return [f"{path}: array max_abs_difference={float(difference.max())}"]
        return []
    if isinstance(reference, dict):
        if not isinstance(resumed, dict):
            return [f"{path}: resumed value is not a mapping"]
        errors: list[str] = []
        if set(reference) != set(resumed):
            errors.append(
                f"{path}: mapping keys differ; missing={sorted(set(reference) - set(resumed), key=repr)}, "
                f"unexpected={sorted(set(resumed) - set(reference), key=repr)}"
            )
            return errors
        for key in sorted(reference, key=repr):
            errors.extend(
                compare_values(reference[key], resumed[key], atol=atol, rtol=rtol, path=f"{path}.{key}")
            )
        return errors
    if isinstance(reference, (list, tuple)):
        if not isinstance(resumed, type(reference)) or len(reference) != len(resumed):
            return [f"{path}: sequence type or length differs"]
        errors = []
        for index, (left, right) in enumerate(zip(reference, resumed)):
            errors.extend(compare_values(left, right, atol=atol, rtol=rtol, path=f"{path}[{index}]"))
        return errors
    if isinstance(reference, float):
        if not isinstance(resumed, (int, float)) or not math.isclose(reference, float(resumed), abs_tol=atol, rel_tol=rtol):
            return [f"{path}: {reference!r} != {resumed!r}"]
        return []
    return [] if reference == resumed else [f"{path}: {reference!r} != {resumed!r}"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify exact/close DreamLite checkpoint-resume continuation")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--resumed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        raise SystemExit("Checkpoint comparison tolerances must be non-negative.")

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    resumed = torch.load(args.resumed, map_location="cpu", weights_only=False)
    required = {
        "schema_version",
        "trainable_state",
        "optimizer",
        "epoch",
        "episode_cursor",
        "optimizer_step",
        "rng_state",
        "manifest",
        "trainer_state",
    }
    missing = {
        "reference": sorted(required - set(reference)),
        "resumed": sorted(required - set(resumed)),
    }
    errors: list[str] = []
    if any(missing.values()):
        errors.append(f"Required checkpoint fields are missing: {missing}")
    else:
        for key in sorted(required):
            errors.extend(compare_values(reference[key], resumed[key], atol=args.atol, rtol=args.rtol, path=key))

    report = {
        "schema_version": 1,
        "reference": str(args.reference.resolve()),
        "resumed": str(args.resumed.resolve()),
        "atol": args.atol,
        "rtol": args.rtol,
        "missing": missing,
        "mismatch_count": len(errors),
        "mismatches": errors[:100],
        "passed": not errors,
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
