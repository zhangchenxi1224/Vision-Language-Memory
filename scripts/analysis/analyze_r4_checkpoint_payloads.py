"""CPU-only analysis of R4 DreamLite checkpoint weights and training metrics.

This deliberately does not load DreamLite or Qwen weights.  It can therefore run
on a CPU notebook and answers whether LoRA parameters are still moving, which
projection/factor dominates the update, and where the recorded training metrics
change across checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def _label(name: str) -> dict[str, str]:
    bounded = f".{name}."
    stage = next(
        (value for value, marker in (
            ("down_blocks", ".down_blocks."),
            ("mid_block", ".mid_block."),
            ("up_blocks", ".up_blocks."),
        ) if marker in bounded),
        "other",
    )
    projection = next(
        (value for value, marker in (
            ("to_q", ".to_q."),
            ("to_k", ".to_k."),
            ("to_v", ".to_v."),
            ("to_out", ".to_out."),
        ) if marker in bounded),
        "other",
    )
    factor = next(
        (value for value, marker in (
            ("lora_A", ".lora_A."),
            ("lora_B", ".lora_B."),
        ) if marker in bounded),
        "other",
    )
    return {"stage": stage, "projection": projection, "factor": factor, "group": f"{projection}|{factor}"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_norm(value: torch.Tensor) -> float:
    return float(value.detach().float().square().sum().sqrt())


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if left.numel() != right.numel():
        return None
    denominator = float(left.norm() * right.norm())
    return None if denominator == 0.0 else float(torch.dot(left, right) / denominator)


def _group_stats(state: dict[str, torch.Tensor], previous: dict[str, torch.Tensor] | None) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"norm_sq": 0.0, "elements": 0, "tensors": 0})
    delta: dict[str, dict[str, Any]] = defaultdict(lambda: {"norm_sq": 0.0, "elements": 0, "cosines": []})
    for name, tensor in state.items():
        label = _label(name)
        group = label["group"]
        value = tensor.detach().float()
        buckets[group]["norm_sq"] += float(value.square().sum())
        buckets[group]["elements"] += value.numel()
        buckets[group]["tensors"] += 1
        if previous is not None and name in previous:
            difference = value - previous[name].detach().float()
            delta[group]["norm_sq"] += float(difference.square().sum())
            delta[group]["elements"] += difference.numel()
            cosine = _cosine(value, previous[name])
            if cosine is not None:
                delta[group]["cosines"].append(cosine)
    rows: list[dict[str, Any]] = []
    for group in sorted(buckets):
        item = buckets[group]
        change = delta[group]
        rows.append({
            "group": group,
            "parameter_norm": math.sqrt(item["norm_sq"]),
            "parameter_elements": item["elements"],
            "tensor_count": item["tensors"],
            "delta_norm_from_previous": math.sqrt(change["norm_sq"]) if previous is not None else None,
            "delta_cosine_mean": (
                sum(change["cosines"]) / len(change["cosines"]) if change["cosines"] else None
            ),
        })
    return rows


def _optimizer_rows(payload: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    optimizer = payload.get("optimizer") or {}
    groups = optimizer.get("param_groups") or []
    ids = list(groups[0].get("params", [])) if groups else []
    state = optimizer.get("state") or {}
    rows: list[dict[str, Any]] = []
    for index, parameter_id in enumerate(ids):
        value = state.get(parameter_id, {})
        row = {
            "parameter_name": names[index] if index < len(names) else f"param_{index}",
            "optimizer_parameter_id": parameter_id,
            "step": float(value.get("step", 0.0)) if value.get("step") is not None else 0.0,
        }
        for key in ("exp_avg", "exp_avg_sq"):
            tensor = value.get(key)
            row[f"{key}_norm"] = _tensor_norm(tensor) if isinstance(tensor, torch.Tensor) else None
        rows.append(row)
    return rows


def _flatten_counts(value: Any, prefix: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {f"{prefix}.{key}": item for key, item in value.items()}


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if value.get("kind") != "optimizer_step":
            continue
        row: dict[str, Any] = {
            "optimizer_step": value.get("optimizer_step"),
            "loss_mean": value.get("loss_mean"),
            "qa_loss_mean": value.get("qa_loss_mean"),
            "identity_normalized_mean": value.get("identity_normalized_mean"),
            "gradient_norm_before_clip": value.get("gradient_norm_before_clip"),
            "rgb_delta_rms_mean": value.get("rgb_delta_rms_mean"),
            "rgb_saturation_fraction_mean": value.get("rgb_saturation_fraction_mean"),
            "elapsed_seconds": value.get("elapsed_seconds"),
        }
        row.update(_flatten_counts(value.get("event_kind_counts"), "event"))
        row.update(_flatten_counts(value.get("diffusion_step_counts"), "diffusion"))
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(run_dir: Path, output_dir: Path, include_endpoints: bool) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "output" / "checkpoints"
    paths = sorted(checkpoint_dir.glob("step-*.pt"))
    if include_endpoints:
        paths.extend(path for path in (run_dir / "output" / "endpoint.pt", run_dir / "output" / "last.pt") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No R4 checkpoints found under {checkpoint_dir}")

    parameter_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    previous: dict[str, torch.Tensor] | None = None
    checkpoints: list[dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = {str(key): value.detach().cpu() for key, value in payload["trainable_state"].items()}
        step = int(payload.get("optimizer_step", 0))
        groups = _group_stats(state, previous)
        for row in groups:
            parameter_rows.append({"checkpoint": path.name, "optimizer_step": step, **row})
        names = list(state)
        for row in _optimizer_rows(payload, names):
            optimizer_rows.append({"checkpoint": path.name, "optimizer_step": step, **row})
        summary_rows.append({
            "checkpoint": path.name,
            "optimizer_step": step,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "parameter_count": sum(value.numel() for value in state.values()),
            "parameter_norm": math.sqrt(sum(float(value.float().square().sum()) for value in state.values())),
            "trainer_state": payload.get("trainer_state", {}),
        })
        checkpoints.append({"name": path.name, "optimizer_step": step, "manifest": payload.get("manifest", {})})
        previous = state

    metrics = _read_metrics(run_dir / "output" / "metrics.jsonl")
    _write_csv(output_dir / "checkpoint_summary.csv", summary_rows)
    _write_csv(output_dir / "checkpoint_parameter_groups.csv", parameter_rows)
    _write_csv(output_dir / "checkpoint_optimizer_state.csv", optimizer_rows)
    _write_csv(output_dir / "metrics_by_step.csv", metrics)
    report = {
        "schema": "vision_memory.r4-checkpoint-payload-analysis.v1",
        "run_dir": str(run_dir),
        "checkpoint_count": len(paths),
        "checkpoints": checkpoints,
        "summary": summary_rows,
        "metrics_rows": len(metrics),
        "notes": [
            "Parameter deltas are relative to the preceding saved checkpoint, not the unrecorded step-0 initialization.",
            "Metrics are copied from metrics.jsonl; no checkpoint is selected by this analysis.",
        ],
    }
    (output_dir / "checkpoint_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--include-endpoints", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "analysis" / "checkpoint_payloads"
    report = analyze(args.run_dir, output_dir, args.include_endpoints)
    print(json.dumps({
        "checkpoint_count": report["checkpoint_count"],
        "metrics_rows": report["metrics_rows"],
        "output_dir": str(output_dir),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
