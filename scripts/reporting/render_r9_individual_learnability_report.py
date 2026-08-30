"""Render and integrity-check a completed eight-target R9 diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import compare_r9_individual_learnability as comparison  # noqa: E402


COLORS = {
    "F2": "#2563eb",
    "F3": "#dc2626",
    "F5": "#059669",
    "F6": "#7c3aed",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"R9 JSONL row is not an object: {path}:{line_number}")
            values.append(value)
    return values


def _optimizer_metrics(root: Path) -> list[dict[str, Any]]:
    values = [
        row
        for row in _load_jsonl(root / "run" / "metrics.jsonl")
        if row.get("kind") == "optimizer_step"
    ]
    if len(values) != 128 or [row.get("optimizer_step") for row in values] != list(range(1, 129)):
        raise ValueError(f"R9 target does not contain the exact 128-step metric sequence: {root}")
    required = ("loss_mean", "learning_rate", "gradient_norm_before_clip", "gradient_clipped")
    for row in values:
        if any(key not in row for key in required):
            raise ValueError(f"R9 optimizer metric lacks required diagnostics: {root}")
        if any(
            not math.isfinite(float(row[key]))
            for key in ("loss_mean", "learning_rate", "gradient_norm_before_clip")
        ):
            raise ValueError(f"R9 optimizer metric is non-finite: {root}")
    return values


def _moving_mean(values: Sequence[float], window: int = 16) -> list[float]:
    return [statistics.fmean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def _update_weight_ratio(row: Mapping[str, Any]) -> float | None:
    value = (
        row.get("optimizer_diagnostics", {})
        .get("updates_after_step", {})
        .get("global", {})
        .get("update_weight_ratio")
    )
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _write_training_csv(
    path: Path,
    targets: Sequence[Mapping[str, Any]],
    metrics: Mapping[int, Sequence[Mapping[str, Any]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "target_index",
                "family",
                "segment_id",
                "optimizer_step",
                "learning_rate",
                "loss",
                "loss_moving_mean_16",
                "gradient_norm_before_clip",
                "gradient_clipped",
                "update_weight_ratio",
                "state_gradient_nonzero_fraction",
            )
        )
        for target in targets:
            index = int(target["target_index"])
            rows = metrics[index]
            smooth = _moving_mean([float(row["loss_mean"]) for row in rows])
            for row, moving in zip(rows, smooth, strict=True):
                writer.writerow(
                    (
                        index,
                        target["target_family"],
                        target["target_segment_id"],
                        row["optimizer_step"],
                        row["learning_rate"],
                        row["loss_mean"],
                        moving,
                        row["gradient_norm_before_clip"],
                        row["gradient_clipped"],
                        _update_weight_ratio(row),
                        row.get("state_gradient_nonzero_fraction"),
                    )
                )


def _training_figure(
    path: Path,
    targets: Sequence[Mapping[str, Any]],
    metrics: Mapping[int, Sequence[Mapping[str, Any]]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    for target in targets:
        index = int(target["target_index"])
        family = str(target["target_family"])
        rows = metrics[index]
        steps = [int(row["optimizer_step"]) for row in rows]
        losses = [float(row["loss_mean"]) for row in rows]
        gradients = [float(row["gradient_norm_before_clip"]) for row in rows]
        ratios = [_update_weight_ratio(row) for row in rows]
        color = COLORS[family]
        label = f"t{index} {family}"
        axes[0, 0].plot(steps, _moving_mean(losses), color=color, alpha=0.78, linewidth=1.2, label=label)
        axes[0, 1].plot(steps, gradients, color=color, alpha=0.68, linewidth=0.9, label=label)
        valid_ratio = [(step, value) for step, value in zip(steps, ratios, strict=True) if value is not None]
        if valid_ratio:
            axes[1, 0].plot(
                [value[0] for value in valid_ratio],
                [value[1] for value in valid_ratio],
                color=color,
                alpha=0.72,
                linewidth=0.9,
                label=label,
            )
        axes[1, 1].plot(
            steps,
            [float(row["learning_rate"]) for row in rows],
            color=color,
            alpha=0.62,
            linewidth=0.9,
            label=label,
        )
    axes[0, 0].set(title="Target training loss (16-step moving mean)", xlabel="optimizer step", ylabel="listwise CE")
    axes[0, 1].axhline(10.0, color="black", linestyle="--", linewidth=0.9, label="clip=10")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Pre-clip gradient norm", xlabel="optimizer step", ylabel="L2 norm (log)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Realized update / weight ratio", xlabel="optimizer step", ylabel="ratio (log)")
    axes[1, 1].set(title="Frozen learning-rate schedule", xlabel="optimizer step", ylabel="learning rate")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _endpoint_figure(path: Path, targets: Sequence[Mapping[str, Any]]) -> None:
    indices = [int(row["target_index"]) for row in targets]
    colors = [COLORS[str(row["target_family"])] for row in targets]
    relative = [float(row["target_statistics"]["relative_change"]) for row in targets]
    views = [int(row["target_statistics"]["improved_choice_views"]) for row in targets]
    accuracy = [float(row["target_statistics"]["accuracy_delta"]) for row in targets]
    did = [float(row["target_statistics"]["normal_reset_difference_in_differences"]) for row in targets]
    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    axes[0, 0].bar(indices, relative, color=colors)
    axes[0, 0].axhline(-0.20, color="black", linestyle="--", label="gate <= -20%")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].set(title="Target normal CE relative change", xlabel="target", ylabel="endpoint / M0 - 1")
    axes[0, 1].bar(indices, views, color=colors)
    axes[0, 1].axhline(4, color="black", linestyle="--", label="gate = 4/4")
    axes[0, 1].set_ylim(0, 4.4)
    axes[0, 1].set(title="Fixed choice views improved", xlabel="target", ylabel="views")
    axes[1, 0].bar(indices, accuracy, color=colors)
    axes[1, 0].axhline(0.25, color="black", linestyle="--", label="gate >= +0.25")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(title="Target normal accuracy change", xlabel="target", ylabel="endpoint - M0")
    axes[1, 1].bar(indices, did, color=colors)
    axes[1, 1].axhline(0.0, color="black", linestyle="--", label="gate < 0")
    axes[1, 1].set(title="Normal/reset difference-in-differences", xlabel="target", ylabel="CE DiD")
    for axis in axes.flat:
        axis.set_xticks(indices)
        axis.grid(axis="y", alpha=0.18)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _refresh_delivery(output_dir: Path, targets: Sequence[Mapping[str, Any]]) -> None:
    artifacts = []
    for path in sorted(value for value in output_dir.iterdir() if value.is_file()):
        if path.name == "DELIVERY_MANIFEST.json":
            continue
        artifacts.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": comparison._sha256(path)}
        )
    comparison._write_json(
        output_dir / "DELIVERY_MANIFEST.json",
        {
            "schema": comparison.DELIVERY_SCHEMA,
            "artifacts": artifacts,
            "source_inventory_sha256": {
                str(row["target_index"]): row["inventory_sha256"] for row in targets
            },
        },
    )


def render(run_root: Path, output_dir: Path) -> dict[str, Any]:
    result = comparison.compare(run_root, output_dir)
    targets = result["targets"]
    metrics = {
        index: _optimizer_metrics(run_root / f"target-{index:02d}")
        for index in range(8)
    }
    _write_training_csv(output_dir / "training_metrics.csv", targets, metrics)
    _training_figure(output_dir / "training_diagnostics.png", targets, metrics)
    _endpoint_figure(output_dir / "endpoint_metrics.png", targets)
    raw_artifacts = {
        "schema": "vision_memory.r9-individual-learnability-source-artifacts.v1",
        "run_root": str(run_root.resolve()),
        "target_roots": {str(row["target_index"]): row["source_root"] for row in targets},
        "summary_sha256": {str(row["target_index"]): row["summary_sha256"] for row in targets},
        "terminal_sha256": {str(row["target_index"]): row["terminal_sha256"] for row in targets},
        "inventory_sha256": {str(row["target_index"]): row["inventory_sha256"] for row in targets},
    }
    comparison._write_json(output_dir / "RAW_ARTIFACTS.json", raw_artifacts)
    analysis = {
        "schema": "vision_memory.r9-individual-learnability-rendered-analysis.v1",
        "status": "completed",
        "formal_success_claim": False,
        "decision": result["decision"],
        "reason": result["reason"],
        "pass_count": result["pass_count"],
        "passed_target_indices": result["passed_target_indices"],
        "failed_target_indices": result["failed_target_indices"],
        "git_commit": result["git_commit"],
        "implementation_revision": result["implementation_revision"],
        "selected_segments_sha256": result["selected_segments_sha256"],
        "targets": targets,
    }
    comparison._write_json(output_dir / "ANALYSIS.json", analysis)
    _refresh_delivery(output_dir, targets)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = render(args.run_root, args.output_dir)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "decision": analysis["decision"],
                "pass_count": analysis["pass_count"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
