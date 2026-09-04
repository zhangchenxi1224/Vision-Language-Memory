"""Render and integrity-check a completed sixteen-run R10 diagnostic."""

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

from scripts.experiments import compare_r10_visual_alignment as comparison  # noqa: E402


ARM_COLORS = {
    "direct-pixel-oracle": "#2563eb",
    "dreamlite-single-set": "#ea580c",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"R10 JSONL row is not an object: {path}:{line_number}")
            values.append(value)
    return values


def _optimizer_metrics(root: Path, arm: str) -> list[dict[str, Any]]:
    values = [
        row
        for row in _load_jsonl(root / "run" / "metrics.jsonl")
        if row.get("kind") == "optimizer_step"
    ]
    if len(values) != 128 or [row.get("optimizer_step") for row in values] != list(range(1, 129)):
        raise ValueError(f"R10 target does not contain the exact 128-step metric sequence: {root}")
    if arm == "direct-pixel-oracle":
        required = (
            "loss_before_step",
            "learning_rate",
            "gradient_norm",
            "gradient_nonzero_fraction",
            "image_min_after_step",
            "image_max_after_step",
            "image_saturation_fraction_after_step",
        )
    else:
        required = (
            "loss_mean",
            "learning_rate",
            "gradient_norm_before_clip",
            "gradient_clipped",
            "state_gradient_nonzero_fraction",
            "image_min",
            "image_max",
            "image_saturation_fraction_mean",
        )
    for row in values:
        if any(key not in row for key in required):
            raise ValueError(f"R10 optimizer metric lacks required diagnostics: {root}")
        numeric = [key for key in required if key != "gradient_clipped"]
        if any(not math.isfinite(float(row[key])) for key in numeric):
            raise ValueError(f"R10 optimizer metric is non-finite: {root}")
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


def _normalized_metric(arm: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if arm == "direct-pixel-oracle":
        return {
            "loss": float(row["loss_before_step"]),
            "gradient_norm": float(row["gradient_norm"]),
            "gradient_clipped": False,
            "state_gradient_nonzero_fraction": float(row["gradient_nonzero_fraction"]),
            "image_min": float(row["image_min_after_step"]),
            "image_max": float(row["image_max_after_step"]),
            "image_saturation_fraction": float(row["image_saturation_fraction_after_step"]),
            "update_weight_ratio": None,
        }
    return {
        "loss": float(row["loss_mean"]),
        "gradient_norm": float(row["gradient_norm_before_clip"]),
        "gradient_clipped": bool(row["gradient_clipped"]),
        "state_gradient_nonzero_fraction": float(row["state_gradient_nonzero_fraction"]),
        "image_min": float(row["image_min"]),
        "image_max": float(row["image_max"]),
        "image_saturation_fraction": float(row["image_saturation_fraction_mean"]),
        "update_weight_ratio": _update_weight_ratio(row),
    }


def _write_training_csv(
    path: Path,
    result: Mapping[str, Any],
    metrics: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "arm", "target_index", "segment_id", "optimizer_step", "learning_rate", "loss",
                "loss_moving_mean_16", "gradient_norm", "gradient_clipped", "update_weight_ratio",
                "state_gradient_nonzero_fraction", "image_min", "image_max", "image_saturation_fraction",
            )
        )
        for arm in comparison.ARMS:
            for target in result["arms"][arm]["targets"]:
                index = int(target["target_index"])
                rows = metrics[(arm, index)]
                normalized = [_normalized_metric(arm, row) for row in rows]
                smooth = _moving_mean([row["loss"] for row in normalized])
                for source, row, moving in zip(rows, normalized, smooth, strict=True):
                    writer.writerow(
                        (
                            arm, index, target["target_segment_id"], source["optimizer_step"],
                            source["learning_rate"], row["loss"], moving, row["gradient_norm"],
                            row["gradient_clipped"], row["update_weight_ratio"],
                            row["state_gradient_nonzero_fraction"], row["image_min"], row["image_max"],
                            row["image_saturation_fraction"],
                        )
                    )


def _training_figure(
    path: Path,
    result: Mapping[str, Any],
    metrics: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    for column, arm in enumerate(comparison.ARMS):
        color = ARM_COLORS[arm]
        for target in result["arms"][arm]["targets"]:
            index = int(target["target_index"])
            rows = metrics[(arm, index)]
            normalized = [_normalized_metric(arm, row) for row in rows]
            steps = [int(row["optimizer_step"]) for row in rows]
            alpha = 0.42 + 0.07 * index
            axes[0, column].plot(
                steps,
                _moving_mean([row["loss"] for row in normalized]),
                color=color,
                alpha=min(alpha, 0.95),
                linewidth=1.0,
                label=f"t{index}",
            )
            axes[1, column].plot(
                steps,
                [row["gradient_norm"] for row in normalized],
                color=color,
                alpha=min(alpha, 0.95),
                linewidth=0.8,
                label=f"t{index}",
            )
        axes[0, column].set_yscale("log")
        axes[0, column].set(
            title=f"{arm}: training loss (16-step mean)",
            xlabel="optimizer step",
            ylabel="listwise CE (log)",
        )
        axes[1, column].set_yscale("log")
        axes[1, column].set(
            title=f"{arm}: gradient norm",
            xlabel="optimizer step",
            ylabel="L2 norm (log)",
        )
        if arm == "dreamlite-single-set":
            axes[1, column].axhline(10.0, color="black", linestyle="--", linewidth=0.9, label="clip=10")
        for row in range(2):
            axes[row, column].grid(alpha=0.18)
            axes[row, column].legend(fontsize=7, ncol=3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _endpoint_figure(path: Path, result: Mapping[str, Any]) -> None:
    indices = list(range(8))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    fields = (
        ("relative_change", "Normal CE relative change", "endpoint / M0 - 1", -0.20),
        ("improved_choice_views", "Fixed reverse-cyclic views improved", "views", 4.0),
        ("accuracy_delta", "Normal accuracy change", "endpoint - M0", 0.25),
        ("normal_reset_difference_in_differences", "Normal/reset difference-in-differences", "CE DiD", 0.0),
    )
    for axis, (field, title, ylabel, threshold) in zip(axes.flat, fields, strict=True):
        for offset, arm in ((-width / 2, comparison.ARMS[0]), (width / 2, comparison.ARMS[1])):
            values = [float(row["target_statistics"][field]) for row in result["arms"][arm]["targets"]]
            axis.bar(
                [index + offset for index in indices],
                values,
                width=width,
                color=ARM_COLORS[arm],
                alpha=0.86,
                label=arm,
            )
        axis.axhline(threshold, color="black", linestyle="--", linewidth=0.9, label=f"gate {threshold:g}")
        if threshold != 0.0:
            axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set(title=title, xlabel="target", ylabel=ylabel)
        axis.set_xticks(indices)
        axis.grid(axis="y", alpha=0.18)
        axis.legend(fontsize=7)
    axes[0, 1].set_ylim(0, 4.4)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _pixel_contact_sheet(path: Path, run_root: Path, result: Mapping[str, Any]) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.5))
    for index, axis in enumerate(axes.flat):
        image_path = run_root / "direct-pixel-oracle" / f"target-{index:02d}" / "run" / "endpoint_raw.png"
        image = plt.imread(image_path)
        target = result["arms"]["direct-pixel-oracle"]["targets"][index]
        axis.imshow(image)
        axis.set_title(f"t{index} | {'PASS' if target['passed'] else 'FAIL'}")
        axis.axis("off")
    figure.suptitle("R10 direct-pixel endpoint images (raw step 128)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _refresh_delivery(output_dir: Path, result: Mapping[str, Any]) -> None:
    artifacts = []
    for path in sorted(value for value in output_dir.iterdir() if value.is_file()):
        if path.name == "DELIVERY_MANIFEST.json":
            continue
        artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": comparison._sha256(path)})
    comparison._write_json(
        output_dir / "DELIVERY_MANIFEST.json",
        {
            "schema": comparison.DELIVERY_SCHEMA,
            "artifacts": artifacts,
            "source_inventory_sha256": {
                arm: {
                    str(row["target_index"]): row["inventory_sha256"]
                    for row in result["arms"][arm]["targets"]
                }
                for arm in comparison.ARMS
            },
        },
    )


def render(run_root: Path, output_dir: Path) -> dict[str, Any]:
    result = comparison.compare(run_root, output_dir)
    metrics = {
        (arm, index): _optimizer_metrics(run_root / arm / f"target-{index:02d}", arm)
        for arm in comparison.ARMS
        for index in range(8)
    }
    _write_training_csv(output_dir / "training_metrics.csv", result, metrics)
    _training_figure(output_dir / "training_diagnostics.png", result, metrics)
    _endpoint_figure(output_dir / "endpoint_metrics.png", result)
    _pixel_contact_sheet(output_dir / "pixel_endpoint_contact_sheet.png", run_root, result)
    raw_artifacts = {
        "schema": "vision_memory.r10-visual-alignment-source-artifacts.v1",
        "run_root": str(run_root.resolve()),
        "arms": {
            arm: {
                str(row["target_index"]): {
                    "source_root": row["source_root"],
                    "summary_sha256": row["summary_sha256"],
                    "terminal_sha256": row["terminal_sha256"],
                    "inventory_sha256": row["inventory_sha256"],
                }
                for row in result["arms"][arm]["targets"]
            }
            for arm in comparison.ARMS
        },
    }
    comparison._write_json(output_dir / "RAW_ARTIFACTS.json", raw_artifacts)
    analysis = {
        "schema": "vision_memory.r10-visual-alignment-rendered-analysis.v1",
        "status": "completed",
        "formal_success_claim": False,
        "formal_success_reason": result["formal_success_reason"],
        "decision": result["decision"],
        "reason": result["reason"],
        "arm_pass_counts": result["arm_pass_counts"],
        "git_commit": result["git_commit"],
        "selected_segments_sha256": result["selected_segments_sha256"],
        "arms": result["arms"],
    }
    comparison._write_json(output_dir / "ANALYSIS.json", analysis)
    _refresh_delivery(output_dir, result)
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
                "arm_pass_counts": analysis["arm_pass_counts"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
