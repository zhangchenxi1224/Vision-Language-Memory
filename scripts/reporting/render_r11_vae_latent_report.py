"""Render integrity-checked tables, plots, and image sheets for completed R11."""

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

from scripts.experiments import compare_r11_vae_latent as comparison  # noqa: E402


CHECKPOINT_STEPS = (0, 64, 128, 192, 256)
TARGET_COLORS = plt.get_cmap("tab10")


def _moving_mean(values: Sequence[float], window: int = 16) -> list[float]:
    return [
        statistics.fmean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def _metrics(root: Path, segment_id: str) -> list[dict[str, Any]]:
    rows, _diagnostic = comparison._validate_metrics(root, segment_id)
    return rows


def _write_training_csv(
    path: Path,
    result: Mapping[str, Any],
    metrics: Mapping[int, Sequence[Mapping[str, Any]]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "target_index",
                "segment_id",
                "optimizer_step",
                "training_view",
                "learning_rate",
                "loss",
                "loss_moving_mean_16",
                "gradient_norm",
                "gradient_nonzero_fraction",
                "latent_delta_norm",
                "latent_rms",
                "image_min",
                "image_max",
                "image_rms",
                "image_saturation_fraction",
            )
        )
        for target in result["targets"]:
            index = int(target["target_index"])
            rows = metrics[index]
            smooth = _moving_mean([float(row["loss_before_step"]) for row in rows])
            for row, moving in zip(rows, smooth, strict=True):
                writer.writerow(
                    (
                        index,
                        target["target_segment_id"],
                        row["optimizer_step"],
                        row["forward_cyclic_training_view"],
                        row["learning_rate"],
                        row["loss_before_step"],
                        moving,
                        row["gradient_norm"],
                        row["gradient_nonzero_fraction"],
                        row["latent_delta_norm_after_step"],
                        row["latent_rms_after_step"],
                        row["image_min_after_step"],
                        row["image_max_after_step"],
                        row["image_rms_after_step"],
                        row["image_saturation_fraction_after_step"],
                    )
                )


def _training_figure(
    path: Path,
    result: Mapping[str, Any],
    metrics: Mapping[int, Sequence[Mapping[str, Any]]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    panels = (
        ("loss", "Training listwise CE (16-step mean)", "listwise CE", True),
        ("gradient", "Gradient norm", "latent gradient L2", True),
        ("latent_delta", "Latent displacement from blank", "latent delta L2", False),
        ("saturation", "Decoded image saturation", "pixel fraction at 0 or 1", False),
    )
    for target in result["targets"]:
        index = int(target["target_index"])
        rows = metrics[index]
        steps = [int(row["optimizer_step"]) for row in rows]
        values = {
            "loss": _moving_mean([float(row["loss_before_step"]) for row in rows]),
            "gradient": [float(row["gradient_norm"]) for row in rows],
            "latent_delta": [float(row["latent_delta_norm_after_step"]) for row in rows],
            "saturation": [float(row["image_saturation_fraction_after_step"]) for row in rows],
        }
        for axis, (key, _title, _ylabel, _log) in zip(axes.flat, panels, strict=True):
            axis.plot(
                steps,
                values[key],
                color=TARGET_COLORS(index),
                linewidth=1.0,
                alpha=0.82,
                label=f"t{index}",
            )
    for axis, (_key, title, ylabel, log_scale) in zip(axes.flat, panels, strict=True):
        if log_scale:
            axis.set_yscale("log")
        axis.set(title=title, xlabel="optimizer step", ylabel=ylabel)
        axis.grid(alpha=0.18)
        axis.legend(fontsize=7, ncol=4)
    figure.suptitle("R11 direct VAE-latent optimization diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _endpoint_figure(path: Path, result: Mapping[str, Any]) -> None:
    indices = list(range(8))
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    fields = (
        ("relative_change", "Normal CE relative change", "endpoint / M0 - 1", -0.20),
        ("improved_choice_views", "Fixed reverse-cyclic views improved", "views", 4.0),
        ("accuracy_delta", "Normal accuracy change", "endpoint - M0", 0.25),
        (
            "normal_reset_difference_in_differences",
            "Normal/reset difference-in-differences",
            "CE DiD",
            0.0,
        ),
    )
    for axis, (field, title, ylabel, threshold) in zip(axes.flat, fields, strict=True):
        values = [float(target["target_statistics"][field]) for target in result["targets"]]
        colors = ["#16a34a" if target["passed"] else "#dc2626" for target in result["targets"]]
        axis.bar(indices, values, color=colors, alpha=0.88)
        axis.axhline(threshold, color="black", linestyle="--", linewidth=0.9, label=f"gate {threshold:g}")
        if threshold != 0.0:
            axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set(title=title, xlabel="target", ylabel=ylabel)
        axis.set_xticks(indices)
        axis.grid(axis="y", alpha=0.18)
        axis.legend(fontsize=7)
    axes[0, 1].set_ylim(0, 4.4)
    figure.suptitle("R11 fixed endpoint and causal-control metrics")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _endpoint_contact_sheet(path: Path, run_root: Path, result: Mapping[str, Any]) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.5))
    for index, axis in enumerate(axes.flat):
        image_path = run_root / f"target-{index:02d}" / "run" / "endpoint_raw.png"
        axis.imshow(plt.imread(image_path))
        target = result["targets"][index]
        stat = target["target_statistics"]
        axis.set_title(
            f"t{index} | {'PASS' if target['passed'] else 'FAIL'}\n"
            f"CE {float(stat['m0_normal_mean_ce']):.2f}→{float(stat['endpoint_normal_mean_ce']):.4f}"
        )
        axis.axis("off")
    figure.suptitle("R11 frozen-VAE decoded endpoint images (raw latent step 256)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _trajectory_contact_sheet(path: Path, run_root: Path) -> None:
    figure, axes = plt.subplots(4, 10, figsize=(20.0, 10.0))
    for target_index in range(8):
        row = target_index // 2
        base_column = (target_index % 2) * len(CHECKPOINT_STEPS)
        for offset, step in enumerate(CHECKPOINT_STEPS):
            axis = axes[row, base_column + offset]
            image_path = (
                run_root
                / f"target-{target_index:02d}"
                / "run"
                / "images"
                / f"step-{step:03d}.png"
            )
            axis.imshow(plt.imread(image_path))
            axis.set_title(f"t{target_index} s{step}", fontsize=8)
            axis.axis("off")
    figure.suptitle("R11 decoded-image trajectories: blank VAE latent to learned visual code")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _refresh_delivery(output_dir: Path, result: Mapping[str, Any]) -> None:
    comparison._refresh_delivery(output_dir, result)


def render(run_root: Path, output_dir: Path) -> dict[str, Any]:
    result = comparison.compare(run_root, output_dir)
    metrics = {
        index: _metrics(
            run_root / f"target-{index:02d}",
            result["targets"][index]["target_segment_id"],
        )
        for index in range(8)
    }
    for rows in metrics.values():
        for row in rows:
            if any(
                not math.isfinite(float(row[key]))
                for key in (
                    "loss_before_step",
                    "gradient_norm",
                    "latent_delta_norm_after_step",
                    "image_saturation_fraction_after_step",
                )
            ):
                raise ValueError("R11 renderer encountered a non-finite training diagnostic.")
    _write_training_csv(output_dir / "training_metrics.csv", result, metrics)
    _training_figure(output_dir / "training_diagnostics.png", result, metrics)
    _endpoint_figure(output_dir / "endpoint_metrics.png", result)
    _endpoint_contact_sheet(output_dir / "latent_endpoint_contact_sheet.png", run_root, result)
    _trajectory_contact_sheet(output_dir / "latent_trajectory_contact_sheet.png", run_root)
    raw_artifacts = {
        "schema": "vision_memory.r11-vae-latent-source-artifacts.v1",
        "run_root": str(run_root.resolve()),
        "targets": {
            str(target["target_index"]): {
                "source_root": target["source_root"],
                "launch_sha256": target["launch_sha256"],
                "summary_sha256": target["summary_sha256"],
                "terminal_sha256": target["terminal_sha256"],
                "inventory_sha256": target["inventory_sha256"],
            }
            for target in result["targets"]
        },
    }
    comparison._write_json(output_dir / "RAW_ARTIFACTS.json", raw_artifacts)
    analysis = {
        "schema": "vision_memory.r11-vae-latent-rendered-analysis.v1",
        "status": "completed",
        "formal_success_claim": False,
        "formal_success_reason": result["formal_success_reason"],
        "decision": result["decision"],
        "reason": result["reason"],
        "target_pass_count": result["target_pass_count"],
        "source_training_git_commit": result["source_training_git_commit"],
        "aggregation_git_commit": result["aggregation_git_commit"],
        "selected_segments_sha256": result["selected_segments_sha256"],
        "targets": result["targets"],
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
                "target_pass_count": analysis["target_pass_count"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
