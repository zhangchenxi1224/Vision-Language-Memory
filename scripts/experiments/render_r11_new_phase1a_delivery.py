"""Render and verify the immutable R11_new Phase 1A delivery bundle.

The figures are derived directly from the independently aggregated comparison
and the archived raw per-step receipts/images.  Trainer/controller summaries
are not used as the scientific source of truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = ROOT / "reports" / "r11-new-phase1a-results-20260905"
EXPECTED_SOURCE_COMMIT = "2cde77ece6f020ab8c747d7c73e19dac4d8fba1b"
EXPECTED_ARCHIVES = {
    "aggregation-v1.tar.gz": {
        "bytes": 43_814,
        "sha256": "64d0cf0dbdfbb9842052a8e1d884a9a6c4da19918cd5f6eb5fe46fbb5112a8ac",
        "entries": 4,
    },
    "targets-00-01-retry.tar.gz": {
        "bytes": 23_366_402,
        "sha256": "67582d05b634f9639b57aebd51f19a7d6ae376b30980e710fbe44d0a3e17a89b",
        "entries": 90,
    },
    "targets-02-03.tar.gz": {
        "bytes": 22_745_949,
        "sha256": "06fc9632ef7858ddd0a02f20e311ebac2bff05b2e9db2c4d5a28630feacf0663",
        "entries": 84,
    },
    "targets-04-05.tar.gz": {
        "bytes": 21_873_812,
        "sha256": "1051dba9d63b837650c0f3979faa30318bfd0867e09e3e3c3d9aaf70017ac4af",
        "entries": 84,
    },
    "targets-06-07.tar.gz": {
        "bytes": 24_568_404,
        "sha256": "238ddf1f137ba2a974fbd7e52c9f4e513c35a3d41ba3104c61a80ba98aa4f44c",
        "entries": 84,
    },
}
TARGET_SOURCES = {
    0: ("targets-00-01-retry.tar.gz", "target-00"),
    1: ("targets-00-01-retry.tar.gz", "target-01-retry01"),
    2: ("targets-02-03.tar.gz", "target-02"),
    3: ("targets-02-03.tar.gz", "target-03"),
    4: ("targets-04-05.tar.gz", "target-04"),
    5: ("targets-04-05.tar.gz", "target-05"),
    6: ("targets-06-07.tar.gz", "target-06"),
    7: ("targets-06-07.tar.gz", "target-07"),
}
CHECKPOINT_STEPS = (0, 64, 128, 192, 256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"Unsafe archive member: {member.name!r}")
    return members


def _read_member(archive: tarfile.TarFile, member: str) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Missing regular archive member: {member}")
    return extracted.read()


def _moving_average(values: np.ndarray, window: int = 16) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[window - 1 :] = np.convolve(values, np.ones(window) / window, mode="valid")
    return result


def _save_figure(path: Path, figure: Any) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render(report_dir: Path) -> None:
    report_dir = report_dir.resolve()
    archive_dir = report_dir / "phase1a-delivery-v1"
    aggregate_dir = report_dir / "phase1a-aggregation-v1"
    comparison_path = aggregate_dir / "comparison.json"
    raw_artifacts_path = aggregate_dir / "RAW_ARTIFACTS.json"
    official_report_path = aggregate_dir / "REPORT.md"

    verified_archives: dict[str, dict[str, Any]] = {}
    handles: dict[str, tarfile.TarFile] = {}
    try:
        for name, expected in EXPECTED_ARCHIVES.items():
            path = archive_dir / name
            actual = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            if actual != {key: expected[key] for key in ("bytes", "sha256")}:
                raise ValueError(f"Archive size/SHA drift: {path}: {actual}")
            archive = tarfile.open(path, "r:gz")
            members = _safe_members(archive)
            if len(members) != expected["entries"]:
                raise ValueError(f"Archive entry-count drift: {path}: {len(members)}")
            handles[name] = archive
            verified_archives[name] = {**actual, "entries": len(members)}

        aggregate_archive = handles["aggregation-v1.tar.gz"]
        for path in (comparison_path, raw_artifacts_path, official_report_path):
            archived = _read_member(aggregate_archive, f"phase1a-aggregation-v1/{path.name}")
            if path.read_bytes() != archived:
                raise ValueError(f"Extracted aggregate artifact differs from immutable archive: {path}")

        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if comparison["source_training_git_commit"] != EXPECTED_SOURCE_COMMIT:
            raise ValueError("Source training commit drifted.")
        if comparison["engineering_gate"] is not True:
            raise ValueError("Expected the independently recomputed engineering gate to pass.")
        if comparison["target_pass_count"] != 6 or comparison["failed_target_indices"] != [1, 7]:
            raise ValueError("Expected immutable Phase 1A diagnostic result 6/8 with failures 1 and 7.")
        if comparison["formal_success"] is not False:
            raise ValueError("Phase 1A must not claim formal scientific success.")

        targets = sorted(comparison["targets"], key=lambda item: item["target_index"])
        if [item["target_index"] for item in targets] != list(range(8)):
            raise ValueError("Expected exactly the eight fixed targets 0..7.")

        metrics_by_target: dict[int, list[dict[str, Any]]] = {}
        training_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        for target in targets:
            index = target["target_index"]
            archive_name, source_name = TARGET_SOURCES[index]
            archive = handles[archive_name]
            payload = _read_member(archive, f"{source_name}/run/metrics.jsonl").decode("utf-8")
            metrics = [json.loads(line) for line in payload.splitlines() if line]
            steps = [row["optimizer_step"] for row in metrics]
            if steps != list(range(1, 257)):
                raise ValueError(f"Target {index} lacks 256 continuous optimizer receipts.")
            if any(row["dreamlite_denoising_steps"] != 4 for row in metrics):
                raise ValueError(f"Target {index} did not execute exactly four DreamLite steps.")
            if any(row["gradient_clipping_applied"] is not False for row in metrics):
                raise ValueError(f"Target {index} unexpectedly applied gradient clipping.")
            metrics_by_target[index] = metrics
            controller_terminal = json.loads(_read_member(archive, f"{source_name}/terminal.json").decode("utf-8"))
            if (
                controller_terminal.get("schema") != "vision_memory.r11-new-phase1a-target-terminal.v1"
                or controller_terminal.get("technical_completed") is not True
            ):
                raise ValueError(f"Target {index} controller terminal is not valid.")

            losses = np.asarray([row["loss_before_step"] for row in metrics], dtype=np.float64)
            gradients = np.asarray([row["gradient_norm"] for row in metrics], dtype=np.float64)
            updates = np.asarray([row["x_T_update_norm"] for row in metrics], dtype=np.float64)
            peak_gradient_offset = int(np.argmax(gradients))
            stats = target["target_statistics"]
            summary_rows.append(
                {
                    "target_index": index,
                    "target_segment_id": target["target_segment_id"],
                    "source_root": target["source_root"],
                    "technical_gate": bool(target["technical_gate"]),
                    "query_reachability_gate": bool(target["target_reachability_gate"]),
                    "m0_normal_mean_ce": stats["m0_normal_mean_ce"],
                    "endpoint_normal_mean_ce": stats["endpoint_normal_mean_ce"],
                    "relative_change": stats["relative_change"],
                    "improved_choice_views": stats["improved_choice_views"],
                    "m0_normal_accuracy": stats["m0_normal_accuracy"],
                    "endpoint_normal_accuracy": stats["endpoint_normal_accuracy"],
                    "accuracy_delta": stats["accuracy_delta"],
                    "normal_reset_did": stats["normal_reset_difference_in_differences"],
                    "trainer_wall_clock_seconds": target["wall_clock_seconds"],
                    "controller_elapsed_seconds": controller_terminal["elapsed_seconds"],
                }
            )
            training_rows.append(
                {
                    "target_index": index,
                    "first_training_ce": float(losses[0]),
                    "minimum_training_ce": float(losses.min()),
                    "minimum_training_ce_step": int(np.argmin(losses)) + 1,
                    "final_training_ce": float(losses[-1]),
                    "median_training_ce": float(np.median(losses)),
                    "peak_gradient_norm": float(gradients[peak_gradient_offset]),
                    "peak_gradient_step": peak_gradient_offset + 1,
                    "median_gradient_norm": float(np.median(gradients)),
                    "maximum_x_T_update_norm": float(updates.max()),
                    "maximum_x_T_update_step": int(np.argmax(updates)) + 1,
                    "median_x_T_update_norm": float(np.median(updates)),
                }
            )

        csv_path = report_dir / "per_target_results.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary_rows)

        diagnostics_path = report_dir / "training_diagnostics.json"
        diagnostics_path.write_bytes(
            (
                json.dumps(
                    {
                        "schema": "vision_memory.r11-new-phase1a-delivery-training-diagnostics.v1",
                        "source": "archived raw metrics.jsonl",
                        "receipt_count": sum(len(rows) for rows in metrics_by_target.values()),
                        "targets": training_rows,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
        )

        pass_color = "#198754"
        fail_color = "#c0392b"
        figure, axes = plt.subplots(2, 4, figsize=(15, 7.8), sharex=True)
        for target, axis in zip(targets, axes.flat, strict=True):
            index = target["target_index"]
            losses = np.asarray([row["loss_before_step"] for row in metrics_by_target[index]], dtype=np.float64)
            steps = np.arange(1, 257)
            color = pass_color if target["target_reachability_gate"] else fail_color
            axis.plot(steps, np.maximum(losses, 1e-6), color=color, alpha=0.28, linewidth=0.75)
            axis.plot(
                steps,
                np.maximum(_moving_average(losses), 1e-6),
                color=color,
                linewidth=1.8,
                label="16-step mean",
            )
            axis.set_yscale("log")
            axis.set_title(
                f"Target {index} — {'PASS' if target['target_reachability_gate'] else 'FAIL'}",
                color=color,
                fontweight="bold",
            )
            axis.grid(True, alpha=0.2)
            axis.set_xlim(1, 256)
            axis.set_xlabel("optimizer step")
            axis.set_ylabel("training-view CE (log)")
        figure.suptitle(
            "R11_new Phase 1A: raw training CE and trailing 16-step mean\n"
            "Four forward-cyclic views repeat every four steps; formal gate uses separate endpoint audit rows.",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        _save_figure(report_dir / "training_loss_trajectories.png", figure)

        figure, axes = plt.subplots(2, 4, figsize=(15, 7.8), sharex=True)
        for target, axis in zip(targets, axes.flat, strict=True):
            index = target["target_index"]
            gradients = np.asarray([row["gradient_norm"] for row in metrics_by_target[index]], dtype=np.float64)
            color = pass_color if target["target_reachability_gate"] else fail_color
            axis.plot(np.arange(1, 257), gradients, color=color, linewidth=0.9)
            axis.set_yscale("log")
            peak = int(np.argmax(gradients))
            axis.scatter([peak + 1], [gradients[peak]], color="#111111", s=12, zorder=3)
            axis.set_title(f"Target {index}; peak={gradients[peak]:.3g} @ {peak + 1}", color=color)
            axis.grid(True, alpha=0.2)
            axis.set_xlim(1, 256)
            axis.set_xlabel("optimizer step")
            axis.set_ylabel("x_T gradient norm (log)")
        figure.suptitle(
            "R11_new Phase 1A: raw x_T gradient norms (no clipping)\n"
            "A spike is an observation, not by itself a causal explanation of endpoint failure.",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        _save_figure(report_dir / "gradient_norm_trajectories.png", figure)

        indices = np.arange(8)
        m0 = np.asarray([row["m0_normal_mean_ce"] for row in summary_rows])
        endpoint = np.asarray([row["endpoint_normal_mean_ce"] for row in summary_rows])
        reductions = -100.0 * np.asarray([row["relative_change"] for row in summary_rows])
        accuracy_delta = np.asarray([row["accuracy_delta"] for row in summary_rows])
        colors = [pass_color if row["query_reachability_gate"] else fail_color for row in summary_rows]
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        width = 0.36
        axes[0].bar(indices - width / 2, m0, width, label="M0", color="#6c757d")
        axes[0].bar(indices + width / 2, endpoint, width, label="raw step 256", color=colors)
        axes[0].set_yscale("log")
        axes[0].set_title("Endpoint audit mean CE")
        axes[0].set_ylabel("reverse-cyclic mean CE (log)")
        axes[0].legend()
        axes[1].bar(indices, reductions, color=colors)
        axes[1].axhline(20.0, color="#111111", linestyle="--", linewidth=1, label="gate ≥20%")
        axes[1].set_title("CE relative reduction")
        axes[1].set_ylabel("reduction from M0 (%)")
        axes[1].legend()
        axes[2].bar(indices, accuracy_delta, color=colors)
        axes[2].axhline(0.25, color="#111111", linestyle="--", linewidth=1, label="gate ≥0.25")
        axes[2].set_title("Accuracy improvement")
        axes[2].set_ylabel("endpoint accuracy − M0 accuracy")
        axes[2].legend()
        for axis in axes:
            axis.set_xticks(indices, [str(index) for index in indices])
            axis.set_xlabel("fixed target index")
            axis.grid(True, axis="y", alpha=0.2)
        figure.suptitle(
            "R11_new Phase 1A endpoint gate decomposition — green PASS, red FAIL",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        _save_figure(report_dir / "endpoint_gate_summary.png", figure)

        figure, axes = plt.subplots(8, 5, figsize=(10.5, 16.5))
        for row_index, target in enumerate(targets):
            index = target["target_index"]
            archive_name, source_name = TARGET_SOURCES[index]
            archive = handles[archive_name]
            for column_index, step in enumerate(CHECKPOINT_STEPS):
                payload = _read_member(archive, f"{source_name}/run/images/step-{step:03d}.png")
                image = Image.open(io.BytesIO(payload)).convert("RGB")
                axis = axes[row_index, column_index]
                axis.imshow(image)
                axis.set_xticks([])
                axis.set_yticks([])
                if row_index == 0:
                    axis.set_title(f"step {step}")
                if column_index == 0:
                    gate = "PASS" if target["target_reachability_gate"] else "FAIL"
                    axis.set_ylabel(f"Target {index}\n{gate}", fontweight="bold")
        figure.suptitle(
            "Frozen-DreamLite decoded checkpoint images\n"
            "Human readability/naturalness was not an optimization objective; these are candidate model-readable codes.",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.5, w_pad=0.2)
        _save_figure(report_dir / "checkpoint_image_montage.png", figure)

        environment_path = report_dir / "render_environment.txt"
        environment_path.write_bytes(
            (
                "\n".join(
                    [
                        f"python_executable={sys.executable}",
                        f"python_version={platform.python_version()}",
                        f"platform={platform.platform()}",
                        f"numpy={np.__version__}",
                        f"matplotlib={matplotlib.__version__}",
                        f"pillow={Image.__version__}",
                        "matplotlib_backend=Agg",
                        "training_source_commit=" + EXPECTED_SOURCE_COMMIT,
                    ]
                )
                + "\n"
            ).encode("utf-8")
        )

        generated_names = [
            "per_target_results.csv",
            "training_diagnostics.json",
            "training_loss_trajectories.png",
            "gradient_norm_trajectories.png",
            "endpoint_gate_summary.png",
            "checkpoint_image_montage.png",
            "render_environment.txt",
        ]
        manifest = {
            "schema": "vision_memory.r11-new-phase1a-delivery-manifest.v1",
            "scientific_source": "independently recomputed raw artifacts",
            "training_source_commit": EXPECTED_SOURCE_COMMIT,
            "official_aggregation": {
                path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in (comparison_path, raw_artifacts_path, official_report_path)
            },
            "archives": verified_archives,
            "valid_target_source_mapping": {
                str(index): {"archive": archive, "root": source} for index, (archive, source) in TARGET_SOURCES.items()
            },
            "invalid_attempt_preserved": {
                "target_index": 1,
                "root": "target-01",
                "classification": "technical failure before model load / optimizer step 0",
                "scientifically_counted": False,
            },
            "aggregate_result": {
                "engineering_gate": comparison["engineering_gate"],
                "query_level_target_pass_count": comparison["target_pass_count"],
                "failed_target_indices": comparison["failed_target_indices"],
                "phase1a_query_level_reachability_gate": comparison["phase1a_query_level_reachability_gate"],
                "formal_success": comparison["formal_success"],
                "decision": comparison["decision"],
            },
            "generated_artifacts": {
                name: {
                    "bytes": (report_dir / name).stat().st_size,
                    "sha256": _sha256(report_dir / name),
                }
                for name in generated_names
            },
            "generator": {
                "path": Path(__file__).relative_to(ROOT).as_posix(),
                "sha256": _sha256(Path(__file__)),
            },
        }
        (report_dir / "DELIVERY_MANIFEST.json").write_bytes(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
    finally:
        for archive in handles.values():
            archive.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    render(args.report_dir)
    print(
        json.dumps(
            {
                "report_dir": str(args.report_dir.resolve()),
                "status": "rendered_and_verified",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
