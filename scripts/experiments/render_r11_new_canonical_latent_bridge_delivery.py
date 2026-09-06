"""Verify and render the R11_new canonical-latent bridge delivery."""

from __future__ import annotations

import argparse
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
DEFAULT_REPORT_DIR = ROOT / "reports" / "r11-new-canonical-latent-bridge-results-20260906"
EXPECTED_COMMIT = "f83519b2863c13747425157f4cd56e55684c12e9"
EXPECTED_ARCHIVES = {
    "aggregation-v1.tar.gz": (
        30_616,
        "660dbf3ffdbe4d85892d40ebf877ce973ce7e554f5d85e683c93548d96e32d27",
        6,
    ),
    "formal-target01.tar.gz": (
        13_751_400,
        "17d34f9920c8f828041845ff482cbf7e99950ba4dece83b477a1a8387a0f61cc",
        45,
    ),
    "technical-preflight.tar.gz": (
        5_512_321,
        "33df13dea63030866dc7e658c6815baa0922603bd8bfc764a9680c1e25981760",
        30,
    ),
}
AGGREGATE_FILES = (
    "RAW_ARTIFACTS.json",
    "REPORT.md",
    "artifact_inventory.json",
    "comparison.json",
    "distance_trajectory.csv",
)
CHECKPOINT_STEPS = (0, 64, 128, 192, 256)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"Unsafe archive member: {member.name!r}")
    return members


def read_member(archive: tarfile.TarFile, member: str) -> bytes:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Missing archive member: {member}")
    return extracted.read()


def save_figure(path: Path, figure: Any) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render(report_dir: Path) -> None:
    report_dir = report_dir.resolve()
    archive_dir = report_dir / "bridge-delivery-v1"
    aggregate_dir = report_dir / "aggregation-v1"
    handles: dict[str, tarfile.TarFile] = {}
    verified_archives: dict[str, dict[str, Any]] = {}
    try:
        for name, (expected_bytes, expected_sha, expected_entries) in EXPECTED_ARCHIVES.items():
            path = archive_dir / name
            actual_sha = sha256(path)
            if path.stat().st_size != expected_bytes or actual_sha != expected_sha:
                raise ValueError(f"Archive size/SHA drift: {path}")
            archive = tarfile.open(path, "r:gz")
            members = safe_members(archive)
            if len(members) != expected_entries:
                raise ValueError(f"Archive member-count drift: {path}")
            handles[name] = archive
            verified_archives[name] = {
                "bytes": expected_bytes,
                "sha256": actual_sha,
                "entries": len(members),
            }

        aggregate_archive = handles["aggregation-v1.tar.gz"]
        for name in AGGREGATE_FILES:
            if (aggregate_dir / name).read_bytes() != read_member(
                aggregate_archive, f"aggregation-v1/{name}"
            ):
                raise ValueError(f"Extracted aggregation drift: {name}")

        comparison = json.loads((aggregate_dir / "comparison.json").read_text(encoding="utf-8"))
        expected_result = {
            "git_commit": EXPECTED_COMMIT,
            "engineering_gate": True,
            "bridge_diagnostic_gate": False,
            "formal_success": False,
            "phase2_allowed": False,
            "optimizer_steps": 256,
            "target_index": 1,
        }
        for key, expected in expected_result.items():
            if comparison[key] != expected:
                raise ValueError(f"Unexpected aggregate {key}: {comparison[key]!r}")

        preflight_archive = handles["technical-preflight.tar.gz"]
        preflight = json.loads(
            read_member(
                preflight_archive,
                "technical-preflight/run/technical_preflight.json",
            ).decode("utf-8")
        )
        audit = preflight["audit"]
        if not preflight["passed"] or (
            audit["full_forward_calls"],
            audit["backward_calls"],
            audit["optimizer_steps"],
        ) != (1, 1, 0):
            raise ValueError("Technical preflight contract failed.")
        if preflight["teacher_replay_statistics"]["accuracy"] != 1.0:
            raise ValueError("Canonical teacher replay failed.")

        formal_archive = handles["formal-target01.tar.gz"]
        summary = json.loads(
            read_member(
                formal_archive,
                "formal-target01/run/r11_new_bridge_summary.json",
            ).decode("utf-8")
        )
        gates = summary["gates"]
        if gates != {
            "bridge_diagnostic_gate": False,
            "bridge_distance_gate": False,
            "endpoint_reader_transfer_gate": False,
            "formal_success_gate": False,
            "teacher_replay_gate": True,
            "technical_gate": True,
        }:
            raise ValueError("Formal gate set drifted.")

        payload = read_member(formal_archive, "formal-target01/run/metrics.jsonl").decode("utf-8")
        metrics = [json.loads(line) for line in payload.splitlines() if line]
        if [row["optimizer_step"] for row in metrics] != list(range(1, 257)):
            raise ValueError("Optimizer receipts are not exactly 1..256.")
        if any(row["dreamlite_denoising_steps"] != 4 for row in metrics):
            raise ValueError("DreamLite step count drifted.")
        if any(row["gradient_clipping_applied"] for row in metrics):
            raise ValueError("Gradient clipping unexpectedly occurred.")
        if any(row["reader_gradient_calls"] != 0 for row in metrics):
            raise ValueError("Reader gradients unexpectedly occurred.")
        if any(row["learning_rate"] != 0.05 for row in metrics):
            raise ValueError("Learning-rate contract drifted.")

        steps = np.asarray([row["optimizer_step"] for row in metrics])
        mse_ratio = np.asarray([row["mse_ratio_to_m0"] for row in metrics])
        l2_ratio = np.asarray([row["l2_distance_ratio_to_m0"] for row in metrics])
        normalized_rmse = np.asarray([row["teacher_normalized_rmse"] for row in metrics])
        gradient_norm = np.asarray([row["gradient_norm"] for row in metrics])
        update_norm = np.asarray([row["x_T_update_norm"] for row in metrics])
        x_t_rms = np.asarray([row["x_T_before_step"]["rms"] for row in metrics])
        best_offset = int(np.argmin(mse_ratio))
        completed_updates = steps - 1
        endpoint = comparison["endpoint_distance_statistics"]

        figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharex=True)
        panels = (
            (mse_ratio, endpoint["mse_ratio_to_m0"], 0.01, "MSE / initial MSE"),
            (l2_ratio, endpoint["l2_distance_ratio_to_m0"], 0.1, "L2 / initial L2"),
            (
                normalized_rmse,
                endpoint["teacher_normalized_rmse"],
                0.1,
                "RMSE / teacher std",
            ),
        )
        for axis, (values, endpoint_value, threshold, title) in zip(axes, panels, strict=True):
            x_values = np.append(completed_updates, 256)
            y_values = np.append(values, endpoint_value)
            axis.plot(x_values, y_values, color="#2563eb", linewidth=1.4)
            axis.axhline(
                threshold,
                color="#c0392b",
                linestyle="--",
                linewidth=1.2,
                label=f"gate <= {threshold:g}",
            )
            axis.scatter(
                [completed_updates[best_offset]],
                [values[best_offset]],
                color="#111827",
                s=24,
                label="best observed pre-update",
            )
            axis.scatter(
                [256],
                [endpoint_value],
                color="#c0392b",
                marker="D",
                s=28,
                label="formal raw step 256",
            )
            axis.set_yscale("log")
            axis.set_title(title)
            axis.set_xlabel("completed optimizer updates")
            axis.grid(True, alpha=0.2)
            axis.legend()
        figure.suptitle(
            "R11_new Target 1 canonical-latent bridge: distance trajectory\n"
            "Best intermediate point is diagnostic only; the gate uses raw step 256.",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.88))
        save_figure(report_dir / "distance_trajectory.png", figure)

        figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharex=True)
        for axis, values, title, color, log_scale in (
            (axes[0], gradient_norm, "x_T gradient norm", "#7c3aed", True),
            (axes[1], update_norm, "x_T update norm", "#d97706", True),
            (axes[2], x_t_rms, "x_T RMS", "#059669", False),
        ):
            axis.plot(completed_updates, values, color=color)
            if log_scale:
                axis.set_yscale("log")
            axis.set_title(title)
            axis.set_xlabel("completed updates before each receipt")
            axis.grid(True, alpha=0.2)
        figure.suptitle(
            "Optimizer diagnostics: finite gradients, no clipping, constant lr=0.05",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.9))
        save_figure(report_dir / "optimizer_diagnostics.png", figure)

        labels = ["M0", "step 256", "teacher"]
        ce = [
            comparison["m0_reader_statistics"]["mean_ce"],
            comparison["endpoint_reader_statistics"]["mean_ce"],
            comparison["teacher_replay_statistics"]["mean_ce"],
        ]
        accuracy = [
            comparison["m0_reader_statistics"]["accuracy"],
            comparison["endpoint_reader_statistics"]["accuracy"],
            comparison["teacher_replay_statistics"]["accuracy"],
        ]
        colors = ["#6b7280", "#c0392b", "#198754"]
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
        axes[0].bar(labels, ce, color=colors)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("mean CE (log)")
        axes[0].set_title("Frozen Reader loss")
        axes[1].bar(labels, accuracy, color=colors)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("accuracy across 4 fixed views")
        axes[1].set_title("Frozen Reader accuracy")
        for axis in axes:
            axis.grid(True, axis="y", alpha=0.2)
        figure.suptitle("Reader transfer fails at the learned endpoint", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.9))
        save_figure(report_dir / "reader_transfer_summary.png", figure)

        image_members = ["formal-target01/run/teacher/canonical_r11_target01.png"] + [
            f"formal-target01/run/images/step-{step:03d}.png" for step in CHECKPOINT_STEPS
        ]
        image_labels = ["canonical teacher"] + [f"step {step}" for step in CHECKPOINT_STEPS]
        figure, axes = plt.subplots(1, len(image_members), figsize=(18, 3.2))
        for axis, member, label in zip(axes, image_members, image_labels, strict=True):
            image = Image.open(io.BytesIO(read_member(formal_archive, member))).convert("RGB")
            axis.imshow(image)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            "Machine-readable teacher versus frozen-DreamLite bridge checkpoints\n"
            "Human readability is not an optimization target or gate.",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.82), w_pad=0.15)
        save_figure(report_dir / "checkpoint_image_montage.png", figure)

        diagnostics = {
            "schema": "vision_memory.r11-new-canonical-latent-bridge-delivery-diagnostics.v1",
            "source_commit": EXPECTED_COMMIT,
            "receipt_count": len(metrics),
            "best_raw_preupdate_mse_ratio": float(mse_ratio[best_offset]),
            "best_raw_preupdate_receipt": int(steps[best_offset]),
            "best_raw_preupdate_completed_updates": int(completed_updates[best_offset]),
            "endpoint_distance_statistics": comparison["endpoint_distance_statistics"],
            "checkpoint_distance_statistics": comparison["checkpoint_distance_statistics"],
            "reader": {
                "m0": comparison["m0_reader_statistics"],
                "endpoint": comparison["endpoint_reader_statistics"],
                "teacher": comparison["teacher_replay_statistics"],
            },
            "gates": gates,
            "decision": comparison["decision"],
        }
        (report_dir / "training_diagnostics.json").write_bytes(
            (json.dumps(diagnostics, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )

        environment_path = report_dir / "render_environment.txt"
        environment_path.write_text(
            "\n".join(
                [
                    f"python_executable={sys.executable}",
                    f"python_version={platform.python_version()}",
                    f"platform={platform.platform()}",
                    f"numpy={np.__version__}",
                    f"matplotlib={matplotlib.__version__}",
                    f"pillow={Image.__version__}",
                    "matplotlib_backend=Agg",
                    "training_source_commit=" + EXPECTED_COMMIT,
                ]
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        sums = "".join(
            f"{verified_archives[name]['sha256']}  bridge-delivery-v1/{name}\n"
            for name in sorted(verified_archives)
        )
        (report_dir / "ARCHIVE_SHA256SUMS.txt").write_text(
            sums, encoding="utf-8", newline="\n"
        )
        generated = (
            "distance_trajectory.png",
            "optimizer_diagnostics.png",
            "reader_transfer_summary.png",
            "checkpoint_image_montage.png",
            "training_diagnostics.json",
            "render_environment.txt",
            "ARCHIVE_SHA256SUMS.txt",
        )
        readme_path = report_dir / "README.md"
        generator_path = Path(__file__).resolve()
        manifest = {
            "schema": "vision_memory.r11-new-canonical-latent-bridge-delivery-manifest.v1",
            "scientific_source": "independently recomputed raw artifacts",
            "training_source_commit": EXPECTED_COMMIT,
            "aggregate_result": {
                "engineering_gate": True,
                "teacher_replay_gate": True,
                "bridge_distance_gate": False,
                "endpoint_reader_transfer_gate": False,
                "bridge_diagnostic_gate": False,
                "formal_success": False,
                "phase2_allowed": False,
                "decision": comparison["decision"],
            },
            "archives": verified_archives,
            "official_aggregation": {
                name: {
                    "bytes": (aggregate_dir / name).stat().st_size,
                    "sha256": sha256(aggregate_dir / name),
                }
                for name in AGGREGATE_FILES
            },
            "report_document": {
                "bytes": readme_path.stat().st_size,
                "sha256": sha256(readme_path),
            },
            "generated_artifacts": {
                name: {
                    "bytes": (report_dir / name).stat().st_size,
                    "sha256": sha256(report_dir / name),
                }
                for name in generated
            },
            "generator": {
                "path": generator_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(generator_path),
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
            {"report_dir": str(args.report_dir.resolve()), "status": "rendered_and_verified"},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
