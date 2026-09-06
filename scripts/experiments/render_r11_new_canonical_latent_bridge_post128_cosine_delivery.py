"""Verify and render the post-step-128 cosine bridge delivery."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
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
DEFAULT_REPORT_DIR = (
    ROOT
    / "reports"
    / "r11-new-canonical-latent-bridge-post128-cosine-results-20260906"
)
PARENT_ARCHIVE = (
    ROOT
    / "reports"
    / "r11-new-canonical-latent-bridge-results-20260906"
    / "bridge-delivery-v1"
    / "formal-target01.tar.gz"
)
EXPECTED_COMMIT = "16318e005b496a16b7712ad4ff3cea50e2be34fa"
EXPECTED_ARCHIVES = {
    "aggregation-v1.tar.gz": (
        31_125,
        "c10222d09804ea5d3322937160eb6037ce0b6a3bda7771d0f143fabe8be26cd9",
        6,
    ),
    "formal-target01.tar.gz": (
        13_856_599,
        "0f8ff790a06d5da2275525cab6e8335d0b1679fa7f5d789ba1e2783a8117afc2",
        45,
    ),
    "technical-preflight.tar.gz": (
        5_512_436,
        "e30bd6f58b025b8af4dd58a14d012a41bacd8896e789a6f3a2d2e7a4c9fc2c7a",
        30,
    ),
}
PARENT_ARCHIVE_SHA256 = "17d34f9920c8f828041845ff482cbf7e99950ba4dece83b477a1a8387a0f61cc"
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


def load_json_member(archive: tarfile.TarFile, member: str) -> dict[str, Any]:
    value = json.loads(read_member(archive, member).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {member}")
    return value


def load_jsonl_member(archive: tarfile.TarFile, member: str) -> list[dict[str, Any]]:
    payload = read_member(archive, member).decode("utf-8")
    rows = [json.loads(line) for line in payload.splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects: {member}")
    return rows


def learning_rate(update: int) -> float:
    if update <= 128:
        return 0.05
    progress = (update - 128) / 128
    return 0.5 * 0.05 * (1.0 + math.cos(math.pi * progress))


def canonical_prefix(rows: list[dict[str, Any]]) -> bytes:
    normalized = []
    for row in rows[:128]:
        value = dict(row)
        value.pop("elapsed_seconds", None)
        normalized.append(value)
    return "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in normalized
    ).encode("utf-8")


def save_figure(path: Path, figure: Any) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render(report_dir: Path) -> None:
    report_dir = report_dir.resolve()
    archive_dir = report_dir / "delivery-v1"
    aggregate_dir = report_dir / "aggregation-v1"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    handles: dict[str, tarfile.TarFile] = {}
    verified_archives: dict[str, dict[str, Any]] = {}
    parent: tarfile.TarFile | None = None
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
            (aggregate_dir / name).write_bytes(
                read_member(aggregate_archive, f"aggregation-v1/{name}")
            )
        comparison = json.loads(
            (aggregate_dir / "comparison.json").read_text(encoding="utf-8")
        )
        expected_result = {
            "git_commit": EXPECTED_COMMIT,
            "engineering_gate": True,
            "bridge_distance_gate": False,
            "endpoint_reader_transfer_gate": False,
            "bridge_diagnostic_gate": False,
            "formal_success": False,
            "phase2_allowed": False,
            "optimizer_steps": 256,
            "target_index": 1,
        }
        for key, expected in expected_result.items():
            if comparison.get(key) != expected:
                raise ValueError(f"Unexpected aggregate {key}: {comparison.get(key)!r}")
        if comparison.get("secondary_solver_hypothesis_audit") != {
            "beats_parent_endpoint": True,
            "eligible": True,
            "passed": True,
            "post128_non_rebound": True,
        }:
            raise ValueError("Secondary schedule audit drifted.")

        preflight_archive = handles["technical-preflight.tar.gz"]
        preflight = load_json_member(
            preflight_archive,
            "technical-preflight/run/technical_preflight.json",
        )
        audit = preflight["audit"]
        if (
            preflight.get("passed") is not True
            or (
                audit.get("full_forward_calls"),
                audit.get("backward_calls"),
                audit.get("optimizer_steps"),
            )
            != (1, 1, 0)
            or audit.get("only_x_T_fp32_trainable") is not True
            or audit.get("frozen_gradients_absent") is not True
        ):
            raise ValueError("Technical preflight contract failed.")

        formal_archive = handles["formal-target01.tar.gz"]
        summary = load_json_member(
            formal_archive,
            "formal-target01/run/r11_new_bridge_summary.json",
        )
        if summary.get("gates") != {
            "bridge_diagnostic_gate": False,
            "bridge_distance_gate": False,
            "endpoint_reader_transfer_gate": False,
            "formal_success_gate": False,
            "teacher_replay_gate": True,
            "technical_gate": True,
        }:
            raise ValueError("Formal gate set drifted.")
        technical = summary["technical_gate"]
        if (
            technical.get("optimizer_lr_schedule_exact") is not True
            or technical.get("pre_intervention_step128_parity_valid") is not True
            or technical.get("finite_nonzero_gradient_every_step") is not True
        ):
            raise ValueError("Formal optimizer or prefix contract failed.")

        metrics = load_jsonl_member(
            formal_archive,
            "formal-target01/run/metrics.jsonl",
        )
        if [row["optimizer_step"] for row in metrics] != list(range(1, 257)):
            raise ValueError("Optimizer receipts are not exactly 1..256.")
        for row in metrics:
            update = int(row["optimizer_step"])
            if (
                not math.isclose(
                    float(row["learning_rate"]),
                    learning_rate(update),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or row["dreamlite_denoising_steps"] != 4
                or row["gradient_clipping_applied"] is not False
                or row["reader_gradient_calls"] != 0
                or float(row["gradient_norm"]) <= 0.0
            ):
                raise ValueError(f"Optimizer receipt drifted at update {update}.")
        if metrics[-1]["learning_rate"] != 0.0 or metrics[-1]["x_T_update_norm"] != 0.0:
            raise ValueError("Update 256 zero-LR receipt drifted.")

        if sha256(PARENT_ARCHIVE) != PARENT_ARCHIVE_SHA256:
            raise ValueError("Parent formal archive hash drifted.")
        parent = tarfile.open(PARENT_ARCHIVE, "r:gz")
        safe_members(parent)
        parent_metrics = load_jsonl_member(
            parent,
            "formal-target01/run/metrics.jsonl",
        )
        parent_summary = load_json_member(
            parent,
            "formal-target01/run/r11_new_bridge_summary.json",
        )
        new_prefix = canonical_prefix(metrics)
        parent_prefix = canonical_prefix(parent_metrics)
        if new_prefix != parent_prefix:
            raise ValueError("Updates 1..128 differ from the parent after removing elapsed time.")
        prefix_parity = {
            "schema": "vision_memory.r11-new-bridge-post128-prefix-parity.v1",
            "rows": 128,
            "excluded_field": "elapsed_seconds",
            "canonical_bytes": len(new_prefix),
            "new_sha256": hashlib.sha256(new_prefix).hexdigest(),
            "parent_sha256": hashlib.sha256(parent_prefix).hexdigest(),
            "exact": True,
        }
        (report_dir / "prefix_parity.json").write_text(
            json.dumps(prefix_parity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        new_x = np.asarray([row["optimizer_step"] - 1 for row in metrics])
        new_ratio = np.asarray([row["mse_ratio_to_m0"] for row in metrics])
        parent_x = np.asarray([row["optimizer_step"] - 1 for row in parent_metrics])
        parent_ratio = np.asarray([row["mse_ratio_to_m0"] for row in parent_metrics])
        new_endpoint = comparison["endpoint_distance_statistics"]["mse_ratio_to_m0"]
        parent_endpoint = parent_summary["endpoint_distance_statistics"]["mse_ratio_to_m0"]
        figure, axis = plt.subplots(figsize=(9.5, 5.2))
        axis.plot(parent_x, parent_ratio, color="#9ca3af", label="parent: constant lr=0.05")
        axis.plot(new_x, new_ratio, color="#2563eb", label="new: cosine decay after 128")
        axis.scatter([256], [parent_endpoint], color="#6b7280", marker="D", s=35)
        axis.scatter([256], [new_endpoint], color="#1d4ed8", marker="D", s=35)
        axis.axvline(128, color="#d97706", linestyle="--", label="first changed update: 129")
        axis.axhline(0.01, color="#c0392b", linestyle=":", label="primary MSE-ratio gate")
        axis.set_yscale("log")
        axis.set_xlabel("completed optimizer updates")
        axis.set_ylabel("MSE / initial MSE")
        axis.set_title("Target 1 bridge distance: matched prefix, changed suffix")
        axis.grid(True, alpha=0.2)
        axis.legend()
        save_figure(report_dir / "distance_trajectory_comparison.png", figure)

        update = np.asarray([row["optimizer_step"] for row in metrics])
        lr = np.asarray([row["learning_rate"] for row in metrics])
        gradient = np.asarray([row["gradient_norm"] for row in metrics])
        update_norm = np.asarray([row["x_T_update_norm"] for row in metrics])
        figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
        axes[0].plot(update, lr, color="#2563eb")
        axes[0].set_title("optimizer learning rate")
        axes[1].plot(update, gradient, color="#7c3aed")
        axes[1].set_yscale("log")
        axes[1].set_title("x_T gradient norm")
        axes[2].plot(update, np.maximum(update_norm, 1e-12), color="#d97706")
        axes[2].set_yscale("log")
        axes[2].set_title("x_T update norm (zero shown at 1e-12)")
        for axis in axes:
            axis.axvline(128, color="#6b7280", linestyle="--", linewidth=1)
            axis.grid(True, alpha=0.2)
        axes[0].set_xlabel("optimizer update")
        axes[1].set_xlabel("update index (gradient measured before update)")
        axes[2].set_xlabel("update index (parameter change from this update)")
        figure.suptitle("Post-step-128 cosine intervention diagnostics", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.9))
        save_figure(report_dir / "optimizer_schedule_diagnostics.png", figure)

        reader_labels = ["M0", "parent 256", "cosine 256", "teacher"]
        reader_stats = (
            comparison["m0_reader_statistics"],
            parent_summary["endpoint_reader_statistics"],
            comparison["endpoint_reader_statistics"],
            comparison["teacher_replay_statistics"],
        )
        figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        colors = ["#6b7280", "#9ca3af", "#2563eb", "#198754"]
        axes[0].bar(reader_labels, [row["mean_ce"] for row in reader_stats], color=colors)
        axes[0].set_yscale("log")
        axes[0].set_title("Frozen Reader mean CE")
        axes[1].bar(reader_labels, [row["accuracy"] for row in reader_stats], color=colors)
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Accuracy across 4 fixed views")
        for axis in axes:
            axis.tick_params(axis="x", rotation=18)
            axis.grid(True, axis="y", alpha=0.2)
        figure.suptitle("Reader transfer remains absent", fontsize=13)
        figure.tight_layout(rect=(0, 0, 1, 0.9))
        save_figure(report_dir / "reader_transfer_summary.png", figure)

        image_members = ["formal-target01/run/teacher/canonical_r11_target01.png"] + [
            f"formal-target01/run/images/step-{step:03d}.png"
            for step in CHECKPOINT_STEPS
        ]
        image_labels = ["canonical teacher"] + [
            f"step {step}" for step in CHECKPOINT_STEPS
        ]
        figure, axes = plt.subplots(1, len(image_members), figsize=(18, 3.2))
        for axis, member, label in zip(axes, image_members, image_labels, strict=True):
            image = Image.open(io.BytesIO(read_member(formal_archive, member))).convert("RGB")
            axis.imshow(image)
            axis.set_title(label)
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            "Canonical model-readable latent and DreamLite bridge checkpoints\n"
            "Human readability is not an optimization objective or gate.",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.82), w_pad=0.15)
        save_figure(report_dir / "checkpoint_image_montage.png", figure)

        best_offset = int(np.argmin(new_ratio))
        diagnostics = {
            "schema": "vision_memory.r11-new-bridge-post128-cosine-delivery-diagnostics.v1",
            "source_commit": EXPECTED_COMMIT,
            "receipt_count": len(metrics),
            "prefix_parity": prefix_parity,
            "parent_endpoint_mse_ratio": parent_endpoint,
            "step128_mse_ratio": comparison["checkpoint_distance_statistics"]["128"][
                "mse_ratio_to_m0"
            ],
            "new_endpoint_mse_ratio": new_endpoint,
            "best_raw_preupdate_receipt": int(metrics[best_offset]["optimizer_step"]),
            "best_raw_preupdate_completed_updates": int(new_x[best_offset]),
            "best_raw_preupdate_mse_ratio": float(new_ratio[best_offset]),
            "endpoint_distance_statistics": comparison["endpoint_distance_statistics"],
            "reader": {
                "m0": comparison["m0_reader_statistics"],
                "parent_endpoint": parent_summary["endpoint_reader_statistics"],
                "new_endpoint": comparison["endpoint_reader_statistics"],
                "teacher": comparison["teacher_replay_statistics"],
            },
            "primary_decision": comparison["decision"],
            "secondary_solver_hypothesis_audit": comparison[
                "secondary_solver_hypothesis_audit"
            ],
            "secondary_solver_hypothesis_decision": comparison[
                "secondary_solver_hypothesis_decision"
            ],
            "formal_success": False,
            "phase2_allowed": False,
        }
        (report_dir / "training_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (report_dir / "render_environment.txt").write_text(
            "\n".join(
                (
                    f"python_executable={sys.executable}",
                    f"python_version={platform.python_version()}",
                    f"platform={platform.platform()}",
                    f"numpy={np.__version__}",
                    f"matplotlib={matplotlib.__version__}",
                    f"pillow={Image.__version__}",
                    "matplotlib_backend=Agg",
                    f"training_source_commit={EXPECTED_COMMIT}",
                )
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (report_dir / "ARCHIVE_SHA256SUMS.txt").write_text(
            "".join(
                f"{verified_archives[name]['sha256']}  delivery-v1/{name}\n"
                for name in sorted(verified_archives)
            ),
            encoding="utf-8",
            newline="\n",
        )

        generated = (
            "distance_trajectory_comparison.png",
            "optimizer_schedule_diagnostics.png",
            "reader_transfer_summary.png",
            "checkpoint_image_montage.png",
            "prefix_parity.json",
            "training_diagnostics.json",
            "render_environment.txt",
            "ARCHIVE_SHA256SUMS.txt",
        )
        readme = report_dir / "README.md"
        generator = Path(__file__).resolve()
        manifest = {
            "schema": "vision_memory.r11-new-bridge-post128-cosine-delivery-manifest.v1",
            "scientific_source": "independently recomputed raw artifacts",
            "training_source_commit": EXPECTED_COMMIT,
            "aggregate_result": {
                key: comparison[key]
                for key in (
                    "engineering_gate",
                    "teacher_replay_gate",
                    "bridge_distance_gate",
                    "endpoint_reader_transfer_gate",
                    "bridge_diagnostic_gate",
                    "formal_success",
                    "phase2_allowed",
                    "decision",
                    "secondary_solver_hypothesis_decision",
                )
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
                "bytes": readme.stat().st_size,
                "sha256": sha256(readme),
            },
            "generated_artifacts": {
                name: {
                    "bytes": (report_dir / name).stat().st_size,
                    "sha256": sha256(report_dir / name),
                }
                for name in generated
            },
            "generator": {
                "path": generator.relative_to(ROOT).as_posix(),
                "sha256": sha256(generator),
            },
        }
        (report_dir / "DELIVERY_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    finally:
        for archive in handles.values():
            archive.close()
        if parent is not None:
            parent.close()


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
