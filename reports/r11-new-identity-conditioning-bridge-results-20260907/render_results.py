"""Render deterministic, claim-bounded figures for the identity-conditioning result."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence"
AGGREGATION = ROOT / "aggregation"
IMAGES = ROOT / "images"
FIGURES = ROOT / "figures"
CHECKPOINT_STEPS = (0, 64, 128, 192, 256)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_metrics() -> list[dict]:
    rows = [
        json.loads(line)
        for line in (EVIDENCE / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 256 or [row["optimizer_step"] for row in rows] != list(range(1, 257)):
        raise ValueError("Expected exactly 256 contiguous optimizer receipts.")
    return rows


def image_statistics(path: Path, teacher: np.ndarray) -> dict[str, float | str]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    blurred = np.asarray(
        Image.fromarray(np.uint8(np.rint(image * 255.0))).filter(ImageFilter.GaussianBlur(radius=2)),
        dtype=np.float64,
    ) / 255.0
    tv = (
        np.abs(np.diff(image, axis=0)).mean()
        + np.abs(np.diff(image, axis=1)).mean()
    ) / 2.0
    return {
        "image": path.name,
        "mean": float(image.mean()),
        "std": float(image.std()),
        "total_variation": float(tv),
        "high_frequency_rms": float(np.sqrt(np.mean((image - blurred) ** 2))),
        "pixel_mse_to_teacher_png": float(np.mean((image - teacher) ** 2)),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics()
    comparison = load_json(AGGREGATION / "comparison.json")
    config = load_json(EVIDENCE / "config.json")
    terminal = load_json(EVIDENCE / "formal-controller-terminal.json")

    endpoint = comparison["endpoint_distance_statistics"]
    reader = comparison["endpoint_reader_statistics"]
    parent = config["parent_bridge"]
    teacher_init = config["teacher_matched_reference"]
    if terminal["technical_gate"] is not True or comparison["engineering_gate"] is not True:
        raise ValueError("Technical evidence is not valid.")
    if comparison["formal_success"] is not False or comparison["phase2_allowed"] is not False:
        raise ValueError("Scientific interpretation boundary drifted.")

    steps = np.asarray([row["optimizer_step"] for row in metrics])
    mse = np.asarray([row["mse"] for row in metrics])
    mse_ratio = np.asarray([row["mse_ratio_to_m0"] for row in metrics])
    l2_ratio = np.asarray([row["l2_distance_ratio_to_m0"] for row in metrics])
    teacher_nrmse = np.asarray([row["teacher_normalized_rmse"] for row in metrics])
    gradient = np.asarray([row["gradient_norm"] for row in metrics])
    update = np.asarray([row["x_T_update_norm"] for row in metrics])
    learning_rate = np.asarray([row["learning_rate"] for row in metrics])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.semilogy(steps, mse, color="#2166ac", linewidth=2, label="identity condition")
    ax.axhline(endpoint["m0_mse"] * 0.01, color="#b2182b", linestyle="--", label="primary MSE gate")
    ax.axhline(parent["endpoint_mse"], color="#ef8a62", linestyle=":", label="source+event parent endpoint")
    for step in CHECKPOINT_STEPS[1:]:
        ax.axvline(step, color="0.8", linewidth=0.7)
    ax.set(title="A. Dense teacher-MSE trajectory", xlabel="Optimizer step", ylabel="Mean squared error (log scale)")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.semilogy(steps, mse_ratio, label="MSE / M0", linewidth=2)
    ax.semilogy(steps, l2_ratio, label="L2 / M0", linewidth=2)
    ax.semilogy(steps, teacher_nrmse, label="teacher-normalized RMSE", linewidth=2)
    ax.axhline(0.01, color="#b2182b", linestyle="--", linewidth=1, label="MSE-ratio gate")
    ax.axhline(0.10, color="#762a83", linestyle=":", linewidth=1, label="L2/nRMSE gate")
    ax.set(title="B. Pre-registered distance gates", xlabel="Optimizer step", ylabel="Normalized distance (log scale)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.semilogy(steps, gradient, color="#1b7837", linewidth=1.8, label="gradient norm")
    positive_update = np.where(update > 0, update, np.nan)
    ax.semilogy(steps, positive_update, color="#5aae61", linewidth=1.2, alpha=0.8, label="xT update norm")
    ax.set(title="C. Optimization remained active", xlabel="Optimizer step", ylabel="Norm (log scale)")
    ax_lr = ax.twinx()
    ax_lr.plot(steps, learning_rate, color="#f46d43", linewidth=1.4, label="learning rate")
    ax_lr.set_ylabel("Learning rate")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_lr.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")

    ax = axes[1, 1]
    labels = ["source+event\nparent", "source+identity\n(this run)", "teacher-init\nreference"]
    mse_relative = np.asarray(
        [1.0, endpoint["mse"] / parent["endpoint_mse"], teacher_init["endpoint_mse"] / parent["endpoint_mse"]]
    )
    ce_relative = np.asarray(
        [1.0, reader["mean_ce"] / parent["endpoint_reader_mean_ce"], teacher_init["endpoint_reader_mean_ce"] / parent["endpoint_reader_mean_ce"]]
    )
    x = np.arange(len(labels))
    width = 0.34
    bars1 = ax.bar(x - width / 2, mse_relative, width, label="endpoint MSE / parent", color="#67a9cf")
    bars2 = ax.bar(x + width / 2, ce_relative, width, label="Reader CE / parent", color="#ef8a62")
    ax.axhline(1.0, color="0.25", linewidth=0.8)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Ratio to direct source+event parent (lower is better)")
    ax.set_title("D. Identity prompt changes little; teacher-init changes more")
    ax.legend(fontsize=8)
    for bars in (bars1, bars2):
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)

    fig.suptitle("R11_new identity-conditioning bridge — diagnostic result, not Picture Memory success", fontsize=14)
    fig.savefig(FIGURES / "optimization_diagnostics.png", dpi=180)
    plt.close(fig)

    teacher_path = IMAGES / "canonical_r11_target01.png"
    teacher_array = np.asarray(Image.open(teacher_path).convert("RGB"), dtype=np.float64) / 255.0
    ordered = [
        ("Canonical R11 teacher", teacher_path),
        *((f"DreamLite step {step}", IMAGES / f"step-{step:03d}.png") for step in CHECKPOINT_STEPS),
    ]
    stats = [image_statistics(path, teacher_array) for _, path in ordered]
    with (EVIDENCE / "image_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stats[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(stats)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    for ax, ((label, path), row) in zip(axes.flat, zip(ordered, stats), strict=True):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.set_title(
            f"{label}\nstd={row['std']:.4f}, TV={row['total_variation']:.4f}, MSE→teacher={row['pixel_mse_to_teacher_png']:.4f}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("The canonical code is textured; frozen DreamLite endpoints remain near-uniform", fontsize=14)
    fig.savefig(FIGURES / "checkpoint_montage.png", dpi=160)
    plt.close(fig)

    summary = {
        "schema": "vision_memory.r11-new-identity-conditioning-rendered-summary.v1",
        "git_commit": comparison["git_commit"],
        "technical_gate": True,
        "teacher_replay_gate": comparison["teacher_replay_gate"],
        "bridge_distance_gate": comparison["bridge_distance_gate"],
        "endpoint_reader_transfer_gate": comparison["endpoint_reader_transfer_gate"],
        "secondary_conditioning_audit": comparison["secondary_solver_hypothesis_audit"],
        "formal_success": False,
        "phase2_allowed": False,
        "endpoint": endpoint,
        "reader": reader,
        "parent": {
            "endpoint_mse": parent["endpoint_mse"],
            "reader_mean_ce": parent["endpoint_reader_mean_ce"],
            "reader_accuracy": parent["endpoint_reader_accuracy"],
        },
        "absolute_improvement_vs_parent": {
            "endpoint_mse": parent["endpoint_mse"] - endpoint["mse"],
            "endpoint_reader_mean_ce": parent["endpoint_reader_mean_ce"] - reader["mean_ce"],
        },
        "relative_improvement_vs_parent": {
            "endpoint_mse": 1.0 - endpoint["mse"] / parent["endpoint_mse"],
            "endpoint_reader_mean_ce": 1.0 - reader["mean_ce"] / parent["endpoint_reader_mean_ce"],
        },
        "image_statistics": stats,
        "interpretation": "diagnostic_only_no_picture_memory_success_claim",
    }
    write_json(EVIDENCE / "rendered_summary.json", summary)

    manifest_path = ROOT / "DELIVERY_MANIFEST.json"
    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        artifacts.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    delivery = {
        "schema": "vision_memory.r11-new-identity-conditioning-local-delivery.v1",
        "implementation_git_commit": comparison["git_commit"],
        "artifact_count_excluding_manifest": len(artifacts),
        "total_bytes_excluding_manifest": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
        "raw_archive_sha256": "7e1734c707682f1d4b93bf15978c02f3f9f7dec0ea5e62dc909752a91816130f",
        "formal_success": False,
        "phase2_allowed": False,
    }
    write_json(manifest_path, delivery)


if __name__ == "__main__":
    main()
