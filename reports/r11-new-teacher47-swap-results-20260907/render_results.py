"""Render deterministic figures and derived summaries for the teacher-swap delivery."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "round02-valid-d0" / "result.json"
RECEIPTS = ROOT / "round02-valid-d0" / "receipts.jsonl"
CONDITIONS = ("own", "donor", "reset")
COLORS = {"own": "#2E8B57", "donor": "#D97706", "reset": "#64748B"}
CHOICES = {0: "no active preference", 1: "blue", 2: "green", 3: "yellow"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    figures = ROOT / "figures"
    derived = ROOT / "derived"
    figures.mkdir(exist_ok=True)
    derived.mkdir(exist_ok=True)
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in RECEIPTS.read_text(encoding="utf-8").splitlines()]
    targets = {item["target_index"]: item for item in result["comparison"]["targets"]}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    x = np.arange(2)
    width = 0.23
    for offset, condition in enumerate(CONDITIONS):
        ce = [targets[index]["scores"][condition]["mean_ce"] for index in (4, 7)]
        acc = [targets[index]["scores"][condition]["accuracy"] for index in (4, 7)]
        positions = x + (offset - 1) * width
        axes[0].bar(positions, ce, width, label=condition, color=COLORS[condition])
        axes[1].bar(positions, acc, width, label=condition, color=COLORS[condition])
    axes[0].axhline(0.001, color="#991B1B", linestyle="--", linewidth=1, label="own CE gate")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean listwise CE (log scale)")
    axes[0].set_title("Reader confidence follows the image")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("Accuracy across 4 permutations")
    axes[1].set_title("Own succeeds; donor/reset fail")
    for axis in axes:
        axis.set_xticks(x, ("Target 4 / blue", "Target 7 / green"))
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(figures / "teacher_swap_metrics.png", dpi=180, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)

    image_paths = [ROOT / "images" / f"image-{key}.png" for key in ("4", "7", "reset")]
    titles = ("Teacher 4 (blue)", "Teacher 7 (green)", "Reset gray")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.15), constrained_layout=True)
    image_stats = {}
    for axis, path, title in zip(axes, image_paths, titles, strict=True):
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
        tv = (np.abs(np.diff(image, axis=0)).mean() + np.abs(np.diff(image, axis=1)).mean()) / 2
        image_stats[path.name] = {
            "sha256": sha256_file(path),
            "shape": list(image.shape),
            "mean": float(image.mean()),
            "std": float(image.std()),
            "mean_total_variation": float(tv),
        }
    fig.suptitle("Canonical teacher inputs (visual appearance is not the causal test)", fontsize=13)
    fig.savefig(figures / "teacher_image_montage.png", dpi=150, metadata={"Software": "Vision-Language-Memory"})
    plt.close(fig)

    predictions: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        predictions[f"target-{row['target_index']}/{row['condition']}"].append(row["predicted_index"])
    summary = {
        "schema": "vision_memory.r11-new-teacher-swap-derived-summary.v1",
        "comparison": result["comparison"],
        "prediction_indices_by_cell": dict(sorted(predictions.items())),
        "prediction_labels_by_cell": {
            key: [CHOICES[index] for index in values] for key, values in sorted(predictions.items())
        },
        "image_statistics": image_stats,
    }
    (derived / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path.name == "DELIVERY_MANIFEST.json" or ".extract" in relative.parts:
            continue
        artifacts.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema": "vision_memory.r11-new-teacher-swap-delivery-manifest.v1",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (ROOT / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
