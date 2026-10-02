"""Exploratory endpoint previews and coordinate histograms, without model inference."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot(source: Path, destination: Path):
    summary = json.loads((source / "summary.json").read_text())
    if summary["run_count"] != 18 or not summary["technical_passed"]:
        raise ValueError("A complete 18-run pilot is required.")
    starts = ["blank"] + [f"noise-seed-{seed:02d}" for seed in range(8)]
    arms = ("mcq", "open")
    payloads = {(arm, start): torch.load(source / "runs" / f"{start}-{arm}" / "endpoint_raw.pt",
                map_location="cpu", weights_only=True) for arm in arms for start in starts}
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 9, figsize=(16.2, 4.6), constrained_layout=True)
    for i, arm in enumerate(arms):
        for j, start in enumerate(starts):
            pixels = payloads[(arm, start)]["image"].float().squeeze(0).permute(1, 2, 0).numpy()
            axes[i, j].imshow(pixels, vmin=0, vmax=1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if j == 0:
                axes[i, j].set_ylabel(arm.upper())
            if i == 0:
                axes[i, j].set_title("blank" if j == 0 else f"seed {j - 1}")
    fig.suptitle("Decoded step-256 endpoint images | Display previews only; evaluation used original BF16 tensors")
    fig.savefig(destination / "endpoint_images.png", dpi=180)
    plt.close(fig)
    arrays = {(arm, start): payloads[(arm, start)]["latent_fp32"].numpy().reshape(4, -1)
              for arm in arms for start in starts[1:]}
    reference = torch.load(source / "initials" / "blank.pt", map_location="cpu",
                           weights_only=True)["latent_fp32"].numpy().reshape(4, -1)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    colors = {"mcq": "#3465a4", "open": "#bd4b24"}
    for channel, ax in enumerate(axes.flat):
        values = [value[channel] for value in arrays.values()] + [reference[channel]]
        edges = np.linspace(min(float(value.min()) for value in values),
                            max(float(value.max()) for value in values), 101)
        centers = (edges[:-1] + edges[1:]) / 2
        for arm in arms:
            densities = []
            for start in starts[1:]:
                value = arrays[(arm, start)][channel]
                density = np.histogram(value, bins=edges, density=True)[0]
                densities.append(density)
                ax.plot(centers, density, color=colors[arm], alpha=0.15, linewidth=0.7)
                rows.append({"arm": arm, "init_id": start, "channel": channel,
                             "coordinate_mean": float(value.mean(dtype=np.float64)),
                             "coordinate_std": float(value.std(dtype=np.float64)),
                             "coordinate_min": float(value.min()), "coordinate_max": float(value.max())})
            ax.plot(centers, np.mean(densities, axis=0), color=colors[arm], linewidth=2, label=arm.upper())
        ax.plot(centers, np.histogram(reference[channel], bins=edges, density=True)[0],
                color="#555555", linestyle="--", linewidth=1, label="blank reference")
        ax.set_title(f"Latent channel {channel}")
        ax.set_xlabel("Coordinate value")
        ax.set_ylabel("Histogram density")
        ax.legend(frameon=False)
        ax.grid(alpha=0.15)
    fig.suptitle("Exploratory coordinate histograms of endpoint tensors\n"
                 "Thin: eight seeds; bold: mean histogram. Coordinates are correlated; this is not a distribution-equality test.",
                 fontsize=11)
    fig.savefig(destination / "coordinate_histograms.png", dpi=180)
    fig.savefig(destination / "coordinate_histograms.pdf")
    plt.close(fig)
    with (destination / "coordinate_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    plot(args.source, args.destination)
