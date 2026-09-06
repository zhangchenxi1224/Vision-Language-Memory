"""Create descriptive tables and scientific figures from a complete R11 pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source: Path, destination: Path):
    summary = read_json(source / "summary.json")
    terminal = read_json(source / "terminal.json")
    if not (summary["complete_records"] and summary["technical_passed"]
            and terminal["status"] == "completed" and summary["run_count"] == 18
            and summary["optimizer_steps"] == 4608):
        raise ValueError("Only complete, technically valid 18-run pilots can be analyzed.")
    geometry = read_json(source / "geometry_summary.json")
    generations = read_rows(source / "generations.jsonl")
    if len(generations) != 108:
        raise ValueError("Expected 108 endpoint generations.")
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "answers.csv", [{key: row[key] for key in
        ("run_id", "init_id", "arm", "condition", "prompt_id", "raw", "truncated")}
        | {"strict_correct": row["scorer"]["strict_correct"]} for row in generations])
    curves, dispersion, pair_ratios, ending = {}, [], [], {}
    for arm in ("mcq", "open"):
        curves[arm] = read_rows(source / f"geometry_{arm}.jsonl")
        if [row["optimizer_step"] for row in curves[arm]] != list(range(257)):
            raise ValueError("Missing or duplicated geometry steps.")
        initial = np.asarray(curves[arm][0]["raw_latent"]["rmse_matrix"])
        for row in curves[arm]:
            raw = row["raw_latent"]
            dispersion.append({"arm": arm, "step": row["optimizer_step"],
                "C_over_C0": row["C_over_C0"], "mean_pairwise_rmse": raw["mean_pairwise_rmse"],
                "raw_cosine": raw["mean_pairwise_cosine"],
                "own_delta_cosine": row["delta_own_start"]["mean_pairwise_cosine"],
                "reference_delta_cosine": row["delta_shared_reference"]["mean_pairwise_cosine"]})
            current = np.asarray(raw["rmse_matrix"])
            for i in range(8):
                for j in range(i + 1, 8):
                    pair_ratios.append({"arm": arm, "step": row["optimizer_step"],
                        "seed_i": i, "seed_j": j, "rmse": current[i, j],
                        "rmse_over_initial": current[i, j] / initial[i, j]})
        steps_last, gradients_last, four_last = [], [], []
        for seed in range(8):
            metrics = read_rows(source / "runs" / f"noise-seed-{seed:02d}-{arm}" / "metrics.jsonl")
            if [row["optimizer_step"] for row in metrics] != list(range(1, 257)):
                raise ValueError("Incomplete optimizer trajectory.")
            steps_last.append(metrics[-1]["actual_update_rms"])
            gradients_last.append(metrics[-1]["gradient_rms"])
            four_last.append(metrics[-1]["four_step_update_rms"])
        last = curves[arm][-1]
        ending[arm] = {"C_over_C0": last["C_over_C0"],
            "endpoint_mean_pairwise_rmse": last["raw_latent"]["mean_pairwise_rmse"],
            "endpoint_raw_cosine": last["raw_latent"]["mean_pairwise_cosine"],
            "endpoint_own_delta_cosine": last["delta_own_start"]["mean_pairwise_cosine"],
            "last_update_rms_min_max": [min(steps_last), max(steps_last)],
            "last_gradient_rms_min_max": [min(gradients_last), max(gradients_last)],
            "last_four_update_rms_min_max": [min(four_last), max(four_last)]}
    write_csv(destination / "dispersion.csv", dispersion)
    write_csv(destination / "pair_distance_ratios.csv", pair_ratios)
    probe_table = []
    for arm in ("mcq", "open"):
        for seed in range(8):
            run = f"noise-seed-{seed:02d}-{arm}"
            probes = read_rows(source / "runs" / run / "fixed_probes.jsonl")
            if len(probes) != 25:
                raise ValueError("Expected 25 fixed probe rows per run.")
            for step in (0, 64, 128, 192, 256):
                rows = [row for row in probes if row["optimizer_step"] == step]
                probe_table.append({"arm": arm, "seed": seed, "step": step,
                    "open_ce": next(row["ce"] for row in rows if row["probe"] == "open_gold_ce"),
                    "mcq_ce_mean": np.mean([row["ce"] for row in rows if row["probe"] == "mcq_forward"])})
    write_csv(destination / "fixed_probes.csv", probe_table)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), constrained_layout=True)
    colors = {"mcq": "#3465a4", "open": "#bd4b24"}
    for arm in ("mcq", "open"):
        ax = axes[0, 0]
        ax.plot(range(257), [row["C_over_C0"] for row in curves[arm]], color=colors[arm], label=arm.upper())
        ax = axes[0, 1]
        ax.plot(range(257), [row["raw_latent"]["mean_pairwise_rmse"] for row in curves[arm]],
                color=colors[arm], label=arm.upper())
        for ax, metric in ((axes[1, 0], "open_ce"), (axes[1, 1], "mcq_ce_mean")):
            for seed in range(8):
                rows = [row for row in probe_table if row["arm"] == arm and row["seed"] == seed]
                ax.plot([row["step"] for row in rows], [max(row[metric], 1e-9) for row in rows],
                        color=colors[arm], alpha=0.23, linewidth=0.9)
            means = [np.mean([row[metric] for row in probe_table if row["arm"] == arm and row["step"] == step])
                     for step in (0, 64, 128, 192, 256)]
            ax.plot((0, 64, 128, 192, 256), np.maximum(means, 1e-9), color=colors[arm],
                    marker="o", linewidth=2, label=arm.upper())
            ax.set_yscale("log")
    axes[0, 0].axhline(1, color="#888888", linestyle="--", linewidth=0.8)
    axes[0, 0].set_title("A  Dispersion relative to initial starts")
    axes[0, 0].set_ylabel("Mean squared pairwise RMSE / initial value")
    axes[0, 1].set_title("B  Absolute distances between starts")
    axes[0, 1].set_ylabel("Mean pairwise latent RMSE")
    axes[1, 0].set_title("C  Common open-answer CE probe")
    axes[1, 1].set_title("D  Common MCQ CE probe")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)
    fig.suptitle("Old R11: one music-preference target, eight paired random starts\n"
                 "Fixed 256 updates; blank starts excluded; thin lines = individual seeds", fontsize=12)
    fig.savefig(destination / "paired_multistart.png", dpi=180)
    fig.savefig(destination / "paired_multistart.pdf")
    plt.close(fig)
    result = {"behavior": summary["by_arm"], "geometry": ending,
        "cross_arm_endpoints": geometry["paired_cross_arm_endpoint"],
        "blank_separate": summary["blank_separate"],
        "scope": "One target, eight random starts; distances are descriptive, not a test of equal distributions."}
    (destination / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    analyze(args.source, args.destination)
