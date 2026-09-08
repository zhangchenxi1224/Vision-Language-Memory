"""Render audited geometry artifacts with Matplotlib; missing results stay pending.

No model calls, synthetic results, reconstructed tensors, or fitted success labels.
B uses one common PCA basis per space across all completed anchor starts and the
11 preregistered snapshots. PCA distances are projections, not full-space distances.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNAPSHOTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256)
SPACES = ("xT", "z", "delta_xT", "delta_z")
LABELS = {"xT": "Control xT", "z": "Endpoint z", "delta_xT": "Control change", "delta_z": "Endpoint change"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default


def wilson(successes: int, count: int) -> tuple[float, float]:
    if not count or not 0 <= successes <= count:
        raise ValueError("Wilson interval requires observed trials")
    z = 1.959963984540054
    p, denominator = successes / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    width = z * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return max(0., center - width), min(1., center + width)


def pending(ax, reason: str = "No completed data available") -> None:
    ax.set_axis_off()
    ax.text(.5, .56, "PENDING", transform=ax.transAxes, ha="center", va="center", fontsize=23, color="#697587")
    ax.text(.5, .41, reason, transform=ax.transAxes, ha="center", va="center", fontsize=10, wrap=True)


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.2)


def success_figure(root: Path):
    rows = read_json(root / "analysis/success_rates.json", [])
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(rows) * .38 + 1.4)))
    ax.set_title("A. Reachability by initialization (95% Wilson interval)", loc="left")
    if not rows:
        pending(ax)
        return fig, {"status": "pending"}
    labels = []
    for index, row in enumerate(rows):
        count, successes, planned = row["completed"], row["successes"], row["planned"]
        labels.append(f"{row['study']} | {row['distribution']} | scale {row['scale']:g}")
        if count:
            rate = successes / count
            lower, upper = wilson(successes, count)
            ax.errorbar(rate, index, xerr=[[rate - lower], [upper - rate]], fmt="o", color="#2667aa", capsize=3)
            text = f"{successes}/{count} passed; {planned - count} unresolved"
        else:
            text = f"PENDING; 0/{planned} completed"
        ax.text(1.04, index, text, va="center", fontsize=9)
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(-.03, 1.01)
    ax.set_xticks(np.linspace(0, 1, 6), [f"{int(value * 100)}%" for value in np.linspace(0, 1, 6)])
    ax.set_xlabel("Success among completed runs; missing/technical failures are separate")
    style(ax)
    fig.subplots_adjust(left=.32, right=.69, bottom=.17, top=.90)
    return fig, {"status": "observed", "groups": len(rows), "interval": "Wilson 95%; completed denominator"}


def anchor_summaries(root: Path) -> list[dict]:
    rows = []
    for path in sorted((root / "runs").glob("A2-*/summary.json")):
        row = read_json(path)
        if row.get("technical_pass") is True and row.get("optimizer_steps") == 256:
            rows.append(row)
    return rows


def common_pca(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """PCA via sample Gram matrix; no [samples,samples,coordinates] allocation."""
    centered = np.asarray(matrix, np.float64)
    centered = centered - centered.mean(0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = np.maximum(eigenvalues[order], 0), eigenvectors[:, order]
    coordinates = eigenvectors[:, :2] * np.sqrt(eigenvalues[:2])[None]
    # Coordinate units correspond to RMSE-scaled full-space PCA scores.
    coordinates /= np.sqrt(centered.shape[1])
    fraction = float(eigenvalues[:2].sum() / eigenvalues.sum()) if eigenvalues.sum() else 0.
    return coordinates, fraction


def trajectory_figure(root: Path):
    summaries = anchor_summaries(root)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("B. Anchor optimization trajectories in a common PCA plane", x=.04, ha="left")
    metadata: dict[str, Any] = {"status": "pending", "snapshots": list(SNAPSHOTS), "completed_anchor_runs": len(summaries)}
    if not summaries:
        for ax in axes:
            pending(ax, "Complete anchor trajectories are required")
        return fig, metadata
    indices = [read_json(Path(row["trajectory_index_path"])) for row in summaries]
    for axis, space in zip(axes, ("xT", "z")):
        matrix = []
        for index in indices:
            by_step = {row["step"]: row for row in index}
            for step in SNAPSHOTS:
                record = by_step[step][space]
                path = Path(record["path"])
                if file_hash(path) != record["file_sha256"]:
                    raise ValueError(f"Trajectory artifact failed SHA verification: {path}")
                value = np.load(path, allow_pickle=False)
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite trajectory: {path}")
                matrix.append(value.reshape(-1))
        matrix = np.stack(matrix)
        coordinates, fraction = common_pca(matrix)
        paths = coordinates.reshape(len(summaries), len(SNAPSHOTS), 2)
        for row, path in zip(summaries, paths):
            color = plt.get_cmap("turbo")(row["seed"] / 31)
            axis.plot(path[:, 0], path[:, 1], color=color, alpha=.65, linewidth=1.2)
            axis.scatter(*path[0], marker="o", s=17, facecolors="none", edgecolors=[color])
            axis.scatter(*path[-1], marker="^", s=25, color=color)
        axis.set_title(f"{LABELS[space]} | 2D explained variance: {fraction:.1%}")
        axis.set_xlabel("PC1 (score / sqrt(full dimensions))")
        axis.set_ylabel("PC2 (score / sqrt(full dimensions))")
        style(axis)
        metadata[space] = {"explained_variance_2d": fraction, "full_dimensions": int(matrix.shape[1]),
                           "common_basis_fit_samples": int(len(matrix)), "distance_unit": "PCA score / sqrt(D)"}
    mapper = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, 31), cmap="turbo")
    color_axis = fig.add_axes((.36, .17, .29, .025))
    fig.colorbar(mapper, cax=color_axis, orientation="horizontal", label="Initialization seed")
    fig.text(.06, .06, "Circle: start; triangle: step 256. Lines join 11 saved states. Projection crossings/closeness do not prove basin identity.", fontsize=9)
    fig.text(.06, .025, "All completed anchor starts are included, including QA failures. The two spaces use separate common PCA bases.", fontsize=9)
    fig.subplots_adjust(left=.08, right=.97, top=.86, bottom=.32, wspace=.29)
    metadata["status"] = "observed"
    return fig, metadata


def distance_figure(root: Path):
    data = read_json(root / "analysis/geometry_statistics.json", {})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("C. Within-task and between-task distances in the full latent space", x=.04, ha="left")
    observed = False
    for ax, population in zip(axes, ("all_completed", "successful_only")):
        values = data.get(population, {})
        if not values:
            pending(ax, population.replace("_", " ") + ": no sufficient data")
            continue
        observed = True
        positions = np.arange(len(SPACES))
        for offset, key, label, color in ((-.18, "within_rmse", "Within same task", "#2667aa"),
                                           (.18, "between_rmse", "Between tasks", "#e59c38")):
            errors = [values.get(space, {}).get(key) for space in SPACES]
            indices = [index for index, value in enumerate(errors) if value is not None]
            ax.bar(positions[indices] + offset, [errors[i] for i in indices], .35, label=label, color=color)
        ax.set_xticks(positions, [LABELS[space] for space in SPACES], rotation=20, ha="right")
        ax.set_title(population.replace("_", " "))
        ax.set_ylabel("RMSE over all coordinates")
        ax.legend(fontsize=9)
        style(ax)
    fig.text(.05, .025, "Distances are computed before projection. Shared initialization seeds can affect between-task distances; inspect the paired-seed control.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .92))
    return fig, {"status": "observed" if observed else "pending", "distance_space": "full dimensional"}


def rank_figure(root: Path):
    data = read_json(root / "analysis/geometry_statistics.json", {}).get("all_completed", {})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("D. Effective rank compared with the sample-rank limit and random null", loc="left")
    if not data:
        pending(ax)
        return fig, {"status": "pending"}
    positions = np.arange(len(SPACES))
    for offset, key, color in ((-.24, "r90", "#9dbde0"), (0, "r95", "#2667aa"), (.24, "r99", "#123c68")):
        pairs = [(i, data[space]["pca"][key]) for i, space in enumerate(SPACES) if space in data]
        ax.bar([i + offset for i, _ in pairs], [v for _, v in pairs], .23, label=key, color=color)
    for key, marker, label, color in (("random_null_r95", "x", "Random null r95", "#cc6e19"),
                                      ("sample_rank_limit", "_", "Maximum sample rank", "#a32222")):
        ax.scatter([i for i, space in enumerate(SPACES) if space in data],
                   [data[space]["pca"][key] for space in SPACES if space in data],
                   marker=marker, s=100, label=label, color=color, zorder=4)
    ax.set_xticks(positions, [LABELS[space] for space in SPACES])
    ax.set_ylabel("Components needed for stated explained variance")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    style(ax)
    fig.text(.06, .025, "A centered sample of N endpoints has rank at most N-1. Low rank relative to 65,536 alone is not evidence of shared structure.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    return fig, {"status": "observed", "population": "all_completed", "null": "matching sample/dimension Gaussian control"}


def functional_cases(root: Path) -> list[dict]:
    payload = read_json(root / "analysis/functional_geometry.json", {})
    if not payload or payload.get("technical_pass") is not True:
        return []
    return payload.get("cases", [])


def interpolation_figure(root: Path):
    rows = [row for row in functional_cases(root) if row.get("case", {}).get("kind") == "interpolation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("E. Reader loss along the sampled straight-line interpolation", x=.04, ha="left")
    for ax, space in zip(axes, ("xT", "z")):
        selected = [row for row in rows if row["case"]["space"] == space]
        if not selected:
            pending(ax, "Actual interpolation Reader evaluations are required")
            continue
        pairs = defaultdict(list)
        for row in selected:
            case = row["case"]
            normal = [evaluation["ce"] for evaluation in row["evaluation_rows"] if evaluation["condition"] == "normal"]
            if len(normal) != 4 or not np.isfinite(normal).all():
                raise ValueError("Interpolation point must contain four finite Reader CE values")
            pairs[(case["left_run_id"], case["right_run_id"])].append((case["lambda"], float(np.mean(normal)), row["qa_pass"]))
        for points in pairs.values():
            points.sort()
            ax.plot([x for x, _, _ in points], [y for _, y, _ in points], color="#2667aa", alpha=.3, linewidth=1)
            failed = [(x, y) for x, y, passed in points if not passed]
            if failed:
                ax.scatter(*zip(*failed), color="#bf3838", marker="x", s=18)
        ax.set_title("xT: full DreamLite -> VAE -> Reader" if space == "xT" else "z: VAE -> Reader diagnostic")
        ax.set_xlabel("Interpolation coefficient lambda")
        ax.set_ylabel("Mean CE over four untrained choice views")
        style(ax)
    fig.text(.06, .025, "Red crosses fail the all-four-views QA gate. A barrier on this finite straight line does not prove disconnected basins.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .92))
    return fig, {"status": "observed" if rows else "pending", "evaluated_points": len(rows)}


def robustness_figure(root: Path):
    rows = [row for row in functional_cases(root) if row.get("case", {}).get("kind") == "perturbation"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("F. Direct functional robustness near one successful control xT", loc="left")
    if not rows:
        pending(ax, "Actual perturbed xT -> DreamLite -> VAE -> Reader evaluations are required")
        return fig, {"status": "pending"}
    grouped = defaultdict(list)
    expected = defaultdict(int)
    for case in read_json(root / "evaluation_inputs/cases.json", {}).get("cases", []):
        if case.get("kind") == "perturbation":
            expected[case["rho"]] += 1
    for row in rows:
        if not row.get("full_dreamlite_path_executed"):
            raise ValueError("xT perturbation robustness must use the complete DreamLite path")
        grouped[row["case"]["rho"]].append(row)
    for rho, points in sorted(grouped.items()):
        count, successes = len(points), sum(row["qa_pass"] is True for row in points)
        rate = successes / count
        lower, upper = wilson(successes, count)
        ax.errorbar(rho, rate, yerr=[[rate - lower], [upper - rate]], fmt="o", color="#2667aa", capsize=4)
        label = f"{successes}/{count} pass"
        if rho in expected:
            label += f"\n{expected[rho] - count} unresolved"
        ax.annotate(label, (rho, rate), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks(sorted(grouped), [f"{rho:g}" for rho in sorted(grouped)])
    ax.set_ylim(-.04, 1.28)
    ax.set_yticks(np.linspace(0, 1, 6), [f"{int(v * 100)}%" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Additive perturbation radius rho (coordinate RMS)")
    ax.set_ylabel("QA success with 95% Wilson interval")
    style(ax)
    fig.text(.06, .025, "Direct evaluation, without reoptimization. One selected anchor and sampled directions do not estimate global basin volume.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, 1))
    return fig, {"status": "observed", "evaluated_directions": len(rows), "reoptimization_included": False}


def optimization_figure(root: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Supplement. Raw training CE and xT gradient norm", x=.04, ha="left")
    count = 0
    for path in sorted((root / "runs").glob("A2-*/metrics.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        count += 1
        axes[0].plot([row["step"] for row in rows], [row["ce"] for row in rows], alpha=.35, linewidth=.8)
        axes[1].plot([row["step"] for row in rows], [row["gradient"]["norm"] for row in rows], alpha=.35, linewidth=.8)
    for ax, label in zip(axes, ("Training-view CE", "Gradient L2 norm (full xT space)")):
        if count:
            ax.set_xlabel("Optimization step (before update)")
            ax.set_ylabel(label)
            style(ax)
        else:
            pending(ax, "Raw optimization logs are not available")
    fig.text(.06, .025, "Training views cycle across steps; training CE is not held-out functional success. Partial logs are explicitly permitted here.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .92))
    return fig, {"status": "observed" if count else "pending", "raw_logs": count, "partial_traces_allowed": True}


def render(root: Path, output: Path, dpi: int = 160) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "semibold", "savefig.facecolor": "white"})
    jobs = [("A_success_rates", success_figure), ("B_common_pca_trajectories", trajectory_figure),
            ("C_within_between", distance_figure), ("D_effective_rank", rank_figure),
            ("E_interpolation_barriers", interpolation_figure), ("F_local_robustness", robustness_figure),
            ("G_optimization_traces", optimization_figure)]
    report = {"schema_version": 1, "input_root": str(root.resolve()), "plots": {}, "errors": [],
              "scientific_success_claim": False, "rendering_does_not_replace_analysis_audit": True}
    for name, function in jobs:
        try:
            figure, metadata = function(root)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            plt.close("all")
            figure, axis = plt.subplots(figsize=(9, 4))
            pending(axis, f"ARTIFACT ERROR: {exc}")
            metadata = {"status": "artifact_error", "error": str(exc)}
            report["errors"].append({"plot": name, "error": str(exc)})
        path = output / f"{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        report["plots"][name] = {**metadata, "path": str(path.resolve()), "file_sha256": file_hash(path)}
    report["analysis_input_hashes"] = {str(path.relative_to(root)): file_hash(path)
                                        for path in sorted((root / "analysis").glob("*.json"))}
    (output / "render_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Campaign artifact root")
    parser.add_argument("--output", type=Path, help="Default: ROOT/analysis/figures")
    parser.add_argument("--dpi", type=int, default=160)
    arguments = parser.parse_args()
    result = render(arguments.root, arguments.output or arguments.root / "analysis/figures", arguments.dpi)
    print(json.dumps({"plots": {key: value["status"] for key, value in result["plots"].items()}, "errors": result["errors"]}))
    raise SystemExit(1 if result["errors"] else 0)
