"""Render and integrity-check a completed paired R7 gradient-balance diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ARMS = ("raw-mean-control", "unit-balanced-norm-matched")
COLORS = {"raw-mean-control": "#d97706", "unit-balanced-norm-matched": "#2563eb"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"Expected JSON objects in {path}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _moving_mean(values: Sequence[float], window: int = 16) -> list[float]:
    return [statistics.fmean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _validate_arm(root: Path, expected_arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    terminal = _load_json(root / "terminal.json")
    if terminal.get("status") != "completed_diagnostic" or terminal.get("passed") is not True:
        raise ValueError(f"R7 arm did not complete cleanly: {root}")
    summary = _load_json(root / "run" / "r7_summary.json")
    if summary.get("schema") != "vision_memory.r7-gradient-balance-summary.v1":
        raise ValueError(f"R7 summary schema mismatch: {root}")
    if summary.get("status") != "completed" or summary.get("arm") != expected_arm:
        raise ValueError(f"R7 summary arm/status mismatch: {root}")
    if summary.get("full_success_claim_allowed") is not False:
        raise ValueError(f"R7 diagnostic improperly permits a formal success claim: {root}")

    inventory = _load_json(root / "artifact_inventory.json")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"R7 artifact inventory is empty: {root}")
    for record in records:
        path = root / str(record["path"])
        if not path.is_file():
            raise ValueError(f"Inventory artifact is missing: {path}")
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise ValueError(f"Inventory artifact failed size/SHA validation: {path}")
    metrics = [
        record for record in _load_jsonl(root / "run" / "metrics.jsonl") if record.get("kind") == "optimizer_step"
    ]
    return summary, metrics


def _validate_pair(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    for key in ("git_commit", "implementation_revision", "selected_segments_sha256", "edit_start_sigma"):
        values = {summary.get(key) for summary in summaries.values()}
        if len(values) != 1 or None in values:
            raise ValueError(f"R7 paired-arm drift in {key}: {values}")
    modes = {summary.get("gradient_aggregation") for summary in summaries.values()}
    if modes != {"raw-mean", "unit-balanced-norm-matched"}:
        raise ValueError(f"R7 aggregation modes are invalid: {modes}")


def _training_csv(output: Path, metrics: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    rows: list[list[Any]] = []
    for arm in ARMS:
        for record in metrics[arm]:
            diagnostics = record.get("optimizer_diagnostics", {})
            update = diagnostics.get("updates_after_step", {}).get("global", {})
            family = record.get("loss_by_family", {})
            aggregation = record["gradient_aggregation"]
            rows.append(
                [
                    arm,
                    record["optimizer_step"],
                    record["learning_rate"],
                    record["loss_mean"],
                    family.get("F2"),
                    family.get("F3"),
                    family.get("F5"),
                    family.get("F6"),
                    record["gradient_norm_before_clip"],
                    record["gradient_clipped"],
                    update.get("update_weight_ratio"),
                    aggregation["raw_vs_applied_cosine"],
                    aggregation["raw_vs_unit_balanced_cosine"],
                    aggregation["norm_match_relative_error"],
                    aggregation["micro_gradient_norm"]["max_to_min_ratio"],
                    aggregation["pairwise_cosine"]["negative_fraction"],
                    record.get("state_gradient_nonzero_fraction"),
                ]
            )
    _write_csv(
        output,
        (
            "arm",
            "optimizer_step",
            "learning_rate",
            "loss_mean",
            "loss_F2",
            "loss_F3",
            "loss_F5",
            "loss_F6",
            "gradient_norm_before_clip",
            "gradient_clipped",
            "update_weight_ratio",
            "raw_vs_applied_cosine",
            "raw_vs_unit_balanced_cosine",
            "norm_match_relative_error",
            "micro_gradient_max_min_ratio",
            "negative_pairwise_cosine_fraction",
            "state_gradient_nonzero_fraction",
        ),
        rows,
    )


def _aggregation_csv(output: Path, metrics: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    rows = []
    for arm in ARMS:
        for record in metrics[arm]:
            aggregation = record["gradient_aggregation"]
            rows.append(
                [
                    arm,
                    record["optimizer_step"],
                    aggregation["mode"],
                    aggregation["raw_mean_norm"],
                    aggregation["unit_mean_norm_before_match"],
                    aggregation["applied_norm_before_clip"],
                    aggregation["norm_match_relative_error"],
                    aggregation["raw_vs_unit_balanced_cosine"],
                    aggregation["raw_vs_applied_cosine"],
                    aggregation["intervention_active"],
                    aggregation["micro_gradient_norm"]["minimum"],
                    aggregation["micro_gradient_norm"]["median"],
                    aggregation["micro_gradient_norm"]["maximum"],
                    aggregation["micro_gradient_norm"]["max_to_min_ratio"],
                    aggregation["pairwise_cosine"]["minimum"],
                    aggregation["pairwise_cosine"]["median"],
                    aggregation["pairwise_cosine"]["maximum"],
                    aggregation["pairwise_cosine"]["negative_fraction"],
                ]
            )
    _write_csv(
        output,
        (
            "arm",
            "optimizer_step",
            "mode",
            "raw_mean_norm",
            "unit_mean_norm_before_match",
            "applied_norm_before_clip",
            "norm_match_relative_error",
            "raw_vs_unit_balanced_cosine",
            "raw_vs_applied_cosine",
            "intervention_active",
            "micro_gradient_norm_minimum",
            "micro_gradient_norm_median",
            "micro_gradient_norm_maximum",
            "micro_gradient_norm_max_min_ratio",
            "pairwise_cosine_minimum",
            "pairwise_cosine_median",
            "pairwise_cosine_maximum",
            "negative_pairwise_cosine_fraction",
        ),
        rows,
    )


def _endpoint_rows(summaries: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    rows = []
    for arm in ARMS:
        summary = summaries[arm]
        comparisons = summary["comparisons"]
        hard8 = comparisons["train_overfit_hard8_endpoint_vs_m0"]
        formal = comparisons["formal_select_32_endpoint_vs_m0"]
        mechanism = comparisons["mechanism_select_32_endpoint_vs_m0"]
        did = comparisons["train_overfit_hard8_state_did"]
        gate = summary["aggregation_technical_gate"]
        rows.append(
            [
                arm,
                summary["gradient_aggregation"],
                hard8["m0_mean_ce"],
                hard8["endpoint_mean_ce"],
                hard8["estimate"],
                hard8["relative_change"],
                hard8["ci95"][0],
                hard8["ci95"][1],
                hard8["improved_pair_units"],
                summary["overfit_accuracy"]["m0"],
                summary["overfit_accuracy"]["endpoint"],
                summary["overfit_accuracy"]["delta"],
                did["estimate"],
                formal["estimate"],
                mechanism["estimate"],
                summary["gates"]["technical_gate"],
                summary["gates"]["hard8_overfit_learnability_gate"],
                summary["gates"]["fixed_dev_generalization_gate"],
                summary["training_summary"]["clip_rate"],
                gate["minimum_raw_vs_applied_cosine"],
                gate["maximum_norm_match_relative_error"],
            ]
        )
    return rows


def _training_figure(output: Path, metrics: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for arm in ARMS:
        records = metrics[arm]
        steps = [int(record["optimizer_step"]) for record in records]
        losses = [float(record["loss_mean"]) for record in records]
        gradients = [float(record["gradient_norm_before_clip"]) for record in records]
        cosines = [float(record["gradient_aggregation"]["raw_vs_applied_cosine"]) for record in records]
        color = COLORS[arm]
        axes[0, 0].plot(steps, losses, color=color, alpha=0.22, linewidth=0.8)
        axes[0, 0].plot(steps, _moving_mean(losses), color=color, linewidth=1.8, label=arm)
        axes[0, 1].plot(steps, gradients, color=color, linewidth=1.2, label=arm)
        axes[1, 0].plot(steps, cosines, color=color, linewidth=1.2, label=arm)
        for family, style in zip(("F2", "F3", "F5", "F6"), ("-", "--", "-.", ":"), strict=True):
            family_losses = [float(record["loss_by_family"][family]) for record in records]
            axes[1, 1].plot(
                steps,
                _moving_mean(family_losses),
                color=color,
                linestyle=style,
                alpha=0.9,
                label=f"{arm} {family}",
            )
    axes[0, 0].set(title="Hard8 training CE", xlabel="optimizer step", ylabel="CE (16-step mean)")
    axes[0, 1].axhline(10.0, color="black", linestyle="--", linewidth=0.9, label="clip=10")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Applied pre-clip gradient norm", xlabel="optimizer step", ylabel="L2 norm (log)")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set(title="Raw versus applied direction", xlabel="optimizer step", ylabel="cosine")
    axes[1, 1].set(title="Per-family training CE", xlabel="optimizer step", ylabel="CE (16-step mean)")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _aggregation_figure(output: Path, metrics: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for arm in ARMS:
        records = metrics[arm]
        steps = [int(record["optimizer_step"]) for record in records]
        reports = [record["gradient_aggregation"] for record in records]
        color = COLORS[arm]
        axes[0, 0].plot(
            steps,
            [float(report["raw_vs_unit_balanced_cosine"]) for report in reports],
            color=color,
            label=arm,
        )
        axes[0, 1].plot(
            steps,
            [float(report["micro_gradient_norm"]["max_to_min_ratio"]) for report in reports],
            color=color,
            label=arm,
        )
        axes[1, 0].plot(
            steps,
            [float(report["pairwise_cosine"]["negative_fraction"]) for report in reports],
            color=color,
            label=arm,
        )
        axes[1, 1].plot(
            steps,
            [max(float(report["norm_match_relative_error"]), 1e-16) for report in reports],
            color=color,
            label=arm,
        )
    axes[0, 0].set(title="Raw versus equal-unit direction", xlabel="optimizer step", ylabel="cosine")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Micro-gradient norm spread", xlabel="optimizer step", ylabel="max/min (log)")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].set(title="Directional conflict", xlabel="optimizer step", ylabel="negative pair fraction")
    axes[1, 1].axhline(1e-5, color="black", linestyle="--", linewidth=0.9, label="gate=1e-5")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(title="Norm-match error", xlabel="optimizer step", ylabel="relative error (log)")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _endpoint_figure(output: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    positions = list(range(len(ARMS)))
    width = 0.36
    hard8 = [summaries[arm]["comparisons"]["train_overfit_hard8_endpoint_vs_m0"] for arm in ARMS]
    axes[0].bar(
        [value - width / 2 for value in positions],
        [value["m0_mean_ce"] for value in hard8],
        width,
        label="M0",
    )
    axes[0].bar(
        [value + width / 2 for value in positions],
        [value["endpoint_mean_ce"] for value in hard8],
        width,
        label="EMA step128",
    )
    axes[0].set(title="Hard8 CE", ylabel="mean CE (lower is better)")
    axes[0].legend(fontsize=8)
    axes[1].bar(
        [value - width / 2 for value in positions],
        [summaries[arm]["overfit_accuracy"]["m0"] for arm in ARMS],
        width,
        label="M0",
    )
    axes[1].bar(
        [value + width / 2 for value in positions],
        [summaries[arm]["overfit_accuracy"]["endpoint"] for arm in ARMS],
        width,
        label="EMA step128",
    )
    axes[1].set(title="Hard8 accuracy", ylabel="accuracy")
    axes[1].legend(fontsize=8)
    suites = ("hard8", "formal", "mechanism")
    keys = (
        "train_overfit_hard8_endpoint_vs_m0",
        "formal_select_32_endpoint_vs_m0",
        "mechanism_select_32_endpoint_vs_m0",
    )
    suite_positions = list(range(len(suites)))
    for offset, arm in ((-width / 2, ARMS[0]), (width / 2, ARMS[1])):
        axes[2].bar(
            [value + offset for value in suite_positions],
            [summaries[arm]["comparisons"][key]["estimate"] for key in keys],
            width,
            color=COLORS[arm],
            label=arm,
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(suite_positions, suites)
    axes[2].set(title="Endpoint minus own M0", ylabel="paired CE delta")
    axes[2].legend(fontsize=7)
    for axis in axes[:2]:
        axis.set_xticks(positions, ["raw", "unit-balanced"])
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _report(summaries: Mapping[str, Mapping[str, Any]], endpoint_rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "# R7 gradient-balance paired diagnostic delivery",
        "",
        "This is a one-seed repeated-hard8 bottleneck diagnostic and cannot establish formal picture-memory success.",
        "",
        f"- Git commit: `{summaries[ARMS[0]]['git_commit']}`",
        f"- Implementation: `{summaries[ARMS[0]]['implementation_revision']}`",
        f"- Selected-segment SHA-256: `{summaries[ARMS[0]]['selected_segments_sha256']}`",
        "",
        "| Arm | hard8 M0 CE | endpoint CE | delta CE | relative | improved units | accuracy delta | hard8 gate | formal delta | mechanism delta | fixed-dev gate | clip rate | min raw/applied cosine | max norm error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for arm, row in zip(ARMS, endpoint_rows, strict=True):
        lines.append(
            "| "
            + " | ".join(
                (
                    arm,
                    _fmt(row[2]),
                    _fmt(row[3]),
                    _fmt(row[4]),
                    _fmt(row[5]),
                    str(row[8]),
                    _fmt(row[11]),
                    _fmt(row[16]),
                    _fmt(row[13]),
                    _fmt(row[14]),
                    _fmt(row[17]),
                    _fmt(row[18]),
                    _fmt(row[19]),
                    f"{float(row[20]):.2e}",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Interpretation must follow the preregistered gates. Both arms share the same M0 architecture and differ only in the applied gradient direction; the unit-balanced norm is matched before the unchanged clip.",
            "",
        )
    )
    return "\n".join(lines)


def render(raw_root: Path, balanced_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R7 renderer refuses a non-empty output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = {ARMS[0]: raw_root.resolve(), ARMS[1]: balanced_root.resolve()}
    summaries: dict[str, dict[str, Any]] = {}
    metrics: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        summaries[arm], metrics[arm] = _validate_arm(roots[arm], arm)
        if len(metrics[arm]) != 128:
            raise ValueError(f"R7 arm does not contain exactly 128 optimizer records: {arm}")
    _validate_pair(summaries)

    _training_csv(output_dir / "training_metrics.csv", metrics)
    _aggregation_csv(output_dir / "aggregation_diagnostics.csv", metrics)
    endpoint_rows = _endpoint_rows(summaries)
    _write_csv(
        output_dir / "endpoint_summary.csv",
        (
            "arm",
            "gradient_aggregation",
            "hard8_m0_ce",
            "hard8_endpoint_ce",
            "hard8_delta_ce",
            "hard8_relative_change",
            "hard8_ci95_low",
            "hard8_ci95_high",
            "hard8_improved_units",
            "hard8_m0_accuracy",
            "hard8_endpoint_accuracy",
            "hard8_accuracy_delta",
            "hard8_state_did",
            "formal_delta_ce",
            "mechanism_delta_ce",
            "technical_gate",
            "hard8_gate",
            "fixed_dev_gate",
            "clip_rate",
            "minimum_raw_vs_applied_cosine",
            "maximum_norm_match_relative_error",
        ),
        endpoint_rows,
    )
    _training_figure(output_dir / "training_diagnostics.png", metrics)
    _aggregation_figure(output_dir / "aggregation_diagnostics.png", metrics)
    _endpoint_figure(output_dir / "endpoint_metrics.png", summaries)
    (output_dir / "REPORT.md").write_text(_report(summaries, endpoint_rows), encoding="utf-8")

    analysis = {
        "schema": "vision_memory.r7-gradient-balance-rendered-analysis.v1",
        "status": "completed",
        "formal_success_claim": False,
        "source_roots": {arm: str(root) for arm, root in roots.items()},
        "git_commit": summaries[ARMS[0]]["git_commit"],
        "implementation_revision": summaries[ARMS[0]]["implementation_revision"],
        "selected_segments_sha256": summaries[ARMS[0]]["selected_segments_sha256"],
        "aggregation_technical_gates": {arm: summaries[arm]["aggregation_technical_gate"] for arm in ARMS},
        "gates": {arm: summaries[arm]["gates"] for arm in ARMS},
        "comparisons": {arm: summaries[arm]["comparisons"] for arm in ARMS},
    }
    _write_json(output_dir / "ANALYSIS.json", analysis)
    inventory = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        if path.name == "DELIVERY_MANIFEST.json":
            continue
        inventory.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    delivery = {
        "schema": "vision_memory.r7-gradient-balance-delivery-manifest.v1",
        "artifacts": inventory,
        "source_inventory_sha256": {arm: _sha256(roots[arm] / "artifact_inventory.json") for arm in ARMS},
    }
    _write_json(output_dir / "DELIVERY_MANIFEST.json", delivery)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--balanced-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = render(args.raw_root, args.balanced_root, args.output_dir)
    print(json.dumps({"status": analysis["status"], "formal_success_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
