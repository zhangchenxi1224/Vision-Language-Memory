"""Render and integrity-check a completed paired R8 common-descent diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import compare_r8_common_descent as comparison  # noqa: E402
from scripts.reporting import render_r7_gradient_balance_report as base  # noqa: E402


ARMS = ("raw-mean-control", "common-descent-projected-norm-matched")
COLORS = {"raw-mean-control": "#d97706", "common-descent-projected-norm-matched": "#059669"}
EXPECTED_MODE = {
    "raw-mean-control": "raw-mean",
    "common-descent-projected-norm-matched": "common-descent-projected-norm-matched",
}


def _validate_arm(root: Path, expected_arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    terminal = base._load_json(root / "terminal.json")
    if terminal.get("status") != "completed_diagnostic" or terminal.get("passed") is not True:
        raise ValueError(f"R8 arm did not complete cleanly: {root}")
    summary = base._load_json(root / "run" / "r8_summary.json")
    if summary.get("schema") != "vision_memory.r8-common-descent-summary.v1":
        raise ValueError(f"R8 summary schema mismatch: {root}")
    if summary.get("status") != "completed" or summary.get("arm") != expected_arm:
        raise ValueError(f"R8 summary arm/status mismatch: {root}")
    if summary.get("gradient_aggregation") != EXPECTED_MODE[expected_arm]:
        raise ValueError(f"R8 aggregation mode drift: {root}")
    if summary.get("full_success_claim_allowed") is not False:
        raise ValueError(f"R8 diagnostic improperly permits a formal success claim: {root}")

    inventory = base._load_json(root / "artifact_inventory.json")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"R8 artifact inventory is empty: {root}")
    for record in records:
        path = root / str(record["path"])
        if not path.is_file():
            raise ValueError(f"Inventory artifact is missing: {path}")
        if path.stat().st_size != int(record["bytes"]) or base._sha256(path) != record["sha256"]:
            raise ValueError(f"Inventory artifact failed size/SHA validation: {path}")
    metrics = [
        record
        for record in base._load_jsonl(root / "run" / "metrics.jsonl")
        if record.get("kind") == "optimizer_step"
    ]
    return summary, metrics


def _validate_pair(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    raw = summaries[ARMS[0]]
    projected = summaries[ARMS[1]]
    comparison._validate(raw, projected)
    for key in ("git_commit", "implementation_revision", "selected_segments_sha256", "edit_start_sigma"):
        values = {summary.get(key) for summary in summaries.values()}
        if len(values) != 1 or None in values:
            raise ValueError(f"R8 paired-arm drift in {key}: {values}")


def _projection_csv(output: Path, metrics: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    rows: list[list[Any]] = []
    for arm in ARMS:
        for record in metrics[arm]:
            aggregation = record["gradient_aggregation"]
            projection = aggregation.get("common_descent_projection", {})
            rows.append(
                [
                    arm,
                    record["optimizer_step"],
                    aggregation["mode"],
                    aggregation["raw_vs_applied_cosine"],
                    aggregation["norm_match_relative_error"],
                    aggregation["intervention_active"],
                    aggregation["pairwise_cosine"]["negative_fraction"],
                    projection.get("minimum_raw_micro_cosine"),
                    projection.get("minimum_projected_micro_cosine"),
                    projection.get("raw_violating_micro_count"),
                    projection.get("projected_violating_micro_count_at_tolerance"),
                    projection.get("active_constraint_count"),
                    projection.get("selected_active_set_mask"),
                    projection.get("projection_distance_squared"),
                    projection.get("projected_norm_before_match"),
                ]
            )
    base._write_csv(
        output,
        (
            "arm",
            "optimizer_step",
            "mode",
            "raw_vs_applied_cosine",
            "norm_match_relative_error",
            "intervention_active",
            "negative_pairwise_cosine_fraction",
            "minimum_raw_micro_cosine",
            "minimum_projected_micro_cosine",
            "raw_violating_micro_count",
            "projected_violating_micro_count_at_tolerance",
            "active_constraint_count",
            "selected_active_set_mask",
            "projection_distance_squared",
            "projected_norm_before_match",
        ),
        rows,
    )


def _projection_figure(output: Path, records: Sequence[Mapping[str, Any]]) -> None:
    steps = [int(record["optimizer_step"]) for record in records]
    projections = [record["gradient_aggregation"]["common_descent_projection"] for record in records]
    raw_minimum = [float(value["minimum_raw_micro_cosine"]) for value in projections]
    projected_minimum = [float(value["minimum_projected_micro_cosine"]) for value in projections]
    raw_violations = [int(value["raw_violating_micro_count"]) for value in projections]
    projected_violations = [int(value["projected_violating_micro_count_at_tolerance"]) for value in projections]
    active = [int(value["active_constraint_count"]) for value in projections]
    distances = [max(float(value["projection_distance_squared"]), 1e-16) for value in projections]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    axes[0, 0].plot(steps, raw_minimum, color="#dc2626", linewidth=1.1, label="raw minimum")
    axes[0, 0].plot(steps, projected_minimum, color="#059669", linewidth=1.1, label="projected minimum")
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].axhline(-1e-5, color="black", linestyle="--", linewidth=0.8, label="gate=-1e-5")
    axes[0, 0].set(title="Worst micro-gradient alignment", xlabel="optimizer step", ylabel="cosine")
    axes[0, 1].plot(steps, raw_violations, color="#dc2626", label="raw")
    axes[0, 1].plot(steps, projected_violations, color="#059669", label="projected at tolerance")
    axes[0, 1].set(title="Micro constraints violated", xlabel="optimizer step", ylabel="count out of 8")
    axes[1, 0].plot(steps, active, color="#7c3aed")
    axes[1, 0].set(title="Active projection constraints", xlabel="optimizer step", ylabel="count")
    axes[1, 1].plot(steps, distances, color="#2563eb")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(title="Projection intervention size", xlabel="optimizer step", ylabel="squared distance (log)")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        if axis.lines and axis in (axes[0, 0], axes[0, 1]):
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _endpoint_rows(summaries: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    rows = base._endpoint_rows(summaries, arms=ARMS)
    for arm, row in zip(ARMS, rows, strict=True):
        gate = summaries[arm]["aggregation_technical_gate"]
        row.extend(
            (
                gate.get("minimum_projected_micro_cosine"),
                gate.get("maximum_projected_violation_count"),
                gate.get("raw_violating_steps"),
            )
        )
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    return base._fmt(value, digits)


def _report(
    summaries: Mapping[str, Mapping[str, Any]],
    endpoint_rows: Sequence[Sequence[Any]],
    *,
    decision: str,
    reason: str,
) -> str:
    lines = [
        "# R8 common-descent paired diagnostic delivery",
        "",
        "This is a one-seed repeated-hard8 bottleneck diagnostic and cannot establish formal picture-memory success.",
        "",
        f"- Decision: `{decision}`",
        f"- Reason: {reason}",
        f"- Git commit: `{summaries[ARMS[0]]['git_commit']}`",
        f"- Implementation: `{summaries[ARMS[0]]['implementation_revision']}`",
        f"- Selected-segment SHA-256: `{summaries[ARMS[0]]['selected_segments_sha256']}`",
        "",
        "| Arm | hard8 M0 CE | endpoint CE | delta CE | relative | improved units | accuracy delta | hard8 gate | formal delta | mechanism delta | fixed-dev gate | clip rate | min raw/applied cosine | min projected/micro cosine | raw-conflict steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
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
                    _fmt(row[21], 6),
                    _fmt(row[23], 0),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "The projected constraint is enforced in pre-AdamW gradient geometry. Adam moments, diagonal preconditioning, weight decay, and finite step size can still alter the realized parameter update.",
            "",
        )
    )
    return "\n".join(lines)


def render(raw_root: Path, projected_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R8 renderer refuses a non-empty output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = {ARMS[0]: raw_root.resolve(), ARMS[1]: projected_root.resolve()}
    summaries: dict[str, dict[str, Any]] = {}
    metrics: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        summaries[arm], metrics[arm] = _validate_arm(roots[arm], arm)
        if len(metrics[arm]) != 128:
            raise ValueError(f"R8 arm does not contain exactly 128 optimizer records: {arm}")
    _validate_pair(summaries)
    decision, reason = comparison._decision(summaries[ARMS[0]], summaries[ARMS[1]])

    base._training_csv(output_dir / "training_metrics.csv", metrics, arms=ARMS)
    base._aggregation_csv(output_dir / "aggregation_diagnostics.csv", metrics, arms=ARMS)
    _projection_csv(output_dir / "projection_diagnostics.csv", metrics)
    endpoint_rows = _endpoint_rows(summaries)
    base._write_csv(
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
            "minimum_projected_micro_cosine",
            "maximum_projected_violation_count",
            "raw_violating_steps",
        ),
        endpoint_rows,
    )
    base._training_figure(output_dir / "training_diagnostics.png", metrics, arms=ARMS, colors=COLORS)
    _projection_figure(output_dir / "projection_diagnostics.png", metrics[ARMS[1]])
    base._endpoint_figure(
        output_dir / "endpoint_metrics.png",
        summaries,
        arms=ARMS,
        colors=COLORS,
        xlabels=("raw", "projected"),
    )
    (output_dir / "REPORT.md").write_text(
        _report(summaries, endpoint_rows, decision=decision, reason=reason),
        encoding="utf-8",
    )

    analysis = {
        "schema": "vision_memory.r8-common-descent-rendered-analysis.v1",
        "status": "completed",
        "formal_success_claim": False,
        "decision": decision,
        "reason": reason,
        "source_roots": {arm: str(root) for arm, root in roots.items()},
        "git_commit": summaries[ARMS[0]]["git_commit"],
        "implementation_revision": summaries[ARMS[0]]["implementation_revision"],
        "selected_segments_sha256": summaries[ARMS[0]]["selected_segments_sha256"],
        "aggregation_technical_gates": {arm: summaries[arm]["aggregation_technical_gate"] for arm in ARMS},
        "gates": {arm: summaries[arm]["gates"] for arm in ARMS},
        "comparisons": {arm: summaries[arm]["comparisons"] for arm in ARMS},
    }
    base._write_json(output_dir / "ANALYSIS.json", analysis)
    artifacts = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        if path.name == "DELIVERY_MANIFEST.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": base._sha256(path),
            }
        )
    delivery = {
        "schema": "vision_memory.r8-common-descent-delivery-manifest.v1",
        "artifacts": artifacts,
        "source_inventory_sha256": {
            arm: base._sha256(roots[arm] / "artifact_inventory.json") for arm in ARMS
        },
    }
    base._write_json(output_dir / "DELIVERY_MANIFEST.json", delivery)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--projected-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = render(args.raw_root, args.projected_root, args.output_dir)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "decision": analysis["decision"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
