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
TRAJECTORY_SCHEMA = "vision_memory.r8-checkpoint-trajectory.v1"
TRAJECTORY_INVENTORY_SCHEMA = "vision_memory.r8-checkpoint-trajectory-inventory.v1"
TRAJECTORY_LABELS = ("m0", "ema_step32", "ema_step64", "ema_step96", "ema_step128")
TRAJECTORY_STEPS = {"m0": 0, "ema_step32": 32, "ema_step64": 64, "ema_step96": 96, "ema_step128": 128}
TRAJECTORY_CONDITIONS = ("normal", "reset", "cross_episode_swap", "temporal_swap")
EXPECTED_MODE = {
    "raw-mean-control": "raw-mean",
    "common-descent-projected-norm-matched": "common-descent-projected-norm-matched",
}


def _validate_inventory(root: Path, *, expected_schema: str) -> dict[str, Any]:
    inventory = base._load_json(root / "artifact_inventory.json")
    if inventory.get("schema") != expected_schema:
        raise ValueError(f"Artifact inventory schema mismatch: {root}")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Artifact inventory is empty: {root}")
    for record in records:
        path = root / str(record["path"])
        if not path.is_file():
            raise ValueError(f"Inventory artifact is missing: {path}")
        if path.stat().st_size != int(record["bytes"]) or base._sha256(path) != record["sha256"]:
            raise ValueError(f"Inventory artifact failed size/SHA validation: {path}")
    return inventory


def _validate_trajectory(
    root: Path,
    *,
    expected_arm: str,
    arm_root: Path,
    source_summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _validate_inventory(root, expected_schema=TRAJECTORY_INVENTORY_SCHEMA)
    summary = base._load_json(root / "trajectory_summary.json")
    if summary.get("schema") != TRAJECTORY_SCHEMA or summary.get("status") != "completed":
        raise ValueError(f"R8 trajectory schema/status mismatch: {root}")
    if summary.get("formal_success_claim") is not False or summary.get("arm") != expected_arm:
        raise ValueError(f"R8 trajectory scope/arm mismatch: {root}")
    if summary.get("gradient_aggregation") != EXPECTED_MODE[expected_arm]:
        raise ValueError(f"R8 trajectory aggregation drift: {root}")
    if summary.get("training_commit") != source_summary.get("git_commit"):
        raise ValueError(f"R8 trajectory training lineage drift: {root}")
    if summary.get("selected_segments_sha256") != source_summary.get("selected_segments_sha256"):
        raise ValueError(f"R8 trajectory hard8 selection drift: {root}")
    trajectory = summary.get("trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError(f"R8 trajectory statistics are missing: {root}")
    if tuple(trajectory.get("checkpoint_order", ())) != TRAJECTORY_LABELS:
        raise ValueError(f"R8 trajectory checkpoint order drift: {root}")
    if trajectory.get("descriptive_only_not_checkpoint_selection") is not True:
        raise ValueError(f"R8 trajectory permits checkpoint selection: {root}")
    if trajectory.get("primary_endpoint_remains") != "ema_step128":
        raise ValueError(f"R8 trajectory changed the primary endpoint: {root}")
    source_validation = summary.get("source_validation", {})
    source_inventory = arm_root / "artifact_inventory.json"
    if source_validation.get("inventory_sha256") != base._sha256(source_inventory):
        raise ValueError(f"R8 trajectory source inventory binding failed: {root}")
    endpoint_binding = summary.get("endpoint_binding", {})
    if endpoint_binding.get("passed") is not True or int(endpoint_binding.get("matched_tensors", 0)) <= 0:
        raise ValueError(f"R8 trajectory endpoint binding failed: {root}")
    hashes = summary.get("checkpoint_sha256", {})
    if set(hashes) != {"step0", "step32", "step64", "step96", "step128"} or any(
        not isinstance(value, str) or len(value) != 64 for value in hashes.values()
    ):
        raise ValueError(f"R8 trajectory checkpoint hashes are incomplete: {root}")

    rows = base._load_jsonl(root / "hard8_checkpoint_rows.jsonl")
    if int(summary.get("rows", -1)) != 640 or len(rows) != 640:
        raise ValueError(f"R8 trajectory does not contain exactly 640 rows: {root}")
    labels = {str(row.get("checkpoint")) for row in rows}
    conditions = {str(row.get("condition")) for row in rows}
    suites = {str(row.get("suite")) for row in rows}
    if labels != set(TRAJECTORY_LABELS) or conditions != set(TRAJECTORY_CONDITIONS):
        raise ValueError(f"R8 trajectory checkpoint/condition coverage drift: {root}")
    if suites != {"train_overfit_hard8"}:
        raise ValueError(f"R8 trajectory suite drift: {root}")
    for label in TRAJECTORY_LABELS:
        for condition in TRAJECTORY_CONDITIONS:
            subset = [
                row
                for row in rows
                if row.get("checkpoint") == label and row.get("condition") == condition
            ]
            if len(subset) != 32 or len({str(row.get("pair_unit")) for row in subset}) != 8:
                raise ValueError(f"R8 trajectory cell coverage drift: {root}:{label}:{condition}")
            if {int(row.get("view_index", -1)) for row in subset} != {0, 1, 2, 3}:
                raise ValueError(f"R8 trajectory view coverage drift: {root}:{label}:{condition}")
            if any(not math.isfinite(float(row["ce"])) for row in subset):
                raise ValueError(f"R8 trajectory contains non-finite CE: {root}")
    reported_accuracy = trajectory.get("normal_accuracy", {})
    comparisons = trajectory.get("normal_ce_vs_m0", {})
    state_did = trajectory.get("normal_reset_difference_in_differences_vs_m0", {})
    if not all(isinstance(value, Mapping) for value in (reported_accuracy, comparisons, state_did)):
        raise ValueError(f"R8 trajectory aggregate structure drift: {root}")
    m0_normal = _trajectory_cell_mean(rows, label="m0", condition="normal", field="ce")
    m0_reset = _trajectory_cell_mean(rows, label="m0", condition="reset", field="ce")
    m0_per_unit = {
        unit: sum(
            float(row["ce"])
            for row in rows
            if row.get("checkpoint") == "m0"
            and row.get("condition") == "normal"
            and str(row.get("pair_unit")) == unit
        )
        / 4.0
        for unit in sorted({str(row.get("pair_unit")) for row in rows})
    }
    for label in TRAJECTORY_LABELS:
        observed_accuracy = _trajectory_cell_mean(rows, label=label, condition="normal", field="correct")
        if label not in reported_accuracy or not math.isclose(
            float(reported_accuracy[label]), observed_accuracy, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(f"R8 trajectory reported accuracy disagrees with raw rows: {root}:{label}")
        if label == "m0":
            continue
        comparison_record = comparisons.get(label)
        did_record = state_did.get(label)
        if not isinstance(comparison_record, Mapping) or not isinstance(did_record, Mapping):
            raise ValueError(f"R8 trajectory aggregate checkpoint is missing: {root}:{label}")
        endpoint_normal = _trajectory_cell_mean(rows, label=label, condition="normal", field="ce")
        endpoint_reset = _trajectory_cell_mean(rows, label=label, condition="reset", field="ce")
        expected_values = {
            "m0_mean_ce": m0_normal,
            "endpoint_mean_ce": endpoint_normal,
            "estimate": endpoint_normal - m0_normal,
            "relative_change": endpoint_normal / m0_normal - 1.0,
        }
        if any(
            key not in comparison_record
            or not math.isclose(
                float(comparison_record[key]), expected, rel_tol=1e-9, abs_tol=1e-9
            )
            for key, expected in expected_values.items()
        ):
            raise ValueError(f"R8 trajectory CE aggregate disagrees with raw rows: {root}:{label}")
        endpoint_per_unit = {
            unit: sum(
                float(row["ce"])
                for row in rows
                if row.get("checkpoint") == label
                and row.get("condition") == "normal"
                and str(row.get("pair_unit")) == unit
            )
            / 4.0
            for unit in m0_per_unit
        }
        improved_units = sum(endpoint_per_unit[unit] < m0_per_unit[unit] for unit in m0_per_unit)
        if int(comparison_record.get("improved_pair_units", -1)) != improved_units:
            raise ValueError(f"R8 trajectory improved-unit count disagrees with raw rows: {root}:{label}")
        observed_did = (endpoint_normal - endpoint_reset) - (m0_normal - m0_reset)
        if "estimate" not in did_record or not math.isclose(
            float(did_record["estimate"]), observed_did, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(f"R8 trajectory state DiD disagrees with raw rows: {root}:{label}")
    return summary, rows


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


def _trajectory_cell_mean(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    condition: str,
    field: str,
) -> float:
    values = [
        float(row[field])
        for row in rows
        if row.get("checkpoint") == label and row.get("condition") == condition
    ]
    if len(values) != 32 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"R8 trajectory cell is incomplete: {label}:{condition}:{field}")
    return sum(values) / len(values)


def _trajectory_rows(
    summaries: Mapping[str, Mapping[str, Any]],
    evaluation_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[list[Any]]:
    output: list[list[Any]] = []
    for arm in ARMS:
        trajectory = summaries[arm]["trajectory"]
        comparisons = trajectory["normal_ce_vs_m0"]
        state_did = trajectory["normal_reset_difference_in_differences_vs_m0"]
        for label in TRAJECTORY_LABELS:
            comparison_record = comparisons.get(label)
            did_record = state_did.get(label)
            output.append(
                [
                    arm,
                    label,
                    TRAJECTORY_STEPS[label],
                    label == "ema_step128",
                    _trajectory_cell_mean(evaluation_rows[arm], label=label, condition="normal", field="ce"),
                    _trajectory_cell_mean(evaluation_rows[arm], label=label, condition="normal", field="correct"),
                    _trajectory_cell_mean(evaluation_rows[arm], label=label, condition="reset", field="ce"),
                    _trajectory_cell_mean(
                        evaluation_rows[arm], label=label, condition="cross_episode_swap", field="ce"
                    ),
                    _trajectory_cell_mean(
                        evaluation_rows[arm], label=label, condition="temporal_swap", field="ce"
                    ),
                    comparison_record.get("estimate") if comparison_record else None,
                    comparison_record.get("relative_change") if comparison_record else None,
                    comparison_record.get("ci95", [None, None])[0] if comparison_record else None,
                    comparison_record.get("ci95", [None, None])[1] if comparison_record else None,
                    comparison_record.get("improved_pair_units") if comparison_record else None,
                    did_record.get("estimate") if did_record else None,
                    did_record.get("ci95", [None, None])[0] if did_record else None,
                    did_record.get("ci95", [None, None])[1] if did_record else None,
                ]
            )
    return output


def _trajectory_figure(output: Path, rows: Sequence[Sequence[Any]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for arm in ARMS:
        selected = [row for row in rows if row[0] == arm]
        steps = [int(row[2]) for row in selected]
        color = COLORS[arm]
        short = "raw" if arm == ARMS[0] else "projected"
        axes[0, 0].plot(steps, [float(row[4]) for row in selected], marker="o", color=color, label=short)
        axes[0, 1].plot(steps, [float(row[5]) for row in selected], marker="o", color=color, label=short)
        axes[1, 0].plot(
            steps[1:], [float(row[9]) for row in selected[1:]], marker="o", color=color, label=short
        )
        axes[1, 1].plot(
            steps[1:], [float(row[14]) for row in selected[1:]], marker="o", color=color, label=short
        )
    axes[0, 0].set(title="Hard8 normal CE trajectory", xlabel="optimizer step", ylabel="mean CE")
    axes[0, 1].set(title="Hard8 normal accuracy trajectory", xlabel="optimizer step", ylabel="accuracy")
    axes[0, 1].set_ylim(-0.03, 1.03)
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set(title="Normal CE change from M0", xlabel="optimizer step", ylabel="delta CE")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set(
        title="Normal/reset difference-in-differences",
        xlabel="optimizer step",
        ylabel="DiD (negative = learned state dependence)",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.18)
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
    trajectory_rows: Sequence[Sequence[Any]] | None = None,
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
    if trajectory_rows is not None:
        lines.extend(
            (
                "## Fixed checkpoint trajectory",
                "",
                "Steps 32/64/96 are descriptive only; step128 remains the preregistered primary endpoint and no intermediate checkpoint may rescue a failed endpoint.",
                "",
                "| Arm | Step | normal CE | delta from M0 | accuracy | normal/reset DiD |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            )
        )
        for row in trajectory_rows:
            lines.append(
                f"| {row[0]} | {row[2]} | {_fmt(row[4])} | {_fmt(row[9])} | {_fmt(row[5])} | {_fmt(row[14])} |"
            )
        lines.append("")
    return "\n".join(lines)


def render(
    raw_root: Path,
    projected_root: Path,
    output_dir: Path,
    *,
    raw_trajectory_root: Path | None = None,
    projected_trajectory_root: Path | None = None,
) -> dict[str, Any]:
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

    supplied_trajectories = (raw_trajectory_root, projected_trajectory_root)
    if sum(value is not None for value in supplied_trajectories) not in (0, 2):
        raise ValueError("R8 renderer requires both checkpoint trajectories or neither.")
    trajectory_roots: dict[str, Path] = {}
    trajectory_summaries: dict[str, dict[str, Any]] = {}
    trajectory_evaluation_rows: dict[str, list[dict[str, Any]]] = {}
    rendered_trajectory_rows: list[list[Any]] | None = None
    if all(value is not None for value in supplied_trajectories):
        assert raw_trajectory_root is not None and projected_trajectory_root is not None
        trajectory_roots = {
            ARMS[0]: raw_trajectory_root.resolve(),
            ARMS[1]: projected_trajectory_root.resolve(),
        }
        for arm in ARMS:
            trajectory_summaries[arm], trajectory_evaluation_rows[arm] = _validate_trajectory(
                trajectory_roots[arm],
                expected_arm=arm,
                arm_root=roots[arm],
                source_summary=summaries[arm],
            )
        rendered_trajectory_rows = _trajectory_rows(trajectory_summaries, trajectory_evaluation_rows)

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
    if rendered_trajectory_rows is not None:
        base._write_csv(
            output_dir / "checkpoint_trajectory.csv",
            (
                "arm",
                "checkpoint",
                "optimizer_step",
                "primary_endpoint",
                "normal_mean_ce",
                "normal_accuracy",
                "reset_mean_ce",
                "cross_episode_swap_mean_ce",
                "temporal_swap_mean_ce",
                "normal_delta_ce_vs_m0",
                "normal_relative_change_vs_m0",
                "normal_delta_ci95_low",
                "normal_delta_ci95_high",
                "improved_pair_units",
                "normal_reset_did",
                "normal_reset_did_ci95_low",
                "normal_reset_did_ci95_high",
            ),
            rendered_trajectory_rows,
        )
        _trajectory_figure(output_dir / "checkpoint_trajectory.png", rendered_trajectory_rows)
    (output_dir / "REPORT.md").write_text(
        _report(
            summaries,
            endpoint_rows,
            decision=decision,
            reason=reason,
            trajectory_rows=rendered_trajectory_rows,
        ),
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
        "checkpoint_trajectory": {
            "included": rendered_trajectory_rows is not None,
            "source_roots": {arm: str(root) for arm, root in trajectory_roots.items()},
            "summaries": {
                arm: trajectory_summaries[arm]["trajectory"] for arm in trajectory_summaries
            },
        },
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
        "source_trajectory_inventory_sha256": {
            arm: base._sha256(trajectory_roots[arm] / "artifact_inventory.json")
            for arm in trajectory_roots
        },
    }
    base._write_json(output_dir / "DELIVERY_MANIFEST.json", delivery)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--projected-root", type=Path, required=True)
    parser.add_argument("--raw-trajectory-root", type=Path)
    parser.add_argument("--projected-trajectory-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = render(
        args.raw_root,
        args.projected_root,
        args.output_dir,
        raw_trajectory_root=args.raw_trajectory_root,
        projected_trajectory_root=args.projected_trajectory_root,
    )
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
