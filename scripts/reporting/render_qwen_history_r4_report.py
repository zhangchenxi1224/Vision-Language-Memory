from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "inspire"))
sys.path.insert(0, str(ROOT / "src"))

from qwen_history_r4_contract import (  # noqa: E402
    ARM_METHODS,
    ARM_ORDER,
    COMPARISON_SCHEMA,
    SCORE_SCHEMA,
    STAGE_EVIDENCE_PROTOCOL,
)


SCHEMA = "vlm.qwen-history-r4-audit-report.v1"
COMBINED_SCHEMA = "vlm.qwen-history-r4-combined-report.v1"
PLOT_NAMES = (
    "condition_accuracy.png",
    "target_position_accuracy.png",
    "event_kind_accuracy.png",
    "form_accuracy.png",
    "nll_margin.png",
    "memory_bytes.png",
    "latency_seconds.png",
    "rotation_state_swap.png",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            values.append(value)
    if not values:
        raise ValueError(f"Prediction file is empty: {path}")
    return values


def atomic_write(path: Path, payload: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if isinstance(payload, bytes):
        temporary.write_bytes(payload)
    else:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 9,
        }
    )


def _empty(axis: plt.Axes, message: str) -> None:
    axis.set_axis_off()
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _accuracy_plot(groups: Mapping[str, Any], title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    names: list[str] = []
    values: list[float] = []
    counts: list[str] = []
    for name, entry in groups.items():
        if isinstance(entry, Mapping) and isinstance(entry.get("accuracy"), (int, float)):
            names.append(str(name))
            values.append(float(entry["accuracy"]))
            counts.append(f"{entry.get('correct', '?')}/{entry.get('count', '?')}")
    if not values:
        _empty(axis, "No observations for this breakdown")
    else:
        bars = axis.bar(names, values, color="#4472C4")
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("Accuracy")
        axis.tick_params(axis="x", rotation=30)
        for bar, label in zip(bars, counts, strict=True):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.02, bar.get_height() + 0.025),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axis.set_title(title)
    _save(fig, path)


def _numeric_histogram(
    rows: Sequence[Mapping[str, Any]], field: str, title: str, path: Path
) -> None:
    values = [
        float(row[field])
        for row in rows
        if row.get("condition") == "standard"
        and isinstance(row.get(field), (int, float))
        and not isinstance(row.get(field), bool)
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    if not values:
        _empty(axis, f"No numeric {field} observations")
    else:
        bins = min(40, max(5, int(len(values) ** 0.5)))
        axis.hist(values, bins=bins, color="#70AD47", alpha=0.82)
        axis.axvline(sum(values) / len(values), color="#C00000", linestyle="--", label="mean")
        axis.legend()
        axis.set_xlabel(field)
        axis.set_ylabel("Records")
    axis.set_title(title)
    _save(fig, path)


def _rotation_swap_plot(metrics: Mapping[str, Any], path: Path) -> None:
    rotation = metrics.get("rotation", {})
    swap = metrics.get("state_swap_donor_answer", {})
    labels = ["rotation agreement", "state-swap donor rate"]
    values = [
        rotation.get("agreement_rate") if isinstance(rotation, Mapping) else None,
        swap.get("rate") if isinstance(swap, Mapping) else None,
    ]
    fig, axis = plt.subplots(figsize=(6.2, 4.2))
    if not any(isinstance(value, (int, float)) for value in values):
        _empty(axis, "Rotation/state-swap metrics are unavailable for this stage")
    else:
        heights = [0.0 if value is None else float(value) for value in values]
        bars = axis.bar(labels, heights, color=["#5B9BD5", "#ED7D31"])
        axis.set_ylim(0, 1.05)
        for bar, value in zip(bars, values, strict=True):
            label = "n/a" if value is None else f"{float(value):.3f}"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.02, bar.get_height() + 0.025),
                label,
                ha="center",
            )
    axis.set_title("Causal and permutation diagnostics")
    _save(fig, path)


def _copy_source(path: Path, destination: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, destination)
    return {
        "label": label,
        "original_path": str(path.resolve()),
        "report_path": str(destination.resolve()),
        "size": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _validate_score_bindings(
    *,
    arm: str,
    score: Mapping[str, Any],
    predictions_a: Path,
    report_a: Path,
    predictions_b: Path,
    report_b: Path,
) -> None:
    method = ARM_METHODS[arm]
    if score.get("schema") != SCORE_SCHEMA or score.get("method") != method:
        raise ValueError(f"R4 {arm} score schema/method mismatch")
    if score.get("passed") is not True or score.get("execution_passed") is not True:
        raise ValueError(f"R4 {arm} score did not pass integrity/replication execution checks")
    integrity = score.get("integrity")
    replication = score.get("replication")
    if not isinstance(integrity, Mapping) or integrity.get("passed") is not True:
        raise ValueError(f"R4 {arm} score failed integrity")
    if (
        not isinstance(replication, Mapping)
        or replication.get("passed") is not True
        or replication.get("bitwise_scientific_payload_match") is not True
    ):
        raise ValueError(f"R4 {arm} score failed exact A/B replication")
    expected = {
        "prediction_sha256": sha256_file(predictions_a),
        "prediction_report_sha256": sha256_file(report_a),
        "replica_b_prediction_sha256": sha256_file(predictions_b),
        "replica_b_report_sha256": sha256_file(report_b),
    }
    for key, value in expected.items():
        if integrity.get(key) != value:
            raise ValueError(f"R4 {arm} score does not bind {key}")


def _validate_prediction_reports(
    *,
    arm: str,
    predictions_a: Path,
    report_a: Mapping[str, Any],
    predictions_b: Path,
    report_b: Mapping[str, Any],
) -> None:
    method = ARM_METHODS[arm]
    for replica, predictions, report in (
        ("A", predictions_a, report_a),
        ("B", predictions_b, report_b),
    ):
        if report.get("status") != "complete" or report.get("method") != method:
            raise ValueError(f"R4 {arm} replica {replica} prediction report is incomplete")
        if report.get("replica_id") != replica or report.get("input_mode") != "blank_image":
            raise ValueError(f"R4 {arm} replica {replica} report role drifted")
        if report.get("output_sha256") != sha256_file(predictions):
            raise ValueError(f"R4 {arm} replica {replica} report does not bind predictions")
        if report.get("choice_view_family") != "reverse-cyclic4":
            raise ValueError(f"R4 {arm} replica {replica} choice family drifted")


def _validate_execution(
    *,
    terminal_path: Path | None,
    evidence_path: Path | None,
    stage: str,
    scientific_paths: Sequence[Path],
    strict: bool,
) -> dict[str, Any]:
    if not strict:
        return {"strict": False, "passed": True, "terminal": None, "evidence": None}
    if terminal_path is None or evidence_path is None:
        raise ValueError("Strict R4 reporting requires terminal and evidence artifacts")
    terminal = load_json(terminal_path)
    evidence = load_json(evidence_path)
    if terminal.get("status") != "succeeded" or terminal.get("passed") is not True:
        raise ValueError("R4 terminal artifact is not a successful terminal state")
    if terminal.get("exit_code") != 0:
        raise ValueError("R4 terminal exit code is nonzero")
    if (
        evidence.get("protocol") != STAGE_EVIDENCE_PROTOCOL
        or evidence.get("stage") != stage
        or evidence.get("passed") is not True
    ):
        raise ValueError("R4 evidence protocol/stage/pass binding is invalid")
    if evidence.get("execution_mode") != "sequential_within_arm":
        raise ValueError("R4 evidence does not prove sequential-within-arm execution")
    output_hashes = {
        str(item.get("sha256"))
        for item in evidence.get("outputs", [])
        if isinstance(item, Mapping)
    }
    missing = [str(path) for path in scientific_paths if sha256_file(path) not in output_hashes]
    if missing:
        raise ValueError(f"R4 evidence does not bind report inputs: {missing}")
    terminal_dir = terminal_path.parent
    for name, key in (("stdout.log", "stdout_sha256"), ("stderr.log", "stderr_sha256")):
        path = terminal_dir / name
        if path.is_file() and terminal.get(key) != sha256_file(path):
            raise ValueError(f"R4 terminal does not bind downloaded {name}")
    return {
        "strict": True,
        "passed": True,
        "terminal": {"path": str(terminal_path.resolve()), "sha256": sha256_file(terminal_path)},
        "evidence": {"path": str(evidence_path.resolve()), "sha256": sha256_file(evidence_path)},
    }


def _metric_rows(score: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = score.get("descriptive_metrics", {})
    rows: list[dict[str, Any]] = []
    sections = {
        "condition": metrics.get("conditions", {}),
        "target_position": metrics.get("by_target_position", {}),
        "event_kind": metrics.get("by_event_kind", {}),
        "form": metrics.get("by_form", {}),
        "ood_group": metrics.get("by_ood_group", {}),
    }
    for section, groups in sections.items():
        if not isinstance(groups, Mapping):
            continue
        for name, entry in groups.items():
            if not isinstance(entry, Mapping):
                continue
            rows.append(
                {
                    "section": section,
                    "name": name,
                    "correct": entry.get("correct"),
                    "count": entry.get("count"),
                    "accuracy": entry.get("accuracy"),
                    "accuracy_drop_from_standard": entry.get("accuracy_drop_from_standard"),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row}) or ["message"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows or [{"message": "no rows"}])
    temporary.replace(path)


def _image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_arm_report(
    *,
    stage: str,
    dataset: str,
    arm: str,
    predictions_a: Path,
    prediction_report_a: Path,
    predictions_b: Path,
    prediction_report_b: Path,
    score_path: Path,
    output_dir: Path,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    if arm not in ARM_ORDER:
        raise ValueError(f"Unknown R4 arm: {arm}")
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite R4 arm report directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    plots = output_dir / "plots"
    plots.mkdir()
    sources = output_dir / "sources"
    sources.mkdir()

    rows_a = load_jsonl(predictions_a)
    rows_b = load_jsonl(predictions_b)
    report_a = load_json(prediction_report_a)
    report_b = load_json(prediction_report_b)
    score = load_json(score_path)
    _validate_prediction_reports(
        arm=arm,
        predictions_a=predictions_a,
        report_a=report_a,
        predictions_b=predictions_b,
        report_b=report_b,
    )
    _validate_score_bindings(
        arm=arm,
        score=score,
        predictions_a=predictions_a,
        report_a=prediction_report_a,
        predictions_b=predictions_b,
        report_b=prediction_report_b,
    )
    if len(rows_a) != len(rows_b):
        raise ValueError(f"R4 {arm} A/B prediction counts differ")
    if any(row.get("method") != ARM_METHODS[arm] for row in rows_a + rows_b):
        raise ValueError(f"R4 {arm} predictions contain a foreign method")

    metrics = score.get("descriptive_metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError(f"R4 {arm} score lacks descriptive metrics")
    _configure_plot_style()
    _accuracy_plot(metrics.get("conditions", {}), "Accuracy by intervention", plots / PLOT_NAMES[0])
    _accuracy_plot(
        metrics.get("by_target_position", {}),
        "Standard accuracy by target position",
        plots / PLOT_NAMES[1],
    )
    _accuracy_plot(
        metrics.get("by_event_kind", {}),
        "Standard accuracy by event kind",
        plots / PLOT_NAMES[2],
    )
    _accuracy_plot(metrics.get("by_form", {}), "Standard accuracy by read form", plots / PLOT_NAMES[3])
    _numeric_histogram(rows_a, "nll_margin", "Target-choice NLL margin", plots / PLOT_NAMES[4])
    _numeric_histogram(
        rows_a, "memory_utf8_bytes", "Visible memory size (UTF-8 bytes)", plots / PLOT_NAMES[5]
    )
    _numeric_histogram(
        rows_a, "latency_seconds", "Frozen Reader latency", plots / PLOT_NAMES[6]
    )
    _rotation_swap_plot(metrics, plots / PLOT_NAMES[7])

    copied = [
        _copy_source(predictions_a, sources / "replica-a-predictions.jsonl", "replica A predictions"),
        _copy_source(prediction_report_a, sources / "replica-a-report.json", "replica A report"),
        _copy_source(predictions_b, sources / "replica-b-predictions.jsonl", "replica B predictions"),
        _copy_source(prediction_report_b, sources / "replica-b-report.json", "replica B report"),
        _copy_source(score_path, sources / "score.json", "replicated score"),
    ]
    metric_rows = _metric_rows(score)
    _write_csv(output_dir / "metrics.csv", metric_rows)
    standard = metrics.get("standard", {})
    scientific_gate = score.get("scientific_gate", {})
    payload = {
        "schema": SCHEMA,
        "stage": stage,
        "dataset": dataset,
        "arm": arm,
        "method": ARM_METHODS[arm],
        "passed": True,
        "execution": dict(execution),
        "replication": score.get("replication"),
        "standard": standard,
        "scientific_gate": scientific_gate,
        "performance_role": (
            "data_readability_gate" if arm == "last_effective" else "descriptive_nonblocking"
        ),
        "training_performed": False,
        "loss_curve_available": False,
        "loss_curve_reason": "Frozen Qwen baseline performs inference only; there is no optimizer or training loss.",
        "plots": [
            {"name": name, "sha256": sha256_file(plots / name), "size": (plots / name).stat().st_size}
            for name in PLOT_NAMES
        ],
        "sources": copied,
        "metrics_csv_sha256": sha256_file(output_dir / "metrics.csv"),
    }
    atomic_write(output_dir / "report.json", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    accuracy = standard.get("accuracy") if isinstance(standard, Mapping) else None
    correct = standard.get("correct") if isinstance(standard, Mapping) else None
    count = standard.get("count") if isinstance(standard, Mapping) else None
    markdown = [
        f"# R4 Qwen baseline report — {stage}/{dataset}/{arm}",
        "",
        f"- Method: `{ARM_METHODS[arm]}`",
        f"- Standard accuracy: `{accuracy}` (`{correct}/{count}`)",
        f"- Exact A/B scientific payload: `{score.get('replication', {}).get('bitwise_scientific_payload_match')}`",
        f"- Performance role: `{payload['performance_role']}`",
        f"- Scientific threshold passed: `{scientific_gate.get('passed') if isinstance(scientific_gate, Mapping) else None}`",
        "- Training performed: `false`",
        "- Loss curve: unavailable by design. This is frozen inference, not training.",
        "",
        "Raw/tagged threshold misses are descriptive and never block later stages. Only the locked last-effective gates may block BH0–BH2; BH3 has no accuracy stop gate.",
        "",
        "## Diagnostic figures",
        "",
    ]
    for name in PLOT_NAMES:
        markdown.extend((f"### {name}", "", f"![{name}](plots/{name})", ""))
    atomic_write(output_dir / "report.md", "\n".join(markdown).rstrip() + "\n")

    images = "".join(
        f"<h3>{html.escape(name)}</h3><img alt='{html.escape(name)}' src='{_image_data_uri(plots / name)}'>"
        for name in PLOT_NAMES
    )
    html_payload = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>R4 {html.escape(stage)} {html.escape(dataset)} {html.escape(arm)}</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1100px}}img{{max-width:100%;border:1px solid #ddd}}code{{background:#eee;padding:.1rem .25rem}}</style></head>
<body><h1>R4 Qwen baseline — {html.escape(stage)}/{html.escape(dataset)}/{html.escape(arm)}</h1>
<ul><li>Method: <code>{html.escape(ARM_METHODS[arm])}</code></li><li>Standard accuracy: <code>{html.escape(str(accuracy))}</code> ({correct}/{count})</li>
<li>Exact A/B payload: <code>{html.escape(str(score.get('replication', {}).get('bitwise_scientific_payload_match')))}</code></li>
<li>Training: <code>false</code>; no loss curve exists for this frozen-inference baseline.</li></ul>
<p>Raw/tagged performance thresholds are descriptive. Only locked last-effective gates can block BH0–BH2; BH3 has no accuracy stop gate.</p>{images}</body></html>"""
    atomic_write(output_dir / "report.html", html_payload)
    manifest_files = [
        output_dir / "report.json",
        output_dir / "report.md",
        output_dir / "report.html",
        output_dir / "metrics.csv",
        *(plots / name for name in PLOT_NAMES),
        *(Path(item["report_path"]) for item in copied),
    ]
    manifest = {
        "schema": "vlm.qwen-history-r4-report-manifest.v1",
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_files
        ],
    }
    atomic_write(output_dir / "sha256_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**payload, "output_dir": str(output_dir.resolve())}


def build_combined_report(
    *,
    stage: str,
    dataset: str,
    arm_reports: Mapping[str, Mapping[str, Any]],
    comparison_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite R4 combined report directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    plots = output_dir / "plots"
    plots.mkdir()
    scores = {
        arm: report.get("standard", {})
        for arm, report in arm_reports.items()
    }
    comparison = None
    if comparison_path is not None:
        comparison = load_json(comparison_path)
        if comparison.get("schema") != COMPARISON_SCHEMA or comparison.get("passed") is not True:
            raise ValueError("R4 combined comparison artifact is invalid")
    _configure_plot_style()
    _accuracy_plot(scores, "Standard accuracy by R4 history arm", plots / "standard_accuracy.png")

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    labels: list[str] = []
    differences: list[float] = []
    if comparison is not None:
        values = comparison.get("comparisons", {})
        if isinstance(values, Mapping):
            for name, entry in values.items():
                bootstrap = entry.get("b_minus_a", {}) if isinstance(entry, Mapping) else {}
                value = bootstrap.get("difference") if isinstance(bootstrap, Mapping) else None
                if isinstance(value, (int, float)):
                    labels.append(str(name))
                    differences.append(float(value))
    if differences:
        axis.bar(labels, differences, color="#A5A5A5")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("Paired accuracy difference")
    else:
        _empty(axis, "No combined comparison for this stage")
    axis.set_title("Strictly paired arm differences")
    _save(fig, plots / "paired_differences.png")

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    names = list(ARM_ORDER)
    record_counts = [int(scores.get(arm, {}).get("count", 0) or 0) for arm in names]
    axis.bar(names, record_counts, color="#FFC000")
    axis.set_ylabel("Standard prediction records")
    axis.set_title("Cross-arm paired inventory")
    _save(fig, plots / "paired_inventory.png")

    rows = [
        {
            "arm": arm,
            "method": ARM_METHODS[arm],
            "correct": scores.get(arm, {}).get("correct"),
            "count": scores.get(arm, {}).get("count"),
            "accuracy": scores.get(arm, {}).get("accuracy"),
            "performance_role": arm_reports[arm].get("performance_role"),
        }
        for arm in ARM_ORDER
    ]
    _write_csv(output_dir / "comparison.csv", rows)
    payload = {
        "schema": COMBINED_SCHEMA,
        "stage": stage,
        "dataset": dataset,
        "passed": True,
        "arms": rows,
        "comparison": comparison,
        "comparison_source_sha256": None if comparison_path is None else sha256_file(comparison_path),
        "training_performed": False,
        "loss_curve_available": False,
        "plots": {
            name: sha256_file(plots / name)
            for name in ("standard_accuracy.png", "paired_differences.png", "paired_inventory.png")
        },
    }
    atomic_write(output_dir / "report.json", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    markdown = [
        f"# R4 combined Qwen comparison — {stage}/{dataset}",
        "",
        "| arm | method | correct | count | accuracy | role |",
        "|---|---|---:|---:|---:|---|",
        *[
            f"| {row['arm']} | `{row['method']}` | {row['correct']} | {row['count']} | {row['accuracy']} | {row['performance_role']} |"
            for row in rows
        ],
        "",
        "This is frozen inference; there is no training loss curve. Raw/tagged performance is descriptive, while last-effective is the locked readability control.",
        "",
        "![standard accuracy](plots/standard_accuracy.png)",
        "",
        "![paired differences](plots/paired_differences.png)",
        "",
        "![paired inventory](plots/paired_inventory.png)",
    ]
    atomic_write(output_dir / "report.md", "\n".join(markdown) + "\n")
    image_tags = "".join(
        f"<img alt='{name}' src='{_image_data_uri(plots / name)}'>"
        for name in ("standard_accuracy.png", "paired_differences.png", "paired_inventory.png")
    )
    atomic_write(
        output_dir / "report.html",
        f"<!doctype html><html><head><meta charset='utf-8'><title>R4 combined</title>"
        "<style>body{font-family:system-ui;margin:2rem;max-width:1100px}img{max-width:100%;display:block;margin:1rem 0}</style>"
        f"</head><body><h1>R4 combined — {html.escape(stage)}/{html.escape(dataset)}</h1>"
        "<p>Frozen inference: no training loss curve. Raw/tagged accuracy is descriptive; last-effective is the readability control.</p>"
        f"{image_tags}</body></html>",
    )
    return {**payload, "output_dir": str(output_dir.resolve())}


def render_stage_reports(
    *,
    stage: str,
    dataset: str,
    arm_inputs: Mapping[str, Mapping[str, Path]],
    comparison: Path | None,
    output_dir: Path,
    terminal: Path | None,
    evidence: Path | None,
    strict_execution: bool,
) -> dict[str, Any]:
    if stage not in {"BH0", "BH1", "BH2", "BH3"}:
        raise ValueError("stage must be BH0, BH1, BH2, or BH3")
    if tuple(arm_inputs) != ARM_ORDER:
        raise ValueError(f"arm inputs must be ordered exactly as {list(ARM_ORDER)}")
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite R4 report root: {output_dir}")
    scientific_paths = [
        path
        for values in arm_inputs.values()
        for path in values.values()
    ]
    if comparison is not None:
        scientific_paths.append(comparison)
    execution = _validate_execution(
        terminal_path=terminal,
        evidence_path=evidence,
        stage=stage,
        scientific_paths=scientific_paths,
        strict=strict_execution,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    reports: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        values = arm_inputs[arm]
        reports[arm] = build_arm_report(
            stage=stage,
            dataset=dataset,
            arm=arm,
            predictions_a=values["predictions_a"],
            prediction_report_a=values["report_a"],
            predictions_b=values["predictions_b"],
            prediction_report_b=values["report_b"],
            score_path=values["score"],
            output_dir=output_dir / arm,
            execution=execution,
        )
    combined = build_combined_report(
        stage=stage,
        dataset=dataset,
        arm_reports=reports,
        comparison_path=comparison,
        output_dir=output_dir / "combined",
    )
    summary = {
        "schema": "vlm.qwen-history-r4-report-set.v1",
        "stage": stage,
        "dataset": dataset,
        "passed": True,
        "strict_execution": execution,
        "arm_reports": {arm: reports[arm]["output_dir"] for arm in ARM_ORDER},
        "combined_report": combined["output_dir"],
    }
    atomic_write(output_dir / "report_set.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render audited plot-rich R4 Qwen history reports")
    parser.add_argument("--stage", choices=("BH0", "BH1", "BH2", "BH3"), required=True)
    parser.add_argument("--dataset", required=True)
    for arm in ARM_ORDER:
        prefix = arm.replace("_", "-")
        parser.add_argument(f"--{prefix}-predictions-a", type=Path, required=True)
        parser.add_argument(f"--{prefix}-report-a", type=Path, required=True)
        parser.add_argument(f"--{prefix}-predictions-b", type=Path, required=True)
        parser.add_argument(f"--{prefix}-report-b", type=Path, required=True)
        parser.add_argument(f"--{prefix}-score", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strict-execution", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    arm_inputs: dict[str, dict[str, Path]] = {}
    for arm in ARM_ORDER:
        key = arm
        arm_inputs[arm] = {
            "predictions_a": getattr(args, f"{key}_predictions_a"),
            "report_a": getattr(args, f"{key}_report_a"),
            "predictions_b": getattr(args, f"{key}_predictions_b"),
            "report_b": getattr(args, f"{key}_report_b"),
            "score": getattr(args, f"{key}_score"),
        }
    try:
        result = render_stage_reports(
            stage=args.stage,
            dataset=args.dataset,
            arm_inputs=arm_inputs,
            comparison=args.comparison,
            output_dir=args.output_dir,
            terminal=args.terminal,
            evidence=args.evidence,
            strict_execution=args.strict_execution,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
