from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCHEMA = "vlm.qwen-history-baseline-report.v1"
SCORE_SCHEMA = "vlm.qwen-history-baseline-score.v1"
FIGURE_NAMES = (
    "accuracy_breakdown.png",
    "condition_drops.png",
    "position_rotation.png",
    "nll_margin.png",
    "latency_vs_tokens.png",
    "history_size.png",
    "input_mode_sensitivity.png",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    if not rows:
        raise ValueError("Cannot render a baseline report from an empty prediction file")
    return rows


def atomic_write(path: Path, payload: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if isinstance(payload, bytes):
        temporary.write_bytes(payload)
    else:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, format="png", bbox_inches="tight", metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    plt.close(fig)


def empty_axis(axis: plt.Axes, message: str) -> None:
    axis.set_axis_off()
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes, color="#64748b")


def accuracy_values(groups: Mapping[str, Any] | None) -> tuple[list[str], list[float], list[int]]:
    labels: list[str] = []
    values: list[float] = []
    counts: list[int] = []
    if not isinstance(groups, Mapping):
        return labels, values, counts
    for label, summary in sorted(groups.items()):
        if not isinstance(summary, Mapping) or summary.get("accuracy") is None:
            continue
        labels.append(str(label))
        values.append(float(summary["accuracy"]))
        counts.append(int(summary.get("count", summary.get("n", 0))))
    return labels, values, counts


def plot_accuracy_breakdown(metrics: Mapping[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, field, title in (
        (axes[0], "by_event_kind", "Standard accuracy by event kind"),
        (axes[1], "by_ood_group", "Standard accuracy by OOD group"),
    ):
        labels, values, counts = accuracy_values(metrics.get(field))
        if not values:
            empty_axis(axis, f"No {field} records")
            continue
        bars = axis.bar(labels, values, color="#2563eb")
        axis.set_ylim(0, 1.05)
        axis.set(title=title, ylabel="accuracy")
        axis.tick_params(axis="x", rotation=25)
        for bar, value, count in zip(bars, values, counts, strict=True):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}\nn={count}", ha="center", fontsize=8)
    save_figure(fig, path)


def plot_condition_drops(metrics: Mapping[str, Any], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    paired = metrics.get("paired_conditions")
    if not isinstance(paired, Mapping) or not paired:
        empty_axis(axis, "No paired reset/shuffle/state-swap conditions")
        save_figure(fig, path)
        return
    labels: list[str] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for label, summary in sorted(paired.items()):
        if not isinstance(summary, Mapping):
            continue
        value = float(summary["accuracy_drop"])
        ci = summary.get("bootstrap", {}).get("ci95") if isinstance(summary.get("bootstrap"), Mapping) else None
        low = float(ci[0]) if isinstance(ci, list) and ci[0] is not None else value
        high = float(ci[1]) if isinstance(ci, list) and ci[1] is not None else value
        labels.append(str(label))
        values.append(value)
        lower.append(max(0.0, value - low))
        upper.append(max(0.0, high - value))
    axis.axhline(0, color="#334155", linewidth=1)
    bars = axis.bar(labels, values, color="#dc2626", yerr=[lower, upper], capsize=4)
    axis.set(title="Paired accuracy drop from standard (95% bootstrap CI)", ylabel="standard - intervention")
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:+.1%}", ha="center", va="bottom")
    save_figure(fig, path)


def plot_position_rotation(metrics: Mapping[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    labels, values, counts = accuracy_values(metrics.get("by_target_position"))
    if labels:
        bars = axes[0].bar(labels, values, color="#0f766e")
        axes[0].set(title="Accuracy by target position", xlabel="target index", ylabel="accuracy", ylim=(0, 1.05))
        for bar, value, count in zip(bars, values, counts, strict=True):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}\nn={count}", ha="center", fontsize=8)
    else:
        empty_axis(axes[0], "Position metrics unavailable")
    rotation = metrics.get("rotation")
    if isinstance(rotation, Mapping) and rotation.get("agreement_rate") is not None:
        agreement = float(rotation["agreement_rate"])
        axes[1].bar(["consistent", "inconsistent"], [agreement, 1.0 - agreement], color=["#16a34a", "#f59e0b"])
        axes[1].set(
            title=f"Predicted-text rotation agreement\ncomplete groups={rotation.get('complete_groups', 0)}",
            ylabel="fraction",
            ylim=(0, 1.05),
        )
    else:
        empty_axis(axes[1], "Rotation metrics unavailable")
    save_figure(fig, path)


def target_margin(row: Mapping[str, Any]) -> float:
    scores = [float(value) for value in row["choice_mean_nll"]]
    target = int(row["target_index"])
    return min(value for index, value in enumerate(scores) if index != target) - scores[target]


def plot_nll_margin(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    standard = [row for row in rows if row.get("condition", "standard") == "standard"]
    margins = [target_margin(row) for row in standard]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    if not margins:
        empty_axis(axis, "No standard NLL margins")
    else:
        bins = min(40, max(8, int(math.sqrt(len(margins)))))
        axis.hist(margins, bins=bins, color="#7c3aed", alpha=0.82)
        axis.axvline(0, color="#dc2626", linestyle="--", label="decision boundary")
        axis.axvline(sum(margins) / len(margins), color="#111827", label="mean")
        axis.set(title="Target NLL margin distribution", xlabel="best wrong NLL - target NLL", ylabel="records")
        axis.legend()
    save_figure(fig, path)


def plot_latency_tokens(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    points = [
        (float(row["history_token_count"]), float(row["latency_seconds"]))
        for row in rows
        if row.get("history_token_count") is not None and row.get("latency_seconds") is not None
    ]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    if not points:
        empty_axis(axis, "Latency/token observations unavailable")
    else:
        axis.scatter([point[0] for point in points], [point[1] for point in points], s=18, alpha=0.55, color="#0284c7")
        axis.set(title="Reader latency versus event-history tokens", xlabel="history tokens", ylabel="latency (seconds)")
    save_figure(fig, path)


def plot_history_size(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    points = [
        (float(row["history_token_count"]), float(row["history_utf8_bytes"]))
        for row in rows
        if row.get("history_token_count") is not None and row.get("history_utf8_bytes") is not None
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    if not points:
        empty_axis(axes[0], "History-size observations unavailable")
        empty_axis(axes[1], "History-size observations unavailable")
    else:
        tokens = [point[0] for point in points]
        byte_values = [point[1] for point in points]
        axes[0].hist(tokens, bins=min(30, max(8, int(math.sqrt(len(tokens))))), color="#0891b2")
        axes[0].set(title="Event-history token distribution", xlabel="tokens", ylabel="records")
        axes[1].scatter(tokens, byte_values, s=18, alpha=0.55, color="#ea580c")
        axes[1].set(title="UTF-8 bytes versus tokens", xlabel="tokens", ylabel="UTF-8 bytes")
    save_figure(fig, path)


def plot_input_mode_sensitivity(score: Mapping[str, Any], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    sensitivity = score.get("text_only_sensitivity")
    if not isinstance(sensitivity, Mapping):
        empty_axis(axis, "Text-only is excluded from formal results or was not run for this micro suite")
        save_figure(fig, path)
        return
    blank = sensitivity.get("blank_image")
    text_only = sensitivity.get("text_only")
    if not isinstance(blank, Mapping) or not isinstance(text_only, Mapping):
        empty_axis(axis, "Text-only sensitivity metrics unavailable")
        save_figure(fig, path)
        return
    values = [float(blank["accuracy"]), float(text_only["accuracy"])]
    bars = axis.bar(["blank image\nformal baseline path", "text only\nmicro sensitivity"], values, color=["#2563eb", "#9333ea"])
    axis.set(title="Micro input-mode sensitivity (not two formal baselines)", ylabel="standard accuracy", ylim=(0, 1.05))
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.1%}", ha="center")
    save_figure(fig, path)


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_cell(row.get(field)) for field in fields})


def flatten_group_metrics(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in ("by_condition", "by_event_kind", "by_ood_group", "by_split", "by_target_position", "by_probe_role"):
        groups = metrics.get(field)
        if not isinstance(groups, Mapping):
            continue
        for group, summary in sorted(groups.items()):
            if isinstance(summary, Mapping):
                rows.append({"dimension": field, "group": group, **summary})
    return rows


def flatten_condition_pairs(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    paired = metrics.get("paired_conditions")
    if not isinstance(paired, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for condition, summary in sorted(paired.items()):
        if not isinstance(summary, Mapping):
            continue
        bootstrap = summary.get("bootstrap") if isinstance(summary.get("bootstrap"), Mapping) else {}
        ci = bootstrap.get("ci95", [None, None])
        rows.append(
            {
                "condition": condition,
                **{key: value for key, value in summary.items() if key != "bootstrap"},
                "bootstrap_iterations": bootstrap.get("iterations"),
                "bootstrap_seed": bootstrap.get("seed"),
                "bootstrap_groups": bootstrap.get("groups"),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
            }
        )
    return rows


def copy_source(path: Path | None, destination: Path, *, root: Path, label: str) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return {
        "label": label,
        "source_path": str(path.resolve()),
        "copied_path": destination.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def terminal_bound_evidence_sha(stdout_path: Path) -> str | None:
    for line in reversed(stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and valid_sha256(value.get("evidence_sha256")):
            return str(value["evidence_sha256"])
    return None


def validate_strict_execution(
    *,
    predictions: Path,
    score_report: Path,
    terminal: Mapping[str, Any] | None,
    evidence: Mapping[str, Any] | None,
    evidence_path: Path | None,
    stdout_path: Path | None,
    stderr_path: Path | None,
    additional_artifacts: Sequence[Path] = (),
) -> None:
    if terminal is None or terminal.get("passed") is not True or terminal.get("exit_code") != 0:
        raise ValueError("Strict baseline report requires a successful terminal")
    if evidence is None or evidence.get("passed") is not True:
        raise ValueError("Strict baseline report requires passing stage evidence")
    for label, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        if path is None or not path.is_file() or terminal.get(f"{label}_sha256") != sha256_file(path):
            raise ValueError(f"Strict baseline report requires terminal-bound {label}")
    if evidence_path is None or stdout_path is None:
        raise ValueError("Strict baseline report requires evidence and stdout paths")
    if terminal_bound_evidence_sha(stdout_path) != sha256_file(evidence_path):
        raise ValueError("Stage evidence does not match the SHA256 bound by terminal stdout")
    outputs = evidence.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("Stage evidence outputs are missing")
    output_hashes = {
        item.get("sha256") for item in outputs if isinstance(item, Mapping) and valid_sha256(item.get("sha256"))
    }
    for path in (predictions, score_report, *additional_artifacts):
        if sha256_file(path) not in output_hashes:
            raise ValueError(f"Stage evidence does not bind required baseline artifact: {path.name}")


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def markdown_table(rows: Sequence[tuple[str, Any]]) -> str:
    result = ["| Field | Value |", "| --- | --- |"]
    for label, value in rows:
        escaped = str(value).replace("|", "\\|").replace("\n", "<br>")
        result.append(f"| {label} | {escaped} |")
    return "\n".join(result)


def html_table(rows: Sequence[tuple[str, Any]]) -> str:
    return "<table>" + "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>" for label, value in rows
    ) + "</table>"


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_baseline_report(
    *,
    predictions: Path,
    score_report: Path,
    output_dir: Path,
    stage: str,
    run_id: str,
    prediction_report: Path | None = None,
    replica_b_predictions: Path | None = None,
    replica_b_prediction_report: Path | None = None,
    text_only_predictions: Path | None = None,
    text_only_prediction_report: Path | None = None,
    terminal_path: Path | None = None,
    stage_evidence_path: Path | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    title: str | None = None,
    strict_complete: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Baseline report output directory must be absent or empty; refusing to overwrite")
    rows = load_jsonl(predictions)
    score = load_json(score_report)
    companion = load_json(prediction_report)
    text_only_companion = load_json(text_only_prediction_report)
    terminal = load_json(terminal_path)
    evidence = load_json(stage_evidence_path)
    if score is None or score.get("schema") != SCORE_SCHEMA:
        raise ValueError("Unsupported or missing baseline score report schema")
    if score.get("predictions_sha256") != sha256_file(predictions):
        raise ValueError("Baseline score report does not bind the supplied prediction JSONL")
    if score.get("record_count") != len(rows):
        raise ValueError("Baseline score report record count differs from predictions")
    if companion is not None and companion.get("output_sha256") != sha256_file(predictions):
        raise ValueError("Prediction companion report does not bind the supplied prediction JSONL")
    if (replica_b_predictions is None) != (replica_b_prediction_report is None):
        raise ValueError("Replica B predictions and companion report must be supplied together")
    replica_b_companion = load_json(replica_b_prediction_report)
    if replica_b_predictions is not None:
        replica_b = score.get("replica_b")
        replication = score.get("replication")
        if not isinstance(replica_b, Mapping) or not isinstance(replication, Mapping):
            raise ValueError("Replica B artifacts require a score with replication provenance")
        if replication.get("passed") is not True:
            raise ValueError("Replica B scientific payload did not pass exact replication")
        if replica_b.get("predictions_sha256") != sha256_file(replica_b_predictions):
            raise ValueError("Baseline score does not bind the supplied replica B predictions")
        if replica_b_companion is None or replica_b_companion.get("output_sha256") != sha256_file(
            replica_b_predictions
        ):
            raise ValueError("Replica B companion report does not bind its prediction JSONL")
    if score.get("suite") == "formal" and any(row.get("method") != "qwen_full_event_history" for row in rows):
        raise ValueError("Formal baseline reporting rejects text-only sensitivity or additional methods")
    if (text_only_predictions is None) != (text_only_prediction_report is None):
        raise ValueError("Text-only predictions and companion report must be supplied together")
    sensitivity = score.get("text_only_sensitivity")
    if sensitivity is not None:
        if score.get("suite") == "formal":
            raise ValueError("Formal baseline reporting rejects text-only sensitivity")
        if text_only_predictions is None or text_only_companion is None:
            raise ValueError("A score containing text-only sensitivity requires its raw predictions and report")
        text_input = score.get("text_only_input")
        if not isinstance(text_input, Mapping):
            raise ValueError("Text-only score input provenance is missing")
        if text_input.get("predictions_sha256") != sha256_file(text_only_predictions):
            raise ValueError("Text-only score does not bind the supplied sensitivity predictions")
        if text_only_companion.get("output_sha256") != sha256_file(text_only_predictions):
            raise ValueError("Text-only companion report does not bind its sensitivity predictions")
    elif text_only_predictions is not None:
        raise ValueError("Raw text-only artifacts were supplied to a score without sensitivity results")
    if strict_complete:
        validate_strict_execution(
            predictions=predictions,
            score_report=score_report,
            terminal=terminal,
            evidence=evidence,
            evidence_path=stage_evidence_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            additional_artifacts=tuple(
                path
                for path in (
                    prediction_report,
                    replica_b_predictions,
                    replica_b_prediction_report,
                    text_only_predictions,
                    text_only_prediction_report,
                )
                if path is not None
            ),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    sources_dir = output_dir / "sources"
    provenance_dir = output_dir / "provenance"
    for directory in (figures_dir, metrics_dir, sources_dir, provenance_dir):
        directory.mkdir(parents=True, exist_ok=True)

    configure_plot_style()
    metrics = score.get("descriptive_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Baseline score report is missing descriptive_metrics")
    plot_accuracy_breakdown(metrics, figures_dir / FIGURE_NAMES[0])
    plot_condition_drops(metrics, figures_dir / FIGURE_NAMES[1])
    plot_position_rotation(metrics, figures_dir / FIGURE_NAMES[2])
    plot_nll_margin(rows, figures_dir / FIGURE_NAMES[3])
    plot_latency_tokens(rows, figures_dir / FIGURE_NAMES[4])
    plot_history_size(rows, figures_dir / FIGURE_NAMES[5])
    plot_input_mode_sensitivity(score, figures_dir / FIGURE_NAMES[6])

    write_csv(metrics_dir / "predictions.csv", rows)
    write_csv(metrics_dir / "group_metrics.csv", flatten_group_metrics(metrics))
    write_csv(metrics_dir / "paired_conditions.csv", flatten_condition_pairs(metrics))
    atomic_write(metrics_dir / "score_report.json", json.dumps(score, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    sources: list[dict[str, Any]] = []
    for value in (
        copy_source(predictions, sources_dir / "predictions.jsonl", root=output_dir, label="predictions"),
        copy_source(score_report, sources_dir / "score_report.json", root=output_dir, label="score_report"),
        copy_source(prediction_report, sources_dir / "prediction_report.json", root=output_dir, label="prediction_report"),
        copy_source(
            replica_b_predictions,
            sources_dir / "replica_b_predictions.jsonl",
            root=output_dir,
            label="replica_b_predictions",
        ),
        copy_source(
            replica_b_prediction_report,
            sources_dir / "replica_b_prediction_report.json",
            root=output_dir,
            label="replica_b_prediction_report",
        ),
        copy_source(terminal_path, sources_dir / "terminal.json", root=output_dir, label="terminal"),
        copy_source(stage_evidence_path, sources_dir / "stage_evidence.json", root=output_dir, label="stage_evidence"),
        copy_source(stdout_path, sources_dir / "stdout.log", root=output_dir, label="stdout"),
        copy_source(stderr_path, sources_dir / "stderr.log", root=output_dir, label="stderr"),
        copy_source(
            text_only_predictions,
            sources_dir / "text_only_predictions.jsonl",
            root=output_dir,
            label="text_only_predictions",
        ),
        copy_source(
            text_only_prediction_report,
            sources_dir / "text_only_prediction_report.json",
            root=output_dir,
            label="text_only_prediction_report",
        ),
    ):
        if value is not None:
            sources.append(value)

    standard = metrics.get("standard") if isinstance(metrics.get("standard"), Mapping) else {}
    rotation = metrics.get("rotation") if isinstance(metrics.get("rotation"), Mapping) else {}
    overview = [
        ("Stage", stage),
        ("Run ID", run_id),
        ("Suite", score.get("suite")),
        ("Method", score.get("method")),
        ("Scientific gate passed", score.get("passed")),
        ("Standard accuracy", standard.get("accuracy")),
        ("Standard correct / count", f"{standard.get('correct')} / {standard.get('count')}"),
        ("Rotation agreement", rotation.get("agreement_rate")),
        ("Reader revision", score.get("reader_revision")),
        ("Episodes SHA256", score.get("episodes_sha256")),
        ("Predictions SHA256", score.get("predictions_sha256")),
        (
            "Replica B predictions SHA256",
            score.get("replica_b", {}).get("predictions_sha256")
            if isinstance(score.get("replica_b"), Mapping)
            else None,
        ),
        ("Scientific payload SHA256", score.get("scientific_payload_sha256")),
        ("Strict terminal/evidence validation", strict_complete),
    ]
    sensitivity = score.get("text_only_sensitivity")
    if isinstance(sensitivity, Mapping):
        overview.extend(
            (
                ("Text-only role", sensitivity.get("role")),
                ("Text-only minus blank accuracy", sensitivity.get("text_only_minus_blank_accuracy")),
                ("Blank/text prediction agreement", sensitivity.get("prediction_text_agreement_rate")),
            )
        )
    report_title = title or f"Qwen full-event-history baseline report: {stage}"
    markdown = [
        f"# {report_title}",
        "",
        "## Outcome",
        "",
        markdown_table(overview),
        "",
        "> This is a frozen-Qwen full event-history baseline with a fixed blank image. It has no training loss curve. "
        "Any text-only result is a micro sensitivity analysis, not a second formal baseline.",
        "",
        "## Diagnostic figures",
        "",
    ]
    for name in FIGURE_NAMES:
        markdown.extend((f"### {name}", "", f"![{name}](figures/{name})", ""))
    markdown.extend(
        (
            "## Scientific score",
            "",
            "```json",
            json.dumps(score, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## Stage evidence",
            "",
            "```json",
            json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) if evidence else "null",
            "```",
            "",
            "## stderr tail",
            "",
            "```text",
            "\n".join(stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:])
            if stderr_path is not None and stderr_path.is_file()
            else "<empty or not supplied>",
            "```",
            "",
        )
    )
    atomic_write(output_dir / "report.md", "\n".join(markdown))

    figure_html = "".join(
        f"<section><h3>{html.escape(name)}</h3><img alt='{html.escape(name)}' src='{image_data_uri(figures_dir / name)}'></section>"
        for name in FIGURE_NAMES
    )
    score_html = html.escape(json.dumps(score, ensure_ascii=False, indent=2, sort_keys=True))
    evidence_html = html.escape(json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) if evidence else "null")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(report_title)}</title><style>
body{{font:15px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif;color:#1f2937;max-width:1180px;margin:0 auto;padding:28px;background:#f8fafc}}
h1,h2,h3{{color:#0f172a}}h2{{margin-top:36px;border-bottom:1px solid #cbd5e1;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{border:1px solid #cbd5e1;padding:8px;text-align:left;vertical-align:top}}th{{width:28%;background:#f1f5f9}}
section{{background:#fff;padding:14px;margin:16px 0;border:1px solid #e2e8f0;border-radius:8px}}img{{width:100%;height:auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0f172a;color:#e2e8f0;padding:14px;border-radius:6px;max-height:720px;overflow:auto}}
.note{{background:#fffbeb;border-left:4px solid #f59e0b;padding:12px}}
</style></head><body><h1>{html.escape(report_title)}</h1>
<p>Rendered {html.escape(utc_now())}; schema {SCHEMA}; Matplotlib {matplotlib.__version__}.</p>
<h2>Outcome</h2>{html_table(overview)}
<p class="note">Frozen-Qwen full event history with a fixed blank image has no training loss curve. Text-only is micro sensitivity only.</p>
<h2>Diagnostic figures</h2>{figure_html}
<h2>Scientific score</h2><pre>{score_html}</pre>
<h2>Stage evidence</h2><pre>{evidence_html}</pre>
</body></html>"""
    atomic_write(output_dir / "report.html", document)

    summary = {
        "schema": SCHEMA,
        "stage": stage,
        "run_id": run_id,
        "title": report_title,
        "suite": score.get("suite"),
        "method": score.get("method"),
        "passed": score.get("passed"),
        "standard": standard,
        "rotation_agreement": rotation.get("agreement_rate"),
        "predictions_sha256": score.get("predictions_sha256"),
        "scientific_payload_sha256": score.get("scientific_payload_sha256"),
        "strict_complete": strict_complete,
        "source_artifacts": sources,
        "figures": [f"figures/{name}" for name in FIGURE_NAMES],
        "generator": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "python": sys.version.split()[0],
            "matplotlib": matplotlib.__version__,
        },
    }
    atomic_write(metrics_dir / "report_summary.json", json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    atomic_write(
        provenance_dir / "manifest.json",
        json.dumps(
            {
                **summary,
                "report_root": str(output_dir),
                "files_before_manifest": sorted(
                    path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    artifact_paths = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "artifacts.sha256"
    )
    atomic_write(
        output_dir / "artifacts.sha256",
        "\n".join(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}" for path in artifact_paths) + "\n",
    )
    return {
        "schema": SCHEMA,
        "output_dir": str(output_dir),
        "report_markdown": str(output_dir / "report.md"),
        "report_html": str(output_dir / "report.html"),
        "artifact_count": len(artifact_paths),
        "passed": score.get("passed"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a plot-rich audited Qwen history baseline report")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--score-report", type=Path, required=True)
    parser.add_argument("--prediction-report", type=Path)
    parser.add_argument("--replica-b-predictions", type=Path)
    parser.add_argument("--replica-b-prediction-report", type=Path)
    parser.add_argument("--text-only-predictions", type=Path)
    parser.add_argument("--text-only-prediction-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--title")
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--stage-evidence", type=Path)
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--strict-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_baseline_report(
        predictions=args.predictions,
        score_report=args.score_report,
        prediction_report=args.prediction_report,
        replica_b_predictions=args.replica_b_predictions,
        replica_b_prediction_report=args.replica_b_prediction_report,
        text_only_predictions=args.text_only_predictions,
        text_only_prediction_report=args.text_only_prediction_report,
        output_dir=args.output_dir,
        stage=args.stage,
        run_id=args.run_id,
        title=args.title,
        terminal_path=args.terminal,
        stage_evidence_path=args.stage_evidence,
        stdout_path=args.stdout_log,
        stderr_path=args.stderr_log,
        strict_complete=args.strict_complete,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
