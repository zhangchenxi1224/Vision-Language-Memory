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


SCHEMA = "vlm.training-report.v1"
DREAMLITE_METRICS_SCHEMA = "vision_memory.dreamlite-training-metrics.v1"
DREAMLITE_SUMMARY_SCHEMA = "vision_memory.dreamlite-training-summary.v1"
FIGURE_NAMES = (
    "loss_total.png",
    "loss_components.png",
    "gradient_norm_and_clip.png",
    "learning_rate.png",
    "dev_loss.png",
    "memory_throughput.png",
)
KNOWN_SOURCE_FILES = (
    "manifest.json",
    "summary.json",
    "metrics.jsonl",
    "state_gradient_audit.json",
    "distill_diagnostics.json",
    "curriculum.json",
    "environment.txt",
)
MAIN_METRIC_COLUMNS = (
    "kind",
    "optimizer_step",
    "epoch",
    "episode_cursor",
    "loss",
    "qa_loss",
    "state_supervision_loss",
    "latent_distill_loss",
    "image_distill_loss",
    "visual_feature_distill_loss",
    "gradient_norm",
    "group_episode_count",
    "elapsed_seconds",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if isinstance(payload, bytes):
        temporary.write_bytes(payload)
    else:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def load_json_object(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Metric row {path}:{line_number} is not an object")
            rows.append(value)
    return rows


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def nested_get(value: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def git_commit(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def ema(values: Sequence[float], span: int) -> list[float]:
    if span <= 0:
        raise ValueError("EMA span must be positive")
    if not values:
        return []
    alpha = 2.0 / (span + 1.0)
    result = [float(values[0])]
    for value in values[1:]:
        result.append(alpha * float(value) + (1.0 - alpha) * result[-1])
    return result


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10.0, 5.5),
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.5,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="png", bbox_inches="tight", metadata={"Software": f"matplotlib {matplotlib.__version__}"})
    plt.close(fig)


def empty_axis(ax: plt.Axes, message: str) -> None:
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#666666")


def train_rows(metrics: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in metrics if row.get("kind") == "train" and isinstance(row.get("optimizer_step"), int)]


def plot_total_loss(rows: Sequence[Mapping[str, Any]], path: Path, *, ema_span: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    values = [(int(row["optimizer_step"]), finite_number(row.get("loss"))) for row in rows]
    values = [(step, value) for step, value in values if value is not None]
    if not values:
        empty_axis(axes[0], "No train loss rows were produced")
        empty_axis(axes[1], "No positive loss values for log scale")
    else:
        steps = [step for step, _ in values]
        losses = [float(value) for _, value in values]
        smooth = ema(losses, ema_span)
        axes[0].plot(steps, losses, alpha=0.35, label="raw total loss")
        axes[0].plot(steps, smooth, label=f"EMA(span={ema_span})")
        axes[0].set(title="Total training loss", xlabel="optimizer step", ylabel="loss")
        axes[0].legend()
        if all(value > 0 for value in losses):
            axes[1].plot(steps, losses, alpha=0.35, label="raw total loss")
            axes[1].plot(steps, smooth, label=f"EMA(span={ema_span})")
            axes[1].set_yscale("log")
            axes[1].set(title="Total training loss (log scale)", xlabel="optimizer step", ylabel="loss")
            axes[1].legend()
        else:
            empty_axis(axes[1], "Log scale omitted because loss is non-positive")
    save_figure(fig, path)


def plot_loss_components(rows: Sequence[Mapping[str, Any]], path: Path, *, ema_span: int) -> None:
    fig, ax = plt.subplots()
    keys = (
        ("qa_loss", "QA"),
        ("state_supervision_loss", "distill composite"),
        ("latent_distill_loss", "latent"),
        ("image_distill_loss", "image"),
        ("visual_feature_distill_loss", "visual feature"),
    )
    plotted = False
    for key, label in keys:
        values = [
            (int(row["optimizer_step"]), finite_number(row.get(key)))
            for row in rows
            if isinstance(row.get("optimizer_step"), int)
        ]
        values = [(step, value) for step, value in values if value is not None]
        if not values:
            continue
        plotted = True
        steps = [step for step, _ in values]
        losses = [float(value) for _, value in values]
        ax.plot(steps, ema(losses, ema_span), label=f"{label} EMA")
        ax.scatter(steps, losses, s=8, alpha=0.18)
    if plotted:
        ax.set(title="Training loss components", xlabel="optimizer step", ylabel="loss")
        ax.legend(ncol=2)
    else:
        empty_axis(ax, "No loss-component metrics were produced")
    save_figure(fig, path)


def plot_gradients(rows: Sequence[Mapping[str, Any]], path: Path, *, gradient_clip: float | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    values = [(int(row["optimizer_step"]), finite_number(row.get("gradient_norm"))) for row in rows]
    values = [(step, value) for step, value in values if value is not None]
    if not values:
        empty_axis(axes[0], "No gradient-norm rows were produced")
        empty_axis(axes[1], "Clip rate unavailable")
    else:
        steps = [step for step, _ in values]
        gradients = [float(value) for _, value in values]
        axes[0].plot(steps, gradients, label="pre-clip global norm")
        if gradient_clip is not None:
            axes[0].axhline(gradient_clip, color="#c44e52", linestyle="--", label=f"clip={gradient_clip:g}")
        axes[0].set(title="Gradient norm and clipping threshold", xlabel="optimizer step", ylabel="L2 norm")
        axes[0].legend()
        if gradient_clip is None:
            empty_axis(axes[1], "Gradient clip is absent from manifest")
        else:
            cumulative: list[float] = []
            clipped = 0
            for index, value in enumerate(gradients, 1):
                clipped += int(value > gradient_clip)
                cumulative.append(clipped / index)
            axes[1].step(steps, cumulative, where="post")
            axes[1].set_ylim(-0.02, 1.02)
            axes[1].set(title="Cumulative clip-trigger rate", xlabel="optimizer step", ylabel="fraction")
    save_figure(fig, path)


def plot_learning_rate(rows: Sequence[Mapping[str, Any]], path: Path, *, learning_rate: float | None) -> None:
    fig, ax = plt.subplots()
    steps = [int(row["optimizer_step"]) for row in rows if isinstance(row.get("optimizer_step"), int)]
    if learning_rate is None or not steps:
        empty_axis(ax, "Learning rate or optimizer steps unavailable")
    else:
        ax.plot(steps, [learning_rate] * len(steps))
        ax.set(title="Learning-rate schedule", xlabel="optimizer step", ylabel="learning rate")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    save_figure(fig, path)


def plot_dev_loss(metrics: Sequence[Mapping[str, Any]], path: Path) -> None:
    fig, ax = plt.subplots()
    values = [
        (int(row["optimizer_step"]), finite_number(row.get("loss")))
        for row in metrics
        if row.get("kind") == "dev" and isinstance(row.get("optimizer_step"), int)
    ]
    values = [(step, value) for step, value in values if value is not None]
    if not values:
        empty_axis(ax, "No dev evaluations were produced during this training run")
    else:
        ax.plot([step for step, _ in values], [float(value) for _, value in values], marker="o")
        ax.set(title="Development loss", xlabel="optimizer step", ylabel="dev loss")
    save_figure(fig, path)


def plot_memory_throughput(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    summary: Mapping[str, Any] | None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    peak = nested_get(summary, "peak_vram_gib", default={})
    if isinstance(peak, Mapping) and peak:
        labels = [str(key) for key in peak]
        values = [finite_number(peak[key]) or 0.0 for key in peak]
        axes[0].bar(labels, values, color="#4c72b0")
        axes[0].set(title="Peak allocated VRAM", ylabel="GiB")
    else:
        empty_axis(axes[0], "Peak VRAM summary unavailable")

    points: list[tuple[int, float]] = []
    previous_elapsed = 0.0
    for row in rows:
        if not isinstance(row.get("optimizer_step"), int):
            continue
        elapsed = finite_number(row.get("elapsed_seconds"))
        group = finite_number(row.get("group_episode_count"))
        if elapsed is None or group is None or elapsed <= previous_elapsed:
            previous_elapsed = max(previous_elapsed, elapsed or previous_elapsed)
            continue
        points.append((int(row["optimizer_step"]), group / (elapsed - previous_elapsed)))
        previous_elapsed = elapsed
    if points:
        axes[1].plot([step for step, _ in points], [rate for _, rate in points])
        axes[1].set(title="Observed training throughput", xlabel="optimizer step", ylabel="episodes / second")
    else:
        empty_axis(axes[1], "Throughput unavailable from elapsed/group metrics")
    save_figure(fig, path)


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    extras = sorted({key for row in rows for key in row if key not in MAIN_METRIC_COLUMNS})
    fields = list(MAIN_METRIC_COLUMNS) + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_cell(row.get(key)) for key in fields})


def copy_source(path: Path | None, destination: Path, label: str) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    return {
        "label": label,
        "source_path": str(path.resolve()),
        "copied_path": destination.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def markdown_table(rows: Sequence[tuple[str, Any]]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    for key, value in rows:
        escaped = str(value).replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {key} | {escaped} |")
    return "\n".join(lines)


def html_table(rows: Sequence[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>" for key, value in rows
    )
    return f"<table>{body}</table>"


def image_data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def tail_text(path: Path | None, *, lines: int = 80) -> str:
    if path is None or not path.is_file():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def report_status(terminal: Mapping[str, Any] | None, summary: Mapping[str, Any] | None) -> tuple[str, bool | None]:
    if terminal is not None:
        passed = terminal.get("passed")
        return str(terminal.get("status", "unknown")), passed if isinstance(passed, bool) else None
    if summary is not None:
        return "training-complete-without-stage-terminal", None
    return "incomplete-or-failed-before-summary", False


def validate_complete_training_inputs(
    *,
    manifest: Mapping[str, Any] | None,
    summary: Mapping[str, Any] | None,
    terminal: Mapping[str, Any] | None,
    metrics: Sequence[Mapping[str, Any]],
    stage: str,
    stdout_path: Path | None,
    stderr_path: Path | None,
) -> None:
    if manifest is None or summary is None:
        raise ValueError("A strict completed-training report requires manifest.json and summary.json")
    if manifest.get("metrics_schema") != DREAMLITE_METRICS_SCHEMA:
        raise ValueError("Training manifest has an unsupported or missing metrics_schema")
    if manifest.get("summary_schema") != DREAMLITE_SUMMARY_SCHEMA:
        raise ValueError("Training manifest has an unsupported or missing summary_schema")
    if summary.get("schema") != DREAMLITE_SUMMARY_SCHEMA:
        raise ValueError("Training summary has an unsupported or missing schema")
    if terminal is None or terminal.get("passed") is not True or terminal.get("exit_code") != 0:
        raise ValueError("A strict completed-training report requires a successful terminal")
    if manifest.get("git_dirty") is not False:
        raise ValueError("A strict completed-training report requires git_dirty=false")
    commit = manifest.get("git_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("Training manifest git_commit must be a lowercase 40-character SHA1")
    if terminal.get("expected_commit") != commit:
        raise ValueError("Stage terminal expected_commit differs from the training manifest")
    normalize_stage = lambda value: str(value).strip().lower().replace("_", "-")  # noqa: E731
    if normalize_stage(terminal.get("stage")) != normalize_stage(stage):
        raise ValueError("Stage terminal stage differs from the requested report stage")
    for label, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        expected = terminal.get(f"{label}_sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Stage terminal is missing a valid {label}_sha256")
        if path is None or not path.is_file():
            raise ValueError(f"A strict completed-training report requires the terminal-bound {label} log")
        if sha256_file(path) != expected:
            raise ValueError(f"Supplied {label} log does not match the stage terminal SHA256")
    if not metrics:
        raise ValueError("A strict completed-training report requires metrics rows")
    allowed_kinds = {"train", "dev", "resume"}
    unknown = sorted({str(row.get("kind")) for row in metrics if row.get("kind") not in allowed_kinds})
    if unknown:
        raise ValueError(f"Training metrics contain unsupported row kinds: {unknown}")
    for index, row in enumerate(metrics):
        if row.get("schema") != DREAMLITE_METRICS_SCHEMA:
            raise ValueError(f"Metric row {index} has an unsupported or missing schema")
        step = row.get("optimizer_step")
        if not isinstance(step, int) or step < 0:
            raise ValueError(f"Metric row {index} has an invalid optimizer_step")
        for key in (
            "loss",
            "qa_loss",
            "state_supervision_loss",
            "latent_distill_loss",
            "image_distill_loss",
            "visual_feature_distill_loss",
            "gradient_norm",
            "elapsed_seconds",
        ):
            if row.get(key) is not None and finite_number(row.get(key)) is None:
                raise ValueError(f"Metric row {index} has non-finite {key}")
    trains = train_rows(metrics)
    if not trains:
        raise ValueError("A strict completed-training report requires at least one train row")
    steps = [int(row["optimizer_step"]) for row in trains]
    if steps != sorted(set(steps)):
        raise ValueError("Train optimizer steps must be strictly increasing and unique")
    if steps != list(range(steps[0], steps[-1] + 1)):
        raise ValueError("Train optimizer steps must be contiguous within the recorded trajectory")
    final_step = summary.get("optimizer_steps")
    if not isinstance(final_step, int) or final_step != steps[-1]:
        raise ValueError("summary.optimizer_steps must equal the final recorded train step")
    arguments = manifest.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("Training manifest arguments are missing")
    for row in trains:
        if row.get("training_regime") != arguments.get("training_regime"):
            raise ValueError("Train metric training_regime differs from the manifest")
        if row.get("objective_stage") != arguments.get("objective_stage"):
            raise ValueError("Train metric objective_stage differs from the manifest")
        for required in ("loss", "gradient_norm"):
            if finite_number(row.get(required)) is None:
                raise ValueError(f"Train metric is missing finite {required}")
        if arguments.get("objective_stage") == "qa" and finite_number(row.get("qa_loss")) is None:
            raise ValueError("QA train metric is missing finite qa_loss")
        if arguments.get("objective_stage") == "distill":
            for required in (
                "state_supervision_loss",
                "latent_distill_loss",
                "image_distill_loss",
                "visual_feature_distill_loss",
            ):
                if finite_number(row.get(required)) is None:
                    raise ValueError(f"Distillation train metric is missing finite {required}")
    if summary.get("training_regime") != arguments.get("training_regime"):
        raise ValueError("Summary training_regime differs from the manifest")
    if summary.get("objective_stage") != arguments.get("objective_stage"):
        raise ValueError("Summary objective_stage differs from the manifest")


def build_training_report(
    *,
    training_dir: Path,
    output_dir: Path,
    stage: str,
    run_id: str,
    title: str | None = None,
    terminal_path: Path | None = None,
    stage_evidence_path: Path | None = None,
    evaluation_paths: Sequence[Path] = (),
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    ema_span: int = 16,
    status_note: str | None = None,
    strict_complete: bool = False,
) -> dict[str, Any]:
    training_dir = training_dir.resolve(strict=True)
    output_dir = output_dir.resolve()
    if not training_dir.is_dir():
        raise ValueError("--training-dir must be a directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Report output directory must be absent or empty: {output_dir}")
    if ema_span <= 0:
        raise ValueError("--ema-span must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    provenance_dir = output_dir / "provenance"
    logs_dir = output_dir / "logs"
    source_dir = output_dir / "source"
    for directory in (figures_dir, metrics_dir, provenance_dir, logs_dir, source_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = training_dir / "manifest.json"
    summary_path = training_dir / "summary.json"
    metrics_path = training_dir / "metrics.jsonl"
    manifest = load_json_object(manifest_path)
    summary = load_json_object(summary_path)
    terminal = load_json_object(terminal_path)
    stage_evidence = load_json_object(stage_evidence_path)
    evaluations = [(path, load_json_object(path)) for path in evaluation_paths]
    metrics = load_metrics(metrics_path)
    trains = train_rows(metrics)
    if strict_complete:
        validate_complete_training_inputs(
            manifest=manifest,
            summary=summary,
            terminal=terminal,
            metrics=metrics,
            stage=stage,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )

    copied_sources: list[dict[str, Any]] = []
    for name in KNOWN_SOURCE_FILES:
        item = copy_source(training_dir / name, source_dir / name, f"training:{name}")
        if item:
            copied_sources.append(item)
    for source, destination, label in (
        (terminal_path, provenance_dir / "terminal.json", "stage:terminal"),
        (stage_evidence_path, provenance_dir / "stage_evidence.json", "stage:evidence"),
        (stdout_path, logs_dir / "stdout.log", "stage:stdout"),
        (stderr_path, logs_dir / "stderr.log", "stage:stderr"),
    ):
        item = copy_source(source, destination, label)
        if item:
            copied_sources.append(item)
    evaluation_objects: list[dict[str, Any]] = []
    for index, (path, value) in enumerate(evaluations):
        if value is None:
            continue
        destination = metrics_dir / f"evaluation_{index:02d}_{path.name}"
        item = copy_source(path, destination, f"evaluation:{index}")
        if item:
            copied_sources.append(item)
        evaluation_objects.append({"path": str(path.resolve()), "sha256": sha256_file(path), "payload": value})

    write_metrics_csv(metrics_dir / "training_curve.csv", metrics)
    configure_plot_style()
    arguments = nested_get(manifest, "arguments", default={})
    gradient_clip = finite_number(arguments.get("gradient_clip")) if isinstance(arguments, Mapping) else None
    learning_rate = finite_number(arguments.get("learning_rate")) if isinstance(arguments, Mapping) else None
    plot_total_loss(trains, figures_dir / "loss_total.png", ema_span=ema_span)
    plot_loss_components(trains, figures_dir / "loss_components.png", ema_span=ema_span)
    plot_gradients(trains, figures_dir / "gradient_norm_and_clip.png", gradient_clip=gradient_clip)
    plot_learning_rate(trains, figures_dir / "learning_rate.png", learning_rate=learning_rate)
    plot_dev_loss(metrics, figures_dir / "dev_loss.png")
    plot_memory_throughput(trains, figures_dir / "memory_throughput.png", summary=summary)

    status, passed = report_status(terminal, summary)
    generated_at = terminal.get("finished_at") if isinstance(terminal, Mapping) else None
    if not isinstance(generated_at, str) or not generated_at:
        generated_at = utc_now()
    final_loss = next(
        (finite_number(row.get("loss")) for row in reversed(trains) if finite_number(row.get("loss")) is not None),
        None,
    )
    gradients = [finite_number(row.get("gradient_norm")) for row in trains]
    gradients = [value for value in gradients if value is not None]
    clip_count = sum(value > gradient_clip for value in gradients) if gradient_clip is not None else None
    clip_rate = clip_count / len(gradients) if clip_count is not None and gradients else None
    optimizer_steps = nested_get(summary, "optimizer_steps", default=None)
    if optimizer_steps is None and trains:
        optimizer_steps = max(int(row["optimizer_step"]) for row in trains)
    commit = nested_get(manifest, "git_commit", default=None)
    seed = nested_get(manifest, "arguments", "seed", default=None)
    regime = nested_get(manifest, "training_lineage", "training_regime", default=None) or nested_get(
        manifest, "arguments", "training_regime", default=None
    )
    objective_stage = nested_get(manifest, "arguments", "objective_stage", default=None)
    overview_rows = (
        ("Status", status),
        ("Passed", passed),
        ("Stage", stage),
        ("Run ID", run_id),
        ("Training regime", regime),
        ("Objective stage", objective_stage),
        ("Seed", seed),
        ("Git commit", commit),
        ("Optimizer steps", optimizer_steps),
        ("Final train loss", final_loss),
        ("Best dev loss", nested_get(summary, "best_dev_loss", default=None)),
        ("Gradient clip", gradient_clip),
        (
            "Clip-trigger count/rate",
            f"{clip_count}/{len(gradients)} ({clip_rate:.4f})" if clip_rate is not None else "n/a",
        ),
        ("Elapsed seconds", nested_get(summary, "elapsed_seconds", default=None)),
        ("Peak VRAM GiB", json_cell(nested_get(summary, "peak_vram_gib", default=None))),
    )
    warnings: list[str] = []
    if not trains:
        warnings.append("No completed optimizer-step metrics were found.")
    if summary is None:
        warnings.append("summary.json is missing; this may be an early failure or interrupted training run.")
    if terminal is None:
        warnings.append("No stage terminal was supplied; pass/fail state is not cryptographically bound here.")
    if passed is False:
        warnings.append(
            "The supplied terminal marks this training as failed; the report preserves the failure without reinterpretation."
        )
    stderr_tail = tail_text(stderr_path)
    if stderr_tail:
        warnings.append("stderr is non-empty; its tail is included below and the full copied log is preserved.")
    if status_note:
        warnings.append(status_note)

    report_summary = {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "stage": stage,
        "run_id": run_id,
        "title": title or f"Vision-Language Memory training report: {stage}",
        "status": status,
        "passed": passed,
        "training_dir": str(training_dir),
        "optimizer_steps": optimizer_steps,
        "final_train_loss": final_loss,
        "best_dev_loss": nested_get(summary, "best_dev_loss", default=None),
        "gradient_clip": gradient_clip,
        "gradient_clip_count": clip_count,
        "gradient_clip_rate": clip_rate,
        "learning_rate": learning_rate,
        "training_regime": regime,
        "objective_stage": objective_stage,
        "seed": seed,
        "git_commit": commit,
        "model_snapshot_manifests": nested_get(manifest, "model_snapshot_manifests", default=None),
        "reader_resize_contract": nested_get(manifest, "reader_resize_contract", default=None),
        "warnings": warnings,
        "strict_complete": strict_complete,
        "complete": bool(strict_complete and passed is True),
        "figure_files": [f"figures/{name}" for name in FIGURE_NAMES],
        "source_artifacts": copied_sources,
        "evaluation_reports": [{"path": item["path"], "sha256": item["sha256"]} for item in evaluation_objects],
        "generator": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "git_commit": git_commit(Path(__file__).resolve().parents[2]),
            "python": sys.version.split()[0],
            "matplotlib": matplotlib.__version__,
            "ema_span": ema_span,
        },
    }
    atomic_write(metrics_dir / "report_summary.json", json.dumps(report_summary, ensure_ascii=False, indent=2) + "\n")

    markdown = [
        f"# {report_summary['title']}",
        "",
        "## Outcome",
        "",
        markdown_table(overview_rows),
        "",
        "## Warnings and termination notes",
        "",
        *(f"- {warning}" for warning in warnings),
        "",
        "## Training curves",
        "",
    ]
    for name in FIGURE_NAMES:
        markdown.extend((f"### {name}", "", f"![{name}](figures/{name})", ""))
    markdown.extend(
        (
            "## Configuration and lineage",
            "",
            "```json",
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) if manifest else "null",
            "```",
            "",
            "## Training summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) if summary else "null",
            "```",
            "",
            "## Stage evidence",
            "",
            "```json",
            json.dumps(stage_evidence, ensure_ascii=False, indent=2, sort_keys=True) if stage_evidence else "null",
            "```",
            "",
            "## Evaluation reports",
            "",
            "```json",
            json.dumps(evaluation_objects, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "## stderr tail",
            "",
            "```text",
            stderr_tail or "<empty or not supplied>",
            "```",
            "",
        )
    )
    atomic_write(output_dir / "report.md", "\n".join(markdown))

    figure_html = "".join(
        f"<section><h3>{html.escape(name)}</h3><img alt='{html.escape(name)}' src='{image_data_uri(figures_dir / name)}'></section>"
        for name in FIGURE_NAMES
    )
    warning_html = "".join(f"<li>{html.escape(warning)}</li>" for warning in warnings) or "<li>None</li>"
    evaluation_html = html.escape(json.dumps(evaluation_objects, ensure_ascii=False, indent=2, sort_keys=True))
    manifest_html = html.escape(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) if manifest else "null"
    )
    summary_html = html.escape(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) if summary else "null")
    evidence_html = html.escape(
        json.dumps(stage_evidence, ensure_ascii=False, indent=2, sort_keys=True) if stage_evidence else "null"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(report_summary["title"]))}</title>
<style>
body{{font:15px/1.55 system-ui,-apple-system,'Segoe UI',sans-serif;color:#1f2937;max-width:1180px;margin:0 auto;padding:28px;background:#f8fafc}}
h1,h2,h3{{color:#0f172a}} h2{{margin-top:36px;border-bottom:1px solid #cbd5e1;padding-bottom:6px}}
table{{border-collapse:collapse;width:100%;background:white}}th,td{{border:1px solid #cbd5e1;padding:8px;text-align:left;vertical-align:top}}th{{width:26%;background:#f1f5f9}}
section{{background:white;padding:14px;margin:16px 0;border:1px solid #e2e8f0;border-radius:8px}}img{{width:100%;height:auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#0f172a;color:#e2e8f0;padding:14px;border-radius:6px;max-height:680px;overflow:auto}}
.status{{font-weight:700;color:{"#166534" if passed else "#991b1b"}}}.meta{{color:#475569}}
</style></head><body>
<h1>{html.escape(str(report_summary["title"]))}</h1>
<p class="meta">Schema {SCHEMA}; generated {html.escape(report_summary["generated_at"])}; Matplotlib {matplotlib.__version__}; EMA span {ema_span}.</p>
<p class="status">Terminal status: {html.escape(status)}; passed: {html.escape(str(passed))}</p>
<h2>Outcome</h2>{html_table(overview_rows)}
<h2>Warnings and termination notes</h2><ul>{warning_html}</ul>
<h2>Training curves</h2>{figure_html}
<h2>Configuration and lineage</h2><pre>{manifest_html}</pre>
<h2>Training summary</h2><pre>{summary_html}</pre>
<h2>Stage evidence</h2><pre>{evidence_html}</pre>
<h2>Evaluation reports</h2><pre>{evaluation_html}</pre>
<h2>stderr tail</h2><pre>{html.escape(stderr_tail or "<empty or not supplied>")}</pre>
</body></html>"""
    atomic_write(output_dir / "report.html", document)

    provenance = {
        **report_summary,
        "report_root": str(output_dir),
        "files_before_provenance_manifest": sorted(
            path.relative_to(output_dir).as_posix() for path in output_dir.rglob("*") if path.is_file()
        ),
    }
    atomic_write(provenance_dir / "manifest.json", json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    artifact_paths = sorted(
        path for path in output_dir.rglob("*") if path.is_file() and path.name != "artifacts.sha256"
    )
    artifact_lines = [f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}" for path in artifact_paths]
    atomic_write(output_dir / "artifacts.sha256", "\n".join(artifact_lines) + "\n")
    return {
        "schema": SCHEMA,
        "output_dir": str(output_dir),
        "report_html": str(output_dir / "report.html"),
        "report_markdown": str(output_dir / "report.md"),
        "artifact_count": len(artifact_paths),
        "passed": passed,
        "status": status,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an immutable, plot-rich report for one completed or failed training run"
    )
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--title")
    parser.add_argument("--terminal", type=Path)
    parser.add_argument("--stage-evidence", type=Path)
    parser.add_argument("--evaluation-report", type=Path, action="append", default=[])
    parser.add_argument("--stdout-log", type=Path)
    parser.add_argument("--stderr-log", type=Path)
    parser.add_argument("--ema-span", type=int, default=16)
    parser.add_argument("--status-note")
    parser.add_argument("--strict-complete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_training_report(
        training_dir=args.training_dir,
        output_dir=args.output_dir,
        stage=args.stage,
        run_id=args.run_id,
        title=args.title,
        terminal_path=args.terminal,
        stage_evidence_path=args.stage_evidence,
        evaluation_paths=args.evaluation_report,
        stdout_path=args.stdout_log,
        stderr_path=args.stderr_log,
        ema_span=args.ema_span,
        status_note=args.status_note,
        strict_complete=args.strict_complete,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
