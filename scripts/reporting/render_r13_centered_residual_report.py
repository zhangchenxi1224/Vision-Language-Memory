"""Render a hash-checked R13 centered-residual experiment delivery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SPLITS = ("train_audit", "dev_select", "dev_replay", "dev_final")
SPLIT_COUNTS = {"train_audit": 36, "dev_select": 24, "dev_replay": 24, "dev_final": 24}
CHECKPOINT_STEPS = (0, 288, 576, 864, 1152)
ENDPOINT = "centered_step1152"
MICRO_STEPS = 4608
OPTIMIZER_STEPS = 1152
MICRO_PER_OPTIMIZER = 4
COLORS = {
    "normal": "#2563eb",
    "reset": "#94a3b8",
    "donor": "#dc2626",
    "base": "#d97706",
}

Criterion = tuple[str, str, Callable[[Mapping[str, Any]], bool]]
CRITERIA: tuple[Criterion, ...] = (
    ("m0_ce", "normal CE <= 80% M0", lambda row: float(row["relative_change"]) <= -0.20),
    ("m0_views", "normal improves 4/4 vs M0", lambda row: int(row["improved_choice_views"]) == 4),
    ("m0_acc", "normal accuracy delta >= .25", lambda row: float(row["accuracy_delta"]) >= 0.25),
    (
        "reset_did",
        "normal/reset DiD < 0",
        lambda row: float(row["normal_reset_difference_in_differences"]) < 0.0,
    ),
    (
        "donor_ce",
        "normal CE <= 80% donor",
        lambda row: float(row["normal_vs_donor_relative_change"]) <= -0.20,
    ),
    (
        "donor_views",
        "normal beats donor 4/4",
        lambda row: int(row["normal_better_than_donor_views"]) == 4,
    ),
    (
        "donor_acc",
        "normal-donor accuracy >= .25",
        lambda row: float(row["normal_vs_donor_accuracy_delta"]) >= 0.25,
    ),
    (
        "donor_did",
        "normal/donor DiD < 0",
        lambda row: float(row["normal_donor_difference_in_differences"]) < 0.0,
    ),
    (
        "base_ce",
        "normal CE <= 80% base",
        lambda row: float(row["relative_normal_ce_vs_base"]) <= -0.20,
    ),
    (
        "base_views",
        "normal beats base 4/4",
        lambda row: int(row["normal_better_than_base_views"]) == 4,
    ),
    (
        "base_acc",
        "normal-base accuracy >= .25",
        lambda row: float(row["normal_accuracy_delta_vs_base"]) >= 0.25,
    ),
    (
        "base_did",
        "normal/base DiD < 0",
        lambda row: float(row["normal_base_difference_in_differences"]) < 0.0,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected JSON object list: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSONL objects: {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _moving_mean(values: Sequence[float], window: int) -> list[float]:
    return [
        statistics.fmean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def _inventory_map(source_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    inventory = _load_object(source_root / "artifact_inventory.json")
    entries = inventory.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("R13 artifact inventory has no artifact list")
    by_path = {str(row["path"]): row for row in entries}
    if len(by_path) != len(entries):
        raise ValueError("R13 artifact inventory contains duplicate paths")
    return inventory, by_path


def _verify_inventory_entry(source_root: Path, by_path: Mapping[str, Mapping[str, Any]], relative: str) -> str:
    path = source_root / relative
    if relative not in by_path:
        raise ValueError(f"source file is absent from authoritative inventory: {relative}")
    entry = by_path[relative]
    if path.stat().st_size != int(entry["bytes"]):
        raise ValueError(f"source byte count differs from inventory: {relative}")
    digest = _sha256(path)
    if digest != entry["sha256"]:
        raise ValueError(f"source hash differs from inventory: {relative}")
    return digest


def _critical_paths() -> list[str]:
    paths = [
        "launch.json",
        "terminal.json",
        "stdout.log",
        "stderr.log",
        "run/manifest.json",
        "run/runtime.json",
        "run/environment.txt",
        "run/selection_audit.json",
        "run/schedule_audit.json",
        "run/event_embedding_audit.json",
        "run/feature_audit.json",
        "run/source_decomposition_audit.json",
        "run/model_snapshot_verification_end.json",
        "run/technical_gate.json",
        "run/r13_centered_residual_summary.json",
        "run/micro_metrics.jsonl",
        "run/optimizer_metrics.jsonl",
        "run/images/fixed_base.png",
        "run/images/reset.png",
    ]
    for split in SPLITS:
        paths.extend((f"run/{split}_evaluation_rows.jsonl", f"run/{split}_statistics.json"))
    for step in CHECKPOINT_STEPS:
        paths.append(f"run/checkpoint_diagnostics/step-{step:04d}.json")
    return paths


def _validate_source(source_root: Path) -> dict[str, Any]:
    inventory, by_path = _inventory_map(source_root)
    critical_sha256 = {
        relative: _verify_inventory_entry(source_root, by_path, relative) for relative in _critical_paths()
    }
    run_root = source_root / "run"
    launch = _load_object(source_root / "launch.json")
    terminal = _load_object(source_root / "terminal.json")
    summary = _load_object(run_root / "r13_centered_residual_summary.json")
    technical = _load_object(run_root / "technical_gate.json")
    manifest = _load_object(run_root / "manifest.json")
    if terminal.get("status") != "completed_diagnostic" or not terminal.get("passed"):
        raise ValueError("R13 terminal did not complete cleanly")
    if terminal.get("formal_success_claim") is not False or terminal.get("scientific_arm_gate") is not False:
        raise ValueError("R13 terminal incorrectly claims scientific success")
    if summary.get("status") != "completed" or summary.get("full_success_claim_allowed") is not False:
        raise ValueError("R13 summary status/success boundary drifted")
    if not technical.get("passed") or not summary["gates"]["technical_gate"]:
        raise ValueError("R13 technical gate did not pass")
    if summary["gates"]["arm_gate"] or summary["gates"]["formal_success_gate"]:
        raise ValueError("R13 source unexpectedly claims a scientific pass")
    if summary.get("git_commit") != launch.get("git_commit"):
        raise ValueError("R13 launch/summary commit mismatch")
    if summary.get("config_sha256") != launch.get("config_sha256"):
        raise ValueError("R13 launch/summary config mismatch")
    terminal_bindings = {
        "summary_sha256": "run/r13_centered_residual_summary.json",
        "technical_gate_sha256": "run/technical_gate.json",
        "manifest_sha256": "run/manifest.json",
        "stdout_sha256": "stdout.log",
        "stderr_sha256": "stderr.log",
    }
    for terminal_key, relative in terminal_bindings.items():
        if terminal.get(terminal_key) != critical_sha256[relative]:
            raise ValueError(f"R13 terminal hash binding failed: {terminal_key}")
    micro = _jsonl(run_root / "micro_metrics.jsonl")
    optimizer = _jsonl(run_root / "optimizer_metrics.jsonl")
    if len(micro) != MICRO_STEPS or len(optimizer) != OPTIMIZER_STEPS:
        raise ValueError("R13 fixed training schedule row count drifted")
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_statistics: dict[str, list[dict[str, Any]]] = {}
    derived_pass_counts: dict[str, int] = {}
    for split in SPLITS:
        rows = _jsonl(run_root / f"{split}_evaluation_rows.jsonl")
        values = _load_list(run_root / f"{split}_statistics.json")
        expected = SPLIT_COUNTS[split]
        if len(rows) != expected * 4 * 4 * 2 or len(values) != expected:
            raise ValueError(f"R13 fixed evaluation count drifted: {split}")
        target_ids = {str(row["item_id"]) for row in rows}
        if len(target_ids) != expected:
            raise ValueError(f"R13 evaluation target count drifted: {split}")
        expected_cells = {
            (target, checkpoint, condition, view)
            for target in target_ids
            for checkpoint in ("m0", ENDPOINT)
            for condition in ("normal", "reset", "donor", "base")
            for view in range(4)
        }
        actual_cells = {
            (str(row["item_id"]), str(row["checkpoint"]), str(row["condition"]), int(row["view_index"]))
            for row in rows
        }
        if actual_cells != expected_cells or len(actual_cells) != len(rows):
            raise ValueError(f"R13 evaluation cells are incomplete or duplicated: {split}")
        split_rows[split] = rows
        split_statistics[split] = values
        derived_pass_counts[split] = sum(bool(value["target_gate"]) for value in values)
    if derived_pass_counts != summary["gates"]["split_target_pass_counts"]:
        raise ValueError("R13 summary pass counts do not reproduce from statistics")
    if summary["gates"]["required_target_pass_counts"] != SPLIT_COUNTS:
        raise ValueError("R13 required target counts drifted")
    if sorted(int(path.stem.removeprefix("step-")) for path in (run_root / "checkpoint_diagnostics").glob("*.json")) != list(
        CHECKPOINT_STEPS
    ):
        raise ValueError("R13 checkpoint diagnostic schedule drifted")
    return {
        "inventory": inventory,
        "inventory_sha256": _sha256(source_root / "artifact_inventory.json"),
        "critical_sha256": critical_sha256,
        "launch": launch,
        "terminal": terminal,
        "summary": summary,
        "technical_gate": technical,
        "manifest": manifest,
        "micro": micro,
        "optimizer": optimizer,
        "split_rows": split_rows,
        "split_statistics": split_statistics,
    }


def _training_rows(micro: Sequence[Mapping[str, Any]], optimizer: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in micro:
        grouped[int(row["optimizer_step_zero"]) + 1].append(row)
    optimizer_by_step = {int(row["optimizer_step"]): row for row in optimizer}
    if set(grouped) != set(range(1, OPTIMIZER_STEPS + 1)) or any(
        len(rows) != MICRO_PER_OPTIMIZER for rows in grouped.values()
    ):
        raise ValueError("R13 gradient-accumulation cells drifted")
    rows = []
    for step in range(1, OPTIMIZER_STEPS + 1):
        micro_rows = grouped[step]
        opt = optimizer_by_step[step]
        row = {
            "optimizer_step": step,
            "epoch_zero": int(opt["epoch_zero"]),
            "learning_rate": float(opt["learning_rate"]),
            "ce_mean": statistics.fmean(float(value["ce"]) for value in micro_rows),
            "objective_mean": statistics.fmean(float(value["objective"]) for value in micro_rows),
            "residual_penalty_mean": statistics.fmean(float(value["residual_penalty"]) for value in micro_rows),
            "coefficient_penalty_mean": statistics.fmean(
                float(value["coefficient_penalty"]) for value in micro_rows
            ),
            "residual_coefficient_norm_mean": statistics.fmean(
                float(value["residual_coefficient_norm"]) for value in micro_rows
            ),
            "residual_rms_mean": statistics.fmean(float(value["residual_rms"]) for value in micro_rows),
            "image_saturation_fraction_mean": statistics.fmean(
                float(value["image_saturation_fraction"]) for value in micro_rows
            ),
            "mean_residual_coefficient_max_abs": max(
                float(value["mean_residual_coefficient_max_abs"]) for value in micro_rows
            ),
            "mean_residual_delta_max_abs": max(
                float(value["mean_residual_delta_max_abs"]) for value in micro_rows
            ),
            "gradient_norm": float(opt["gradient_norm"]),
            "gradient_nonzero_fraction": float(opt["gradient_nonzero_fraction"]),
            "basis_parameter_norm": float(opt["basis_parameter_norm"]),
            "gradient_clipped": bool(opt["gradient_clipped"]),
        }
        numeric = [value for key, value in row.items() if key != "gradient_clipped"]
        if row["gradient_clipped"] or any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"R13 invalid optimizer diagnostic at step {step}")
        rows.append(row)
    smooth = _moving_mean([row["ce_mean"] for row in rows], window=16)
    for row, value in zip(rows, smooth, strict=True):
        row["ce_moving_mean_16_optimizer_steps"] = value
    return rows


def _training_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    steps = [int(row["optimizer_step"]) for row in rows]
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 8.5))
    panels = (
        ("ce_moving_mean_16_optimizer_steps", "Reader CE (64-micro moving mean)", "CE", True),
        ("gradient_norm", "Writer gradient norm", "L2 norm", True),
        ("residual_coefficient_norm_mean", "Residual coefficient norm", "coefficient L2", False),
        ("residual_rms_mean", "Latent residual RMS", "latent RMS", False),
        ("image_saturation_fraction_mean", "Decoded image saturation", "fraction", False),
        ("mean_residual_delta_max_abs", "Exact train-mean residual error", "max abs", True),
    )
    for axis, (field, title, ylabel, log_scale) in zip(axes.flat, panels, strict=True):
        axis.plot(steps, [float(row[field]) for row in rows], color="#2563eb", linewidth=1.0)
        for checkpoint in CHECKPOINT_STEPS[1:]:
            axis.axvline(checkpoint, color="black", linewidth=0.5, alpha=0.18)
        if log_scale:
            axis.set_yscale("log")
        axis.set(title=title, xlabel="optimizer step", ylabel=ylabel)
        axis.grid(alpha=0.18)
    figure.suptitle("R13 centered-residual writer: fixed training diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _endpoint_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for split in SPLITS:
        rows = data["split_rows"][split]
        endpoint_rows = [row for row in rows if row["checkpoint"] == ENDPOINT]
        m0 = [row for row in rows if row["checkpoint"] == "m0" and row["condition"] == "normal"]
        result: dict[str, Any] = {
            "split": split,
            "targets": SPLIT_COUNTS[split],
            "target_pass_count": data["summary"]["gates"]["split_target_pass_counts"][split],
            "required_target_pass_count": SPLIT_COUNTS[split],
            "m0_accuracy": statistics.fmean(float(row["correct"]) for row in m0),
            "m0_ce": statistics.fmean(float(row["ce"]) for row in m0),
        }
        for condition in ("normal", "reset", "donor", "base"):
            selected = [row for row in endpoint_rows if row["condition"] == condition]
            result[f"{condition}_accuracy"] = statistics.fmean(float(row["correct"]) for row in selected)
            result[f"{condition}_ce"] = statistics.fmean(float(row["ce"]) for row in selected)
        result["normal_relative_ce_vs_donor"] = result["normal_ce"] / result["donor_ce"] - 1.0
        result["normal_relative_ce_vs_base"] = result["normal_ce"] / result["base_ce"] - 1.0
        result["normal_accuracy_delta_vs_donor"] = result["normal_accuracy"] - result["donor_accuracy"]
        result["normal_accuracy_delta_vs_base"] = result["normal_accuracy"] - result["base_accuracy"]
        output.append(result)
    return output


def _endpoint_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    x = np.arange(len(SPLITS), dtype=np.float64)
    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    width = 0.19
    for index, condition in enumerate(("normal", "reset", "donor", "base")):
        offset = (index - 1.5) * width
        axes[0, 0].bar(
            x + offset,
            [float(row[f"{condition}_accuracy"]) for row in rows],
            width=width,
            color=COLORS[condition],
            label=condition,
        )
        axes[0, 1].bar(
            x + offset,
            [float(row[f"{condition}_ce"]) for row in rows],
            width=width,
            color=COLORS[condition],
            label=condition,
        )
    axes[1, 0].bar(x, [int(row["target_pass_count"]) for row in rows], color="#2563eb")
    axes[1, 0].plot(x, [int(row["required_target_pass_count"]) for row in rows], "k--", label="required")
    axes[1, 1].bar(
        x - width / 2,
        [float(row["normal_accuracy_delta_vs_donor"]) for row in rows],
        width=width,
        color=COLORS["donor"],
        label="normal - donor",
    )
    axes[1, 1].bar(
        x + width / 2,
        [float(row["normal_accuracy_delta_vs_base"]) for row in rows],
        width=width,
        color=COLORS["base"],
        label="normal - base",
    )
    panels = (
        (axes[0, 0], "Endpoint accuracy by causal condition", "accuracy"),
        (axes[0, 1], "Endpoint CE by causal condition", "mean CE"),
        (axes[1, 0], "Targets passing every preregistered gate", "count"),
        (axes[1, 1], "Aggregate accuracy attribution", "accuracy difference"),
    )
    for axis, title, ylabel in panels:
        axis.set(title=title, ylabel=ylabel)
        axis.set_xticks(x, SPLITS, rotation=12)
        axis.grid(axis="y", alpha=0.18)
        axis.legend(fontsize=8)
    axes[0, 1].set_yscale("log")
    axes[1, 1].axhline(0.25, color="black", linestyle="--", linewidth=0.8, label="gate .25")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.6)
    figure.suptitle("R13 fixed endpoint: effectiveness is not causal generalization")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _criterion_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split in SPLITS:
        statistics_rows = data["split_statistics"][split]
        for key, label, predicate in CRITERIA:
            rows.append(
                {
                    "split": split,
                    "criterion": key,
                    "criterion_label": label,
                    "passed_targets": sum(predicate(row) for row in statistics_rows),
                    "targets": len(statistics_rows),
                }
            )
    return rows


def _criterion_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    by_key = {(row["split"], row["criterion"]): row for row in rows}
    x = np.arange(len(CRITERIA), dtype=np.float64)
    labels = [label for _key, label, _predicate in CRITERIA]
    figure, axes = plt.subplots(2, 2, figsize=(17.0, 10.0))
    for axis, split in zip(axes.flat, SPLITS, strict=True):
        values = [int(by_key[(split, key)]["passed_targets"]) for key, _label, _predicate in CRITERIA]
        total = SPLIT_COUNTS[split]
        axis.bar(x, values, color=["#16a34a" if value == total else "#dc2626" for value in values])
        axis.axhline(total, color="black", linestyle="--", linewidth=0.8)
        axis.set(title=split, ylabel=f"targets (of {total})")
        axis.set_xticks(x, labels, rotation=55, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle("R13 per-criterion causal gate counts (green only when every target passes)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _checkpoint_rows(source_root: Path) -> list[dict[str, Any]]:
    rows = []
    for step in CHECKPOINT_STEPS:
        value = _load_object(source_root / "run" / "checkpoint_diagnostics" / f"step-{step:04d}.json")
        for representative in value["rows"]:
            rows.append(
                {
                    "step": step,
                    "split": representative["split"],
                    "segment_id": representative["segment_id"],
                    "residual_coefficient_norm": representative["residual_coefficient_norm"],
                    "residual_rms": representative["residual_rms"],
                    "image_saturation_fraction": representative["image_saturation_fraction"],
                    "basis_norm_min": value["basis_norm_min"],
                    "basis_norm_max": value["basis_norm_max"],
                    "train_mean_residual_coefficient_max_abs": value[
                        "train_mean_residual_coefficient_max_abs"
                    ],
                    "train_mean_residual_delta_max_abs": value["train_mean_residual_delta_max_abs"],
                    "fixed_base_latent_sha256": value["fixed_base_latent_sha256"],
                }
            )
    return rows


def _checkpoint_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    panels = (
        ("residual_coefficient_norm", "Representative residual coefficient norm", False),
        ("residual_rms", "Representative latent residual RMS", False),
        ("image_saturation_fraction", "Representative image saturation", False),
        ("train_mean_residual_delta_max_abs", "Exact train-mean latent residual error", True),
    )
    for axis, (field, title, log_scale) in zip(axes.flat, panels, strict=True):
        means = [
            statistics.fmean(float(row[field]) for row in rows if int(row["step"]) == step)
            for step in CHECKPOINT_STEPS
        ]
        axis.plot(CHECKPOINT_STEPS, means, marker="o", color="#2563eb")
        if log_scale:
            axis.set_yscale("log")
        axis.set(title=title, xlabel="fixed checkpoint", ylabel=field)
        axis.grid(alpha=0.18)
    figure.suptitle("R13 checkpoint trajectory (descriptive; endpoint was fixed in advance)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _image_contact_sheet(path: Path, source_root: Path, data: Mapping[str, Any]) -> list[dict[str, Any]]:
    image_root = source_root / "run" / "images"
    figure, axes = plt.subplots(len(SPLITS) + 1, 2, figsize=(8.0, 3.1 * (len(SPLITS) + 1)))
    for column, (filename, title) in enumerate((("fixed_base.png", "fixed common base"), ("reset.png", "blank reset"))):
        axes[0, column].imshow(plt.imread(image_root / filename))
        axes[0, column].set_title(title)
        axes[0, column].axis("off")
    differences = []
    base = np.asarray(plt.imread(image_root / "fixed_base.png"), dtype=np.float64)[..., :3]
    for row_index, split in enumerate(SPLITS, start=1):
        target = data["split_statistics"][split][0]
        segment_id = str(target["target_segment_id"])
        images = {}
        for column, condition in enumerate(("normal", "donor")):
            image_path = image_root / "endpoint" / split / condition / f"{segment_id}.png"
            pixels = np.asarray(plt.imread(image_path), dtype=np.float64)[..., :3]
            images[condition] = pixels
            axes[row_index, column].imshow(pixels)
            axes[row_index, column].set_title(
                f"{split} · {condition}\nquery target={target['target_value']}"
            )
            axes[row_index, column].axis("off")
        differences.append(
            {
                "split": split,
                "segment_id": segment_id,
                "target_value": target["target_value"],
                "normal_base_pixel_rms": float(np.sqrt(np.mean((images["normal"] - base) ** 2))),
                "donor_base_pixel_rms": float(np.sqrt(np.mean((images["donor"] - base) ** 2))),
                "normal_donor_pixel_rms": float(
                    np.sqrt(np.mean((images["normal"] - images["donor"]) ** 2))
                ),
                "normal_base_mean_abs": float(np.mean(np.abs(images["normal"] - base))),
            }
        )
    figure.suptitle("R13 machine-oriented visual codes: own-event normal vs fixed wrong-event donor")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return differences


def _trajectory_contact_sheet(path: Path, source_root: Path, data: Mapping[str, Any]) -> None:
    target = data["split_statistics"]["train_audit"][0]
    segment_id = str(target["target_segment_id"])
    figure, axes = plt.subplots(1, len(CHECKPOINT_STEPS), figsize=(17.0, 3.7))
    for axis, step in zip(axes, CHECKPOINT_STEPS, strict=True):
        image_path = (
            source_root
            / "run"
            / "images"
            / "trajectory"
            / f"step-{step:04d}"
            / "train_audit"
            / f"{segment_id}.png"
        )
        axis.imshow(plt.imread(image_path))
        axis.set_title(f"step {step}")
        axis.axis("off")
    figure.suptitle(f"R13 fixed-checkpoint visual trajectory · target={target['target_value']}")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _aggregate_criterion_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        split: {
            str(row["criterion"]): int(row["passed_targets"])
            for row in rows
            if row["split"] == split
        }
        for split in SPLITS
    }


def _comparison(r12_analysis: Path, endpoint_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    r12 = _load_object(r12_analysis)
    r12_aggregates = r12["conditioned_aggregates"]
    r13_by_split = {str(row["split"]): row for row in endpoint_rows}
    shared_splits = ("train_audit", "dev_select", "dev_final")
    return {
        "schema": "vision_memory.r12-r13-comparison.v1",
        "status": "completed",
        "formal_success_claim": False,
        "r12_analysis_path": str(r12_analysis.resolve()),
        "r12_analysis_sha256": _sha256(r12_analysis),
        "target_pass_counts": {
            split: {
                "r12_conditioned": int(r12_aggregates[split]["target_pass_count"]),
                "r13_centered_residual": int(r13_by_split[split]["target_pass_count"]),
                "required": SPLIT_COUNTS[split],
            }
            for split in shared_splits
        },
        "interpretation": (
            "R13 converts R12's event-independent visual-prompt collapse into partial target-specific "
            "fit (14/36 train-audit), but it still fails the fixed boundary and does not generalize "
            "causally to held-out entities."
        ),
        "next_preregistered_test": (
            "Explicit symmetric own-versus-wrong-donor ranking during training, with the R13 "
            "normal/reset/donor/base evaluation held unchanged."
        ),
    }


def _delivery_manifest(output_dir: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    files = [
        path for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "DELIVERY_MANIFEST.json"
    ]
    return {
        "schema": "vision_memory.r13-centered-residual-delivery-manifest.v1",
        "status": "completed",
        "formal_success_claim": False,
        "source_sha256": source,
        "artifacts": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files
        ],
    }


def render(
    source_root: Path,
    output_dir: Path,
    r12_analysis: Path,
    remote_source_root: str,
    report_source_archive: str,
    report_source_archive_sha256: str,
    report_source_archive_bytes: int,
) -> dict[str, Any]:
    data = _validate_source(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_rows = _training_rows(data["micro"], data["optimizer"])
    _write_csv(output_dir / "training_metrics.csv", training_rows)
    _training_figure(output_dir / "training_diagnostics.png", training_rows)
    endpoint_rows = _endpoint_rows(data)
    _write_csv(output_dir / "endpoint_summary.csv", endpoint_rows)
    _endpoint_figure(output_dir / "endpoint_metrics.png", endpoint_rows)
    criterion_rows = _criterion_rows(data)
    _write_csv(output_dir / "causal_gate_counts.csv", criterion_rows)
    _criterion_figure(output_dir / "causal_gate_counts.png", criterion_rows)
    checkpoint_rows = _checkpoint_rows(source_root)
    _write_csv(output_dir / "checkpoint_trajectory.csv", checkpoint_rows)
    _checkpoint_figure(output_dir / "checkpoint_trajectory.png", checkpoint_rows)
    visual_differences = _image_contact_sheet(output_dir / "endpoint_image_contact_sheet.png", source_root, data)
    _write_csv(output_dir / "visual_code_difference.csv", visual_differences)
    _trajectory_contact_sheet(output_dir / "latent_trajectory_contact_sheet.png", source_root, data)
    comparison = _comparison(r12_analysis, endpoint_rows)
    _write_json(output_dir / "comparison.json", comparison)

    endpoint_by_split = {str(row["split"]): row for row in endpoint_rows}
    gradient_norms = [float(row["gradient_norm"]) for row in training_rows]
    inventory = data["inventory"]
    inventory_artifacts = inventory["artifacts"]
    source_sha256 = {
        "artifact_inventory": data["inventory_sha256"],
        "manifest": data["critical_sha256"]["run/manifest.json"],
        "summary": data["critical_sha256"]["run/r13_centered_residual_summary.json"],
        "technical_gate": data["critical_sha256"]["run/technical_gate.json"],
        "terminal": data["critical_sha256"]["terminal.json"],
        "report_source_archive": report_source_archive_sha256,
    }
    analysis = {
        "schema": "vision_memory.r13-centered-residual-rendered-analysis.v1",
        "status": "completed",
        "formal_success_claim": False,
        "renderer_sha256": _sha256(Path(__file__).resolve()),
        "git_commit": data["summary"]["git_commit"],
        "config_sha256": data["summary"]["config_sha256"],
        "technical_gate": True,
        "scientific_arm_gate": False,
        "decision": data["summary"]["decision"],
        "split_target_pass_counts": data["summary"]["gates"]["split_target_pass_counts"],
        "required_target_pass_counts": SPLIT_COUNTS,
        "training": {
            "micro_steps": MICRO_STEPS,
            "optimizer_steps": OPTIMIZER_STEPS,
            "first_64_micro_ce": statistics.fmean(float(row["ce"]) for row in data["micro"][:64]),
            "last_64_micro_ce": statistics.fmean(float(row["ce"]) for row in data["micro"][-64:]),
            "minimum_64_micro_ce": min(float(row["ce_moving_mean_16_optimizer_steps"]) for row in training_rows),
            "maximum_gradient_norm": max(gradient_norms),
            "optimizer_steps_gradient_norm_gt_10": sum(value > 10.0 for value in gradient_norms),
            "gradient_clipped_steps": sum(bool(row["gradient_clipped"]) for row in training_rows),
        },
        "endpoint_aggregates": endpoint_by_split,
        "criterion_pass_counts": _aggregate_criterion_counts(criterion_rows),
        "visual_code_representative_pixel_differences": visual_differences,
        "fixed_base_latent_sha256": checkpoint_rows[-1]["fixed_base_latent_sha256"],
        "final_center_constraints": {
            "train_mean_residual_coefficient_max_abs": checkpoint_rows[-1][
                "train_mean_residual_coefficient_max_abs"
            ],
            "train_mean_residual_delta_max_abs": checkpoint_rows[-1]["train_mean_residual_delta_max_abs"],
        },
        "first_principles_interpretation": {
            "what_worked": (
                "The fixed base plus centered conditional residual reached 100% train-audit normal accuracy, "
                "0% reset accuracy, and 14/36 full causal target gates, improving on R12's 0/36."
            ),
            "what_failed": (
                "Wrong-event donor and zero-residual base remained correct too often in train-audit, while "
                "held-out normal accuracy fell to 19.8%-44.8% and the conditional residual often underperformed "
                "the fixed base. The shared event-to-code map is therefore overfit and insufficiently discriminative."
            ),
            "meaning_of_unreadable_images": (
                "The decoded images are measurable high-frequency visual codes without a human-readable semantic "
                "payload. This is compatible with Picture Memory, but it neither proves uniform all-pixel storage "
                "nor establishes causal/generalizable memory by itself."
            ),
            "next_action": comparison["next_preregistered_test"],
        },
        "source_sha256": source_sha256,
    }
    _write_json(output_dir / "ANALYSIS.json", analysis)
    raw_artifacts = {
        "schema": "vision_memory.r13-centered-residual-source-artifacts.v1",
        "authoritative_remote_source_root": remote_source_root,
        "local_report_source_root": str(source_root.resolve()),
        "full_inventory": {
            "artifact_count": len(inventory_artifacts),
            "total_bytes": sum(int(row["bytes"]) for row in inventory_artifacts),
            "inventory_sha256": data["inventory_sha256"],
        },
        "report_source_archive": {
            "remote_path": report_source_archive,
            "bytes": report_source_archive_bytes,
            "sha256": report_source_archive_sha256,
        },
        "critical_source_sha256": data["critical_sha256"],
    }
    _write_json(output_dir / "RAW_ARTIFACTS.json", raw_artifacts)
    _write_json(
        output_dir / "SOURCE_DELIVERY.json",
        {
            "schema": "vision_memory.r13-centered-residual-source-delivery.v1",
            "status": "completed",
            "formal_success_claim": False,
            "source_root": remote_source_root,
            "launch": data["launch"],
            "terminal": data["terminal"],
            "technical_gate": data["technical_gate"],
            "artifact_inventory": inventory,
            "report_source_archive": raw_artifacts["report_source_archive"],
        },
    )

    pass_counts = data["summary"]["gates"]["split_target_pass_counts"]
    report = "\n".join(
        (
            "# R13 mean-centered conditional residual writer: result",
            "",
            "**Decision:** technical pass; scientific diagnostic fail. No formal Picture Memory success is claimed.",
            "",
            "## Fixed causal result",
            "",
            "| split | passed all gates | required | normal accuracy | donor accuracy | base accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *(
                f"| {split} | {pass_counts[split]} | {SPLIT_COUNTS[split]} | "
                f"{endpoint_by_split[split]['normal_accuracy']:.1%} | "
                f"{endpoint_by_split[split]['donor_accuracy']:.1%} | "
                f"{endpoint_by_split[split]['base_accuracy']:.1%} |"
                for split in SPLITS
            ),
            "",
            "## Interpretation",
            "",
            analysis["first_principles_interpretation"]["what_worked"],
            "",
            analysis["first_principles_interpretation"]["what_failed"],
            "",
            analysis["first_principles_interpretation"]["meaning_of_unreadable_images"],
            "",
            "## Locked next experiment",
            "",
            comparison["next_preregistered_test"],
            "",
        )
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    report_zh = "\n".join(
        (
            "# R13 均值中心化条件残差写入器：实验结果",
            "",
            "**判定：技术通过，科学诊断失败；不得宣称 Picture Memory 已成功。**",
            "",
            "## 固定因果结果",
            "",
            "| 数据切分 | 全部门槛通过数 | 要求 | normal 准确率 | donor 准确率 | base 准确率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *(
                f"| {split} | {pass_counts[split]} | {SPLIT_COUNTS[split]} | "
                f"{endpoint_by_split[split]['normal_accuracy']:.1%} | "
                f"{endpoint_by_split[split]['donor_accuracy']:.1%} | "
                f"{endpoint_by_split[split]['base_accuracy']:.1%} |"
                for split in SPLITS
            ),
            "",
            "## 第一性原理结论",
            "",
            "1. R13 的普通 CE 确实学会训练样本：normal 为 100%，reset 为 0%，严格门槛从 R12 的 0/36 提升到 14/36。",
            "2. 但 donor/base 仍经常答对，说明训练图像还没有形成足够强的事件特异判别边界。",
            "3. dev-select/dev-replay/fresh-final 的 normal 仅为 19.8%/42.7%/44.8%，且条件残差常不如固定 base，说明事件到视觉码的共享映射过拟合。",
            "4. 图片在人眼上是不可语义解码的高频纹理，像素差异真实存在；这符合机器可读视觉码设想，但本身不证明记忆因果性、泛化性或全像素均匀承载。",
            "5. 按预注册决策，下一项最小实验是在训练中加入对称 own-vs-wrong-donor 排序，网络、数据和 normal/reset/donor/base 评测全部保持不变。",
            "",
        )
    )
    (output_dir / "REPORT.zh-CN.md").write_text(report_zh, encoding="utf-8")
    conclusion = "\n".join(
        (
            "# R13 first-principles conclusion",
            "",
            "1. Technical validity is established by the clean terminal, exact schedule, frozen snapshots, fixed checkpoints, complete evaluation cells, and hash-bound source inventory.",
            "2. Mean-centering removed the trainable event-independent code and produced partial target-specific fit, so R12's collapse was not an immutable VLM limitation.",
            "3. Low own-image CE is insufficient: target-wise wrong-donor and zero-residual-base interventions remain the causal test.",
            "4. R13 fails that boundary (14/36 train-audit; 1/24, 5/24, and 6/24 held-out), so recurrence and full-scale ID/OOD claims remain premature.",
            "5. The next minimal intervention is symmetric donor-ranking credit assignment with every evaluation gate held fixed.",
            "",
        )
    )
    (output_dir / "FIRST_PRINCIPLES_CONCLUSION.md").write_text(conclusion, encoding="utf-8")
    _write_json(output_dir / "DELIVERY_MANIFEST.json", _delivery_manifest(output_dir, source_sha256))
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r12-analysis", type=Path, required=True)
    parser.add_argument("--remote-source-root", required=True)
    parser.add_argument("--report-source-archive", required=True)
    parser.add_argument("--report-source-archive-sha256", required=True)
    parser.add_argument("--report-source-archive-bytes", type=int, required=True)
    args = parser.parse_args()
    analysis = render(
        source_root=args.source_root,
        output_dir=args.output_dir,
        r12_analysis=args.r12_analysis,
        remote_source_root=args.remote_source_root,
        report_source_archive=args.report_source_archive,
        report_source_archive_sha256=args.report_source_archive_sha256,
        report_source_archive_bytes=args.report_source_archive_bytes,
    )
    print(
        json.dumps(
            {
                "decision": analysis["decision"],
                "split_target_pass_counts": analysis["split_target_pass_counts"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
