"""Render a hash-checked R14 symmetric-donor experiment delivery."""

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
ENDPOINT = "symmetric_step1152"
MICRO_STEPS = 4608
OPTIMIZER_STEPS = 1152
MICRO_PER_OPTIMIZER = 4
EXPECTED_READER_CALLS = 9216
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
    return [statistics.fmean(values[max(0, index - window + 1) : index + 1]) for index in range(len(values))]


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
        "run/pairing_audit.json",
        "run/event_embedding_audit.json",
        "run/feature_audit.json",
        "run/source_decomposition_audit.json",
        "run/model_snapshot_verification_end.json",
        "run/technical_gate.json",
        "run/r14_symmetric_donor_summary.json",
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


def _inventory_map(source_root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    inventory = _load_object(source_root / "artifact_inventory.json")
    entries = inventory.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError("R14 artifact inventory has no artifact list")
    by_path = {str(row["path"]): row for row in entries}
    if len(by_path) != len(entries):
        raise ValueError("R14 artifact inventory contains duplicate paths")
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


def _validate_source(source_root: Path) -> dict[str, Any]:
    inventory, by_path = _inventory_map(source_root)
    critical_sha256 = {
        relative: _verify_inventory_entry(source_root, by_path, relative) for relative in _critical_paths()
    }
    run_root = source_root / "run"
    launch = _load_object(source_root / "launch.json")
    terminal = _load_object(source_root / "terminal.json")
    summary = _load_object(run_root / "r14_symmetric_donor_summary.json")
    technical = _load_object(run_root / "technical_gate.json")
    manifest = _load_object(run_root / "manifest.json")
    pairing = _load_object(run_root / "pairing_audit.json")
    feature = _load_object(run_root / "feature_audit.json")
    if terminal.get("status") != "completed_diagnostic" or not terminal.get("passed"):
        raise ValueError("R14 terminal did not complete cleanly")
    if terminal.get("formal_success_claim") is not False or terminal.get("scientific_arm_gate") is not False:
        raise ValueError("R14 terminal incorrectly claims scientific success")
    if summary.get("status") != "completed" or summary.get("full_success_claim_allowed") is not False:
        raise ValueError("R14 summary status/success boundary drifted")
    if not technical.get("passed") or not summary["gates"]["technical_gate"]:
        raise ValueError("R14 technical gate did not pass")
    if summary["gates"]["arm_gate"] or summary["gates"]["formal_success_gate"]:
        raise ValueError("R14 source unexpectedly claims a scientific pass")
    if summary.get("git_commit") != launch.get("git_commit"):
        raise ValueError("R14 launch/summary commit mismatch")
    if summary.get("config_sha256") != launch.get("config_sha256"):
        raise ValueError("R14 launch/summary config mismatch")
    if pairing.get("pairs_sha256") != launch.get("pairing_sha256"):
        raise ValueError("R14 launch/pairing hash mismatch")
    terminal_bindings = {
        "summary_sha256": "run/r14_symmetric_donor_summary.json",
        "technical_gate_sha256": "run/technical_gate.json",
        "manifest_sha256": "run/manifest.json",
        "stdout_sha256": "stdout.log",
        "stderr_sha256": "stderr.log",
    }
    for terminal_key, relative in terminal_bindings.items():
        if terminal.get(terminal_key) != critical_sha256[relative]:
            raise ValueError(f"R14 terminal hash binding failed: {terminal_key}")
    micro = _jsonl(run_root / "micro_metrics.jsonl")
    optimizer = _jsonl(run_root / "optimizer_metrics.jsonl")
    if len(micro) != MICRO_STEPS or len(optimizer) != OPTIMIZER_STEPS:
        raise ValueError("R14 fixed training schedule row count drifted")
    if sum(int(row["reader_calls"]) for row in micro) != EXPECTED_READER_CALLS:
        raise ValueError("R14 Reader-call accounting drifted")
    if not all(bool(row["same_query_and_permutation_for_own_and_donor"]) for row in micro):
        raise ValueError("R14 own/donor query controls drifted")
    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_statistics: dict[str, list[dict[str, Any]]] = {}
    derived_pass_counts: dict[str, int] = {}
    for split in SPLITS:
        rows = _jsonl(run_root / f"{split}_evaluation_rows.jsonl")
        values = _load_list(run_root / f"{split}_statistics.json")
        expected = SPLIT_COUNTS[split]
        if len(rows) != expected * 4 * 4 * 2 or len(values) != expected:
            raise ValueError(f"R14 fixed evaluation count drifted: {split}")
        target_ids = {str(row["item_id"]) for row in rows}
        expected_cells = {
            (target, checkpoint, condition, view)
            for target in target_ids
            for checkpoint in ("m0", ENDPOINT)
            for condition in ("normal", "reset", "donor", "base")
            for view in range(4)
        }
        actual_cells = {
            (str(row["item_id"]), str(row["checkpoint"]), str(row["condition"]), int(row["view_index"])) for row in rows
        }
        if len(target_ids) != expected or actual_cells != expected_cells or len(actual_cells) != len(rows):
            raise ValueError(f"R14 evaluation cells are incomplete or duplicated: {split}")
        split_rows[split] = rows
        split_statistics[split] = values
        derived_pass_counts[split] = sum(bool(value["target_gate"]) for value in values)
    if derived_pass_counts != summary["gates"]["split_target_pass_counts"]:
        raise ValueError("R14 summary pass counts do not reproduce from statistics")
    if summary["gates"]["required_target_pass_counts"] != SPLIT_COUNTS:
        raise ValueError("R14 required target counts drifted")
    observed_checkpoints = sorted(
        int(path.stem.removeprefix("step-")) for path in (run_root / "checkpoint_diagnostics").glob("*.json")
    )
    if observed_checkpoints != list(CHECKPOINT_STEPS):
        raise ValueError("R14 checkpoint diagnostic schedule drifted")
    checkpoint_inventory = {
        relative: by_path[relative]
        for relative in (f"run/checkpoints/step-{step:04d}.pt" for step in CHECKPOINT_STEPS)
        if relative in by_path
    }
    if len(checkpoint_inventory) != len(CHECKPOINT_STEPS):
        raise ValueError("R14 complete checkpoint inventory is missing")
    return {
        "inventory": inventory,
        "inventory_by_path": by_path,
        "inventory_sha256": _sha256(source_root / "artifact_inventory.json"),
        "critical_sha256": critical_sha256,
        "checkpoint_inventory": checkpoint_inventory,
        "launch": launch,
        "terminal": terminal,
        "summary": summary,
        "technical_gate": technical,
        "manifest": manifest,
        "pairing": pairing,
        "feature": feature,
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
        raise ValueError("R14 gradient-accumulation cells drifted")
    rows = []
    for step in range(1, OPTIMIZER_STEPS + 1):
        micros = grouped[step]
        opt = optimizer_by_step[step]
        row = {
            "optimizer_step": step,
            "epoch_zero": int(opt["epoch_zero"]),
            "learning_rate": float(opt["learning_rate"]),
            "own_ce_mean": statistics.fmean(float(value["own_ce"]) for value in micros),
            "donor_ce_mean": statistics.fmean(float(value["donor_ce"]) for value in micros),
            "ranking_loss_mean": statistics.fmean(float(value["ranking_loss"]) for value in micros),
            "ranking_satisfied_fraction": statistics.fmean(float(bool(value["ranking_satisfied"])) for value in micros),
            "objective_mean": statistics.fmean(float(value["objective"]) for value in micros),
            "residual_coefficient_norm_mean": statistics.fmean(
                float(value["residual_coefficient_norm"]) for value in micros
            ),
            "residual_rms_mean": statistics.fmean(float(value["residual_rms"]) for value in micros),
            "image_saturation_fraction_mean": statistics.fmean(
                float(value["image_saturation_fraction"]) for value in micros
            ),
            "mean_residual_coefficient_max_abs": max(
                float(value["mean_residual_coefficient_max_abs"]) for value in micros
            ),
            "mean_residual_delta_max_abs": max(float(value["mean_residual_delta_max_abs"]) for value in micros),
            "gradient_norm": float(opt["gradient_norm"]),
            "gradient_nonzero_fraction": float(opt["gradient_nonzero_fraction"]),
            "basis_parameter_norm": float(opt["basis_parameter_norm"]),
            "gradient_clipped": bool(opt["gradient_clipped"]),
        }
        numeric = [value for key, value in row.items() if key != "gradient_clipped"]
        if row["gradient_clipped"] or any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError(f"R14 invalid optimizer diagnostic at step {step}")
        rows.append(row)
    for field in ("own_ce_mean", "donor_ce_mean", "ranking_loss_mean", "ranking_satisfied_fraction"):
        smooth = _moving_mean([float(row[field]) for row in rows], window=16)
        for row, value in zip(rows, smooth, strict=True):
            row[f"{field}_moving_mean_16"] = value
    return rows


def _training_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    steps = [int(row["optimizer_step"]) for row in rows]
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 8.5))
    axes[0, 0].plot(steps, [row["own_ce_mean_moving_mean_16"] for row in rows], label="own CE")
    axes[0, 0].plot(steps, [row["donor_ce_mean_moving_mean_16"] for row in rows], label="donor CE")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set(title="Reader CE (64-micro moving mean)", ylabel="CE")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, [row["ranking_loss_mean_moving_mean_16"] for row in rows], label="ranking loss")
    rank_axis = axes[0, 1].twinx()
    rank_axis.plot(
        steps,
        [row["ranking_satisfied_fraction_moving_mean_16"] for row in rows],
        color="#16a34a",
        alpha=0.8,
        label="satisfied",
    )
    axes[0, 1].set(title="Own-vs-donor ranking", ylabel="hinge loss")
    rank_axis.set_ylabel("satisfied fraction")
    rank_axis.set_ylim(-0.02, 1.02)
    panels = (
        (axes[0, 2], "gradient_norm", "Writer gradient norm", True),
        (axes[1, 0], "residual_coefficient_norm_mean", "Residual coefficient norm", False),
        (axes[1, 1], "image_saturation_fraction_mean", "Decoded image saturation", False),
        (axes[1, 2], "mean_residual_delta_max_abs", "Exact train-mean residual error", True),
    )
    for axis, field, title, log_scale in panels:
        axis.plot(steps, [float(row[field]) for row in rows], color="#2563eb", linewidth=1.0)
        if log_scale:
            axis.set_yscale("log")
        axis.set(title=title, ylabel=field)
    for axis in axes.flat:
        for checkpoint in CHECKPOINT_STEPS[1:]:
            axis.axvline(checkpoint, color="black", linewidth=0.5, alpha=0.16)
        axis.set_xlabel("optimizer step")
        axis.grid(alpha=0.18)
    figure.suptitle("R14 symmetric fixed-donor ranking: training diagnostics")
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
            x + offset, [row[f"{condition}_accuracy"] for row in rows], width, color=COLORS[condition], label=condition
        )
        axes[0, 1].bar(
            x + offset, [row[f"{condition}_ce"] for row in rows], width, color=COLORS[condition], label=condition
        )
    axes[1, 0].bar(x, [row["target_pass_count"] for row in rows], color="#2563eb")
    axes[1, 0].plot(x, [row["required_target_pass_count"] for row in rows], "k--", label="required")
    axes[1, 1].bar(
        x - width / 2,
        [row["normal_accuracy_delta_vs_donor"] for row in rows],
        width,
        color=COLORS["donor"],
        label="normal - donor",
    )
    axes[1, 1].bar(
        x + width / 2,
        [row["normal_accuracy_delta_vs_base"] for row in rows],
        width,
        color=COLORS["base"],
        label="normal - base",
    )
    for axis, title, ylabel in (
        (axes[0, 0], "Endpoint accuracy by causal condition", "accuracy"),
        (axes[0, 1], "Endpoint CE by causal condition", "mean CE"),
        (axes[1, 0], "Targets passing every preregistered gate", "count"),
        (axes[1, 1], "Aggregate accuracy attribution", "accuracy difference"),
    ):
        axis.set(title=title, ylabel=ylabel)
        axis.set_xticks(x, SPLITS, rotation=12)
        axis.grid(axis="y", alpha=0.18)
        axis.legend(fontsize=8)
    axes[0, 1].set_yscale("log")
    axes[1, 1].axhline(0.25, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].axhline(0.0, color="black", linewidth=0.6)
    figure.suptitle("R14 fixed endpoint: technical validity without causal generalization")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _criterion_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "criterion": key,
            "criterion_label": label,
            "passed_targets": sum(predicate(row) for row in data["split_statistics"][split]),
            "targets": len(data["split_statistics"][split]),
        }
        for split in SPLITS
        for key, label, predicate in CRITERIA
    ]


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
    figure.suptitle("R14 per-criterion causal gates (green only when every target passes)")
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
                    "train_mean_residual_coefficient_max_abs": value["train_mean_residual_coefficient_max_abs"],
                    "train_mean_residual_delta_max_abs": value["train_mean_residual_delta_max_abs"],
                    "fixed_base_latent_sha256": value["fixed_base_latent_sha256"],
                }
            )
    return rows


def _checkpoint_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    for axis, field, title, log_scale in (
        (axes[0, 0], "residual_coefficient_norm", "Representative residual coefficient norm", False),
        (axes[0, 1], "residual_rms", "Representative latent residual RMS", False),
        (axes[1, 0], "image_saturation_fraction", "Representative image saturation", False),
        (axes[1, 1], "train_mean_residual_delta_max_abs", "Exact train-mean latent residual error", True),
    ):
        means = [
            statistics.fmean(float(row[field]) for row in rows if int(row["step"]) == step) for step in CHECKPOINT_STEPS
        ]
        axis.plot(CHECKPOINT_STEPS, means, marker="o", color="#2563eb")
        if log_scale:
            axis.set_yscale("log")
        axis.set(title=title, xlabel="fixed checkpoint", ylabel=field)
        axis.grid(alpha=0.18)
    figure.suptitle("R14 fixed-checkpoint trajectory (descriptive, never selected post hoc)")
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
            axes[row_index, column].set_title(f"{split} · {condition}\nquery target={target['target_value']}")
            axes[row_index, column].axis("off")
        differences.append(
            {
                "split": split,
                "segment_id": segment_id,
                "target_value": target["target_value"],
                "normal_base_pixel_rms": float(np.sqrt(np.mean((images["normal"] - base) ** 2))),
                "donor_base_pixel_rms": float(np.sqrt(np.mean((images["donor"] - base) ** 2))),
                "normal_donor_pixel_rms": float(np.sqrt(np.mean((images["normal"] - images["donor"]) ** 2))),
                "normal_base_mean_abs": float(np.mean(np.abs(images["normal"] - base))),
            }
        )
    figure.suptitle("R14 machine-oriented visual codes: own-event normal vs wrong-event donor")
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
            source_root / "run" / "images" / "trajectory" / f"step-{step:04d}" / "train_audit" / f"{segment_id}.png"
        )
        axis.imshow(plt.imread(image_path))
        axis.set_title(f"step {step}")
        axis.axis("off")
    figure.suptitle(f"R14 fixed-checkpoint visual trajectory · target={target['target_value']}")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _pair_update_lag_rows(micro: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in micro:
        left, right = sorted((str(row["segment_id"]), str(row["donor_segment_id"])))
        grouped[(int(row["epoch_zero"]), left, right)].append(row)
    expected_instances = 32 * 72
    if len(grouped) != expected_instances:
        raise ValueError("R14 pair/epoch count drifted")
    output = []
    for (epoch, left, right), rows in sorted(grouped.items()):
        if len(rows) != 2 or {str(row["segment_id"]) for row in rows} != {left, right}:
            raise ValueError("R14 pair directions are incomplete")
        steps = sorted(int(row["optimizer_step_zero"]) for row in rows)
        lag = steps[1] - steps[0]
        output.append(
            {
                "epoch_zero": epoch,
                "left_segment_id": left,
                "right_segment_id": right,
                "first_optimizer_step_zero": steps[0],
                "second_optimizer_step_zero": steps[1],
                "optimizer_step_lag": lag,
                "same_optimizer_update": lag == 0,
            }
        )
    lags = [int(row["optimizer_step_lag"]) for row in output]
    same = sum(lag == 0 for lag in lags)
    audit = {
        "schema": "vision_memory.r14-pair-update-lag-audit.v1",
        "pair_epoch_instances": len(lags),
        "same_optimizer_update_count": same,
        "same_optimizer_update_fraction": same / len(lags),
        "mean_optimizer_step_lag": statistics.fmean(lags),
        "median_optimizer_step_lag": statistics.median(lags),
        "p90_optimizer_step_lag": float(np.percentile(lags, 90)),
        "maximum_optimizer_step_lag": max(lags),
    }
    return output, audit


def _pair_update_lag_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lags = [int(row["optimizer_step_lag"]) for row in rows]
    figure, axis = plt.subplots(figsize=(9.0, 5.0))
    axis.hist(lags, bins=np.arange(-0.5, max(lags) + 1.5, 1.0), color="#2563eb")
    axis.axvline(
        statistics.median(lags), color="#dc2626", linestyle="--", label=f"median={statistics.median(lags):.0f}"
    )
    axis.set(
        title="R14 paired directions were rarely optimized together",
        xlabel="optimizer-step lag within pair/epoch",
        ylabel="pair-epoch instances",
    )
    axis.grid(axis="y", alpha=0.18)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _negative_coverage_rows(data: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_donors: dict[str, set[str]] = defaultdict(set)
    target_by_id: dict[str, str] = {}
    for row in data["micro"]:
        segment_id = str(row["segment_id"])
        donor_id = str(row["donor_segment_id"])
        training_donors[segment_id].add(donor_id)
        target_by_id[segment_id] = str(row["target_value"])
        target_by_id[donor_id] = str(row["donor_target_value"])
    endpoint_donor_rows = [
        row
        for row in data["split_rows"]["train_audit"]
        if row["checkpoint"] == ENDPOINT and row["condition"] == "donor"
    ]
    by_item: dict[str, set[str]] = defaultdict(set)
    for row in endpoint_donor_rows:
        by_item[str(row["item_id"])].add(str(row["donor_item_id"]))
    output = []
    for item_id, eval_donors in sorted(by_item.items()):
        if len(eval_donors) != 1:
            raise ValueError("R14 train-audit donor intervention is not fixed across views")
        eval_donor = next(iter(eval_donors))
        trained = training_donors[item_id]
        trained_values = {target_by_id[value] for value in trained}
        eval_value = target_by_id[eval_donor]
        output.append(
            {
                "item_id": item_id,
                "target_value": target_by_id[item_id],
                "training_donor_count": len(trained),
                "training_donor_target_value_count": len(trained_values),
                "evaluation_donor_item_id": eval_donor,
                "evaluation_donor_target_value": eval_value,
                "exact_donor_seen_in_training": eval_donor in trained,
                "donor_target_value_seen_in_training": eval_value in trained_values,
            }
        )
    audit = {
        "schema": "vision_memory.r14-negative-coverage-audit.v1",
        "train_event_count": len(training_donors),
        "train_audit_relation_count": len(output),
        "minimum_unique_training_donor_items_per_event": min(len(value) for value in training_donors.values()),
        "maximum_unique_training_donor_items_per_event": max(len(value) for value in training_donors.values()),
        "exact_evaluation_donor_overlap_count": sum(bool(row["exact_donor_seen_in_training"]) for row in output),
        "evaluation_donor_target_value_overlap_count": sum(
            bool(row["donor_target_value_seen_in_training"]) for row in output
        ),
    }
    return output, audit


def _feature_audit(feature: Mapping[str, Any]) -> dict[str, Any]:
    records = feature["records"]
    entropies = [float(row["normalized_attention_entropy"]) for row in records]
    maxima = [float(row["attention_max"]) for row in records]
    return {
        "schema": "vision_memory.r14-frozen-feature-pooling-audit.v1",
        "feature_count": len(records),
        "normalized_attention_entropy": {
            "minimum": min(entropies),
            "mean": statistics.fmean(entropies),
            "maximum": max(entropies),
        },
        "attention_max": {"minimum": min(maxima), "mean": statistics.fmean(maxima), "maximum": max(maxima)},
        "interpretation": "Frozen token pooling is nearly uniform; this is a secondary representation hypothesis, not a causal result.",
    }


def _aggregate_criterion_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        split: {str(row["criterion"]): int(row["passed_targets"]) for row in rows if row["split"] == split}
        for split in SPLITS
    }


def _comparison(r13_analysis: Path, endpoint_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    r13 = _load_object(r13_analysis)
    r14 = {str(row["split"]): row for row in endpoint_rows}
    pass_counts = {
        split: {
            "r13_centered_residual": int(r13["split_target_pass_counts"][split]),
            "r14_symmetric_fixed_donor": int(r14[split]["target_pass_count"]),
            "delta": int(r14[split]["target_pass_count"]) - int(r13["split_target_pass_counts"][split]),
            "required": SPLIT_COUNTS[split],
        }
        for split in SPLITS
    }
    return {
        "schema": "vision_memory.r13-r14-comparison.v1",
        "status": "completed",
        "formal_success_claim": False,
        "r13_analysis_path": str(r13_analysis.resolve()),
        "r13_analysis_sha256": _sha256(r13_analysis),
        "target_pass_counts": pass_counts,
        "interpretation": "R14 changed only 3/-0/-1/+1 target gates versus R13. Fixed-donor ranking produced no robust held-out improvement and did not solve shared causal generalization.",
        "next_preregistered_test": "Use update-synchronous bidirectional pair loss and deterministic rotating negatives while preserving the R14 writer, Reader-call budget, optimizer, endpoint, data, and normal/reset/donor/base gates.",
    }


def _delivery_manifest(output_dir: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    files = [path for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "DELIVERY_MANIFEST.json"]
    return {
        "schema": "vision_memory.r14-symmetric-donor-delivery-manifest.v1",
        "status": "completed",
        "formal_success_claim": False,
        "source_sha256": source,
        "artifacts": [{"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files],
    }


def render(
    source_root: Path,
    output_dir: Path,
    r13_analysis: Path,
    remote_source_root: str,
    report_source_archive: str,
    report_source_archive_sha256: str,
    report_source_archive_bytes: int,
    complete_archive: str,
    complete_archive_sha256: str,
    complete_archive_bytes: int,
) -> dict[str, Any]:
    data = _validate_source(source_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_rows = _training_rows(data["micro"], data["optimizer"])
    _write_csv(output_dir / "training_metrics.csv", training_rows)
    _training_figure(output_dir / "training_diagnostics.png", training_rows)
    endpoint_rows = _endpoint_rows(data)
    endpoint_by_split = {str(row["split"]): row for row in endpoint_rows}
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
    pair_rows, pair_audit = _pair_update_lag_rows(data["micro"])
    _write_csv(output_dir / "pair_update_lag.csv", pair_rows)
    _write_json(output_dir / "pair_update_lag_audit.json", pair_audit)
    _pair_update_lag_figure(output_dir / "pair_update_lag.png", pair_rows)
    coverage_rows, coverage_audit = _negative_coverage_rows(data)
    _write_csv(output_dir / "negative_coverage.csv", coverage_rows)
    _write_json(output_dir / "negative_coverage_audit.json", coverage_audit)
    feature_audit = _feature_audit(data["feature"])
    _write_json(output_dir / "feature_pooling_audit.json", feature_audit)
    comparison = _comparison(r13_analysis, endpoint_rows)
    _write_json(output_dir / "comparison.json", comparison)

    inventory = data["inventory"]
    inventory_artifacts = inventory["artifacts"]
    source_sha256 = {
        "artifact_inventory": data["inventory_sha256"],
        "manifest": data["critical_sha256"]["run/manifest.json"],
        "summary": data["critical_sha256"]["run/r14_symmetric_donor_summary.json"],
        "technical_gate": data["critical_sha256"]["run/technical_gate.json"],
        "terminal": data["critical_sha256"]["terminal.json"],
        "report_source_archive": report_source_archive_sha256,
        "complete_archive": complete_archive_sha256,
    }
    gradient_norms = [float(row["gradient_norm"]) for row in training_rows]
    analysis = {
        "schema": "vision_memory.r14-symmetric-donor-rendered-analysis.v1",
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
            "reader_calls": sum(int(row["reader_calls"]) for row in data["micro"]),
            "first_64_mean_own_ce": statistics.fmean(float(row["own_ce"]) for row in data["micro"][:64]),
            "last_64_mean_own_ce": statistics.fmean(float(row["own_ce"]) for row in data["micro"][-64:]),
            "first_64_mean_donor_ce": statistics.fmean(float(row["donor_ce"]) for row in data["micro"][:64]),
            "last_64_mean_donor_ce": statistics.fmean(float(row["donor_ce"]) for row in data["micro"][-64:]),
            "first_64_ranking_satisfied_fraction": statistics.fmean(
                float(bool(row["ranking_satisfied"])) for row in data["micro"][:64]
            ),
            "last_64_ranking_satisfied_fraction": statistics.fmean(
                float(bool(row["ranking_satisfied"])) for row in data["micro"][-64:]
            ),
            "all_micro_ranking_satisfied_fraction": statistics.fmean(
                float(bool(row["ranking_satisfied"])) for row in data["micro"]
            ),
            "maximum_gradient_norm": max(gradient_norms),
            "optimizer_steps_gradient_norm_gt_10": sum(value > 10.0 for value in gradient_norms),
            "gradient_clipped_steps": sum(bool(row["gradient_clipped"]) for row in training_rows),
        },
        "endpoint_aggregates": endpoint_by_split,
        "criterion_pass_counts": _aggregate_criterion_counts(criterion_rows),
        "pair_update_lag_audit": pair_audit,
        "negative_coverage_audit": coverage_audit,
        "feature_pooling_audit": feature_audit,
        "visual_code_representative_pixel_differences": visual_differences,
        "fixed_base_latent_sha256": checkpoint_rows[-1]["fixed_base_latent_sha256"],
        "final_center_constraints": {
            "train_mean_residual_coefficient_max_abs": checkpoint_rows[-1]["train_mean_residual_coefficient_max_abs"],
            "train_mean_residual_delta_max_abs": checkpoint_rows[-1]["train_mean_residual_delta_max_abs"],
        },
        "first_principles_interpretation": {
            "what_worked": "The objective learned the fixed training contrast: last-64 own CE fell while donor CE rose and 60.9% of late micro-steps satisfied the ln(4) margin.",
            "what_failed": "Only 17/36 train-audit and 1/24, 4/24, 7/24 held-out targets passed every causal gate. This is a scientific failure, not a full Picture Memory result.",
            "identified_mechanism": "The objective was symmetric only over an epoch: just 44/2304 pair-epoch directions shared an optimizer update, with median lag 11. Each event also saw one fixed negative, while 0/36 train-audit evaluation donor identities and 0/36 donor target values overlapped that negative.",
            "meaning_of_unreadable_images": "Noise-like images remain valid machine-oriented visual codes, but visual appearance alone is neither success nor failure; causal attribution and held-out generalization decide the claim.",
            "next_action": comparison["next_preregistered_test"],
        },
        "source_sha256": source_sha256,
    }
    _write_json(output_dir / "ANALYSIS.json", analysis)
    raw_artifacts = {
        "schema": "vision_memory.r14-symmetric-donor-source-artifacts.v1",
        "authoritative_remote_source_root": remote_source_root,
        "local_report_source_root": str(source_root.resolve()),
        "full_inventory": {
            "artifact_count": len(inventory_artifacts),
            "total_bytes": sum(int(row["bytes"]) for row in inventory_artifacts),
            "inventory_sha256": data["inventory_sha256"],
        },
        "checkpoint_inventory": data["checkpoint_inventory"],
        "report_source_archive": {
            "remote_path": report_source_archive,
            "bytes": report_source_archive_bytes,
            "sha256": report_source_archive_sha256,
        },
        "complete_archive": {
            "remote_path": complete_archive,
            "bytes": complete_archive_bytes,
            "sha256": complete_archive_sha256,
        },
        "critical_source_sha256": data["critical_sha256"],
    }
    _write_json(output_dir / "RAW_ARTIFACTS.json", raw_artifacts)
    _write_json(
        output_dir / "SOURCE_DELIVERY.json",
        {
            "schema": "vision_memory.r14-symmetric-donor-source-delivery.v1",
            "status": "completed",
            "formal_success_claim": False,
            "source_root": remote_source_root,
            "launch": data["launch"],
            "terminal": data["terminal"],
            "technical_gate": data["technical_gate"],
            "artifact_inventory": inventory,
            "archives": {
                "report_source": raw_artifacts["report_source_archive"],
                "complete": raw_artifacts["complete_archive"],
            },
        },
    )
    pass_counts = analysis["split_target_pass_counts"]
    table = tuple(
        f"| {split} | {pass_counts[split]} | {SPLIT_COUNTS[split]} | {endpoint_by_split[split]['normal_accuracy']:.1%} | {endpoint_by_split[split]['donor_accuracy']:.1%} | {endpoint_by_split[split]['base_accuracy']:.1%} |"
        for split in SPLITS
    )
    (output_dir / "REPORT.md").write_text(
        "\n".join(
            (
                "# R14 symmetric fixed-donor ranking: result",
                "",
                "**Decision: technical pass; scientific diagnostic fail. No formal Picture Memory success is claimed.**",
                "",
                "| split | passed all gates | required | normal accuracy | donor accuracy | base accuracy |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *table,
                "",
                "## First-principles diagnosis",
                "",
                analysis["first_principles_interpretation"]["what_worked"],
                "",
                analysis["first_principles_interpretation"]["what_failed"],
                "",
                analysis["first_principles_interpretation"]["identified_mechanism"],
                "",
                "## Locked next test",
                "",
                comparison["next_preregistered_test"],
                "",
            )
        ),
        encoding="utf-8",
    )
    (output_dir / "REPORT.zh-CN.md").write_text(
        "\n".join(
            (
                "# R14 固定错误记忆对称排序：实验结果",
                "",
                "**判定：技术通过，科学诊断失败；不得宣称 Picture Memory 已成功。**",
                "",
                "| 数据切分 | 全部门槛通过数 | 要求 | normal 准确率 | donor 准确率 | base 准确率 |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                *table,
                "",
                "## 第一性原理结论",
                "",
                "1. R14 确实学到固定训练对比：最后 64 个 micro-step 的 own CE 为 0.615、donor CE 为 2.983，排序满足率为 60.9%。",
                "2. 但严格门槛只有 train-audit 17/36、dev-select 1/24、dev-replay 4/24、dev-final 7/24；相对 R13 仅为 +3/0/-1/+1，不构成稳健泛化改善。",
                "3. 目标函数只在整个 epoch 上对称：2304 个 pair-epoch 中仅 44 个正反方向位于同一次更新，优化器步中位间隔为 11，形成彼此追逐与振荡。",
                "4. 每个训练事件 32 个 epoch 始终只见同一个负例；train-audit 的评测 donor 身份和目标值与训练 donor 均为 0/36 重合，因此评测要求的是训练中没有覆盖的关系泛化。",
                "5. 人眼不可读图片符合机器视觉码假设，但图片外观本身不构成成功证据；因果替换和未见样本泛化才是判定标准。",
                "6. 下一轮保持网络、数据、Reader 调用预算、优化器、固定端点和四条件评测不变，只把训练对比改为同一步双向且确定性轮换多负例。",
                "",
            )
        ),
        encoding="utf-8",
    )
    (output_dir / "FIRST_PRINCIPLES_CONCLUSION.md").write_text(
        "\n".join(
            (
                "# R14 first-principles conclusion",
                "",
                "1. Exact schedules, frozen snapshots, finite unclipped gradients, complete causal cells, fixed endpoint, and hash bindings establish technical validity.",
                "2. Fixed-donor ranking improved only local fit and did not pass the preregistered causal boundary.",
                "3. Update-asynchronous symmetry and one-negative coverage are directly measured design failures, not post-hoc guesses from loss shape.",
                "4. R15 must synchronize both pair directions and rotate negatives without changing the causal evaluation or success threshold.",
                "5. Full-data, multi-seed ID/OOD recurrence claims remain forbidden until a diagnostic arm passes and is confirmed on a newly sealed protocol.",
                "",
            )
        ),
        encoding="utf-8",
    )
    _write_json(output_dir / "DELIVERY_MANIFEST.json", _delivery_manifest(output_dir, source_sha256))
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--r13-analysis", type=Path, required=True)
    parser.add_argument("--remote-source-root", required=True)
    parser.add_argument("--report-source-archive", required=True)
    parser.add_argument("--report-source-archive-sha256", required=True)
    parser.add_argument("--report-source-archive-bytes", type=int, required=True)
    parser.add_argument("--complete-archive", required=True)
    parser.add_argument("--complete-archive-sha256", required=True)
    parser.add_argument("--complete-archive-bytes", type=int, required=True)
    args = parser.parse_args()
    analysis = render(
        source_root=args.source_root,
        output_dir=args.output_dir,
        r13_analysis=args.r13_analysis,
        remote_source_root=args.remote_source_root,
        report_source_archive=args.report_source_archive,
        report_source_archive_sha256=args.report_source_archive_sha256,
        report_source_archive_bytes=args.report_source_archive_bytes,
        complete_archive=args.complete_archive,
        complete_archive_sha256=args.complete_archive_sha256,
        complete_archive_bytes=args.complete_archive_bytes,
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
