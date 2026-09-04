"""Render a hash-checked R12 paired-arm delivery with causal diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis import analyze_r12_shared_writer_collapse as collapse  # noqa: E402
from scripts.experiments import compare_r12_shared_writer as comparison  # noqa: E402


ARM_COLORS = {"conditioned": "#2563eb", "constant-control": "#dc2626"}
CRITERIA = (
    ("m0_ce", "normal CE ≤ 80% M0", lambda row: float(row["relative_change"]) <= -0.20),
    ("m0_views", "normal improves 4/4 vs M0", lambda row: int(row["improved_choice_views"]) == 4),
    ("m0_acc", "normal accuracy Δ ≥ .25", lambda row: float(row["accuracy_delta"]) >= 0.25),
    (
        "reset_did",
        "normal/reset DiD < 0",
        lambda row: float(row["normal_reset_difference_in_differences"]) < 0.0,
    ),
    (
        "donor_ce",
        "normal CE ≤ 80% donor",
        lambda row: float(row["normal_vs_donor_relative_change"]) <= -0.20,
    ),
    (
        "donor_views",
        "normal beats donor 4/4",
        lambda row: int(row["normal_better_than_donor_views"]) == 4,
    ),
    (
        "donor_acc",
        "normal-donor accuracy ≥ .25",
        lambda row: float(row["normal_vs_donor_accuracy_delta"]) >= 0.25,
    ),
    (
        "donor_did",
        "normal/donor DiD < 0",
        lambda row: float(row["normal_donor_difference_in_differences"]) < 0.0,
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R12 renderer expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"R12 renderer expected JSON objects: {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _moving_mean(values: Sequence[float], window: int = 16) -> list[float]:
    return [
        statistics.fmean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def _training_rows(root: Path, arm: str) -> list[dict[str, Any]]:
    micro = _jsonl(root / "run" / "micro_metrics.jsonl")
    optimizer = _jsonl(root / "run" / "optimizer_metrics.jsonl")
    if len(micro) != comparison.EXPECTED_MICRO_STEPS or len(optimizer) != comparison.EXPECTED_OPTIMIZER_STEPS:
        raise ValueError(f"R12 renderer training row count drift: {arm}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in micro:
        grouped[int(row["optimizer_step_zero"]) + 1].append(row)
    if set(grouped) != set(range(1, comparison.EXPECTED_OPTIMIZER_STEPS + 1)) or any(
        len(rows) != 4 for rows in grouped.values()
    ):
        raise ValueError(f"R12 renderer gradient accumulation cells drifted: {arm}")
    optimizer_by_step = {int(row["optimizer_step"]): row for row in optimizer}
    rows = []
    for step in range(1, comparison.EXPECTED_OPTIMIZER_STEPS + 1):
        values = grouped[step]
        opt = optimizer_by_step[step]
        row = {
            "arm": arm,
            "optimizer_step": step,
            "epoch_zero": int(opt["epoch_zero"]),
            "learning_rate": float(opt["learning_rate"]),
            "ce_mean": statistics.fmean(float(value["ce"]) for value in values),
            "objective_mean": statistics.fmean(float(value["objective"]) for value in values),
            "latent_penalty_mean": statistics.fmean(float(value["latent_penalty"]) for value in values),
            "coefficient_penalty_mean": statistics.fmean(
                float(value["coefficient_penalty"]) for value in values
            ),
            "coefficient_norm_mean": statistics.fmean(
                float(value["coefficient_norm"]) for value in values
            ),
            "latent_delta_rms_mean": statistics.fmean(
                float(value["latent_delta_rms"]) for value in values
            ),
            "image_saturation_fraction_mean": statistics.fmean(
                float(value["image_saturation_fraction"]) for value in values
            ),
            "gradient_norm": float(opt["gradient_norm"]),
            "gradient_nonzero_fraction": float(opt["gradient_nonzero_fraction"]),
            "basis_parameter_norm": float(opt["basis_parameter_norm"]),
            "attention_query_norm": float(opt["attention_query_norm"]),
            "gradient_clipped": bool(opt["gradient_clipped"]),
        }
        if row["gradient_clipped"] or any(
            not math.isfinite(float(value))
            for key, value in row.items()
            if key not in {"arm", "gradient_clipped"}
        ):
            raise ValueError(f"R12 renderer invalid training diagnostic: {arm}:{step}")
        rows.append(row)
    smooth = _moving_mean([row["ce_mean"] for row in rows], window=16)
    for row, value in zip(rows, smooth, strict=True):
        row["ce_moving_mean_16_optimizer_steps"] = value
    return rows


def _write_training_csv(path: Path, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fields = list(next(iter(rows.values()))[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in comparison.ARMS:
            writer.writerows(rows[arm])


def _training_figure(path: Path, rows: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    panels = (
        ("ce_moving_mean_16_optimizer_steps", "Training Reader CE (64-micro moving mean)", "CE", True),
        ("gradient_norm", "Writer gradient norm", "L2 norm", True),
        ("coefficient_norm_mean", "Generated coefficient norm", "coefficient L2", False),
        ("image_saturation_fraction_mean", "Decoded image saturation", "fraction at 0 or 1", False),
    )
    for arm in comparison.ARMS:
        values = rows[arm]
        steps = [int(row["optimizer_step"]) for row in values]
        for axis, (field, _title, _ylabel, _log) in zip(axes.flat, panels, strict=True):
            axis.plot(
                steps,
                [float(row[field]) for row in values],
                label=arm,
                color=ARM_COLORS[arm],
                linewidth=1.1,
                alpha=0.9,
            )
    for axis, (_field, title, ylabel, log_scale) in zip(axes.flat, panels, strict=True):
        if log_scale:
            axis.set_yscale("log")
        for checkpoint in comparison.EXPECTED_CHECKPOINT_STEPS[1:]:
            axis.axvline(checkpoint, color="black", linewidth=0.5, alpha=0.18)
        axis.set(title=title, xlabel="optimizer step", ylabel=ylabel)
        axis.grid(alpha=0.18)
        axis.legend(fontsize=8)
    figure.suptitle("R12 shared-writer training diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _endpoint_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for arm in comparison.ARMS:
        for split in comparison.SPLITS:
            aggregate = result["aggregates"][arm][split]
            rows.append(
                {
                    "arm": arm,
                    "split": split,
                    **aggregate,
                    "normal_relative_ce_change_vs_m0": (
                        aggregate["endpoint_normal_mean_ce"] / aggregate["m0_normal_mean_ce"] - 1.0
                    ),
                    "normal_relative_ce_change_vs_donor": (
                        aggregate["endpoint_normal_mean_ce"] / aggregate["endpoint_donor_mean_ce"]
                        - 1.0
                    ),
                }
            )
    return rows


def _write_dict_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _endpoint_figure(path: Path, endpoint_rows: Sequence[Mapping[str, Any]]) -> None:
    by_key = {(row["arm"], row["split"]): row for row in endpoint_rows}
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    split_labels = list(comparison.SPLITS)
    x = list(range(len(split_labels)))
    width = 0.18
    for arm_index, arm in enumerate(comparison.ARMS):
        offset_base = (-1.5 + arm_index * 2) * width
        for condition_index, (field, suffix) in enumerate(
            (("endpoint_normal_mean_ce", "normal"), ("endpoint_donor_mean_ce", "donor"))
        ):
            offset = offset_base + condition_index * width
            axes[0, 0].bar(
                [value + offset for value in x],
                [float(by_key[(arm, split)][field]) for split in split_labels],
                width=width,
                label=f"{arm} {suffix}",
                color=ARM_COLORS[arm],
                alpha=0.9 if suffix == "normal" else 0.4,
                hatch="" if suffix == "normal" else "//",
            )
        axes[0, 1].bar(
            [value + (arm_index - 0.5) * 0.34 for value in x],
            [float(by_key[(arm, split)]["endpoint_normal_accuracy"]) for split in split_labels],
            width=0.34,
            label=arm,
            color=ARM_COLORS[arm],
            alpha=0.85,
        )
        axes[1, 0].bar(
            [value + (arm_index - 0.5) * 0.34 for value in x],
            [int(by_key[(arm, split)]["target_pass_count"]) for split in split_labels],
            width=0.34,
            label=arm,
            color=ARM_COLORS[arm],
            alpha=0.85,
        )
        axes[1, 1].bar(
            [value + (arm_index - 0.5) * 0.34 for value in x],
            [
                float(by_key[(arm, split)]["normal_relative_ce_change_vs_donor"])
                for split in split_labels
            ],
            width=0.34,
            label=arm,
            color=ARM_COLORS[arm],
            alpha=0.85,
        )
    panels = (
        (axes[0, 0], "Endpoint normal vs donor CE", "mean CE"),
        (axes[0, 1], "Endpoint normal accuracy", "accuracy"),
        (axes[1, 0], "Targets passing all causal gates", "count"),
        (axes[1, 1], "Normal CE relative to donor", "normal / donor - 1"),
    )
    for axis, title, ylabel in panels:
        axis.set(title=title, ylabel=ylabel)
        axis.set_xticks(x, split_labels, rotation=12)
        axis.grid(axis="y", alpha=0.18)
        axis.legend(fontsize=7)
    axes[1, 1].axhline(-0.20, color="black", linestyle="--", linewidth=0.9, label="gate -0.20")
    axes[1, 1].axhline(0.0, color="black", linewidth=0.6)
    figure.suptitle("R12 fixed endpoint and causal-control results")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _criterion_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for arm in comparison.ARMS:
        for split in comparison.SPLITS:
            statistics = result["arms"][arm]["split_statistics"][split]
            for key, label, predicate in CRITERIA:
                rows.append(
                    {
                        "arm": arm,
                        "split": split,
                        "criterion": key,
                        "criterion_label": label,
                        "passed_targets": sum(predicate(row) for row in statistics),
                        "targets": len(statistics),
                    }
                )
    return rows


def _criterion_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16.0, 8.0), sharey=False)
    by_key = {(row["arm"], row["split"], row["criterion"]): row for row in rows}
    labels = [label for _key, label, _predicate in CRITERIA]
    x = list(range(len(CRITERIA)))
    for axis, (arm, split) in zip(
        axes.flat,
        ((arm, split) for arm in comparison.ARMS for split in comparison.SPLITS),
        strict=True,
    ):
        values = [int(by_key[(arm, split, key)]["passed_targets"]) for key, _label, _fn in CRITERIA]
        total = int(by_key[(arm, split, CRITERIA[0][0])]["targets"])
        colors = ["#16a34a" if value == total else "#dc2626" for value in values]
        axis.bar(x, values, color=colors, alpha=0.85)
        axis.axhline(total, color="black", linestyle="--", linewidth=0.8)
        axis.set(title=f"{arm} · {split}", ylabel=f"targets (of {total})")
        axis.set_xticks(x, labels, rotation=58, ha="right", fontsize=7)
        axis.grid(axis="y", alpha=0.18)
    figure.suptitle("R12 per-criterion target pass counts (green only when every target passes)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _checkpoint_rows(roots: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for arm, root in roots.items():
        for step in comparison.EXPECTED_CHECKPOINT_STEPS:
            value = _load(root / "run" / "checkpoint_diagnostics" / f"step-{step:04d}.json")
            for row in value["rows"]:
                rows.append(
                    {
                        "arm": arm,
                        "step": step,
                        "split": row["split"],
                        "segment_id": row["segment_id"],
                        "coefficient_norm": row["coefficient_norm"],
                        "coefficient_max_abs": row["coefficient_max_abs"],
                        "latent_delta_rms": row["latent_delta_rms"],
                        "image_saturation_fraction": row["image_saturation_fraction"],
                        "basis_norm_min": value["basis_norm_min"],
                        "basis_norm_max": value["basis_norm_max"],
                    }
                )
    return rows


def _checkpoint_figure(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    fields = (
        ("coefficient_norm", "Representative coefficient norm"),
        ("latent_delta_rms", "Representative latent delta RMS"),
        ("image_saturation_fraction", "Representative image saturation"),
    )
    for arm in comparison.ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        for axis, (field, title) in zip(axes, fields, strict=True):
            steps = comparison.EXPECTED_CHECKPOINT_STEPS
            means = [
                statistics.fmean(float(row[field]) for row in arm_rows if int(row["step"]) == step)
                for step in steps
            ]
            axis.plot(steps, means, marker="o", label=arm, color=ARM_COLORS[arm])
            axis.set(title=title, xlabel="fixed checkpoint", ylabel=field)
            axis.grid(alpha=0.18)
            axis.legend(fontsize=8)
    figure.suptitle("R12 fixed checkpoint trajectory (descriptive; no checkpoint selection)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _image_contact_sheet(path: Path, roots: Mapping[str, Path], result: Mapping[str, Any]) -> None:
    selections = []
    for arm in comparison.ARMS:
        summary = result["arms"][arm]
        for split in comparison.SPLITS:
            target = summary["split_statistics"][split][0]
            selections.append((arm, split, target))
    figure, axes = plt.subplots(len(selections), 2, figsize=(8.0, 3.0 * len(selections)))
    for row_index, (arm, split, target) in enumerate(selections):
        segment_id = target["target_segment_id"]
        for column, condition in enumerate(("normal", "donor")):
            image_path = roots[arm] / "run" / "images" / "endpoint" / split / condition / f"{segment_id}.png"
            axes[row_index, column].imshow(plt.imread(image_path))
            axes[row_index, column].set_title(
                f"{arm} · {split} · {condition}\nvalue={target['target_value']}"
            )
            axes[row_index, column].axis("off")
    figure.suptitle("R12 endpoint visual codes: own-event normal vs fixed donor")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _delivery_manifest(output_dir: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    files = [
        path for path in sorted(output_dir.iterdir()) if path.is_file() and path.name != "DELIVERY_MANIFEST.json"
    ]
    return {
        "schema": "vision_memory.r12-shared-writer-delivery-manifest.v1",
        "status": "completed",
        "formal_success_claim": False,
        "source_sha256": source,
        "artifacts": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in files
        ],
    }


def render(conditioned_root: Path, control_root: Path, output_dir: Path) -> dict[str, Any]:
    roots = {"conditioned": conditioned_root, "constant-control": control_root}
    result = comparison.compare(conditioned_root, control_root, output_dir)
    training_rows = {arm: _training_rows(roots[arm], arm) for arm in comparison.ARMS}
    _write_training_csv(output_dir / "training_metrics.csv", training_rows)
    _training_figure(output_dir / "training_diagnostics.png", training_rows)
    endpoint_rows = _endpoint_rows(result)
    _write_dict_csv(output_dir / "endpoint_summary.csv", endpoint_rows)
    _endpoint_figure(output_dir / "endpoint_metrics.png", endpoint_rows)
    criterion_rows = _criterion_rows(result)
    _write_dict_csv(output_dir / "causal_gate_counts.csv", criterion_rows)
    _criterion_figure(output_dir / "causal_gate_counts.png", criterion_rows)
    checkpoint_rows = _checkpoint_rows(roots)
    _write_dict_csv(output_dir / "checkpoint_trajectory.csv", checkpoint_rows)
    _checkpoint_figure(output_dir / "checkpoint_trajectory.png", checkpoint_rows)
    _image_contact_sheet(output_dir / "endpoint_image_contact_sheet.png", roots, result)

    conditioned_run = conditioned_root / "run"
    collapse_path = output_dir / "conditioned_collapse_audit.json"
    collapse_result = collapse.analyze(
        summary_path=conditioned_run / "r12_shared_writer_summary.json",
        checkpoint_path=conditioned_run / "checkpoints" / "step-1152.pt",
        embedding_cache_path=conditioned_run / "event_embedding_cache.pt",
        micro_metrics_path=conditioned_run / "micro_metrics.jsonl",
        output_path=collapse_path,
    )
    token_probe = collapse_result["representations"]["uniform_token_mean"][
        "ridge_target_value_probe"
    ]["accuracy"]
    coefficient_probe = collapse_result["representations"]["coefficients"][
        "ridge_target_value_probe"
    ]["accuracy"]
    conditioned_aggregates = result["aggregates"]["conditioned"]
    analysis = {
        "schema": "vision_memory.r12-shared-writer-rendered-analysis.v1",
        "status": "completed",
        "aggregation_git_commit": result["aggregation_git_commit"],
        "renderer_sha256": _sha256(Path(__file__).resolve()),
        "decision": result["decision"],
        "reason": result["reason"],
        "r12_diagnostic_gate": result["r12_diagnostic_gate"],
        "formal_success_claim": False,
        "formal_success_reason": result["formal_success_reason"],
        "conditioned_aggregates": conditioned_aggregates,
        "constant_control_aggregates": result["aggregates"]["constant-control"],
        "posthoc_collapse_localization": {
            "uniform_token_mean_ridge_accuracy": token_probe,
            "writer_coefficient_ridge_accuracy": coefficient_probe,
            "attention": collapse_result["attention"],
            "coefficient_anchor": collapse_result["coefficient_anchor"],
            "interpretation": (
                "Frozen event states retain target-value information, but the learned coefficient head compresses most of it into a dominant event-independent visual code. This is post-hoc localization, not a changed R12 outcome."
            ),
        },
    }
    _write_json(output_dir / "ANALYSIS.json", analysis)
    _write_json(
        output_dir / "RAW_ARTIFACTS.json",
        {
            "schema": "vision_memory.r12-shared-writer-source-artifacts.v1",
            "source_roots": result["source_roots"],
            "source_sha256": result["source_sha256"],
        },
    )

    condition_counts = result["arms"]["conditioned"]["gates"]["split_target_pass_counts"]
    control_counts = result["arms"]["constant-control"]["gates"]["split_target_pass_counts"]
    report = "\n".join(
        (
            "# R12 shared event-to-latent writer: paired result",
            "",
            f"**Decision:** `{result['decision']}`",
            "",
            result["reason"],
            "",
            "## Fixed causal gates",
            "",
            "| arm | train audit | dev select | sealed dev final | arm gate |",
            "| --- | ---: | ---: | ---: | --- |",
            (
                f"| conditioned | {condition_counts['train_audit']}/36 | "
                f"{condition_counts['dev_select']}/24 | {condition_counts['dev_final']}/24 | "
                f"{'PASS' if result['arms']['conditioned']['gates']['arm_gate'] else 'FAIL'} |"
            ),
            (
                f"| constant-control | {control_counts['train_audit']}/36 | "
                f"{control_counts['dev_select']}/24 | {control_counts['dev_final']}/24 | "
                f"{'PASS' if result['arms']['constant-control']['gates']['arm_gate'] else 'FAIL'} |"
            ),
            "",
            "## First-principles localization",
            "",
            (
                "The frozen event representation is linearly predictive of the target value "
                f"(ridge audit: train-audit {token_probe['train_audit']:.1%}, "
                f"dev-select {token_probe['dev_select']:.1%}, dev-final {token_probe['dev_final']:.1%})."
            ),
            (
                "After the learned coefficient head, the same audit falls to "
                f"{coefficient_probe['train_audit']:.1%}, {coefficient_probe['dev_select']:.1%}, and "
                f"{coefficient_probe['dev_final']:.1%}. The dominant failure is therefore conditional-code "
                "collapse after the event encoder, not absence of event information in the frozen encoder."
            ),
            "",
            "The R12 scientific outcome remains unchanged by this post-hoc audit. R12 is diagnostic-only and cannot establish full Picture Memory success.",
            "",
        )
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    conditioned_endpoint = result["aggregates"]["conditioned"]
    control_endpoint = result["aggregates"]["constant-control"]
    report_zh = "\n".join(
        (
            "# R12 共享事件到视觉 latent 写入器：双臂结果",
            "",
            f"**判定：** `{result['decision']}`",
            "",
            "## 固定因果门槛",
            "",
            "| 实验臂 | train-audit | dev-select | sealed dev-final | 总门槛 |",
            "| --- | ---: | ---: | ---: | --- |",
            (
                f"| conditioned | {condition_counts['train_audit']}/36 | "
                f"{condition_counts['dev_select']}/24 | {condition_counts['dev_final']}/24 | "
                f"{'通过' if result['arms']['conditioned']['gates']['arm_gate'] else '未通过'} |"
            ),
            (
                f"| constant-control | {control_counts['train_audit']}/36 | "
                f"{control_counts['dev_select']}/24 | {control_counts['dev_final']}/24 | "
                f"{'通过' if result['arms']['constant-control']['gates']['arm_gate'] else '未通过'} |"
            ),
            "",
            "## 核心事实",
            "",
            (
                "conditioned 的平均 CE（M0→normal→donor）分别为："
                f"train-audit {conditioned_endpoint['train_audit']['m0_normal_mean_ce']:.3f}→"
                f"{conditioned_endpoint['train_audit']['endpoint_normal_mean_ce']:.3f}→"
                f"{conditioned_endpoint['train_audit']['endpoint_donor_mean_ce']:.3f}；"
                f"dev-select {conditioned_endpoint['dev_select']['m0_normal_mean_ce']:.3f}→"
                f"{conditioned_endpoint['dev_select']['endpoint_normal_mean_ce']:.3f}→"
                f"{conditioned_endpoint['dev_select']['endpoint_donor_mean_ce']:.3f}；"
                f"dev-final {conditioned_endpoint['dev_final']['m0_normal_mean_ce']:.3f}→"
                f"{conditioned_endpoint['dev_final']['endpoint_normal_mean_ce']:.3f}→"
                f"{conditioned_endpoint['dev_final']['endpoint_donor_mean_ce']:.3f}。"
            ),
            (
                "constant-control 的 normal CE 为："
                f"train-audit {control_endpoint['train_audit']['endpoint_normal_mean_ce']:.3f}、"
                f"dev-select {control_endpoint['dev_select']['endpoint_normal_mean_ce']:.3f}、"
                f"dev-final {control_endpoint['dev_final']['endpoint_normal_mean_ce']:.3f}。"
            ),
            "",
            "## 第一性原理归因",
            "",
            (
                "冻结事件表征仍能预测目标值：线性探针在 train-audit/dev-select/dev-final 上为 "
                f"{token_probe['train_audit']:.1%}/{token_probe['dev_select']:.1%}/"
                f"{token_probe['dev_final']:.1%}。"
            ),
            (
                "经过写入器系数头后降为 "
                f"{coefficient_probe['train_audit']:.1%}/{coefficient_probe['dev_select']:.1%}/"
                f"{coefficient_probe['dev_final']:.1%}。事件特异信息主要在表征到视觉码的映射阶段丢失，"
                "而不是冻结事件编码器中不存在。"
            ),
            "",
            "R12 仅是单步 SET 诊断，不能作为完整 Picture Memory 成功。下一轮必须保持现有 normal/reset/donor 门槛，并结构性消除事件无关通用视觉码捷径。",
            "",
        )
    )
    (output_dir / "REPORT.zh-CN.md").write_text(report_zh, encoding="utf-8")
    conclusion = "\n".join(
        (
            "# R12 first-principles conclusion",
            "",
            "1. The end-to-end gradient and artifact pipeline is valid: both arms must pass the locked technical gate.",
            "2. A learned event-independent image can sharply reduce frozen-Reader CE, so M0 improvement alone is not memory.",
            "3. The unchanged donor intervention is decisive: own-event images must outperform wrong-event images target by target.",
            "4. The post-hoc probe shows target information before the writer head but substantial loss after it, localizing the next intervention to conditional credit assignment and coefficient generation.",
            "5. The next experiment must preserve all evaluation gates and directly remove or penalize the event-independent shortcut; it must not advance to recurrence until one-SET conditionality passes.",
            "",
        )
    )
    (output_dir / "FIRST_PRINCIPLES_CONCLUSION.md").write_text(conclusion, encoding="utf-8")
    manifest = _delivery_manifest(output_dir, result["source_sha256"])
    _write_json(output_dir / "DELIVERY_MANIFEST.json", manifest)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioned-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = render(args.conditioned_root, args.control_root, args.output_dir)
    print(
        json.dumps(
            {
                "decision": analysis["decision"],
                "r12_diagnostic_gate": analysis["r12_diagnostic_gate"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
