"""Render a teacher-assisted initialization diagnostic from verified raw evidence.

This CPU-capable delivery renderer binds the CUDA aggregator's inventories and
recomputes plotted checkpoint distances and Reader rows. It deliberately does
not re-run CUDA initialization arithmetic or replace the independent aggregator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import compare_r11_new_canonical_latent_bridge as aggregate  # noqa: E402
from vision_memory.repro import canonical_object_sha256, canonical_tensor_sha256  # noqa: E402


core = aggregate.core
STEPS = (0, 64, 128, 192, 256)
SCHEMA = "vision_memory.r11-new-bridge-teacher-init-delivery.v1"
CHECKPOINT_AUDIT_FIELDS = {"optimizer_step", "completed_updates", "local_cpu_recomputed_distance_statistics"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same_statistics(declared: Mapping[str, Any], actual: Mapping[str, Any], label: str) -> None:
    _require(set(declared) == set(actual), f"{label} statistic fields drifted.")
    for key, value in actual.items():
        _require(aggregate._equal_float(declared[key], value), f"{label}/{key} arithmetic drifted.")


def _bound_inventory(root: Path, binding: Mapping[str, Any], schema: str) -> dict[str, Any]:
    result = aggregate._validate_inventory(root, schema=schema)
    _require(result["sha256"] == binding.get("sha256"), f"RAW inventory binding drifted: {root}")
    return result


def _read_metrics(path: Path, binding: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float]:
    """Recompute display arithmetic without substituting Windows libm for Linux.

The same cosine expression differs by a few binary64 ULPs across libm builds.
The source-runtime aggregator already checked LR exactly. Its byte-bound audit
is authoritative for LR; this renderer neither loosens that gate nor edits LR.
"""
    _require(sha256(path) == binding.get("sha256") and binding.get("checks", {}).get("optimizer") is True,
             "Source-runtime exact optimizer audit or receipt hash binding drifted.")
    rows = aggregate._load_jsonl(path)
    _require([row.get("optimizer_step") for row in rows] == list(range(1, 257)),
             "Receipt steps must be exactly 1..256.")
    m0_values = {row.get("m0_mse") for row in rows}
    _require(len(m0_values) == 1, "Receipt M0 is inconsistent.")
    m0 = float(next(iter(m0_values)))
    for row in rows:
        statistics = core.bridge_distance_statistics(mse=float(row["mse"]), m0_mse=m0, tensor_numel=65536)
        _require(all(aggregate._equal_float(row.get(key), value) for key, value in statistics.items())
                 and row.get("loss_before_step") == row["mse"], "Receipt distance arithmetic drifted.")
        _require(
            row.get("schema") == aggregate.TRAINER_METRICS_SCHEMA
            and row.get("target_index") == 1 and row.get("target_segment_id") == core.BRIDGE_TARGET_SEGMENT_ID
            and row.get("objective") == "canonical_r11_latent_fp32_mean_mse"
            and row.get("teacher_tensor_sha256") == core.BRIDGE_TEACHER_TENSOR_SHA256
            and row.get("full_dreamlite_forward_executed") is True
            and row.get("dreamlite_denoising_steps") == 4 and row.get("trajectory_points") == 5
            and aggregate._sigmas_exact(row.get("effective_sigmas"))
            and row.get("gradient_clipping_applied") is False and row.get("reader_gradient_calls") == 0
            and row.get("weight_decay") == 0
            and all(isinstance(row.get(field), (float, int)) and not isinstance(row[field], bool)
                    and math.isfinite(row[field]) and row[field] >= 0 for field in (
                        "gradient_norm", "gradient_nonzero_fraction", "x_T_update_norm", "learning_rate", "elapsed_seconds",
                    ))
            and row["gradient_norm"] > 0 and row["gradient_nonzero_fraction"] > 0,
            "Receipt path/finite-gradient contract drifted.",
        )
    _require(all(row["learning_rate"] == 0.05 for row in rows[:128]) and rows[-1]["learning_rate"] == 0,
             "Receipt constant-prefix/zero-final-LR boundary drifted.")
    return rows, m0


def _checkpoint_distance(
    run: Path, record: Mapping[str, Any], teacher: torch.Tensor, m0_mse: float,
) -> dict[str, Any]:
    """CPU distance recomputation; CUDA h0 closure remains the bound aggregator's job."""
    step = record["optimizer_step"]
    path = run / "checkpoints" / f"step-{step:03d}.pt"
    image = run / "images" / f"step-{step:03d}.png"
    hash_path = run / "checkpoint_hashes" / f"step-{step:03d}.json"
    for item, field in ((path, "checkpoint_sha256"), (image, "image_sha256"),
                        (hash_path, "record_sha256")):
        _require(sha256(item) == record.get(field), f"RAW checkpoint file binding drifted: {item}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x_t, z_t, trajectory = (payload.get(name) for name in ("x_T_fp32", "z_t_fp32", "trajectory_fp32"))
    _require(
        payload.get("optimizer_step") == step
        and isinstance(x_t, torch.Tensor) and isinstance(z_t, torch.Tensor)
        and isinstance(trajectory, (list, tuple)) and len(trajectory) == 5
        and all(isinstance(t, torch.Tensor) and t.dtype == torch.float32
                and tuple(t.shape) == (1, 4, 128, 128) and bool(torch.isfinite(t).all())
                for t in (x_t, z_t, *trajectory))
        and torch.equal(trajectory[-1], z_t),
        f"Checkpoint raw tensor/trajectory contract drifted at {step}.",
    )
    hashes = {
        "x_T_fp32": canonical_tensor_sha256(x_t),
        "z_t_fp32": canonical_tensor_sha256(z_t),
        "trajectory_fp32": [canonical_tensor_sha256(t) for t in trajectory],
    }
    _require(hashes == record.get("tensor_sha256") == payload.get("tensor_sha256"),
             f"Checkpoint tensor hash drifted at {step}.")
    _require(
        aggregate._optimizer_state_matches_step(payload.get("optimizer"), expected_step=step)
        and canonical_object_sha256(payload["optimizer"]) == record.get("optimizer_state_sha256")
        == payload.get("optimizer_state_sha256"),
        f"Checkpoint Adam state/count drifted at {step}.",
    )
    statistics = core.bridge_distance_statistics(
        mse=float((z_t - teacher).square().mean()), m0_mse=m0_mse, tensor_numel=teacher.numel(),
    )
    _same_statistics(record["distance_statistics"], statistics, f"checkpoint {step}")
    _same_statistics(payload["distance_statistics"], statistics, f"payload {step}")
    return {
        "optimizer_step": step, "completed_updates": step,
        **dict(record["distance_statistics"]),
        "local_cpu_recomputed_distance_statistics": statistics,
    }


def collect_evidence(
    *, preflight_root: Path, formal_root: Path, aggregation_root: Path,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Read-only verification. No output directory is touched before this passes."""
    roots = [path.resolve() for path in (preflight_root, formal_root, aggregation_root)]
    _require(len(set(roots)) == 3 and all(path.is_dir() for path in roots),
             "Three distinct existing input directories are required.")
    preflight_root, formal_root, aggregation_root = roots
    agg_inventory = aggregate._validate_inventory(aggregation_root, schema=aggregate.INVENTORY_SCHEMA)
    comparison = _json(aggregation_root / "comparison.json")
    raw_path = aggregation_root / "RAW_ARTIFACTS.json"
    raw = _json(raw_path)
    commit = comparison.get("git_commit")
    _require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None,
             "A full training commit is required.")
    _require(expected_commit is None or commit == expected_commit, "Unexpected training commit.")
    required = {
        "schema": aggregate.COMPARISON_SCHEMA, "status": "completed", "protocol": core.BRIDGE_PROTOCOL,
        "engineering_gate": True, "teacher_replay_gate": True, "optimizer_steps": 256,
        "target_index": 1, "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID,
        "primary_endpoint": core.BRIDGE_PRIMARY_ENDPOINT, "teacher_assisted_initialization": True,
        "answer_independent_writer_usable": False, "formal_success": False, "phase2_allowed": False,
        "scientific_success_claim": False, "interpretation": "diagnostic_only",
        "raw_artifacts_sha256": sha256(raw_path),
    }
    _require(all(comparison.get(key) == value for key, value in required.items()),
             "Aggregate protocol, evidence hash, or diagnostic-only boundary drifted.")
    _require(raw.get("schema") == aggregate.RAW_SCHEMA and raw.get("expected_commit") == commit,
             "RAW schema/commit binding drifted.")
    inventories = {"aggregation": agg_inventory}
    for label, root in (("preflight", preflight_root), ("formal", formal_root)):
        binding = raw[f"{label}_controller"]
        inventories[label] = _bound_inventory(root, binding["root_inventory"], aggregate.controller.INVENTORY_SCHEMA)
        _bound_inventory(root / "run", binding["trainer_inventory"], aggregate.TRAINER_INVENTORY_SCHEMA)
        terminal = _json(root / "terminal.json")
        _require(terminal.get("git_commit") == commit and terminal.get("technical_gate") is True
                 and terminal.get("formal_success") is False and terminal.get("phase2_allowed") is False,
                 f"{label} terminal boundary/commit drifted.")
    run = formal_root / "run"
    preflight_run = preflight_root / "run"
    preflight = _json(preflight_run / "technical_preflight.json")
    audit = preflight.get("audit", {})
    _require(
        preflight == _json(preflight_run / "r11_new_bridge_summary.json")
        and preflight.get("passed") is True and preflight.get("bridge_result_evaluated") is False
        and preflight.get("formal_success_gate") is False and preflight.get("phase2_allowed") is False
        and (audit.get("full_forward_calls"), audit.get("backward_calls"), audit.get("optimizer_steps")) == (1, 1, 0)
        and all(audit.get(key) is True for key in (
            "four_dreamlite_steps", "finite_nonzero_x_T_gradient", "only_x_T_fp32_trainable",
            "frozen_gradients_absent", "teacher_matched_initialization_artifact_valid", "snapshots_unchanged",
        )) and not (preflight_run / "metrics.jsonl").exists(),
        "Preflight must be one full forward, one backward, zero updates, and diagnostic unevaluated.",
    )
    _require(sha256(preflight_run / "r11_new_bridge_summary.json") == raw["preflight"]["summary_sha256"],
             "Preflight summary binding drifted.")
    for label, root in (("preflight", preflight_run), ("formal", run)):
        initialization = raw["initialization"][label]
        _require(initialization.get("passed") is True and initialization.get("compute_device_type") == "cuda"
                 and initialization.get("compute_dtype") == "torch.bfloat16"
                 and 0 <= initialization["trajectory_point0_teacher_normalized_rmse"] <= 0.01
                 and sha256(root / "initialization" / "teacher_matched_initialization.pt")
                 == initialization.get("artifact_sha256"),
                 "The renderer requires an unchanged CUDA-verified initialization artifact.")
    teacher, _ = aggregate._load_teacher(run, preflight_run)
    metrics, m0_mse = _read_metrics(run / "metrics.jsonl", raw["metrics"])
    checkpoints = raw["checkpoints"]
    _require([row.get("optimizer_step") for row in checkpoints] == list(STEPS), "Checkpoint roster drifted.")
    checkpoint_data = [
        _checkpoint_distance(run, record, teacher, m0_mse) for record in checkpoints
    ]
    _require(aggregate._equal_float(checkpoint_data[0]["mse"], m0_mse),
             "M0 must be the unoptimized full-chain endpoint.")
    for row in checkpoint_data:
        stats = {key: value for key, value in row.items() if key not in CHECKPOINT_AUDIT_FIELDS}
        _same_statistics(comparison["checkpoint_distance_statistics"][str(row["optimizer_step"])], stats,
                         f"comparison checkpoint {row['optimizer_step']}")
        _require(comparison["checkpoint_distance_statistics"][str(row["optimizer_step"])] == stats,
                 "Source checkpoint/aggregate arithmetic declarations differ; refuse unlabelled display mixture.")
    _require(sha256(run / "endpoint_raw.pt") == sha256(run / "checkpoints" / "step-256.pt")
             and sha256(run / "endpoint_raw.png") == sha256(run / "images" / "step-256.png"),
             "Raw endpoint must be byte-identical to checkpoint 256.")
    target = raw["locked_parent_target"]["target_segment"]
    reader = aggregate._validate_rows(run / "evaluation_rows.jsonl", target_segment=target)
    _require(reader["sha256"] == raw["evaluation_rows"]["sha256"], "Reader raw-row hash binding drifted.")
    groups = reader["statistics"]
    for field, group in (
        ("teacher_replay_statistics", "canonical_teacher/teacher"), ("m0_reader_statistics", "m0/normal"),
        ("m0_reset_statistics", "m0/reset"),
        ("endpoint_reader_statistics", f"{core.BRIDGE_PRIMARY_ENDPOINT}/normal"),
        ("endpoint_reset_statistics", f"{core.BRIDGE_PRIMARY_ENDPOINT}/reset"),
    ):
        _require(comparison[field] == groups[group], f"Reader aggregate arithmetic drifted: {field}")
    endpoint = {key: value for key, value in checkpoint_data[-1].items()
                if key not in CHECKPOINT_AUDIT_FIELDS}
    _same_statistics(comparison["endpoint_distance_statistics"], endpoint, "raw endpoint")
    _require(comparison["endpoint_distance_statistics"] == endpoint,
             "Source endpoint arithmetic declarations differ; refuse unlabelled display mixture.")
    teacher_gate = core.teacher_replay_gate(groups["canonical_teacher/teacher"])
    distance_gate = core.bridge_distance_gate(endpoint, technical_gate=teacher_gate)
    reader_gate = core.endpoint_reader_transfer_gate(comparison["endpoint_reader_statistics"])
    secondary = core.bridge_initialization_hypothesis_audit(
        endpoint_mse=endpoint["mse"], endpoint_reader_mean_ce=comparison["endpoint_reader_statistics"]["mean_ce"],
        technical_gate=True, teacher_replay_gate=teacher_gate, distance_pass=distance_gate,
        reader_transfer_pass=reader_gate,
    )
    _require(
        teacher_gate and comparison["bridge_distance_gate"] is distance_gate
        and comparison["endpoint_reader_transfer_gate"] is reader_gate
        and comparison["bridge_diagnostic_gate"] is (distance_gate and reader_gate)
        and comparison["decision"] == core.bridge_decision(distance_pass=distance_gate, reader_transfer_pass=reader_gate)
        and comparison["secondary_solver_hypothesis_audit"] == secondary
        and comparison["secondary_solver_hypothesis_decision"] == core.bridge_initialization_hypothesis_decision(
            distance_pass=distance_gate, reader_transfer_pass=reader_gate, audit=secondary,
        ), "Primary or secondary decision differs from recomputed evidence.",
    )
    parent = raw["locked_parent_bridge"]
    config = aggregate._validate_config()["parent_bridge"]
    _require(all(parent.get(key) == config[key] for key in (
        "training_git_commit", "comparison_sha256", "raw_artifacts_sha256", "config_sha256",
    )), "Locked absolute parent reference provenance drifted.")
    return {
        "schema": SCHEMA, "training_commit": commit, "comparison": comparison,
        "receipts": [{"completed_updates": row["optimizer_step"] - 1, **row} for row in metrics],
        "checkpoints": checkpoint_data, "reader_rows": reader["rows"],
        "parent_absolute_reference": {
            "training_commit": parent["training_git_commit"], "comparison_sha256": parent["comparison_sha256"],
            "endpoint_mse": core.BRIDGE_PARENT_ENDPOINT_MSE, "endpoint_reader_mean_ce": core.BRIDGE_PARENT_ENDPOINT_READER_CE,
            "m0_denominator_is_different": True,
        },
        "preflight_audit": audit, "initialization": raw["initialization"],
        "source_inventories": inventories,
        "source_comparison_sha256": sha256(aggregation_root / "comparison.json"),
        "source_raw_artifacts_sha256": sha256(raw_path),
        "verification_scope": {
            "all_bound_source_file_bytes_verified": True, "checkpoint_distance_recomputed_from_tensors": True,
            "reader_statistics_recomputed_from_logits": True, "cuda_initializer_recomputed_by_this_renderer": False,
            "cuda_initializer_evidence": "byte-bound independent CUDA aggregation; no CPU fallback/reconstruction",
            "optimizer_learning_rate_exact_evidence": "byte-bound independent source-runtime aggregation; original LR plotted unchanged",
            "checkpoint_display_statistics": "unchanged hash-bound source record statistics, identical to aggregate declarations",
            "local_cpu_reduction_statistics": "separate checkpoints[*].local_cpu_recomputed_distance_statistics; existing tolerance verification only",
        },
        "formal_success": False, "phase2_allowed": False, "phase1a_passed": 6, "phase1a_total": 8,
    }


def _save_figure(path: Path, figure: Any) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot(evidence: Mapping[str, Any], formal_run: Path, output: Path) -> None:
    metrics, checkpoints = evidence["receipts"], evidence["checkpoints"]
    x = [row["completed_updates"] for row in metrics]
    cx = [row["completed_updates"] for row in checkpoints]
    parent = evidence["parent_absolute_reference"]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.5))
    for axis, field, title in (
        (axes[0], "mse", "Absolute endpoint-to-teacher MSE"),
        (axes[1], "mse_ratio_to_m0", "MSE / NEW full-chain M0 MSE"),
    ):
        axis.plot(x, [row[field] for row in metrics], color="#2563eb", label="receipt: before update u (x=u-1)")
        axis.scatter(cx, [row[field] for row in checkpoints], color="#d97706", label="checkpoint: after u updates", zorder=3)
        axis.scatter([256], [checkpoints[-1][field]], color="#dc2626", marker="D", s=55, label="raw256: only primary endpoint", zorder=4)
        axis.set_title(title)
        axis.set_xlabel("completed optimizer updates")
        axis.set_yscale("symlog", linthresh=1e-8)
        axis.grid(alpha=0.2)
    axes[0].axhline(parent["endpoint_mse"], color="#737373", linestyle="--", label="parent raw256 absolute MSE")
    axes[1].axhline(0.01, color="#dc2626", linestyle=":", label="primary MSE-ratio gate")
    for axis in axes:
        axis.legend(fontsize=7)
    figure.suptitle("Teacher-assisted initialization diagnostic | parent and new M0 are different")
    figure.tight_layout()
    _save_figure(output / "distance_trajectory.png", figure)

    figure, axes = plt.subplots(1, 3, figsize=(13.3, 4.1))
    updates = [row["optimizer_step"] for row in metrics]
    for axis, field, title, xpos, xlabel in (
        (axes[0], "learning_rate", "Adam learning rate", updates, "update call u"),
        (axes[1], "gradient_norm", "x_T gradient norm (before update)", x, "completed updates u-1"),
        (axes[2], "x_T_update_norm", "x_T parameter change from update", updates, "update call u"),
    ):
        axis.plot(xpos, [row[field] for row in metrics], color="#2563eb")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.2)
    axes[1].set_yscale("log")
    axes[2].set_yscale("symlog", linthresh=1e-8)
    figure.suptitle("LR(256)=0; gradient and Adam counter remain required (zeros are not replaced)")
    figure.tight_layout()
    _save_figure(output / "optimizer_diagnostics.png", figure)

    comparison = evidence["comparison"]
    labels = ["new M0", "new raw256", "teacher", "M0 reset", "raw256 reset"]
    statistics = [comparison[key] for key in (
        "m0_reader_statistics", "endpoint_reader_statistics", "teacher_replay_statistics",
        "m0_reset_statistics", "endpoint_reset_statistics",
    )]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    colors = ["#64748b", "#2563eb", "#198754", "#94a3b8", "#93c5fd"]
    axes[0].bar(labels, [row["mean_ce"] for row in statistics], color=colors)
    axes[0].axhline(parent["endpoint_reader_mean_ce"], color="#737373", linestyle="--", label="parent raw256 CE")
    axes[0].set_yscale("symlog", linthresh=1e-6)
    axes[0].set_title("Reader mean CE (no endpoint CE success threshold)")
    axes[0].legend(fontsize=8)
    axes[1].bar(labels, [row["accuracy"] for row in statistics], color=colors)
    axes[1].set_ylim(0, 1.08)
    axes[1].set_title("Accuracy on four fixed views of ONE target")
    for index, row in enumerate(statistics):
        axes[1].text(index, row["accuracy"] + 0.025, f"{round(row['accuracy'] * 4)}/4", ha="center")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    _save_figure(output / "reader_transfer.png", figure)

    images = [(formal_run / "teacher" / "canonical_r11_target01.png", "canonical teacher")]
    images += [(formal_run / "images" / f"step-{step:03d}.png", f"after {step} updates") for step in STEPS]
    figure, axes = plt.subplots(1, 6, figsize=(17, 3.3))
    for axis, (path, title) in zip(axes, images, strict=True):
        with Image.open(path) as image:
            axis.imshow(image.convert("RGB"))
        axis.set_title(title, fontsize=10)
        axis.set_axis_off()
    figure.suptitle("Actual saved RGB | human readability is not a scientific gate")
    figure.tight_layout()
    _save_figure(output / "checkpoint_images.png", figure)


def _readme(evidence: Mapping[str, Any]) -> str:
    comparison = evidence["comparison"]
    endpoint = comparison["endpoint_distance_statistics"]
    reader = comparison["endpoint_reader_statistics"]
    def status(value: bool) -> str:
        return "通过" if value else "失败"

    secondary = comparison["secondary_solver_hypothesis_audit"]
    parent = evidence["parent_absolute_reference"]
    rows = "\n".join(
        f"| {row['optimizer_step']} | {row['mse']:.10g} | {row['mse_ratio_to_m0']:.10g} | {row['teacher_normalized_rmse']:.10g} |"
        for row in evidence["checkpoints"]
    )
    return f"""# R11_new：Teacher-matched initialization 真实结果

训练锁定提交：`{evidence['training_commit']}`。固定 target 1，唯一干预是使用 canonical teacher 反解 initial xT；完整四步 DreamLite 保持冻结，只优化 FP32 xT。

## 结论边界

工程门：**通过**；teacher replay：**通过**；距离门：**{status(comparison['bridge_distance_gate'])}**；Reader 门：**{status(comparison['endpoint_reader_transfer_gate'])}**；本轮 bridge 诊断：**{status(comparison['bridge_diagnostic_gate'])}**。

本轮使用答案相关 teacher 辅助初始化。无论诊断结果如何，`formal_success=false`、`phase2_allowed=false`。Phase 1A 仍为 **6/8**；不能将本轮 target 1 或 canonical R11 结果拼入主线通过数，也不能声称长期 memory/shared writer 已学会。

## 实际门槛与结果

| raw256 指标 | 本轮真实值 | 固定门槛 |
| --- | ---: | --- |
| MSE / 新 M0 MSE | {endpoint['mse_ratio_to_m0']:.10g} | ≤ 0.01 |
| L2 / 新 M0 L2 | {endpoint['l2_distance_ratio_to_m0']:.10g} | ≤ 0.1 |
| RMSE / teacher population std | {endpoint['teacher_normalized_rmse']:.10g} | ≤ 0.1 |
| Reader accuracy | {round(reader['accuracy'] * 4)}/4 | 4/4 |
| Reader mean CE | {reader['mean_ce']:.10g} | 原样报告，不新增 endpoint CE 成功阈值 |

四视图是同一个 target 的固定排列，并非四个独立样本。仅 raw256 判主门；中间 checkpoint、最低 train loss、二级改善均不能挽救主门失败。

## 真实训练轨迹

![距离轨迹](distance_trajectory.png)

Receipt 第 u 行的 loss 在更新前测量，横坐标为 u−1；checkpoint 保存更新后 endpoint，横坐标为 u。新 M0 是新初始化完整跑完四步后、零更新的 endpoint，不是初始 flow state 或 teacher。父实验具有不同 M0，因此只画父实验绝对 MSE 参考线，不作同分母比值对比。

| 已完成更新数 | checkpoint MSE | MSE / 新 M0 | teacher NRMSE |
| --- | ---: | ---: | ---: |
{rows}

![优化器与梯度](optimizer_diagnostics.png)

真实 receipts 共 256 行。Adam 前 128 次学习率 0.05，之后按锁定 cosine 曲线衰减；第 256 次学习率为零，但仍须有有限非零梯度与第 256 次 Adam 计数。图中零值保持为零。

![Reader 结果](reader_transfer.png)

![真实 checkpoint 图像](checkpoint_images.png)

图像仅展示保存的真实 RGB；人类可读性不是本实验优化目标或门槛。

## 二级审计与下一项决策

主决策：`{comparison['decision']}`。

二级决策：`{comparison['secondary_solver_hypothesis_decision']}`。资格 eligible=`{str(secondary['eligible']).lower()}`，通过 passed=`{str(secondary['passed']).lower()}`。

二级仅在技术/teacher 有效且两主门均失败时参与解释，要求 endpoint MSE 严格小于父值 `{parent['endpoint_mse']}`，且 Reader CE 严格小于父值 `{parent['endpoint_reader_mean_ce']}`。只改善一项或相等均不通过，二级不是成功门。

| 距离 / Reader | 另行预注册的候选下一项 |
| --- | --- |
| 通过 / 通过 | 答案无关 learned initializer / inverse writer |
| 通过 / 失败 | teacher 邻域 Reader 鲁棒性 |
| 失败 / 通过 | 原始 QA 目标诊断 |
| 失败 / 失败且二级通过 | 隔离答案无关初始化或 conditioning |
| 失败 / 失败且二级未通过 | reachable-teacher control，区分优化失败与 teacher/flow 不匹配 |

这些是需新预注册的候选，不是此配置内追加实验许可。初始化接近 teacher 不证明全程局部可控，也不证明初始化是主导根因；失败不证明数学不可达。

## 证据与复核范围

Preflight 为一次完整四步前向、一次反向、零更新，不评判 bridge 结果。源目录 inventory、聚合 comparison/RAW 及所有绑定文件字节已检查；本地另从原始 checkpoint tensor 复算距离并按既定容差核验，从原始四选项 logits 复算 CE/正确率，并对原始聚合指标重算主次决策。

所有表格、图表和展示字段统一使用已哈希绑定的原始 record/聚合统计。FP32 reduction 在不同平台或线程实现下可能出现末位差异；本地 CPU 复算值仅用于原有严格容差校验，单独保存在 `checkpoints[*].local_cpu_recomputed_distance_statistics`，不替换远端原值、不改变科学门槛。机器可读文件同时保留两组明确标注的数值。

CUDA 初始化与每 checkpoint 起点闭包由独立 CUDA aggregator 核验。本渲染器仅核对其证据文件字节，不在 CPU 上伪装重跑 CUDA 逐位算术。初始化接近阈值只约束 step 0，其余起点由各自当前 xT 决定；不要求旧 Gaussian M0 或旧优化前缀相等。

逐行学习率的精确合同由原 Linux 运行环境的独立 aggregator 验证并以文件哈希绑定。Windows/Linux 的 cosine 库可产生极小末位差异，渲染器不以另一平台的重算替换原始学习率，也不放宽原有门槛；图表保留原 receipts 数值。

机器可读数据：[training_diagnostics.json](training_diagnostics.json)；逐行原始 Reader 数据和带明确前后坐标的 receipts/checkpoints 均在其中。交付文件哈希：[DELIVERY_MANIFEST.json](DELIVERY_MANIFEST.json)。
"""


def render(
    *, preflight_root: Path, formal_root: Path, aggregation_root: Path, output_dir: Path,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    output = output_dir.absolute()
    _require(not output.exists() and not output.is_symlink(), "A fresh nonexistent output directory is required.")
    for root in (preflight_root, formal_root, aggregation_root):
        _require(not output.resolve().is_relative_to(root.resolve()), "Output must not be inside immutable input roots.")
    evidence = collect_evidence(
        preflight_root=preflight_root, formal_root=formal_root, aggregation_root=aggregation_root,
        expected_commit=expected_commit,
    )
    output.mkdir(parents=True, exist_ok=False)
    _plot(evidence, formal_root.resolve() / "run", output)
    _write_json(output / "training_diagnostics.json", evidence)
    (output / "README.md").write_text(_readme(evidence), encoding="utf-8", newline="\n")
    environment = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "platform": platform.platform(), "torch": str(torch.__version__),
        "matplotlib": matplotlib.__version__, "numpy": np.__version__, "pillow": Image.__version__,
        "matplotlib_backend": str(matplotlib.get_backend()),
    }
    _write_json(output / "render_environment.json", environment)
    manifest = {
        "schema": SCHEMA, "training_commit": evidence["training_commit"],
        "source_comparison_sha256": evidence["source_comparison_sha256"],
        "source_raw_artifacts_sha256": evidence["source_raw_artifacts_sha256"],
        "generator": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": sha256(Path(__file__))},
        "verification_scope": evidence["verification_scope"], "formal_success": False, "phase2_allowed": False,
        "artifacts": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                      for path in sorted(output.iterdir()) if path.is_file()},
    }
    _write_json(output / "DELIVERY_MANIFEST.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--aggregation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args(argv)
    manifest = render(**vars(args))
    print(json.dumps({"status": "rendered_and_verified", "output_dir": str(args.output_dir.resolve()),
                      "training_commit": manifest["training_commit"], "formal_success": False,
                      "phase2_allowed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
