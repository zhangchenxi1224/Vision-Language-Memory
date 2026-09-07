"""Verify, summarize, and render the delivered FP32 L-BFGS trust-region run."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from vision_memory.training import r11_new_fp32_lbfgs_trust_region as core  # noqa: E402


ARCHIVE_NAME = "r11-new-lbfgs-f995248-20260907-round01-full.tar.gz"
ARCHIVE_BYTES = 702_895_016
ARCHIVE_SHA256 = "64acb0c4bc16b8aa5ad0fc4eafcb064c11b84f12c031e3129cb377856f5ef005"
PREFLIGHT = "r11-new-lbfgs-f995248-20260907-round01-preflight"
FORMAL = "r11-new-lbfgs-f995248-20260907-round01-formal"
READINESS = "r11-new-lbfgs-f995248-20260907-readiness-remote"
PREFLIGHT_AUDIT = "r11-new-lbfgs-f995248-20260907-round01-preflight-audit.json"
FORMAL_AUDIT = "r11-new-lbfgs-f995248-20260907-round01-formal-audit.json"
PREFLIGHT_AUDIT_SHA256 = "a9cfa61f4edc391ae7185fd95048a52cc9921cf9b078ccd8e938d3c2c036b1eb"
FORMAL_AUDIT_SHA256 = "04004effc2b779ddbaed3a7429ef3a4fd8748fb7d1c03f5210ea28a0727852ac"
JUNIT_SHA256 = "fa9c0d47d860f3450435ef105c0a7a3e07d7e71eb5282bc6268546414704cc72"
EXPERIMENT_COMMIT = "f995248e5d730a478d2398a49c4e533fd05aa2ae"

PART_SPECS = (
    ("00", 47_185_920, "a4ea40df8ac3c087fb93f2eb89642364348e1e6e70cedaf964f4e0a0ed480703"),
    ("01", 47_185_920, "8f51d5ada766bead104789c0cc9b91745e8613158ec227a6fadebce548d3443f"),
    ("02", 47_185_920, "5fc5b5b0573907dacb49dac121ef254077ebcb279c4d9dc65f81e2e2ec30256b"),
    ("03", 47_185_920, "0fde5254850077c0f907f93973fcc74773642ed8bd3973abbcaa5708f4583ccc"),
    ("04", 47_185_920, "6f354d36b36f7abdf3b1f8ceae7dd3449507442b78a87e2653c660c01682a04e"),
    ("05", 47_185_920, "b5e4268952a27b3404b8d18abfaca4b186e3546b22cb0d300062eb14c69817be"),
    ("06", 47_185_920, "d8c665cc15210b32315d903b81920b131fc1c2b270e8d2c8b283809cb828b35b"),
    ("07", 47_185_920, "54e7aa3600355c79cf68f839e87d41c30ffbf46aa870b863319f01bfd60185b2"),
    ("08", 47_185_920, "f0aee5213569b25794b83375279164801ddee3b26eb0bc6811488b742d22305c"),
    ("09", 47_185_920, "217c5575348b90f5494d1c56dfe2bd8903584a6ffbf186f9db5920e5ae8d7e43"),
    ("10", 47_185_920, "d080afb470eab2570cdc01d388749bda61c046eeed30d6c9febadbe611aa19c5"),
    ("11", 47_185_920, "62104e306e4ca8e4b9642dd67120578e6dfb2cb3379816b012a8aaf9be0c477c"),
    ("12", 47_185_920, "e74339813e696e395489976f4662d41f7bbabc227862674fe5fa6a335a76f1bf"),
    ("13", 47_185_920, "0a02af0227605e514460f88d2afb42545ceb806357b349afc92c1ffad40bc801"),
    ("14", 42_292_136, "483a545a801a38f4d04e23794eb9d93a2fce5eadf58d6851feaadb6e4ab26404"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise ValueError(f"Missing terminal newline: {path}")
    rows = [json.loads(line) for line in text.splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects: {path}")
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def part_path(index: str) -> Path:
    return ROOT / "raw" / f"{ARCHIVE_NAME}.part-{index}"


def reconstruct_archive(destination: Path) -> list[dict[str, Any]]:
    expected_paths = [part_path(index) for index, _, _ in PART_SPECS]
    observed_paths = sorted((ROOT / "raw").glob(f"{ARCHIVE_NAME}.part-*"))
    if observed_paths != expected_paths:
        raise ValueError("Raw archive part set is incomplete or contains unexpected files.")
    records: list[dict[str, Any]] = []
    complete_hash = hashlib.sha256()
    complete_bytes = 0
    with destination.open("xb") as output:
        for (index, expected_bytes, expected_hash), path in zip(PART_SPECS, expected_paths, strict=True):
            observed_hash = sha256_file(path)
            if path.stat().st_size != expected_bytes or observed_hash != expected_hash:
                raise ValueError(f"Archive part {index} failed its bound size or SHA-256.")
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    output.write(chunk)
                    complete_hash.update(chunk)
                    complete_bytes += len(chunk)
            records.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": expected_bytes,
                    "sha256": expected_hash,
                }
            )
    if complete_bytes != ARCHIVE_BYTES or complete_hash.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Parts do not reconstruct the exact remote archive.")
    return records


def validate_members(bundle: tarfile.TarFile) -> None:
    allowed_roots = {PREFLIGHT, FORMAL, READINESS, PREFLIGHT_AUDIT, FORMAL_AUDIT}
    for member in bundle.getmembers():
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or not path.parts
            or ".." in path.parts
            or path.parts[0] not in allowed_roots
            or member.issym()
            or member.islnk()
            or member.isdev()
        ):
            raise ValueError(f"Unsafe archive member: {member.name}")


def verify_inventory(root: Path) -> dict[str, Any]:
    inventory_path = root / "artifact_inventory.json"
    inventory = load_json(inventory_path)
    listed: set[str] = set()
    for item in inventory["artifacts"]:
        relative = item["path"]
        path = PurePosixPath(relative)
        if relative in listed or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe inventory path: {relative}")
        artifact = root.joinpath(*path.parts)
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or artifact.stat().st_size != item["bytes"]
            or sha256_file(artifact) != item["sha256"]
        ):
            raise ValueError(f"Inventory mismatch: {root.name}/{relative}")
        listed.add(relative)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != inventory_path
    }
    if observed != listed or inventory["artifact_count"] != len(listed):
        raise ValueError(f"Incomplete inventory: {root.name}")
    return {
        "passed": True,
        "artifact_count": len(listed),
        "inventory_sha256": sha256_file(inventory_path),
    }


def portable_trace_audit(
    root: Path,
    saved_audit: dict[str, Any],
    inventory: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    iteration_rows = read_jsonl(root / "iteration_metrics.jsonl")
    candidate_rows = read_jsonl(root / "candidate_metrics.jsonl")
    summary = core.summarize_trace(iteration_rows, candidate_rows, config)
    result = load_json(root / "result.json")
    classification = None if result["mode"] == "technical-preflight" else core.classify_outcome(summary, config)
    if not (
        saved_audit["passed"] is True
        and saved_audit["artifact_count"] == inventory["artifact_count"]
        and saved_audit["git_commit"] == EXPERIMENT_COMMIT
        and saved_audit["mode"] == result["mode"]
        and saved_audit["classification"] == classification == result["classification"]
        and saved_audit["trace_summary"] == summary == result["trace_summary"]
        and saved_audit["fixed_target_optimization_success"]
        == result["fixed_target_optimization_success"]
        and saved_audit["formal_success"] is False
        and result["formal_success"] is False
        and saved_audit["phase2_allowed"] is False
        and result["phase2_allowed"] is False
        and result["final_replay_bitwise_equal"] is True
    ):
        raise ValueError(f"Portable trace or signed-audit binding drift: {root.name}")
    return {
        "passed": True,
        "scope": "inventory_hashes_and_jsonl_trace_semantics",
        "tensor_arithmetic_recomputed": False,
        "authoritative_locked_linux_tensor_audit_sha256": (
            PREFLIGHT_AUDIT_SHA256 if result["mode"] == "technical-preflight" else FORMAL_AUDIT_SHA256
        ),
        "trace_summary": summary,
    }


def verify_junit(path: Path) -> dict[str, Any]:
    if sha256_file(path) != JUNIT_SHA256:
        raise ValueError("Remote full-suite JUnit SHA-256 drift.")
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise ValueError("JUnit has no testsuite.")
    summary = {
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib.get("failures", "0")),
        "errors": int(suite.attrib.get("errors", "0")),
        "skipped": int(suite.attrib.get("skipped", "0")),
        "time_seconds": float(suite.attrib["time"]),
        "sha256": JUNIT_SHA256,
    }
    if summary != {
        "tests": 1506,
        "failures": 0,
        "errors": 0,
        "skipped": 5,
        "time_seconds": summary["time_seconds"],
        "sha256": JUNIT_SHA256,
    }:
        raise ValueError(f"Unexpected remote test result: {summary}")
    summary["passed"] = True
    summary["collected_items"] = 1418
    summary["ordinary_passed"] = 1413
    summary["subtests_passed"] = 88
    return summary


def copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size or sha256_file(destination) != sha256_file(source):
            raise ValueError(f"Existing selected artifact drifted: {destination}")
        return
    shutil.copy2(source, destination)


def copy_selected_run(source: Path, destination: Path) -> list[str]:
    selected: list[Path] = []
    readable_suffixes = {".json", ".jsonl", ".md", ".txt", ".log"}
    selected.extend(
        path for path in source.iterdir() if path.is_file() and path.suffix.casefold() in readable_suffixes
    )
    for folder in ("target", "final", "accepted_images"):
        root = source / folder
        if root.is_dir():
            selected.extend(path for path in root.rglob("*.png") if path.is_file())
    copied: list[str] = []
    for path in sorted(selected):
        relative = path.relative_to(source)
        copy_exact(path, destination / relative)
        copied.append(relative.as_posix())
    return copied


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def trajectory(summary: dict[str, Any]) -> list[float]:
    trace = summary["formal_result"]["trace_summary"]
    initial = float(trace["initial_loss"])
    return [1.0, *(float(value) / initial for value in trace["accepted_loss_sequence"])]


def image_statistics(path: Path, target: np.ndarray) -> dict[str, Any]:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if pixels.shape != target.shape:
        raise ValueError(f"Image shape drift: {path}")
    delta = pixels - target
    horizontal = np.abs(pixels[:, 1:] - pixels[:, :-1]).mean()
    vertical = np.abs(pixels[1:, :] - pixels[:-1, :]).mean()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "shape": list(pixels.shape),
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "mean_total_variation": float((horizontal + vertical) / 2.0),
        "difference_from_fixed_target_png": {
            "pixel_mae": float(np.abs(delta).mean()),
            "pixel_mse": float(np.square(delta).mean()),
            "pixel_max_abs": float(np.abs(delta).max()),
        },
    }


def make_loss_figure(
    horizon: list[float], prp: list[float], lbfgs: list[float], destination: Path
) -> None:
    figure, axis = plt.subplots(figsize=(9.5, 5.7))
    axis.semilogy(range(len(horizon)), horizon, label="Normalized -gradient (Horizon-128)", linewidth=1.8)
    axis.semilogy(range(len(prp)), prp, label="PRP+", linewidth=1.8)
    axis.semilogy(range(len(lbfgs)), lbfgs, label="L-BFGS (m=10)", linewidth=2.4)
    axis.axhline(0.01, color="black", linestyle="--", linewidth=1.2, label="Fixed target gate = 0.01")
    axis.axhline(0.027161627667846616, color="#888888", linestyle=":", linewidth=1.2, label="5% vs PRP+ gate")
    axis.set(xlabel="Accepted updates", ylabel="Loss ratio to initial plateau", title="Matched-budget fixed-target optimization")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def make_dynamics_figure(rows: list[dict[str, Any]], destination: Path) -> None:
    updates = np.asarray([int(row["iteration"]) + 1 for row in rows])
    gradients = np.asarray([float(row["gradient_norm"]) for row in rows])
    radii = np.asarray([float(row["selected_radius_l2"]) for row in rows])
    cosines = np.asarray([
        np.nan if row.get("curvature_cosine") is None else float(row["curvature_cosine"]) for row in rows
    ])
    scales = np.asarray([
        np.nan
        if row.get("initial_inverse_hessian_scale") is None
        else float(row["initial_inverse_hessian_scale"])
        for row in rows
    ])
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    axes[0, 0].semilogy(updates, gradients, color="#1f77b4")
    axes[0, 0].set_ylabel("Gradient norm")
    axes[0, 1].semilogy(updates, radii, color="#ff7f0e")
    axes[0, 1].set_ylabel("Selected radius L2")
    axes[1, 0].plot(updates, cosines, color="#2ca02c")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_ylabel("Curvature cosine")
    axes[1, 1].semilogy(updates, scales, color="#9467bd")
    axes[1, 1].set_ylabel("Initial inverse-Hessian scale")
    for axis in axes.flat:
        axis.set_xlabel("Accepted update")
        axis.grid(True, which="both", alpha=0.25)
    figure.suptitle("L-BFGS trust-region dynamics")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def make_matched_figure(
    horizon: list[float], prp: list[float], lbfgs: list[float], destination: Path
) -> None:
    updates = [1, 8, 16, 32, 64]
    figure, axis = plt.subplots(figsize=(9.5, 5.7))
    axis.semilogy(updates, [horizon[index] for index in updates], marker="o", label="Normalized -gradient")
    axis.semilogy(updates, [prp[index] for index in updates], marker="o", label="PRP+")
    axis.semilogy(updates, [lbfgs[index] for index in updates], marker="o", label="L-BFGS")
    axis.axhline(0.01, color="black", linestyle="--", linewidth=1.2)
    axis.set(xlabel="Matched accepted updates", ylabel="Loss ratio", title="Same-update comparison")
    axis.set_xticks(updates)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def make_final_figure(horizon: float, prp: float, lbfgs: float, destination: Path) -> None:
    labels = ["Horizon-128\n(update 128)", "PRP+\n(update 64)", "L-BFGS\n(update 115)"]
    values = [horizon, prp, lbfgs]
    figure, axis = plt.subplots(figsize=(8, 5.5))
    bars = axis.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b"])
    axis.set_yscale("log")
    axis.axhline(0.01, color="black", linestyle="--", linewidth=1.2, label="gate = 0.01")
    axis.set(ylabel="Final loss ratio", title="Fixed-target final outcomes")
    axis.grid(True, axis="y", which="both", alpha=0.25)
    axis.legend()
    for bar, value in zip(bars, values, strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value * 1.08, f"{value:.6f}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def make_montage(destination: Path) -> list[str]:
    formal = ROOT / "formal"
    items: list[tuple[str, Path]] = [
        ("Fixed target", formal / "target" / "fixed_parent_target.png"),
        ("FP32 teacher", formal / "target" / "fp32_teacher.png"),
    ]
    images = sorted((formal / "accepted_images").glob("iteration-*.png"))
    if images:
        requested = [0, 15, 31, 63, 95, 114]
        by_iteration = {int(path.stem.split("-")[-1]): path for path in images}
        available = sorted(by_iteration)
        chosen: set[int] = set()
        for target in requested:
            nearest = min(available, key=lambda value: (abs(value - target), value))
            chosen.add(nearest)
        items.extend((f"Accepted update {value + 1}", by_iteration[value]) for value in sorted(chosen))
    items.append(("Final replay", formal / "final" / "final.png"))
    for _, path in items:
        if not path.is_file():
            raise ValueError(f"Missing montage image: {path}")
    columns = 3
    rows = (len(items) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for axis, (label, path) in zip(axes_array, items, strict=False):
        axis.imshow(Image.open(path).convert("RGB"))
        axis.set_title(label)
        axis.axis("off")
    for axis in axes_array[len(items) :]:
        axis.axis("off")
    figure.suptitle("Decoded image trajectory (visual evidence only)")
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return [path.relative_to(ROOT).as_posix() for _, path in items]


def render_readme(summary: dict[str, Any]) -> None:
    formal = summary["formal_result"]
    trace = formal["trace_summary"]
    comparison = summary["comparison_to_parents"]
    diagnostics = summary["optimization_diagnostics"]
    tests = summary["remote_full_suite"]
    text = f"""# R11_new FP32 L-BFGS trust-region：正式结果交付

## 核心结论

L-BFGS 在**完全固定的单 target oracle 口径**下真实跨过了预注册阈值：第 `{diagnostics['first_fixed_target_success_update']}` 次接受更新后，loss ratio 降至 `{trace['final_loss_ratio_to_plateau']:.10f}`，分类为 `{formal['classification']}`。相对 PRP+ 最终 ratio 降低 `{comparison['relative_reduction_vs_prpplus_final']:.2%}`，相对 Horizon-128 最终值降低 `{comparison['relative_reduction_vs_horizon128_final']:.2%}`。

这证明固定目标位于当前 DreamLite 参数化的可达域内，之前长尾主要是搜索方向/局部曲率问题，而不是断梯度或绝对不可达。但这**仍不是 Picture Memory 正式成功**：本轮没有训练 event→state 共享 writer，没有 Reader、multi-target、multi-seed、ID/OOD、SET/overwrite 或因果替图评测。因此 `formal_success=false`、`phase2_allowed=false` 保持不变。

## 实验口径与结果

| 项目 | Horizon-128 | PRP+ | L-BFGS 本轮 |
| --- | ---: | ---: | ---: |
| 搜索方向 | 单位负梯度 | PRP+ 共轭方向 | L-BFGS two-loop，m=10 |
| 最大预算 | 128 更新 / 771 前向 | 128 更新 / 771 前向 | 128 更新 / 771 前向 |
| 实际执行 | 128 更新 / 771 前向 | 64 更新 / 393 前向 | `{trace['accepted_update_count']}` 更新 / `{formal['counters']['full_chain_forward_calls']}` 前向 |
| 最终 loss | `9.4914816e-7` | `5.1399650e-7` | `{trace['final_loss']:.10e}` |
| 最终 ratio | `0.05279661` | `0.02859119` | `{trace['final_loss_ratio_to_plateau']:.10f}` |
| 停止原因 | 最大更新数 | 无可接受候选 | `{trace['stop_reason']}` |
| ratio ≤ 0.01 | 否 | 否 | **是** |

唯一科学变量是首步后的搜索方向。target、原始 plateau、DreamLite 快照、FP32 map、endpoint MSE、五个半径 `(0.1, 0.03, 0.01, 0.003, 0.001)`、候选选择、`1e-6` 接受规则、128-update/771-forward 上限及 0.01 门槛均未改变。没有 optimizer、gradient clipping 或 Reader forward。

## 关键观察

- `{trace['accepted_update_count']}/{trace['iteration_count']}` 次更新被接受，loss 全程严格单调；`{diagnostics['curvature_pairs_accepted']}` 个后续曲率对全部通过门控，历史最大长度 `{trace['maximum_history_size_used']}`，除首步初始化外无重启。
- 首次进入 ratio≤0.1 在 update `{diagnostics['first_strong_capture_update']}`；首次达到 ratio≤0.01 在 update `{diagnostics['first_fixed_target_success_update']}`。
- 最终 ratio 是门槛的 `{comparison['fraction_of_success_threshold']:.4f}` 倍，属于刚跨线而非大裕量成功；下一阶段需要多 target 与多 seed 验证，不能把单点结果外推。
- 解码图像依旧近似人类不可读的灰色纹理。图像只能展示同一 latent 优化轨迹，不能替代 Reader 正确率、因果替图或泛化证据。
- 正式运行期间 Inspire 控制 WebSocket 曾断开，但后台迭代继续增长，最终终态、运行锁释放和 255 个产物独立审计均正常；不存在重复启动或残缺结果。

## 测试、审计与完整性

- 远端锁定环境全量测试：`{tests['ordinary_passed']} passed / {tests['skipped']} skipped / {tests['failures']} failed`，另有 `{tests['subtests_passed']}` subtests passed；跳过项仅为节点无指定 TrueType 字体。
- 技术预检审计：27 个产物，1/1 更新接受；正式审计：255 个产物，分类、曲率历史、two-loop、候选、checkpoint、inventory、模型快照与最终 bitwise replay 全部通过。
- Windows 交付脚本可移植地重算 inventory、JSONL 候选选择、接受规则、状态链和分类，并校验远端权威审计 SHA；逐张量浮点重放必须使用锁定 Linux/PyTorch 环境，避免把跨平台归约差异伪装成 bitwise 一致。
- 完整原始归档：`{ARCHIVE_BYTES}` bytes，SHA-256 `{ARCHIVE_SHA256}`；15 个 GitHub 分片各自有 size/SHA 绑定。
- 运行 commit：`{EXPERIMENT_COMMIT}`。其中 `9196f9a…` 冻结 L-BFGS 方案，`f995248…` 仅修复旧 R3 DAG 对 Linux venv 符号链接的等价路径校验，发生在首次 DreamLite forward 之前，不改变科学口径。

## 下一步主线

停止继续微调单 target 局部优化器。下一最小判别实验应把本轮证明可达的 oracle latent 当作教师信号，预注册并训练**一个跨多个事件/目标共享、推理时非 oracle 的 writer**；先检验 held-in multi-target 写入能力，再接冻结 Reader 与因果替图，最后扩展到 multi-seed、ID/OOD、SET/overwrite。只有这条链路通过，才可讨论 Picture Memory 成功。

## 交付内容

- `raw/`：15 个完整原始归档分片及重组清单。
- `preflight/`、`formal/`：配置、环境、日志、指标、审计 inventory 与代表性图片。
- `audits/`、`readiness/`：两份独立审计和远端全量测试 JUnit。
- `derived/`：重算摘要、逐轮 CSV 与同 update 对照表。
- `figures/`：loss 对照、L-BFGS 动力学、matched-update、最终值与图像轨迹。
- `render_results.py`：从分片重组并重新验证上述全部交付。

复核命令：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python reports/r11-new-fp32-lbfgs-trust-region-results-20260907/render_results.py
```
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8", newline="\n")


def delivery_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if (
            path.is_file()
            and path.name != "DELIVERY_MANIFEST.json"
            and "__pycache__" not in path.parts
            and path.suffix.casefold() != ".pyc"
        ):
            files.append(
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema": "vision_memory.r11-new-fp32-lbfgs-trust-region-delivery.v1",
        "generated_from_experiment_commit": EXPERIMENT_COMMIT,
        "classification": summary["formal_result"]["classification"],
        "fixed_target_optimization_success": True,
        "formal_picture_memory_success": False,
        "phase2_allowed": False,
        "file_count_excluding_manifest": len(files),
        "files": files,
    }


def verify_delivery_manifest(path: Path) -> str:
    manifest = load_json(path)
    listed = {item["path"]: item for item in manifest["files"]}
    if len(listed) != manifest["file_count_excluding_manifest"]:
        raise ValueError("Delivery manifest has duplicate or miscounted paths.")
    observed = {
        item.relative_to(ROOT).as_posix(): item
        for item in ROOT.rglob("*")
        if item.is_file()
        and item != path
        and "__pycache__" not in item.parts
        and item.suffix.casefold() != ".pyc"
    }
    if set(observed) != set(listed):
        raise ValueError("Delivery manifest file coverage drift.")
    for relative, item in observed.items():
        record = listed[relative]
        if item.stat().st_size != record["bytes"] or sha256_file(item) != record["sha256"]:
            raise ValueError(f"Delivery manifest hash drift: {relative}")
    return sha256_file(path)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="vlm-r11-lbfgs-delivery-") as temporary:
        temporary_root = Path(temporary)
        archive = temporary_root / ARCHIVE_NAME
        parts = reconstruct_archive(archive)
        with tarfile.open(archive, "r:gz") as bundle:
            validate_members(bundle)
            bundle.extractall(temporary_root, filter="data")

        extracted_preflight = temporary_root / PREFLIGHT
        extracted_formal = temporary_root / FORMAL
        extracted_readiness = temporary_root / READINESS
        preflight_inventory = verify_inventory(extracted_preflight)
        formal_inventory = verify_inventory(extracted_formal)

        preflight_saved = load_json(temporary_root / PREFLIGHT_AUDIT)
        formal_saved = load_json(temporary_root / FORMAL_AUDIT)
        if sha256_file(temporary_root / PREFLIGHT_AUDIT) != PREFLIGHT_AUDIT_SHA256:
            raise ValueError("Preflight audit SHA-256 drift.")
        if sha256_file(temporary_root / FORMAL_AUDIT) != FORMAL_AUDIT_SHA256:
            raise ValueError("Formal audit SHA-256 drift.")
        config = core.load_config()
        preflight_portable = portable_trace_audit(
            extracted_preflight,
            preflight_saved,
            preflight_inventory,
            config,
        )
        formal_portable = portable_trace_audit(
            extracted_formal,
            formal_saved,
            formal_inventory,
            config,
        )
        if not (
            formal_saved["passed"] is True
            and formal_saved["classification"] == "fixed_target_trust_region_success"
            and formal_saved["fixed_target_optimization_success"] is True
            and formal_saved["formal_success"] is False
            and formal_saved["phase2_allowed"] is False
        ):
            raise ValueError("Formal scientific boundary drift.")

        junit = verify_junit(extracted_readiness / "full_suite.junit.xml")
        preflight_selected = copy_selected_run(extracted_preflight, ROOT / "preflight")
        formal_selected = copy_selected_run(extracted_formal, ROOT / "formal")
        copy_exact(temporary_root / PREFLIGHT_AUDIT, ROOT / "audits" / PREFLIGHT_AUDIT)
        copy_exact(temporary_root / FORMAL_AUDIT, ROOT / "audits" / FORMAL_AUDIT)
        copy_exact(extracted_readiness / "full_suite.junit.xml", ROOT / "readiness" / "full_suite.junit.xml")

        rows = read_jsonl(extracted_formal / "iteration_metrics.jsonl")
        if len(rows) != 115:
            raise ValueError("Unexpected L-BFGS iteration count.")
        iteration_fields = [
            "update",
            "current_loss",
            "current_loss_ratio_to_plateau",
            "selected_loss",
            "selected_loss_ratio_to_plateau",
            "selected_radius_l2",
            "relative_improvement",
            "gradient_norm",
            "gradient_nonzero_fraction",
            "analytic_directional_derivative",
            "negative_gradient_cosine",
            "curvature_pair_accepted",
            "curvature_s_dot_y",
            "curvature_cosine",
            "history_size_used",
            "initial_inverse_hessian_scale",
            "restart_reason",
        ]
        csv_rows = [
            {"update": int(row["iteration"]) + 1, **{name: row.get(name) for name in iteration_fields[1:]}}
            for row in rows
        ]
        write_csv(ROOT / "derived" / "iteration_metrics.csv", csv_rows, iteration_fields)

        horizon_summary = load_json(
            REPO / "reports/r11-new-fp32-trust-region-horizon128-results-20260907/derived/summary.json"
        )
        prp_summary = load_json(
            REPO / "reports/r11-new-fp32-prpplus-trust-region-results-20260907/derived/summary.json"
        )
        horizon = trajectory(horizon_summary)
        prp = trajectory(prp_summary)
        formal_result = load_json(extracted_formal / "result.json")
        initial_loss = float(formal_result["trace_summary"]["initial_loss"])
        lbfgs = [1.0, *(float(row["selected_loss"]) / initial_loss for row in rows)]
        if lbfgs[-1] != formal_result["trace_summary"]["final_loss_ratio_to_plateau"]:
            raise ValueError("Recomputed final ratio drift.")

        matched_updates = [1, 8, 16, 32, 64]
        matched = [
            {
                "accepted_update": update,
                "normalized_negative_gradient": horizon[update],
                "prpplus": prp[update],
                "lbfgs": lbfgs[update],
            }
            for update in matched_updates
        ]
        write_csv(
            ROOT / "derived" / "matched_update_comparison.csv",
            matched,
            ["accepted_update", "normalized_negative_gradient", "prpplus", "lbfgs"],
        )

        first_strong = next(index for index, ratio in enumerate(lbfgs[1:], start=1) if ratio <= 0.1)
        first_success = next(index for index, ratio in enumerate(lbfgs[1:], start=1) if ratio <= 0.01)
        radius_counts = Counter(str(row["selected_radius_l2"]) for row in rows)
        curvature_accepted = sum(bool(row["curvature_pair_accepted"]) for row in rows)
        if curvature_accepted != 114 or first_success != 115:
            raise ValueError("Unexpected L-BFGS trajectory classification details.")

        figures = ROOT / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        make_loss_figure(horizon, prp, lbfgs, figures / "lbfgs_loss_comparison.png")
        make_dynamics_figure(rows, figures / "lbfgs_dynamics.png")
        make_matched_figure(horizon, prp, lbfgs, figures / "lbfgs_matched_updates.png")
        make_final_figure(horizon[-1], prp[-1], lbfgs[-1], figures / "lbfgs_final_comparison.png")
        montage_sources = make_montage(figures / "lbfgs_montage.png")

        target_path = ROOT / "formal" / "target" / "fixed_parent_target.png"
        target_pixels = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.float32) / 255.0
        image_paths = {
            "fixed_target": target_path,
            "fp32_teacher": ROOT / "formal" / "target" / "fp32_teacher.png",
            "final_replay": ROOT / "formal" / "final" / "final.png",
        }
        image_stats = {name: image_statistics(path, target_pixels) for name, path in image_paths.items()}

        final_ratio = float(formal_result["trace_summary"]["final_loss_ratio_to_plateau"])
        prp_final = float(prp[-1])
        horizon_final = float(horizon[-1])
        summary = {
            "schema": "vision_memory.r11-new-fp32-lbfgs-trust-region-derived-summary.v1",
            "archive": {
                "remote_complete_archive_bytes": ARCHIVE_BYTES,
                "remote_complete_archive_sha256": ARCHIVE_SHA256,
                "github_delivery_parts": parts,
            },
            "experiment_commit": EXPERIMENT_COMMIT,
            "remote_full_suite": junit,
            "independent_audits": {
                "authoritative_locked_linux_preflight": preflight_saved,
                "authoritative_locked_linux_formal": formal_saved,
                "portable_preflight": preflight_portable,
                "portable_formal": formal_portable,
                "saved_preflight_sha256": PREFLIGHT_AUDIT_SHA256,
                "saved_formal_sha256": FORMAL_AUDIT_SHA256,
                "preflight_inventory": preflight_inventory,
                "formal_inventory": formal_inventory,
            },
            "selected_delivery": {
                "preflight_files": preflight_selected,
                "formal_files": formal_selected,
                "montage_sources": montage_sources,
            },
            "formal_result": formal_result,
            "comparison_to_parents": {
                "horizon128_final_ratio": horizon_final,
                "prpplus_final_ratio": prp_final,
                "lbfgs_final_ratio": final_ratio,
                "relative_reduction_vs_horizon128_final": (horizon_final - final_ratio) / horizon_final,
                "relative_reduction_vs_prpplus_final": (prp_final - final_ratio) / prp_final,
                "fraction_of_success_threshold": final_ratio / 0.01,
                "matched_update_ratios": {str(row["accepted_update"]): row for row in matched},
            },
            "optimization_diagnostics": {
                "first_strong_capture_update": first_strong,
                "first_fixed_target_success_update": first_success,
                "curvature_pairs_accepted": curvature_accepted,
                "curvature_pairs_rejected": len(rows) - 1 - curvature_accepted,
                "selected_radius_counts": dict(sorted(radius_counts.items())),
                "all_gradients_dense": all(float(row["gradient_nonzero_fraction"]) == 1.0 for row in rows),
                "all_directions_descent": all(float(row["analytic_directional_derivative"]) < 0.0 for row in rows),
                "all_updates_accepted": all(bool(row["accepted"]) for row in rows),
            },
            "image_statistics": image_stats,
            "scientific_boundary": {
                "single_fixed_oracle_target_only": True,
                "fixed_target_optimization_success": True,
                "shared_writer_evaluated": False,
                "reader_evaluated": False,
                "multi_target_evaluated": False,
                "multi_seed_evaluated": False,
                "id_ood_evaluated": False,
                "set_overwrite_evaluated": False,
                "causal_image_swap_evaluated": False,
                "picture_memory_success": False,
            },
        }
        atomic_json(ROOT / "raw" / "archive_parts.json", summary["archive"])
        atomic_json(ROOT / "derived" / "summary.json", summary)
        render_readme(summary)
        atomic_json(ROOT / "DELIVERY_MANIFEST.json", delivery_manifest(summary))
        manifest_sha256 = verify_delivery_manifest(ROOT / "DELIVERY_MANIFEST.json")
        print(
            json.dumps(
                {
                    "passed": True,
                    "classification": formal_result["classification"],
                    "final_loss_ratio_to_plateau": final_ratio,
                    "fixed_target_optimization_success": True,
                    "formal_picture_memory_success": False,
                    "archive_sha256": ARCHIVE_SHA256,
                    "delivery_manifest_sha256": manifest_sha256,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
