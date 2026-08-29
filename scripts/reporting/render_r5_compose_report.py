"""Render the complete R5-Compose evidence package and loss/causal figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCHEMA = "vision_memory.r5-compose-final-report.v1"


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _rows(path: Path, *, kind: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict) and (kind is None or value.get("kind") == kind):
            values.append(value)
    return values


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if not math.isfinite(number) else f"{number:.{digits}f}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _gradient_figure(report: Mapping[str, Any], output: Path) -> None:
    records = report.get("records", [])
    groups = {
        "K1": [float(row["cosine_to_full"]) for row in records if row.get("count_group") == "K1"],
        "K2": [float(row["cosine_to_full"]) for row in records if row.get("count_group") == "K2"],
    }
    if not all(groups.values()):
        return
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.boxplot([groups["K1"], groups["K2"]], tick_labels=["K=1", "K=2"], showmeans=True)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axhline(0.3, color="#d97706", linestyle="--", linewidth=1.0, label="preferred median target")
    axis.set_ylabel("cosine(approximate gradient, full gradient)")
    axis.set_title("R5 gradient fidelity")
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _pilot_figure(pilots: Sequence[Mapping[str, Any]], output: Path) -> None:
    eligible = [value for value in pilots if isinstance(value.get("pilot_selection"), Mapping)]
    if not eligible:
        return
    labels = [f"{value['persistent_state']} h{value['tbptt_horizon']}" for value in eligible]
    m0 = [float(value["pilot_selection"]["delayed_mechanism_ce"]["m0"]) for value in eligible]
    endpoint = [float(value["pilot_selection"]["delayed_mechanism_ce"]["endpoint"]) for value in eligible]
    positions = list(range(len(labels)))
    fig, axis = plt.subplots(figsize=(max(7.0, len(labels) * 1.8), 4.2))
    width = 0.36
    axis.bar([value - width / 2 for value in positions], m0, width=width, label="M0")
    axis.bar([value + width / 2 for value in positions], endpoint, width=width, label="EMA step128")
    axis.set_xticks(positions, labels, rotation=15, ha="right")
    axis.set_ylabel("mechanism-select mean CE (lower is better)")
    axis.set_title("R5 2×2 pilot endpoints")
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _loss_figure(root: Path, seeds: Sequence[int], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.4, 4.2))
    plotted = False
    for seed in seeds:
        records = _rows(root / "runs" / f"main-seed{seed}" / "metrics.jsonl", kind="optimizer_step")
        if not records:
            continue
        steps = [int(record["optimizer_step"]) for record in records]
        losses = [float(record["loss_mean"]) for record in records]
        axis.plot(steps, losses, alpha=0.35, linewidth=0.8, label=f"seed {seed} raw step loss")
        window = 16
        smoothed = [sum(losses[max(0, index - window + 1) : index + 1]) / min(index + 1, window) for index in range(len(losses))]
        axis.plot(steps, smoothed, linewidth=1.8, label=f"seed {seed} moving mean ({window})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("mean delayed-query training CE")
    axis.set_title("R5 main training loss")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _causal_figure(evaluations: Sequence[tuple[int, Mapping[str, Any]]], output: Path) -> None:
    controls = ("normal", "reset", "cross_episode_swap", "temporal_swap", "updater_disabled")
    if not evaluations:
        return
    fig, axes = plt.subplots(1, len(evaluations), figsize=(6.0 * len(evaluations), 4.2), squeeze=False)
    for axis, (seed, value) in zip(axes[0], evaluations, strict=True):
        metrics = value.get("by_checkpoint_suite_condition", {})
        values = [metrics.get(f"ema_step640|mechanism_final_128|{condition}", {}).get("mean_ce") for condition in controls]
        if any(item is None for item in values):
            axis.set_visible(False)
            continue
        axis.bar(range(len(controls)), [float(item) for item in values], color="#4c78a8")
        axis.set_xticks(range(len(controls)), [item.replace("_", "\n") for item in controls], fontsize=8)
        axis.set_ylabel("mean CE (lower is better)")
        axis.set_title(f"Seed {seed}: causal state controls")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _pilot_rows(pilots: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for value in pilots:
        selection = value.get("pilot_selection")
        if not isinstance(selection, Mapping):
            continue
        rows.append(
            [
                f"{value['persistent_state']} + h{value['tbptt_horizon']}",
                _fmt(selection["delayed_mechanism_ce"]["m0"]),
                _fmt(selection["delayed_mechanism_ce"]["endpoint"]),
                _fmt(selection["delayed_mechanism_ce"]["delta"]),
                _fmt(selection["endpoint_normal_minus_reset_ce"]),
                "PASS" if selection["technical_gate_passed"] else "FAIL",
                "PASS" if selection["mechanism_gate_passed"] else "FAIL",
                _fmt(value.get("elapsed_seconds"), 1),
            ]
        )
    return rows


def _evaluation_row(seed: int, value: Mapping[str, Any]) -> list[Any]:
    comparison = value.get("endpoint_comparisons", {})
    formal = comparison.get("formal_final_128", {})
    mechanism = comparison.get("mechanism_final_128", {})
    causal = value.get("paired_bootstrap", {}).get("ema_step640|mechanism_final_128|normal_vs_reset", {})
    swap = value.get("paired_bootstrap", {}).get(
        "ema_step640|mechanism_final_128|normal_vs_cross_episode_swap", {}
    )
    passed = (
        float(formal.get("primary_minus_m0_ce", float("inf"))) < 0
        and float(mechanism.get("primary_minus_m0_ce", float("inf"))) < 0
        and float(causal.get("estimate", float("inf"))) < 0
    )
    return [
        seed,
        _fmt(formal.get("m0_ce")),
        _fmt(formal.get("raw_ce")),
        _fmt(formal.get("primary_ema_ce")),
        _fmt(formal.get("primary_minus_m0_ce")),
        _fmt(mechanism.get("m0_ce")),
        _fmt(mechanism.get("primary_ema_ce")),
        _fmt(mechanism.get("primary_minus_m0_ce")),
        _fmt(causal.get("estimate")),
        _fmt(causal.get("ci95")),
        _fmt(swap.get("estimate")),
        "PASS" if passed else "FAIL",
    ]


def render(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    topology = _load(root / "topology_topology_decision.json")
    gradient = _load(root / "runs" / "gradient-audit-latent-h4" / "gradient_fidelity.json")
    pilot_selection = _load(root / "pilot_selection.json")
    rescue = _load(root / "rescue_decision.json")
    pilot_summaries = [
        value
        for path in sorted((root / "runs").glob("pilot-*/summary.json"))
        if (value := _load(path)) is not None
    ]
    main_summaries = {
        int(path.parent.name.removeprefix("main-seed")): value
        for path in sorted((root / "runs").glob("main-seed*/summary.json"))
        if (value := _load(path)) is not None
    }
    evaluations = [
        (int(path.parent.name.removeprefix("final-eval-seed")), value)
        for path in sorted((root / "runs").glob("final-eval-seed*/evaluation_summary.json"))
        if (value := _load(path)) is not None
    ]
    evaluations.sort(key=lambda item: item[0])

    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    if gradient is not None:
        _gradient_figure(gradient, figures / "gradient_fidelity.png")
    _pilot_figure(pilot_summaries, figures / "pilot_delayed_ce.png")
    _loss_figure(root, sorted(main_summaries), figures / "main_training_loss.png")
    _causal_figure(evaluations, figures / "causal_controls.png")

    report_value = {
        "schema": SCHEMA,
        "status": "completed" if evaluations or rescue else "incomplete",
        "topology": topology,
        "gradient_fidelity": gradient,
        "pilot_selection": pilot_selection,
        "pilot_summaries": pilot_summaries,
        "main_summaries": main_summaries,
        "evaluations": {str(seed): value for seed, value in evaluations},
        "rescue": rescue,
    }
    (output / "FINAL_REPORT.json").write_text(
        json.dumps(report_value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    gradient_selection = gradient.get("selection", {}) if gradient else {}
    pilot_winner = pilot_selection.get("winner") if pilot_selection else None
    lines = [
        "# R5-Compose 最终实验报告",
        "",
        "> 固定目标：检验多步视觉状态能否保持、早期写入能否获得未来查询梯度，以及 latent persistence 是否减少 RGB/VAE 往返损伤。",
        "",
        "## 结论先行",
        "",
    ]
    if evaluations:
        passed = sum(_evaluation_row(seed, value)[-1] == "PASS" for seed, value in evaluations)
        lines.append(
            f"已完成 {len(evaluations)} 个主种子的固定 EMA step640 评测，其中 {passed}/{len(evaluations)} 通过“formal 与 mechanism CE 均优于 M0，且 Normal 优于 Reset”的最低因果门。"
        )
    elif rescue:
        lines.append("四臂或主线未形成可接受胜者，已按预注册规则完成 latent+h4、τ=0.5 residual rescue；结果应解释为机制诊断，而非正向记忆证明。")
    else:
        lines.append("实验链尚无完整主评测或条件 rescue，不能给出终局科学结论。")
    lines.extend(
        [
            "",
            "## 1. 固定方法",
            "",
            "- DreamLite-mobile U-Net 仅训练 rank-4 LoRA；Qwen3-VL-4B Reader 与 DreamLite 基座冻结。",
            "- `drtune_stateful`：selected U-Net step 保留状态输入梯度，非 selected step 不计算参数梯度，scheduler 仍在 autograd 内。",
            "- TBPTT 只在 segment 边界 detach；比较 h=2 与 h=4。",
            "- NOOP 返回同一个 Tensor，不调用 DreamLite、不使用 identity loss。",
            "- 噪声只由 global seed、source episode ID、source turn ID 决定。",
            "- 训练课程为 F1–F6，640 optimizer steps × 8 micro-segments = 5,120 个组合段。",
            "- AdamW，clip=10，16-step 线性 warmup 后 cosine decay；EMA=0.995，主 endpoint 固定为 EMA step640。",
            "",
            "## 2. 梯度 fidelity 与设备拓扑",
            "",
            f"- 梯度决策：`{gradient_selection.get('decision', '—')}`；selected-step count：`{gradient_selection.get('selected_step_count', '—')}`。",
            f"- 设备决策：`{topology.get('decision', '—') if topology else '—'}`。",
        ]
    )
    if gradient:
        summaries = gradient_selection.get("group_summaries", {})
        lines.extend(
            [
                "",
                _table(
                    ["近似", "median cosine", "positive fraction", "median norm ratio"],
                    [
                        [
                            key,
                            _fmt(value.get("median_cosine")),
                            _fmt(value.get("positive_fraction")),
                            _fmt(value.get("median_norm_ratio")),
                        ]
                        for key, value in summaries.items()
                    ],
                ),
                "",
                "![Gradient fidelity](figures/gradient_fidelity.png)",
            ]
        )
    lines.extend(["", "## 3. Pilot 结果", ""])
    pilot_rows = _pilot_rows(pilot_summaries)
    if pilot_rows:
        lines.extend(
            [
                _table(
                    ["Arm", "M0 delayed CE", "Endpoint delayed CE", "ΔCE", "Normal−Reset", "技术门", "机制门", "秒"],
                    pilot_rows,
                ),
                "",
                f"预注册胜者：`{pilot_winner.get('name') if isinstance(pilot_winner, Mapping) else '无合格胜者'}`。",
                "",
                "![Pilot delayed CE](figures/pilot_delayed_ce.png)",
            ]
        )
    else:
        lines.append("无完整 pilot summary。")
    lines.extend(["", "## 4. 主训练与最终因果评测", ""])
    if evaluations:
        lines.extend(
            [
                _table(
                    [
                        "Seed",
                        "Formal M0",
                        "Formal raw",
                        "Formal EMA",
                        "Formal Δ",
                        "Mechanism M0",
                        "Mechanism EMA",
                        "Mechanism Δ",
                        "Normal−Reset",
                        "Reset 95% CI",
                        "Normal−CrossSwap",
                        "门",
                    ],
                    [_evaluation_row(seed, value) for seed, value in evaluations],
                ),
                "",
                "![Main training loss](figures/main_training_loss.png)",
                "",
                "![Causal controls](figures/causal_controls.png)",
            ]
        )
    else:
        lines.append("主训练未进入完整固定 endpoint 评测；见 conditional rescue。")
    if main_summaries:
        lines.extend(["", "### 训练诊断", ""])
        lines.append(
            _table(
                ["Seed", "秒", "clip rate", "peak updater GiB", "peak reader GiB", "技术门"],
                [
                    [
                        seed,
                        _fmt(value.get("elapsed_seconds"), 1),
                        _fmt(value.get("clip_rate")),
                        _fmt(value.get("updater_peak_memory_gib"), 2),
                        _fmt(value.get("reader_peak_memory_gib"), 2),
                        "PASS" if value.get("technical_gate", {}).get("passed") else "FAIL",
                    ]
                    for seed, value in sorted(main_summaries.items())
                ],
            )
        )
    lines.extend(
        [
            "",
            "## 5. 状态图片与可解释性边界",
            "",
            "每个最终评测目录都包含 `state_examples/index.json` 以及 F1–F6 的初始/中间/最终图片。图片是否人类可读不是本实验成功标准；核心证据是固定 Reader 的 CE、状态干预和配对 bootstrap。",
            "",
            "## 6. 严谨解释",
            "",
            "- 正向结论只有在多 seed endpoint 改善且 Normal 显著优于 Reset/Cross-swap 时才成立。",
            "- train loss 下降本身不能证明状态被使用；因果对照是必要条件。",
            "- 本轮只评估 ID dev 与受控 mechanism dev，不主张 OOD、跨 Reader 或闭源 API 泛化。",
            "- F1–F6 最长为 3 次 updater；因此原方案中的 gap-4 tie-break 实际退化为最长可观察 gap=3，这一偏差已在 pilot selection 中显式记录。",
            "- hard NOOP 是机制隔离，不代表生成式 updater 已学会 NOOP。",
        ]
    )
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (output / "main_seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "seed",
                "formal_m0_ce",
                "formal_ema_ce",
                "formal_delta_ce",
                "mechanism_m0_ce",
                "mechanism_ema_ce",
                "mechanism_delta_ce",
                "normal_minus_reset_ce",
                "pass",
            ]
        )
        for seed, value in evaluations:
            row = _evaluation_row(seed, value)
            writer.writerow([seed, row[1], row[3], row[4], row[5], row[6], row[7], row[8], row[-1]])

    indexed: list[dict[str, Any]] = []
    critical = [
        output / "FINAL_REPORT.md",
        output / "FINAL_REPORT.json",
        output / "main_seed_summary.csv",
        *sorted(figures.glob("*.png")),
    ]
    for path in critical:
        if path.is_file():
            indexed.append(
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest = {"schema": "vision_memory.r5-compose-delivery-manifest.v1", "artifacts": indexed}
    (output / "DELIVERY_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render(args.experiment_root.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "output": str(args.output_dir.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
