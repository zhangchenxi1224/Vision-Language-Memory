"""Render and integrity-check a completed paired R6 source-anchor diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ARMS = ("legacy-pure-noise", "source-anchored")
COLORS = {"legacy-pure-noise": "#d97706", "source-anchored": "#2563eb"}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"Expected JSON objects in {path}")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _moving_mean(values: Sequence[float], window: int = 16) -> list[float]:
    return [
        statistics.fmean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _validate_arm(root: Path, expected_arm: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    terminal = _load_json(root / "terminal.json")
    if terminal.get("status") != "completed_diagnostic" or terminal.get("passed") is not True:
        raise ValueError(f"R6 arm did not complete cleanly: {root}")
    summary = _load_json(root / "run" / "r6_summary.json")
    if summary.get("status") != "completed" or summary.get("arm") != expected_arm:
        raise ValueError(f"R6 summary arm/status mismatch: {root}")
    if summary.get("full_success_claim_allowed") is not False:
        raise ValueError(f"R6 diagnostic improperly permits a formal success claim: {root}")

    inventory = _load_json(root / "artifact_inventory.json")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"R6 artifact inventory is empty: {root}")
    for record in records:
        path = root / str(record["path"])
        if not path.is_file():
            raise ValueError(f"Inventory artifact is missing: {path}")
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise ValueError(f"Inventory artifact failed size/SHA validation: {path}")
    return summary, _load_jsonl(root / "run" / "metrics.jsonl")


def _validate_pair(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    for key in ("git_commit", "implementation_revision", "selected_segments_sha256"):
        values = {summary.get(key) for summary in summaries.values()}
        if len(values) != 1 or None in values:
            raise ValueError(f"R6 paired-arm drift in {key}: {values}")


def _training_csv(output: Path, metrics: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    rows: list[list[Any]] = []
    for arm in ARMS:
        for record in metrics[arm]:
            diagnostics = record.get("optimizer_diagnostics", {})
            update = diagnostics.get("updates_after_step", {}).get("global", {})
            family = record.get("loss_by_family", {})
            rows.append(
                [
                    arm,
                    record["optimizer_step"],
                    record["learning_rate"],
                    record["loss_mean"],
                    family.get("F2"),
                    family.get("F3"),
                    family.get("F5"),
                    family.get("F6"),
                    record["gradient_norm_before_clip"],
                    record["gradient_clipped"],
                    update.get("update_weight_ratio"),
                    record.get("state_gradient_nonzero_fraction"),
                    record.get("image_saturation_fraction_mean"),
                ]
            )
    _write_csv(
        output,
        (
            "arm",
            "optimizer_step",
            "learning_rate",
            "loss_mean",
            "loss_F2",
            "loss_F3",
            "loss_F5",
            "loss_F6",
            "gradient_norm_before_clip",
            "gradient_clipped",
            "update_weight_ratio",
            "state_gradient_nonzero_fraction",
            "image_saturation_fraction_mean",
        ),
        rows,
    )


def _gradient_csv(output: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    rows = []
    for arm in ARMS:
        for record in summaries[arm]["gradient_conflict_audit"]["records"]:
            rows.append(
                [
                    arm,
                    record["index"],
                    record["segment_id"],
                    record["family"],
                    record["loss"],
                    record["gradient_norm"],
                    record["norm_share_of_sum_of_norms"],
                    record["cosine_to_raw_batch_gradient"],
                    record["cosine_to_unit_balanced_gradient"],
                ]
            )
    _write_csv(
        output,
        (
            "arm",
            "index",
            "segment_id",
            "family",
            "loss",
            "gradient_norm",
            "norm_share_of_sum_of_norms",
            "cosine_to_raw_batch_gradient",
            "cosine_to_unit_balanced_gradient",
        ),
        rows,
    )


def _endpoint_rows(summaries: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    rows = []
    for arm in ARMS:
        summary = summaries[arm]
        comparisons = summary["comparisons"]
        hard8 = comparisons["train_overfit_hard8_endpoint_vs_m0"]
        formal = comparisons["formal_select_32_endpoint_vs_m0"]
        mechanism = comparisons["mechanism_select_32_endpoint_vs_m0"]
        did = comparisons["train_overfit_hard8_state_did"]
        conflict = summary["gradient_conflict_audit"]
        training = summary["training_summary"]
        rows.append(
            [
                arm,
                summary["edit_start_sigma"],
                hard8["m0_mean_ce"],
                hard8["endpoint_mean_ce"],
                hard8["estimate"],
                hard8["relative_change"],
                hard8["ci95"][0],
                hard8["ci95"][1],
                hard8["improved_pair_units"],
                summary["overfit_accuracy"]["m0"],
                summary["overfit_accuracy"]["endpoint"],
                summary["overfit_accuracy"]["delta"],
                did["estimate"],
                formal["estimate"],
                mechanism["estimate"],
                summary["gates"]["technical_gate"],
                summary["gates"]["hard8_overfit_learnability_gate"],
                summary["gates"]["fixed_dev_generalization_gate"],
                training["clip_rate"],
                conflict["gradient_norm"]["median"],
                conflict["gradient_norm"]["max_to_min_ratio"],
                conflict["off_diagonal"]["negative_fraction"],
                conflict["off_diagonal"]["median"],
                conflict["raw_vs_unit_balanced_cosine"],
            ]
        )
    return rows


def _training_figure(output: Path, metrics: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    for arm in ARMS:
        records = metrics[arm]
        steps = [int(record["optimizer_step"]) for record in records]
        losses = [float(record["loss_mean"]) for record in records]
        gradients = [float(record["gradient_norm_before_clip"]) for record in records]
        update_ratios = [
            float(record["optimizer_diagnostics"]["updates_after_step"]["global"]["update_weight_ratio"])
            for record in records
        ]
        color = COLORS[arm]
        axes[0, 0].plot(steps, losses, color=color, alpha=0.22, linewidth=0.8)
        axes[0, 0].plot(steps, _moving_mean(losses), color=color, linewidth=1.8, label=arm)
        axes[0, 1].plot(steps, gradients, color=color, linewidth=1.2, label=arm)
        axes[1, 0].plot(steps, update_ratios, color=color, linewidth=1.2, label=arm)
        for family, style in zip(("F2", "F3", "F5", "F6"), ("-", "--", "-.", ":"), strict=True):
            family_losses = [float(record["loss_by_family"][family]) for record in records]
            axes[1, 1].plot(
                steps,
                _moving_mean(family_losses),
                color=color,
                linestyle=style,
                alpha=0.9,
                label=f"{arm} {family}",
            )
    axes[0, 0].set(title="Hard8 training CE", xlabel="optimizer step", ylabel="CE (16-step mean)")
    axes[0, 1].axhline(10.0, color="black", linestyle="--", linewidth=0.9, label="clip=10")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Pre-clip gradient norm", xlabel="optimizer step", ylabel="L2 norm (log)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Optimizer update / weight", xlabel="optimizer step", ylabel="ratio (log)")
    axes[1, 1].set(title="Per-family training CE", xlabel="optimizer step", ylabel="CE (16-step mean)")
    for axis in axes.flat:
        axis.grid(alpha=0.18)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _gradient_figure(output: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    image = None
    for axis, arm in zip(axes[:2], ARMS, strict=True):
        audit = summaries[arm]["gradient_conflict_audit"]
        image = axis.imshow(audit["pairwise_cosine"], vmin=-1.0, vmax=1.0, cmap="coolwarm")
        labels = [record["family"] for record in audit["records"]]
        axis.set_xticks(range(8), labels, rotation=45)
        axis.set_yticks(range(8), labels)
        axis.set_title(f"{arm}: gradient cosine")
    if image is not None:
        fig.colorbar(image, ax=axes[:2].tolist(), fraction=0.03, pad=0.03)
    positions = list(range(8))
    width = 0.38
    for offset, arm in ((-width / 2, ARMS[0]), (width / 2, ARMS[1])):
        records = summaries[arm]["gradient_conflict_audit"]["records"]
        axes[2].bar(
            [position + offset for position in positions],
            [record["gradient_norm"] for record in records],
            width,
            color=COLORS[arm],
            label=arm,
        )
    axes[2].set_yscale("log")
    axes[2].set_xticks(positions, [record["family"] for record in summaries[ARMS[0]]["gradient_conflict_audit"]["records"]])
    axes[2].set(title="Per-segment gradient norm", xlabel="segment family", ylabel="L2 norm (log)")
    axes[2].legend(fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.16, top=0.90, wspace=0.34)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _endpoint_figure(output: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    positions = list(range(len(ARMS)))
    width = 0.36
    hard8 = [summaries[arm]["comparisons"]["train_overfit_hard8_endpoint_vs_m0"] for arm in ARMS]
    axes[0].bar([value - width / 2 for value in positions], [value["m0_mean_ce"] for value in hard8], width, label="M0")
    axes[0].bar([value + width / 2 for value in positions], [value["endpoint_mean_ce"] for value in hard8], width, label="EMA step128")
    axes[0].set(title="Hard8 CE", ylabel="mean CE (lower is better)")
    axes[0].legend(fontsize=8)
    axes[1].bar([value - width / 2 for value in positions], [summaries[arm]["overfit_accuracy"]["m0"] for arm in ARMS], width, label="M0")
    axes[1].bar([value + width / 2 for value in positions], [summaries[arm]["overfit_accuracy"]["endpoint"] for arm in ARMS], width, label="EMA step128")
    axes[1].set(title="Hard8 accuracy", ylabel="accuracy")
    axes[1].legend(fontsize=8)
    suites = ("hard8", "formal", "mechanism")
    keys = (
        "train_overfit_hard8_endpoint_vs_m0",
        "formal_select_32_endpoint_vs_m0",
        "mechanism_select_32_endpoint_vs_m0",
    )
    suite_positions = list(range(len(suites)))
    for offset, arm in ((-width / 2, ARMS[0]), (width / 2, ARMS[1])):
        axes[2].bar(
            [value + offset for value in suite_positions],
            [summaries[arm]["comparisons"][key]["estimate"] for key in keys],
            width,
            color=COLORS[arm],
            label=arm,
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(suite_positions, suites)
    axes[2].set(title="Endpoint minus own M0", ylabel="paired CE delta")
    axes[2].legend(fontsize=7)
    for axis in axes[:2]:
        axis.set_xticks(positions, ["legacy", "anchored"])
    for axis in axes:
        axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _report(summaries: Mapping[str, Mapping[str, Any]], endpoint_rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "# R6 source-anchor paired diagnostic delivery",
        "",
        "This is a one-seed repeated-hard8 bottleneck diagnostic and cannot establish formal picture-memory success.",
        "",
        f"- Git commit: `{summaries[ARMS[0]]['git_commit']}`",
        f"- Implementation: `{summaries[ARMS[0]]['implementation_revision']}`",
        f"- Selected-segment SHA-256: `{summaries[ARMS[0]]['selected_segments_sha256']}`",
        "",
        "| Arm | hard8 M0 CE | endpoint CE | delta CE | relative | improved units | accuracy delta | hard8 gate | formal delta | mechanism delta | fixed-dev gate | clip rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | ---: |",
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
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "Interpretation must follow the preregistered gates. Absolute CE across arms is secondary because each update law has a different M0; the primary comparison is endpoint versus that arm's own M0 with paired uncertainty and causal state controls.",
            "",
        )
    )
    return "\n".join(lines)


def render(legacy_root: Path, anchored_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R6 renderer refuses a non-empty output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    roots = {ARMS[0]: legacy_root.resolve(), ARMS[1]: anchored_root.resolve()}
    summaries: dict[str, dict[str, Any]] = {}
    metrics: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        summaries[arm], metrics[arm] = _validate_arm(roots[arm], arm)
        if len(metrics[arm]) != 128:
            raise ValueError(f"R6 arm does not contain exactly 128 optimizer records: {arm}")
    _validate_pair(summaries)

    _training_csv(output_dir / "training_metrics.csv", metrics)
    _gradient_csv(output_dir / "gradient_conflict.csv", summaries)
    endpoint_rows = _endpoint_rows(summaries)
    _write_csv(
        output_dir / "endpoint_summary.csv",
        (
            "arm",
            "edit_start_sigma",
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
            "audit_gradient_median",
            "audit_gradient_max_min_ratio",
            "negative_gradient_cosine_fraction",
            "median_gradient_cosine",
            "raw_vs_unit_balanced_cosine",
        ),
        endpoint_rows,
    )
    _training_figure(output_dir / "training_diagnostics.png", metrics)
    _gradient_figure(output_dir / "gradient_conflict.png", summaries)
    _endpoint_figure(output_dir / "endpoint_metrics.png", summaries)
    (output_dir / "REPORT.md").write_text(_report(summaries, endpoint_rows), encoding="utf-8")

    analysis = {
        "schema": "vision_memory.r6-source-anchor-rendered-analysis.v1",
        "status": "completed",
        "formal_success_claim": False,
        "source_roots": {arm: str(root) for arm, root in roots.items()},
        "git_commit": summaries[ARMS[0]]["git_commit"],
        "implementation_revision": summaries[ARMS[0]]["implementation_revision"],
        "selected_segments_sha256": summaries[ARMS[0]]["selected_segments_sha256"],
        "gates": {arm: summaries[arm]["gates"] for arm in ARMS},
        "comparisons": {arm: summaries[arm]["comparisons"] for arm in ARMS},
    }
    _write_json(output_dir / "ANALYSIS.json", analysis)
    inventory = []
    for path in sorted(value for value in output_dir.rglob("*") if value.is_file()):
        if path.name == "DELIVERY_MANIFEST.json":
            continue
        inventory.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    delivery = {
        "schema": "vision_memory.r6-source-anchor-delivery-manifest.v1",
        "artifacts": inventory,
        "source_inventory_sha256": {
            arm: _sha256(roots[arm] / "artifact_inventory.json") for arm in ARMS
        },
    }
    _write_json(output_dir / "DELIVERY_MANIFEST.json", delivery)
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--anchored-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = render(args.legacy_root, args.anchored_root, args.output_dir)
    print(json.dumps({"status": analysis["status"], "formal_success_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
