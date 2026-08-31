"""Validate and render all eight R10 DreamLite raw-endpoint attributions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiments import compare_r10_visual_alignment as r10  # noqa: E402
from scripts.experiments import evaluate_r10_raw_endpoint as raw  # noqa: E402
from vision_memory.training.r10_alignment import target_gate  # noqa: E402


ANALYSIS_COMMIT = "05814af7bd8adada14327f708c713f1ee1eea19d"
ANALYSIS_SCHEMA = "vision_memory.r10-raw-endpoint-attribution-analysis.v1"
DELIVERY_SCHEMA = "vision_memory.r10-raw-endpoint-attribution-delivery.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _validate_inventory(root: Path) -> str:
    path = root / "artifact_inventory.json"
    inventory = _load(path)
    if inventory.get("schema") != raw.INVENTORY_SCHEMA:
        raise ValueError(f"R10 raw attribution inventory schema mismatch: {root}")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"R10 raw attribution inventory is empty: {root}")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"R10 raw attribution inventory record is malformed: {root}")
        relative = str(record.get("path"))
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative in names:
            raise ValueError(f"R10 raw attribution inventory path is invalid: {relative}")
        names.add(relative)
        artifact = root / relative_path
        if not artifact.is_file():
            raise ValueError(f"R10 raw attribution artifact is missing: {artifact}")
        if artifact.stat().st_size != int(record.get("bytes", -1)) or r10._sha256(artifact) != record.get("sha256"):
            raise ValueError(f"R10 raw attribution artifact failed size/SHA validation: {artifact}")
    required = {
        "environment.txt",
        "runtime.json",
        "manifest.json",
        "raw_endpoint_evaluation_rows.jsonl",
        "raw_endpoint_state.png",
        "summary.json",
        "terminal.json",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"R10 raw attribution inventory lacks required artifacts: {root}:{missing}")
    return r10._sha256(path)


def _numeric_statistics(statistics: Mapping[str, Any], *, label: str) -> None:
    fields = (
        "m0_normal_mean_ce",
        "endpoint_normal_mean_ce",
        "delta_ce",
        "relative_change",
        "accuracy_delta",
        "normal_reset_difference_in_differences",
    )
    if any(not math.isfinite(float(statistics.get(field, math.nan))) for field in fields):
        raise ValueError(f"R10 raw attribution contains non-finite {label} statistics.")


def _validate_target(root: Path, index: int) -> dict[str, Any]:
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    terminal_path = root / "terminal.json"
    summary = _load(summary_path)
    manifest = _load(manifest_path)
    terminal = _load(terminal_path)
    expected_id = r10.R10_TARGET_IDS[index]
    if (
        terminal.get("schema") != raw.TERMINAL_SCHEMA
        or terminal.get("status") != "completed_attribution"
        or terminal.get("passed") is not True
        or terminal.get("formal_success_claim") is not False
        or terminal.get("target_index") != index
        or terminal.get("target_segment_id") != expected_id
        or terminal.get("summary_sha256") != r10._sha256(summary_path)
        or terminal.get("manifest_sha256") != r10._sha256(manifest_path)
    ):
        raise ValueError(f"R10 raw attribution terminal integrity failed: target {index}")
    if (
        manifest.get("schema") != raw.MANIFEST_SCHEMA
        or manifest.get("status") != "completed"
        or manifest.get("formal_success_claim") is not False
        or manifest.get("cannot_replace_preregistered_ema_endpoint") is not True
        or manifest.get("analysis_git_commit") != ANALYSIS_COMMIT
        or manifest.get("source_training_git_commit") != r10.EXPECTED_GIT_COMMIT
        or manifest.get("target_index") != index
        or manifest.get("target_segment_id") != expected_id
        or manifest.get("selected_segments_sha256") != r10.R10_SELECTED_SEGMENTS_SHA256
    ):
        raise ValueError(f"R10 raw attribution manifest integrity failed: target {index}")
    if (
        summary.get("schema") != raw.SUMMARY_SCHEMA
        or summary.get("status") != "completed_attribution"
        or summary.get("formal_success_claim") is not False
        or summary.get("cannot_replace_preregistered_ema_endpoint") is not True
        or summary.get("target_index") != index
        or summary.get("target_segment_id") != expected_id
        or summary.get("technical_gate") is not True
    ):
        raise ValueError(f"R10 raw attribution summary integrity failed: target {index}")
    raw_statistics = summary.get("raw_statistics")
    ema_statistics = summary.get("existing_ema_statistics")
    if not isinstance(raw_statistics, Mapping) or not isinstance(ema_statistics, Mapping):
        raise ValueError(f"R10 raw attribution statistics are missing: target {index}")
    _numeric_statistics(raw_statistics, label="raw")
    _numeric_statistics(ema_statistics, label="EMA")
    raw_gate = target_gate(raw_statistics, technical_gate=True)
    if summary.get("raw_descriptive_gate") is not raw_gate:
        raise ValueError(f"R10 raw attribution gate drifted: target {index}")
    source_root = Path(str(manifest.get("source_root")))
    source = r10._validate_target(source_root, "dreamlite-single-set", index)
    if (
        source["inventory_sha256"] != manifest.get("source_inventory_sha256")
        or source["summary_sha256"] != manifest.get("source_summary_sha256")
        or source["terminal_sha256"] != manifest.get("source_terminal_sha256")
        or source["target_statistics"] != dict(ema_statistics)
        or source["passed"] is not summary.get("existing_ema_gate")
    ):
        raise ValueError(f"R10 raw attribution/source binding failed: target {index}")
    artifacts = summary.get("artifacts")
    expected_artifacts = {
        "manifest_sha256": r10._sha256(manifest_path),
        "rows_sha256": r10._sha256(root / "raw_endpoint_evaluation_rows.jsonl"),
        "state_image_sha256": r10._sha256(root / "raw_endpoint_state.png"),
    }
    if artifacts != expected_artifacts:
        raise ValueError(f"R10 raw attribution output hash binding failed: target {index}")
    inventory_sha = _validate_inventory(root)
    return {
        "target_index": index,
        "target_segment_id": expected_id,
        "raw_descriptive_gate": raw_gate,
        "raw_statistics": dict(raw_statistics),
        "ema_gate": source["passed"],
        "ema_statistics": dict(ema_statistics),
        "raw_minus_ema_normal_ce": float(summary["raw_minus_ema_normal_ce"]),
        "source_root": str(source_root.resolve()),
        "attribution_root": str(root.resolve()),
        "inventory_sha256": inventory_sha,
        "summary_sha256": r10._sha256(summary_path),
        "terminal_sha256": r10._sha256(terminal_path),
        "state_image_sha256": expected_artifacts["state_image_sha256"],
    }


def _decision(pass_count: int) -> tuple[str, str]:
    if pass_count == 8:
        return (
            "ema_lag_is_sufficient_endpoint_bottleneck",
            "All raw step128 targets pass while the preregistered EMA arm failed. Repair endpoint averaging or training horizon before changing representation.",
        )
    if pass_count > 0:
        return (
            "ema_contributes_but_updater_remains_insufficient_run_vae_latent_oracle",
            "Raw weights rescue only a subset. EMA may contribute, but the updater remains target-dependent and insufficient; test VAE-latent reachability next.",
        )
    return (
        "ema_is_not_sufficient_explanation_run_vae_latent_oracle",
        "No raw step128 target passes. EMA lag is not a sufficient explanation; test whether the VAE latent space itself contains readable codes before redesigning the writer.",
    )


def _write_csv(path: Path, targets: list[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "target_index", "segment_id", "raw_gate", "ema_gate", "raw_relative_change",
                "ema_relative_change", "raw_views", "ema_views", "raw_accuracy_delta",
                "ema_accuracy_delta", "raw_did", "ema_did", "raw_minus_ema_normal_ce",
            )
        )
        for target in targets:
            raw_stat = target["raw_statistics"]
            ema_stat = target["ema_statistics"]
            writer.writerow(
                (
                    target["target_index"], target["target_segment_id"], target["raw_descriptive_gate"],
                    target["ema_gate"], raw_stat["relative_change"], ema_stat["relative_change"],
                    raw_stat["improved_choice_views"], ema_stat["improved_choice_views"],
                    raw_stat["accuracy_delta"], ema_stat["accuracy_delta"],
                    raw_stat["normal_reset_difference_in_differences"],
                    ema_stat["normal_reset_difference_in_differences"],
                    target["raw_minus_ema_normal_ce"],
                )
            )


def _comparison_figure(path: Path, targets: list[Mapping[str, Any]]) -> None:
    indices = list(range(8))
    width = 0.36
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    fields = (
        ("relative_change", "Normal CE relative change", "endpoint / M0 - 1", -0.20),
        ("improved_choice_views", "Held-out views improved", "views", 4.0),
        ("accuracy_delta", "Normal accuracy change", "endpoint - M0", 0.25),
        ("normal_reset_difference_in_differences", "Normal/reset difference-in-differences", "CE DiD", 0.0),
    )
    for axis, (field, title, ylabel, gate) in zip(axes.flat, fields, strict=True):
        raw_values = [float(target["raw_statistics"][field]) for target in targets]
        ema_values = [float(target["ema_statistics"][field]) for target in targets]
        axis.bar([index - width / 2 for index in indices], ema_values, width, color="#ea580c", label="EMA step128")
        axis.bar([index + width / 2 for index in indices], raw_values, width, color="#7c3aed", label="raw step128")
        axis.axhline(gate, color="black", linestyle="--", linewidth=0.9, label=f"gate {gate:g}")
        if gate != 0.0:
            axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set(title=title, xlabel="target", ylabel=ylabel)
        axis.set_xticks(indices)
        axis.grid(axis="y", alpha=0.18)
        axis.legend(fontsize=7)
    axes[0, 1].set_ylim(0, 4.4)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _contact_sheet(path: Path, attribution_root: Path, targets: list[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.5))
    for index, axis in enumerate(axes.flat):
        image = plt.imread(attribution_root / f"target-{index:02d}" / "raw_endpoint_state.png")
        axis.imshow(image)
        relative = float(targets[index]["raw_statistics"]["relative_change"])
        axis.set_title(f"t{index} | raw {relative:+.1%}")
        axis.axis("off")
    figure.suptitle("R10 DreamLite raw step128 state images")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# R10 DreamLite raw-endpoint attribution",
        "",
        "This post-registered attribution cannot replace or rescue the preregistered EMA endpoint.",
        "",
        f"- Raw descriptive gates: {analysis['raw_pass_count']}/8.",
        f"- Existing EMA gates: {analysis['ema_pass_count']}/8.",
        f"- Decision: `{analysis['decision']}`",
        f"- Reason: {analysis['reason']}",
        "",
        "| Target | Raw gate | Raw relative CE | EMA relative CE | Raw views | Raw accuracy delta | Raw/EMA CE difference |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in analysis["targets"]:
        raw_stat = target["raw_statistics"]
        ema_stat = target["ema_statistics"]
        lines.append(
            f"| {target['target_index']} | {'PASS' if target['raw_descriptive_gate'] else 'FAIL'} | "
            f"{float(raw_stat['relative_change']):+.2%} | {float(ema_stat['relative_change']):+.2%} | "
            f"{raw_stat['improved_choice_views']}/4 | {float(raw_stat['accuracy_delta']):+.2f} | "
            f"{float(target['raw_minus_ema_normal_ce']):+.4f} |"
        )
    lines.extend(("", "Raw results are root-cause evidence only; formal success remains false.", ""))
    return "\n".join(lines)


def render(attribution_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R10 raw attribution renderer refuses a non-empty output directory.")
    targets = [_validate_target(attribution_root / f"target-{index:02d}", index) for index in range(8)]
    raw_pass_count = sum(bool(target["raw_descriptive_gate"]) for target in targets)
    ema_pass_count = sum(bool(target["ema_gate"]) for target in targets)
    decision, reason = _decision(raw_pass_count)
    analysis = {
        "schema": ANALYSIS_SCHEMA,
        "status": "completed",
        "formal_success_claim": False,
        "cannot_replace_preregistered_ema_endpoint": True,
        "analysis_git_commit": ANALYSIS_COMMIT,
        "source_training_git_commit": r10.EXPECTED_GIT_COMMIT,
        "selected_segments_sha256": r10.R10_SELECTED_SEGMENTS_SHA256,
        "raw_pass_count": raw_pass_count,
        "ema_pass_count": ema_pass_count,
        "decision": decision,
        "reason": reason,
        "targets": targets,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    r10._write_json(output_dir / "ANALYSIS.json", analysis)
    _write_csv(output_dir / "raw_vs_ema.csv", targets)
    _comparison_figure(output_dir / "raw_vs_ema_metrics.png", targets)
    _contact_sheet(output_dir / "raw_state_contact_sheet.png", attribution_root, targets)
    (output_dir / "REPORT.md").write_text(_report(analysis), encoding="utf-8")
    artifacts = []
    for path in sorted(value for value in output_dir.iterdir() if value.is_file()):
        if path.name == "DELIVERY_MANIFEST.json":
            continue
        artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": r10._sha256(path)})
    r10._write_json(
        output_dir / "DELIVERY_MANIFEST.json",
        {
            "schema": DELIVERY_SCHEMA,
            "artifacts": artifacts,
            "source_inventory_sha256": {
                str(target["target_index"]): target["inventory_sha256"] for target in targets
            },
        },
    )
    return analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analysis = render(args.attribution_root, args.output_dir)
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "raw_pass_count": analysis["raw_pass_count"],
                "ema_pass_count": analysis["ema_pass_count"],
                "decision": analysis["decision"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
