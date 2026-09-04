"""Validate and aggregate both preregistered R10 visual-alignment arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_OPTIMIZER_STEPS,
    R10_PROTOCOL,
    R10_SELECTED_SEGMENTS_SHA256,
    R10_TARGET_IDS,
    target_gate,
)


ARMS = ("direct-pixel-oracle", "dreamlite-single-set")
SUMMARY_SCHEMAS = {
    "direct-pixel-oracle": "vision_memory.r10-direct-pixel-oracle-summary.v1",
    "dreamlite-single-set": "vision_memory.r10-dreamlite-single-set-summary.v1",
}
SUMMARY_FILES = {
    "direct-pixel-oracle": "r10_pixel_summary.json",
    "dreamlite-single-set": "r10_dreamlite_summary.json",
}
IMPLEMENTATION_REVISIONS = {
    "direct-pixel-oracle": "direct-pixel-logit-v1",
    "dreamlite-single-set": "dreamlite-single-set-full-gradient-v1",
}
TERMINAL_SCHEMA = "vision_memory.r10-alignment-target-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r10-alignment-target-inventory.v1"
COMPARISON_SCHEMA = "vision_memory.r10-visual-alignment-comparison.v1"
DELIVERY_SCHEMA = "vision_memory.r10-visual-alignment-delivery.v1"
EXPECTED_GIT_COMMIT = "ba86e2c5b6a4d55f97ba386ad135fea546a22dbf"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _required_artifacts(arm: str) -> set[str]:
    common = {
        "launch.json",
        "terminal.json",
        "run/manifest.json",
        "run/metrics.jsonl",
        "run/target_evaluation_rows.jsonl",
        f"run/{SUMMARY_FILES[arm]}",
    }
    if arm == "direct-pixel-oracle":
        return common | {
            "run/endpoint_raw.pt",
            "run/endpoint_raw.png",
            "run/snapshot_end_verification.json",
        }
    return common | {
        "run/micro_metrics.jsonl",
        "run/endpoint_ema.pt",
        "run/endpoint_raw.pt",
    }


def _validate_inventory(root: Path, arm: str) -> str:
    inventory_path = root / "artifact_inventory.json"
    inventory = _load(inventory_path)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"R10 inventory schema mismatch: {root}")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"R10 artifact inventory is empty: {root}")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"R10 inventory record is malformed: {root}")
        relative = str(record.get("path"))
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"R10 inventory path escapes its root: {relative}")
        if relative in names:
            raise ValueError(f"R10 inventory contains a duplicate path: {relative}")
        names.add(relative)
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"R10 inventory artifact is missing: {path}")
        if path.stat().st_size != int(record.get("bytes", -1)) or _sha256(path) != record.get("sha256"):
            raise ValueError(f"R10 inventory artifact failed size/SHA validation: {path}")
    missing = sorted(_required_artifacts(arm) - names)
    if missing:
        raise ValueError(f"R10 inventory lacks required artifacts for {root}: {missing}")
    return _sha256(inventory_path)


def _summary_artifact_paths(arm: str) -> dict[str, str]:
    if arm == "direct-pixel-oracle":
        return {
            "manifest_sha256": "manifest.json",
            "metrics_sha256": "metrics.jsonl",
            "evaluation_rows_sha256": "target_evaluation_rows.jsonl",
            "endpoint_raw_sha256": "endpoint_raw.pt",
            "endpoint_png_sha256": "endpoint_raw.png",
            "snapshot_end_verification_sha256": "snapshot_end_verification.json",
        }
    return {
        "manifest_sha256": "manifest.json",
        "metrics_sha256": "metrics.jsonl",
        "micro_metrics_sha256": "micro_metrics.jsonl",
        "evaluation_rows_sha256": "target_evaluation_rows.jsonl",
        "endpoint_ema_sha256": "endpoint_ema.pt",
        "endpoint_raw_sha256": "endpoint_raw.pt",
    }


def _validate_summary_artifacts(root: Path, arm: str, summary: Mapping[str, Any]) -> None:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"R10 summary artifact bindings are missing: {root}")
    for key, relative in _summary_artifact_paths(arm).items():
        path = root / "run" / relative
        if artifacts.get(key) != _sha256(path):
            raise ValueError(f"R10 summary artifact hash binding failed: {root}:{key}")


def _validate_target(root: Path, arm: str, index: int) -> dict[str, Any]:
    expected_segment = R10_TARGET_IDS[index]
    terminal_path = root / "terminal.json"
    summary_path = root / "run" / SUMMARY_FILES[arm]
    terminal = _load(terminal_path)
    summary = _load(summary_path)
    if terminal.get("schema") != TERMINAL_SCHEMA:
        raise ValueError(f"R10 terminal schema mismatch: {arm} target {index}")
    checks = terminal.get("checks")
    if (
        terminal.get("status") != "completed_diagnostic"
        or terminal.get("passed") is not True
        or terminal.get("child_exit_code") != 0
        or terminal.get("scientific_success_claim") is not False
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError(f"R10 target did not complete its technical contract: {arm} target {index}")
    if terminal.get("summary_sha256") != _sha256(summary_path):
        raise ValueError(f"R10 terminal/summary hash binding failed: {arm} target {index}")
    manifest_path = root / "run" / "manifest.json"
    if terminal.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError(f"R10 terminal/manifest hash binding failed: {arm} target {index}")
    if (
        summary.get("schema") != SUMMARY_SCHEMAS[arm]
        or summary.get("status") != "completed"
        or summary.get("protocol") != R10_PROTOCOL
        or summary.get("arm") != arm
    ):
        raise ValueError(f"R10 summary identity mismatch: {arm} target {index}")
    if (
        summary.get("target_index") != index
        or summary.get("target_family") != "F1"
        or summary.get("target_segment_id") != expected_segment
        or terminal.get("arm") != arm
        or terminal.get("target_index") != index
        or terminal.get("target_segment_id") != expected_segment
    ):
        raise ValueError(f"R10 target identity drift: {arm} target {index}")
    if (
        summary.get("selected_segments") != list(R10_TARGET_IDS)
        or summary.get("selected_segments_sha256") != R10_SELECTED_SEGMENTS_SHA256
        or summary.get("checkpoint_steps_observed") != [0, 32, 64, 96, 128]
    ):
        raise ValueError(f"R10 fixed selection/execution drift: {arm} target {index}")
    if (
        summary.get("git_commit") != EXPECTED_GIT_COMMIT
        or summary.get("implementation_revision") != IMPLEMENTATION_REVISIONS[arm]
    ):
        raise ValueError(f"R10 immutable code drift: {arm} target {index}")
    gates = summary.get("gates")
    technical = summary.get("technical_gate")
    if (
        not isinstance(gates, Mapping)
        or not isinstance(technical, Mapping)
        or technical.get("passed") is not True
        or gates.get("technical_gate") is not True
    ):
        raise ValueError(f"R10 target technical gate is invalid: {arm} target {index}")
    if (
        gates.get("formal_success_gate") is not False
        or summary.get("full_success_claim_allowed") is not False
        or summary.get("diagnostic_only_not_formal_success") is not True
    ):
        raise ValueError(f"R10 target incorrectly permits formal success: {arm} target {index}")
    statistics = summary.get("target_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError(f"R10 target statistics are missing: {arm} target {index}")
    numeric = (
        "m0_normal_mean_ce",
        "endpoint_normal_mean_ce",
        "delta_ce",
        "relative_change",
        "accuracy_delta",
        "normal_reset_difference_in_differences",
    )
    if any(not math.isfinite(float(statistics.get(key, math.nan))) for key in numeric):
        raise ValueError(f"R10 target statistics are non-finite: {arm} target {index}")
    expected_gate = target_gate(statistics, technical_gate=True)
    if gates.get("target_lower_bound_gate") is not expected_gate:
        raise ValueError(f"R10 summary gate differs from fixed thresholds: {arm} target {index}")
    if terminal.get("target_lower_bound_gate") is not expected_gate:
        raise ValueError(f"R10 terminal gate differs from its summary: {arm} target {index}")
    if arm == "direct-pixel-oracle":
        if summary.get("optimizer_steps") != R10_OPTIMIZER_STEPS:
            raise ValueError(f"R10 pixel optimizer-step drift: target {index}")
    elif (
        summary.get("gradient_aggregation") != "single-target-full"
        or not math.isclose(float(summary.get("gradient_coefficient", math.nan)), 1.0)
    ):
        raise ValueError(f"R10 DreamLite gradient contract drift: target {index}")
    _validate_summary_artifacts(root, arm, summary)
    inventory_sha = _validate_inventory(root, arm)
    training_summary = summary.get("training_summary", {})
    clip_rate = training_summary.get("clip_rate") if isinstance(training_summary, Mapping) else None
    return {
        "arm": arm,
        "target_index": index,
        "target_family": "F1",
        "target_segment_id": expected_segment,
        "passed": expected_gate,
        "target_statistics": dict(statistics),
        "clip_rate": float(clip_rate) if clip_rate is not None else None,
        "wall_clock_seconds": summary.get("wall_clock_seconds"),
        "source_root": str(root.resolve()),
        "summary_sha256": _sha256(summary_path),
        "terminal_sha256": _sha256(terminal_path),
        "inventory_sha256": inventory_sha,
        "git_commit": summary.get("git_commit"),
        "implementation_revision": summary.get("implementation_revision"),
    }


def _decision(pixel_passes: int, dreamlite_passes: int) -> tuple[str, str]:
    if pixel_passes == 8 and dreamlite_passes == 8:
        return (
            "advance_shared_f1_train_heldout_alignment",
            "All eight targets are readable from optimized pixels and writable by one-step DreamLite. Advance to shared multi-item F1 train/held-out-dev alignment before recurrence.",
        )
    if pixel_passes == 8:
        return (
            "redesign_dreamlite_updater_only",
            "The frozen Reader/image channel is usable on all eight targets, but the current DreamLite updater does not pass all eight. Restrict the next repair to updater parameterization, conditioning, and optimization.",
        )
    if pixel_passes > 0:
        return (
            "diagnose_target_dependent_visual_channel",
            "Direct-pixel learnability is target-dependent. Diagnose fixed query/option/token properties before claiming a general visual code or changing recurrence.",
        )
    return (
        "test_post_resize_pixel_token_oracle",
        "The current differentiable Reader/loss/preprocessing path did not establish a readable code. Test a post-resize pixel/token oracle before changing recurrence.",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = (
        "arm", "target_index", "family", "segment_id", "gate", "m0_normal_ce",
        "endpoint_normal_ce", "delta_ce", "relative_change", "improved_choice_views",
        "accuracy_delta", "normal_reset_did", "clip_rate", "wall_clock_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            stat = row["target_statistics"]
            writer.writerow(
                (
                    row["arm"], row["target_index"], row["target_family"],
                    row["target_segment_id"], row["passed"], stat["m0_normal_mean_ce"],
                    stat["endpoint_normal_mean_ce"], stat["delta_ce"], stat["relative_change"],
                    stat["improved_choice_views"], stat["accuracy_delta"],
                    stat["normal_reset_difference_in_differences"], row["clip_rate"],
                    row["wall_clock_seconds"],
                )
            )


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# R10 visual-alignment lower bound",
        "",
        "R10 is a one-seed, repeated-target bottleneck diagnostic. It cannot establish formal picture-memory success.",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Direct-pixel oracle: {result['arm_pass_counts']['direct-pixel-oracle']}/8.",
        f"- DreamLite single SET: {result['arm_pass_counts']['dreamlite-single-set']}/8.",
        f"- Reason: {result['reason']}",
        f"- Git commit: `{result['git_commit']}`",
        f"- Fixed F1 payload SHA-256: `{R10_SELECTED_SEGMENTS_SHA256}`",
        "",
        "| Arm | Target | Gate | M0 CE | Endpoint CE | Relative change | Improved views | Accuracy delta | Normal/reset DiD | Clip rate |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        for row in result["arms"][arm]["targets"]:
            stat = row["target_statistics"]
            clip = "n/a" if row["clip_rate"] is None else f"{float(row['clip_rate']):.2%}"
            lines.append(
                f"| {arm} | {row['target_index']} | {'PASS' if row['passed'] else 'FAIL'} | "
                f"{float(stat['m0_normal_mean_ce']):.4f} | {float(stat['endpoint_normal_mean_ce']):.4f} | "
                f"{float(stat['relative_change']):+.2%} | {stat['improved_choice_views']}/4 | "
                f"{float(stat['accuracy_delta']):+.3f} | "
                f"{float(stat['normal_reset_difference_in_differences']):+.4f} | {clip} |"
            )
    lines.extend(
        (
            "",
            "Every target must pass every preregistered endpoint condition. Training loss, an intermediate checkpoint, or partial target success cannot override this decision.",
            "",
        )
    )
    return "\n".join(lines)


def compare(run_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R10 comparison refuses a non-empty output directory.")
    arms: dict[str, dict[str, Any]] = {}
    all_targets: list[dict[str, Any]] = []
    for arm in ARMS:
        targets = [
            _validate_target(run_root / arm / f"target-{index:02d}", arm, index)
            for index in range(8)
        ]
        all_targets.extend(targets)
        arms[arm] = {
            "pass_count": sum(bool(row["passed"]) for row in targets),
            "passed_target_indices": [row["target_index"] for row in targets if row["passed"]],
            "failed_target_indices": [row["target_index"] for row in targets if not row["passed"]],
            "targets": targets,
        }
    commits = {row["git_commit"] for row in all_targets}
    if commits != {EXPECTED_GIT_COMMIT}:
        raise ValueError(f"R10 immutable git drift across targets: {commits}")
    pixel_passes = arms["direct-pixel-oracle"]["pass_count"]
    dreamlite_passes = arms["dreamlite-single-set"]["pass_count"]
    decision, reason = _decision(pixel_passes, dreamlite_passes)
    result = {
        "schema": COMPARISON_SCHEMA,
        "status": "completed",
        "formal_success_claim": False,
        "formal_success_reason": (
            "R10 is a one-seed per-target visual read/write lower-bound diagnostic; formal success "
            "still requires shared training, held-out ID/OOD evaluation, multiple seeds, and causal controls."
        ),
        "decision": decision,
        "reason": reason,
        "arm_pass_counts": {arm: arms[arm]["pass_count"] for arm in ARMS},
        "git_commit": EXPECTED_GIT_COMMIT,
        "selected_segments_sha256": R10_SELECTED_SEGMENTS_SHA256,
        "arms": arms,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", result)
    _write_csv(output_dir / "target_summary.csv", all_targets)
    (output_dir / "REPORT.md").write_text(_report(result), encoding="utf-8")
    artifacts = []
    for path in sorted(value for value in output_dir.iterdir() if value.is_file()):
        if path.name == "DELIVERY_MANIFEST.json":
            continue
        artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    _write_json(
        output_dir / "DELIVERY_MANIFEST.json",
        {
            "schema": DELIVERY_SCHEMA,
            "artifacts": artifacts,
            "source_inventory_sha256": {
                arm: {str(row["target_index"]): row["inventory_sha256"] for row in arms[arm]["targets"]}
                for arm in ARMS
            },
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run_root, args.output_dir)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "arm_pass_counts": result["arm_pass_counts"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
