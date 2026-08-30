"""Validate and aggregate the eight preregistered R9 target diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA = "vision_memory.r9-individual-learnability-summary.v1"
TERMINAL_SCHEMA = "vision_memory.r9-individual-target-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r9-individual-target-inventory.v1"
COMPARISON_SCHEMA = "vision_memory.r9-individual-learnability-comparison.v1"
DELIVERY_SCHEMA = "vision_memory.r9-individual-learnability-delivery.v1"
SELECTED_SEGMENTS_SHA256 = "eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6"
TARGETS = (
    ("F2", "r5-f2-d1803aae745ed83f142b0bae"),
    ("F2", "r5-f2-6db739261bbb0fc74169dc1f"),
    ("F3", "r5-f3-dbc9791cf4ba3569afc7a3d0"),
    ("F3", "r5-f3-913c95f41d8e72f9b0aabac4"),
    ("F5", "r5-f5-9fec14ebb444f5f1fe201087"),
    ("F5", "r5-f5-56ef217cc7251269d003b5c8"),
    ("F6", "r5-f6-b6302052a821cfd9e9d9261e"),
    ("F6", "r5-f6-1ccc652bfea4b3e698683232"),
)


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_inventory(root: Path) -> str:
    inventory_path = root / "artifact_inventory.json"
    inventory = _load(inventory_path)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"R9 inventory schema mismatch: {root}")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"R9 artifact inventory is empty: {root}")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"R9 inventory record is malformed: {root}")
        relative = str(record.get("path"))
        if relative in names:
            raise ValueError(f"R9 inventory contains a duplicate path: {relative}")
        names.add(relative)
        path = root / relative
        if not path.is_file():
            raise ValueError(f"R9 inventory artifact is missing: {path}")
        if path.stat().st_size != int(record.get("bytes", -1)) or _sha256(path) != record.get("sha256"):
            raise ValueError(f"R9 inventory artifact failed size/SHA validation: {path}")
    required = {
        "launch.json",
        "terminal.json",
        "run/manifest.json",
        "run/metrics.jsonl",
        "run/micro_metrics.jsonl",
        "run/r9_summary.json",
        "run/endpoint_ema.pt",
        "run/endpoint_raw.pt",
        "run/overfit_evaluation_rows.jsonl",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"R9 inventory lacks required artifacts for {root}: {missing}")
    return _sha256(inventory_path)


def _expected_gate(statistics: Mapping[str, Any], *, technical_gate: bool) -> bool:
    return (
        technical_gate
        and float(statistics["relative_change"]) <= -0.20
        and int(statistics["improved_choice_views"]) == 4
        and float(statistics["accuracy_delta"]) >= 0.25
        and float(statistics["normal_reset_difference_in_differences"]) < 0.0
    )


def _validate_target(root: Path, index: int) -> dict[str, Any]:
    expected_family, expected_segment = TARGETS[index]
    terminal_path = root / "terminal.json"
    summary_path = root / "run" / "r9_summary.json"
    terminal = _load(terminal_path)
    summary = _load(summary_path)
    if terminal.get("schema") != TERMINAL_SCHEMA:
        raise ValueError(f"R9 terminal schema mismatch: target {index}")
    if terminal.get("status") != "completed_diagnostic" or terminal.get("passed") is not True:
        raise ValueError(f"R9 target {index} did not complete its technical gate")
    if terminal.get("child_exit_code") != 0 or terminal.get("scientific_success_claim") is not False:
        raise ValueError(f"R9 target {index} terminal integrity failed")
    if terminal.get("summary_sha256") != _sha256(summary_path):
        raise ValueError(f"R9 target {index} terminal/summary hash binding failed")
    if summary.get("schema") != SUMMARY_SCHEMA or summary.get("status") != "completed":
        raise ValueError(f"R9 target {index} summary is incomplete")
    if (
        summary.get("target_index") != index
        or summary.get("target_family") != expected_family
        or summary.get("target_segment_id") != expected_segment
        or terminal.get("target_index") != index
        or terminal.get("target_segment_id") != expected_segment
    ):
        raise ValueError(f"R9 target identity drift: target {index}")
    if summary.get("selected_segments") != [segment for _family, segment in TARGETS]:
        raise ValueError(f"R9 selected segment order drift: target {index}")
    if summary.get("selected_segments_sha256") != SELECTED_SEGMENTS_SHA256:
        raise ValueError(f"R9 hard8 SHA drift: target {index}")
    if (
        summary.get("gradient_aggregation") != "single-target-one-eighth"
        or not math.isclose(float(summary.get("gradient_coefficient", math.nan)), 0.125)
        or summary.get("executed_micro_segments") != 128
        or summary.get("schedule_cursor_segments") != 1024
        or summary.get("checkpoint_steps_observed") != [0, 32, 64, 96, 128]
    ):
        raise ValueError(f"R9 fixed execution contract drift: target {index}")
    gates = summary.get("gates")
    if not isinstance(gates, Mapping) or gates.get("technical_gate") is not True:
        raise ValueError(f"R9 target {index} technical gate is not valid")
    if gates.get("formal_success_gate") is not False or summary.get("full_success_claim_allowed") is not False:
        raise ValueError(f"R9 target {index} incorrectly permits formal success")
    statistics = summary.get("target_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError(f"R9 target {index} statistics are missing")
    numeric = (
        "m0_normal_mean_ce",
        "endpoint_normal_mean_ce",
        "delta_ce",
        "relative_change",
        "accuracy_delta",
        "normal_reset_difference_in_differences",
    )
    if any(not math.isfinite(float(statistics.get(key, math.nan))) for key in numeric):
        raise ValueError(f"R9 target {index} statistics are non-finite")
    expected_gate = _expected_gate(statistics, technical_gate=True)
    if gates.get("target_individual_learnability_gate") is not expected_gate:
        raise ValueError(f"R9 target {index} summary gate does not match fixed thresholds")
    if terminal.get("target_individual_learnability_gate") is not expected_gate:
        raise ValueError(f"R9 target {index} terminal gate does not match its summary")
    inventory_sha = _validate_inventory(root)
    aggregation = summary.get("aggregation_technical_gate", {})
    if aggregation.get("passed") is not True:
        raise ValueError(f"R9 target {index} gradient aggregation gate failed")
    return {
        "target_index": index,
        "target_family": expected_family,
        "target_segment_id": expected_segment,
        "passed": expected_gate,
        "target_statistics": dict(statistics),
        "clip_rate": float(summary["training_summary"]["clip_rate"]),
        "minimum_raw_vs_applied_cosine": aggregation.get("minimum_raw_vs_applied_cosine"),
        "maximum_scale_relative_error": aggregation.get("maximum_scale_relative_error"),
        "wall_clock_seconds": summary.get("wall_clock_seconds"),
        "source_root": str(root.resolve()),
        "summary_sha256": _sha256(summary_path),
        "terminal_sha256": _sha256(terminal_path),
        "inventory_sha256": inventory_sha,
        "git_commit": summary.get("git_commit"),
        "implementation_revision": summary.get("implementation_revision"),
    }


def _decision(pass_count: int) -> tuple[str, str]:
    if pass_count == 8:
        return (
            "all_targets_individually_learnable_diagnose_realized_shared_update_interference",
            "All eight transitions are learnable alone at their original one-eighth coefficient; basic per-transition representability is supported and the next test must target realized shared-parameter/optimizer interference.",
        )
    if pass_count > 0:
        return (
            "transition_heterogeneity_repair_failing_structural_property",
            "Only a subset of transitions is learnable in isolation; fixed transition structure is causal and failing targets must be repaired without averaging them into passing targets.",
        )
    return (
        "reject_batch_aggregation_reopen_recurrent_alignment_and_temporal_credit",
        "No transition is learnable alone at its original coefficient; simultaneous aggregation is not the sufficient bottleneck, so the recurrent visual alignment/update law and temporal credit path must be reopened against the single-step SET positive control.",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    headers = (
        "target_index", "family", "segment_id", "gate", "m0_normal_ce", "endpoint_normal_ce",
        "delta_ce", "relative_change", "improved_choice_views", "accuracy_delta", "normal_reset_did",
        "clip_rate", "min_raw_applied_cosine", "max_scale_relative_error", "wall_clock_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            stat = row["target_statistics"]
            writer.writerow(
                (
                    row["target_index"], row["target_family"], row["target_segment_id"], row["passed"],
                    stat["m0_normal_mean_ce"], stat["endpoint_normal_mean_ce"], stat["delta_ce"],
                    stat["relative_change"], stat["improved_choice_views"], stat["accuracy_delta"],
                    stat["normal_reset_difference_in_differences"], row["clip_rate"],
                    row["minimum_raw_vs_applied_cosine"], row["maximum_scale_relative_error"],
                    row["wall_clock_seconds"],
                )
            )


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# R9 individual-learnability decomposition",
        "",
        "This is a one-seed repeated-hard8 bottleneck diagnostic. It cannot establish formal picture-memory success.",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Result: {result['pass_count']}/8 target gates passed.",
        f"- Reason: {result['reason']}",
        f"- Git commit: `{result['git_commit']}`",
        f"- Hard8 SHA-256: `{SELECTED_SEGMENTS_SHA256}`",
        "",
        "| Target | Family | Gate | M0 CE | Endpoint CE | Relative change | Improved views | Accuracy delta | Normal/reset DiD | Clip rate |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["targets"]:
        stat = row["target_statistics"]
        lines.append(
            f"| {row['target_index']} | {row['target_family']} | {'PASS' if row['passed'] else 'FAIL'} | "
            f"{float(stat['m0_normal_mean_ce']):.4f} | {float(stat['endpoint_normal_mean_ce']):.4f} | "
            f"{float(stat['relative_change']):+.2%} | {stat['improved_choice_views']}/4 | "
            f"{float(stat['accuracy_delta']):+.3f} | "
            f"{float(stat['normal_reset_difference_in_differences']):+.4f} | {float(row['clip_rate']):.2%} |"
        )
    lines.extend(
        (
            "",
            "A target passes only when all preregistered endpoint conditions hold. Training loss, non-target improvement, or an intermediate checkpoint cannot rescue a failed target.",
            "",
        )
    )
    return "\n".join(lines)


def compare(run_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R9 comparison refuses a non-empty output directory.")
    targets = [_validate_target(run_root / f"target-{index:02d}", index) for index in range(8)]
    for key in ("git_commit", "implementation_revision"):
        values = {row.get(key) for row in targets}
        if len(values) != 1 or None in values:
            raise ValueError(f"R9 immutable drift in {key}: {values}")
    pass_count = sum(bool(row["passed"]) for row in targets)
    decision, reason = _decision(pass_count)
    result = {
        "schema": COMPARISON_SCHEMA,
        "status": "completed",
        "formal_success_claim": False,
        "formal_success_reason": "R9 is a one-seed repeated-subset bottleneck diagnostic.",
        "decision": decision,
        "reason": reason,
        "pass_count": pass_count,
        "failed_target_indices": [row["target_index"] for row in targets if not row["passed"]],
        "passed_target_indices": [row["target_index"] for row in targets if row["passed"]],
        "git_commit": targets[0]["git_commit"],
        "implementation_revision": targets[0]["implementation_revision"],
        "selected_segments_sha256": SELECTED_SEGMENTS_SHA256,
        "targets": targets,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", result)
    _write_csv(output_dir / "target_summary.csv", targets)
    (output_dir / "REPORT.md").write_text(_report(result), encoding="utf-8")
    artifacts = []
    for path in sorted(value for value in output_dir.iterdir() if value.is_file()):
        if path.name == "DELIVERY_MANIFEST.json":
            continue
        artifacts.append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    _write_json(
        output_dir / "DELIVERY_MANIFEST.json",
        {
            "schema": DELIVERY_SCHEMA,
            "artifacts": artifacts,
            "source_inventory_sha256": {
                str(row["target_index"]): row["inventory_sha256"] for row in targets
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
                "pass_count": result["pass_count"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
