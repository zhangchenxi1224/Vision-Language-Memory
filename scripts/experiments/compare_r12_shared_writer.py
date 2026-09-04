"""Integrity-check and compare the preregistered R12 shared-writer arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARMS = ("conditioned", "constant-control")
SPLITS = ("train_audit", "dev_select", "dev_final")
SPLIT_COUNTS = {"train_audit": 36, "dev_select": 24, "dev_final": 24}
EXPECTED_COMMIT = "c401954e5624e99347306a60c3f86202c941ab34"
EXPECTED_CONFIG_SHA256 = "85712f6e53fee8c83366cd3ca41f132e9152288b6226d7f330bcf5f49e7705e1"
EXPECTED_OPTIMIZER_STEPS = 1152
EXPECTED_MICRO_STEPS = 4608
EXPECTED_CHECKPOINT_STEPS = [0, 288, 576, 864, 1152]
SUMMARY_SCHEMA = "vision_memory.r12-shared-event-latent-writer-summary.v1"
MANIFEST_SCHEMA = "vision_memory.r12-shared-event-latent-writer-manifest.v1"
TECHNICAL_SCHEMA = "vision_memory.r12-shared-event-latent-writer-technical-gate.v1"
TERMINAL_SCHEMA = "vision_memory.r12-shared-writer-arm-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r12-shared-writer-arm-inventory.v1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R12 expected a JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"R12 expected JSON objects in {path}.")
    return rows


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


def _validate_inventory(root: Path) -> dict[str, Any]:
    inventory_path = root / "artifact_inventory.json"
    inventory = _load(inventory_path)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"R12 invalid inventory schema: {root}")
    entries = inventory.get("artifacts")
    if not isinstance(entries, list):
        raise ValueError(f"R12 inventory artifacts are missing: {root}")
    listed = {str(entry.get("path")): entry for entry in entries if isinstance(entry, Mapping)}
    observed = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.name != inventory_path.name
    }
    if set(listed) != set(observed):
        missing = sorted(set(observed) - set(listed))
        extra = sorted(set(listed) - set(observed))
        raise ValueError(f"R12 inventory/file-set mismatch: missing={missing}, extra={extra}")
    for relative, path in observed.items():
        entry = listed[relative]
        if int(entry.get("bytes", -1)) != path.stat().st_size or entry.get("sha256") != _sha256(path):
            raise ValueError(f"R12 inventory binding mismatch: {root}:{relative}")
    return inventory


def _required_artifacts() -> set[str]:
    required = {
        "launch.json",
        "stdout.log",
        "stderr.log",
        "terminal.json",
        "run/manifest.json",
        "run/technical_gate.json",
        "run/r12_shared_writer_summary.json",
        "run/micro_metrics.jsonl",
        "run/optimizer_metrics.jsonl",
    }
    for step in EXPECTED_CHECKPOINT_STEPS:
        required.add(f"run/checkpoints/step-{step:04d}.pt")
        required.add(f"run/checkpoint_diagnostics/step-{step:04d}.json")
    for split in SPLITS:
        required.add(f"run/{split}_evaluation_rows.jsonl")
        required.add(f"run/{split}_statistics.json")
    return required


def _finite_fields(values: Iterable[Any], *, label: str) -> None:
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError(f"R12 non-finite metric in {label}.")


def _validate_evaluation_rows(root: Path, split: str, rows: Sequence[Mapping[str, Any]]) -> None:
    expected_targets = SPLIT_COUNTS[split]
    if len(rows) != expected_targets * 2 * 3 * 4:
        raise ValueError(f"R12 {split} evaluation row count drift: {len(rows)}")
    cells = {
        (
            str(row.get("pair_unit")),
            str(row.get("checkpoint")),
            str(row.get("condition")),
            int(row.get("view_index", -1)),
        )
        for row in rows
    }
    if len(cells) != len(rows):
        raise ValueError(f"R12 duplicate evaluation cells: {root}:{split}")
    targets = {cell[0] for cell in cells}
    checkpoints = {cell[1] for cell in cells}
    conditions = {cell[2] for cell in cells}
    views = {cell[3] for cell in cells}
    if (
        len(targets) != expected_targets
        or checkpoints != {"m0", "shared_step1152"}
        or conditions != {"normal", "reset", "donor"}
        or views != {0, 1, 2, 3}
    ):
        raise ValueError(f"R12 incomplete evaluation cells: {root}:{split}")
    _finite_fields((row["ce"] for row in rows), label=f"{split} CE")


def _validate_arm(root: Path, expected_arm: str) -> dict[str, Any]:
    _validate_inventory(root)
    observed_files = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    missing = sorted(_required_artifacts() - observed_files)
    if missing:
        raise ValueError(f"R12 required artifacts are missing for {expected_arm}: {missing}")

    terminal = _load(root / "terminal.json")
    summary_path = root / "run" / "r12_shared_writer_summary.json"
    manifest_path = root / "run" / "manifest.json"
    technical_path = root / "run" / "technical_gate.json"
    summary = _load(summary_path)
    manifest = _load(manifest_path)
    technical = _load(technical_path)
    if (
        terminal.get("schema") != TERMINAL_SCHEMA
        or terminal.get("status") != "completed_diagnostic"
        or terminal.get("passed") is not True
        or terminal.get("arm") != expected_arm
        or not all(terminal.get("execution_checks", {}).values())
    ):
        raise ValueError(f"R12 invalid controller terminal: {expected_arm}")
    expected_hashes = {
        "summary_sha256": _sha256(summary_path),
        "manifest_sha256": _sha256(manifest_path),
        "technical_gate_sha256": _sha256(technical_path),
    }
    if any(terminal.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError(f"R12 terminal hash mismatch: {expected_arm}")
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("status") != "completed"
        or summary.get("arm") != expected_arm
        or summary.get("git_commit") != EXPECTED_COMMIT
        or summary.get("config_sha256") != EXPECTED_CONFIG_SHA256
        or summary.get("optimizer_steps") != EXPECTED_OPTIMIZER_STEPS
        or summary.get("micro_steps") != EXPECTED_MICRO_STEPS
        or summary.get("checkpoint_steps_observed") != EXPECTED_CHECKPOINT_STEPS
        or summary.get("endpoint") != "shared_step1152"
        or summary.get("full_success_claim_allowed") is not False
        or summary.get("diagnostic_only_not_formal_success") is not True
    ):
        raise ValueError(f"R12 invalid scientific summary: {expected_arm}")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("arm") != expected_arm
        or manifest.get("git_commit") != EXPECTED_COMMIT
        or technical.get("schema") != TECHNICAL_SCHEMA
        or technical.get("passed") is not True
        or summary.get("gates", {}).get("technical_gate") is not True
        or summary.get("gates", {}).get("formal_success_gate") is not False
    ):
        raise ValueError(f"R12 technical or manifest gate failed: {expected_arm}")

    pass_counts = summary.get("gates", {}).get("split_target_pass_counts")
    if not isinstance(pass_counts, Mapping) or set(pass_counts) != set(SPLITS):
        raise ValueError(f"R12 split pass counts are incomplete: {expected_arm}")
    statistics = summary.get("split_statistics")
    if not isinstance(statistics, Mapping):
        raise ValueError(f"R12 split statistics are missing: {expected_arm}")
    for split in SPLITS:
        values = statistics.get(split)
        if not isinstance(values, list) or len(values) != SPLIT_COUNTS[split]:
            raise ValueError(f"R12 {split} statistics count drift: {expected_arm}")
        if int(pass_counts[split]) != sum(value.get("target_gate") is True for value in values):
            raise ValueError(f"R12 {split} pass count does not match statistics: {expected_arm}")
        rows = _jsonl(root / "run" / f"{split}_evaluation_rows.jsonl")
        _validate_evaluation_rows(root, split, rows)
    arm_gate = bool(summary["gates"]["arm_gate"])
    if terminal.get("scientific_arm_gate") is not arm_gate:
        raise ValueError(f"R12 controller/scientific gate mismatch: {expected_arm}")
    if expected_arm == "constant-control" and terminal.get("constant_control_arm_gate_false") is not (
        not arm_gate
    ):
        raise ValueError("R12 constant-control terminal gate mismatch.")
    return {
        "root": str(root.resolve()),
        "terminal": terminal,
        "summary": summary,
        "manifest": manifest,
        "technical_gate": technical,
        "source_sha256": {
            **expected_hashes,
            "terminal_sha256": _sha256(root / "terminal.json"),
            "inventory_sha256": _sha256(root / "artifact_inventory.json"),
        },
    }


def _baseline_cells(root: Path) -> dict[tuple[str, str, str, int], tuple[float, bool]]:
    result: dict[tuple[str, str, str, int], tuple[float, bool]] = {}
    for split in SPLITS:
        for row in _jsonl(root / "run" / f"{split}_evaluation_rows.jsonl"):
            if row["checkpoint"] != "m0":
                continue
            key = (split, str(row["pair_unit"]), str(row["condition"]), int(row["view_index"]))
            result[key] = (float(row["ce"]), bool(row["correct"]))
    return result


def _paired_invariants(conditioned: Mapping[str, Any], control: Mapping[str, Any]) -> None:
    left = conditioned["summary"]
    right = control["summary"]
    for key in ("protocol", "implementation_revision", "git_commit", "config_sha256", "endpoint"):
        if left.get(key) != right.get(key):
            raise ValueError(f"R12 paired-arm immutable drift: {key}")
    if left.get("selection_audits") != right.get("selection_audits"):
        raise ValueError("R12 paired-arm selection drift.")
    left_baseline = _baseline_cells(Path(conditioned["root"]))
    right_baseline = _baseline_cells(Path(control["root"]))
    if left_baseline.keys() != right_baseline.keys():
        raise ValueError("R12 paired-arm M0 cell drift.")
    for key in left_baseline:
        left_ce, left_correct = left_baseline[key]
        right_ce, right_correct = right_baseline[key]
        if abs(left_ce - right_ce) > 1e-9 or left_correct != right_correct:
            raise ValueError(f"R12 paired-arm M0 value drift: {key}")


def _aggregate(summary: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        rows = summary["split_statistics"][split]
        result[split] = {
            "targets": len(rows),
            "target_pass_count": sum(row["target_gate"] is True for row in rows),
            "m0_normal_mean_ce": sum(float(row["m0_normal_mean_ce"]) for row in rows) / len(rows),
            "endpoint_normal_mean_ce": sum(float(row["endpoint_normal_mean_ce"]) for row in rows)
            / len(rows),
            "endpoint_donor_mean_ce": sum(float(row["endpoint_donor_mean_ce"]) for row in rows)
            / len(rows),
            "m0_normal_accuracy": sum(float(row["m0_normal_accuracy"]) for row in rows) / len(rows),
            "endpoint_normal_accuracy": sum(float(row["endpoint_normal_accuracy"]) for row in rows)
            / len(rows),
            "endpoint_donor_accuracy": sum(float(row["endpoint_donor_accuracy"]) for row in rows)
            / len(rows),
        }
    return result


def _decision(conditioned: Mapping[str, Any], control: Mapping[str, Any]) -> tuple[str, str, bool]:
    conditioned_gate = bool(conditioned["summary"]["gates"]["arm_gate"])
    control_gate = bool(control["summary"]["gates"]["arm_gate"])
    conditioned_counts = conditioned["summary"]["gates"]["split_target_pass_counts"]
    if conditioned_gate and not control_gate:
        return (
            "advance_to_recurrent_state_algebra",
            "The event-conditioned shared writer passed every fixed train/dev target while the matched constant-input control did not; causal one-SET visual writing generalized to held-out entities.",
            True,
        )
    if conditioned_gate and control_gate:
        return (
            "reject_universal_image_false_positive",
            "Both arms passed, so event conditioning is not necessary and the apparent memory effect fails the matched-control test.",
            False,
        )
    if int(conditioned_counts["train_audit"]) == SPLIT_COUNTS["train_audit"]:
        return (
            "diagnose_shared_writer_generalization",
            "The conditioned writer fit every held-in target but failed at least one held-out target; improve the event-to-code mapping before recurrent training.",
            False,
        )
    return (
        "diagnose_shared_writer_fit_boundary",
        "The conditioned writer did not fit every held-in F1 target; localize event representation, coefficient mapping, and shared latent-basis limitations before expanding scope.",
        False,
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def compare(conditioned_root: Path, control_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R12 comparison refuses a non-empty output directory.")
    conditioned = _validate_arm(conditioned_root, "conditioned")
    control = _validate_arm(control_root, "constant-control")
    _paired_invariants(conditioned, control)
    decision, reason, diagnostic_gate = _decision(conditioned, control)
    aggregates = {
        "conditioned": _aggregate(conditioned["summary"]),
        "constant-control": _aggregate(control["summary"]),
    }
    result = {
        "schema": "vision_memory.r12-shared-writer-comparison.v1",
        "status": "completed",
        "decision": decision,
        "reason": reason,
        "r12_diagnostic_gate": diagnostic_gate,
        "formal_success_claim": False,
        "formal_success_reason": (
            "R12 is a one-seed, one-SET shared-writer diagnostic; formal Picture Memory success still requires recurrent composition, overwrite/clear/interference, fixed full ID/OOD evaluation, multiple seeds, and causal state controls."
        ),
        "source_roots": {arm: value["root"] for arm, value in zip(ARMS, (conditioned, control), strict=True)},
        "source_sha256": {
            arm: value["source_sha256"] for arm, value in zip(ARMS, (conditioned, control), strict=True)
        },
        "aggregates": aggregates,
        "arms": {"conditioned": conditioned["summary"], "constant-control": control["summary"]},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", result)
    rows = []
    for arm in ARMS:
        summary = result["arms"][arm]
        counts = summary["gates"]["split_target_pass_counts"]
        rows.append(
            (
                arm,
                f"{counts['train_audit']}/{SPLIT_COUNTS['train_audit']}",
                f"{counts['dev_select']}/{SPLIT_COUNTS['dev_select']}",
                f"{counts['dev_final']}/{SPLIT_COUNTS['dev_final']}",
                "PASS" if summary["gates"]["arm_gate"] else "FAIL",
            )
        )
    table = _markdown_table(("Arm", "train audit", "dev select", "sealed dev final", "arm gate"), rows)
    report = "\n".join(
        (
            "# R12 shared event-to-latent writer paired diagnostic",
            "",
            f"**Decision:** `{decision}`",
            "",
            reason,
            "",
            table,
            "",
            "This is not formal Picture Memory success. The fixed one-SET diagnostic remains bounded by the preregistered success boundary.",
            "",
        )
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditioned-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.conditioned_root, args.control_root, args.output_dir)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "r12_diagnostic_gate": result["r12_diagnostic_gate"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
