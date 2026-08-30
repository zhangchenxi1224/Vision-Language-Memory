"""Compare the two preregistered R7 gradient-aggregation arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = ("raw-mean-control", "unit-balanced-norm-matched")
EXPECTED_MODE = {
    "raw-mean-control": "raw-mean",
    "unit-balanced-norm-matched": "unit-balanced-norm-matched",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R7 summary is not an object: {path}")
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


def _validate(raw: Mapping[str, Any], balanced: Mapping[str, Any]) -> None:
    for expected_arm, value in zip(ARMS, (raw, balanced), strict=True):
        if value.get("schema") != "vision_memory.r7-gradient-balance-summary.v1":
            raise ValueError(f"Invalid R7 schema for {expected_arm}.")
        if value.get("status") != "completed" or value.get("arm") != expected_arm:
            raise ValueError(f"Incomplete or mislabelled R7 summary for {expected_arm}.")
        if value.get("gradient_aggregation") != EXPECTED_MODE[expected_arm]:
            raise ValueError(f"R7 aggregation mode drift for {expected_arm}.")
        if value.get("edit_start_sigma") != 0.5:
            raise ValueError(f"R7 source-anchor sigma drift for {expected_arm}.")
        if value.get("full_success_claim_allowed") is not False:
            raise ValueError(f"R7 diagnostic incorrectly allows formal success: {expected_arm}.")
        for required in ("implementation_revision", "git_commit"):
            if not value.get(required):
                raise ValueError(f"R7 summary is missing {required}: {expected_arm}.")
    immutable = (
        "selected_segments_sha256",
        "implementation_revision",
        "git_commit",
        "edit_start_sigma",
    )
    drift = {key: (raw.get(key), balanced.get(key)) for key in immutable if raw.get(key) != balanced.get(key)}
    if drift:
        raise ValueError(f"R7 paired-arm immutable drift: {drift}")
    for comparison in (
        "train_overfit_hard8_endpoint_vs_m0",
        "formal_select_32_endpoint_vs_m0",
        "mechanism_select_32_endpoint_vs_m0",
    ):
        raw_m0 = float(raw["comparisons"][comparison]["m0_mean_ce"])
        balanced_m0 = float(balanced["comparisons"][comparison]["m0_mean_ce"])
        if abs(raw_m0 - balanced_m0) > 1e-9:
            raise ValueError(f"R7 M0 drift for {comparison}: {raw_m0} != {balanced_m0}")


def _decision(raw: Mapping[str, Any], balanced: Mapping[str, Any]) -> tuple[str, str]:
    raw_overfit = bool(raw["gates"]["hard8_overfit_learnability_gate"])
    balanced_overfit = bool(balanced["gates"]["hard8_overfit_learnability_gate"])
    raw_dev = bool(raw["gates"]["fixed_dev_generalization_gate"])
    balanced_dev = bool(balanced["gates"]["fixed_dev_generalization_gate"])
    if balanced_overfit and not raw_overfit:
        return (
            "advance_unit_balanced_source_anchor_full_data_pilot",
            "Only equal-unit aggregation passed the unchanged hard8 gate; raw norm dominance is supported as a primary local failure cause.",
        )
    if raw_overfit and not balanced_overfit:
        return (
            "reject_unit_balance_retain_raw_aggregation",
            "Only the raw-mean control passed local learnability; unit balancing is rejected.",
        )
    if not raw_overfit and not balanced_overfit:
        return (
            "reject_unit_balance_as_sufficient_test_conflict_projection",
            "Neither arm learned the identical hard8 state algebra; equal weighting is insufficient and deterministic conflict projection is the next fixed-bottleneck test.",
        )
    if balanced_dev and not raw_dev:
        return (
            "advance_unit_balanced_source_anchor_full_data_pilot",
            "Both arms learn locally, but only unit balancing passes unchanged fixed-dev transfer.",
        )
    if raw_dev and not balanced_dev:
        return (
            "retain_raw_source_anchor_full_data_pilot",
            "Both arms learn locally, but only raw aggregation passes unchanged fixed-dev transfer.",
        )
    if not raw_dev and not balanced_dev:
        return (
            "local_learning_without_dev_transfer_retain_raw_diagnose_generalization",
            "Both arms overfit but neither transfers; retain the simpler raw law while diagnosing generalization.",
        )
    raw_formal = float(raw["comparisons"]["formal_select_32_endpoint_vs_m0"]["estimate"])
    balanced_formal = float(balanced["comparisons"]["formal_select_32_endpoint_vs_m0"]["estimate"])
    return (
        (
            "advance_unit_balanced_source_anchor_full_data_pilot"
            if balanced_formal < raw_formal
            else "retain_raw_source_anchor_full_data_pilot"
        ),
        "Both arms pass local and fixed-dev gates; lower paired formal endpoint-versus-M0 CE delta breaks the tie.",
    )


def _optimizer_aggregation_records(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return value["aggregation_technical_gate"]


def _row(value: Mapping[str, Any]) -> list[str]:
    comparisons = value["comparisons"]
    train = comparisons["train_overfit_hard8_endpoint_vs_m0"]
    formal = comparisons["formal_select_32_endpoint_vs_m0"]
    mechanism = comparisons["mechanism_select_32_endpoint_vs_m0"]
    aggregation = _optimizer_aggregation_records(value)
    return [
        str(value["arm"]),
        str(value["gradient_aggregation"]),
        f"{float(train['estimate']):.4f}",
        f"{float(value['overfit_accuracy']['delta']):+.3f}",
        "PASS" if value["gates"]["hard8_overfit_learnability_gate"] else "FAIL",
        f"{float(formal['estimate']):.4f}",
        f"{float(mechanism['estimate']):.4f}",
        "PASS" if value["gates"]["fixed_dev_generalization_gate"] else "FAIL",
        f"{float(aggregation['minimum_raw_vs_applied_cosine']):.4f}",
        f"{float(aggregation['maximum_norm_match_relative_error']):.2e}",
        f"{float(value['training_summary']['clip_rate']):.4f}",
    ]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def compare(raw_path: Path, balanced_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R7 comparison refuses a non-empty output directory.")
    raw = _load(raw_path)
    balanced = _load(balanced_path)
    _validate(raw, balanced)
    decision, reason = _decision(raw, balanced)
    result = {
        "schema": "vision_memory.r7-gradient-balance-comparison.v1",
        "status": "completed",
        "decision": decision,
        "reason": reason,
        "formal_success_claim": False,
        "formal_success_reason": "R7 is a one-seed repeated-subset bottleneck diagnostic.",
        "summary_paths": {
            "raw-mean-control": str(raw_path.resolve()),
            "unit-balanced-norm-matched": str(balanced_path.resolve()),
        },
        "summary_sha256": {
            "raw-mean-control": _sha256(raw_path),
            "unit-balanced-norm-matched": _sha256(balanced_path),
        },
        "arms": {
            "raw-mean-control": raw,
            "unit-balanced-norm-matched": balanced,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", result)
    table = _markdown_table(
        (
            "Arm",
            "aggregation",
            "hard8 delta CE",
            "hard8 delta acc",
            "hard8 gate",
            "formal delta CE",
            "mechanism delta CE",
            "fixed-dev gate",
            "min raw/applied cosine",
            "max norm error",
            "clip rate",
        ),
        (_row(raw), _row(balanced)),
    )
    report = "\n".join(
        (
            "# R7 gradient-balance paired diagnostic",
            "",
            f"**Decision:** `{decision}`",
            "",
            reason,
            "",
            table,
            "",
            "This result cannot be called formal picture-memory success. Any advancing arm must still pass the unchanged full-data endpoint, ID/OOD, reset/swap, and multi-seed gates.",
            "",
        )
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--balanced", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.raw, args.balanced, args.output_dir)
    print(json.dumps({"decision": result["decision"], "formal_success_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
