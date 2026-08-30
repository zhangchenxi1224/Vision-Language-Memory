"""Compare the two preregistered R8 common-descent aggregation arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = ("raw-mean-control", "common-descent-projected-norm-matched")
EXPECTED_MODE = {
    "raw-mean-control": "raw-mean",
    "common-descent-projected-norm-matched": "common-descent-projected-norm-matched",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R8 summary is not an object: {path}")
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


def _validate(raw: Mapping[str, Any], projected: Mapping[str, Any]) -> None:
    for expected_arm, value in zip(ARMS, (raw, projected), strict=True):
        if value.get("schema") != "vision_memory.r8-common-descent-summary.v1":
            raise ValueError(f"Invalid R8 schema for {expected_arm}.")
        if value.get("status") != "completed" or value.get("arm") != expected_arm:
            raise ValueError(f"Incomplete or mislabelled R8 summary for {expected_arm}.")
        if value.get("gradient_aggregation") != EXPECTED_MODE[expected_arm]:
            raise ValueError(f"R8 aggregation mode drift for {expected_arm}.")
        if value.get("edit_start_sigma") != 0.5:
            raise ValueError(f"R8 source-anchor sigma drift for {expected_arm}.")
        if value.get("full_success_claim_allowed") is not False:
            raise ValueError(f"R8 diagnostic incorrectly allows formal success: {expected_arm}.")
        for required in ("implementation_revision", "git_commit"):
            if not value.get(required):
                raise ValueError(f"R8 summary is missing {required}: {expected_arm}.")
    immutable = (
        "selected_segments_sha256",
        "implementation_revision",
        "git_commit",
        "edit_start_sigma",
    )
    drift = {key: (raw.get(key), projected.get(key)) for key in immutable if raw.get(key) != projected.get(key)}
    if drift:
        raise ValueError(f"R8 paired-arm immutable drift: {drift}")
    for comparison in (
        "train_overfit_hard8_endpoint_vs_m0",
        "formal_select_32_endpoint_vs_m0",
        "mechanism_select_32_endpoint_vs_m0",
    ):
        raw_m0 = float(raw["comparisons"][comparison]["m0_mean_ce"])
        projected_m0 = float(projected["comparisons"][comparison]["m0_mean_ce"])
        if abs(raw_m0 - projected_m0) > 1e-9:
            raise ValueError(f"R8 M0 drift for {comparison}: {raw_m0} != {projected_m0}")


def _decision(raw: Mapping[str, Any], projected: Mapping[str, Any]) -> tuple[str, str]:
    raw_technical = bool(raw["gates"]["technical_gate"])
    projected_technical = bool(projected["gates"]["technical_gate"])
    if not raw_technical:
        return "raw_control_technical_failure_no_scientific_decision", "The unchanged raw control failed its technical gate."
    if not projected_technical:
        return (
            "common_descent_technical_failure_no_scientific_decision",
            "The projected arm failed its feasibility, integrity, or intervention gate; repair the solver before interpretation.",
        )
    raw_overfit = bool(raw["gates"]["hard8_overfit_learnability_gate"])
    projected_overfit = bool(projected["gates"]["hard8_overfit_learnability_gate"])
    raw_dev = bool(raw["gates"]["fixed_dev_generalization_gate"])
    projected_dev = bool(projected["gates"]["fixed_dev_generalization_gate"])
    if projected_overfit and not raw_overfit:
        return (
            "advance_common_descent_source_anchor_full_data_pilot",
            "Only common-descent projection passed hard8; destructive batch directions are supported as a primary local bottleneck.",
        )
    if raw_overfit and not projected_overfit:
        return (
            "reject_common_descent_retain_raw_aggregation",
            "Only raw mean passed local learnability; the common-descent projection is rejected.",
        )
    if not raw_overfit and not projected_overfit:
        return (
            "reject_batch_conflict_as_sufficient_test_per_segment_learnability",
            "Neither arm learned hard8 even when every pre-AdamW micro-gradient had nonnegative alignment with the applied direction; decompose learnability per segment before changing rank, data, Reader, or loss.",
        )
    if projected_dev and not raw_dev:
        return (
            "advance_common_descent_source_anchor_full_data_pilot",
            "Both arms overfit, but only common-descent projection passes unchanged fixed-dev transfer.",
        )
    if raw_dev and not projected_dev:
        return (
            "retain_raw_source_anchor_full_data_pilot",
            "Both arms overfit, but only raw mean passes unchanged fixed-dev transfer.",
        )
    if not raw_dev and not projected_dev:
        return (
            "local_learning_without_dev_transfer_retain_raw_diagnose_generalization",
            "Both arms learn locally but neither transfers; retain the simpler raw law and diagnose generalization.",
        )
    raw_formal = float(raw["comparisons"]["formal_select_32_endpoint_vs_m0"]["estimate"])
    projected_formal = float(projected["comparisons"]["formal_select_32_endpoint_vs_m0"]["estimate"])
    return (
        (
            "advance_common_descent_source_anchor_full_data_pilot"
            if projected_formal < raw_formal
            else "retain_raw_source_anchor_full_data_pilot"
        ),
        "Both arms pass local and fixed-dev gates; lower paired formal endpoint-versus-M0 CE delta breaks the tie.",
    )


def _row(value: Mapping[str, Any]) -> list[str]:
    comparisons = value["comparisons"]
    train = comparisons["train_overfit_hard8_endpoint_vs_m0"]
    formal = comparisons["formal_select_32_endpoint_vs_m0"]
    mechanism = comparisons["mechanism_select_32_endpoint_vs_m0"]
    aggregation = value["aggregation_technical_gate"]
    projected_min = aggregation.get("minimum_projected_micro_cosine")
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
        "n/a" if projected_min is None else f"{float(projected_min):.2e}",
        f"{float(value['training_summary']['clip_rate']):.4f}",
    ]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def compare(raw_path: Path, projected_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R8 comparison refuses a non-empty output directory.")
    raw = _load(raw_path)
    projected = _load(projected_path)
    _validate(raw, projected)
    decision, reason = _decision(raw, projected)
    result = {
        "schema": "vision_memory.r8-common-descent-comparison.v1",
        "status": "completed",
        "decision": decision,
        "reason": reason,
        "formal_success_claim": False,
        "formal_success_reason": "R8 is a one-seed repeated-subset bottleneck diagnostic.",
        "summary_paths": {ARMS[0]: str(raw_path.resolve()), ARMS[1]: str(projected_path.resolve())},
        "summary_sha256": {ARMS[0]: _sha256(raw_path), ARMS[1]: _sha256(projected_path)},
        "arms": {ARMS[0]: raw, ARMS[1]: projected},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", result)
    table = _markdown_table(
        (
            "Arm", "aggregation", "hard8 delta CE", "hard8 delta acc", "hard8 gate",
            "formal delta CE", "mechanism delta CE", "fixed-dev gate", "min raw/applied cosine",
            "min applied/micro cosine", "clip rate",
        ),
        (_row(raw), _row(projected)),
    )
    report = "\n".join(
        (
            "# R8 common-descent paired diagnostic", "", f"**Decision:** `{decision}`", "", reason,
            "", table, "",
            "This diagnostic cannot establish formal picture-memory success. Any advancing arm must still pass the unchanged full-data endpoint, ID/OOD, reset/swap, and multi-seed gates.", "",
        )
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--projected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.raw, args.projected, args.output_dir)
    print(json.dumps({"decision": result["decision"], "formal_success_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
