"""Compare the two preregistered R6 source-anchor diagnostic arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R6 summary is not an object: {path}")
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


def _validate(legacy: Mapping[str, Any], anchored: Mapping[str, Any]) -> None:
    for expected_arm, value in (("legacy-pure-noise", legacy), ("source-anchored", anchored)):
        if value.get("schema") != "vision_memory.r6-source-anchor-summary.v1":
            raise ValueError(f"Invalid R6 schema for {expected_arm}.")
        if value.get("status") != "completed" or value.get("arm") != expected_arm:
            raise ValueError(f"Incomplete or mislabelled R6 summary for {expected_arm}.")
        if value.get("full_success_claim_allowed") is not False:
            raise ValueError(f"R6 diagnostic incorrectly allows a formal success claim: {expected_arm}.")
    immutable = ("selected_segments_sha256",)
    drift = {key: (legacy.get(key), anchored.get(key)) for key in immutable if legacy.get(key) != anchored.get(key)}
    if drift:
        raise ValueError(f"R6 paired-arm immutable drift: {drift}")


def _decision(legacy: Mapping[str, Any], anchored: Mapping[str, Any]) -> tuple[str, str]:
    legacy_overfit = bool(legacy["gates"]["hard8_overfit_learnability_gate"])
    anchored_overfit = bool(anchored["gates"]["hard8_overfit_learnability_gate"])
    legacy_dev = bool(legacy["gates"]["fixed_dev_generalization_gate"])
    anchored_dev = bool(anchored["gates"]["fixed_dev_generalization_gate"])
    if anchored_overfit and not legacy_overfit:
        return (
            "advance_source_anchor_full_data_pilot",
            "Only sigma=0.5 passed the matched multi-step learnability gate; pure-noise redraw is supported as a primary R5 failure cause.",
        )
    if legacy_overfit and not anchored_overfit:
        return (
            "reject_source_anchor_keep_legacy_diagnose_generalization",
            "Only the exact R5 update law passed local learnability; source anchoring is rejected.",
        )
    if not legacy_overfit and not anchored_overfit:
        return (
            "reject_sigma_as_sufficient_test_gradient_balancing",
            "Neither arm can overfit the identical hard8 state algebra; edit start sigma is not sufficient and the recorded gradient conflict becomes the next main hypothesis.",
        )
    if anchored_dev and not legacy_dev:
        return (
            "advance_source_anchor_full_data_pilot",
            "Both arms learn locally, but only source anchoring passes unchanged fixed-dev transfer.",
        )
    if legacy_dev and not anchored_dev:
        return (
            "reject_source_anchor_keep_legacy",
            "Both arms learn locally, but only the legacy arm passes unchanged fixed-dev transfer.",
        )
    if not legacy_dev and not anchored_dev:
        return (
            "local_learning_without_generalization_test_gradient_balancing",
            "Both arms overfit but neither transfers to fixed dev; data-gradient interference, not local state reachability, is next.",
        )
    legacy_formal = float(
        legacy["comparisons"]["formal_select_32_endpoint_vs_m0"]["estimate"]
    )
    anchored_formal = float(
        anchored["comparisons"]["formal_select_32_endpoint_vs_m0"]["estimate"]
    )
    return (
        ("advance_source_anchor_full_data_pilot" if anchored_formal < legacy_formal else "keep_legacy_full_data_pilot"),
        "Both arms pass local and fixed-dev gates; the preregistered lower formal endpoint-versus-M0 CE delta breaks the tie.",
    )


def _row(value: Mapping[str, Any]) -> list[str]:
    comparisons = value["comparisons"]
    train = comparisons["train_overfit_hard8_endpoint_vs_m0"]
    formal = comparisons["formal_select_32_endpoint_vs_m0"]
    mechanism = comparisons["mechanism_select_32_endpoint_vs_m0"]
    conflict = value["gradient_conflict_audit"]
    return [
        str(value["arm"]),
        f"{float(value['edit_start_sigma']):.2f}",
        f"{float(train['estimate']):.4f}",
        f"{float(value['overfit_accuracy']['delta']):+.3f}",
        "PASS" if value["gates"]["hard8_overfit_learnability_gate"] else "FAIL",
        f"{float(formal['estimate']):.4f}",
        f"{float(mechanism['estimate']):.4f}",
        "PASS" if value["gates"]["fixed_dev_generalization_gate"] else "FAIL",
        f"{float(conflict['off_diagonal']['negative_fraction']):.3f}",
        f"{float(conflict['gradient_norm']['max_to_min_ratio']):.1f}",
    ]


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def compare(legacy_path: Path, anchored_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R6 comparison refuses a non-empty output directory.")
    legacy = _load(legacy_path)
    anchored = _load(anchored_path)
    _validate(legacy, anchored)
    decision, reason = _decision(legacy, anchored)
    result = {
        "schema": "vision_memory.r6-source-anchor-comparison.v1",
        "status": "completed",
        "decision": decision,
        "reason": reason,
        "formal_success_claim": False,
        "formal_success_reason": "R6 is a one-seed repeated-subset bottleneck diagnostic.",
        "summary_paths": {
            "legacy-pure-noise": str(legacy_path.resolve()),
            "source-anchored": str(anchored_path.resolve()),
        },
        "summary_sha256": {
            "legacy-pure-noise": _sha256(legacy_path),
            "source-anchored": _sha256(anchored_path),
        },
        "arms": {
            "legacy-pure-noise": legacy,
            "source-anchored": anchored,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", result)
    table = _markdown_table(
        (
            "Arm",
            "sigma",
            "hard8 delta CE",
            "hard8 delta acc",
            "hard8 gate",
            "formal delta CE",
            "mechanism delta CE",
            "fixed-dev gate",
            "negative grad cosine",
            "grad max/min",
        ),
        (_row(legacy), _row(anchored)),
    )
    report = "\n".join(
        (
            "# R6 source-anchor paired diagnostic",
            "",
            f"**Decision:** `{decision}`",
            "",
            reason,
            "",
            table,
            "",
            "This result cannot be called formal picture-memory success.  Any advancing arm must next pass the unchanged full-data endpoint, ID/OOD, reset/swap, and multi-seed gates.",
            "",
        )
    )
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--anchored", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.legacy, args.anchored, args.output_dir)
    print(json.dumps({"decision": result["decision"], "formal_success_claim": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
