"""Hash-aware in-process steps for the immutable R3 teacher-preparation DAG.

Raw TC0/TF0 reports do not have a digest until their preceding command has
finished.  The scientific validators intentionally require those digests as
explicit inputs.  This helper computes the just-materialized file digest and
immediately invokes the existing fail-closed validator/probe without a shell,
leaving the outer immutable stage runner to bind every final artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
PROBES = ROOT / "scripts" / "probes"
sys.path.insert(0, str(PROBES))

import r3_teacher_feature_compatibility as tf0  # noqa: E402
import validate_r3_teacher_cache_compatibility as validate_tc0  # noqa: E402
import validate_r3_teacher_feature_compatibility as validate_tf0  # noqa: E402
from r3_dag_contract import load_json_object, require_json_values, sha256_file  # noqa: E402


def _existing_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    return resolved


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one hash-aware R3 teacher-preparation substep")
    subparsers = parser.add_subparsers(dest="action", required=True)

    tc0 = subparsers.add_parser("validate-tc0")
    tc0.add_argument("--report", type=Path, required=True)
    tc0.add_argument("--preregistration", type=Path, required=True)
    tc0.add_argument("--expected-commit", required=True)
    tc0.add_argument("--output", type=Path, required=True)

    tf0_raw = subparsers.add_parser("run-tf0")
    tf0_raw.add_argument("--set8-cache", type=Path, required=True)
    tf0_raw.add_argument("--transition16-cache", type=Path, required=True)
    tf0_raw.add_argument("--reader", type=Path, required=True)
    tf0_raw.add_argument("--tc0-validation", type=Path, required=True)
    tf0_raw.add_argument("--preregistration", type=Path, required=True)
    tf0_raw.add_argument("--device", default="cuda:0")
    tf0_raw.add_argument("--output", type=Path, required=True)

    tf0_validation = subparsers.add_parser("validate-tf0")
    tf0_validation.add_argument("--report", type=Path, required=True)
    tf0_validation.add_argument("--tc0-validation", type=Path, required=True)
    tf0_validation.add_argument("--preregistration", type=Path, required=True)
    tf0_validation.add_argument("--expected-commit", required=True)
    tf0_validation.add_argument("--output", type=Path, required=True)

    final = subparsers.add_parser("finalize")
    final.add_argument("--expected-commit", required=True)
    for name in (
        "tc0-raw",
        "tc0-validation",
        "tf0-raw",
        "tf0-validation",
        "t0",
        "set8-calibration",
        "set8-calibration-report",
        "transition16-calibration",
        "transition16-calibration-report",
    ):
        final.add_argument(f"--{name}", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "validate-tc0":
        report = _existing_file(args.report, label="raw TC0 report")
        preregistration = _existing_file(args.preregistration, label="R3 preregistration")
        return validate_tc0.main(
            [
                "--report",
                str(report),
                "--report-sha256",
                sha256_file(report),
                "--preregistration",
                str(preregistration),
                "--preregistration-sha256",
                sha256_file(preregistration),
                "--expected-commit",
                args.expected_commit,
                "--output",
                str(args.output),
            ]
        )
    if args.action == "run-tf0":
        tc0_validation = _existing_file(args.tc0_validation, label="TC0 validation")
        return tf0.main(
            [
                "--set8-cache",
                str(args.set8_cache),
                "--transition16-cache",
                str(args.transition16_cache),
                "--reader",
                str(args.reader),
                "--tc0-validation-report",
                str(tc0_validation),
                "--tc0-validation-report-sha256",
                sha256_file(tc0_validation),
                "--preregistration",
                str(args.preregistration),
                "--device",
                args.device,
                "--output-json",
                str(args.output),
            ]
        )
    if args.action == "validate-tf0":
        report = _existing_file(args.report, label="raw TF0 report")
        tc0_validation = _existing_file(args.tc0_validation, label="TC0 validation")
        preregistration = _existing_file(args.preregistration, label="R3 preregistration")
        return validate_tf0.main(
            [
                "--report",
                str(report),
                "--report-sha256",
                sha256_file(report),
                "--tc0-validation-report",
                str(tc0_validation),
                "--tc0-validation-report-sha256",
                sha256_file(tc0_validation),
                "--preregistration",
                str(preregistration),
                "--preregistration-sha256",
                sha256_file(preregistration),
                "--expected-commit",
                args.expected_commit,
                "--output",
                str(args.output),
            ]
        )

    sources = {
        "tc0_raw": args.tc0_raw,
        "tc0_validation": args.tc0_validation,
        "tf0_raw": args.tf0_raw,
        "tf0_validation": args.tf0_validation,
        "t0": args.t0,
        "set8_calibration": args.set8_calibration,
        "set8_calibration_report": args.set8_calibration_report,
        "transition16_calibration": args.transition16_calibration,
        "transition16_calibration_report": args.transition16_calibration_report,
    }
    artifacts = {
        name: {
            "path": str(_existing_file(path, label=name)),
            "sha256": sha256_file(_existing_file(path, label=name)),
        }
        for name, path in sources.items()
    }
    checks = {
        "tc0_raw": {"passed": True},
        "tc0_validation": {
            "passed": True,
            "protocol": "R3-TC0-cache-forward-compatibility-validation.v1",
        },
        "tf0_raw": {"passed": True},
        "tf0_validation": {
            "passed": True,
            "protocol": "R3-TF0-feature-backend-compatibility-validation.v1",
        },
        "t0": {"passed": True, "probe": "teacher_t0_real_qwen_integrity_upper_bound"},
        "set8_calibration": {"schema": "vision_memory.teacher-calibration-file.v1", "split": "train"},
        "set8_calibration_report": {
            "schema": "vision_memory.r3-teacher-calibration-report.v1",
            "suite": "set8",
        },
        "transition16_calibration": {
            "schema": "vision_memory.teacher-calibration-file.v1",
            "split": "train",
        },
        "transition16_calibration_report": {
            "schema": "vision_memory.r3-teacher-calibration-report.v1",
            "suite": "transition16",
        },
    }
    for name, required in checks.items():
        value = load_json_object(Path(artifacts[name]["path"]))
        require_json_values(value, required, name)
    for suite in ("set8", "transition16"):
        report = load_json_object(Path(artifacts[f"{suite}_calibration_report"]["path"]))
        if report.get("calibration_file_sha256") != artifacts[f"{suite}_calibration"]["sha256"]:
            raise ValueError(f"{suite} calibration report does not bind its generated calibration file")
    payload = {
        "schema_version": 1,
        "protocol": "r3-inspire-teacher-preparation-final.v1",
        "passed": True,
        "expected_commit": args.expected_commit,
        "strict_order": ["R3-TC0", "R3-TF0", "T0", "CAL-Set8", "CAL-Transition16"],
        "artifacts": artifacts,
    }
    destination = args.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
