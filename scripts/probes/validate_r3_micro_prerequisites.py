from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def validate_prerequisites(
    *,
    scorer_s0: Mapping[str, Any],
    technical: Mapping[str, Any],
    teacher_t0: Mapping[str, Any] | None,
    training_regime: str,
    expected_commit: str,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a lowercase full Git commit.")
    errors: list[str] = []
    if scorer_s0.get("schema_version") != 1 or scorer_s0.get("probe") != "r3_s0_qwen_scorer_contract":
        errors.append("R3-S0 scorer report identity is invalid")
    if scorer_s0.get("passed") is not True:
        errors.append("R3-S0 scorer contract did not pass")
    s0_contract = scorer_s0.get("contract")
    if not isinstance(s0_contract, Mapping) or s0_contract.get("reader_loss_mode") != "listwise-choice":
        errors.append("R3-S0 scorer report is not listwise-choice")
    if scorer_s0.get("summary") != {
        "views_passed": 8,
        "views_required": 8,
        "joint_tokenization_views_passed": 8,
        "train_eval_views_passed": 8,
        "repeat_eval_views_passed": 8,
    }:
        errors.append("R3-S0 scorer report does not pass all eight cyclic/reverse views")
    s0_provenance = scorer_s0.get("provenance")
    s0_git = s0_provenance.get("git") if isinstance(s0_provenance, Mapping) else None
    if not isinstance(s0_git, Mapping) or s0_git.get("commit") != expected_commit or s0_git.get("clean") is not True:
        errors.append("R3-S0 provenance is not the expected clean commit")
    expected_gates = ["G4-L", "G5-L", "G6-L", "DL-S"]
    if technical.get("protocol") != "R3-technical-listwise-v1":
        errors.append("technical protocol is not R3-technical-listwise-v1")
    if technical.get("through") != "DL-S" or technical.get("required_gates") != expected_gates:
        errors.append("technical report does not cover the complete G4-L/G5-L/G6-L/DL-S chain")
    if technical.get("passed") is not True or technical.get("errors") not in ([], None):
        errors.append("technical gate report did not pass fail-closed")
    if technical.get("git_commit") != expected_commit:
        errors.append("technical gate report is not bound to the expected clean commit")
    technical_checks = technical.get("checks")
    if not isinstance(technical_checks, Mapping) or not all(
        isinstance(technical_checks.get(gate), Mapping) and technical_checks[gate].get("valid") is True
        for gate in expected_gates
    ):
        errors.append("technical report is missing a valid required gate")

    if training_regime not in {"qa_only", "teacher_assisted"}:
        raise ValueError("training_regime must be qa_only or teacher_assisted.")
    if training_regime == "teacher_assisted":
        if not isinstance(teacher_t0, Mapping):
            errors.append("teacher T0 is required for teacher_assisted stages")
        else:
            if teacher_t0.get("schema_version") != 1:
                errors.append("teacher T0 schema_version is not 1")
            if teacher_t0.get("probe") != "teacher_t0_real_qwen_integrity_upper_bound":
                errors.append("teacher T0 probe identity is invalid")
            if teacher_t0.get("passed") is not True:
                errors.append("teacher T0 did not pass")
            preregistered_inputs = teacher_t0.get("preregistered_inputs")
            if (
                not isinstance(preregistered_inputs, Mapping)
                or preregistered_inputs.get("passed") is not True
                or not isinstance(preregistered_inputs.get("checks"), Mapping)
                or not preregistered_inputs["checks"]
                or not all(value is True for value in preregistered_inputs["checks"].values())
            ):
                errors.append("teacher T0 is not bound to every prospective preregistered input")
            for field in ("cache_integrity", "cross_split_fail_closed", "upper_bound"):
                value = teacher_t0.get(field)
                if not isinstance(value, Mapping) or value.get("passed") is not True:
                    errors.append(f"teacher T0 {field} did not pass")
            mutations = teacher_t0.get("identity_mutations")
            if (
                not isinstance(mutations, Mapping)
                or not mutations
                or not all(isinstance(value, Mapping) and value.get("passed") is True for value in mutations.values())
            ):
                errors.append("teacher T0 identity-mutation audit did not pass")
            provenance = teacher_t0.get("provenance")
            git = provenance.get("git") if isinstance(provenance, Mapping) else None
            if not isinstance(git, Mapping) or git.get("commit") != expected_commit or git.get("clean") is not True:
                errors.append("teacher T0 provenance is not the expected clean commit")
            frozen = teacher_t0.get("frozen_gradients")
            reader = frozen.get("reader") if isinstance(frozen, Mapping) else None
            if not isinstance(reader, Mapping) or reader.get("frozen_tensors_with_grad") != 0:
                errors.append("teacher T0 does not prove a frozen Reader")

    return {
        "schema_version": 1,
        "protocol": "R3-micro-prerequisites-v1",
        "training_regime": training_regime,
        "expected_commit": expected_commit,
        "technical_complete": not any(error.startswith("technical") for error in errors),
        "scorer_s0_complete": not any(error.startswith("R3-S0") for error in errors),
        "teacher_t0_required": training_regime == "teacher_assisted",
        "teacher_t0_complete": (
            None if training_regime == "qa_only" else not any(error.startswith("teacher T0") for error in errors)
        ),
        "errors": errors,
        "passed": not errors,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed R3 micro-overfit prerequisite validator")
    parser.add_argument("--scorer-s0-report", type=Path, required=True)
    parser.add_argument("--scorer-s0-report-sha256", required=True)
    parser.add_argument("--technical-report", type=Path, required=True)
    parser.add_argument("--technical-report-sha256", required=True)
    parser.add_argument("--training-regime", choices=("qa_only", "teacher_assisted"), required=True)
    parser.add_argument("--teacher-t0-report", type=Path)
    parser.add_argument("--teacher-t0-report-sha256")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    digest_pairs: list[tuple[str, str]] = [
        ("scorer-s0-report", args.scorer_s0_report_sha256),
        ("technical-report", args.technical_report_sha256),
    ]
    if args.training_regime == "teacher_assisted":
        if args.teacher_t0_report is None or args.teacher_t0_report_sha256 is None:
            errors.append("teacher-assisted validation requires teacher T0 path and SHA256")
        else:
            digest_pairs.append(("teacher-t0-report", args.teacher_t0_report_sha256))
    for label, digest in digest_pairs:
        if _SHA256.fullmatch(digest) is None:
            errors.append(f"{label} expected SHA256 is malformed")
    try:
        if not errors and sha256_file(args.scorer_s0_report) != args.scorer_s0_report_sha256:
            errors.append("scorer-s0-report SHA256 mismatch")
        if not errors and sha256_file(args.technical_report) != args.technical_report_sha256:
            errors.append("technical-report SHA256 mismatch")
        if (
            not errors
            and args.training_regime == "teacher_assisted"
            and args.teacher_t0_report is not None
            and sha256_file(args.teacher_t0_report) != args.teacher_t0_report_sha256
        ):
            errors.append("teacher-t0-report SHA256 mismatch")
        if errors:
            raise ValueError("; ".join(errors))
        report = validate_prerequisites(
            scorer_s0=_load_object(args.scorer_s0_report),
            technical=_load_object(args.technical_report),
            teacher_t0=(
                _load_object(args.teacher_t0_report)
                if args.training_regime == "teacher_assisted" and args.teacher_t0_report is not None
                else None
            ),
            training_regime=args.training_regime,
            expected_commit=args.expected_commit,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "protocol": "R3-micro-prerequisites-v1",
            "errors": [str(exc)],
            "passed": False,
        }
    report["inputs"] = {
        "scorer_s0_report": str(args.scorer_s0_report),
        "scorer_s0_report_sha256": args.scorer_s0_report_sha256,
        "technical_report": str(args.technical_report),
        "technical_report_sha256": args.technical_report_sha256,
        "training_regime": args.training_regime,
        "teacher_t0_report": None if args.teacher_t0_report is None else str(args.teacher_t0_report),
        "teacher_t0_report_sha256": args.teacher_t0_report_sha256,
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
