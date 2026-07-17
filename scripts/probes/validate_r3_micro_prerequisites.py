from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.reader import R3_QWEN_READER_RESIZE_CONTRACT  # noqa: E402
from vision_memory.teacher import (  # noqa: E402
    CALIBRATION_SAMPLE_SELECTION,
    CALIBRATION_SUITES,
    FrozenTeacherLossCalibration,
    load_teacher_calibration_input_lock,
)
from validate_r3_technical_gates import validate_resize_contract_report  # noqa: E402


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


def _locked_model_revision(provenance: Mapping[str, Any], name: str) -> str | None:
    models = provenance.get("models")
    model = models.get(name) if isinstance(models, Mapping) else None
    if not isinstance(model, Mapping) or model.get("revision_matches_lock") is not True:
        return None
    expected = model.get("expected_revision")
    observed = model.get("observed_revision")
    if (
        not isinstance(expected, str)
        or _COMMIT.fullmatch(expected) is None
        or observed != expected
    ):
        return None
    return expected


def validate_teacher_calibration_report(
    report: Mapping[str, Any],
    *,
    expected_commit: str,
    expected_calibration_file_sha256: str,
    expected_suite: str,
    expected_preregistration_sha256: str,
    expected_train_sha256: str,
    expected_manifest_sha256: str,
    expected_sidecar_sha256: str,
    expected_reader_revision: str,
    expected_dreamlite_revision: str,
) -> list[str]:
    errors: list[str] = []
    expected_input_binding = {
        "suite": expected_suite,
        "preregistration_sha256": expected_preregistration_sha256,
        "train_sha256": expected_train_sha256,
        "manifest_sha256": expected_manifest_sha256,
        "sidecar_sha256": expected_sidecar_sha256,
    }
    if expected_suite not in CALIBRATION_SUITES or any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for field, value in expected_input_binding.items()
        if field != "suite"
    ):
        errors.append("teacher calibration expected input binding is malformed")
        preregistered_lock = None
    else:
        try:
            preregistered_lock = load_teacher_calibration_input_lock(
                ROOT / "configs" / "experiments" / "r3_preregistration.json",
                suite=expected_suite,
            )
            locked_binding = preregistered_lock.to_dict()
            for field, expected in expected_input_binding.items():
                if locked_binding[field] != expected:
                    errors.append(
                        f"teacher calibration command binding for {field} differs from preregistration"
                    )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            preregistered_lock = None
            errors.append(f"teacher calibration preregistration lock is invalid: {exc}")
    if report.get("schema") != "vision_memory.r3-teacher-calibration-report.v1":
        errors.append("teacher calibration report identity is invalid")
    if report.get("reader_resize_contract") != R3_QWEN_READER_RESIZE_CONTRACT:
        errors.append("teacher calibration has the wrong Reader resize contract")
    if report.get("calibration_file_sha256") != expected_calibration_file_sha256:
        errors.append("teacher calibration report does not bind the supplied calibration file")
    for field, expected in expected_input_binding.items():
        if report.get(field) != expected:
            errors.append(f"teacher calibration report does not bind the expected {field}")
    if report.get("seed") != 0 or report.get("adapter_seed") != 0 or report.get("lora_rank") != 4:
        errors.append("teacher calibration seed/adapter/rank differs from the R3 lock")
    transition_count = report.get("transition_count")
    if isinstance(transition_count, bool) or not isinstance(transition_count, int) or transition_count <= 0:
        errors.append("teacher calibration transition_count must be positive")
    elif preregistered_lock is not None and transition_count != preregistered_lock.transition_count:
        errors.append("teacher calibration transition_count differs from preregistration")
    initial = report.get("initial_state")
    if (
        not isinstance(initial, Mapping)
        or initial.get("origin") != "blank_fixture"
        or initial.get("mode") != "RGB"
        or initial.get("size") != [1024, 1024]
    ):
        errors.append("teacher calibration initial state is not blank-1024 RGB")
    selection = report.get("sample_selection")
    if selection != CALIBRATION_SAMPLE_SELECTION:
        errors.append("teacher calibration sample selection differs from the preregistration")
    try:
        scales = report.get("scales")
        if not isinstance(scales, Mapping):
            raise ValueError("missing scales")
        calibration = FrozenTeacherLossCalibration.from_dict(scales)
        if report.get("calibration_contract_sha256") != calibration.contract_sha256:
            errors.append("teacher calibration contract SHA does not match its scales")
    except (TypeError, ValueError):
        errors.append("teacher calibration scales are invalid")
    ranges = report.get("raw_component_ranges")
    if not isinstance(ranges, Mapping) or set(ranges) != {"latent", "image", "feature"}:
        errors.append("teacher calibration raw component ranges are missing")
    else:
        for name, bounds in ranges.items():
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bounds)
                or not all(math.isfinite(float(value)) and float(value) >= 0 for value in bounds)
                or float(bounds[0]) > float(bounds[1])
            ):
                errors.append(f"teacher calibration {name} range is invalid")
    determinism = report.get("strict_determinism")
    if (
        not isinstance(determinism, Mapping)
        or determinism.get("deterministic_algorithms") is not True
        or determinism.get("deterministic_warn_only") is not False
        or determinism.get("sdpa")
        != {"flash": False, "memory_efficient": False, "cudnn": False, "math": True}
    ):
        errors.append("teacher calibration did not use strict math-only determinism")
    provenance = report.get("provenance")
    git = provenance.get("git") if isinstance(provenance, Mapping) else None
    if not isinstance(git, Mapping) or git.get("commit") != expected_commit or git.get("clean") is not True:
        errors.append("teacher calibration provenance is not the expected clean commit")
    if not isinstance(provenance, Mapping) or _locked_model_revision(provenance, "reader") != expected_reader_revision:
        errors.append("teacher calibration Reader revision differs from the technical chain")
    if (
        not isinstance(provenance, Mapping)
        or _locked_model_revision(provenance, "dreamlite") != expected_dreamlite_revision
    ):
        errors.append("teacher calibration DreamLite revision differs from the technical chain")
    return errors


def validate_teacher_compatibility_reports(
    *,
    tc0: Mapping[str, Any],
    tc0_file_sha256: str,
    tf0: Mapping[str, Any],
    tf0_file_sha256: str,
    expected_commit: str,
    expected_reader_revision: str,
) -> list[str]:
    """Validate the prospective read-only teacher-cache compatibility chain."""

    errors: list[str] = []
    for label, digest in (("TC0", tc0_file_sha256), ("TF0", tf0_file_sha256)):
        if _SHA256.fullmatch(digest) is None:
            errors.append(f"teacher {label} file SHA256 is malformed")
    expected_tc0 = {
        "protocol": "R3-TC0-cache-forward-compatibility-validation.v1",
        "expected_commit": expected_commit,
        "reader_revision": expected_reader_revision,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "validated_suites": ["set8", "transition16"],
        "validated_state_count": 30,
        "validated_artifact_tensor_count": 90,
        "validated_image_forward_count": 30,
        "cache_forward_compatibility_complete": True,
        "feature_backend_compatibility_complete": False,
        "teacher_t0_unlocked": False,
        "teacher_calibration_unlocked": False,
        "teacher_assisted_training_unlocked": False,
        "errors": [],
        "passed": True,
    }
    for field, expected in expected_tc0.items():
        if tc0.get(field) != expected:
            errors.append(f"teacher TC0 field {field!r} differs from the locked chain")

    expected_tf0 = {
        "protocol": "R3-TF0-feature-backend-compatibility-validation.v1",
        "expected_commit": expected_commit,
        "reader_revision": expected_reader_revision,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "tc0_validation_sha256": tc0_file_sha256,
        "validated_suites": ["set8", "transition16"],
        "validated_state_count": 30,
        "validated_feature_comparison_count": 30,
        "validated_feature_pass_count": 30,
        "teacher_t0_unlocked": True,
        "teacher_calibration_unlocked": True,
        "teacher_assisted_training_unlocked": True,
        "qa_only_dependency": False,
        "errors": [],
        "passed": True,
    }
    for field, expected in expected_tf0.items():
        if tf0.get(field) != expected:
            errors.append(f"teacher TF0 field {field!r} differs from the locked chain")
    tc0_preregistration = tc0.get("preregistration_sha256")
    if (
        not isinstance(tc0_preregistration, str)
        or _SHA256.fullmatch(tc0_preregistration) is None
        or tf0.get("preregistration_sha256") != tc0_preregistration
    ):
        errors.append("teacher TC0/TF0 preregistration SHA256 lineage is invalid")
    if not isinstance(tf0.get("feature_gate_sha256"), str) or _SHA256.fullmatch(
        str(tf0.get("feature_gate_sha256"))
    ) is None:
        errors.append("teacher TF0 feature gate SHA256 is invalid")
    return errors


def validate_prerequisites(
    *,
    resize_contract: Mapping[str, Any],
    scorer_s0: Mapping[str, Any],
    technical: Mapping[str, Any],
    teacher_t0: Mapping[str, Any] | None,
    teacher_calibration: Mapping[str, Any] | None,
    teacher_calibration_file_sha256: str | None,
    training_regime: str,
    expected_commit: str,
    teacher_tc0: Mapping[str, Any] | None = None,
    teacher_tc0_file_sha256: str | None = None,
    teacher_tf0: Mapping[str, Any] | None = None,
    teacher_tf0_file_sha256: str | None = None,
    teacher_calibration_suite: str | None = None,
    teacher_calibration_preregistration_sha256: str | None = None,
    teacher_calibration_train_sha256: str | None = None,
    teacher_calibration_manifest_sha256: str | None = None,
    teacher_calibration_sidecar_sha256: str | None = None,
) -> dict[str, Any]:
    if _COMMIT.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a lowercase full Git commit.")
    errors: list[str] = []
    resize_check: dict[str, Any] = {}
    try:
        resize_check = validate_resize_contract_report(resize_contract)
        if resize_check.get("git_commit") != expected_commit:
            errors.append("R3-R0 resize contract is not bound to the expected clean commit")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"R3-R0 resize contract is invalid: {exc}")
    if scorer_s0.get("schema_version") != 1 or scorer_s0.get("probe") != "r3_s0_qwen_scorer_contract":
        errors.append("R3-S0 scorer report identity is invalid")
    if scorer_s0.get("passed") is not True:
        errors.append("R3-S0 scorer contract did not pass")
    s0_contract = scorer_s0.get("contract")
    if not isinstance(s0_contract, Mapping) or s0_contract.get("reader_loss_mode") != "listwise-choice":
        errors.append("R3-S0 scorer report is not listwise-choice")
    if (
        not isinstance(s0_contract, Mapping)
        or s0_contract.get("reader_resize_contract") != R3_QWEN_READER_RESIZE_CONTRACT
    ):
        errors.append("R3-S0 scorer report has the wrong Reader resize contract")
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
    s0_runtime = s0_provenance.get("runtime") if isinstance(s0_provenance, Mapping) else None
    if (
        not isinstance(s0_runtime, Mapping)
        or s0_runtime.get("torch") != "2.7.0a0+ecf3bae40a.nv25.02"
        or s0_runtime.get("cuda_runtime") != "12.8"
    ):
        errors.append("R3-S0 runtime is not the locked Inspire H200 software stack")
    s0_determinism = scorer_s0.get("strict_determinism")
    expected_s0_determinism = {
        "seed": 0,
        "environment": {
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MKL_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "sdpa": {"flash": False, "memory_efficient": False, "cudnn": False, "math": True},
    }
    if s0_determinism != expected_s0_determinism:
        errors.append("R3-S0 did not use the locked strict math-only deterministic backend")
    expected_gates = ["R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L", "DL-S"]
    if technical.get("protocol") != "R3-technical-listwise-resize-v2":
        errors.append("technical protocol is not R3-technical-listwise-resize-v2")
    if technical.get("through") != "DL-S" or technical.get("required_gates") != expected_gates:
        errors.append("technical report does not cover the complete R3-R0/G4-L/G5-L/G6-L/DL-S chain")
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
    else:
        expected_reader_revision = technical_checks["DL-S"].get("reader_revision")
        if resize_check.get("reader_revision") != expected_reader_revision:
            errors.append("R3-R0 standalone report and technical chain use different Reader revisions")
        if (
            not isinstance(s0_provenance, Mapping)
            or _locked_model_revision(s0_provenance, "reader") != expected_reader_revision
        ):
            errors.append("R3-S0 Reader revision differs from the technical chain")

    if training_regime not in {"qa_only", "teacher_assisted"}:
        raise ValueError("training_regime must be qa_only or teacher_assisted.")
    if training_regime == "qa_only" and any(
        value is not None
        for value in (
            teacher_t0,
            teacher_calibration,
            teacher_calibration_file_sha256,
            teacher_tc0,
            teacher_tc0_file_sha256,
            teacher_tf0,
            teacher_tf0_file_sha256,
            teacher_calibration_suite,
            teacher_calibration_preregistration_sha256,
            teacher_calibration_train_sha256,
            teacher_calibration_manifest_sha256,
            teacher_calibration_sidecar_sha256,
        )
    ):
        errors.append("qa_only prerequisite validation rejects every teacher artifact")
    if training_regime == "teacher_assisted":
        expected_reader_revision = (
            technical_checks.get("DL-S", {}).get("reader_revision")
            if isinstance(technical_checks, Mapping)
            and isinstance(technical_checks.get("DL-S"), Mapping)
            else None
        )
        if (
            not isinstance(teacher_tc0, Mapping)
            or not isinstance(teacher_tf0, Mapping)
            or not isinstance(teacher_tc0_file_sha256, str)
            or not isinstance(teacher_tf0_file_sha256, str)
            or not isinstance(expected_reader_revision, str)
        ):
            errors.append("teacher TC0/TF0 reports and file SHA256 bindings are required")
        else:
            errors.extend(
                validate_teacher_compatibility_reports(
                    tc0=teacher_tc0,
                    tc0_file_sha256=teacher_tc0_file_sha256,
                    tf0=teacher_tf0,
                    tf0_file_sha256=teacher_tf0_file_sha256,
                    expected_commit=expected_commit,
                    expected_reader_revision=expected_reader_revision,
                )
            )
        if not isinstance(teacher_t0, Mapping):
            errors.append("teacher T0 is required for teacher_assisted stages")
        else:
            if teacher_t0.get("schema_version") != 1:
                errors.append("teacher T0 schema_version is not 1")
            if teacher_t0.get("probe") != "teacher_t0_real_qwen_integrity_upper_bound":
                errors.append("teacher T0 probe identity is invalid")
            if teacher_t0.get("passed") is not True:
                errors.append("teacher T0 did not pass")
            if teacher_t0.get("reader_resize_contract") != R3_QWEN_READER_RESIZE_CONTRACT:
                errors.append("teacher T0 has the wrong Reader resize contract")
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
            if teacher_t0.get("strict_determinism") != expected_s0_determinism:
                errors.append("teacher T0 did not use the locked strict math-only deterministic backend")
            runtime = provenance.get("runtime") if isinstance(provenance, Mapping) else None
            if (
                not isinstance(runtime, Mapping)
                or runtime.get("torch") != "2.7.0a0+ecf3bae40a.nv25.02"
                or runtime.get("cuda_runtime") != "12.8"
            ):
                errors.append("teacher T0 runtime differs from the locked Inspire stack")
            frozen = teacher_t0.get("frozen_gradients")
            reader = frozen.get("reader") if isinstance(frozen, Mapping) else None
            if (
                not isinstance(reader, Mapping)
                or reader.get("trainable_parameter_tensors") != 0
                or reader.get("frozen_tensors_with_grad") != 0
                or reader.get("frozen_nonfinite_grad_elements") != 0
            ):
                errors.append("teacher T0 does not prove a frozen Reader")
            memory = teacher_t0.get("cuda_peak_memory")
            memory_record = memory.get("cuda:0") if isinstance(memory, Mapping) else None
            if not isinstance(memory_record, Mapping) or memory_record.get("name") != "NVIDIA H200":
                errors.append("teacher T0 did not run on the locked H200")
            expected_reader_revision = (
                technical_checks.get("DL-S", {}).get("reader_revision")
                if isinstance(technical_checks, Mapping)
                and isinstance(technical_checks.get("DL-S"), Mapping)
                else None
            )
            if (
                not isinstance(provenance, Mapping)
                or _locked_model_revision(provenance, "reader") != expected_reader_revision
            ):
                errors.append("teacher T0 Reader revision differs from the technical chain")

        expected_reader_revision = (
            technical_checks.get("DL-S", {}).get("reader_revision")
            if isinstance(technical_checks, Mapping)
            and isinstance(technical_checks.get("DL-S"), Mapping)
            else None
        )
        expected_dreamlite_revision = (
            technical_checks.get("DL-S", {}).get("dreamlite_revision")
            if isinstance(technical_checks, Mapping)
            and isinstance(technical_checks.get("DL-S"), Mapping)
            else None
        )
        if (
            not isinstance(teacher_calibration, Mapping)
            or not isinstance(teacher_calibration_file_sha256, str)
            or _SHA256.fullmatch(teacher_calibration_file_sha256) is None
            or not isinstance(expected_reader_revision, str)
            or not isinstance(expected_dreamlite_revision, str)
            or teacher_calibration_suite not in CALIBRATION_SUITES
            or not isinstance(teacher_calibration_preregistration_sha256, str)
            or _SHA256.fullmatch(teacher_calibration_preregistration_sha256) is None
            or not isinstance(teacher_calibration_train_sha256, str)
            or _SHA256.fullmatch(teacher_calibration_train_sha256) is None
            or not isinstance(teacher_calibration_manifest_sha256, str)
            or _SHA256.fullmatch(teacher_calibration_manifest_sha256) is None
            or not isinstance(teacher_calibration_sidecar_sha256, str)
            or _SHA256.fullmatch(teacher_calibration_sidecar_sha256) is None
        ):
            errors.append(
                "teacher calibration report/file and suite/preregistration/train/manifest/sidecar "
                "bindings are required for teacher_assisted stages"
            )
        else:
            errors.extend(
                validate_teacher_calibration_report(
                    teacher_calibration,
                    expected_commit=expected_commit,
                    expected_calibration_file_sha256=teacher_calibration_file_sha256,
                    expected_suite=teacher_calibration_suite,
                    expected_preregistration_sha256=teacher_calibration_preregistration_sha256,
                    expected_train_sha256=teacher_calibration_train_sha256,
                    expected_manifest_sha256=teacher_calibration_manifest_sha256,
                    expected_sidecar_sha256=teacher_calibration_sidecar_sha256,
                    expected_reader_revision=expected_reader_revision,
                    expected_dreamlite_revision=expected_dreamlite_revision,
                )
            )

    return {
        "schema_version": 2,
        "protocol": "R3-micro-prerequisites-resize-v2",
        "training_regime": training_regime,
        "expected_commit": expected_commit,
        "resize_r0_complete": not any(error.startswith("R3-R0") for error in errors),
        "technical_complete": not any(error.startswith("technical") for error in errors),
        "scorer_s0_complete": not any(error.startswith("R3-S0") for error in errors),
        "teacher_t0_required": training_regime == "teacher_assisted",
        "teacher_tc0_complete": (
            None
            if training_regime == "qa_only"
            else not any(error.startswith(("teacher TC0", "teacher TF0")) for error in errors)
        ),
        "teacher_tf0_complete": (
            None
            if training_regime == "qa_only"
            else not any(error.startswith(("teacher TC0", "teacher TF0")) for error in errors)
        ),
        "teacher_t0_complete": (
            None if training_regime == "qa_only" else not any(error.startswith("teacher T0") for error in errors)
        ),
        "teacher_calibration_complete": (
            None
            if training_regime == "qa_only"
            else not any(error.startswith("teacher calibration") for error in errors)
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
    parser.add_argument("--resize-contract-report", type=Path, required=True)
    parser.add_argument("--resize-contract-report-sha256", required=True)
    parser.add_argument("--scorer-s0-report", type=Path, required=True)
    parser.add_argument("--scorer-s0-report-sha256", required=True)
    parser.add_argument("--technical-report", type=Path, required=True)
    parser.add_argument("--technical-report-sha256", required=True)
    parser.add_argument("--training-regime", choices=("qa_only", "teacher_assisted"), required=True)
    parser.add_argument("--teacher-t0-report", type=Path)
    parser.add_argument("--teacher-t0-report-sha256")
    parser.add_argument("--teacher-tc0-report", type=Path)
    parser.add_argument("--teacher-tc0-report-sha256")
    parser.add_argument("--teacher-tf0-report", type=Path)
    parser.add_argument("--teacher-tf0-report-sha256")
    parser.add_argument("--teacher-calibration", type=Path)
    parser.add_argument("--teacher-calibration-sha256")
    parser.add_argument("--teacher-calibration-report", type=Path)
    parser.add_argument("--teacher-calibration-report-sha256")
    parser.add_argument("--teacher-calibration-suite", choices=CALIBRATION_SUITES)
    parser.add_argument("--teacher-calibration-preregistration-sha256")
    parser.add_argument("--teacher-calibration-train-sha256")
    parser.add_argument("--teacher-calibration-manifest-sha256")
    parser.add_argument("--teacher-calibration-sidecar-sha256")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    digest_pairs: list[tuple[str, str]] = [
        ("resize-contract-report", args.resize_contract_report_sha256),
        ("scorer-s0-report", args.scorer_s0_report_sha256),
        ("technical-report", args.technical_report_sha256),
    ]
    if args.training_regime == "teacher_assisted":
        if any(
            value is None
            for value in (
                args.teacher_t0_report,
                args.teacher_t0_report_sha256,
                args.teacher_tc0_report,
                args.teacher_tc0_report_sha256,
                args.teacher_tf0_report,
                args.teacher_tf0_report_sha256,
                args.teacher_calibration,
                args.teacher_calibration_sha256,
                args.teacher_calibration_report,
                args.teacher_calibration_report_sha256,
                args.teacher_calibration_suite,
                args.teacher_calibration_preregistration_sha256,
                args.teacher_calibration_train_sha256,
                args.teacher_calibration_manifest_sha256,
                args.teacher_calibration_sidecar_sha256,
            )
        ):
            errors.append(
                "teacher-assisted validation requires TC0, TF0, T0, and calibration paths/SHA256 values"
            )
        else:
            digest_pairs.append(("teacher-t0-report", args.teacher_t0_report_sha256))
            digest_pairs.append(("teacher-tc0-report", args.teacher_tc0_report_sha256))
            digest_pairs.append(("teacher-tf0-report", args.teacher_tf0_report_sha256))
            digest_pairs.append(("teacher-calibration", args.teacher_calibration_sha256))
            digest_pairs.append(("teacher-calibration-report", args.teacher_calibration_report_sha256))
            digest_pairs.append(
                (
                    "teacher-calibration-preregistration",
                    args.teacher_calibration_preregistration_sha256,
                )
            )
            digest_pairs.append(("teacher-calibration-train", args.teacher_calibration_train_sha256))
            digest_pairs.append(
                ("teacher-calibration-manifest", args.teacher_calibration_manifest_sha256)
            )
            digest_pairs.append(
                ("teacher-calibration-sidecar", args.teacher_calibration_sidecar_sha256)
            )
    elif any(
        value is not None
        for value in (
            args.teacher_t0_report,
            args.teacher_t0_report_sha256,
            args.teacher_tc0_report,
            args.teacher_tc0_report_sha256,
            args.teacher_tf0_report,
            args.teacher_tf0_report_sha256,
            args.teacher_calibration,
            args.teacher_calibration_sha256,
            args.teacher_calibration_report,
            args.teacher_calibration_report_sha256,
            args.teacher_calibration_suite,
            args.teacher_calibration_preregistration_sha256,
            args.teacher_calibration_train_sha256,
            args.teacher_calibration_manifest_sha256,
            args.teacher_calibration_sidecar_sha256,
        )
    ):
        errors.append("qa_only validation rejects every teacher artifact path/SHA256 value")
    for label, digest in digest_pairs:
        if _SHA256.fullmatch(digest) is None:
            errors.append(f"{label} expected SHA256 is malformed")
    try:
        if not errors and sha256_file(args.resize_contract_report) != args.resize_contract_report_sha256:
            errors.append("resize-contract-report SHA256 mismatch")
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
        if (
            not errors
            and args.training_regime == "teacher_assisted"
            and args.teacher_tc0_report is not None
            and sha256_file(args.teacher_tc0_report) != args.teacher_tc0_report_sha256
        ):
            errors.append("teacher-tc0-report SHA256 mismatch")
        if (
            not errors
            and args.training_regime == "teacher_assisted"
            and args.teacher_tf0_report is not None
            and sha256_file(args.teacher_tf0_report) != args.teacher_tf0_report_sha256
        ):
            errors.append("teacher-tf0-report SHA256 mismatch")
        if (
            not errors
            and args.training_regime == "teacher_assisted"
            and args.teacher_calibration is not None
            and sha256_file(args.teacher_calibration) != args.teacher_calibration_sha256
        ):
            errors.append("teacher-calibration SHA256 mismatch")
        if (
            not errors
            and args.training_regime == "teacher_assisted"
            and args.teacher_calibration_report is not None
            and sha256_file(args.teacher_calibration_report) != args.teacher_calibration_report_sha256
        ):
            errors.append("teacher-calibration-report SHA256 mismatch")
        if errors:
            raise ValueError("; ".join(errors))
        report = validate_prerequisites(
            resize_contract=_load_object(args.resize_contract_report),
            scorer_s0=_load_object(args.scorer_s0_report),
            technical=_load_object(args.technical_report),
            teacher_t0=(
                _load_object(args.teacher_t0_report)
                if args.training_regime == "teacher_assisted" and args.teacher_t0_report is not None
                else None
            ),
            teacher_calibration=(
                _load_object(args.teacher_calibration_report)
                if args.training_regime == "teacher_assisted"
                and args.teacher_calibration_report is not None
                else None
            ),
            teacher_calibration_file_sha256=(
                args.teacher_calibration_sha256 if args.training_regime == "teacher_assisted" else None
            ),
            training_regime=args.training_regime,
            expected_commit=args.expected_commit,
            teacher_tc0=(
                _load_object(args.teacher_tc0_report)
                if args.training_regime == "teacher_assisted" and args.teacher_tc0_report is not None
                else None
            ),
            teacher_tc0_file_sha256=(
                args.teacher_tc0_report_sha256 if args.training_regime == "teacher_assisted" else None
            ),
            teacher_tf0=(
                _load_object(args.teacher_tf0_report)
                if args.training_regime == "teacher_assisted" and args.teacher_tf0_report is not None
                else None
            ),
            teacher_tf0_file_sha256=(
                args.teacher_tf0_report_sha256 if args.training_regime == "teacher_assisted" else None
            ),
            teacher_calibration_suite=(
                args.teacher_calibration_suite if args.training_regime == "teacher_assisted" else None
            ),
            teacher_calibration_preregistration_sha256=(
                args.teacher_calibration_preregistration_sha256
                if args.training_regime == "teacher_assisted"
                else None
            ),
            teacher_calibration_train_sha256=(
                args.teacher_calibration_train_sha256
                if args.training_regime == "teacher_assisted"
                else None
            ),
            teacher_calibration_manifest_sha256=(
                args.teacher_calibration_manifest_sha256
                if args.training_regime == "teacher_assisted"
                else None
            ),
            teacher_calibration_sidecar_sha256=(
                args.teacher_calibration_sidecar_sha256
                if args.training_regime == "teacher_assisted"
                else None
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 2,
            "protocol": "R3-micro-prerequisites-resize-v2",
            "errors": [str(exc)],
            "passed": False,
        }
    report["inputs"] = {
        "resize_contract_report": str(args.resize_contract_report),
        "resize_contract_report_sha256": args.resize_contract_report_sha256,
        "scorer_s0_report": str(args.scorer_s0_report),
        "scorer_s0_report_sha256": args.scorer_s0_report_sha256,
        "technical_report": str(args.technical_report),
        "technical_report_sha256": args.technical_report_sha256,
        "training_regime": args.training_regime,
        "teacher_t0_report": None if args.teacher_t0_report is None else str(args.teacher_t0_report),
        "teacher_t0_report_sha256": args.teacher_t0_report_sha256,
        "teacher_tc0_report": None if args.teacher_tc0_report is None else str(args.teacher_tc0_report),
        "teacher_tc0_report_sha256": args.teacher_tc0_report_sha256,
        "teacher_tf0_report": None if args.teacher_tf0_report is None else str(args.teacher_tf0_report),
        "teacher_tf0_report_sha256": args.teacher_tf0_report_sha256,
        "teacher_calibration": (
            None if args.teacher_calibration is None else str(args.teacher_calibration)
        ),
        "teacher_calibration_sha256": args.teacher_calibration_sha256,
        "teacher_calibration_report": (
            None if args.teacher_calibration_report is None else str(args.teacher_calibration_report)
        ),
        "teacher_calibration_report_sha256": args.teacher_calibration_report_sha256,
        "teacher_calibration_suite": args.teacher_calibration_suite,
        "teacher_calibration_preregistration_sha256": (
            args.teacher_calibration_preregistration_sha256
        ),
        "teacher_calibration_train_sha256": args.teacher_calibration_train_sha256,
        "teacher_calibration_manifest_sha256": args.teacher_calibration_manifest_sha256,
        "teacher_calibration_sidecar_sha256": args.teacher_calibration_sidecar_sha256,
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
