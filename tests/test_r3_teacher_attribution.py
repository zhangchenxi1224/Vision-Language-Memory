from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from vision_memory.eval.r3_teacher_attribution import score_r3_teacher_attribution
from vision_memory.eval.teacher_retrieval import TEACHER_RETRIEVAL_SCHEMA, validate_teacher_checkpoint_lineage


CONTROLS = ("correct", "shuffled", "random-moment-matched")


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summary(*, control: str, stage: str) -> dict[str, Any]:
    diagnostics = None
    if stage == "distill":
        initial = {"distill_loss": 1.0, "latent_raw": 2.0, "image_raw": 3.0, "feature_raw": 4.0}
        final = {"distill_loss": 0.4, "latent_raw": 1.0, "image_raw": 2.0, "feature_raw": 3.0}
        diagnostics = {
            "initial": initial,
            "final": final,
            "final_over_initial": {key: final[key] / initial[key] for key in initial},
            "checks": {
                "composite_drop_at_least_50_percent": True,
                "all_raw_components_decreased": True,
            },
        }
    return {
        "training_regime": "teacher_assisted",
        "objective_stage": stage,
        "reader_loss_mode": "listwise-choice",
        "choice_view_schedule": "cyclic4",
        "teacher_manifest_sha256": digest("manifest"),
        "teacher_control": control,
        "teacher_control_sha256": digest(f"control-{control}"),
        "distill_diagnostics": diagnostics,
    }


def lineage(*, control: str, stage: str, distill_checkpoint: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "training_regime": "teacher_assisted",
        "parent_checkpoint_regime": None if stage == "distill" else "teacher_assisted",
        "parent_checkpoint_sha256": None if stage == "distill" else distill_checkpoint,
        "objective_stage": stage,
        "reader_loss_mode": "listwise-choice",
        "choice_view_schedule": "cyclic4",
        "choice_permutation_family_sha256": digest("cyclic4"),
        "eval_choice_permutation_family_sha256": digest("reverse-cyclic4"),
        "teacher_manifest_sha256": digest("manifest"),
        "teacher_sidecar_sha256": digest("sidecar"),
        "teacher_calibration_sha256": digest("calibration"),
        "teacher_control": control,
        "teacher_control_sha256": digest(f"control-{control}"),
        "presentations_per_state": 256 if stage == "distill" else 512,
        "distill_presentations": 256,
        "qa_presentations": 0 if stage == "distill" else 256,
        "teacher_supervision_loaded": stage == "distill",
        "teacher_checkpoint_is_qa_only_eligible": False,
    }


def retrieval(
    *,
    control: str,
    stage: str,
    distill_checkpoint: str,
    correct: int,
) -> dict[str, Any]:
    checkpoint = distill_checkpoint if stage == "distill" else digest(f"{control}-qa-checkpoint")
    value = {
        "schema": TEACHER_RETRIEVAL_SCHEMA,
        "objective_stage": stage,
        "training_regime": "teacher_assisted",
        "teacher_control": control,
        "checkpoint_sha256": checkpoint,
        "episodes_sha256": digest("set8-train"),
        "teacher_cache_lock_sha256": digest("cache-lock"),
        "episode_state_contract_sha256": digest("state-contract"),
        "teacher_cache_files": {
            "manifest_sha256": digest("manifest"),
            "sidecar_sha256": digest("sidecar"),
            "calibration_sha256": digest("calibration"),
            "manifest_payload_sha256": digest("manifest-payload"),
        },
        "training_lineage": lineage(
            control=control,
            stage=stage,
            distill_checkpoint=distill_checkpoint,
        ),
        "summary": {
            "n_episodes": 8,
            "correct": correct,
            "top1_accuracy": correct / 8,
            "minimum_correct": 7,
            "gate_passed": correct >= 7,
            "ambiguous_top_ties": 0,
        },
    }
    return value


def gate(*, control: str, qa_retrieval: dict[str, Any], passed: bool) -> dict[str, Any]:
    checks = {
        "accuracy": passed,
        "positions": passed,
        "per_state": passed,
        "rotation_consistency": passed,
        "reset_drop": passed,
        "shuffle_drop": passed,
    }
    payload = {
        "schema_version": "vlm.r3.set8_gate.v1",
        "suite": "set8",
        "correct": 30 if passed else 24,
        "count": 32,
        "positions": {},
        "states": {},
        "consistent_state_count": 7 if passed else 3,
        "interventions": {},
        "checks": checks,
        "passed": passed,
    }
    lineage_value = qa_retrieval["training_lineage"]
    provenance = {
        "schema": "vlm.r3.micro_artifact_provenance.v1",
        "predictions_sha256": digest(f"{control}-predictions"),
        "prediction_report_sha256": digest(f"{control}-prediction-report"),
        "checkpoint_path": f"/audit/{control}/qa.pt",
        "checkpoint_sha256": qa_retrieval["checkpoint_sha256"],
        "training_summary_sha256": digest(f"{control}-training-summary"),
        "dreamlite_snapshot_manifest_sha256": digest("dreamlite-snapshot-manifest"),
        "reader_snapshot_manifest_sha256": digest("reader-snapshot-manifest"),
        "training_regime": "teacher_assisted",
        "parent_checkpoint_regime": "teacher_assisted",
        "objective_stage": "qa",
        "reader_loss_mode": "listwise-choice",
        "choice_permutation_family_sha256": lineage_value["choice_permutation_family_sha256"],
        "eval_choice_permutation_family_sha256": lineage_value["eval_choice_permutation_family_sha256"],
        "teacher_control": control,
        "teacher_control_sha256": lineage_value["teacher_control_sha256"],
        "teacher_manifest_sha256": lineage_value["teacher_manifest_sha256"],
        "teacher_sidecar_sha256": lineage_value["teacher_sidecar_sha256"],
        "teacher_calibration_sha256": lineage_value["teacher_calibration_sha256"],
        "presentations_per_state": 512,
        "distill_presentations": 256,
        "qa_presentations": 256,
        "recurrence_mode": "direct_latent",
        "detach_between_events": False,
        "noop_policy": "update",
        "initial_state_mode": "blank",
        "learn_initial_state": False,
        "lora_rank": 4,
        "seed": 0,
        "adapter_seed": 0,
        "strict_determinism": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "sdpa": {"flash": False, "memory_efficient": False, "cudnn": False, "math": True},
        },
        "state_gradient_audit": {
            "schema": "vision_memory.r3-state-gradient-audit.v1",
            "enabled": True,
            "objective_stage": "qa",
            "passed": True,
        },
        "training_trace": {
            "schema": "vlm.r3.micro_training_trace.v1",
            "suite": "set8",
            "episodes": 8,
            "epochs": 256,
            "presentations_per_state": 256,
            "optimizer_steps": 256,
            "choice_rotation_counts": [512, 512, 512, 512],
            "dev_optimizer_steps": list(range(64, 257, 32)),
            "checkpoint_count": 8,
            "metrics_sha256": digest(f"{control}-metrics"),
            "passed": True,
        },
    }
    return {
        **payload,
        "scientific_payload_sha256": canonical_sha(payload),
        "artifact_provenance": provenance,
    }


def arm(*, control: str, retrieval_correct: int, qa_gate_passed: bool) -> dict[str, Any]:
    distill_checkpoint = digest(f"{control}-distill-checkpoint")
    distill_retrieval = retrieval(
        control=control,
        stage="distill",
        distill_checkpoint=distill_checkpoint,
        correct=retrieval_correct,
    )
    qa_retrieval = retrieval(
        control=control,
        stage="qa",
        distill_checkpoint=distill_checkpoint,
        correct=retrieval_correct,
    )
    return {
        "distill_summary": summary(control=control, stage="distill"),
        "qa_summary": summary(control=control, stage="qa"),
        "distill_retrieval": distill_retrieval,
        "qa_retrieval": qa_retrieval,
        "qa_gate": gate(control=control, qa_retrieval=qa_retrieval, passed=qa_gate_passed),
    }


def passing_arms() -> dict[str, dict[str, Any]]:
    values = {
        "correct": arm(control="correct", retrieval_correct=8, qa_gate_passed=True),
        "shuffled": arm(control="shuffled", retrieval_correct=2, qa_gate_passed=False),
        "random-moment-matched": arm(
            control="random-moment-matched",
            retrieval_correct=1,
            qa_gate_passed=False,
        ),
    }
    values["correct"]["qa_retrieval"]["retention"] = {
        "reference_correct": 8,
        "current_correct": 8,
        "retention": 1.0,
        "minimum_retention": 0.9,
        "gate_passed": True,
    }
    return values


def test_complete_teacher_attribution_package_passes() -> None:
    report = score_r3_teacher_attribution(passing_arms())
    assert report["passed"]
    assert all(report["checks"].values())
    assert not report["arms"]["shuffled"]["substitution_candidate"]
    assert len(report["scientific_payload_sha256"]) == 64


@pytest.mark.parametrize("raw_key", ["latent_raw", "image_raw", "feature_raw"])
def test_each_non_decreasing_raw_component_fails_correct_teacher_gate(raw_key: str) -> None:
    values = passing_arms()
    diagnostics = values["correct"]["distill_summary"]["distill_diagnostics"]
    diagnostics["final"][raw_key] = diagnostics["initial"][raw_key]
    diagnostics["final_over_initial"][raw_key] = 1.0
    diagnostics["checks"]["all_raw_components_decreased"] = False
    report = score_r3_teacher_attribution(values)
    assert not report["passed"]
    assert not report["checks"]["correct_all_raw_components_decreased"]


def test_composite_must_drop_by_at_least_half() -> None:
    values = passing_arms()
    diagnostics = values["correct"]["distill_summary"]["distill_diagnostics"]
    diagnostics["final"]["distill_loss"] = 0.5000001
    diagnostics["final_over_initial"]["distill_loss"] = 0.5000001
    diagnostics["checks"]["composite_drop_at_least_50_percent"] = False
    report = score_r3_teacher_attribution(values)
    assert not report["checks"]["correct_composite_drop_at_least_50_percent"]
    assert not report["passed"]


def test_correct_teacher_qa_retrieval_must_retain_ninety_percent() -> None:
    values = passing_arms()
    values["correct"]["qa_retrieval"]["summary"].update({"correct": 7, "top1_accuracy": 7 / 8, "gate_passed": True})
    values["correct"]["qa_retrieval"]["retention"] = {
        "reference_correct": 8,
        "current_correct": 7,
        "retention": 0.875,
        "minimum_retention": 0.9,
        "gate_passed": False,
    }
    report = score_r3_teacher_attribution(values)
    assert not report["checks"]["correct_qa_retrieval_retention_at_least_90_percent"]
    assert not report["passed"]


@pytest.mark.parametrize("control", ["shuffled", "random-moment-matched"])
def test_control_cannot_substitute_for_correct_teacher(control: str) -> None:
    values = passing_arms()
    control_arm = values[control]
    control_arm["distill_retrieval"]["summary"].update({"correct": 7, "top1_accuracy": 7 / 8, "gate_passed": True})
    control_arm["qa_gate"] = gate(
        control=control,
        qa_retrieval=control_arm["qa_retrieval"],
        passed=True,
    )
    report = score_r3_teacher_attribution(values)
    check_name = "shuffled_teacher_cannot_substitute" if control == "shuffled" else "random_teacher_cannot_substitute"
    assert not report["checks"][check_name]
    assert not report["passed"]


def test_cross_wired_parent_checkpoint_is_an_integrity_error() -> None:
    values = passing_arms()
    values["correct"]["qa_retrieval"]["training_lineage"]["parent_checkpoint_sha256"] = digest("wrong")
    with pytest.raises(ValueError, match="not descended"):
        score_r3_teacher_attribution(values)


def test_qa_gate_must_be_bound_to_the_same_checkpoint_and_lineage() -> None:
    values = passing_arms()
    values["correct"]["qa_gate"]["artifact_provenance"]["checkpoint_sha256"] = digest("other")
    with pytest.raises(ValueError, match="provenance field 'checkpoint_sha256'"):
        score_r3_teacher_attribution(values)


def test_tampered_qa_scientific_payload_is_an_integrity_error() -> None:
    values = passing_arms()
    values["correct"]["qa_gate"]["correct"] = 31
    with pytest.raises(ValueError, match="scientific payload SHA256 mismatch"):
        score_r3_teacher_attribution(values)


def test_retrieval_lineage_can_be_explicitly_bound_to_control_arm() -> None:
    control = "shuffled"
    manifest = {
        "schema_version": 2,
        "training_lineage": lineage(
            control=control,
            stage="distill",
            distill_checkpoint=digest("unused"),
        ),
    }
    validated = validate_teacher_checkpoint_lineage(
        manifest,
        manifest_file_sha256=digest("manifest"),
        sidecar_file_sha256=digest("sidecar"),
        calibration_file_sha256=digest("calibration"),
        expected_teacher_control="shuffled",
    )
    assert validated["teacher_control"] == "shuffled"
    with pytest.raises(ValueError, match="shuffled/random"):
        validate_teacher_checkpoint_lineage(
            manifest,
            manifest_file_sha256=digest("manifest"),
            sidecar_file_sha256=digest("sidecar"),
            calibration_file_sha256=digest("calibration"),
            expected_teacher_control="correct",
        )
