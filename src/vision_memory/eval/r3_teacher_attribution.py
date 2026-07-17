"""Fail-closed offline scoring for the R3 teacher-assisted Set8 gate.

The scorer deliberately consumes only immutable JSON artifacts.  A malformed or
cross-wired artifact raises ``ValueError``; a well-formed scientific failure is
returned as ``passed=False``.  This distinction prevents a missing lineage field
from being silently counted as a negative experimental result.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from .r3_micro import validate_r3_micro_artifact_provenance
from .teacher_retrieval import TEACHER_RETRIEVAL_SCHEMA, compare_retrieval_retention


R3_TEACHER_ATTRIBUTION_SCHEMA = "vision_memory.r3-teacher-attribution-gate.v1"
TEACHER_CONTROLS = ("correct", "shuffled", "random-moment-matched")
DISTILL_KEYS = ("distill_loss", "latent_raw", "image_raw", "feature_raw")
RAW_COMPONENT_KEYS = ("latent_raw", "image_raw", "feature_raw")
TEACHER_HASH_FIELDS = (
    "teacher_manifest_sha256",
    "teacher_sidecar_sha256",
    "teacher_calibration_sha256",
)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a JSON object.")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise ValueError(f"{field} must be a lowercase SHA256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a lowercase SHA256 digest.") from exc
    return value


def _finite(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}.")
    return result


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}.")
    return value


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean.")
    return value


def _validate_summary(
    summary: Mapping[str, Any],
    *,
    control: str,
    stage: str,
) -> dict[str, Any]:
    prefix = f"{control}.{stage}_summary"
    if summary.get("training_regime") != "teacher_assisted":
        raise ValueError(f"{prefix} is not teacher_assisted.")
    if summary.get("objective_stage") != stage:
        raise ValueError(f"{prefix} objective_stage must be {stage!r}.")
    if summary.get("reader_loss_mode") != "listwise-choice":
        raise ValueError(f"{prefix} must use listwise-choice Reader loss.")
    if summary.get("choice_view_schedule") != "cyclic4":
        raise ValueError(f"{prefix} must use the cyclic4 training schedule.")
    if summary.get("teacher_control") != control:
        raise ValueError(f"{prefix} teacher_control mismatch.")
    manifest_sha = _sha256(summary.get("teacher_manifest_sha256"), field=f"{prefix}.teacher_manifest_sha256")
    control_sha = _sha256(summary.get("teacher_control_sha256"), field=f"{prefix}.teacher_control_sha256")
    diagnostics = summary.get("distill_diagnostics")
    if stage == "qa":
        if diagnostics is not None:
            raise ValueError(f"{prefix} must not contain distillation diagnostics after teacher unload.")
        return {
            "teacher_manifest_sha256": manifest_sha,
            "teacher_control_sha256": control_sha,
            "distill_progress_passed": None,
        }
    diagnostics = _mapping(diagnostics, field=f"{prefix}.distill_diagnostics")
    initial = _mapping(diagnostics.get("initial"), field=f"{prefix}.distill_diagnostics.initial")
    final = _mapping(diagnostics.get("final"), field=f"{prefix}.distill_diagnostics.final")
    reported_ratios = _mapping(
        diagnostics.get("final_over_initial"),
        field=f"{prefix}.distill_diagnostics.final_over_initial",
    )
    ratios: dict[str, float] = {}
    for key in DISTILL_KEYS:
        initial_value = _finite(initial.get(key), field=f"{prefix}.initial.{key}", minimum=0.0)
        if initial_value <= 0.0:
            raise ValueError(f"{prefix}.initial.{key} must be strictly positive.")
        final_value = _finite(final.get(key), field=f"{prefix}.final.{key}", minimum=0.0)
        ratio = final_value / initial_value
        reported = _finite(reported_ratios.get(key), field=f"{prefix}.final_over_initial.{key}", minimum=0.0)
        if not math.isclose(ratio, reported, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{prefix} reported ratio for {key!r} does not match initial/final values.")
        ratios[key] = ratio
    composite_passed = ratios["distill_loss"] <= 0.5
    raw_components_passed = all(
        _finite(final[key], field=f"{prefix}.final.{key}") < _finite(initial[key], field=f"{prefix}.initial.{key}")
        for key in RAW_COMPONENT_KEYS
    )
    checks = _mapping(diagnostics.get("checks"), field=f"{prefix}.distill_diagnostics.checks")
    if (
        _boolean(
            checks.get("composite_drop_at_least_50_percent"),
            field=f"{prefix}.checks.composite_drop_at_least_50_percent",
        )
        is not composite_passed
    ):
        raise ValueError(f"{prefix} stored composite check disagrees with recomputed values.")
    if (
        _boolean(
            checks.get("all_raw_components_decreased"),
            field=f"{prefix}.checks.all_raw_components_decreased",
        )
        is not raw_components_passed
    ):
        raise ValueError(f"{prefix} stored raw-component check disagrees with recomputed values.")
    return {
        "teacher_manifest_sha256": manifest_sha,
        "teacher_control_sha256": control_sha,
        "ratios": ratios,
        "composite_drop_at_least_50_percent": composite_passed,
        "all_raw_components_decreased": raw_components_passed,
        "distill_progress_passed": composite_passed and raw_components_passed,
    }


def _validate_retrieval(
    report: Mapping[str, Any],
    *,
    control: str,
    stage: str,
) -> dict[str, Any]:
    prefix = f"{control}.{stage}_retrieval"
    if report.get("schema") != TEACHER_RETRIEVAL_SCHEMA:
        raise ValueError(f"{prefix} has an unsupported schema.")
    if report.get("training_regime") != "teacher_assisted" or report.get("objective_stage") != stage:
        raise ValueError(f"{prefix} regime/stage mismatch.")
    if report.get("teacher_control") != control:
        raise ValueError(f"{prefix} teacher_control mismatch.")
    checkpoint_sha = _sha256(report.get("checkpoint_sha256"), field=f"{prefix}.checkpoint_sha256")
    episodes_sha = _sha256(report.get("episodes_sha256"), field=f"{prefix}.episodes_sha256")
    cache_lock_sha = _sha256(
        report.get("teacher_cache_lock_sha256"),
        field=f"{prefix}.teacher_cache_lock_sha256",
    )
    state_contract_sha = _sha256(
        report.get("episode_state_contract_sha256"),
        field=f"{prefix}.episode_state_contract_sha256",
    )
    lineage = _mapping(report.get("training_lineage"), field=f"{prefix}.training_lineage")
    if _integer(lineage.get("schema_version"), field=f"{prefix}.training_lineage.schema_version") < 2:
        raise ValueError(f"{prefix} requires schema-v2 lineage.")
    expected_lineage = {
        "training_regime": "teacher_assisted",
        "objective_stage": stage,
        "reader_loss_mode": "listwise-choice",
        "choice_view_schedule": "cyclic4",
        "teacher_control": control,
    }
    for field, expected in expected_lineage.items():
        if lineage.get(field) != expected:
            raise ValueError(f"{prefix}.training_lineage.{field} must equal {expected!r}.")
    if lineage.get("teacher_checkpoint_is_qa_only_eligible") is not False:
        raise ValueError(f"{prefix} is incorrectly marked QA-only eligible.")
    if lineage.get("teacher_supervision_loaded") is not (stage == "distill"):
        raise ValueError(f"{prefix} teacher_supervision_loaded conflicts with its stage.")
    family_sha = _sha256(
        lineage.get("choice_permutation_family_sha256"),
        field=f"{prefix}.training_lineage.choice_permutation_family_sha256",
    )
    eval_family_sha = _sha256(
        lineage.get("eval_choice_permutation_family_sha256"),
        field=f"{prefix}.training_lineage.eval_choice_permutation_family_sha256",
    )
    if family_sha == eval_family_sha:
        raise ValueError(f"{prefix} train and gate permutation families must be disjoint contracts.")
    lineage_hashes = {
        field: _sha256(lineage.get(field), field=f"{prefix}.training_lineage.{field}") for field in TEACHER_HASH_FIELDS
    }
    control_sha = _sha256(
        lineage.get("teacher_control_sha256"),
        field=f"{prefix}.training_lineage.teacher_control_sha256",
    )
    cache_files = _mapping(report.get("teacher_cache_files"), field=f"{prefix}.teacher_cache_files")
    cache_to_lineage = {
        "manifest_sha256": "teacher_manifest_sha256",
        "sidecar_sha256": "teacher_sidecar_sha256",
        "calibration_sha256": "teacher_calibration_sha256",
    }
    for cache_field, lineage_field in cache_to_lineage.items():
        actual = _sha256(cache_files.get(cache_field), field=f"{prefix}.teacher_cache_files.{cache_field}")
        if actual != lineage_hashes[lineage_field]:
            raise ValueError(f"{prefix} cache file {cache_field!r} differs from checkpoint lineage.")
    distill_presentations = _integer(
        lineage.get("distill_presentations"),
        field=f"{prefix}.training_lineage.distill_presentations",
    )
    qa_presentations = _integer(
        lineage.get("qa_presentations"),
        field=f"{prefix}.training_lineage.qa_presentations",
    )
    total_presentations = _integer(
        lineage.get("presentations_per_state"),
        field=f"{prefix}.training_lineage.presentations_per_state",
    )
    expected_presentations = (256, 0, 256) if stage == "distill" else (256, 256, 512)
    if (distill_presentations, qa_presentations, total_presentations) != expected_presentations:
        raise ValueError(
            f"{prefix} presentations must be distill/QA/total={expected_presentations}, "
            f"got {(distill_presentations, qa_presentations, total_presentations)}."
        )
    parent_checkpoint_sha = lineage.get("parent_checkpoint_sha256")
    parent_regime = lineage.get("parent_checkpoint_regime")
    if stage == "distill":
        if parent_checkpoint_sha is not None or parent_regime is not None:
            raise ValueError(f"{prefix} must have fresh distillation lineage.")
    else:
        parent_checkpoint_sha = _sha256(
            parent_checkpoint_sha,
            field=f"{prefix}.training_lineage.parent_checkpoint_sha256",
        )
        if parent_regime != "teacher_assisted":
            raise ValueError(f"{prefix} must retain teacher_assisted parent lineage.")
    summary = _mapping(report.get("summary"), field=f"{prefix}.summary")
    n_episodes = _integer(summary.get("n_episodes"), field=f"{prefix}.summary.n_episodes")
    correct = _integer(summary.get("correct"), field=f"{prefix}.summary.correct")
    minimum_correct = _integer(summary.get("minimum_correct"), field=f"{prefix}.summary.minimum_correct")
    if n_episodes != 8 or not 0 <= correct <= 8 or minimum_correct != 7:
        raise ValueError(f"{prefix} must score the locked Set8 7/8 retrieval gate.")
    retrieval_passed = correct >= 7
    if _boolean(summary.get("gate_passed"), field=f"{prefix}.summary.gate_passed") is not retrieval_passed:
        raise ValueError(f"{prefix} stored retrieval gate disagrees with recomputed counts.")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "episodes_sha256": episodes_sha,
        "teacher_cache_lock_sha256": cache_lock_sha,
        "episode_state_contract_sha256": state_contract_sha,
        "lineage": dict(lineage),
        "teacher_hashes": lineage_hashes,
        "teacher_control_sha256": control_sha,
        "correct": correct,
        "retrieval_passed": retrieval_passed,
        "parent_checkpoint_sha256": parent_checkpoint_sha,
    }


def _validate_qa_gate(
    report: Mapping[str, Any],
    *,
    control: str,
    qa_retrieval: Mapping[str, Any],
) -> dict[str, Any]:
    prefix = f"{control}.qa_gate"
    if report.get("schema_version") != "vlm.r3.set8_gate.v1" or report.get("suite") != "set8":
        raise ValueError(f"{prefix} is not an R3 Set8 gate report.")
    if _integer(report.get("count"), field=f"{prefix}.count") != 32:
        raise ValueError(f"{prefix} must contain exactly 32 held-out views.")
    checks = _mapping(report.get("checks"), field=f"{prefix}.checks")
    required_checks = {
        "accuracy",
        "positions",
        "per_state",
        "rotation_consistency",
        "reset_drop",
        "shuffle_drop",
    }
    if set(checks) != required_checks or any(not isinstance(value, bool) for value in checks.values()):
        raise ValueError(f"{prefix} has an incomplete or non-boolean check set.")
    passed = all(checks.values())
    if _boolean(report.get("passed"), field=f"{prefix}.passed") is not passed:
        raise ValueError(f"{prefix} stored pass flag disagrees with its checks.")
    payload = dict(report)
    reported_payload_sha = _sha256(
        payload.pop("scientific_payload_sha256", None),
        field=f"{prefix}.scientific_payload_sha256",
    )
    provenance = _mapping(payload.pop("artifact_provenance", None), field=f"{prefix}.artifact_provenance")
    try:
        validate_r3_micro_artifact_provenance(provenance, suite="set8")
    except ValueError as exc:
        raise ValueError(f"{prefix} has invalid artifact provenance: {exc}") from exc
    if _canonical_sha256(payload) != reported_payload_sha:
        raise ValueError(f"{prefix} scientific payload SHA256 mismatch.")
    lineage = qa_retrieval["lineage"]
    expected_provenance = {
        "checkpoint_sha256": qa_retrieval["checkpoint_sha256"],
        "training_regime": "teacher_assisted",
        "parent_checkpoint_regime": "teacher_assisted",
        "objective_stage": "qa",
        "reader_loss_mode": "listwise-choice",
        "choice_permutation_family_sha256": lineage["choice_permutation_family_sha256"],
        "eval_choice_permutation_family_sha256": lineage["eval_choice_permutation_family_sha256"],
        "teacher_control": control,
        "teacher_control_sha256": qa_retrieval["teacher_control_sha256"],
        "presentations_per_state": 512,
        "distill_presentations": 256,
        "qa_presentations": 256,
        **qa_retrieval["teacher_hashes"],
    }
    for field, expected in expected_provenance.items():
        actual = provenance.get(field)
        if field.endswith("_sha256"):
            actual = _sha256(actual, field=f"{prefix}.artifact_provenance.{field}")
        if actual != expected:
            raise ValueError(f"{prefix} provenance field {field!r} differs from QA checkpoint lineage.")
    return {"passed": passed, "scientific_payload_sha256": reported_payload_sha}


def _validate_arm(control: str, arm: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "distill_summary",
        "qa_summary",
        "distill_retrieval",
        "qa_retrieval",
        "qa_gate",
    }
    if set(arm) != expected_keys:
        raise ValueError(f"{control} arm must contain exactly {sorted(expected_keys)}.")
    distill_summary = _validate_summary(
        _mapping(arm["distill_summary"], field=f"{control}.distill_summary"),
        control=control,
        stage="distill",
    )
    qa_summary = _validate_summary(
        _mapping(arm["qa_summary"], field=f"{control}.qa_summary"),
        control=control,
        stage="qa",
    )
    distill_retrieval = _validate_retrieval(
        _mapping(arm["distill_retrieval"], field=f"{control}.distill_retrieval"),
        control=control,
        stage="distill",
    )
    qa_retrieval = _validate_retrieval(
        _mapping(arm["qa_retrieval"], field=f"{control}.qa_retrieval"),
        control=control,
        stage="qa",
    )
    if qa_retrieval["parent_checkpoint_sha256"] != distill_retrieval["checkpoint_sha256"]:
        raise ValueError(f"{control} QA checkpoint is not descended from its supplied distill checkpoint.")
    for field in (
        "episodes_sha256",
        "teacher_cache_lock_sha256",
        "episode_state_contract_sha256",
        "teacher_hashes",
        "teacher_control_sha256",
    ):
        if qa_retrieval[field] != distill_retrieval[field]:
            raise ValueError(f"{control} distill/QA retrieval reports differ on locked field {field!r}.")
    for summary_name, summary in (("distill", distill_summary), ("qa", qa_summary)):
        if summary["teacher_manifest_sha256"] != distill_retrieval["teacher_hashes"]["teacher_manifest_sha256"]:
            raise ValueError(f"{control} {summary_name} summary teacher manifest differs from checkpoint lineage.")
        if summary["teacher_control_sha256"] != distill_retrieval["teacher_control_sha256"]:
            raise ValueError(f"{control} {summary_name} summary teacher-control SHA differs from checkpoint lineage.")
    qa_gate = _validate_qa_gate(
        _mapping(arm["qa_gate"], field=f"{control}.qa_gate"),
        control=control,
        qa_retrieval=qa_retrieval,
    )
    substitution_candidate = bool(
        distill_summary["distill_progress_passed"] and distill_retrieval["retrieval_passed"] and qa_gate["passed"]
    )
    return {
        "distill": distill_summary,
        "distill_retrieval": {
            "correct": distill_retrieval["correct"],
            "passed": distill_retrieval["retrieval_passed"],
        },
        "qa_retrieval_correct": qa_retrieval["correct"],
        "qa_gate": qa_gate,
        "substitution_candidate": substitution_candidate,
        "locks": {
            "episodes_sha256": distill_retrieval["episodes_sha256"],
            "teacher_cache_lock_sha256": distill_retrieval["teacher_cache_lock_sha256"],
            "episode_state_contract_sha256": distill_retrieval["episode_state_contract_sha256"],
            "teacher_hashes": distill_retrieval["teacher_hashes"],
            "teacher_control_sha256": distill_retrieval["teacher_control_sha256"],
            "distill_checkpoint_sha256": distill_retrieval["checkpoint_sha256"],
            "qa_checkpoint_sha256": qa_retrieval["checkpoint_sha256"],
        },
    }


def score_r3_teacher_attribution(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Score the complete correct/shuffled/random R3 Set8 teacher package."""

    if set(arms) != set(TEACHER_CONTROLS):
        raise ValueError(f"arms must contain exactly {sorted(TEACHER_CONTROLS)}.")
    results = {
        control: _validate_arm(control, _mapping(arms[control], field=f"arms.{control}"))
        for control in TEACHER_CONTROLS
    }
    correct_arm = arms["correct"]
    retention = compare_retrieval_retention(
        reference_report=_mapping(correct_arm["distill_retrieval"], field="correct.distill_retrieval"),
        current_report=_mapping(correct_arm["qa_retrieval"], field="correct.qa_retrieval"),
        minimum_retention=0.9,
    )
    reported_retention = _mapping(
        correct_arm["qa_retrieval"].get("retention"),
        field="correct.qa_retrieval.retention",
    )
    for field in ("reference_correct", "current_correct"):
        if _integer(reported_retention.get(field), field=f"correct.qa_retrieval.retention.{field}") != retention[field]:
            raise ValueError(f"correct QA retrieval stored retention field {field!r} disagrees with recomputation.")
    for field in ("retention", "minimum_retention"):
        reported = _finite(reported_retention.get(field), field=f"correct.qa_retrieval.retention.{field}")
        if not math.isclose(reported, float(retention[field]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"correct QA retrieval stored retention field {field!r} disagrees with recomputation.")
    if (
        _boolean(
            reported_retention.get("gate_passed"),
            field="correct.qa_retrieval.retention.gate_passed",
        )
        is not retention["gate_passed"]
    ):
        raise ValueError("correct QA retrieval stored retention pass flag disagrees with recomputation.")

    correct_locks = results["correct"]["locks"]
    for control in ("shuffled", "random-moment-matched"):
        control_locks = results[control]["locks"]
        for field in (
            "episodes_sha256",
            "teacher_cache_lock_sha256",
            "episode_state_contract_sha256",
            "teacher_hashes",
        ):
            if control_locks[field] != correct_locks[field]:
                raise ValueError(f"{control} arm differs from correct arm on locked field {field!r}.")
    control_contracts = [results[control]["locks"]["teacher_control_sha256"] for control in TEACHER_CONTROLS]
    if len(set(control_contracts)) != len(control_contracts):
        raise ValueError("correct/shuffled/random teacher-control contract SHA256 values must be distinct.")
    checkpoint_shas = [
        results[control]["locks"][field]
        for control in TEACHER_CONTROLS
        for field in ("distill_checkpoint_sha256", "qa_checkpoint_sha256")
    ]
    if len(set(checkpoint_shas)) != len(checkpoint_shas):
        raise ValueError("Every teacher arm/stage must use a distinct checkpoint artifact.")

    checks = {
        "correct_composite_drop_at_least_50_percent": results["correct"]["distill"][
            "composite_drop_at_least_50_percent"
        ],
        "correct_all_raw_components_decreased": results["correct"]["distill"]["all_raw_components_decreased"],
        "correct_distill_retrieval_at_least_7_of_8": results["correct"]["distill_retrieval"]["passed"],
        "correct_qa_retrieval_retention_at_least_90_percent": bool(retention["gate_passed"]),
        "correct_qa_set8_gate_passed": results["correct"]["qa_gate"]["passed"],
        "shuffled_teacher_cannot_substitute": not results["shuffled"]["substitution_candidate"],
        "random_teacher_cannot_substitute": not results["random-moment-matched"]["substitution_candidate"],
    }
    payload = {
        "schema": R3_TEACHER_ATTRIBUTION_SCHEMA,
        "thresholds": {
            "composite_final_over_initial_max": 0.5,
            "all_raw_components_strictly_decrease": True,
            "distill_teacher_retrieval_correct_min": 7,
            "distill_teacher_retrieval_count": 8,
            "qa_retrieval_retention_min": 0.9,
            "controls_must_not_pass_distill_retrieval_and_qa_gate": True,
        },
        "arms": results,
        "correct_retention": retention,
        "checks": checks,
        "passed": all(checks.values()),
    }
    return {**payload, "scientific_payload_sha256": _canonical_sha256(payload)}


__all__ = [
    "R3_TEACHER_ATTRIBUTION_SCHEMA",
    "TEACHER_CONTROLS",
    "score_r3_teacher_attribution",
]
