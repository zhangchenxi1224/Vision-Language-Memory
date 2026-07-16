"""Fail-closed nearest-neighbour diagnostics for recurrent teacher-state latents."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from vision_memory.data.schema import Episode
from vision_memory.teacher import TeacherTransitionRecord, normalize_latent_per_channel


TEACHER_RETRIEVAL_SCHEMA = "vision_memory.teacher-state-retrieval.v1"


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA256 digest.")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a lowercase SHA256 digest.") from exc
    if value != value.lower():
        raise ValueError(f"{field} must be a lowercase SHA256 digest.")
    return value


def teacher_cache_lock_sha256(
    *,
    manifest_file_sha256: str,
    sidecar_file_sha256: str,
    calibration_file_sha256: str,
    manifest_payload_sha256: str,
) -> str:
    """Hash every metadata lock needed to interpret the immutable tensor cache."""

    payload = {
        "schema": "vision_memory.teacher-cache-lock.v1",
        "manifest_file_sha256": _require_sha256(manifest_file_sha256, field="manifest_file_sha256"),
        "sidecar_file_sha256": _require_sha256(sidecar_file_sha256, field="sidecar_file_sha256"),
        "calibration_file_sha256": _require_sha256(calibration_file_sha256, field="calibration_file_sha256"),
        "manifest_payload_sha256": _require_sha256(manifest_payload_sha256, field="manifest_payload_sha256"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_teacher_checkpoint_lineage(
    checkpoint_manifest: Mapping[str, Any],
    *,
    manifest_file_sha256: str,
    sidecar_file_sha256: str,
    calibration_file_sha256: str,
    expected_teacher_control: str = "correct",
) -> dict[str, Any]:
    """Require an R3 teacher checkpoint tied to the supplied cache files."""

    if expected_teacher_control not in {"correct", "shuffled", "random-moment-matched"}:
        raise ValueError("expected_teacher_control has an unsupported R3 teacher control.")

    if int(checkpoint_manifest.get("schema_version", 0)) < 2:
        raise ValueError("Teacher retrieval requires a schema-v2 R3 checkpoint manifest.")
    lineage = checkpoint_manifest.get("training_lineage")
    if not isinstance(lineage, Mapping) or int(lineage.get("schema_version", 0)) < 2:
        raise ValueError("Checkpoint is missing schema-v2 training_lineage.")
    if lineage.get("training_regime") != "teacher_assisted":
        raise ValueError("Teacher-state retrieval accepts only teacher_assisted lineage.")
    objective_stage = lineage.get("objective_stage")
    if objective_stage not in {"distill", "qa"}:
        raise ValueError("Teacher checkpoint objective_stage must be 'distill' or 'qa'.")
    if lineage.get("reader_loss_mode") != "listwise-choice":
        raise ValueError("R3 teacher retrieval requires listwise-choice Reader lineage.")
    if lineage.get("teacher_control") != expected_teacher_control:
        if expected_teacher_control == "correct":
            raise ValueError("Correct-state retrieval cannot be attributed to shuffled/random teacher controls.")
        raise ValueError("Checkpoint teacher_control differs from the expected retrieval arm.")
    if lineage.get("teacher_checkpoint_is_qa_only_eligible") is not False:
        raise ValueError("Teacher checkpoint is incorrectly marked QA-only eligible.")
    expected_loaded = objective_stage == "distill"
    if lineage.get("teacher_supervision_loaded") is not expected_loaded:
        raise ValueError("teacher_supervision_loaded conflicts with checkpoint objective_stage.")
    expected_hashes = {
        "teacher_manifest_sha256": manifest_file_sha256,
        "teacher_sidecar_sha256": sidecar_file_sha256,
        "teacher_calibration_sha256": calibration_file_sha256,
    }
    for field, expected in expected_hashes.items():
        _require_sha256(expected, field=field)
        actual = _require_sha256(lineage.get(field), field=f"training_lineage.{field}")
        if actual != expected:
            raise ValueError(f"Checkpoint {field} does not match the supplied locked teacher cache.")
    if objective_stage == "distill":
        if lineage.get("parent_checkpoint_sha256") is not None:
            raise ValueError("A distill-only checkpoint must use fresh LoRA lineage.")
        if int(lineage.get("distill_presentations", 0)) <= 0:
            raise ValueError("Distill-only checkpoint records no distillation presentations.")
    else:
        _require_sha256(
            lineage.get("parent_checkpoint_sha256"),
            field="training_lineage.parent_checkpoint_sha256",
        )
        if lineage.get("parent_checkpoint_regime") != "teacher_assisted":
            raise ValueError("Teacher-assisted QA checkpoint must retain its teacher parent regime.")
        if int(lineage.get("distill_presentations", 0)) <= 0 or int(lineage.get("qa_presentations", 0)) <= 0:
            raise ValueError("Teacher-assisted QA checkpoint must record both distill and QA presentations.")
    return dict(lineage)


def final_teacher_state_ids(
    episodes: Iterable[Episode],
    transitions: Iterable[TeacherTransitionRecord],
    *,
    require_exact_episode_set: bool = True,
) -> dict[str, str]:
    """Resolve each episode's final post-event semantic state with route-level checks."""

    episode_values = tuple(episodes)
    transition_values = tuple(transitions)
    episode_ids = [episode.episode_id for episode in episode_values]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Training episode input contains duplicate episode_id values.")
    if any(episode.split != "train" for episode in episode_values):
        raise ValueError("Teacher-state retrieval is train-only.")
    by_episode: dict[str, list[TeacherTransitionRecord]] = defaultdict(list)
    for transition in transition_values:
        by_episode[transition.episode_id].append(transition)
    if require_exact_episode_set and set(by_episode) != set(episode_ids):
        missing = sorted(set(episode_ids) - set(by_episode))
        unexpected = sorted(set(by_episode) - set(episode_ids))
        raise ValueError(
            f"Teacher sidecar episode set differs from input episodes: missing={missing}, unexpected={unexpected}"
        )
    result: dict[str, str] = {}
    for episode in episode_values:
        expected_routes = [
            (turn_index, turn.event_kind.value)
            for turn_index, turn in enumerate(episode.turns)
            if turn.calls_updater and turn.event_kind is not None
        ]
        actual = sorted(by_episode.get(episode.episode_id, ()), key=lambda record: record.turn_id)
        actual_routes = [(record.turn_id, record.event_kind) for record in actual]
        if actual_routes != expected_routes:
            raise ValueError(
                f"Teacher sidecar route differs for {episode.episode_id!r}: "
                f"expected={expected_routes}, actual={actual_routes}"
            )
        if not actual:
            raise ValueError(f"Episode {episode.episode_id!r} has no teacher-supervised update.")
        result[episode.episode_id] = actual[-1].after_state_id
    return result


def latent_smooth_l1_distance(student: Tensor, teacher: Tensor) -> Tensor:
    """The exact latent metric used by ``L_z``, before frozen-scale division."""

    if not isinstance(student, Tensor) or not isinstance(teacher, Tensor):
        raise TypeError("student and teacher latents must be tensors.")
    if tuple(student.shape) != tuple(teacher.shape):
        raise ValueError(f"Student latent shape {tuple(student.shape)} differs from teacher {tuple(teacher.shape)}.")
    if student.requires_grad:
        student = student.detach()
    target = teacher.detach().to(device=student.device, dtype=student.dtype)
    distance = F.smooth_l1_loss(
        normalize_latent_per_channel(student),
        normalize_latent_per_channel(target),
    )
    if distance.numel() != 1 or not torch.isfinite(distance):
        raise ValueError("Teacher latent distance must be one finite scalar.")
    return distance


@dataclass(frozen=True)
class TeacherRetrievalMatch:
    episode_id: str
    expected_state_id: str
    predicted_state_id: str
    correct: bool
    expected_distance: float
    predicted_distance: float
    runner_up_distance: float | None
    margin: float | None
    top_tie_count: int
    candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def retrieve_teacher_state(
    *,
    episode_id: str,
    student_latent: Tensor,
    expected_state_id: str,
    teacher_latents: Mapping[str, Tensor],
    tie_tolerance: float = 0.0,
) -> TeacherRetrievalMatch:
    """Rank the complete teacher cache and treat a top-distance tie as incorrect."""

    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id must be non-empty.")
    _require_sha256(expected_state_id, field="expected_state_id")
    if expected_state_id not in teacher_latents:
        raise KeyError("Expected teacher state is absent from the candidate cache.")
    if not teacher_latents:
        raise ValueError("teacher_latents cannot be empty.")
    if not isinstance(tie_tolerance, (float, int)) or not math.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError("tie_tolerance must be a finite non-negative scalar.")
    distances: list[tuple[float, str]] = []
    for state_id, teacher_latent in teacher_latents.items():
        _require_sha256(state_id, field="teacher state_id")
        value = float(latent_smooth_l1_distance(student_latent, teacher_latent).item())
        distances.append((value, state_id))
    distances.sort(key=lambda item: (item[0], item[1]))
    predicted_distance, predicted_state_id = distances[0]
    tie_count = sum(abs(distance - predicted_distance) <= tie_tolerance for distance, _ in distances)
    expected_distance = next(distance for distance, state_id in distances if state_id == expected_state_id)
    runner_up = distances[1][0] if len(distances) > 1 else None
    margin = None if runner_up is None else runner_up - predicted_distance
    return TeacherRetrievalMatch(
        episode_id=episode_id,
        expected_state_id=expected_state_id,
        predicted_state_id=predicted_state_id,
        correct=predicted_state_id == expected_state_id and tie_count == 1,
        expected_distance=expected_distance,
        predicted_distance=predicted_distance,
        runner_up_distance=runner_up,
        margin=margin,
        top_tie_count=tie_count,
        candidate_count=len(distances),
    )


def score_teacher_retrieval(
    matches: Iterable[TeacherRetrievalMatch],
    *,
    expected_episodes: int,
    minimum_correct: int,
) -> dict[str, Any]:
    values = tuple(matches)
    if len(values) != expected_episodes:
        raise ValueError(f"Expected exactly {expected_episodes} retrieval rows, found {len(values)}.")
    if len({value.episode_id for value in values}) != len(values):
        raise ValueError("Teacher retrieval rows contain duplicate episode IDs.")
    if not 0 <= minimum_correct <= expected_episodes:
        raise ValueError("minimum_correct must lie in [0, expected_episodes].")
    correct = sum(value.correct for value in values)
    return {
        "n_episodes": len(values),
        "correct": correct,
        "top1_accuracy": correct / len(values),
        "minimum_correct": minimum_correct,
        "gate_passed": correct >= minimum_correct,
        "ambiguous_top_ties": sum(value.top_tie_count != 1 for value in values),
    }


def compare_retrieval_retention(
    *,
    reference_report: Mapping[str, Any],
    current_report: Mapping[str, Any],
    minimum_retention: float = 0.9,
) -> dict[str, Any]:
    """Compare QA-end retrieval with its exact distill-parent report."""

    if not 0.0 <= minimum_retention <= 1.0:
        raise ValueError("minimum_retention must lie in [0, 1].")
    for name, report in (("reference", reference_report), ("current", current_report)):
        if report.get("schema") != TEACHER_RETRIEVAL_SCHEMA:
            raise ValueError(f"{name} report has an unsupported schema.")
    if reference_report.get("objective_stage") != "distill":
        raise ValueError("Retrieval retention reference must be a distill-only checkpoint report.")
    if current_report.get("objective_stage") != "qa":
        raise ValueError("Retrieval retention current report must be a teacher-assisted QA checkpoint report.")
    for field in ("episodes_sha256", "teacher_cache_lock_sha256", "episode_state_contract_sha256"):
        if current_report.get(field) != reference_report.get(field):
            raise ValueError(f"Retrieval reports differ on locked field {field!r}.")
    current_lineage = current_report.get("training_lineage")
    if not isinstance(current_lineage, Mapping):
        raise ValueError("Current retrieval report is missing training_lineage.")
    if current_lineage.get("parent_checkpoint_sha256") != reference_report.get("checkpoint_sha256"):
        raise ValueError("QA retrieval checkpoint is not descended from the distill reference checkpoint.")
    reference_summary = reference_report.get("summary")
    current_summary = current_report.get("summary")
    if not isinstance(reference_summary, Mapping) or not isinstance(current_summary, Mapping):
        raise ValueError("Retrieval report summaries are missing.")
    reference_correct = int(reference_summary.get("correct", -1))
    current_correct = int(current_summary.get("correct", -1))
    if reference_correct <= 0:
        raise ValueError("Cannot define retrieval retention from a zero-correct reference.")
    retention = current_correct / reference_correct
    return {
        "reference_correct": reference_correct,
        "current_correct": current_correct,
        "retention": retention,
        "minimum_retention": minimum_retention,
        "gate_passed": retention >= minimum_retention,
    }


__all__ = [
    "TEACHER_RETRIEVAL_SCHEMA",
    "TeacherRetrievalMatch",
    "compare_retrieval_retention",
    "final_teacher_state_ids",
    "latent_smooth_l1_distance",
    "retrieve_teacher_state",
    "score_teacher_retrieval",
    "teacher_cache_lock_sha256",
    "validate_teacher_checkpoint_lineage",
]
