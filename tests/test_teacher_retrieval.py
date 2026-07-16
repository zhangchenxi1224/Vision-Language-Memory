from __future__ import annotations

import hashlib

import pytest
import torch

from vision_memory.data.schema import Episode, EventKind, QuerySpec, Turn, TurnType
from vision_memory.eval.teacher_retrieval import (
    TEACHER_RETRIEVAL_SCHEMA,
    compare_retrieval_retention,
    final_teacher_state_ids,
    latent_smooth_l1_distance,
    retrieve_teacher_state,
    score_teacher_retrieval,
    teacher_cache_lock_sha256,
    validate_teacher_checkpoint_lineage,
)
from vision_memory.teacher import TeacherTransitionRecord


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def episode() -> Episode:
    return Episode(
        episode_id="set8-0",
        split="train",
        seed=0,
        entity_id="entity-0",
        template_id="template-a",
        turns=(
            Turn(type=TurnType.EVENT, event_kind=EventKind.SET, event_text="Use red."),
            Turn(
                type=TurnType.QUERY,
                query=QuerySpec(
                    text="Which color?",
                    choices=("red", "blue", "green", "yellow"),
                    target_index=0,
                ),
            ),
        ),
        pair_id="pair-0",
        counterfactual_episode_id="set8-1",
        topic="color",
    )


def transition() -> TeacherTransitionRecord:
    return TeacherTransitionRecord(
        episode_id="set8-0",
        turn_id=0,
        before_state_id=digest("empty"),
        after_state_id=digest("red"),
        event_kind="set",
        teacher_key=digest("teacher-red"),
    )


def lineage(*, stage: str = "distill") -> dict[str, object]:
    manifest_sha = digest("manifest-file")
    sidecar_sha = digest("sidecar-file")
    calibration_sha = digest("calibration-file")
    return {
        "schema_version": 2,
        "training_regime": "teacher_assisted",
        "objective_stage": stage,
        "reader_loss_mode": "listwise-choice",
        "teacher_control": "correct",
        "teacher_checkpoint_is_qa_only_eligible": False,
        "teacher_supervision_loaded": stage == "distill",
        "teacher_manifest_sha256": manifest_sha,
        "teacher_sidecar_sha256": sidecar_sha,
        "teacher_calibration_sha256": calibration_sha,
        "parent_checkpoint_sha256": None if stage == "distill" else digest("distill-checkpoint"),
        "parent_checkpoint_regime": None if stage == "distill" else "teacher_assisted",
        "distill_presentations": 256,
        "qa_presentations": 0 if stage == "distill" else 256,
    }


def test_latent_metric_matches_per_channel_affine_invariance() -> None:
    teacher = torch.tensor([[[[0.0, 1.0], [2.0, 4.0]], [[-2.0, 3.0], [1.0, 8.0]]]], dtype=torch.float32)
    scale = torch.tensor([3.0, 2.0]).view(1, 2, 1, 1)
    offset = torch.tensor([7.0, -5.0]).view(1, 2, 1, 1)
    student = teacher * scale + offset
    assert float(latent_smooth_l1_distance(student, teacher)) == pytest.approx(0.0, abs=1e-12)


def test_retrieve_correct_state_from_complete_candidate_set() -> None:
    expected = digest("state-red")
    other = digest("state-blue")
    teacher = torch.tensor([[[[0.0, 1.0], [2.0, 4.0]]]], dtype=torch.float32)
    different = torch.tensor([[[[4.0, 2.0], [1.0, 0.0]]]], dtype=torch.float32)
    match = retrieve_teacher_state(
        episode_id="set8-0",
        student_latent=teacher.clone(),
        expected_state_id=expected,
        teacher_latents={other: different, expected: teacher},
    )
    assert match.correct
    assert match.predicted_state_id == expected
    assert match.expected_distance == pytest.approx(0.0)
    assert match.top_tie_count == 1
    summary = score_teacher_retrieval([match], expected_episodes=1, minimum_correct=1)
    assert summary["gate_passed"]


def test_normalization_collision_is_ambiguous_and_not_counted_correct() -> None:
    first = digest("a")
    second = digest("b")
    latent = torch.tensor([[[[0.0, 1.0], [2.0, 4.0]]]], dtype=torch.float32)
    match = retrieve_teacher_state(
        episode_id="set8-0",
        student_latent=latent,
        expected_state_id=first,
        teacher_latents={first: latent, second: latent * 3.0 + 8.0},
        tie_tolerance=1e-10,
    )
    assert match.top_tie_count == 2
    assert not match.correct


def test_final_state_contract_checks_exact_event_route() -> None:
    assert final_teacher_state_ids([episode()], [transition()]) == {"set8-0": digest("red")}
    bad = TeacherTransitionRecord(
        episode_id="set8-0",
        turn_id=1,
        before_state_id=digest("empty"),
        after_state_id=digest("red"),
        event_kind="set",
        teacher_key=digest("teacher-red"),
    )
    with pytest.raises(ValueError, match="route differs"):
        final_teacher_state_ids([episode()], [bad])


def test_checkpoint_lineage_is_bound_to_all_teacher_lock_files() -> None:
    values = lineage()
    manifest = {"schema_version": 2, "training_lineage": values}
    validated = validate_teacher_checkpoint_lineage(
        manifest,
        manifest_file_sha256=digest("manifest-file"),
        sidecar_file_sha256=digest("sidecar-file"),
        calibration_file_sha256=digest("calibration-file"),
    )
    assert validated["objective_stage"] == "distill"
    with pytest.raises(ValueError, match="does not match"):
        validate_teacher_checkpoint_lineage(
            manifest,
            manifest_file_sha256=digest("tampered"),
            sidecar_file_sha256=digest("sidecar-file"),
            calibration_file_sha256=digest("calibration-file"),
        )
    random_teacher = dict(values, teacher_control="random")
    with pytest.raises(ValueError, match="shuffled/random"):
        validate_teacher_checkpoint_lineage(
            {"schema_version": 2, "training_lineage": random_teacher},
            manifest_file_sha256=digest("manifest-file"),
            sidecar_file_sha256=digest("sidecar-file"),
            calibration_file_sha256=digest("calibration-file"),
        )


def test_cache_lock_is_stable_and_sensitive_to_every_component() -> None:
    arguments = {
        "manifest_file_sha256": digest("manifest-file"),
        "sidecar_file_sha256": digest("sidecar-file"),
        "calibration_file_sha256": digest("calibration-file"),
        "manifest_payload_sha256": digest("manifest-payload"),
    }
    first = teacher_cache_lock_sha256(**arguments)
    assert first == teacher_cache_lock_sha256(**arguments)
    assert first != teacher_cache_lock_sha256(**{**arguments, "manifest_payload_sha256": digest("changed-payload")})


def test_qa_retention_requires_exact_distill_parent_and_same_lockbox() -> None:
    cache_lock = digest("cache-lock")
    episode_sha = digest("episodes")
    contract_sha = digest("episode-contract")
    distill_checkpoint = digest("distill-checkpoint")
    reference = {
        "schema": TEACHER_RETRIEVAL_SCHEMA,
        "objective_stage": "distill",
        "checkpoint_sha256": distill_checkpoint,
        "episodes_sha256": episode_sha,
        "teacher_cache_lock_sha256": cache_lock,
        "episode_state_contract_sha256": contract_sha,
        "summary": {"correct": 8},
    }
    current = {
        "schema": TEACHER_RETRIEVAL_SCHEMA,
        "objective_stage": "qa",
        "checkpoint_sha256": digest("qa-checkpoint"),
        "episodes_sha256": episode_sha,
        "teacher_cache_lock_sha256": cache_lock,
        "episode_state_contract_sha256": contract_sha,
        "training_lineage": {"parent_checkpoint_sha256": distill_checkpoint},
        "summary": {"correct": 7},
    }
    comparison = compare_retrieval_retention(
        reference_report=reference,
        current_report=current,
        minimum_retention=0.9,
    )
    assert comparison["retention"] == pytest.approx(0.875)
    assert not comparison["gate_passed"]
    with pytest.raises(ValueError, match="not descended"):
        compare_retrieval_retention(
            reference_report=reference,
            current_report={
                **current,
                "training_lineage": {"parent_checkpoint_sha256": digest("other")},
            },
        )
