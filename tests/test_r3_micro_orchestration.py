from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cluster"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from render_r3_micro_gates import (  # noqa: E402
    MicroPaths,
    SuiteSpec,
    build_stages,
    materialize_dry_run,
    render_stage_sbatch,
)
from score_r3_micro import (  # noqa: E402
    build_artifact_provenance,
    scientific_prediction_payload,
    sha256_file,
)
from dreamlite_episode import audit_episode_gradients, gradient_audit_summary  # noqa: E402
from validate_r3_micro_prerequisites import validate_prerequisites  # noqa: E402
from validate_r3_micro_replication import validate_replication  # noqa: E402


SHA = "a" * 64
COMMIT = "b" * 40


def _technical_report() -> dict:
    return {
        "protocol": "R3-technical-listwise-v1",
        "through": "DL-S",
        "required_gates": ["G4-L", "G5-L", "G6-L", "DL-S"],
        "checks": {
            "G4-L": {"valid": True},
            "G5-L": {"valid": True},
            "G6-L": {"valid": True},
            "DL-S": {"valid": True},
        },
        "errors": [],
        "git_commit": COMMIT,
        "passed": True,
    }


def _s0_report() -> dict:
    return {
        "schema_version": 1,
        "probe": "r3_s0_qwen_scorer_contract",
        "passed": True,
        "contract": {"reader_loss_mode": "listwise-choice"},
        "summary": {
            "views_passed": 8,
            "views_required": 8,
            "joint_tokenization_views_passed": 8,
            "train_eval_views_passed": 8,
            "repeat_eval_views_passed": 8,
        },
        "provenance": {"git": {"commit": COMMIT, "clean": True}},
    }


def _teacher_t0_report() -> dict:
    return {
        "schema_version": 1,
        "probe": "teacher_t0_real_qwen_integrity_upper_bound",
        "passed": True,
        "cache_integrity": {"passed": True},
        "cross_split_fail_closed": {"passed": True},
        "upper_bound": {"passed": True},
        "identity_mutations": {"state": {"passed": True}},
        "provenance": {"git": {"commit": COMMIT, "clean": True}},
        "frozen_gradients": {"reader": {"frozen_tensors_with_grad": 0}},
    }


def _paths(root: Path) -> MicroPaths:
    return MicroPaths(
        project=root / "project",
        environment=root / "environment",
        model_root=root / "models",
        run_root=root / "run",
        scorer_s0_report=root / "s0.json",
        technical_report=root / "technical.json",
        teacher_t0_report=root / "teacher_t0.json",
    )


def _suite(root: Path, name: str, episodes: int) -> SuiteSpec:
    return SuiteSpec(
        name=name,
        train=root / "data" / name / "train.jsonl",
        gate=root / "data" / name / "gate.jsonl",
        teacher_cache=root / "teacher" / name,
        train_sha256=SHA,
        gate_sha256=SHA,
        teacher_manifest_sha256=SHA,
        teacher_sidecar_sha256=SHA,
        teacher_calibration_sha256=SHA,
        episodes=episodes,
    )


def test_prerequisites_require_both_complete_reports_and_clean_commit() -> None:
    passed = validate_prerequisites(
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=_teacher_t0_report(),
        training_regime="teacher_assisted",
        expected_commit=COMMIT,
    )
    assert passed["passed"] is True
    failed_t0 = _teacher_t0_report()
    failed_t0["upper_bound"] = {"passed": False}
    failed = validate_prerequisites(
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=failed_t0,
        training_regime="teacher_assisted",
        expected_commit=COMMIT,
    )
    assert failed["passed"] is False
    assert any("upper_bound" in error for error in failed["errors"])
    qa_only = validate_prerequisites(
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=None,
        training_regime="qa_only",
        expected_commit=COMMIT,
    )
    assert qa_only["passed"] is True
    assert qa_only["teacher_t0_required"] is False
    assert qa_only["teacher_t0_complete"] is None


def test_micro_gradient_audit_requires_finite_positive_state_and_image_gradients() -> None:
    state = torch.tensor([2.0], requires_grad=True)
    image = state.square()
    image.retain_grad()
    loss = image.sum()
    loss.backward()
    accumulator: dict[str, list[float]] = {}
    audit_episode_gradients(
        [("final_state", state), ("query_image", image)],
        accumulator,
    )
    summary = gradient_audit_summary(accumulator, enabled=True, objective_stage="qa")
    assert summary["passed"] is True
    assert summary["categories"]["final_state"]["positive_finite"] == 1

    zero = torch.tensor([1.0], requires_grad=True)
    (zero * 0.0).sum().backward()
    with pytest.raises(RuntimeError, match="non-positive"):
        audit_episode_gradients([("final_state", zero)], {})


def test_score_provenance_binds_rows_report_checkpoint_and_lineage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "last.pt"
        checkpoint.write_bytes(b"checkpoint")
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "optimizer_steps": 256,
                    "state_gradient_audit": {"enabled": True, "passed": True},
                }
            ),
            encoding="utf-8",
        )
        metric_rows = [
            {
                "kind": "train",
                "optimizer_step": step,
                "choice_rotation_counts": [2, 2, 2, 2],
                "gradient_norm": 1.0,
                "state_gradient_audit": {"passed": True},
            }
            for step in range(1, 257)
        ]
        metric_rows.extend({"kind": "dev", "optimizer_step": step, "loss": 1.0} for step in range(64, 257, 32))
        (root / "metrics.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in metric_rows),
            encoding="utf-8",
        )
        for step in range(32, 257, 32):
            (root / f"checkpoint-{step:06d}.pt").write_bytes(b"checkpoint")
        predictions = root / "predictions.jsonl"
        row = {
            "checkpoint": str(checkpoint),
            "training_regime": "teacher_assisted",
            "parent_checkpoint_regime": "teacher_assisted",
            "teacher_control": "correct",
            "recurrence_mode": "direct_latent",
            "initial_state_mode": "blank",
            "seed": 0,
            "diffusion_seed": 0,
            "deterministic_ce": True,
        }
        predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
        lineage = {
            "schema_version": 2,
            "training_regime": "teacher_assisted",
            "parent_checkpoint_regime": "teacher_assisted",
            "objective_stage": "qa",
            "reader_loss_mode": "listwise-choice",
            "choice_view_schedule": "cyclic4",
            "teacher_control": "correct",
            "teacher_control_sha256": SHA,
            "teacher_manifest_sha256": SHA,
            "teacher_sidecar_sha256": SHA,
            "teacher_calibration_sha256": SHA,
            "presentations_per_state": 512,
            "distill_presentations": 256,
            "qa_presentations": 256,
            "parent_checkpoint_sha256": SHA,
        }
        arguments = {
            "reader_loss_mode": "listwise-choice",
            "choice_view_schedule": "cyclic4",
            "recurrence_mode": "direct_latent",
            "detach_between_events": False,
            "noop_policy": "update",
            "initial_state_mode": "blank",
            "learn_initial_state": False,
            "lora_rank": 4,
            "seed": 0,
            "adapter_seed": 0,
            "strict_determinism": True,
            "audit_state_gradients": True,
            "disable_early_stopping": True,
            "require_mixed_delayed_probe": True,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "gradient_accumulation": 8,
            "gradient_clip": 1.0,
            "resolution": 1024,
            "checkpoint_unet": True,
            "curriculum": "full",
            "max_optimizer_steps": None,
            "max_train_episodes": 8,
            "epochs": 256,
            "presentations_per_state": 256,
            "distill_presentations": 256,
            "qa_presentations": 256,
            "checkpoint_every": 32,
            "eval_start_step": 64,
            "eval_every": 32,
            "eval_limit": 8,
        }
        companion = root / "predictions.jsonl.report.json"
        companion.write_text(
            json.dumps(
                {
                    "output_sha256": sha256_file(predictions),
                    "choice_view_family": "reverse-cyclic4",
                    "conditions": ["standard", "reset", "shuffle"],
                    "noop_policy": "keep",
                    "episodes_sha256": SHA,
                    "deterministic_ce": True,
                    "checkpoint_manifest": {
                        "training_lineage": lineage,
                        "arguments": arguments,
                        "strict_determinism": {"enabled": True},
                        "dev_sha256": SHA,
                    },
                }
            ),
            encoding="utf-8",
        )
        provenance = build_artifact_provenance(
            predictions=predictions,
            rows=[row],
            prediction_report=companion,
            suite="set8",
        )
        assert provenance["checkpoint_sha256"] == sha256_file(checkpoint)
        assert provenance["objective_stage"] == "qa"
        assert provenance["distill_presentations"] == 256


def _gate_report(path: str, *, payload_sha: str = SHA) -> dict:
    return {
        "schema_version": "vlm.r3.set8_gate.v1",
        "suite": "set8",
        "passed": True,
        "scientific_payload_sha256": payload_sha,
        "scientific_prediction_payload": {"sha256": "d" * 64},
        "artifact_provenance": {
            "checkpoint_path": path,
            "checkpoint_sha256": SHA,
            "training_regime": "qa_only",
            "objective_stage": "qa",
            "teacher_control": "none",
        },
    }


def test_replication_requires_two_fresh_paths_and_identical_scientific_payload() -> None:
    passed = validate_replication(
        _gate_report("/run/A/last.pt"),
        _gate_report("/run/B/last.pt"),
        suite="set8",
        training_regime="qa_only",
        teacher_control="correct",
    )
    assert passed["passed"] is True
    drift = validate_replication(
        _gate_report("/run/A/last.pt"),
        _gate_report("/run/B/last.pt", payload_sha="c" * 64),
        suite="set8",
        training_regime="qa_only",
        teacher_control="correct",
    )
    assert drift["passed"] is False
    assert drift["bitwise_scientific_payload_match"] is False
    prediction_drift_b = _gate_report("/run/B/last.pt")
    prediction_drift_b["scientific_prediction_payload"]["sha256"] = "e" * 64
    prediction_drift = validate_replication(
        _gate_report("/run/A/last.pt"),
        prediction_drift_b,
        suite="set8",
        training_regime="qa_only",
        teacher_control="correct",
    )
    assert prediction_drift["passed"] is False


def test_scientific_prediction_hash_excludes_latency_but_includes_choice_scores() -> None:
    row = {
        "episode_id": "r3-set8-r0-v0",
        "query_ordinal": 0,
        "probe_role": "delayed",
        "choice_view_family": "reverse-cyclic4",
        "choice_view_index": 0,
        "condition": "standard",
        "choices": ["a", "b", "c", "d"],
        "target_index": 0,
        "target_text": "a",
        "prediction_index": 0,
        "prediction_text": "a",
        "choice_mean_nll": [1.0, 2.0, 3.0, 4.0],
        "latency_seconds": 1.0,
    }
    first = scientific_prediction_payload([row])
    latency_changed = scientific_prediction_payload([{**row, "latency_seconds": 99.0}])
    score_changed = scientific_prediction_payload([{**row, "choice_mean_nll": [1.1, 2.0, 3.0, 4.0]}])
    assert first["sha256"] == latency_changed["sha256"]
    assert first["sha256"] != score_changed["sha256"]


def test_micro_dag_locks_tracks_controls_budgets_and_unlocks() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = _paths(root)
        set8 = _suite(root, "set8", 8)
        transition16 = _suite(root, "transition16", 16)
        stages = build_stages(paths, set8, transition16)
        by_name = {stage.name: stage for stage in stages}
        assert list(by_name) == [
            "QA8-A",
            "QA8-B",
            "QA16-A",
            "QA16-B",
            "TD8-CORRECT-A",
            "TD8-SHUFFLED-A",
            "TD8-RANDOM-A",
            "TD8-ATTRIBUTION-A",
            "TD8-CORRECT-B",
            "TD16-CORRECT-A",
            "TD16-CORRECT-B",
        ]
        assert by_name["QA8-B"].dependencies == ("QA8-A",)
        assert by_name["QA16-A"].dependencies == ("QA8-B",)
        assert by_name["TD8-ATTRIBUTION-A"].dependencies == (
            "TD8-CORRECT-A",
            "TD8-SHUFFLED-A",
            "TD8-RANDOM-A",
        )
        assert by_name["TD16-CORRECT-A"].dependencies == ("TD8-CORRECT-B",)

        qa8 = "\n".join(by_name["QA8-A"].commands)
        assert "--training-regime qa_only" in qa8
        assert "--epochs 512" in qa8
        assert "--presentations-per-state 512" in qa8
        assert "--eval-start-step 64" in qa8
        assert "--eval-every 32" in qa8
        assert "--choice-view-family reverse-cyclic4" in qa8
        assert "--require-mixed-delayed-probe" in qa8
        assert "--conditions standard reset shuffle" in qa8

        qa16 = "\n".join(by_name["QA16-A"].commands)
        assert "--eval-start-step 128" in qa16
        assert "--eval-every 64" in qa16
        assert "state_swap" in qa16

        correct = "\n".join(by_name["TD8-CORRECT-A"].commands)
        assert "--objective-stage distill" in correct
        assert "--epochs 256" in correct
        assert "--distill-presentations 256" in correct
        assert "--objective-stage qa" in correct
        assert "--qa-presentations 256" in correct
        assert "--initialize-from" in correct
        assert "--expected-teacher-control correct" in correct
        shuffled = "\n".join(by_name["TD8-SHUFFLED-A"].commands)
        random = "\n".join(by_name["TD8-RANDOM-A"].commands)
        assert "--teacher-control shuffled" in shuffled
        assert "--no-fail-on-gate" in shuffled
        assert "--teacher-control random-moment-matched" in random
        attribution = "\n".join(by_name["TD8-ATTRIBUTION-A"].commands)
        assert "score_r3_teacher_attribution.py" in attribution
        assert "--random-qa-gate" in attribution

        text = render_stage_sbatch(
            by_name["QA8-A"],
            paths=paths,
            expected_commit=COMMIT,
            expected_torch="2.7.1+cu118",
            scorer_s0_report_sha256=SHA,
            technical_report_sha256=SHA,
            teacher_t0_report_sha256=SHA,
        )
        assert "#SBATCH --nodes=1" in text
        assert "#SBATCH --gres=gpu:2" in text
        assert "#SBATCH --partition=a800" in text
        assert "validate_r3_micro_prerequisites.py" in text
        assert "--training-regime qa_only" in text
        assert "--teacher-t0-report" not in text
        assert "transitions.jsonl" not in text
        assert "R3_SUBMISSION_SUPPORTED=0" in text

        teacher_text = render_stage_sbatch(
            by_name["TD8-CORRECT-A"],
            paths=paths,
            expected_commit=COMMIT,
            expected_torch="2.7.1+cu118",
            scorer_s0_report_sha256=SHA,
            technical_report_sha256=SHA,
            teacher_t0_report_sha256=SHA,
        )
        assert "--training-regime teacher_assisted" in teacher_text
        assert "--teacher-t0-report" in teacher_text
        assert "transitions.jsonl" in teacher_text


def test_materialized_plan_is_template_only_and_has_separate_unlocks() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = _paths(root)
        set8 = _suite(root, "set8", 8)
        transition16 = _suite(root, "transition16", 16)
        stages = build_stages(paths, set8, transition16)
        manifest = materialize_dry_run(
            paths=paths,
            stages=stages,
            set8=set8,
            transition16=transition16,
            expected_commit=COMMIT,
            expected_torch="2.7.1+cu118",
            scorer_s0_report_sha256=SHA,
            technical_report_sha256=SHA,
            teacher_t0_report_sha256=SHA,
        )
        assert manifest["dry_run"] is True
        assert manifest["submission_supported"] is False
        assert manifest["unlock_rules"]["cross_track_substitution"] is False
        assert "T0" not in manifest["jobs"]["QA8-A"]["hard_prerequisites"]
        assert "T0" in manifest["jobs"]["TD8-CORRECT-A"]["hard_prerequisites"]
        assert manifest["fixed_protocol"]["qa_only_presentations_per_state"] == 512
        assert manifest["fixed_protocol"]["teacher_distill_presentations_per_state"] == 256
        assert all(job["nodes"] == 1 and job["gpus_per_node"] == 2 for job in manifest["jobs"].values())
        assert len(list(paths.sbatch.glob("*.sbatch"))) == len(stages)
