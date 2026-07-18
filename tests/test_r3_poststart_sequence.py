from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import pytest

from scripts.inspire import run_r3_technical_sequence as sequence
from scripts.inspire.r3_dag_contract import atomic_json as atomic_contract_json


def make_args(tmp_path: Path) -> argparse.Namespace:
    root = tmp_path.resolve()
    train = root / "train.jsonl"
    dev = root / "dev.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    dev.write_text("{}\n", encoding="utf-8")
    python = root / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("fixture\n", encoding="utf-8")
    platform_status = root / "platform-status.json"
    platform_sha256 = atomic_contract_json(
        platform_status,
        {
            "schema_version": 1,
            "protocol": sequence.PLATFORM_STATUS_PROTOCOL,
            "captured_at": datetime.now(UTC).isoformat(),
            "instance": "instance",
            "status": "RUNNING",
            "node": "node",
            "node_status": "READY",
            "image": "image",
            "image_source": "SOURCE_OFFICIAL",
            "workspace": "workspace",
            "project": "project",
            "project_priority": "10",
            "gpu_product": "H200",
            "gpu_count": 2,
            "cpu_count": 40,
            "memory_gib": 400,
            "shared_memory_gib": 128,
            "auto_stop": False,
        },
    )
    return argparse.Namespace(
        repo=root / "repo",
        python=python,
        model_root=root / "models",
        runs_root=root / "runs",
        run_name="r3-technical-unit",
        preflight=root / "runs" / "preflight" / "formal.json",
        train=train,
        train_sha256="a" * 64,
        dev=dev,
        dev_sha256="b" * 64,
        expected_commit="c" * 40,
        expected_instance="instance",
        expected_node="node",
        expected_image="image",
        expected_workspace="workspace",
        expected_project="project",
        platform_status=platform_status,
        platform_status_sha256=platform_sha256,
        max_platform_status_age_seconds=1800,
        expected_driver="driver",
        hf_home=root / "hf",
        torch_home=root / "torch",
        poll_seconds=0.01,
    )


def test_poststart_sequence_runs_only_locked_technical_order(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    labels: list[str] = []
    authorized: list[str] = []

    def fake_run_checked(command, *, cwd, environment, events_path, label):
        del command, cwd, environment
        labels.append(label)
        if label == "formal_preflight":
            atomic_contract_json(args.preflight, {"passed": True, "formal_ready": True})
            plan["formal_preflight"]["sha256"] = sequence.sha256_file(args.preflight)
        sequence.append_event(events_path, {"event": "fake", "label": label})

    plan = {
        "formal_preflight": {"path": str(args.preflight), "sha256": "placeholder"},
        "stages": {
            stage: {
                "slug": f"{index:02d}-{stage.lower()}",
                "run_dir": str(tmp_path / "runs" / args.run_name / "stages" / stage),
            }
            for index, stage in enumerate(sequence.STAGES)
        },
    }

    def materialize(stage: str) -> dict:
        definition = plan["stages"][stage]
        launch = {
            "schema_version": 1,
            "protocol": sequence.LAUNCH_COMMAND_PROTOCOL,
            "stage": stage,
            "run_dir": definition["run_dir"],
            "executed": False,
            "argv": ["true"],
        }
        path = args.runs_root / args.run_name / "launch_commands" / f"{definition['slug']}.json"
        digest = atomic_contract_json(path, launch)
        return {"launch_path": str(path), "launch_sha256": digest, "launch": launch}

    def fake_initialize(**kwargs):
        run_root = kwargs["run_root"]
        (run_root / "launch_commands").mkdir(parents=True)
        return {"plan_sha256": "d" * 64, "first_stage": materialize(sequence.STAGES[0])}

    def fake_authorize(_root, *, stage, dry_run):
        del dry_run
        authorized.append(stage)
        return materialize(stage)

    with (
        mock.patch.object(sequence, "require_clean_commit"),
        mock.patch.object(sequence, "run_checked", side_effect=fake_run_checked),
        mock.patch.object(sequence, "initialize_technical_dag", side_effect=fake_initialize),
        mock.patch.object(sequence, "_load_verified_plan", return_value=(plan, Path("plan"), "d" * 64)),
        mock.patch.object(sequence, "wait_for_stage", return_value={"passed": True, "exit_code": 0}),
        mock.patch.object(sequence, "_verify_completed_stage", return_value=[{"sha256": "e" * 64}]),
        mock.patch.object(sequence, "authorize_stage", side_effect=fake_authorize),
    ):
        assert sequence.run_sequence(args) == 0

    assert labels == ["formal_preflight", *(f"launch:{stage}" for stage in sequence.STAGES)]
    assert authorized == list(sequence.STAGES[1:])
    terminal = json.loads((args.runs_root / "control" / args.run_name / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["passed"] is True
    assert terminal["completed_stages"] == list(sequence.STAGES)
    assert terminal["micro_or_pilot_authorized"] is False
    assert terminal["dag_plan_sha256"] == "d" * 64
    assert set(terminal["final_stage_bindings"]) == set(sequence.STAGES)


def test_wait_for_stage_rejects_failed_terminal(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage"
    run_dir.mkdir()
    sequence.atomic_json(run_dir / "terminal.json", {"passed": False, "exit_code": 1})
    plan = {"stages": {"G4-L": {"run_dir": str(run_dir)}}}
    with pytest.raises(RuntimeError, match="terminal is not passing"):
        sequence.wait_for_stage(
            plan=plan,
            stage="G4-L",
            timeout_seconds=1,
            poll_seconds=0.01,
            progress_path=tmp_path / "progress.json",
            progress_base={},
        )


def test_environment_is_explicit_and_deterministic(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    environment = sequence.build_environment(args)
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["VLM_INSPIRE_NODE"] == "node"
    assert environment["VLM_MODEL_ROOT"] == str(args.model_root)
    assert environment["VLM_RUN_ROOT"] == str(args.runs_root)


def test_venv_python_symlink_identity_is_not_dereferenced(tmp_path: Path) -> None:
    target = tmp_path / "base-python"
    target.write_text("python\n", encoding="utf-8")
    linked = tmp_path / "venv" / "bin" / "python"
    linked.parent.mkdir(parents=True)
    try:
        linked.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    normalized = sequence.require_absolute_executable(linked, "--python")
    assert normalized == Path(os.path.abspath(linked))
    assert normalized != target.resolve()


def test_launch_binding_rejects_sidecar_and_in_memory_tampering(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    stage = "R3-R0"
    plan = {"stages": {stage: {"slug": "00-r3-r0", "run_dir": str(run_root / "stage")}}}
    launch = {
        "schema_version": 1,
        "protocol": sequence.LAUNCH_COMMAND_PROTOCOL,
        "stage": stage,
        "run_dir": str(run_root / "stage"),
        "executed": False,
        "argv": [sys.executable, "-c", "pass"],
    }
    path = run_root / "launch_commands" / "00-r3-r0.json"
    digest = atomic_contract_json(path, launch)
    materialized = {"launch_path": str(path), "launch_sha256": digest, "launch": launch}
    path.write_text(json.dumps({**launch, "argv": ["tampered"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        sequence.verify_launch_binding(run_root=run_root, plan=plan, stage=stage, materialized=materialized)

    tampered = {**launch, "argv": ["tampered"]}
    atomic_contract_json(path, tampered)
    materialized["launch_sha256"] = sequence.sha256_file(path)
    with pytest.raises(ValueError, match="materializer return value"):
        sequence.verify_launch_binding(run_root=run_root, plan=plan, stage=stage, materialized=materialized)


def test_wait_timeout_reports_live_worker_without_authorizing(tmp_path: Path) -> None:
    run_dir = tmp_path / "stage"
    run_dir.mkdir()
    sequence.atomic_json(run_dir / "running.json", {"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()})
    plan = {"stages": {"G4-L": {"run_dir": str(run_dir)}}}
    with pytest.raises(sequence.StageOrchestrationTimeout) as raised:
        sequence.wait_for_stage(
            plan=plan,
            stage="G4-L",
            timeout_seconds=0,
            poll_seconds=0.01,
            progress_path=tmp_path / "progress.json",
            progress_base={},
        )
    assert raised.value.worker_may_still_be_running is True
    assert raised.value.snapshot["pid_alive"] is True


def test_duplicate_control_is_refused_without_mutating_original(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    control = args.runs_root / "control" / args.run_name
    control.mkdir(parents=True)
    marker = control / "progress.json"
    marker.write_bytes(b"original\n")
    with (
        mock.patch.object(sequence, "run_checked") as run_checked,
        mock.patch.object(sequence, "initialize_technical_dag") as initialize,
    ):
        assert sequence.run_sequence(args) == 2
    run_checked.assert_not_called()
    initialize.assert_not_called()
    assert marker.read_bytes() == b"original\n"
    attempts = list((args.runs_root / "control" / "_attempts" / args.run_name).glob("*.json"))
    assert len(attempts) == 1
    assert json.loads(attempts[0].read_text(encoding="utf-8"))["state"] == "duplicate_control_refused"


def test_sequence_timeout_never_authorizes_successor(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    first = sequence.STAGES[0]
    run_root = args.runs_root / args.run_name
    run_dir = run_root / "stages" / first
    plan = {
        "formal_preflight": {"path": str(args.preflight), "sha256": "placeholder"},
        "stages": {
            stage: {
                "slug": f"{index:02d}-{stage.lower()}",
                "run_dir": str(run_root / "stages" / stage),
            }
            for index, stage in enumerate(sequence.STAGES)
        },
    }

    def fake_run_checked(command, *, cwd, environment, events_path, label):
        del command, cwd, environment
        if label == "formal_preflight":
            atomic_contract_json(args.preflight, {"passed": True, "formal_ready": True})
            plan["formal_preflight"]["sha256"] = sequence.sha256_file(args.preflight)
        sequence.append_event(events_path, {"event": "fake", "label": label})

    def fake_initialize(**kwargs):
        launch = {
            "schema_version": 1,
            "protocol": sequence.LAUNCH_COMMAND_PROTOCOL,
            "stage": first,
            "run_dir": str(run_dir),
            "executed": False,
            "argv": ["true"],
        }
        path = kwargs["run_root"] / "launch_commands" / "00-r3-r0.json"
        digest = atomic_contract_json(path, launch)
        return {
            "plan_sha256": "d" * 64,
            "first_stage": {"launch_path": str(path), "launch_sha256": digest, "launch": launch},
        }

    timeout = sequence.StageOrchestrationTimeout(
        first,
        run_dir,
        {"status": "running", "pid": os.getpid(), "pid_alive": True},
    )
    with (
        mock.patch.object(sequence, "require_clean_commit"),
        mock.patch.object(sequence, "run_checked", side_effect=fake_run_checked),
        mock.patch.object(sequence, "initialize_technical_dag", side_effect=fake_initialize),
        mock.patch.object(sequence, "_load_verified_plan", return_value=(plan, Path("plan"), "d" * 64)),
        mock.patch.object(sequence, "wait_for_stage", side_effect=timeout),
        mock.patch.object(sequence, "authorize_stage") as authorize,
        mock.patch.object(sequence, "_verify_completed_stage") as verify,
    ):
        assert sequence.run_sequence(args) == 124
    authorize.assert_not_called()
    verify.assert_not_called()
    terminal = json.loads((args.runs_root / "control" / args.run_name / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["state"] == "orchestration_timeout_unknown"
    assert terminal["worker_may_still_be_running"] is True
    assert terminal["current_stage"] == first


def test_final_evidence_drift_prevents_success(tmp_path: Path) -> None:
    args = make_args(tmp_path)
    plan = {
        "formal_preflight": {"path": str(args.preflight), "sha256": "placeholder"},
        "stages": {
            stage: {
                "slug": f"{index:02d}-{stage.lower()}",
                "run_dir": str(args.runs_root / args.run_name / "stages" / stage),
            }
            for index, stage in enumerate(sequence.STAGES)
        },
    }

    def materialize(stage: str) -> dict:
        definition = plan["stages"][stage]
        launch = {
            "schema_version": 1,
            "protocol": sequence.LAUNCH_COMMAND_PROTOCOL,
            "stage": stage,
            "run_dir": definition["run_dir"],
            "executed": False,
            "argv": ["true"],
        }
        path = args.runs_root / args.run_name / "launch_commands" / f"{definition['slug']}.json"
        digest = atomic_contract_json(path, launch)
        return {"launch_path": str(path), "launch_sha256": digest, "launch": launch}

    def fake_run_checked(command, *, cwd, environment, events_path, label):
        del command, cwd, environment
        if label == "formal_preflight":
            atomic_contract_json(args.preflight, {"passed": True, "formal_ready": True})
            plan["formal_preflight"]["sha256"] = sequence.sha256_file(args.preflight)
        sequence.append_event(events_path, {"event": "fake", "label": label})

    def fake_initialize(**kwargs):
        (kwargs["run_root"] / "launch_commands").mkdir(parents=True)
        return {"plan_sha256": "d" * 64, "first_stage": materialize(sequence.STAGES[0])}

    verify_calls = 0

    def verify_stage(_plan, _stage):
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == len(sequence.STAGES) + 1:
            raise ValueError("final evidence drift")
        return [{"sha256": "e" * 64}]

    with (
        mock.patch.object(sequence, "require_clean_commit"),
        mock.patch.object(sequence, "run_checked", side_effect=fake_run_checked),
        mock.patch.object(sequence, "initialize_technical_dag", side_effect=fake_initialize),
        mock.patch.object(sequence, "_load_verified_plan", return_value=(plan, Path("plan"), "d" * 64)),
        mock.patch.object(sequence, "wait_for_stage", return_value={"passed": True, "exit_code": 0}),
        mock.patch.object(sequence, "_verify_completed_stage", side_effect=verify_stage),
        mock.patch.object(
            sequence,
            "authorize_stage",
            side_effect=lambda _root, *, stage, dry_run: materialize(stage),
        ),
    ):
        assert sequence.run_sequence(args) == 1
    terminal = json.loads((args.runs_root / "control" / args.run_name / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["passed"] is False
    assert terminal["state"] == "failed"
    assert "final evidence drift" in terminal["detail"]
