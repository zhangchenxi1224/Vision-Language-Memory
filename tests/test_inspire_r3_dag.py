from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSPIRE = ROOT / "scripts" / "inspire"
sys.path.insert(0, str(INSPIRE))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from materialize_r3_dag import (  # noqa: E402
    TECHNICAL_ORDER,
    TEACHER_PREPARATION_ORDER,
    _completed_teacher_preparation_parent,
    _load_micro_command_contract,
    _require_run_child,
    _teacher_preregistered_contract,
    _validate_set8_parent_role,
    _validate_set8_parent_dag_identity,
    _validate_teacher_parent_policy,
    _validate_micro_runtime_bindings,
    _verify_completed_stage,
    _validate_micro_prerequisites,
    authorize_stage,
    initialize_micro_extension,
    initialize_teacher_preparation_dag,
    initialize_technical_dag,
    main as materialize_main,
)
from launch_background import STRICT_ENVIRONMENT  # noqa: E402
from model_snapshot_manifest import (  # noqa: E402
    SNAPSHOT_MANIFEST_NAME,
    create_snapshot_manifest,
    verify_snapshot_manifest,
)
from r3_dag_contract import (  # noqa: E402
    MICRO_COMMAND_PROTOCOL,
    PLAN_PROTOCOL,
    STAGE_EVIDENCE_PROTOCOL,
    STAGE_SPEC_PROTOCOL,
    atomic_json,
    load_json_object,
    require_absolute_executable,
    sha256_file,
)
from run_r3_stage import run_bound_stage  # noqa: E402
from dreamlite_episode import parse_args as parse_training_args, training_lineage  # noqa: E402


FIXED_MICRO_PROTOCOL = {
    "reader_loss_mode": "listwise-choice",
    "train_choice_family": "cyclic4",
    "gate_choice_family": "reverse-cyclic4",
    "dreamlite": "DreamLite-mobile-4-step",
    "lora_rank": 4,
    "strict_determinism": True,
}


def _qa_micro_contract_payload(tmp_path: Path, suite: str) -> dict:
    train = tmp_path / f"{suite}-train.jsonl"
    gate = tmp_path / f"{suite}-gate.jsonl"
    if not train.exists():
        train.write_text('{"episode_id":"train"}\n', encoding="utf-8")
    if not gate.exists():
        gate.write_text('{"episode_id":"gate"}\n', encoding="utf-8")
    episodes = 8 if suite == "set8" else 16
    final_step = 512 * episodes // 8
    arm_ids = ["A"] if suite == "set8" else ["A", "B"]
    arms = []
    commands: list[list[str]] = []
    outputs: list[dict] = []
    gate_reports: dict[str, Path] = {}

    for arm_id in arm_ids:
        output_dir = (tmp_path / "run" / arm_id / "qa").resolve()
        checkpoint = output_dir / f"checkpoint-{final_step:06d}.pt"
        predictions = output_dir / "gate_predictions.jsonl"
        prediction_report = predictions.with_suffix(predictions.suffix + ".report.json")
        score = output_dir / "gate_report.json"
        gate_reports[arm_id] = score
        arms.append(
            {
                "arm_id": arm_id,
                "teacher_control": "none",
                "qa_output_dir": str(output_dir),
                "distill_output_dir": None,
            }
        )
        commands.extend(
            [
                [
                    sys.executable,
                    str(ROOT / "scripts" / "train" / "dreamlite_episode.py"),
                    "--training-regime",
                    "qa_only",
                    "--objective-stage",
                    "qa",
                    "--teacher-control",
                    "correct",
                    "--train",
                    str(train.resolve()),
                    "--dev",
                    str(gate.resolve()),
                    "--dreamlite",
                    str((tmp_path / "models" / "DreamLite-mobile").resolve()),
                    "--reader",
                    str((tmp_path / "models" / "Qwen3-VL-4B-Instruct").resolve()),
                    "--output-dir",
                    str(output_dir),
                    "--presentations-per-state",
                    "512",
                    "--epochs",
                    "512",
                    "--gradient-accumulation",
                    "8",
                    "--max-train-episodes",
                    str(episodes),
                ],
                [
                    sys.executable,
                    str(ROOT / "scripts" / "eval" / "dreamlite_mcq.py"),
                    "--episodes",
                    str(gate.resolve()),
                    "--format",
                    "synthetic",
                    "--dreamlite",
                    str((tmp_path / "models" / "DreamLite-mobile").resolve()),
                    "--reader",
                    str((tmp_path / "models" / "Qwen3-VL-4B-Instruct").resolve()),
                    "--checkpoint",
                    str(checkpoint),
                    "--expected-training-regime",
                    "qa_only",
                    "--output",
                    str(predictions),
                    "--method",
                    f"r3-{suite}-qa-only-{arm_id}",
                ],
                [
                    sys.executable,
                    str(ROOT / "scripts" / "eval" / "score_r3_micro.py"),
                    "--predictions",
                    str(predictions),
                    "--prediction-report",
                    str(prediction_report),
                    "--suite",
                    suite,
                    "--output",
                    str(score),
                    "--expected-git-commit",
                    "a" * 40,
                    "--expected-reader-revision",
                    "b" * 40,
                    "--expected-dreamlite-revision",
                    "c" * 40,
                    "--expected-train-sha256",
                    sha256_file(train),
                    "--expected-dev-sha256",
                    sha256_file(gate),
                ],
            ]
        )
        score_kind = "scientific_report" if suite == "set8" else "artifact"
        score_requirements = (
            {
                "passed": True,
                "suite": suite,
                "training_regime": "qa_only",
                "artifact_provenance_validated": True,
            }
            if score_kind == "scientific_report"
            else {}
        )
        outputs.extend(
            [
                {
                    "kind": "artifact",
                    "label": f"{arm_id}-checkpoint",
                    "path": str(checkpoint),
                    "required_values": {},
                },
                {
                    "kind": "artifact",
                    "label": f"{arm_id}-summary",
                    "path": str(output_dir / "summary.json"),
                    "required_values": {},
                },
                {
                    "kind": "artifact",
                    "label": f"{arm_id}-metrics",
                    "path": str(output_dir / "metrics.jsonl"),
                    "required_values": {},
                },
                {
                    "kind": "artifact",
                    "label": f"{arm_id}-predictions",
                    "path": str(predictions),
                    "required_values": {},
                },
                {
                    "kind": "artifact",
                    "label": f"{arm_id}-prediction-report",
                    "path": str(prediction_report),
                    "required_values": {},
                },
                {
                    "kind": score_kind,
                    "label": f"{arm_id}-score",
                    "path": str(score),
                    "required_values": score_requirements,
                },
            ]
        )

    if suite == "transition16":
        replication_report = (tmp_path / "run" / "replication_report.json").resolve()
        commands.append(
            [
                sys.executable,
                str(ROOT / "scripts" / "probes" / "validate_r3_micro_replication.py"),
                "--a",
                str(gate_reports["A"]),
                "--b",
                str(gate_reports["B"]),
                "--suite",
                "transition16",
                "--training-regime",
                "qa_only",
                "--teacher-control",
                "none",
                "--output",
                str(replication_report),
            ]
        )
        outputs.append(
            {
                "kind": "scientific_report",
                "label": "replication",
                "path": str(replication_report),
                "required_values": {
                    "passed": True,
                    "suite": suite,
                    "training_regime": "qa_only",
                    "artifact_provenance_validated": True,
                },
            }
        )

    return {
        "schema_version": 2,
        "protocol": MICRO_COMMAND_PROTOCOL,
        "stage": f"{suite}-qa",
        "suite": suite,
        "training_regime": "qa_only",
        "teacher_control": "none",
        "execution_shape": "single" if suite == "set8" else "paired-replica",
        "arms": arms,
        "data_binding": {
            "preregistration_sha256": "e" * 64,
            "train_sha256": sha256_file(train),
            "gate_sha256": sha256_file(gate),
        },
        "lineage_binding": {
            "git_commit": "a" * 40,
            "reader_revision": "b" * 40,
            "dreamlite_revision": "c" * 40,
        },
        "teacher_calibration_binding": None,
        "fixed_protocol": FIXED_MICRO_PROTOCOL,
        "commands": commands,
        "outputs": outputs,
    }


def _make_repo(root: Path) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    (repo / "models.lock.json").write_text(
        json.dumps(
            {
                "models": {
                    "dreamlite_mobile": {
                        "repo_id": "fixture/dreamlite",
                        "revision": "c" * 40,
                        "local_dir": "models/DreamLite-mobile",
                        "snapshot_manifest": SNAPSHOT_MANIFEST_NAME,
                    },
                    "qwen_reader": {
                        "repo_id": "fixture/qwen",
                        "revision": "b" * 40,
                        "local_dir": "models/Qwen3-VL-4B-Instruct",
                        "snapshot_manifest": SNAPSHOT_MANIFEST_NAME,
                    },
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "README.md", "models.lock.json"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "fixture"], check=True)
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repo, commit


def _write_preflight(
    path: Path,
    commit: str,
    *,
    model_root: Path | None = None,
    runs_root: Path | None = None,
    python: Path | None = None,
    model_snapshots: dict[str, dict] | None = None,
) -> str:
    payload = {
        "passed": True,
        "formal_ready": True,
        "git": {"commit": commit},
        "paths": {
            "VLM_MODEL_ROOT": {"value": str(model_root)} if model_root is not None else {},
            "VLM_RUN_ROOT": {"value": str(runs_root)} if runs_root is not None else {},
        },
        "python": {"executable": str(python)} if python is not None else {},
        "models": {
            name: {"snapshot_manifest": binding}
            for name, binding in (model_snapshots or {}).items()
        },
    }
    return atomic_json(path, payload)


def _technical_fixture(root: Path, *, through: str = "DL-S", dry_run: bool = False) -> tuple[dict, dict]:
    repo, commit = _make_repo(root)
    model_root = root / "models"
    model_root.mkdir()
    (model_root / "DreamLite-mobile").mkdir()
    (model_root / "Qwen3-VL-4B-Instruct").mkdir()
    model_specs = {
        "dreamlite_mobile": ("DreamLite-mobile", "fixture/dreamlite", "c" * 40),
        "qwen_reader": ("Qwen3-VL-4B-Instruct", "fixture/qwen", "b" * 40),
    }
    model_snapshots: dict[str, dict] = {}
    for name, (directory, repo_id, revision) in model_specs.items():
        model_dir = model_root / directory
        (model_dir / ".locked_revision").write_text(revision + "\n", encoding="utf-8")
        (model_dir / ".snapshot_complete").write_text(revision + "\n", encoding="utf-8")
        (model_dir / "config.json").write_text('{"fixture":true}\n', encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(f"{name}-weights".encode())
        create_snapshot_manifest(model_dir=model_dir, repo_id=repo_id, revision=revision)
        model_snapshots[name] = verify_snapshot_manifest(
            manifest_path=model_dir / SNAPSHOT_MANIFEST_NAME,
            model_dir=model_dir,
            expected_repo_id=repo_id,
            expected_revision=revision,
        )
    train = root / "train.jsonl"
    dev = root / "dev.jsonl"
    train.write_text('{"episode_id":"train"}\n', encoding="utf-8")
    dev.write_text('{"episode_id":"dev"}\n', encoding="utf-8")
    preflight = root / "formal_preflight.json"
    run_root = root / "runs" / "r3-technical-unit"
    preflight_sha = _write_preflight(
        preflight,
        commit,
        model_root=model_root,
        runs_root=run_root.parent,
        python=Path(sys.executable).resolve(),
        model_snapshots=model_snapshots,
    )
    result = initialize_technical_dag(
        repo=repo,
        python=Path(sys.executable),
        model_root=model_root,
        train=train,
        train_sha256=sha256_file(train),
        dev=dev,
        dev_sha256=sha256_file(dev),
        run_root=run_root,
        preflight=preflight,
        expected_commit=commit,
        through=through,
        dry_run=dry_run,
    )
    context = {
        "repo": repo,
        "commit": commit,
        "model_root": model_root,
        "train": train,
        "dev": dev,
        "preflight": preflight,
        "preflight_sha": preflight_sha,
        "run_root": run_root,
    }
    return result, context


def _complete_technical_parent(root: Path) -> tuple[dict, dict]:
    _, context = _technical_fixture(root)
    plan = load_json_object(context["run_root"] / "dag_plan.json")
    for index, stage in enumerate(TECHNICAL_ORDER):
        _record_fake_success(plan, stage)
        if index + 1 < len(TECHNICAL_ORDER):
            authorize_stage(context["run_root"], stage=TECHNICAL_ORDER[index + 1])
    return plan, context


def _teacher_input_fixture(root: Path, context: dict) -> dict[str, Path]:
    reader_revision = "b" * 40
    dreamlite_revision = "c" * 40
    (context["model_root"] / "Qwen3-VL-4B-Instruct" / ".locked_revision").write_text(
        reader_revision + "\n", encoding="utf-8"
    )
    (context["model_root"] / "DreamLite-mobile" / ".locked_revision").write_text(
        dreamlite_revision + "\n", encoding="utf-8"
    )
    data = root / "teacher-inputs"
    data.mkdir()
    paths = {
        "set8_train": data / "set8_train.jsonl",
        "transition16_train": data / "transition16_train.jsonl",
        "transition16_gate": data / "transition16_gate.jsonl",
        "transition16_raw_sidecar": data / "transition16_raw_sidecar.jsonl",
        "font": data / "DejaVuSans.ttf",
    }
    for name, path in paths.items():
        path.write_text(f"{name}\n", encoding="utf-8")
    for suite in ("set8", "transition16"):
        cache = data / f"{suite}-cache"
        cache.mkdir()
        (cache / "manifest.json").write_text(f"{suite}-manifest\n", encoding="utf-8")
        (cache / "transitions.jsonl").write_text(f"{suite}-transitions\n", encoding="utf-8")
        (cache / "build_report.json").write_text(f"{suite}-build\n", encoding="utf-8")
        paths[f"{suite}_cache"] = cache
    set8_cache = paths["set8_cache"]
    transition16_cache = paths["transition16_cache"]
    preregistration = data / "r3_preregistration.json"
    preregistration.write_text(
        json.dumps(
            {
                "schema": "vision_memory.r3-preregistration.v1",
                "models": {
                    "reader": {"revision": reader_revision},
                    "updater": {"revision": dreamlite_revision},
                },
                "micro_execution": {
                    "teacher_preparation_order": list(TEACHER_PREPARATION_ORDER)
                },
                "micro_data": {
                    "set8": {"train_sha256": sha256_file(paths["set8_train"])},
                    "transition16": {
                        "train_sha256": sha256_file(paths["transition16_train"]),
                        "gate_sha256": sha256_file(paths["transition16_gate"]),
                        "raw_teacher_sidecar_sha256": sha256_file(
                            paths["transition16_raw_sidecar"]
                        ),
                    },
                },
                "teacher_contract": {
                    "font_sha256": sha256_file(paths["font"]),
                    "cache_manifest_sha256": {
                        "set8": sha256_file(set8_cache / "manifest.json"),
                        "transition16": sha256_file(transition16_cache / "manifest.json"),
                    },
                    "cache_builds": {
                        "set8": {
                            "transitions_sha256": sha256_file(set8_cache / "transitions.jsonl"),
                            "build_report_sha256": sha256_file(set8_cache / "build_report.json"),
                        },
                        "transition16": {
                            "transitions_sha256": sha256_file(
                                transition16_cache / "transitions.jsonl"
                            ),
                            "build_report_sha256": sha256_file(
                                transition16_cache / "build_report.json"
                            ),
                        },
                    },
                    "calibration_protocol": {
                        "global_seed": 0,
                        "adapter_seed": 0,
                        "lora_rank": 4,
                        "initial_state": "blank_1024",
                        "strict_cuda_determinism": True,
                        "sdpa_backend": "math_only",
                        "sample_unit": "one_unweighted_sample_per_updater_transition",
                        "query_turns_excluded": True,
                        "duplicate_semantic_after_states_retained": True,
                        "component_reduction": "median",
                        "set8_transition_samples": 8,
                        "transition16_transition_samples": 28,
                    },
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["preregistration"] = preregistration
    return paths


def _teacher_preparation_fixture(root: Path, *, dry_run: bool = False) -> tuple[dict, dict]:
    _, context = _complete_technical_parent(root)
    inputs = _teacher_input_fixture(root, context)
    teacher_run_root = root / "runs" / "r3-teacher-unit"
    result = initialize_teacher_preparation_dag(
        technical_run_root=context["run_root"],
        teacher_run_root=teacher_run_root,
        preregistration=inputs["preregistration"],
        set8_train=inputs["set8_train"],
        transition16_train=inputs["transition16_train"],
        transition16_gate=inputs["transition16_gate"],
        transition16_raw_sidecar=inputs["transition16_raw_sidecar"],
        set8_cache=inputs["set8_cache"],
        transition16_cache=inputs["transition16_cache"],
        font=inputs["font"],
        dry_run=dry_run,
    )
    return result, {**context, **inputs, "teacher_run_root": teacher_run_root}


def _record_fake_success(plan: dict, stage: str) -> None:
    definition = plan["stages"][stage]
    spec_path = Path(plan["run_root"]) / "authorizations" / f"{definition['slug']}.json"
    spec_sha256 = sha256_file(spec_path)
    spec = load_json_object(spec_path)
    for output in definition["outputs"]:
        path = Path(output["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        required = dict(output.get("required_values", {}))
        payload = required or {"passed": True}
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    if stage in {"CAL-Set8", "CAL-Transition16"}:
        suite = "set8" if stage == "CAL-Set8" else "transition16"
        calibration_label = f"teacher_{suite}_calibration"
        report_label = f"teacher_{suite}_calibration_report"
        calibration = next(output for output in definition["outputs"] if output["label"] == calibration_label)
        report = next(output for output in definition["outputs"] if output["label"] == report_label)
        report_path = Path(report["path"])
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        report_payload["calibration_file_sha256"] = sha256_file(Path(calibration["path"]))
        report_path.write_text(json.dumps(report_payload, sort_keys=True) + "\n", encoding="utf-8")
    if stage == "CAL-Transition16":
        final = next(
            output for output in definition["outputs"] if output["label"] == "teacher_preparation_final"
        )
        labels = {
            "tc0_raw": "teacher_tc0_raw",
            "tc0_validation": "teacher_tc0_validation",
            "tf0_raw": "teacher_tf0_raw",
            "tf0_validation": "teacher_tf0_validation",
            "t0": "teacher_t0",
            "set8_calibration": "teacher_set8_calibration",
            "set8_calibration_report": "teacher_set8_calibration_report",
            "transition16_calibration": "teacher_transition16_calibration",
            "transition16_calibration_report": "teacher_transition16_calibration_report",
        }
        all_outputs = {
            output["label"]: output
            for planned_stage in plan["strict_order"]
            for output in plan["stages"][planned_stage]["outputs"]
        }
        payload = {
            **final["required_values"],
            "artifacts": {
                name: {
                    "path": all_outputs[label]["path"],
                    "sha256": sha256_file(Path(all_outputs[label]["path"])),
                }
                for name, label in labels.items()
            },
        }
        Path(final["path"]).write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )
    run_dir = Path(definition["run_dir"])
    run_dir.mkdir(parents=True)
    worker_input = run_dir / "worker_input.json"
    configuration_sha = atomic_json(
        worker_input,
        {
            "stage": definition["launcher_stage"],
            "repo": plan["repo"],
            "run_root": plan["run_root"],
            "run_dir": definition["run_dir"],
            "expected_commit": plan["expected_commit"],
            "preflight": plan["formal_preflight"]["path"],
            "preflight_sha256": plan["formal_preflight"]["sha256"],
            "infrastructure_stage": False,
            "strict_environment": STRICT_ENVIRONMENT,
            "command": [
                plan["python"],
                str(Path(plan["repo"]) / "scripts" / "inspire" / "run_r3_stage.py"),
                "--spec",
                str(spec_path),
                "--spec-sha256",
                spec_sha256,
            ],
        },
    )
    (run_dir / "stdout.log").write_text("fixture stdout\n", encoding="utf-8")
    (run_dir / "stderr.log").write_text("", encoding="utf-8")
    atomic_json(
        run_dir / "terminal.json",
        {
            "status": "succeeded",
            "passed": True,
            "exit_code": 0,
            "expected_commit": plan["expected_commit"],
            "configuration_sha256": configuration_sha,
            "stdout_sha256": sha256_file(run_dir / "stdout.log"),
            "stderr_sha256": sha256_file(run_dir / "stderr.log"),
        },
    )
    outputs = [
        {
            "label": output["label"],
            "path": output["path"],
            "sha256": sha256_file(Path(output["path"])),
            "required_values": output["required_values"],
        }
        for output in definition["outputs"]
    ]
    atomic_json(
        Path(definition["evidence_path"]),
        {
            "schema_version": 1,
            "protocol": STAGE_EVIDENCE_PROTOCOL,
            "passed": True,
            "stage": stage,
            "stage_slug": definition["slug"],
            "launcher_stage": definition["launcher_stage"],
            "expected_commit": plan["expected_commit"],
            "configuration_sha256": configuration_sha,
            "stage_spec_sha256": spec_sha256,
            "plan_sha256": spec["plan_sha256"],
            "worker_input_path": str(worker_input),
            "formal_preflight_sha256": plan["formal_preflight"]["sha256"],
            "model_snapshots": plan["model_snapshots"],
            "prerequisites": spec["prerequisites"],
            "commands": [
                {"index": index, "exit_code": 0} for index, _ in enumerate(definition["commands"])
            ],
            "outputs": outputs,
        },
    )


def test_technical_dry_run_is_complete_inspire_only_and_writes_nothing(tmp_path: Path) -> None:
    result, context = _technical_fixture(tmp_path, dry_run=True)
    plan = result["plan"]
    assert result["dry_run"] is True
    assert plan["strict_order"] == list(TECHNICAL_ORDER)
    assert plan["stages"]["R3-R0"]["launcher_stage"] == "r3-r0"
    assert plan["formal_preflight"]["sha256"] == context["preflight_sha"]
    assert plan["external_scheduler_submission"] is False
    assert plan["submission_backend"] == "scripts/inspire/launch_background.py"
    serialized = json.dumps(plan).lower()
    assert "sbatch" not in serialized
    assert "fudan" not in serialized
    assert not context["run_root"].exists()


def test_initialization_materializes_only_r0_and_refuses_run_root_reuse(tmp_path: Path) -> None:
    result, context = _technical_fixture(tmp_path)
    run_root = context["run_root"]
    plan = load_json_object(run_root / "dag_plan.json")
    assert result["plan_sha256"] == sha256_file(run_root / "dag_plan.json")
    assert (run_root / "authorizations" / "00-r3-r0.json").is_file()
    assert not (run_root / "authorizations" / "01-r3-s0.json").exists()
    launch = load_json_object(run_root / "launch_commands" / "00-r3-r0.json")
    stage_index = launch["argv"].index("--stage")
    assert launch["argv"][stage_index + 1] == "r3-r0"
    assert launch["executed"] is False
    assert launch["run_dir"] == plan["stages"]["R3-R0"]["run_dir"]

    with pytest.raises(ValueError, match="run root already exists"):
        initialize_technical_dag(
            repo=context["repo"],
            python=Path(sys.executable),
            model_root=context["model_root"],
            train=context["train"],
            train_sha256=sha256_file(context["train"]),
            dev=context["dev"],
            dev_sha256=sha256_file(context["dev"]),
            run_root=run_root,
            preflight=context["preflight"],
            expected_commit=context["commit"],
        )


def test_next_stage_requires_terminal_and_sha_bound_predecessor_outputs(tmp_path: Path) -> None:
    _, context = _technical_fixture(tmp_path)
    run_root = context["run_root"]
    with pytest.raises((ValueError, FileNotFoundError), match="terminal.json"):
        authorize_stage(run_root, stage="R3-S0")

    plan = load_json_object(run_root / "dag_plan.json")
    _record_fake_success(plan, "R3-R0")
    result = authorize_stage(run_root, stage="R3-S0")
    prerequisites = result["stage_spec"]["prerequisites"]
    labels = {binding["label"] for binding in prerequisites}
    assert labels == {
        "R3-R0:terminal",
        "R3-R0:evidence",
        "upstream:r0_raw",
        "upstream:r0_validation",
    }
    assert all(len(binding["sha256"]) == 64 for binding in prerequisites)


def test_completed_chain_revalidation_rejects_rewritten_early_artifact(tmp_path: Path) -> None:
    _, context = _technical_fixture(tmp_path)
    run_root = context["run_root"]
    plan = load_json_object(run_root / "dag_plan.json")
    for index, stage in enumerate(TECHNICAL_ORDER):
        _record_fake_success(plan, stage)
        _verify_completed_stage(plan, stage)
        if index + 1 < len(TECHNICAL_ORDER):
            authorize_stage(run_root, stage=TECHNICAL_ORDER[index + 1])

    g4_output = Path(plan["stages"]["G4-L"]["outputs"][0]["path"])
    g4_output.write_text('{"passed":true,"rewritten":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _verify_completed_stage(plan, "G4-L")

    (Path(plan["stages"]["DL-S"]["run_dir"]) / "stdout.log").write_text(
        "rewritten log\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not bind stdout.log"):
        _verify_completed_stage(plan, "DL-S")


def test_upstream_mutation_blocks_next_stage_authorization(tmp_path: Path) -> None:
    _, context = _technical_fixture(tmp_path)
    run_root = context["run_root"]
    plan = load_json_object(run_root / "dag_plan.json")
    _record_fake_success(plan, "R3-R0")
    raw = Path(plan["stages"]["R3-R0"]["outputs"][0]["path"])
    raw.write_text('{"passed":false}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        authorize_stage(run_root, stage="R3-S0")
    assert not (run_root / "authorizations" / "01-r3-s0.json").exists()


def test_teacher_preparation_dry_run_binds_exact_parent_inputs_and_external_calibrations(
    tmp_path: Path,
) -> None:
    result, context = _teacher_preparation_fixture(tmp_path, dry_run=True)
    plan = result["plan"]
    assert plan["kind"] == "teacher-preparation"
    assert plan["strict_order"] == list(TEACHER_PREPARATION_ORDER)
    assert plan["technical_parent"]["run_root"] == str(context["run_root"])
    assert plan["stages"]["R3-TC0"]["external_prerequisites"]
    assert len(plan["stages"]["R3-TC0"]["commands"]) == 2
    assert len(plan["stages"]["R3-TF0"]["commands"]) == 2
    assert plan["stages"]["CAL-Transition16"]["commands"][-1][2] == "finalize"
    for label in ("teacher_set8_calibration", "teacher_transition16_calibration"):
        output = next(
            output
            for stage in plan["strict_order"]
            for output in plan["stages"][stage]["outputs"]
            if output["label"] == label
        )
        assert Path(output["path"]).is_relative_to(context["teacher_run_root"])
        assert not Path(output["path"]).is_relative_to(context["set8_cache"])
        assert not Path(output["path"]).is_relative_to(context["transition16_cache"])
    assert not context["teacher_run_root"].exists()


def test_teacher_preparation_rejects_preregistered_stage_identity_drift(tmp_path: Path) -> None:
    _, context = _teacher_preparation_fixture(tmp_path, dry_run=True)
    preregistration = context["preregistration"]
    payload = load_json_object(preregistration)
    payload["micro_execution"]["teacher_preparation_order"] = [
        "TC0",
        "TF0",
        "T0",
        "CAL-Set8",
        "CAL-Transition16",
    ]
    preregistration.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="teacher_preparation_order"):
        _teacher_preregistered_contract(preregistration)


def test_teacher_preparation_materializes_strictly_one_stage_at_a_time(tmp_path: Path) -> None:
    _, context = _teacher_preparation_fixture(tmp_path)
    run_root = context["teacher_run_root"]
    plan = load_json_object(run_root / "dag_plan.json")
    assert (run_root / "authorizations" / "00-r3-tc0.json").is_file()
    assert not (run_root / "authorizations" / "01-r3-tf0.json").exists()
    with pytest.raises((ValueError, FileNotFoundError), match="terminal.json"):
        authorize_stage(run_root, stage="R3-TF0")
    _record_fake_success(plan, "R3-TC0")
    tf0 = authorize_stage(run_root, stage="R3-TF0")
    labels = {binding["label"] for binding in tf0["stage_spec"]["prerequisites"]}
    assert {"R3-TC0:terminal", "R3-TC0:evidence"}.issubset(labels)
    assert {"upstream:teacher_tc0_raw", "upstream:teacher_tc0_validation"}.issubset(labels)


def test_completed_teacher_parent_exposes_sha_index_and_rejects_artifact_rewrite(
    tmp_path: Path,
) -> None:
    _, context = _teacher_preparation_fixture(tmp_path)
    run_root = context["teacher_run_root"]
    teacher = load_json_object(run_root / "dag_plan.json")
    for index, stage in enumerate(TEACHER_PREPARATION_ORDER):
        _record_fake_success(teacher, stage)
        if index + 1 < len(TEACHER_PREPARATION_ORDER):
            authorize_stage(run_root, stage=TEACHER_PREPARATION_ORDER[index + 1])
    suite_lock = teacher["teacher_contract"]["calibration_input_locks"]["set8"]
    calibration = next(
        output
        for output in teacher["stages"]["CAL-Set8"]["outputs"]
        if output["label"] == "teacher_set8_calibration"
    )
    contract = {
        "suite": "set8",
        "data_binding": {
            "preregistration_sha256": suite_lock["preregistration_sha256"],
            "train_sha256": suite_lock["train_sha256"],
            "gate_sha256": "d" * 64,
        },
        "teacher_calibration_binding": {
            **{
                field: suite_lock[field]
                for field in (
                    "suite",
                    "preregistration_sha256",
                    "train_sha256",
                    "manifest_sha256",
                    "sidecar_sha256",
                )
            },
            "calibration_sha256": sha256_file(Path(calibration["path"])),
        },
    }
    technical = load_json_object(context["run_root"] / "dag_plan.json")
    parent = _completed_teacher_preparation_parent(
        teacher_run_root=run_root,
        technical=technical,
        technical_run_root=context["run_root"],
        contract=contract,
    )
    assert parent["calibration_binding"]["calibration_sha256"] == sha256_file(
        Path(calibration["path"])
    )
    final = load_json_object(run_root / "results" / "teacher_preparation_final.json")
    assert final["artifacts"]["set8_calibration"]["path"] == calibration["path"]

    Path(calibration["path"]).write_text('{"rewritten":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _completed_teacher_preparation_parent(
            teacher_run_root=run_root,
            technical=technical,
            technical_run_root=context["run_root"],
            contract=contract,
        )


def test_teacher_preparation_cli_uses_its_explicit_branch(tmp_path: Path) -> None:
    technical_root = tmp_path / "technical"
    runs_root = tmp_path / "runs"
    argv = [
        "materialize_r3_dag.py",
        "init-teacher-preparation",
        "--technical-run-root",
        str(technical_root),
        "--runs-root",
        str(runs_root),
        "--run-name",
        "teacher-unit",
        "--preregistration",
        str(tmp_path / "prereg.json"),
        "--set8-train",
        str(tmp_path / "set8.jsonl"),
        "--transition16-train",
        str(tmp_path / "transition16.jsonl"),
        "--transition16-gate",
        str(tmp_path / "gate.jsonl"),
        "--transition16-raw-sidecar",
        str(tmp_path / "raw.jsonl"),
        "--set8-cache",
        str(tmp_path / "set8-cache"),
        "--transition16-cache",
        str(tmp_path / "transition16-cache"),
        "--font",
        str(tmp_path / "font.ttf"),
    ]
    with patch.object(sys, "argv", argv), patch(
        "materialize_r3_dag._load_verified_plan",
        return_value=({"expected_commit": "a" * 40}, Path("plan"), "b" * 64),
    ), patch(
        "materialize_r3_dag.initialize_teacher_preparation_dag",
        return_value={"dry_run": False},
    ) as initialize:
        assert materialize_main() == 0
    assert initialize.call_args.kwargs["teacher_run_root"] == runs_root.resolve() / "teacher-unit"


def test_teacher_assisted_policy_requires_parent_and_rejects_loose_evidence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires a completed teacher-preparation parent"):
        _validate_teacher_parent_policy(
            regime="teacher_assisted",
            teacher_preparation_run_root=None,
            loose_teacher_inputs={},
        )
    with pytest.raises(ValueError, match="rejects loose"):
        _validate_teacher_parent_policy(
            regime="teacher_assisted",
            teacher_preparation_run_root=tmp_path,
            loose_teacher_inputs={"teacher_t0": tmp_path / "T0.json"},
        )
    _validate_teacher_parent_policy(
        regime="teacher_assisted",
        teacher_preparation_run_root=tmp_path,
        loose_teacher_inputs={},
    )


def test_micro_run_root_must_be_a_strict_formal_run_child(tmp_path: Path) -> None:
    approved = (tmp_path / "formal-runs").resolve()
    approved.mkdir()
    child = approved / "r3-set8-qa"
    assert _require_run_child(child, approved_root=approved, label="micro_run_root") == child
    with pytest.raises(ValueError, match="strict child"):
        _require_run_child(approved, approved_root=approved, label="micro_run_root")
    with pytest.raises(ValueError, match="strict child"):
        _require_run_child(tmp_path / "outside", approved_root=approved, label="micro_run_root")


def test_bound_runner_verifies_launcher_and_materializes_output_evidence(tmp_path: Path) -> None:
    repo, commit = _make_repo(tmp_path)
    run_root = tmp_path / "runs" / "unit-runner"
    run_dir = run_root / "stages" / "00-unit"
    spec_path = run_root / "authorizations" / "00-unit.json"
    evidence_path = run_root / "evidence" / "00-unit.json"
    output = run_root / "results" / "unit.json"
    preflight = tmp_path / "preflight.json"
    preflight_sha = _write_preflight(preflight, commit)
    command = [
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(output)!r}).parent.mkdir(parents=True); "
        f"Path({str(output)!r}).write_text('\\u007b\"passed\":true\\u007d\\n')",
    ]
    definition = {
        "index": 0,
        "slug": "00-unit",
        "launcher_stage": "unit-stage",
        "dependency": None,
        "run_dir": str(run_dir),
        "evidence_path": str(evidence_path),
        "commands": [command],
        "outputs": [{"label": "unit", "path": str(output), "required_values": {"passed": True}}],
    }
    plan = {
        "schema_version": 1,
        "protocol": PLAN_PROTOCOL,
        "kind": "unit",
        "expected_commit": commit,
        "repo": str(repo),
        "python": sys.executable,
        "run_root": str(run_root),
        "formal_preflight": {"path": str(preflight), "sha256": preflight_sha},
        "strict_order": ["UNIT"],
        "stages": {"UNIT": definition},
    }
    plan_path = run_root / "dag_plan.json"
    plan_sha = atomic_json(plan_path, plan)
    spec = {
        "schema_version": 1,
        "protocol": STAGE_SPEC_PROTOCOL,
        "plan_path": str(plan_path),
        "plan_sha256": plan_sha,
        "stage": "UNIT",
        "stage_index": 0,
        "stage_slug": "00-unit",
        "launcher_stage": "unit-stage",
        "run_root": str(run_root),
        "run_dir": str(run_dir),
        "repo": str(repo),
        "expected_commit": commit,
        "formal_preflight": {"path": str(preflight), "sha256": preflight_sha},
        "prerequisites": [],
        "commands": [command],
        "outputs": definition["outputs"],
        "evidence_path": str(evidence_path),
    }
    spec_sha = atomic_json(spec_path, spec)
    run_dir.mkdir(parents=True)
    runner_command = [
        sys.executable,
        str(repo / "scripts" / "inspire" / "run_r3_stage.py"),
        "--spec",
        str(spec_path),
        "--spec-sha256",
        spec_sha,
    ]
    worker_input = run_dir / "worker_input.json"
    configuration_sha = atomic_json(
        worker_input,
        {
            "stage": "unit-stage",
            "repo": str(repo),
            "run_root": str(run_root),
            "run_dir": str(run_dir),
            "expected_commit": commit,
            "preflight": str(preflight),
            "preflight_sha256": preflight_sha,
            "infrastructure_stage": False,
            "strict_environment": STRICT_ENVIRONMENT,
            "command": runner_command,
        },
    )
    environment = {
        "VLM_STAGE_WORKER_INPUT": str(worker_input),
        "VLM_STAGE_CONFIGURATION_SHA256": configuration_sha,
        "VLM_STAGE_PREFLIGHT": str(preflight),
        "VLM_STAGE_PREFLIGHT_SHA256": preflight_sha,
    }
    with patch.dict(os.environ, environment, clear=False):
        report = run_bound_stage(spec_path, spec_sha)
    assert report["passed"] is True
    assert report["configuration_sha256"] == configuration_sha
    assert report["outputs"][0]["sha256"] == sha256_file(output)
    assert sha256_file(evidence_path) == report["evidence_sha256"]


@pytest.mark.parametrize("suite", ["set8", "transition16"])
def test_micro_extension_contract_supports_both_locked_suites(tmp_path: Path, suite: str) -> None:
    payload = _qa_micro_contract_payload(tmp_path, suite)
    contract = tmp_path / f"{suite}.json"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with patch(
        "materialize_r3_dag._preregistered_micro_data_binding",
        return_value=payload["data_binding"],
    ):
        loaded = _load_micro_command_contract(contract)
    assert loaded["suite"] == suite
    expected_shape = "single" if suite == "set8" else "paired-replica"
    assert loaded["execution_shape"] == expected_shape
    assert [arm["arm_id"] for arm in loaded["arms"]] == (
        ["A"] if suite == "set8" else ["A", "B"]
    )


def test_transition_micro_rejects_replica_b_starting_before_a_score(tmp_path: Path) -> None:
    payload = _qa_micro_contract_payload(tmp_path, "transition16")
    # Valid order is A train/eval/score, then fresh B train/eval/score.
    payload["commands"][2], payload["commands"][3] = (
        payload["commands"][3],
        payload["commands"][2],
    )
    contract = tmp_path / "transition-b-before-a-score.json"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with patch(
        "materialize_r3_dag._preregistered_micro_data_binding",
        return_value=payload["data_binding"],
    ), pytest.raises(ValueError, match="arms must execute serially"):
        _load_micro_command_contract(contract)


def test_transition_micro_rejects_replication_of_wrong_gate_report(tmp_path: Path) -> None:
    payload = _qa_micro_contract_payload(tmp_path, "transition16")
    replication = payload["commands"][-1]
    replication[replication.index("--a") + 1] = str(
        Path(payload["arms"][1]["qa_output_dir"]) / "gate_report.json"
    )
    contract = tmp_path / "transition-wrong-replication-input.json"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with patch(
        "materialize_r3_dag._preregistered_micro_data_binding",
        return_value=payload["data_binding"],
    ), pytest.raises(ValueError, match="does not consume this stage's A/B gate reports"):
        _load_micro_command_contract(contract)


def test_teacher_transition_accepts_composite_set8_attribution_parent() -> None:
    parent = {
        "suite": "set8",
        "training_regime": "teacher_assisted",
        "teacher_control": "composite",
        "outputs": [
            {
                "label": "teacher-attribution",
                "kind": "scientific_report",
                "required_values": {"passed": True, "artifact_provenance_validated": True},
            }
        ],
    }
    child = {
        "suite": "transition16",
        "training_regime": "teacher_assisted",
        "teacher_control": "correct",
    }
    _validate_set8_parent_role(
        set8_contract=parent,
        transition_contract=child,
        regime="teacher_assisted",
    )

    parent["outputs"] = []
    with pytest.raises(ValueError, match="composite Set8 attribution"):
        _validate_set8_parent_role(
            set8_contract=parent,
            transition_contract=child,
            regime="teacher_assisted",
        )


def test_transition_parent_must_share_exact_technical_and_teacher_dags() -> None:
    technical = {"run_root": "/runs/technical-a", "plan_sha256": "a" * 64}
    teacher = {"run_root": "/runs/teacher-a", "final_index_sha256": "b" * 64}
    parent = {
        "technical_parent": dict(technical),
        "teacher_preparation_parent": dict(teacher),
    }
    _validate_set8_parent_dag_identity(
        set8_parent=parent,
        technical_parent=technical,
        teacher_parent=teacher,
        regime="teacher_assisted",
    )
    parent["technical_parent"] = {**technical, "run_root": "/runs/technical-b"}
    with pytest.raises(ValueError, match="different technical DAG"):
        _validate_set8_parent_dag_identity(
            set8_parent=parent,
            technical_parent=technical,
            teacher_parent=teacher,
            regime="teacher_assisted",
        )


def test_qa_micro_command_parses_and_normalizes_teacher_control_to_none(tmp_path: Path) -> None:
    payload = _qa_micro_contract_payload(tmp_path, "set8")
    training_command = payload["commands"][0]
    with patch.object(sys, "argv", training_command[1:]):
        args = parse_training_args()
    assert args.teacher_control == "correct"
    assert training_lineage(args)["teacher_control"] == "none"


def test_micro_runtime_binding_rejects_other_python_or_model_root(tmp_path: Path) -> None:
    payload = _qa_micro_contract_payload(tmp_path, "set8")
    technical = {"python": sys.executable, "model_root": str((tmp_path / "models").resolve())}
    _validate_micro_runtime_bindings(payload, technical)

    drifted_python = json.loads(json.dumps(payload))
    drifted_python["commands"][0][0] = str((tmp_path / "other-python").resolve())
    with pytest.raises(ValueError, match="Python differs"):
        _validate_micro_runtime_bindings(drifted_python, technical)

    drifted_model = json.loads(json.dumps(payload))
    train_command = drifted_model["commands"][0]
    train_command[train_command.index("--reader") + 1] = str((tmp_path / "other-reader").resolve())
    with pytest.raises(ValueError, match="model paths differ"):
        _validate_micro_runtime_bindings(drifted_model, technical)


def test_venv_python_symlink_identity_is_not_dereferenced(tmp_path: Path) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    linked_python = venv_bin / "python"
    try:
        linked_python.symlink_to(Path(sys.executable))
    except OSError as exc:  # Windows without developer-mode symlink permission.
        pytest.skip(f"symlink creation is unavailable: {exc}")
    normalized = require_absolute_executable(linked_python, "python")
    assert normalized == linked_python.absolute()
    assert normalized != linked_python.resolve()


def test_micro_contract_rejects_teacher_smuggling_and_unbound_outputs(tmp_path: Path) -> None:
    contract_path = tmp_path / "qa.json"
    payload = _qa_micro_contract_payload(tmp_path, "set8")
    payload["commands"][0].extend(("--teacher-manifest", str(tmp_path / "teacher.json")))
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    with patch(
        "materialize_r3_dag._preregistered_micro_data_binding", return_value=payload["data_binding"]
    ), pytest.raises(ValueError, match="must not receive teacher inputs"):
        _load_micro_command_contract(contract_path)

    payload = _qa_micro_contract_payload(tmp_path, "set8")
    payload["outputs"][-1]["required_values"] = {}
    contract_path.write_text(json.dumps(payload), encoding="utf-8")
    with patch(
        "materialize_r3_dag._preregistered_micro_data_binding", return_value=payload["data_binding"]
    ), pytest.raises(ValueError, match="scientific reports"):
        _load_micro_command_contract(contract_path)


def test_micro_materialization_refuses_incomplete_technical_chain(tmp_path: Path) -> None:
    _, context = _technical_fixture(tmp_path, through="G6-L")
    contract = tmp_path / "set8.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "protocol": MICRO_COMMAND_PROTOCOL,
                "stage": "qa8-a",
                "suite": "set8",
                "training_regime": "qa_only",
                "teacher_control": "none",
                "fixed_protocol": FIXED_MICRO_PROTOCOL,
                "commands": _qa_micro_contract_payload(tmp_path, "set8")["commands"],
                "outputs": _qa_micro_contract_payload(tmp_path, "set8")["outputs"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete six-stage technical plan"):
        initialize_micro_extension(
            technical_run_root=context["run_root"],
            micro_run_root=tmp_path / "runs" / "micro",
            command_contract_path=contract,
        )


def test_qa_micro_adapter_explicitly_tracks_current_calibration_signature() -> None:
    report = _validate_micro_prerequisites(
        resize={},
        scorer={},
        technical={},
        teacher_t0=None,
        teacher_calibration_report=None,
        teacher_calibration_file_sha256=None,
        teacher_tc0=None,
        teacher_tc0_file_sha256=None,
        teacher_tf0=None,
        teacher_tf0_file_sha256=None,
        training_regime="qa_only",
        expected_commit="a" * 40,
    )
    assert report["training_regime"] == "qa_only"
    assert report["teacher_t0_required"] is False
    assert report["teacher_calibration_complete"] is None
