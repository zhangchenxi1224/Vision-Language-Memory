from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from r3_dag_contract import MICRO_COMMAND_PROTOCOL, atomic_json, sha256_file  # noqa: E402


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FIXED_PROTOCOL = {
    "reader_loss_mode": "listwise-choice",
    "train_choice_family": "cyclic4",
    "gate_choice_family": "reverse-cyclic4",
    "dreamlite": "DreamLite-mobile-4-step",
    "lora_rank": 4,
    "strict_determinism": True,
}
TEACHER_CACHE_FILES = {
    "manifest": "manifest.json",
    "sidecar": "transitions.jsonl",
}


@dataclass(frozen=True)
class RenderInputs:
    repo: Path
    python: Path
    model_root: Path
    run_root: Path
    suite: str
    training_regime: str
    train: Path
    gate: Path
    expected_commit: str
    reader_revision: str
    dreamlite_revision: str
    teacher_cache: Path | None = None
    teacher_calibration: Path | None = None

    @property
    def episodes(self) -> int:
        return 8 if self.suite == "set8" else 16

    @property
    def step_per_presentation(self) -> int:
        return self.episodes // 8

    @property
    def eval_start_step(self) -> int:
        return 64 * self.step_per_presentation

    @property
    def eval_every(self) -> int:
        return 32 * self.step_per_presentation

    @property
    def dreamlite(self) -> Path:
        return self.model_root / "DreamLite-mobile"

    @property
    def reader(self) -> Path:
        return self.model_root / "Qwen3-VL-4B-Instruct"

    @property
    def preregistration(self) -> Path:
        return self.repo / "configs" / "experiments" / "r3_preregistration.json"


@dataclass(frozen=True)
class Arm:
    arm_id: str
    teacher_control: str
    qa_output_dir: Path
    distill_output_dir: Path | None


def _absolute(path: Path, name: str) -> Path:
    value = path.expanduser().resolve()
    if not value.is_absolute():  # pragma: no cover - resolve is absolute on supported platforms.
        raise ValueError(f"{name} must be absolute")
    return value


def _absolute_executable(path: Path) -> Path:
    """Make an executable path absolute without dereferencing a venv symlink."""

    return Path(os.path.abspath(path.expanduser()))


def _require_file(path: Path, name: str) -> Path:
    value = _absolute(path, name)
    if not value.is_file():
        raise ValueError(f"{name} is not a file: {value}")
    return value


def _require_directory(path: Path, name: str) -> Path:
    value = _absolute(path, name)
    if not value.is_dir():
        raise ValueError(f"{name} is not a directory: {value}")
    return value


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _verify_cli_repository(inputs: RenderInputs) -> None:
    if _git(inputs.repo, "rev-parse", "HEAD") != inputs.expected_commit:
        raise ValueError("--expected-commit differs from the repository HEAD")
    if _git(inputs.repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("R3 production contract rendering requires a clean repository")


def _locked_revision(path: Path, *, expected: str, name: str) -> None:
    marker = _require_file(path / ".locked_revision", f"{name} revision marker")
    observed = marker.read_text(encoding="utf-8").strip()
    if observed != expected:
        raise ValueError(f"{name} revision marker differs from the explicit lineage binding")


def _validate_inputs(inputs: RenderInputs) -> dict[str, Any]:
    if inputs.suite not in {"set8", "transition16"}:
        raise ValueError("suite must be set8 or transition16")
    if inputs.training_regime not in {"qa_only", "teacher_assisted"}:
        raise ValueError("training_regime must be qa_only or teacher_assisted")
    if inputs.run_root.exists():
        raise ValueError("the immutable micro run root must not exist before materialization")
    for label, value in (
        ("expected_commit", inputs.expected_commit),
        ("reader_revision", inputs.reader_revision),
        ("dreamlite_revision", inputs.dreamlite_revision),
    ):
        if COMMIT_PATTERN.fullmatch(value) is None:
            raise ValueError(f"{label} must be a lowercase 40-character Git revision")

    _require_directory(inputs.repo, "repo")
    if inputs.repo.resolve() != ROOT.resolve():
        raise ValueError("--repo must be the same checkout that owns this renderer")
    _require_file(inputs.python, "python")
    train = _require_file(inputs.train, "train")
    gate = _require_file(inputs.gate, "gate")
    preregistration = _require_file(inputs.preregistration, "R3 preregistration")
    for relative in (
        "scripts/train/dreamlite_episode.py",
        "scripts/eval/dreamlite_mcq.py",
        "scripts/eval/score_r3_micro.py",
        "scripts/eval/teacher_state_retrieval.py",
        "scripts/eval/score_r3_teacher_attribution.py",
        "scripts/probes/validate_r3_micro_replication.py",
    ):
        _require_file(inputs.repo / relative, relative)
    _require_directory(inputs.model_root, "model_root")
    _locked_revision(inputs.reader, expected=inputs.reader_revision, name="Reader")
    _locked_revision(inputs.dreamlite, expected=inputs.dreamlite_revision, name="DreamLite")

    preregistered = json.loads(preregistration.read_text(encoding="utf-8"))
    try:
        suite_lock = preregistered["micro_data"][inputs.suite]
        expected_manifest_sha = str(suite_lock["suite_manifest_sha256"])
        expected_train_sha = str(suite_lock["train_sha256"])
        expected_gate_sha = str(suite_lock["gate_sha256"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"R3 preregistration lacks the {inputs.suite} data lock") from exc
    train_sha = sha256_file(train)
    gate_sha = sha256_file(gate)
    suite_manifest = _require_file(
        train.parent / f"{inputs.suite}_manifest.json",
        f"{inputs.suite} suite manifest",
    )
    if gate.parent != train.parent:
        raise ValueError("train, gate, and suite manifest must share one immutable data directory")
    suite_manifest_sha = sha256_file(suite_manifest)
    if (
        suite_manifest_sha != expected_manifest_sha
        or train_sha != expected_train_sha
        or gate_sha != expected_gate_sha
    ):
        raise ValueError("suite manifest/train/gate files differ from the preregistered micro-data lock")
    manifest_value = json.loads(suite_manifest.read_text(encoding="utf-8"))
    manifest_artifacts = manifest_value.get("artifacts")
    expected_artifacts = {
        train.name: train_sha,
        gate.name: gate_sha,
    }
    if not isinstance(manifest_artifacts, dict) or any(
        not isinstance(manifest_artifacts.get(name), dict)
        or manifest_artifacts[name].get("sha256") != expected_sha
        for name, expected_sha in expected_artifacts.items()
    ):
        raise ValueError("suite manifest does not bind the selected train/gate artifacts")

    teacher_binding = None
    teacher_files = None
    if inputs.training_regime == "teacher_assisted":
        if inputs.teacher_cache is None:
            raise ValueError("teacher_assisted rendering requires --teacher-cache")
        cache = _require_directory(inputs.teacher_cache, "teacher_cache")
        teacher_files = {
            label: _require_file(cache / filename, f"teacher {label}")
            for label, filename in TEACHER_CACHE_FILES.items()
        }
        if inputs.teacher_calibration is None:
            raise ValueError("teacher_assisted rendering requires --teacher-calibration")
        teacher_files["calibration"] = _require_file(
            inputs.teacher_calibration,
            "teacher calibration",
        )
        teacher_binding = {
            "suite": inputs.suite,
            "preregistration_sha256": sha256_file(preregistration),
            "train_sha256": train_sha,
            "manifest_sha256": sha256_file(teacher_files["manifest"]),
            "sidecar_sha256": sha256_file(teacher_files["sidecar"]),
            "calibration_sha256": sha256_file(teacher_files["calibration"]),
        }
    elif inputs.teacher_cache is not None or inputs.teacher_calibration is not None:
        raise ValueError("qa_only rendering forbids --teacher-cache and --teacher-calibration")

    return {
        "data_binding": {
            "preregistration_sha256": sha256_file(preregistration),
            "suite_manifest_sha256": suite_manifest_sha,
            "train_sha256": train_sha,
            "gate_sha256": gate_sha,
        },
        "teacher_binding": teacher_binding,
        "teacher_files": teacher_files,
    }


def _arm_specs(inputs: RenderInputs) -> tuple[str, str, list[Arm]]:
    root = inputs.run_root / "arms"
    if inputs.suite == "set8" and inputs.training_regime == "qa_only":
        values = [("A", "none")]
        shape, top_control = "single", "none"
    elif inputs.suite == "transition16" and inputs.training_regime == "qa_only":
        values = [("A", "none"), ("B", "none")]
        shape, top_control = "paired-replica", "none"
    elif inputs.suite == "set8":
        values = [
            ("correct", "correct"),
            ("shuffled", "shuffled"),
            ("random", "random-moment-matched"),
        ]
        shape, top_control = "teacher-control-composite", "composite"
    else:
        values = [("A", "correct"), ("B", "correct")]
        shape, top_control = "paired-replica", "correct"
    return (
        shape,
        top_control,
        [
            Arm(
                arm_id=arm_id,
                teacher_control=control,
                qa_output_dir=root / arm_id / "qa",
                distill_output_dir=(
                    root / arm_id / "distill" if inputs.training_regime == "teacher_assisted" else None
                ),
            )
            for arm_id, control in values
        ],
    )


def _optimizer_steps(inputs: RenderInputs, presentations: int) -> int:
    return presentations * inputs.episodes // 8


def _final_checkpoint(inputs: RenderInputs, output_dir: Path, presentations: int) -> Path:
    return output_dir / f"checkpoint-{_optimizer_steps(inputs, presentations):06d}.pt"


def _training_command(
    inputs: RenderInputs,
    *,
    output_dir: Path,
    objective: str,
    teacher_control: str,
    teacher_files: dict[str, Path] | None,
    initialize_from: Path | None = None,
) -> list[str]:
    qa_only = inputs.training_regime == "qa_only"
    presentations = 512 if qa_only else 256
    distill_presentations = 0 if qa_only else 256
    qa_presentations = 512 if qa_only else (0 if objective == "distill" else 256)
    command = [
        str(inputs.python),
        str(inputs.repo / "scripts" / "train" / "dreamlite_episode.py"),
        "--train",
        str(inputs.train),
        "--dev",
        str(inputs.gate),
        "--dataset-format",
        "synthetic",
        "--dreamlite",
        str(inputs.dreamlite),
        "--reader",
        str(inputs.reader),
        "--reader-loss-mode",
        "listwise-choice",
        "--choice-view-schedule",
        "cyclic4",
        "--training-regime",
        inputs.training_regime,
        "--objective-stage",
        objective,
        "--teacher-control",
        "correct" if qa_only else teacher_control,
        "--presentations-per-state",
        str(presentations),
        "--distill-presentations",
        str(distill_presentations),
        "--qa-presentations",
        str(qa_presentations),
        "--initial-state-mode",
        "blank",
        "--output-dir",
        str(output_dir),
        "--resolution",
        "1024",
        "--seed",
        "0",
        "--adapter-seed",
        "0",
        "--learning-rate",
        "1e-4",
        "--weight-decay",
        "0.01",
        "--lora-rank",
        "4",
        "--epochs",
        str(presentations),
        "--gradient-accumulation",
        "8",
        "--gradient-clip",
        "1.0",
        "--checkpoint-every",
        str(inputs.eval_every),
        "--eval-start-step",
        str(inputs.eval_start_step),
        "--eval-every",
        str(inputs.eval_every),
        "--eval-limit",
        str(inputs.episodes),
        "--early-stopping-patience",
        "100000",
        "--disable-early-stopping",
        "--max-train-episodes",
        str(inputs.episodes),
        "--recurrence-mode",
        "direct_latent",
        "--noop-policy",
        "update",
        "--curriculum",
        "full",
        "--checkpoint-unet",
        "--audit-state-gradients",
        "--require-mixed-delayed-probe",
        "--strict-determinism",
        "--dreamlite-device",
        "cuda:0",
        "--reader-device",
        "cuda:1",
    ]
    if objective == "distill":
        if teacher_files is None:
            raise ValueError("distillation command requires teacher files")
        command.extend(
            [
                "--teacher-manifest",
                str(teacher_files["manifest"]),
                "--teacher-sidecar",
                str(teacher_files["sidecar"]),
                "--teacher-calibration",
                str(teacher_files["calibration"]),
            ]
        )
    if initialize_from is not None:
        command.extend(["--initialize-from", str(initialize_from)])
    return command


def _evaluation_command(
    inputs: RenderInputs,
    *,
    checkpoint: Path,
    predictions: Path,
    arm: Arm,
) -> list[str]:
    conditions = ["standard", "reset", "shuffle"]
    if inputs.suite == "transition16":
        conditions.append("state_swap")
    return [
        str(inputs.python),
        str(inputs.repo / "scripts" / "eval" / "dreamlite_mcq.py"),
        "--episodes",
        str(inputs.gate),
        "--format",
        "synthetic",
        "--dreamlite",
        str(inputs.dreamlite),
        "--reader",
        str(inputs.reader),
        "--checkpoint",
        str(checkpoint),
        "--expected-training-regime",
        inputs.training_regime,
        "--output",
        str(predictions),
        "--method",
        f"r3-{inputs.suite}-{inputs.training_regime}-{arm.arm_id}",
        "--conditions",
        *conditions,
        "--noop-policy",
        "keep",
        "--recurrence-mode",
        "direct_latent",
        "--seed",
        "0",
        "--training-seed",
        "0",
        "--adapter-seed",
        "0",
        "--lora-rank",
        "4",
        "--resolution",
        "1024",
        "--initial-state-mode",
        "blank",
        "--choice-view-family",
        "reverse-cyclic4",
        "--strict-determinism",
        "--dreamlite-device",
        "cuda:0",
        "--reader-device",
        "cuda:1",
    ]


def _score_command(
    inputs: RenderInputs,
    *,
    predictions: Path,
    gate_report: Path,
    allow_scientific_failure: bool,
    data_binding: dict[str, str],
) -> list[str]:
    command = [
        str(inputs.python),
        str(inputs.repo / "scripts" / "eval" / "score_r3_micro.py"),
        "--predictions",
        str(predictions),
        "--prediction-report",
        str(predictions.with_suffix(predictions.suffix + ".report.json")),
        "--suite",
        inputs.suite,
        "--expected-git-commit",
        inputs.expected_commit,
        "--expected-reader-revision",
        inputs.reader_revision,
        "--expected-dreamlite-revision",
        inputs.dreamlite_revision,
        "--expected-train-sha256",
        data_binding["train_sha256"],
        "--expected-dev-sha256",
        data_binding["gate_sha256"],
        "--output",
        str(gate_report),
    ]
    if allow_scientific_failure:
        command.append("--no-fail-on-gate")
    return command


def _retrieval_command(
    inputs: RenderInputs,
    *,
    checkpoint: Path,
    output: Path,
    arm: Arm,
    distill_reference: Path | None,
    fail_on_gate: bool,
) -> list[str]:
    if inputs.teacher_cache is None:
        raise ValueError("teacher retrieval requires a teacher cache")
    command = [
        str(inputs.python),
        str(inputs.repo / "scripts" / "eval" / "teacher_state_retrieval.py"),
        "--episodes",
        str(inputs.train),
        "--teacher-cache",
        str(inputs.teacher_cache),
        "--teacher-calibration",
        str(inputs.teacher_calibration),
        "--dreamlite",
        str(inputs.dreamlite),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        "cuda:0",
        "--expected-episodes",
        str(inputs.episodes),
        "--minimum-correct",
        "7" if inputs.suite == "set8" else "0",
        "--expected-teacher-control",
        arm.teacher_control,
    ]
    if distill_reference is not None:
        command.extend(
            [
                "--distill-reference-report",
                str(distill_reference),
                "--minimum-retention",
                "0.9",
            ]
        )
    if not fail_on_gate:
        command.append("--no-fail-on-gate")
    return command


def _artifact(label: str, path: Path) -> dict[str, Any]:
    return {"kind": "artifact", "label": label, "path": str(path), "required_values": {}}


def _scientific_report(label: str, path: Path, inputs: RenderInputs) -> dict[str, Any]:
    return {
        "kind": "scientific_report",
        "label": label,
        "path": str(path),
        "required_values": {
            "passed": True,
            "suite": inputs.suite,
            "training_regime": inputs.training_regime,
            "artifact_provenance_validated": True,
        },
    }


def _training_outputs(
    inputs: RenderInputs,
    *,
    arm: Arm,
    objective: str,
    output_dir: Path,
    presentations: int,
) -> list[dict[str, Any]]:
    prefix = f"{arm.arm_id}-{objective}"
    final_step = _optimizer_steps(inputs, presentations)
    values = [
        _artifact(f"{prefix}-manifest", output_dir / "manifest.json"),
        _artifact(f"{prefix}-environment", output_dir / "environment.txt"),
        _artifact(f"{prefix}-curriculum", output_dir / "curriculum.json"),
        _artifact(f"{prefix}-metrics", output_dir / "metrics.jsonl"),
        _artifact(f"{prefix}-state-gradient-audit", output_dir / "state_gradient_audit.json"),
        _artifact(f"{prefix}-summary", output_dir / "summary.json"),
        _artifact(f"{prefix}-last", output_dir / "last.pt"),
    ]
    if objective == "distill":
        values.append(_artifact(f"{prefix}-diagnostics", output_dir / "distill_diagnostics.json"))
    for step in range(inputs.eval_every, final_step + 1, inputs.eval_every):
        values.append(
            _artifact(
                f"{prefix}-checkpoint-{step:06d}",
                output_dir / f"checkpoint-{step:06d}.pt",
            )
        )
    return values


def _replication_command(
    inputs: RenderInputs,
    *,
    reports: dict[str, Path],
    output: Path,
) -> list[str]:
    return [
        str(inputs.python),
        str(inputs.repo / "scripts" / "probes" / "validate_r3_micro_replication.py"),
        "--a",
        str(reports["A"]),
        "--b",
        str(reports["B"]),
        "--suite",
        inputs.suite,
        "--training-regime",
        inputs.training_regime,
        "--teacher-control",
        "none" if inputs.training_regime == "qa_only" else "correct",
        "--output",
        str(output),
    ]


def _attribution_command(
    inputs: RenderInputs,
    *,
    artifacts: dict[str, dict[str, Path]],
    output: Path,
) -> list[str]:
    command = [
        str(inputs.python),
        str(inputs.repo / "scripts" / "eval" / "score_r3_teacher_attribution.py"),
    ]
    for cli_name, arm_id in (("correct", "correct"), ("shuffled", "shuffled"), ("random", "random")):
        arm = artifacts[arm_id]
        command.extend(
            [
                f"--{cli_name}-distill-summary",
                str(arm["distill_summary"]),
                f"--{cli_name}-qa-summary",
                str(arm["qa_summary"]),
                f"--{cli_name}-distill-retrieval",
                str(arm["distill_retrieval"]),
                f"--{cli_name}-qa-retrieval",
                str(arm["qa_retrieval"]),
                f"--{cli_name}-qa-gate",
                str(arm["gate_report"]),
            ]
        )
    command.extend(["--output", str(output)])
    return command


def _unique_outputs(outputs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(outputs)
    labels = [str(value["label"]) for value in values]
    paths = [str(value["path"]) for value in values]
    if len(labels) != len(set(labels)) or len(paths) != len(set(paths)):
        raise RuntimeError("renderer generated duplicate output labels or paths")
    return values


def render_contract(inputs: RenderInputs) -> dict[str, Any]:
    inputs = RenderInputs(
        repo=_absolute(inputs.repo, "repo"),
        python=_absolute_executable(inputs.python),
        model_root=_absolute(inputs.model_root, "model_root"),
        run_root=_absolute(inputs.run_root, "run_root"),
        suite=inputs.suite,
        training_regime=inputs.training_regime,
        train=_absolute(inputs.train, "train"),
        gate=_absolute(inputs.gate, "gate"),
        expected_commit=inputs.expected_commit,
        reader_revision=inputs.reader_revision,
        dreamlite_revision=inputs.dreamlite_revision,
        teacher_cache=(
            None if inputs.teacher_cache is None else _absolute(inputs.teacher_cache, "teacher_cache")
        ),
        teacher_calibration=(
            None
            if inputs.teacher_calibration is None
            else _absolute(inputs.teacher_calibration, "teacher_calibration")
        ),
    )
    validated = _validate_inputs(inputs)
    data_binding: dict[str, str] = validated["data_binding"]
    teacher_files: dict[str, Path] | None = validated["teacher_files"]
    shape, top_control, arms = _arm_specs(inputs)
    commands: list[list[str]] = []
    outputs: list[dict[str, Any]] = []
    gate_reports: dict[str, Path] = {}
    attribution_artifacts: dict[str, dict[str, Path]] = {}

    for arm in arms:
        distill_checkpoint = None
        distill_retrieval = None
        if inputs.training_regime == "teacher_assisted":
            assert arm.distill_output_dir is not None
            distill_checkpoint = _final_checkpoint(inputs, arm.distill_output_dir, 256)
            distill_retrieval = arm.distill_output_dir.parent / "distill_retrieval.json"
            commands.append(
                _training_command(
                    inputs,
                    output_dir=arm.distill_output_dir,
                    objective="distill",
                    teacher_control=arm.teacher_control,
                    teacher_files=teacher_files,
                )
            )
            outputs.extend(
                _training_outputs(
                    inputs,
                    arm=arm,
                    objective="distill",
                    output_dir=arm.distill_output_dir,
                    presentations=256,
                )
            )
            retrieval_is_gate = inputs.suite == "set8" and arm.teacher_control == "correct"
            commands.append(
                _retrieval_command(
                    inputs,
                    checkpoint=distill_checkpoint,
                    output=distill_retrieval,
                    arm=arm,
                    distill_reference=None,
                    fail_on_gate=retrieval_is_gate,
                )
            )
            outputs.append(_artifact(f"{arm.arm_id}-distill-retrieval", distill_retrieval))

        qa_presentations = 512 if inputs.training_regime == "qa_only" else 256
        qa_checkpoint = _final_checkpoint(inputs, arm.qa_output_dir, qa_presentations)
        commands.append(
            _training_command(
                inputs,
                output_dir=arm.qa_output_dir,
                objective="qa",
                teacher_control=arm.teacher_control,
                teacher_files=None,
                initialize_from=distill_checkpoint,
            )
        )
        outputs.extend(
            _training_outputs(
                inputs,
                arm=arm,
                objective="qa",
                output_dir=arm.qa_output_dir,
                presentations=qa_presentations,
            )
        )

        qa_retrieval = None
        if inputs.training_regime == "teacher_assisted":
            assert distill_retrieval is not None
            qa_retrieval = arm.qa_output_dir.parent / "qa_retrieval.json"
            retrieval_is_gate = inputs.suite == "set8" and arm.teacher_control == "correct"
            commands.append(
                _retrieval_command(
                    inputs,
                    checkpoint=qa_checkpoint,
                    output=qa_retrieval,
                    arm=arm,
                    distill_reference=(distill_retrieval if arm.teacher_control == "correct" else None),
                    fail_on_gate=retrieval_is_gate,
                )
            )
            outputs.append(_artifact(f"{arm.arm_id}-qa-retrieval", qa_retrieval))

        predictions = arm.qa_output_dir / "gate_predictions.jsonl"
        prediction_report = predictions.with_suffix(predictions.suffix + ".report.json")
        gate_report = arm.qa_output_dir / "gate_report.json"
        gate_reports[arm.arm_id] = gate_report
        commands.append(
            _evaluation_command(
                inputs,
                checkpoint=qa_checkpoint,
                predictions=predictions,
                arm=arm,
            )
        )
        allow_failure = (
            inputs.suite == "set8"
            and inputs.training_regime == "teacher_assisted"
            and arm.teacher_control != "correct"
        )
        commands.append(
            _score_command(
                inputs,
                predictions=predictions,
                gate_report=gate_report,
                allow_scientific_failure=allow_failure,
                data_binding=data_binding,
            )
        )
        outputs.extend(
            [
                _artifact(f"{arm.arm_id}-predictions", predictions),
                _artifact(f"{arm.arm_id}-prediction-report", prediction_report),
                (
                    _scientific_report(f"{arm.arm_id}-gate", gate_report, inputs)
                    if inputs.suite == "set8" and inputs.training_regime == "qa_only"
                    else _artifact(f"{arm.arm_id}-gate", gate_report)
                ),
            ]
        )
        if inputs.training_regime == "teacher_assisted":
            assert arm.distill_output_dir is not None
            assert distill_retrieval is not None and qa_retrieval is not None
            attribution_artifacts[arm.arm_id] = {
                "distill_summary": arm.distill_output_dir / "summary.json",
                "qa_summary": arm.qa_output_dir / "summary.json",
                "distill_retrieval": distill_retrieval,
                "qa_retrieval": qa_retrieval,
                "gate_report": gate_report,
            }

    if inputs.suite == "transition16":
        replication = inputs.run_root / "results" / "replication.json"
        commands.append(_replication_command(inputs, reports=gate_reports, output=replication))
        outputs.append(_scientific_report("replication", replication, inputs))
    elif inputs.training_regime == "teacher_assisted":
        attribution = inputs.run_root / "results" / "teacher_attribution.json"
        commands.append(
            _attribution_command(
                inputs,
                artifacts=attribution_artifacts,
                output=attribution,
            )
        )
        outputs.append(_scientific_report("teacher-attribution", attribution, inputs))

    return {
        "schema_version": 2,
        "protocol": MICRO_COMMAND_PROTOCOL,
        "stage": f"{inputs.suite}-{inputs.training_regime.replace('_', '-')}",
        "suite": inputs.suite,
        "training_regime": inputs.training_regime,
        "teacher_control": top_control,
        "execution_shape": shape,
        "arms": [
            {
                "arm_id": arm.arm_id,
                "teacher_control": arm.teacher_control,
                "qa_output_dir": str(arm.qa_output_dir),
                "distill_output_dir": (
                    None if arm.distill_output_dir is None else str(arm.distill_output_dir)
                ),
            }
            for arm in arms
        ],
        "data_binding": data_binding,
        "lineage_binding": {
            "git_commit": inputs.expected_commit,
            "reader_revision": inputs.reader_revision,
            "dreamlite_revision": inputs.dreamlite_revision,
        },
        "teacher_calibration_binding": validated["teacher_binding"],
        "fixed_protocol": dict(FIXED_PROTOCOL),
        "commands": commands,
        "outputs": _unique_outputs(outputs),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one immutable, arm-aware R3 Inspire micro command contract; never launch it"
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--suite", choices=("set8", "transition16"), required=True)
    parser.add_argument("--training-regime", choices=("qa_only", "teacher_assisted"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--reader-revision", required=True)
    parser.add_argument("--dreamlite-revision", required=True)
    parser.add_argument("--teacher-cache", type=Path)
    parser.add_argument("--teacher-calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        inputs = RenderInputs(
            repo=args.repo,
            python=args.python,
            model_root=args.model_root,
            run_root=args.run_root,
            suite=args.suite,
            training_regime=args.training_regime,
            train=args.train,
            gate=args.gate,
            expected_commit=args.expected_commit,
            reader_revision=args.reader_revision,
            dreamlite_revision=args.dreamlite_revision,
            teacher_cache=args.teacher_cache,
            teacher_calibration=args.teacher_calibration,
        )
        normalized_output = _absolute(args.output, "output")
        if _is_within(normalized_output, _absolute(args.run_root, "run_root")):
            raise ValueError("--output must be outside --run-root so rendering cannot pre-create the immutable run")
        normalized_inputs = RenderInputs(
            **{
                **inputs.__dict__,
                "repo": _absolute(inputs.repo, "repo"),
                "python": _absolute_executable(inputs.python),
                "model_root": _absolute(inputs.model_root, "model_root"),
                "run_root": _absolute(inputs.run_root, "run_root"),
                "train": _absolute(inputs.train, "train"),
                "gate": _absolute(inputs.gate, "gate"),
                "teacher_cache": (
                    None
                    if inputs.teacher_cache is None
                    else _absolute(inputs.teacher_cache, "teacher_cache")
                ),
                "teacher_calibration": (
                    None
                    if inputs.teacher_calibration is None
                    else _absolute(inputs.teacher_calibration, "teacher_calibration")
                ),
            }
        )
        _verify_cli_repository(normalized_inputs)
        contract = render_contract(normalized_inputs)
        digest = atomic_json(normalized_output, contract)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(normalized_output),
                "sha256": digest,
                "suite": contract["suite"],
                "training_regime": contract["training_regime"],
                "execution_shape": contract["execution_shape"],
                "launched": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
