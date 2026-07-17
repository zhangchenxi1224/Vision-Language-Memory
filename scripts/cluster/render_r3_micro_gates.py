from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
TEACHER_CONTROLS = ("correct", "shuffled", "random-moment-matched")


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    train: Path
    gate: Path
    teacher_cache: Path
    train_sha256: str
    gate_sha256: str
    teacher_manifest_sha256: str
    teacher_sidecar_sha256: str
    teacher_calibration_sha256: str
    episodes: int

    def __post_init__(self) -> None:
        expected_episodes = {"set8": 8, "transition16": 16}
        if self.name not in expected_episodes or self.episodes != expected_episodes[self.name]:
            raise ValueError("R3 micro suites are locked to Set8=8 and Transition16=16 episodes.")
        for field in (
            "train_sha256",
            "gate_sha256",
            "teacher_manifest_sha256",
            "teacher_sidecar_sha256",
            "teacher_calibration_sha256",
        ):
            if _SHA256.fullmatch(str(getattr(self, field))) is None:
                raise ValueError(f"{self.name}.{field} must be a lowercase SHA256 digest.")

    @property
    def optimizer_steps_per_32_presentations(self) -> int:
        if self.episodes % 8:
            raise ValueError("R3 micro suites must be divisible by accumulation=8.")
        return 32 * (self.episodes // 8)

    @property
    def optimizer_steps_per_64_presentations(self) -> int:
        return 2 * self.optimizer_steps_per_32_presentations


@dataclass(frozen=True)
class MicroPaths:
    project: Path
    environment: Path
    model_root: Path
    run_root: Path
    resize_contract_report: Path
    scorer_s0_report: Path
    technical_report: Path
    teacher_t0_report: Path

    @property
    def results(self) -> Path:
        return self.run_root / "results"

    @property
    def logs(self) -> Path:
        return self.run_root / "logs"

    @property
    def sbatch(self) -> Path:
        return self.run_root / "sbatch"


@dataclass(frozen=True)
class MicroStage:
    name: str
    dependencies: tuple[str, ...]
    suite: str
    training_regime: str
    walltime: str
    commands: tuple[str, ...]


def shell_join(parts: Iterable[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha_contract(path: Path, digest: str) -> str:
    return f"test \"$(sha256sum {shlex.quote(str(path))} | awk '{{print $1}}')\" = {shlex.quote(digest)}"


def _suite_contract(spec: SuiteSpec, *, include_teacher: bool) -> tuple[str, ...]:
    data_contract = (
        _sha_contract(spec.train, spec.train_sha256),
        _sha_contract(spec.gate, spec.gate_sha256),
    )
    if not include_teacher:
        return data_contract
    return (
        *data_contract,
        _sha_contract(spec.teacher_cache / "manifest.json", spec.teacher_manifest_sha256),
        _sha_contract(spec.teacher_cache / "transitions.jsonl", spec.teacher_sidecar_sha256),
        _sha_contract(spec.teacher_cache / "calibration.json", spec.teacher_calibration_sha256),
    )


def _training_command(
    *,
    paths: MicroPaths,
    spec: SuiteSpec,
    output_dir: Path,
    training_regime: str,
    objective_stage: str,
    teacher_control: str = "correct",
    initialize_from: Path | None = None,
) -> str:
    if training_regime == "qa_only":
        epochs = 512
        distill_presentations = 0
        qa_presentations = 512
    elif training_regime == "teacher_assisted" and objective_stage == "distill":
        epochs = 256
        distill_presentations = 256
        qa_presentations = 0
    elif training_regime == "teacher_assisted" and objective_stage == "qa":
        epochs = 256
        distill_presentations = 256
        qa_presentations = 256
    else:
        raise ValueError("Unsupported R3 micro training stage.")

    parts: list[str | Path] = [
        "python",
        paths.project / "scripts" / "train" / "dreamlite_episode.py",
        "--train",
        spec.train,
        "--dev",
        spec.gate,
        "--dataset-format",
        "synthetic",
        "--dreamlite",
        paths.model_root / "DreamLite-mobile",
        "--reader",
        paths.model_root / "Qwen3-VL-4B-Instruct",
        "--reader-loss-mode",
        "listwise-choice",
        "--choice-view-schedule",
        "cyclic4",
        "--training-regime",
        training_regime,
        "--objective-stage",
        objective_stage,
        "--teacher-control",
        teacher_control,
        "--presentations-per-state",
        str(epochs),
        "--distill-presentations",
        str(distill_presentations),
        "--qa-presentations",
        str(qa_presentations),
        "--initial-state-mode",
        "blank",
        "--output-dir",
        output_dir,
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
        str(epochs),
        "--gradient-accumulation",
        "8",
        "--gradient-clip",
        "1.0",
        "--checkpoint-every",
        str(spec.optimizer_steps_per_32_presentations),
        "--eval-start-step",
        str(spec.optimizer_steps_per_64_presentations),
        "--eval-every",
        str(spec.optimizer_steps_per_32_presentations),
        "--eval-limit",
        str(spec.episodes),
        "--early-stopping-patience",
        "100000",
        "--disable-early-stopping",
        "--max-train-episodes",
        str(spec.episodes),
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
    if objective_stage == "distill":
        parts.extend(
            (
                "--teacher-manifest",
                spec.teacher_cache / "manifest.json",
                "--teacher-sidecar",
                spec.teacher_cache / "transitions.jsonl",
                "--teacher-calibration",
                spec.teacher_cache / "calibration.json",
            )
        )
    if initialize_from is not None:
        parts.extend(("--initialize-from", initialize_from))
    return shell_join(parts)


def _evaluation_commands(
    *,
    paths: MicroPaths,
    spec: SuiteSpec,
    checkpoint: Path,
    output_dir: Path,
    training_regime: str,
    method: str,
    fail_on_gate: bool = True,
) -> tuple[str, str]:
    predictions = output_dir / "gate_predictions.jsonl"
    conditions = ["standard", "reset", "shuffle"]
    if spec.name == "transition16":
        conditions.append("state_swap")
    evaluate = shell_join(
        [
            "python",
            paths.project / "scripts" / "eval" / "dreamlite_mcq.py",
            "--episodes",
            spec.gate,
            "--format",
            "synthetic",
            "--dreamlite",
            paths.model_root / "DreamLite-mobile",
            "--reader",
            paths.model_root / "Qwen3-VL-4B-Instruct",
            "--checkpoint",
            checkpoint,
            "--expected-training-regime",
            training_regime,
            "--output",
            predictions,
            "--method",
            method,
            "--conditions",
            *conditions,
            "--noop-policy",
            "keep",
            "--recurrence-mode",
            "direct_latent",
            "--seed",
            "0",
            "--choice-view-family",
            "reverse-cyclic4",
            "--dreamlite-device",
            "cuda:0",
            "--reader-device",
            "cuda:1",
        ]
    )
    score_parts: list[str | Path] = [
        "python",
        paths.project / "scripts" / "eval" / "score_r3_micro.py",
        "--predictions",
        predictions,
        "--prediction-report",
        predictions.with_suffix(predictions.suffix + ".report.json"),
        "--suite",
        spec.name,
        "--output",
        output_dir / "gate_report.json",
    ]
    if not fail_on_gate:
        score_parts.append("--no-fail-on-gate")
    return evaluate, shell_join(score_parts)


def _retrieval_command(
    *,
    paths: MicroPaths,
    spec: SuiteSpec,
    checkpoint: Path,
    output: Path,
    teacher_control: str,
    distill_reference: Path | None = None,
    fail_on_gate: bool,
) -> str:
    minimum = 7 if spec.name == "set8" else 0
    parts: list[str | Path] = [
        "python",
        paths.project / "scripts" / "eval" / "teacher_state_retrieval.py",
        "--episodes",
        spec.train,
        "--teacher-cache",
        spec.teacher_cache,
        "--dreamlite",
        paths.model_root / "DreamLite-mobile",
        "--checkpoint",
        checkpoint,
        "--output",
        output,
        "--device",
        "cuda:0",
        "--expected-episodes",
        str(spec.episodes),
        "--minimum-correct",
        str(minimum),
        "--expected-teacher-control",
        teacher_control,
    ]
    if distill_reference is not None:
        parts.extend(("--distill-reference-report", distill_reference, "--minimum-retention", "0.9"))
    if not fail_on_gate:
        parts.append("--no-fail-on-gate")
    return shell_join(parts)


def _qa_run_commands(
    *,
    paths: MicroPaths,
    spec: SuiteSpec,
    replica: str,
) -> tuple[str, ...]:
    output_dir = paths.run_root / spec.name / "qa_only" / replica / "qa"
    train = _training_command(
        paths=paths,
        spec=spec,
        output_dir=output_dir,
        training_regime="qa_only",
        objective_stage="qa",
    )
    evaluate, score = _evaluation_commands(
        paths=paths,
        spec=spec,
        checkpoint=output_dir / "last.pt",
        output_dir=output_dir,
        training_regime="qa_only",
        method=f"r3-{spec.name}-qa-only-{replica}",
    )
    return (*_suite_contract(spec, include_teacher=False), train, evaluate, score)


def _teacher_run_commands(
    *,
    paths: MicroPaths,
    spec: SuiteSpec,
    replica: str,
    teacher_control: str,
    fail_scientific_gate: bool,
) -> tuple[str, ...]:
    if teacher_control not in TEACHER_CONTROLS:
        raise ValueError(f"Unknown teacher control: {teacher_control}.")
    root = paths.run_root / spec.name / "teacher_assisted" / teacher_control / replica
    distill_dir = root / "distill"
    qa_dir = root / "qa"
    distill_retrieval = root / "distill_retrieval.json"
    qa_retrieval = root / "qa_retrieval.json"
    distill_train = _training_command(
        paths=paths,
        spec=spec,
        output_dir=distill_dir,
        training_regime="teacher_assisted",
        objective_stage="distill",
        teacher_control=teacher_control,
    )
    qa_train = _training_command(
        paths=paths,
        spec=spec,
        output_dir=qa_dir,
        training_regime="teacher_assisted",
        objective_stage="qa",
        teacher_control=teacher_control,
        initialize_from=distill_dir / "last.pt",
    )
    evaluate, score = _evaluation_commands(
        paths=paths,
        spec=spec,
        checkpoint=qa_dir / "last.pt",
        output_dir=qa_dir,
        training_regime="teacher_assisted",
        method=f"r3-{spec.name}-teacher-{teacher_control}-{replica}",
        fail_on_gate=fail_scientific_gate,
    )
    commands: list[str] = [*_suite_contract(spec, include_teacher=True), distill_train]
    if spec.name == "set8":
        commands.append(
            _retrieval_command(
                paths=paths,
                spec=spec,
                checkpoint=distill_dir / "last.pt",
                output=distill_retrieval,
                teacher_control=teacher_control,
                fail_on_gate=teacher_control == "correct",
            )
        )
    commands.append(qa_train)
    if spec.name == "set8":
        commands.append(
            _retrieval_command(
                paths=paths,
                spec=spec,
                checkpoint=qa_dir / "last.pt",
                output=qa_retrieval,
                teacher_control=teacher_control,
                distill_reference=distill_retrieval if teacher_control == "correct" else None,
                fail_on_gate=teacher_control == "correct",
            )
        )
    commands.extend((evaluate, score))
    return tuple(commands)


def _replication_command(
    *,
    paths: MicroPaths,
    spec: SuiteSpec,
    training_regime: str,
    teacher_control: str = "correct",
) -> str:
    if training_regime == "qa_only":
        base = paths.run_root / spec.name / training_regime
        a = base / "A" / "qa" / "gate_report.json"
        b = base / "B" / "qa" / "gate_report.json"
    else:
        base = paths.run_root / spec.name / training_regime / teacher_control
        a = base / "A" / "qa" / "gate_report.json"
        b = base / "B" / "qa" / "gate_report.json"
    return shell_join(
        [
            "python",
            paths.project / "scripts" / "probes" / "validate_r3_micro_replication.py",
            "--a",
            a,
            "--b",
            b,
            "--suite",
            spec.name,
            "--training-regime",
            training_regime,
            "--teacher-control",
            teacher_control,
            "--output",
            paths.results / f"{spec.name}_{training_regime}_{teacher_control}_A_B.json",
        ]
    )


def _teacher_attribution_command(paths: MicroPaths, spec: SuiteSpec) -> str:
    parts: list[str | Path] = [
        "python",
        paths.project / "scripts" / "eval" / "score_r3_teacher_attribution.py",
    ]
    labels = {
        "correct": "correct",
        "shuffled": "shuffled",
        "random": "random-moment-matched",
    }
    for cli_label, directory_label in labels.items():
        root = paths.run_root / spec.name / "teacher_assisted" / directory_label / "A"
        parts.extend(
            (
                f"--{cli_label}-distill-summary",
                root / "distill" / "summary.json",
                f"--{cli_label}-qa-summary",
                root / "qa" / "summary.json",
                f"--{cli_label}-distill-retrieval",
                root / "distill_retrieval.json",
                f"--{cli_label}-qa-retrieval",
                root / "qa_retrieval.json",
                f"--{cli_label}-qa-gate",
                root / "qa" / "gate_report.json",
            )
        )
    parts.extend(("--output", paths.results / f"{spec.name}_teacher_attribution_A.json"))
    return shell_join(parts)


def build_stages(paths: MicroPaths, set8: SuiteSpec, transition16: SuiteSpec) -> list[MicroStage]:
    """Build a fail-stop R3 micro DAG; no function in this module can submit it."""

    qa8_a = MicroStage(
        "QA8-A",
        (),
        "set8",
        "qa_only",
        "48:00:00",
        _qa_run_commands(paths=paths, spec=set8, replica="A"),
    )
    qa8_b = MicroStage(
        "QA8-B",
        ("QA8-A",),
        "set8",
        "qa_only",
        "48:00:00",
        (
            *_qa_run_commands(paths=paths, spec=set8, replica="B"),
            _replication_command(paths=paths, spec=set8, training_regime="qa_only"),
        ),
    )
    qa16_a = MicroStage(
        "QA16-A",
        ("QA8-B",),
        "transition16",
        "qa_only",
        "72:00:00",
        _qa_run_commands(paths=paths, spec=transition16, replica="A"),
    )
    qa16_b = MicroStage(
        "QA16-B",
        ("QA16-A",),
        "transition16",
        "qa_only",
        "72:00:00",
        (
            *_qa_run_commands(paths=paths, spec=transition16, replica="B"),
            _replication_command(paths=paths, spec=transition16, training_regime="qa_only"),
        ),
    )

    td8_correct_a = MicroStage(
        "TD8-CORRECT-A",
        (),
        "set8",
        "teacher_assisted",
        "72:00:00",
        _teacher_run_commands(
            paths=paths,
            spec=set8,
            replica="A",
            teacher_control="correct",
            fail_scientific_gate=True,
        ),
    )
    td8_shuffled_a = MicroStage(
        "TD8-SHUFFLED-A",
        ("TD8-CORRECT-A",),
        "set8",
        "teacher_assisted",
        "72:00:00",
        _teacher_run_commands(
            paths=paths,
            spec=set8,
            replica="A",
            teacher_control="shuffled",
            fail_scientific_gate=False,
        ),
    )
    td8_random_a = MicroStage(
        "TD8-RANDOM-A",
        ("TD8-CORRECT-A",),
        "set8",
        "teacher_assisted",
        "72:00:00",
        _teacher_run_commands(
            paths=paths,
            spec=set8,
            replica="A",
            teacher_control="random-moment-matched",
            fail_scientific_gate=False,
        ),
    )
    td8_attribution = MicroStage(
        "TD8-ATTRIBUTION-A",
        ("TD8-CORRECT-A", "TD8-SHUFFLED-A", "TD8-RANDOM-A"),
        "set8",
        "teacher_assisted",
        "01:00:00",
        (*_suite_contract(set8, include_teacher=True), _teacher_attribution_command(paths, set8)),
    )
    td8_correct_b = MicroStage(
        "TD8-CORRECT-B",
        ("TD8-ATTRIBUTION-A",),
        "set8",
        "teacher_assisted",
        "72:00:00",
        (
            *_teacher_run_commands(
                paths=paths,
                spec=set8,
                replica="B",
                teacher_control="correct",
                fail_scientific_gate=True,
            ),
            _replication_command(
                paths=paths,
                spec=set8,
                training_regime="teacher_assisted",
            ),
        ),
    )
    td16_correct_a = MicroStage(
        "TD16-CORRECT-A",
        ("TD8-CORRECT-B",),
        "transition16",
        "teacher_assisted",
        "96:00:00",
        _teacher_run_commands(
            paths=paths,
            spec=transition16,
            replica="A",
            teacher_control="correct",
            fail_scientific_gate=True,
        ),
    )
    td16_correct_b = MicroStage(
        "TD16-CORRECT-B",
        ("TD16-CORRECT-A",),
        "transition16",
        "teacher_assisted",
        "96:00:00",
        (
            *_teacher_run_commands(
                paths=paths,
                spec=transition16,
                replica="B",
                teacher_control="correct",
                fail_scientific_gate=True,
            ),
            _replication_command(
                paths=paths,
                spec=transition16,
                training_regime="teacher_assisted",
            ),
        ),
    )
    return [
        qa8_a,
        qa8_b,
        qa16_a,
        qa16_b,
        td8_correct_a,
        td8_shuffled_a,
        td8_random_a,
        td8_attribution,
        td8_correct_b,
        td16_correct_a,
        td16_correct_b,
    ]


def _preflight(paths: MicroPaths, *, expected_torch: str, output: Path) -> str:
    return shell_join(
        [
            "python",
            paths.project / "scripts" / "bootstrap" / "preflight.py",
            "--mode",
            "cluster",
            "--model-root",
            paths.model_root,
            "--expected-torch",
            expected_torch,
            "--min-gpus",
            "2",
            "--min-gpu-memory-gib",
            "70",
            "--output",
            output,
        ]
    )


def _prerequisite_command(
    paths: MicroPaths,
    *,
    stage: str,
    training_regime: str,
    expected_commit: str,
    resize_contract_report_sha256: str,
    scorer_s0_report_sha256: str,
    technical_report_sha256: str,
    teacher_t0_report_sha256: str,
) -> str:
    parts: list[str | Path] = [
        "python",
        paths.project / "scripts" / "probes" / "validate_r3_micro_prerequisites.py",
        "--resize-contract-report",
        paths.resize_contract_report,
        "--resize-contract-report-sha256",
        resize_contract_report_sha256,
        "--scorer-s0-report",
        paths.scorer_s0_report,
        "--scorer-s0-report-sha256",
        scorer_s0_report_sha256,
        "--technical-report",
        paths.technical_report,
        "--technical-report-sha256",
        technical_report_sha256,
        "--training-regime",
        training_regime,
        "--expected-commit",
        expected_commit,
        "--output",
        paths.results / f"{stage}_prerequisites.json",
    ]
    if training_regime == "teacher_assisted":
        parts.extend(
            (
                "--teacher-t0-report",
                paths.teacher_t0_report,
                "--teacher-t0-report-sha256",
                teacher_t0_report_sha256,
            )
        )
    return shell_join(parts)


def _gpu_contract() -> str:
    code = (
        "import torch; "
        "names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; "
        "assert len(names)==2, names; assert all('A800' in name for name in names), names"
    )
    return shell_join(["python", "-c", code])


def render_stage_sbatch(
    stage: MicroStage,
    *,
    paths: MicroPaths,
    expected_commit: str,
    expected_torch: str,
    resize_contract_report_sha256: str,
    scorer_s0_report_sha256: str,
    technical_report_sha256: str,
    teacher_t0_report_sha256: str,
) -> str:
    dependencies = "NONE" if not stage.dependencies else ",".join(stage.dependencies)
    commands = "\n".join(stage.commands)
    return f"""#!/usr/bin/env bash
# R3_DRY_RUN_TEMPLATE=1
# R3_SUBMISSION_SUPPORTED=0
# R3_DEPENDS_ON={dependencies}
# R3_REQUIRED_DEPENDENCY_MODE=afterok
# R3_SCIENCE_FAILURE_POLICY=no-extra-presentations-no-threshold-change-no-rescue
# R3_WALLTIME_MUST_BE_CONFIRMED_BY_SEPARATE_FIXED_THROUGHPUT_PROBE=1
#SBATCH --job-name=r3_{stage.name.lower().replace("-", "_")}
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time={stage.walltime}
#SBATCH --output={paths.logs}/{stage.name}_%j.out
#SBATCH --error={paths.logs}/{stage.name}_%j.err

set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
source /etc/profile.d/modules.sh
module purge
module load cuda/11.8
source {shlex.quote(str(paths.environment / "bin" / "activate"))}
cd {shlex.quote(str(paths.project))}

test "$(git rev-parse HEAD)" = {shlex.quote(expected_commit)}
test -z "$(git status --porcelain --untracked-files=all)"
{_preflight(paths, expected_torch=expected_torch, output=paths.results / f"{stage.name}_preflight.json")}
{_gpu_contract()}
{_prerequisite_command(paths, stage=stage.name, training_regime=stage.training_regime, expected_commit=expected_commit, resize_contract_report_sha256=resize_contract_report_sha256, scorer_s0_report_sha256=scorer_s0_report_sha256, technical_report_sha256=technical_report_sha256, teacher_t0_report_sha256=teacher_t0_report_sha256)}

{commands}
"""


def materialize_dry_run(
    *,
    paths: MicroPaths,
    stages: list[MicroStage],
    set8: SuiteSpec,
    transition16: SuiteSpec,
    expected_commit: str,
    expected_torch: str,
    resize_contract_report_sha256: str,
    scorer_s0_report_sha256: str,
    technical_report_sha256: str,
    teacher_t0_report_sha256: str,
) -> dict[str, Any]:
    if paths.run_root.exists() and any(paths.run_root.iterdir()):
        raise ValueError("R3 micro dry-run refuses to overwrite a non-empty run directory.")
    stage_names = [stage.name for stage in stages]
    if len(stage_names) != len(set(stage_names)):
        raise ValueError("R3 micro stage names must be unique.")
    seen: set[str] = set()
    for stage in stages:
        unknown_or_forward = set(stage.dependencies) - seen
        if unknown_or_forward:
            raise ValueError(f"Stage {stage.name} has unknown/forward dependencies: {sorted(unknown_or_forward)}.")
        seen.add(stage.name)
    for directory in (paths.run_root, paths.results, paths.logs, paths.sbatch):
        directory.mkdir(parents=True, exist_ok=directory != paths.sbatch)
    jobs: dict[str, Any] = {}
    for stage in stages:
        destination = paths.sbatch / f"{stage.name}.sbatch"
        destination.write_text(
            render_stage_sbatch(
                stage,
                paths=paths,
                expected_commit=expected_commit,
                expected_torch=expected_torch,
                resize_contract_report_sha256=resize_contract_report_sha256,
                scorer_s0_report_sha256=scorer_s0_report_sha256,
                technical_report_sha256=technical_report_sha256,
                teacher_t0_report_sha256=teacher_t0_report_sha256,
            ),
            encoding="utf-8",
            newline="\n",
        )
        jobs[stage.name] = {
            "status": "template_only",
            "job_id": None,
            "dependencies": list(stage.dependencies),
            "dependency_mode": "afterok",
            "suite": stage.suite,
            "training_regime": stage.training_regime,
            "hard_prerequisites": (
                ["R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L", "DL-S"]
                if stage.training_regime == "qa_only"
                else ["R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L", "DL-S", "T0"]
            ),
            "partition": "a800",
            "nodes": 1,
            "gpus_per_node": 2,
            "sbatch": str(destination),
        }
    manifest = {
        "schema_version": 2,
        "protocol": "R3-Set8-Transition16-micro-resize-dry-run-v2",
        "dry_run": True,
        "submission_supported": False,
        "commit": expected_commit,
        "expected_torch": expected_torch,
        "external_hard_dependencies": {
            "all_tracks": {
                "resize_contract_report": str(paths.resize_contract_report),
                "resize_contract_report_sha256": resize_contract_report_sha256,
                "scorer_s0_report": str(paths.scorer_s0_report),
                "scorer_s0_report_sha256": scorer_s0_report_sha256,
                "technical_report": str(paths.technical_report),
                "technical_report_sha256": technical_report_sha256,
            },
            "teacher_assisted_only": {
                "teacher_t0_report": str(paths.teacher_t0_report),
                "teacher_t0_report_sha256": teacher_t0_report_sha256,
            },
        },
        "fixed_protocol": {
            "reader_loss_mode": "listwise-choice",
            "train_choice_family": "cyclic4",
            "gate_choice_family": "reverse-cyclic4",
            "dreamlite": "DreamLite-mobile-4-step",
            "lora_rank": 4,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "gradient_accumulation": 8,
            "gradient_clip": 1.0,
            "seed": 0,
            "adapter_seed": 0,
            "eval_diffusion_seed": 0,
            "initial_state_mode": "blank",
            "recurrence_mode": "direct_latent",
            "detach_between_events": False,
            "noop_policy": "update",
            "strict_determinism": True,
            "state_image_gradient_audit": True,
            "require_mixed_delayed_probe": True,
            "dev_loss_early_stopping": False,
            "qa_only_presentations_per_state": 512,
            "teacher_distill_presentations_per_state": 256,
            "teacher_qa_presentations_per_state": 256,
            "checkpoint_cadence_presentations": 32,
            "first_eval_presentations": 64,
            "eval_cadence_presentations": 32,
            "scientific_failure_policy": "no additional presentations, relaxed gates, or post-hoc rescue",
        },
        "datasets": {
            spec.name: {
                "train": str(spec.train),
                "train_sha256": spec.train_sha256,
                "gate": str(spec.gate),
                "gate_sha256": spec.gate_sha256,
                "teacher_cache": str(spec.teacher_cache),
                "teacher_manifest_sha256": spec.teacher_manifest_sha256,
                "teacher_sidecar_sha256": spec.teacher_sidecar_sha256,
                "teacher_calibration_sha256": spec.teacher_calibration_sha256,
                "episodes": spec.episodes,
            }
            for spec in (set8, transition16)
        },
        "unlock_rules": {
            "QA16": "QA8 A and fresh B both pass and scientific payload SHA is identical",
            "TD16": "TD8 correct A passes attribution versus shuffled/random, then fresh B matches exactly",
            "qa_only_pilot": "QA16-B afterok only",
            "teacher_assisted_pilot": "TD16-CORRECT-B afterok only",
            "cross_track_substitution": False,
        },
        "teacher_controls": list(TEACHER_CONTROLS),
        "stage_order": [stage.name for stage in stages],
        "jobs": jobs,
    }
    _atomic_json(paths.run_root / "dry_run_manifest.json", manifest)
    return manifest


def _git(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _require_sha(label: str, value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase 64-character SHA256 digest.")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fail-stop R3 Set8/Transition16 micro sbatch templates; never submit jobs"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--resize-contract-report", type=Path, required=True)
    parser.add_argument("--resize-contract-report-sha256", required=True)
    parser.add_argument("--scorer-s0-report", type=Path, required=True)
    parser.add_argument("--scorer-s0-report-sha256", required=True)
    parser.add_argument("--technical-report", type=Path, required=True)
    parser.add_argument("--technical-report-sha256", required=True)
    parser.add_argument("--teacher-t0-report", type=Path, required=True)
    parser.add_argument("--teacher-t0-report-sha256", required=True)
    for prefix in ("set8", "transition16"):
        parser.add_argument(f"--{prefix}-train", type=Path, required=True)
        parser.add_argument(f"--{prefix}-train-sha256", required=True)
        parser.add_argument(f"--{prefix}-gate", type=Path, required=True)
        parser.add_argument(f"--{prefix}-gate-sha256", required=True)
        parser.add_argument(f"--{prefix}-teacher-cache", type=Path, required=True)
        parser.add_argument(f"--{prefix}-teacher-manifest-sha256", required=True)
        parser.add_argument(f"--{prefix}-teacher-sidecar-sha256", required=True)
        parser.add_argument(f"--{prefix}-teacher-calibration-sha256", required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-torch", default="2.7.1+cu118")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve(strict=True)
    commit = _git("rev-parse", "HEAD", cwd=project)
    if _COMMIT.fullmatch(commit) is None:
        raise SystemExit("Could not resolve a full clean Git commit.")
    if args.expected_commit is not None and commit != args.expected_commit:
        raise SystemExit(f"Commit mismatch: expected {args.expected_commit}, found {commit}.")
    if _git("status", "--porcelain", "--untracked-files=all", cwd=project):
        raise SystemExit("R3 micro dry-run rendering refuses a dirty checkout.")
    sha_fields = {
        name: _require_sha(name, str(getattr(args, name)))
        for name in (
            "technical_report_sha256",
            "resize_contract_report_sha256",
            "scorer_s0_report_sha256",
            "teacher_t0_report_sha256",
            "set8_train_sha256",
            "set8_gate_sha256",
            "set8_teacher_manifest_sha256",
            "set8_teacher_sidecar_sha256",
            "set8_teacher_calibration_sha256",
            "transition16_train_sha256",
            "transition16_gate_sha256",
            "transition16_teacher_manifest_sha256",
            "transition16_teacher_sidecar_sha256",
            "transition16_teacher_calibration_sha256",
        )
    }
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    paths = MicroPaths(
        project=project,
        environment=args.environment,
        model_root=args.model_root,
        run_root=args.runs_root / (args.run_name or f"r3-micro-{stamp}-{commit[:8]}"),
        resize_contract_report=args.resize_contract_report,
        scorer_s0_report=args.scorer_s0_report,
        technical_report=args.technical_report,
        teacher_t0_report=args.teacher_t0_report,
    )
    set8 = SuiteSpec(
        name="set8",
        train=args.set8_train,
        gate=args.set8_gate,
        teacher_cache=args.set8_teacher_cache,
        train_sha256=sha_fields["set8_train_sha256"],
        gate_sha256=sha_fields["set8_gate_sha256"],
        teacher_manifest_sha256=sha_fields["set8_teacher_manifest_sha256"],
        teacher_sidecar_sha256=sha_fields["set8_teacher_sidecar_sha256"],
        teacher_calibration_sha256=sha_fields["set8_teacher_calibration_sha256"],
        episodes=8,
    )
    transition16 = SuiteSpec(
        name="transition16",
        train=args.transition16_train,
        gate=args.transition16_gate,
        teacher_cache=args.transition16_teacher_cache,
        train_sha256=sha_fields["transition16_train_sha256"],
        gate_sha256=sha_fields["transition16_gate_sha256"],
        teacher_manifest_sha256=sha_fields["transition16_teacher_manifest_sha256"],
        teacher_sidecar_sha256=sha_fields["transition16_teacher_sidecar_sha256"],
        teacher_calibration_sha256=sha_fields["transition16_teacher_calibration_sha256"],
        episodes=16,
    )
    manifest = materialize_dry_run(
        paths=paths,
        stages=build_stages(paths, set8, transition16),
        set8=set8,
        transition16=transition16,
        expected_commit=commit,
        expected_torch=args.expected_torch,
        resize_contract_report_sha256=sha_fields["resize_contract_report_sha256"],
        scorer_s0_report_sha256=sha_fields["scorer_s0_report_sha256"],
        technical_report_sha256=sha_fields["technical_report_sha256"],
        teacher_t0_report_sha256=sha_fields["teacher_t0_report_sha256"],
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
