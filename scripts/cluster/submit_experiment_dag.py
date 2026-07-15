from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PREFEVAL_REVISION = "50795054b5ff5f418d2b768a331d71e480f93331"
STAGE_ORDER = {"data": 0, "sanity": 1, "lightweight": 2, "pilot": 3, "full": 4, "eval": 5}


@dataclass(frozen=True)
class Resources:
    gpus: int
    cpus: int
    memory: str
    walltime: str


@dataclass(frozen=True)
class Job:
    name: str
    stage: str
    resource_class: str
    commands: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class Paths:
    project: Path
    environment: Path
    model_root: Path
    prefeval_root: Path
    run_root: Path

    @property
    def dreamlite(self) -> Path:
        return self.model_root / "DreamLite-mobile"

    @property
    def reader(self) -> Path:
        return self.model_root / "Qwen3-VL-4B-Instruct"

    @property
    def synthetic(self) -> Path:
        return self.run_root / "data" / "synthetic_v2"

    @property
    def synthetic_set_only(self) -> Path:
        return self.run_root / "data" / "synthetic_set_only_v2"

    @property
    def results(self) -> Path:
        return self.run_root / "results"

    @property
    def outputs(self) -> Path:
        return self.run_root / "outputs"

    @property
    def prefeval_zero(self) -> Path:
        return self.run_root / "data" / "prefeval_zero_shot.jsonl"

    @property
    def prefeval_forced(self) -> Path:
        return self.run_root / "data" / "prefeval_forced_write_200.jsonl"

    def prefeval_adapt(self, split: str) -> Path:
        return self.run_root / "data" / f"prefeval_{split}.jsonl"


DEFAULT_RESOURCES = {
    "cpu": Resources(0, 8, "32G", "02:00:00"),
    "score": Resources(0, 12, "48G", "12:00:00"),
    # These are fail-safe ceilings, not expected runtimes.  The A800 partition
    # advertises MaxTime=3-12:00:00; actual elapsed/GPU-hours are harvested from
    # sacct.  Full-Reader sweeps perform four teacher-forced option forwards per
    # MCQ, and recurrent evaluation also executes every routed updater event.
    "qwen": Resources(1, 12, "64G", "36:00:00"),
    "pilot": Resources(2, 24, "96G", "36:00:00"),
    "full": Resources(2, 24, "96G", "3-00:00:00"),
    "eval": Resources(2, 24, "96G", "36:00:00"),
}


def shell_join(parts: Iterable[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def python_command(paths: Paths, relative: str, *arguments: str | Path) -> str:
    return shell_join(["python", paths.project / relative, *arguments])


def run_text(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_configuration(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"resources": {}, "command_overrides": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--config-json must contain a JSON object.")
    unknown = set(value) - {"resources", "command_overrides"}
    if unknown:
        raise ValueError(f"Unknown config-json sections: {sorted(unknown)}")
    resources = value.get("resources", {})
    overrides = value.get("command_overrides", {})
    if not isinstance(resources, dict) or not isinstance(overrides, dict):
        raise ValueError("resources and command_overrides must be JSON objects.")
    return {"resources": resources, "command_overrides": overrides}


def configured_resources(configuration: dict[str, Any]) -> dict[str, Resources]:
    result = dict(DEFAULT_RESOURCES)
    for name, raw in configuration["resources"].items():
        if name not in result or not isinstance(raw, dict):
            raise ValueError(f"Unknown or invalid resource class: {name!r}")
        unknown = set(raw) - {"gpus", "cpus", "memory", "walltime"}
        if unknown:
            raise ValueError(f"Unknown resource fields for {name}: {sorted(unknown)}")
        merged = {**asdict(result[name]), **raw}
        resource = Resources(**merged)
        if resource.gpus not in (0, 1, 2) or resource.cpus <= 0:
            raise ValueError(f"Invalid resource request for {name}: {resource}")
        if not resource.memory or not resource.walltime:
            raise ValueError(f"memory and walltime must be non-empty for {name}")
        result[name] = resource
    return result


def training_arguments(
    paths: Paths,
    *,
    train: Path,
    dev: Path,
    output: Path,
    seed: int,
    learning_rate: str,
    extra: Sequence[str | Path] = (),
    max_train_episodes: int | None = None,
    dataset_format: str = "synthetic",
) -> list[str | Path]:
    arguments: list[str | Path] = [
        "--train",
        train,
        "--dev",
        dev,
        "--dataset-format",
        dataset_format,
        "--dreamlite",
        paths.dreamlite,
        "--reader",
        paths.reader,
        "--output-dir",
        output,
        "--seed",
        str(seed),
        "--adapter-seed",
        str(seed),
        "--learning-rate",
        learning_rate,
        "--lora-rank",
        "4",
        "--epochs",
        "2",
        "--gradient-accumulation",
        "8",
        "--gradient-clip",
        "1.0",
        "--checkpoint-every",
        "100",
        "--eval-every",
        "250",
        "--early-stopping-patience",
        "3",
        "--checkpoint-unet",
        "--dreamlite-device",
        "cuda:0",
        "--reader-device",
        "cuda:1",
    ]
    if max_train_episodes is not None:
        arguments.extend(["--max-train-episodes", str(max_train_episodes)])
    arguments.extend(extra)
    return arguments


def dreamlite_eval_command(
    paths: Paths,
    *,
    episodes: Path,
    data_format: str,
    checkpoint: Path | None,
    output: Path,
    method: str,
    training_seed: int | None,
    conditions: Sequence[str] = ("standard",),
    noop_policy: str = "keep",
    recurrence_mode: str = "direct_latent",
    diffusion_seed: int = 0,
    lora_rank: int = 4,
    limit: int | None = None,
) -> str:
    arguments: list[str | Path] = [
        "--episodes",
        episodes,
        "--format",
        data_format,
        "--dreamlite",
        paths.dreamlite,
        "--reader",
        paths.reader,
        "--output",
        output,
        "--method",
        method,
        "--conditions",
        *conditions,
        "--noop-policy",
        noop_policy,
        "--recurrence-mode",
        recurrence_mode,
        "--seed",
        str(diffusion_seed),
        "--adapter-seed",
        str(training_seed or 0),
        "--lora-rank",
        str(lora_rank),
        "--dreamlite-device",
        "cuda:0",
        "--reader-device",
        "cuda:1",
    ]
    if checkpoint is not None:
        arguments.extend(["--checkpoint", checkpoint])
    if training_seed is not None:
        arguments.extend(["--training-seed", str(training_seed)])
    if limit is not None:
        arguments.extend(["--limit", str(limit)])
    return python_command(paths, "scripts/eval/dreamlite_mcq.py", *arguments)


def add_data_jobs(
    jobs: list[Job],
    paths: Paths,
    *,
    fetch_prefeval: bool,
    generate_set_only: bool,
) -> None:
    commands: list[str] = []
    if fetch_prefeval:
        commands.append(
            python_command(
                paths,
                "scripts/bootstrap/fetch_datasets.py",
                "--data-root",
                paths.prefeval_root.parent,
            )
        )
    commands.extend(
        [
            f"test \"$(git -C {shlex.quote(str(paths.prefeval_root))} rev-parse HEAD)\" = {PREFEVAL_REVISION}",
            f"test -z \"$(git -C {shlex.quote(str(paths.prefeval_root))} status --porcelain)\"",
            python_command(
                paths,
                "scripts/data/generate_synthetic.py",
                "--output-dir",
                paths.synthetic,
                "--seed",
                "2026",
                "--train",
                "5000",
                "--dev",
                "500",
                "--test-id",
                "1000",
                "--test-ood",
                "1000",
            ),
            python_command(
                paths,
                "scripts/data/validate_synthetic.py",
                paths.synthetic,
                "--output",
                paths.results / "synthetic_v2_validation.json",
            ),
            python_command(
                paths,
                "scripts/eval/prepare_prefeval.py",
                "--prefeval-root",
                paths.prefeval_root,
                "--output",
                paths.prefeval_zero,
                "--protocol",
                "oracle-sparse",
                "--forms",
                "explicit",
                "implicit_choice",
                "implicit_persona",
                "--expected-base-pairs",
                "1000",
                "--expected-records",
                "3000",
            ),
            python_command(
                paths,
                "scripts/eval/prepare_prefeval.py",
                "--prefeval-root",
                paths.prefeval_root,
                "--output",
                paths.prefeval_forced,
                "--protocol",
                "forced-write",
                "--forced-write-k",
                "0",
                "2",
                "5",
                "10",
                "--forms",
                "explicit",
                "implicit_choice",
                "implicit_persona",
                "--max-base-pairs-per-topic",
                "10",
                "--expected-base-pairs",
                "1000",
                "--expected-records",
                "2400",
            ),
            python_command(
                paths,
                "scripts/bootstrap/freeze_environment.py",
                "--output",
                paths.results / "environment-lock.json",
            ),
        ]
    )
    if generate_set_only:
        commands.extend(
            [
                python_command(
                    paths,
                    "scripts/data/generate_synthetic.py",
                    "--output-dir",
                    paths.synthetic_set_only,
                    "--seed",
                    "2026",
                    "--train",
                    "5000",
                    "--dev",
                    "500",
                    "--test-id",
                    "1000",
                    "--test-ood",
                    "1000",
                    "--transition-profile",
                    "set-only",
                ),
                python_command(
                    paths,
                    "scripts/data/validate_synthetic.py",
                    paths.synthetic_set_only,
                    "--output",
                    paths.results / "synthetic_set_only_v2_validation.json",
                ),
            ]
        )
    # The locked PrefEval snapshot has unequal topic sizes.  Seed-2026 selects
    # 188 base pairs in four held-out topics; the remaining 812 pairs split
    # into 730 train and 82 dev base pairs under the per-topic 90/10 rule.
    # Each base pair is exported in all three forms.
    expected_adapt_records = {"adapt_train": 2190, "adapt_dev": 246, "adapt_ood": 564}
    for split in ("adapt_train", "adapt_dev", "adapt_ood"):
        commands.append(
            python_command(
                paths,
                "scripts/eval/prepare_prefeval.py",
                "--prefeval-root",
                paths.prefeval_root,
                "--output",
                paths.prefeval_adapt(split),
                "--protocol",
                "oracle-sparse",
                "--forms",
                "explicit",
                "implicit_choice",
                "implicit_persona",
                "--subset",
                split,
                "--expected-base-pairs",
                "1000",
                "--expected-records",
                str(expected_adapt_records[split]),
            )
        )
    jobs.append(Job("D0_data", "data", "cpu", tuple(commands)))


def add_gate_jobs(jobs: list[Job], paths: Paths) -> None:
    jobs.append(
        Job(
            "D1_qwen_sanity",
            "sanity",
            "qwen",
            (
                python_command(
                    paths,
                    "scripts/data/qwen_sanity.py",
                    "--dataset",
                    paths.synthetic / "dev.jsonl",
                    "--reader",
                    paths.reader,
                    "--output-json",
                    paths.results / "qwen_sanity.json",
                    "--limit",
                    "200",
                    "--oracle-threshold",
                    "0.95",
                    "--query-only-ceiling",
                    "0.30",
                    "--device",
                    "cuda:0",
                ),
            ),
            ("D0_data",),
        )
    )
    jobs.append(
        Job(
            "D2_lightweight_overfit",
            "lightweight",
            "qwen",
            (
                python_command(
                    paths,
                    "scripts/train/lightweight_episode.py",
                    "--train",
                    paths.synthetic / "train.jsonl",
                    "--dev",
                    paths.synthetic / "dev.jsonl",
                    "--reader",
                    paths.reader,
                    "--output-dir",
                    paths.outputs / "lightweight_overfit",
                    "--method",
                    "recurrent",
                    "--overfit-gate",
                    "--overfit-episodes",
                    "64",
                    "--max-optimizer-steps",
                    "2000",
                    "--gradient-accumulation",
                    "1",
                    "--overfit-threshold",
                    "0.90",
                    "--eval-every",
                    "100",
                    "--seed",
                    "0",
                    "--device",
                    "cuda:0",
                ),
            ),
            ("D1_qwen_sanity",),
        )
    )


def add_pilot_jobs(jobs: list[Job], paths: Paths, candidate_specification: Path, selected_env: Path) -> None:
    # Pilot selection and all pilot stop-gate evidence stay on dev.  Test-ID is
    # first touched only after the learning rate has been selected and frozen.
    blank = paths.outputs / "pilot_baselines" / "blank_dev.jsonl"
    frozen = paths.outputs / "pilot_baselines" / "frozen_dev.jsonl"
    jobs.append(
        Job(
            "P0_blank",
            "pilot",
            "qwen",
            (
                python_command(
                    paths,
                    "scripts/eval/qwen_text_baselines.py",
                    "--episodes",
                    paths.synthetic / "dev.jsonl",
                    "--format",
                    "synthetic",
                    "--reader",
                    paths.reader,
                    "--output",
                    blank,
                    "--methods",
                    "query_only",
                    "--seed",
                    "0",
                    "--device",
                    "cuda:0",
                ),
            ),
            ("D2_lightweight_overfit",),
        )
    )
    jobs.append(
        Job(
            "P0_frozen",
            "pilot",
            "eval",
            (
                dreamlite_eval_command(
                    paths,
                    episodes=paths.synthetic / "dev.jsonl",
                    data_format="synthetic",
                    checkpoint=None,
                    output=frozen,
                    method="frozen_dreamlite",
                    training_seed=0,
                ),
            ),
            ("D2_lightweight_overfit",),
        )
    )
    pilot_names: list[str] = []
    for index, learning_rate in enumerate(("3e-5", "1e-4", "3e-4")):
        name = f"P{index + 1}_lr_{learning_rate.replace('-', 'm')}"
        pilot_names.append(name)
        output = paths.outputs / "pilots" / f"lr_{learning_rate}"
        train = python_command(
            paths,
            "scripts/train/dreamlite_episode.py",
            *training_arguments(
                paths,
                train=paths.synthetic / "train.jsonl",
                dev=paths.synthetic / "dev.jsonl",
                output=output,
                seed=0,
                learning_rate=learning_rate,
                max_train_episodes=1000,
            ),
        )
        evaluate = dreamlite_eval_command(
            paths,
            episodes=paths.synthetic / "dev.jsonl",
            data_format="synthetic",
            checkpoint=output / "best.pt",
            output=output / "dev_predictions.jsonl",
            method=f"pilot_lr_{learning_rate}",
            training_seed=0,
            conditions=("standard", "reset", "shuffle", "state_swap"),
        )
        jobs.append(Job(name, "pilot", "pilot", (train, evaluate), ("D2_lightweight_overfit",)))

    jobs.append(
        Job(
            "P4_select",
            "pilot",
            "cpu",
            (
                python_command(
                    paths,
                    "scripts/cluster/select_pilot.py",
                    "--specification",
                    candidate_specification,
                    "--output",
                    paths.results / "pilot_selection.json",
                    "--env-output",
                    selected_env,
                    "--minimum-gain",
                    "0.10",
                    "--minimum-intervention-drop",
                    "0.10",
                ),
            ),
            tuple(["P0_blank", "P0_frozen", *pilot_names]),
        )
    )
    audit_output = paths.outputs / "pilot_resume_audit"
    resume_command = "\n".join(
        [
            f"source {shlex.quote(str(selected_env))}",
            python_command(
                paths,
                "scripts/train/dreamlite_episode.py",
                *training_arguments(
                    paths,
                    train=paths.synthetic / "train.jsonl",
                    dev=paths.synthetic / "dev.jsonl",
                    output=audit_output,
                    seed=0,
                    learning_rate="${VLM_SELECTED_LR}",
                    max_train_episodes=1000,
                    extra=("--resume", "${VLM_SELECTED_RESUME_CHECKPOINT}"),
                ),
            ).replace("'${VLM_SELECTED_LR}'", '"${VLM_SELECTED_LR}"').replace(
                "'${VLM_SELECTED_RESUME_CHECKPOINT}'", '"${VLM_SELECTED_RESUME_CHECKPOINT}"'
            ),
            python_command(
                paths,
                "scripts/cluster/compare_checkpoints.py",
                "--reference",
                "${VLM_SELECTED_DIR}/last.pt",
                "--resumed",
                audit_output / "last.pt",
                "--output",
                paths.results / "pilot_resume_equivalence.json",
                "--atol",
                "1e-6",
                "--rtol",
                "1e-5",
            ).replace("'${VLM_SELECTED_DIR}/last.pt'", '"${VLM_SELECTED_DIR}/last.pt"'),
        ]
    )
    jobs.append(Job("P5_resume_gate", "pilot", "pilot", (resume_command,), ("P4_select",)))


def lightweight_matrix_commands(paths: Paths, *, seed: int, method: str, output: Path) -> tuple[str, ...]:
    train_arguments: list[str | Path] = [
        "--train",
        paths.synthetic / "train.jsonl",
        "--dev",
        paths.synthetic / "dev.jsonl",
        "--reader",
        paths.reader,
        "--output-dir",
        output,
        "--method",
        method,
        "--epochs",
        "2",
        "--gradient-accumulation",
        "8",
        "--eval-every",
        "250",
        "--seed",
        str(seed),
        "--device",
        "cuda:0",
    ]
    commands = [python_command(paths, "scripts/train/lightweight_episode.py", *train_arguments)]
    commands.append(
        python_command(
            paths,
            "scripts/train/lightweight_predict.py",
            "--episodes",
            paths.synthetic / "test_id.jsonl",
            "--checkpoint",
            output / "best.pt",
            "--reader",
            paths.reader,
            "--output",
            output / "synthetic.jsonl",
            "--device",
            "cuda:0",
        )
    )
    commands.append(
        python_command(
            paths,
            "scripts/train/lightweight_predict.py",
            "--episodes",
            paths.synthetic / "test_ood.jsonl",
            "--checkpoint",
            output / "best.pt",
            "--reader",
            paths.reader,
            "--output",
            output / "synthetic_ood.jsonl",
            "--device",
            "cuda:0",
        )
    )
    for split in ("adapt_train", "adapt_dev", "adapt_ood"):
        for protocol, episodes in (("prefeval", paths.prefeval_zero), ("prefeval_forced", paths.prefeval_forced)):
            commands.append(
                python_command(
                    paths,
                    "scripts/train/lightweight_predict.py",
                    "--episodes",
                    episodes,
                    "--dataset-format",
                    "prefeval-export",
                    "--prefeval-split",
                    split,
                    "--checkpoint",
                    output / "best.pt",
                    "--reader",
                    paths.reader,
                    "--output",
                    output / f"{protocol}_{split}.jsonl",
                    "--device",
                    "cuda:0",
                )
            )
    return tuple(commands)


def add_full_jobs(
    jobs: list[Job],
    paths: Paths,
    *,
    selected_env: Path,
    include_ablations: bool,
    set_only_train: Path | None,
    set_only_dev: Path | None,
    include_prefeval_adaptation: bool,
) -> None:
    for seed in (0, 1, 2):
        output = paths.outputs / "dreamlite_main" / f"seed_{seed}"
        command = python_command(
            paths,
            "scripts/train/dreamlite_episode.py",
            *training_arguments(
                paths,
                train=paths.synthetic / "train.jsonl",
                dev=paths.synthetic / "dev.jsonl",
                output=output,
                seed=seed,
                learning_rate="${VLM_SELECTED_LR}",
            ),
        ).replace("'${VLM_SELECTED_LR}'", '"${VLM_SELECTED_LR}"')
        jobs.append(
            Job(
                f"F_dreamlite_seed_{seed}",
                "full",
                "full",
                (f"source {shlex.quote(str(selected_env))}\n{command}",),
                ("P5_resume_gate",),
            )
        )
        for method, name in (("recurrent", "lightweight"), ("static-initial-image", "static")):
            baseline_output = paths.outputs / f"{name}_main" / f"seed_{seed}"
            jobs.append(
                Job(
                    f"F_{name}_seed_{seed}",
                    "full",
                    "qwen",
                    lightweight_matrix_commands(paths, seed=seed, method=method, output=baseline_output),
                    ("P5_resume_gate",),
                )
            )

    if include_ablations:
        comparator_output = paths.outputs / "ablations" / "rank4_blank_1k" / "seed_0"
        comparator_evaluation = dreamlite_eval_command(
            paths,
            episodes=paths.synthetic / "test_id.jsonl",
            data_format="synthetic",
            checkpoint=Path("${VLM_SELECTED_CHECKPOINT}"),
            output=comparator_output / "test_id.jsonl",
            method="ablation_rank4_blank_1k",
            training_seed=0,
        ).replace("'${VLM_SELECTED_CHECKPOINT}'", '"${VLM_SELECTED_CHECKPOINT}"')
        jobs.append(
            Job(
                "A_rank4_blank_1k_seed_0",
                "full",
                "eval",
                (f"source {shlex.quote(str(selected_env))}\n{comparator_evaluation}",),
                ("P5_resume_gate",),
            )
        )
        variants: list[tuple[str, tuple[str, ...], str, int]] = [
            ("detach", ("--detach-between-events",), "direct_latent", 4),
            ("decode_reencode", ("--recurrence-mode", "decode_reencode"), "decode_reencode", 4),
            ("noop_skip", ("--noop-policy", "skip"), "direct_latent", 4),
        ]
        for variant, extra, recurrence_mode, rank in variants:
            for seed in (0, 1, 2):
                add_ablation_job(
                    jobs,
                    paths,
                    selected_env=selected_env,
                    variant=variant,
                    seed=seed,
                    train=paths.synthetic / "train.jsonl",
                    dev=paths.synthetic / "dev.jsonl",
                    extra=extra,
                    recurrence_mode=recurrence_mode,
                    rank=rank,
                )
        add_ablation_job(
            jobs,
            paths,
            selected_env=selected_env,
            variant="rank8",
            seed=0,
            train=paths.synthetic / "train.jsonl",
            dev=paths.synthetic / "dev.jsonl",
            extra=("--lora-rank", "8", "--max-train-episodes", "1000"),
            recurrence_mode="direct_latent",
            rank=8,
        )
        add_ablation_job(
            jobs,
            paths,
            selected_env=selected_env,
            variant="learned_initial",
            seed=0,
            train=paths.synthetic / "train.jsonl",
            dev=paths.synthetic / "dev.jsonl",
            extra=("--learn-initial-state", "--max-train-episodes", "1000"),
            recurrence_mode="direct_latent",
            rank=4,
        )
        if set_only_train is not None and set_only_dev is not None:
            for seed in (0, 1, 2):
                add_ablation_job(
                    jobs,
                    paths,
                    selected_env=selected_env,
                    variant="set_only",
                    seed=seed,
                    train=set_only_train,
                    dev=set_only_dev,
                    extra=("--curriculum", "set-only"),
                    recurrence_mode="direct_latent",
                    rank=4,
                )

    if include_prefeval_adaptation:
        output = paths.outputs / "prefeval_adapted_seed_0"
        command = python_command(
            paths,
            "scripts/train/dreamlite_episode.py",
            *training_arguments(
                paths,
                train=paths.prefeval_adapt("adapt_train"),
                dev=paths.prefeval_adapt("adapt_dev"),
                output=output,
                seed=0,
                learning_rate="${VLM_SELECTED_LR}",
                dataset_format="prefeval-export",
            ),
        ).replace("'${VLM_SELECTED_LR}'", '"${VLM_SELECTED_LR}"')
        jobs.append(
            Job(
                "F_prefeval_adapted_seed_0",
                "full",
                "full",
                (f"source {shlex.quote(str(selected_env))}\n{command}",),
                ("P5_resume_gate",),
            )
        )


def add_ablation_job(
    jobs: list[Job],
    paths: Paths,
    *,
    selected_env: Path,
    variant: str,
    seed: int,
    train: Path,
    dev: Path,
    extra: Sequence[str],
    recurrence_mode: str,
    rank: int,
) -> None:
    output = paths.outputs / "ablations" / variant / f"seed_{seed}"
    # Explicit variant flags follow defaults, so duplicate scalar arguments resolve to the
    # final CLI occurrence and remain visible in the generated sbatch/manifest.
    train_command = python_command(
        paths,
        "scripts/train/dreamlite_episode.py",
        *training_arguments(
            paths,
            train=train,
            dev=dev,
            output=output,
            seed=seed,
            learning_rate="${VLM_SELECTED_LR}",
            extra=extra,
        ),
    ).replace("'${VLM_SELECTED_LR}'", '"${VLM_SELECTED_LR}"')
    evaluation = dreamlite_eval_command(
        paths,
        episodes=paths.synthetic / "test_id.jsonl",
        data_format="synthetic",
        checkpoint=output / "best.pt",
        output=output / "test_id.jsonl",
        method=f"ablation_{variant}",
        training_seed=seed,
        recurrence_mode=recurrence_mode,
        lora_rank=rank,
        noop_policy="skip" if variant == "noop_skip" else "keep",
    )
    jobs.append(
        Job(
            f"A_{variant}_seed_{seed}",
            "full",
            "full",
            (f"source {shlex.quote(str(selected_env))}\n{train_command}", evaluation),
            ("P5_resume_gate",),
        )
    )


def add_evaluation_jobs(
    jobs: list[Job],
    paths: Paths,
    *,
    include_ablations: bool,
    include_prefeval_adaptation: bool,
) -> None:
    evaluation_names: list[str] = []
    for seed in (0, 1, 2):
        main = paths.outputs / "dreamlite_main" / f"seed_{seed}"
        name = f"E_dreamlite_seed_{seed}"
        evaluation_names.append(name)
        commands = (
            dreamlite_eval_command(
                paths,
                episodes=paths.synthetic / "test_id.jsonl",
                data_format="synthetic",
                checkpoint=main / "best.pt",
                output=main / "synthetic_predictions.jsonl",
                method="dreamlite_latent",
                training_seed=seed,
                conditions=("standard", "reset", "shuffle", "state_swap"),
                noop_policy="both",
            ),
            dreamlite_eval_command(
                paths,
                episodes=paths.synthetic / "test_ood.jsonl",
                data_format="synthetic",
                checkpoint=main / "best.pt",
                output=main / "synthetic_ood_predictions.jsonl",
                method="dreamlite_latent",
                training_seed=seed,
            ),
            dreamlite_eval_command(
                paths,
                episodes=paths.prefeval_zero,
                data_format="prefeval",
                checkpoint=main / "best.pt",
                output=main / "prefeval_predictions.jsonl",
                method="dreamlite_latent",
                training_seed=seed,
            ),
            dreamlite_eval_command(
                paths,
                episodes=paths.prefeval_forced,
                data_format="prefeval",
                checkpoint=main / "best.pt",
                output=main / "prefeval_forced_predictions.jsonl",
                method="dreamlite_latent",
                training_seed=seed,
                noop_policy="both",
            ),
        )
        jobs.append(Job(name, "eval", "eval", commands, (f"F_dreamlite_seed_{seed}",)))

        text_name = f"E_text_seed_{seed}"
        evaluation_names.append(text_name)
        text_root = paths.outputs / "text_baselines" / f"seed_{seed}"
        jobs.append(
            Job(
                text_name,
                "eval",
                "qwen",
                (
                    python_command(
                        paths,
                        "scripts/eval/qwen_text_baselines.py",
                        "--episodes",
                        paths.synthetic / "test_id.jsonl",
                        "--format",
                        "synthetic",
                        "--reader",
                        paths.reader,
                        "--output",
                        text_root / "synthetic.jsonl",
                        "--methods",
                        "query_only",
                        "full_history",
                        "oracle_target",
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda:0",
                    ),
                    python_command(
                        paths,
                        "scripts/eval/qwen_text_baselines.py",
                        "--episodes",
                        paths.synthetic / "test_ood.jsonl",
                        "--format",
                        "synthetic",
                        "--reader",
                        paths.reader,
                        "--output",
                        text_root / "synthetic_ood.jsonl",
                        "--methods",
                        "query_only",
                        "full_history",
                        "oracle_target",
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda:0",
                    ),
                    python_command(
                        paths,
                        "scripts/eval/qwen_text_baselines.py",
                        "--episodes",
                        paths.prefeval_zero,
                        "--format",
                        "prefeval",
                        "--reader",
                        paths.reader,
                        "--output",
                        text_root / "prefeval.jsonl",
                        "--methods",
                        "query_only",
                        "full_history",
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda:0",
                    ),
                    python_command(
                        paths,
                        "scripts/eval/qwen_text_baselines.py",
                        "--episodes",
                        paths.prefeval_forced,
                        "--format",
                        "prefeval",
                        "--reader",
                        paths.reader,
                        "--output",
                        text_root / "prefeval_forced.jsonl",
                        "--methods",
                        "query_only",
                        "full_history",
                        "--seed",
                        str(seed),
                        "--device",
                        "cuda:0",
                    ),
                ),
                ("P5_resume_gate",),
            )
        )

        frozen_name = f"E_frozen_seed_{seed}"
        evaluation_names.append(frozen_name)
        frozen_root = paths.outputs / "frozen_baseline" / f"seed_{seed}"
        jobs.append(
            Job(
                frozen_name,
                "eval",
                "eval",
                (
                    dreamlite_eval_command(
                        paths,
                        episodes=paths.synthetic / "test_id.jsonl",
                        data_format="synthetic",
                        checkpoint=None,
                        output=frozen_root / "synthetic.jsonl",
                        method="frozen_dreamlite",
                        training_seed=seed,
                    ),
                    dreamlite_eval_command(
                        paths,
                        episodes=paths.synthetic / "test_ood.jsonl",
                        data_format="synthetic",
                        checkpoint=None,
                        output=frozen_root / "synthetic_ood.jsonl",
                        method="frozen_dreamlite",
                        training_seed=seed,
                    ),
                    dreamlite_eval_command(
                        paths,
                        episodes=paths.prefeval_zero,
                        data_format="prefeval",
                        checkpoint=None,
                        output=frozen_root / "prefeval.jsonl",
                        method="frozen_dreamlite",
                        training_seed=seed,
                    ),
                    dreamlite_eval_command(
                        paths,
                        episodes=paths.prefeval_forced,
                        data_format="prefeval",
                        checkpoint=None,
                        output=frozen_root / "prefeval_forced.jsonl",
                        method="frozen_dreamlite",
                        training_seed=seed,
                    ),
                ),
                ("P5_resume_gate",),
            )
        )

    score_dependencies = [
        *evaluation_names,
        *(f"F_lightweight_seed_{seed}" for seed in (0, 1, 2)),
        *(f"F_static_seed_{seed}" for seed in (0, 1, 2)),
    ]
    synthetic_inputs: list[Path] = []
    prefeval_inputs: list[Path] = []
    forced_inputs: list[Path] = []
    for seed in (0, 1, 2):
        synthetic_inputs.extend(
            [
                paths.outputs / "dreamlite_main" / f"seed_{seed}" / "synthetic_predictions.jsonl",
                paths.outputs / "dreamlite_main" / f"seed_{seed}" / "synthetic_ood_predictions.jsonl",
                paths.outputs / "text_baselines" / f"seed_{seed}" / "synthetic.jsonl",
                paths.outputs / "text_baselines" / f"seed_{seed}" / "synthetic_ood.jsonl",
                paths.outputs / "frozen_baseline" / f"seed_{seed}" / "synthetic.jsonl",
                paths.outputs / "frozen_baseline" / f"seed_{seed}" / "synthetic_ood.jsonl",
                paths.outputs / "lightweight_main" / f"seed_{seed}" / "synthetic.jsonl",
                paths.outputs / "lightweight_main" / f"seed_{seed}" / "synthetic_ood.jsonl",
                paths.outputs / "static_main" / f"seed_{seed}" / "synthetic.jsonl",
                paths.outputs / "static_main" / f"seed_{seed}" / "synthetic_ood.jsonl",
            ]
        )
        prefeval_inputs.extend(
            [
                paths.outputs / "dreamlite_main" / f"seed_{seed}" / "prefeval_predictions.jsonl",
                paths.outputs / "text_baselines" / f"seed_{seed}" / "prefeval.jsonl",
                paths.outputs / "frozen_baseline" / f"seed_{seed}" / "prefeval.jsonl",
            ]
        )
        forced_inputs.extend(
            [
                paths.outputs / "dreamlite_main" / f"seed_{seed}" / "prefeval_forced_predictions.jsonl",
                paths.outputs / "text_baselines" / f"seed_{seed}" / "prefeval_forced.jsonl",
                paths.outputs / "frozen_baseline" / f"seed_{seed}" / "prefeval_forced.jsonl",
            ]
        )
        for method_root in ("lightweight_main", "static_main"):
            prefeval_inputs.extend(
                paths.outputs / method_root / f"seed_{seed}" / f"prefeval_{split}.jsonl"
                for split in ("adapt_train", "adapt_dev", "adapt_ood")
            )
            forced_inputs.extend(
                paths.outputs / method_root / f"seed_{seed}" / f"prefeval_forced_{split}.jsonl"
                for split in ("adapt_train", "adapt_dev", "adapt_ood")
            )

    merged_synthetic = paths.results / "synthetic_predictions.jsonl"
    merged_prefeval = paths.results / "prefeval_predictions.jsonl"
    merged_forced = paths.results / "prefeval_protocol_predictions.jsonl"
    merge_synthetic = python_command(
        paths,
        "scripts/cluster/merge_jsonl.py",
        *(item for path in synthetic_inputs for item in ("--input", path)),
        "--output",
        merged_synthetic,
        "--method-map",
        "recurrent=lightweight_recurrent",
        "--method-map",
        "static-initial-image=static_learned_initial_image",
    )
    merge_prefeval = python_command(
        paths,
        "scripts/cluster/merge_jsonl.py",
        *(item for path in prefeval_inputs for item in ("--input", path)),
        "--output",
        merged_prefeval,
        "--method-map",
        "recurrent=lightweight_recurrent",
        "--method-map",
        "static-initial-image=static_learned_initial_image",
    )
    merge_forced = python_command(
        paths,
        "scripts/cluster/merge_jsonl.py",
        *(item for path in [*prefeval_inputs, *forced_inputs] for item in ("--input", path)),
        "--output",
        merged_forced,
        "--method-map",
        "recurrent=lightweight_recurrent",
        "--method-map",
        "static-initial-image=static_learned_initial_image",
    )
    main_score_commands: list[str] = [
        merge_synthetic,
        python_command(
            paths,
            "scripts/eval/score_synthetic.py",
            "--predictions",
            merged_synthetic,
            "--output",
            paths.results / "synthetic_scores.json",
        ),
        python_command(
            paths,
            "scripts/eval/assess_core_hypothesis.py",
            "--scores",
            paths.results / "synthetic_scores.json",
            "--output",
            paths.results / "core_hypothesis_assessment.json",
            "--minimum-intervention-drop",
            "0.10",
        ),
        merge_prefeval,
    ]
    for form in ("explicit", "implicit_choice", "implicit_persona"):
        main_score_commands.append(
            python_command(
                paths,
                "scripts/eval/score_prefeval.py",
                "--predictions",
                merged_prefeval,
                "--output",
                paths.results / f"prefeval_scores_{form}.json",
                "--headline-protocol",
                "oracle-sparse",
                "--headline-form",
                form,
            )
        )
    main_score_commands.append(merge_forced)
    for forced_write_k in (0, 2, 5, 10):
        for form in ("explicit", "implicit_choice", "implicit_persona"):
            main_score_commands.append(
                python_command(
                    paths,
                    "scripts/eval/score_prefeval.py",
                    "--predictions",
                    merged_forced,
                    "--output",
                    paths.results / f"prefeval_forced_k{forced_write_k}_scores_{form}.json",
                    "--headline-protocol",
                    "forced-write",
                    "--headline-forced-write-k",
                    str(forced_write_k),
                    "--headline-form",
                    form,
                )
            )
    jobs.append(
        Job(
            "E_score_main",
            "eval",
            "score",
            tuple(main_score_commands),
            tuple(score_dependencies),
        )
    )

    if include_prefeval_adaptation:
        adapted_output = paths.outputs / "prefeval_adapted_seed_0"
        predictions = adapted_output / "adapt_ood_predictions.jsonl"
        jobs.append(
            Job(
                "E_prefeval_adapted_seed_0",
                "eval",
                "eval",
                (
                    dreamlite_eval_command(
                        paths,
                        episodes=paths.prefeval_adapt("adapt_ood"),
                        data_format="prefeval",
                        checkpoint=adapted_output / "best.pt",
                        output=predictions,
                        method="dreamlite_prefeval_adapted",
                        training_seed=0,
                    ),
                    python_command(
                        paths,
                        "scripts/cluster/summarize_predictions.py",
                        "--predictions",
                        predictions,
                        "--output",
                        paths.results / "prefeval_adapted_state_streaming.json",
                        "--label",
                        "PrefEval adapted state-streaming protocol (seed 0, held-out topics)",
                    ),
                ),
                ("F_prefeval_adapted_seed_0",),
            )
        )

    if include_ablations:
        ablation_runs = [
            ("rank4_blank_1k", 0),
            *(('detach', seed) for seed in (0, 1, 2)),
            *(('decode_reencode', seed) for seed in (0, 1, 2)),
            *(('noop_skip', seed) for seed in (0, 1, 2)),
            ("rank8", 0),
            ("learned_initial", 0),
            *(('set_only', seed) for seed in (0, 1, 2)),
        ]
        ablation_inputs = [
            paths.outputs / "ablations" / variant / f"seed_{seed}" / "test_id.jsonl"
            for variant, seed in ablation_runs
        ]
        # Include the matched 5k/rank-4/blank main runs as controls for the
        # three-seed recurrent ablations.  The selected 1k pilot above is the
        # budget-matched control for rank-8 and learned-initial-state.
        ablation_inputs.extend(
            paths.outputs / "dreamlite_main" / f"seed_{seed}" / "synthetic_predictions.jsonl"
            for seed in (0, 1, 2)
        )
        merged_ablations = paths.results / "ablation_predictions.jsonl"
        jobs.append(
            Job(
                "E_score_ablations",
                "eval",
                "score",
                (
                    python_command(
                        paths,
                        "scripts/cluster/merge_jsonl.py",
                        *(item for path in ablation_inputs for item in ("--input", path)),
                        "--output",
                        merged_ablations,
                    ),
                    python_command(
                        paths,
                        "scripts/eval/score_ablations.py",
                        "--predictions",
                        merged_ablations,
                        "--output",
                        paths.results / "ablation_summary.json",
                    ),
                ),
                tuple(
                    [
                        *(f"A_{variant}_seed_{seed}" for variant, seed in ablation_runs),
                        *(f"E_dreamlite_seed_{seed}" for seed in (0, 1, 2)),
                    ]
                ),
            )
        )
        noise_jobs: list[str] = []
        noise_inputs: list[Path] = []
        for seed in (0, 1, 2):
            name = f"E_noise_seed_{seed}"
            noise_jobs.append(name)
            output_root = paths.outputs / "dreamlite_main" / f"seed_{seed}" / "noise"
            noise_inputs.extend(output_root / f"diffusion_seed_{noise_seed}.jsonl" for noise_seed in range(5))
            commands = tuple(
                dreamlite_eval_command(
                    paths,
                    episodes=paths.synthetic / "test_id.jsonl",
                    data_format="synthetic",
                    checkpoint=paths.outputs / "dreamlite_main" / f"seed_{seed}" / "best.pt",
                    output=output_root / f"diffusion_seed_{noise_seed}.jsonl",
                    method="dreamlite_latent",
                    training_seed=seed,
                    diffusion_seed=noise_seed,
                    limit=200,
                )
                for noise_seed in range(5)
            )
            jobs.append(Job(name, "eval", "eval", commands, (f"F_dreamlite_seed_{seed}",)))
        merged_noise = paths.results / "noise_robustness_predictions.jsonl"
        jobs.append(
            Job(
                "E_score_noise",
                "eval",
                "score",
                (
                    python_command(
                        paths,
                        "scripts/cluster/merge_jsonl.py",
                        *(item for path in noise_inputs for item in ("--input", path)),
                        "--output",
                        merged_noise,
                    ),
                    python_command(
                        paths,
                        "scripts/eval/score_noise_robustness.py",
                        "--predictions",
                        merged_noise,
                        "--output",
                        paths.results / "noise_robustness_summary.json",
                        "--expected-episodes-per-training-seed",
                        "200",
                        "--expected-diffusion-seeds",
                        "0",
                        "1",
                        "2",
                        "3",
                        "4",
                    ),
                ),
                tuple(noise_jobs),
            )
        )


def apply_command_overrides(jobs: list[Job], configuration: dict[str, Any]) -> list[Job]:
    overrides = configuration["command_overrides"]
    unknown = set(overrides) - {job.name for job in jobs}
    if unknown:
        raise ValueError(f"Command overrides reference unknown jobs: {sorted(unknown)}")
    result: list[Job] = []
    for job in jobs:
        raw = overrides.get(job.name)
        if raw is None:
            result.append(job)
            continue
        values = [raw] if isinstance(raw, str) else raw
        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value.strip() for value in values):
            raise ValueError(f"Command override for {job.name} must be a string or non-empty string list.")
        result.append(Job(job.name, job.stage, job.resource_class, tuple(values), job.dependencies))
    return result


def validate_jobs(jobs: Sequence[Job]) -> None:
    names = [job.name for job in jobs]
    if len(names) != len(set(names)):
        raise ValueError("Experiment DAG has duplicate job names.")
    seen: set[str] = set()
    for job in jobs:
        if job.stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage for {job.name}: {job.stage}")
        missing = set(job.dependencies) - seen
        if missing:
            raise ValueError(f"Job {job.name} has non-topological dependencies: {sorted(missing)}")
        seen.add(job.name)


def render_sbatch(
    job: Job,
    resources: Resources,
    *,
    paths: Paths,
    partition: str,
    expected_commit: str,
    expected_torch: str,
) -> str:
    gpu_line = "" if resources.gpus == 0 else f"#SBATCH --gres=gpu:{resources.gpus}\n"
    runtime_guard = (
        "python -c "
        + shlex.quote(
            "import pathlib,sys,torch; "
            f"assert torch.__version__ == {expected_torch!r}, torch.__version__; "
            f"assert pathlib.Path(sys.prefix).resolve() == pathlib.Path({str(paths.environment)!r}).resolve(), sys.prefix"
        )
    )
    preflight = ""
    if resources.gpus:
        preflight = python_command(
            paths,
            "scripts/bootstrap/preflight.py",
            "--mode",
            "cluster",
            "--model-root",
            paths.model_root,
            "--expected-torch",
            expected_torch,
            "--min-gpus",
            str(resources.gpus),
            "--min-gpu-memory-gib",
            "40",
            "--output",
            paths.results / "preflight" / f"{job.name}.json",
        )
    body = "\n".join(job.commands)
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=vlm_{job.name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
{gpu_line}#SBATCH --cpus-per-task={resources.cpus}
#SBATCH --mem={resources.memory}
#SBATCH --time={resources.walltime}
#SBATCH --output={paths.run_root}/logs/{job.name}_%j.out
#SBATCH --error={paths.run_root}/logs/{job.name}_%j.err

set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load cuda/11.8
source {shlex.quote(str(paths.environment / 'bin' / 'activate'))}
cd {shlex.quote(str(paths.project))}
test "$(git rev-parse HEAD)" = {shlex.quote(expected_commit)}
test -z "$(git status --porcelain)"
{runtime_guard}
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
{preflight}
{body}
"""


def build_jobs(
    paths: Paths,
    *,
    fetch_prefeval: bool,
    include_ablations: bool,
    set_only_train: Path | None,
    set_only_dev: Path | None,
    include_prefeval_adaptation: bool,
) -> tuple[list[Job], dict[str, Any]]:
    jobs: list[Job] = []
    selected_env = paths.run_root / "config" / "selected_pilot.env"
    candidate_specification = paths.run_root / "config" / "pilot_candidates.json"
    generate_set_only = include_ablations and set_only_train is None and set_only_dev is None
    if generate_set_only:
        set_only_train = paths.synthetic_set_only / "train.jsonl"
        set_only_dev = paths.synthetic_set_only / "dev.jsonl"
    add_data_jobs(
        jobs,
        paths,
        fetch_prefeval=fetch_prefeval,
        generate_set_only=generate_set_only,
    )
    add_gate_jobs(jobs, paths)
    add_pilot_jobs(jobs, paths, candidate_specification, selected_env)
    add_full_jobs(
        jobs,
        paths,
        selected_env=selected_env,
        include_ablations=include_ablations,
        set_only_train=set_only_train,
        set_only_dev=set_only_dev,
        include_prefeval_adaptation=include_prefeval_adaptation,
    )
    add_evaluation_jobs(
        jobs,
        paths,
        include_ablations=include_ablations,
        include_prefeval_adaptation=include_prefeval_adaptation,
    )
    candidates = []
    for learning_rate in ("3e-5", "1e-4", "3e-4"):
        directory = paths.outputs / "pilots" / f"lr_{learning_rate}"
        candidates.append(
            {
                "learning_rate": float(learning_rate),
                "candidate_dir": str(directory),
                "summary": str(directory / "summary.json"),
                "predictions": str(directory / "dev_predictions.jsonl"),
                "checkpoint": str(directory / "best.pt"),
                "resume_checkpoint": str(directory / "checkpoint-000100.pt"),
            }
        )
    pilot_specification = {
        "schema_version": 1,
        "selection_split": "dev",
        "blank_predictions": str(paths.outputs / "pilot_baselines" / "blank_dev.jsonl"),
        "frozen_predictions": str(paths.outputs / "pilot_baselines" / "frozen_dev.jsonl"),
        "candidates": candidates,
    }
    validate_jobs(jobs)
    return jobs, pilot_specification


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or submit the strict post-G6 experiment DAG")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_runs/vision-language-memory"),
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_envs/vision_memory_py310_cu118_torch271"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_models/vision-language-memory"),
    )
    parser.add_argument(
        "--prefeval-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_runs/vision-language-memory/data/external/PrefEval"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_runs/vision-language-memory-runs"),
    )
    parser.add_argument("--partition", default="a800")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-torch", default="2.7.1+cu118")
    parser.add_argument("--through", choices=tuple(STAGE_ORDER), default="eval")
    parser.add_argument("--run-name")
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--fetch-prefeval", action="store_true")
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--include-prefeval-adaptation", action="store_true")
    parser.add_argument("--set-only-train", type=Path)
    parser.add_argument("--set-only-dev", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="Actually call sbatch; default is dry-run generation")
    mode.add_argument("--dry-run", action="store_true", help="Explicitly select the default no-submit mode")
    args = parser.parse_args()

    if (args.set_only_train is None) != (args.set_only_dev is None):
        raise SystemExit("--set-only-train and --set-only-dev must be supplied together.")
    if args.set_only_train is not None and not args.include_ablations:
        raise SystemExit("Explicit set-only paths require --include-ablations.")
    set_only_train = None if args.set_only_train is None else args.set_only_train.expanduser().resolve()
    set_only_dev = None if args.set_only_dev is None else args.set_only_dev.expanduser().resolve()
    if args.include_ablations and args.set_only_train is None:
        set_only_status = "planned from separately generated synthetic_set_only_v2"
    elif args.include_ablations:
        set_only_status = "planned from explicit independent dataset"
    else:
        set_only_status = "omitted: --include-ablations not selected"

    project = args.project_root.expanduser().resolve(strict=True)
    commit = run_text("git", "rev-parse", "HEAD", cwd=project)
    if args.expected_commit and commit != args.expected_commit:
        raise SystemExit(f"Commit mismatch: expected {args.expected_commit}, found {commit}")
    if args.submit and not args.expected_commit:
        raise SystemExit("Actual submission requires an explicit --expected-commit.")
    status = run_text("git", "status", "--porcelain", cwd=project)
    if status:
        raise SystemExit("Experiment DAG refuses a dirty project checkout.")

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_name = args.run_name or f"experiment-{stamp}-{commit[:8]}"
    if "/" in run_name or "\\" in run_name or run_name in {"", ".", ".."}:
        raise SystemExit("--run-name must be one safe path component.")
    run_root = args.runs_root.expanduser().resolve() / run_name
    if run_root.exists():
        raise SystemExit(f"Refusing to reuse an existing run directory: {run_root}")
    for directory in (run_root / "sbatch", run_root / "logs", run_root / "results", run_root / "config"):
        directory.mkdir(parents=True, exist_ok=False if directory == run_root / "sbatch" else True)

    paths = Paths(
        project=project,
        environment=args.environment.expanduser().resolve(),
        model_root=args.model_root.expanduser().resolve(),
        prefeval_root=args.prefeval_root.expanduser().resolve(),
        run_root=run_root,
    )
    configuration = load_configuration(args.config_json)
    resources = configured_resources(configuration)
    jobs, pilot_specification = build_jobs(
        paths,
        fetch_prefeval=args.fetch_prefeval,
        include_ablations=args.include_ablations,
        set_only_train=set_only_train,
        set_only_dev=set_only_dev,
        include_prefeval_adaptation=args.include_prefeval_adaptation,
    )
    jobs = apply_command_overrides(jobs, configuration)
    validate_jobs(jobs)
    selected_jobs = [job for job in jobs if STAGE_ORDER[job.stage] <= STAGE_ORDER[args.through]]
    selected_names = {job.name for job in selected_jobs}
    for job in selected_jobs:
        if not set(job.dependencies).issubset(selected_names):
            raise RuntimeError(f"Selected job {job.name} has an excluded dependency.")

    write_json_atomic(run_root / "config" / "pilot_candidates.json", pilot_specification)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dry_run": not args.submit,
        "status": "generating",
        "project": str(project),
        "commit": commit,
        "expected_torch": args.expected_torch,
        "environment": str(paths.environment),
        "model_root": str(paths.model_root),
        "prefeval_root": str(paths.prefeval_root),
        "prefeval_revision": PREFEVAL_REVISION,
        "run_root": str(run_root),
        "through": args.through,
        "include_ablations": args.include_ablations,
        "include_prefeval_adaptation": args.include_prefeval_adaptation,
        "config_json": None if args.config_json is None else str(args.config_json.resolve()),
        "command_overrides": sorted(configuration["command_overrides"]),
        "set_only_ablation": {
            "status": set_only_status,
            "train": str(
                set_only_train
                or (paths.synthetic_set_only / "train.jsonl" if args.include_ablations else "")
            ),
            "dev": str(
                set_only_dev
                or (paths.synthetic_set_only / "dev.jsonl" if args.include_ablations else "")
            ),
        },
        "resources": {name: asdict(value) for name, value in resources.items()},
        "jobs": {},
    }
    manifest_path = run_root / "submission.json"
    write_json_atomic(manifest_path, manifest)
    job_ids: dict[str, str] = {}
    for job in selected_jobs:
        resource = resources[job.resource_class]
        sbatch_path = run_root / "sbatch" / f"{job.name}.sbatch"
        sbatch_path.write_text(
            render_sbatch(
                job,
                resource,
                paths=paths,
                partition=args.partition,
                expected_commit=commit,
                expected_torch=args.expected_torch,
            ),
            encoding="utf-8",
        )
        record = {
            "stage": job.stage,
            "resource_class": job.resource_class,
            "resources": asdict(resource),
            "dependencies": list(job.dependencies),
            "dependency_job_ids": [job_ids[name] for name in job.dependencies if name in job_ids],
            "sbatch": str(sbatch_path),
            "job_id": None,
            "status": "generated",
        }
        manifest["jobs"][job.name] = record
        write_json_atomic(manifest_path, manifest)
        if args.submit:
            dependency_ids = [job_ids[name] for name in job.dependencies]
            command = ["sbatch", "--parsable", "--kill-on-invalid-dep=yes"]
            if dependency_ids:
                command.append(f"--dependency=afterok:{':'.join(dependency_ids)}")
            command.append(str(sbatch_path))
            try:
                job_id = run_text(*command).split(";", 1)[0]
            except Exception as exc:
                record["status"] = "submission_failed"
                record["error"] = repr(exc)
                manifest["status"] = "partial_failure"
                write_json_atomic(manifest_path, manifest)
                raise
            job_ids[job.name] = job_id
            record["job_id"] = job_id
            record["dependency_job_ids"] = dependency_ids
            record["status"] = "submitted"
            write_json_atomic(manifest_path, manifest)

    manifest["status"] = "submitted" if args.submit else "dry_run_complete"
    write_json_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
