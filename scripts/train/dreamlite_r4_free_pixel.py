"""Teacher-free R4-FreePixel transition training for DreamLite-mobile.

R4 is intentionally separate from the historical R3 episode trainer.  Its
persistent recurrent state is a real RGB tensor, prefix replay is no-grad, and
only one target event receives gradient on each micro-transition.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train.dreamlite_episode import (  # noqa: E402
    append_jsonl,
    assert_frozen_contract,
    choice_reader_callable,
    compute_dtype,
    git_value,
    locked_revision,
    set_all_seeds,
    sha256_file,
    begin_optimizer_diagnostics,
    finalize_optimizer_diagnostics_after_step,
    record_optimizer_diagnostics_after_clip,
    grouped_tensor_norms,
)
from scripts.inspire.model_snapshot_manifest import (  # noqa: E402
    SNAPSHOT_MANIFEST_NAME,
    verify_snapshot_binding,
    verify_snapshot_manifest,
)
from vision_memory.data import CYCLIC4, REVERSE_CYCLIC4, read_jsonl as read_episode_jsonl  # noqa: E402
from vision_memory.dreamlite import DreamLiteRecurrentUpdater, freeze_module  # noqa: E402
from vision_memory.reader import R3_QWEN_READER_RESIZE_CONTRACT  # noqa: E402
from vision_memory.repro import configure_strict_cuda_determinism, load_initial_image  # noqa: E402
from vision_memory.training import format_mcq_query, load_training_checkpoint, save_training_checkpoint  # noqa: E402
from vision_memory.training.r4_transition import (  # noqa: E402
    R4_DIFFUSION_STEPS,
    R4_EVENT_KINDS,
    R4ScheduleCursor,
    ScheduledTransition,
    TransitionExample,
    build_transition_index,
    iter_balanced_schedule,
    make_r4_manifest_contract,
    target_update_kwargs,
    trainable_transition_examples,
    transition_index_audit,
    validate_r4_manifest_contract,
    validate_teacher_free_bindings,
)


R4_METRICS_SCHEMA = "vision_memory.r4-free-pixel-training-metrics.v1"
R4_SUMMARY_SCHEMA = "vision_memory.r4-free-pixel-training-summary.v1"
R4_CHECKPOINT_STATE_SCHEMA = "vision_memory.r4-free-pixel-checkpoint-state.v1"
R4_IDENTITY_CALIBRATION_SCHEMA = "vision_memory.r4-noop-identity-calibration.v1"
R4_WARMUP_SCHEMA = "vision_memory.r4-set-from-blank-warmup.v1"
R4_MICRO_METRICS_SCHEMA = "vision_memory.r4-free-pixel-micro-metrics.v1"
R4_PROFILE = "mechanism-rescue-v1"

FORMAL_OPTIMIZER_STEPS = 256
FORMAL_WARMUP_OPTIMIZER_STEPS = 32
FORMAL_BALANCED_OPTIMIZER_STEPS = 224
GRADIENT_ACCUMULATION = 8
LORA_RANK = 4
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
FORMAL_EVAL_LIMIT = 8
FORMAL_CALIBRATION_COUNT = 8
FORMAL_CHECKPOINT_EVERY = 16
FORMAL_TRAIN_SHA256 = "24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184"
FORMAL_DEV_SHA256 = "8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303"
FORMAL_TRANSITION_INDEX_SHA256 = "2b96ae090955c25092b0d415b5b5a7e36a3e9f9fc91bf693725c197ae94f4964"


@dataclass(frozen=True)
class RunBudget:
    scope: str
    optimizer_steps: int
    warmup_optimizer_steps: int
    balanced_optimizer_steps: int
    gradient_accumulation: int
    eval_limit: int
    identity_calibration_count: int
    checkpoint_every: int

    @property
    def total_micro_transitions(self) -> int:
        return self.optimizer_steps * self.gradient_accumulation

    @property
    def warmup_micro_transitions(self) -> int:
        return self.warmup_optimizer_steps * self.gradient_accumulation

    @property
    def balanced_micro_transitions(self) -> int:
        return self.balanced_optimizer_steps * self.gradient_accumulation

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": R4_PROFILE,
            "scope": self.scope,
            "optimizer_steps": self.optimizer_steps,
            "warmup_optimizer_steps": self.warmup_optimizer_steps,
            "balanced_optimizer_steps": self.balanced_optimizer_steps,
            "gradient_accumulation": self.gradient_accumulation,
            "total_micro_transitions": self.total_micro_transitions,
            "warmup_micro_transitions": self.warmup_micro_transitions,
            "balanced_micro_transitions": self.balanced_micro_transitions,
            "eval_limit": self.eval_limit,
            "identity_calibration_count": self.identity_calibration_count,
            "checkpoint_every": self.checkpoint_every,
            "endpoint_selection": "fixed_final_only",
            "intermediate_dev_selection": False,
        }


@dataclass(frozen=True)
class TrainingUnit:
    global_micro_index: int
    phase: str
    phase_unit_index: int
    selected_step_index: int
    transition: TransitionExample
    selected_step_count: int = 1
    balanced_schedule: ScheduledTransition | None = None

    def updater_kwargs(self) -> dict[str, Any]:
        selected = _selected_step_indices(self.selected_step_index, self.selected_step_count)
        if self.balanced_schedule is None:
            kwargs: dict[str, Any] = {
                "gradient_mode": "drtune",
                "selected_step_indices": selected,
                "persistent_state": "float_rgb",
            }
        else:
            kwargs = {**target_update_kwargs(self.balanced_schedule), "selected_step_indices": selected}
        expected = {
            "gradient_mode": "drtune",
            "selected_step_indices": selected,
            "persistent_state": "float_rgb",
        }
        if kwargs != expected:
            raise RuntimeError(f"R4 target updater contract drifted: expected={expected}, observed={kwargs}")
        return {**kwargs, "presentation_index": self.global_micro_index}

    def receipt(self) -> dict[str, Any]:
        if self.balanced_schedule is None:
            return {
                "schedule_schema": R4_WARMUP_SCHEMA,
                "global_micro_index": self.global_micro_index,
                "phase_unit_index": self.phase_unit_index,
                "transition_id": self.transition.transition_id,
                "event_kind": self.transition.event_kind,
                "selected_step_index": self.selected_step_index,
            }
        return {**self.balanced_schedule.receipt(), "global_micro_index": self.global_micro_index}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Teacher-free R4-FreePixel DreamLite transition training")
    parser.add_argument("--experimental", action="store_true", help="Allow short research arms with explicit hyperparameters.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-seed", type=int, default=0)
    parser.add_argument("--schedule-seed", type=int, default=0)
    parser.add_argument("--max-optimizer-steps", type=int, default=FORMAL_OPTIMIZER_STEPS)
    parser.add_argument(
        "--set-warmup-optimizer-steps",
        type=int,
        default=FORMAL_WARMUP_OPTIMIZER_STEPS,
    )
    parser.add_argument("--eval-limit", type=int, default=FORMAL_EVAL_LIMIT)
    parser.add_argument("--identity-calibration-count", type=int, default=FORMAL_CALIBRATION_COUNT)
    parser.add_argument("--checkpoint-every", type=int, default=FORMAL_CHECKPOINT_EVERY)
    parser.add_argument("--checkpoint-unet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gradient-accumulation", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--gradient-clip", type=float, default=GRADIENT_CLIP)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--selected-step-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--optimizer-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-micro-metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-step-zero", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--record-validation-metrics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--validation-every", type=int)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def resolve_budget(args: argparse.Namespace) -> RunBudget:
    values = {
        "max_optimizer_steps": args.max_optimizer_steps,
        "set_warmup_optimizer_steps": args.set_warmup_optimizer_steps,
        "eval_limit": args.eval_limit,
        "identity_calibration_count": args.identity_calibration_count,
        "checkpoint_every": args.checkpoint_every,
        "gradient_accumulation": getattr(args, "gradient_accumulation", GRADIENT_ACCUMULATION),
    }
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values.values()):
        raise ValueError("R4 budget fields must be integers.")
    if any(value <= 0 for key, value in values.items() if key != "set_warmup_optimizer_steps"):
        raise ValueError("R4 budget fields other than warmup must be positive.")
    if args.set_warmup_optimizer_steps < 0:
        raise ValueError("SET warmup optimizer steps cannot be negative.")
    if args.set_warmup_optimizer_steps > args.max_optimizer_steps:
        raise ValueError("SET warmup cannot exceed the optimizer endpoint.")
    if getattr(args, "smoke", False):
        if args.max_optimizer_steps != 2 or args.set_warmup_optimizer_steps != 0:
            raise ValueError("R4 smoke requires --max-optimizer-steps 2 --set-warmup-optimizer-steps 0.")
        scope = "technical_smoke"
    elif getattr(args, "experimental", False):
        scope = "experimental"
    else:
        formal = {
            "max_optimizer_steps": FORMAL_OPTIMIZER_STEPS,
            "set_warmup_optimizer_steps": FORMAL_WARMUP_OPTIMIZER_STEPS,
            "eval_limit": FORMAL_EVAL_LIMIT,
            "identity_calibration_count": FORMAL_CALIBRATION_COUNT,
            "checkpoint_every": FORMAL_CHECKPOINT_EVERY,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
        }
        mismatches = {name: (formal[name], values[name]) for name in formal if formal[name] != values[name]}
        if mismatches:
            raise ValueError(f"Formal R4 profile is fixed; argument mismatches: {mismatches}")
        scope = "mechanism_rescue"
    budget = RunBudget(
        scope=scope,
        optimizer_steps=args.max_optimizer_steps,
        warmup_optimizer_steps=args.set_warmup_optimizer_steps,
        balanced_optimizer_steps=args.max_optimizer_steps - args.set_warmup_optimizer_steps,
        gradient_accumulation=values["gradient_accumulation"],
        eval_limit=args.eval_limit,
        identity_calibration_count=args.identity_calibration_count,
        checkpoint_every=args.checkpoint_every,
    )
    if budget.balanced_micro_transitions % 16:
        raise ValueError("Balanced R4 phase must contain complete 16-unit event-kind x diffusion-step blocks.")
    if budget.warmup_micro_transitions % 4:
        raise ValueError("SET warmup must contain complete four-step cycles.")
    return budget


def validate_runtime_args(args: argparse.Namespace) -> RunBudget:
    budget = resolve_budget(args)
    if budget.scope == "mechanism_rescue" and not args.strict_determinism:
        raise ValueError("Formal R4 requires --strict-determinism.")
    if args.resolution != 1024:
        raise ValueError("R4-FreePixel locks 1024x1024 RGB state and the Reader resize contract.")
    if torch.device(args.dreamlite_device) == torch.device(args.reader_device):
        raise ValueError("DreamLite and the frozen Reader must use distinct devices.")
    validation_every = getattr(args, "validation_every", None)
    if validation_every is not None and (isinstance(validation_every, bool) or not isinstance(validation_every, int) or validation_every <= 0):
        raise ValueError("--validation-every must be a positive integer when provided.")
    return budget


def validate_formal_data_contract(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    audit: Mapping[str, Any],
    transition_digest: str,
) -> None:
    if budget.scope != "mechanism_rescue":
        return
    observed_hashes = {"train": sha256_file(args.train), "dev": sha256_file(args.dev)}
    expected_hashes = {"train": FORMAL_TRAIN_SHA256, "dev": FORMAL_DEV_SHA256}
    if observed_hashes != expected_hashes:
        raise ValueError(f"Formal R4 dataset SHA256 mismatch: {observed_hashes}")
    expected = {
        "total_transitions": 12500,
        "trainable_transitions": 11876,
        "prefix_only_transitions": 624,
        "local_query_count": 12496,
        "by_event_kind": {"set": 7504, "overwrite": 1872, "clear": 624, "noop": 2500},
        "trainable_by_event_kind": {"set": 6880, "overwrite": 1872, "clear": 624, "noop": 2500},
        "by_objective": {
            "qa_only": 9376,
            "identity_only": 1876,
            "qa_and_identity": 624,
            "prefix_only": 624,
        },
    }
    mismatches = {
        key: {"expected": value, "observed": audit.get(key)}
        for key, value in expected.items()
        if audit.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Formal R4 transition-index contract mismatch: {mismatches}")
    if transition_digest != FORMAL_TRANSITION_INDEX_SHA256:
        raise ValueError(f"Formal R4 canonical transition digest mismatch: {transition_digest}")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _warmup_pool(examples: Iterable[TransitionExample]) -> tuple[TransitionExample, ...]:
    pool = tuple(
        example
        for example in examples
        if example.event_kind == "set"
        and example.is_trainable
        and example.uses_qa_loss
        and not example.prefix_updater_indices
    )
    if not pool:
        raise ValueError("R4 SET-from-blank warmup requires a trainable SET target with no updater prefix.")
    return pool


def _stable_warmup_order(
    pool: Sequence[TransitionExample],
    *,
    seed: int,
    epoch: int,
) -> tuple[TransitionExample, ...]:
    return tuple(
        sorted(
            pool,
            key=lambda example: (
                hashlib.sha256(f"{R4_WARMUP_SCHEMA}\x1f{seed}\x1f{epoch}\x1f{example.transition_id}".encode()).digest(),
                example.transition_id,
            ),
        )
    )


def training_unit_for_micro(
    examples: Sequence[TransitionExample],
    *,
    schedule_seed: int,
    global_micro_index: int,
    warmup_micro_transitions: int,
    selected_step_count: int = 1,
) -> TrainingUnit:
    if global_micro_index < 0:
        raise ValueError("global_micro_index must be non-negative.")
    if selected_step_count not in (1, 2):
        raise ValueError("selected_step_count must be 1 or 2.")
    if global_micro_index < warmup_micro_transitions:
        pool = _warmup_pool(examples)
        epoch, offset = divmod(global_micro_index, len(pool))
        return TrainingUnit(
            global_micro_index=global_micro_index,
            phase="set_from_blank_warmup",
            phase_unit_index=global_micro_index,
            selected_step_index=R4_DIFFUSION_STEPS[global_micro_index % len(R4_DIFFUSION_STEPS)],
            transition=_stable_warmup_order(pool, seed=schedule_seed, epoch=epoch)[offset],
            selected_step_count=selected_step_count,
        )
    balanced_index = global_micro_index - warmup_micro_transitions
    scheduled = next(
        iter_balanced_schedule(
            examples,
            seed=schedule_seed,
            start_unit_index=balanced_index,
            num_units=1,
        )
    )
    return TrainingUnit(
        global_micro_index=global_micro_index,
        phase="balanced_transition",
        phase_unit_index=balanced_index,
        selected_step_index=scheduled.selected_step_index,
        transition=scheduled.transition,
        balanced_schedule=scheduled,
        selected_step_count=selected_step_count,
    )


def _selected_step_indices(selected_step_index: int, selected_step_count: int) -> tuple[int, ...]:
    if selected_step_count == 1:
        return (selected_step_index,)
    if selected_step_count == 2:
        return (0, 2) if selected_step_index % 2 == 0 else (1, 3)
    raise ValueError("selected_step_count must be 1 or 2.")


def expected_schedule_counts(budget: RunBudget) -> dict[str, dict[str, int]]:
    if budget.balanced_micro_transitions % 16 or budget.warmup_micro_transitions % 4:
        raise ValueError("Expected counts require complete balanced blocks and warmup step cycles.")
    repeats = budget.balanced_micro_transitions // 16
    kind_count = repeats * len(R4_DIFFUSION_STEPS)
    step_count = repeats * len(R4_EVENT_KINDS) + budget.warmup_micro_transitions // 4
    by_kind = {kind: kind_count for kind in R4_EVENT_KINDS}
    by_kind["set"] += budget.warmup_micro_transitions
    return {
        "by_event_kind": by_kind,
        "by_diffusion_step": {str(step): step_count for step in R4_DIFFUSION_STEPS},
        "balanced_by_event_kind_step": {
            f"{kind}:{step}": repeats for kind in R4_EVENT_KINDS for step in R4_DIFFUSION_STEPS
        },
    }


def _field(value: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _enum_text(value: Any) -> str | None:
    value = getattr(value, "value", value)
    return None if value is None else str(value)


def _save_checkpoint(
    path: Path,
    *,
    model: R4FreePixelModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    manifest: Mapping[str, Any],
    budget: RunBudget,
    schedule_seed: int,
    identity_scale: float,
    m0: Mapping[str, Any],
) -> Path:
    return save_training_checkpoint(
        path,
        trainable_module=model,
        optimizer=optimizer,
        epoch=0,
        episode_cursor=step * budget.gradient_accumulation,
        optimizer_step=step,
        manifest=manifest,
        trainer_state=checkpoint_trainer_state(
            optimizer_step=step,
            budget=budget,
            schedule_seed=schedule_seed,
            identity_scale=identity_scale,
            m0_evaluation=m0,
        ),
    )


def _run_training(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    model: R4FreePixelModel,
    pipe: Any,
    reader: nn.Module,
    examples: Sequence[TransitionExample],
    dev_records: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    named_trainable: Sequence[tuple[str, nn.Parameter]],
    trainable: Sequence[nn.Parameter],
    reader_fn: Any,
    eval_reader: Any,
    manifest: Mapping[str, Any],
    identity_scale: float,
    m0: Mapping[str, Any],
    start_step: int,
    prior_elapsed: float,
) -> float:
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for step_zero in range(start_step, budget.optimizer_steps):
        metric = _run_one_step(
            args=args,
            budget=budget,
            step_zero=step_zero,
            model=model,
            pipe=pipe,
            reader=reader,
            examples=examples,
            optimizer=optimizer,
            named_trainable=named_trainable,
            trainable=trainable,
            reader_fn=reader_fn,
            identity_scale=identity_scale,
            elapsed=0.0,
        )
        elapsed = prior_elapsed + time.monotonic() - started
        metric["elapsed_seconds"] = elapsed
        micro_records = metric.pop("micro_records", None)
        if micro_records:
            for micro_record in micro_records:
                append_jsonl(args.output_dir / "micro_metrics.jsonl", micro_record)
        append_jsonl(args.output_dir / "metrics.jsonl", metric)
        print(
            json.dumps(
                {
                    "milestone": "optimizer_step",
                    "optimizer_step": metric["optimizer_step"],
                    "phase_counts": metric["phase_counts"],
                    "event_kind_counts": metric["event_kind_counts"],
                    "loss_mean": metric["loss_mean"],
                    "gradient_norm": metric["gradient_norm_before_clip"],
                    "elapsed_seconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if bool(getattr(args, "record_validation_metrics", False)):
            validation_every = int(getattr(args, "validation_every", budget.checkpoint_every) or budget.checkpoint_every)
            if (step_zero + 1) % validation_every == 0:
                was_training = model.training
                model.eval()
                try:
                    validation = evaluate_r4(model=model, records=dev_records, reader_fn=eval_reader)
                finally:
                    model.train(was_training)
                append_jsonl(
                    args.output_dir / "metrics.jsonl",
                    {
                        "schema": R4_METRICS_SCHEMA,
                        "kind": "validation",
                        "optimizer_step": step_zero + 1,
                        "validation_loss": validation["mean_episode_listwise_choice_ce"],
                        **validation,
                        "elapsed_seconds": elapsed,
                    },
                )
        _periodic_checkpoint(
            args=args,
            budget=budget,
            step=step_zero + 1,
            model=model,
            optimizer=optimizer,
            manifest=manifest,
            identity_scale=identity_scale,
            m0=m0,
        )
    return prior_elapsed + time.monotonic() - started


def _prepare_metrics_resume(path: Path, *, optimizer_step: int) -> float:
    if not path.is_file():
        return 0.0
    retained: list[str] = []
    elapsed = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        step = value.get("optimizer_step")
        if step is None or isinstance(step, int) and step <= optimizer_step:
            retained.append(json.dumps(value, ensure_ascii=False))
            if step == optimizer_step:
                elapsed = float(value.get("elapsed_seconds", elapsed))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    os.replace(temporary, path)
    return elapsed


def _prepare_micro_metrics_resume(path: Path, *, global_micro_index: int) -> None:
    if not path.is_file():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        index = value.get("global_micro_index")
        if isinstance(index, int) and index < global_micro_index:
            retained.append(json.dumps(value, ensure_ascii=False))
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    budget = _startup(args)
    determinism = configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    set_all_seeds(args.seed)
    data = _load_data(args, budget)
    static_manifest = make_static_manifest(
        args=args,
        budget=budget,
        all_transitions=data.all_transitions,
        transition_audit=data.audit,
        warmup_pool=data.warmup_pool,
        determinism_report=determinism,
    )
    print(
        json.dumps(
            {
                "milestone": "startup_validated",
                "scope": budget.scope,
                "optimizer_steps": budget.optimizer_steps,
                "micro_transitions": budget.total_micro_transitions,
                "transition_index_sha256": static_manifest["transition_index_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    runtime = _load_runtime(args)
    train_reader = _reader_fn(runtime, args, require_grad=True)
    eval_reader = _reader_fn(runtime, args, require_grad=False)
    state = _prepare_run_state(
        args=args,
        budget=budget,
        runtime=runtime,
        data=data,
        static_manifest=static_manifest,
        eval_fn=eval_reader,
    )
    print(
        json.dumps(
            {
                "milestone": "m0_ready",
                "start_step": state.start_step,
                "identity_scale": state.identity_scale,
                "m0_episode_ce": state.m0["mean_episode_listwise_choice_ce"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats(runtime.updater_device)
    torch.cuda.reset_peak_memory_stats(runtime.reader_device)
    elapsed = _run_training(
        args=args,
        budget=budget,
        model=runtime.model,
        pipe=runtime.pipe,
        reader=runtime.reader,
        examples=data.trainable,
        dev_records=getattr(data, "dev_records", ()),
        optimizer=runtime.optimizer,
        named_trainable=runtime.named_trainable,
        trainable=runtime.trainable,
        reader_fn=train_reader,
        eval_reader=eval_reader,
        manifest=state.manifest,
        identity_scale=state.identity_scale,
        m0=state.m0,
        start_step=state.start_step,
        prior_elapsed=state.prior_elapsed,
    )
    endpoint, final = _finish_evaluation(
        args=args,
        budget=budget,
        runtime=runtime,
        data=data,
        state=state,
        eval_fn=eval_reader,
        elapsed=elapsed,
    )
    last = _save_last(args=args, budget=budget, runtime=runtime, state=state)
    _write_summary(
        args=args,
        budget=budget,
        runtime=runtime,
        state=state,
        final=final,
        endpoint=endpoint,
        last=last,
        elapsed=elapsed,
    )
    print(
        json.dumps(
            {
                "milestone": "completed",
                "optimizer_steps": budget.optimizer_steps,
                "final_episode_ce": final["mean_episode_listwise_choice_ce"],
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _write_summary(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    runtime: RuntimeBundle,
    state: ResumeState,
    final: Mapping[str, Any],
    endpoint: Path,
    last: Path,
    elapsed: float,
) -> None:
    metric_name = "mean_episode_listwise_choice_ce"
    summary = {
        "schema": R4_SUMMARY_SCHEMA,
        "status": "completed",
        "profile": R4_PROFILE,
        "scope": budget.scope,
        "optimizer_steps": budget.optimizer_steps,
        "micro_transitions": budget.total_micro_transitions,
        "expected_schedule_counts": expected_schedule_counts(budget),
        "fixed_evaluation_metric": metric_name,
        "m0_episode_listwise_choice_ce": state.m0[metric_name],
        "final_episode_listwise_choice_ce": final[metric_name],
        "episode_listwise_choice_ce_delta": final[metric_name] - state.m0[metric_name],
        "identity_calibration_scale": state.identity_scale,
        "elapsed_seconds": elapsed,
        "updater_peak_memory_gib": _peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": _peak_gib(runtime.reader_device),
        "endpoint_checkpoint_sha256": sha256_file(endpoint),
        "last_checkpoint_sha256": sha256_file(last),
        "checkpoint_selection": "fixed_endpoint_not_best_dev",
    }
    _write_json(args.output_dir / "summary.json", summary)


def _save_last(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    runtime: RuntimeBundle,
    state: ResumeState,
) -> Path:
    return _save_checkpoint(
        args.output_dir / "last.pt",
        model=runtime.model,
        optimizer=runtime.optimizer,
        step=budget.optimizer_steps,
        manifest=state.manifest,
        budget=budget,
        schedule_seed=args.schedule_seed,
        identity_scale=state.identity_scale,
        m0=state.m0,
    )


def _finish_evaluation(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    runtime: RuntimeBundle,
    data: DataBundle,
    state: ResumeState,
    eval_fn: Any,
    elapsed: float,
) -> tuple[Path, dict[str, Any]]:
    endpoint = _save_checkpoint(
        args.output_dir / "endpoint.pt",
        model=runtime.model,
        optimizer=runtime.optimizer,
        step=budget.optimizer_steps,
        manifest=state.manifest,
        budget=budget,
        schedule_seed=args.schedule_seed,
        identity_scale=state.identity_scale,
        m0=state.m0,
    )
    final = evaluate_r4(model=runtime.model, records=data.dev_records, reader_fn=eval_fn)
    print(json.dumps({"milestone": "model_snapshot_end_verify_started"}), flush=True)
    snapshot_end = _verify_end_snapshot_bindings(state.manifest)
    _write_json(args.output_dir / "model_snapshot_verification_end.json", snapshot_end)
    print(json.dumps({"milestone": "model_snapshot_end_verify_passed"}), flush=True)
    _write_json(
        args.output_dir / "evaluation.json",
        {
            "schema": "vision_memory.r4-fixed-endpoint-evaluation.v1",
            "selection_policy": "fixed_m0_and_final_only_no_best_dev",
            "m0": state.m0,
            "final": final,
            "model_snapshot_end_verification": snapshot_end,
        },
    )
    append_jsonl(
        args.output_dir / "metrics.jsonl",
        {
            "schema": R4_METRICS_SCHEMA,
            "kind": "fixed_evaluation_final",
            "optimizer_step": budget.optimizer_steps,
            **final,
            "elapsed_seconds": elapsed,
        },
    )
    return endpoint, final


def _prepare_run_state(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    runtime: RuntimeBundle,
    data: DataBundle,
    static_manifest: Mapping[str, Any],
    eval_fn: Any,
) -> ResumeState:
    evaluation_path = args.output_dir / "evaluation.json"
    if args.resume is not None:
        state = _resume_state(
            args=args,
            budget=budget,
            runtime=runtime,
            static_manifest=static_manifest,
        )
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        if evaluation.get("m0") != state.m0:
            raise ValueError("R4 resume M0 differs between checkpoint and evaluation.json.")
        return state
    manifest, scale = _fresh_manifest(
        args=args,
        budget=budget,
        runtime=runtime,
        static_manifest=static_manifest,
        examples=data.trainable,
    )
    m0 = evaluate_r4(model=runtime.model, records=data.dev_records, reader_fn=eval_fn)
    _write_json(
        evaluation_path,
        {
            "schema": "vision_memory.r4-fixed-endpoint-evaluation.v1",
            "selection_policy": "fixed_m0_and_final_only_no_best_dev",
            "m0": m0,
            "final": None,
        },
    )
    append_jsonl(
        args.output_dir / "metrics.jsonl",
        {
            "schema": R4_METRICS_SCHEMA,
            "kind": "fixed_evaluation_m0",
            "optimizer_step": 0,
            **m0,
            "elapsed_seconds": 0.0,
        },
    )
    if bool(getattr(args, "save_step_zero", False)):
        _save_checkpoint(
            args.output_dir / "checkpoints" / "step-000000.pt",
            model=runtime.model,
            optimizer=runtime.optimizer,
            step=0,
            manifest=manifest,
            budget=budget,
            schedule_seed=args.schedule_seed,
            identity_scale=scale,
            m0=m0,
        )
    return ResumeState(manifest, scale, 0, 0.0, m0)


def _startup(args: argparse.Namespace) -> RunBudget:
    try:
        budget = validate_runtime_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("R4-FreePixel requires two visible CUDA GPUs.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise SystemExit("A fresh R4 run refuses a non-empty output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return budget


def _reader_fn(runtime: RuntimeBundle, args: argparse.Namespace, *, require_grad: bool) -> Any:
    return choice_reader_callable(
        reader=runtime.reader,
        processor=runtime.processor,
        reader_device=runtime.reader_device,
        require_grad=require_grad,
        deterministic_ce=args.strict_determinism,
    )


@dataclass(frozen=True)
class DataBundle:
    dev_records: list[Any]
    all_transitions: tuple[TransitionExample, ...]
    trainable: tuple[TransitionExample, ...]
    audit: dict[str, Any]
    warmup_pool: tuple[TransitionExample, ...]


def _load_data(args: argparse.Namespace, budget: RunBudget) -> DataBundle:
    train_records = read_episode_jsonl(args.train)
    dev_records = read_episode_jsonl(args.dev)[: budget.eval_limit]
    if not train_records or not dev_records:
        raise ValueError("R4 train and fixed dev subsets must both be non-empty.")
    all_transitions = build_transition_index(train_records)
    trainable = trainable_transition_examples(all_transitions)
    audit = transition_index_audit(all_transitions)
    validate_formal_data_contract(
        args=args,
        budget=budget,
        audit=audit,
        transition_digest=transition_index_digest(all_transitions),
    )
    warmup_pool = _warmup_pool(trainable)
    _write_json(args.output_dir / "transition_index_audit.json", audit)
    return DataBundle(dev_records, all_transitions, trainable, audit, warmup_pool)


def _fresh_manifest(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    runtime: RuntimeBundle,
    static_manifest: Mapping[str, Any],
    examples: Sequence[TransitionExample],
) -> tuple[dict[str, Any], float]:
    calibration = calibrate_identity_scale(
        model=runtime.model,
        examples=examples,
        schedule_seed=args.schedule_seed,
        count=budget.identity_calibration_count,
    )
    manifest = {**static_manifest, "identity_calibration": calibration}
    scale = _validate_identity_calibration(calibration, count=budget.identity_calibration_count)
    _write_json(args.output_dir / "manifest.json", manifest)
    _write_json(args.output_dir / "identity_calibration.json", calibration)
    _write_environment(args.output_dir / "environment.txt")
    set_all_seeds(args.seed)
    return manifest, scale


@dataclass(frozen=True)
class ResumeState:
    manifest: dict[str, Any]
    identity_scale: float
    start_step: int
    prior_elapsed: float
    m0: dict[str, Any]


def _resume_state(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    runtime: RuntimeBundle,
    static_manifest: Mapping[str, Any],
) -> ResumeState:
    assert args.resume is not None
    manifest = _load_resume_manifest(args.resume, static_manifest)
    calibration = manifest.get("identity_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("R4 resume manifest is missing identity_calibration.")
    scale = _validate_identity_calibration(calibration, count=budget.identity_calibration_count)
    payload = load_training_checkpoint(
        args.resume,
        trainable_module=runtime.model,
        optimizer=runtime.optimizer,
        expected_manifest=manifest,
    )
    step = int(payload["optimizer_step"])
    if step >= budget.optimizer_steps:
        raise RuntimeError("R4 resume checkpoint already reached the fixed endpoint.")
    trainer_state = payload.get("trainer_state")
    if not isinstance(trainer_state, Mapping):
        raise ValueError("R4 resume checkpoint is missing trainer_state.")
    validate_checkpoint_trainer_state(
        trainer_state,
        optimizer_step=step,
        budget=budget,
        schedule_seed=args.schedule_seed,
        identity_scale=scale,
    )
    if int(payload["episode_cursor"]) != step * budget.gradient_accumulation:
        raise ValueError("R4 resume micro-transition cursor mismatch.")
    elapsed = _prepare_metrics_resume(args.output_dir / "metrics.jsonl", optimizer_step=step)
    _prepare_micro_metrics_resume(
        args.output_dir / "micro_metrics.jsonl",
        global_micro_index=step * budget.gradient_accumulation,
    )
    return ResumeState(manifest, scale, step, elapsed, dict(trainer_state["m0_evaluation"]))


@dataclass(frozen=True)
class RuntimeBundle:
    pipe: Any
    processor: Any
    reader: nn.Module
    model: R4FreePixelModel
    optimizer: torch.optim.Optimizer
    named_trainable: list[tuple[str, nn.Parameter]]
    trainable: list[nn.Parameter]
    updater_device: torch.device
    reader_device: torch.device


def _load_runtime(args: argparse.Namespace) -> RuntimeBundle:
    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    pipe = _load_pipeline(args, updater_device, compute_dtype(updater_device))
    processor, reader = _load_reader(args, reader_device, compute_dtype(reader_device))
    model = R4FreePixelModel(
        pipeline=pipe,
        initial_rgb=_initial_rgb_tensor(
            resolution=args.resolution,
            device=updater_device,
            dtype=compute_dtype(updater_device),
        ),
        global_seed=args.seed,
        checkpoint_unet=args.checkpoint_unet,
    )
    named, trainable = _force_trainable_fp32(model)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(getattr(args, "learning_rate", LEARNING_RATE)),
        weight_decay=float(getattr(args, "weight_decay", WEIGHT_DECAY)),
    )
    return RuntimeBundle(
        pipe=pipe,
        processor=processor,
        reader=reader,
        model=model,
        optimizer=optimizer,
        named_trainable=named,
        trainable=trainable,
        updater_device=updater_device,
        reader_device=reader_device,
    )


def _load_reader(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> tuple[Any, nn.Module]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("R4 requires the fast tensor-native Qwen image processor.")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False
    return processor, reader


def _load_pipeline(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Any:
    from diffusers import DreamLiteMobilePipeline
    from peft import LoraConfig, get_peft_model

    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.dreamlite,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    pipe.unet.requires_grad_(False)
    set_all_seeds(args.adapter_seed)
    pipe.unet = get_peft_model(
        pipe.unet,
        LoraConfig(
            r=int(getattr(args, "lora_rank", LORA_RANK)),
            lora_alpha=int(getattr(args, "lora_rank", LORA_RANK)),
            lora_dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
    )
    pipe.unet.eval()
    return pipe


def _write_environment(path: Path) -> None:
    installed = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    path.write_text("\n".join(installed) + "\n", encoding="utf-8")


def _manifest_without_calibration(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "identity_calibration"}


def _load_resume_manifest(path: Path, static_manifest: Mapping[str, Any]) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("R4 resume checkpoint is missing its manifest.")
    if _manifest_without_calibration(manifest) != dict(static_manifest):
        raise ValueError("R4 resume static manifest differs from the current invocation.")
    return manifest


def _periodic_checkpoint(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    step: int,
    model: R4FreePixelModel,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    identity_scale: float,
    m0: Mapping[str, Any],
) -> None:
    if step % budget.checkpoint_every:
        return
    _save_checkpoint(
        args.output_dir / "checkpoints" / f"step-{step:06d}.pt",
        model=model,
        optimizer=optimizer,
        step=step,
        manifest=manifest,
        budget=budget,
        schedule_seed=args.schedule_seed,
        identity_scale=identity_scale,
        m0=m0,
    )


def _run_one_step(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    step_zero: int,
    model: R4FreePixelModel,
    pipe: Any,
    reader: nn.Module,
    examples: Sequence[TransitionExample],
    optimizer: torch.optim.Optimizer,
    named_trainable: Sequence[tuple[str, nn.Parameter]],
    trainable: Sequence[nn.Parameter],
    reader_fn: Any,
    identity_scale: float,
    elapsed: float,
) -> dict[str, Any]:
    outcomes: list[MicroOutcome] = []
    for accumulation_index in range(budget.gradient_accumulation):
        micro = step_zero * budget.gradient_accumulation + accumulation_index
        unit = training_unit_for_micro(
            examples,
            schedule_seed=args.schedule_seed,
            global_micro_index=micro,
            warmup_micro_transitions=budget.warmup_micro_transitions,
            selected_step_count=int(getattr(args, "selected_step_count", 1)),
        )
        outcomes.append(
            _micro_forward_backward(
                unit=unit,
                model=model,
                reader_fn=reader_fn,
                identity_scale=identity_scale,
                accumulation=budget.gradient_accumulation,
                named_trainable=named_trainable,
                record_micro_gradient=bool(getattr(args, "record_micro_metrics", False)),
            )
        )
    assert_frozen_contract(pipe, reader)
    diagnostic_snapshot = None
    diagnostic_report = None
    if bool(getattr(args, "optimizer_diagnostics", False)):
        diagnostic_snapshot, diagnostic_report = begin_optimizer_diagnostics(named_trainable)
    gradient_norm = _clip_gradients(
        named_trainable,
        trainable,
        max_norm=float(getattr(args, "gradient_clip", GRADIENT_CLIP)),
    )
    if diagnostic_report is not None:
        record_optimizer_diagnostics_after_clip(
            diagnostic_report,
            named_trainable,
            gradient_norm=gradient_norm,
            max_norm=float(getattr(args, "gradient_clip", GRADIENT_CLIP)),
        )
    optimizer.step()
    if diagnostic_report is not None and diagnostic_snapshot is not None:
        diagnostic_report = finalize_optimizer_diagnostics_after_step(
            diagnostic_report,
            named_trainable,
            diagnostic_snapshot,
        )
    optimizer.zero_grad(set_to_none=True)
    step = step_zero + 1
    balanced_next = max(0, step * budget.gradient_accumulation - budget.warmup_micro_transitions)
    return _step_metric(
        outcomes,
        step=step,
        balanced_next=balanced_next,
        gradient_norm=gradient_norm,
        gradient_accumulation=budget.gradient_accumulation,
        optimizer_diagnostics=diagnostic_report,
        include_micro_metrics=bool(getattr(args, "record_micro_metrics", False)),
        elapsed=elapsed,
        updater_device=torch.device(args.dreamlite_device),
        reader_device=torch.device(args.reader_device),
    )


def _step_metric(
    outcomes: Sequence[MicroOutcome],
    *,
    step: int,
    balanced_next: int,
    gradient_norm: float,
    gradient_accumulation: int,
    optimizer_diagnostics: Mapping[str, Any] | None,
    include_micro_metrics: bool,
    elapsed: float,
    updater_device: torch.device,
    reader_device: torch.device,
) -> dict[str, Any]:
    qa = [item.qa_loss for item in outcomes if item.qa_loss is not None]
    raw = [item.identity_raw for item in outcomes if item.identity_raw is not None]
    normalized = [item.identity_normalized for item in outcomes if item.identity_normalized is not None]
    return {
        "schema": R4_METRICS_SCHEMA,
        "kind": "optimizer_step",
        "optimizer_step": step,
        "next_global_micro_index": step * gradient_accumulation,
        "balanced_next_unit_index": balanced_next,
        "loss_mean": _mean([item.loss for item in outcomes]),
        "qa_loss_mean": _mean(qa),
        "identity_raw_smooth_l1_mean": _mean(raw),
        "identity_normalized_mean": _mean(normalized),
        "gradient_norm_before_clip": gradient_norm,
        "phase_counts": _counter(outcomes, "phase"),
        "event_kind_counts": _counter(outcomes, "event_kind"),
        "diffusion_step_counts": _counter(outcomes, "step_index"),
        "objective_counts": _counter(outcomes, "objective"),
        "rgb_min": min(item.rgb_min for item in outcomes),
        "rgb_max": max(item.rgb_max for item in outcomes),
        "rgb_saturation_fraction_mean": _mean([item.saturation for item in outcomes]),
        "rgb_rms_mean": _mean([item.rgb_rms for item in outcomes]),
        "rgb_delta_rms_mean": _mean([item.delta_rms for item in outcomes]),
        "schedule_receipts": [item.receipt for item in outcomes],
        "elapsed_seconds": elapsed,
        "updater_peak_memory_gib": _peak_gib(updater_device),
        "reader_peak_memory_gib": _peak_gib(reader_device),
        "optimizer_diagnostics": optimizer_diagnostics,
        "micro_records": [
            item.to_dict()
            for item in outcomes
        ] if include_micro_metrics else None,
    }


def _clip_gradients(
    named_trainable: Sequence[tuple[str, nn.Parameter]],
    trainable: Sequence[nn.Parameter],
    *,
    max_norm: float = GRADIENT_CLIP,
) -> float:
    observed = 0
    for name, parameter in named_trainable:
        if parameter.dtype is not torch.float32:
            raise RuntimeError(f"R4 adapter changed dtype: {name} -> {parameter.dtype}")
        if parameter.grad is not None:
            observed += 1
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"R4 adapter has non-finite gradient: {name}")
    if observed == 0:
        raise RuntimeError("R4 optimizer group has no adapter gradients.")
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("gradient clip must be finite and positive.")
    norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm)
    if not torch.isfinite(norm) or float(norm) <= 0:
        raise RuntimeError(f"R4 optimizer group has invalid gradient norm: {float(norm)}")
    return float(norm)


def _counter(items: Sequence[MicroOutcome], field: str) -> dict[str, int]:
    return dict(Counter(str(getattr(item, field)) for item in items))


def _micro_forward_backward(
    *,
    unit: TrainingUnit,
    model: R4FreePixelModel,
    reader_fn: Any,
    identity_scale: float,
    accumulation: int,
    named_trainable: Sequence[tuple[str, nn.Parameter]],
    record_micro_gradient: bool,
) -> MicroOutcome:
    source = _replay_prefix(
        model=model,
        transition=unit.transition,
        presentation_index=unit.global_micro_index,
    )
    output = _target_update(model, unit, source)
    qa = _qa_loss_for_transition(unit=unit, output_rgb=output, reader_loss_fn=reader_fn)
    identity_raw = _identity_loss(output, source) if unit.transition.uses_identity_loss else None
    identity = normalized_identity_loss(identity_raw, identity_scale) if identity_raw is not None else None
    total = combine_objective_losses(qa, identity)
    if not torch.isfinite(total):
        raise RuntimeError("R4 transition loss is non-finite.")
    previous_gradients = (
        {name: parameter.grad.detach().clone() if parameter.grad is not None else None
         for name, parameter in named_trainable}
        if record_micro_gradient
        else None
    )
    (total / accumulation).backward()
    micro_gradient_norms = None
    if previous_gradients is not None:
        micro_gradient_norms = grouped_tensor_norms(
            (
                name,
                None if parameter.grad is None else (
                    parameter.grad.detach() - (previous_gradients[name] if previous_gradients[name] is not None else 0)
                ),
            )
            for name, parameter in named_trainable
        )
    detached = output.detach().float()
    selected_step_indices = tuple(unit.updater_kwargs()["selected_step_indices"])
    delta = detached - source.detach().float()
    return MicroOutcome(
        loss=float(total.detach()),
        qa_loss=_optional_float(qa),
        identity_raw=_optional_float(identity_raw),
        identity_normalized=_optional_float(identity),
        phase=unit.phase,
        event_kind=unit.transition.event_kind,
        step_index=unit.selected_step_index,
        selected_step_indices=selected_step_indices,
        micro_gradient_norms=micro_gradient_norms,
        objective=unit.transition.objective.value,
        transition_id=unit.transition.transition_id,
        global_micro_index=unit.global_micro_index,
        rgb_min=float(detached.min()),
        rgb_max=float(detached.max()),
        saturation=float(((detached <= 1 / 255) | (detached >= 254 / 255)).float().mean()),
        rgb_rms=float(detached.square().mean().sqrt()),
        delta_rms=float(delta.square().mean().sqrt()),
        receipt=unit.receipt(),
    )


@dataclass(frozen=True)
class MicroOutcome:
    loss: float
    qa_loss: float | None
    identity_raw: float | None
    identity_normalized: float | None
    phase: str
    event_kind: str
    step_index: int
    selected_step_indices: tuple[int, ...]
    micro_gradient_norms: dict[str, Any] | None
    objective: str
    transition_id: str
    global_micro_index: int
    rgb_min: float
    rgb_max: float
    saturation: float
    rgb_rms: float
    delta_rms: float
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R4_MICRO_METRICS_SCHEMA,
            "kind": "micro_transition",
            "global_micro_index": self.global_micro_index,
            "transition_id": self.transition_id,
            "phase": self.phase,
            "event_kind": self.event_kind,
            "selected_step_index": self.step_index,
            "selected_step_indices": list(self.selected_step_indices),
            "micro_gradient_norms": self.micro_gradient_norms,
            "objective": self.objective,
            "loss": self.loss,
            "qa_loss": self.qa_loss,
            "identity_raw_smooth_l1": self.identity_raw,
            "identity_normalized": self.identity_normalized,
            "rgb_min": self.rgb_min,
            "rgb_max": self.rgb_max,
            "rgb_saturation_fraction": self.saturation,
            "rgb_rms": self.rgb_rms,
            "rgb_delta_rms": self.delta_rms,
            "receipt": self.receipt,
        }


def _optional_float(value: Tensor | None) -> float | None:
    return None if value is None else float(value.detach())


def validate_checkpoint_trainer_state(
    value: Mapping[str, Any],
    *,
    optimizer_step: int,
    budget: RunBudget,
    schedule_seed: int,
    identity_scale: float,
) -> None:
    expected = checkpoint_trainer_state(
        optimizer_step=optimizer_step,
        budget=budget,
        schedule_seed=schedule_seed,
        identity_scale=identity_scale,
        m0_evaluation=value.get("m0_evaluation", {}),
    )
    if dict(value) != expected:
        raise ValueError(f"R4 checkpoint cursor/phase mismatch: expected={expected}, observed={dict(value)}")


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _peak_gib(device: torch.device) -> float:
    return float(torch.cuda.max_memory_allocated(device) / (1024**3))


def checkpoint_trainer_state(
    *,
    optimizer_step: int,
    budget: RunBudget,
    schedule_seed: int,
    identity_scale: float,
    m0_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    next_micro = optimizer_step * budget.gradient_accumulation
    balanced_next = max(0, next_micro - budget.warmup_micro_transitions)
    return {
        "schema": R4_CHECKPOINT_STATE_SCHEMA,
        "next_global_micro_index": next_micro,
        "warmup_next_unit_index": min(next_micro, budget.warmup_micro_transitions),
        "balanced_schedule_cursor": R4ScheduleCursor(
            seed=schedule_seed,
            next_unit_index=balanced_next,
        ).to_dict(),
        "identity_scale": identity_scale,
        "m0_evaluation": dict(m0_evaluation),
    }


def evaluate_r4(*, model: R4FreePixelModel, records: Sequence[Any], reader_fn: Any) -> dict[str, Any]:
    presentation_means: list[float] = []
    query_count = 0
    updates = 0
    with torch.no_grad():
        for episode in records:
            for permutation in REVERSE_CYCLIC4:
                episode_losses, episode_updates = _evaluate_episode(model, episode, permutation, reader_fn)
                if not episode_losses:
                    raise ValueError("Every R4 dev episode presentation must contain a query.")
                presentation_means.append(sum(episode_losses) / len(episode_losses))
                query_count += len(episode_losses)
                updates += episode_updates
    if not presentation_means:
        raise ValueError("R4 fixed evaluation produced no query losses.")
    return {
        "choice_view_family": "reverse_cyclic4",
        "noise_presentation_index": 0,
        "episode_count": len(records),
        "choice_presentations_per_episode": len(REVERSE_CYCLIC4),
        "query_loss_count": query_count,
        "updater_call_count": updates,
        "mean_episode_listwise_choice_ce": sum(presentation_means) / len(presentation_means),
    }


def _evaluate_episode(
    model: R4FreePixelModel,
    episode: Any,
    permutation: tuple[int, int, int, int],
    reader_fn: Any,
) -> tuple[list[float], int]:
    episode_id = _required_text(episode, "episode_id")
    state = model.reset_state()
    losses: list[float] = []
    updates = 0
    for turn_index, turn in enumerate(_episode_turns(episode)):
        kind = _enum_text(_field(turn, "type", _field(turn, "kind")))
        if kind in {"event", "mixed"}:
            state = model.updater(
                state,
                _required_text(turn, "event_text"),
                episode_id,
                turn_index,
                gradient_mode="full",
                persistent_state="float_rgb",
                presentation_index=0,
            )
            _assert_unit_rgb(state, name="evaluation state RGB")
            updates += 1
        if kind in {"query", "mixed"}:
            losses.append(_eval_query(state, turn, permutation, reader_fn))
    return losses, updates


def _validate_identity_calibration(value: Mapping[str, Any], *, count: int) -> float:
    if value.get("schema") != R4_IDENTITY_CALIBRATION_SCHEMA:
        raise ValueError("Resume identity calibration schema mismatch.")
    if value.get("source_split") != "train_only" or value.get("count") != count:
        raise ValueError("Resume identity calibration differs from the train-only contract.")
    scale = value.get("frozen_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Resume identity calibration scale is invalid.")
    return float(scale)


def _eval_query(state: Tensor, turn: Any, permutation: tuple[int, int, int, int], reader_fn: Any) -> float:
    text, choices, target_index = _query_payload(turn)
    ordered = tuple(choices[index] for index in permutation)
    target = permutation.index(target_index)
    return float(_loss_tensor(reader_fn(state, format_mcq_query(text, ordered), ordered, target)))


def calibrate_identity_scale(
    *,
    model: R4FreePixelModel,
    examples: Sequence[TransitionExample],
    schedule_seed: int,
    count: int,
) -> dict[str, Any]:
    selected = _calibration_examples(examples, schedule_seed=schedule_seed, count=count)
    values: list[float] = []
    with torch.no_grad():
        for index, transition in enumerate(selected):
            source = _replay_prefix(model=model, transition=transition, presentation_index=index)
            output = model.updater(
                source,
                _required_text(transition.target_turn, "event_text"),
                transition.episode_id,
                transition.target_turn_index,
                gradient_mode="drtune",
                selected_step_indices=(index % 4,),
                persistent_state="float_rgb",
                presentation_index=index,
            )
            _assert_unit_rgb(output, name="calibration output RGB")
            value = float(_identity_loss(output, source))
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"Invalid identity calibration loss: {value}")
            values.append(value)
    median = float(np.median(np.asarray(values, dtype=np.float64)))
    return {
        "schema": R4_IDENTITY_CALIBRATION_SCHEMA,
        "source_split": "train_only",
        "parameter_state": "initial_lora_before_optimizer_step_1",
        "count": count,
        "transition_ids_sha256": _canonical_sha256([item.transition_id for item in selected]),
        "raw_smooth_l1": values,
        "median_raw_smooth_l1": median,
        "frozen_scale": max(median, 1e-6),
    }


def _calibration_examples(
    examples: Sequence[TransitionExample],
    *,
    schedule_seed: int,
    count: int,
) -> list[TransitionExample]:
    noop = sorted(
        (item for item in examples if item.event_kind == "noop" and item.uses_identity_loss),
        key=lambda item: (
            hashlib.sha256(
                f"{R4_IDENTITY_CALIBRATION_SCHEMA}\x1f{schedule_seed}\x1f{item.transition_id}".encode()
            ).digest(),
            item.transition_id,
        ),
    )
    if not noop:
        raise ValueError("R4 identity calibration requires train-only NOOP transitions.")
    return [noop[index % len(noop)] for index in range(count)]


def _qa_loss_for_transition(*, unit: TrainingUnit, output_rgb: Tensor, reader_loss_fn: Any) -> Tensor | None:
    if not unit.transition.uses_qa_loss:
        return None
    losses: list[Tensor] = []
    for query_index, turn in zip(
        unit.transition.qa_query_indices,
        unit.transition.qa_query_turns,
        strict=True,
    ):
        text, choices, target_index = _query_payload(turn)
        permutation = _training_choice_view(unit.transition.transition_id, unit.global_micro_index, query_index)
        ordered = tuple(choices[index] for index in permutation)
        ordered_target = permutation.index(target_index)
        result = reader_loss_fn(output_rgb, format_mcq_query(text, ordered), ordered, ordered_target)
        losses.append(_loss_tensor(result))
    if not losses:
        raise RuntimeError("R4 QA transition has no causally local query loss.")
    return torch.stack(losses).mean()


def _loss_tensor(result: Any) -> Tensor:
    loss = result if isinstance(result, Tensor) else getattr(result, "loss", None)
    if not isinstance(loss, Tensor) or loss.numel() != 1:
        raise TypeError("Reader must return a scalar Tensor or an object with scalar .loss.")
    if not torch.isfinite(loss):
        raise RuntimeError("Reader returned a non-finite loss.")
    return loss


def _training_choice_view(transition_id: str, micro: int, query_index: int) -> tuple[int, int, int, int]:
    key = f"{transition_id}\x1f{query_index}".encode()
    phase = int.from_bytes(hashlib.sha256(key).digest()[:2], "big") % len(CYCLIC4)
    return CYCLIC4[(micro + phase) % len(CYCLIC4)]


def _identity_loss(output_rgb: Tensor, source_rgb: Tensor) -> Tensor:
    return torch.nn.functional.smooth_l1_loss(output_rgb, source_rgb.detach(), reduction="mean")


def normalized_identity_loss(raw: Tensor, scale: float) -> Tensor:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Frozen identity calibration scale must be finite and positive.")
    return raw / raw.new_tensor(scale)


def combine_objective_losses(qa: Tensor | None, identity: Tensor | None) -> Tensor:
    if qa is None and identity is None:
        raise ValueError("R4 transition has no admissible objective.")
    if qa is None:
        assert identity is not None
        return identity
    if identity is None:
        return qa
    return qa + identity.to(device=qa.device)


def _replay_prefix(
    *,
    model: R4FreePixelModel,
    transition: TransitionExample,
    presentation_index: int,
) -> Tensor:
    state = model.reset_state()
    turns = _episode_turns(transition.episode)
    with torch.no_grad():
        for turn_index in transition.prefix_updater_indices:
            state = model.updater(
                state,
                _required_text(turns[turn_index], "event_text"),
                transition.episode_id,
                turn_index,
                gradient_mode="full",
                persistent_state="float_rgb",
                presentation_index=presentation_index,
            )
            _assert_unit_rgb(state, name="prefix output RGB")
    return state.detach()


def _target_update(model: R4FreePixelModel, unit: TrainingUnit, source_rgb: Tensor) -> Tensor:
    _assert_unit_rgb(source_rgb, name="target source RGB")
    output = model.updater(
        source_rgb.detach(),
        _required_text(unit.transition.target_turn, "event_text"),
        unit.transition.episode_id,
        unit.transition.target_turn_index,
        **unit.updater_kwargs(),
    )
    _assert_unit_rgb(output, name="target output RGB")
    return output


def _force_trainable_fp32(module: nn.Module) -> tuple[list[tuple[str, nn.Parameter]], list[nn.Parameter]]:
    named = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    unexpected = [name for name, _ in named if ".lora_A." not in name and ".lora_B." not in name]
    if unexpected:
        raise RuntimeError(f"R4 permits only LoRA adapter parameters: {unexpected}")
    if not named:
        raise RuntimeError("R4 found no trainable LoRA parameters.")
    for _name, parameter in named:
        parameter.data = parameter.data.float()
        if parameter.dtype is not torch.float32:
            raise RuntimeError("R4 LoRA adapters must remain FP32.")
    return named, [parameter for _name, parameter in named]


def _assert_unit_rgb(tensor: Tensor, *, name: str) -> None:
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise RuntimeError(f"{name} must be batch-one BCHW RGB, got {tuple(tensor.shape)}.")
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} must be finite floating-point RGB.")
    minimum, maximum = float(tensor.detach().min()), float(tensor.detach().max())
    if minimum < 0.0 or maximum > 1.0:
        raise RuntimeError(f"{name} escaped [0,1]: min={minimum}, max={maximum}.")


class R4FreePixelModel(nn.Module):
    def __init__(
        self,
        *,
        pipeline: Any,
        initial_rgb: Tensor,
        global_seed: int,
        checkpoint_unet: bool,
    ) -> None:
        super().__init__()
        self.updater = DreamLiteRecurrentUpdater(
            pipeline=pipeline,
            global_seed=global_seed,
            checkpoint_unet=checkpoint_unet,
        )
        self.register_buffer("initial_rgb", initial_rgb.detach().clone(), persistent=False)

    def reset_state(self) -> Tensor:
        return self.initial_rgb.clone()


def _initial_rgb_tensor(*, resolution: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    image, _metadata = load_initial_image("blank", None, resolution=resolution)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)


def make_static_manifest(
    *,
    args: argparse.Namespace,
    budget: RunBudget,
    all_transitions: Sequence[TransitionExample],
    transition_audit: Mapping[str, Any],
    warmup_pool: Sequence[TransitionExample],
    determinism_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    validate_teacher_free_bindings(
        teacher_manifest=None,
        teacher_sidecar=None,
        canonical_canvas=None,
        pixel_target=None,
        latent_target=None,
        feature_target=None,
        codebook=None,
    )
    commit = git_value("rev-parse", "HEAD")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise RuntimeError("Formal R4 training refuses a dirty worktree; --allow-dirty is diagnostic only.")
    protocol = validate_r4_manifest_contract(make_r4_manifest_contract(schedule_seed=args.schedule_seed))
    _image, initial = load_initial_image("blank", None, resolution=args.resolution)
    return {
        "schema": "vision_memory.r4-free-pixel-training-manifest.v1",
        "metrics_schema": R4_METRICS_SCHEMA,
        "summary_schema": R4_SUMMARY_SCHEMA,
        "git_commit": commit,
        "git_dirty": bool(status),
        "protocol_contract": protocol,
        "teacher_bindings": None,
        "training_profile": _training_profile(budget, warmup_pool, args),
        "historical_comparator": _historical_comparator(),
        "transition_index_audit": dict(transition_audit),
        "transition_index_sha256": transition_index_digest(all_transitions),
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        **_model_bindings(args),
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "initial_state": {
            **initial,
            "representation": "BCHW_float_rgb_unit_interval",
            "learnable": False,
        },
        "arguments": _serializable_args(args),
        "strict_determinism": dict(determinism_report) if determinism_report is not None else None,
        "environment": _runtime_versions(),
    }


def _model_bindings(args: argparse.Namespace) -> dict[str, Any]:
    payloads = {
        "dreamlite_mobile": _verified_snapshot_payload(
            model_dir=args.dreamlite,
            model_key="dreamlite_mobile",
            env_name="VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256",
            required=args.strict_determinism,
        ),
        "qwen_reader": _verified_snapshot_payload(
            model_dir=args.reader,
            model_key="qwen_reader",
            env_name="VLM_READER_SNAPSHOT_MANIFEST_SHA256",
            required=args.strict_determinism,
        ),
    }
    return {
        "dreamlite_revision": locked_revision(args.dreamlite),
        "reader_revision": locked_revision(args.reader),
        "model_snapshot_manifests": {name: binding["manifest_sha256"] for name, binding in payloads.items()},
        "model_snapshot_payloads_start": payloads,
    }


def _verified_snapshot_payload(
    *,
    model_dir: Path,
    model_key: str,
    env_name: str,
    required: bool,
) -> dict[str, Any]:
    lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
    specification = lock["models"][model_key]
    declared_manifest_sha = _snapshot_binding(model_dir, env_name, required=required)
    binding = verify_snapshot_manifest(
        manifest_path=model_dir / SNAPSHOT_MANIFEST_NAME,
        model_dir=model_dir,
        expected_repo_id=str(specification["repo_id"]),
        expected_revision=str(specification["revision"]),
    )
    if binding["manifest_sha256"] != declared_manifest_sha:
        raise RuntimeError(f"{model_key} full snapshot verification disagrees with {env_name}.")
    return binding


def _verify_end_snapshot_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected = manifest.get("model_snapshot_payloads_start")
    if not isinstance(expected, Mapping) or set(expected) != {"dreamlite_mobile", "qwen_reader"}:
        raise ValueError("R4 manifest is missing the start-of-run full snapshot bindings.")
    observed = {
        name: verify_snapshot_binding(binding)
        for name, binding in sorted(expected.items())
        if isinstance(binding, Mapping)
    }
    if observed != dict(expected):
        raise RuntimeError("R4 model snapshot payload changed between start and end verification.")
    return {
        "schema": "vision_memory.r4-model-snapshot-end-verification.v1",
        "passed": True,
        "bindings": observed,
    }


def _historical_comparator() -> dict[str, Any]:
    return {
        "role": "metadata_only_not_loaded_or_rerun",
        "name": "M_hist_R3_fullscale_20260720",
        "git_commit": "10bde565d30d119a68e8460757d979b1c35e1b8f",
        "optimizer_steps": 256,
        "elapsed_seconds": 11233.8007,
        "scheduled_dev_loss": 10.0844607353,
    }


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "diffusers": importlib.metadata.version("diffusers"),
        "transformers": importlib.metadata.version("transformers"),
        "peft": importlib.metadata.version("peft"),
    }


def transition_index_digest(examples: Sequence[TransitionExample]) -> str:
    payload = [
        {
            "transition_id": item.transition_id,
            "episode_id": item.episode_id,
            "target_turn_index": item.target_turn_index,
            "event_kind": item.event_kind,
            "prefix_updater_indices": list(item.prefix_updater_indices),
            "local_query_indices": list(item.local_query_indices),
            "objective": item.objective.value,
        }
        for item in examples
    ]
    return _canonical_sha256(payload)


def _training_profile(budget: RunBudget, warmup_pool: Sequence[TransitionExample], args: argparse.Namespace | None = None) -> dict[str, Any]:
    warmup_ids = sorted(item.transition_id for item in warmup_pool)
    return {
        **budget.to_dict(),
        "lora_rank": int(getattr(args, "lora_rank", LORA_RANK)),
        "learning_rate": float(getattr(args, "learning_rate", LEARNING_RATE)),
        "weight_decay": float(getattr(args, "weight_decay", WEIGHT_DECAY)),
        "gradient_clip": float(getattr(args, "gradient_clip", GRADIENT_CLIP)),
        "selected_step_count": int(getattr(args, "selected_step_count", 1)),
        "optimizer_diagnostics": bool(getattr(args, "optimizer_diagnostics", False)),
        "record_micro_metrics": bool(getattr(args, "record_micro_metrics", False)),
        "record_validation_metrics": bool(getattr(args, "record_validation_metrics", False)),
        "validation_every": int(getattr(args, "validation_every", None) or budget.checkpoint_every),
        "optimizer": "AdamW",
        "identity_loss": "SmoothL1(output_rgb,stopgrad(input_rgb))/frozen_train_median",
        "qa_loss": "frozen_qwen_listwise_choice_ce",
        "objective_weights": {"qa": 1.0, "identity": 1.0},
        "warmup_schema": R4_WARMUP_SCHEMA,
        "warmup_pool_rule": "trainable_set_with_empty_updater_prefix",
        "warmup_pool_count": len(warmup_ids),
        "warmup_pool_ids_sha256": _canonical_sha256(warmup_ids),
        "expected_schedule_counts": expected_schedule_counts(budget),
    }


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"resume", "output_dir", "allow_dirty"}
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }


def _snapshot_binding(model_dir: Path, env_name: str, *, required: bool) -> str:
    path = model_dir / ".snapshot_manifest.json"
    if not path.is_file():
        raise RuntimeError(f"Model snapshot is missing {path.name}: {model_dir}")
    observed = sha256_file(path)
    declared = os.environ.get(env_name)
    if declared is None:
        if required:
            raise RuntimeError(f"Strict R4 training requires {env_name}.")
        return observed
    if declared != declared.lower() or len(declared) != 64 or declared != observed:
        raise RuntimeError(f"{env_name} is not the SHA256 of {path}.")
    return observed


def _query_payload(turn: Mapping[str, Any] | Any) -> tuple[str, tuple[str, ...], int]:
    query = _field(turn, "query")
    if query is None:
        raise ValueError("R4 QA objective requires an explicit query object.")
    text = _required_text(query, "text")
    choices = tuple(str(item) for item in _field(query, "choices", ()))
    target_index = _field(query, "target_index")
    if len(choices) != 4 or len(set(choices)) != 4:
        raise ValueError("R4 listwise Reader requires four distinct choices.")
    if isinstance(target_index, bool) or not isinstance(target_index, int) or not 0 <= target_index < 4:
        raise ValueError("R4 query target_index must be an integer in [0,3].")
    return text, choices, target_index


def _required_text(value: Mapping[str, Any] | Any, name: str) -> str:
    item = _field(value, name)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"R4 model-visible field {name!r} must be non-empty.")
    return item.strip()


def _episode_turns(episode: Mapping[str, Any] | Any) -> Sequence[Any]:
    turns = _field(episode, "turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)) or not turns:
        raise ValueError("R4 episode must contain turns.")
    return turns


if __name__ == "__main__":
    raise SystemExit(main())
