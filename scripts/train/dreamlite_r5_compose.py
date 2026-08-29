"""R5-Compose stateful DreamLite training, audit, and endpoint evaluation.

The script is intentionally fail-closed around the preregistered R5 mechanism:
teacher-free F1--F6 segments, hard NOOP identity, fixed event noise, h=2/h=4
TBPTT, LoRA-only optimization, and fixed raw/EMA endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.inspire.model_snapshot_manifest import (  # noqa: E402
    verify_snapshot_binding,
)
from scripts.train.dreamlite_episode import (  # noqa: E402
    append_jsonl,
    assert_frozen_contract,
    begin_optimizer_diagnostics,
    choice_reader_callable,
    compute_dtype,
    finalize_optimizer_diagnostics_after_step,
    git_value,
    grouped_tensor_norms,
    record_optimizer_diagnostics_after_clip,
    set_all_seeds,
    sha256_file,
)
from scripts.train.dreamlite_r4_free_pixel import (  # noqa: E402
    _force_trainable_fp32,
    _initial_rgb_tensor,
    _load_pipeline,
    _load_reader,
    _peak_gib,
    _runtime_versions,
    _verified_snapshot_payload,
    _write_environment,
)
from vision_memory.data import REVERSE_CYCLIC4, read_jsonl as read_episode_jsonl  # noqa: E402
from vision_memory.data.schema import EventKind, QuerySpec  # noqa: E402
from vision_memory.dreamlite import DreamLiteRecurrentUpdater  # noqa: E402
from vision_memory.repro import configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import format_mcq_query, load_trainable_weights  # noqa: E402
from vision_memory.training.checkpoint import load_training_checkpoint, save_training_checkpoint  # noqa: E402
from vision_memory.training.r5_compose import (  # noqa: E402
    R5_FAMILIES,
    R5_PROTOCOL,
    R5Event,
    R5Segment,
    ScheduledR5Segment,
    build_r5_family_pools,
    build_r5_schedule,
    canonical_sha256,
    family_pool_audit,
    make_r5_manifest_contract,
    mechanism_subsets,
    schedule_audit,
    stable_record_split,
)


R5_METRICS_SCHEMA = "vision_memory.r5-compose-training-metrics.v1"
R5_MICRO_SCHEMA = "vision_memory.r5-compose-micro-metrics.v1"
R5_TRAINER_STATE_SCHEMA = "vision_memory.r5-compose-checkpoint-state.v1"
R5_TRAINING_MANIFEST_SCHEMA = "vision_memory.r5-compose-training-manifest.v1"
R5_GRADIENT_AUDIT_SCHEMA = "vision_memory.r5-compose-gradient-fidelity.v1"
R5_EVALUATION_SCHEMA = "vision_memory.r5-compose-causal-evaluation.v1"
R5_SUMMARY_SCHEMA = "vision_memory.r5-compose-summary.v1"

GRADIENT_ACCUMULATION = 8
LORA_RANK = 4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 10.0
EMA_DECAY = 0.995
LR_START = 1e-5
LR_PEAK = 3e-5
LR_FINAL = 1e-5
LR_WARMUP_STEPS = 16
MAIN_OPTIMIZER_STEPS = 640
PILOT_OPTIMIZER_STEPS = 128
CHECKPOINT_EVERY = 64

PROFILE_STEPS = {
    "smoke": 2,
    "topology": 10,
    "pilot": PILOT_OPTIMIZER_STEPS,
    "rescue": PILOT_OPTIMIZER_STEPS,
    "main": MAIN_OPTIMIZER_STEPS,
}


def _write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _field(value: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _enum_text(value: Any) -> str | None:
    value = getattr(value, "value", value)
    return None if value is None else str(value)


def _required_text(value: Mapping[str, Any] | Any, name: str) -> str:
    item = _field(value, name)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"R5 requires a non-empty {name}.")
    return item.strip()


def _query_payload(turn: Mapping[str, Any] | Any) -> QuerySpec:
    value = _field(turn, "query")
    if value is None:
        raise ValueError("R5 query turn has no query payload.")
    return QuerySpec.from_dict(value) if isinstance(value, Mapping) else value


def _loss_tensor(result: Any) -> Tensor:
    loss = result if isinstance(result, Tensor) else getattr(result, "loss", None)
    if not isinstance(loss, Tensor) or loss.numel() != 1:
        raise TypeError("R5 Reader must return a scalar Tensor or an object with scalar .loss.")
    if not torch.isfinite(loss):
        raise RuntimeError("R5 Reader returned non-finite loss.")
    return loss


def _mean(values: Iterable[float]) -> float | None:
    items = list(values)
    return None if not items else sum(items) / len(items)


def r5_learning_rate(step_zero: int, *, endpoint_steps: int = MAIN_OPTIMIZER_STEPS) -> float:
    """Preregistered 16-step warmup followed by cosine decay to step 639."""

    if isinstance(step_zero, bool) or not isinstance(step_zero, int) or step_zero < 0:
        raise ValueError("R5 LR step must be a non-negative integer.")
    if endpoint_steps != MAIN_OPTIMIZER_STEPS:
        # Pilots and technical runs intentionally consume a prefix of the same
        # 640-step schedule instead of silently changing the optimization law.
        endpoint_steps = MAIN_OPTIMIZER_STEPS
    if step_zero < LR_WARMUP_STEPS:
        if LR_WARMUP_STEPS == 1:
            return LR_PEAK
        fraction = step_zero / (LR_WARMUP_STEPS - 1)
        return LR_START + fraction * (LR_PEAK - LR_START)
    clipped = min(step_zero, endpoint_steps - 1)
    fraction = (clipped - LR_WARMUP_STEPS) / (endpoint_steps - 1 - LR_WARMUP_STEPS)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return LR_FINAL + cosine * (LR_PEAK - LR_FINAL)


def _training_choice_view(segment_id: str, global_micro_index: int) -> tuple[int, int, int, int]:
    phase = int.from_bytes(hashlib.sha256(segment_id.encode("utf-8")).digest()[:2], "big") % 4
    rotation = (global_micro_index + phase) % 4
    base = (0, 1, 2, 3)
    return base[rotation:] + base[:rotation]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="R5-Compose stateful DreamLite experiments")
    parser.add_argument(
        "--profile",
        choices=("smoke", "topology", "gradient-audit", "pilot", "rescue", "main", "final-eval"),
        required=True,
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--persistent-state", choices=("float_rgb", "latent"), required=True)
    parser.add_argument("--tbptt-horizon", type=int, choices=(2, 4), required=True)
    parser.add_argument("--gradient-mode", choices=("drtune_stateful", "full"), default="drtune_stateful")
    parser.add_argument("--selected-step-count", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-seed", type=int)
    parser.add_argument("--schedule-seed", type=int, default=0)
    parser.add_argument("--pairing-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--gradient-accumulation", type=int, default=GRADIENT_ACCUMULATION)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--gradient-clip", type=float, default=GRADIENT_CLIP)
    parser.add_argument("--lora-rank", type=int, default=LORA_RANK)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument("--residual-blend", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--checkpoint-unet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--audit-state-gradients", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-audit-size", type=int, default=24)
    parser.add_argument("--health-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--record-micro-metrics", action=argparse.BooleanOptionalAction, default=False)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.adapter_seed is None:
        args.adapter_seed = args.seed
    return args


def _validate_args(args: argparse.Namespace) -> int:
    if args.resolution != 1024:
        raise ValueError("R5 locks the 1024x1024 visual-state and Reader resize contract.")
    if args.gradient_mode == "full" and args.selected_step_count != 0:
        raise ValueError("R5 full gradient requires --selected-step-count 0.")
    if args.gradient_mode == "drtune_stateful" and args.selected_step_count not in {1, 2}:
        raise ValueError("R5 drtune_stateful requires selected-step-count 1 or 2.")
    if args.gradient_accumulation != GRADIENT_ACCUMULATION:
        raise ValueError("R5 fixes gradient accumulation at 8.")
    if args.lora_rank != LORA_RANK or args.weight_decay != WEIGHT_DECAY or args.gradient_clip != GRADIENT_CLIP:
        raise ValueError("R5 fixes rank=4, weight_decay=1e-4, and gradient_clip=10.")
    if not 0.0 < args.ema_decay < 1.0 or args.ema_decay != EMA_DECAY:
        raise ValueError("R5 fixes EMA decay at 0.995.")
    if args.checkpoint_every <= 0:
        raise ValueError("R5 checkpoint interval must be positive.")
    if args.profile == "rescue":
        if args.residual_blend != 0.5 or args.persistent_state != "latent":
            raise ValueError("R5 conditional rescue fixes latent persistence and residual_blend=0.5.")
    elif args.profile not in {"final-eval"} and args.residual_blend != 1.0:
        raise ValueError("R5 non-rescue training profiles forbid residual blending.")
    elif args.profile == "final-eval" and args.residual_blend not in {0.5, 1.0}:
        raise ValueError("R5 final evaluation residual_blend must match a main (1.0) or rescue (0.5) checkpoint.")
    if args.profile in PROFILE_STEPS:
        expected = PROFILE_STEPS[args.profile]
        steps = expected if args.max_optimizer_steps is None else args.max_optimizer_steps
        if args.profile in {"pilot", "rescue", "main"} and steps != expected:
            raise ValueError(f"R5 {args.profile} profile fixes optimizer steps at {expected}.")
        if args.profile == "smoke" and steps != 2:
            raise ValueError("R5 smoke profile fixes optimizer steps at 2.")
        if args.profile == "topology" and steps != 10:
            raise ValueError("R5 topology profile fixes optimizer steps at 10.")
        return steps
    if args.max_optimizer_steps is not None:
        raise ValueError(f"R5 profile {args.profile} does not accept --max-optimizer-steps.")
    if args.profile == "gradient-audit" and not 24 <= args.gradient_audit_size <= 32:
        raise ValueError("R5 gradient fidelity audit requires 24--32 fixed segments.")
    if args.profile == "final-eval" and not args.checkpoint:
        raise ValueError("R5 final-eval requires at least one --checkpoint.")
    return 0


@dataclass(frozen=True)
class R5DataBundle:
    train_records: tuple[Any, ...]
    train_pools: dict[str, tuple[R5Segment, ...]]
    schedule: tuple[ScheduledR5Segment, ...]
    dev_select_records: tuple[Any, ...]
    dev_final_records: tuple[Any, ...]
    dev_reserve_records: tuple[Any, ...]
    mechanism_select: tuple[R5Segment, ...]
    mechanism_final: tuple[R5Segment, ...]
    pool_audit: dict[str, Any]
    schedule_audit: dict[str, Any] | None
    split_audit: dict[str, Any]


def _load_data(args: argparse.Namespace, *, optimizer_steps: int) -> R5DataBundle:
    train_records = tuple(read_episode_jsonl(args.train))
    dev_records = tuple(read_episode_jsonl(args.dev))
    if not train_records or len(dev_records) < 160:
        raise ValueError("R5 requires a non-empty train set and at least 160 dev episodes.")
    train_pools = build_r5_family_pools(train_records, pairing_seed=args.pairing_seed)
    pool_report = family_pool_audit(train_pools)
    selected_count = args.selected_step_count if args.gradient_mode == "drtune_stateful" else 0
    schedule = (
        build_r5_schedule(
            train_pools,
            optimizer_steps=optimizer_steps,
            schedule_seed=args.schedule_seed,
            selected_step_count=selected_count,
        )
        if optimizer_steps
        else ()
    )
    dev_split = stable_record_split(dev_records, seed=args.split_seed)
    dev_pools = build_r5_family_pools(dev_records, pairing_seed=args.split_seed)
    mechanism = mechanism_subsets(dev_pools, seed=args.split_seed)
    split_report = {
        "schema": "vision_memory.r5-compose-dev-split.v1",
        "split_seed": args.split_seed,
        "formal_counts": {key: len(value) for key, value in dev_split.items()},
        "formal_ids": {
            key: [_required_text(record, "episode_id") for record in value]
            for key, value in dev_split.items()
        },
        "formal_ids_sha256": {
            key: canonical_sha256([_required_text(record, "episode_id") for record in value])
            for key, value in dev_split.items()
        },
        "mechanism_select_ids": [segment.segment_id for segment in mechanism["select"]],
        "mechanism_final_ids": [segment.segment_id for segment in mechanism["final"]],
        "mechanism_category_counts": mechanism["category_counts"],
        "mechanism_select_sha256": canonical_sha256([segment.to_dict() for segment in mechanism["select"]]),
        "mechanism_final_sha256": canonical_sha256([segment.to_dict() for segment in mechanism["final"]]),
    }
    return R5DataBundle(
        train_records=train_records,
        train_pools=train_pools,
        schedule=schedule,
        dev_select_records=tuple(dev_split["select"]),
        dev_final_records=tuple(dev_split["final"]),
        dev_reserve_records=tuple(dev_split["reserve"]),
        mechanism_select=tuple(mechanism["select"]),
        mechanism_final=tuple(mechanism["final"]),
        pool_audit=pool_report,
        schedule_audit=schedule_audit(schedule) if schedule else None,
        split_audit=split_report,
    )


class R5ComposeModel(nn.Module):
    def __init__(
        self,
        *,
        pipeline: Any,
        initial_rgb: Tensor,
        global_seed: int,
        checkpoint_unet: bool,
        persistent_state: str,
        residual_blend: float,
    ) -> None:
        super().__init__()
        self.updater = DreamLiteRecurrentUpdater(
            pipeline=pipeline,
            global_seed=global_seed,
            checkpoint_unet=checkpoint_unet,
        )
        self.persistent_state = persistent_state
        if not 0.0 < residual_blend <= 1.0:
            raise ValueError("R5 residual_blend must lie in (0,1].")
        self.residual_blend = float(residual_blend)
        with torch.no_grad():
            initial = (
                initial_rgb.detach().clone()
                if persistent_state == "float_rgb"
                else self.updater.encode_persistent_rgb(initial_rgb.detach())
            )
        self.register_buffer("initial_state", initial.detach().clone(), persistent=False)

    def reset_state(self) -> Tensor:
        return self.initial_state.clone()

    def apply_event(
        self,
        state: Tensor,
        event: R5Event,
        *,
        gradient_mode: str,
        selected_step_indices: tuple[int, ...] | None,
    ) -> Tensor:
        if event.event_kind == EventKind.NOOP.value:
            # Hard identity is deliberately object-identical and graph-preserving.
            return state
        updated = self.updater(
            state,
            event.event_text,
            event.source_episode_id,
            event.noise_turn_id,
            gradient_mode=gradient_mode,
            selected_step_indices=selected_step_indices,
            persistent_state=self.persistent_state,
            presentation_index=0,
            noise_include_presentation_index=False,
        )
        if self.residual_blend == 1.0:
            return updated
        return state.mul(1.0 - self.residual_blend).add(updated, alpha=self.residual_blend)

    def reader_image(self, state: Tensor) -> Tensor:
        return self.updater.decode_for_reader(state, persistent_state=self.persistent_state)


@dataclass(frozen=True)
class RuntimeBundle:
    pipe: Any
    processor: Any
    reader: nn.Module
    model: R5ComposeModel
    optimizer: torch.optim.Optimizer
    named_trainable: list[tuple[str, nn.Parameter]]
    trainable: list[nn.Parameter]
    updater_device: torch.device
    reader_device: torch.device


def _load_runtime(args: argparse.Namespace) -> RuntimeBundle:
    if not torch.cuda.is_available():
        raise RuntimeError("R5 requires CUDA.")
    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    pipe = _load_pipeline(args, updater_device, compute_dtype(updater_device))
    processor, reader = _load_reader(args, reader_device, compute_dtype(reader_device))
    model = R5ComposeModel(
        pipeline=pipe,
        initial_rgb=_initial_rgb_tensor(
            resolution=args.resolution,
            device=updater_device,
            dtype=compute_dtype(updater_device),
        ),
        global_seed=args.seed,
        checkpoint_unet=args.checkpoint_unet,
        persistent_state=args.persistent_state,
        residual_blend=args.residual_blend,
    )
    named, trainable = _force_trainable_fp32(model)
    optimizer = torch.optim.AdamW(trainable, lr=LR_START, weight_decay=args.weight_decay)
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


class TrainableEMA:
    def __init__(self, named_parameters: Sequence[tuple[str, nn.Parameter]], *, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {name: parameter.detach().clone() for name, parameter in named_parameters}

    @torch.no_grad()
    def update(self, named_parameters: Sequence[tuple[str, nn.Parameter]]) -> None:
        observed = {name for name, _ in named_parameters}
        if observed != set(self.shadow):
            raise ValueError("R5 EMA parameter names changed during training.")
        for name, parameter in named_parameters:
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def cpu_state_dict(self) -> dict[str, Tensor]:
        return {name: value.detach().cpu() for name, value in self.shadow.items()}

    def load_state_dict(self, value: Mapping[str, Tensor]) -> None:
        if set(value) != set(self.shadow):
            raise ValueError("R5 EMA checkpoint parameter names do not match.")
        for name, destination in self.shadow.items():
            destination.copy_(value[name].to(device=destination.device, dtype=destination.dtype))

    @contextmanager
    def apply(self, named_parameters: Sequence[tuple[str, nn.Parameter]]) -> Iterator[None]:
        backup = {name: parameter.detach().clone() for name, parameter in named_parameters}
        try:
            with torch.no_grad():
                for name, parameter in named_parameters:
                    parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))
            yield
        finally:
            with torch.no_grad():
                for name, parameter in named_parameters:
                    parameter.copy_(backup[name])


@dataclass
class SegmentForward:
    loss: Tensor
    reader_output: Any
    boundary_state: Tensor
    earliest_output_state: Tensor
    final_state: Tensor
    final_image: Tensor
    prefix_event_count: int


def _segment_forward(
    *,
    model: R5ComposeModel,
    segment: R5Segment,
    reader_fn: Any,
    tbptt_horizon: int,
    gradient_mode: str,
    selected_step_indices: tuple[int, ...] | None,
    choice_permutation: tuple[int, int, int, int],
    audit_state_gradients: bool,
) -> SegmentForward:
    event_count = len(segment.events)
    prefix_count = max(0, event_count - tbptt_horizon)
    state = model.reset_state()
    with torch.no_grad():
        for event in segment.events[:prefix_count]:
            state = model.apply_event(
                state,
                event,
                gradient_mode="full",
                selected_step_indices=None,
            )
    boundary = state.detach()
    if audit_state_gradients:
        boundary.requires_grad_(True)
        boundary.retain_grad()
    # The graph-bearing suffix must start from this explicit TBPTT boundary;
    # continuing from the pre-detach ``state`` would silently defeat truncation
    # and make the source-gradient audit observe an unused tensor.
    state = boundary
    earliest: Tensor | None = None
    for event in segment.events[prefix_count:]:
        state = model.apply_event(
            state,
            event,
            gradient_mode=gradient_mode,
            selected_step_indices=selected_step_indices,
        )
        if earliest is None:
            earliest = state
            if audit_state_gradients and earliest.requires_grad:
                earliest.retain_grad()
    if earliest is None:
        raise RuntimeError("R5 segment unroll contains no events.")
    image = model.reader_image(state)
    query = segment.query
    ordered_choices = tuple(query.choices[index] for index in choice_permutation)
    ordered_target = choice_permutation.index(query.target_index)
    reader_output = reader_fn(
        image,
        format_mcq_query(query.text, ordered_choices),
        ordered_choices,
        ordered_target,
    )
    return SegmentForward(
        loss=_loss_tensor(reader_output),
        reader_output=reader_output,
        boundary_state=boundary,
        earliest_output_state=earliest,
        final_state=state,
        final_image=image,
        prefix_event_count=prefix_count,
    )


def _state_gradient_record(forward: SegmentForward) -> dict[str, Any]:
    def record(tensor: Tensor) -> dict[str, Any]:
        gradient = tensor.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all())
        norm = None if gradient is None else float(gradient.detach().float().norm())
        return {
            "present": gradient is not None,
            "finite": finite,
            "norm": norm,
            "nonzero": finite and norm is not None and norm > 0.0,
        }

    return {
        "segment_boundary": record(forward.boundary_state),
        "earliest_unrolled_output": record(forward.earliest_output_state),
    }


def _assert_finite_state(state: Tensor, *, name: str) -> None:
    if not state.is_floating_point() or not bool(torch.isfinite(state).all()):
        raise RuntimeError(f"R5 {name} is not a finite floating-point tensor.")
    if state.ndim != 4 or state.shape[0] != 1:
        raise RuntimeError(f"R5 {name} must be batch-one BCHW, got {tuple(state.shape)}.")


@dataclass(frozen=True)
class MicroOutcome:
    global_micro_index: int
    segment_id: str
    family: str
    phase: str
    loss: float
    selected_step_indices: tuple[int, ...] | None
    prefix_event_count: int
    updater_count: int
    query_gap: int
    target_event_kind: str
    state_gradient: dict[str, Any] | None
    image_min: float
    image_max: float
    image_saturation_fraction: float
    image_rms: float
    micro_gradient_norms: dict[str, Any] | None
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R5_MICRO_SCHEMA,
            "global_micro_index": self.global_micro_index,
            "segment_id": self.segment_id,
            "family": self.family,
            "phase": self.phase,
            "loss": self.loss,
            "selected_step_indices": (
                list(self.selected_step_indices) if self.selected_step_indices is not None else None
            ),
            "prefix_event_count": self.prefix_event_count,
            "updater_count": self.updater_count,
            "query_gap": self.query_gap,
            "target_event_kind": self.target_event_kind,
            "state_gradient": self.state_gradient,
            "image_min": self.image_min,
            "image_max": self.image_max,
            "image_saturation_fraction": self.image_saturation_fraction,
            "image_rms": self.image_rms,
            "micro_gradient_norms": self.micro_gradient_norms,
            "receipt": self.receipt,
        }


def _micro_forward_backward(
    *,
    unit: ScheduledR5Segment,
    model: R5ComposeModel,
    reader_fn: Any,
    args: argparse.Namespace,
    named_trainable: Sequence[tuple[str, nn.Parameter]],
) -> MicroOutcome:
    permutation = _training_choice_view(unit.segment.segment_id, unit.global_micro_index)
    forward = _segment_forward(
        model=model,
        segment=unit.segment,
        reader_fn=reader_fn,
        tbptt_horizon=args.tbptt_horizon,
        gradient_mode=args.gradient_mode,
        selected_step_indices=unit.selected_step_indices,
        choice_permutation=permutation,
        audit_state_gradients=args.audit_state_gradients,
    )
    if not torch.isfinite(forward.loss):
        raise RuntimeError("R5 segment loss is non-finite.")
    previous = (
        {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in named_trainable
        }
        if args.record_micro_metrics
        else None
    )
    (forward.loss / args.gradient_accumulation).backward()
    micro_gradient_norms = None
    if previous is not None:
        micro_gradient_norms = grouped_tensor_norms(
            (
                name,
                None
                if parameter.grad is None
                else parameter.grad.detach() - (previous[name] if previous[name] is not None else 0),
            )
            for name, parameter in named_trainable
        )
    state_gradient = _state_gradient_record(forward) if args.audit_state_gradients else None
    _assert_finite_state(forward.final_state, name="final persistent state")
    detached_image = forward.final_image.detach().float()
    if detached_image.ndim != 4 or detached_image.shape[1] != 3:
        raise RuntimeError(f"R5 Reader image must be BCHW RGB, got {tuple(detached_image.shape)}.")
    return MicroOutcome(
        global_micro_index=unit.global_micro_index,
        segment_id=unit.segment.segment_id,
        family=unit.segment.family,
        phase=unit.phase,
        loss=float(forward.loss.detach()),
        selected_step_indices=unit.selected_step_indices,
        prefix_event_count=forward.prefix_event_count,
        updater_count=unit.segment.updater_count,
        query_gap=unit.segment.query_gap,
        target_event_kind=unit.segment.target_event_kind,
        state_gradient=state_gradient,
        image_min=float(detached_image.min()),
        image_max=float(detached_image.max()),
        image_saturation_fraction=float(
            ((detached_image <= 1 / 255) | (detached_image >= 254 / 255)).float().mean()
        ),
        image_rms=float(detached_image.square().mean().sqrt()),
        micro_gradient_norms=micro_gradient_norms,
        receipt=unit.receipt(),
    )


def _clip_gradients(
    named_trainable: Sequence[tuple[str, nn.Parameter]],
    trainable: Sequence[nn.Parameter],
    *,
    max_norm: float,
) -> float:
    observed = 0
    for name, parameter in named_trainable:
        if parameter.dtype is not torch.float32:
            raise RuntimeError(f"R5 LoRA parameter changed dtype: {name} -> {parameter.dtype}")
        if parameter.grad is not None:
            observed += 1
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"R5 LoRA gradient is non-finite: {name}")
    if not observed:
        raise RuntimeError("R5 optimizer received no LoRA gradients.")
    norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm)
    value = float(norm)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"R5 optimizer gradient norm is invalid: {value}")
    return value


def _run_optimizer_step(
    *,
    step_zero: int,
    units: Sequence[ScheduledR5Segment],
    runtime: RuntimeBundle,
    ema: TrainableEMA,
    reader_fn: Any,
    args: argparse.Namespace,
    elapsed: float,
) -> dict[str, Any]:
    if len(units) != args.gradient_accumulation:
        raise ValueError("R5 optimizer step requires exactly eight micro-segments.")
    learning_rate = r5_learning_rate(step_zero)
    for group in runtime.optimizer.param_groups:
        group["lr"] = learning_rate
    runtime.optimizer.zero_grad(set_to_none=True)
    outcomes = [
        _micro_forward_backward(
            unit=unit,
            model=runtime.model,
            reader_fn=reader_fn,
            args=args,
            named_trainable=runtime.named_trainable,
        )
        for unit in units
    ]
    assert_frozen_contract(runtime.pipe, runtime.reader)
    diagnostic_snapshot, diagnostic_report = begin_optimizer_diagnostics(runtime.named_trainable)
    gradient_norm = _clip_gradients(
        runtime.named_trainable,
        runtime.trainable,
        max_norm=args.gradient_clip,
    )
    record_optimizer_diagnostics_after_clip(
        diagnostic_report,
        runtime.named_trainable,
        gradient_norm=gradient_norm,
        max_norm=args.gradient_clip,
    )
    runtime.optimizer.step()
    diagnostic_report = finalize_optimizer_diagnostics_after_step(
        diagnostic_report,
        runtime.named_trainable,
        diagnostic_snapshot,
    )
    ema.update(runtime.named_trainable)
    runtime.optimizer.zero_grad(set_to_none=True)
    state_gradients = [
        item.state_gradient["segment_boundary"]["nonzero"]
        for item in outcomes
        if item.state_gradient is not None
    ]
    return {
        "schema": R5_METRICS_SCHEMA,
        "kind": "optimizer_step",
        "optimizer_step": step_zero + 1,
        "next_global_micro_index": (step_zero + 1) * args.gradient_accumulation,
        "learning_rate": learning_rate,
        "loss_mean": _mean(item.loss for item in outcomes),
        "loss_by_family": {
            family: _mean(item.loss for item in outcomes if item.family == family)
            for family in sorted({item.family for item in outcomes})
        },
        "gradient_norm_before_clip": gradient_norm,
        "gradient_clip_threshold": args.gradient_clip,
        "gradient_clipped": gradient_norm > args.gradient_clip,
        "state_gradient_nonzero_fraction": (
            None if not state_gradients else sum(state_gradients) / len(state_gradients)
        ),
        "family_counts": dict(sorted(Counter(item.family for item in outcomes).items())),
        "phase_counts": dict(sorted(Counter(item.phase for item in outcomes).items())),
        "target_event_kind_counts": dict(
            sorted(Counter(item.target_event_kind for item in outcomes).items())
        ),
        "selected_step_set_counts": dict(
            sorted(Counter(str(item.selected_step_indices) for item in outcomes).items())
        ),
        "image_min": min(item.image_min for item in outcomes),
        "image_max": max(item.image_max for item in outcomes),
        "image_saturation_fraction_mean": _mean(item.image_saturation_fraction for item in outcomes),
        "image_rms_mean": _mean(item.image_rms for item in outcomes),
        "optimizer_diagnostics": diagnostic_report,
        "schedule_receipts": [item.receipt for item in outcomes],
        "elapsed_seconds": elapsed,
        "updater_peak_memory_gib": _peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": _peak_gib(runtime.reader_device),
        "micro_records": [item.to_dict() for item in outcomes] if args.record_micro_metrics else None,
    }


def _manifest_model_bindings(args: argparse.Namespace) -> dict[str, Any]:
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
        "model_snapshot_manifests": {name: value["manifest_sha256"] for name, value in payloads.items()},
        "model_snapshot_payloads_start": payloads,
    }


def _serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    excluded = {"resume", "output_dir", "checkpoint"}
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }


def _make_manifest(
    *,
    args: argparse.Namespace,
    optimizer_steps: int,
    data: R5DataBundle,
    determinism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    commit = git_value("rev-parse", "HEAD")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise RuntimeError("R5 formal runs refuse a dirty worktree; use --allow-dirty for diagnostics only.")
    contract = make_r5_manifest_contract(
        persistent_state=args.persistent_state,
        tbptt_horizon=args.tbptt_horizon,
        selected_step_count=args.selected_step_count,
        schedule_seed=args.schedule_seed,
        gradient_mode=args.gradient_mode,
    )
    return {
        "schema": R5_TRAINING_MANIFEST_SCHEMA,
        "protocol": R5_PROTOCOL,
        "profile": args.profile,
        "git_commit": commit,
        "git_dirty": bool(status),
        "protocol_contract": contract,
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "family_pool_audit": data.pool_audit,
        "schedule_audit": data.schedule_audit,
        "dev_split_audit": data.split_audit,
        "optimizer": {
            "name": "AdamW",
            "optimizer_steps": optimizer_steps,
            "gradient_accumulation": args.gradient_accumulation,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "lora_rank": args.lora_rank,
            "lr_schedule": {
                "warmup_steps": LR_WARMUP_STEPS,
                "start": LR_START,
                "peak": LR_PEAK,
                "final": LR_FINAL,
                "cosine_endpoint_step": MAIN_OPTIMIZER_STEPS - 1,
                "short_runs_use_main_schedule_prefix": True,
            },
            "ema_decay": args.ema_decay,
            "primary_endpoint": "ema_fixed_final_step",
            "secondary_endpoint": "raw_fixed_final_step",
        },
        "supervision": {
            "final_query_weight": 1.0,
            "aux_query_total_weight": 0.0,
            "noop_identity_loss": False,
            "teacher_image": None,
            "latent_target": None,
            "feature_target": None,
            "canonical_canvas": None,
            "residual_blend": args.residual_blend,
        },
        "arguments": _serializable_args(args),
        "strict_determinism": dict(determinism) if determinism is not None else None,
        "environment": _runtime_versions(),
        **_manifest_model_bindings(args),
    }


def _verify_end_bindings(manifest: Mapping[str, Any]) -> dict[str, Any]:
    expected = manifest.get("model_snapshot_payloads_start")
    if not isinstance(expected, Mapping) or set(expected) != {"dreamlite_mobile", "qwen_reader"}:
        raise ValueError("R5 manifest lacks complete model snapshot bindings.")
    observed = {
        name: verify_snapshot_binding(binding)
        for name, binding in sorted(expected.items())
        if isinstance(binding, Mapping)
    }
    if observed != dict(expected):
        raise RuntimeError("R5 model snapshot changed between start and endpoint.")
    return {"schema": "vision_memory.r5-model-snapshot-end-verification.v1", "passed": True, "bindings": observed}


def _trainer_state(
    *,
    optimizer_step: int,
    ema: TrainableEMA,
    prior_elapsed: float,
    health_baseline: float | None,
    consecutive_health_failures: int,
) -> dict[str, Any]:
    return {
        "schema": R5_TRAINER_STATE_SCHEMA,
        "next_optimizer_step": optimizer_step,
        "next_global_micro_index": optimizer_step * GRADIENT_ACCUMULATION,
        "ema_decay": ema.decay,
        "ema_state": ema.cpu_state_dict(),
        "prior_elapsed_seconds": prior_elapsed,
        "health_baseline_ce": health_baseline,
        "consecutive_health_failures": consecutive_health_failures,
    }


def _save_checkpoint(
    path: Path,
    *,
    runtime: RuntimeBundle,
    manifest: Mapping[str, Any],
    optimizer_step: int,
    ema: TrainableEMA,
    elapsed: float,
    health_baseline: float | None,
    consecutive_health_failures: int,
) -> Path:
    return save_training_checkpoint(
        path,
        trainable_module=runtime.model,
        optimizer=runtime.optimizer,
        epoch=0,
        episode_cursor=optimizer_step * GRADIENT_ACCUMULATION,
        optimizer_step=optimizer_step,
        manifest=manifest,
        trainer_state=_trainer_state(
            optimizer_step=optimizer_step,
            ema=ema,
            prior_elapsed=elapsed,
            health_baseline=health_baseline,
            consecutive_health_failures=consecutive_health_failures,
        ),
    )


def _save_ema_checkpoint(
    path: Path,
    *,
    runtime: RuntimeBundle,
    manifest: Mapping[str, Any],
    optimizer_step: int,
    ema: TrainableEMA,
    elapsed: float,
    health_baseline: float | None,
    consecutive_health_failures: int,
) -> Path:
    with ema.apply(runtime.named_trainable):
        return _save_checkpoint(
            path,
            runtime=runtime,
            manifest=manifest,
            optimizer_step=optimizer_step,
            ema=ema,
            elapsed=elapsed,
            health_baseline=health_baseline,
            consecutive_health_failures=consecutive_health_failures,
        )


def _load_resume(
    *,
    args: argparse.Namespace,
    runtime: RuntimeBundle,
    manifest: Mapping[str, Any],
    ema: TrainableEMA,
) -> tuple[int, float, float | None, int]:
    assert args.resume is not None
    payload = load_training_checkpoint(
        args.resume,
        trainable_module=runtime.model,
        optimizer=runtime.optimizer,
        expected_manifest=manifest,
    )
    state = payload.get("trainer_state")
    if not isinstance(state, Mapping) or state.get("schema") != R5_TRAINER_STATE_SCHEMA:
        raise ValueError("R5 resume checkpoint has invalid trainer_state.")
    step = int(payload["optimizer_step"])
    if state.get("next_optimizer_step") != step or state.get("next_global_micro_index") != step * 8:
        raise ValueError("R5 resume checkpoint cursor mismatch.")
    if float(state.get("ema_decay")) != ema.decay or not isinstance(state.get("ema_state"), Mapping):
        raise ValueError("R5 resume checkpoint EMA contract mismatch.")
    ema.load_state_dict(state["ema_state"])
    return (
        step,
        float(state.get("prior_elapsed_seconds", 0.0)),
        None if state.get("health_baseline_ce") is None else float(state["health_baseline_ce"]),
        int(state.get("consecutive_health_failures", 0)),
    )


def _hard_noop_gate(model: R5ComposeModel) -> dict[str, Any]:
    state = model.reset_state()
    event = R5Event("technical-noop", 0, "technical-entity", "noop", "Unrelated update.")
    output = model.apply_event(state, event, gradient_mode="full", selected_step_indices=None)
    return {
        "same_object": output is state,
        "same_data_ptr": output.data_ptr() == state.data_ptr(),
        "bitwise_equal": bool(torch.equal(output, state)),
        "passed": output is state and output.data_ptr() == state.data_ptr() and bool(torch.equal(output, state)),
    }


def _reset_peak_memory(runtime: RuntimeBundle) -> None:
    devices = {runtime.updater_device, runtime.reader_device}
    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)


def _append_metric(path: Path, metric: Mapping[str, Any]) -> None:
    append_jsonl(path, dict(metric))


def _truncate_jsonl_for_resume(path: Path, *, optimizer_step: int) -> None:
    if not path.is_file():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        step = value.get("optimizer_step")
        if step is None or isinstance(step, int) and step <= optimizer_step:
            retained.append(json.dumps(value, ensure_ascii=False))
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    temporary.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class EvalItem:
    item_id: str
    pair_unit: str
    episode_id: str
    query_id: str
    query: QuerySpec
    normal_state: Tensor
    temporal_state: Tensor
    family: str
    target_event_kind: str | None
    query_gap: int
    updater_count: int
    cross_slot_interference: bool
    stale_target_text: str | None


def _cpu_state(state: Tensor) -> Tensor:
    return state.detach().to(device="cpu", copy=True)


def _segment_eval_items(model: R5ComposeModel, segments: Sequence[R5Segment]) -> list[EvalItem]:
    items: list[EvalItem] = []
    with torch.no_grad():
        for segment in segments:
            state = model.reset_state()
            timeline = [_cpu_state(state)]
            for event in segment.events:
                state = model.apply_event(
                    state,
                    event,
                    gradient_mode="full",
                    selected_step_indices=None,
                )
                timeline.append(_cpu_state(state))
            items.append(
                EvalItem(
                    item_id=segment.segment_id,
                    pair_unit=segment.segment_id,
                    episode_id=segment.query_source_episode_id,
                    query_id=f"{segment.query_source_episode_id}:{segment.query_turn_index}",
                    query=segment.query,
                    normal_state=timeline[-1],
                    temporal_state=timeline[-2],
                    family=segment.family,
                    target_event_kind=segment.target_event_kind,
                    query_gap=segment.query_gap,
                    updater_count=segment.updater_count,
                    cross_slot_interference=segment.cross_slot_interference,
                    stale_target_text=segment.stale_target_text,
                )
            )
    return items


def _formal_eval_items(model: R5ComposeModel, records: Sequence[Any]) -> list[EvalItem]:
    items: list[EvalItem] = []
    with torch.no_grad():
        for episode in records:
            episode_id = _required_text(episode, "episode_id")
            turns = _field(episode, "turns")
            if not isinstance(turns, Sequence):
                raise ValueError("R5 formal evaluation episode has invalid turns.")
            state = model.reset_state()
            temporal = state
            updater_count = 0
            last_event_kind: str | None = None
            for turn_index, turn in enumerate(turns):
                turn_type = _enum_text(_field(turn, "type", _field(turn, "kind")))
                if turn_type in {"event", "mixed"}:
                    event = R5Event(
                        source_episode_id=episode_id,
                        source_turn_index=turn_index,
                        entity_id=_required_text(episode, "entity_id"),
                        event_kind=_enum_text(_field(turn, "event_kind")) or "",
                        event_text=_required_text(turn, "event_text"),
                    )
                    temporal = state
                    state = model.apply_event(
                        state,
                        event,
                        gradient_mode="full",
                        selected_step_indices=None,
                    )
                    updater_count += 1
                    if event.event_kind != EventKind.NOOP.value:
                        last_event_kind = event.event_kind
                if turn_type in {"query", "mixed"}:
                    items.append(
                        EvalItem(
                            item_id=f"{episode_id}:q{turn_index}",
                            pair_unit=episode_id,
                            episode_id=episode_id,
                            query_id=f"{episode_id}:{turn_index}",
                            query=_query_payload(turn),
                            normal_state=_cpu_state(state),
                            temporal_state=_cpu_state(temporal),
                            family="formal",
                            target_event_kind=last_event_kind,
                            query_gap=updater_count,
                            updater_count=updater_count,
                            cross_slot_interference=False,
                            stale_target_text=None,
                        )
                    )
    if not items:
        raise ValueError("R5 formal evaluation produced no query items.")
    return items


def _cross_donor_indices(items: Sequence[EvalItem]) -> list[int]:
    if len(items) < 2:
        raise ValueError("R5 cross-episode swap requires at least two evaluation items.")
    ordered = sorted(
        range(len(items)),
        key=lambda index: (
            hashlib.sha256(f"r5-cross-swap\x1f{items[index].item_id}".encode()).digest(),
            items[index].item_id,
        ),
    )
    donors: list[int] = []
    for recipient_index, recipient in enumerate(items):
        start = ordered.index(recipient_index)
        donor: int | None = None
        for offset in range(1, len(ordered) + 1):
            candidate_index = ordered[(start + offset) % len(ordered)]
            candidate = items[candidate_index]
            if (
                candidate.episode_id != recipient.episode_id
                and candidate.query.target != recipient.query.target
                and tuple(candidate.normal_state.shape) == tuple(recipient.normal_state.shape)
            ):
                donor = candidate_index
                break
        if donor is None:
            for candidate_index, candidate in enumerate(items):
                if candidate_index != recipient_index and tuple(candidate.normal_state.shape) == tuple(
                    recipient.normal_state.shape
                ):
                    donor = candidate_index
                    break
        if donor is None:
            raise RuntimeError("R5 failed to construct a cross-item state donor.")
        donors.append(donor)
    return donors


def _choice_row(
    *,
    reader_output: Any,
    item: EvalItem,
    checkpoint_label: str,
    suite: str,
    condition: str,
    permutation: tuple[int, int, int, int],
    view_index: int,
    donor_item_id: str | None,
) -> dict[str, Any]:
    logits = getattr(reader_output, "choice_logits", None)
    if not isinstance(logits, Tensor) or logits.numel() != 4:
        raise TypeError("R5 evaluation requires Reader choice_logits with four elements.")
    values = logits.detach().float().cpu()
    predicted_ordered = int(values.argmax())
    ordered_target = permutation.index(item.query.target_index)
    alternatives = torch.cat((values[:ordered_target], values[ordered_target + 1 :]))
    margin = float(values[ordered_target] - alternatives.max())
    predicted_original = permutation[predicted_ordered]
    return {
        "schema": R5_EVALUATION_SCHEMA,
        "checkpoint": checkpoint_label,
        "suite": suite,
        "condition": condition,
        "item_id": item.item_id,
        "pair_unit": item.pair_unit,
        "episode_id": item.episode_id,
        "query_id": item.query_id,
        "view_index": view_index,
        "permutation": list(permutation),
        "target_index": item.query.target_index,
        "predicted_index": predicted_original,
        "target_text": item.query.target,
        "predicted_text": item.query.choices[predicted_original],
        "correct": predicted_original == item.query.target_index,
        "ce": float(_loss_tensor(reader_output).detach()),
        "margin": margin,
        "choice_logits_ordered": values.tolist(),
        "family": item.family,
        "target_event_kind": item.target_event_kind,
        "query_gap": item.query_gap,
        "updater_count": item.updater_count,
        "cross_slot_interference": item.cross_slot_interference,
        "donor_item_id": donor_item_id,
        "stale_target_text": item.stale_target_text,
        "stale_error": (
            item.stale_target_text is not None
            and item.query.choices[predicted_original] == item.stale_target_text
            and predicted_original != item.query.target_index
        ),
    }


def _score_state(
    *,
    model: R5ComposeModel,
    reader_fn: Any,
    state: Tensor,
    item: EvalItem,
    checkpoint_label: str,
    suite: str,
    condition: str,
    permutations: Sequence[tuple[int, int, int, int]],
    donor_item_id: str | None = None,
) -> list[dict[str, Any]]:
    device_state = state.to(device=model.initial_state.device, dtype=model.initial_state.dtype)
    with torch.no_grad():
        image = model.reader_image(device_state)
        rows: list[dict[str, Any]] = []
        for view_index, permutation in enumerate(permutations):
            ordered = tuple(item.query.choices[index] for index in permutation)
            target = permutation.index(item.query.target_index)
            output = reader_fn(
                image,
                format_mcq_query(item.query.text, ordered),
                ordered,
                target,
            )
            rows.append(
                _choice_row(
                    reader_output=output,
                    item=item,
                    checkpoint_label=checkpoint_label,
                    suite=suite,
                    condition=condition,
                    permutation=permutation,
                    view_index=view_index,
                    donor_item_id=donor_item_id,
                )
            )
    return rows


def evaluate_items(
    *,
    model: R5ComposeModel,
    reader_fn: Any,
    items: Sequence[EvalItem],
    checkpoint_label: str,
    suite: str,
    controls: Sequence[str],
    permutations: Sequence[tuple[int, int, int, int]],
) -> list[dict[str, Any]]:
    allowed = {"normal", "reset", "cross_episode_swap", "temporal_swap", "updater_disabled"}
    if not controls or not set(controls).issubset(allowed):
        raise ValueError(f"R5 evaluation controls must be a non-empty subset of {sorted(allowed)}.")
    donors = _cross_donor_indices(items) if "cross_episode_swap" in controls else []
    initial = _cpu_state(model.reset_state())
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        reset_rows: list[dict[str, Any]] | None = None
        for condition in controls:
            if condition == "normal":
                state = item.normal_state
                donor_id = None
            elif condition == "temporal_swap":
                state = item.temporal_state
                donor_id = None
            elif condition == "cross_episode_swap":
                donor = items[donors[index]]
                state = donor.normal_state
                donor_id = donor.item_id
            else:
                state = initial
                donor_id = None
            if condition == "updater_disabled" and reset_rows is not None:
                for row in reset_rows:
                    rows.append({**row, "condition": "updater_disabled"})
                continue
            scored = _score_state(
                model=model,
                reader_fn=reader_fn,
                state=state,
                item=item,
                checkpoint_label=checkpoint_label,
                suite=suite,
                condition=condition,
                permutations=permutations,
                donor_item_id=donor_id,
            )
            rows.extend(scored)
            if condition == "reset":
                reset_rows = scored
    return rows


def _group_rows(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    grouped: defaultdict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in fields)].append(row)
    return dict(grouped)


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "n_rows": len(rows),
        "n_pair_units": len({str(row["pair_unit"]) for row in rows}),
        "mean_ce": _mean(float(row["ce"]) for row in rows),
        "accuracy": _mean(float(bool(row["correct"])) for row in rows),
        "mean_margin": _mean(float(row["margin"]) for row in rows),
    }


def _paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    condition_a: str,
    condition_b: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    per_unit_condition: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        condition = str(row["condition"])
        if condition in {condition_a, condition_b}:
            per_unit_condition[(str(row["pair_unit"]), condition)].append(float(row["ce"]))
    units_a = {unit for unit, condition in per_unit_condition if condition == condition_a}
    units_b = {unit for unit, condition in per_unit_condition if condition == condition_b}
    if units_a != units_b or not units_a:
        raise ValueError(f"R5 paired bootstrap inputs are not matched: {condition_a} versus {condition_b}.")
    units = sorted(units_a)
    differences = np.asarray(
        [
            np.mean(per_unit_condition[(unit, condition_a)])
            - np.mean(per_unit_condition[(unit, condition_b)])
            for unit in units
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = differences[rng.integers(0, len(differences), size=len(differences))].mean()
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "definition": "mean_ce(condition_a)-mean_ce(condition_b)",
        "pair_units": len(units),
        "iterations": iterations,
        "seed": seed,
        "estimate": float(differences.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "fraction_below_zero": float((draws < 0).mean()),
    }


def summarize_evaluation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_checkpoint_suite_condition = {
        "|".join(str(part) for part in key): _metric_summary(values)
        for key, values in sorted(_group_rows(rows, ("checkpoint", "suite", "condition")).items())
    }
    breakdowns: dict[str, Any] = {}
    for field in ("family", "target_event_kind", "query_gap", "updater_count", "cross_slot_interference"):
        breakdowns[field] = {
            "|".join(str(part) for part in key): _metric_summary(values)
            for key, values in sorted(
                _group_rows(rows, ("checkpoint", "suite", "condition", field)).items(),
                key=lambda item: tuple(str(value) for value in item[0]),
            )
        }
    bootstraps: dict[str, Any] = {}
    for (checkpoint, suite), subset in _group_rows(rows, ("checkpoint", "suite")).items():
        conditions = {str(row["condition"]) for row in subset}
        if "normal" not in conditions:
            continue
        for control in ("reset", "cross_episode_swap", "temporal_swap", "updater_disabled"):
            if control in conditions:
                key = f"{checkpoint}|{suite}|normal_vs_{control}"
                bootstraps[key] = _paired_bootstrap(
                    subset,
                    condition_a="normal",
                    condition_b=control,
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big"),
                )
    stale_rows = [row for row in rows if row.get("stale_target_text") is not None and row.get("condition") == "normal"]
    return {
        "schema": R5_EVALUATION_SCHEMA,
        "rows": len(rows),
        "by_checkpoint_suite_condition": by_checkpoint_suite_condition,
        "breakdowns": breakdowns,
        "paired_bootstrap": bootstraps,
        "stale_error_proxy": {
            "definition": "fraction of labeled overwrite/clear rows whose prediction equals the pre-update target",
            "n": len(stale_rows),
            "rate": _mean(float(bool(row["stale_error"])) for row in stale_rows),
            "coverage_note": "Clear-only source episodes without an intermediate pre-clear query have no stale label and are excluded.",
        },
    }


def _normal_ce(
    *,
    model: R5ComposeModel,
    reader_fn: Any,
    records: Sequence[Any],
    checkpoint_label: str,
    permutations: Sequence[tuple[int, int, int, int]],
) -> float:
    items = _formal_eval_items(model, records)
    rows = evaluate_items(
        model=model,
        reader_fn=reader_fn,
        items=items,
        checkpoint_label=checkpoint_label,
        suite="health_formal_select",
        controls=("normal",),
        permutations=permutations,
    )
    value = _mean(float(row["ce"]) for row in rows)
    if value is None or not math.isfinite(value):
        raise RuntimeError("R5 health evaluation returned invalid CE.")
    return value


def _pilot_evaluation(
    *,
    model: R5ComposeModel,
    reader_fn: Any,
    data: R5DataBundle,
    checkpoint_label: str,
    bootstrap_iterations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = evaluate_items(
        model=model,
        reader_fn=reader_fn,
        items=_formal_eval_items(model, data.dev_select_records),
        checkpoint_label=checkpoint_label,
        suite="formal_select_32",
        controls=("normal",),
        permutations=REVERSE_CYCLIC4,
    )
    rows.extend(
        evaluate_items(
            model=model,
            reader_fn=reader_fn,
            items=_segment_eval_items(model, data.mechanism_select),
            checkpoint_label=checkpoint_label,
            suite="mechanism_select_32",
            controls=("normal", "reset"),
            permutations=REVERSE_CYCLIC4,
        )
    )
    return rows, summarize_evaluation_rows(
        rows,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=20260730,
    )


def _gradient_vector(named_trainable: Sequence[tuple[str, nn.Parameter]]) -> Tensor:
    chunks: list[Tensor] = []
    for _name, parameter in named_trainable:
        gradient = parameter.grad
        chunks.append(
            torch.zeros(parameter.numel(), dtype=torch.float32)
            if gradient is None
            else gradient.detach().float().reshape(-1).cpu()
        )
    if not chunks:
        raise RuntimeError("R5 gradient audit found no trainable parameters.")
    vector = torch.cat(chunks)
    if not bool(torch.isfinite(vector).all()) or float(vector.norm()) <= 0.0:
        raise RuntimeError("R5 gradient audit produced an invalid gradient vector.")
    return vector


def _audit_segment_selection(data: R5DataBundle, *, size: int, seed: int) -> tuple[R5Segment, ...]:
    families = ("F2", "F3", "F4", "F5", "F6")
    selected: list[R5Segment] = []
    cursors = {family: 0 for family in families}
    ordered = {
        family: tuple(
            sorted(
                data.train_pools[family],
                key=lambda segment: (
                    hashlib.sha256(
                        f"{R5_GRADIENT_AUDIT_SCHEMA}\x1f{seed}\x1f{family}\x1f{segment.segment_id}".encode()
                    ).digest(),
                    segment.segment_id,
                ),
            )
        )
        for family in families
    }
    for index in range(size):
        family = families[index % len(families)]
        values = ordered[family]
        selected.append(values[cursors[family] % len(values)])
        cursors[family] += 1
    return tuple(selected)


def _audit_gradient_for_policy(
    *,
    runtime: RuntimeBundle,
    reader_fn: Any,
    segment: R5Segment,
    tbptt_horizon: int,
    gradient_mode: str,
    selected_steps: tuple[int, ...] | None,
) -> tuple[Tensor, dict[str, Any]]:
    runtime.optimizer.zero_grad(set_to_none=True)
    forward = _segment_forward(
        model=runtime.model,
        segment=segment,
        reader_fn=reader_fn,
        tbptt_horizon=tbptt_horizon,
        gradient_mode=gradient_mode,
        selected_step_indices=selected_steps,
        choice_permutation=(0, 1, 2, 3),
        audit_state_gradients=True,
    )
    forward.loss.backward()
    state_gradient = _state_gradient_record(forward)
    vector = _gradient_vector(runtime.named_trainable)
    assert_frozen_contract(runtime.pipe, runtime.reader)
    runtime.optimizer.zero_grad(set_to_none=True)
    return vector, {
        "loss": float(forward.loss.detach()),
        "state_gradient": state_gradient,
        "gradient_norm": float(vector.norm()),
    }


def _cosine_and_ratio(approximation: Tensor, reference: Tensor) -> tuple[float, float]:
    reference_norm = reference.double().norm()
    approximation_norm = approximation.double().norm()
    cosine = torch.dot(approximation.double(), reference.double()) / (approximation_norm * reference_norm)
    return float(cosine), float(approximation_norm / reference_norm)


def _gradient_group_summary(records: Sequence[Mapping[str, Any]], group: str) -> dict[str, Any]:
    values = [record for record in records if record["count_group"] == group]
    cosines = np.asarray([float(record["cosine_to_full"]) for record in values], dtype=np.float64)
    ratios = np.asarray([float(record["norm_ratio_to_full"]) for record in values], dtype=np.float64)
    by_kind: dict[str, Any] = {}
    for kind in ("set", "overwrite", "clear"):
        subset = [float(record["cosine_to_full"]) for record in values if record["target_event_kind"] == kind]
        by_kind[kind] = {
            "n": len(subset),
            "positive_fraction": None if not subset else float(np.mean(np.asarray(subset) > 0)),
            "median_cosine": None if not subset else float(np.median(subset)),
        }
    return {
        "n": len(values),
        "median_cosine": float(np.median(cosines)),
        "positive_fraction": float(np.mean(cosines > 0)),
        "median_norm_ratio": float(np.median(ratios)),
        "norm_ratio_q05_q95": [float(np.quantile(ratios, 0.05)), float(np.quantile(ratios, 0.95))],
        "by_target_event_kind": by_kind,
    }


def _select_gradient_policy(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = {group: _gradient_group_summary(records, group) for group in ("K1", "K2")}
    eligible: list[str] = []
    for group, summary in summaries.items():
        event_gate = all(
            value["n"] > 0 and value["positive_fraction"] >= 0.5
            for value in summary["by_target_event_kind"].values()
        )
        norm_gate = 0.02 <= summary["median_norm_ratio"] <= 20.0 and summary["norm_ratio_q05_q95"][1] <= 50.0
        if summary["positive_fraction"] >= 0.75 and event_gate and norm_gate:
            eligible.append(group)
    if not eligible:
        return {
            "decision": "fallback_full_gradient",
            "gradient_mode": "full",
            "selected_step_count": 0,
            "persistent_state": "latent",
            "tbptt_horizon": 2,
            "reason": "No K1/K2 approximation passed positive-alignment, event-kind, and norm-ratio gates.",
            "group_summaries": summaries,
        }
    selected = max(eligible, key=lambda group: (summaries[group]["median_cosine"], group))
    return {
        "decision": f"select_{selected}",
        "gradient_mode": "drtune_stateful",
        "selected_step_count": 1 if selected == "K1" else 2,
        "persistent_state": None,
        "tbptt_horizon": None,
        "median_cosine_preference_target": 0.3,
        "eligible_groups": eligible,
        "group_summaries": summaries,
    }


def run_gradient_audit(
    *,
    args: argparse.Namespace,
    runtime: RuntimeBundle,
    data: R5DataBundle,
    reader_fn: Any,
) -> dict[str, Any]:
    segments = _audit_segment_selection(data, size=args.gradient_audit_size, seed=args.schedule_seed)
    policies = (
        ("K1-step0", "K1", (0,)),
        ("K1-step1", "K1", (1,)),
        ("K1-step2", "K1", (2,)),
        ("K1-step3", "K1", (3,)),
        ("K2-steps02", "K2", (0, 2)),
        ("K2-steps13", "K2", (1, 3)),
    )
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for segment_index, segment in enumerate(segments):
        full, full_metadata = _audit_gradient_for_policy(
            runtime=runtime,
            reader_fn=reader_fn,
            segment=segment,
            tbptt_horizon=args.tbptt_horizon,
            gradient_mode="full",
            selected_steps=None,
        )
        if not all(
            record["nonzero"] and record["finite"]
            for record in full_metadata["state_gradient"].values()
        ):
            raise RuntimeError(f"R5 full-gradient source-state gate failed on {segment.segment_id}.")
        for policy, count_group, selected_steps in policies:
            approximation, metadata = _audit_gradient_for_policy(
                runtime=runtime,
                reader_fn=reader_fn,
                segment=segment,
                tbptt_horizon=args.tbptt_horizon,
                gradient_mode="drtune_stateful",
                selected_steps=selected_steps,
            )
            cosine, ratio = _cosine_and_ratio(approximation, full)
            state_gate = all(
                record["nonzero"] and record["finite"]
                for record in metadata["state_gradient"].values()
            )
            records.append(
                {
                    "segment_index": segment_index,
                    "segment_id": segment.segment_id,
                    "family": segment.family,
                    "target_event_kind": segment.target_event_kind,
                    "updater_count": segment.updater_count,
                    "policy": policy,
                    "count_group": count_group,
                    "selected_step_indices": list(selected_steps),
                    "cosine_to_full": cosine,
                    "norm_ratio_to_full": ratio,
                    "full_gradient_norm": full_metadata["gradient_norm"],
                    "approximation_gradient_norm": metadata["gradient_norm"],
                    "full_loss": full_metadata["loss"],
                    "approximation_loss": metadata["loss"],
                    "state_gradient_gate_passed": state_gate,
                    "state_gradient": metadata["state_gradient"],
                }
            )
        print(
            json.dumps(
                {
                    "milestone": "gradient_audit_segment",
                    "completed": segment_index + 1,
                    "total": len(segments),
                    "segment_id": segment.segment_id,
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    decision = _select_gradient_policy(records)
    report = {
        "schema": R5_GRADIENT_AUDIT_SCHEMA,
        "status": "completed",
        "segment_count": len(segments),
        "segment_ids": [segment.segment_id for segment in segments],
        "segment_ids_sha256": canonical_sha256([segment.segment_id for segment in segments]),
        "persistent_state": args.persistent_state,
        "tbptt_horizon": args.tbptt_horizon,
        "records": records,
        "selection": decision,
        "all_state_gradient_gates_passed": all(record["state_gradient_gate_passed"] for record in records),
        "elapsed_seconds": time.monotonic() - started,
        "updater_peak_memory_gib": _peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": _peak_gib(runtime.reader_device),
    }
    _write_json(args.output_dir / "gradient_fidelity.json", report)
    return report


def _metric_ce(summary: Mapping[str, Any], key: str) -> float:
    value = summary["by_checkpoint_suite_condition"][key]["mean_ce"]
    if not isinstance(value, (int, float)):
        raise ValueError(f"R5 evaluation summary lacks numeric CE for {key}.")
    return float(value)


def _pilot_comparison(
    *,
    baseline_summary: Mapping[str, Any],
    endpoint_summary: Mapping[str, Any],
    technical_gate: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_formal = _metric_ce(baseline_summary, "m0|formal_select_32|normal")
    endpoint_formal = _metric_ce(endpoint_summary, "ema_step128|formal_select_32|normal")
    baseline_delayed = _metric_ce(baseline_summary, "m0|mechanism_select_32|normal")
    endpoint_delayed = _metric_ce(endpoint_summary, "ema_step128|mechanism_select_32|normal")
    reset = _metric_ce(endpoint_summary, "ema_step128|mechanism_select_32|reset")
    technical_pass = bool(technical_gate.get("passed"))
    mechanism_pass = endpoint_delayed < baseline_delayed and endpoint_delayed < reset
    return {
        "technical_gate_passed": technical_pass,
        "mechanism_gate_passed": mechanism_pass,
        "eligible_for_selection": technical_pass and mechanism_pass,
        "formal_select_ce": {
            "m0": baseline_formal,
            "endpoint": endpoint_formal,
            "delta": endpoint_formal - baseline_formal,
        },
        "delayed_mechanism_ce": {
            "m0": baseline_delayed,
            "endpoint": endpoint_delayed,
            "delta": endpoint_delayed - baseline_delayed,
        },
        "endpoint_reset_ce": reset,
        "endpoint_normal_minus_reset_ce": endpoint_delayed - reset,
        "selection_order": [
            "lowest_delayed_dev_ce",
            "lowest_formal_dev_ce",
            "highest_gap4_margin",
            "shortest_wall_clock",
        ],
    }


def _technical_gate_from_metrics(
    *,
    hard_noop: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    runtime: RuntimeBundle,
) -> dict[str, Any]:
    fractions = [
        float(metric["state_gradient_nonzero_fraction"])
        for metric in metrics
        if metric.get("state_gradient_nonzero_fraction") is not None
    ]
    state_fraction = _mean(fractions)
    finite_losses = all(math.isfinite(float(metric["loss_mean"])) for metric in metrics)
    passed = (
        bool(hard_noop.get("passed"))
        and finite_losses
        and state_fraction is not None
        and state_fraction >= 0.95
    )
    return {
        "hard_noop": dict(hard_noop),
        "finite_losses": finite_losses,
        "state_gradient_nonzero_fraction": state_fraction,
        "state_gradient_required_fraction": 0.95,
        "updater_peak_memory_gib": _peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": _peak_gib(runtime.reader_device),
        "passed": passed,
    }


def _checkpoint_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items()}


def run_training_profile(
    *,
    args: argparse.Namespace,
    optimizer_steps: int,
    data: R5DataBundle,
    runtime: RuntimeBundle,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    train_reader = choice_reader_callable(
        reader=runtime.reader,
        processor=runtime.processor,
        reader_device=runtime.reader_device,
        require_grad=True,
        deterministic_ce=args.strict_determinism,
    )
    eval_reader = choice_reader_callable(
        reader=runtime.reader,
        processor=runtime.processor,
        reader_device=runtime.reader_device,
        require_grad=False,
        deterministic_ce=args.strict_determinism,
    )
    ema = TrainableEMA(runtime.named_trainable, decay=args.ema_decay)
    hard_noop = _hard_noop_gate(runtime.model)
    if not hard_noop["passed"]:
        raise RuntimeError(f"R5 hard-NOOP gate failed: {hard_noop}")

    if args.resume is None:
        start_step = 0
        prior_elapsed = 0.0
        health_baseline: float | None = None
        consecutive_health_failures = 0
        _save_checkpoint(
            args.output_dir / "checkpoints" / "step-000000.pt",
            runtime=runtime,
            manifest=manifest,
            optimizer_step=0,
            ema=ema,
            elapsed=0.0,
            health_baseline=None,
            consecutive_health_failures=0,
        )
    else:
        start_step, prior_elapsed, health_baseline, consecutive_health_failures = _load_resume(
            args=args,
            runtime=runtime,
            manifest=manifest,
            ema=ema,
        )
        if start_step >= optimizer_steps:
            raise ValueError("R5 resume checkpoint already reached this profile endpoint.")
        _truncate_jsonl_for_resume(args.output_dir / "metrics.jsonl", optimizer_step=start_step)
        _truncate_jsonl_for_resume(args.output_dir / "micro_metrics.jsonl", optimizer_step=start_step)

    pilot_baseline_rows: list[dict[str, Any]] = []
    pilot_baseline_summary: dict[str, Any] | None = None
    pilot_like = args.profile in {"pilot", "rescue"}
    if pilot_like and start_step == 0:
        pilot_baseline_rows, pilot_baseline_summary = _pilot_evaluation(
            model=runtime.model,
            reader_fn=eval_reader,
            data=data,
            checkpoint_label="m0",
            bootstrap_iterations=args.bootstrap_iterations,
        )
        _write_json(args.output_dir / "pilot_m0_evaluation.json", pilot_baseline_summary)
        for row in pilot_baseline_rows:
            append_jsonl(args.output_dir / "pilot_evaluation_rows.jsonl", row)
    elif pilot_like:
        path = args.output_dir / "pilot_m0_evaluation.json"
        if not path.is_file():
            raise ValueError("R5 pilot resume is missing its fixed M0 evaluation.")
        pilot_baseline_summary = json.loads(path.read_text(encoding="utf-8"))

    if args.profile == "main" and args.health_eval and health_baseline is None:
        health_baseline = _normal_ce(
            model=runtime.model,
            reader_fn=eval_reader,
            records=data.dev_select_records,
            checkpoint_label="m0",
            permutations=(REVERSE_CYCLIC4[0],),
        )
        _append_metric(
            args.output_dir / "metrics.jsonl",
            {
                "schema": R5_METRICS_SCHEMA,
                "kind": "health_validation",
                "optimizer_step": 0,
                "endpoint": "raw_m0",
                "mean_ce": health_baseline,
                "resource_protection_only_not_selection": True,
                "elapsed_seconds": prior_elapsed,
            },
        )

    _reset_peak_memory(runtime)
    started = time.monotonic()
    step_metrics: list[dict[str, Any]] = []
    for step_zero in range(start_step, optimizer_steps):
        units = data.schedule[step_zero * 8 : (step_zero + 1) * 8]
        elapsed = prior_elapsed + time.monotonic() - started
        metric = _run_optimizer_step(
            step_zero=step_zero,
            units=units,
            runtime=runtime,
            ema=ema,
            reader_fn=train_reader,
            args=args,
            elapsed=elapsed,
        )
        elapsed = prior_elapsed + time.monotonic() - started
        metric["elapsed_seconds"] = elapsed
        micro_records = metric.pop("micro_records", None)
        _append_metric(args.output_dir / "metrics.jsonl", metric)
        if micro_records:
            for micro_record in micro_records:
                append_jsonl(args.output_dir / "micro_metrics.jsonl", micro_record)
        step_metrics.append(metric)
        print(
            json.dumps(
                {
                    "milestone": "optimizer_step",
                    "profile": args.profile,
                    "optimizer_step": step_zero + 1,
                    "loss_mean": metric["loss_mean"],
                    "gradient_norm": metric["gradient_norm_before_clip"],
                    "learning_rate": metric["learning_rate"],
                    "elapsed_seconds": elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        step = step_zero + 1
        checkpoint_due = step % args.checkpoint_every == 0 or step == optimizer_steps
        if checkpoint_due:
            _save_checkpoint(
                args.output_dir / "checkpoints" / f"step-{step:06d}.pt",
                runtime=runtime,
                manifest=manifest,
                optimizer_step=step,
                ema=ema,
                elapsed=elapsed,
                health_baseline=health_baseline,
                consecutive_health_failures=consecutive_health_failures,
            )
        if args.profile == "main" and args.health_eval and step % CHECKPOINT_EVERY == 0:
            assert health_baseline is not None
            with ema.apply(runtime.named_trainable):
                health_ce = _normal_ce(
                    model=runtime.model,
                    reader_fn=eval_reader,
                    records=data.dev_select_records,
                    checkpoint_label=f"ema_step{step}",
                    permutations=(REVERSE_CYCLIC4[0],),
                )
            degraded = health_ce > health_baseline * 1.30
            consecutive_health_failures = consecutive_health_failures + 1 if degraded else 0
            health_record = {
                "schema": R5_METRICS_SCHEMA,
                "kind": "health_validation",
                "optimizer_step": step,
                "endpoint": "ema",
                "mean_ce": health_ce,
                "m0_mean_ce": health_baseline,
                "relative_change": health_ce / health_baseline - 1.0,
                "degraded_over_30_percent": degraded,
                "consecutive_failures": consecutive_health_failures,
                "resource_protection_only_not_selection": True,
                "elapsed_seconds": prior_elapsed + time.monotonic() - started,
            }
            _append_metric(args.output_dir / "metrics.jsonl", health_record)
            if consecutive_health_failures >= 2:
                _write_json(
                    args.output_dir / "terminal_failure.json",
                    {
                        "schema": "vision_memory.r5-resource-protection-stop.v1",
                        "status": "stopped",
                        "reason": "two consecutive health evaluations exceeded M0 CE by more than 30%",
                        "last_health": health_record,
                    },
                )
                raise RuntimeError("R5 resource-protection rule triggered after two health degradations.")

    elapsed = prior_elapsed + time.monotonic() - started
    raw_endpoint = _save_checkpoint(
        args.output_dir / "endpoint_raw.pt",
        runtime=runtime,
        manifest=manifest,
        optimizer_step=optimizer_steps,
        ema=ema,
        elapsed=elapsed,
        health_baseline=health_baseline,
        consecutive_health_failures=consecutive_health_failures,
    )
    ema_endpoint = _save_ema_checkpoint(
        args.output_dir / "endpoint_ema.pt",
        runtime=runtime,
        manifest=manifest,
        optimizer_step=optimizer_steps,
        ema=ema,
        elapsed=elapsed,
        health_baseline=health_baseline,
        consecutive_health_failures=consecutive_health_failures,
    )
    snapshot_verification = _verify_end_bindings(manifest)
    _write_json(args.output_dir / "model_snapshot_verification_end.json", snapshot_verification)
    technical_gate = _technical_gate_from_metrics(
        hard_noop=hard_noop,
        metrics=step_metrics,
        runtime=runtime,
    )

    pilot_endpoint_summary: dict[str, Any] | None = None
    pilot_comparison: dict[str, Any] | None = None
    if pilot_like:
        assert pilot_baseline_summary is not None
        with ema.apply(runtime.named_trainable):
            endpoint_rows, pilot_endpoint_summary = _pilot_evaluation(
                model=runtime.model,
                reader_fn=eval_reader,
                data=data,
                checkpoint_label="ema_step128",
                bootstrap_iterations=args.bootstrap_iterations,
            )
        _write_json(args.output_dir / "pilot_endpoint_evaluation.json", pilot_endpoint_summary)
        for row in endpoint_rows:
            append_jsonl(args.output_dir / "pilot_evaluation_rows.jsonl", row)
        pilot_comparison = _pilot_comparison(
            baseline_summary=pilot_baseline_summary,
            endpoint_summary=pilot_endpoint_summary,
            technical_gate=technical_gate,
        )
        _write_json(args.output_dir / "pilot_selection_record.json", pilot_comparison)

    checkpoint_paths = {"raw": raw_endpoint, "ema": ema_endpoint}
    throughput_seconds = max(elapsed - prior_elapsed, 1e-9)
    summary = {
        "schema": R5_SUMMARY_SCHEMA,
        "status": "completed",
        "profile": args.profile,
        "persistent_state": args.persistent_state,
        "tbptt_horizon": args.tbptt_horizon,
        "gradient_mode": args.gradient_mode,
        "selected_step_count": args.selected_step_count,
        "residual_blend": args.residual_blend,
        "seed": args.seed,
        "optimizer_steps": optimizer_steps,
        "micro_segments": optimizer_steps * 8,
        "elapsed_seconds": elapsed,
        "training_interval_seconds": throughput_seconds,
        "optimizer_steps_per_second": (optimizer_steps - start_step) / throughput_seconds,
        "micro_segments_per_second": (optimizer_steps - start_step) * 8 / throughput_seconds,
        "updater_peak_memory_gib": _peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": _peak_gib(runtime.reader_device),
        "same_device": runtime.updater_device == runtime.reader_device,
        "technical_gate": technical_gate,
        "clip_rate": _mean(float(bool(metric["gradient_clipped"])) for metric in step_metrics),
        "checkpoint_sha256": _checkpoint_hashes(checkpoint_paths),
        "primary_endpoint": "endpoint_ema.pt",
        "secondary_endpoint": "endpoint_raw.pt",
        "checkpoint_selection": "fixed_endpoint_not_best_dev",
        "pilot_m0_evaluation": pilot_baseline_summary,
        "pilot_endpoint_evaluation": pilot_endpoint_summary,
        "pilot_selection": pilot_comparison,
        "model_snapshot_end_verification": snapshot_verification,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def _checkpoint_label(path: Path) -> str:
    normalized = path.name.casefold().replace("_", "-")
    if "step-000000" in normalized or normalized.startswith("m0"):
        return "m0"
    if "ema" in normalized:
        return "ema_step640"
    if "raw" in normalized:
        return "raw_step640"
    return path.stem


def _checkpoint_manifest(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("schema") != R5_TRAINING_MANIFEST_SCHEMA:
        raise ValueError(f"R5 endpoint checkpoint has invalid manifest: {path}")
    return manifest


def _validate_evaluation_checkpoint(
    *,
    path: Path,
    args: argparse.Namespace,
    reference_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest = _checkpoint_manifest(path)
    contract = manifest.get("protocol_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"R5 checkpoint lacks protocol contract: {path}")
    expected = {
        "persistent_state": args.persistent_state,
        "tbptt_horizon": args.tbptt_horizon,
        "gradient_mode": args.gradient_mode,
        "selected_steps_per_transition": args.selected_step_count,
    }
    mismatches = {key: (value, contract.get(key)) for key, value in expected.items() if contract.get(key) != value}
    if mismatches:
        raise ValueError(f"R5 evaluation checkpoint contract mismatch for {path}: {mismatches}")
    invocation = manifest.get("arguments")
    if not isinstance(invocation, Mapping) or int(invocation.get("seed", -1)) != args.seed:
        raise ValueError(f"R5 evaluation seed differs from checkpoint: {path}")
    if manifest.get("train_sha256") != sha256_file(args.train) or manifest.get("dev_sha256") != sha256_file(args.dev):
        raise ValueError(f"R5 evaluation dataset binding differs from checkpoint: {path}")
    supervision = manifest.get("supervision")
    if not isinstance(supervision, Mapping) or float(supervision.get("residual_blend", -1.0)) != args.residual_blend:
        raise ValueError(f"R5 evaluation residual blend differs from checkpoint: {path}")
    if reference_manifest is not None:
        immutable = (
            "git_commit",
            "train_sha256",
            "dev_sha256",
            "protocol_contract",
            "family_pool_audit",
            "schedule_audit",
            "dev_split_audit",
            "model_snapshot_manifests",
        )
        drift = {key: (reference_manifest.get(key), manifest.get(key)) for key in immutable if reference_manifest.get(key) != manifest.get(key)}
        if drift:
            raise ValueError(f"R5 endpoint checkpoints do not share immutable lineage: {drift}")
    return manifest


def _export_state_examples(
    *,
    model: R5ComposeModel,
    pools: Mapping[str, Sequence[R5Segment]],
    output_dir: Path,
) -> dict[str, Any]:
    from torchvision.transforms.functional import to_pil_image

    destination = output_dir / "state_examples"
    destination.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []
    with torch.no_grad():
        for family in R5_FAMILIES:
            for case_index, segment in enumerate(pools[family][:2]):
                state = model.reset_state()
                images: list[str] = []
                initial_image = model.reader_image(state).detach().float().clamp(0.0, 1.0)[0].cpu()
                initial_name = f"{family.casefold()}-{case_index}-state-00-initial.png"
                to_pil_image(initial_image).save(destination / initial_name)
                images.append(initial_name)
                for event_index, event in enumerate(segment.events, start=1):
                    state = model.apply_event(
                        state,
                        event,
                        gradient_mode="full",
                        selected_step_indices=None,
                    )
                    image = model.reader_image(state).detach().float().clamp(0.0, 1.0)[0].cpu()
                    name = f"{family.casefold()}-{case_index}-state-{event_index:02d}-{event.event_kind}.png"
                    to_pil_image(image).save(destination / name)
                    images.append(name)
                cases.append(
                    {
                        "family": family,
                        "case_index": case_index,
                        "segment": segment.to_dict(),
                        "images": images,
                    }
                )
    index = {
        "schema": "vision_memory.r5-compose-state-examples.v1",
        "checkpoint": "primary_ema_endpoint",
        "persistent_state": model.persistent_state,
        "cases": cases,
    }
    _write_json(destination / "index.json", index)
    return index


def _endpoint_comparisons(summary: Mapping[str, Any], *, primary: str) -> dict[str, Any]:
    metrics = summary["by_checkpoint_suite_condition"]
    comparisons: dict[str, Any] = {}
    for suite in ("formal_final_128", "mechanism_final_128"):
        baseline_key = f"m0|{suite}|normal"
        primary_key = f"{primary}|{suite}|normal"
        raw_key = f"raw_step640|{suite}|normal"
        if baseline_key not in metrics or primary_key not in metrics:
            continue
        baseline = float(metrics[baseline_key]["mean_ce"])
        endpoint = float(metrics[primary_key]["mean_ce"])
        value: dict[str, Any] = {
            "m0_ce": baseline,
            "primary_ema_ce": endpoint,
            "primary_minus_m0_ce": endpoint - baseline,
            "relative_ce_change": endpoint / baseline - 1.0,
        }
        if raw_key in metrics:
            raw = float(metrics[raw_key]["mean_ce"])
            value.update({"raw_ce": raw, "raw_minus_m0_ce": raw - baseline, "ema_minus_raw_ce": endpoint - raw})
        comparisons[suite] = value
    return comparisons


def run_final_evaluation(
    *,
    args: argparse.Namespace,
    data: R5DataBundle,
    runtime: RuntimeBundle,
) -> dict[str, Any]:
    eval_reader = choice_reader_callable(
        reader=runtime.reader,
        processor=runtime.processor,
        reader_device=runtime.reader_device,
        require_grad=False,
        deterministic_ce=args.strict_determinism,
    )
    checkpoint_paths = [path.resolve() for path in args.checkpoint]
    labels = [_checkpoint_label(path) for path in checkpoint_paths]
    if len(set(labels)) != len(labels):
        raise ValueError(f"R5 final evaluation checkpoint labels are not unique: {labels}")
    reference_manifest: dict[str, Any] | None = None
    manifests: dict[str, dict[str, Any]] = {}
    for label, path in zip(labels, checkpoint_paths, strict=True):
        manifest = _validate_evaluation_checkpoint(
            path=path,
            args=args,
            reference_manifest=reference_manifest,
        )
        if reference_manifest is None:
            reference_manifest = manifest
        manifests[label] = manifest
    if "m0" not in labels:
        raise ValueError("R5 final evaluation requires the fixed step-zero checkpoint.")
    primary = "ema_step640" if "ema_step640" in labels else labels[-1]
    all_controls = (
        "normal",
        "reset",
        "cross_episode_swap",
        "temporal_swap",
        "updater_disabled",
    )
    rows: list[dict[str, Any]] = []
    rows_path = args.output_dir / "evaluation_rows.jsonl"
    if rows_path.exists():
        rows_path.unlink()
    started = time.monotonic()
    for label, path in zip(labels, checkpoint_paths, strict=True):
        load_trainable_weights(path, trainable_module=runtime.model)
        controls = all_controls if label == primary else ("normal",)
        formal_rows = evaluate_items(
            model=runtime.model,
            reader_fn=eval_reader,
            items=_formal_eval_items(runtime.model, data.dev_final_records),
            checkpoint_label=label,
            suite="formal_final_128",
            controls=controls,
            permutations=REVERSE_CYCLIC4,
        )
        mechanism_rows = evaluate_items(
            model=runtime.model,
            reader_fn=eval_reader,
            items=_segment_eval_items(runtime.model, data.mechanism_final),
            checkpoint_label=label,
            suite="mechanism_final_128",
            controls=controls,
            permutations=REVERSE_CYCLIC4,
        )
        checkpoint_rows = formal_rows + mechanism_rows
        rows.extend(checkpoint_rows)
        for row in checkpoint_rows:
            append_jsonl(rows_path, row)
        print(
            json.dumps(
                {
                    "milestone": "final_evaluation_checkpoint",
                    "checkpoint": label,
                    "rows": len(checkpoint_rows),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = summarize_evaluation_rows(
        rows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=20260730 + args.seed,
    )
    comparisons = _endpoint_comparisons(summary, primary=primary)
    load_trainable_weights(checkpoint_paths[labels.index(primary)], trainable_module=runtime.model)
    state_examples = _export_state_examples(
        model=runtime.model,
        pools=data.train_pools,
        output_dir=args.output_dir,
    )
    assert reference_manifest is not None
    snapshot_verification = _verify_end_bindings(reference_manifest)
    result = {
        **summary,
        "status": "completed",
        "seed": args.seed,
        "primary_checkpoint": primary,
        "checkpoint_paths": {label: str(path) for label, path in zip(labels, checkpoint_paths, strict=True)},
        "checkpoint_sha256": {label: sha256_file(path) for label, path in zip(labels, checkpoint_paths, strict=True)},
        "endpoint_comparisons": comparisons,
        "causal_controls": list(all_controls),
        "choice_views": "reverse_cyclic4",
        "bootstrap_unit": "episode for formal; composed segment for mechanism",
        "state_examples_index": str(args.output_dir / "state_examples" / "index.json"),
        "state_example_count": len(state_examples["cases"]),
        "elapsed_seconds": time.monotonic() - started,
        "updater_peak_memory_gib": _peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": _peak_gib(runtime.reader_device),
        "model_snapshot_end_verification": snapshot_verification,
    }
    _write_json(args.output_dir / "evaluation_summary.json", result)
    return result


def _prepare_output(args: argparse.Namespace) -> None:
    if args.resume is None and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("A fresh R5 run refuses a non-empty output directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        optimizer_steps = _validate_args(args)
        _prepare_output(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    determinism = configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    set_all_seeds(args.seed)
    data = _load_data(args, optimizer_steps=optimizer_steps)
    _write_json(args.output_dir / "family_pool_audit.json", data.pool_audit)
    _write_json(args.output_dir / "dev_split_audit.json", data.split_audit)
    if data.schedule_audit is not None:
        _write_json(args.output_dir / "schedule_audit.json", data.schedule_audit)
    runtime = _load_runtime(args)
    _write_environment(args.output_dir / "environment.txt")
    _write_json(args.output_dir / "runtime.json", _runtime_versions())
    if args.profile == "gradient-audit":
        audit_manifest = _make_manifest(
            args=args,
            optimizer_steps=0,
            data=data,
            determinism=determinism,
        )
        _write_json(args.output_dir / "manifest.json", audit_manifest)
        reader_fn = choice_reader_callable(
            reader=runtime.reader,
            processor=runtime.processor,
            reader_device=runtime.reader_device,
            require_grad=True,
            deterministic_ce=args.strict_determinism,
        )
        _reset_peak_memory(runtime)
        report = run_gradient_audit(args=args, runtime=runtime, data=data, reader_fn=reader_fn)
        print(json.dumps({"milestone": "completed", "profile": args.profile, "selection": report["selection"]}, sort_keys=True), flush=True)
        return 0
    if args.profile == "final-eval":
        result = run_final_evaluation(args=args, data=data, runtime=runtime)
        print(
            json.dumps(
                {
                    "milestone": "completed",
                    "profile": args.profile,
                    "primary_checkpoint": result["primary_checkpoint"],
                    "elapsed_seconds": result["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    manifest = _make_manifest(
        args=args,
        optimizer_steps=optimizer_steps,
        data=data,
        determinism=determinism,
    )
    _write_json(args.output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "milestone": "startup_validated",
                "profile": args.profile,
                "optimizer_steps": optimizer_steps,
                "micro_segments": optimizer_steps * 8,
                "persistent_state": args.persistent_state,
                "tbptt_horizon": args.tbptt_horizon,
                "same_device": runtime.updater_device == runtime.reader_device,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    summary = run_training_profile(
        args=args,
        optimizer_steps=optimizer_steps,
        data=data,
        runtime=runtime,
        manifest=manifest,
    )
    print(
        json.dumps(
            {
                "milestone": "completed",
                "profile": args.profile,
                "optimizer_steps": optimizer_steps,
                "elapsed_seconds": summary["elapsed_seconds"],
                "technical_gate_passed": summary["technical_gate"]["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
