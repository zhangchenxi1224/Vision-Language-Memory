"""R6 paired bottleneck diagnostic for source-anchored DreamLite updates.

This is deliberately not a formal-success run.  It compares the R5 pure-noise
event rewrite (start sigma 1.0) with a flow-matching-consistent, source-anchored
event update (start sigma 0.5) while keeping the frozen Reader, LoRA contract,
data, loss, optimizer, and fixed dev suites unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train.dreamlite_episode import (  # noqa: E402
    choice_reader_callable,
    compute_dtype,
    git_value,
    set_all_seeds,
    sha256_file,
)
from vision_memory.data import REVERSE_CYCLIC4  # noqa: E402
from vision_memory.data.schema import EventKind  # noqa: E402
from vision_memory.dreamlite import DreamLiteRecurrentUpdater  # noqa: E402
from vision_memory.repro import configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import load_trainable_weights  # noqa: E402
from vision_memory.training.r5_compose import (  # noqa: E402
    R5Event,
    R5Segment,
    ScheduledR5Segment,
    canonical_sha256,
    schedule_audit,
)


R6_PROTOCOL = "R6-SourceAnchor-Bottleneck"
R6_IMPLEMENTATION_REVISION = "scheduler-effective-sigma-v2"
R6_SCHEMA = "vision_memory.r6-source-anchor-summary.v1"
R6_MANIFEST_SCHEMA = "vision_memory.r6-source-anchor-manifest.v1"
R6_GRADIENT_SCHEMA = "vision_memory.r6-gradient-conflict-audit.v1"
R6_OPTIMIZER_STEPS = 128
R6_OVERFIT_FAMILIES = ("F2", "F3", "F5", "F6")
R6_SEGMENTS_PER_FAMILY = 2
ARM_SIGMA = {
    "legacy-pure-noise": 1.0,
    "source-anchored": 0.5,
}
SOURCE_ANCHOR_EFFECTIVE_SIGMAS = (0.5, 0.375, 0.25, 0.125)


class R6SourceAnchorModel(nn.Module):
    """Latent picture state with a fixed event-level diffusion start sigma."""

    def __init__(
        self,
        *,
        pipeline: Any,
        initial_rgb: Tensor,
        global_seed: int,
        checkpoint_unet: bool,
        edit_start_sigma: float,
    ) -> None:
        super().__init__()
        self.updater = DreamLiteRecurrentUpdater(
            pipeline=pipeline,
            global_seed=global_seed,
            checkpoint_unet=checkpoint_unet,
        )
        self.persistent_state = "latent"
        self.residual_blend = 1.0
        self.edit_start_sigma = float(edit_start_sigma)
        with torch.no_grad():
            initial = self.updater.encode_persistent_rgb(initial_rgb.detach())
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
            return state
        return self.updater(
            state,
            event.event_text,
            event.source_episode_id,
            event.noise_turn_id,
            gradient_mode=gradient_mode,
            selected_step_indices=selected_step_indices,
            persistent_state="latent",
            presentation_index=0,
            noise_include_presentation_index=False,
            edit_start_sigma=self.edit_start_sigma,
        )

    def reader_image(self, state: Tensor) -> Tensor:
        return self.updater.decode_for_reader(state, persistent_state="latent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=R6_PROTOCOL)
    parser.add_argument("--arm", choices=tuple(ARM_SIGMA), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-seed", type=int)
    parser.add_argument("--schedule-seed", type=int, default=20260829)
    parser.add_argument("--pairing-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--checkpoint-unet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.adapter_seed is None:
        args.adapter_seed = args.seed
    # Fields consumed by the audited R5 training core.  They are deliberately
    # fixed rather than exposed as R6 sweep dimensions.
    args.profile = "pilot"
    args.persistent_state = "latent"
    args.tbptt_horizon = 4
    args.gradient_mode = "full"
    args.selected_step_count = 0
    args.gradient_accumulation = r5.GRADIENT_ACCUMULATION
    args.weight_decay = r5.WEIGHT_DECAY
    args.gradient_clip = r5.GRADIENT_CLIP
    args.lora_rank = r5.LORA_RANK
    args.ema_decay = r5.EMA_DECAY
    args.residual_blend = 1.0
    args.checkpoint_every = 32
    args.max_optimizer_steps = R6_OPTIMIZER_STEPS
    args.resume = None
    args.checkpoint = []
    args.audit_state_gradients = True
    args.gradient_audit_size = 24
    args.health_eval = False
    args.record_micro_metrics = True
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.resolution != 1024:
        raise ValueError("R6 preserves the fixed 1024x1024 visual/Reader contract.")
    for name in ("train", "dev"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R6 {name} path is not a file: {getattr(args, name)}")
    for name in ("dreamlite", "reader"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R6 {name} path is not a directory: {getattr(args, name)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("A fresh R6 arm refuses a non-empty output directory.")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise ValueError("R6 refuses a dirty source tree unless --allow-dirty is explicit.")


def _stable_segments(
    pools: Mapping[str, Sequence[R5Segment]],
    *,
    schedule_seed: int,
) -> tuple[R5Segment, ...]:
    selected: list[R5Segment] = []
    for family in R6_OVERFIT_FAMILIES:
        ordered = sorted(
            pools[family],
            key=lambda segment: (
                hashlib.sha256(
                    f"{R6_PROTOCOL}\x1f{schedule_seed}\x1f{family}\x1f{segment.segment_id}".encode()
                ).digest(),
                segment.segment_id,
            ),
        )
        if len(ordered) < R6_SEGMENTS_PER_FAMILY:
            raise ValueError(f"R6 needs at least two {family} segments.")
        selected.extend(ordered[:R6_SEGMENTS_PER_FAMILY])
    if len({segment.segment_id for segment in selected}) != 8:
        raise RuntimeError("R6 overfit segment selection is not unique.")
    return tuple(selected)


def _overfit_schedule(
    segments: Sequence[R5Segment],
    *,
    optimizer_steps: int,
    schedule_seed: int,
) -> tuple[ScheduledR5Segment, ...]:
    draws: defaultdict[str, int] = defaultdict(int)
    units: list[ScheduledR5Segment] = []
    global_micro = 0
    for step_zero in range(optimizer_steps):
        order = sorted(
            segments,
            key=lambda segment: (
                hashlib.sha256(
                    f"{R6_PROTOCOL}\x1f{schedule_seed}\x1f{step_zero}\x1f{segment.segment_id}".encode()
                ).digest(),
                segment.segment_id,
            ),
        )
        for micro_in_step, segment in enumerate(order):
            units.append(
                ScheduledR5Segment(
                    global_micro_index=global_micro,
                    optimizer_step_zero=step_zero,
                    micro_in_step=micro_in_step,
                    phase="r6_bottleneck_overfit",
                    family_draw_index=draws[segment.family],
                    segment=segment,
                    selected_step_indices=None,
                )
            )
            draws[segment.family] += 1
            global_micro += 1
    return tuple(units)


def _load_data(args: argparse.Namespace) -> tuple[r5.R5DataBundle, tuple[R5Segment, ...]]:
    base = r5._load_data(args, optimizer_steps=R6_OPTIMIZER_STEPS)
    selected = _stable_segments(base.train_pools, schedule_seed=args.schedule_seed)
    scheduled = _overfit_schedule(
        selected,
        optimizer_steps=R6_OPTIMIZER_STEPS,
        schedule_seed=args.schedule_seed,
    )
    audit = {
        **schedule_audit(scheduled),
        "r6_protocol": R6_PROTOCOL,
        "repeated_fixed_subset_diagnostic_only": True,
        "selected_segment_ids": [segment.segment_id for segment in selected],
        "selected_segments_sha256": canonical_sha256([segment.to_dict() for segment in selected]),
        "family_counts_per_optimizer_step": dict(Counter(segment.family for segment in selected)),
    }
    return replace(base, schedule=scheduled, schedule_audit=audit), selected


def _load_runtime(args: argparse.Namespace) -> r5.RuntimeBundle:
    if not torch.cuda.is_available():
        raise RuntimeError("R6 requires CUDA.")
    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    pipe = r5._load_pipeline(args, updater_device, compute_dtype(updater_device))
    processor, reader = r5._load_reader(args, reader_device, compute_dtype(reader_device))
    model = R6SourceAnchorModel(
        pipeline=pipe,
        initial_rgb=r5._initial_rgb_tensor(
            resolution=args.resolution,
            device=updater_device,
            dtype=compute_dtype(updater_device),
        ),
        global_seed=args.seed,
        checkpoint_unet=args.checkpoint_unet,
        edit_start_sigma=ARM_SIGMA[args.arm],
    )
    named, trainable = r5._force_trainable_fp32(model)
    optimizer = torch.optim.AdamW(trainable, lr=r5.LR_START, weight_decay=args.weight_decay)
    return r5.RuntimeBundle(
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


def _manifest(
    *,
    args: argparse.Namespace,
    data: r5.R5DataBundle,
    selected: Sequence[R5Segment],
    determinism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": R6_MANIFEST_SCHEMA,
        "protocol": R6_PROTOCOL,
        "implementation_revision": R6_IMPLEMENTATION_REVISION,
        "diagnostic_only_not_formal_success": True,
        "hypothesis": "R5 recurrence fails because every non-NOOP event restarts from pure noise and redraws the full state.",
        "arm": args.arm,
        "edit_start_sigma": ARM_SIGMA[args.arm],
        "flow_initialization": "x_sigma=(1-sigma)*source_latent+sigma*fixed_event_noise",
        "edit_start_sigma_semantics": "effective post-scheduler-shift flow sigma",
        "source_anchor_effective_sigma_schedule": list(SOURCE_ANCHOR_EFFECTIVE_SIGMAS),
        "scheduler_alignment_fail_closed": True,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(git_value("status", "--porcelain")),
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "selected_segments": [segment.to_dict() for segment in selected],
        "selected_segments_sha256": canonical_sha256([segment.to_dict() for segment in selected]),
        "family_pool_audit": data.pool_audit,
        "schedule_audit": data.schedule_audit,
        "dev_split_audit": data.split_audit,
        "fixed_contract": {
            "persistent_state": "latent",
            "tbptt_horizon": 4,
            "gradient_mode": "full",
            "diffusion_steps": 4,
            "hard_noop_identity": True,
            "fixed_episode_turn_noise": True,
            "lora_rank": 4,
            "optimizer": "AdamW",
            "optimizer_steps": R6_OPTIMIZER_STEPS,
            "gradient_accumulation": 8,
            "gradient_clip": 10.0,
            "weight_decay": 1e-4,
            "reader": "frozen Qwen3-VL-4B-Instruct",
            "updater_base": "frozen DreamLite-mobile with U-Net LoRA only",
            "loss": "four-choice listwise CE from frozen Reader",
            "ema_decay": 0.995,
            "primary_endpoint": "fixed EMA step128",
        },
        "success_boundary": {
            "this_run_can_only_diagnose": True,
            "formal_success_requires_fixed_full_data_multi_seed_id_ood_and_causal_controls": True,
        },
        "strict_determinism": dict(determinism) if determinism is not None else None,
        "environment": r5._runtime_versions(),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in sorted(vars(args).items())
            if key not in {"output_dir", "resume", "checkpoint"}
        },
        **r5._manifest_model_bindings(args),
    }


def _cosine(first: Tensor, second: Tensor) -> float:
    denominator = first.double().norm() * second.double().norm()
    if float(denominator) == 0.0:
        raise ValueError("R6 gradient cosine received a zero vector.")
    return float(torch.dot(first.double(), second.double()) / denominator)


def _gradient_conflict_audit(
    *,
    runtime: r5.RuntimeBundle,
    reader_fn: Any,
    segments: Sequence[R5Segment],
) -> dict[str, Any]:
    vectors: list[Tensor] = []
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, segment in enumerate(segments):
        vector, metadata = r5._audit_gradient_for_policy(
            runtime=runtime,
            reader_fn=reader_fn,
            segment=segment,
            tbptt_horizon=4,
            gradient_mode="full",
            selected_steps=None,
        )
        vectors.append(vector)
        records.append(
            {
                "index": index,
                "segment_id": segment.segment_id,
                "family": segment.family,
                "loss": metadata["loss"],
                "gradient_norm": metadata["gradient_norm"],
                "state_gradient": metadata["state_gradient"],
            }
        )
        print(
            json.dumps(
                {
                    "milestone": "r6_gradient_conflict_segment",
                    "completed": index + 1,
                    "total": len(segments),
                    "family": segment.family,
                    "gradient_norm": metadata["gradient_norm"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    matrix = [[_cosine(first, second) for second in vectors] for first in vectors]
    off_diagonal = [matrix[i][j] for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    norms = np.asarray([float(vector.double().norm()) for vector in vectors], dtype=np.float64)
    raw_sum = torch.stack(vectors).sum(dim=0)
    unit_sum = torch.stack([vector / vector.norm() for vector in vectors]).sum(dim=0)
    for record, vector, norm in zip(records, vectors, norms, strict=True):
        record["norm_share_of_sum_of_norms"] = float(norm / norms.sum())
        record["cosine_to_raw_batch_gradient"] = _cosine(vector, raw_sum)
        record["cosine_to_unit_balanced_gradient"] = _cosine(vector, unit_sum)
    return {
        "schema": R6_GRADIENT_SCHEMA,
        "status": "completed",
        "segment_count": len(segments),
        "records": records,
        "pairwise_cosine": matrix,
        "off_diagonal": {
            "median": float(np.median(off_diagonal)),
            "minimum": float(np.min(off_diagonal)),
            "maximum": float(np.max(off_diagonal)),
            "negative_fraction": float(np.mean(np.asarray(off_diagonal) < 0.0)),
        },
        "gradient_norm": {
            "minimum": float(norms.min()),
            "median": float(np.median(norms)),
            "maximum": float(norms.max()),
            "max_to_min_ratio": float(norms.max() / norms.min()),
        },
        "raw_vs_unit_balanced_cosine": _cosine(raw_sum, unit_sum),
        "elapsed_seconds": time.monotonic() - started,
    }


def _evaluation_rows(
    *,
    model: R6SourceAnchorModel,
    reader_fn: Any,
    segments: Sequence[R5Segment],
    checkpoint: str,
) -> list[dict[str, Any]]:
    return r5.evaluate_items(
        model=model,
        reader_fn=reader_fn,
        items=r5._segment_eval_items(model, segments),
        checkpoint_label=checkpoint,
        suite="train_overfit_hard8",
        controls=("normal", "reset", "cross_episode_swap", "temporal_swap"),
        permutations=REVERSE_CYCLIC4,
    )


def _per_unit_means(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str,
    suite: str,
    condition: str,
) -> dict[str, float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            str(row["checkpoint"]) == checkpoint
            and str(row["suite"]) == suite
            and str(row["condition"]) == condition
        ):
            grouped[str(row["pair_unit"])].append(float(row["ce"]))
    return {unit: float(np.mean(values)) for unit, values in grouped.items()}


def _bootstrap_differences(
    differences: Sequence[float],
    *,
    iterations: int,
    seed: int,
    definition: str,
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("R6 paired bootstrap requires finite matched differences.")
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = values[rng.integers(0, len(values), size=len(values))].mean()
    return {
        "definition": definition,
        "pair_units": len(values),
        "iterations": iterations,
        "seed": seed,
        "estimate": float(values.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "fraction_below_zero": float(np.mean(draws < 0.0)),
    }


def _checkpoint_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    endpoint: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    m0 = _per_unit_means(rows, checkpoint="m0", suite=suite, condition="normal")
    final = _per_unit_means(rows, checkpoint=endpoint, suite=suite, condition="normal")
    if set(m0) != set(final) or not m0:
        raise ValueError(f"R6 endpoint comparison is not paired for {suite}.")
    units = sorted(m0)
    differences = [final[unit] - m0[unit] for unit in units]
    bootstrap = _bootstrap_differences(
        differences,
        iterations=iterations,
        seed=seed,
        definition=f"mean_ce({endpoint},normal)-mean_ce(m0,normal) on {suite}",
    )
    m0_mean = float(np.mean([m0[unit] for unit in units]))
    endpoint_mean = float(np.mean([final[unit] for unit in units]))
    return {
        **bootstrap,
        "m0_mean_ce": m0_mean,
        "endpoint_mean_ce": endpoint_mean,
        "relative_change": endpoint_mean / m0_mean - 1.0,
        "improved_pair_units": sum(difference < 0.0 for difference in differences),
        "per_pair_unit_delta": dict(zip(units, differences, strict=True)),
    }


def _difference_in_differences(
    rows: Sequence[Mapping[str, Any]],
    *,
    suite: str,
    endpoint: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    values = {
        (checkpoint, condition): _per_unit_means(
            rows,
            checkpoint=checkpoint,
            suite=suite,
            condition=condition,
        )
        for checkpoint in ("m0", endpoint)
        for condition in ("normal", "reset")
    }
    unit_sets = [set(value) for value in values.values()]
    if not unit_sets or any(unit_set != unit_sets[0] for unit_set in unit_sets[1:]):
        raise ValueError(f"R6 difference-in-differences is not paired for {suite}.")
    units = sorted(unit_sets[0])
    differences = [
        (
            values[(endpoint, "normal")][unit]
            - values[(endpoint, "reset")][unit]
        )
        - (
            values[("m0", "normal")][unit]
            - values[("m0", "reset")][unit]
        )
        for unit in units
    ]
    return _bootstrap_differences(
        differences,
        iterations=iterations,
        seed=seed,
        definition="(endpoint_normal-endpoint_reset)-(m0_normal-m0_reset); negative means training-induced state dependence",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        r5.append_jsonl(path, dict(row))


def _metric(summary: Mapping[str, Any], key: str, field: str) -> float:
    value = summary["by_checkpoint_suite_condition"][key][field]
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"R6 missing finite metric {key}:{field}.")
    return float(value)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    determinism = configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    set_all_seeds(args.seed)
    data, selected = _load_data(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    r5._write_json(args.output_dir / "family_pool_audit.json", data.pool_audit)
    r5._write_json(args.output_dir / "dev_split_audit.json", data.split_audit)
    r5._write_json(args.output_dir / "schedule_audit.json", data.schedule_audit)
    runtime = _load_runtime(args)
    r5._write_environment(args.output_dir / "environment.txt")
    r5._write_json(args.output_dir / "runtime.json", r5._runtime_versions())
    manifest = _manifest(
        args=args,
        data=data,
        selected=selected,
        determinism=determinism,
    )
    r5._write_json(args.output_dir / "manifest.json", manifest)

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

    print(json.dumps({"milestone": "r6_m0_overfit_eval", "arm": args.arm}, sort_keys=True), flush=True)
    m0_rows = _evaluation_rows(
        model=runtime.model,
        reader_fn=eval_reader,
        segments=selected,
        checkpoint="m0",
    )
    overfit_rows_path = args.output_dir / "overfit_evaluation_rows.jsonl"
    _append_rows(overfit_rows_path, m0_rows)

    gradient_audit = _gradient_conflict_audit(
        runtime=runtime,
        reader_fn=train_reader,
        segments=selected,
    )
    r5._write_json(args.output_dir / "gradient_conflict_audit.json", gradient_audit)

    print(
        json.dumps(
            {
                "milestone": "r6_training_start",
                "arm": args.arm,
                "edit_start_sigma": ARM_SIGMA[args.arm],
                "optimizer_steps": R6_OPTIMIZER_STEPS,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    training_summary = r5.run_training_profile(
        args=args,
        optimizer_steps=R6_OPTIMIZER_STEPS,
        data=data,
        runtime=runtime,
        manifest=manifest,
    )

    endpoint_path = args.output_dir / "endpoint_ema.pt"
    load_trainable_weights(endpoint_path, trainable_module=runtime.model)
    endpoint_label = "ema_step128"
    print(json.dumps({"milestone": "r6_endpoint_overfit_eval", "arm": args.arm}, sort_keys=True), flush=True)
    endpoint_rows = _evaluation_rows(
        model=runtime.model,
        reader_fn=eval_reader,
        segments=selected,
        checkpoint=endpoint_label,
    )
    _append_rows(overfit_rows_path, endpoint_rows)
    overfit_rows = m0_rows + endpoint_rows
    overfit_summary = r5.summarize_evaluation_rows(
        overfit_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=20260829,
    )
    r5._write_json(args.output_dir / "overfit_evaluation_summary.json", overfit_summary)

    pilot_rows = _read_jsonl(args.output_dir / "pilot_evaluation_rows.jsonl")
    all_rows = overfit_rows + pilot_rows
    comparisons = {
        "train_overfit_hard8_endpoint_vs_m0": _checkpoint_comparison(
            all_rows,
            suite="train_overfit_hard8",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260830,
        ),
        "train_overfit_hard8_state_did": _difference_in_differences(
            all_rows,
            suite="train_overfit_hard8",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260831,
        ),
        "formal_select_32_endpoint_vs_m0": _checkpoint_comparison(
            all_rows,
            suite="formal_select_32",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260832,
        ),
        "mechanism_select_32_endpoint_vs_m0": _checkpoint_comparison(
            all_rows,
            suite="mechanism_select_32",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260833,
        ),
        "mechanism_select_32_state_did": _difference_in_differences(
            all_rows,
            suite="mechanism_select_32",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260834,
        ),
    }
    overfit_delta = comparisons["train_overfit_hard8_endpoint_vs_m0"]
    overfit_did = comparisons["train_overfit_hard8_state_did"]
    formal_delta = comparisons["formal_select_32_endpoint_vs_m0"]
    mechanism_delta = comparisons["mechanism_select_32_endpoint_vs_m0"]
    mechanism_did = comparisons["mechanism_select_32_state_did"]
    endpoint_overfit_key = f"{endpoint_label}|train_overfit_hard8|normal"
    m0_overfit_key = "m0|train_overfit_hard8|normal"
    m0_accuracy = _metric(overfit_summary, m0_overfit_key, "accuracy")
    endpoint_accuracy = _metric(overfit_summary, endpoint_overfit_key, "accuracy")
    overfit_gate = (
        bool(training_summary["technical_gate"]["passed"])
        and overfit_delta["relative_change"] <= -0.20
        and overfit_delta["improved_pair_units"] >= 7
        and overfit_delta["ci95"][1] < 0.0
        and endpoint_accuracy - m0_accuracy >= 0.20
        and overfit_did["estimate"] < 0.0
    )
    fixed_dev_gate = (
        formal_delta["ci95"][1] < 0.0
        and mechanism_delta["ci95"][1] < 0.0
        and mechanism_did["ci95"][1] < 0.0
    )

    summary = {
        "schema": R6_SCHEMA,
        "status": "completed",
        "protocol": R6_PROTOCOL,
        "implementation_revision": R6_IMPLEMENTATION_REVISION,
        "git_commit": manifest["git_commit"],
        "arm": args.arm,
        "edit_start_sigma": ARM_SIGMA[args.arm],
        "edit_start_sigma_semantics": "effective post-scheduler-shift flow sigma",
        "source_anchor_effective_sigma_schedule": list(SOURCE_ANCHOR_EFFECTIVE_SIGMAS),
        "diagnostic_only_not_formal_success": True,
        "full_success_claim_allowed": False,
        "selected_segments": [segment.segment_id for segment in selected],
        "selected_segments_sha256": manifest["selected_segments_sha256"],
        "gradient_conflict_audit": gradient_audit,
        "training_summary": training_summary,
        "overfit_evaluation_summary": overfit_summary,
        "comparisons": comparisons,
        "gates": {
            "technical_gate": bool(training_summary["technical_gate"]["passed"]),
            "hard8_overfit_learnability_gate": overfit_gate,
            "fixed_dev_generalization_gate": fixed_dev_gate,
            "formal_success_gate": False,
            "formal_success_gate_reason": "single-seed repeated-subset diagnostic cannot establish fixed-full-data ID/OOD success",
        },
        "overfit_accuracy": {
            "m0": m0_accuracy,
            "endpoint": endpoint_accuracy,
            "delta": endpoint_accuracy - m0_accuracy,
        },
        "elapsed_seconds": training_summary["elapsed_seconds"],
        "artifacts": {
            "manifest": str((args.output_dir / "manifest.json").resolve()),
            "manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
            "metrics": str((args.output_dir / "metrics.jsonl").resolve()),
            "metrics_sha256": sha256_file(args.output_dir / "metrics.jsonl"),
            "micro_metrics": str((args.output_dir / "micro_metrics.jsonl").resolve()),
            "micro_metrics_sha256": sha256_file(args.output_dir / "micro_metrics.jsonl"),
            "pilot_rows": str((args.output_dir / "pilot_evaluation_rows.jsonl").resolve()),
            "pilot_rows_sha256": sha256_file(args.output_dir / "pilot_evaluation_rows.jsonl"),
            "overfit_rows": str(overfit_rows_path.resolve()),
            "overfit_rows_sha256": sha256_file(overfit_rows_path),
            "endpoint_ema": str(endpoint_path.resolve()),
            "endpoint_ema_sha256": sha256_file(endpoint_path),
            "endpoint_raw": str((args.output_dir / "endpoint_raw.pt").resolve()),
            "endpoint_raw_sha256": sha256_file(args.output_dir / "endpoint_raw.pt"),
        },
    }
    r5._write_json(args.output_dir / "r6_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    started = time.monotonic()
    summary = _run(args)
    summary["wall_clock_seconds"] = time.monotonic() - started
    r5._write_json(args.output_dir / "r6_summary.json", summary)
    print(
        json.dumps(
            {
                "milestone": "r6_completed",
                "arm": args.arm,
                "hard8_overfit_learnability_gate": summary["gates"]["hard8_overfit_learnability_gate"],
                "fixed_dev_generalization_gate": summary["gates"]["fixed_dev_generalization_gate"],
                "formal_success_gate": False,
                "wall_clock_seconds": summary["wall_clock_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
