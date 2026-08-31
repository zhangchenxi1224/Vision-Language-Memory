"""R10 DreamLite one-step SET lower-bound diagnostic for one fixed F1 target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r6_source_anchor as r6  # noqa: E402
from scripts.train import dreamlite_r7_gradient_balance as r8  # noqa: E402
from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_OPTIMIZER_STEPS,
    R10_PROTOCOL,
    R10_SELECTED_SEGMENTS_SHA256,
    R10_SELECTION_SEED,
    build_single_target_schedule,
    select_f1_targets,
    target_gate,
    target_statistics,
    target_training_view_counts,
)


IMPLEMENTATION_REVISION = "dreamlite-single-set-full-gradient-v1"
SUMMARY_SCHEMA = "vision_memory.r10-dreamlite-single-set-summary.v1"
MANIFEST_SCHEMA = "vision_memory.r10-dreamlite-single-set-manifest.v1"
AGGREGATION_SCHEMA = "vision_memory.r10-single-target-gradient-step.v1"
TECHNICAL_GATE_SCHEMA = "vision_memory.r10-dreamlite-technical-gate.v1"
SUITE = "r10_f1_single_target"
GRADIENT_COEFFICIENT = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index", type=int, choices=range(8), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-seed", type=int)
    parser.add_argument("--pairing-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--checkpoint-unet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.adapter_seed is None:
        args.adapter_seed = args.seed
    args.arm = "source-anchored"
    args.profile = "r10-diagnostic"
    args.protocol_revision = "r10"
    args.schedule_seed = R10_SELECTION_SEED
    args.persistent_state = "latent"
    args.tbptt_horizon = 4
    args.gradient_mode = "full"
    args.selected_step_count = 0
    args.gradient_accumulation = r5.GRADIENT_ACCUMULATION
    args.gradient_aggregation = "single-target-full"
    args.weight_decay = r5.WEIGHT_DECAY
    args.gradient_clip = r5.GRADIENT_CLIP
    args.lora_rank = r5.LORA_RANK
    args.ema_decay = r5.EMA_DECAY
    args.residual_blend = 1.0
    args.checkpoint_every = 32
    args.max_optimizer_steps = R10_OPTIMIZER_STEPS
    args.resume = None
    args.checkpoint = []
    args.audit_state_gradients = True
    args.gradient_audit_size = 24
    args.health_eval = False
    args.record_micro_metrics = True
    args.bootstrap_iterations = 10_000
    args.target_segment_id = None
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.resolution != 1024:
        raise ValueError("R10 preserves the fixed 1024x1024 visual/Reader contract.")
    for name in ("train", "dev"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R10 {name} path is not a file: {getattr(args, name)}")
    for name in ("dreamlite", "reader"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R10 {name} path is not a directory: {getattr(args, name)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("A fresh R10 target refuses a non-empty output directory.")
    if r8.git_value("status", "--porcelain") and not args.allow_dirty:
        raise ValueError("R10 refuses a dirty source tree unless --allow-dirty is explicit.")


def _load_data(args: argparse.Namespace) -> tuple[r5.R5DataBundle, tuple[r5.R5Segment, ...]]:
    base = r5._load_data(args, optimizer_steps=R10_OPTIMIZER_STEPS)
    selected = select_f1_targets(base.train_pools)
    schedule = build_single_target_schedule(
        selected,
        target_index=args.target_index,
        schedule_seed=args.schedule_seed,
    )
    target = selected[args.target_index]
    audit = {
        **r5.schedule_audit(schedule),
        "r10_protocol": R10_PROTOCOL,
        "selected_segment_ids": [segment.segment_id for segment in selected],
        "selected_segments_sha256": r5.canonical_sha256([segment.to_dict() for segment in selected]),
        "target_index": args.target_index,
        "target_segment_id": target.segment_id,
        "target_training_view_counts": target_training_view_counts(
            schedule,
            target_segment_id=target.segment_id,
        ),
        "unoptimized_decoys_per_step": 7,
    }
    return replace(base, schedule=schedule, schedule_audit=audit), selected


def _load_runtime(args: argparse.Namespace) -> r5.RuntimeBundle:
    return r8._load_runtime(args)


def _manifest(
    *,
    args: argparse.Namespace,
    data: r5.R5DataBundle,
    selected: Sequence[r5.R5Segment],
    target: r5.R5Segment,
    determinism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parent_args = argparse.Namespace(**vars(args))
    parent_args.arm = "source-anchored"
    payload = r6._manifest(
        args=parent_args,
        data=data,
        selected=selected,
        determinism=determinism,
    )
    payload.update(
        {
            "schema": MANIFEST_SCHEMA,
            "protocol": R10_PROTOCOL,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "arm": "dreamlite-single-set",
            "target_index": args.target_index,
            "target_segment": target.to_dict(),
            "target_segment_id": target.segment_id,
            "selected_segments_sha256": R10_SELECTED_SEGMENTS_SHA256,
            "gradient_aggregation": args.gradient_aggregation,
            "gradient_coefficient": GRADIENT_COEFFICIENT,
            "activation_condition": "technically valid R9 pass count is at most seven",
            "hypothesis": (
                "If a single F1 target remains unlearnable at full coefficient, the current DreamLite "
                "write parameterization is not a demonstrated lower bound for recurrent picture memory."
            ),
            "diagnostic_only_not_formal_success": True,
        }
    )
    payload["fixed_contract"] = {
        **payload["fixed_contract"],
        "selected_segments_sha256": R10_SELECTED_SEGMENTS_SHA256,
        "optimizer_steps": R10_OPTIMIZER_STEPS,
        "target_exposures": R10_OPTIMIZER_STEPS,
        "target_gradient_coefficient": GRADIENT_COEFFICIENT,
        "forward_cyclic_training_view_exposures": {str(index): 32 for index in range(4)},
        "heldout_reverse_cyclic_endpoint_views": 4,
        "checkpoint_steps": [0, 32, 64, 96, 128],
        "primary_endpoint": "EMA step128",
        "evaluation_controls": ["normal", "reset"],
    }
    payload["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in {"output_dir", "resume", "checkpoint"}
    }
    return payload


def _cosine(first: Tensor, second: Tensor) -> float:
    denominator = first.double().norm() * second.double().norm()
    if float(denominator) == 0.0:
        raise ValueError("R10 gradient cosine received a zero vector.")
    return float(torch.dot(first.double(), second.double()) / denominator)


def _run_optimizer_step(
    *,
    step_zero: int,
    units: Sequence[r5.ScheduledR5Segment],
    runtime: r5.RuntimeBundle,
    ema: r5.TrainableEMA,
    reader_fn: Any,
    args: argparse.Namespace,
    elapsed: float,
) -> dict[str, Any]:
    if len(units) != 8 or not args.target_segment_id:
        raise ValueError("R10 requires one target inside each fixed eight-item schedule block.")
    matches = [unit for unit in units if unit.segment.segment_id == args.target_segment_id]
    if len(matches) != 1:
        raise RuntimeError(f"R10 schedule contains {len(matches)} target copies at step {step_zero}.")
    unit = matches[0]
    learning_rate = r5.r5_learning_rate(step_zero)
    for group in runtime.optimizer.param_groups:
        group["lr"] = learning_rate

    runtime.optimizer.zero_grad(set_to_none=True)
    outcome = r5._micro_forward_backward(
        unit=unit,
        model=runtime.model,
        reader_fn=reader_fn,
        args=args,
        named_trainable=runtime.named_trainable,
        loss_divisor=1.0,
    )
    raw, activity = r8._gradient_vector(runtime.named_trainable)
    raw_norm = float(raw.double().norm())
    if not math.isfinite(raw_norm) or raw_norm <= 0.0:
        raise RuntimeError("R10 target gradient is non-finite or zero.")
    applied = raw * GRADIENT_COEFFICIENT
    applied_norm = float(applied.double().norm())
    expected_norm = raw_norm * GRADIENT_COEFFICIENT
    scale_error = abs(applied_norm - expected_norm) / expected_norm
    if not math.isfinite(applied_norm) or scale_error > 1e-6:
        raise RuntimeError(f"R10 target gradient scaling failed: {scale_error}")
    runtime.optimizer.zero_grad(set_to_none=True)
    r8._install_gradient_vector(runtime.named_trainable, applied, active_parameters=activity)
    r8.assert_frozen_contract(runtime.pipe, runtime.reader)

    snapshot, diagnostics = r8.begin_optimizer_diagnostics(runtime.named_trainable)
    gradient_norm = r5._clip_gradients(
        runtime.named_trainable,
        runtime.trainable,
        max_norm=args.gradient_clip,
    )
    r8.record_optimizer_diagnostics_after_clip(
        diagnostics,
        runtime.named_trainable,
        gradient_norm=gradient_norm,
        max_norm=args.gradient_clip,
    )
    runtime.optimizer.step()
    diagnostics = r8.finalize_optimizer_diagnostics_after_step(
        diagnostics,
        runtime.named_trainable,
        snapshot,
    )
    ema.update(runtime.named_trainable)
    runtime.optimizer.zero_grad(set_to_none=True)

    state_gradient = None
    if outcome.state_gradient is not None:
        state_gradient = float(bool(outcome.state_gradient["segment_boundary"]["nonzero"]))
    phase = int.from_bytes(hashlib.sha256(args.target_segment_id.encode()).digest()[:2], "big") % 4
    training_view = (unit.global_micro_index + phase) % 4
    return {
        "schema": r5.R5_METRICS_SCHEMA,
        "kind": "optimizer_step",
        "optimizer_step": step_zero + 1,
        "next_global_micro_index": (step_zero + 1) * 8,
        "learning_rate": learning_rate,
        "loss_mean": outcome.loss,
        "loss_by_family": {outcome.family: outcome.loss},
        "gradient_norm_before_clip": gradient_norm,
        "gradient_clip_threshold": args.gradient_clip,
        "gradient_clipped": gradient_norm > args.gradient_clip,
        "gradient_aggregation": {
            "schema": AGGREGATION_SCHEMA,
            "mode": args.gradient_aggregation,
            "target_segment_id": args.target_segment_id,
            "target_index": args.target_index,
            "raw_target_norm": raw_norm,
            "gradient_coefficient": GRADIENT_COEFFICIENT,
            "applied_norm_before_clip": applied_norm,
            "expected_applied_norm": expected_norm,
            "scale_relative_error": scale_error,
            "raw_vs_applied_cosine": _cosine(raw, applied),
            "active_parameter_tensors": sum(activity),
            "total_parameter_tensors": len(activity),
            "forward_cyclic_training_view": training_view,
        },
        "state_gradient_nonzero_fraction": state_gradient,
        "family_counts": {outcome.family: 1},
        "phase_counts": dict(Counter([outcome.phase])),
        "target_event_kind_counts": dict(Counter([outcome.target_event_kind])),
        "selected_step_set_counts": dict(Counter([str(outcome.selected_step_indices)])),
        "image_min": outcome.image_min,
        "image_max": outcome.image_max,
        "image_saturation_fraction_mean": outcome.image_saturation_fraction,
        "image_rms_mean": outcome.image_rms,
        "optimizer_diagnostics": diagnostics,
        "schedule_receipts": [outcome.receipt],
        "elapsed_seconds": elapsed,
        "updater_peak_memory_gib": r5._peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": r5._peak_gib(runtime.reader_device),
        "micro_records": [outcome.to_dict()] if args.record_micro_metrics else None,
    }


def _technical_gate(
    metrics: Sequence[Mapping[str, Any]],
    micro_metrics: Sequence[Mapping[str, Any]],
    *,
    target_segment_id: str,
    training_gate: Mapping[str, Any],
    checkpoint_steps: set[int],
) -> dict[str, Any]:
    records = [
        metric["gradient_aggregation"]
        for metric in metrics
        if metric.get("kind") == "optimizer_step"
        and isinstance(metric.get("gradient_aggregation"), Mapping)
    ]
    errors = [float(record["scale_relative_error"]) for record in records]
    cosines = [float(record["raw_vs_applied_cosine"]) for record in records]
    views = Counter(int(record["forward_cyclic_training_view"]) for record in records)
    expected_views = {0: 32, 1: 32, 2: 32, 3: 32}
    passed = bool(
        training_gate.get("passed")
        and len(records) == R10_OPTIMIZER_STEPS
        and len(micro_metrics) == R10_OPTIMIZER_STEPS
        and all(record.get("target_segment_id") == target_segment_id for record in records)
        and all(record.get("mode") == "single-target-full" for record in records)
        and all(math.isclose(float(record["gradient_coefficient"]), 1.0) for record in records)
        and all(record.get("segment_id") == target_segment_id for record in micro_metrics)
        and bool(errors)
        and max(errors) <= 1e-6
        and bool(cosines)
        and min(cosines) >= 1.0 - 1e-6
        and dict(sorted(views.items())) == expected_views
        and checkpoint_steps == {0, 32, 64, 96, 128}
    )
    return {
        "schema": TECHNICAL_GATE_SCHEMA,
        "passed": passed,
        "training_gate_passed": bool(training_gate.get("passed")),
        "optimizer_step_records": len(records),
        "micro_records": len(micro_metrics),
        "maximum_scale_relative_error": max(errors) if errors else None,
        "minimum_raw_vs_applied_cosine": min(cosines) if cosines else None,
        "training_view_counts": dict(sorted(views.items())),
        "expected_training_view_counts": expected_views,
        "checkpoint_steps_observed": sorted(checkpoint_steps),
    }


def _evaluation_rows(
    *,
    model: Any,
    reader_fn: Any,
    target: r5.R5Segment,
    checkpoint: str,
) -> list[dict[str, Any]]:
    return r5.evaluate_items(
        model=model,
        reader_fn=reader_fn,
        items=r5._segment_eval_items(model, (target,)),
        checkpoint_label=checkpoint,
        suite=SUITE,
        controls=("normal", "reset"),
        permutations=r5.REVERSE_CYCLIC4,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    determinism = r8.configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    r8.set_all_seeds(args.seed)
    data, selected = _load_data(args)
    selected_sha = r5.canonical_sha256([segment.to_dict() for segment in selected])
    if selected_sha != R10_SELECTED_SEGMENTS_SHA256:
        raise RuntimeError(f"R10 F1 selection drifted: {selected_sha}")
    target = selected[args.target_index]
    args.target_segment_id = target.segment_id
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
        target=target,
        determinism=determinism,
    )
    r5._write_json(args.output_dir / "manifest.json", manifest)

    eval_reader = r8.choice_reader_callable(
        reader=runtime.reader,
        processor=runtime.processor,
        reader_device=runtime.reader_device,
        require_grad=False,
        deterministic_ce=args.strict_determinism,
    )
    m0_rows = _evaluation_rows(
        model=runtime.model,
        reader_fn=eval_reader,
        target=target,
        checkpoint="m0",
    )
    rows_path = args.output_dir / "target_evaluation_rows.jsonl"
    r6._append_rows(rows_path, m0_rows)

    training_summary = r5.run_training_profile(
        args=args,
        optimizer_steps=R10_OPTIMIZER_STEPS,
        data=data,
        runtime=runtime,
        manifest=manifest,
        optimizer_step_fn=_run_optimizer_step,
    )
    metrics = r6._read_jsonl(args.output_dir / "metrics.jsonl")
    micro_metrics = r6._read_jsonl(args.output_dir / "micro_metrics.jsonl")
    checkpoint_steps = {
        int(path.stem.removeprefix("step-"))
        for path in (args.output_dir / "checkpoints").glob("step-*.pt")
    }
    technical = _technical_gate(
        metrics,
        micro_metrics,
        target_segment_id=target.segment_id,
        training_gate=training_summary["technical_gate"],
        checkpoint_steps=checkpoint_steps,
    )

    endpoint_path = args.output_dir / "endpoint_ema.pt"
    r8.load_trainable_weights(endpoint_path, trainable_module=runtime.model)
    endpoint_rows = _evaluation_rows(
        model=runtime.model,
        reader_fn=eval_reader,
        target=target,
        checkpoint="ema_step128",
    )
    r8.assert_frozen_contract(runtime.pipe, runtime.reader)
    r6._append_rows(rows_path, endpoint_rows)
    all_rows = m0_rows + endpoint_rows
    statistics = target_statistics(
        all_rows,
        suite=SUITE,
        target_segment_id=target.segment_id,
        endpoint="ema_step128",
    )
    scientific_gate = target_gate(statistics, technical_gate=bool(technical["passed"]))
    evaluation_summary = r5.summarize_evaluation_rows(
        all_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=20261000 + args.target_index,
    )
    r5._write_json(args.output_dir / "target_evaluation_summary.json", evaluation_summary)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "protocol": R10_PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": manifest["git_commit"],
        "arm": "dreamlite-single-set",
        "target_index": args.target_index,
        "target_segment_id": target.segment_id,
        "target_family": target.family,
        "selected_segments": [segment.segment_id for segment in selected],
        "selected_segments_sha256": selected_sha,
        "gradient_aggregation": args.gradient_aggregation,
        "gradient_coefficient": GRADIENT_COEFFICIENT,
        "training_summary": training_summary,
        "technical_gate": technical,
        "target_statistics": statistics,
        "gates": {
            "technical_gate": bool(technical["passed"]),
            "target_lower_bound_gate": scientific_gate,
            "formal_success_gate": False,
        },
        "diagnostic_only_not_formal_success": True,
        "full_success_claim_allowed": False,
        "checkpoint_steps_observed": sorted(checkpoint_steps),
        "artifacts": {
            "manifest_sha256": r8.sha256_file(args.output_dir / "manifest.json"),
            "metrics_sha256": r8.sha256_file(args.output_dir / "metrics.jsonl"),
            "micro_metrics_sha256": r8.sha256_file(args.output_dir / "micro_metrics.jsonl"),
            "evaluation_rows_sha256": r8.sha256_file(rows_path),
            "endpoint_ema_sha256": r8.sha256_file(endpoint_path),
            "endpoint_raw_sha256": r8.sha256_file(args.output_dir / "endpoint_raw.pt"),
        },
    }
    r5._write_json(args.output_dir / "r10_dreamlite_summary.json", summary)
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
    r5._write_json(args.output_dir / "r10_dreamlite_summary.json", summary)
    print(
        json.dumps(
            {
                "milestone": "r10_dreamlite_completed",
                "target_index": args.target_index,
                "target_segment_id": summary["target_segment_id"],
                "target_lower_bound_gate": summary["gates"]["target_lower_bound_gate"],
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
