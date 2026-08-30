"""Conditional R9 decomposition of hard8 learnability into eight independent targets."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r6_source_anchor as r6  # noqa: E402
from scripts.train import dreamlite_r7_gradient_balance as r8  # noqa: E402


PROTOCOL = "R9-IndividualLearnability-Decomposition"
IMPLEMENTATION_REVISION = "single-target-one-eighth-gradient-v1"
SUMMARY_SCHEMA = "vision_memory.r9-individual-learnability-summary.v1"
MANIFEST_SCHEMA = "vision_memory.r9-individual-learnability-manifest.v1"
AGGREGATION_SCHEMA = "vision_memory.r9-single-target-gradient-step.v1"
OPTIMIZER_STEPS = 128
TARGET_COEFFICIENT = 1.0 / 8.0
SELECTED_SEGMENTS_SHA256 = r8.R8_SELECTED_SEGMENTS_SHA256


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
    args.protocol_revision = "r9"
    args.arm = f"target-{args.target_index}"
    args.profile = "pilot"
    args.persistent_state = "latent"
    args.tbptt_horizon = 4
    args.gradient_mode = "full"
    args.selected_step_count = 0
    args.gradient_accumulation = r5.GRADIENT_ACCUMULATION
    args.gradient_aggregation = "single-target-one-eighth"
    args.weight_decay = r5.WEIGHT_DECAY
    args.gradient_clip = r5.GRADIENT_CLIP
    args.lora_rank = r5.LORA_RANK
    args.ema_decay = r5.EMA_DECAY
    args.residual_blend = 1.0
    args.checkpoint_every = 32
    args.max_optimizer_steps = OPTIMIZER_STEPS
    args.resume = None
    args.checkpoint = []
    args.audit_state_gradients = True
    args.gradient_audit_size = 24
    args.health_eval = False
    args.record_micro_metrics = True
    args.target_segment_id = None
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.resolution != 1024:
        raise ValueError("R9 preserves the fixed 1024x1024 visual/Reader contract.")
    for name in ("train", "dev"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R9 {name} path is not a file: {getattr(args, name)}")
    for name in ("dreamlite", "reader"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R9 {name} path is not a directory: {getattr(args, name)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("A fresh R9 target refuses a non-empty output directory.")
    status = r8.git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise ValueError("R9 refuses a dirty source tree unless --allow-dirty is explicit.")


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
    parent_args.protocol_revision = "r8"
    parent_args.arm = "raw-mean-control"
    parent_args.gradient_aggregation = "raw-mean"
    payload = r8._manifest(
        args=parent_args,
        data=data,
        selected=selected,
        determinism=determinism,
    )
    payload.update(
        {
            "schema": MANIFEST_SCHEMA,
            "protocol": PROTOCOL,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "arm": args.arm,
            "target_index": args.target_index,
            "target_segment": target.to_dict(),
            "target_segment_id": target.segment_id,
            "gradient_aggregation": args.gradient_aggregation,
            "gradient_coefficient": TARGET_COEFFICIENT,
            "activation_condition": (
                "both technically valid R8 arms fail hard8_overfit_learnability_gate"
            ),
            "hypothesis": (
                "If common-descent aggregation is insufficient, test whether each hard8 transition "
                "is learnable alone at its original raw-mean coefficient."
            ),
            "single_changed_variable": {
                "name": "simultaneous_training_segments",
                "r8": "eight independently computed segment gradients per optimizer step",
                "r9": "one fixed target gradient per optimizer step, multiplied by 1/8",
            },
        }
    )
    payload["fixed_contract"] = {
        **payload["fixed_contract"],
        "selected_segments_sha256": SELECTED_SEGMENTS_SHA256,
        "optimizer_steps": OPTIMIZER_STEPS,
        "target_exposures": OPTIMIZER_STEPS,
        "executed_micro_segments": OPTIMIZER_STEPS,
        "schedule_cursor_segments": OPTIMIZER_STEPS * 8,
        "gradient_aggregation": args.gradient_aggregation,
        "target_gradient_coefficient": TARGET_COEFFICIENT,
        "checkpoint_steps": [0, 32, 64, 96, 128],
        "primary_endpoint": "EMA step128",
    }
    if isinstance(payload.get("arguments"), dict):
        payload["arguments"].update(
            {
                "protocol_revision": "r9",
                "arm": args.arm,
                "target_index": args.target_index,
                "target_segment_id": target.segment_id,
                "gradient_aggregation": args.gradient_aggregation,
                "gradient_coefficient": TARGET_COEFFICIENT,
            }
        )
    return payload


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
        raise ValueError("R9 requires one target inside each fixed eight-segment schedule block.")
    matches = [unit for unit in units if unit.segment.segment_id == args.target_segment_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"R9 schedule block contains {len(matches)} copies of target {args.target_segment_id}."
        )
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
        raise RuntimeError("R9 target gradient is non-finite or zero.")
    applied = raw * TARGET_COEFFICIENT
    applied_norm = float(applied.double().norm())
    expected_norm = raw_norm * TARGET_COEFFICIENT
    scale_error = abs(applied_norm - expected_norm) / expected_norm
    if not math.isfinite(applied_norm) or scale_error > 1e-6:
        raise RuntimeError(f"R9 one-eighth target gradient scaling failed: {scale_error}")
    runtime.optimizer.zero_grad(set_to_none=True)
    r8._install_gradient_vector(runtime.named_trainable, applied, active_parameters=activity)
    r8.assert_frozen_contract(runtime.pipe, runtime.reader)

    diagnostic_snapshot, diagnostic_report = r8.begin_optimizer_diagnostics(runtime.named_trainable)
    gradient_norm = r5._clip_gradients(
        runtime.named_trainable,
        runtime.trainable,
        max_norm=args.gradient_clip,
    )
    r8.record_optimizer_diagnostics_after_clip(
        diagnostic_report,
        runtime.named_trainable,
        gradient_norm=gradient_norm,
        max_norm=args.gradient_clip,
    )
    runtime.optimizer.step()
    diagnostic_report = r8.finalize_optimizer_diagnostics_after_step(
        diagnostic_report,
        runtime.named_trainable,
        diagnostic_snapshot,
    )
    ema.update(runtime.named_trainable)
    runtime.optimizer.zero_grad(set_to_none=True)

    state_gradient = None
    if outcome.state_gradient is not None:
        state_gradient = float(bool(outcome.state_gradient["segment_boundary"]["nonzero"]))
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
            "gradient_coefficient": TARGET_COEFFICIENT,
            "applied_norm_before_clip": applied_norm,
            "expected_applied_norm": expected_norm,
            "scale_relative_error": scale_error,
            "raw_vs_applied_cosine": r8._cosine(raw, applied),
            "active_parameter_tensors": sum(activity),
            "total_parameter_tensors": len(activity),
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
        "optimizer_diagnostics": diagnostic_report,
        "schedule_receipts": [outcome.receipt],
        "elapsed_seconds": elapsed,
        "updater_peak_memory_gib": r5._peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": r5._peak_gib(runtime.reader_device),
        "micro_records": [outcome.to_dict()] if args.record_micro_metrics else None,
    }


def _aggregation_technical_gate(
    metrics: Sequence[Mapping[str, Any]],
    micro_metrics: Sequence[Mapping[str, Any]],
    *,
    target_segment_id: str,
) -> dict[str, Any]:
    records = [
        metric["gradient_aggregation"]
        for metric in metrics
        if metric.get("kind") == "optimizer_step"
        and isinstance(metric.get("gradient_aggregation"), Mapping)
    ]
    errors = [float(record["scale_relative_error"]) for record in records]
    cosines = [float(record["raw_vs_applied_cosine"]) for record in records]
    micro_target_matches = bool(micro_metrics) and all(
        record.get("segment_id") == target_segment_id for record in micro_metrics
    )
    checkpoints = {0, 32, 64, 96, 128}
    return {
        "schema": "vision_memory.r9-single-target-gradient-gate.v1",
        "optimizer_step_records": len(records),
        "micro_records": len(micro_metrics),
        "exact_optimizer_step_count": len(records) == OPTIMIZER_STEPS,
        "exact_micro_record_count": len(micro_metrics) == OPTIMIZER_STEPS,
        "target_matches": bool(records)
        and all(record.get("target_segment_id") == target_segment_id for record in records)
        and all(record.get("mode") == "single-target-one-eighth" for record in records)
        and micro_target_matches,
        "coefficient_matches": bool(records)
        and all(math.isclose(float(record["gradient_coefficient"]), TARGET_COEFFICIENT) for record in records),
        "maximum_scale_relative_error": max(errors) if errors else None,
        "minimum_raw_vs_applied_cosine": min(cosines) if cosines else None,
        "checkpoint_steps_expected": sorted(checkpoints),
        "passed": (
            len(records) == OPTIMIZER_STEPS
            and len(micro_metrics) == OPTIMIZER_STEPS
            and all(record.get("target_segment_id") == target_segment_id for record in records)
            and all(record.get("mode") == "single-target-one-eighth" for record in records)
            and micro_target_matches
            and all(math.isclose(float(record["gradient_coefficient"]), TARGET_COEFFICIENT) for record in records)
            and bool(errors)
            and max(errors) <= 1e-6
            and bool(cosines)
            and min(cosines) >= 1.0 - 1e-6
        ),
    }


def _cell(
    rows: Sequence[Mapping[str, Any]],
    *,
    checkpoint: str,
    condition: str,
    target_segment_id: str,
) -> list[Mapping[str, Any]]:
    values = [
        row
        for row in rows
        if row.get("suite") == "train_overfit_hard8"
        and row.get("checkpoint") == checkpoint
        and row.get("condition") == condition
        and row.get("pair_unit") == target_segment_id
    ]
    if len(values) != 4 or {int(row["view_index"]) for row in values} != {0, 1, 2, 3}:
        raise ValueError(
            f"R9 target cell is incomplete: {target_segment_id}:{checkpoint}:{condition}"
        )
    return values


def _target_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_segment_id: str,
) -> dict[str, Any]:
    cells = {
        (checkpoint, condition): _cell(
            rows,
            checkpoint=checkpoint,
            condition=condition,
            target_segment_id=target_segment_id,
        )
        for checkpoint in ("m0", "ema_step128")
        for condition in ("normal", "reset", "cross_episode_swap", "temporal_swap")
    }

    def mean(checkpoint: str, condition: str, field: str) -> float:
        return r5._mean(float(row[field]) for row in cells[(checkpoint, condition)])

    m0_normal = mean("m0", "normal", "ce")
    endpoint_normal = mean("ema_step128", "normal", "ce")
    per_view = {
        int(endpoint["view_index"]): float(endpoint["ce"])
        - float(next(row["ce"] for row in cells[("m0", "normal")] if row["view_index"] == endpoint["view_index"]))
        for endpoint in cells[("ema_step128", "normal")]
    }
    accuracy_m0 = mean("m0", "normal", "correct")
    accuracy_endpoint = mean("ema_step128", "normal", "correct")
    did = (
        endpoint_normal - mean("ema_step128", "reset", "ce")
    ) - (
        m0_normal - mean("m0", "reset", "ce")
    )
    return {
        "target_segment_id": target_segment_id,
        "m0_normal_mean_ce": m0_normal,
        "endpoint_normal_mean_ce": endpoint_normal,
        "delta_ce": endpoint_normal - m0_normal,
        "relative_change": endpoint_normal / m0_normal - 1.0,
        "per_view_delta_ce": dict(sorted(per_view.items())),
        "improved_choice_views": sum(value < 0.0 for value in per_view.values()),
        "m0_normal_accuracy": accuracy_m0,
        "endpoint_normal_accuracy": accuracy_endpoint,
        "accuracy_delta": accuracy_endpoint - accuracy_m0,
        "normal_reset_difference_in_differences": did,
        "condition_mean_ce": {
            checkpoint: {
                condition: mean(checkpoint, condition, "ce")
                for condition in ("normal", "reset", "cross_episode_swap", "temporal_swap")
            }
            for checkpoint in ("m0", "ema_step128")
        },
        "bootstrap_ci_used": False,
        "bootstrap_ci_reason": (
            "one independent target pair unit; deterministic choice views are not independent units"
        ),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    determinism = r8.configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    r8.set_all_seeds(args.seed)
    data, selected = r6._load_data(args)
    selected_sha = r5.canonical_sha256([segment.to_dict() for segment in selected])
    if selected_sha != SELECTED_SEGMENTS_SHA256 or len(selected) != 8:
        raise RuntimeError(f"R9 hard8 selection drifted: {selected_sha}")
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
    print(
        json.dumps(
            {"milestone": "r9_m0_hard8_eval", "target_index": args.target_index},
            sort_keys=True,
        ),
        flush=True,
    )
    m0_rows = r6._evaluation_rows(
        model=runtime.model,
        reader_fn=eval_reader,
        segments=selected,
        checkpoint="m0",
    )
    overfit_rows_path = args.output_dir / "overfit_evaluation_rows.jsonl"
    r6._append_rows(overfit_rows_path, m0_rows)

    training_summary = r5.run_training_profile(
        args=args,
        optimizer_steps=OPTIMIZER_STEPS,
        data=data,
        runtime=runtime,
        manifest=manifest,
        optimizer_step_fn=_run_optimizer_step,
    )
    metrics = r6._read_jsonl(args.output_dir / "metrics.jsonl")
    micro_metrics = r6._read_jsonl(args.output_dir / "micro_metrics.jsonl")
    aggregation_gate = _aggregation_technical_gate(
        metrics,
        micro_metrics,
        target_segment_id=target.segment_id,
    )
    checkpoint_steps = {
        int(path.stem.removeprefix("step-"))
        for path in (args.output_dir / "checkpoints").glob("step-*.pt")
    }
    checkpoint_gate = checkpoint_steps == {0, 32, 64, 96, 128}
    technical_gate = (
        bool(training_summary["technical_gate"]["passed"])
        and bool(aggregation_gate["passed"])
        and checkpoint_gate
    )

    endpoint_path = args.output_dir / "endpoint_ema.pt"
    r8.load_trainable_weights(endpoint_path, trainable_module=runtime.model)
    print(
        json.dumps(
            {"milestone": "r9_endpoint_hard8_eval", "target_index": args.target_index},
            sort_keys=True,
        ),
        flush=True,
    )
    endpoint_rows = r6._evaluation_rows(
        model=runtime.model,
        reader_fn=eval_reader,
        segments=selected,
        checkpoint="ema_step128",
    )
    r8.assert_frozen_contract(runtime.pipe, runtime.reader)
    r6._append_rows(overfit_rows_path, endpoint_rows)
    overfit_rows = m0_rows + endpoint_rows
    overfit_summary = r5.summarize_evaluation_rows(
        overfit_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=20260890 + args.target_index,
    )
    r5._write_json(args.output_dir / "overfit_evaluation_summary.json", overfit_summary)
    target_statistics = _target_statistics(overfit_rows, target_segment_id=target.segment_id)
    target_gate = (
        technical_gate
        and target_statistics["relative_change"] <= -0.20
        and target_statistics["improved_choice_views"] == 4
        and target_statistics["accuracy_delta"] >= 0.25
        and target_statistics["normal_reset_difference_in_differences"] < 0.0
    )

    pilot_rows = r6._read_jsonl(args.output_dir / "pilot_evaluation_rows.jsonl")
    all_rows = overfit_rows + pilot_rows
    descriptive_comparisons = {
        "all_hard8_endpoint_vs_m0": r6._checkpoint_comparison(
            all_rows,
            suite="train_overfit_hard8",
            endpoint="ema_step128",
            iterations=args.bootstrap_iterations,
            seed=20260900 + args.target_index,
        ),
        "all_hard8_state_did": r6._difference_in_differences(
            all_rows,
            suite="train_overfit_hard8",
            endpoint="ema_step128",
            iterations=args.bootstrap_iterations,
            seed=20260910 + args.target_index,
        ),
        "formal_select_32_endpoint_vs_m0": r6._checkpoint_comparison(
            all_rows,
            suite="formal_select_32",
            endpoint="ema_step128",
            iterations=args.bootstrap_iterations,
            seed=20260920 + args.target_index,
        ),
        "mechanism_select_32_endpoint_vs_m0": r6._checkpoint_comparison(
            all_rows,
            suite="mechanism_select_32",
            endpoint="ema_step128",
            iterations=args.bootstrap_iterations,
            seed=20260930 + args.target_index,
        ),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "protocol": PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": manifest["git_commit"],
        "arm": args.arm,
        "target_index": args.target_index,
        "target_segment_id": target.segment_id,
        "target_family": target.family,
        "selected_segments": [segment.segment_id for segment in selected],
        "selected_segments_sha256": selected_sha,
        "gradient_aggregation": args.gradient_aggregation,
        "gradient_coefficient": TARGET_COEFFICIENT,
        "executed_micro_segments": len(micro_metrics),
        "schedule_cursor_segments": OPTIMIZER_STEPS * 8,
        "source_anchor_effective_sigma_schedule": list(r6.SOURCE_ANCHOR_EFFECTIVE_SIGMAS),
        "diagnostic_only_not_formal_success": True,
        "full_success_claim_allowed": False,
        "training_summary": training_summary,
        "aggregation_technical_gate": aggregation_gate,
        "checkpoint_steps_observed": sorted(checkpoint_steps),
        "overfit_evaluation_summary": overfit_summary,
        "target_statistics": target_statistics,
        "descriptive_comparisons": descriptive_comparisons,
        "gates": {
            "technical_gate": technical_gate,
            "target_individual_learnability_gate": target_gate,
            "formal_success_gate": False,
            "formal_success_gate_reason": (
                "single-target single-seed repeated-subset diagnostic cannot establish full-data ID/OOD success"
            ),
        },
        "artifacts": {
            "manifest_sha256": r8.sha256_file(args.output_dir / "manifest.json"),
            "metrics_sha256": r8.sha256_file(args.output_dir / "metrics.jsonl"),
            "micro_metrics_sha256": r8.sha256_file(args.output_dir / "micro_metrics.jsonl"),
            "pilot_rows_sha256": r8.sha256_file(args.output_dir / "pilot_evaluation_rows.jsonl"),
            "overfit_rows_sha256": r8.sha256_file(overfit_rows_path),
            "endpoint_ema_sha256": r8.sha256_file(endpoint_path),
            "endpoint_raw_sha256": r8.sha256_file(args.output_dir / "endpoint_raw.pt"),
        },
    }
    r5._write_json(args.output_dir / "r9_summary.json", summary)
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
    r5._write_json(args.output_dir / "r9_summary.json", summary)
    print(
        json.dumps(
            {
                "milestone": "r9_completed",
                "target_index": args.target_index,
                "target_segment_id": summary["target_segment_id"],
                "target_individual_learnability_gate": summary["gates"][
                    "target_individual_learnability_gate"
                ],
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
