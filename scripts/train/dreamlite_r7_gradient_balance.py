"""R7/R8 paired hard8 diagnostics for explicit micro-gradient aggregation.

Both arms use the corrected R6 source-anchored DreamLite update.  The only
changed variable is how eight independently computed segment gradients are
aggregated before the unchanged global clip and AdamW step.  R7 tests equal
directional votes; R8 tests an exact deterministic projection of the raw mean
onto the current micro-gradients' common first-order descent cone.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r6_source_anchor as r6  # noqa: E402
from scripts.train.dreamlite_episode import (  # noqa: E402
    assert_frozen_contract,
    begin_optimizer_diagnostics,
    choice_reader_callable,
    finalize_optimizer_diagnostics_after_step,
    git_value,
    record_optimizer_diagnostics_after_clip,
    set_all_seeds,
    sha256_file,
)
from vision_memory.repro import configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import load_trainable_weights  # noqa: E402


R7_PROTOCOL = "R7-GradientBalance-Bottleneck"
R7_IMPLEMENTATION_REVISION = "explicit-micro-gradient-unit-balance-v1"
R7_SCHEMA = "vision_memory.r7-gradient-balance-summary.v1"
R7_MANIFEST_SCHEMA = "vision_memory.r7-gradient-balance-manifest.v1"
R7_AGGREGATION_SCHEMA = "vision_memory.r7-gradient-aggregation-step.v1"
R7_OPTIMIZER_STEPS = 128
R7_SELECTED_SEGMENTS_SHA256 = "eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6"
R8_PROTOCOL = "R8-CommonDescent-Bottleneck"
R8_IMPLEMENTATION_REVISION = "exact-active-set-common-descent-projection-v1"
R8_SCHEMA = "vision_memory.r8-common-descent-summary.v1"
R8_MANIFEST_SCHEMA = "vision_memory.r8-common-descent-manifest.v1"
R8_AGGREGATION_SCHEMA = "vision_memory.r8-common-descent-step.v1"
R8_OPTIMIZER_STEPS = 128
R8_SELECTED_SEGMENTS_SHA256 = R7_SELECTED_SEGMENTS_SHA256

R7_AGGREGATION_MODE = {
    "raw-mean-control": "raw-mean",
    "unit-balanced-norm-matched": "unit-balanced-norm-matched",
}
R8_AGGREGATION_MODE = {
    "raw-mean-control": "raw-mean",
    "common-descent-projected-norm-matched": "common-descent-projected-norm-matched",
}
AGGREGATION_MODE = {**R7_AGGREGATION_MODE, **R8_AGGREGATION_MODE}
PROTOCOL_CONTRACT = {
    "r7": {
        "protocol": R7_PROTOCOL,
        "implementation_revision": R7_IMPLEMENTATION_REVISION,
        "summary_schema": R7_SCHEMA,
        "manifest_schema": R7_MANIFEST_SCHEMA,
        "aggregation_schema": R7_AGGREGATION_SCHEMA,
        "optimizer_steps": R7_OPTIMIZER_STEPS,
        "selected_segments_sha256": R7_SELECTED_SEGMENTS_SHA256,
        "summary_filename": "r7_summary.json",
        "allowed_arms": R7_AGGREGATION_MODE,
    },
    "r8": {
        "protocol": R8_PROTOCOL,
        "implementation_revision": R8_IMPLEMENTATION_REVISION,
        "summary_schema": R8_SCHEMA,
        "manifest_schema": R8_MANIFEST_SCHEMA,
        "aggregation_schema": R8_AGGREGATION_SCHEMA,
        "optimizer_steps": R8_OPTIMIZER_STEPS,
        "selected_segments_sha256": R8_SELECTED_SEGMENTS_SHA256,
        "summary_filename": "r8_summary.json",
        "allowed_arms": R8_AGGREGATION_MODE,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=R7_PROTOCOL)
    parser.add_argument("--protocol-revision", choices=tuple(PROTOCOL_CONTRACT), default="r7")
    parser.add_argument("--arm", choices=tuple(AGGREGATION_MODE), required=True)
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
    contract = PROTOCOL_CONTRACT[args.protocol_revision]
    allowed_arms = contract["allowed_arms"]
    if args.arm not in allowed_arms:
        raise ValueError(f"{args.protocol_revision.upper()} does not define arm {args.arm}.")
    if args.adapter_seed is None:
        args.adapter_seed = args.seed
    args.profile = "pilot"
    args.persistent_state = "latent"
    args.tbptt_horizon = 4
    args.gradient_mode = "full"
    args.selected_step_count = 0
    args.gradient_accumulation = r5.GRADIENT_ACCUMULATION
    args.gradient_aggregation = allowed_arms[args.arm]
    args.weight_decay = r5.WEIGHT_DECAY
    args.gradient_clip = r5.GRADIENT_CLIP
    args.lora_rank = r5.LORA_RANK
    args.ema_decay = r5.EMA_DECAY
    args.residual_blend = 1.0
    args.checkpoint_every = 32
    args.max_optimizer_steps = int(contract["optimizer_steps"])
    args.resume = None
    args.checkpoint = []
    args.audit_state_gradients = True
    args.gradient_audit_size = 24
    args.health_eval = False
    args.record_micro_metrics = True
    return args


def _validate_args(args: argparse.Namespace) -> None:
    contract = PROTOCOL_CONTRACT[args.protocol_revision]
    protocol = str(contract["protocol"])
    if args.resolution != 1024:
        raise ValueError(f"{protocol} preserves the fixed 1024x1024 visual/Reader contract.")
    allowed_arms = contract["allowed_arms"]
    if args.arm not in allowed_arms or args.gradient_aggregation != allowed_arms[args.arm]:
        raise ValueError(f"{protocol} aggregation arm and implementation mode drifted.")
    for name in ("train", "dev"):
        if not getattr(args, name).is_file():
            raise ValueError(f"{protocol} {name} path is not a file: {getattr(args, name)}")
    for name in ("dreamlite", "reader"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"{protocol} {name} path is not a directory: {getattr(args, name)}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"A fresh {protocol} arm refuses a non-empty output directory.")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise ValueError(f"{protocol} refuses a dirty source tree unless --allow-dirty is explicit.")


def _source_anchored_args(args: argparse.Namespace) -> argparse.Namespace:
    payload = argparse.Namespace(**vars(args))
    payload.arm = "source-anchored"
    return payload


def _load_runtime(args: argparse.Namespace) -> r5.RuntimeBundle:
    return r6._load_runtime(_source_anchored_args(args))


def _manifest(
    *,
    args: argparse.Namespace,
    data: r5.R5DataBundle,
    selected: Sequence[r5.R5Segment],
    determinism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    contract = PROTOCOL_CONTRACT[args.protocol_revision]
    payload = r6._manifest(
        args=_source_anchored_args(args),
        data=data,
        selected=selected,
        determinism=determinism,
    )
    payload.update(
        {
            "schema": contract["manifest_schema"],
            "protocol": contract["protocol"],
            "implementation_revision": contract["implementation_revision"],
            "arm": args.arm,
            "gradient_aggregation": args.gradient_aggregation,
            "edit_start_sigma": r6.ARM_SIGMA["source-anchored"],
        }
    )
    if args.protocol_revision == "r7":
        payload.update(
            {
                "hypothesis": (
                    "After source anchoring repairs gradient scale, raw micro-segment aggregation "
                    "still blocks hard8 learning through norm dominance and cancellation."
                ),
                "parent_r6": {
                    "git_commit": "e1ab129ae86a39814a9ce0ce17ac06965f2e835c",
                    "decision": "reject_sigma_as_sufficient_test_gradient_balancing",
                    "selected_segments_sha256": R7_SELECTED_SEGMENTS_SHA256,
                },
                "single_changed_variable": {
                    "name": "micro_segment_gradient_aggregation",
                    "raw_mean": "g_raw=(1/8)*sum_i(g_i)",
                    "unit_balanced": "u=(1/8)*sum_i(g_i/||g_i||)",
                    "norm_match": "g_applied=u*||g_raw||/||u|| before unchanged global clip",
                },
            }
        )
    else:
        payload.update(
            {
                "hypothesis": (
                    "R7 equal weighting failed because it did not prevent the aggregate from being "
                    "a first-order ascent direction for individual micro-segments."
                ),
                "parent_r7": {
                    "git_commit": "c720f6b28e3ce6ef4e8f838a576d3b042d35cd58",
                    "decision": "reject_unit_balance_as_sufficient_test_conflict_projection",
                    "raw_summary_sha256": "1c78f9e4d10c6d2e0ffee5b9764120306f13687f13aa1bffecf2779c66e14096",
                    "balanced_summary_sha256": "e80c2f88dfe127693f4328294d92877480ce6abc14d6515809df420b98220185",
                    "selected_segments_sha256": R8_SELECTED_SEGMENTS_SHA256,
                },
                "single_changed_variable": {
                    "name": "micro_segment_gradient_aggregation",
                    "raw_mean": "g_raw=(1/8)*sum_i(g_i)",
                    "common_descent_projection": (
                        "d=argmin_v 0.5*||v-g_raw||^2 subject to dot(g_i,v)>=0 for all eight micros"
                    ),
                    "exact_solver": "deterministic exhaustive active-set KKT solve over at most 2^8 subsets",
                    "norm_match": "g_applied=d*||g_raw||/||d|| before unchanged global clip",
                    "optimizer_geometry_caveat": (
                        "the constraint is enforced in pre-AdamW gradient geometry, not on the "
                        "preconditioned finite optimizer update"
                    ),
                },
            }
        )
    payload["fixed_contract"] = {
        **payload["fixed_contract"],
        "source_anchor_effective_sigma": 0.5,
        "source_anchor_effective_sigma_schedule": list(r6.SOURCE_ANCHOR_EFFECTIVE_SIGMAS),
        "micro_gradients_computed_independently": True,
        "gradient_aggregation": args.gradient_aggregation,
        "unit_balanced_norm_match_relative_tolerance": 1e-5,
    }
    if args.protocol_revision == "r8":
        payload["fixed_contract"].update(
            {
                "common_descent_primal_cosine_tolerance": 1e-5,
                "common_descent_norm_match_relative_tolerance": 1e-5,
                "active_set_order": "ascending integer bitmask with smallest-mask objective tie break",
            }
        )
    payload["arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in {"output_dir", "resume", "checkpoint"}
    }
    return payload


def _gradient_vector(
    named_trainable: Sequence[tuple[str, nn.Parameter]],
) -> tuple[Tensor, tuple[bool, ...]]:
    chunks: list[Tensor] = []
    active: list[bool] = []
    device: torch.device | None = None
    for name, parameter in named_trainable:
        if parameter.dtype is not torch.float32:
            raise RuntimeError(f"R7 trainable parameter changed dtype: {name} -> {parameter.dtype}")
        if device is None:
            device = parameter.device
        elif parameter.device != device:
            raise RuntimeError("R7 explicit aggregation requires all trainable parameters on one device.")
        gradient = parameter.grad
        active.append(gradient is not None)
        chunks.append(
            torch.zeros(parameter.numel(), device=parameter.device, dtype=torch.float32)
            if gradient is None
            else gradient.detach().float().reshape(-1).clone()
        )
    if not chunks:
        raise RuntimeError("R7 found no trainable parameters.")
    vector = torch.cat(chunks)
    norm = vector.norm()
    if not bool(torch.isfinite(vector).all()) or not math.isfinite(float(norm)) or float(norm) <= 0.0:
        raise RuntimeError("R7 produced a non-finite or zero micro-gradient.")
    return vector, tuple(active)


def _install_gradient_vector(
    named_trainable: Sequence[tuple[str, nn.Parameter]],
    vector: Tensor,
    *,
    active_parameters: Sequence[bool],
) -> None:
    if not bool(torch.isfinite(vector).all()):
        raise RuntimeError("R7 refuses to install a non-finite aggregate gradient.")
    expected = sum(parameter.numel() for _name, parameter in named_trainable)
    if vector.numel() != expected:
        raise ValueError(f"R7 aggregate size mismatch: {vector.numel()} != {expected}")
    if len(active_parameters) != len(named_trainable):
        raise ValueError("R7 active-parameter mask topology mismatch.")
    offset = 0
    for (_name, parameter), active in zip(named_trainable, active_parameters, strict=True):
        count = parameter.numel()
        parameter.grad = vector[offset : offset + count].reshape_as(parameter).clone() if active else None
        offset += count


def _project_raw_mean_to_common_descent(
    vectors: Sequence[Tensor],
    *,
    raw_mean: Tensor,
    norms: Sequence[float],
) -> tuple[Tensor, dict[str, Any]]:
    """Project ``raw_mean`` onto ``dot(g_i, d) >= 0`` by exact active-set search.

    Positive rescaling of a constraint leaves its half-space unchanged, so the
    KKT system uses unit micro-gradients for conditioning.  With exactly eight
    micros, enumerating all 256 active sets is cheap and removes the randomized
    task-order ambiguity of sequential PCGrad.
    """

    count = len(vectors)
    if count != r5.GRADIENT_ACCUMULATION:
        raise ValueError("R8 common-descent projection requires exactly eight micro-gradients.")
    raw_norm = float(raw_mean.norm())
    if not math.isfinite(raw_norm) or raw_norm <= 0.0:
        raise RuntimeError("R8 common-descent projection received an invalid raw mean.")
    unit_stack = torch.stack(
        [vector / norm for vector, norm in zip(vectors, norms, strict=True)],
        dim=0,
    )
    gram = (unit_stack @ unit_stack.T).detach().double().cpu()
    raw_constraint = (unit_stack @ raw_mean).detach().double().cpu()
    if not bool(torch.isfinite(gram).all()) or not bool(torch.isfinite(raw_constraint).all()):
        raise RuntimeError("R8 common-descent KKT inputs are non-finite.")

    primal_tolerance = 2e-6 * max(raw_norm, 1.0)
    dual_tolerance = 2e-8 * max(raw_norm, 1.0)
    solve_tolerance = 2e-7 * max(raw_norm, 1.0)
    best: tuple[float, int, Tensor, Tensor] | None = None
    for mask in range(1 << count):
        indices = [index for index in range(count) if mask & (1 << index)]
        coefficients = torch.zeros(count, dtype=torch.float64)
        if indices:
            submatrix = gram[indices][:, indices]
            right_hand_side = -raw_constraint[indices]
            try:
                solution = torch.linalg.lstsq(submatrix, right_hand_side, rcond=1e-12).solution
            except RuntimeError:
                continue
            if not bool(torch.isfinite(solution).all()):
                continue
            residual = submatrix @ solution - right_hand_side
            if float(residual.abs().max()) > solve_tolerance:
                continue
            if float(solution.min()) < -dual_tolerance:
                continue
            solution = solution.clamp_min(0.0)
            coefficients[indices] = solution
        constraints = raw_constraint + gram @ coefficients
        if float(constraints.min()) < -primal_tolerance:
            continue
        objective = float(coefficients @ gram @ coefficients)
        if not math.isfinite(objective) or objective < -1e-10:
            continue
        objective = max(objective, 0.0)
        if best is None:
            best = (objective, mask, coefficients, constraints)
            continue
        tie_tolerance = 1e-12 * max(1.0, best[0], objective)
        if objective < best[0] - tie_tolerance or (
            abs(objective - best[0]) <= tie_tolerance and mask < best[1]
        ):
            best = (objective, mask, coefficients, constraints)
    if best is None:
        raise RuntimeError("R8 found no numerically feasible common-descent projection active set.")

    objective, mask, coefficients_cpu, _constraints_cpu = best
    coefficients = coefficients_cpu.to(device=raw_mean.device, dtype=raw_mean.dtype)
    projected = raw_mean + torch.sum(unit_stack * coefficients[:, None], dim=0)
    projected_norm = float(projected.norm())
    if not math.isfinite(projected_norm) or projected_norm <= 1e-8 * raw_norm:
        raise RuntimeError("R8 common-descent projection collapsed to a zero direction.")
    raw_micro_cosines = [
        float(torch.dot(vector, raw_mean) / (norm * raw_norm))
        for vector, norm in zip(vectors, norms, strict=True)
    ]
    projected_micro_cosines = [
        float(torch.dot(vector, projected) / (norm * projected_norm))
        for vector, norm in zip(vectors, norms, strict=True)
    ]
    minimum_projected_cosine = min(projected_micro_cosines)
    if minimum_projected_cosine < -1e-5:
        raise RuntimeError(
            "R8 common-descent projection violated a micro constraint: "
            f"minimum cosine={minimum_projected_cosine}"
        )
    active = [index for index, value in enumerate(coefficients_cpu.tolist()) if value > dual_tolerance]
    return projected, {
        "solver": "exhaustive-active-set-kkt",
        "candidate_active_sets": 1 << count,
        "selected_active_set_mask": mask,
        "selected_active_constraints": active,
        "active_constraint_count": len(active),
        "projection_distance_squared": objective,
        "raw_micro_cosines": raw_micro_cosines,
        "projected_micro_cosines": projected_micro_cosines,
        "raw_violating_micro_count": sum(value < 0.0 for value in raw_micro_cosines),
        "projected_violating_micro_count_at_tolerance": sum(
            value < -1e-5 for value in projected_micro_cosines
        ),
        "minimum_raw_micro_cosine": min(raw_micro_cosines),
        "minimum_projected_micro_cosine": minimum_projected_cosine,
        "primal_tolerance_dot_units": primal_tolerance,
        "dual_tolerance": dual_tolerance,
        "projected_norm_before_match": projected_norm,
    }


def _aggregate_micro_gradients(
    vectors: Sequence[Tensor],
    *,
    mode: str,
    protocol_revision: str = "r7",
) -> tuple[Tensor, dict[str, Any]]:
    if len(vectors) != r5.GRADIENT_ACCUMULATION:
        raise ValueError("R7 aggregation requires exactly eight micro-gradients.")
    if mode not in set(AGGREGATION_MODE.values()):
        raise ValueError(f"Unknown R7 aggregation mode: {mode}")
    reference = vectors[0]
    if any(vector.shape != reference.shape or vector.device != reference.device for vector in vectors):
        raise ValueError("R7 micro-gradient vector topology drifted within an optimizer step.")
    norms = [float(vector.norm()) for vector in vectors]
    if any(not math.isfinite(norm) or norm <= 0.0 for norm in norms):
        raise RuntimeError(f"R7 invalid micro-gradient norms: {norms}")

    raw_mean = torch.zeros_like(reference)
    unit_mean = torch.zeros_like(reference)
    weight = 1.0 / len(vectors)
    for vector, norm in zip(vectors, norms, strict=True):
        raw_mean.add_(vector, alpha=weight)
        unit_mean.add_(vector, alpha=weight / norm)
    raw_norm = float(raw_mean.norm())
    unit_norm = float(unit_mean.norm())
    if not math.isfinite(raw_norm) or raw_norm <= 0.0:
        raise RuntimeError("R7 raw-mean aggregate is non-finite or zero.")
    if not math.isfinite(unit_norm) or unit_norm <= 0.0:
        raise RuntimeError("R7 unit-balanced aggregate is non-finite or zero.")

    unit_matched = unit_mean * (raw_norm / unit_norm)
    projection_report: dict[str, Any] | None = None
    if mode == "common-descent-projected-norm-matched":
        projected, projection_report = _project_raw_mean_to_common_descent(
            vectors,
            raw_mean=raw_mean,
            norms=norms,
        )
        projected_norm = float(projected.norm())
        applied = projected * (raw_norm / projected_norm)
    elif mode == "unit-balanced-norm-matched":
        applied = unit_matched
    else:
        applied = raw_mean
    applied_norm = float(applied.norm())
    relative_norm_error = abs(applied_norm / raw_norm - 1.0)
    if relative_norm_error > 1e-5:
        raise RuntimeError(f"R7 aggregate norm match failed: {relative_norm_error}")
    if not bool(torch.isfinite(applied).all()):
        raise RuntimeError("R7 applied aggregate is non-finite.")

    pairwise = [
        float(torch.dot(vectors[first], vectors[second]) / (norms[first] * norms[second]))
        for first in range(len(vectors))
        for second in range(first + 1, len(vectors))
    ]
    norm_array = np.asarray(norms, dtype=np.float64)
    raw_vs_unit = float(torch.dot(raw_mean, unit_matched) / (raw_norm * float(unit_matched.norm())))
    raw_vs_applied = float(torch.dot(raw_mean, applied) / (raw_norm * applied_norm))
    report = {
        "schema": (
            R8_AGGREGATION_SCHEMA
            if protocol_revision == "r8"
            else R7_AGGREGATION_SCHEMA
        ),
        "mode": mode,
        "micro_count": len(vectors),
        "micro_gradient_norms": norms,
        "micro_gradient_norm": {
            "minimum": float(norm_array.min()),
            "median": float(np.median(norm_array)),
            "maximum": float(norm_array.max()),
            "max_to_min_ratio": float(norm_array.max() / norm_array.min()),
        },
        "pairwise_cosine": {
            "minimum": float(np.min(pairwise)),
            "median": float(np.median(pairwise)),
            "maximum": float(np.max(pairwise)),
            "negative_fraction": float(np.mean(np.asarray(pairwise) < 0.0)),
        },
        "raw_mean_norm": raw_norm,
        "unit_mean_norm_before_match": unit_norm,
        "applied_norm_before_clip": applied_norm,
        "norm_match_relative_error": relative_norm_error,
        "raw_vs_unit_balanced_cosine": raw_vs_unit,
        "raw_vs_applied_cosine": raw_vs_applied,
        "intervention_active": mode != "raw-mean" and raw_vs_applied < 1.0 - 1e-6,
    }
    if projection_report is not None:
        report["common_descent_projection"] = projection_report
    return applied, report


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
    if len(units) != args.gradient_accumulation:
        raise ValueError("R7 optimizer step requires exactly eight micro-segments.")
    learning_rate = r5.r5_learning_rate(step_zero)
    for group in runtime.optimizer.param_groups:
        group["lr"] = learning_rate

    outcomes: list[r5.MicroOutcome] = []
    vectors: list[Tensor] = []
    activity_masks: list[tuple[bool, ...]] = []
    runtime.optimizer.zero_grad(set_to_none=True)
    for unit in units:
        runtime.optimizer.zero_grad(set_to_none=True)
        outcomes.append(
            r5._micro_forward_backward(
                unit=unit,
                model=runtime.model,
                reader_fn=reader_fn,
                args=args,
                named_trainable=runtime.named_trainable,
                loss_divisor=1.0,
            )
        )
        vector, activity = _gradient_vector(runtime.named_trainable)
        vectors.append(vector)
        activity_masks.append(activity)
    runtime.optimizer.zero_grad(set_to_none=True)
    applied, aggregation = _aggregate_micro_gradients(
        vectors,
        mode=args.gradient_aggregation,
        protocol_revision=args.protocol_revision,
    )
    active_parameters = tuple(any(mask[index] for mask in activity_masks) for index in range(len(activity_masks[0])))
    _install_gradient_vector(
        runtime.named_trainable,
        applied,
        active_parameters=active_parameters,
    )
    aggregation["active_parameter_tensors"] = sum(active_parameters)
    aggregation["total_parameter_tensors"] = len(active_parameters)
    assert_frozen_contract(runtime.pipe, runtime.reader)

    diagnostic_snapshot, diagnostic_report = begin_optimizer_diagnostics(runtime.named_trainable)
    gradient_norm = r5._clip_gradients(
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
        item.state_gradient["segment_boundary"]["nonzero"] for item in outcomes if item.state_gradient is not None
    ]
    return {
        "schema": r5.R5_METRICS_SCHEMA,
        "kind": "optimizer_step",
        "optimizer_step": step_zero + 1,
        "next_global_micro_index": (step_zero + 1) * args.gradient_accumulation,
        "learning_rate": learning_rate,
        "loss_mean": r5._mean(item.loss for item in outcomes),
        "loss_by_family": {
            family: r5._mean(item.loss for item in outcomes if item.family == family)
            for family in sorted({item.family for item in outcomes})
        },
        "gradient_norm_before_clip": gradient_norm,
        "gradient_clip_threshold": args.gradient_clip,
        "gradient_clipped": gradient_norm > args.gradient_clip,
        "gradient_aggregation": aggregation,
        "state_gradient_nonzero_fraction": (
            None if not state_gradients else sum(state_gradients) / len(state_gradients)
        ),
        "family_counts": dict(sorted(Counter(item.family for item in outcomes).items())),
        "phase_counts": dict(sorted(Counter(item.phase for item in outcomes).items())),
        "target_event_kind_counts": dict(sorted(Counter(item.target_event_kind for item in outcomes).items())),
        "selected_step_set_counts": dict(sorted(Counter(str(item.selected_step_indices) for item in outcomes).items())),
        "image_min": min(item.image_min for item in outcomes),
        "image_max": max(item.image_max for item in outcomes),
        "image_saturation_fraction_mean": r5._mean(item.image_saturation_fraction for item in outcomes),
        "image_rms_mean": r5._mean(item.image_rms for item in outcomes),
        "optimizer_diagnostics": diagnostic_report,
        "schedule_receipts": [item.receipt for item in outcomes],
        "elapsed_seconds": elapsed,
        "updater_peak_memory_gib": r5._peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": r5._peak_gib(runtime.reader_device),
        "micro_records": [item.to_dict() for item in outcomes] if args.record_micro_metrics else None,
    }


def _aggregation_technical_gate(
    metrics: Sequence[Mapping[str, Any]],
    *,
    expected_mode: str,
    protocol_revision: str = "r7",
) -> dict[str, Any]:
    records = [
        metric["gradient_aggregation"]
        for metric in metrics
        if metric.get("kind") == "optimizer_step" and isinstance(metric.get("gradient_aggregation"), Mapping)
    ]
    errors = [float(record["norm_match_relative_error"]) for record in records]
    cosines = [float(record["raw_vs_applied_cosine"]) for record in records]
    modes_match = bool(records) and all(record.get("mode") == expected_mode for record in records)
    finite = bool(records) and all(
        math.isfinite(value)
        for record in records
        for value in (
            float(record["raw_mean_norm"]),
            float(record["unit_mean_norm_before_match"]),
            float(record["applied_norm_before_clip"]),
            float(record["raw_vs_unit_balanced_cosine"]),
            float(record["raw_vs_applied_cosine"]),
        )
    )
    norm_match = bool(errors) and max(errors) <= 1e-5
    expected_step_count = int(PROTOCOL_CONTRACT[protocol_revision]["optimizer_steps"])
    exact_step_count = len(records) == expected_step_count
    if expected_mode == "raw-mean":
        intervention_check = bool(cosines) and max(abs(value - 1.0) for value in cosines) <= 1e-6
    else:
        intervention_check = any(bool(record.get("intervention_active")) for record in records)
    common_descent_records = [
        record.get("common_descent_projection")
        for record in records
        if isinstance(record.get("common_descent_projection"), Mapping)
    ]
    common_descent_gate = True
    minimum_projected_micro_cosine = None
    maximum_projected_violation_count = None
    raw_violating_steps = None
    if expected_mode == "common-descent-projected-norm-matched":
        projected_cosines = [
            float(record["minimum_projected_micro_cosine"])
            for record in common_descent_records
        ]
        projected_violation_counts = [
            int(record["projected_violating_micro_count_at_tolerance"])
            for record in common_descent_records
        ]
        raw_violating_steps = sum(int(record["raw_violating_micro_count"]) > 0 for record in common_descent_records)
        minimum_projected_micro_cosine = min(projected_cosines) if projected_cosines else None
        maximum_projected_violation_count = max(projected_violation_counts) if projected_violation_counts else None
        common_descent_gate = (
            len(common_descent_records) == len(records)
            and bool(projected_cosines)
            and all(math.isfinite(value) and value >= -1e-5 for value in projected_cosines)
            and max(projected_violation_counts) == 0
            and raw_violating_steps > 0
        )
    result = {
        "schema": (
            "vision_memory.r8-common-descent-gate.v1"
            if protocol_revision == "r8"
            else "vision_memory.r7-gradient-aggregation-gate.v1"
        ),
        "expected_mode": expected_mode,
        "optimizer_step_records": len(records),
        "exact_step_count": exact_step_count,
        "modes_match": modes_match,
        "finite": finite,
        "norm_match": norm_match,
        "maximum_norm_match_relative_error": None if not errors else max(errors),
        "minimum_raw_vs_applied_cosine": None if not cosines else min(cosines),
        "intervention_check": intervention_check,
        "common_descent_constraint_check": common_descent_gate,
        "minimum_projected_micro_cosine": minimum_projected_micro_cosine,
        "maximum_projected_violation_count": maximum_projected_violation_count,
        "raw_violating_steps": raw_violating_steps,
        "passed": (
            exact_step_count
            and modes_match
            and finite
            and norm_match
            and intervention_check
            and common_descent_gate
        ),
    }
    return result


def _run(args: argparse.Namespace) -> dict[str, Any]:
    contract = PROTOCOL_CONTRACT[args.protocol_revision]
    protocol_tag = args.protocol_revision
    optimizer_steps = int(contract["optimizer_steps"])
    determinism = configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    set_all_seeds(args.seed)
    data, selected = r6._load_data(args)
    selected_sha = r5.canonical_sha256([segment.to_dict() for segment in selected])
    if selected_sha != contract["selected_segments_sha256"]:
        raise RuntimeError(f"{contract['protocol']} hard8 selection drifted: {selected_sha}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    r5._write_json(args.output_dir / "family_pool_audit.json", data.pool_audit)
    r5._write_json(args.output_dir / "dev_split_audit.json", data.split_audit)
    r5._write_json(args.output_dir / "schedule_audit.json", data.schedule_audit)
    runtime = _load_runtime(args)
    r5._write_environment(args.output_dir / "environment.txt")
    r5._write_json(args.output_dir / "runtime.json", r5._runtime_versions())
    manifest = _manifest(args=args, data=data, selected=selected, determinism=determinism)
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

    print(
        json.dumps({"milestone": f"{protocol_tag}_m0_overfit_eval", "arm": args.arm}, sort_keys=True),
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

    gradient_audit = r6._gradient_conflict_audit(
        runtime=runtime,
        reader_fn=train_reader,
        segments=selected,
    )
    r5._write_json(args.output_dir / "gradient_conflict_audit.json", gradient_audit)

    print(
        json.dumps(
            {
                "milestone": f"{protocol_tag}_training_start",
                "arm": args.arm,
                "gradient_aggregation": args.gradient_aggregation,
                "edit_start_sigma": 0.5,
                "optimizer_steps": optimizer_steps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    training_summary = r5.run_training_profile(
        args=args,
        optimizer_steps=optimizer_steps,
        data=data,
        runtime=runtime,
        manifest=manifest,
        optimizer_step_fn=_run_optimizer_step,
    )
    training_metrics = r6._read_jsonl(args.output_dir / "metrics.jsonl")
    aggregation_gate = _aggregation_technical_gate(
        training_metrics,
        expected_mode=args.gradient_aggregation,
        protocol_revision=args.protocol_revision,
    )
    combined_technical_gate = bool(training_summary["technical_gate"]["passed"]) and bool(aggregation_gate["passed"])

    endpoint_path = args.output_dir / "endpoint_ema.pt"
    load_trainable_weights(endpoint_path, trainable_module=runtime.model)
    endpoint_label = "ema_step128"
    print(
        json.dumps(
            {"milestone": f"{protocol_tag}_endpoint_overfit_eval", "arm": args.arm}, sort_keys=True
        ),
        flush=True,
    )
    endpoint_rows = r6._evaluation_rows(
        model=runtime.model,
        reader_fn=eval_reader,
        segments=selected,
        checkpoint=endpoint_label,
    )
    r6._append_rows(overfit_rows_path, endpoint_rows)
    overfit_rows = m0_rows + endpoint_rows
    overfit_summary = r5.summarize_evaluation_rows(
        overfit_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=20260840,
    )
    r5._write_json(args.output_dir / "overfit_evaluation_summary.json", overfit_summary)

    pilot_rows = r6._read_jsonl(args.output_dir / "pilot_evaluation_rows.jsonl")
    all_rows = overfit_rows + pilot_rows
    comparisons = {
        "train_overfit_hard8_endpoint_vs_m0": r6._checkpoint_comparison(
            all_rows,
            suite="train_overfit_hard8",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260841,
        ),
        "train_overfit_hard8_state_did": r6._difference_in_differences(
            all_rows,
            suite="train_overfit_hard8",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260842,
        ),
        "formal_select_32_endpoint_vs_m0": r6._checkpoint_comparison(
            all_rows,
            suite="formal_select_32",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260843,
        ),
        "mechanism_select_32_endpoint_vs_m0": r6._checkpoint_comparison(
            all_rows,
            suite="mechanism_select_32",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260844,
        ),
        "mechanism_select_32_state_did": r6._difference_in_differences(
            all_rows,
            suite="mechanism_select_32",
            endpoint=endpoint_label,
            iterations=args.bootstrap_iterations,
            seed=20260845,
        ),
    }
    overfit_delta = comparisons["train_overfit_hard8_endpoint_vs_m0"]
    overfit_did = comparisons["train_overfit_hard8_state_did"]
    formal_delta = comparisons["formal_select_32_endpoint_vs_m0"]
    mechanism_delta = comparisons["mechanism_select_32_endpoint_vs_m0"]
    mechanism_did = comparisons["mechanism_select_32_state_did"]
    endpoint_overfit_key = f"{endpoint_label}|train_overfit_hard8|normal"
    m0_overfit_key = "m0|train_overfit_hard8|normal"
    m0_accuracy = r6._metric(overfit_summary, m0_overfit_key, "accuracy")
    endpoint_accuracy = r6._metric(overfit_summary, endpoint_overfit_key, "accuracy")
    overfit_gate = (
        combined_technical_gate
        and overfit_delta["relative_change"] <= -0.20
        and overfit_delta["improved_pair_units"] >= 7
        and overfit_delta["ci95"][1] < 0.0
        and endpoint_accuracy - m0_accuracy >= 0.20
        and overfit_did["estimate"] < 0.0
    )
    fixed_dev_gate = (
        formal_delta["ci95"][1] < 0.0 and mechanism_delta["ci95"][1] < 0.0 and mechanism_did["ci95"][1] < 0.0
    )

    summary = {
        "schema": contract["summary_schema"],
        "status": "completed",
        "protocol": contract["protocol"],
        "protocol_revision": args.protocol_revision,
        "implementation_revision": contract["implementation_revision"],
        "git_commit": manifest["git_commit"],
        "arm": args.arm,
        "gradient_aggregation": args.gradient_aggregation,
        "edit_start_sigma": 0.5,
        "edit_start_sigma_semantics": "effective post-scheduler-shift flow sigma",
        "source_anchor_effective_sigma_schedule": list(r6.SOURCE_ANCHOR_EFFECTIVE_SIGMAS),
        "diagnostic_only_not_formal_success": True,
        "full_success_claim_allowed": False,
        "selected_segments": [segment.segment_id for segment in selected],
        "selected_segments_sha256": selected_sha,
        "gradient_conflict_audit": gradient_audit,
        "aggregation_technical_gate": aggregation_gate,
        "training_summary": training_summary,
        "overfit_evaluation_summary": overfit_summary,
        "comparisons": comparisons,
        "gates": {
            "technical_gate": combined_technical_gate,
            "hard8_overfit_learnability_gate": overfit_gate,
            "fixed_dev_generalization_gate": fixed_dev_gate,
            "formal_success_gate": False,
            "formal_success_gate_reason": (
                "single-seed repeated-subset diagnostic cannot establish fixed-full-data ID/OOD success"
            ),
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
    r5._write_json(args.output_dir / str(contract["summary_filename"]), summary)
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
    contract = PROTOCOL_CONTRACT[args.protocol_revision]
    r5._write_json(args.output_dir / str(contract["summary_filename"]), summary)
    print(
        json.dumps(
            {
                "milestone": f"{args.protocol_revision}_completed",
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
