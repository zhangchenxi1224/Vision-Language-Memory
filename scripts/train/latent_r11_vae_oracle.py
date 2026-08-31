"""R11 direct VAE-latent reachability oracle for one fixed F1 target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.inspire.model_snapshot_manifest import verify_snapshot_binding  # noqa: E402
from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r7_gradient_balance as r8  # noqa: E402
from vision_memory.data import CYCLIC4  # noqa: E402
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402
from vision_memory.training.checkpoint import save_training_checkpoint  # noqa: E402
from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_SELECTED_SEGMENTS_SHA256,
    R10_SELECTION_SEED,
    select_f1_targets,
    target_gate,
    target_statistics,
)
from vision_memory.training.r11_latent import (  # noqa: E402
    R11_CHECKPOINT_STEPS,
    R11_LEARNING_RATE,
    R11_OPTIMIZER_STEPS,
    R11_PROTOCOL,
    training_view_counts,
)


IMPLEMENTATION_REVISION = "direct-vae-latent-fp32-v1"
MANIFEST_SCHEMA = "vision_memory.r11-vae-latent-oracle-manifest.v1"
METRICS_SCHEMA = "vision_memory.r11-vae-latent-oracle-metrics.v1"
SUMMARY_SCHEMA = "vision_memory.r11-vae-latent-oracle-summary.v1"
TECHNICAL_GATE_SCHEMA = "vision_memory.r11-vae-latent-oracle-technical-gate.v1"
SUITE = "r11_f1_vae_latent"


def _posterior_mean(output: Any) -> Tensor:
    direct = getattr(output, "latents", None)
    if isinstance(direct, Tensor):
        return direct
    distribution = getattr(output, "latent_dist", None)
    if distribution is None and isinstance(output, (tuple, list)) and output:
        distribution = output[0]
    if isinstance(distribution, Tensor):
        return distribution
    if distribution is None:
        raise TypeError(f"Unsupported VAE posterior output: {type(output)!r}")
    mode = getattr(distribution, "mode", None)
    if callable(mode):
        return mode()
    mean = getattr(distribution, "mean", None)
    if isinstance(mean, Tensor):
        return mean
    raise TypeError("VAE posterior exposes neither mode() nor mean.")


def encode_model_latent(vae: nn.Module, unit_image: Tensor) -> Tensor:
    raw_image = unit_image * 2.0 - 1.0
    posterior = vae.encode(raw_image, return_dict=True)
    latent = _posterior_mean(posterior)
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    shift = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    return (latent - shift) * scaling


class VAELatentOracle(nn.Module):
    """Exactly one FP32 model-space latent decoded through a frozen VAE."""

    def __init__(self, *, vae: nn.Module, initial_latent: Tensor, compute_dtype: torch.dtype) -> None:
        super().__init__()
        if initial_latent.ndim != 4 or initial_latent.shape[0] != 1:
            raise ValueError(f"R11 latent must be batch-one BCHW, got {tuple(initial_latent.shape)}.")
        if not torch.isfinite(initial_latent).all():
            raise ValueError("R11 initial latent is non-finite.")
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae
        self.compute_dtype = compute_dtype
        self.latent_fp32 = nn.Parameter(initial_latent.detach().float().clone())
        self.register_buffer("initial_latent_fp32", initial_latent.detach().float().clone(), persistent=False)

    def image(self) -> Tensor:
        latent = self.latent_fp32.to(dtype=self.compute_dtype)
        return decode_model_latents_unit_interval(self.vae, latent, clamp=True)


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
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.adapter_seed is None:
        args.adapter_seed = args.seed
    args.resolution = 1024
    args.schedule_seed = R10_SELECTION_SEED
    args.bootstrap_iterations = 10_000
    args.lora_rank = 4
    args.selected_step_count = 0
    args.gradient_mode = "full"
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("train", "dev"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R11 {name} path is not a file: {getattr(args, name)}")
    for name in ("dreamlite", "reader"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R11 {name} path is not a directory: {getattr(args, name)}")
    if torch.device(args.dreamlite_device) == torch.device(args.reader_device):
        raise ValueError("R11 VAE and Reader must use distinct devices.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("A fresh R11 target refuses a non-empty output directory.")
    if r8.git_value("status", "--porcelain") and not args.allow_dirty:
        raise ValueError("R11 refuses a dirty source tree unless --allow-dirty is explicit.")


def _append_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def _save_image(path: Path, image: Tensor) -> None:
    value = image.detach().float().cpu()
    if tuple(value.shape) != (1, 3, 1024, 1024):
        raise ValueError(f"R11 image artifact has invalid shape: {tuple(value.shape)}")
    array = (
        value[0].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _checkpoint(
    *,
    step: int,
    oracle: VAELatentOracle,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> None:
    save_training_checkpoint(
        output_dir / "checkpoints" / f"step-{step:03d}.pt",
        trainable_module=oracle,
        optimizer=optimizer,
        epoch=0,
        episode_cursor=step,
        optimizer_step=step,
        manifest=manifest,
        trainer_state={
            "schema": "vision_memory.r11-vae-latent-checkpoint-state.v1",
            "next_optimizer_step": step,
        },
    )
    _save_image(output_dir / "images" / f"step-{step:03d}.png", oracle.image())


def _item(segment: r5.R5Segment, state: Tensor) -> r5.EvalItem:
    cpu = state.detach().to(device="cpu", copy=True)
    return r5.EvalItem(
        item_id=segment.segment_id,
        pair_unit=segment.segment_id,
        episode_id=segment.query_source_episode_id,
        query_id=f"{segment.query_source_episode_id}:{segment.query_turn_index}",
        query=segment.query,
        normal_state=cpu,
        temporal_state=cpu,
        family=segment.family,
        target_event_kind=segment.target_event_kind,
        query_gap=segment.query_gap,
        updater_count=segment.updater_count,
        cross_slot_interference=segment.cross_slot_interference,
        stale_target_text=segment.stale_target_text,
    )


def _evaluation_rows(
    *,
    reader_fn: Any,
    segment: r5.R5Segment,
    normal_image: Tensor,
    reset_image: Tensor,
    checkpoint: str,
) -> list[dict[str, Any]]:
    item = _item(segment, normal_image)
    rows = []
    with torch.no_grad():
        for condition, image in (("normal", normal_image), ("reset", reset_image)):
            for view_index, permutation in enumerate(r5.REVERSE_CYCLIC4):
                ordered = tuple(segment.query.choices[index] for index in permutation)
                target_index = permutation.index(segment.query.target_index)
                output = reader_fn(
                    image,
                    r5.format_mcq_query(segment.query.text, ordered),
                    ordered,
                    target_index,
                )
                rows.append(
                    r5._choice_row(
                        reader_output=output,
                        item=item,
                        checkpoint_label=checkpoint,
                        suite=SUITE,
                        condition=condition,
                        permutation=permutation,
                        view_index=view_index,
                        donor_item_id=None,
                    )
                )
    return rows


def _target_phase(segment_id: str) -> int:
    return int.from_bytes(hashlib.sha256(segment_id.encode()).digest()[:2], "big") % 4


def _train_step(
    *,
    step_zero: int,
    oracle: VAELatentOracle,
    optimizer: torch.optim.Optimizer,
    reader_fn: Any,
    segment: r5.R5Segment,
) -> dict[str, Any]:
    view_index = (step_zero + _target_phase(segment.segment_id)) % 4
    permutation = CYCLIC4[view_index]
    ordered = tuple(segment.query.choices[index] for index in permutation)
    target_index = permutation.index(segment.query.target_index)
    optimizer.zero_grad(set_to_none=True)
    image = oracle.image()
    output = reader_fn(
        image,
        r5.format_mcq_query(segment.query.text, ordered),
        ordered,
        target_index,
    )
    loss = output.loss
    if not isinstance(loss, Tensor) or loss.numel() != 1 or not torch.isfinite(loss):
        raise RuntimeError("R11 Reader returned an invalid scalar loss.")
    loss.backward()
    gradient = oracle.latent_fp32.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("R11 latent received no finite gradient.")
    gradient_norm = float(gradient.double().norm())
    nonzero_fraction = float((gradient != 0).double().mean())
    if gradient_norm <= 0.0 or nonzero_fraction <= 0.0:
        raise RuntimeError("R11 latent received a zero gradient.")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updated_image = oracle.image().detach()
    latent = oracle.latent_fp32.detach()
    delta = latent - oracle.initial_latent_fp32
    return {
        "schema": METRICS_SCHEMA,
        "kind": "optimizer_step",
        "optimizer_step": step_zero + 1,
        "target_segment_id": segment.segment_id,
        "forward_cyclic_training_view": view_index,
        "permutation": list(permutation),
        "loss_before_step": float(loss.detach()),
        "gradient_norm": gradient_norm,
        "gradient_nonzero_fraction": nonzero_fraction,
        "latent_min_after_step": float(latent.min()),
        "latent_max_after_step": float(latent.max()),
        "latent_rms_after_step": float(latent.square().mean().sqrt()),
        "latent_delta_norm_after_step": float(delta.double().norm()),
        "image_min_after_step": float(updated_image.min()),
        "image_max_after_step": float(updated_image.max()),
        "image_rms_after_step": float(updated_image.square().mean().sqrt()),
        "image_saturation_fraction_after_step": float(
            ((updated_image <= 0.0) | (updated_image >= 1.0)).double().mean()
        ),
        "learning_rate": R11_LEARNING_RATE,
    }


def _technical_gate(
    metrics: Sequence[Mapping[str, Any]],
    *,
    target_segment_id: str,
    oracle: VAELatentOracle,
    reader: nn.Module,
    checkpoint_steps: set[int],
    png_steps: set[int],
    snapshots_unchanged: bool,
) -> dict[str, Any]:
    expected_steps = list(range(1, R11_OPTIMIZER_STEPS + 1))
    views = training_view_counts(metrics)
    expected_views = {0: 64, 1: 64, 2: 64, 3: 64}
    trainable = [name for name, parameter in oracle.named_parameters() if parameter.requires_grad]
    finite_fields = (
        "loss_before_step",
        "gradient_norm",
        "gradient_nonzero_fraction",
        "latent_min_after_step",
        "latent_max_after_step",
        "latent_rms_after_step",
        "latent_delta_norm_after_step",
        "image_min_after_step",
        "image_max_after_step",
    )
    passed = bool(
        len(metrics) == R11_OPTIMIZER_STEPS
        and [int(row["optimizer_step"]) for row in metrics] == expected_steps
        and all(row.get("target_segment_id") == target_segment_id for row in metrics)
        and all(math.isfinite(float(row[field])) for row in metrics for field in finite_fields)
        and all(float(row["gradient_norm"]) > 0.0 for row in metrics)
        and all(float(row["gradient_nonzero_fraction"]) > 0.0 for row in metrics)
        and views == expected_views
        and checkpoint_steps == set(R11_CHECKPOINT_STEPS)
        and png_steps == set(R11_CHECKPOINT_STEPS)
        and trainable == ["latent_fp32"]
        and all(not parameter.requires_grad and parameter.grad is None for parameter in oracle.vae.parameters())
        and all(not parameter.requires_grad and parameter.grad is None for parameter in reader.parameters())
        and snapshots_unchanged
    )
    return {
        "schema": TECHNICAL_GATE_SCHEMA,
        "passed": passed,
        "optimizer_step_records": len(metrics),
        "training_view_counts": views,
        "expected_training_view_counts": expected_views,
        "checkpoint_steps_observed": sorted(checkpoint_steps),
        "png_steps_observed": sorted(png_steps),
        "trainable_parameter_names": trainable,
        "vae_frozen": all(not parameter.requires_grad for parameter in oracle.vae.parameters()),
        "reader_frozen": all(not parameter.requires_grad for parameter in reader.parameters()),
        "snapshots_unchanged": snapshots_unchanged,
        "minimum_gradient_norm": min(float(row["gradient_norm"]) for row in metrics),
        "minimum_gradient_nonzero_fraction": min(
            float(row["gradient_nonzero_fraction"]) for row in metrics
        ),
    }


def _manifest(
    *,
    args: argparse.Namespace,
    data: r5.R5DataBundle,
    selected: Sequence[r5.R5Segment],
    target: r5.R5Segment,
    initial_latent: Tensor,
    snapshot_bindings: Mapping[str, Any],
    determinism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol": R11_PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": r8.git_value("rev-parse", "HEAD"),
        "git_dirty": bool(r8.git_value("status", "--porcelain")),
        "target_index": args.target_index,
        "target_segment": target.to_dict(),
        "target_segment_id": target.segment_id,
        "selected_segment_ids": [segment.segment_id for segment in selected],
        "selected_segments_sha256": R10_SELECTED_SEGMENTS_SHA256,
        "train_sha256": r5.sha256_file(args.train),
        "dev_sha256": r5.sha256_file(args.dev),
        "family_pool_audit": data.pool_audit,
        "dev_split_audit": data.split_audit,
        "model_snapshot_payloads_start": dict(snapshot_bindings),
        "strict_determinism": dict(determinism) if determinism is not None else None,
        "initial_latent": {
            "shape": list(initial_latent.shape),
            "dtype": str(initial_latent.dtype),
            "sha256": canonical_tensor_sha256(initial_latent.detach().float().cpu()),
        },
        "fixed_contract": {
            "resolution": 1024,
            "only_trainable": "one unconstrained fp32 DreamLite model-space latent tensor",
            "vae_frozen": True,
            "dreamlite_unet_executed": False,
            "semantic_prompt_used": False,
            "optimizer": "Adam",
            "learning_rate": R11_LEARNING_RATE,
            "weight_decay": 0.0,
            "optimizer_steps": R11_OPTIMIZER_STEPS,
            "checkpoint_steps": list(R11_CHECKPOINT_STEPS),
            "primary_endpoint": "raw_latent_step256",
            "best_checkpoint_selection_forbidden": True,
        },
        "diagnostic_only_not_formal_success": True,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("R11 VAE-latent oracle requires CUDA.")
    determinism = r8.configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    r8.set_all_seeds(args.seed)
    data = r5._load_data(args, optimizer_steps=0)
    selected = select_f1_targets(data.train_pools)
    selected_sha = r5.canonical_sha256([segment.to_dict() for segment in selected])
    if selected_sha != R10_SELECTED_SEGMENTS_SHA256:
        raise RuntimeError(f"R11 F1 selection drifted: {selected_sha}")
    target = selected[args.target_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    updater_dtype = r5.compute_dtype(updater_device)
    reader_dtype = r5.compute_dtype(reader_device)
    pipe = r5._load_pipeline(args, updater_device, updater_dtype)
    pipe.unet.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    processor, reader = r5._load_reader(args, reader_device, reader_dtype)
    snapshot_bindings = {
        "dreamlite_mobile": r5._verified_snapshot_payload(
            model_dir=args.dreamlite,
            model_key="dreamlite_mobile",
            env_name="VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256",
            required=args.strict_determinism,
        ),
        "qwen_reader": r5._verified_snapshot_payload(
            model_dir=args.reader,
            model_key="qwen_reader",
            env_name="VLM_READER_SNAPSHOT_MANIFEST_SHA256",
            required=args.strict_determinism,
        ),
    }
    initial_rgb = r5._initial_rgb_tensor(
        resolution=1024,
        device=updater_device,
        dtype=updater_dtype,
    )
    with torch.no_grad():
        initial_latent = encode_model_latent(pipe.vae, initial_rgb)
    oracle = VAELatentOracle(
        vae=pipe.vae,
        initial_latent=initial_latent,
        compute_dtype=updater_dtype,
    )
    optimizer = torch.optim.Adam((oracle.latent_fp32,), lr=R11_LEARNING_RATE, weight_decay=0.0)
    r5._write_environment(args.output_dir / "environment.txt")
    r5._write_json(args.output_dir / "runtime.json", r5._runtime_versions())
    r5._write_json(args.output_dir / "family_pool_audit.json", data.pool_audit)
    r5._write_json(args.output_dir / "dev_split_audit.json", data.split_audit)
    manifest = _manifest(
        args=args,
        data=data,
        selected=selected,
        target=target,
        initial_latent=initial_latent,
        snapshot_bindings=snapshot_bindings,
        determinism=determinism,
    )
    r5._write_json(args.output_dir / "manifest.json", manifest)
    train_reader = r8.choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=reader_device,
        require_grad=True,
        deterministic_ce=args.strict_determinism,
    )
    eval_reader = r8.choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=reader_device,
        require_grad=False,
        deterministic_ce=args.strict_determinism,
    )
    reset_image = oracle.image().detach()
    rows_path = args.output_dir / "target_evaluation_rows.jsonl"
    m0_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        normal_image=reset_image,
        reset_image=reset_image,
        checkpoint="m0",
    )
    _append_jsonl(rows_path, m0_rows)
    _checkpoint(step=0, oracle=oracle, optimizer=optimizer, manifest=manifest, output_dir=args.output_dir)
    metrics = []
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    for step_zero in range(R11_OPTIMIZER_STEPS):
        metric = _train_step(
            step_zero=step_zero,
            oracle=oracle,
            optimizer=optimizer,
            reader_fn=train_reader,
            segment=target,
        )
        metric["elapsed_seconds"] = time.monotonic() - started
        metrics.append(metric)
        _append_jsonl(metrics_path, (metric,))
        step = step_zero + 1
        if step in R11_CHECKPOINT_STEPS:
            _checkpoint(
                step=step,
                oracle=oracle,
                optimizer=optimizer,
                manifest=manifest,
                output_dir=args.output_dir,
            )
        print(
            json.dumps(
                {
                    "milestone": "r11_optimizer_step",
                    "target_index": args.target_index,
                    "optimizer_step": step,
                    "loss": metric["loss_before_step"],
                    "gradient_norm": metric["gradient_norm"],
                    "elapsed_seconds": metric["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    endpoint_image = oracle.image().detach()
    endpoint_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        normal_image=endpoint_image,
        reset_image=reset_image,
        checkpoint="raw_latent_step256",
    )
    _append_jsonl(rows_path, endpoint_rows)
    torch.save(
        {
            "schema": "vision_memory.r11-vae-latent-endpoint.v1",
            "latent_fp32": oracle.latent_fp32.detach().cpu(),
            "image": endpoint_image.cpu(),
            "manifest_sha256": r5.sha256_file(args.output_dir / "manifest.json"),
        },
        args.output_dir / "endpoint_raw.pt",
    )
    _save_image(args.output_dir / "endpoint_raw.png", endpoint_image)
    observed_bindings = {
        name: verify_snapshot_binding(binding) for name, binding in snapshot_bindings.items()
    }
    snapshots_unchanged = observed_bindings == snapshot_bindings
    r5._write_json(
        args.output_dir / "model_snapshot_verification_end.json",
        {
            "schema": "vision_memory.r11-model-snapshot-end-verification.v1",
            "passed": snapshots_unchanged,
            "bindings": observed_bindings,
        },
    )
    checkpoint_steps = {
        int(path.stem.removeprefix("step-"))
        for path in (args.output_dir / "checkpoints").glob("step-*.pt")
    }
    png_steps = {
        int(path.stem.removeprefix("step-"))
        for path in (args.output_dir / "images").glob("step-*.png")
    }
    technical = _technical_gate(
        metrics,
        target_segment_id=target.segment_id,
        oracle=oracle,
        reader=reader,
        checkpoint_steps=checkpoint_steps,
        png_steps=png_steps,
        snapshots_unchanged=snapshots_unchanged,
    )
    all_rows = m0_rows + endpoint_rows
    statistics = target_statistics(
        all_rows,
        suite=SUITE,
        target_segment_id=target.segment_id,
        endpoint="raw_latent_step256",
    )
    scientific_gate = target_gate(statistics, technical_gate=bool(technical["passed"]))
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "protocol": R11_PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": manifest["git_commit"],
        "target_index": args.target_index,
        "target_segment_id": target.segment_id,
        "target_family": target.family,
        "selected_segments": [segment.segment_id for segment in selected],
        "selected_segments_sha256": selected_sha,
        "optimizer_steps": R11_OPTIMIZER_STEPS,
        "technical_gate": technical,
        "target_statistics": statistics,
        "gates": {
            "technical_gate": bool(technical["passed"]),
            "target_latent_reachability_gate": scientific_gate,
            "formal_success_gate": False,
        },
        "diagnostic_only_not_formal_success": True,
        "full_success_claim_allowed": False,
        "checkpoint_steps_observed": sorted(checkpoint_steps),
        "artifacts": {
            "manifest_sha256": r5.sha256_file(args.output_dir / "manifest.json"),
            "metrics_sha256": r5.sha256_file(metrics_path),
            "evaluation_rows_sha256": r5.sha256_file(rows_path),
            "endpoint_raw_sha256": r5.sha256_file(args.output_dir / "endpoint_raw.pt"),
            "endpoint_png_sha256": r5.sha256_file(args.output_dir / "endpoint_raw.png"),
            "snapshot_end_sha256": r5.sha256_file(
                args.output_dir / "model_snapshot_verification_end.json"
            ),
        },
        "wall_clock_seconds": time.monotonic() - started,
    }
    r5._write_json(args.output_dir / "r11_latent_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_args(args)
        summary = _run(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "milestone": "r11_vae_latent_completed",
                "target_index": summary["target_index"],
                "target_latent_reachability_gate": summary["gates"]["target_latent_reachability_gate"],
                "formal_success_gate": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
