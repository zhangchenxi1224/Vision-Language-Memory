"""R10 direct-pixel visual-alignment lower-bound diagnostic for one fixed F1 target."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
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
from vision_memory.training.checkpoint import save_training_checkpoint  # noqa: E402
from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_OPTIMIZER_STEPS,
    R10_PROTOCOL,
    R10_SELECTED_SEGMENTS_SHA256,
    R10_SELECTION_SEED,
    select_f1_targets,
    target_gate,
    target_statistics,
)


IMPLEMENTATION_REVISION = "direct-pixel-logit-v1"
MANIFEST_SCHEMA = "vision_memory.r10-direct-pixel-oracle-manifest.v1"
METRICS_SCHEMA = "vision_memory.r10-direct-pixel-oracle-metrics.v1"
SUMMARY_SCHEMA = "vision_memory.r10-direct-pixel-oracle-summary.v1"
TECHNICAL_GATE_SCHEMA = "vision_memory.r10-direct-pixel-oracle-technical-gate.v1"
SUITE = "r10_f1_single_target"
LEARNING_RATE = 0.05
CHECKPOINT_STEPS = (0, 32, 64, 96, 128)


class PixelOracle(nn.Module):
    """A single unconstrained RGB-logit tensor exposed through a sigmoid image."""

    def __init__(self, initial_rgb: Tensor) -> None:
        super().__init__()
        if tuple(initial_rgb.shape) != (1, 3, 1024, 1024):
            raise ValueError(f"R10 pixel oracle requires [1,3,1024,1024], got {tuple(initial_rgb.shape)}.")
        if initial_rgb.dtype is not torch.float32:
            raise TypeError("R10 pixel oracle logits must be initialized from FP32 RGB.")
        if not torch.isfinite(initial_rgb).all() or bool((initial_rgb <= 0).any()) or bool((initial_rgb >= 1).any()):
            raise ValueError("R10 pixel oracle initialization must be finite and strictly inside (0,1).")
        self.image_logits = nn.Parameter(torch.logit(initial_rgb.detach().clone()))

    def image(self) -> Tensor:
        return torch.sigmoid(self.image_logits)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index", type=int, choices=range(8), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairing-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=20260730)
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.resolution = 1024
    args.schedule_seed = R10_SELECTION_SEED
    args.selected_step_count = 0
    args.gradient_mode = "full"
    args.bootstrap_iterations = 10_000
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("train", "dev"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R10 {name} path is not a file: {getattr(args, name)}")
    if not args.reader.is_dir():
        raise ValueError(f"R10 reader path is not a directory: {args.reader}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("A fresh R10 target refuses a non-empty output directory.")
    if r8.git_value("status", "--porcelain") and not args.allow_dirty:
        raise ValueError("R10 refuses a dirty source tree unless --allow-dirty is explicit.")


def _append_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def _save_image(path: Path, image: Tensor) -> None:
    value = image.detach().float().cpu()
    if tuple(value.shape) != (1, 3, 1024, 1024):
        raise ValueError(f"R10 image artifact has invalid shape: {tuple(value.shape)}")
    array = (
        value[0]
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def _checkpoint(
    *,
    step: int,
    oracle: PixelOracle,
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
            "schema": "vision_memory.r10-direct-pixel-oracle-checkpoint-state.v1",
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
    rows: list[dict[str, Any]] = []
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
    oracle: PixelOracle,
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
        raise RuntimeError("R10 direct-pixel Reader returned an invalid scalar loss.")
    loss.backward()
    gradient = oracle.image_logits.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("R10 direct-pixel logits received no finite gradient.")
    gradient_norm = float(gradient.double().norm())
    nonzero_fraction = float((gradient != 0).double().mean())
    if not math.isfinite(gradient_norm) or gradient_norm <= 0.0 or nonzero_fraction <= 0.0:
        raise RuntimeError("R10 direct-pixel logits received a zero or invalid gradient.")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    updated = oracle.image().detach()
    minimum = float(updated.min())
    maximum = float(updated.max())
    if not torch.isfinite(updated).all() or minimum < 0.0 or maximum > 1.0:
        raise RuntimeError("R10 direct-pixel image escaped its finite [0,1] contract.")
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
        "image_min_after_step": minimum,
        "image_max_after_step": maximum,
        "image_mean_after_step": float(updated.mean()),
        "image_rms_after_step": float(updated.square().mean().sqrt()),
        "image_saturation_fraction_after_step": float(((updated <= 0.0) | (updated >= 1.0)).double().mean()),
        "learning_rate": LEARNING_RATE,
    }


def _reader_frozen(reader: nn.Module) -> bool:
    return all(not parameter.requires_grad and parameter.grad is None for parameter in reader.parameters())


def _technical_gate(
    metrics: Sequence[Mapping[str, Any]],
    *,
    target_segment_id: str,
    checkpoint_steps: set[int],
    png_steps: set[int],
    reader_frozen: bool,
    snapshot_end_passed: bool,
) -> dict[str, Any]:
    records = [value for value in metrics if value.get("kind") == "optimizer_step"]
    views = Counter(int(value["forward_cyclic_training_view"]) for value in records)
    expected_views = {0: 32, 1: 32, 2: 32, 3: 32}
    expected_steps = list(range(1, R10_OPTIMIZER_STEPS + 1))
    finite_fields = (
        "loss_before_step",
        "gradient_norm",
        "gradient_nonzero_fraction",
        "image_min_after_step",
        "image_max_after_step",
    )
    passed = bool(
        len(records) == R10_OPTIMIZER_STEPS
        and [int(value["optimizer_step"]) for value in records] == expected_steps
        and all(value.get("target_segment_id") == target_segment_id for value in records)
        and all(math.isfinite(float(value[field])) for value in records for field in finite_fields)
        and all(float(value["gradient_norm"]) > 0.0 for value in records)
        and all(float(value["gradient_nonzero_fraction"]) > 0.0 for value in records)
        and all(0.0 <= float(value["image_min_after_step"]) <= float(value["image_max_after_step"]) <= 1.0 for value in records)
        and dict(sorted(views.items())) == expected_views
        and checkpoint_steps == set(CHECKPOINT_STEPS)
        and png_steps == set(CHECKPOINT_STEPS)
        and reader_frozen
        and snapshot_end_passed
    )
    return {
        "schema": TECHNICAL_GATE_SCHEMA,
        "passed": passed,
        "optimizer_step_records": len(records),
        "training_view_counts": dict(sorted(views.items())),
        "expected_training_view_counts": expected_views,
        "checkpoint_steps_observed": sorted(checkpoint_steps),
        "png_steps_observed": sorted(png_steps),
        "reader_frozen": reader_frozen,
        "snapshot_end_passed": snapshot_end_passed,
        "minimum_gradient_norm": min((float(value["gradient_norm"]) for value in records), default=None),
        "minimum_gradient_nonzero_fraction": min(
            (float(value["gradient_nonzero_fraction"]) for value in records),
            default=None,
        ),
    }


def _manifest(
    *,
    args: argparse.Namespace,
    data: r5.R5DataBundle,
    selected: Sequence[r5.R5Segment],
    target: r5.R5Segment,
    reader_binding: Mapping[str, Any],
    determinism: Mapping[str, Any] | None,
) -> dict[str, Any]:
    status = r8.git_value("status", "--porcelain")
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol": R10_PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": r8.git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "arm": "direct-pixel-oracle",
        "scientific_role": "capacity oracle only; never a deployable memory method or formal success claim",
        "target_index": args.target_index,
        "target_segment": target.to_dict(),
        "target_segment_id": target.segment_id,
        "selected_segment_ids": [segment.segment_id for segment in selected],
        "selected_segments_sha256": R10_SELECTED_SEGMENTS_SHA256,
        "train_sha256": r5.sha256_file(args.train),
        "dev_sha256": r5.sha256_file(args.dev),
        "family_pool_audit": data.pool_audit,
        "dev_split_audit": data.split_audit,
        "reader_snapshot_payload_start": dict(reader_binding),
        "strict_determinism": dict(determinism) if determinism is not None else None,
        "fixed_contract": {
            "resolution": 1024,
            "parameterization": "image=sigmoid(one_unconstrained_fp32_rgb_logit_tensor)",
            "initial_rgb": 127 / 255,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "optimizer_steps": R10_OPTIMIZER_STEPS,
            "forward_cyclic_training_view_exposures": {str(index): 32 for index in range(4)},
            "heldout_reverse_cyclic_endpoint_views": 4,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "primary_endpoint": "raw_pixel_logits_step128",
            "evaluation_controls": ["normal", "reset"],
            "reader_loss": "four-choice listwise CE",
            "best_checkpoint_selection_forbidden": True,
        },
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in sorted(vars(args).items())
            if key != "output_dir"
        },
        "diagnostic_only_not_formal_success": True,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("R10 direct-pixel oracle requires CUDA.")
    determinism = r8.configure_strict_cuda_determinism(args.seed) if args.strict_determinism else None
    r8.set_all_seeds(args.seed)
    data = r5._load_data(args, optimizer_steps=0)
    selected = select_f1_targets(data.train_pools)
    selected_sha = r5.canonical_sha256([segment.to_dict() for segment in selected])
    if selected_sha != R10_SELECTED_SEGMENTS_SHA256:
        raise RuntimeError(f"R10 F1 selection drifted: {selected_sha}")
    target = selected[args.target_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.reader_device)
    processor, reader = r5._load_reader(args, device, r5.compute_dtype(device))
    reader_binding = r5._verified_snapshot_payload(
        model_dir=args.reader,
        model_key="qwen_reader",
        env_name="VLM_READER_SNAPSHOT_MANIFEST_SHA256",
        required=args.strict_determinism,
    )
    r5._write_environment(args.output_dir / "environment.txt")
    r5._write_json(args.output_dir / "runtime.json", r5._runtime_versions())
    r5._write_json(args.output_dir / "family_pool_audit.json", data.pool_audit)
    r5._write_json(args.output_dir / "dev_split_audit.json", data.split_audit)

    initial = r5._initial_rgb_tensor(resolution=1024, device=device, dtype=torch.float32)
    expected = torch.full_like(initial, 127 / 255)
    if not torch.equal(initial, expected):
        raise RuntimeError("R10 direct-pixel initialization drifted from exact uniform 127/255 RGB.")
    oracle = PixelOracle(initial)
    if float((oracle.image().detach() - initial).abs().max()) > 1e-7:
        raise RuntimeError("R10 logit parameterization failed to reproduce the locked initial RGB.")
    optimizer = torch.optim.Adam(oracle.parameters(), lr=LEARNING_RATE, weight_decay=0.0)
    manifest = _manifest(
        args=args,
        data=data,
        selected=selected,
        target=target,
        reader_binding=reader_binding,
        determinism=determinism,
    )
    r5._write_json(args.output_dir / "manifest.json", manifest)

    train_reader = r8.choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=device,
        require_grad=True,
        deterministic_ce=args.strict_determinism,
    )
    eval_reader = r8.choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=device,
        require_grad=False,
        deterministic_ce=args.strict_determinism,
    )
    rows_path = args.output_dir / "target_evaluation_rows.jsonl"
    m0_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        normal_image=oracle.image().detach(),
        reset_image=initial,
        checkpoint="m0",
    )
    _append_jsonl(rows_path, m0_rows)
    _checkpoint(step=0, oracle=oracle, optimizer=optimizer, manifest=manifest, output_dir=args.output_dir)

    metrics: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    for step_zero in range(R10_OPTIMIZER_STEPS):
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
        if step in CHECKPOINT_STEPS:
            _checkpoint(
                step=step,
                oracle=oracle,
                optimizer=optimizer,
                manifest=manifest,
                output_dir=args.output_dir,
            )

    endpoint = oracle.image().detach()
    endpoint_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        normal_image=endpoint,
        reset_image=initial,
        checkpoint="raw_step128",
    )
    _append_jsonl(rows_path, endpoint_rows)
    torch.save(
        {
            "schema": "vision_memory.r10-direct-pixel-oracle-endpoint.v1",
            "image_logits": oracle.image_logits.detach().cpu(),
            "image": endpoint.cpu(),
            "manifest_sha256": r5.sha256_file(args.output_dir / "manifest.json"),
        },
        args.output_dir / "endpoint_raw.pt",
    )
    _save_image(args.output_dir / "endpoint_raw.png", endpoint)

    observed_binding = verify_snapshot_binding(reader_binding)
    snapshot_end_passed = observed_binding == dict(reader_binding)
    r5._write_json(
        args.output_dir / "snapshot_end_verification.json",
        {
            "schema": "vision_memory.r10-reader-snapshot-end-verification.v1",
            "passed": snapshot_end_passed,
            "binding": observed_binding,
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
        checkpoint_steps=checkpoint_steps,
        png_steps=png_steps,
        reader_frozen=_reader_frozen(reader),
        snapshot_end_passed=snapshot_end_passed,
    )
    all_rows = m0_rows + endpoint_rows
    statistics = target_statistics(
        all_rows,
        suite=SUITE,
        target_segment_id=target.segment_id,
        endpoint="raw_step128",
    )
    scientific_gate = target_gate(statistics, technical_gate=bool(technical["passed"]))
    evaluation_summary = r5.summarize_evaluation_rows(
        all_rows,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=20261100 + args.target_index,
    )
    r5._write_json(args.output_dir / "target_evaluation_summary.json", evaluation_summary)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "protocol": R10_PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": manifest["git_commit"],
        "arm": "direct-pixel-oracle",
        "target_index": args.target_index,
        "target_segment_id": target.segment_id,
        "target_family": target.family,
        "selected_segments": [segment.segment_id for segment in selected],
        "selected_segments_sha256": selected_sha,
        "optimizer_steps": R10_OPTIMIZER_STEPS,
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
            "manifest_sha256": r5.sha256_file(args.output_dir / "manifest.json"),
            "metrics_sha256": r5.sha256_file(metrics_path),
            "evaluation_rows_sha256": r5.sha256_file(rows_path),
            "endpoint_raw_sha256": r5.sha256_file(args.output_dir / "endpoint_raw.pt"),
            "endpoint_png_sha256": r5.sha256_file(args.output_dir / "endpoint_raw.png"),
            "snapshot_end_verification_sha256": r5.sha256_file(
                args.output_dir / "snapshot_end_verification.json"
            ),
        },
        "wall_clock_seconds": time.monotonic() - started,
    }
    r5._write_json(args.output_dir / "r10_pixel_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_args(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    summary = _run(args)
    print(
        json.dumps(
            {
                "milestone": "r10_direct_pixel_completed",
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
