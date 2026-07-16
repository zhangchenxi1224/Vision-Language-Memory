from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train.lightweight_episode import (  # noqa: E402
    episode_value,
    event_payload,
    query_payload,
    training_subset_audit,
    turn_kind,
    validate_overfit_gate_episodes,
)
from vision_memory.data import read_jsonl  # noqa: E402
from vision_memory.dreamlite import assert_no_frozen_parameter_grads, freeze_module  # noqa: E402
from vision_memory.lightweight import LightweightVisualUpdater  # noqa: E402
from vision_memory.reader import qwen3vl_choice_nll, qwen3vl_target_only_ce  # noqa: E402
from vision_memory.repro import (  # noqa: E402
    canonical_object_sha256,
    canonical_tensor_sha256,
    configure_strict_cuda_determinism,
    model_optimizer_rng_manifest,
    named_tensors_manifest,
)
from vision_memory.training import format_mcq_query, run_episode  # noqa: E402


EPISODE_COUNT = 64
SEED = 0
STATE_CHANNELS = 64
STATE_SIZE = 64
PRODUCTION_OUTPUT_SIZE = 256
DETERMINISTIC_READER_SIZE = 256
EXPECTED_QWEN_IMAGE_GRID = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
}
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 5.0
GRADIENT_ACCUMULATION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bitwise reproducibility probe for the exact-64 lightweight updater")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(1, 100), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    if not marker.is_file() or not marker.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"Reader has no non-empty revision lock: {marker}")
    return marker.read_text(encoding="utf-8").strip()


def validate_qwen_image_grid_contract(image_processor: Any, *, image_size: int) -> dict[str, int]:
    actual: dict[str, int] = {}
    for field, expected in EXPECTED_QWEN_IMAGE_GRID.items():
        value = getattr(image_processor, field, None)
        if value is None:
            raise RuntimeError(f"Qwen image processor does not expose required {field}.")
        actual[field] = int(value)
        if actual[field] != expected:
            raise RuntimeError(f"Qwen image processor {field} drifted: expected {expected}, got {actual[field]}.")
    spatial_factor = actual["patch_size"] * actual["merge_size"]
    if image_size % spatial_factor:
        raise RuntimeError(
            "Deterministic reader size must be divisible by patch_size * merge_size; "
            f"got image_size={image_size}, spatial_factor={spatial_factor}."
        )
    return {**actual, "spatial_factor": spatial_factor}


def git_value(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def float_hex(value: float | torch.Tensor) -> str:
    number = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"Canonical trace encountered a non-finite scalar: {number}")
    return number.hex()


def episode_schedule(count: int, steps: int) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    epoch = 0
    while len(schedule) < steps:
        order = list(range(count))
        random.Random((SEED << 16) ^ epoch).shuffle(order)
        schedule.extend((epoch, index) for index in order)
        epoch += 1
    return schedule[:steps]


def gradient_manifest(model: torch.nn.Module) -> dict[str, Any]:
    missing = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
    ]
    if missing:
        raise RuntimeError(f"Trainable tensors have no gradient: {missing[:8]}")
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite:
        raise RuntimeError(f"Trainable tensors have non-finite gradients: {nonfinite[:8]}")
    return named_tensors_manifest(
        (name, parameter.grad) for name, parameter in model.named_parameters() if parameter.requires_grad
    )


def module_gradient_norms(model: LightweightVisualUpdater) -> dict[str, str]:
    modules = {
        "event_encoder": model.event_encoder,
        "event_projection": model.event_projection,
        "event_spatial_projection": model.event_spatial_projection,
        "film": model.film,
        "cell": model.cell,
        "rgb_head": model.rgb_head,
    }
    result: dict[str, str] = {}
    for name, module in modules.items():
        gradients = [
            parameter.grad.detach().float()
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError(f"Module {name!r} has no gradients.")
        norm = torch.sqrt(torch.stack([gradient.square().sum() for gradient in gradients]).sum())
        result[name] = float_hex(norm)
    return result


def runtime_metadata(device: torch.device, determinism: dict[str, Any]) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "-L"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    nvidia_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return {
        "pid": os.getpid(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": importlib.metadata.version("transformers"),
        "gpu": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "uuid": str(getattr(properties, "uuid", "")) or None,
        },
        "nvidia_smi_L": nvidia_smi.stdout.strip() if nvidia_smi.returncode == 0 else None,
        "nvidia_smi_inventory": nvidia_query.stdout.strip() if nvidia_query.returncode == 0 else None,
        "determinism": determinism,
    }


def evaluate_canonical_predictions(
    *,
    episodes: list[Any],
    updater: LightweightVisualUpdater,
    reader: Any,
    processor: Any,
    device: torch.device,
) -> list[dict[str, Any]]:
    updater.eval()
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for episode in episodes:
            state = updater.initial_state(batch_size=1, device=device, dtype=torch.float32)
            query_ordinal = 0
            for turn_id, turn in enumerate(episode_value(episode, "turns")):
                kind = turn_kind(turn)
                if kind in {"event", "mixed"}:
                    event_text, _event_kind = event_payload(turn)
                    state = updater.update(state, event_text)
                if kind in {"query", "mixed"}:
                    query, choices, target_index, comparison_id = query_payload(turn)
                    image = updater.render_deterministic_repro(state, target_size=DETERMINISTIC_READER_SIZE)[0]
                    score = qwen3vl_choice_nll(
                        model=reader,
                        processor=processor,
                        image=image,
                        query=format_mcq_query(query, choices),
                        choices=choices,
                        device=device,
                        do_resize=False,
                        deterministic_ce=True,
                    )
                    predictions.append(
                        {
                            "episode_id": str(episode_value(episode, "episode_id")),
                            "turn_id": turn_id,
                            "query_ordinal": query_ordinal,
                            "comparison_id": comparison_id,
                            "target_index": target_index,
                            "predicted_index": score.predicted_index,
                            "correct": score.predicted_index == target_index,
                            "choice_mean_nll_float_hex": [float(value).hex() for value in score.mean_nll],
                        }
                    )
                    query_ordinal += 1
    updater.train()
    return predictions


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The bitwise lightweight reproducibility probe requires CUDA.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("--device must select CUDA.")
    git_commit = git_value("rev-parse", "HEAD")
    git_status = git_value("status", "--porcelain=v1", "--untracked-files=all")
    if git_commit is None or git_status is None:
        raise RuntimeError("The reproducibility probe requires an inspectable Git worktree.")
    if git_status:
        raise RuntimeError("The reproducibility probe refuses a dirty Git worktree.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("The reproducibility probe refuses a non-empty --output-dir.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    determinism = configure_strict_cuda_determinism(SEED)
    episodes = list(read_jsonl(args.train))[:EPISODE_COUNT]
    validate_overfit_gate_episodes(episodes)
    subset = training_subset_audit(episodes)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=DETERMINISTIC_READER_SIZE * DETERMINISTIC_READER_SIZE,
        max_pixels=DETERMINISTIC_READER_SIZE * DETERMINISTIC_READER_SIZE,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("The deterministic probe requires a tensor-native fast Qwen processor.")
    qwen_image_grid = validate_qwen_image_grid_contract(
        processor.image_processor,
        image_size=DETERMINISTIC_READER_SIZE,
    )
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False

    updater = LightweightVisualUpdater(
        state_channels=STATE_CHANNELS,
        state_size=STATE_SIZE,
        output_size=PRODUCTION_OUTPUT_SIZE,
        learned_initial_state=False,
    ).to(device=device, dtype=torch.float32)
    trainable = [parameter for parameter in updater.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        foreach=False,
    )
    initial = model_optimizer_rng_manifest(updater, optimizer)

    def update_fn(state: torch.Tensor, event_text: str, _episode_id: str, _turn_id: str | int) -> torch.Tensor:
        return updater.update(state, event_text)

    def reader_loss(image: torch.Tensor, query: str, target: str):
        return qwen3vl_target_only_ce(
            model=reader,
            processor=processor,
            image=image[0],
            query=query,
            target=target,
            device=device,
            require_image_grad=True,
            do_resize=False,
            deterministic_ce=True,
        )

    trace: list[dict[str, Any]] = []
    milestones: dict[str, Any] = {}
    step_one_gradients: dict[str, Any] | None = None
    milestone_steps = {1, 2, 10, args.steps}
    torch.cuda.reset_peak_memory_stats(device)
    for optimizer_step, (epoch, episode_index) in enumerate(
        episode_schedule(len(episodes), args.steps),
        start=1,
    ):
        episode = episodes[episode_index]
        optimizer.zero_grad(set_to_none=True)
        result = run_episode(
            episode=episode,
            initial_state=updater.initial_state(batch_size=1, device=device, dtype=torch.float32),
            update_fn=update_fn,
            decode_fn=lambda state: updater.render_deterministic_repro(state, target_size=DETERMINISTIC_READER_SIZE),
            reader_loss_fn=reader_loss,
            noop_policy="update",
            collect_states=False,
        )
        if not torch.isfinite(result.loss):
            raise RuntimeError(f"Non-finite loss at optimizer step {optimizer_step}.")
        result.loss.backward()
        assert_no_frozen_parameter_grads(reader, "Qwen Reader")
        raw_gradients = gradient_manifest(updater)
        raw_module_gradient_norms = module_gradient_norms(updater)
        norm_before_clip = torch.nn.utils.clip_grad_norm_(
            trainable,
            GRADIENT_CLIP,
            error_if_nonfinite=True,
            foreach=False,
        )
        clipped_gradients = gradient_manifest(updater)
        clipped_module_gradient_norms = module_gradient_norms(updater)
        clipping_factor = min(1.0, GRADIENT_CLIP / (float(norm_before_clip.item()) + 1e-6))
        if optimizer_step == 1:
            step_one_gradients = {
                "raw": raw_gradients,
                "clipped": clipped_gradients,
            }
        optimizer.step()
        trace.append(
            {
                "optimizer_step": optimizer_step,
                "epoch": epoch,
                "episode_index": episode_index,
                "episode_id": str(episode_value(episode, "episode_id")),
                "loss_tensor_sha256": canonical_tensor_sha256(result.loss.detach()),
                "loss_float_hex": float_hex(result.loss),
                "gradient_norm_before_clip_float_hex": float_hex(norm_before_clip),
                "gradient_clipping_factor_float_hex": clipping_factor.hex(),
                "raw_gradient_bundle_sha256": raw_gradients["bundle_sha256"],
                "clipped_gradient_bundle_sha256": clipped_gradients["bundle_sha256"],
                "raw_module_gradient_norms_float_hex": raw_module_gradient_norms,
                "clipped_module_gradient_norms_float_hex": clipped_module_gradient_norms,
            }
        )
        if optimizer_step in milestone_steps:
            milestones[str(optimizer_step)] = model_optimizer_rng_manifest(updater, optimizer)

    if step_one_gradients is None:
        raise RuntimeError("The probe did not execute optimizer step 1.")

    predictions = evaluate_canonical_predictions(
        episodes=episodes,
        updater=updater,
        reader=reader,
        processor=processor,
        device=device,
    )
    assert_no_frozen_parameter_grads(reader, "Qwen Reader")
    trace_path = args.output_dir / "canonical_trace.jsonl"
    predictions_path = args.output_dir / "canonical_predictions.jsonl"
    write_jsonl(trace_path, trace)
    write_jsonl(predictions_path, predictions)
    runtime = runtime_metadata(device, determinism)

    protocol = {
        "schema_version": "vision_memory.lightweight_determinism_protocol.v2",
        "episode_count": EPISODE_COUNT,
        "seed": SEED,
        "steps": args.steps,
        "state_channels": STATE_CHANNELS,
        "state_size": STATE_SIZE,
        "production_output_size": PRODUCTION_OUTPUT_SIZE,
        "deterministic_reader_size": DETERMINISTIC_READER_SIZE,
        "renderer": "integer-repeat-without-crop",
        "qwen_do_resize": False,
        "qwen_image_grid": qwen_image_grid,
        "reader_ce": "fp32-logsumexp-minus-target-score",
        "attention": "sdpa-math-only",
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "learning_rate_float_hex": LEARNING_RATE.hex(),
        "weight_decay_float_hex": WEIGHT_DECAY.hex(),
        "gradient_clip_float_hex": GRADIENT_CLIP.hex(),
        "optimizer": "AdamW(foreach=False)",
        "dtype": str(dtype).removeprefix("torch."),
        "determinism": determinism,
    }
    comparison_payload = {
        "protocol": protocol,
        "git": {"commit": git_commit, "clean": True},
        "runtime_fingerprint": {
            key: runtime[key]
            for key in (
                "hostname",
                "slurm_job_id",
                "cuda_visible_devices",
                "python",
                "torch",
                "cuda_runtime",
                "cudnn",
                "transformers",
                "gpu",
                "nvidia_smi_L",
                "nvidia_smi_inventory",
            )
        },
        "train_sha256": sha256_file(args.train),
        "train_subset": subset,
        "reader_revision": locked_revision(args.reader),
        "initial": initial,
        "step_one_gradients": step_one_gradients,
        "trace": trace,
        "trace_sha256": canonical_object_sha256(trace),
        "trace_file_sha256": sha256_file(trace_path),
        "milestones": milestones,
        "final_predictions_sha256": canonical_object_sha256(predictions),
        "final_predictions_file_sha256": sha256_file(predictions_path),
        "final_prediction_count": len(predictions),
        "final_correct": sum(int(record["correct"]) for record in predictions),
    }
    report = {
        "schema_version": "vision_memory.lightweight_determinism_report.v1",
        "status": "complete",
        "comparison_payload_sha256": canonical_object_sha256(comparison_payload),
        "comparison_payload": comparison_payload,
        "runtime": runtime,
        "provenance": {
            "git_commit": git_commit,
            "git_clean": True,
            "train": str(args.train.resolve()),
            "reader": str(args.reader.resolve()),
            "output_dir": str(args.output_dir.resolve()),
        },
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    return report


def main() -> int:
    args = parse_args()
    refused_preexisting_output = args.output_dir.exists() and any(args.output_dir.iterdir())
    try:
        report = run_probe(args)
    except Exception as error:
        if not refused_preexisting_output:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": "vision_memory.lightweight_determinism_report.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            write_json(args.output_dir / "report.json", failure)
        raise
    write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
