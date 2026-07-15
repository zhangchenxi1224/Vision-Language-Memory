from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import assert_no_frozen_parameter_grads, freeze_module  # noqa: E402
from vision_memory.data import read_jsonl as read_episode_jsonl  # noqa: E402
from vision_memory.reader import qwen3vl_target_only_ce  # noqa: E402
from vision_memory.repro import load_source_image  # noqa: E402
from vision_memory.training import (  # noqa: E402
    DreamLiteEpisodeModel,
    load_training_checkpoint,
    run_episode,
    save_training_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Episode-level DreamLite latent-memory training")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, help="Omit to use the locked deterministic 1024 RGB fixture")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0, help="Episode order and per-event noise seed")
    parser.add_argument("--adapter-seed", type=int, default=0, help="LoRA initialization seed")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--max-train-episodes", type=int, default=None)
    parser.add_argument("--recurrence-mode", choices=["direct_latent", "decode_reencode"], default="direct_latent")
    parser.add_argument("--detach-between-events", action="store_true")
    parser.add_argument("--learn-initial-state", action="store_true")
    parser.add_argument("--checkpoint-unet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.strip()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else "unknown"


def compute_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        raise ValueError("Real DreamLite episode training requires CUDA devices.")
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}


def make_manifest(args: argparse.Namespace) -> dict[str, Any]:
    commit = git_value("rev-parse", "HEAD")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise RuntimeError("Formal training refuses a dirty worktree. Commit or pass --allow-dirty for debugging only.")
    compatibility_args = serializable_args(args)
    for excluded in ("resume", "output_dir", "allow_dirty"):
        compatibility_args.pop(excluded, None)
    return {
        "schema_version": 1,
        "git_commit": commit,
        "git_dirty": bool(status),
        "dreamlite_revision": locked_revision(args.dreamlite),
        "reader_revision": locked_revision(args.reader),
        "train_sha256": sha256_file(args.train),
        "dev_sha256": sha256_file(args.dev),
        "source_image": load_source_image(args.source_image, resolution=args.resolution)[1],
        "arguments": compatibility_args,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "diffusers": importlib.metadata.version("diffusers"),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
        },
    }


def assert_frozen_contract(pipe: Any, reader: torch.nn.Module) -> None:
    assert_no_frozen_parameter_grads(pipe.vae, "VAE")
    assert_no_frozen_parameter_grads(pipe.text_encoder, "DreamLite conditioner")
    assert_no_frozen_parameter_grads(pipe.unet, "DreamLite base U-Net")
    assert_no_frozen_parameter_grads(reader, "Qwen Reader")


def reader_callable(*, reader: Any, processor: Any, reader_device: torch.device, require_grad: bool):
    def call(image: torch.Tensor, query: str, target: str):
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise ValueError("Reader currently supports one state image per episode.")
            image = image[0]
        return qwen3vl_target_only_ce(
            model=reader,
            processor=processor,
            image=image.to(reader_device),
            query=query,
            target=target,
            device=reader_device,
            require_image_grad=require_grad,
        )

    return call


def episode_order(records: list[dict[str, Any]], seed: int, epoch: int) -> list[int]:
    order = list(range(len(records)))
    random.Random((seed << 16) ^ epoch).shuffle(order)
    return order


def evaluate_dev(
    *,
    model: DreamLiteEpisodeModel,
    records: Iterable[dict[str, Any]],
    recurrence_mode: str,
    detach_between_events: bool,
    reader_loss_fn,
) -> float:
    losses: list[float] = []
    with torch.no_grad():
        for episode in records:
            result = run_episode(
                episode=episode,
                initial_state=model.reset_state(),
                update_fn=model.updater,
                decode_fn=model.updater.decode_for_reader,
                reencode_fn=model.updater.reencode_posterior_mean,
                reencode_decode_fn=model.updater.decode_for_reencode,
                reader_loss_fn=reader_loss_fn,
                recurrence_mode=recurrence_mode,
                detach_between_events=detach_between_events,
                collect_states=False,
            )
            losses.append(float(result.loss.item()))
    return sum(losses) / len(losses)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def truncate_metrics_for_resume(path: Path, *, optimizer_step: int) -> float:
    if not path.exists():
        return 0.0
    kept: list[dict[str, Any]] = []
    prior_elapsed = 0.0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            step = value.get("optimizer_step")
            if not isinstance(step, int):
                raise ValueError(f"Invalid optimizer_step in {path}:{line_number}")
            if step <= optimizer_step:
                kept.append(value)
                if value.get("elapsed_seconds") is not None:
                    prior_elapsed = max(prior_elapsed, float(value["elapsed_seconds"]))
    temporary = path.with_suffix(path.suffix + ".resume.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in kept:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.write(json.dumps({"kind": "resume", "optimizer_step": optimizer_step}) + "\n")
    temporary.replace(path)
    return prior_elapsed


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("DreamLite episode training requires two visible CUDA GPUs.")
    positive_values = {
        "epochs": args.epochs,
        "gradient_accumulation": args.gradient_accumulation,
        "gradient_clip": args.gradient_clip,
        "checkpoint_every": args.checkpoint_every,
        "eval_every": args.eval_every,
        "eval_limit": args.eval_limit,
        "early_stopping_patience": args.early_stopping_patience,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0}
    if invalid:
        raise SystemExit(f"Training arguments must be positive: {invalid}")
    if args.max_train_episodes is not None and args.max_train_episodes <= 0:
        raise SystemExit("--max-train-episodes must be positive when supplied.")
    if torch.device(args.dreamlite_device) == torch.device(args.reader_device):
        raise SystemExit("DreamLite and Reader must use distinct CUDA devices for the formal run.")

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise SystemExit("A fresh run refuses a non-empty --output-dir; use --resume or choose a new directory.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = make_manifest(args)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    installed = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    (args.output_dir / "environment.txt").write_text("\n".join(installed) + "\n", encoding="utf-8")

    train_records = read_episode_jsonl(args.train)
    dev_records = read_episode_jsonl(args.dev)[: args.eval_limit]
    if args.max_train_episodes is not None:
        train_records = train_records[: args.max_train_episodes]
    if not train_records or not dev_records:
        raise SystemExit("Train and dev datasets must remain non-empty after limits are applied.")

    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    updater_dtype = compute_dtype(updater_device)
    reader_dtype = compute_dtype(reader_device)
    set_all_seeds(args.seed)

    from diffusers import DreamLiteMobilePipeline
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    pipe = DreamLiteMobilePipeline.from_pretrained(
        args.dreamlite,
        local_files_only=True,
        torch_dtype=updater_dtype,
    ).to(updater_device)
    freeze_module(pipe.vae)
    freeze_module(pipe.text_encoder)
    pipe.unet.requires_grad_(False)
    set_all_seeds(args.adapter_seed)
    pipe.unet = get_peft_model(
        pipe.unet,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            lora_dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
    )
    pipe.unet.eval()

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("A fast tensor-native Qwen image processor is required.")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=reader_dtype,
        attn_implementation="sdpa",
    ).to(reader_device)
    freeze_module(reader)
    reader.config.use_cache = False

    source_pil, _source_metadata = load_source_image(args.source_image, resolution=args.resolution)
    source_image = pipe.image_processor.preprocess(source_pil, height=args.resolution, width=args.resolution)
    with torch.no_grad():
        initial_state = pipe.prepare_image_latents(source_image, dtype=updater_dtype, device=updater_device)
    model = DreamLiteEpisodeModel(
        pipeline=pipe,
        initial_state=initial_state,
        global_seed=args.seed,
        checkpoint_unet=args.checkpoint_unet,
        learn_initial_state=args.learn_initial_state,
    )
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    allowed_trainable = {
        name
        for name in trainable_names
        if ".lora_A." in name or ".lora_B." in name or (args.learn_initial_state and name == "initial_state")
    }
    unexpected_trainable = sorted(trainable_names - allowed_trainable)
    if unexpected_trainable:
        raise RuntimeError(f"Unexpected trainable parameters outside the LoRA/initial-state whitelist: {unexpected_trainable}")
    trainable = [parameter for name, parameter in model.named_parameters() if name in allowed_trainable]
    if not trainable:
        raise RuntimeError("No trainable LoRA/initial-state parameters were found.")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    start_epoch = 0
    start_cursor = 0
    optimizer_step = 0
    best_dev = float("inf")
    stale_evals = 0
    if args.resume:
        payload = load_training_checkpoint(
            args.resume,
            trainable_module=model,
            optimizer=optimizer,
            expected_manifest=manifest,
        )
        start_epoch = int(payload["epoch"])
        start_cursor = int(payload["episode_cursor"])
        optimizer_step = int(payload["optimizer_step"])
        saved_trainer_state = payload.get("trainer_state", {})
        best_dev = float(saved_trainer_state.get("best_dev", float("inf")))
        stale_evals = int(saved_trainer_state.get("stale_evals", 0))

    train_reader = reader_callable(reader=reader, processor=processor, reader_device=reader_device, require_grad=True)
    eval_reader = reader_callable(reader=reader, processor=processor, reader_device=reader_device, require_grad=False)
    metrics_path = args.output_dir / "metrics.jsonl"
    prior_elapsed = truncate_metrics_for_resume(metrics_path, optimizer_step=optimizer_step) if args.resume else 0.0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(updater_device)
    torch.cuda.reset_peak_memory_stats(reader_device)

    final_epoch = start_epoch
    final_cursor = start_cursor
    epoch_iterator = range(start_epoch, args.epochs) if stale_evals < args.early_stopping_patience else ()
    for epoch in epoch_iterator:
        final_epoch = epoch
        order = episode_order(train_records, args.seed, epoch)
        cursor = start_cursor if epoch == start_epoch else 0
        accumulation_count = 0
        for position in range(cursor, len(order)):
            final_cursor = position + 1
            episode = train_records[order[position]]
            result = run_episode(
                episode=episode,
                initial_state=model.reset_state(),
                update_fn=model.updater,
                decode_fn=model.updater.decode_for_reader,
                reencode_fn=model.updater.reencode_posterior_mean,
                reencode_decode_fn=model.updater.decode_for_reencode,
                reader_loss_fn=train_reader,
                recurrence_mode=args.recurrence_mode,
                detach_between_events=args.detach_between_events,
                collect_states=False,
            )
            (result.loss / args.gradient_accumulation).backward()
            assert_frozen_contract(pipe, reader)
            accumulation_count += 1

            is_last = position + 1 == len(order)
            if accumulation_count == args.gradient_accumulation or is_last:
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
                if not torch.isfinite(gradient_norm) or gradient_norm.item() <= 0:
                    raise RuntimeError(f"Invalid trainable gradient norm: {gradient_norm.item()}")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                accumulation_count = 0
                append_jsonl(
                    metrics_path,
                    {
                        "kind": "train",
                        "epoch": epoch,
                        "episode_cursor": position + 1,
                        "optimizer_step": optimizer_step,
                        "loss": float(result.loss.item()),
                        "gradient_norm": float(gradient_norm.item()),
                        "elapsed_seconds": prior_elapsed + time.monotonic() - started,
                    },
                )

                stop_after_step = False
                if optimizer_step % args.eval_every == 0:
                    dev_loss = evaluate_dev(
                        model=model,
                        records=dev_records,
                        recurrence_mode=args.recurrence_mode,
                        detach_between_events=args.detach_between_events,
                        reader_loss_fn=eval_reader,
                    )
                    append_jsonl(metrics_path, {"kind": "dev", "optimizer_step": optimizer_step, "loss": dev_loss})
                    if dev_loss < best_dev:
                        best_dev = dev_loss
                        stale_evals = 0
                        save_training_checkpoint(
                            args.output_dir / "best.pt",
                            trainable_module=model,
                            optimizer=optimizer,
                            epoch=epoch,
                            episode_cursor=position + 1,
                            optimizer_step=optimizer_step,
                            manifest=manifest,
                            trainer_state={"best_dev": best_dev, "stale_evals": stale_evals},
                        )
                    else:
                        stale_evals += 1
                    stop_after_step = stale_evals >= args.early_stopping_patience

                if optimizer_step % args.checkpoint_every == 0:
                    save_training_checkpoint(
                        args.output_dir / f"checkpoint-{optimizer_step:06d}.pt",
                        trainable_module=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        episode_cursor=position + 1,
                        optimizer_step=optimizer_step,
                        manifest=manifest,
                        trainer_state={"best_dev": best_dev, "stale_evals": stale_evals},
                    )
                if stop_after_step:
                    break
        start_cursor = 0
        if stale_evals >= args.early_stopping_patience:
            break
        final_cursor = len(order)

    save_training_checkpoint(
        args.output_dir / "last.pt",
        trainable_module=model,
        optimizer=optimizer,
        epoch=final_epoch,
        episode_cursor=final_cursor,
        optimizer_step=optimizer_step,
        manifest=manifest,
        trainer_state={"best_dev": best_dev, "stale_evals": stale_evals},
    )
    summary = {
        "optimizer_steps": optimizer_step,
        "best_dev_loss": None if best_dev == float("inf") else best_dev,
        "elapsed_seconds": prior_elapsed + time.monotonic() - started,
        "peak_vram_gib": {
            str(updater_device): torch.cuda.max_memory_allocated(updater_device) / 2**30,
            str(reader_device): torch.cuda.max_memory_allocated(reader_device) / 2**30,
        },
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
