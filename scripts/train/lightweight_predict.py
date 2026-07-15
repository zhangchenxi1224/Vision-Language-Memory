from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl as read_synthetic_jsonl  # noqa: E402
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.training import read_prefeval_adapted_jsonl, read_prefeval_supervised_jsonl  # noqa: E402

from lightweight_episode import build_model, evaluate_accuracy, locked_revision, set_seeds  # noqa: E402


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a formal lightweight/static-image checkpoint with Qwen")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("synthetic", "prefeval-export", "prefeval-supervised"),
        default="synthetic",
    )
    parser.add_argument(
        "--prefeval-split", choices=("adapt_train", "adapt_dev", "adapt_ood"), default="adapt_ood"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise SystemExit("Formal prediction requires a CUDA device and the real Qwen Reader.")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite predictions: {args.output}")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit("Formal prediction refuses a dirty worktree.")

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1:
        raise RuntimeError("Unsupported lightweight checkpoint schema.")
    manifest = payload.get("manifest")
    config = payload.get("model_config")
    if not isinstance(manifest, dict) or not isinstance(config, dict):
        raise RuntimeError("Checkpoint is missing manifest/model_config.")
    if manifest.get("git_commit") != git_value("rev-parse", "HEAD"):
        raise RuntimeError("Checkpoint git commit does not match the checked-out code.")
    if manifest.get("reader_revision") != locked_revision(args.reader):
        raise RuntimeError("Checkpoint Reader revision does not match --reader.")
    training_args = manifest.get("arguments")
    if not isinstance(training_args, dict):
        raise RuntimeError("Checkpoint manifest is missing training arguments.")
    seed = int(training_args["seed"])
    noop_policy = str(training_args["noop_policy"])
    set_seeds(seed)
    device = torch.device(args.device)
    model = build_model(SimpleNamespace(**config)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)

    if args.dataset_format == "synthetic":
        episodes = list(read_synthetic_jsonl(args.episodes))
    elif args.dataset_format == "prefeval-export":
        episodes = read_prefeval_adapted_jsonl(args.episodes, allowed_splits={args.prefeval_split})
    else:
        episodes = read_prefeval_supervised_jsonl(args.episodes, allowed_splits={args.prefeval_split})

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    output_size = int(config["output_size"])
    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=output_size * output_size,
        max_pixels=output_size * output_size,
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("A tensor-native fast Qwen image processor is required.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False
    torch.cuda.reset_peak_memory_stats(device)
    accuracy = evaluate_accuracy(
        episodes=episodes,
        model=model,
        reader=reader,
        processor=processor,
        device=device,
        noop_policy=noop_policy,
        predictions_path=args.output,
        method=str(config["method"]),
        seed=seed,
    )
    report = {
        "predictions": str(args.output.resolve()),
        "predictions_sha256": sha256_file(args.output),
        "episode_source": str(args.episodes.resolve()),
        "episode_source_sha256": sha256_file(args.episodes),
        "episodes": len(episodes),
        "accuracy": accuracy,
        "method": config["method"],
        "seed": seed,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_manifest": manifest,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
