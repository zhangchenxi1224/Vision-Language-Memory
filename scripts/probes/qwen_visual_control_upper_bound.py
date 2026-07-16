from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import Episode, read_jsonl  # noqa: E402
from vision_memory.dreamlite import assert_no_frozen_parameter_grads, freeze_module  # noqa: E402
from vision_memory.reader import qwen3vl_choice_nll, qwen3vl_target_only_ce  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402

from scripts.data.qwen_sanity import (  # noqa: E402
    UniqueQuery,
    collect_unique_queries,
    locked_revision,
    sha256_file,
    validate_query_inventory,
)
from scripts.train.lightweight_episode import (  # noqa: E402
    FORMAL_OVERFIT_EPISODES,
    training_subset_audit,
    validate_overfit_gate_episodes,
)


CONTROL_CODE_COUNT = 4
IMAGE_SHAPE = (3, 256, 256)
DISCLAIMER = (
    "TARGET-SUPERVISED DIAGNOSTIC ONLY: target_index selects one of four learned images. "
    "This is not a memory updater, baseline, ablation, or publishable method result."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic upper bound for target-selected visual control of frozen Qwen"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_exact_gate_subset(dataset: Path) -> tuple[list[Episode], dict[str, Any]]:
    episodes = list(read_jsonl(dataset))[:FORMAL_OVERFIT_EPISODES]
    validate_overfit_gate_episodes(episodes)
    audit = training_subset_audit(episodes)
    return episodes, audit


def deterministic_query_order(query_count: int, *, steps: int, seed: int) -> list[int]:
    if query_count < 1 or steps < 1:
        raise ValueError("query_count and steps must be positive")
    order: list[int] = []
    epoch = 0
    while len(order) < steps:
        indices = list(range(query_count))
        random.Random((seed << 16) ^ epoch).shuffle(indices)
        order.extend(indices)
        epoch += 1
    return order[:steps]


def query_pattern(item: UniqueQuery) -> str:
    marker = "-pattern-"
    if marker not in item.template_id:
        return "unknown"
    suffix = item.template_id.split(marker, maxsplit=1)[1]
    number = suffix.split("-", maxsplit=1)[0]
    return f"pattern_{number}"


def grouped_accuracy(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        counts = groups[str(record[key])]
        counts[0] += int(bool(record["correct"]))
        counts[1] += 1
    return {
        name: {"correct": counts[0], "count": counts[1], "accuracy": counts[0] / counts[1]}
        for name, counts in sorted(groups.items())
    }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.learning_rate <= 0:
        raise SystemExit("--steps and --learning-rate must be positive")
    if not 0.0 < args.threshold <= 1.0:
        raise SystemExit("--threshold must be in (0, 1]")
    if not torch.cuda.is_available():
        raise SystemExit("The visual-control diagnostic requires CUDA and the real frozen Qwen Reader.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("The visual-control diagnostic refuses a non-empty --output-dir.")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit("The visual-control diagnostic refuses a dirty worktree.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must select CUDA")
    set_seeds(args.seed)
    episodes, subset_audit = select_exact_gate_subset(args.dataset)
    raw_query_count, queries = collect_unique_queries(episodes)
    validate_query_inventory(
        raw_query_count,
        queries,
        expected_raw_queries=128,
        expected_comparison_queries=64,
        expected_target_position_count=16,
    )
    args.output_dir.mkdir(parents=True)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=IMAGE_SHAPE[1] * IMAGE_SHAPE[2],
        max_pixels=IMAGE_SHAPE[1] * IMAGE_SHAPE[2],
    )
    if "Fast" not in type(processor.image_processor).__name__:
        raise RuntimeError("A tensor-native fast Qwen image processor is required.")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False

    image_logits = torch.nn.Parameter(torch.zeros(CONTROL_CODE_COUNT, *IMAGE_SHAPE, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([image_logits], lr=args.learning_rate)
    query_order = deterministic_query_order(len(queries), steps=args.steps, seed=args.seed)
    metrics_path = args.output_dir / "metrics.jsonl"
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for optimizer_step, query_index in enumerate(query_order, start=1):
        item = queries[query_index]
        query = item.query
        optimizer.zero_grad(set_to_none=True)
        image = torch.sigmoid(image_logits[query.target_index])
        output = qwen3vl_target_only_ce(
            model=reader,
            processor=processor,
            image=image,
            query=format_mcq_query(query.text, query.choices),
            target=query.target,
            device=device,
            require_image_grad=True,
        )
        output.loss.backward()
        assert_no_frozen_parameter_grads(reader, "Qwen Reader")
        gradient_norm = image_logits.grad.norm() if image_logits.grad is not None else None
        if gradient_norm is None or not torch.isfinite(gradient_norm) or gradient_norm.item() <= 0:
            raise RuntimeError(f"Invalid visual-control gradient at step {optimizer_step}: {gradient_norm}")
        optimizer.step()
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "optimizer_step": optimizer_step,
                        "comparison_id": item.comparison_id,
                        "target_index": query.target_index,
                        "loss": float(output.loss.item()),
                        "gradient_norm": float(gradient_norm.item()),
                    }
                )
                + "\n"
            )

    records: list[dict[str, Any]] = []
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in queries:
            query = item.query
            score = qwen3vl_choice_nll(
                model=reader,
                processor=processor,
                image=torch.sigmoid(image_logits[query.target_index]).detach(),
                query=format_mcq_query(query.text, query.choices),
                choices=query.choices,
                device=device,
            )
            record = {
                "comparison_id": item.comparison_id,
                "member_episode_ids": sorted(member.episode_id for member in item.members),
                "topic": item.topic,
                "template_id": item.template_id,
                "pattern": query_pattern(item),
                "target_index": query.target_index,
                "target_text": query.target,
                "predicted_index": score.predicted_index,
                "choice_mean_nll": list(score.mean_nll),
                "correct": score.predicted_index == query.target_index,
                "diagnostic_disclaimer": DISCLAIMER,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    code_payload = {
        "schema_version": "vision_memory.target_supervised_visual_control.v1",
        "diagnostic_disclaimer": DISCLAIMER,
        "target_index_selects_code": True,
        "image_logits": image_logits.detach().cpu(),
    }
    codes_path = args.output_dir / "target_selected_control_codes.pt"
    torch.save(code_payload, codes_path)
    accuracy = sum(int(record["correct"]) for record in records) / len(records)
    target_positions = Counter(record["target_index"] for record in records)
    summary = {
        "schema_version": "vision_memory.target_supervised_visual_control_summary.v1",
        "diagnostic_disclaimer": DISCLAIMER,
        "passed_threshold": accuracy >= args.threshold,
        "threshold": args.threshold,
        "accuracy": accuracy,
        "accuracy_by_pattern": grouped_accuracy(records, "pattern"),
        "accuracy_by_target_position": grouped_accuracy(records, "target_index"),
        "target_position_counts": dict(sorted(target_positions.items())),
        "raw_query_count": raw_query_count,
        "comparison_query_count": len(queries),
        "train_subset": subset_audit,
        "dataset_sha256": sha256_file(args.dataset),
        "reader_revision": locked_revision(args.reader),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "dtype": str(dtype).removeprefix("torch."),
        },
        "artifacts": {
            "metrics": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "predictions": str(predictions_path.resolve()),
            "predictions_sha256": sha256_file(predictions_path),
            "control_codes": str(codes_path.resolve()),
            "control_codes_sha256": sha256_file(codes_path),
        },
        "control_code_tensor_sha256": sha256_bytes(image_logits.detach().cpu().contiguous().numpy().tobytes()),
        "frozen_reader_parameter_grad_count": sum(parameter.grad is not None for parameter in reader.parameters()),
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
