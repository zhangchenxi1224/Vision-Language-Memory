from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl  # noqa: E402
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import qwen3vl_choice_nll  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    if not marker.is_file():
        raise RuntimeError(f"Reader snapshot is missing required revision marker: {marker}")
    revision = marker.read_text(encoding="utf-8").strip()
    if not revision:
        raise RuntimeError(f"Reader snapshot has an empty revision marker: {marker}")
    return revision


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen-Qwen oracle-text and query-only synthetic-data gates")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--oracle-threshold", type=float, default=0.95)
    parser.add_argument("--query-only-ceiling", type=float, default=0.30)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Qwen data sanity requires CUDA.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    reader_revision = locked_revision(args.reader)
    episodes = read_jsonl(args.dataset)[: args.limit]
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=256 * 256,
        max_pixels=256 * 256,
    )
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)
    freeze_module(reader)
    reader.config.use_cache = False
    blank = torch.full((3, 256, 256), 0.5, device=device, dtype=torch.float32)

    oracle_correct = 0
    blank_correct = 0
    query_count = 0
    started = time.monotonic()
    for episode in episodes:
        for turn in episode.turns:
            if not turn.calls_reader:
                continue
            query = turn.query
            rendered = format_mcq_query(query.text, query.choices)
            blank_result = qwen3vl_choice_nll(
                model=reader,
                processor=processor,
                image=blank,
                query=rendered,
                choices=query.choices,
                device=device,
            )
            oracle_result = qwen3vl_choice_nll(
                model=reader,
                processor=processor,
                image=blank,
                query=f"Current preference memory: {query.target}\n{rendered}",
                choices=query.choices,
                device=device,
            )
            blank_correct += int(blank_result.predicted_index == query.target_index)
            oracle_correct += int(oracle_result.predicted_index == query.target_index)
            query_count += 1

    oracle_accuracy = oracle_correct / query_count
    query_only_accuracy = blank_correct / query_count
    passed = oracle_accuracy >= args.oracle_threshold and query_only_accuracy <= args.query_only_ceiling
    report = {
        "schema_version": 1,
        "episodes": len(episodes),
        "queries": query_count,
        "dataset_sha256": sha256_file(args.dataset),
        "reader_revision": reader_revision,
        "oracle_text_accuracy": oracle_accuracy,
        "oracle_threshold": args.oracle_threshold,
        "query_only_blank_accuracy": query_only_accuracy,
        "query_only_ceiling": args.query_only_ceiling,
        "passed": passed,
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
