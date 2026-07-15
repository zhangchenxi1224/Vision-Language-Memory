from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl as read_synthetic_jsonl  # noqa: E402
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import qwen3vl_choice_nll  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402


METHODS = ("query_only", "full_history", "full_history_reminder", "rag_top5", "oracle_target")


def raw_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def synthetic_queries(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    episodes = read_synthetic_jsonl(path)
    if limit is not None:
        episodes = episodes[:limit]
    for episode in episodes:
        history: list[str] = []
        query_number = 0
        for turn in episode.turns:
            if turn.calls_updater:
                history.append(turn.event_text)
            if turn.calls_reader:
                query = turn.query
                yield {
                    "metadata": {
                        "episode_id": episode.episode_id,
                        "query_id": f"{episode.episode_id}:q{query_number}",
                        "counterfactual_pair_id": episode.pair_id,
                        "topic": episode.topic,
                        "split": episode.split,
                        "ood_group": episode.ood_group,
                    },
                    "query": query.text,
                    "choices": tuple(query.choices),
                    "target_index": query.target_index,
                    "history": tuple(history),
                }
                query_number += 1


def prefeval_queries(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    records = raw_jsonl(path)
    if limit is not None:
        records = records[:limit]
    for record in records:
        model_input = record["model_input"]
        history: list[str] = []
        query_number = 0
        for turn in model_input["turns"]:
            if turn["type"] == "event":
                history.append(turn["text"])
            elif turn["type"] == "query":
                yield {
                    "metadata": {
                        "episode_id": model_input["sample_id"],
                        "query_id": f"{model_input['sample_id']}:q{query_number}",
                        "base_pair_id": model_input["base_pair_id"],
                        "topic": model_input["topic"],
                        "form": model_input["form"],
                        "split": model_input["split"],
                        "protocol": model_input["protocol"],
                        "forced_write_k": model_input["forced_write_k"],
                    },
                    "query": turn["text"],
                    "choices": tuple(turn["options"]),
                    "target_index": int(record["label"]["target_index"]),
                    "history": tuple(history),
                }
                query_number += 1


def method_prompt(method: str, item: dict[str, Any]) -> str:
    query = format_mcq_query(item["query"], item["choices"])
    if method == "query_only":
        return query
    if method == "oracle_target":
        target = item["choices"][item["target_index"]]
        return f"Current preference memory: {target}\n{query}"
    events = list(item["history"])
    if method == "rag_top5":
        events = events[-5:]
    history = "\n".join(f"- {event}" for event in events)
    reminder = "\nUse the user's remembered preference when choosing." if method == "full_history_reminder" else ""
    return f"Conversation memory:\n{history}{reminder}\n{query}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen-Qwen text-history controls with the same blank image")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--format", choices=("synthetic", "prefeval"), required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Qwen baselines require CUDA.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
    blank_image = torch.full((3, 256, 256), 0.5, device=device, dtype=torch.float32)
    items = list(synthetic_queries(args.episodes, args.limit) if args.format == "synthetic" else prefeval_queries(args.episodes, args.limit))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for method in args.methods:
            for item in items:
                started = time.monotonic()
                result = qwen3vl_choice_nll(
                    model=reader,
                    processor=processor,
                    image=blank_image,
                    query=method_prompt(method, item),
                    choices=item["choices"],
                    device=device,
                )
                record = {
                    **item["metadata"],
                    "method": method,
                    "seed": args.seed,
                    "condition": "standard",
                    "prediction_index": result.predicted_index,
                    "target_index": item["target_index"],
                    "choice_mean_nll": list(result.mean_nll),
                    "latency_seconds": time.monotonic() - started,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "queries": len(items), "methods": args.methods}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
