from __future__ import annotations

import argparse
import hashlib
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    if not marker.is_file():
        raise ValueError(f"Reader snapshot is missing required revision marker: {marker}")
    revision = marker.read_text(encoding="utf-8").strip()
    if not revision:
        raise ValueError(f"Reader snapshot has an empty revision marker: {marker}")
    return revision


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
        last_transition = "unknown"
        previous_target_text: str | None = None
        event_count_since_query = 0
        noop_count_since_query = 0
        for turn in episode.turns:
            if turn.calls_updater:
                history.append(turn.event_text)
                last_transition = turn.event_kind.value
                event_count_since_query += 1
                noop_count_since_query += int(turn.event_kind.value == "noop")
            if turn.calls_reader:
                query = turn.query
                target_text = query.choices[query.target_index]
                stale_index = None
                if previous_target_text is not None and previous_target_text != target_text:
                    try:
                        stale_index = tuple(query.choices).index(previous_target_text)
                    except ValueError:
                        pass
                metadata = {
                    "episode_id": episode.episode_id,
                    "query_id": f"{episode.episode_id}:q{query_number}",
                    "pair_id": episode.pair_id,
                    "counterfactual_pair_id": episode.pair_id,
                    "semantic_counterfactual_pair_id": episode.pair_id,
                    "counterfactual_episode_id": episode.counterfactual_episode_id,
                    "distractor_pair_id": episode.distractor_pair_id,
                    "distractor_episode_id": episode.distractor_episode_id,
                    "distractor_variant": (
                        episode.distractor_variant.value
                        if episode.distractor_variant is not None
                        else None
                    ),
                    "query_comparison_id": query.comparison_id,
                    "topic": episode.topic,
                    "subtype": last_transition,
                    "form": last_transition,
                    "split": episode.split,
                    "ood_group": episode.ood_group,
                    "protocol": "synthetic",
                    "route": "event_then_query" if turn.calls_updater else "query_read_only",
                    "query_turn_type": turn.type.value,
                    "event_latency_seconds": 0.0,
                    "updater_calls_since_query": event_count_since_query,
                    "noop_events_since_query": noop_count_since_query,
                    "noop_events_applied_since_query": 0,
                    "noop_policy": "keep",
                }
                if previous_target_text is not None and previous_target_text != target_text:
                    metadata["stale_target_text"] = previous_target_text
                    metadata["stale_target_mapped"] = stale_index is not None
                    if stale_index is not None:
                        metadata["stale_target_index"] = stale_index
                yield {
                    "metadata": metadata,
                    "query": query.text,
                    "choices": tuple(query.choices),
                    "target_index": query.target_index,
                    "history": tuple(history),
                }
                previous_target_text = target_text
                query_number += 1
                event_count_since_query = 0
                noop_count_since_query = 0


def prefeval_queries(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    records = raw_jsonl(path)
    if limit is not None:
        records = records[:limit]
    for record in records:
        model_input = record["model_input"]
        history: list[str] = []
        query_number = 0
        event_count_since_query = 0
        noop_count_since_query = 0
        for turn in model_input["turns"]:
            if turn["type"] == "event":
                history.append(turn["text"])
                event_count_since_query += 1
                noop_count_since_query += int(turn.get("event_type") == "noop")
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
                        "route": "query_read_only",
                        "query_turn_type": "query",
                        "event_latency_seconds": 0.0,
                        "updater_calls_since_query": event_count_since_query,
                        "noop_events_since_query": noop_count_since_query,
                        "noop_events_applied_since_query": 0,
                        "noop_policy": "keep",
                    },
                    "query": turn["text"],
                    "choices": tuple(turn["options"]),
                    "target_index": int(record["label"]["target_index"]),
                    "history": tuple(history),
                }
                query_number += 1
                event_count_since_query = 0
                noop_count_since_query = 0


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

    if args.format == "prefeval" and "oracle_target" in args.methods:
        raise SystemExit(
            "oracle_target injects the label and is prohibited for PrefEval model inputs; "
            "use only query_only/full_history controls for the external benchmark."
        )
    if not torch.cuda.is_available():
        raise SystemExit("Qwen baselines require CUDA.")
    reader_revision = locked_revision(args.reader)
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
    torch.cuda.reset_peak_memory_stats(device)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for method in args.methods:
            for item in items:
                torch.cuda.synchronize(device)
                started = time.monotonic()
                result = qwen3vl_choice_nll(
                    model=reader,
                    processor=processor,
                    image=blank_image,
                    query=method_prompt(method, item),
                    choices=item["choices"],
                    device=device,
                )
                torch.cuda.synchronize(device)
                query_latency = time.monotonic() - started
                record = {
                    **item["metadata"],
                    "method": method,
                    "seed": args.seed,
                    "diffusion_seed": 0,
                    "recurrence_mode": "direct_latent",
                    "condition": "standard",
                    "prediction_index": result.predicted_index,
                    "target_index": item["target_index"],
                    "choice_mean_nll": list(result.mean_nll),
                    "query_latency_seconds": query_latency,
                    "latency_seconds": query_latency,
                    "state_bytes": int(blank_image.numel() * blank_image.element_size()),
                    "peak_reader_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                }
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "episodes": str(args.episodes.resolve()),
        "episodes_sha256": sha256_file(args.episodes),
        "queries": len(items),
        "prediction_records": len(items) * len(args.methods),
        "methods": args.methods,
        "seed": args.seed,
        "reader_revision": reader_revision,
        "blank_image": {"shape": [3, 256, 256], "float_value": 0.5},
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
