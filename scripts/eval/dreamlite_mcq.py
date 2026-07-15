from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl as read_synthetic_jsonl  # noqa: E402
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import qwen3vl_choice_nll  # noqa: E402
from vision_memory.repro import load_source_image  # noqa: E402
from vision_memory.training import DreamLiteEpisodeModel, format_mcq_query, load_trainable_weights  # noqa: E402


@dataclass
class QueryState:
    metadata: dict[str, Any]
    query: str
    choices: tuple[str, ...]
    target_index: int
    state: torch.Tensor


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number} must contain a JSON object.")
                values.append(value)
    return values


def collect_synthetic(
    model: DreamLiteEpisodeModel,
    path: Path,
    limit: int | None,
    recurrence_mode: str,
) -> list[QueryState]:
    episodes = read_synthetic_jsonl(path)
    if limit is not None:
        episodes = episodes[:limit]
    queries: list[QueryState] = []
    with torch.no_grad():
        for episode in episodes:
            state = model.reset_state()
            query_number = 0
            last_transition = "unknown"
            previous_target = None
            for turn_index, turn in enumerate(episode.turns):
                if turn.calls_updater:
                    state = model.updater(state, turn.event_text, episode.episode_id, turn_index)
                    if recurrence_mode == "decode_reencode":
                        state = model.updater.reencode_posterior_mean(model.updater.decode_for_reencode(state))
                    last_transition = turn.event_kind.value
                if turn.calls_reader:
                    query = turn.query
                    metadata = {
                        "episode_id": episode.episode_id,
                        "query_id": f"{episode.episode_id}:q{query_number}",
                        "query_ordinal": query_number,
                        "counterfactual_pair_id": episode.pair_id,
                        "counterfactual_episode_id": episode.counterfactual_episode_id,
                        "topic": episode.topic,
                        "subtype": last_transition,
                        "split": episode.split,
                        "ood_group": episode.ood_group,
                        "update_count": sum(item.calls_updater for item in episode.turns[: turn_index + 1]),
                    }
                    if previous_target is not None and previous_target != query.target_index:
                        metadata["stale_target_index"] = previous_target
                    queries.append(
                        QueryState(
                            metadata=metadata,
                            query=query.text,
                            choices=tuple(query.choices),
                            target_index=query.target_index,
                            state=state.detach().cpu().clone(),
                        )
                    )
                    previous_target = query.target_index
                    query_number += 1
    return queries


def collect_prefeval(
    model: DreamLiteEpisodeModel,
    path: Path,
    limit: int | None,
    recurrence_mode: str,
) -> list[QueryState]:
    records = read_raw_jsonl(path)
    if limit is not None:
        records = records[:limit]
    queries: list[QueryState] = []
    with torch.no_grad():
        for record in records:
            model_input = record["model_input"]
            label = record["label"]
            state = model.reset_state()
            query_number = 0
            for turn_index, turn in enumerate(model_input["turns"]):
                if turn["type"] == "event":
                    state = model.updater(state, turn["text"], model_input["sample_id"], turn_index)
                    if recurrence_mode == "decode_reencode":
                        state = model.updater.reencode_posterior_mean(model.updater.decode_for_reencode(state))
                elif turn["type"] == "query":
                    options = tuple(turn["options"])
                    queries.append(
                        QueryState(
                            metadata={
                                "episode_id": model_input["sample_id"],
                                "query_id": f"{model_input['sample_id']}:q{query_number}",
                                "base_pair_id": model_input["base_pair_id"],
                                "topic": model_input["topic"],
                                "form": model_input["form"],
                                "split": model_input["split"],
                                "protocol": model_input["protocol"],
                                "forced_write_k": model_input["forced_write_k"],
                            },
                            query=turn["text"],
                            choices=options,
                            target_index=int(label["target_index"]),
                            state=state.detach().cpu().clone(),
                        )
                    )
                    query_number += 1
                else:
                    raise ValueError(f"Unsupported PrefEval turn type: {turn['type']}")
    return queries


def intervention_states(
    items: list[QueryState],
    *,
    condition: str,
    initial_state: torch.Tensor,
    seed: int,
) -> list[torch.Tensor]:
    if condition == "standard":
        return [item.state for item in items]
    if condition == "reset":
        return [initial_state.detach().cpu().clone() for _ in items]
    if condition == "shuffle":
        order = list(range(len(items)))
        random.Random(seed).shuffle(order)
        if len(order) > 1 and all(index == value for index, value in enumerate(order)):
            order = order[1:] + order[:1]
        return [items[index].state for index in order]
    if condition == "state_swap":
        by_episode_ordinal = {
            (item.metadata.get("episode_id"), item.metadata.get("query_ordinal", 0)): item.state for item in items
        }
        shuffled = intervention_states(items, condition="shuffle", initial_state=initial_state, seed=seed + 1)
        states: list[torch.Tensor] = []
        for index, item in enumerate(items):
            donor_key = (
                item.metadata.get("counterfactual_episode_id"),
                item.metadata.get("query_ordinal", 0),
            )
            states.append(by_episode_ordinal.get(donor_key, shuffled[index]))
        return states
    raise ValueError(f"Unknown condition: {condition}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate DreamLite visual states with frozen-Qwen MCQ NLL")
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--format", choices=("synthetic", "prefeval"), required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--conditions", nargs="+", choices=("standard", "reset", "shuffle", "state_swap"), default=["standard"])
    parser.add_argument("--recurrence-mode", choices=("direct_latent", "decode_reencode"), default="direct_latent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adapter-seed", type=int, default=0)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("DreamLite MCQ evaluation requires two visible GPUs.")
    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    updater_dtype = torch.bfloat16 if torch.cuda.get_device_capability(updater_device)[0] >= 8 else torch.float16
    reader_dtype = torch.bfloat16 if torch.cuda.get_device_capability(reader_device)[0] >= 8 else torch.float16
    torch.manual_seed(args.adapter_seed)
    torch.cuda.manual_seed_all(args.adapter_seed)

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
    torch.manual_seed(args.adapter_seed)
    torch.cuda.manual_seed_all(args.adapter_seed)
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
    source_pil, _ = load_source_image(args.source_image, resolution=args.resolution)
    source_tensor = pipe.image_processor.preprocess(source_pil, height=args.resolution, width=args.resolution)
    with torch.no_grad():
        initial = pipe.prepare_image_latents(source_tensor, dtype=updater_dtype, device=updater_device)
    model = DreamLiteEpisodeModel(
        pipeline=pipe,
        initial_state=initial,
        global_seed=args.seed,
        checkpoint_unet=False,
    )
    checkpoint_manifest = None
    if args.checkpoint:
        checkpoint_manifest = load_trainable_weights(args.checkpoint, trainable_module=model).get("manifest")

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
        torch_dtype=reader_dtype,
        attn_implementation="sdpa",
    ).to(reader_device)
    freeze_module(reader)
    reader.config.use_cache = False

    if args.format == "synthetic":
        items = collect_synthetic(model, args.episodes, args.limit, args.recurrence_mode)
    else:
        items = collect_prefeval(model, args.episodes, args.limit, args.recurrence_mode)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for condition in args.conditions:
            states = intervention_states(items, condition=condition, initial_state=initial, seed=args.seed)
            for item, cpu_state in zip(items, states, strict=True):
                started = time.monotonic()
                with torch.no_grad():
                    image = model.updater.decode_for_reader(cpu_state.to(updater_device))[0].to(reader_device)
                    result = qwen3vl_choice_nll(
                        model=reader,
                        processor=processor,
                        image=image,
                        query=format_mcq_query(item.query, item.choices),
                        choices=item.choices,
                        device=reader_device,
                    )
                row = {
                    **item.metadata,
                    "method": args.method,
                    "seed": args.seed,
                    "diffusion_seed": args.seed,
                    "recurrence_mode": args.recurrence_mode,
                    "condition": condition,
                    "prediction_index": result.predicted_index,
                    "target_index": item.target_index,
                    "choice_mean_nll": list(result.mean_nll),
                    "latency_seconds": time.monotonic() - started,
                    "state_bytes": int(cpu_state.numel() * cpu_state.element_size()),
                    "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "output": str(args.output),
        "queries": len(items),
        "conditions": args.conditions,
        "checkpoint_manifest": checkpoint_manifest,
        "peak_vram_gib": {
            str(updater_device): torch.cuda.max_memory_allocated(updater_device) / 2**30,
            str(reader_device): torch.cuda.max_memory_allocated(reader_device) / 2**30,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
