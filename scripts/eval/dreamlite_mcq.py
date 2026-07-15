from __future__ import annotations

import argparse
import hashlib
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
from vision_memory.prefeval import prefeval_noise_episode_key  # noqa: E402
from vision_memory.reader import qwen3vl_choice_nll  # noqa: E402
from vision_memory.repro import load_initial_image  # noqa: E402
from vision_memory.training import DreamLiteEpisodeModel, format_mcq_query, load_trainable_weights  # noqa: E402


@dataclass
class QueryState:
    metadata: dict[str, Any]
    query: str
    choices: tuple[str, ...]
    target_index: int
    state: torch.Tensor


@dataclass(frozen=True)
class InterventionState:
    state: torch.Tensor
    donor_target_index: int | None = None
    donor_episode_id: str | None = None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _target_index_in_choices(target_text: str, choices: tuple[str, ...]) -> int | None:
    try:
        return choices.index(target_text)
    except ValueError:
        return None


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    if not marker.is_file():
        raise ValueError(f"Model snapshot is missing required revision marker: {marker}")
    revision = marker.read_text(encoding="utf-8").strip()
    if not revision:
        raise ValueError(f"Model snapshot has an empty revision marker: {marker}")
    return revision


def read_checkpoint_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("manifest"), dict):
        raise ValueError(f"Unsupported or unmanifested DreamLite checkpoint: {path}")
    return dict(payload["manifest"])


def resolve_checkpoint_argument(
    *,
    name: str,
    supplied: Any,
    manifest_args: dict[str, Any] | None,
    default: Any,
) -> Any:
    checkpoint_value = None if manifest_args is None else manifest_args.get(name)
    if manifest_args is not None and checkpoint_value is None:
        raise ValueError(f"Checkpoint manifest does not record required argument {name!r}.")
    if supplied is not None and checkpoint_value is not None and supplied != checkpoint_value:
        raise ValueError(
            f"Evaluation argument {name!r}={supplied!r} conflicts with checkpoint value {checkpoint_value!r}."
        )
    return supplied if supplied is not None else (checkpoint_value if checkpoint_value is not None else default)


def collect_synthetic(
    model: DreamLiteEpisodeModel,
    path: Path,
    limit: int | None,
    recurrence_mode: str,
    *,
    skip_noop: bool,
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
            previous_target_text: str | None = None
            event_latency_since_query = 0.0
            updater_calls_since_query = 0
            noop_events_since_query = 0
            noop_events_applied_since_query = 0
            for turn_index, turn in enumerate(episode.turns):
                if turn.calls_updater:
                    is_noop = turn.event_kind.value == "noop"
                    noop_events_since_query += int(is_noop)
                    if not (skip_noop and is_noop):
                        _synchronize(state.device)
                        event_started = time.monotonic()
                        state = model.updater(state, turn.event_text, episode.episode_id, turn_index)
                        if recurrence_mode == "decode_reencode":
                            state = model.updater.reencode_posterior_mean(model.updater.decode_for_reencode(state))
                        _synchronize(state.device)
                        event_latency_since_query += time.monotonic() - event_started
                        updater_calls_since_query += 1
                        noop_events_applied_since_query += int(is_noop)
                    last_transition = turn.event_kind.value
                if turn.calls_reader:
                    query = turn.query
                    target_text = query.choices[query.target_index]
                    metadata = {
                        "episode_id": episode.episode_id,
                        "query_id": f"{episode.episode_id}:q{query_number}",
                        "query_ordinal": query_number,
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
                        "split": episode.split,
                        "ood_group": episode.ood_group,
                        "update_count": sum(item.calls_updater for item in episode.turns[: turn_index + 1]),
                        "route": "event_then_query" if turn.calls_updater else "query_read_only",
                        "query_turn_type": turn.type.value,
                        "event_latency_seconds": event_latency_since_query,
                        "updater_calls_since_query": updater_calls_since_query,
                        "noop_events_since_query": noop_events_since_query,
                        "noop_events_applied_since_query": noop_events_applied_since_query,
                        "noop_policy": "skip" if skip_noop else "keep",
                    }
                    if previous_target_text is not None and previous_target_text != target_text:
                        stale_index = _target_index_in_choices(previous_target_text, tuple(query.choices))
                        metadata["stale_target_text"] = previous_target_text
                        metadata["stale_target_mapped"] = stale_index is not None
                        if stale_index is not None:
                            metadata["stale_target_index"] = stale_index
                    queries.append(
                        QueryState(
                            metadata=metadata,
                            query=query.text,
                            choices=tuple(query.choices),
                            target_index=query.target_index,
                            state=state.detach().cpu().clone(),
                        )
                    )
                    previous_target_text = target_text
                    query_number += 1
                    event_latency_since_query = 0.0
                    updater_calls_since_query = 0
                    noop_events_since_query = 0
                    noop_events_applied_since_query = 0
    return queries


def collect_prefeval(
    model: DreamLiteEpisodeModel,
    path: Path,
    limit: int | None,
    recurrence_mode: str,
    *,
    skip_noop: bool,
) -> list[QueryState]:
    records = read_raw_jsonl(path)
    if limit is not None:
        records = records[:limit]
    queries: list[QueryState] = []
    with torch.no_grad():
        for record in records:
            model_input = record["model_input"]
            label = record["label"]
            noise_episode_key = prefeval_noise_episode_key(
                str(model_input["base_pair_id"]),
                str(model_input["form"]),
            )
            state = model.reset_state()
            query_number = 0
            event_latency_since_query = 0.0
            updater_calls_since_query = 0
            noop_events_since_query = 0
            noop_events_applied_since_query = 0
            for turn_index, turn in enumerate(model_input["turns"]):
                if turn["type"] == "event":
                    is_noop = turn.get("event_type") == "noop"
                    noop_events_since_query += int(is_noop)
                    if not (skip_noop and is_noop):
                        _synchronize(state.device)
                        event_started = time.monotonic()
                        state = model.updater(state, turn["text"], noise_episode_key, turn_index)
                        if recurrence_mode == "decode_reencode":
                            state = model.updater.reencode_posterior_mean(model.updater.decode_for_reencode(state))
                        _synchronize(state.device)
                        event_latency_since_query += time.monotonic() - event_started
                        updater_calls_since_query += 1
                        noop_events_applied_since_query += int(is_noop)
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
                                "noise_episode_key": noise_episode_key,
                                "route": "query_read_only",
                                "query_turn_type": "query",
                                "event_latency_seconds": event_latency_since_query,
                                "updater_calls_since_query": updater_calls_since_query,
                                "noop_events_since_query": noop_events_since_query,
                                "noop_events_applied_since_query": noop_events_applied_since_query,
                                "noop_policy": "skip" if skip_noop else "keep",
                            },
                            query=turn["text"],
                            choices=options,
                            target_index=int(label["target_index"]),
                            state=state.detach().cpu().clone(),
                        )
                    )
                    query_number += 1
                    event_latency_since_query = 0.0
                    updater_calls_since_query = 0
                    noop_events_since_query = 0
                    noop_events_applied_since_query = 0
                else:
                    raise ValueError(f"Unsupported PrefEval turn type: {turn['type']}")
    return queries


def intervention_states(
    items: list[QueryState],
    *,
    condition: str,
    initial_state: torch.Tensor,
    seed: int,
) -> list[InterventionState]:
    if condition == "standard":
        return [InterventionState(item.state) for item in items]
    if condition == "reset":
        return [InterventionState(initial_state.detach().cpu().clone()) for _ in items]
    if condition == "shuffle":
        groups: dict[tuple[Any, ...], list[int]] = {}
        for index, item in enumerate(items):
            key = (
                item.metadata.get("split"),
                item.metadata.get("protocol", "synthetic"),
                item.metadata.get("forced_write_k"),
                item.metadata.get("form", item.metadata.get("subtype")),
                item.metadata.get("query_ordinal", 0),
                item.metadata.get("distractor_variant"),
                item.metadata.get("noop_policy"),
            )
            groups.setdefault(key, []).append(index)
        order = list(range(len(items)))
        for group_number, key in enumerate(sorted(groups, key=repr)):
            recipients = groups[key]
            donors = recipients.copy()
            random.Random(seed + group_number).shuffle(donors)
            if len(donors) > 1 and donors == recipients:
                donors = donors[1:] + donors[:1]
            for recipient, donor in zip(recipients, donors, strict=True):
                order[recipient] = donor
        return [InterventionState(items[index].state) for index in order]
    if condition == "state_swap":
        by_episode_ordinal = {
            (
                item.metadata.get("episode_id"),
                item.metadata.get("query_ordinal", 0),
                item.metadata.get("noop_policy"),
            ): item
            for item in items
        }
        states: list[InterventionState] = []
        for item in items:
            donor_key = (
                item.metadata.get("counterfactual_episode_id"),
                item.metadata.get("query_ordinal", 0),
                item.metadata.get("noop_policy"),
            )
            donor = by_episode_ordinal.get(donor_key)
            if donor is None:
                raise ValueError(
                    "state_swap requires the semantic counterfactual episode/query under the same no-op policy; "
                    f"missing donor for {donor_key!r}"
                )
            donor_target_text = donor.choices[donor.target_index]
            donor_target_index = _target_index_in_choices(donor_target_text, item.choices)
            if donor_target_index is None:
                raise ValueError(
                    "state_swap donor semantic target is absent from recipient choices: "
                    f"{donor_target_text!r} not in {item.choices!r}"
                )
            states.append(
                InterventionState(
                    donor.state,
                    donor_target_index=donor_target_index,
                    donor_episode_id=str(donor.metadata.get("episode_id")),
                )
            )
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
    parser.add_argument(
        "--noop-policy",
        choices=("keep", "skip", "both"),
        default="keep",
        help="Keep no-op/distractor events, prefilter them, or emit a paired keep/skip evaluation.",
    )
    parser.add_argument("--recurrence-mode", choices=("direct_latent", "decode_reencode"), default="direct_latent")
    parser.add_argument("--seed", type=int, default=0, help="Diffusion/noise seed used for this evaluation pass.")
    parser.add_argument(
        "--training-seed",
        type=int,
        help="Checkpoint training seed; inferred from the checkpoint manifest when available.",
    )
    parser.add_argument("--adapter-seed", type=int)
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--initial-state-mode", choices=("blank", "fixture", "file"))
    parser.add_argument("--source-image", type=Path, help="Accepted only for the explicit file initialization mode")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    args = parser.parse_args()

    if args.format == "prefeval" and "state_swap" in args.conditions:
        raise SystemExit("state_swap requires synthetic matched counterfactual episodes.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("DreamLite MCQ evaluation requires two visible GPUs.")
    checkpoint_manifest = read_checkpoint_manifest(args.checkpoint)
    manifest_args = None
    if checkpoint_manifest is not None:
        raw_manifest_args = checkpoint_manifest.get("arguments")
        if not isinstance(raw_manifest_args, dict):
            raise SystemExit("Checkpoint manifest has no arguments mapping.")
        manifest_args = raw_manifest_args
    try:
        initial_state_mode = resolve_checkpoint_argument(
            name="initial_state_mode",
            supplied=args.initial_state_mode,
            manifest_args=manifest_args,
            default="blank",
        )
        adapter_seed = int(
            resolve_checkpoint_argument(
                name="adapter_seed",
                supplied=args.adapter_seed,
                manifest_args=manifest_args,
                default=0,
            )
        )
        lora_rank = int(
            resolve_checkpoint_argument(
                name="lora_rank",
                supplied=args.lora_rank,
                manifest_args=manifest_args,
                default=4,
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    checkpoint_initial_image = None if checkpoint_manifest is None else checkpoint_manifest.get("initial_image")
    if checkpoint_manifest is not None and not isinstance(checkpoint_initial_image, dict):
        raise SystemExit("Checkpoint manifest has no initial_image provenance mapping.")
    checkpoint_source = (
        checkpoint_initial_image.get("path") if isinstance(checkpoint_initial_image, dict) else None
    )
    source_image = args.source_image
    if initial_state_mode == "file":
        if source_image is None and isinstance(checkpoint_source, str) and checkpoint_source:
            source_image = Path(checkpoint_source)
        if source_image is None:
            raise SystemExit("File initialization requires --source-image or a checkpoint-recorded source path.")
    elif source_image is not None:
        raise SystemExit("--source-image is accepted only with --initial-state-mode file.")
    learn_initial_state = bool(manifest_args.get("learn_initial_state", False)) if manifest_args else False

    dreamlite_revision = locked_revision(args.dreamlite)
    reader_revision = locked_revision(args.reader)
    if checkpoint_manifest is not None:
        if checkpoint_manifest.get("dreamlite_revision") != dreamlite_revision:
            raise SystemExit("DreamLite revision differs from the checkpoint manifest.")
        if checkpoint_manifest.get("reader_revision") != reader_revision:
            raise SystemExit("Reader revision differs from the checkpoint manifest.")

    updater_device = torch.device(args.dreamlite_device)
    reader_device = torch.device(args.reader_device)
    updater_dtype = torch.bfloat16 if torch.cuda.get_device_capability(updater_device)[0] >= 8 else torch.float16
    reader_dtype = torch.bfloat16 if torch.cuda.get_device_capability(reader_device)[0] >= 8 else torch.float16
    torch.manual_seed(adapter_seed)
    torch.cuda.manual_seed_all(adapter_seed)

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
    torch.manual_seed(adapter_seed)
    torch.cuda.manual_seed_all(adapter_seed)
    pipe.unet = get_peft_model(
        pipe.unet,
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            lora_dropout=0.0,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ),
    )
    pipe.unet.eval()
    source_pil, initial_image_metadata = load_initial_image(
        initial_state_mode,
        source_image,
        resolution=args.resolution,
    )
    if isinstance(checkpoint_initial_image, dict):
        provenance_fields = (
            "initial_state_mode",
            "origin",
            "fixture_id",
            "file_sha256",
            "rgb_sha256",
            "mode",
            "size",
        )
        drift = {
            field: (checkpoint_initial_image.get(field), initial_image_metadata.get(field))
            for field in provenance_fields
            if checkpoint_initial_image.get(field) != initial_image_metadata.get(field)
        }
        if drift:
            raise SystemExit(f"Initial image provenance differs from the checkpoint manifest: {drift}")
    source_tensor = pipe.image_processor.preprocess(source_pil, height=args.resolution, width=args.resolution)
    with torch.no_grad():
        initial = pipe.prepare_image_latents(source_tensor, dtype=updater_dtype, device=updater_device)
    model = DreamLiteEpisodeModel(
        pipeline=pipe,
        initial_state=initial,
        global_seed=args.seed,
        checkpoint_unet=False,
        learn_initial_state=learn_initial_state,
    )
    if args.checkpoint:
        loaded_manifest = load_trainable_weights(args.checkpoint, trainable_module=model).get("manifest")
        if loaded_manifest != checkpoint_manifest:
            raise RuntimeError("Checkpoint manifest changed between metadata inspection and weight loading.")
    training_seed = args.training_seed
    if training_seed is None and isinstance(checkpoint_manifest, dict):
        manifest_args = checkpoint_manifest.get("arguments")
        if isinstance(manifest_args, dict) and isinstance(manifest_args.get("seed"), int):
            training_seed = int(manifest_args["seed"])
    if training_seed is None:
        training_seed = adapter_seed

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

    torch.cuda.reset_peak_memory_stats(updater_device)
    torch.cuda.reset_peak_memory_stats(reader_device)
    skip_policies = (False, True) if args.noop_policy == "both" else (args.noop_policy == "skip",)
    items: list[QueryState] = []
    for skip_noop in skip_policies:
        if args.format == "synthetic":
            items.extend(
                collect_synthetic(
                    model,
                    args.episodes,
                    args.limit,
                    args.recurrence_mode,
                    skip_noop=skip_noop,
                )
            )
        else:
            items.extend(
                collect_prefeval(
                    model,
                    args.episodes,
                    args.limit,
                    args.recurrence_mode,
                    skip_noop=skip_noop,
                )
            )
    if args.noop_policy == "both":
        for item in items:
            item.metadata["noop_intervention_pair_id"] = str(item.metadata["query_id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for condition in args.conditions:
            states = intervention_states(items, condition=condition, initial_state=initial, seed=args.seed)
            for item, intervention in zip(items, states, strict=True):
                cpu_state = intervention.state
                _synchronize(updater_device)
                decode_started = time.monotonic()
                with torch.no_grad():
                    image = model.updater.decode_for_reader(cpu_state.to(updater_device))[0].to(reader_device)
                _synchronize(updater_device)
                _synchronize(reader_device)
                decode_latency = time.monotonic() - decode_started
                reader_started = time.monotonic()
                with torch.no_grad():
                    result = qwen3vl_choice_nll(
                        model=reader,
                        processor=processor,
                        image=image,
                        query=format_mcq_query(item.query, item.choices),
                        choices=item.choices,
                        device=reader_device,
                    )
                _synchronize(reader_device)
                reader_latency = time.monotonic() - reader_started
                updater_peak = torch.cuda.max_memory_allocated(updater_device) / 2**30
                reader_peak = torch.cuda.max_memory_allocated(reader_device) / 2**30
                row = {
                    **item.metadata,
                    "method": args.method,
                    "seed": training_seed,
                    "diffusion_seed": args.seed,
                    "recurrence_mode": args.recurrence_mode,
                    "initial_state_mode": initial_state_mode,
                    "learn_initial_state": learn_initial_state,
                    "condition": condition,
                    "prediction_index": result.predicted_index,
                    "target_index": item.target_index,
                    "choice_mean_nll": list(result.mean_nll),
                    "decode_latency_seconds": decode_latency,
                    "reader_latency_seconds": reader_latency,
                    "query_latency_seconds": decode_latency + reader_latency,
                    "latency_seconds": float(item.metadata["event_latency_seconds"])
                    + decode_latency
                    + reader_latency,
                    "state_bytes": int(cpu_state.numel() * cpu_state.element_size()),
                    "peak_updater_vram_gib": updater_peak,
                    "peak_reader_vram_gib": reader_peak,
                    "peak_vram_gib": max(updater_peak, reader_peak),
                    "peak_vram_gib_by_device": {
                        str(updater_device): updater_peak,
                        str(reader_device): reader_peak,
                    },
                    "donor_target_index": intervention.donor_target_index,
                    "donor_episode_id": intervention.donor_episode_id,
                    "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
                }
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "episodes": str(args.episodes.resolve()),
        "episodes_sha256": sha256_file(args.episodes),
        "queries": len(items),
        "prediction_records": len(items) * len(args.conditions),
        "conditions": args.conditions,
        "noop_policy": args.noop_policy,
        "checkpoint_manifest": checkpoint_manifest,
        "dreamlite_revision": dreamlite_revision,
        "reader_revision": reader_revision,
        "initial_image": initial_image_metadata,
        "learn_initial_state": learn_initial_state,
        "peak_vram_gib": {
            str(updater_device): torch.cuda.max_memory_allocated(updater_device) / 2**30,
            str(reader_device): torch.cuda.max_memory_allocated(reader_device) / 2**30,
        },
    }
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
