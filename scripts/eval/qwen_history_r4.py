from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.eval.qwen_text_baselines import (  # noqa: E402
    BLANK_IMAGE_SHAPE,
    BLANK_IMAGE_VALUE,
    EXPECTED_READER_REVISION,
    audit_context,
    locked_revision,
    sha256_file,
    sha256_text,
)
from vision_memory.data import REVERSE_CYCLIC4, Episode, read_jsonl  # noqa: E402
from vision_memory.data.r5_baseline_lockbox import (  # noqa: E402
    validate_same_entity_pair_contract,
)
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.eval.r4_history_representations import (  # noqa: E402
    R4_HISTORY_METHODS,
    VisibleEvent,
    render_history_representation,
    representation_contract_sha256,
    state_swap_event_stream,
)
from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_RESIZE_CONTRACT,
    deterministic_qwen_reader_resize,
    qwen3vl_choice_nll,
)
from vision_memory.repro import canonical_object_sha256, configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402


SCHEMA_VERSION = "vision_memory.qwen_r4_history_predictions.v1"
REPORT_SCHEMA_VERSION = "vision_memory.qwen_r4_history_report.v1"
SCIENTIFIC_PAYLOAD_SCHEMA_VERSION = "vision_memory.qwen_r4_history_scientific_payload.v1"
CONDITIONS = ("standard", "reset", "shuffle", "state_swap")
BLANK_IMAGE_CONTRACT = {
    "shape": [3, 1024, 1024],
    "dtype": "float32",
    "value": 0.5,
    "bytes": 12_582_912,
    "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
}
RUNTIME_FIELDS = frozenset(
    {
        "replica_id",
        "event_latency_seconds",
        "reader_latency_seconds",
        "query_latency_seconds",
        "latency_seconds",
        "peak_reader_vram_gib",
        "peak_vram_gib",
    }
)


@dataclass(frozen=True)
class R4HistoryQuery:
    metadata: dict[str, Any]
    query: str
    choices: tuple[str, str, str, str]
    target_index: int
    events: tuple[VisibleEvent, ...]


@dataclass(frozen=True)
class R4HistoryIntervention:
    events: tuple[VisibleEvent, ...]
    donor_target_text: str | None = None
    donor_episode_id: str | None = None


@dataclass(frozen=True)
class R4ChoiceView:
    metadata: dict[str, Any]
    query: str
    choices: tuple[str, str, str, str]
    target_index: int
    events: tuple[VisibleEvent, ...]
    donor_target_index: int | None
    donor_episode_id: str | None


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"git rev-parse returned an invalid commit: {commit!r}")
    return commit


def _target_index_in_choices(target_text: str, choices: Sequence[str]) -> int | None:
    try:
        return tuple(choices).index(target_text)
    except ValueError:
        return None


def _read_form(episode: Episode, has_mixed_prefix: bool) -> str:
    identifiers = f"{episode.episode_id}\0{episode.template_id}".casefold()
    if has_mixed_prefix or "mixed" in identifiers:
        return "mixed"
    return "separate"


def synthetic_queries(path: Path, limit: int | None) -> Iterator[R4HistoryQuery]:
    """Extract causal query snapshots from the public Episode schema only.

    A mixed turn appends its event before taking the query snapshot. Query text,
    options, labels, episode metadata and all later events are absent from the
    structured representation input.
    """

    episodes = read_jsonl(path)
    r5_micro = [episode for episode in episodes if episode.episode_id.startswith("r5-")]
    if r5_micro:
        if len(r5_micro) != len(episodes) or any(
            episode.split != "lockbox" for episode in episodes
        ):
            raise ValueError("R5 micro validation refuses mixed namespaces or non-lockbox episodes.")
        delayed_states = sum(
            turn.type.value == "query"
            for episode in episodes
            for turn in episode.turns
            if turn.calls_reader
        )
        validate_same_entity_pair_contract(
            episodes,
            expected_delayed_states=delayed_states,
        )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive when provided.")
        episodes = episodes[:limit]
    for episode in episodes:
        events: list[VisibleEvent] = []
        query_number = 0
        last_transition = "unknown"
        previous_target_text: str | None = None
        events_since_query = 0
        noops_since_query = 0
        has_mixed_prefix = False
        for turn_index, turn in enumerate(episode.turns):
            if turn.calls_updater:
                if turn.event_kind is None or turn.event_text is None:
                    raise RuntimeError("Schema invariant violated: updater turn lacks event kind/text.")
                events.append(VisibleEvent(turn.event_kind, turn.event_text))
                last_transition = turn.event_kind.value
                events_since_query += 1
                noops_since_query += int(turn.event_kind.value == "noop")
                has_mixed_prefix = has_mixed_prefix or turn.type.value == "mixed"
            if not turn.calls_reader:
                continue
            if turn.query is None:
                raise RuntimeError("Schema invariant violated: Reader turn lacks query payload.")
            query = turn.query
            target_text = query.choices[query.target_index]
            stale_index = None
            if previous_target_text is not None and previous_target_text != target_text:
                stale_index = _target_index_in_choices(previous_target_text, query.choices)
            metadata: dict[str, Any] = {
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
                    episode.distractor_variant.value if episode.distractor_variant is not None else None
                ),
                "query_comparison_id": query.comparison_id,
                "semantic_group_id": episode.semantic_group_id,
                "topic": episode.topic,
                "template_id": episode.template_id,
                "subtype": last_transition,
                "form": _read_form(episode, has_mixed_prefix),
                "split": episode.split,
                "ood_group": episode.ood_group,
                "protocol": "synthetic",
                "update_count": len(events),
                "route": "event_then_query" if turn.calls_updater else "query_read_only",
                "query_turn_type": turn.type.value,
                "probe_role": "immediate" if turn.calls_updater else "delayed",
                "updater_calls_since_query": events_since_query,
                "noop_events_since_query": noops_since_query,
                "noop_events_applied_since_query": noops_since_query,
                "noop_policy": "keep",
                "turn_index": turn_index,
            }
            if previous_target_text is not None and previous_target_text != target_text:
                metadata["stale_target_text"] = previous_target_text
                metadata["stale_target_mapped"] = stale_index is not None
                if stale_index is not None:
                    metadata["stale_target_index"] = stale_index
            yield R4HistoryQuery(
                metadata=metadata,
                query=query.text,
                choices=query.choices,
                target_index=query.target_index,
                events=tuple(events),
            )
            previous_target_text = target_text
            query_number += 1
            events_since_query = 0
            noops_since_query = 0


def _different_target_derangement(
    items: Sequence[R4HistoryQuery], recipients: Sequence[int], seed: int
) -> dict[int, int]:
    buckets: dict[str, list[int]] = {}
    for recipient in recipients:
        item = items[recipient]
        buckets.setdefault(item.choices[item.target_index], []).append(recipient)
    if not buckets:
        raise ValueError("Cannot derange an empty query group.")
    maximum_bucket = max(map(len, buckets.values()))
    if len(recipients) < 2 or maximum_bucket > len(recipients) // 2:
        counts = dict(sorted((name, len(values)) for name, values in buckets.items()))
        raise ValueError(f"history shuffle requires a different-target derangement; target_counts={counts}")
    rng = random.Random(seed)
    ordered: list[int] = []
    for target_text in sorted(buckets, key=lambda name: (-len(buckets[name]), name)):
        values = list(buckets[target_text])
        rng.shuffle(values)
        ordered.extend(values)
    donors = ordered[maximum_bucket:] + ordered[:maximum_bucket]
    pairs = dict(zip(ordered, donors, strict=True))
    if any(
        recipient == donor
        or items[recipient].choices[items[recipient].target_index]
        == items[donor].choices[items[donor].target_index]
        for recipient, donor in pairs.items()
    ):
        raise RuntimeError("Internal different-target derangement construction failed.")
    return pairs


def intervention_event_streams(
    items: Sequence[R4HistoryQuery], *, condition: str, seed: int
) -> list[R4HistoryIntervention]:
    if condition == "standard":
        return [R4HistoryIntervention(item.events) for item in items]
    if condition == "reset":
        return [R4HistoryIntervention(()) for _ in items]
    if condition == "shuffle":
        groups: dict[tuple[Any, ...], list[int]] = {}
        for index, item in enumerate(items):
            key = (
                item.metadata.get("split"),
                item.metadata.get("query_ordinal", 0),
                item.metadata.get("probe_role", "delayed"),
                item.metadata.get("noop_policy", "keep"),
            )
            groups.setdefault(key, []).append(index)
        order = list(range(len(items)))
        for group_number, key in enumerate(sorted(groups, key=repr)):
            pairs = _different_target_derangement(items, groups[key], seed + group_number)
            for recipient, donor in pairs.items():
                order[recipient] = donor
        return [
            R4HistoryIntervention(
                events=state_swap_event_stream(items[recipient].events, items[donor].events),
                donor_episode_id=str(items[donor].metadata.get("episode_id")),
            )
            for recipient, donor in enumerate(order)
        ]
    if condition == "state_swap":
        by_episode_ordinal = {
            (item.metadata.get("episode_id"), item.metadata.get("query_ordinal", 0)): item
            for item in items
        }
        interventions: list[R4HistoryIntervention] = []
        for item in items:
            donor_key = (
                item.metadata.get("counterfactual_episode_id"),
                item.metadata.get("query_ordinal", 0),
            )
            donor = by_episode_ordinal.get(donor_key)
            if donor is None:
                raise ValueError(f"state_swap requires a matched counterfactual query; missing {donor_key!r}")
            interventions.append(
                R4HistoryIntervention(
                    events=state_swap_event_stream(item.events, donor.events),
                    donor_target_text=donor.choices[donor.target_index],
                    donor_episode_id=str(donor.metadata.get("episode_id")),
                )
            )
        return interventions
    raise ValueError(f"Unknown condition: {condition!r}")


def expand_reverse_cyclic_views(
    item: R4HistoryQuery, intervention: R4HistoryIntervention
) -> tuple[R4ChoiceView, ...]:
    target_text = item.choices[item.target_index]
    base_query_id = str(item.metadata["query_id"])
    views: list[R4ChoiceView] = []
    for view_index, permutation in enumerate(REVERSE_CYCLIC4):
        choices = tuple(item.choices[index] for index in permutation)
        donor_target_index = None
        if intervention.donor_target_text is not None:
            donor_target_index = _target_index_in_choices(intervention.donor_target_text, choices)
            if donor_target_index is None:
                raise ValueError("Counterfactual donor target is absent from recipient choices.")
        metadata = {
            **item.metadata,
            "base_query_id": base_query_id,
            "query_id": f"{base_query_id}:reverse{view_index}",
            "choice_view_family": "reverse-cyclic4",
            "choice_view_index": view_index,
        }
        stale_target_text = metadata.get("stale_target_text")
        if isinstance(stale_target_text, str):
            stale_index = _target_index_in_choices(stale_target_text, choices)
            metadata["stale_target_mapped"] = stale_index is not None
            if stale_index is None:
                metadata.pop("stale_target_index", None)
            else:
                metadata["stale_target_index"] = stale_index
        views.append(
            R4ChoiceView(
                metadata=metadata,
                query=item.query,
                choices=choices,  # type: ignore[arg-type]
                target_index=choices.index(target_text),
                events=intervention.events,
                donor_target_index=donor_target_index,
                donor_episode_id=intervention.donor_episode_id,
            )
        )
    return tuple(views)


def method_prompt(memory_text: str, query: str, choices: Sequence[str]) -> str:
    return f"{memory_text}\n\n{format_mcq_query(query, tuple(choices))}"


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else getattr(encoded, "input_ids", None)
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise TypeError("Qwen tokenizer must return input_ids with shape [1, sequence].")
    return int(input_ids.shape[1])


def _nll_margin(scores: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in scores)
    return ordered[1] - ordered[0]


def scientific_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in RUNTIME_FIELDS}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prospective R4 frozen-Qwen history-representation evaluator"
    )
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=R4_HISTORY_METHODS, required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--probe-role", choices=("all", "delayed"), default="all")
    parser.add_argument("--choice-view-family", choices=("reverse-cyclic4",), default="reverse-cyclic4")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replica-id", choices=("A", "B"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--strict-determinism", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    if args.output.exists() or report_path.exists():
        raise SystemExit(f"Refusing to overwrite existing R4 artifact: {args.output} / {report_path}")
    if not args.strict_determinism:
        raise SystemExit("R4 Qwen history evaluation requires strict determinism.")
    if len(set(args.conditions)) != len(args.conditions):
        raise SystemExit("--conditions must not contain duplicates.")

    # Keep this before the first CUDA availability/capability/device probe.
    strict_determinism = configure_strict_cuda_determinism(seed=args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("R4 Qwen history evaluation requires CUDA.")

    reader_revision = locked_revision(args.reader)
    if reader_revision != EXPECTED_READER_REVISION:
        raise SystemExit(
            f"Reader revision drift: expected {EXPECTED_READER_REVISION}, observed {reader_revision}."
        )
    items = list(synthetic_queries(args.episodes, args.limit))
    if args.probe_role == "delayed":
        items = [item for item in items if item.metadata.get("probe_role") == "delayed"]
    if not items:
        raise SystemExit("No query states remain after applying the requested filters.")
    episodes_sha256 = sha256_file(args.episodes)

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("The audited R4 evaluator requires a CUDA Reader device.")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
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
    reader.eval()
    reader.config.use_cache = False

    blank_image = torch.full(BLANK_IMAGE_SHAPE, BLANK_IMAGE_VALUE, device=device, dtype=torch.float32)
    observed_blank_contract = {
        "shape": list(blank_image.shape),
        "dtype": str(blank_image.dtype).removeprefix("torch."),
        "value": float(blank_image.reshape(-1)[0].item()),
        "bytes": int(blank_image.numel() * blank_image.element_size()),
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
    }
    if observed_blank_contract != BLANK_IMAGE_CONTRACT or not bool(
        torch.all(blank_image == BLANK_IMAGE_VALUE)
    ):
        raise RuntimeError(
            "Fixed blank-image tensor drifted from its locked scientific contract: "
            f"expected={BLANK_IMAGE_CONTRACT!r}, observed={observed_blank_contract!r}."
        )
    resized_blank_image = deterministic_qwen_reader_resize(
        blank_image, contract=R3_QWEN_READER_RESIZE_CONTRACT
    )
    contract_sha = representation_contract_sha256(args.method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    scientific_rows: list[dict[str, Any]] = []
    prediction_count = 0
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for condition in args.conditions:
            interventions = intervention_event_streams(items, condition=condition, seed=args.seed)
            for item, intervention in zip(items, interventions, strict=True):
                for view in expand_reverse_cyclic_views(item, intervention):
                    representation = render_history_representation(args.method, view.events)
                    if representation.representation_contract_sha256 != contract_sha:
                        raise RuntimeError("Representation contract SHA drifted during evaluation.")
                    prompt = method_prompt(representation.memory_text, view.query, view.choices)
                    context = audit_context(
                        model=reader,
                        processor=processor,
                        prompt=prompt,
                        choices=view.choices,
                        input_mode="blank_image",
                        resized_blank_image=resized_blank_image,
                    )
                    torch.cuda.synchronize(device)
                    started = time.monotonic()
                    result = qwen3vl_choice_nll(
                        model=reader,
                        processor=processor,
                        image=blank_image,
                        query=prompt,
                        choices=view.choices,
                        device=device,
                        reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
                        deterministic_ce=True,
                    )
                    torch.cuda.synchronize(device)
                    reader_latency = time.monotonic() - started
                    peak_vram = torch.cuda.max_memory_allocated(device) / 2**30
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        **view.metadata,
                        "method": args.method,
                        "input_mode": "blank_image",
                        "seed": args.seed,
                        "replica_id": args.replica_id,
                        "condition": condition,
                        "query_text_sha256": sha256_text(view.query),
                        "dataset_sha256": episodes_sha256,
                        "episodes_sha256": episodes_sha256,
                        "prediction_index": result.predicted_index,
                        "target_index": view.target_index,
                        "choices": list(view.choices),
                        "prediction_text": view.choices[result.predicted_index],
                        "target_text": view.choices[view.target_index],
                        "choice_mean_nll": list(result.mean_nll),
                        "nll_margin": _nll_margin(result.mean_nll),
                        "representation_contract_sha256": representation.representation_contract_sha256,
                        "source_event_stream_sha256": representation.source_event_stream_sha256,
                        "memory_text_sha256": representation.memory_text_sha256,
                        "source_event_count": representation.source_event_count,
                        "retained_event_count": representation.retained_event_count,
                        "memory_token_count": _token_count(processor.tokenizer, representation.memory_text),
                        "memory_utf8_bytes": len(representation.memory_text.encode("utf-8")),
                        "state_bytes": len(representation.memory_text.encode("utf-8")),
                        "prompt_sha256": sha256_text(prompt),
                        "chat_prompt_sha256": context.chat_prompt_sha256,
                        "prompt_token_count": context.prompt_token_count,
                        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                        "choice_context_token_counts": list(context.choice_context_token_counts),
                        "choice_target_token_counts": list(context.choice_target_token_counts),
                        "context_limit": context.context_limit,
                        "context_truncated": False,
                        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
                        "blank_image": dict(BLANK_IMAGE_CONTRACT),
                        "constant_visual_input_bytes": int(blank_image.numel() * blank_image.element_size()),
                        "event_latency_seconds": 0.0,
                        "reader_latency_seconds": reader_latency,
                        "query_latency_seconds": reader_latency,
                        "latency_seconds": reader_latency,
                        "peak_reader_vram_gib": peak_vram,
                        "peak_vram_gib": peak_vram,
                        "donor_target_index": view.donor_target_index,
                        "donor_episode_id": view.donor_episode_id,
                        "checkpoint": None,
                        "training_regime": "frozen_baseline",
                        "deterministic_ce": True,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    scientific_rows.append(scientific_row(row))
                    prediction_count += 1

    scientific_payload = {
        "schema_version": SCIENTIFIC_PAYLOAD_SCHEMA_VERSION,
        "records": scientific_rows,
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "method": args.method,
        "input_mode": "blank_image",
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "episodes": str(args.episodes.resolve()),
        "episodes_sha256": episodes_sha256,
        "dataset_sha256": episodes_sha256,
        "git_commit": git_commit(),
        "query_states": len(items),
        "prediction_records": prediction_count,
        "conditions": list(args.conditions),
        "probe_role": args.probe_role,
        "choice_view_family": args.choice_view_family,
        "seed": args.seed,
        "replica_id": args.replica_id,
        "reader_revision": reader_revision,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "representation_contract_sha256": contract_sha,
        "blank_image": dict(BLANK_IMAGE_CONTRACT),
        "strict_determinism": strict_determinism,
        "deterministic_ce": True,
        "context_truncation_policy": "fail_closed",
        "representation_input_contract": "VisibleEvent(kind,text)-only; mixed-write-before-read; no-query-choice-label-ledger-sidecar-future",
        "scientific_payload_sha256": canonical_object_sha256(scientific_payload),
        "peak_vram_gib": {str(device): torch.cuda.max_memory_allocated(device) / 2**30},
    }
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
