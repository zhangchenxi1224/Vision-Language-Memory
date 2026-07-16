from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import Episode, QuerySpec, read_jsonl  # noqa: E402
from vision_memory.dreamlite import assert_no_frozen_parameter_grads, freeze_module  # noqa: E402
from vision_memory.reader import qwen3vl_choice_nll, qwen3vl_target_only_ce  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402

from scripts.data.qwen_sanity import locked_revision, sha256_file  # noqa: E402
from scripts.train.lightweight_episode import (  # noqa: E402
    FORMAL_OVERFIT_EPISODES,
    training_subset_audit,
    validate_overfit_gate_episodes,
)


IMAGE_SHAPE = (3, 256, 256)
EXPECTED_RAW_READS = 128
PERMUTATION_COUNT = 24
PERMUTATIONS_PER_SPLIT = 12
DISCLAIMER = (
    "EVENT-PREFIX CODEBOOK DIAGNOSTIC ONLY: a transductive image code is selected solely "
    "by the ordered, model-visible event_text prefix before a query. Target supervision is "
    "used only in the training loss and evaluation. This is not a learned updater, baseline, "
    "ablation, generalization result, or publishable method result."
)
KEY_SCHEMA = "vision_memory.visible_event_prefix.v1"
FORBIDDEN_SELECTOR_FIELDS = (
    "query.text",
    "query.choices",
    "query.target_index",
    "query.target",
    "query.comparison_id",
    "episode_id",
    "entity_id",
    "template_id",
    "topic",
)


@dataclass(frozen=True)
class EventPrefixRead:
    episode_id: str
    turn_index: int
    query_ordinal: int
    event_texts: tuple[str, ...]
    state_key: str
    query: QuerySpec
    topic: str
    template_id: str
    distractor_variant: str | None


@dataclass(frozen=True)
class PermutationView:
    read_index: int
    split: str
    choices: tuple[str, str, str, str]
    target_index: int
    parity: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Transductive event-prefix visual-code diagnostic with held-out choice permutations")
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4_096)
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


def _canonical_prefix_payload(event_texts: Sequence[str]) -> bytes:
    texts = tuple(event_texts)
    if not texts:
        raise ValueError("An event-prefix code requires at least one visible event_text")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("event_texts must contain only non-empty strings")
    return json.dumps(
        {"schema": KEY_SCHEMA, "event_texts": list(texts)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def event_prefix_key(event_texts: Sequence[str]) -> str:
    """Hash only ordered event text already visible before the current read."""

    return hashlib.sha256(_canonical_prefix_payload(event_texts)).hexdigest()


def select_code_index(state_key: str, code_index_by_key: Mapping[str, int]) -> int:
    """Select a code without accepting query or target metadata as arguments."""

    try:
        return int(code_index_by_key[state_key])
    except KeyError as error:
        raise KeyError(f"No visual code exists for event-prefix key {state_key}") from error


def collect_event_prefix_reads(episodes: Sequence[Episode]) -> list[EventPrefixRead]:
    reads: list[EventPrefixRead] = []
    payload_by_key: dict[str, bytes] = {}
    for episode in episodes:
        event_texts: list[str] = []
        query_ordinal = 0
        for turn_index, turn in enumerate(episode.turns):
            # Mixed turns intentionally append their event before the read, matching run_episode.
            if turn.calls_updater:
                if turn.event_text is None:
                    raise RuntimeError(
                        f"Schema invariant violated: {episode.episode_id} turn {turn_index} has no event_text"
                    )
                event_texts.append(turn.event_text)
            if not turn.calls_reader:
                continue
            if turn.query is None:
                raise RuntimeError(f"Schema invariant violated: {episode.episode_id} turn {turn_index} has no query")
            payload = _canonical_prefix_payload(event_texts)
            state_key = hashlib.sha256(payload).hexdigest()
            existing_payload = payload_by_key.setdefault(state_key, payload)
            if existing_payload != payload:
                raise RuntimeError(f"SHA256 collision for event-prefix key {state_key}")
            reads.append(
                EventPrefixRead(
                    episode_id=episode.episode_id,
                    turn_index=turn_index,
                    query_ordinal=query_ordinal,
                    event_texts=tuple(event_texts),
                    state_key=state_key,
                    query=turn.query,
                    topic=episode.topic,
                    template_id=episode.template_id,
                    distractor_variant=(
                        episode.distractor_variant.value if episode.distractor_variant is not None else None
                    ),
                )
            )
            query_ordinal += 1
    if not reads:
        raise ValueError("The event-prefix diagnostic dataset contains no queries")
    return reads


def permutation_parity(permutation: Sequence[int]) -> int:
    values = tuple(permutation)
    if sorted(values) != list(range(len(values))):
        raise ValueError("permutation must contain each integer in [0, n) exactly once")
    inversions = sum(
        values[left] > values[right] for left in range(len(values)) for right in range(left + 1, len(values))
    )
    return inversions % 2


def split_choice_permutations(
    choices: Sequence[str],
) -> tuple[
    tuple[tuple[str, str, str, str], ...],
    tuple[tuple[str, str, str, str], ...],
]:
    if len(choices) != 4 or len(set(choices)) != 4:
        raise ValueError("Exactly four distinct choices are required")
    canonical = tuple(sorted(choices))
    train: list[tuple[str, str, str, str]] = []
    heldout: list[tuple[str, str, str, str]] = []
    for indices in itertools.permutations(range(4)):
        permuted = tuple(canonical[index] for index in indices)
        destination = train if permutation_parity(indices) == 0 else heldout
        destination.append(permuted)  # type: ignore[arg-type]
    if len(train) != PERMUTATIONS_PER_SPLIT or len(heldout) != PERMUTATIONS_PER_SPLIT:
        raise RuntimeError("Parity split did not produce two 12-permutation partitions")
    if set(train) & set(heldout) or len(set(train) | set(heldout)) != PERMUTATION_COUNT:
        raise RuntimeError("Choice permutation partitions are not disjoint and exhaustive")
    return tuple(train), tuple(heldout)


def build_permutation_views(
    reads: Sequence[EventPrefixRead],
) -> tuple[list[PermutationView], list[PermutationView]]:
    train_views: list[PermutationView] = []
    heldout_views: list[PermutationView] = []
    for read_index, read in enumerate(reads):
        target_text = read.query.target
        train, heldout = split_choice_permutations(read.query.choices)
        for split, permutations, destination in (
            ("train_even", train, train_views),
            ("heldout_odd", heldout, heldout_views),
        ):
            parity = 0 if split == "train_even" else 1
            for choices in permutations:
                destination.append(
                    PermutationView(
                        read_index=read_index,
                        split=split,
                        choices=choices,
                        target_index=choices.index(target_text),
                        parity=parity,
                    )
                )
    return train_views, heldout_views


def deterministic_view_order(view_count: int, *, steps: int, seed: int) -> list[int]:
    if view_count < 1 or steps < 1:
        raise ValueError("view_count and steps must be positive")
    order: list[int] = []
    epoch = 0
    while len(order) < steps:
        indices = list(range(view_count))
        random.Random((seed << 16) ^ epoch).shuffle(indices)
        order.extend(indices)
        epoch += 1
    return order[:steps]


def target_position_counts(views: Sequence[PermutationView]) -> dict[int, int]:
    return dict(sorted(Counter(view.target_index for view in views).items()))


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


def selector_leakage_audit(reads: Sequence[EventPrefixRead]) -> dict[str, Any]:
    signature = tuple(inspect.signature(select_code_index).parameters)
    if signature != ("state_key", "code_index_by_key"):
        raise RuntimeError(f"Unexpected selector signature: {signature}")
    mismatches = sum(event_prefix_key(read.event_texts) != read.state_key for read in reads)
    if mismatches:
        raise RuntimeError(f"Found {mismatches} event-prefix key recomputation mismatches")
    return {
        "passed": True,
        "state_key_schema": KEY_SCHEMA,
        "state_key_source_fields": ["ordered turns[:query].event_text"],
        "selector_signature": list(signature),
        "forbidden_selector_fields": list(FORBIDDEN_SELECTOR_FIELDS),
        "forbidden_selector_fields_used": False,
        "state_key_recomputation_mismatches": mismatches,
        "target_supervision_role": "training loss and evaluation only",
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
        raise SystemExit("The event-prefix visual-code diagnostic requires CUDA and real frozen Qwen.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("The diagnostic refuses a non-empty --output-dir.")
    status = git_value("status", "--porcelain")
    if status and not args.allow_dirty:
        raise SystemExit("The diagnostic refuses a dirty worktree.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must select CUDA")
    set_seeds(args.seed)
    episodes = list(read_jsonl(args.dataset))[:FORMAL_OVERFIT_EPISODES]
    validate_overfit_gate_episodes(episodes)
    subset_audit = training_subset_audit(episodes)
    reads = collect_event_prefix_reads(episodes)
    if len(reads) != EXPECTED_RAW_READS:
        raise RuntimeError(f"Expected {EXPECTED_RAW_READS} raw reads, found {len(reads)}")
    leakage_audit = selector_leakage_audit(reads)
    train_views, heldout_views = build_permutation_views(reads)
    expected_views = EXPECTED_RAW_READS * PERMUTATIONS_PER_SPLIT
    if len(train_views) != expected_views or len(heldout_views) != expected_views:
        raise RuntimeError(
            f"Expected {expected_views} views per split, found {len(train_views)} and {len(heldout_views)}"
        )
    expected_position_count = EXPECTED_RAW_READS * 3
    expected_positions = {index: expected_position_count for index in range(4)}
    if target_position_counts(train_views) != expected_positions:
        raise RuntimeError("Training permutations are not exactly target-position balanced")
    if target_position_counts(heldout_views) != expected_positions:
        raise RuntimeError("Held-out permutations are not exactly target-position balanced")

    state_keys = sorted({read.state_key for read in reads})
    code_index_by_key = {state_key: index for index, state_key in enumerate(state_keys)}
    for read in reads:
        select_code_index(read.state_key, code_index_by_key)

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

    image_logits = torch.nn.Parameter(torch.zeros(len(state_keys), *IMAGE_SHAPE, device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([image_logits], lr=args.learning_rate)
    view_order = deterministic_view_order(len(train_views), steps=args.steps, seed=args.seed)
    metrics_path = args.output_dir / "training_metrics.jsonl"
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with metrics_path.open("w", encoding="utf-8", newline="\n") as handle:
        for optimizer_step, view_index in enumerate(view_order, start=1):
            view = train_views[view_index]
            read = reads[view.read_index]
            code_index = select_code_index(read.state_key, code_index_by_key)
            optimizer.zero_grad(set_to_none=True)
            image = torch.sigmoid(image_logits[code_index])
            output = qwen3vl_target_only_ce(
                model=reader,
                processor=processor,
                image=image,
                query=format_mcq_query(read.query.text, view.choices),
                target=read.query.target,
                device=device,
                require_image_grad=True,
            )
            output.loss.backward()
            assert_no_frozen_parameter_grads(reader, "Qwen Reader")
            gradient_norm = image_logits.grad.norm() if image_logits.grad is not None else None
            if gradient_norm is None or not torch.isfinite(gradient_norm) or gradient_norm.item() <= 0:
                raise RuntimeError(f"Invalid event-prefix control gradient at step {optimizer_step}: {gradient_norm}")
            optimizer.step()
            handle.write(
                json.dumps(
                    {
                        "optimizer_step": optimizer_step,
                        "episode_id": read.episode_id,
                        "turn_index": read.turn_index,
                        "state_key": read.state_key,
                        "code_index": code_index,
                        "permutation_split": view.split,
                        "permutation_parity": view.parity,
                        "target_index": view.target_index,
                        "loss": float(output.loss.item()),
                        "gradient_norm": float(gradient_norm.item()),
                        "diagnostic_disclaimer": DISCLAIMER,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    predictions_path = args.output_dir / "heldout_predictions.jsonl"
    records: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for view in heldout_views:
            read = reads[view.read_index]
            code_index = select_code_index(read.state_key, code_index_by_key)
            score = qwen3vl_choice_nll(
                model=reader,
                processor=processor,
                image=torch.sigmoid(image_logits[code_index]).detach(),
                query=format_mcq_query(read.query.text, view.choices),
                choices=view.choices,
                device=device,
            )
            record = {
                "episode_id": read.episode_id,
                "turn_index": read.turn_index,
                "query_ordinal": read.query_ordinal,
                "state_key": read.state_key,
                "code_index": code_index,
                "event_count": len(read.event_texts),
                "topic": read.topic,
                "template_id": read.template_id,
                "distractor_variant": read.distractor_variant,
                "permutation_split": view.split,
                "permutation_parity": view.parity,
                "choices": list(view.choices),
                "target_text": read.query.target,
                "target_index": view.target_index,
                "predicted_index": score.predicted_index,
                "choice_mean_nll": list(score.mean_nll),
                "correct": score.predicted_index == view.target_index,
                "diagnostic_disclaimer": DISCLAIMER,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    accuracy = sum(int(record["correct"]) for record in records) / len(records)
    code_payload = {
        "schema_version": "vision_memory.event_prefix_visual_codes.v1",
        "diagnostic_disclaimer": DISCLAIMER,
        "state_key_schema": KEY_SCHEMA,
        "state_keys_in_code_index_order": state_keys,
        "image_logits": image_logits.detach().cpu(),
    }
    codes_path = args.output_dir / "event_prefix_control_codes.pt"
    torch.save(code_payload, codes_path)
    summary = {
        "schema_version": "vision_memory.event_prefix_semantic_upper_bound_summary.v1",
        "diagnostic_disclaimer": DISCLAIMER,
        "passed_threshold": accuracy >= args.threshold,
        "threshold": args.threshold,
        "heldout_permutation_accuracy": accuracy,
        "heldout_accuracy_by_target_position": grouped_accuracy(records, "target_index"),
        "heldout_accuracy_by_topic": grouped_accuracy(records, "topic"),
        "heldout_accuracy_by_distractor_variant": grouped_accuracy(records, "distractor_variant"),
        "episode_count": len(episodes),
        "raw_read_count": len(reads),
        "unique_event_prefix_code_count": len(state_keys),
        "train_permutation_view_count": len(train_views),
        "heldout_permutation_view_count": len(heldout_views),
        "train_target_position_counts": target_position_counts(train_views),
        "heldout_target_position_counts": target_position_counts(heldout_views),
        "permutation_protocol": {
            "total_per_read": PERMUTATION_COUNT,
            "train": "12 even permutations",
            "heldout": "12 odd permutations",
            "train_heldout_disjoint": True,
            "heldout_is_candidate_permutation_only": True,
            "heldout_is_not_episode_or_state_generalization": True,
        },
        "selector_leakage_audit": leakage_audit,
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
            "training_metrics": str(metrics_path.resolve()),
            "training_metrics_sha256": sha256_file(metrics_path),
            "heldout_predictions": str(predictions_path.resolve()),
            "heldout_predictions_sha256": sha256_file(predictions_path),
            "control_codes": str(codes_path.resolve()),
            "control_codes_sha256": sha256_file(codes_path),
        },
        "control_code_tensor_sha256": sha256_bytes(image_logits.detach().cpu().contiguous().numpy().tobytes()),
        "state_key_inventory_sha256": sha256_bytes(json.dumps(state_keys, separators=(",", ":")).encode("utf-8")),
        "frozen_reader_parameter_grad_count": sum(parameter.grad is not None for parameter in reader.parameters()),
        "elapsed_seconds": time.monotonic() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["passed_threshold"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
