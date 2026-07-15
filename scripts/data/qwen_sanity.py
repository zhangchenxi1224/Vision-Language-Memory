from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import Episode, QuerySpec, read_jsonl  # noqa: E402
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import ChoiceScoreOutput, qwen3vl_choice_nll  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402


CLEAR_TARGET = "no active preference"
BLANK_IMAGE_SPEC = {"shape": [3, 256, 256], "float_value": 0.5}


@dataclass(frozen=True)
class QueryMember:
    episode_id: str
    query_id: str
    pair_id: str
    counterfactual_episode_id: str
    distractor_pair_id: str | None
    distractor_episode_id: str | None
    distractor_variant: str | None
    turn_index: int
    turn_type: str


@dataclass
class UniqueQuery:
    comparison_id: str
    query: QuerySpec
    query_ordinal: int
    topic: str
    entity_id: str
    template_id: str
    template_family: str | None
    members: list[QueryMember] = field(default_factory=list)

    @property
    def audit_signature(self) -> tuple[Any, ...]:
        return (
            self.query.text,
            self.query.choices,
            self.query.target_index,
            self.query.target_token_count,
            self.query_ordinal,
            self.topic,
            self.entity_id,
            self.template_id,
            self.template_family,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_sha256(prompt: str) -> str:
    payload = json.dumps(
        {"image": BLANK_IMAGE_SPEC, "prompt": prompt},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def locked_revision(path: Path) -> str:
    marker = path / ".locked_revision"
    if not marker.is_file():
        raise RuntimeError(f"Reader snapshot is missing required revision marker: {marker}")
    revision = marker.read_text(encoding="utf-8").strip()
    if not revision:
        raise RuntimeError(f"Reader snapshot has an empty revision marker: {marker}")
    return revision


def require_exact_episode_count(
    episodes: Sequence[Episode],
    *,
    expected: int,
) -> list[Episode]:
    if len(episodes) != expected:
        raise ValueError(
            f"Formal Qwen sanity requires exactly {expected} episodes, found {len(episodes)}"
        )
    return list(episodes)


def collect_unique_queries(episodes: Sequence[Episode]) -> tuple[int, list[UniqueQuery]]:
    """Deduplicate model-identical clean/distractor reads and fail on ID collisions."""

    groups: dict[str, UniqueQuery] = {}
    raw_query_count = 0
    for episode in episodes:
        query_ordinal = 0
        for turn_index, turn in enumerate(episode.turns):
            if not turn.calls_reader:
                continue
            query = turn.query
            if query is None:  # pragma: no cover - Episode schema already enforces this
                raise ValueError(f"{episode.episode_id} turn {turn_index} has no query payload")
            if query.comparison_id is None:
                raise ValueError(f"{episode.episode_id} query {query_ordinal} has no comparison_id")

            candidate = UniqueQuery(
                comparison_id=query.comparison_id,
                query=query,
                query_ordinal=query_ordinal,
                topic=episode.topic,
                entity_id=episode.entity_id,
                template_id=episode.template_id,
                template_family=episode.template_family,
            )
            existing = groups.get(query.comparison_id)
            if existing is None:
                existing = candidate
                groups[query.comparison_id] = existing
            elif existing.audit_signature != candidate.audit_signature:
                raise ValueError(
                    f"comparison_id {query.comparison_id!r} has inconsistent payload/target metadata"
                )

            existing.members.append(
                QueryMember(
                    episode_id=episode.episode_id,
                    query_id=f"{episode.episode_id}:q{query_ordinal}",
                    pair_id=episode.pair_id,
                    counterfactual_episode_id=episode.counterfactual_episode_id,
                    distractor_pair_id=episode.distractor_pair_id,
                    distractor_episode_id=episode.distractor_episode_id,
                    distractor_variant=(
                        episode.distractor_variant.value
                        if episode.distractor_variant is not None
                        else None
                    ),
                    turn_index=turn_index,
                    turn_type=turn.type.value,
                )
            )
            raw_query_count += 1
            query_ordinal += 1

    if not groups:
        raise ValueError("Qwen sanity dataset contains no queries")
    for comparison_id, group in groups.items():
        if len(group.members) > 2:
            raise ValueError(
                f"comparison_id {comparison_id!r} has {len(group.members)} members; expected at most two"
            )
        if len(group.members) == 2:
            variants = {member.distractor_variant for member in group.members}
            if variants != {"clean", "distractor"}:
                raise ValueError(
                    f"comparison_id {comparison_id!r} duplicates are not one clean/distractor pair: "
                    f"{sorted(str(value) for value in variants)}"
                )
    return raw_query_count, list(groups.values())


def validate_query_inventory(
    raw_query_count: int,
    unique_queries: Sequence[UniqueQuery],
    *,
    expected_raw_queries: int | None,
    expected_comparison_queries: int | None,
    expected_target_position_count: int | None,
) -> None:
    if expected_raw_queries is not None and raw_query_count != expected_raw_queries:
        raise ValueError(
            f"Raw query inventory mismatch: {raw_query_count} != {expected_raw_queries}"
        )
    if expected_comparison_queries is not None and len(unique_queries) != expected_comparison_queries:
        raise ValueError(
            "Comparison-query inventory mismatch: "
            f"{len(unique_queries)} != {expected_comparison_queries}"
        )
    if expected_target_position_count is not None:
        actual = Counter(item.query.target_index for item in unique_queries)
        expected = {index: expected_target_position_count for index in range(4)}
        if dict(actual) != expected:
            raise ValueError(
                f"Unique target-position inventory mismatch: {dict(sorted(actual.items()))} != {expected}"
            )


def model_input_inventory(
    unique_queries: Sequence[UniqueQuery],
) -> dict[str, Counter[int]]:
    """Group comparison reads by the exact blank-image prompt seen by Qwen."""

    groups: dict[str, Counter[int]] = defaultdict(Counter)
    for item in unique_queries:
        rendered = format_mcq_query(item.query.text, item.query.choices)
        groups[input_sha256(rendered)][item.query.target_index] += 1
    return groups


def validate_model_input_inventory(
    unique_queries: Sequence[UniqueQuery],
    *,
    expected_model_inputs: int | None,
) -> None:
    """Fail closed on the full-profile query-only counterfactual certificate."""

    if expected_model_inputs is None:
        return
    groups = model_input_inventory(unique_queries)
    if len(groups) != expected_model_inputs:
        raise ValueError(
            f"Model-input inventory mismatch: {len(groups)} != {expected_model_inputs}"
        )
    for input_hash, target_counts in groups.items():
        count = sum(target_counts.values())
        if any(target_counts[index] * 4 != count for index in range(4)):
            raise ValueError(
                "Each model-visible input must pair with all four target positions equally: "
                f"input_sha256={input_hash}, counts={dict(sorted(target_counts.items()))}"
            )


def prediction_record(
    item: UniqueQuery,
    *,
    blank_result: ChoiceScoreOutput,
    oracle_result: ChoiceScoreOutput,
) -> dict[str, Any]:
    query = item.query
    choices = list(query.choices)
    target_text = query.target
    rendered = format_mcq_query(query.text, query.choices)
    oracle_prompt = f"Current preference memory: {target_text}\n{rendered}"
    members = sorted(
        item.members,
        key=lambda member: (
            {"clean": 0, "distractor": 1, "unpaired": 2}.get(member.distractor_variant or "", 3),
            member.episode_id,
        ),
    )
    representative = members[0]
    blank_index = blank_result.predicted_index
    oracle_index = oracle_result.predicted_index
    return {
        "episode_id": representative.episode_id,
        "episode_ids": [member.episode_id for member in members],
        "query_id": representative.query_id,
        "query_ids": [member.query_id for member in members],
        "comparison_id": item.comparison_id,
        "pair_ids": [member.pair_id for member in members],
        "counterfactual_episode_ids": [member.counterfactual_episode_id for member in members],
        "distractor_pair_id": representative.distractor_pair_id,
        "distractor_episode_ids": [member.distractor_episode_id for member in members],
        "distractor_variants": [member.distractor_variant for member in members],
        "member_count": len(members),
        "topic": item.topic,
        "entity_id": item.entity_id,
        "template_id": item.template_id,
        "template_family": item.template_family,
        "query_ordinal": item.query_ordinal,
        "turn_index": representative.turn_index,
        "turn_type": representative.turn_type,
        "member_turn_indices": [member.turn_index for member in members],
        "member_turn_types": [member.turn_type for member in members],
        "query_text": query.text,
        "choices": choices,
        "target_index": query.target_index,
        "target": target_text,
        "target_text": target_text,
        "target_is_clear": target_text == CLEAR_TARGET,
        "candidate_has_clear_sentinel": CLEAR_TARGET in choices,
        "input_sha256": input_sha256(rendered),
        "oracle_input_sha256": input_sha256(oracle_prompt),
        "blank_predicted_index": blank_index,
        "blank_predicted_text": choices[blank_index],
        "blank_choice_mean_nll": list(blank_result.mean_nll),
        "blank_correct": blank_index == query.target_index,
        "oracle_predicted_index": oracle_index,
        "oracle_predicted_text": choices[oracle_index],
        "oracle_choice_mean_nll": list(oracle_result.mean_nll),
        "oracle_correct": oracle_index == query.target_index,
    }


def _accuracy_bucket(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    blank_correct = sum(bool(record["blank_correct"]) for record in records)
    oracle_correct = sum(bool(record["oracle_correct"]) for record in records)
    return {
        "queries": count,
        "raw_queries": sum(int(record["member_count"]) for record in records),
        "blank_correct": blank_correct,
        "blank_accuracy": blank_correct / count,
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / count,
    }


def _breakdown(
    records: Sequence[dict[str, Any]],
    *,
    key,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(key(record))].append(record)
    return {name: _accuracy_bucket(grouped[name]) for name in sorted(grouped)}


def build_summary(
    records: Sequence[dict[str, Any]],
    *,
    episodes: int,
    raw_query_count: int,
    dataset_sha256: str,
    reader_revision: str,
    predictions_path: Path,
    oracle_threshold: float,
    query_only_ceiling: float,
    elapsed_seconds: float,
    peak_vram_gib: float,
    requested_limit: int,
    expected_raw_queries: int | None,
    expected_comparison_queries: int | None,
    expected_target_position_count: int | None,
    expected_model_inputs: int | None,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot summarize an empty prediction set")
    oracle_accuracy = sum(bool(record["oracle_correct"]) for record in records) / len(records)
    query_only_accuracy = sum(bool(record["blank_correct"]) for record in records) / len(records)
    passed = oracle_accuracy >= oracle_threshold and query_only_accuracy <= query_only_ceiling

    def counts(field_name: str) -> dict[str, int]:
        values = Counter(str(record[field_name]) for record in records)
        return dict(sorted(values.items()))

    raw_position_counts: Counter[str] = Counter()
    for record in records:
        raw_position_counts[str(record["target_index"])] += int(record["member_count"])
    input_targets: dict[str, Counter[int]] = defaultdict(Counter)
    for record in records:
        input_targets[str(record["input_sha256"])][int(record["target_index"])] += 1
    input_target_patterns = Counter(
        ",".join(str(target_counts[index]) for index in range(4))
        for target_counts in input_targets.values()
    )
    input_deviations = [
        abs(target_counts[index] / sum(target_counts.values()) - 0.25)
        for target_counts in input_targets.values()
        for index in range(4)
    ]
    return {
        "schema_version": 3,
        "episodes": episodes,
        "requested_episode_limit": requested_limit,
        "raw_query_count": raw_query_count,
        "comparison_query_count": len(records),
        "clean_distractor_duplicates_removed": raw_query_count - len(records),
        "unique_model_input_count": len(input_targets),
        "model_input_target_pattern_counts": dict(sorted(input_target_patterns.items())),
        "max_model_input_target_share_deviation": max(input_deviations, default=0.0),
        "comparison_member_count_counts": counts("member_count"),
        "expected_inventory": {
            "raw_queries": expected_raw_queries,
            "comparison_queries": expected_comparison_queries,
            "queries_per_target_position": expected_target_position_count,
            "model_inputs": expected_model_inputs,
        },
        "dataset_sha256": dataset_sha256,
        "reader_revision": reader_revision,
        "device": device,
        "dtype": dtype,
        "predictions_jsonl": str(predictions_path.resolve()),
        "predictions_sha256": sha256_file(predictions_path),
        "oracle_text_accuracy": oracle_accuracy,
        "oracle_threshold": oracle_threshold,
        "query_only_blank_accuracy": query_only_accuracy,
        "query_only_ceiling": query_only_ceiling,
        "target_position_counts": counts("target_index"),
        "raw_target_position_counts": dict(sorted(raw_position_counts.items())),
        "target_position_breakdown": _breakdown(records, key=lambda record: record["target_index"]),
        "clear_breakdown": _breakdown(
            records,
            key=lambda record: "clear" if record["target_is_clear"] else "active",
        ),
        "sentinel_presence_breakdown": _breakdown(
            records,
            key=lambda record: (
                "contains_clear_sentinel"
                if record["candidate_has_clear_sentinel"]
                else "no_clear_sentinel"
            ),
        ),
        "blank_prediction_index_counts": counts("blank_predicted_index"),
        "blank_prediction_text_counts": counts("blank_predicted_text"),
        "oracle_prediction_index_counts": counts("oracle_predicted_index"),
        "oracle_prediction_text_counts": counts("oracle_predicted_text"),
        "passed": passed,
        "elapsed_seconds": elapsed_seconds,
        "peak_vram_gib": peak_vram_gib,
        "blank_image": BLANK_IMAGE_SPEC,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen-Qwen oracle-text and query-only synthetic-data gates")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--expected-raw-queries", type=int)
    parser.add_argument("--expected-comparison-queries", type=int)
    parser.add_argument("--expected-target-position-count", type=int)
    parser.add_argument("--expected-model-inputs", type=int)
    parser.add_argument("--oracle-threshold", type=float, default=0.95)
    parser.add_argument("--query-only-ceiling", type=float, default=0.30)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("Qwen data sanity requires CUDA.")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    reader_revision = locked_revision(args.reader)
    all_episodes = read_jsonl(args.dataset)
    try:
        episodes = require_exact_episode_count(all_episodes, expected=args.limit)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    raw_query_count, unique_queries = collect_unique_queries(episodes)
    validate_query_inventory(
        raw_query_count,
        unique_queries,
        expected_raw_queries=args.expected_raw_queries,
        expected_comparison_queries=args.expected_comparison_queries,
        expected_target_position_count=args.expected_target_position_count,
    )
    validate_model_input_inventory(
        unique_queries,
        expected_model_inputs=args.expected_model_inputs,
    )

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

    records: list[dict[str, Any]] = []
    args.predictions_jsonl.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with args.predictions_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for item in unique_queries:
            query = item.query
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
            record = prediction_record(
                item,
                blank_result=blank_result,
                oracle_result=oracle_result,
            )
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    report = build_summary(
        records,
        episodes=len(episodes),
        raw_query_count=raw_query_count,
        dataset_sha256=sha256_file(args.dataset),
        reader_revision=reader_revision,
        predictions_path=args.predictions_jsonl,
        oracle_threshold=args.oracle_threshold,
        query_only_ceiling=args.query_only_ceiling,
        elapsed_seconds=time.monotonic() - started,
        peak_vram_gib=torch.cuda.max_memory_allocated(device) / 2**30,
        requested_limit=args.limit,
        expected_raw_queries=args.expected_raw_queries,
        expected_comparison_queries=args.expected_comparison_queries,
        expected_target_position_count=args.expected_target_position_count,
        expected_model_inputs=args.expected_model_inputs,
        device=str(device),
        dtype=str(dtype).removeprefix("torch."),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
