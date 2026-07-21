from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import REVERSE_CYCLIC4, read_jsonl as read_synthetic_jsonl  # noqa: E402
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_RESIZE_CONTRACT,
    deterministic_qwen_reader_resize,
    qwen3vl_choice_nll,
)
from vision_memory.repro import canonical_object_sha256, configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import format_mcq_query  # noqa: E402


METHOD = "qwen_full_event_history"
TEXT_ONLY_METHOD = "qwen_full_event_history_text_only"
SCHEMA_VERSION = "vision_memory.qwen_full_event_history_predictions.v1"
REPORT_SCHEMA_VERSION = "vision_memory.qwen_full_event_history_report.v1"
SCIENTIFIC_PAYLOAD_SCHEMA_VERSION = "vision_memory.qwen_full_event_history_scientific_payload.v1"
EXPECTED_READER_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
INPUT_MODES = ("blank_image", "text_only")
CONDITIONS = ("standard", "reset", "shuffle", "state_swap")
MICRO_EPISODE_PREFIXES = ("r3-set8-", "r3-transition-")
BLANK_IMAGE_SHAPE = (3, 1024, 1024)
BLANK_IMAGE_VALUE = 0.5


@dataclass(frozen=True)
class HistoryQuery:
    metadata: dict[str, Any]
    query: str
    choices: tuple[str, str, str, str]
    target_index: int
    history: tuple[str, ...]


@dataclass(frozen=True)
class HistoryIntervention:
    history: tuple[str, ...]
    donor_target_text: str | None = None
    donor_episode_id: str | None = None


@dataclass(frozen=True)
class ChoiceView:
    metadata: dict[str, Any]
    query: str
    choices: tuple[str, str, str, str]
    target_index: int
    history: tuple[str, ...]
    donor_target_index: int | None
    donor_episode_id: str | None


@dataclass(frozen=True)
class ContextAudit:
    chat_prompt_sha256: str
    prompt_token_count: int
    context_limit: int
    choice_context_token_counts: tuple[int, ...]
    choice_target_token_counts: tuple[int, ...]


@dataclass(frozen=True)
class TextChoiceScore:
    mean_nll: tuple[float, ...]
    predicted_index: int


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


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


def raw_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            values.append(value)
    return values


def _target_index_in_choices(target_text: str, choices: Sequence[str]) -> int | None:
    try:
        return tuple(choices).index(target_text)
    except ValueError:
        return None


def _read_form(episode_id: str, template_id: str, has_mixed_prefix: bool) -> str:
    identifiers = f"{episode_id}\0{template_id}".casefold()
    if has_mixed_prefix or "mixed" in identifiers:
        return "mixed"
    if "separate" in identifiers:
        return "separate"
    return "separate"


def synthetic_queries(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    """Yield query snapshots containing only the preceding model-visible event texts.

    The tuple is copied at each query boundary, so later events cannot enter an earlier
    query. A mixed turn appends its event before the snapshot and therefore implements the
    locked oracle route: write first, then read. Query text, choices, labels, answers and
    episode metadata never enter ``history``.
    """

    episodes = read_synthetic_jsonl(path)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive when provided.")
        episodes = episodes[:limit]
    for episode in episodes:
        history: list[str] = []
        query_number = 0
        last_transition = "unknown"
        previous_target_text: str | None = None
        event_count_since_query = 0
        noop_count_since_query = 0
        has_mixed_prefix = False
        for turn_index, turn in enumerate(episode.turns):
            if turn.calls_updater:
                if turn.event_text is None or turn.event_kind is None:
                    raise RuntimeError("Schema invariant violated: updater turn lacks event text/kind.")
                history.append(turn.event_text)
                last_transition = turn.event_kind.value
                event_count_since_query += 1
                noop_count_since_query += int(turn.event_kind.value == "noop")
                has_mixed_prefix = has_mixed_prefix or turn.type.value == "mixed"
            if turn.calls_reader:
                if turn.query is None:
                    raise RuntimeError("Schema invariant violated: Reader turn lacks query payload.")
                query = turn.query
                target_text = query.choices[query.target_index]
                stale_index = None
                if previous_target_text is not None and previous_target_text != target_text:
                    stale_index = _target_index_in_choices(previous_target_text, query.choices)
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
                        episode.distractor_variant.value if episode.distractor_variant is not None else None
                    ),
                    "query_comparison_id": query.comparison_id,
                    "semantic_group_id": episode.semantic_group_id,
                    "topic": episode.topic,
                    "template_id": episode.template_id,
                    "subtype": last_transition,
                    "form": _read_form(episode.episode_id, episode.template_id, has_mixed_prefix),
                    "split": episode.split,
                    "ood_group": episode.ood_group,
                    "protocol": "synthetic",
                    "update_count": sum(item.calls_updater for item in episode.turns[: turn_index + 1]),
                    "route": "event_then_query" if turn.calls_updater else "query_read_only",
                    "query_turn_type": turn.type.value,
                    "probe_role": "immediate" if turn.calls_updater else "delayed",
                    "event_latency_seconds": 0.0,
                    "updater_calls_since_query": event_count_since_query,
                    "noop_events_since_query": noop_count_since_query,
                    "noop_events_applied_since_query": noop_count_since_query,
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
        if limit < 1:
            raise ValueError("--limit must be positive when provided.")
        records = records[:limit]
    for record in records:
        model_input = record["model_input"]
        history: list[str] = []
        query_number = 0
        event_count_since_query = 0
        noop_count_since_query = 0
        for turn in model_input["turns"]:
            if turn["type"] == "event":
                history.append(str(turn["text"]))
                event_count_since_query += 1
                noop_count_since_query += int(turn.get("event_type") == "noop")
            elif turn["type"] == "query":
                label = record["label"]
                yield {
                    "metadata": {
                        "episode_id": model_input["sample_id"],
                        "query_id": f"{model_input['sample_id']}:q{query_number}",
                        "query_ordinal": query_number,
                        "base_pair_id": model_input["base_pair_id"],
                        "counterfactual_episode_id": None,
                        "semantic_group_id": model_input.get("semantic_group_id"),
                        "topic": model_input["topic"],
                        "subtype": model_input["form"],
                        "form": model_input["form"],
                        "split": model_input["split"],
                        "ood_group": None,
                        "protocol": model_input["protocol"],
                        "forced_write_k": model_input["forced_write_k"],
                        "route": "query_read_only",
                        "query_turn_type": "query",
                        "probe_role": "delayed",
                        "event_latency_seconds": 0.0,
                        "updater_calls_since_query": event_count_since_query,
                        "noop_events_since_query": noop_count_since_query,
                        "noop_events_applied_since_query": noop_count_since_query,
                        "noop_policy": "keep",
                    },
                    "query": turn["text"],
                    "choices": tuple(turn["options"]),
                    "target_index": int(label["target_index"]),
                    "history": tuple(history),
                }
                query_number += 1
                event_count_since_query = 0
                noop_count_since_query = 0
            else:
                raise ValueError(f"Unsupported PrefEval turn type: {turn['type']!r}")


def _as_history_query(item: Mapping[str, Any]) -> HistoryQuery:
    choices = tuple(str(value) for value in item["choices"])
    if len(choices) != 4 or len(set(choices)) != 4:
        raise ValueError("Full-history baseline requires four distinct choices.")
    target_index = int(item["target_index"])
    if not 0 <= target_index < 4:
        raise ValueError("target_index must be in [0, 3].")
    history = tuple(str(value) for value in item["history"])
    if any(not value.strip() for value in history):
        raise ValueError("Event history cannot contain empty entries.")
    return HistoryQuery(
        metadata=dict(item["metadata"]),
        query=str(item["query"]),
        choices=choices,  # type: ignore[arg-type]
        target_index=target_index,
        history=history,
    )


def _different_target_derangement(items: Sequence[HistoryQuery], recipients: Sequence[int], seed: int) -> dict[int, int]:
    buckets: dict[str, list[int]] = {}
    for recipient in recipients:
        item = items[recipient]
        target_text = item.choices[item.target_index]
        buckets.setdefault(target_text, []).append(recipient)
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
        raise RuntimeError("Internal history-shuffle derangement construction failed.")
    return pairs


def intervention_histories(
    items: Sequence[HistoryQuery],
    *,
    condition: str,
    seed: int,
) -> list[HistoryIntervention]:
    """Apply causal history interventions before candidate-view expansion.

    Expanding views afterwards guarantees that all four reverse-cyclic views of one query
    receive the exact same donor history.
    """

    if condition == "standard":
        return [HistoryIntervention(item.history) for item in items]
    if condition == "reset":
        return [HistoryIntervention(()) for _ in items]
    if condition == "shuffle":
        groups: dict[tuple[Any, ...], list[int]] = {}
        for index, item in enumerate(items):
            key = (
                item.metadata.get("split"),
                item.metadata.get("protocol", "synthetic"),
                item.metadata.get("forced_write_k"),
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
            HistoryIntervention(
                history=items[donor].history,
                # A shuffle is an unrelated-history control and may cross topic/
                # choice vocabularies in the formal data. Donor-answer attribution
                # is meaningful only for the matched counterfactual state swap.
                donor_target_text=None,
                donor_episode_id=str(items[donor].metadata.get("episode_id")),
            )
            for donor in order
        ]
    if condition == "state_swap":
        by_episode_ordinal = {
            (item.metadata.get("episode_id"), item.metadata.get("query_ordinal", 0)): item
            for item in items
        }
        states: list[HistoryIntervention] = []
        for item in items:
            donor_key = (
                item.metadata.get("counterfactual_episode_id"),
                item.metadata.get("query_ordinal", 0),
            )
            donor = by_episode_ordinal.get(donor_key)
            if donor is None:
                raise ValueError(f"state_swap requires a matched counterfactual query; missing donor {donor_key!r}")
            states.append(
                HistoryIntervention(
                    history=donor.history,
                    donor_target_text=donor.choices[donor.target_index],
                    donor_episode_id=str(donor.metadata.get("episode_id")),
                )
            )
        return states
    raise ValueError(f"Unknown condition: {condition!r}")


def expand_reverse_cyclic_views(
    item: HistoryQuery,
    intervention: HistoryIntervention,
) -> tuple[ChoiceView, ...]:
    target_text = item.choices[item.target_index]
    base_query_id = str(item.metadata["query_id"])
    views: list[ChoiceView] = []
    for view_index, permutation in enumerate(REVERSE_CYCLIC4):
        choices = tuple(item.choices[index] for index in permutation)
        donor_target_index = None
        if intervention.donor_target_text is not None:
            donor_target_index = _target_index_in_choices(intervention.donor_target_text, choices)
            if donor_target_index is None:
                raise ValueError(
                    "Intervention donor target is absent from recipient choices: "
                    f"{intervention.donor_target_text!r} not in {choices!r}"
                )
        metadata = {
            **item.metadata,
            "base_query_id": base_query_id,
            "query_id": f"{base_query_id}:reverse{view_index}",
            "choice_view_family": "reverse-cyclic4",
            "choice_view_index": view_index,
        }
        stale_target_text = metadata.get("stale_target_text")
        if isinstance(stale_target_text, str):
            stale_target_index = _target_index_in_choices(stale_target_text, choices)
            metadata["stale_target_mapped"] = stale_target_index is not None
            if stale_target_index is None:
                metadata.pop("stale_target_index", None)
            else:
                metadata["stale_target_index"] = stale_target_index
        views.append(
            ChoiceView(
                metadata=metadata,
                query=item.query,
                choices=choices,  # type: ignore[arg-type]
                target_index=choices.index(target_text),
                history=intervention.history,
                donor_target_index=donor_target_index,
                donor_episode_id=intervention.donor_episode_id,
            )
        )
    return tuple(views)


def render_history(history: Sequence[str]) -> str:
    body = "<empty>" if not history else "\n".join(f"- {event}" for event in history)
    return f"Conversation memory:\n{body}"


def method_prompt(method: str, item: Mapping[str, Any], *, history: Sequence[str] | None = None) -> str:
    if method not in {METHOD, TEXT_ONLY_METHOD}:
        raise ValueError(f"Unknown full-history method: {method!r}.")
    selected_history = tuple(item["history"] if history is None else history)
    query = format_mcq_query(str(item["query"]), tuple(item["choices"]))
    return f"{render_history(selected_history)}\n{query}"


def _tokenizer_ids(tokenizer: Any, text: str) -> Tensor:
    encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else getattr(encoded, "input_ids", None)
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise TypeError("Qwen tokenizer must return input_ids with shape [1, sequence].")
    return input_ids


def _joint_prompt_target_tokenization(processor: Any, prompt: str, target: str) -> tuple[str, Tensor]:
    tokenizer = processor.tokenizer
    prompt_ids = _tokenizer_ids(tokenizer, prompt)
    joint_text = prompt + target
    joint_ids = _tokenizer_ids(tokenizer, joint_text)
    prompt_length = int(prompt_ids.shape[1])
    if joint_ids.shape[1] <= prompt_length:
        raise ValueError("Choice target tokenized to an empty continuation.")
    if not torch.equal(joint_ids[:, :prompt_length], prompt_ids):
        raise RuntimeError("Appending a choice retokenized the chat-template prefix.")
    return joint_text, joint_ids[:, prompt_length:]


def _chat_prompt(processor: Any, prompt: str, input_mode: str) -> str:
    if input_mode == "blank_image":
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
    elif input_mode == "text_only":
        content = [{"type": "text", "text": prompt}]
    else:
        raise ValueError(f"Unknown input mode: {input_mode!r}")
    return str(
        processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def _context_limit(model: Any, tokenizer: Any) -> int:
    candidates: list[int] = []
    for config in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        value = getattr(config, "max_position_embeddings", None)
        if isinstance(value, int) and 1 < value < 10**9:
            candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 1 < tokenizer_limit < 10**9:
        candidates.append(tokenizer_limit)
    if not candidates:
        raise RuntimeError("Cannot establish a finite Qwen context limit; refusing to score.")
    return min(candidates)


def _batch_input_ids(batch: Any) -> Tensor:
    input_ids = batch.get("input_ids") if isinstance(batch, Mapping) else getattr(batch, "input_ids", None)
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise TypeError("Prepared Qwen batch must contain input_ids with shape [1, sequence].")
    return input_ids


def audit_context(
    *,
    model: Any,
    processor: Any,
    prompt: str,
    choices: Sequence[str],
    input_mode: str,
    resized_blank_image: Tensor | None,
) -> ContextAudit:
    chat_prompt = _chat_prompt(processor, prompt, input_mode)
    context_limit = _context_limit(model, processor.tokenizer)
    prefix_counts: set[int] = set()
    context_counts: list[int] = []
    target_counts: list[int] = []
    for choice in choices:
        joint_text, target_ids = _joint_prompt_target_tokenization(processor, chat_prompt, choice)
        if input_mode == "blank_image":
            if resized_blank_image is None:
                raise ValueError("blank_image context audit requires the locked resized image.")
            batch = processor(
                text=[joint_text],
                images=[resized_blank_image],
                return_tensors="pt",
                do_rescale=False,
                do_resize=False,
            )
        else:
            batch = processor.tokenizer(joint_text, add_special_tokens=False, return_tensors="pt")
        input_ids = _batch_input_ids(batch)
        target_length = int(target_ids.shape[1])
        if input_ids.shape[1] <= target_length:
            raise RuntimeError("Joint Qwen input contains no non-target chat prefix.")
        if not torch.equal(input_ids[:, -target_length:].cpu(), target_ids.cpu()):
            raise RuntimeError("Prepared Qwen input changed the jointly tokenized choice suffix.")
        context_count = int(input_ids.shape[1])
        if context_count > context_limit:
            raise RuntimeError(
                f"Qwen context overflow: required {context_count} tokens, finite limit is {context_limit}; "
                "history truncation is forbidden."
            )
        prefix_counts.add(context_count - target_length)
        context_counts.append(context_count)
        target_counts.append(target_length)
    if len(prefix_counts) != 1:
        raise RuntimeError(f"Choice-independent Reader prefix length drifted: {sorted(prefix_counts)}")
    return ContextAudit(
        chat_prompt_sha256=sha256_text(chat_prompt),
        prompt_token_count=prefix_counts.pop(),
        context_limit=context_limit,
        choice_context_token_counts=tuple(context_counts),
        choice_target_token_counts=tuple(target_counts),
    )


def _hidden_states(output: Any) -> Tensor:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    raise TypeError(f"Unsupported Qwen base-model output: {type(output)!r}")


def qwen3vl_text_choice_nll(
    *,
    model: Any,
    processor: Any,
    query: str,
    choices: Sequence[str],
    device: torch.device,
    deterministic_ce: bool,
) -> TextChoiceScore:
    """True text-only joint-continuation scorer used only for micro sensitivity.

    No image placeholder, pixel tensor, image grid or multimodal token type is constructed.
    Option scores remain negative mean assistant-token log likelihood, identical in meaning to
    the locked multimodal scorer.
    """

    if len(choices) != 4 or len(set(choices)) != 4:
        raise ValueError("Text-only MCQ scoring requires four distinct choices.")
    chat_prompt = _chat_prompt(processor, query, "text_only")
    scores: list[float] = []
    with torch.no_grad():
        for choice in choices:
            joint_text, target_ids = _joint_prompt_target_tokenization(processor, chat_prompt, choice)
            batch = processor.tokenizer(joint_text, add_special_tokens=False, return_tensors="pt")
            input_ids = _batch_input_ids(batch).to(device)
            attention_mask = batch.get("attention_mask")
            if not isinstance(attention_mask, Tensor):
                attention_mask = torch.ones_like(input_ids)
            else:
                attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)
            target_length = int(target_ids.shape[1])
            if not torch.equal(input_ids[:, -target_length:], target_ids):
                raise RuntimeError("Text-only tokenizer changed the joint choice suffix.")
            prefix_length = int(input_ids.shape[1]) - target_length
            output = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = _hidden_states(output)
            positions = torch.arange(prefix_length - 1, prefix_length + target_length - 1, device=device)
            logits = model.lm_head(hidden.index_select(dim=1, index=positions))
            flat_logits = logits.float().reshape(-1, logits.shape[-1])
            flat_targets = target_ids.reshape(-1)
            if deterministic_ce:
                target_scores = flat_logits.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
                loss = (torch.logsumexp(flat_logits, dim=-1) - target_scores).mean()
            else:
                loss = F.cross_entropy(flat_logits, flat_targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Text-only choice scorer produced NaN or Inf.")
            scores.append(float(loss.item()))
    predicted_index = min(range(4), key=scores.__getitem__)
    return TextChoiceScore(mean_nll=tuple(scores), predicted_index=predicted_index)


def _history_token_count(tokenizer: Any, history_text: str) -> int:
    return int(_tokenizer_ids(tokenizer, history_text).shape[1])


def _nll_margin(scores: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in scores)
    return ordered[1] - ordered[0]


def _is_micro_suite(items: Sequence[HistoryQuery]) -> bool:
    return bool(items) and all(
        str(item.metadata.get("episode_id", "")).startswith(MICRO_EPISODE_PREFIXES) for item in items
    )


def _scientific_row(row: Mapping[str, Any]) -> dict[str, Any]:
    runtime_fields = {
        "replica_id",
        "event_latency_seconds",
        "reader_latency_seconds",
        "query_latency_seconds",
        "latency_seconds",
        "peak_reader_vram_gib",
        "peak_vram_gib",
    }
    return {key: value for key, value in row.items() if key not in runtime_fields}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen Qwen full-event-history baseline with R3 causal interventions"
    )
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--format", choices=("synthetic", "prefeval"), default="synthetic")
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=(METHOD, TEXT_ONLY_METHOD), default=METHOD)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--probe-role", choices=("all", "delayed"), default="all")
    parser.add_argument("--choice-view-family", choices=("reverse-cyclic4",), default="reverse-cyclic4")
    parser.add_argument("--input-mode", choices=INPUT_MODES, default="blank_image")
    parser.add_argument(
        "--micro-sensitivity",
        action="store_true",
        help="Required for true text-only input and accepted only on the preregistered R3 micro suites.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replica-id", choices=("A", "B"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--strict-determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enabled by default and mandatory for this formal baseline.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    if args.output.exists() or report_path.exists():
        raise SystemExit(f"Refusing to overwrite existing baseline artifact: {args.output} / {report_path}")
    if not args.strict_determinism:
        raise SystemExit("qwen_full_event_history requires strict deterministic evaluation.")
    if len(set(args.conditions)) != len(args.conditions):
        raise SystemExit("--conditions must not contain duplicates.")
    if args.format == "prefeval" and "state_swap" in args.conditions:
        raise SystemExit("state_swap requires synthetic matched counterfactual episodes.")

    # This call must remain before the first CUDA availability/capability/device probe.
    strict_determinism = configure_strict_cuda_determinism(seed=args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("Qwen full-event-history evaluation requires CUDA.")

    reader_revision = locked_revision(args.reader)
    if reader_revision != EXPECTED_READER_REVISION:
        raise SystemExit(
            f"Reader revision drift: expected {EXPECTED_READER_REVISION}, observed {reader_revision}."
        )
    source = synthetic_queries if args.format == "synthetic" else prefeval_queries
    items = [_as_history_query(value) for value in source(args.episodes, args.limit)]
    if args.probe_role == "delayed":
        items = [item for item in items if item.metadata.get("probe_role") == "delayed"]
    if not items:
        raise SystemExit("No query states remain after applying the requested filters.")
    if args.input_mode == "text_only":
        if not args.micro_sensitivity or args.format != "synthetic" or not _is_micro_suite(items):
            raise SystemExit(
                "text_only is restricted to explicit --micro-sensitivity runs on R3 Set8/Transition16."
            )
        if args.method != TEXT_ONLY_METHOD:
            raise SystemExit(f"text_only sensitivity must use --method {TEXT_ONLY_METHOD}.")
    elif args.micro_sensitivity:
        raise SystemExit("--micro-sensitivity is reserved for --input-mode text_only.")
    elif args.method != METHOD:
        raise SystemExit(f"The formal blank-image baseline must use --method {METHOD}.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("The audited baseline requires a CUDA Reader device.")
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

    blank_image: Tensor | None = None
    resized_blank_image: Tensor | None = None
    if args.input_mode == "blank_image":
        blank_image = torch.full(BLANK_IMAGE_SHAPE, BLANK_IMAGE_VALUE, device=device, dtype=torch.float32)
        resized_blank_image = deterministic_qwen_reader_resize(
            blank_image,
            contract=R3_QWEN_READER_RESIZE_CONTRACT,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    scientific_rows: list[dict[str, Any]] = []
    prediction_count = 0
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        for condition in args.conditions:
            interventions = intervention_histories(items, condition=condition, seed=args.seed)
            for item, intervention in zip(items, interventions, strict=True):
                for view in expand_reverse_cyclic_views(item, intervention):
                    prompt_item = {
                        "query": view.query,
                        "choices": view.choices,
                        "history": view.history,
                    }
                    prompt = method_prompt(args.method, prompt_item)
                    history_text = render_history(view.history)
                    context = audit_context(
                        model=reader,
                        processor=processor,
                        prompt=prompt,
                        choices=view.choices,
                        input_mode=args.input_mode,
                        resized_blank_image=resized_blank_image,
                    )
                    torch.cuda.synchronize(device)
                    started = time.monotonic()
                    if args.input_mode == "blank_image":
                        if blank_image is None:
                            raise RuntimeError("Internal blank-image initialization failure.")
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
                    else:
                        result = qwen3vl_text_choice_nll(
                            model=reader,
                            processor=processor,
                            query=prompt,
                            choices=view.choices,
                            device=device,
                            deterministic_ce=True,
                        )
                    torch.cuda.synchronize(device)
                    reader_latency = time.monotonic() - started
                    peak_vram = torch.cuda.max_memory_allocated(device) / 2**30
                    dynamic_state_bytes = len(history_text.encode("utf-8"))
                    row = {
                        "schema_version": SCHEMA_VERSION,
                        **view.metadata,
                        "method": args.method,
                        "input_mode": args.input_mode,
                        "micro_sensitivity": args.micro_sensitivity,
                        "seed": args.seed,
                        "diffusion_seed": 0,
                        "replica_id": args.replica_id,
                        "recurrence_mode": "full_event_text_history",
                        "condition": condition,
                        "prediction_index": result.predicted_index,
                        "target_index": view.target_index,
                        "choices": list(view.choices),
                        "prediction_text": view.choices[result.predicted_index],
                        "target_text": view.choices[view.target_index],
                        "choice_mean_nll": list(result.mean_nll),
                        "nll_margin": _nll_margin(result.mean_nll),
                        "history_sha256": sha256_text(history_text),
                        "prompt_sha256": sha256_text(prompt),
                        "chat_prompt_sha256": context.chat_prompt_sha256,
                        "history_event_count": len(view.history),
                        "history_token_count": _history_token_count(processor.tokenizer, history_text),
                        "history_utf8_bytes": dynamic_state_bytes,
                        "prompt_token_count": context.prompt_token_count,
                        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                        "choice_context_token_counts": list(context.choice_context_token_counts),
                        "choice_target_token_counts": list(context.choice_target_token_counts),
                        "context_limit": context.context_limit,
                        "context_truncated": False,
                        "reader_resize_contract": (
                            R3_QWEN_READER_RESIZE_CONTRACT if args.input_mode == "blank_image" else None
                        ),
                        "state_bytes": dynamic_state_bytes,
                        "constant_visual_input_bytes": (
                            0 if blank_image is None else int(blank_image.numel() * blank_image.element_size())
                        ),
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
                    scientific_rows.append(_scientific_row(row))
                    prediction_count += 1

    scientific_payload = {
        "schema_version": SCIENTIFIC_PAYLOAD_SCHEMA_VERSION,
        "records": scientific_rows,
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "method": args.method,
        "input_mode": args.input_mode,
        "micro_sensitivity": args.micro_sensitivity,
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "episodes": str(args.episodes.resolve()),
        "episodes_sha256": sha256_file(args.episodes),
        "git_commit": git_commit(),
        "query_states": len(items),
        "prediction_records": prediction_count,
        "conditions": list(args.conditions),
        "probe_role": args.probe_role,
        "choice_view_family": args.choice_view_family,
        "seed": args.seed,
        "replica_id": args.replica_id,
        "reader_revision": reader_revision,
        "reader_resize_contract": (
            R3_QWEN_READER_RESIZE_CONTRACT if args.input_mode == "blank_image" else None
        ),
        "blank_image": (
            None
            if args.input_mode == "text_only"
            else {
                "shape": list(BLANK_IMAGE_SHAPE),
                "float_value": BLANK_IMAGE_VALUE,
                "dtype": "float32",
                "bytes": 3 * 1024 * 1024 * 4,
            }
        ),
        "strict_determinism": strict_determinism,
        "deterministic_ce": True,
        "context_truncation_policy": "fail_closed",
        "history_contract": "events-and-noops-only; mixed-write-before-read; no-query-answer-ledger-label-future",
        "scientific_payload_sha256": canonical_object_sha256(scientific_payload),
        "peak_vram_gib": {str(device): torch.cuda.max_memory_allocated(device) / 2**30},
    }
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
