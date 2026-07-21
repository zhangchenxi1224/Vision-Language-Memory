from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.eval import qwen_text_baselines as baseline  # noqa: E402
from vision_memory.data import (  # noqa: E402
    Episode,
    EventKind,
    QuerySpec,
    Turn,
    TurnType,
    write_jsonl,
)


CHOICES = ("red", "blue", "green", "yellow")


def query(text: str, target_index: int, comparison_id: str) -> QuerySpec:
    return QuerySpec(
        text=text,
        choices=CHOICES,
        target_index=target_index,
        comparison_id=comparison_id,
    )


def episode(
    episode_id: str,
    *,
    target_index: int = 0,
    counterfactual_episode_id: str = "counterfactual",
) -> Episode:
    return Episode(
        episode_id=episode_id,
        split="gate",
        seed=2026,
        entity_id=f"entity-{episode_id}",
        template_id="r3-transition-mixed-gate-b",
        pair_id=f"pair-{episode_id}",
        counterfactual_episode_id=counterfactual_episode_id,
        topic="color",
        semantic_group_id=f"semantic-{episode_id}",
        turns=(
            Turn(type=TurnType.EVENT, event_kind=EventKind.SET, event_text="EVENT_ONE"),
            Turn(
                type=TurnType.MIXED,
                event_kind=EventKind.OVERWRITE,
                event_text="MIXED_EVENT",
                query=query("MIXED_QUERY", target_index, f"{episode_id}:immediate"),
            ),
            Turn(type=TurnType.EVENT, event_kind=EventKind.NOOP, event_text="FUTURE_NOOP"),
            Turn(
                type=TurnType.QUERY,
                query=query("DELAYED_QUERY", target_index, f"{episode_id}:delayed"),
            ),
        ),
    )


def item(name: str, target_index: int, *, counterfactual: str | None = None) -> baseline.HistoryQuery:
    return baseline.HistoryQuery(
        metadata={
            "episode_id": name,
            "query_id": f"{name}:q0",
            "query_ordinal": 0,
            "probe_role": "delayed",
            "split": "gate",
            "protocol": "synthetic",
            "noop_policy": "keep",
            "counterfactual_episode_id": counterfactual,
        },
        query="Which color?",
        choices=CHOICES,
        target_index=target_index,
        history=(f"event-{name}",),
    )


def test_event_prefix_is_write_before_read_and_future_safe(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    write_jsonl(path, [episode("r3-transition-overwrite-mixed-r0")])

    records = list(baseline.synthetic_queries(path, None))

    assert len(records) == 2
    assert records[0]["history"] == ("EVENT_ONE", "MIXED_EVENT")
    assert records[0]["metadata"]["route"] == "event_then_query"
    assert records[0]["metadata"]["probe_role"] == "immediate"
    assert records[1]["history"] == ("EVENT_ONE", "MIXED_EVENT", "FUTURE_NOOP")
    assert records[1]["metadata"]["probe_role"] == "delayed"
    for record in records:
        joined = "\n".join(record["history"])
        assert "QUERY" not in joined
        assert not any(choice in joined for choice in CHOICES)
        assert "ledger" not in joined.casefold()


def test_prompt_is_label_index_blind_and_reset_is_explicitly_empty() -> None:
    base = {"query": "Which color?", "choices": CHOICES, "history": ("I prefer red.",), "target_index": 0}
    changed_label = {**base, "target_index": 3}

    assert baseline.method_prompt(baseline.METHOD, base) == baseline.method_prompt(
        baseline.METHOD, changed_label
    )
    assert baseline.render_history(()) == "Conversation memory:\n<empty>"


def test_shuffle_is_different_target_and_view_stable() -> None:
    items = [item(f"e{index}", index) for index in range(4)]

    interventions = baseline.intervention_histories(items, condition="shuffle", seed=17)

    for source, intervention in zip(items, interventions, strict=True):
        assert intervention.donor_episode_id != source.metadata["episode_id"]
        assert intervention.donor_target_text is None
        views = baseline.expand_reverse_cyclic_views(source, intervention)
        assert len(views) == 4
        assert {view.donor_episode_id for view in views} == {intervention.donor_episode_id}
        assert {view.history for view in views} == {intervention.history}
        assert {view.target_index for view in views} == {0, 1, 2, 3}
        for view in views:
            assert view.choices[view.target_index] == source.choices[source.target_index]
            assert view.donor_target_index is None


def test_shuffle_allows_cross_vocabulary_donor_histories() -> None:
    first = item("cross-topic-a", 0)
    second = baseline.HistoryQuery(
        metadata={
            **item("cross-topic-b", 1).metadata,
            "query_id": "cross-topic-b:q0",
            "episode_id": "cross-topic-b",
        },
        query="Which style?",
        choices=("rustic", "modern", "formal", "minimal"),
        target_index=1,
        history=("The user prefers a modern style.",),
    )

    interventions = baseline.intervention_histories([first, second], condition="shuffle", seed=0)

    for source, intervention in zip((first, second), interventions, strict=True):
        views = baseline.expand_reverse_cyclic_views(source, intervention)
        assert intervention.history != source.history
        assert all(view.donor_target_index is None for view in views)


def test_state_swap_uses_counterfactual_history_and_semantic_donor_index() -> None:
    first = item("a", 0, counterfactual="b")
    second = item("b", 2, counterfactual="a")

    swapped = baseline.intervention_histories([first, second], condition="state_swap", seed=0)
    views = baseline.expand_reverse_cyclic_views(first, swapped[0])

    assert swapped[0].history == second.history
    assert swapped[0].donor_episode_id == "b"
    assert all(view.choices[view.donor_target_index] == "green" for view in views)


def test_reverse_views_remap_stale_target_index() -> None:
    source = item("stale", 0)
    source = baseline.HistoryQuery(
        metadata={
            **source.metadata,
            "stale_target_text": "blue",
            "stale_target_index": 1,
            "stale_target_mapped": True,
        },
        query=source.query,
        choices=source.choices,
        target_index=source.target_index,
        history=source.history,
    )

    views = baseline.expand_reverse_cyclic_views(
        source,
        baseline.HistoryIntervention(source.history),
    )

    assert {view.metadata["stale_target_index"] for view in views} == {0, 1, 2, 3}
    assert all(
        view.choices[view.metadata["stale_target_index"]] == "blue"
        for view in views
    )


class CharacterTokenizer:
    model_max_length = 8

    def __call__(self, text: str, *, add_special_tokens: bool, return_tensors: str) -> dict[str, torch.Tensor]:
        del add_special_tokens, return_tensors
        ids = torch.arange(1, len(text) + 1, dtype=torch.long).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TextProcessor:
    tokenizer = CharacterTokenizer()

    @staticmethod
    def apply_chat_template(messages, *, tokenize: bool, add_generation_prompt: bool) -> str:
        del tokenize, add_generation_prompt
        return str(messages[0]["content"][0]["text"]) + "|"


def test_context_overflow_fails_closed_without_truncation() -> None:
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8))
    with pytest.raises(RuntimeError, match="context overflow"):
        baseline.audit_context(
            model=model,
            processor=TextProcessor(),
            prompt="12345678",
            choices=CHOICES,
            input_mode="text_only",
            resized_blank_image=None,
        )


def test_text_only_scorer_never_constructs_multimodal_inputs() -> None:
    class BaseModel:
        calls: list[set[str]] = []

        def __call__(self, **kwargs):
            self.calls.append(set(kwargs))
            assert "pixel_values" not in kwargs
            assert "image_grid_thw" not in kwargs
            input_ids = kwargs["input_ids"]
            return SimpleNamespace(
                last_hidden_state=torch.zeros((1, input_ids.shape[1], 2), dtype=torch.float32)
            )

    class Model:
        model = BaseModel()

        @staticmethod
        def lm_head(hidden: torch.Tensor) -> torch.Tensor:
            return torch.zeros((*hidden.shape[:2], 128), dtype=torch.float32)

    result = baseline.qwen3vl_text_choice_nll(
        model=Model(),
        processor=TextProcessor(),
        query="q",
        choices=("a", "bb", "ccc", "dddd"),
        device=torch.device("cpu"),
        deterministic_ce=True,
    )

    assert len(result.mean_nll) == 4
    assert len(Model.model.calls) == 4
    assert all(
        call == {"input_ids", "attention_mask", "use_cache", "return_dict"}
        for call in Model.model.calls
    )


def test_text_only_scope_accepts_only_r3_micro_ids() -> None:
    micro = item("r3-set8-r0-v0", 0)
    formal = item("r3-formal-dev-0001", 0)
    assert baseline._is_micro_suite([micro])
    assert not baseline._is_micro_suite([formal])
    assert not baseline._is_micro_suite([micro, formal])


def test_scientific_payload_omits_replica_and_runtime_fields() -> None:
    first = baseline._scientific_row(
        {"replica_id": "A", "prediction_index": 2, "latency_seconds": 1.0, "peak_vram_gib": 10.0}
    )
    second = baseline._scientific_row(
        {"replica_id": "B", "prediction_index": 2, "latency_seconds": 9.0, "peak_vram_gib": 20.0}
    )
    assert first == second == {"prediction_index": 2}


def test_strict_determinism_is_configured_before_cuda_probe(tmp_path: Path) -> None:
    argv = [
        "qwen_text_baselines.py",
        "--episodes",
        str(tmp_path / "episodes.jsonl"),
        "--reader",
        str(tmp_path / "reader"),
        "--output",
        str(tmp_path / "predictions.jsonl"),
        "--replica-id",
        "A",
        "--seed",
        "7",
    ]
    sentinel = RuntimeError("strict-runtime-configured")
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(
            baseline,
            "configure_strict_cuda_determinism",
            side_effect=sentinel,
        ) as configure,
        mock.patch.object(baseline.torch.cuda, "is_available") as cuda_available,
        pytest.raises(RuntimeError, match="strict-runtime-configured"),
    ):
        baseline.main()
    configure.assert_called_once_with(seed=7)
    cuda_available.assert_not_called()


def test_existing_output_is_rejected_before_runtime_initialization(tmp_path: Path) -> None:
    output = tmp_path / "predictions.jsonl"
    output.write_text("owned\n", encoding="utf-8")
    argv = [
        "qwen_text_baselines.py",
        "--episodes",
        str(tmp_path / "episodes.jsonl"),
        "--reader",
        str(tmp_path / "reader"),
        "--output",
        str(output),
        "--replica-id",
        "A",
    ]
    with (
        mock.patch.object(sys, "argv", argv),
        mock.patch.object(baseline, "configure_strict_cuda_determinism") as configure,
        pytest.raises(SystemExit, match="Refusing to overwrite"),
    ):
        baseline.main()
    configure.assert_not_called()
