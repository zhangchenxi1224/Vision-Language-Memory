import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("multiq_prepare", Path(__file__).resolve().parents[1] /
                                            "scripts/experiments/prepare_r11_open_eos_multiquestion.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def source_episodes():
    rows = []
    for topic, gold in prepare.STRATA:
        for i in range(3):
            eid = f"{topic}-{gold}-{i}"
            rows.append({"episode_id": eid, "semantic_group_id": eid,
                         "split": "train", "distractor_variant": "clean", "topic": topic,
                         "turns": [{"query": {"text": f"Before any later update, what is the current {topic} preference for the table? Choose exactly one option.",
                                              "choices": [gold, "foil"], "target_index": 0}}]})
    return rows


def test_selection_is_outcome_independent_and_unique_by_group():
    rows = source_episodes()
    a = prepare.select_questions(rows, set())
    b = prepare.select_questions(list(reversed(rows)), set())
    assert [t["segment_id"] for t in a] == [t["segment_id"] for t in b]
    assert len(a) == len({t["semantic_group_id"] for t in a}) == 16
    assert all(t["original"]["choices"][t["original"]["answer_index"]] == t["scorer_metadata"]["gold"] for t in a)
    assert all(t["inputs"]["paraphrase_open"].startswith("Before any later update,") for t in a)


def test_counterfactual_duplicates_cannot_supply_independent_questions():
    rows = source_episodes()
    for row in rows:
        row["semantic_group_id"] = row["topic"]
    with pytest.raises(ValueError, match="independent source groups"):
        prepare.select_questions(rows, set())


def test_exclusion_prevents_old_task_reuse():
    rows = source_episodes()
    old_group = prepare.select_questions(rows, set())[0]["semantic_group_id"]
    selected = prepare.select_questions(rows, {old_group})
    assert old_group not in {t["semantic_group_id"] for t in selected}


def test_query_transformation_rejects_unknown_format_and_does_not_insert_gold():
    with pytest.raises(ValueError, match="suffix"):
        prepare.make_prompts("What material is preferred?")
    query = "After applying this update first, what is the current drink preference for the desk? Choose exactly one option."
    prompts = prepare.make_prompts(query)
    assert all(text.startswith("After applying this update first,") for text in prompts.values())
    assert all("juice" not in text and "no active preference" not in text for text in prompts.values())
