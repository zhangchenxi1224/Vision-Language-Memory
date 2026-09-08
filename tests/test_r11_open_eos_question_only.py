import copy
import json

import pytest

from scripts.experiments.run_r11_open_eos_question_only import (
    ROOT, SUFFIX, QUESTION_IDS, locked_prompts, check_anchor_parity, summarize,
)


def fixture():
    config = json.loads((ROOT / "configs/experiments/r11_open_eos_question_only.json").read_text())
    parent = json.loads((ROOT / "configs/experiments/r11_open_eos_paired.json").read_text())
    target = json.loads((ROOT / "configs/experiments/r11_open_answer_replay.json").read_text())["targets"][1]
    return config, target, parent


def test_fixed_instruction_is_byte_identical_and_restored_question_is_unchanged():
    config, target, parent = fixture()
    prompts = locked_prompts(config, target, parent)
    for name in ["original_open", "paraphrase_open"] + QUESTION_IDS:
        assert prompts[name].split("\n", 1)[1].encode() == SUFFIX[1:].encode()
    assert prompts["question_only_restored"].split("\n")[0] == parent["new_rewrite_open"].split("\n")[0]


@pytest.mark.parametrize("mutation", ["instruction", "gold", "identity", "duplicate"])
def test_reject_confounds_and_leakage(mutation):
    config, target, parent = fixture()
    if mutation == "instruction":
        config["fixed_suffix"] = "\nReturn only a short phrase."
    elif mutation == "gold":
        config["question_sentences"]["question_only_02"] += " ambient"
    elif mutation == "identity":
        config["question_sentences"]["question_only_02"] = "What music does another person prefer?"
    else:
        config["question_sentences"]["question_only_02"] = config["question_sentences"]["question_only_03"]
    with pytest.raises(ValueError):
        locked_prompts(config, target, parent)


def test_reject_parity_drift_and_incomplete_evaluation():
    row = {"arm": "B", "seed": 0, "condition": "matched", "prompt_id": "original_open",
           "generated_token_ids": [1, 2], "input_token_ids": [3], "chat_prompt": "p",
           "raw": "ambient", "strict_correct": True}
    parent = copy.deepcopy(row)
    row["generated_token_ids"] = [1, 4]
    with pytest.raises(RuntimeError):
        check_anchor_parity([row], {("B", 0, "matched", "original_open"): parent})
    with pytest.raises(ValueError):
        summarize([row])
