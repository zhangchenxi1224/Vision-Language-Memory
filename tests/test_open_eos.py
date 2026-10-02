from types import SimpleNamespace
import pytest
import torch
from vision_memory.reader.open_eos import (assistant_termination_contract, split_answer_eos_loss,
    generation_diagnostics, deployment_max_tokens, normalize_deployment_answer)
from scripts.experiments.run_r11_open_eos_paired import training_prompt, load_config, ROOT, aggregate


class Tokenizer:
    eos_token_id = 9
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[10, 11] if text == "<turn>\n" else [10]])}
    def decode(self, ids, **kwargs):
        return "<turn>"


class Processor:
    tokenizer = Tokenizer()
    def apply_chat_template(self, messages, **kwargs):
        if len(messages) == 1:
            return "prompt "
        return "prompt " + messages[-1]["content"] + "<turn>\n"


def test_eot_selected_from_template_and_generation_config_not_tokenizer_eos():
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9, 10]))
    contract = assistant_termination_contract(model, Processor())
    assert contract["assistant_end_token_id"] == 10
    assert contract["tokenizer_eos_token_id"] == 9
    with pytest.raises(ValueError, match="not a generation"):
        assistant_termination_contract(SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[9])), Processor())


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_eos_weight_not_diluted_by_multitoken_answer(n):
    torch.manual_seed(4)
    logits = torch.randn(1, n+1, 13, requires_grad=True)
    targets = torch.arange(n+1).reshape(1, -1)
    result = split_answer_eos_loss(logits, targets, answer_length=n)
    losses = torch.nn.functional.cross_entropy(logits[0], targets[0], reduction="none")
    torch.testing.assert_close(result.loss, losses[:n].mean() + losses[-1])
    result.loss.backward()
    grad = logits.grad.clone()
    logits.grad = None
    (losses[:n].mean() + losses[-1]).backward()
    torch.testing.assert_close(grad, logits.grad)
    assert result.answer_token_count == n


def test_full_sequence_metrics_do_not_award_prefix_or_trim_internal_spaces():
    generated = {"raw": "ambient trance", "generated_token_ids": [1, 2, 10], "eos_token_ids": [10]}
    result = generation_diagnostics(generated, "ambient", [1])
    assert result["answer_prefix_token_exact"] and result["overgeneration"]
    assert not result["strict_correct"]
    assert normalize_deployment_answer("  Light Blue。\nExplanation") == "light blue"
    assert normalize_deployment_answer("ambient trance") == "ambient trance"
    assert normalize_deployment_answer("orange juice") == "orange juice"
    generated.update(raw="light blue", generated_token_ids=[1, 2, 10])
    result = generation_diagnostics(generated, "light blue", [1, 2])
    assert result["strict_correct"] and not result["overgeneration"]
    assert result["answer_followed_immediately_by_eos"]


def test_schema_budget_and_balanced_prompt_schedule():
    assert deployment_max_tokens("short_word") == 4
    assert deployment_max_tokens("short_phrase") == 8
    with pytest.raises(ValueError):
        deployment_max_tokens("ambient")
    assert [training_prompt("C", s) for s in range(256)].count("paraphrase_open") == 128
    for arm in ("A", "B"):
        assert {training_prompt(arm, s) for s in range(256)} == {"original_open"}
    config = load_config(ROOT / "configs/experiments/r11_open_eos_paired.json")
    assert "new_rewrite_open" not in config["arms"]["C"]["training_prompts"]


def test_eos_is_one_position_and_partial_answer_accuracy_counts_missing_tokens_wrong():
    with pytest.raises(ValueError):
        split_answer_eos_loss(torch.randn(1, 4, 8), torch.tensor([[1, 2, 3, 4]]), answer_length=2)
    result = generation_diagnostics({"raw": "light", "generated_token_ids": [1, 10], "eos_token_ids": [10]},
                                    "light blue", [1, 2])
    assert result["generated_answer_token_accuracy"] == .5
    assert not result["answer_prefix_token_exact"]
