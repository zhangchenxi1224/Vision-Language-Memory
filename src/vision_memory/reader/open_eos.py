"""Explicit assistant end-of-turn supervision, with unchanged scientific decoding.

The completion terminator is derived from the checkpoint's actual chat template
and checked against generate()'s stopping IDs. The tokenizer EOS alone is not an
adequate specification of a chat model's end-of-turn token.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor

from .open_answer import _eos_token_ids, normalize_short_answer, score_short_answer
from .qwen3vl import _tokenizer_ids, qwen3vl_target_only_ce


def assistant_termination_contract(model: Any, processor: Any) -> dict[str, Any]:
    sentinel = "R11EOSBoundaryProbe"
    user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Return a short answer."}]}
    prompt = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    complete = processor.apply_chat_template(
        [user, {"role": "assistant", "content": sentinel}], tokenize=False, add_generation_prompt=False)
    expected = prompt + sentinel
    if not complete.startswith(expected):
        raise ValueError("Assistant template is not the generation prefix plus continuation; inspect it before training.")
    suffix = complete[len(expected):]
    ids = _tokenizer_ids(processor.tokenizer, suffix)[0].tolist()
    stops = _eos_token_ids(model, processor)
    if not ids or ids[0] not in stops:
        raise ValueError(f"First assistant suffix token {ids[:3]} is not a generation stopping token {stops}.")
    token_id = int(ids[0])
    token_text = processor.tokenizer.decode([token_id], skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
    if _tokenizer_ids(processor.tokenizer, token_text)[0].tolist() != [token_id]:
        raise ValueError("Assistant terminator does not round-trip as one special token.")
    return {"assistant_end_token_id": token_id, "assistant_end_token_text": token_text,
            "generation_eos_token_ids": list(stops), "chat_suffix": suffix,
            "chat_suffix_token_ids": ids,
            "tokenizer_eos_token_id": getattr(processor.tokenizer, "eos_token_id", None),
            "selection": "first token after an assistant answer in actual chat template, checked against generation EOS"}


@dataclass(frozen=True)
class SplitAnswerEOSOutput:
    loss: Tensor
    answer_loss: Tensor
    eos_loss: Tensor
    target_ids: Tensor
    target_logits: Tensor
    answer_token_count: int
    teacher_forced_answer_accuracy: Tensor
    teacher_forced_eos_accuracy: Tensor


def split_answer_eos_loss(logits: Tensor, targets: Tensor, *, answer_length: int,
                          lambda_eos: float = 1.0) -> SplitAnswerEOSOutput:
    if logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape:
        raise ValueError("Expected batch-one logits [1,L,V] and targets [1,L].")
    if targets.shape[0] != 1 or answer_length < 1 or targets.shape[1] != answer_length + 1:
        raise ValueError("Loss requires one or more answer tokens followed by exactly one end-of-turn token.")
    if lambda_eos < 0:
        raise ValueError("EOS weight must be nonnegative.")
    values = logits.float()
    nll = torch.logsumexp(values, -1) - values.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    answer_loss, eos_loss = nll[:, :answer_length].mean(), nll[:, answer_length].mean()
    correct = values.argmax(-1).eq(targets).float()
    return SplitAnswerEOSOutput(answer_loss + lambda_eos * eos_loss, answer_loss, eos_loss,
                                targets, logits, answer_length, correct[:, :answer_length].mean(),
                                correct[:, answer_length].mean())


def qwen3vl_answer_eos_ce(*, target: str, termination: dict[str, Any], lambda_eos: float = 1.0,
                         **kwargs: Any) -> SplitAnswerEOSOutput:
    processor = kwargs["processor"]
    query = kwargs["query"]
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": query}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from .qwen3vl import _joint_prompt_target_tokenization
    _, answer_ids = _joint_prompt_target_tokenization(processor, prompt, target)
    output = qwen3vl_target_only_ce(target=target + termination["assistant_end_token_text"], **kwargs)
    n = answer_ids.shape[1]
    if (output.target_ids.shape[1] != n + 1
            or not torch.equal(output.target_ids[:, :n].cpu(), answer_ids.cpu())
            or int(output.target_ids[0, -1]) != termination["assistant_end_token_id"]):
        raise ValueError("Appending end-of-turn changed answer tokenization or produced extra target tokens.")
    return split_answer_eos_loss(output.target_logits, output.target_ids, answer_length=n, lambda_eos=lambda_eos)


def generation_diagnostics(generated: dict[str, Any], gold: str, gold_token_ids: list[int]) -> dict[str, Any]:
    raw = generated["raw"]
    score = score_short_answer(raw, gold)
    tokens = generated["generated_token_ids"]
    prefix_exact = len(tokens) >= len(gold_token_ids) and tokens[:len(gold_token_ids)] == gold_token_ids
    answer_accuracy = sum(int(i < len(tokens) and tokens[i] == token) for i, token in enumerate(gold_token_ids)) / len(gold_token_ids)
    after_answer = tokens[len(gold_token_ids):] if prefix_exact else []
    eos_ids = generated["eos_token_ids"]
    return {**score, "answer_prefix_token_exact": prefix_exact, "generated_answer_token_accuracy": answer_accuracy,
            "overgeneration": bool(prefix_exact and any(token not in eos_ids for token in after_answer)),
            "answer_followed_immediately_by_eos": bool(prefix_exact and after_answer and after_answer[0] in eos_ids),
            "answer_token_count": len(gold_token_ids), "gold_token_ids": gold_token_ids}


def deployment_max_tokens(answer_type: str) -> int:
    """Input is a task schema field, never computed from gold answer/token length."""
    if answer_type == "short_word":
        return 4
    if answer_type == "short_phrase":
        return 8
    raise ValueError("Task schema must declare short_word or short_phrase.")


def normalize_deployment_answer(raw: str) -> str:
    # A newline is an explicit response boundary; spaces inside a phrase are not.
    # Raw 32-token output remains the scientific metric and is never replaced.
    return normalize_short_answer(raw.split("\n", 1)[0])


def newline_stopping_criteria(processor: Any, prompt_token_count: int):
    from transformers import StoppingCriteria
    class NewlineStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            text = processor.tokenizer.decode(input_ids[0, prompt_token_count:], skip_special_tokens=False,
                                              clean_up_tokenization_spaces=False)
            return torch.tensor(["\n" in text], dtype=torch.bool, device=input_ids.device)
    return NewlineStop()
