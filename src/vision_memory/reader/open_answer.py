"""Answer-blind Qwen generation and conservative exact short-answer scoring.

Generation receives only an image and the question. Expected answers belong exclusively
to the post-generation scorer; diagnostics never relax the primary exact-match score.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import torch
from torch import Tensor

from .qwen3vl import (
    R3_QWEN_READER_RESIZE_CONTRACT,
    _assert_locked_reader_processor_output,
    _prepare_reader_image,
)


def normalize_short_answer(text: str) -> str:
    """Normalize case, whitespace and trailing punctuation, preserving answer content.

    No synonym lookup, substring extraction, option matching or negation removal is
    performed. In particular, ``not green`` remains different from ``green``.
    Only trailing Unicode punctuation (category P) and whitespace are removed;
    leading or internal punctuation is preserved exactly.
    """
    if not isinstance(text, str):
        raise TypeError("Short answers must be strings.")
    normalized = " ".join(text.casefold().split())
    while normalized and (normalized[-1].isspace() or unicodedata.category(normalized[-1]).startswith("P")):
        normalized = normalized[:-1]
    return normalized


def score_short_answer(raw: str, expected_answer: str) -> dict[str, Any]:
    """Score complete normalized strings; report extra wording without awarding credit.

    ``expected_text_present`` is a lexical diagnostic, not semantic correctness. It
    can be true for a negated answer. ``format_status`` likewise does not establish
    that an answer with an explanation or extra text is factually correct.
    """
    normalized = normalize_short_answer(raw)
    expected = normalize_short_answer(expected_answer)
    if not expected:
        raise ValueError("Expected answer must be nonempty after normalization.")
    strict_correct = normalized == expected
    expected_present = bool(re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", normalized))
    extra_text = expected_present and not strict_correct
    if not normalized:
        format_status = "empty"
    elif strict_correct:
        format_status = "exact_short_answer"
    elif extra_text:
        format_status = "extra_text_or_explanation"
    else:
        format_status = "different_answer"
    return {
        "raw": raw,
        "normalized": normalized,
        "normalized_expected": expected,
        "strict_correct": strict_correct,
        "format_status": format_status,
        "has_extra_text": extra_text,
        "expected_text_present": expected_present,
    }


def _eos_token_ids(model: Any, processor: Any) -> tuple[int, ...]:
    # Mirror generate's model generation-config precedence. Tokenizer is only a
    # fallback, and resolved IDs are explicitly passed so the audit matches stopping.
    candidates = (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(getattr(processor, "tokenizer", None), "eos_token_id", None),
    )
    for value in candidates:
        if value is not None:
            if isinstance(value, Tensor):
                value = value.detach().cpu().reshape(-1).tolist()
            if isinstance(value, int):
                return (value,)
            return tuple(int(item) for item in value)
    return ()


@torch.no_grad()
def generate_short_answer(
    *,
    model: Any,
    processor: Any,
    image: Tensor,
    query: str,
    device: torch.device | str,
    max_new_tokens: int = 32,
    reader_resize_contract: str | None = R3_QWEN_READER_RESIZE_CONTRACT,
    do_sample: bool = False,
) -> dict[str, Any]:
    """Generate from image + question, without any answer/choice interface.

    Uses the same tensor resize and processor contract as R11 teacher-forced scoring.
    The caller supplies the complete question, including any short-answer instruction.
    Evaluation temporarily uses ``eval`` and ``no_grad``; the prior model training
    state is restored, and parameter ``requires_grad`` flags are untouched.

    ``truncated`` conservatively marks every output lacking an observed EOS. A
    separate finish reason distinguishes a token-limit stop from another stop.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty question string.")
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer.")
    if not isinstance(do_sample, bool):
        raise TypeError("do_sample must be a bool.")
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError("image must be one floating RGB tensor [3,H,W] or [1,3,H,W].")
    image, do_resize = _prepare_reader_image(
        image, do_resize=None, reader_resize_contract=reader_resize_contract
    )
    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": query}],
    }]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs: dict[str, Any] = {
        "text": [prompt], "images": [image], "return_tensors": "pt", "do_rescale": False,
    }
    if do_resize is not None:
        processor_kwargs["do_resize"] = do_resize
    batch = processor(**processor_kwargs)
    # Forward processor metadata, including Qwen3-VL's mm_token_type_ids, unchanged.
    model_inputs = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}
    if reader_resize_contract is not None:
        _assert_locked_reader_processor_output(model_inputs["pixel_values"], model_inputs["image_grid_thw"])
    input_ids = model_inputs["input_ids"]
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Short-answer generation requires one input token sequence.")
    prompt_length = input_ids.shape[1]
    eos_ids = _eos_token_ids(model, processor)
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": 1,
        "num_return_sequences": 1,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    if eos_ids:
        generation_kwargs["eos_token_id"] = list(eos_ids)
    prior_training = model.training
    model.eval()
    try:
        generated = model.generate(**model_inputs, **generation_kwargs)
    finally:
        model.train(prior_training)
    sequences = generated if isinstance(generated, Tensor) else generated.sequences
    if not isinstance(sequences, Tensor) or sequences.ndim != 2 or sequences.shape[0] != 1:
        raise RuntimeError("Qwen generate must return exactly one full token sequence.")
    if sequences.shape[1] < prompt_length or not torch.equal(sequences[:, :prompt_length], input_ids):
        raise RuntimeError("Qwen generate changed the input prefix; refusing ambiguous output trimming.")
    continuation = sequences[:, prompt_length:]
    token_ids = [int(item) for item in continuation[0].detach().cpu().tolist()]
    raw = processor.batch_decode(
        continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    raw_with_special_tokens = processor.batch_decode(
        continuation, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )[0]
    eos_reached = any(token_id in eos_ids for token_id in token_ids)
    finish_reason = "eos" if eos_reached else "token_limit" if len(token_ids) >= max_new_tokens else "other_stop"
    return {
        "raw": raw,
        "raw_with_special_tokens": raw_with_special_tokens,
        "chat_prompt": prompt,
        "input_token_ids": [int(item) for item in input_ids[0].detach().cpu().tolist()],
        "generated_token_ids": token_ids,
        "prompt_token_count": prompt_length,
        "generated_token_count": len(token_ids),
        "eos_token_ids": list(eos_ids),
        "eos_reached": eos_reached,
        "truncated": not eos_reached,
        "finish_reason": finish_reason,
    }


__all__ = ["generate_short_answer", "normalize_short_answer", "score_short_answer"]
