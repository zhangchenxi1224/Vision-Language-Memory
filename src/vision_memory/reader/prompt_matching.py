"""Frozen-Reader prompt matching on exact generated continuation token IDs.

The teacher and student may have different query prefixes, but must be scored on
the same continuation. No answer text is decoded, retokenized, or added to the
processor's prompt. Soft targets cover the complete vocabulary, including EOS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .qwen3vl import (
    R3_QWEN_READER_RESIZE_CONTRACT,
    ReaderLossOutput,
    _assert_locked_reader_processor_output,
    _hidden_states,
    _prepare_reader_image,
)


def _require_frozen_eval(model: Any) -> None:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Prompt matching requires a torch.nn.Module Reader.")
    if any(module.training for module in model.modules()):
        raise ValueError("Prompt matching requires an eval-mode Reader, including its submodules.")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Prompt matching requires all Reader parameters to be frozen.")


def _continuation_ids(target_ids: Tensor | Sequence[int], device: torch.device | str) -> Tensor:
    if isinstance(target_ids, Tensor):
        if target_ids.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("target_ids must contain integer token IDs, not floats or bools.")
        ids = target_ids.detach()
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError("target_ids must be a nonempty sequence with shape [L] or [1,L].")
        if torch.any(ids < 0):
            raise ValueError("target_ids cannot contain negative token IDs.")
        return ids.to(device=device, dtype=torch.long)
    if not isinstance(target_ids, Sequence) or isinstance(target_ids, (str, bytes)):
        raise TypeError("target_ids must be an integer sequence or tensor.")
    if not target_ids or any(type(token) is not int or token < 0 for token in target_ids):
        raise ValueError("target_ids must be a nonempty sequence of nonnegative integer IDs.")
    return torch.tensor([list(target_ids)], device=device, dtype=torch.long)


def qwen3vl_continuation_logits(
    *,
    model: Any,
    processor: Any,
    image: Tensor,
    query: str,
    target_ids: Tensor | Sequence[int],
    device: torch.device | str,
    require_image_grad: bool = True,
    reader_resize_contract: str | None = R3_QWEN_READER_RESIZE_CONTRACT,
) -> ReaderLossOutput:
    """Score exact generated IDs after an image/query generation prefix.

    Teacher callers can use ``no_grad`` and ``require_image_grad=False``. Student
    callers retain input-image gradients through the frozen Reader. The returned
    ``loss`` is mean hard CE for diagnostics; prompt matching uses target_logits.
    The caller must validate a generated continuation's real EOS before caching.
    """
    _require_frozen_eval(model)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string.")
    if not isinstance(require_image_grad, bool):
        raise TypeError("require_image_grad must be a bool.")
    if not isinstance(image, Tensor):
        raise TypeError("image must be a torch.Tensor.")
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError("image must be one floating RGB tensor [3,H,W] or [1,3,H,W].")
    if not torch.isfinite(image).all():
        raise ValueError("image must contain finite pixels.")
    ids = _continuation_ids(target_ids, device)
    image, do_resize = _prepare_reader_image(
        image, do_resize=None, reader_resize_contract=reader_resize_contract
    )
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": query},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": [prompt], "images": [image], "return_tensors": "pt", "do_rescale": False}
    if do_resize is not None:
        kwargs["do_resize"] = do_resize
    batch = {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in processor(**kwargs).items()
    }
    pixels, grid = batch["pixel_values"], batch["image_grid_thw"]
    if reader_resize_contract is not None:
        _assert_locked_reader_processor_output(pixels, grid)
    if require_image_grad and (not pixels.requires_grad or pixels.grad_fn is None):
        raise RuntimeError("Qwen processor detached the image; prompt matching requires image gradients.")
    prefix = batch["input_ids"]
    attention = batch["attention_mask"]
    if not isinstance(prefix, Tensor) or prefix.ndim != 2 or prefix.shape[0] != 1 or prefix.shape[1] == 0:
        raise ValueError("Prompt matching requires one nonempty processed input prefix.")
    if prefix.dtype != torch.long or torch.any(prefix < 0):
        raise ValueError("Processed input prefix must contain nonnegative int64 token IDs.")
    if not isinstance(attention, Tensor) or attention.shape != prefix.shape or not torch.all(attention == 1):
        raise ValueError("Prompt matching requires an unpadded prefix attention mask.")
    prefix_length = prefix.shape[1]  # Includes processor-expanded visual tokens.
    inputs = {
        "input_ids": torch.cat((prefix, ids), dim=1),
        "attention_mask": torch.cat((attention, attention.new_ones(ids.shape)), dim=1),
        "pixel_values": pixels,
        "image_grid_thw": grid,
        "use_cache": False,
        "return_dict": True,
    }
    if "mm_token_type_ids" in batch:
        types = batch["mm_token_type_ids"]
        if not isinstance(types, Tensor) or types.shape != prefix.shape:
            raise ValueError("mm_token_type_ids must match the processed prefix shape.")
        inputs["mm_token_type_ids"] = torch.cat((types, types.new_zeros(ids.shape)), dim=1)
    hidden = _hidden_states(model.model(**inputs))
    if hidden.ndim != 3 or hidden.shape[:2] != inputs["input_ids"].shape:
        raise RuntimeError("Reader hidden states do not match the complete input sequence.")
    # Logits at prefix_length-1 predict the FIRST continuation token.
    logits = model.lm_head(hidden[:, prefix_length - 1:prefix_length + ids.shape[1] - 1])
    if logits.ndim != 3 or logits.shape[:2] != ids.shape or logits.shape[-1] == 0:
        raise RuntimeError("Reader returned invalid continuation vocabulary logits.")
    if torch.any(ids >= logits.shape[-1]):
        raise ValueError("A continuation token ID is outside the Reader vocabulary.")
    flat = logits.float().reshape(-1, logits.shape[-1])
    if not torch.isfinite(flat).all():
        raise RuntimeError("Reader continuation logits contain NaN or Inf.")
    scores = flat.gather(-1, ids.reshape(-1, 1)).squeeze(-1)
    hard_ce = (torch.logsumexp(flat, dim=-1) - scores).mean()
    return ReaderLossOutput(loss=hard_ce, pixel_values=pixels, target_ids=ids, target_logits=logits)


def _soft_terms(
    student_logits: Tensor, teacher_logits: Tensor, temperature: float, mask: Tensor | None
) -> tuple[Tensor, Tensor, Tensor, Tensor, float]:
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise ValueError("temperature must be a finite positive number.")
    temperature = float(temperature)
    fp32 = torch.finfo(torch.float32)
    if (not math.isfinite(temperature) or temperature <= 0
            or not fp32.tiny * fp32.eps <= temperature * temperature <= fp32.max):
        raise ValueError("temperature and its square must be finite, positive and representable in FP32.")
    if not isinstance(student_logits, Tensor) or not isinstance(teacher_logits, Tensor):
        raise TypeError("Student and teacher logits must be tensors.")
    if (student_logits.shape != teacher_logits.shape or student_logits.ndim < 2
            or student_logits.numel() == 0):
        raise ValueError("Student and teacher logits require identical nonempty [...,vocabulary] shapes.")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise ValueError("Student and teacher logits must be floating point.")
    student = student_logits.float() / temperature
    teacher = teacher_logits.detach().to(device=student.device, dtype=torch.float32) / temperature
    if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
        raise ValueError("Scaled student and teacher logits must be finite.")
    if mask is None:
        weights = torch.ones(student.shape[:-1], device=student.device, dtype=torch.float32)
    else:
        if not isinstance(mask, Tensor) or mask.shape != student.shape[:-1]:
            raise ValueError("mask must match the logits' token dimensions exactly.")
        if mask.is_complex():
            raise ValueError("mask must be real binary values.")
        weights = mask.detach().to(device=student.device, dtype=torch.float32)
        if not torch.isfinite(weights).all() or not torch.all((weights == 0) | (weights == 1)):
            raise ValueError("mask must contain only zero/one (or boolean) values.")
        if not torch.any(weights):
            raise ValueError("mask must select at least one token.")
    student_logp, teacher_logp = F.log_softmax(student, dim=-1), F.log_softmax(teacher, dim=-1)
    if not torch.isfinite(student_logp).all() or not torch.isfinite(teacher_logp).all():
        raise ValueError("Log-softmax overflowed; logits exceed the finite FP32 loss domain.")
    return student_logp, teacher_logp, teacher_logp.exp(), weights, temperature


def soft_target_cross_entropy(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
    mask: Tensor | None = None,
) -> Tensor:
    """Full-vocabulary teacher-to-student CE, averaged over selected tokens × T².

    Teacher logits are always detached. A binary mask has shape logits.shape[:-1];
    it never selects only top-k vocabulary entries or changes target probabilities.
    """
    student_logp, _, teacher_p, weights, temp = _soft_terms(student_logits, teacher_logits, temperature, mask)
    per_token = -(teacher_p * student_logp).sum(-1)
    return (per_token * weights).sum() / weights.sum() * (temp * temp)


def soft_target_kl_divergence(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
    mask: Tensor | None = None,
) -> Tensor:
    """Teacher || student KL diagnostic, using the same token reduction and T²."""
    student_logp, teacher_logp, teacher_p, weights, temp = _soft_terms(
        student_logits, teacher_logits, temperature, mask
    )
    per_token = (teacher_p * (teacher_logp - student_logp)).sum(-1)
    return (per_token * weights).sum() / weights.sum() * (temp * temp)


def validate_generated_target_ids(
    generation: Mapping[str, Any],
    *,
    assistant_end_token_id: int,
    pad_token_id: int | None = None,
) -> tuple[int, ...]:
    """Accept an unmodified, unpadded generation ending at the actual assistant EOS.

    The input is the receipt from ``generate_short_answer``. A different stopping
    token, missing EOS, post-EOS padding, or inconsistent metadata fails closed.
    Never appends an EOS, trims tokens, or decodes/retokenizes a truncated answer.
    """
    if not isinstance(generation, Mapping):
        raise TypeError("generation must be a generation receipt mapping.")
    if type(assistant_end_token_id) is not int or assistant_end_token_id < 0:
        raise ValueError("assistant_end_token_id must be a nonnegative integer.")
    if pad_token_id is not None and (type(pad_token_id) is not int or pad_token_id < 0):
        raise ValueError("pad_token_id must be a nonnegative integer or None.")
    raw_ids = generation.get("generated_token_ids")
    if not isinstance(raw_ids, (list, tuple)) or not raw_ids or any(type(t) is not int or t < 0 for t in raw_ids):
        raise ValueError("Generation must contain the exact nonempty integer generated_token_ids.")
    ids = tuple(raw_ids)
    stop_ids = generation.get("eos_token_ids")
    if (not isinstance(stop_ids, (list, tuple)) or not stop_ids
            or any(type(t) is not int or t < 0 for t in stop_ids)
            or assistant_end_token_id not in stop_ids):
        raise ValueError("Generation stopping metadata must include the actual assistant EOS.")
    if (generation.get("eos_reached") is not True or generation.get("truncated") is not False
            or generation.get("finish_reason") != "eos"):
        raise ValueError("Generation must reach EOS naturally without truncation or another stopping condition.")
    count = generation.get("generated_token_count")
    if type(count) is not int or count != len(ids):
        raise ValueError("Generation token count metadata does not match its exact token IDs.")
    if ids[-1] != assistant_end_token_id or any(t in stop_ids for t in ids[:-1]):
        raise ValueError("Continuation must end exactly once at the actual assistant EOS, with no post-EOS tokens.")
    # Some model configurations use EOS as padding. Permit that single terminal EOS only.
    if pad_token_id is not None and (pad_token_id in ids[:-1]
                                    or (ids[-1] == pad_token_id and pad_token_id != assistant_end_token_id)):
        raise ValueError("Generated continuation contains padding.")
    return ids


__all__ = [
    "qwen3vl_continuation_logits",
    "soft_target_cross_entropy",
    "soft_target_kl_divergence",
    "validate_generated_target_ids",
]
