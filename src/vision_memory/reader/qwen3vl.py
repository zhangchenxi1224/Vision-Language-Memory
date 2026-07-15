"""Qwen3-VL target-only CE without materializing vocabulary logits for the prefix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class ReaderLossOutput:
    loss: Tensor
    pixel_values: Tensor
    target_ids: Tensor
    target_logits: Tensor


def _hidden_states(output: Any) -> Tensor:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported Qwen base-model output: {type(output)!r}")


def qwen3vl_target_only_ce(
    *,
    model: Any,
    processor: Any,
    image: Tensor,
    query: str,
    target: str,
    device: torch.device,
) -> ReaderLossOutput:
    """Compute teacher-forced CE while retaining image-to-loss autograd.

    model parameters should already be frozen with requires_grad_(False). Do not call this
    function under no_grad/inference_mode. image is expected to contain floats in [0, 1].
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        do_rescale=False,
    ).to(device)

    pixel_values = batch["pixel_values"]
    if not pixel_values.requires_grad or pixel_values.grad_fn is None:
        raise RuntimeError(
            "Qwen processor detached the image. Require the fast tensor processor or add a tensor-native adapter."
        )

    target_ids = processor.tokenizer(
        target,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    if target_ids.numel() == 0:
        raise ValueError("The target tokenized to an empty sequence.")

    prefix_len = batch["input_ids"].shape[1]
    input_ids = torch.cat([batch["input_ids"], target_ids], dim=1)
    attention_mask = torch.cat([batch["attention_mask"], torch.ones_like(target_ids)], dim=1)

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": batch["image_grid_thw"],
        "use_cache": False,
        "return_dict": True,
    }
    if "mm_token_type_ids" in batch:
        suffix_types = torch.zeros_like(target_ids, dtype=batch["mm_token_type_ids"].dtype)
        model_inputs["mm_token_type_ids"] = torch.cat([batch["mm_token_type_ids"], suffix_types], dim=1)

    output = model.model(**model_inputs)
    hidden = _hidden_states(output)

    # Token at input position j is predicted from hidden state j - 1.
    positions = torch.arange(
        prefix_len - 1,
        prefix_len + target_ids.shape[1] - 1,
        device=device,
    )
    target_hidden = hidden.index_select(dim=1, index=positions)
    logits = model.lm_head(target_hidden)
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target_ids.reshape(-1))
    return ReaderLossOutput(loss=loss, pixel_values=pixel_values, target_ids=target_ids, target_logits=logits)

