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


@dataclass(frozen=True)
class ChoiceScoreOutput:
    mean_nll: tuple[float, ...]
    predicted_index: int


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
    require_image_grad: bool = True,
    do_resize: bool | None = None,
    deterministic_ce: bool = False,
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
    processor_kwargs = {
        "text": [prompt],
        "images": [image],
        "return_tensors": "pt",
        "do_rescale": False,
    }
    if do_resize is not None:
        processor_kwargs["do_resize"] = do_resize
    batch = processor(
        **processor_kwargs,
    ).to(device)

    pixel_values = batch["pixel_values"]
    if require_image_grad and (not pixel_values.requires_grad or pixel_values.grad_fn is None):
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
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_targets = target_ids.reshape(-1)
    if deterministic_ce:
        # CUDA NLLLoss backward is rejected by torch.use_deterministic_algorithms.
        # This mathematically equivalent FP32 target log-probability path is used
        # only by the explicit reproducibility probe; the production default above
        # remains PyTorch cross entropy.
        target_scores = flat_logits.gather(dim=-1, index=flat_targets.unsqueeze(-1)).squeeze(-1)
        loss = (torch.logsumexp(flat_logits, dim=-1) - target_scores).mean()
    else:
        loss = F.cross_entropy(flat_logits, flat_targets)
    return ReaderLossOutput(loss=loss, pixel_values=pixel_values, target_ids=target_ids, target_logits=logits)


def qwen3vl_choice_nll(
    *,
    model: Any,
    processor: Any,
    image: Tensor,
    query: str,
    choices: list[str] | tuple[str, ...],
    device: torch.device,
    do_resize: bool | None = None,
    deterministic_ce: bool = False,
) -> ChoiceScoreOutput:
    """Score MCQ option texts by teacher-forced mean NLL for evaluation."""

    if len(choices) < 2:
        raise ValueError("Choice scoring requires at least two options.")
    scores: list[float] = []
    with torch.no_grad():
        for choice in choices:
            output = qwen3vl_target_only_ce(
                model=model,
                processor=processor,
                image=image,
                query=query,
                target=choice,
                device=device,
                require_image_grad=False,
                do_resize=do_resize,
                deterministic_ce=deterministic_ce,
            )
            scores.append(float(output.loss.item()))
    predicted_index = min(range(len(scores)), key=scores.__getitem__)
    return ChoiceScoreOutput(mean_nll=tuple(scores), predicted_index=predicted_index)
