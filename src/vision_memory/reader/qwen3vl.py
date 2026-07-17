"""Qwen3-VL target-only CE without materializing vocabulary logits for the prefix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .deterministic_resize import (
    R3_QWEN_READER_GRID_THW,
    R3_QWEN_READER_PIXEL_VALUES_SHAPE,
    R3_QWEN_READER_RESIZE_CONTRACT,
    deterministic_qwen_reader_resize,
)


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


@dataclass(frozen=True)
class ListwiseChoiceLossOutput:
    loss: Tensor
    choice_mean_nll: Tensor
    choice_logits: Tensor
    target_ids: Tensor
    choice_token_counts: tuple[int, ...]


@dataclass(frozen=True)
class VisualFeatureOutput:
    """Query-free post-merger Qwen visual tokens with the tensor preprocessing trace."""

    features: Tensor
    pixel_values: Tensor
    image_grid_thw: Tensor


def _prepare_reader_image(
    image: Tensor,
    *,
    do_resize: bool | None,
    reader_resize_contract: str | None,
) -> tuple[Tensor, bool | None]:
    if reader_resize_contract is None:
        return image, do_resize
    if reader_resize_contract != R3_QWEN_READER_RESIZE_CONTRACT:
        raise ValueError(f"Unknown Qwen Reader resize contract: {reader_resize_contract!r}.")
    if do_resize is not None and do_resize is not False:
        raise ValueError("The locked R3 Reader resize contract requires processor do_resize=False.")
    return deterministic_qwen_reader_resize(image, contract=reader_resize_contract), False


def _assert_locked_reader_processor_output(pixel_values: Tensor, image_grid_thw: Tensor) -> None:
    if tuple(pixel_values.shape) != R3_QWEN_READER_PIXEL_VALUES_SHAPE:
        raise RuntimeError(
            "Locked R3 Qwen processor pixel shape drifted: "
            f"expected {R3_QWEN_READER_PIXEL_VALUES_SHAPE}, got {tuple(pixel_values.shape)}."
        )
    if pixel_values.dtype != torch.float32:
        raise RuntimeError(
            f"Locked R3 Qwen processor must emit float32 pixels, got {pixel_values.dtype}."
        )
    if not torch.isfinite(pixel_values).all():
        raise RuntimeError("Locked R3 Qwen processor pixels contain NaN or Inf.")
    if tuple(image_grid_thw.shape) != (1, 3):
        raise RuntimeError(f"Locked R3 Qwen image grid must have shape [1,3], got {tuple(image_grid_thw.shape)}.")
    if image_grid_thw.dtype != torch.int64:
        raise RuntimeError(f"Locked R3 Qwen image grid must be int64, got {image_grid_thw.dtype}.")
    actual_grid = tuple(int(value) for value in image_grid_thw.detach().cpu().reshape(-1).tolist())
    if actual_grid != R3_QWEN_READER_GRID_THW:
        raise RuntimeError(
            "Locked R3 Qwen image grid drifted: "
            f"expected {R3_QWEN_READER_GRID_THW}, got {actual_grid}."
        )


def _hidden_states(output: Any) -> Tensor:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported Qwen base-model output: {type(output)!r}")


def qwen3vl_query_free_visual_features(
    *,
    model: Any,
    processor: Any,
    image: Tensor,
    device: torch.device,
    require_image_grad: bool = True,
    do_resize: bool | None = None,
    reader_resize_contract: str | None = None,
) -> VisualFeatureOutput:
    """Extract post-merger visual tokens without accepting any text/query input."""

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Query-free Qwen visual features support exactly one image.")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3 or not image.is_floating_point():
        raise ValueError("image must be a floating RGB tensor with shape [3,H,W] or [1,3,H,W].")
    image, do_resize = _prepare_reader_image(
        image,
        do_resize=do_resize,
        reader_resize_contract=reader_resize_contract,
    )
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not callable(image_processor):
        raise TypeError("Qwen processor must expose a callable tensor image_processor.")
    kwargs: dict[str, Any] = {
        "images": [image],
        "return_tensors": "pt",
        "do_rescale": False,
    }
    if do_resize is not None:
        kwargs["do_resize"] = do_resize
    batch = image_processor(**kwargs)
    pixel_values = batch["pixel_values"].to(device)
    image_grid_thw = batch["image_grid_thw"].to(device)
    if reader_resize_contract is not None:
        _assert_locked_reader_processor_output(pixel_values, image_grid_thw)
    if require_image_grad and (not pixel_values.requires_grad or pixel_values.grad_fn is None):
        raise RuntimeError("Qwen image processor detached the query-free visual feature input.")
    encoded = model.get_image_features(pixel_values, image_grid_thw)
    image_features = encoded[0] if isinstance(encoded, (tuple, list)) else encoded
    if isinstance(image_features, (tuple, list)):
        if len(image_features) != 1 or not isinstance(image_features[0], Tensor):
            raise TypeError("Qwen get_image_features must return exactly one image tensor.")
        features = image_features[0]
    elif isinstance(image_features, Tensor):
        features = image_features
    else:
        raise TypeError("Unsupported Qwen image feature output.")
    if features.ndim == 2:
        features = features.unsqueeze(0)
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError(f"Qwen visual features must have shape [1,tokens,hidden], got {tuple(features.shape)}.")
    if not torch.isfinite(features).all():
        raise RuntimeError("Qwen query-free visual features contain NaN or Inf.")
    return VisualFeatureOutput(
        features=features,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )


def _tokenizer_ids(tokenizer: Any, text: str) -> Tensor:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise TypeError("Qwen tokenizer must return input_ids with shape [1, sequence].")
    return input_ids


def _joint_prompt_target_tokenization(processor: Any, prompt: str, target: str) -> tuple[str, Tensor]:
    """Tokenize the assistant continuation in the prompt's actual left context.

    Tokenizing ``target`` in isolation can choose different BPE tokens from tokenizing the
    same text after the chat-template generation prompt.  The generation prompt is required
    to end on a stable tokenizer boundary; fail closed if appending the target retokenizes
    any prefix token.
    """

    tokenizer = processor.tokenizer
    prompt_ids = _tokenizer_ids(tokenizer, prompt)
    joint_text = prompt + target
    joint_ids = _tokenizer_ids(tokenizer, joint_text)
    prompt_length = prompt_ids.shape[1]
    if joint_ids.shape[1] <= prompt_length:
        raise ValueError("The target tokenized to an empty continuation in joint chat context.")
    if not torch.equal(joint_ids[:, :prompt_length], prompt_ids):
        raise RuntimeError(
            "Appending the target retokenized the chat-template prefix. "
            "The generation prompt must end on a stable tokenizer boundary."
        )
    return joint_text, joint_ids[:, prompt_length:]


def _qwen3vl_target_only_ce_prepared(
    *,
    model: Any,
    processor: Any,
    image: Tensor,
    query: str,
    target: str,
    device: torch.device,
    require_image_grad: bool,
    do_resize: bool | None,
    locked_resize_contract: bool,
    deterministic_ce: bool,
) -> ReaderLossOutput:
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
    joint_text, target_ids = _joint_prompt_target_tokenization(processor, prompt, target)
    processor_kwargs = {
        "text": [joint_text],
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
    if locked_resize_contract:
        _assert_locked_reader_processor_output(pixel_values, batch["image_grid_thw"])
    if require_image_grad and (not pixel_values.requires_grad or pixel_values.grad_fn is None):
        raise RuntimeError(
            "Qwen processor detached the image. Require the fast tensor processor or add a tensor-native adapter."
        )

    target_ids = target_ids.to(device)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    target_length = target_ids.shape[1]
    if input_ids.shape[1] <= target_length:
        raise RuntimeError("Joint Qwen input contains no non-target chat prefix.")
    if not torch.equal(input_ids[:, -target_length:], target_ids):
        raise RuntimeError(
            "Qwen processor changed the jointly tokenized target suffix; "
            "train/eval choice scoring requires an exact shared continuation."
        )
    prefix_len = input_ids.shape[1] - target_length

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": batch["image_grid_thw"],
        "use_cache": False,
        "return_dict": True,
    }
    if "mm_token_type_ids" in batch:
        model_inputs["mm_token_type_ids"] = batch["mm_token_type_ids"]

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
    reader_resize_contract: str | None = None,
    deterministic_ce: bool = False,
) -> ReaderLossOutput:
    """Compute teacher-forced CE while retaining image-to-loss autograd.

    model parameters should already be frozen with requires_grad_(False). Do not call this
    function under no_grad/inference_mode. image is expected to contain floats in [0, 1].
    """

    image, do_resize = _prepare_reader_image(
        image,
        do_resize=do_resize,
        reader_resize_contract=reader_resize_contract,
    )
    return _qwen3vl_target_only_ce_prepared(
        model=model,
        processor=processor,
        image=image,
        query=query,
        target=target,
        device=device,
        require_image_grad=require_image_grad,
        do_resize=do_resize,
        locked_resize_contract=reader_resize_contract is not None,
        deterministic_ce=deterministic_ce,
    )


def qwen3vl_listwise_choice_ce(
    *,
    model: Any,
    processor: Any,
    image: Tensor,
    query: str,
    choices: list[str] | tuple[str, ...],
    target_index: int,
    device: torch.device,
    require_image_grad: bool = True,
    do_resize: bool | None = None,
    reader_resize_contract: str | None = None,
    deterministic_ce: bool = False,
) -> ListwiseChoiceLossOutput:
    """Compute differentiable listwise CE over four teacher-forced option scores.

    Each option score is the negative mean token NLL, matching ``qwen3vl_choice_nll``
    evaluation. The target index is used only at the loss boundary and is never added to
    the Reader prompt.
    """

    if len(choices) != 4:
        raise ValueError("Listwise choice CE requires exactly four options.")
    if any(not isinstance(choice, str) or not choice.strip() for choice in choices):
        raise ValueError("Listwise choice CE requires four non-empty string options.")
    if len(set(choices)) != 4:
        raise ValueError("Listwise choice CE requires four distinct options.")
    if isinstance(target_index, bool) or not isinstance(target_index, int) or not 0 <= target_index < 4:
        raise ValueError("target_index must be an integer in [0, 3].")

    image, do_resize = _prepare_reader_image(
        image,
        do_resize=do_resize,
        reader_resize_contract=reader_resize_contract,
    )
    locked_resize_contract = reader_resize_contract is not None
    option_outputs: list[ReaderLossOutput] = []
    for choice in choices:
        option_outputs.append(
            _qwen3vl_target_only_ce_prepared(
                model=model,
                processor=processor,
                image=image,
                query=query,
                target=choice,
                device=device,
                require_image_grad=require_image_grad,
                do_resize=do_resize,
                locked_resize_contract=locked_resize_contract,
                deterministic_ce=deterministic_ce,
            )
        )

    choice_mean_nll = torch.stack([output.loss for output in option_outputs]).float()
    choice_logits = -choice_mean_nll
    target = torch.tensor([target_index], device=choice_logits.device, dtype=torch.long)
    if deterministic_ce:
        # CUDA NLLLoss backward is disallowed under strict deterministic algorithms.
        target_score = choice_logits.gather(dim=0, index=target).squeeze(0)
        loss = torch.logsumexp(choice_logits, dim=0) - target_score
    else:
        loss = F.cross_entropy(choice_logits.unsqueeze(0), target)

    return ListwiseChoiceLossOutput(
        loss=loss,
        choice_mean_nll=choice_mean_nll,
        choice_logits=choice_logits,
        target_ids=option_outputs[target_index].target_ids,
        choice_token_counts=tuple(int(output.target_ids.numel()) for output in option_outputs),
    )


def qwen3vl_choice_nll(
    *,
    model: Any,
    processor: Any,
    image: Tensor,
    query: str,
    choices: list[str] | tuple[str, ...],
    device: torch.device,
    do_resize: bool | None = None,
    reader_resize_contract: str | None = None,
    deterministic_ce: bool = False,
) -> ChoiceScoreOutput:
    """Score MCQ option texts by teacher-forced mean NLL for evaluation."""

    if len(choices) < 2:
        raise ValueError("Choice scoring requires at least two options.")
    scores: list[float] = []
    with torch.no_grad():
        image, do_resize = _prepare_reader_image(
            image,
            do_resize=do_resize,
            reader_resize_contract=reader_resize_contract,
        )
        locked_resize_contract = reader_resize_contract is not None
        for choice in choices:
            output = _qwen3vl_target_only_ce_prepared(
                model=model,
                processor=processor,
                image=image,
                query=query,
                target=choice,
                device=device,
                require_image_grad=False,
                do_resize=do_resize,
                locked_resize_contract=locked_resize_contract,
                deterministic_ce=deterministic_ce,
            )
            scores.append(float(output.loss.item()))
    predicted_index = min(range(len(scores)), key=scores.__getitem__)
    return ChoiceScoreOutput(mean_nll=tuple(scores), predicted_index=predicted_index)
