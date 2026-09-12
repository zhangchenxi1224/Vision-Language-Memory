"""DreamLite edit conditioning for the first latent-path BPTT milestone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .latent_codec import decode_model_latents_raw


@dataclass(frozen=True)
class EditConditioning:
    prompt_embeds: Tensor
    attention_mask: Tensor


def official_mobile_edit_prompt(event_text: str) -> str:
    return (
        "[Edit]: A diptych with two side-by-side images of the same scene. "
        f"Compared to the right side, the left one has {event_text}"
    )


@torch.no_grad()
def encode_image_edit_condition(pipeline: Any, source_image: Any, event_text: str,
                                *, device: Any, dtype: Any,
                                prompt_style: str = "official_raw") -> EditConditioning:
    """Use the original source PIL image, as upstream LoRA training does.

    The LoRA example passes raw editing instructions; Mobile inference wraps
    them in a diptych description. Choose explicitly and use the same choice
    in training and evaluation rather than silently changing the condition.
    """
    if prompt_style not in {"official_raw", "mobile_diptych"}:
        raise ValueError("Unknown DreamLite prompt style")
    prompt = event_text if prompt_style == "official_raw" else official_mobile_edit_prompt(event_text)
    embeds, mask = pipeline.encode_prompt(mode="edit", prompts=[prompt], image=source_image,
                                         device=device, dtype=dtype)
    return EditConditioning(embeds.detach(), mask.detach())


@torch.no_grad()
def encode_latent_path_condition(pipeline: Any, source_latents: Tensor, event_text: str) -> EditConditioning:
    """Build the internal Qwen3-VL-2B condition with an explicit stop-gradient.

    Forward values still depend on the current state image. Only the auxiliary
    source-image-to-condition gradient is stopped. Recurrent gradients remain available
    through the U-Net's direct source-latent spatial-concatenation path.
    """

    if source_latents.ndim != 4 or source_latents.shape[0] != 1:
        raise ValueError("The first DreamLite-mobile milestone supports batch size 1 only.")

    decoded = decode_model_latents_raw(pipeline.vae, source_latents.detach())
    source_pil = pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
    prompt = official_mobile_edit_prompt(event_text)
    embeds, mask = pipeline.encode_prompt(
        mode="edit",
        prompts=[prompt],
        image=source_pil,
        device=source_latents.device,
        dtype=source_latents.dtype,
    )
    return EditConditioning(prompt_embeds=embeds.detach(), attention_mask=mask.detach())

