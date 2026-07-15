"""High-level recurrent DreamLite updater for episode training."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from vision_memory.event_noise import make_event_generator

from .conditioning import encode_latent_path_condition
from .differentiable_mobile import DifferentiableDreamLiteMobileSampler
from .latent_codec import decode_model_latents_unit_interval


def assert_no_frozen_parameter_grads(module: nn.Module, name: str) -> None:
    offenders = [parameter_name for parameter_name, parameter in module.named_parameters() if not parameter.requires_grad and parameter.grad is not None]
    if offenders:
        preview = ", ".join(offenders[:8])
        raise RuntimeError(f"Frozen {name} parameters accumulated gradients: {preview}")


def _encoded_latent(output: Any) -> Tensor:
    direct_latents = getattr(output, "latents", None)
    if isinstance(direct_latents, Tensor):
        return direct_latents
    distribution = getattr(output, "latent_dist", None)
    if distribution is None and isinstance(output, (tuple, list)) and output:
        distribution = output[0]
    if isinstance(distribution, Tensor):
        return distribution
    if distribution is None:
        raise TypeError(f"Unsupported VAE encode output: {type(output)!r}")
    mode = getattr(distribution, "mode", None)
    if callable(mode):
        return mode()
    mean = getattr(distribution, "mean", None)
    if isinstance(mean, Tensor):
        return mean
    raise TypeError(f"VAE posterior exposes neither mode() nor mean: {type(distribution)!r}")


class DreamLiteRecurrentUpdater(nn.Module):
    """Direct-latent RNN state update with explicit stop-gradient conditioning."""

    def __init__(
        self,
        *,
        pipeline: Any,
        global_seed: int,
        checkpoint_unet: bool = True,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.global_seed = int(global_seed)
        self.sampler = DifferentiableDreamLiteMobileSampler.from_pipeline(
            pipeline,
            checkpoint_unet=checkpoint_unet,
        )

    def forward(self, state: Tensor, event_text: str, episode_id: str, turn_id: str | int) -> Tensor:
        condition = encode_latent_path_condition(self.pipeline, state, event_text)
        generator = make_event_generator(
            device=state.device,
            global_seed=self.global_seed,
            episode_id=episode_id,
            turn_id=turn_id,
        )
        noise = torch.randn(state.shape, generator=generator, device=state.device, dtype=state.dtype)
        return self.sampler(
            source_latents=state,
            noise_latents=noise,
            prompt_embeds=condition.prompt_embeds,
            prompt_attention_mask=condition.attention_mask,
            return_trajectory=False,
        ).latents

    def decode_for_reader(self, state: Tensor) -> Tensor:
        return decode_model_latents_unit_interval(self.pipeline.vae, state, clamp=True)

    def decode_for_reencode(self, state: Tensor) -> Tensor:
        return decode_model_latents_unit_interval(self.pipeline.vae, state, clamp=False)

    def reencode_posterior_mean(self, unit_image: Tensor) -> Tensor:
        """Differentiable RGB bottleneck used only by the decode/re-encode ablation."""

        if unit_image.ndim == 3:
            unit_image = unit_image.unsqueeze(0)
        raw_image = unit_image * 2.0 - 1.0
        posterior = self.pipeline.vae.encode(raw_image, return_dict=True)
        vae_latents = _encoded_latent(posterior)
        scaling_factor = float(getattr(self.pipeline.vae.config, "scaling_factor", 1.0))
        shift_factor = float(getattr(self.pipeline.vae.config, "shift_factor", 0.0) or 0.0)
        return (vae_latents - shift_factor) * scaling_factor
