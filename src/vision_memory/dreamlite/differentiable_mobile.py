"""A narrow differentiable copy of the DreamLite-mobile denoising core.

The official pipeline remains the inference/numerical reference. This module intentionally
does not implement generation, CFG, PIL output, model offload hooks, or implicit noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class DreamLiteSamplerOutput:
    latents: Tensor
    trajectory: tuple[Tensor, ...] | None = None


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    return image_seq_len * slope + intercept


def _config_value(config: Any, key: str, default: Any) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _extract_unet_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    raise TypeError(f"Unsupported U-Net output type: {type(output)!r}")


class DifferentiableDreamLiteMobileSampler(nn.Module):
    """Edit-only, explicit-noise, latent-returning DreamLite-mobile sampler."""

    def __init__(
        self,
        *,
        unet: nn.Module,
        scheduler: Any,
        vae_scale_factor: int = 8,
        checkpoint_unet: bool = False,
    ) -> None:
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
        self.vae_scale_factor = int(vae_scale_factor)
        self.checkpoint_unet = bool(checkpoint_unet)

    @classmethod
    def from_pipeline(cls, pipeline: Any, *, checkpoint_unet: bool = False) -> "DifferentiableDreamLiteMobileSampler":
        return cls(
            unet=pipeline.unet,
            scheduler=pipeline.scheduler,
            vae_scale_factor=int(pipeline.vae_scale_factor),
            checkpoint_unet=checkpoint_unet,
        )

    def _validate_inputs(
        self,
        source_latents: Tensor,
        noise_latents: Tensor,
        prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
    ) -> None:
        if source_latents.ndim != 4:
            raise ValueError(f"source_latents must be BCHW, got {tuple(source_latents.shape)}")
        if source_latents.shape != noise_latents.shape:
            raise ValueError(
                "source_latents and noise_latents must have identical shapes; "
                f"got {tuple(source_latents.shape)} and {tuple(noise_latents.shape)}"
            )
        if source_latents.shape[0] != 1:
            raise ValueError("The first DreamLite-mobile milestone supports batch size 1 only.")
        if prompt_embeds.shape[0] != source_latents.shape[0]:
            raise ValueError("prompt_embeds batch does not match latent batch.")
        if prompt_attention_mask.shape[:2] != prompt_embeds.shape[:2]:
            raise ValueError("prompt_attention_mask must match the first two prompt_embeds dimensions.")
        if source_latents.device != noise_latents.device:
            raise ValueError("source_latents and noise_latents must be on the same device.")
        if source_latents.dtype != noise_latents.dtype:
            raise ValueError("source_latents and noise_latents must use the same dtype.")
        if prompt_embeds.device != source_latents.device:
            raise ValueError("prompt_embeds and source_latents must be on the same device.")
        if prompt_attention_mask.device != source_latents.device:
            raise ValueError("prompt_attention_mask and source_latents must be on the same device.")
        if prompt_embeds.dtype != source_latents.dtype:
            raise ValueError("prompt_embeds and source_latents must use the same dtype.")
        if not source_latents.is_floating_point() or not prompt_embeds.is_floating_point():
            raise ValueError("DreamLite latent and prompt tensors must use floating-point dtypes.")

    def _prepare_timesteps(self, latents: Tensor, num_steps: int, sigmas: Iterable[float] | None) -> Tensor:
        if num_steps != 4:
            raise ValueError("The first training wrapper is deliberately restricted to DreamLite-mobile's 4 steps.")
        sigma_values = list(sigmas) if sigmas is not None else torch.linspace(1.0, 1.0 / num_steps, num_steps).tolist()
        if len(sigma_values) != num_steps:
            raise ValueError(f"Expected {num_steps} sigma values, got {len(sigma_values)}")

        image_seq_len = latents.shape[2] * latents.shape[3] // 4
        config = self.scheduler.config
        mu = calculate_shift(
            image_seq_len,
            _config_value(config, "base_image_seq_len", 256),
            _config_value(config, "max_image_seq_len", 4096),
            _config_value(config, "base_shift", 0.5),
            _config_value(config, "max_shift", 1.16),
        )

        # set_timesteps resets the scheduler's mutable step index for every event update.
        self.scheduler.set_timesteps(sigmas=sigma_values, device=latents.device, mu=mu)
        return self.scheduler.timesteps

    def _unet_step(
        self,
        model_input: Tensor,
        timestep: Tensor,
        prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        time_ids: Tensor,
    ) -> Tensor:
        output = self.unet(
            model_input,
            timestep=timestep.expand(model_input.shape[0]).to(model_input.dtype),
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            added_cond_kwargs={"time_ids": time_ids},
            return_dict=False,
        )
        return _extract_unet_tensor(output)

    def forward(
        self,
        *,
        source_latents: Tensor,
        noise_latents: Tensor,
        prompt_embeds: Tensor,
        prompt_attention_mask: Tensor,
        num_steps: int = 4,
        sigmas: Iterable[float] | None = None,
        time_ids: Tensor | None = None,
        return_trajectory: bool = False,
    ) -> DreamLiteSamplerOutput:
        self._validate_inputs(source_latents, noise_latents, prompt_embeds, prompt_attention_mask)

        latents = noise_latents
        timesteps = self._prepare_timesteps(latents, num_steps, sigmas)
        if time_ids is None:
            height = source_latents.shape[-2] * self.vae_scale_factor
            width = source_latents.shape[-1] * self.vae_scale_factor
            time_ids = torch.tensor([[width, height]], device=latents.device, dtype=latents.dtype)
        else:
            time_ids = time_ids.to(device=latents.device, dtype=latents.dtype)

        trajectory = [latents] if return_trajectory else None
        for timestep in timesteps:
            model_input = torch.cat([latents, source_latents], dim=3)
            if self.checkpoint_unet and torch.is_grad_enabled():
                noise_pair = checkpoint(
                    self._unet_step,
                    model_input,
                    timestep,
                    prompt_embeds,
                    prompt_attention_mask,
                    time_ids,
                    use_reentrant=False,
                )
            else:
                noise_pair = self._unet_step(
                    model_input,
                    timestep,
                    prompt_embeds,
                    prompt_attention_mask,
                    time_ids,
                )

            noise_prediction = noise_pair[..., : latents.shape[-1]]
            latents = self.scheduler.step(noise_prediction, timestep, latents, return_dict=False)[0]
            if trajectory is not None:
                trajectory.append(latents)

        return DreamLiteSamplerOutput(
            latents=latents,
            trajectory=tuple(trajectory) if trajectory is not None else None,
        )
