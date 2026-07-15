"""Canonical DreamLite model-latent decoding helpers.

These functions deliberately do not use torch.no_grad(). Freezing VAE parameters and
preserving input gradients are separate concerns.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze parameters without disabling autograd with respect to module inputs."""

    module.requires_grad_(False)
    module.eval()
    return module


def _first_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    raise TypeError(f"Unsupported VAE output type: {type(output)!r}")


def decode_model_latents_raw(vae: nn.Module, latents: Tensor) -> Tensor:
    """Decode DreamLite model-space latents to the VAE's raw image tensor.

    The scale/shift convention mirrors Diffusers 0.39.0 DreamLiteMobilePipeline.
    VAE weights may be frozen, but this forward must remain in the autograd graph when
    a downstream Reader loss should reach the latent or DreamLite LoRA parameters.
    """

    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    shift_factor = float(getattr(vae.config, "shift_factor", 0.0) or 0.0)
    vae_latents = latents / scaling_factor + shift_factor
    return _first_tensor(vae.decode(vae_latents, return_dict=False))


def decode_model_latents_unit_interval(vae: nn.Module, latents: Tensor, *, clamp: bool = True) -> Tensor:
    """Decode to RGB-like floats in [0, 1] for a tensor-native Reader processor."""

    image = decode_model_latents_raw(vae, latents) * 0.5 + 0.5
    return image.clamp(0.0, 1.0) if clamp else image
