"""Differentiable DreamLite adapters that leave the upstream pipeline untouched."""

from .differentiable_mobile import DifferentiableDreamLiteMobileSampler, DreamLiteSamplerOutput
from .latent_codec import decode_model_latents_raw, decode_model_latents_unit_interval, freeze_module
from .recurrent import DreamLiteRecurrentUpdater, assert_no_frozen_parameter_grads

__all__ = [
    "DifferentiableDreamLiteMobileSampler",
    "DreamLiteSamplerOutput",
    "DreamLiteRecurrentUpdater",
    "assert_no_frozen_parameter_grads",
    "decode_model_latents_raw",
    "decode_model_latents_unit_interval",
    "freeze_module",
]
