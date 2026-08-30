"""High-level recurrent DreamLite updater for episode training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch
from torch import Tensor, nn

from vision_memory.event_noise import make_event_generator

from .conditioning import encode_latent_path_condition
from .differentiable_mobile import DifferentiableDreamLiteMobileSampler
from .latent_codec import decode_model_latents_unit_interval


@dataclass(frozen=True)
class DreamLiteUpdateTrace:
    """Trace of one recurrent DreamLite event update for checkpoint analysis."""
    output_state: Tensor
    source_latents: Tensor
    output_latents: Tensor
    latent_trajectory: tuple[Tensor, ...]
    selected_step_indices: tuple[int, ...]
    persistent_state: str
    edit_start_sigma: float
    effective_sigma_schedule: tuple[float, ...]


def assert_no_frozen_parameter_grads(module: nn.Module, name: str) -> None:
    offenders = [
        parameter_name
        for parameter_name, parameter in module.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
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
    """DreamLite update with opt-in, externally visible float-RGB persistence."""

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

    @staticmethod
    def _validate_persistent_state(value: str) -> Literal["latent", "float_rgb"]:
        if value not in {"latent", "float_rgb"}:
            raise ValueError("persistent_state must be exactly 'latent' or 'float_rgb'.")
        return value

    @staticmethod
    def _validate_presentation_index(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("presentation_index must be a non-negative integer, not bool.")
        return value

    @staticmethod
    def _validate_persistent_rgb(state: Tensor) -> None:
        if state.ndim != 4 or state.shape[0] != 1 or state.shape[1] != 3:
            raise ValueError(
                "persistent_state='float_rgb' requires a batch-one BCHW tensor "
                f"with three channels, got {tuple(state.shape)}."
            )
        if not state.is_floating_point():
            raise ValueError("persistent_state='float_rgb' requires a floating-point tensor.")

    def encode_persistent_rgb(self, unit_image: Tensor) -> Tensor:
        """Encode the externally persisted RGB state at the start of an event."""

        self._validate_persistent_rgb(unit_image)
        return self.reencode_posterior_mean(unit_image.clamp(0.0, 1.0))

    def persist_rgb_state(self, state: Tensor) -> Tensor:
        """Decode a transient model latent to the only state allowed to persist."""

        unit_image = decode_model_latents_unit_interval(self.pipeline.vae, state, clamp=True)
        self._validate_persistent_rgb(unit_image)
        return unit_image

    def roundtrip_float_rgb(self, state: Tensor) -> Tensor:
        """Decode, clamp, and reload a latent without any hidden latent carry."""

        return self.encode_persistent_rgb(self.persist_rgb_state(state))

    def forward(
        self,
        state: Tensor,
        event_text: str,
        episode_id: str,
        turn_id: str | int,
        *,
        gradient_mode: Literal["full", "drtune", "drtune_stateful"] = "full",
        selected_step_indices: Iterable[int] | None = None,
        persistent_state: Literal["latent", "float_rgb"] = "latent",
        presentation_index: int = 0,
        noise_include_presentation_index: bool = True,
        edit_start_sigma: float = 1.0,
    ) -> Tensor:
        return self._forward_impl(
            state,
            event_text,
            episode_id,
            turn_id,
            gradient_mode=gradient_mode,
            selected_step_indices=selected_step_indices,
            persistent_state=persistent_state,
            presentation_index=presentation_index,
            noise_include_presentation_index=noise_include_presentation_index,
            edit_start_sigma=edit_start_sigma,
            return_trace=False,
        )

    def forward_with_trace(
        self,
        state: Tensor,
        event_text: str,
        episode_id: str,
        turn_id: str | int,
        *,
        gradient_mode: Literal["full", "drtune", "drtune_stateful"] = "full",
        selected_step_indices: Iterable[int] | None = None,
        persistent_state: Literal["latent", "float_rgb"] = "latent",
        presentation_index: int = 0,
        noise_include_presentation_index: bool = True,
        edit_start_sigma: float = 1.0,
    ) -> DreamLiteUpdateTrace:
        """Run one update while retaining the five-point four-step latent trajectory."""
        result = self._forward_impl(
            state,
            event_text,
            episode_id,
            turn_id,
            gradient_mode=gradient_mode,
            selected_step_indices=selected_step_indices,
            persistent_state=persistent_state,
            presentation_index=presentation_index,
            noise_include_presentation_index=noise_include_presentation_index,
            edit_start_sigma=edit_start_sigma,
            return_trace=True,
        )
        if not isinstance(result, DreamLiteUpdateTrace):
            raise RuntimeError("DreamLite trace implementation returned an invalid result.")
        return result

    def _forward_impl(
        self,
        state: Tensor,
        event_text: str,
        episode_id: str,
        turn_id: str | int,
        *,
        gradient_mode: Literal["full", "drtune", "drtune_stateful"],
        selected_step_indices: Iterable[int] | None,
        persistent_state: Literal["latent", "float_rgb"],
        presentation_index: int,
        noise_include_presentation_index: bool,
        edit_start_sigma: float,
        return_trace: bool,
    ) -> Tensor | DreamLiteUpdateTrace:
        resolved_selected_step_indices = (
            tuple(selected_step_indices) if selected_step_indices is not None else None
        )
        resolved_persistence = self._validate_persistent_state(persistent_state)
        resolved_presentation_index = self._validate_presentation_index(presentation_index)
        if not isinstance(noise_include_presentation_index, bool):
            raise TypeError("noise_include_presentation_index must be bool.")
        source_latents = self.encode_persistent_rgb(state) if resolved_persistence == "float_rgb" else state
        condition = encode_latent_path_condition(self.pipeline, source_latents, event_text)
        noise_episode_id = episode_id
        if noise_include_presentation_index and resolved_presentation_index:
            noise_episode_id = f"{episode_id}\0vlm-presentation-index-v1\0{resolved_presentation_index}"
        generator = make_event_generator(
            device=source_latents.device,
            global_seed=self.global_seed,
            episode_id=noise_episode_id,
            turn_id=turn_id,
        )
        noise = torch.randn(
            source_latents.shape,
            generator=generator,
            device=source_latents.device,
            dtype=source_latents.dtype,
        )
        sampler_output = self.sampler(
            source_latents=source_latents,
            noise_latents=noise,
            prompt_embeds=condition.prompt_embeds,
            prompt_attention_mask=condition.attention_mask,
            return_trajectory=return_trace,
            gradient_mode=gradient_mode,
            selected_step_indices=resolved_selected_step_indices,
            edit_start_sigma=edit_start_sigma,
        )
        updated_latents = sampler_output.latents
        output_state = (
            self.persist_rgb_state(updated_latents)
            if resolved_persistence == "float_rgb"
            else updated_latents
        )
        if not return_trace:
            return output_state
        trajectory = sampler_output.trajectory
        if trajectory is None:
            raise RuntimeError("DreamLite trace requested without a sampler trajectory.")
        return DreamLiteUpdateTrace(
            output_state=output_state,
            source_latents=source_latents,
            output_latents=updated_latents,
            latent_trajectory=trajectory,
            selected_step_indices=tuple(resolved_selected_step_indices or ()),
            persistent_state=resolved_persistence,
            edit_start_sigma=float(edit_start_sigma),
            effective_sigma_schedule=sampler_output.effective_sigmas,
        )

    def decode_for_reader(
        self,
        state: Tensor,
        *,
        persistent_state: Literal["latent", "float_rgb"] = "latent",
    ) -> Tensor:
        if self._validate_persistent_state(persistent_state) == "float_rgb":
            self._validate_persistent_rgb(state)
            return state
        return decode_model_latents_unit_interval(self.pipeline.vae, state, clamp=True)

    def decode_for_reencode(
        self,
        state: Tensor,
        *,
        persistent_state: Literal["latent", "float_rgb"] = "latent",
    ) -> Tensor:
        if self._validate_persistent_state(persistent_state) == "float_rgb":
            self._validate_persistent_rgb(state)
            return state
        return decode_model_latents_unit_interval(self.pipeline.vae, state, clamp=False)

    def reencode_posterior_mean(self, unit_image: Tensor) -> Tensor:
        """Differentiable unit-RGB to DreamLite model-latent encoding."""

        if unit_image.ndim == 3:
            unit_image = unit_image.unsqueeze(0)
        raw_image = unit_image * 2.0 - 1.0
        posterior = self.pipeline.vae.encode(raw_image, return_dict=True)
        vae_latents = _encoded_latent(posterior)
        scaling_factor = float(getattr(self.pipeline.vae.config, "scaling_factor", 1.0))
        shift_factor = float(getattr(self.pipeline.vae.config, "shift_factor", 0.0) or 0.0)
        return (vae_latents - shift_factor) * scaling_factor
