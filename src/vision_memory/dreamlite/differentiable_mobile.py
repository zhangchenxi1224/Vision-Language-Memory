"""A narrow differentiable copy of the DreamLite-mobile denoising core.

The official pipeline remains the inference/numerical reference. This module intentionally
does not implement generation, CFG, PIL output, model offload hooks, or implicit noise.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any, Iterable, Literal

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class DreamLiteSamplerOutput:
    latents: Tensor
    trajectory: tuple[Tensor, ...] | None = None
    effective_sigmas: tuple[float, ...] = ()


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

    @staticmethod
    def _validate_gradient_mode(value: str) -> Literal["full", "drtune", "drtune_stateful"]:
        if value not in {"full", "drtune", "drtune_stateful"}:
            raise ValueError("gradient_mode must be exactly 'full', 'drtune', or 'drtune_stateful'.")
        return value

    @staticmethod
    def _normalize_selected_step_indices(values: Iterable[int] | None) -> tuple[int, ...] | None:
        if values is None:
            return None
        normalized = tuple(values)
        if not normalized:
            raise ValueError("selected_step_indices must not be empty when provided.")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in normalized):
            raise TypeError("selected_step_indices must contain integers only.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("selected_step_indices must not contain duplicates.")
        return normalized

    @staticmethod
    def _validate_edit_start_sigma(value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("edit_start_sigma must be a real number in (0, 1].")
        resolved = float(value)
        if not math.isfinite(resolved) or not 0.0 < resolved <= 1.0:
            raise ValueError("edit_start_sigma must be finite and lie in (0, 1].")
        return resolved

    def _resolve_gradient_policy(
        self,
        *,
        gradient_mode: str,
        selected_step_indices: Iterable[int] | None,
        num_steps: int,
    ) -> tuple[Literal["full", "drtune", "drtune_stateful"], frozenset[int]]:
        mode = self._validate_gradient_mode(gradient_mode)
        selected = self._normalize_selected_step_indices(selected_step_indices)
        if mode == "full":
            if selected is not None:
                raise ValueError("selected_step_indices is valid only when gradient_mode='drtune'.")
            return mode, frozenset()
        if selected is None:
            raise ValueError(f"gradient_mode={mode!r} requires selected_step_indices.")
        if any(index < 0 or index >= num_steps for index in selected):
            raise ValueError(f"selected_step_indices must be in [0, {num_steps - 1}].")
        return mode, frozenset(selected)

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

    @staticmethod
    def _raw_sigmas_for_effective_schedule(
        effective_sigmas: Iterable[float],
        *,
        config: Any,
        mu: float,
    ) -> list[float]:
        """Invert the scheduler shift so its *effective* sigmas match the flow state.

        ``FlowMatchEulerDiscreteScheduler.set_timesteps`` shifts even explicitly
        supplied sigmas.  An img2img state must be mixed with the post-shift
        sigma consumed by ``scheduler.step`` and the denoiser timestep.  R6
        therefore specifies effective flow sigmas and maps them back to the raw
        values expected by ``set_timesteps``.
        """

        values = [float(value) for value in effective_sigmas]
        if not values or any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in values):
            raise ValueError("Effective scheduler sigmas must be finite values in (0, 1].")
        unsupported = {
            name: _config_value(config, name, False)
            for name in (
                "use_karras_sigmas",
                "use_exponential_sigmas",
                "use_beta_sigmas",
                "invert_sigmas",
            )
            if _config_value(config, name, False)
        }
        if _config_value(config, "shift_terminal", None):
            unsupported["shift_terminal"] = _config_value(config, "shift_terminal", None)
        if unsupported:
            raise ValueError(
                "Effective-sigma inversion is unsupported for scheduler post-processing: "
                f"{unsupported}"
            )

        if _config_value(config, "use_dynamic_shifting", False):
            shift_type = _config_value(config, "time_shift_type", "exponential")
            if shift_type == "exponential":
                scale = math.exp(float(mu))
            elif shift_type == "linear":
                scale = float(mu)
            else:
                raise ValueError(f"Unsupported dynamic time_shift_type: {shift_type!r}")
            if not math.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Invalid scheduler shift scale: {scale}")
            return [1.0 / (1.0 + scale * (1.0 / value - 1.0)) for value in values]

        shift = float(_config_value(config, "shift", 1.0))
        if not math.isfinite(shift) or shift <= 0.0:
            raise ValueError(f"Invalid static scheduler shift: {shift}")
        return [value / (shift - value * (shift - 1.0)) for value in values]

    def _prepare_timesteps(
        self,
        latents: Tensor,
        num_steps: int,
        sigmas: Iterable[float] | None,
        *,
        sigmas_are_effective: bool = False,
    ) -> tuple[Tensor, tuple[float, ...]]:
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

        scheduler_inputs = (
            self._raw_sigmas_for_effective_schedule(sigma_values, config=config, mu=mu)
            if sigmas_are_effective
            else sigma_values
        )
        # set_timesteps resets the scheduler's mutable step index for every event update.
        self.scheduler.set_timesteps(sigmas=scheduler_inputs, device=latents.device, mu=mu)
        effective = self.scheduler.sigmas[:num_steps]
        if not isinstance(effective, Tensor) or effective.numel() != num_steps:
            raise RuntimeError("DreamLite scheduler did not expose the expected effective sigma schedule.")
        if sigmas_are_effective:
            expected = torch.tensor(sigma_values, device=effective.device, dtype=effective.dtype)
            if not torch.allclose(effective, expected, rtol=2e-6, atol=2e-6):
                raise RuntimeError(
                    "DreamLite scheduler effective sigmas do not match the requested flow schedule: "
                    f"expected={expected.detach().cpu().tolist()}, "
                    f"observed={effective.detach().cpu().tolist()}"
                )
        return self.scheduler.timesteps, tuple(float(value) for value in effective.detach().cpu().tolist())

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
        gradient_mode: Literal["full", "drtune", "drtune_stateful"] = "full",
        selected_step_indices: Iterable[int] | None = None,
        edit_start_sigma: float = 1.0,
    ) -> DreamLiteSamplerOutput:
        self._validate_inputs(source_latents, noise_latents, prompt_embeds, prompt_attention_mask)
        resolved_start_sigma = self._validate_edit_start_sigma(edit_start_sigma)
        resolved_gradient_mode, resolved_selected_steps = self._resolve_gradient_policy(
            gradient_mode=gradient_mode,
            selected_step_indices=selected_step_indices,
            num_steps=num_steps,
        )

        if sigmas is not None and resolved_start_sigma != 1.0:
            raise ValueError("Explicit sigmas cannot be combined with edit_start_sigma != 1.")
        resolved_sigmas = (
            sigmas
            if sigmas is not None or resolved_start_sigma == 1.0
            else torch.linspace(
                resolved_start_sigma,
                resolved_start_sigma / num_steps,
                num_steps,
            ).tolist()
        )
        timesteps, effective_sigmas = self._prepare_timesteps(
            source_latents,
            num_steps,
            resolved_sigmas,
            sigmas_are_effective=resolved_start_sigma != 1.0,
        )
        # DreamLite is flow-matching trained with
        # x_sigma=(1-sigma)*x_0 + sigma*epsilon.  R5's sigma=1 path therefore
        # starts every event from pure noise.  A smaller start sigma is a
        # pretrained-manifold-consistent image-to-image update: it anchors the
        # ODE trajectory in the previous persistent state instead of redrawing
        # the complete state from scratch.  The scheduler contract above makes
        # this sigma the *effective post-shift* sigma, not merely its raw input.
        effective_start_sigma = effective_sigmas[0]
        if resolved_start_sigma != 1.0 and not math.isclose(
            effective_start_sigma,
            resolved_start_sigma,
            rel_tol=2e-6,
            abs_tol=2e-6,
        ):
            raise RuntimeError(
                "Source interpolation and scheduler start sigma diverged: "
                f"mix={resolved_start_sigma}, scheduler={effective_start_sigma}"
            )
        latents = (
            noise_latents
            if resolved_start_sigma == 1.0
            else source_latents.mul(1.0 - effective_start_sigma).add(
                noise_latents, alpha=effective_start_sigma
            )
        )
        if time_ids is None:
            height = source_latents.shape[-2] * self.vae_scale_factor
            width = source_latents.shape[-1] * self.vae_scale_factor
            time_ids = torch.tensor([[width, height]], device=latents.device, dtype=latents.dtype)
        else:
            time_ids = time_ids.to(device=latents.device, dtype=latents.dtype)

        trajectory = [latents] if return_trajectory else None
        for step_index, timestep in enumerate(timesteps):
            model_input = torch.cat([latents, source_latents], dim=3)
            if resolved_gradient_mode == "full":
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
            elif step_index in resolved_selected_steps:
                # Classic DRTune stops gradients at the selected denoiser input.
                # R5's stateful variant keeps that input connected so a delayed
                # query can assign credit to prior recurrent states.  Both modes
                # still expose parameter gradients at selected U-Net steps only;
                # scheduler updates deliberately remain outside ``no_grad``.
                selected_model_input = (
                    model_input if resolved_gradient_mode == "drtune_stateful" else model_input.detach()
                )
                if self.checkpoint_unet and torch.is_grad_enabled():
                    noise_pair = checkpoint(
                        self._unet_step,
                        selected_model_input,
                        timestep,
                        prompt_embeds,
                        prompt_attention_mask,
                        time_ids,
                        use_reentrant=False,
                    )
                else:
                    noise_pair = self._unet_step(
                        selected_model_input,
                        timestep,
                        prompt_embeds,
                        prompt_attention_mask,
                        time_ids,
                    )
            else:
                with torch.no_grad():
                    noise_pair = self._unet_step(
                        model_input.detach(),
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
            effective_sigmas=effective_sigmas,
        )
