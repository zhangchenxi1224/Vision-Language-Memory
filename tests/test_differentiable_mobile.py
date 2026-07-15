from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import DifferentiableDreamLiteMobileSampler  # noqa: E402


class MockFlowScheduler:
    order = 1
    config = {
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
        "base_shift": 0.5,
        "max_shift": 1.16,
    }

    def set_timesteps(self, *, sigmas, device, mu):
        del mu
        values = torch.tensor(list(sigmas) + [0.0], device=device, dtype=torch.float32)
        self.sigmas = values
        self.timesteps = values[:-1] * 1000.0
        self._step_index = 0

    def step(self, model_output, timestep, sample, return_dict=False):
        del timestep, return_dict
        dt = self.sigmas[self._step_index + 1] - self.sigmas[self._step_index]
        self._step_index += 1
        return (sample + dt.to(sample.dtype) * model_output,)


class MockDreamLiteUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        sample,
        *,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        added_cond_kwargs,
        return_dict,
    ):
        del timestep, encoder_attention_mask, added_cond_kwargs, return_dict
        target, source = sample.chunk(2, dim=-1)
        prompt_term = encoder_hidden_states.mean().to(sample.dtype)
        target_velocity = self.gain * (target + 0.25 * source) + 0.01 * prompt_term
        source_velocity = torch.zeros_like(source)
        return (torch.cat([target_velocity, source_velocity], dim=-1),)


def make_inputs():
    source = torch.ones(1, 4, 8, 8, requires_grad=True)
    noise = torch.zeros_like(source)
    prompt = torch.ones(1, 3, 6)
    mask = torch.ones(1, 3, dtype=torch.long)
    return source, noise, prompt, mask


class DifferentiableSamplerContractTest(unittest.TestCase):
    def make_sampler(self, *, checkpoint_unet=False):
        return DifferentiableDreamLiteMobileSampler(
            unet=MockDreamLiteUNet(),
            scheduler=MockFlowScheduler(),
            vae_scale_factor=8,
            checkpoint_unet=checkpoint_unet,
        )

    def test_single_event_gradients_reach_source_and_trainable_parameter(self):
        sampler = self.make_sampler()
        source, noise, prompt, mask = make_inputs()
        output = sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
        )
        loss = output.latents.square().mean()
        loss.backward()

        self.assertIsNotNone(source.grad)
        self.assertGreater(source.grad.norm().item(), 0.0)
        self.assertIsNotNone(sampler.unet.gain.grad)
        self.assertGreater(abs(sampler.unet.gain.grad.item()), 0.0)

    def test_two_event_bptt_reaches_intermediate_state(self):
        sampler = self.make_sampler()
        source, noise, prompt, mask = make_inputs()
        z1 = sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
        ).latents
        z1.retain_grad()
        z2 = sampler(
            source_latents=z1,
            noise_latents=torch.zeros_like(z1),
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
        ).latents
        z2.square().mean().backward()

        self.assertIsNotNone(z1.grad)
        self.assertGreater(z1.grad.norm().item(), 0.0)
        self.assertGreater(source.grad.norm().item(), 0.0)

    def test_detach_negative_control_breaks_intermediate_path(self):
        sampler = self.make_sampler()
        source, noise, prompt, mask = make_inputs()
        z1 = sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
        ).latents
        z1.retain_grad()
        z2 = sampler(
            source_latents=z1.detach(),
            noise_latents=torch.zeros_like(z1),
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
        ).latents
        z2.square().mean().backward()

        self.assertIsNone(z1.grad)

    def test_scheduler_state_is_reset_for_each_call(self):
        sampler = self.make_sampler()
        source, noise, prompt, mask = make_inputs()
        kwargs = dict(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
        )
        first = sampler(**kwargs).latents
        second = sampler(**kwargs).latents
        torch.testing.assert_close(first, second)

    def test_non_reentrant_checkpoint_preserves_gradients(self):
        sampler = self.make_sampler(checkpoint_unet=True)
        source, noise, prompt, mask = make_inputs()
        output = sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
        )
        output.latents.square().mean().backward()
        self.assertGreater(source.grad.norm().item(), 0.0)
        self.assertGreater(abs(sampler.unet.gain.grad.item()), 0.0)


if __name__ == "__main__":
    unittest.main()

