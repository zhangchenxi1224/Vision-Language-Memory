from __future__ import annotations

import math
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

    def __init__(self):
        self.step_grad_enabled = []
        self.requested_sigmas = []

    def set_timesteps(self, *, sigmas, device, mu):
        del mu
        self.requested_sigmas = list(sigmas)
        values = torch.tensor(self.requested_sigmas + [0.0], device=device, dtype=torch.float32)
        self.sigmas = values
        self.timesteps = values[:-1] * 1000.0
        self._step_index = 0
        self.step_grad_enabled = []

    def step(self, model_output, timestep, sample, return_dict=False):
        del timestep, return_dict
        self.step_grad_enabled.append(torch.is_grad_enabled())
        dt = self.sigmas[self._step_index + 1] - self.sigmas[self._step_index]
        self._step_index += 1
        return (sample + dt.to(sample.dtype) * model_output,)


class ShiftedMockFlowScheduler(MockFlowScheduler):
    """Faithful dynamic-shift behavior used by DreamLite's FlowMatch scheduler."""

    config = {
        "base_image_seq_len": 16,
        "max_image_seq_len": 4096,
        "base_shift": 0.5,
        "max_shift": 1.16,
        "use_dynamic_shifting": True,
        "time_shift_type": "exponential",
        "shift_terminal": None,
        "use_karras_sigmas": False,
        "use_exponential_sigmas": False,
        "use_beta_sigmas": False,
        "invert_sigmas": False,
    }

    def set_timesteps(self, *, sigmas, device, mu):
        self.requested_sigmas = list(sigmas)
        scale = math.exp(mu)
        shifted = [scale / (scale + (1.0 / value - 1.0)) for value in self.requested_sigmas]
        values = torch.tensor(shifted + [0.0], device=device, dtype=torch.float32)
        self.sigmas = values
        self.timesteps = values[:-1] * 1000.0
        self._step_index = 0
        self.step_grad_enabled = []


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


class StepwiseDreamLiteUNet(nn.Module):
    """Mock with one independent parameter per denoising step."""

    def __init__(self):
        super().__init__()
        self.gains = nn.Parameter(torch.tensor([0.2, 0.3, 0.4, 0.5]))
        self.call_grad_enabled = []
        self.call_input_requires_grad = []

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
        del encoder_attention_mask, added_cond_kwargs, return_dict
        self.call_grad_enabled.append(torch.is_grad_enabled())
        self.call_input_requires_grad.append(sample.requires_grad)
        step_index = int(round((1000.0 - float(timestep[0].item())) / 250.0))
        target, source = sample.chunk(2, dim=-1)
        prompt_term = encoder_hidden_states.mean().to(sample.dtype)
        target_velocity = self.gains[step_index] * (target + 0.25 * source) + 0.01 * prompt_term
        return (torch.cat([target_velocity, torch.zeros_like(source)], dim=-1),)


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

    def test_source_anchored_edit_uses_flow_matching_interpolation_and_scaled_schedule(self):
        source, noise, prompt, mask = make_inputs()
        noise = torch.full_like(noise, 3.0)
        scheduler = MockFlowScheduler()
        sampler = DifferentiableDreamLiteMobileSampler(
            unet=MockDreamLiteUNet(),
            scheduler=scheduler,
        )
        output = sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
            edit_start_sigma=0.5,
            return_trajectory=True,
        )
        self.assertEqual(scheduler.requested_sigmas, [0.5, 0.375, 0.25, 0.125])
        self.assertEqual(output.effective_sigmas, (0.5, 0.375, 0.25, 0.125))
        self.assertIsNotNone(output.trajectory)
        expected_start = source * 0.5 + noise * 0.5
        torch.testing.assert_close(output.trajectory[0], expected_start)
        output.latents.square().mean().backward()
        self.assertIsNotNone(source.grad)
        self.assertGreater(source.grad.norm().item(), 0.0)

    def test_source_anchor_inverts_dynamic_shift_to_match_effective_flow_sigma(self):
        source, noise, prompt, mask = make_inputs()
        noise = torch.full_like(noise, 3.0)
        scheduler = ShiftedMockFlowScheduler()
        sampler = DifferentiableDreamLiteMobileSampler(
            unet=MockDreamLiteUNet(),
            scheduler=scheduler,
        )
        output = sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
            edit_start_sigma=0.5,
            return_trajectory=True,
        )

        expected_effective = torch.tensor([0.5, 0.375, 0.25, 0.125])
        torch.testing.assert_close(scheduler.sigmas[:4], expected_effective)
        self.assertEqual(output.effective_sigmas, (0.5, 0.375, 0.25, 0.125))
        scale = math.exp(0.5)
        expected_raw = [
            1.0 / (1.0 + scale * (1.0 / value - 1.0))
            for value in expected_effective.tolist()
        ]
        for observed, expected in zip(scheduler.requested_sigmas, expected_raw, strict=True):
            self.assertAlmostEqual(observed, expected, places=7)
        self.assertIsNotNone(output.trajectory)
        torch.testing.assert_close(output.trajectory[0], source * 0.5 + noise * 0.5)

    def test_dynamic_shift_does_not_change_legacy_raw_sigma_schedule(self):
        source, noise, prompt, mask = make_inputs()
        scheduler = ShiftedMockFlowScheduler()
        output = DifferentiableDreamLiteMobileSampler(
            unet=MockDreamLiteUNet(),
            scheduler=scheduler,
        )(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
            edit_start_sigma=1.0,
            return_trajectory=True,
        )
        self.assertEqual(scheduler.requested_sigmas, [1.0, 0.75, 0.5, 0.25])
        self.assertEqual(output.effective_sigmas[0], 1.0)
        self.assertIsNotNone(output.trajectory)
        self.assertTrue(torch.equal(output.trajectory[0], noise))

    def test_sigma_one_remains_exactly_the_legacy_pure_noise_path(self):
        source, noise, prompt, mask = make_inputs()
        default_sampler = self.make_sampler()
        explicit_sampler = self.make_sampler()
        explicit_sampler.unet.load_state_dict(default_sampler.unet.state_dict())
        default = default_sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
            return_trajectory=True,
        )
        explicit = explicit_sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
            edit_start_sigma=1.0,
            return_trajectory=True,
        )
        self.assertTrue(torch.equal(default.latents, explicit.latents))
        self.assertTrue(torch.equal(default.trajectory[0], noise))
        self.assertTrue(torch.equal(explicit.trajectory[0], noise))

    def test_invalid_source_anchor_contract_fails_closed(self):
        sampler = self.make_sampler()
        source, noise, prompt, mask = make_inputs()
        kwargs = {
            "source_latents": source,
            "noise_latents": noise,
            "prompt_embeds": prompt,
            "prompt_attention_mask": mask,
        }
        for value in (0.0, -0.1, 1.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sampler(**kwargs, edit_start_sigma=value)
        with self.assertRaises(TypeError):
            sampler(**kwargs, edit_start_sigma=True)
        with self.assertRaisesRegex(ValueError, "Explicit sigmas"):
            sampler(**kwargs, edit_start_sigma=0.5, sigmas=[0.5, 0.375, 0.25, 0.125])

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

    def test_drtune_forward_is_bitwise_equal_to_full_forward(self):
        source, noise, prompt, mask = make_inputs()
        full_unet = StepwiseDreamLiteUNet()
        full_sampler = DifferentiableDreamLiteMobileSampler(
            unet=full_unet,
            scheduler=MockFlowScheduler(),
        )
        full = full_sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
        ).latents

        for selected_step in range(4):
            with self.subTest(selected_step=selected_step):
                drtune_unet = StepwiseDreamLiteUNet()
                drtune_unet.load_state_dict(full_unet.state_dict())
                drtune_sampler = DifferentiableDreamLiteMobileSampler(
                    unet=drtune_unet,
                    scheduler=MockFlowScheduler(),
                )
                drtune = drtune_sampler(
                    source_latents=source,
                    noise_latents=noise,
                    prompt_embeds=prompt,
                    prompt_attention_mask=mask,
                    gradient_mode="drtune",
                    selected_step_indices=(selected_step,),
                ).latents
                self.assertTrue(torch.equal(full, drtune))

    def test_each_drtune_step_has_exclusive_parameter_grad_and_scheduler_graph(self):
        for selected_step in range(4):
            with self.subTest(selected_step=selected_step):
                source, noise, prompt, mask = make_inputs()
                unet = StepwiseDreamLiteUNet()
                scheduler = MockFlowScheduler()
                sampler = DifferentiableDreamLiteMobileSampler(
                    unet=unet,
                    scheduler=scheduler,
                )
                output = sampler(
                    source_latents=source,
                    noise_latents=noise,
                    prompt_embeds=prompt,
                    prompt_attention_mask=mask,
                    gradient_mode="drtune",
                    selected_step_indices=(selected_step,),
                )
                output.latents.square().mean().backward()

                expected_grad = torch.zeros_like(unet.gains.grad)
                self.assertGreater(abs(unet.gains.grad[selected_step].item()), 0.0)
                expected_grad[selected_step] = unet.gains.grad[selected_step]
                torch.testing.assert_close(unet.gains.grad, expected_grad)
                expected_grad_modes = [index == selected_step for index in range(4)]
                self.assertEqual(unet.call_grad_enabled, expected_grad_modes)
                self.assertEqual(unet.call_input_requires_grad, [False, False, False, False])
                self.assertEqual(scheduler.step_grad_enabled, [True, True, True, True])
                self.assertIsNone(source.grad)

    def test_drtune_stateful_keeps_source_path_and_selected_parameter_only(self):
        for selected_step in range(4):
            with self.subTest(selected_step=selected_step):
                source, noise, prompt, mask = make_inputs()
                unet = StepwiseDreamLiteUNet()
                scheduler = MockFlowScheduler()
                sampler = DifferentiableDreamLiteMobileSampler(unet=unet, scheduler=scheduler)
                output = sampler(
                    source_latents=source,
                    noise_latents=noise,
                    prompt_embeds=prompt,
                    prompt_attention_mask=mask,
                    gradient_mode="drtune_stateful",
                    selected_step_indices=(selected_step,),
                )
                output.latents.square().mean().backward()

                self.assertIsNotNone(source.grad)
                self.assertGreater(source.grad.norm().item(), 0.0)
                nonzero = unet.gains.grad.ne(0).nonzero().flatten().tolist()
                self.assertEqual(nonzero, [selected_step])
                self.assertEqual(
                    unet.call_input_requires_grad,
                    [index == selected_step for index in range(4)],
                )
                self.assertEqual(scheduler.step_grad_enabled, [True, True, True, True])

    def test_drtune_stateful_forward_is_bitwise_equal_to_full(self):
        source, noise, prompt, mask = make_inputs()
        reference_unet = StepwiseDreamLiteUNet()
        reference = DifferentiableDreamLiteMobileSampler(
            unet=reference_unet,
            scheduler=MockFlowScheduler(),
        )(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
        ).latents
        for selected in ((0,), (3,), (0, 2), (1, 3)):
            unet = StepwiseDreamLiteUNet()
            unet.load_state_dict(reference_unet.state_dict())
            observed = DifferentiableDreamLiteMobileSampler(
                unet=unet,
                scheduler=MockFlowScheduler(),
            )(
                source_latents=source,
                noise_latents=noise,
                prompt_embeds=prompt,
                prompt_attention_mask=mask,
                gradient_mode="drtune_stateful",
                selected_step_indices=selected,
            ).latents
            self.assertTrue(torch.equal(reference, observed))

    def test_drtune_selected_step_checkpoint_keeps_parameter_gradients(self):
        source, noise, prompt, mask = make_inputs()
        unet = StepwiseDreamLiteUNet()
        sampler = DifferentiableDreamLiteMobileSampler(
            unet=unet,
            scheduler=MockFlowScheduler(),
            checkpoint_unet=True,
        )
        output = sampler(
            source_latents=source,
            noise_latents=noise,
            prompt_embeds=prompt,
            prompt_attention_mask=mask,
            gradient_mode="drtune",
            selected_step_indices=(0,),
        )
        output.latents.square().mean().backward()
        self.assertGreater(abs(unet.gains.grad[0].item()), 0.0)

    def test_drtune_policy_fails_closed(self):
        sampler = self.make_sampler()
        source, noise, prompt, mask = make_inputs()
        kwargs = {
            "source_latents": source,
            "noise_latents": noise,
            "prompt_embeds": prompt,
            "prompt_attention_mask": mask,
        }
        with self.assertRaisesRegex(ValueError, "requires selected_step_indices"):
            sampler(**kwargs, gradient_mode="drtune")
        with self.assertRaisesRegex(ValueError, "must be in"):
            sampler(**kwargs, gradient_mode="drtune", selected_step_indices=(4,))
        with self.assertRaisesRegex(ValueError, "valid only"):
            sampler(**kwargs, gradient_mode="full", selected_step_indices=(0,))


if __name__ == "__main__":
    unittest.main()
