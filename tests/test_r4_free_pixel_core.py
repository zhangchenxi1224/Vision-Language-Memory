from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite.recurrent import DreamLiteRecurrentUpdater  # noqa: E402
from vision_memory.event_noise import make_event_generator  # noqa: E402


class MockFlowScheduler:
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
        self.gain = nn.Parameter(torch.tensor(0.25))
        self.target_inputs = []

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
        self.target_inputs.append(target.detach().clone())
        prompt_term = encoder_hidden_states.mean().to(sample.dtype)
        velocity = self.gain * (target + 0.1 * source) + 0.01 * prompt_term
        return (torch.cat([velocity, torch.zeros_like(source)], dim=-1),)


class MockVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)
        self.encoded_inputs = []

    def encode(self, raw_image, return_dict=True):
        if not return_dict:
            raise AssertionError("R4 core expects a return_dict VAE encode.")
        self.encoded_inputs.append(raw_image.detach().clone())
        pooled = functional.avg_pool2d(raw_image, kernel_size=8)
        fourth_channel = pooled.mean(dim=1, keepdim=True)
        mean = torch.cat([pooled, fourth_channel], dim=1)
        return SimpleNamespace(latent_dist=SimpleNamespace(mean=mean))

    def decode(self, latents, return_dict=False):
        rgb = functional.interpolate(latents[:, :3], scale_factor=8, mode="nearest")
        if return_dict:
            return SimpleNamespace(sample=rgb)
        return (rgb,)


class MockImageProcessor:
    def postprocess(self, decoded, output_type):
        del decoded
        if output_type != "pil":
            raise AssertionError("Conditioning must request PIL output.")
        return [object()]


class MockPipeline:
    def __init__(self):
        self.unet = MockDreamLiteUNet()
        self.scheduler = MockFlowScheduler()
        self.vae = MockVAE()
        self.vae_scale_factor = 8
        self.image_processor = MockImageProcessor()

    def encode_prompt(self, *, mode, prompts, image, device, dtype):
        del prompts, image
        if mode != "edit":
            raise AssertionError("DreamLite updater must use edit conditioning.")
        return (
            torch.ones(1, 2, 3, device=device, dtype=dtype),
            torch.ones(1, 2, device=device, dtype=torch.long),
        )


class R4FreePixelCoreTest(unittest.TestCase):
    def make_updater(self):
        pipeline = MockPipeline()
        updater = DreamLiteRecurrentUpdater(
            pipeline=pipeline,
            global_seed=37,
            checkpoint_unet=False,
        )
        return pipeline, updater

    def test_float_rgb_state_is_decoded_clamped_and_reencoded_on_next_event(self):
        pipeline, updater = self.make_updater()
        rgb0 = torch.full((1, 3, 16, 16), 0.6)
        rgb1 = updater(
            rgb0,
            "set the memory",
            "episode-rgb",
            "turn-0",
            gradient_mode="drtune",
            selected_step_indices=(0,),
            persistent_state="float_rgb",
        )

        self.assertEqual(tuple(rgb1.shape), tuple(rgb0.shape))
        self.assertEqual(rgb1.shape[1], 3)
        self.assertTrue(rgb1.is_floating_point())
        self.assertGreaterEqual(rgb1.min().item(), 0.0)
        self.assertLessEqual(rgb1.max().item(), 1.0)
        self.assertEqual(len(pipeline.vae.encoded_inputs), 1)
        torch.testing.assert_close(pipeline.vae.encoded_inputs[0], rgb0 * 2.0 - 1.0)
        rgb1.square().mean().backward()
        self.assertIsNotNone(pipeline.unet.gain.grad)
        self.assertGreater(abs(pipeline.unet.gain.grad.item()), 0.0)

        rgb1_persisted = rgb1.detach()
        rgb2 = updater(
            rgb1_persisted,
            "overwrite the memory",
            "episode-rgb",
            "turn-1",
            gradient_mode="drtune",
            selected_step_indices=(1,),
            persistent_state="float_rgb",
            presentation_index=1,
        )
        self.assertEqual(tuple(rgb2.shape), tuple(rgb0.shape))
        self.assertEqual(len(pipeline.vae.encoded_inputs), 2)
        torch.testing.assert_close(
            pipeline.vae.encoded_inputs[1],
            rgb1_persisted.clamp(0.0, 1.0) * 2.0 - 1.0,
        )
        reader_rgb = updater.decode_for_reader(rgb2, persistent_state="float_rgb")
        self.assertIs(reader_rgb, rgb2)

    def test_float_rgb_mode_rejects_hidden_latent_carry(self):
        _, updater = self.make_updater()
        hidden_latent = torch.zeros(1, 4, 2, 2)
        with self.assertRaisesRegex(ValueError, "three channels"):
            updater(
                hidden_latent,
                "event",
                "episode-rgb",
                0,
                gradient_mode="drtune",
                selected_step_indices=(0,),
                persistent_state="float_rgb",
            )

    def test_persist_and_reload_helpers_enforce_the_float_rgb_bottleneck(self):
        pipeline, updater = self.make_updater()
        latent = torch.tensor(
            [[[[4.0]], [[-4.0]], [[0.0]], [[1.0]]]],
            dtype=torch.float32,
        )
        rgb = updater.persist_rgb_state(latent)
        self.assertEqual(tuple(rgb.shape), (1, 3, 8, 8))
        self.assertEqual(rgb.min().item(), 0.0)
        self.assertEqual(rgb.max().item(), 1.0)
        reloaded = updater.roundtrip_float_rgb(latent)
        self.assertEqual(tuple(reloaded.shape), (1, 4, 1, 1))
        self.assertEqual(len(pipeline.vae.encoded_inputs), 1)

    def test_default_presentation_zero_preserves_original_event_seed(self):
        pipeline, updater = self.make_updater()
        source = torch.zeros(1, 4, 2, 2)
        updater(source, "event", "episode-seed", "turn-seed")
        expected_generator = make_event_generator(
            device=source.device,
            global_seed=37,
            episode_id="episode-seed",
            turn_id="turn-seed",
        )
        expected_noise = torch.randn(
            source.shape,
            generator=expected_generator,
            device=source.device,
            dtype=source.dtype,
        )
        self.assertTrue(torch.equal(pipeline.unet.target_inputs[0], expected_noise))

        updater(source, "event", "episode-seed", "turn-seed", presentation_index=1)
        self.assertFalse(torch.equal(pipeline.unet.target_inputs[4], expected_noise))

    def test_r5_fixed_event_noise_ignores_presentation_index(self):
        pipeline, updater = self.make_updater()
        source = torch.zeros(1, 4, 2, 2)
        updater(
            source,
            "event",
            "episode-seed",
            "turn-seed",
            presentation_index=1,
            noise_include_presentation_index=False,
        )
        updater(
            source,
            "event",
            "episode-seed",
            "turn-seed",
            presentation_index=99,
            noise_include_presentation_index=False,
        )
        self.assertTrue(torch.equal(pipeline.unet.target_inputs[0], pipeline.unet.target_inputs[4]))

    def test_presentation_index_and_persistent_state_fail_closed(self):
        _, updater = self.make_updater()
        source = torch.zeros(1, 4, 2, 2)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            updater(source, "event", "episode", 0, presentation_index=True)
        with self.assertRaisesRegex(ValueError, "persistent_state"):
            updater(source, "event", "episode", 0, persistent_state="pixels")


if __name__ == "__main__":
    unittest.main()
