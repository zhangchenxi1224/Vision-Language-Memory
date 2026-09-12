from types import SimpleNamespace

import pytest
import torch

from scripts.probes.official_base_guidance import TrainingConditionNativeSampler


def test_cached_training_condition_reaches_native_call_and_hook_restores_on_error():
    condition = SimpleNamespace(prompt_embeds=torch.randn(1, 7, 4), attention_mask=torch.ones(1, 7))
    original = lambda **kwargs: (_ for _ in ()).throw(AssertionError("must use cached tensors"))
    pipe = SimpleNamespace(encode_prompt=original)

    class Native:
        guidance_scale = 1.
        pipeline = pipe
        fail = False
        def __call__(self, **kwargs):
            embeds, mask = pipe.encode_prompt(mode="edit", prompts=["", "", "wrapped event"])
            for row in range(3):
                assert torch.equal(embeds[row], condition.prompt_embeds[0])
                assert torch.equal(mask[row], condition.attention_mask[0])
            if self.fail:
                raise RuntimeError("denoising failed")
            return kwargs["noise_latents"]

    native = Native()
    sampler = TrainingConditionNativeSampler(native, condition)
    noise = torch.randn(1, 4, 2, 2)
    assert sampler(noise_latents=noise) is noise
    assert pipe.encode_prompt is original
    native.fail = True
    with pytest.raises(RuntimeError, match="denoising failed"):
        sampler(noise_latents=noise)
    assert pipe.encode_prompt is original


def test_cached_condition_rejects_cfg_extrapolation_or_unconsumed_override():
    condition = SimpleNamespace(prompt_embeds=torch.randn(1, 7, 4), attention_mask=torch.ones(1, 7))
    with pytest.raises(ValueError, match="guidance_scale=1"):
        TrainingConditionNativeSampler(SimpleNamespace(guidance_scale=7.5), condition)

    class Native:
        guidance_scale = 1.
        pipeline = SimpleNamespace(encode_prompt=lambda: None)
        def __call__(self, **kwargs):
            return None

    sampler = TrainingConditionNativeSampler(Native(), condition)
    with pytest.raises(RuntimeError, match="exactly once"):
        sampler()
