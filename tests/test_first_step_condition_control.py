from types import SimpleNamespace
import pytest
import torch

from scripts.probes.logical_first_step_condition import first_native_step
from vision_memory.dreamlite.native_base import NativeBaseEditSampler


class FirstStepPipeline:
    def __init__(self):
        self.prepare_latents=lambda:torch.zeros(1,4,2,2)
        self.prepare_image_latents=lambda:torch.zeros(1,4,2,2)
        self.encode_prompt=lambda **kwargs:(torch.ones(3,2,1),torch.ones(3,2))
        self.scheduler=SimpleNamespace(step=lambda velocity,time,state,**kwargs:(state-.1*velocity,))
        self.timestep=1000.
    def __call__(self,**kwargs):
        noise=self.prepare_latents()
        self.prepare_image_latents()
        condition,_=self.encode_prompt(mode='edit',prompts=['','','native wrapped event'])
        velocity=torch.ones_like(noise)*condition[-1].mean()
        self.scheduler.step(velocity,torch.tensor(self.timestep),noise,return_dict=False)
        raise AssertionError('Probe must stop after the unchanged first Euler update')


def test_first_step_capture_replays_actual_update_and_restores_every_hook():
    pipe=FirstStepPipeline()
    original=(pipe.prepare_latents,pipe.prepare_image_latents,pipe.encode_prompt,pipe.scheduler.step)
    native=NativeBaseEditSampler(pipe,source_image=object(),event_text='event',guidance_scale=1.)
    noise=torch.randn(1,4,2,2)
    source=torch.zeros_like(noise)
    result=first_native_step(native,source=source,noise=noise)
    assert torch.equal(result['state'],noise)
    assert torch.equal(result['next_state'],noise-.1)
    assert original==(pipe.prepare_latents,pipe.prepare_image_latents,pipe.encode_prompt,pipe.scheduler.step)
    condition=SimpleNamespace(prompt_embeds=torch.full((1,2,1),2.),attention_mask=torch.ones(1,2))
    raw=first_native_step(native,source=source,noise=noise,condition=condition)
    assert torch.equal(raw['velocity'],torch.full_like(noise,2.))
    assert torch.equal(raw['next_state'],noise-.2)
    assert original==(pipe.prepare_latents,pipe.prepare_image_latents,pipe.encode_prompt,pipe.scheduler.step)
    pipe.timestep=999.
    with pytest.raises(ValueError,match='sigma1'):
        first_native_step(native,source=source,noise=noise,condition=condition)
    assert original==(pipe.prepare_latents,pipe.prepare_image_latents,pipe.encode_prompt,pipe.scheduler.step)
