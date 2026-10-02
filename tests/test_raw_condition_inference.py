from types import SimpleNamespace
from PIL import Image
import pytest
import torch

from scripts.probes.official_base_guidance import TrainingConditionNativeSampler
from vision_memory.dreamlite.conditioning import encode_image_edit_condition
from vision_memory.dreamlite.native_base import NativeBaseEditSampler
from vision_memory.dreamlite.writer_package import (export_completed_writer, inspect_package,
    package_inference_condition, RAW_CONDITION_SCHEMA, SCHEMA, file_sha)
from test_writer_package import parent_fixture, write_json


class Pipeline:
    def __init__(self):
        self.calls = []
        self.prepare_latents = lambda: torch.zeros(1, 4, 2, 2)
        self.prepare_image_latents = lambda: torch.zeros(1, 4, 2, 2)
        self.scheduler = SimpleNamespace(sigmas=torch.tensor([1., .5, 0.]),
            step=lambda velocity, state: (state-.5*velocity,))
        self.fail = False

    def encode_prompt(self, **kwargs):
        self.calls.append(kwargs)
        assert len(kwargs['prompts']) == 1
        value = len(kwargs['prompts'][0])+kwargs['image'].getpixel((0, 0))[0]
        return torch.full((1, 3, 2), float(value)), torch.tensor([[1, 1, 0]])

    def __call__(self, **kwargs):
        state = self.prepare_latents()
        self.prepare_image_latents()
        embeddings, mask = self.encode_prompt(mode='edit', prompts=['', '', 'wrapped event'])
        assert torch.equal(embeddings[0], embeddings[2]) and torch.equal(mask[0], mask[2])
        if self.fail:
            raise RuntimeError('denoising failed')
        velocity = torch.ones_like(state)*embeddings[-1].mean()
        for _ in range(2):
            state = self.scheduler.step(velocity, state)[0]
        return SimpleNamespace(images=state)


def test_dynamic_raw_condition_matches_frozen_control_and_reencodes_current_event_image():
    pipe = Pipeline()
    noise = torch.randn(1, 4, 2, 2)
    source = torch.zeros_like(noise)
    original = (pipe.prepare_latents, pipe.prepare_image_latents, pipe.scheduler.step, pipe.encode_prompt)
    outputs = []
    for event, gray in [('set the value', 128), ('keep it unchanged', 64)]:
        image = Image.new('RGB', (1024, 1024), (gray,)*3)
        condition = encode_image_edit_condition(pipe, image, event, device='cpu', dtype=torch.float32)
        cached = TrainingConditionNativeSampler(NativeBaseEditSampler(pipe, source_image=image,
            event_text=event, guidance_scale=1., num_steps=2), condition)
        before = cached(source_latents=source, noise_latents=noise, num_steps=2)
        dynamic = NativeBaseEditSampler(pipe, source_image=image, event_text=event,
            guidance_scale=1., num_steps=2, inference_condition='training_raw')
        after = dynamic(source_latents=source, noise_latents=noise, num_steps=2)
        assert all(torch.equal(a, b) for a, b in zip(before.trajectory, after.trajectory, strict=True))
        assert original == (pipe.prepare_latents, pipe.prepare_image_latents, pipe.scheduler.step, pipe.encode_prompt)
        assert pipe.calls[-1]['image'] is image and pipe.calls[-1]['prompts'] == [event]
        outputs.append(after.latents)
    assert not torch.equal(*outputs)
    pipe.fail = True
    with pytest.raises(RuntimeError, match='denoising failed'):
        dynamic(source_latents=source, noise_latents=noise, num_steps=2)
    assert original == (pipe.prepare_latents, pipe.prepare_image_latents, pipe.scheduler.step, pipe.encode_prompt)


def test_raw_policy_rejects_cfg_extrapolation():
    with pytest.raises(ValueError, match='guidance_scale=1'):
        NativeBaseEditSampler(Pipeline(), source_image=object(), event_text='event', inference_condition='training_raw')


def test_raw_package_explicit_schema_cannot_silently_load_as_native(tmp_path):
    parent, package = tmp_path/'parent', tmp_path/'package'
    parent_fixture(parent, prompt_style='official_raw')
    manifest = export_completed_writer(parent, package, inference_condition='training_raw')
    assert manifest['schema'] == RAW_CONDITION_SCHEMA
    assert package_inference_condition(inspect_package(package)) == 'training_raw'
    assert 'SECRET' not in (package/'manifest.json').read_text()
    manifest['schema'] = SCHEMA
    write_json(package/'manifest.json', manifest)
    write_json(package/'complete.json', {'manifest_sha256': file_sha(package/'manifest.json')})
    with pytest.raises(ValueError, match='condition protocol'):
        inspect_package(package)


def test_raw_export_requires_matching_training_condition(tmp_path):
    parent = tmp_path/'parent'
    parent_fixture(parent, prompt_style='native_base')
    with pytest.raises(ValueError, match='raw-event training'):
        export_completed_writer(parent, tmp_path/'package', inference_condition='training_raw')
