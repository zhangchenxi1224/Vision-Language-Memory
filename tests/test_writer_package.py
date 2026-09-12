import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from PIL import Image

from scripts.inference.rgb_memory import run_commands, validate_command
from vision_memory.dreamlite.writer_package import export_completed_writer, file_sha, inspect_package, load_parameter_export, UPSTREAM


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def parent_fixture(root):
    module = torch.nn.Linear(2, 3)
    identity = {'model_variant': 'base', 'trainable_scope': 'full_unet', 'flow_protocol': 'official',
                'steps': 2, 'git_commit': 'training-commit', 'bank_manifest_sha256': 'bank-sha',
                'semantic_question_count': 1, 'conditional_group_count': 3,
                'teacher_data_that_must_not_be_exported': 'SECRET_TEACHER_ANSWER'}
    runtime = {'additional_protocol_binding': {'official_source_commit': UPSTREAM, 'inference_steps': 28,
                 'inference_guidance_scale': 1., 'base_snapshot': {'revision': 'base-revision'},
                 'source_bindings': {'SECRET_TEACHER_QUERY': 'bank-context'}},
               'snapshots': {'qwen_reader': {'repo_id': 'reader', 'revision': 'reader-revision',
                   'manifest_sha256': 'reader-manifest', 'snapshot_payload_sha256': 'reader-payload'}}}
    for name, value in (('identity.json', identity), ('runtime.json', runtime)):
        write_json(root / 'train' / name, value)
    checkpoint = root / 'train/checkpoint-final.pt'
    torch.save({'schema_version': 1, 'manifest': {**identity, **runtime}, 'optimizer_step': 2,
                'trainable_state': dict(module.named_parameters()), 'optimizer': {'SECRET_OPTIMIZER': 3},
                'rng_state': {'SECRET_RNG': 7}}, checkpoint)
    result = {'checkpoint_sha256': file_sha(checkpoint), 'optimizer_steps': 2,
              'unet_trainable_parameters': sum(p.numel() for p in module.parameters())}
    write_json(root / 'train/result.json', result)
    write_json(root / 'terminal.json', {'state': 'completed', 'training_result_sha256': file_sha(root / 'train/result.json')})
    return module


def test_export_loads_without_parent_bank_optimizer_or_query_data(tmp_path):
    parent, package = tmp_path / 'parent', tmp_path / 'package'
    source = parent_fixture(parent)
    manifest = export_completed_writer(parent, package)
    assert 'requires_independent_functional_validation' in manifest['status']
    assert 'SECRET' not in (package / 'manifest.json').read_text()
    state = torch.load(package / 'unet-parameters.pt', weights_only=True)
    assert set(state) == {'weight', 'bias'}
    # There is no bank file in this fixture, and inference never reads the parent.
    destination = torch.nn.Linear(2, 3).requires_grad_(False)
    load_parameter_export(destination, package, inspect_package(package))
    assert not destination.training and all(not p.requires_grad for p in destination.parameters())
    for left, right in zip(source.parameters(), destination.parameters(), strict=True):
        assert torch.equal(left, right)
    with (package / 'unet-parameters.pt').open('ab') as stream:
        stream.write(b'tamper')
    with pytest.raises(ValueError, match='changed'):
        inspect_package(package)


def test_wrong_final_parameter_rejects_entire_load_before_any_mutation(tmp_path):
    parent, package = tmp_path / 'parent', tmp_path / 'package'
    parent_fixture(parent)
    manifest = export_completed_writer(parent, package)
    state = torch.load(package / 'unet-parameters.pt', weights_only=True)
    state['bias'] = torch.ones(4)
    torch.save(state, package / 'unet-parameters.pt')
    manifest['weights_sha256'] = file_sha(package / 'unet-parameters.pt')
    destination = torch.nn.Linear(2, 3)
    before = copy.deepcopy(destination.state_dict())
    with pytest.raises(ValueError, match='mismatch'):
        load_parameter_export(destination, package, manifest)
    assert all(torch.equal(value, destination.state_dict()[key]) for key, value in before.items())


def test_command_api_rejects_gold_and_query_never_invokes_writer(tmp_path):
    with pytest.raises(ValueError):
        validate_command({'op': 'write', 'event': 'remember music', 'seed': 1, 'gold': 'ambient'})
    with pytest.raises(ValueError):
        validate_command({'op': 'write', 'event': 'remember music', 'seed': True})
    class Memory:
        def __init__(self):
            self.state = Image.new('RGB', (4, 4), (128, 128, 128))
            self.events = []
        @property
        def image(self):
            return self.state.copy()
        def write(self, event, seed):
            self.events.append((event, seed))
            self.state = Image.new('RGB', (4, 4), (1, 2, 3))
            return SimpleNamespace(image=self.image)
    memory = Memory()
    def read(image, query):
        assert image.getpixel((0, 0)) == (1, 2, 3)
        image.putpixel((0, 0), (200, 200, 200))
        return {'raw': 'reader output'}
    result = run_commands(memory, [{'op': 'write', 'event': 'event only', 'seed': 1},
                                  {'op': 'read', 'query': 'question one'}, {'op': 'read', 'query': 'question two'}], read, tmp_path / 'out')
    assert memory.events == [('event only', 1)] and result['writes'] == 1 and result['reads'] == 2
    assert memory.image.getpixel((0, 0)) == (1, 2, 3)
