import copy
import json
from types import SimpleNamespace
import pytest
import torch

from vision_memory.dreamlite.writer_package import SCHEMA, UPSTREAM, file_sha
from vision_memory.training.writer_initialization import initial_writer_binding, apply_initial_writer


def setup_initialization(tmp_path):
    package = tmp_path / 'package'
    package.mkdir()
    source = torch.nn.Linear(2, 3)
    torch.save(source.state_dict(), package / 'unet-parameters.pt')
    base = {'model_dir': '/original/base', 'revision': 'base-revision'}
    (package / 'base-seal.json').write_text(json.dumps(base))
    reader = {'repo_id': 'reader', 'revision': 'reader-revision',
              'manifest_sha256': 'reader-manifest', 'snapshot_payload_sha256': 'reader-payload'}
    manifest = {'schema': SCHEMA, 'status': 'experimental_endpoint_requires_independent_functional_validation',
        'weights': 'unet-parameters.pt', 'weights_sha256': file_sha(package / 'unet-parameters.pt'),
        'base_seal_sha256': file_sha(package / 'base-seal.json'), 'official_source_commit': UPSTREAM,
        'native_steps': 28, 'guidance_scale': 1., 'image_guidance_scale': 1., 'reader_snapshot': reader,
        'parameter_count': sum(p.numel() for p in source.parameters()), 'parent_commit': 'parent-commit',
        'parent_result_sha256': 'parent-result', 'parent_checkpoint_sha256': 'parent-checkpoint', 'optimizer_steps': 2880}
    (package / 'manifest.json').write_text(json.dumps(manifest))
    digest = file_sha(package / 'manifest.json')
    (package / 'complete.json').write_text(json.dumps({'manifest_sha256': digest}))
    args = SimpleNamespace(initial_writer_package=package, initial_writer_package_sha256=digest,
                           model_variant='base', trainable_scope='full_unet', flow_protocol='official', baseline_reference=None)
    destination = torch.nn.Linear(2, 3).eval()
    runtime = {'pipe': SimpleNamespace(unet=destination), 'snapshots': {'qwen_reader': reader},
               'protocol_binding': {'base_snapshot': {**base, 'model_dir': '/relocated/base'},
                   'official_source_commit': UPSTREAM, 'inference_steps': 28, 'inference_guidance_scale': 1.}}
    return source, args, runtime


def test_warm_start_preserves_rng_loads_all_parameters_and_supports_fresh_training(tmp_path):
    source, args, runtime = setup_initialization(tmp_path)
    binding = initial_writer_binding(args)
    rng = torch.get_rng_state().clone()
    apply_initial_writer(args, runtime, binding)
    target = runtime['pipe'].unet
    assert torch.equal(rng, torch.get_rng_state())
    assert not target.training and all(p.requires_grad for p in target.parameters())
    assert all(torch.equal(a, b) for a, b in zip(source.parameters(), target.parameters(), strict=True))
    assert binding['parent_optimizer_steps'] == 2880 and 'fresh AdamW' in binding['optimizer_and_rng']
    optimizer = torch.optim.AdamW(target.parameters(), lr=5e-5)
    assert not optimizer.state
    before = copy.deepcopy(target.state_dict())
    target(torch.ones(1, 2)).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in target.parameters())
    optimizer.step()
    assert all(float(state['step']) == 1 for state in optimizer.state.values())
    assert any(not torch.equal(value, target.state_dict()[key]) for key, value in before.items())


@pytest.mark.parametrize('mismatch', ['base', 'reader', 'guidance'])
def test_runtime_mismatch_rejected_before_any_parameter_changes(tmp_path, mismatch):
    _, args, runtime = setup_initialization(tmp_path)
    binding = initial_writer_binding(args)
    before = copy.deepcopy(runtime['pipe'].unet.state_dict())
    if mismatch == 'base':
        runtime['protocol_binding']['base_snapshot']['revision'] = 'different'
    elif mismatch == 'reader':
        runtime['snapshots']['qwen_reader']['revision'] = 'different'
    else:
        runtime['protocol_binding']['inference_guidance_scale'] = 2.
    with pytest.raises(ValueError, match='differs'):
        apply_initial_writer(args, runtime, binding)
    assert all(torch.equal(value, runtime['pipe'].unet.state_dict()[key]) for key, value in before.items())


def test_initialization_is_explicit_and_tampered_package_is_rejected(tmp_path):
    _, args, runtime = setup_initialization(tmp_path)
    binding = initial_writer_binding(args)
    assert initial_writer_binding(SimpleNamespace()) is None
    without_digest = copy.copy(args)
    without_digest.initial_writer_package_sha256 = None
    with pytest.raises(ValueError, match='SHA256'):
        initial_writer_binding(without_digest)
    reused_baseline = copy.copy(args)
    reused_baseline.baseline_reference = tmp_path / 'old-untrained-baseline'
    with pytest.raises(ValueError, match='new measured baseline'):
        initial_writer_binding(reused_baseline)
    before = copy.deepcopy(runtime['pipe'].unet.state_dict())
    with (args.initial_writer_package / 'unet-parameters.pt').open('ab') as stream:
        stream.write(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        apply_initial_writer(args, runtime, binding)
    assert all(torch.equal(value, runtime['pipe'].unet.state_dict()[key]) for key, value in before.items())
