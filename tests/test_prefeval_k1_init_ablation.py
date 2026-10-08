import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.experiments import prefeval_k1_init_ablation as controls
from vision_memory.training.latent_bank_unet import stable_seed


class TinyUNet(torch.nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.config = {'width': width}
        self.linear = torch.nn.Linear(width, width)
        self.register_buffer('fixed_scale', torch.ones(1))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def args(**updates):
    values = dict(unet_init='parent', learning_rate=5e-5, train_seed=20260924,
                  init_seed=20261009, conditioning_seed=0, fresh_start=False,
                  audit_updates=True, fm_probe_interval=0, fm_probe_count=32,
                  fm_probe_seed=20261009, mode='train', stage='write', steps=128,
                  checkpoint='parent.pt', device='cpu', output=None, expected_init_audit=None)
    values.update(updates)
    return SimpleNamespace(**values)


def pipe():
    result = SimpleNamespace(unet=TinyUNet(), vae=torch.nn.Linear(4, 2), text_encoder=torch.nn.Linear(4, 2))
    for module in (result.unet, result.vae, result.text_encoder):
        module.eval().requires_grad_(False)
    return result


def test_random_standard_constructor_and_rng_are_identical_across_learning_rates():
    first, second = pipe(), pipe()
    original_schema = controls.tensor_schema(first.unet)
    frozen_hashes = {name: controls.frozen_hash(getattr(first, name)) for name in ('vae', 'text_encoder')}
    before_rng = torch.get_rng_state().clone()
    before_py, before_np = random.getstate(), np.random.get_state()

    def forbidden(*a, **kw):
        raise AssertionError('Random branch loaded parent weights')

    controls.initialize_unet(first, args(unet_init='random', learning_rate=5e-5), forbidden)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert random.getstate() == before_py
    assert np.array_equal(np.random.get_state()[1], before_np[1])
    controls.initialize_unet(second, args(unet_init='random', learning_rate=5e-4), forbidden)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert random.getstate() == before_py
    assert np.array_equal(np.random.get_state()[1], before_np[1])
    with controls.isolated_rng(20261009):
        expected = TinyUNet.from_config({'width': 4})
    assert controls.model_hash(first.unet) == controls.model_hash(second.unet) == controls.model_hash(expected)
    assert controls.tensor_schema(first.unet) == original_schema
    assert not first.unet.training and all(p.requires_grad for p in first.unet.parameters())
    controls.verify_frozen(first, frozen_hashes)


def test_parent_loads_exact_weights_and_parent_random_architectures_match():
    original = TinyUNet()
    def loader(path, *, trainable_module):
        assert path == 'parent.pt'
        trainable_module.load_state_dict(original.state_dict())
    parent, random_pipe = pipe(), pipe()
    controls.initialize_unet(parent, args(), loader)
    controls.initialize_unet(random_pipe, args(unet_init='random'), loader)
    assert controls.model_hash(parent.unet) == controls.model_hash(original)
    assert controls.model_hash(random_pipe.unet) != controls.model_hash(original)
    assert controls.initialization_manifest(parent, args())['architecture_sha256'] == controls.initialization_manifest(random_pipe, args())['architecture_sha256']


def test_initialization_does_not_change_condition_or_training_random_streams():
    def condition():
        return (torch.randn(5), random.random(), np.random.rand())
    before = controls.seeded_factory(condition, 55)
    random_pipe = pipe()
    controls.initialize_unet(random_pipe, args(unet_init='random', init_seed=7), lambda *a, **k: None)
    after = controls.seeded_factory(condition, 55)
    assert torch.equal(before[0], after[0]) and before[1:] == after[1:]
    for draw in range(12):
        seed = stable_seed(20260924, 'noise', draw)
        one = torch.randn(4, generator=torch.Generator().manual_seed(seed))
        with controls.isolated_rng(999):
            torch.randn(100)
        two = torch.randn(4, generator=torch.Generator().manual_seed(seed))
        assert torch.equal(one, two)


def test_fresh_start_rejects_resume_and_any_existing_output(tmp_path):
    controls.ensure_fresh_output(tmp_path)
    (tmp_path/'resume.pt').write_text('old')
    with pytest.raises(ValueError, match='fresh-start'):
        controls.ensure_fresh_output(tmp_path)


def test_audit_measures_real_adam_update_and_detects_nonfinite_state(tmp_path):
    model = TinyUNet().requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    assert not optimizer.state
    audit = controls.UpdateAudit(model)
    model.linear(torch.ones(2, 4)).square().mean().backward()
    controls.clip_gradients(model.parameters(), 1)
    before = audit.capture()
    optimizer.step()
    measured = audit.after_step(before, optimizer, 1)
    assert measured['sampled_update_l2'] > 0
    assert measured['sampled_changed_coordinates'] > 0
    assert measured['sampled_parameter_count'] == len(list(model.parameters()))
    first = next(model.parameters())
    optimizer.state[first]['exp_avg'].flatten()[0] = float('nan')
    with pytest.raises(controls.NumericFailure) as result:
        audit.after_step(audit.capture(), optimizer, 2)
    controls.write_numeric_failure(tmp_path, result.value)
    record = json.loads((tmp_path/'numeric-failure.json').read_text())
    assert record['reason'] == 'nonfinite_optimizer_state' and record['step'] == 2


def test_nonfinite_gradient_norm_has_specific_failure_but_other_errors_do_not(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.tensor([float('inf'), 0.])
    with pytest.raises(controls.NumericFailure, match='nonfinite_gradient_norm'):
        controls.clip_gradients([parameter], 1)
    def oom(*a, **k):
        raise RuntimeError('CUDA out of memory')
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', oom)
    with pytest.raises(RuntimeError, match='out of memory') as result:
        controls.clip_gradients([parameter], 1)
    assert not isinstance(result.value, controls.NumericFailure)


def test_shared_identity_checks_actual_condition_contents_but_excludes_arm_and_budget():
    rows = [{'base_pair_id': 'p:0', 'history': [{'role': 'user', 'content': 'sample'}]}]
    cache = {('p:0', 0): {'source': torch.zeros(1), 'embeds': torch.ones(2), 'mask': torch.ones(2)}}
    hashes = controls.conditions_manifest(cache)
    left = controls.shared_identity(rows, {'p:0': 'target'}, hashes, args(), 'variants')
    right = controls.shared_identity(rows, {'p:0': 'target'}, hashes,
                                    args(unet_init='random', learning_rate=5e-4, steps=23360), 'variants')
    assert left == right
    cache['p:0', 0]['source'].add_(1)
    changed = controls.shared_identity(rows, {'p:0': 'target'}, controls.conditions_manifest(cache), args(), 'variants')
    assert changed['conditions_sha256'] != left['conditions_sha256']
    assert controls.shared_identity(rows, {'p:0': 'other'}, hashes, args(), 'variants')['targets_sha256'] != left['targets_sha256']


def test_sparse_probe_tail_recovery_and_probe_does_not_perturb_rng(tmp_path):
    path = tmp_path/'fm-probe.jsonl'
    path.write_text(''.join(json.dumps({'step': step})+'\n' for step in [0, 128, 256]))
    controls.archive_probe_tail(path, 128)
    assert [json.loads(x)['step'] for x in path.read_text().splitlines()] == [0, 128]
    controls.archive_probe_tail(path, 0)
    assert path.read_text() == ''
    before = torch.get_rng_state().clone()
    with controls.isolated_rng(777):
        torch.randn(20)
    assert torch.equal(before, torch.get_rng_state())


def test_frozen_weights_change_is_detected():
    model = pipe()
    expected = {name: controls.frozen_hash(getattr(model, name)) for name in ('vae', 'text_encoder')}
    with torch.no_grad():
        next(model.vae.parameters()).add_(1)
    with pytest.raises(RuntimeError, match='Frozen'):
        controls.verify_frozen(model, expected)


def test_frozen_hash_excludes_only_nonpersistent_caches():
    model = pipe()
    model.text_encoder.register_buffer('transient_cache', torch.zeros(2), persistent=False)
    model.text_encoder.register_buffer('persistent_scale', torch.ones(1), persistent=True)
    expected = {name: controls.frozen_hash(getattr(model, name)) for name in ('vae', 'text_encoder')}
    full_before = controls.model_hash(model.text_encoder)
    model.text_encoder.transient_cache.add_(1)
    controls.verify_frozen(model, expected)
    assert controls.model_hash(model.text_encoder) != full_before
    model.text_encoder.persistent_scale.add_(1)
    with pytest.raises(RuntimeError, match='Frozen'):
        controls.verify_frozen(model, expected)


def test_formal_initialization_gate_ignores_budget_but_rejects_changed_inputs(tmp_path):
    reference = {'initial_state_sha256': 'initial', 'architecture_sha256': 'architecture',
                 'frozen_module_hashes': {'vae': 'v', 'text_encoder': 't'}, 'runtime': {'torch': 'pinned'},
                 'shared_training_identity': {'conditions_sha256': 'conditions', 'targets_sha256': 'targets'},
                 'unet_init': 'parent', 'learning_rate': 5e-4, 'train_seed': 20260924,
                 'init_seed': 20261009, 'conditioning_seed': 0, 'steps': 128, 'snapshot_steps': []}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(reference))
    formal = {**reference, 'steps': 23360, 'snapshot_steps': [2048]}
    first = controls.expected_init_audit(formal, path)
    assert len(first) == 64
    for field in ('initial_state_sha256', 'learning_rate', 'conditioning_seed', 'shared_training_identity'):
        changed = {**formal, field: 'wrong'}
        with pytest.raises(RuntimeError, match='audit mismatch'):
            controls.expected_init_audit(changed, path)
    incomplete = dict(reference)
    del incomplete['runtime']
    path.write_text(json.dumps(incomplete))
    with pytest.raises(RuntimeError, match='missing'):
        controls.expected_init_audit(formal, path)


def test_real_writer_loop_probes_do_not_change_updates_and_fresh_optimizer(tmp_path):
    """Execute the real train function with small CPU tensors, without GPU model imports."""
    import ast
    import copy
    import hashlib
    from pathlib import Path
    import time
    from scripts.experiments.prefeval_k1_variants import training_variant, select_fm_target
    from scripts.experiments.prefeval_k1_refresh_bank import archive_uncommitted_tail
    from vision_memory.training.latent_bank_unet import official_flow_bridge, OFFICIAL_REFERENCE_COMMIT
    from vision_memory.training.checkpoint import save_training_checkpoint, load_training_checkpoint
    from vision_memory.repro import canonical_tensor_sha256

    root = Path(__file__).resolve().parents[1]
    writer_path = root/'scripts/experiments/prefeval_k1_writer.py'
    tree = ast.parse(writer_path.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {'train', 'save_inference_checkpoint'}]

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def save_json(path, value):
        Path(path).write_text(json.dumps(value))

    def append(path, value):
        with Path(path).open('a') as stream:
            stream.write(json.dumps(value)+'\n')

    def condition(*arguments):
        return {'source': torch.randn(1, 4, 4, 4), 'embeds': torch.randn(1, 2, 4),
                'mask': torch.ones(1, 2)}

    env = dict(torch=torch, json=json, Path=Path, time=time, ROOT=root, __file__=str(writer_path),
               init_control=controls, sha=sha, save_json=save_json, append=append,
               atomic_save=lambda path, value: torch.save(value, path),
               cache_condition=condition, event_text=lambda exchange: json.dumps(exchange),
               training_variant=training_variant, select_fm_target=select_fm_target,
               stable_seed=stable_seed, official_flow_bridge=official_flow_bridge,
               OFFICIAL_REFERENCE_COMMIT=OFFICIAL_REFERENCE_COMMIT,
               canonical_tensor_sha256=canonical_tensor_sha256,
               save_training_checkpoint=save_training_checkpoint, load_training_checkpoint=load_training_checkpoint,
               archive_uncommitted_tail=archive_uncommitted_tail,
               DifferentiableDreamLiteMobileSampler=SimpleNamespace(from_pipeline=lambda p, **kw: p),
               predict_velocity=lambda sampler, state, *a, **kw: sampler.unet.linear(state))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(writer_path), 'exec'), env)
    teachers = tmp_path/'teachers'
    rows = []
    for index in range(2):
        pid = f'p:{index}'
        directory = teachers/pid.replace(':', '_')
        directory.mkdir(parents=True)
        torch.save(torch.full((1, 4, 4, 4), float(index)), directory/'latent.pt')
        save_json(directory/'complete.json', {'step': 288, 'binding': {'arm': 'B'},
                                             'latent_sha256': sha(directory/'latent.pt')})
        rows.append({'base_pair_id': pid, 'history': [{'role': 'user', 'content': str(index)},
                                                      {'role': 'assistant', 'content': 'noted'}]})
    parent_file = tmp_path/'parent.pt'
    parent_file.write_bytes(b'parent-reference')
    baseline = pipe()
    baseline.unet.requires_grad_(True)
    trained, probed = copy.deepcopy(baseline), copy.deepcopy(baseline)
    results = []
    for label, model, interval in [('plain', trained, 0), ('probed', probed, 1)]:
        configuration = args(output=tmp_path/label, checkpoint=parent_file, split='train',
                             arm='B', stage='write', sources=None, target_bank=None,
                             initial_variants=None, retain_source_mode='recursive',
                             retain_target_mode='teacher', snapshot_steps=[2], steps=4,
                             refresh_round=0, teachers=teachers, fresh_start=True,
                             fm_probe_interval=interval, fm_probe_count=2)
        configuration.initialization_manifest = controls.initialization_manifest(model, configuration)
        env['train'](configuration, model, rows)
        complete = json.loads((configuration.output/'complete.json').read_text())
        manifest = json.loads((configuration.output/'manifest.json').read_text())
        initial = json.loads((configuration.output/'initial-state.json').read_text())
        resume = torch.load(configuration.output/'resume.pt', weights_only=False)
        assert initial['fresh_optimizer'] and initial['optimizer_state_entries'] == 0
        assert complete['frozen_verified'] and complete['steps'] == 4
        assert resume['optimizer_step'] == 4 and resume['episode_cursor'] == 16
        assert all(float(state['step']) == 4 for state in resume['optimizer']['state'].values())
        assert (configuration.output/'checkpoint-step-000002.pt').exists()
        results.append((complete, manifest))
    assert controls.model_hash(trained.unet) == controls.model_hash(probed.unet)
    assert results[0][1]['shared_training_identity'] == results[1][1]['shared_training_identity']
    assert results[0][0]['initial_state_sha256'] == results[1][0]['initial_state_sha256']
    probes = [json.loads(line) for line in (tmp_path/'probed/fm-probe.jsonl').read_text().splitlines()]
    assert [record['step'] for record in probes] == [0, 1, 2, 3, 4]

    # Exercise the real 128-step recovery boundary, including a legacy missing snapshot.
    expected_end = copy.deepcopy(baseline)
    recovery_args = args(output=tmp_path/'reference130', checkpoint=parent_file, split='train',
                         arm='B', stage='write', sources=None, target_bank=None,
                         initial_variants=None, retain_source_mode='recursive',
                         retain_target_mode='teacher', snapshot_steps=[128], steps=130,
                         refresh_round=0, teachers=teachers, fresh_start=True,
                         fm_probe_interval=0, fm_probe_count=2)
    recovery_args.initialization_manifest = controls.initialization_manifest(expected_end, recovery_args)
    env['train'](recovery_args, expected_end, rows)
    reference_snapshot = torch.load(recovery_args.output/'checkpoint-step-000128.pt', weights_only=False)

    interrupted = copy.deepcopy(baseline)
    recovery_args.output = tmp_path/'interrupted130'
    calls = []

    class SimulatedInterruption(Exception):
        pass

    def interrupt_after_resume_commit(path, **kwargs):
        snapshot = Path(path).parent/'checkpoint-step-000128.pt'
        if kwargs['optimizer_step'] == 128:
            # Proves the new ordering closes the original missing-snapshot window.
            assert snapshot.exists()
            calls.append('snapshot_committed_before_resume')
        result = save_training_checkpoint(path, **kwargs)
        if kwargs['optimizer_step'] == 128:
            # Also simulate a snapshot absent from a previously committed recovery point.
            snapshot.unlink()
            raise SimulatedInterruption('resume committed; legacy snapshot missing')
        return result

    env['save_training_checkpoint'] = interrupt_after_resume_commit
    with pytest.raises(SimulatedInterruption):
        env['train'](recovery_args, interrupted, rows)
    assert calls == ['snapshot_committed_before_resume']
    durable = torch.load(recovery_args.output/'resume.pt', weights_only=False)
    assert durable['optimizer_step'] == 128
    assert not (recovery_args.output/'checkpoint-step-000128.pt').exists()

    env['save_training_checkpoint'] = save_training_checkpoint
    recovery_args.fresh_start = False
    recovered = copy.deepcopy(baseline)
    env['train'](recovery_args, recovered, rows)
    repaired_snapshot = torch.load(recovery_args.output/'checkpoint-step-000128.pt', weights_only=False)
    assert repaired_snapshot['optimizer_step'] == 128
    for name, value in repaired_snapshot['trainable_state'].items():
        assert torch.equal(value, durable['trainable_state'][name])
        assert torch.equal(value, reference_snapshot['trainable_state'][name])
    assert controls.model_hash(recovered.unet) == controls.model_hash(expected_end.unet)
    records = [json.loads(line) for line in (recovery_args.output/'optimization.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records] == list(range(1, 131))
    final_resume = torch.load(recovery_args.output/'resume.pt', weights_only=False)
    assert final_resume['optimizer_step'] == 130 and final_resume['episode_cursor'] == 520
    assert all(float(state['step']) == 130 for state in final_resume['optimizer']['state'].values())
