import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from scripts.experiments import prefeval_aug4_writer as writer
from scripts.experiments import prefeval_k1_init_ablation as control


def rows(n=730):
    return [{'base_pair_id': f'topic:{i:04}', 'writer_user_forms': {f'W{k}': f'preference {i} wording {k}'
            for k in range(4)}, 'teacher_question_forms': {f'T{k}': 'SECRET QUESTION' for k in range(1, 4)},
            'options': ['SECRET ANSWER', 'b', 'c', 'd']} for i in range(n)]


def test_registered_schedule_exact_exposure_and_pairing():
    data = rows()
    a, b = writer.BalancedSchedule(data, 101), writer.BalancedSchedule(data, 101)
    for cycle in range(8):
        observed = [a.draw(cycle * 730 + i) for i in range(730)]
        assert len({row['base_pair_id'] for row, _ in observed}) == 730
        assert {v for _, v in observed} == {f'W{cycle % 4}'}
    for step, exposures in zip(writer.SNAPSHOTS, (32, 64, 128, 256, 512)):
        counts = a.exposure(step * 4)
        assert len(counts) == 730
        assert all(set(c.values()) == {exposures // 4} for c in counts.values())
    for index in (0, 729, 730, 731, 2920, 15055):
        assert a.draw(index) == b.draw(index)
    partial = a.exposure(733)
    assert sum(sum(c.values()) for c in partial.values()) == 733
    assert sum(c['W1'] for c in partial.values()) == 3


def test_writer_input_uses_only_requested_user_wording_and_neutral_ack():
    row = rows(1)[0]
    for variant in writer.VARIANTS:
        text = writer.writer_text(row, variant)
        assert text == f'user: {row["writer_user_forms"][variant]}\nassistant: Understood.'
        assert 'SECRET' not in text


class TinyUnet(torch.nn.Module):
    def __init__(self, width=2):
        super().__init__()
        self.config = {'width': width}
        self.layer = torch.nn.Linear(width, width)
        self.register_buffer('scale', torch.ones(1))

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def pipe():
    p = SimpleNamespace(unet=TinyUnet(), vae=torch.nn.Linear(2, 2), text_encoder=torch.nn.Linear(2, 2))
    for m in (p.unet, p.vae, p.text_encoder):
        m.eval().requires_grad_(False)
    return p


def test_official_init_leaves_official_weights_intact_random_pair_matches():
    original = pipe()
    before = control.model_hash(original.unet)
    args = SimpleNamespace(init='pretrained', mode='train', init_seed=10, device='cpu')
    writer.initialize_unet(original, args)
    assert control.model_hash(original.unet) == before
    assert all(p.requires_grad for p in original.unet.parameters())
    first, second = pipe(), pipe()
    args.init = 'random'
    rng = torch.get_rng_state().clone()
    writer.initialize_unet(first, args)
    writer.initialize_unet(second, args)
    assert torch.equal(rng, torch.get_rng_state())
    assert control.model_hash(first.unet) == control.model_hash(second.unet)
    assert control.model_hash(first.unet) != before
    assert control.tensor_schema(first.unet) == control.tensor_schema(original.unet)


def test_official_file_hash_is_a_hard_gate(tmp_path):
    root = tmp_path / 'unet'
    root.mkdir()
    (root / 'diffusion_pytorch_model.safetensors').write_bytes(b'wrong task weights')
    (root / 'config.json').write_text('{}')
    with pytest.raises(RuntimeError, match='SHA256 mismatch'):
        writer.official_files(SimpleNamespace(base=tmp_path))


def loop_args(output, **changes):
    values = dict(output=output, steps=12, snapshot_steps=[4, 8, 12], learning_rate=5e-4,
                  resume=False, train_seed=42, device='cpu', mode='train', save_interval=2,
                  audit_interval=2, fm_probe_interval=0, fm_probe_count=3)
    values.update(changes)
    return SimpleNamespace(**values)


def fixture(p):
    data = rows(5)
    targets = {row['base_pair_id']: torch.tensor([[.1 + i, .2 - i]]) for i, row in enumerate(data)}
    cache = {(row['base_pair_id'], v): {'source': torch.zeros(1, 2), 'embeds': torch.tensor([[j / 10., .3]]),
             'mask': torch.ones(1, 2)} for row in data for j, v in enumerate(writer.VARIANTS)}
    p.unet.requires_grad_(True)
    return data, targets, cache


def test_real_loop_crash_resume_matches_and_published_snapshots_are_immutable(tmp_path, monkeypatch):
    base = pipe()
    p_full, p_resume = copy.deepcopy(base), copy.deepcopy(base)
    data, targets, cache = fixture(p_full)
    fixture(p_resume)
    full, recovering = tmp_path / 'full', tmp_path / 'recovering'
    full.mkdir()
    recovering.mkdir()
    calls = [0]

    def prediction(predictor, state, source, sigma, embeds, mask, integer_timestep):
        return predictor.layer(state + embeds)

    monkeypatch.setattr(writer, 'predict_velocity', prediction)
    manifest = {'fixture': 'real FM loop'}
    writer.train_loop(loop_args(full), p_full, targets, cache, data, manifest, p_full.unet)

    def interrupted(*a, **kw):
        calls[0] += 1
        if calls[0] == 29:
            raise RuntimeError('simulated interruption after unsaved step 7')
        return prediction(*a, **kw)

    monkeypatch.setattr(writer, 'predict_velocity', interrupted)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        writer.train_loop(loop_args(recovering), p_resume, targets, cache, data, manifest, p_resume.unet)
    assert torch.load(recovering / 'resume.pt', weights_only=False)['optimizer_step'] == 6
    published = recovering / 'checkpoint-step-000004.pt'
    published_sha = writer.sha(published)
    monkeypatch.setattr(writer, 'predict_velocity', prediction)
    writer.train_loop(loop_args(recovering, resume=True), p_resume, targets, cache, data, manifest, p_resume.unet)
    assert control.model_hash(p_full.unet) == control.model_hash(p_resume.unet)
    assert writer.sha(published) == published_sha
    logs = [json.loads(line) for line in (recovering / 'optimization.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs] == list(range(1, 13))
    assert len(list(recovering.glob('optimization-uncommitted-*'))) == 1
    full_state = torch.load(full / 'resume.pt', weights_only=False)
    resumed_state = torch.load(recovering / 'resume.pt', weights_only=False)
    assert full_state['optimizer_step'] == resumed_state['optimizer_step'] == 12
    for key, values in full_state['optimizer']['state'].items():
        for field, value in values.items():
            assert torch.equal(value, resumed_state['optimizer']['state'][key][field])
    counts = json.loads((recovering / 'checkpoint-step-000012.exposures.json').read_text())['counts']
    expected = Counter((d['pair_id'], d['variant']) for r in logs for d in r['draws'])
    assert all(counts[pid][variant] == expected[pid, variant] for pid in counts for variant in writer.VARIANTS)
    # Replacing resume must not mutate a hardlinked, published full-state snapshot.
    old_full = torch.load(recovering / 'checkpoint-step-000004.full.pt', weights_only=False)
    assert old_full['optimizer_step'] == 4
    writer.save_snapshot(recovering, p_resume.unet, 4, manifest, writer.BalancedSchedule(data, 42))
    with published.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(RuntimeError, match='immutable'):
        writer.save_snapshot(recovering, p_resume.unet, 4, manifest, writer.BalancedSchedule(data, 42))


def test_teacher_binding_rejects_legacy_bank(tmp_path):
    dataset = tmp_path / 'data.json'
    dataset.write_text('{}')
    root = tmp_path / 'teachers/topic_0000'
    root.mkdir(parents=True)
    writer.save_json(root / 'complete.json', {'step': 288, 'binding': {'data_sha256': 'old dataset'}})
    with pytest.raises(RuntimeError, match='registered'):
        writer.load_targets(SimpleNamespace(data=dataset, teacher_root=tmp_path / 'teachers'), rows(1))
