import collections
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.experiments.prefeval_k1_source_bank import (
    SOURCE_ORDER, freeze_bank, load_bank, noise_namespace, retention_draw,
)
from scripts.experiments.prefeval_k1_condition_store import DiskConditions
from scripts.experiments.prefeval_k1_data import sha
from vision_memory.training.latent_bank_unet import stable_seed


def test_registered_budget_covers_every_source_event_without_parity_aliasing():
    per_source = collections.Counter()
    per_event = collections.Counter()
    joint = collections.Counter()
    initial = collections.Counter()
    for cycle in range(64):
        source, position = retention_draw(cycle)
        per_source[source] += 1
        per_event[position] += 1
        joint[source, position] += 1
        initial[cycle % 2] += 1
    assert per_source == {0: 16, 1: 16, 2: 16, 3: 16}
    assert initial == {0: 32, 1: 32}
    assert set(per_event) == set(range(1, 11)) and set(per_event.values()) == {6, 7}
    assert len(joint) == 40 and set(joint.values()) == {1, 2}
    assert 730 * (sum(per_source.values()) + sum(initial.values())) == 23360 * 4


def bank_fixture(tmp_path):
    root = tmp_path/'bank'
    rows = [{'base_pair_id': 'p:0'}, {'base_pair_id': 'p:1'}]
    checkpoint = tmp_path/'checkpoint-final.pt'
    checkpoint.write_bytes(b'fixed parent')
    (tmp_path/'complete.json').write_text(json.dumps({'steps': 23360, 'checkpoint_sha256': sha(checkpoint)}))
    variants = tmp_path/'variants.json'
    variants.write_text('{}')
    for row in rows:
        pid = row['base_pair_id']
        for variant, chain in SOURCE_ORDER:
            directory = root/f'V{variant}'/pid.replace(':', '_')/f'seed-{chain}'
            directory.mkdir(parents=True)
            png = directory/'prefix-00.png'
            png.write_bytes(f'{pid}:{variant}:{chain}'.encode())
            binding = {'checkpoint_sha256': sha(checkpoint), 'split': 'train', 'inter_turns': 0,
                       'noise_chains': 2, 'noise_domain': 'source-bank', 'steps': 28, 'cfg': 1,
                       'initial_variant': variant, 'initial_variants_sha256': sha(variants)}
            (directory/'complete.json').write_text(json.dumps({'binding': binding, 'png_hashes': {png.name: sha(png)}}))
            seed = stable_seed(20260924, noise_namespace(pid, chain, 'source-bank'), 0)
            (directory/'writes.jsonl').write_text(json.dumps({'position': 0, 'source_png_sha256': None,
                                                           'output_png_sha256': sha(png), 'noise_seed': seed})+'\n')
    return root, rows, checkpoint, variants


def test_bank_preserves_all_sources_and_rejects_parent_data_or_png_changes(tmp_path):
    root, rows, checkpoint, variants = bank_fixture(tmp_path)
    freeze_bank(root, rows, checkpoint, variants)
    paths = load_bank(root, rows, sha(checkpoint), sha(variants))
    assert set(paths) == {'p:0', 'p:1'} and all(len(x) == 4 for x in paths.values())
    with pytest.raises(AssertionError):
        load_bank(root, rows[:1], sha(checkpoint), sha(variants))
    with pytest.raises(AssertionError):
        load_bank(root, rows, 'another parent', sha(variants))
    paths['p:0'][0].write_bytes(b'changed PNG')
    with pytest.raises(AssertionError):
        load_bank(root, rows, sha(checkpoint), sha(variants))


def test_bank_rejects_incomplete_or_eval_noise_sources(tmp_path):
    root, rows, checkpoint, variants = bank_fixture(tmp_path)
    path = root/'V1/p_1/seed-1/complete.json'
    saved = path.read_text()
    data = json.loads(saved)
    data['binding']['noise_domain'] = 'eval'
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        freeze_bank(root, rows, checkpoint, variants)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        freeze_bank(root, rows, checkpoint, variants)


def test_disk_conditions_share_exact_float32_tensors_only_with_same_binding(tmp_path):
    condition = {'source': torch.randn(1, 4, 8, 8), 'embeds': torch.randn(1, 20, 12),
                 'mask': torch.ones(1, 20, dtype=torch.int64)}
    c = DiskConditions(tmp_path, {'bank': 'same', 'code': 'fixed'})
    c.ensure(('p:0', 1, 0), lambda: condition)
    i = DiskConditions(tmp_path, {'bank': 'same', 'code': 'fixed'})
    def must_not_reencode():
        raise AssertionError('Shared cache should reuse the same encoded source')
    i.ensure(('p:0', 1, 0), must_not_reencode)
    for key, expected in condition.items():
        assert torch.equal(i['p:0', 1, 0][key], expected)
        assert i['p:0', 1, 0][key].dtype == expected.dtype
    other = DiskConditions(tmp_path, {'bank': 'changed', 'code': 'fixed'})
    with pytest.raises(FileNotFoundError):
        other['p:0', 1, 0]


def test_source_noise_is_separate_and_default_eval_seeds_unchanged():
    from scripts.experiments.prefeval_k1_data import load_training_records
    bank, evaluation = set(), set()
    for row in load_training_records('train'):
        pid = row['base_pair_id']
        for chain in range(2):
            assert noise_namespace(pid, chain) == f'rollout:{pid}:{chain}'
            bank.add(stable_seed(20260924, noise_namespace(pid, chain, 'source-bank'), 0))
            evaluation.update(stable_seed(20260924, noise_namespace(pid, chain), p) for p in range(11))
    assert len(bank) == 1460 and bank.isdisjoint(evaluation)
