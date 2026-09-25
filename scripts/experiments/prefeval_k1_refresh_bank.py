"""Current-student PNG replay: four fixed rounds, no evaluation samples or filtering."""
import json
from pathlib import Path

from scripts.experiments.prefeval_k1_data import sha
from scripts.experiments.prefeval_k1_source_bank import noise_namespace

TOTAL_STEPS = 23360
ROUND_STEPS = 5840
ROUNDS = 4
DEPTHS = 10


def round_bounds(index):
    assert index in range(ROUNDS)
    return index * ROUND_STEPS, (index + 1) * ROUND_STEPS


def source_checkpoint(train, parent, index):
    start, _ = round_bounds(index)
    return Path(parent) if index == 0 else Path(train) / f'checkpoint-step-{start:06d}.pt'


def save_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Frozen record changed: {path}'
    else:
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(value, indent=2) + '\n')
        temp.replace(path)


def freeze_refresh_bank(root, rows, checkpoint, variants, index, shards=2):
    from vision_memory.training.latent_bank_unet import stable_seed
    root = Path(root)
    parent, variants_hash = sha(checkpoint), sha(variants)
    entries = {}
    for row_index, row in enumerate(rows):
        pid = row['base_pair_id']
        directory = root / f'shard-{row_index % shards}' / pid.replace(':', '_') / 'seed-0'
        done = json.loads((directory / 'complete.json').read_text())
        b = done['binding']
        assert b['checkpoint_sha256'] == parent and b['split'] == 'train'
        assert b['steps'] == 28 and b['cfg'] == 1
        assert b['noise_chains'] == 1 and b['inter_turns'] == DEPTHS - 1
        assert b['noise_domain'] == f'refresh-{index}'
        assert b['initial_variant'] == index % 2
        assert b['initial_variants_sha256'] == variants_hash
        assert 'probe_initial_sources' not in b
        assert set(done['png_hashes']) == {f'prefix-{d:02d}.png' for d in range(DEPTHS)}
        writes = [json.loads(line) for line in (directory / 'writes.jsonl').read_text().splitlines()][-DEPTHS:]
        assert [w['position'] for w in writes] == list(range(DEPTHS))
        entries[pid] = []
        for depth, write in enumerate(writes):
            path = directory / f'prefix-{depth:02d}.png'
            digest = sha(path)
            assert digest == done['png_hashes'][path.name] == write['output_png_sha256']
            assert write['source_png_sha256'] == (entries[pid][-1]['sha256'] if depth else None)
            assert write['noise_seed'] == stable_seed(20260924, noise_namespace(pid, 0, f'refresh-{index}'), depth)
            entries[pid].append({'depth': depth, 'png': str(path.relative_to(root)), 'sha256': digest})
    artifact = {'schema': 1, 'round': index, 'split': 'train', 'source_checkpoint_sha256': parent,
                'source_optimizer_step': round_bounds(index)[0], 'variants_sha256': variants_hash,
                'filter': 'none', 'items': entries}
    save_once(root / 'bank.json', artifact)
    return artifact


def load_refresh_bank(root, rows, checkpoint_hash, variants_hash, index):
    root = Path(root)
    artifact = json.loads((root / 'bank.json').read_text())
    assert artifact['schema'] == 1 and artifact['split'] == 'train' and artifact['filter'] == 'none'
    assert artifact['round'] == index and artifact['source_optimizer_step'] == round_bounds(index)[0]
    assert artifact['source_checkpoint_sha256'] == checkpoint_hash
    assert artifact['variants_sha256'] == variants_hash
    assert set(artifact['items']) == {r['base_pair_id'] for r in rows}
    result = {}
    for pid, entries in artifact['items'].items():
        assert [x['depth'] for x in entries] == list(range(DEPTHS))
        result[pid] = []
        for item in entries:
            path = (root / item['png']).resolve()
            assert path.is_relative_to(root.resolve()) and sha(path) == item['sha256']
            result[pid].append(path)
    return result


def validate_resume(first, index, state, bank_hash):
    start, end = round_bounds(index)
    assert start <= first <= end, 'Resume cursor is outside the registered round'
    if first > start:
        assert state == {'refresh_round': index, 'source_bank_sha256': bank_hash}
    elif first:
        assert state['refresh_round'] == index - 1, 'A refresh must follow the preceding trained round'


def archive_uncommitted_tail(path, first):
    """Interrupted updates after the saved optimizer cursor are evidence, not extra steps."""
    path = Path(path)
    if not path.exists():
        assert first == 0
        return
    lines = path.read_text().splitlines()
    valid = [line for line in lines if json.loads(line)['step'] <= first]
    tail = [line for line in lines if json.loads(line)['step'] > first]
    assert [json.loads(line)['step'] for line in valid] == list(range(1, first + 1))
    if tail:
        import time
        archive = path.with_name(f'optimization-uncommitted-{time.time_ns()}.jsonl')
        archive.write_text('\n'.join(tail) + '\n')
        temp = path.with_suffix('.tmp')
        temp.write_text(('\n'.join(valid) + '\n') if valid else '')
        temp.replace(path)
