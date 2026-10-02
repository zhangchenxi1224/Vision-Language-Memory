"""Shared, unfiltered four-PNG bank for the registered C730/I730 comparison."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.prefeval_k1_data import load_training_records, sha

SOURCE_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]
POSITION_OFFSETS = [0, 2, 5, 7]


def retention_draw(cycle):
    """64 visits: each source 16 times, each event 6/7 times, no source/event aliasing."""
    source = cycle % 4
    position = 1 + (cycle // 4 + POSITION_OFFSETS[source]) % 10
    return source, position


def noise_namespace(pair_id, chain, domain='eval'):
    assert domain in {'eval', 'source-bank', 'refresh-0', 'refresh-1', 'refresh-2', 'refresh-3'}
    original = f'rollout:{pair_id}:{chain}'
    return original if domain == 'eval' else domain + ':' + original


def freeze_bank(root, rows, checkpoint, variants):
    from vision_memory.training.latent_bank_unet import stable_seed
    root, checkpoint, variants = map(Path, (root, checkpoint, variants))
    parent = sha(checkpoint)
    final = json.loads((checkpoint.parent / 'complete.json').read_text())
    assert final == {'steps': 23360, 'checkpoint_sha256': parent}
    variants_hash = sha(variants)
    entries, bank_seeds, eval_seeds = {}, set(), set()
    for row in rows:
        pid = row['base_pair_id']
        entries[pid] = []
        for variant, chain in SOURCE_ORDER:
            directory = root / f'V{variant}' / pid.replace(':', '_') / f'seed-{chain}'
            done = json.loads((directory / 'complete.json').read_text())
            binding = done['binding']
            assert binding['checkpoint_sha256'] == parent
            assert binding['split'] == 'train' and binding['inter_turns'] == 0
            assert binding['noise_chains'] == 2 and binding['noise_domain'] == 'source-bank'
            assert binding['steps'] == 28 and binding['cfg'] == 1
            assert binding['initial_variant'] == variant
            assert binding['initial_variants_sha256'] == variants_hash
            assert 'probe_initial_sources' not in binding
            png = directory / 'prefix-00.png'
            digest = sha(png)
            assert done['png_hashes'] == {'prefix-00.png': digest}
            writes = [json.loads(line) for line in (directory / 'writes.jsonl').read_text().splitlines()]
            seed = stable_seed(20260924, noise_namespace(pid, chain, 'source-bank'), 0)
            assert writes[-1]['position'] == 0 and writes[-1]['source_png_sha256'] is None
            assert writes[-1]['output_png_sha256'] == digest and writes[-1]['noise_seed'] == seed
            bank_seeds.add(seed)
            eval_seeds.update(stable_seed(20260924, noise_namespace(pid, chain), p) for p in range(11))
            entries[pid].append({'variant': variant, 'chain': chain, 'noise_seed': seed,
                                 'png': str(png.relative_to(root)), 'sha256': digest})
    assert bank_seeds.isdisjoint(eval_seeds)
    artifact = {'schema': 1, 'split': 'train', 'parent_sha256': parent, 'parent_steps': 23360,
                'variants_sha256': variants_hash, 'source_order': SOURCE_ORDER,
                'filter': 'none; retain all registered training preferences', 'items': entries}
    path = root / 'bank.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(artifact)), 'Frozen bank changed'
    else:
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(artifact, indent=2) + '\n')
        temp.replace(path)
    return artifact


def load_bank(root, rows, parent_hash, variants_hash):
    root = Path(root)
    artifact = json.loads((root / 'bank.json').read_text())
    assert artifact['schema'] == 1 and artifact['split'] == 'train'
    assert artifact['parent_sha256'] == parent_hash and artifact['parent_steps'] == 23360
    assert artifact['variants_sha256'] == variants_hash
    assert artifact['source_order'] == [list(x) for x in SOURCE_ORDER]
    assert set(artifact['items']) == {r['base_pair_id'] for r in rows}, 'Missing or extra preference'
    paths = {}
    for pid, entries in artifact['items'].items():
        assert [(x['variant'], x['chain']) for x in entries] == SOURCE_ORDER
        paths[pid] = []
        for item in entries:
            path = (root / item['png']).resolve()
            assert path.is_relative_to(root.resolve())
            assert sha(path) == item['sha256'], 'Source PNG changed after freezing'
            paths[pid].append(path)
    return paths


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--variants', type=Path, required=True)
    args = p.parse_args()
    rows = load_training_records('train')
    assert len(rows) == 730
    frozen = freeze_bank(args.root, rows, args.checkpoint, args.variants)
    load_bank(args.root, rows, frozen['parent_sha256'], frozen['variants_sha256'])
    print(json.dumps({'preferences': len(rows), 'pngs': len(rows) * 4,
                      'bank_sha256': sha(args.root / 'bank.json')}))
