"""Fixed B730 initialization-ablation evaluation using the existing PNG/Reader path."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.prefeval_k1_data import load_records, official_mcq, option_order, sha, event_text
from scripts.experiments.prefeval_k1_variants import load_variants, apply_variant

FAMILIES = ('T1', 'T2', 'T3')
CONTROLS = ('memory', 'mismatch', 'blank', 'text')
POSITION_CONTROLS = ('memory', 'mismatch', 'blank')
POSITIONS = ('official', 0, 1, 2, 3)
WRITER = ROOT / 'scripts/experiments/prefeval_k1_writer.py'
READER = ROOT / 'scripts/experiments/prefeval_k1_evaluate.py'
PROBE = ROOT / 'scripts/experiments/prefeval_k1_position_probe.py'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def complete_receipt(binding, binding_path, summary_path, summary):
    return {'status': 'complete', 'binding_sha256': sha(binding_path), 'summary_sha256': sha(summary_path),
            'optimizer_step': binding['optimizer_step'], 'checkpoint_sha256': binding['checkpoint_sha256'],
            'split': binding['split'], 'preferences': summary['preferences'],
            'primary_records': summary['primary_record_count'], 'position_records': summary['position_record_count'],
            'actual_pngs': summary['actual_png_count']}


def donor_map(rows):
    topics = defaultdict(list)
    for row in rows:
        topics[row['topic']].append(row['base_pair_id'])
    result = {pid: ids[(i + 1) % len(ids)] for ids in topics.values() for i, pid in enumerate(ids)}
    assert all(k != v for k, v in result.items()), 'Each topic needs a distinct mismatch donor'
    return result


def cyclic_order(position):
    shift = 0 if position == 'official' else position
    return list(range(4))[-shift:] + list(range(4))[:-shift] if shift else list(range(4))


def expected_primary(rows, chains):
    return {(r['base_pair_id'], c, 0, control, family, 'mcq')
            for r in rows for control in CONTROLS
            for c in (range(chains) if control in ('memory', 'mismatch') else range(1))
            for family in FAMILIES}


def expected_positions(rows, chains):
    return {(r['base_pair_id'], c, control, position)
            for r in rows for control in POSITION_CONTROLS
            for c in (range(1) if control == 'blank' else range(chains)) for position in POSITIONS}


def validate_records(records, rows, hashes, split, chains, mcq, *, positions=False):
    """Reject incomplete/duplicated matrices and recompute labels from original options."""
    by_id = {r['base_pair_id']: r for r in rows}
    donors = donor_map(rows)
    expected = expected_positions(rows, chains) if positions else expected_primary(rows, chains)
    indexed = {}
    for rec in records:
        pid, chain, control = rec['pair_id'], rec['chain'], rec['control']
        assert pid in by_id and rec['split'] == split and rec['endpoint_kind'] == 'student'
        assert rec['prefix'] == 0 and rec['max_new_tokens'] == 32
        row = by_id[pid]
        if positions:
            key = (pid, chain, control, rec['position'])
            assert rec['family'] == 'T1' and rec['order_mode'] == 'official-cyclic'
            order = cyclic_order(rec['position'])
            assert rec['order'] == order
            family = 'T1'
        else:
            key = (pid, chain, rec['prefix'], control, rec['family'], rec['task'])
            family = rec['family']
            step = int.from_bytes(hashlib.sha256(f'eval:{pid}:{family}'.encode()).digest()[:4], 'big')
            order, _ = option_order(pid, step)
            assert rec['option_order'] == order and rec['question'] == row['forms'][family]
        assert key in expected, f'Unexpected evaluation key: {key}'
        assert key not in indexed, f'Duplicate evaluation key: {key}'
        gold = 'ABCD'[order.index(0)]
        query = row['forms'][family] + mcq['get_mcq_question_format']([row['options'][i] for i in order])
        assert rec['reader_query'] == query, 'Question/options changed or extra context leaked'
        donor = donors[pid] if control == 'mismatch' else None
        digest = hashes[(donor or pid, chain)] if control in ('memory', 'mismatch') else None
        assert rec['donor_pair_id'] == donor and rec['png_sha256'] == digest
        predicted = mcq['extract_choice'](rec['generated']['raw'])
        assert rec['correct_letter'] == gold and rec['predicted_letter'] == predicted
        assert rec['correct'] == (predicted == gold) and rec['parse_failure'] == (predicted is None)
        assert isinstance(rec['generated']['truncated'], bool)
        indexed[key] = rec
    assert set(indexed) == expected, f'Incomplete matrix: {len(indexed)}/{len(expected)}'
    return indexed


def metric(records):
    n = len(records)
    correct = sum(int(r['correct']) for r in records)
    return {'correct': correct, 'total': n, 'accuracy': correct / n,
            'parse_failures': sum(int(r['parse_failure']) for r in records),
            'truncations': sum(int(r['generated']['truncated']) for r in records)}


def aggregate(primary, positions, rows, chains):
    ids = [r['base_pair_id'] for r in rows]
    result = {'preferences': len(ids), 'noise_chains': chains, 'families': {}, 'positions': {}}
    for family in FAMILIES:
        controls = {control: metric([r for k, r in primary.items() if k[3] == control and k[4] == family])
                    for control in CONTROLS}
        paired = [(primary[pid, c, 0, 'memory', family, 'mcq']['correct'],
                   primary[pid, c, 0, 'mismatch', family, 'mcq']['correct']) for pid in ids for c in range(chains)]
        result['families'][family] = {
            'controls': controls,
            'memory_minus_mismatch_pp': 100 * (controls['memory']['accuracy'] - controls['mismatch']['accuracy']),
            'memory_minus_blank_pp': 100 * (controls['memory']['accuracy'] - controls['blank']['accuracy']),
            'paired_memory_vs_mismatch': {'repaired': sum(a and not b for a, b in paired),
                                         'regressed': sum(b and not a for a, b in paired), 'total': len(paired)},
            'both_noise_correct': {'correct': sum(all(primary[pid, c, 0, 'memory', family, 'mcq']['correct']
                                                     for c in range(chains)) for pid in ids), 'total': len(ids)}}
    result['all_three_questions_and_both_noise_correct'] = {
        'correct': sum(all(primary[pid, c, 0, 'memory', f, 'mcq']['correct']
                           for c in range(chains) for f in FAMILIES) for pid in ids), 'total': len(ids)}
    for control in POSITION_CONTROLS:
        cs = list(range(1) if control == 'blank' else range(chains))
        result['positions'][control] = {
            'by_position': {str(p): metric([positions[pid, c, control, p] for pid in ids for c in cs]) for p in POSITIONS},
            'all_four_cyclic_correct_per_image': {
                'correct': sum(all(positions[pid, c, control, p]['correct'] for p in range(4)) for pid in ids for c in cs),
                'total': len(ids) * len(cs)},
            'all_four_cyclic_and_all_noise_correct_per_preference': {
                'correct': sum(all(positions[pid, c, control, p]['correct'] for p in range(4) for c in cs) for pid in ids),
                'total': len(ids)}}
    result['note'] = ('Preference IDs are the statistical units. Questions and noise chains are repeated measurements. '
                      'Official order duplicates cyclic shift zero and is reported separately, not counted as a fifth independent position.')
    return result


def validate_images(args, rows, binding):
    from PIL import Image
    from vision_memory.training.latent_bank_unet import stable_seed
    images = args.output / 'images'
    manifest = json.loads((images / 'manifest.json').read_text())
    required = {'checkpoint_sha256': binding['checkpoint_sha256'], 'split': args.split,
                'steps': 28, 'cfg': 1, 'noise_chains': args.noise_chains, 'inter_turns': 0}
    if args.split == 'official':
        required['benchmark_history_sha256'] = binding['history_sha256']
        assert 'initial_variants_sha256' not in manifest
    else:
        required.update(initial_variants_sha256=binding['initial_variants_sha256'], initial_variant=args.initial_variant)
    assert all(manifest.get(k) == v for k, v in required.items())
    assert manifest['state'] == 'only reopened uint8 RGB PNG; fresh Gaussian each write'
    assert manifest.get('noise_domain', 'eval') == 'eval' and 'probe_initial_sources' not in manifest
    variants = load_variants(args.initial_variants, rows) if args.initial_variants else None
    hashes = {}
    for row in rows:
        pid = row['base_pair_id']
        actual_row = apply_variant(row, variants, args.initial_variant) if variants else row
        event = event_text(actual_row['history'][:2])
        for chain in range(args.noise_chains):
            directory = images / pid.replace(':', '_') / f'seed-{chain}'
            png = directory / 'prefix-00.png'
            digest = sha(png)
            with Image.open(png) as image:
                assert image.format == 'PNG' and image.mode == 'RGB' and image.size == (1024, 1024)
                image.verify()
            done = json.loads((directory / 'complete.json').read_text())
            assert done['binding'] == manifest and done['png_hashes'] == {'prefix-00.png': digest}
            writes = jsonl(directory / 'writes.jsonl')
            # A killed rollout may have an uncommitted write. The committed image's last write is authoritative.
            assert writes and all(w['position'] == 0 for w in writes)
            last = writes[-1]
            assert last['source_png_sha256'] is None and last['output_png_sha256'] == digest
            assert last['noise_seed'] == stable_seed(20260924, f'rollout:{pid}:{chain}', 0)
            assert last['event'] == event
            hashes[pid, chain] = digest
    return hashes


@contextmanager
def endpoint_lock(path):
    import fcntl
    with path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def run_child(args, label, command):
    command = [str(x) for x in command]
    environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
                       PYTHONOPTIMIZE='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    receipt = {'command': command, 'status': 'running'}
    save(args.output / f'{label}-process.json', receipt)
    with (args.output / f'{label}.log').open('a', encoding='utf-8') as log:
        proc = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    receipt.update(status='complete' if proc.returncode == 0 else 'failed', exit_code=proc.returncode)
    save(args.output / f'{label}-process.json', receipt)
    assert proc.returncode == 0, f'{label} failed; see {args.output / (label + ".log")}'


def make_binding(args, rows, step):
    dependencies = [Path(__file__), WRITER, READER, PROBE,
                    ROOT / 'scripts/experiments/prefeval_k1_data.py',
                    ROOT / 'scripts/experiments/prefeval_k1_variants.py',
                    ROOT / 'scripts/experiments/prefeval_k1_init_ablation.py',
                    ROOT / 'scripts/eval/prefeval_rgb.py',
                    ROOT / 'src/vision_memory/reader/open_answer.py',
                    ROOT / 'src/vision_memory/reader/qwen3vl.py',
                    ROOT / 'src/vision_memory/dreamlite/conditioning.py',
                    ROOT / 'src/vision_memory/dreamlite/native_base.py',
                    ROOT / 'src/vision_memory/dreamlite/latent_codec.py',
                    ROOT / 'src/vision_memory/training/latent_bank_unet.py',
                    ROOT / 'src/vision_memory/training/checkpoint.py',
                    ROOT / 'third_party/prefeval_reference/utils/utils_mcq.py']
    return {'schema': 'prefeval-unet-init-evaluation/v1', 'checkpoint_sha256': sha(args.checkpoint),
            'optimizer_step': step, 'split': args.split, 'base': str(args.base.resolve()),
            'reader': str(args.reader.resolve()), 'official_source': str(args.official_source.resolve()),
            'noise_chains': args.noise_chains, 'initial_variant': args.initial_variant if args.initial_variants else None,
            'initial_variants_sha256': sha(args.initial_variants) if args.initial_variants else None,
            'history_sha256': sha(args.history_file) if args.history_file else None,
            'record_content_sha256': canonical_sha([{k: r[k] for k in ('base_pair_id', 'topic', 'history', 'forms', 'options')} for r in rows]),
            'protocol': {'steps': 28, 'cfg': 1, 'seed': 20260924, 'inter_turns': 0, 'families': list(FAMILIES),
                         'controls': list(CONTROLS), 'position_controls': list(POSITION_CONTROLS),
                         'position_order': 'official-cyclic', 'max_new_tokens': 32},
            'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in dependencies},
            'exposure_label': {'train': 'trained_contents', 'dev': 'trainheldout_unverified',
                               'official': 'heldout_topics_with_historical_research_exposure'}[args.split],
            'selection_policy': 'fixed_endpoints_only; official_only_after_train_dev; no_official_checkpoint_selection'}


def main(args):
    if not __debug__:
        raise RuntimeError('Run without Python optimization: evidence assertions must remain enabled')
    import torch
    for name in ('checkpoint', 'output', 'base', 'official_source', 'reader', 'initial_variants', 'history_file'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    assert args.noise_chains == 2 and args.initial_variant == 1
    if args.split == 'official':
        assert args.history_file and not args.initial_variants
    else:
        assert args.initial_variants and not args.history_file
    rows = load_records(args.split, history_file=args.history_file)
    assert len(rows) == {'train': 730, 'dev': 90, 'official': 180}[args.split]
    assert all(set(FAMILIES) <= set(row['forms']) for row in rows)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    step = checkpoint['optimizer_step']
    assert step in (2048, 23360) and (args.expected_step is None or step == args.expected_step)
    assert args.split != 'official' or step == 23360, 'Official topics are evaluated only at the frozen final endpoint'
    del checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    with endpoint_lock(args.output / 'evaluation.lock'):
        binding = make_binding(args, rows, step)
        path = args.output / 'binding.json'
        if path.exists():
            assert json.loads(path.read_text()) == binding, 'Evaluation identity changed; use a new output directory'
        else:
            assert not any((args.output / name).exists() for name in ('images', 'readback', 'positions.jsonl'))
            save(path, binding)
        complete = args.output / 'complete.json'
        if not complete.exists():
            history_args = (['--history-file', args.history_file] if args.split == 'official' else
                            ['--initial-variants', args.initial_variants, '--initial-variant', '1'])
            run_child(args, 'rollout', [sys.executable, WRITER, 'rollout', '--arm', 'B', '--checkpoint', args.checkpoint,
                '--base', args.base, '--official-source', args.official_source, '--split', args.split,
                '--output', args.output / 'images', *history_args,
                '--inter-turns', '0', '--noise-chains', '2', '--device', args.device])
            validate_images(args, rows, binding)
            run_child(args, 'readback', [sys.executable, READER, '--kind', 'student', '--split', args.split,
                '--reader', args.reader, '--images', args.output / 'images', '--output', args.output / 'readback',
                *history_args, '--families', ','.join(FAMILIES),
                '--controls', ','.join(CONTROLS), '--tasks', 'mcq', '--prefixes', '0', '--noise-chains', '2', '--device', args.device])
            run_child(args, 'positions', [sys.executable, PROBE, '--kind', 'student', '--split', args.split,
                '--reader', args.reader, '--images', args.output / 'images', '--output', args.output / 'positions.jsonl',
                '--order-mode', 'official-cyclic', '--controls', ','.join(POSITION_CONTROLS),
                '--prefix', '0', '--noise-chains', '2', '--device', args.device,
                *(['--history-file', args.history_file] if args.split == 'official' else [])])
        hashes = validate_images(args, rows, binding)
        mcq = official_mcq(ROOT / 'third_party/prefeval_reference')
        primary_paths = sorted((args.output / 'readback').glob('readback-*.jsonl'))
        assert primary_paths == [args.output / 'readback/readback-0.jsonl']
        primary = validate_records([r for p in primary_paths for r in jsonl(p)], rows, hashes, args.split, 2, mcq)
        positions = validate_records(jsonl(args.output / 'positions.jsonl'), rows, hashes, args.split, 2, mcq, positions=True)
        summary = aggregate(primary, positions, rows, 2)
        summary.update(binding=binding, matrix_complete=True, primary_record_count=len(primary),
                       position_record_count=len(positions), actual_png_count=len(hashes),
                       actual_png_hashes={f'{pid}/seed-{chain}': digest for (pid, chain), digest in hashes.items()})
        files = primary_paths + [args.output / 'positions.jsonl']
        summary['raw_sha256'] = {str(p.relative_to(args.output)): sha(p) for p in files}
        if complete.exists():
            saved = json.loads(complete.read_text())
            assert json.loads((args.output / 'summary.json').read_text()) == summary, 'Completed evidence changed'
            assert saved == complete_receipt(binding, path, args.output / 'summary.json', summary), 'Completion receipt changed'
        else:
            save(args.output / 'summary.json', summary)
            save(complete, complete_receipt(binding, path, args.output / 'summary.json', summary))
        print(json.dumps({'status': 'complete', 'output': str(args.output), 'step': step, 'split': args.split}))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'output', 'base', 'official-source', 'reader'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--initial-variants', type=Path)
    p.add_argument('--history-file', type=Path)
    p.add_argument('--split', choices=('train', 'dev', 'official'), required=True)
    p.add_argument('--expected-step', type=int, choices=(2048, 23360))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--noise-chains', type=int, choices=(2,), default=2)
    p.add_argument('--initial-variant', type=int, choices=(1,), default=1)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
