"""Verify the fixed 2880-update endpoint, paired raw generations, and every draw."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
BANK_SHA = '962f02846ed1a1933e6c219604bc22ee520e28f2dfe2721e26f111dc36ea122e'
COMMIT = '9628d7142db5a81a9d11a35b89d0515ef32d2e4f'
PROMPTS = ('original_open', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3', 'paraphrase_4')
# Verified from the sealed frozen Reader's 45/45 direct positive controls.
GOLD_IDS = {'ambient': [59614], 'jazz': [73, 9802], 'no active preference': [2152, 4541, 21933]}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def seed(index):
    return int.from_bytes(hashlib.sha256(f'20260913:heldout-evaluation-noise:{index}'.encode()).digest()[:8], 'big') % (2**63 - 1)


def phase_summary(rows, bank, phase):
    groups = {g['question_id']: g for g in bank['groups']}
    expected = {(qid, condition, noise, prompt) for qid in groups for prompt in PROMPTS
                for condition, noise in [('blank', None), ('donor', None), *(('matched', seed(i)) for i in range(4))]}
    seen, cells, images, raw_by_cell, paired = set(), {}, {}, {}, {}
    for row in rows:
        key = row['question_id'], row['condition'], row['noise_seed'], row['prompt_id']
        if key not in expected or key in seen or row['phase'] != phase:
            raise ValueError('Unexpected, repeated, or wrong-phase evaluation row')
        seen.add(key)
        group = groups[row['question_id']]
        if row['query'] != group['question_variants'][row['prompt_id']] or row['gold'] != group['answer']:
            raise ValueError('Query or gold differs from the bank')
        if row['scorer']['gold_token_ids'] != GOLD_IDS[row['gold']]:
            raise ValueError('Gold token IDs differ from the sealed Reader positive controls')
        passed = row['generated_token_ids'] == GOLD_IDS[row['gold']] + [151645]
        if passed != bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']):
            raise ValueError('Raw answer tokens disagree with the score')
        image_key = key[:-1]
        image = images.setdefault(image_key, {'hash': row['image_sha256'], 'answers': []})
        if image['hash'] != row['image_sha256']:
            raise ValueError('Query variants read different images')
        image['answers'].append(passed)
        if row['condition'] == 'matched':
            cell_key = '/'.join((group['source_state'], group['operation'], group['target_state'], f"wording-{group['wording_index']}"))
            cell = cells.setdefault(cell_key, {'n': 0, 'correct_eos': 0})
            cell['n'] += 1
            cell['correct_eos'] += int(passed)
            raw_by_cell.setdefault(cell_key, Counter())[row['raw']] += 1
        paired[key] = {'passed': passed, 'image_sha256': row['image_sha256'],
                       'generated_token_ids': row['generated_token_ids']}
    if seen != expected or len(expected) != 1350 or len(cells) != 45 or any(c['n'] != 20 for c in cells.values()):
        raise ValueError('Incomplete fixed 45-group evaluation coverage')
    matched_images = [v for k, v in images.items() if k[1] == 'matched']
    summary = {'phase': phase, 'raw_rows': len(rows), 'matched_rows': 900,
               'correct_eos': sum(c['correct_eos'] for c in cells.values()),
               'generated_images': len(matched_images),
               'images_passing_all_five': sum(len(v['answers']) == 5 and all(v['answers']) for v in matched_images),
               'cells': cells, 'raw_by_cell': raw_by_cell}
    return summary, paired


def collect(run, bank_path, *, text_only=False):
    run, bank_path = Path(run), Path(bank_path)
    if sha(bank_path) != BANK_SHA:
        raise ValueError('Wrong transition bank')
    bank = read(bank_path)
    identity = read(run / 'train/identity.json')
    for key, value in {'git_commit': COMMIT, 'steps': 2880, 'seed': 20260913, 'eval_seeds': 4,
                       'bank_manifest_sha256': BANK_SHA, 'model_variant': 'base', 'flow_protocol': 'official',
                       'trainable_scope': 'full_unet', 'gradient_accumulation_steps': 4}.items():
        if identity.get(key) != value:
            raise ValueError('Unexpected training identity: ' + key)
    runtime = read(run / 'train/runtime.json')['additional_protocol_binding']
    if runtime['inference_guidance_scale'] != 1. or runtime['inference_steps'] != 28:
        raise ValueError('Unexpected native inference protocol')
    bindings = runtime['source_bindings']
    if set(bindings) != {g['question_id'] for g in bank['groups']}:
        raise ValueError('Incomplete source bindings')
    for group in bank['groups']:
        if group['source_kind'] == 'sealed_rgb_1024':
            source = bindings[group['question_id']]
            if (source['rms_difference'] != 0 or source['official_source_sha256'] != group['source_latent_sha256']
                    or source['source_image_file_sha256'] != group['source_image_file_sha256']):
                raise ValueError('Source PNG binding changed')
    terminal, result = read(run / 'terminal.json'), read(run / 'train/result.json')
    if terminal['state'] != 'completed' or terminal['training_result_sha256'] != sha(run / 'train/result.json') or result['optimizer_steps'] != 2880:
        raise ValueError('Require a sealed completed fixed endpoint')
    omitted = []
    checkpoint = run / 'train/checkpoint-final.pt'
    if text_only and not checkpoint.exists():
        omitted.append('train/checkpoint-final.pt')
    elif sha(checkpoint) != result['checkpoint_sha256']:
        raise ValueError('Final checkpoint changed')
    phases, pairs = {}, {}
    for phase in ('baseline', 'trained'):
        directory = run / 'train' / phase
        complete = read(directory / 'complete.json')
        expected_artifacts = {'generations.jsonl', 'summary.json'} | {
            hashlib.sha256(g['question_id'].encode()).hexdigest()[:16] + f'-seed-{i:02d}.pt'
            for g in bank['groups'] for i in range(4)}
        if set(complete['artifact_hashes']) != expected_artifacts or complete['generation_rows'] != 1350:
            raise ValueError('Missing fixed endpoint artifacts')
        for name, digest in complete['artifact_hashes'].items():
            if Path(name).name != name:
                raise ValueError('Invalid artifact name')
            path = directory / name
            if text_only and path.suffix == '.pt' and not path.exists():
                omitted.append(str(path.relative_to(run)))
            elif sha(path) != digest:
                raise ValueError('Changed phase artifact: ' + name)
        phases[phase], pairs[phase] = phase_summary(jsonl(directory / 'generations.jsonl'), bank, phase)
        phases[phase]['complete_sha256'] = sha(directory / 'complete.json')
    matched_pairs = Counter()
    for key, left in pairs['baseline'].items():
        right = pairs['trained'][key]
        if key[1] != 'matched':
            if left != right:
                raise ValueError('Blank/donor control changed across the paired evaluation')
        else:
            matched_pairs[f"{int(left['passed'])}->{int(right['passed'])}"] += 1
    from vision_memory.training.latent_bank_unet import balanced_draw
    metrics = jsonl(run / 'train/training.jsonl')
    draws = []
    if len(metrics) != 2880:
        raise ValueError('Wrong optimizer update count')
    for index, row in enumerate(metrics):
        if row['optimizer_step'] != index + 1 or len(row['microbatches']) != 4:
            raise ValueError('Optimizer sequence or accumulation changed')
        for micro, draw in enumerate(row['microbatches']):
            group, teacher, noise, sigma = balanced_draw(bank['groups'], 20260913, index * 4 + micro)
            if (draw['question_id'], draw['teacher_id'], draw['noise_seed'], draw['effective_sigma']) != (group['question_id'], teacher, noise, sigma):
                raise ValueError('Training draw differs from deterministic replay')
            if not math.isfinite(draw['flow_matching_mse']):
                raise ValueError('Nonfinite training loss')
            draws.append(draw)
    counts = Counter(d['question_id'] for d in draws)
    if set(counts) != {g['question_id'] for g in bank['groups']} or any(n != 256 for n in counts.values()):
        raise ValueError('Training exposure differs from the fixed design')
    return {'identity': identity, 'result_sha256': sha(run / 'train/result.json'),
            'checkpoint_sha256': result['checkpoint_sha256'], 'phases': phases, 'matched_pairs': matched_pairs,
            'optimizer_steps': 2880, 'exact_draws_replayed': len(draws), 'draws_per_group': counts,
            'sigma_min': min(d['effective_sigma'] for d in draws), 'sigma_max': max(d['effective_sigma'] for d in draws),
            'sigma_above_half': sum(d['effective_sigma'] > .5 for d in draws),
            'artifacts_omitted_locally': omitted, 'all_remote_artifacts_verified_here': not omitted,
            'development_all_correct_eos': phases['trained']['correct_eos'] == 900,
            'scope': 'One entity, three states. Complete development is not independent RGB-chain or unseen-entity validation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bank', 'output-prefix'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--text-only', action='store_true')
    a = p.parse_args()
    summary = collect(a.run, a.bank, text_only=a.text_only)
    out = Path(str(a.output_prefix) + '-summary.json')
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    if not a.text_only:
        with tarfile.open(str(a.output_prefix) + '-evidence.tgz', 'w:gz') as archive:
            for path in sorted(a.run.rglob('*')):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.log'):
                    archive.add(path, arcname=str(path.relative_to(a.run)))
            archive.add(out, arcname='verified-summary.json')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('identity', 'phases', 'draws_per_group', 'artifacts_omitted_locally')}))


if __name__ == '__main__':
    main()
