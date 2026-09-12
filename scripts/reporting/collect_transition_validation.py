"""Verify new transition confirmation/chain evidence without rewriting failed gates."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_transition_endpoint import BANK_SHA, COMMIT, GOLD_IDS, PROMPTS, sha, read, jsonl, phase_summary
PROBE_COMMIT = '0f4076788bf4125bbca8851b6b270d5f8418d534'


def expected_rows(resolved, variants, mode):
    expected, artifacts = {}, {'identity.json', 'generations.jsonl'}
    if mode == 'single_writes':
        for case in resolved['single_writes']:
            name = case['state'] + '_' + case['style']
            for index, seed in enumerate(case['noise_seeds']):
                label = name + f'-seed-{index:02d}'
                artifacts.update((label + '.pt', label + '.png'))
                for prompt in PROMPTS:
                    expected[name, seed, prompt] = {'case': name, 'state': case['state'], 'gold': case['gold'],
                        'event_text': case['event_text'], 'noise_seed': seed, 'query': variants[prompt],
                        'image_artifact': label + '.pt'}
        for state, gold in (('ambient', 'ambient'), ('jazz', 'jazz'), ('clear', 'no active preference')):
            for control in ('blank', 'donor'):
                label = state + '_' + control
                artifacts.add(label + '.pt')
                for prompt in PROMPTS:
                    expected[label, None, prompt] = {'case': label, 'state': state, 'gold': gold, 'event_text': None,
                        'noise_seed': None, 'query': variants[prompt], 'image_artifact': label + '.pt'}
    elif mode == 'rgb_chains':
        for sequence in resolved['rgb_chains']:
            previous = None
            for step in sequence['steps']:
                label = f"sequence-{sequence['sequence']}-rep-{sequence['repetition']}-step-{step['step']}"
                artifacts.update((label + '.pt', label + '.png'))
                for prompt in PROMPTS:
                    expected[label, step['noise_seed'], prompt] = {'case': label, 'sequence': sequence['sequence'],
                        'repetition': sequence['repetition'], **step, 'query': variants[prompt],
                        'source_artifact': previous, 'image_artifact': label + '.png'}
                previous = label + '.png'
    else:
        raise ValueError('Unknown validation mode')
    return expected, artifacts


def summarize_rows(rows, resolved, variants, mode):
    expected, artifacts = expected_rows(resolved, variants, mode)
    seen, cells, groups, images = set(), {}, {}, {}
    for row in rows:
        key = row['case'], row['noise_seed'], row['prompt_id']
        if key not in expected or key in seen:
            raise ValueError('Unexpected or duplicated validation row')
        seen.add(key)
        if any(row.get(k) != v for k, v in expected[key].items()):
            raise ValueError('Query, event, seed, state or RGB source link changed')
        if row['scorer']['gold_token_ids'] != GOLD_IDS[row['gold']]:
            raise ValueError('Gold tokenization changed')
        passed = row['generated_token_ids'] == GOLD_IDS[row['gold']] + [151645]
        if passed != bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']):
            raise ValueError('Raw tokens disagree with the score')
        cell = cells.setdefault(row['case'] + '/' + row['prompt_id'], {'n': 0, 'correct_eos': 0})
        cell['n'] += 1
        cell['correct_eos'] += int(passed)
        if mode == 'rgb_chains':
            category = '/'.join((row['event_style'], 'initial_write' if row['step'] == 0 else row['operation'], row['expected_state']))
            control = False
        else:
            category = row['case']
            control = row['noise_seed'] is None
        group = groups.setdefault(category, {'n': 0, 'correct_eos': 0, 'raw': Counter(), 'control': control})
        group['n'] += 1
        group['correct_eos'] += int(passed)
        group['raw'][row['raw']] += 1
        if not control:
            image = images.setdefault(key[:-1], {'hash': row['image_sha256'], 'answers': [],
                'chain': (row['sequence'], row['repetition']) if mode == 'rgb_chains' else None})
            if image['hash'] != row['image_sha256']:
                raise ValueError('Reader variants saw different images')
            image['answers'].append(passed)
    if seen != set(expected):
        raise ValueError('Missing preregistered validation rows')
    matched = [g for g in groups.values() if not g['control']]
    summary = {'mode': mode, 'raw_rows': len(rows), 'groups': groups, 'cells': cells,
        'matched_rows': sum(g['n'] for g in matched), 'matched_correct_eos': sum(g['correct_eos'] for g in matched),
        'generated_images': len(images), 'all_five_prompt_images': sum(len(v['answers']) == 5 and all(v['answers']) for v in images.values()),
        'all_generated_correct_eos': all(g['n'] == g['correct_eos'] for g in matched)}
    if mode == 'rgb_chains':
        chains = {}
        for image in images.values():
            chains.setdefault(image['chain'], []).extend(image['answers'])
        summary['chain_count'] = len(chains)
        summary['complete_chains_correct_eos'] = sum(len(v) == 30 and all(v) for v in chains.values())
    return summary, artifacts


def collect(run, parent, bank_path, plan_path, *, text_only=False):
    from scripts.probes.official_transition_confirmation import development_gate, resolve_events
    from scripts.probes.transition_validation_plan import plan
    run, parent, bank_path, plan_path = map(Path, (run, parent, bank_path, plan_path))
    if sha(bank_path) != BANK_SHA or read(plan_path) != plan(20260913):
        raise ValueError('Bank or preregistered validation plan changed')
    bank = read(bank_path)
    original, resolved = resolve_events(read(plan_path), bank)
    complete = read(run / 'complete.json')
    identity = complete['identity']
    if read(run / 'identity.json') != identity:
        raise ValueError('Identity file differs from the seal')
    for key, value in {'parent_commit': COMMIT, 'probe_commit': PROBE_COMMIT, 'bank_sha256': BANK_SHA,
                       'optimizer_updates': 0, 'guidance_scale': 1., 'image_guidance_scale': 1., 'native_steps': 28,
                       'registered_plan': read(plan_path), 'resolved_plan': resolved, 'plan_file_sha256': sha(plan_path)}.items():
        if identity.get(key) != value:
            raise ValueError('Unexpected validation identity: ' + key)
    terminal, result = read(parent / 'terminal.json'), read(parent / 'train/result.json')
    parent_identity = read(parent / 'train/identity.json')
    for key, value in {'git_commit': COMMIT, 'steps': 2880, 'bank_manifest_sha256': BANK_SHA,
                       'trainable_scope': 'full_unet', 'model_variant': 'base', 'flow_protocol': 'official'}.items():
        if parent_identity.get(key) != value:
            raise ValueError('Unexpected parent identity: ' + key)
    if (terminal['state'] != 'completed' or terminal['training_result_sha256'] != sha(parent / 'train/result.json')
            or identity['parent_result_sha256'] != terminal['training_result_sha256']
            or identity['checkpoint_sha256'] != result['checkpoint_sha256']):
        raise ValueError('Validation is not bound to this completed parent')
    parent_rows_path = parent / 'train/trained/generations.jsonl'
    if sha(parent_rows_path) != read(parent / 'train/trained/complete.json')['artifact_hashes']['generations.jsonl']:
        raise ValueError('Parent development raw changed')
    parent_rows = jsonl(parent_rows_path)
    phase_summary(parent_rows, bank, 'trained')
    gate = json.loads(json.dumps(development_gate(parent_rows, bank)))
    interpretation = 'fresh_confirmation' if gate['all_correct_eos'] else 'diagnostic_after_development_failure'
    if identity['development_gate'] != gate or identity['interpretation'] != interpretation:
        raise ValueError('The development failure or validation interpretation was changed')
    summary, expected_artifacts = summarize_rows(jsonl(run / 'generations.jsonl'), resolved,
                                                 original['ambient']['question_variants'], identity['mode'])
    if (complete['cells'] != summary['cells'] or complete['all_generated_correct_eos'] != summary['all_generated_correct_eos']
            or set(complete['artifact_hashes']) != expected_artifacts):
        raise ValueError('Sealed cells or artifacts disagree with full raw coverage')
    omitted, verified = [], []
    checkpoint = parent / 'train/checkpoint-final.pt'
    if text_only and not checkpoint.exists():
        omitted.append('parent/train/checkpoint-final.pt')
    elif sha(checkpoint) != identity['checkpoint_sha256']:
        raise ValueError('Parent checkpoint changed')
    for name, digest in complete['artifact_hashes'].items():
        if Path(name).name != name:
            raise ValueError('Invalid artifact path')
        path = run / name
        if text_only and path.suffix == '.pt' and not path.exists():
            omitted.append(name)
        elif sha(path) != digest:
            raise ValueError('Validation artifact changed: ' + name)
        else:
            verified.append(name)
    if identity['mode'] == 'rgb_chains':
        for row in jsonl(run / 'generations.jsonl'):
            if sha(run / row['image_artifact']) != row['image_file_sha256']:
                raise ValueError('Chain PNG differs from its raw row')
    return {**summary, 'identity': identity, 'complete_sha256': sha(run / 'complete.json'),
            'verified_files': verified, 'artifacts_omitted_locally': omitted, 'all_files_verified_here': not omitted,
            'interpretation': interpretation, 'preregistered_plan_sha256': sha(plan_path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'parent', 'bank', 'plan', 'output-prefix'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--text-only', action='store_true')
    a = p.parse_args()
    summary = collect(a.run, a.parent, a.bank, a.plan, text_only=a.text_only)
    out = Path(str(a.output_prefix) + '-summary.json')
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    if not a.text_only:
        with tarfile.open(str(a.output_prefix) + '-evidence.tgz', 'w:gz') as archive:
            for path in sorted(a.run.iterdir()):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.png'):
                    archive.add(path, arcname=path.name)
            archive.add(out, arcname='verified-summary.json')
            archive.add(a.plan, arcname='preregistered-plan.json')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('identity', 'groups', 'cells', 'verified_files', 'artifacts_omitted_locally')}))


if __name__ == '__main__':
    main()
