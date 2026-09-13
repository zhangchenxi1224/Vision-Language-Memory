"""Verify every registered broader validation row and captured native artifact."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_broader_endpoint import parent_binding, phase_summary, strict_pass, COMMIT, BANK_SHA, PLAN_SHA
from scripts.reporting.collect_transition_endpoint import read, jsonl, sha
from scripts.probes.official_transition_confirmation import resolve_events
from scripts.probes.official_broader_confirmation import selected_cases


def expected_rows(registered, bank, mode, lane):
    original, resolved = resolve_events(registered['transition_validation'],
        {**bank, 'groups': [group for group in bank['groups'] if 'source_state' in group]})
    cases = selected_cases({**registered, 'transition_validation': resolved}, mode, lane)
    expected, artifacts = {}, {'identity.json', 'generations.jsonl'}
    def add(meta, variants):
        for prompt, query in variants.items():
            key = meta['case'], meta['noise_seed'], prompt
            if key in expected:
                raise ValueError('Repeated registered cell')
            expected[key] = {**meta, 'prompt_id': prompt, 'query': query}
    if mode == 'rgb_chains':
        for sequence in cases:
            previous = None
            for step in sequence['steps']:
                name = f"sequence-{sequence['sequence']}-rep-{sequence['repetition']}-step-{step['step']}"
                artifacts.update((name + '.pt', name + '.png'))
                add({'case': name, 'sequence': sequence['sequence'], 'repetition': sequence['repetition'], **step,
                    'condition': 'matched', 'source_artifact': previous, 'image_artifact': name + '.png'},
                    original['ambient']['question_variants'])
                previous = name + '.png'
    else:
        groups = {group['question_id']: group for group in bank['groups']}
        for case in cases:
            group = original[case['state']] if mode == 'single_writes' else groups[case['question_id']]
            label = case['state'] + '_' + case['style'] if mode == 'single_writes' else case['case']
            if case['gold'] != group['answer'] or ('question_variants' in case and case['question_variants'] != group['question_variants']):
                raise ValueError('Registered answer or historical question differs')
            for index, seed in enumerate(case['noise_seeds']):
                name = label + f'-seed-{index:02d}'
                artifacts.update((name + '.pt', name + '.png'))
                add({'case': label, 'question_id': group['question_id'], 'condition': 'matched', 'gold': group['answer'],
                    'event_text': case['event_text'], 'noise_seed': seed, 'image_artifact': name + '.pt'}, group['question_variants'])
        controls = list(original.items()) if mode == 'single_writes' else [
            ('target-' + str(groups[qid]['historical_target_index']), groups[qid]) for qid in sorted({case['question_id'] for case in cases})]
        for label, group in controls:
            for condition in ('blank', 'donor'):
                name = label + '_' + condition
                artifacts.add(name + '.pt')
                add({'case': name, 'question_id': group['question_id'], 'condition': condition, 'gold': group['answer'],
                    'event_text': None, 'noise_seed': None, 'image_artifact': name + '.pt'}, group['question_variants'])
    return expected, artifacts, cases


def summarize_rows(rows, registered, bank, mode, lane):
    expected, artifacts, cases = expected_rows(registered, bank, mode, lane)
    seen, cells, groups, images, chains = set(), {}, {}, {}, {}
    for row in rows:
        key = row['case'], row['noise_seed'], row['prompt_id']
        if key not in expected or key in seen or any(row.get(k) != v for k, v in expected[key].items()):
            raise ValueError('Missing, duplicated or changed registered query/event/seed/source cell')
        seen.add(key)
        passed = strict_pass(row, row['gold'])
        cell = cells.setdefault(row['case'] + '/' + row['prompt_id'], {'n': 0, 'correct_eos': 0, 'condition': row['condition']})
        cell['n'] += 1
        cell['correct_eos'] += int(passed)
        group = groups.setdefault(row['case'], {'n': 0, 'correct_eos': 0, 'condition': row['condition'], 'raw': Counter()})
        group['n'] += 1
        group['correct_eos'] += int(passed)
        group['raw'][row['raw']] += 1
        image = images.setdefault(key[:-1], {'hash': row['image_sha256'], 'answers': [], 'condition': row['condition']})
        if image['hash'] != row['image_sha256']:
            raise ValueError('Query variants read different images')
        image['answers'].append(passed)
        if mode == 'rgb_chains':
            chains.setdefault((row['sequence'], row['repetition']), []).append(passed)
    expected_count = {'single_writes': 390, 'rgb_chains': 480, 'historical_prefixes': 560}[mode]
    if seen != set(expected) or len(rows) != expected_count:
        raise ValueError('Incomplete registered validation matrix')
    matched = [group for group in groups.values() if group['condition'] == 'matched']
    generated = [image for image in images.values() if image['condition'] == 'matched']
    summary = {'mode': mode, 'prefix_lane': lane, 'raw_rows': len(rows), 'groups': groups, 'cells': cells,
        'matched_rows': sum(group['n'] for group in matched), 'matched_correct_eos': sum(group['correct_eos'] for group in matched),
        'generated_images': len(generated), 'all_five_prompt_images': sum(len(image['answers']) == 5 and all(image['answers']) for image in generated),
        'all_generated_correct_eos': all(group['n'] == group['correct_eos'] for group in matched)}
    if mode == 'rgb_chains':
        summary.update(chain_count=len(chains), complete_chains_correct_eos=sum(len(values) == 30 and all(values) for values in chains.values()))
    return summary, artifacts, cases


def verify_tensors(run, rows, mode):
    import numpy as np
    import torch
    from PIL import Image
    from vision_memory.repro import canonical_tensor_sha256
    selected = [row for row in rows if row['prompt_id'] == 'original_open']
    for row in selected:
        path = run / row['image_artifact']
        payload = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=True)
        pixels = payload['image']
        if canonical_tensor_sha256(pixels) != row['image_sha256']:
            raise ValueError('Reader row differs from captured pixels')
        if row['condition'] != 'matched':
            continue
        trajectory = payload['trajectory']
        if (len(trajectory) != 29 or not torch.equal(trajectory[0], payload['noise'])
                or not torch.equal(trajectory[-1], payload['latent'])):
            raise ValueError('Native trajectory start or endpoint differs')
        noise = torch.randn(payload['noise'].shape, generator=torch.Generator().manual_seed(row['noise_seed']), dtype=torch.float32)
        if not torch.equal(noise, payload['noise']):
            raise ValueError('Not the registered pure Gaussian initial state')
        with Image.open(path.with_suffix('.png')) as image:
            if image.mode != 'RGB' or image.size != (1024, 1024):
                raise ValueError('Unexpected image format')
            array = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0)
        if not torch.equal((pixels * 255).round().byte(), array):
            raise ValueError('Visualization differs from captured FP32 pixels')
        if mode == 'rgb_chains' and not torch.equal(pixels, array.float() / 255.):
            raise ValueError('Chain Reader did not see actual stored RGB pixels')
    return len(selected)


def collect(run, parent, bank_path, expected_probe_commit, *, text_only=False):
    run, parent, bank_path = map(Path, (run, parent, bank_path))
    bank, _, result = parent_binding(parent, bank_path)
    registered = read(parent / 'preregistered-experiment.json')
    complete = read(run / 'complete.json')
    identity = complete['identity']
    if read(run / 'identity.json') != identity or len(expected_probe_commit) != 40:
        raise ValueError('Validation identity or explicit probe commit missing')
    for key, value in {'probe_commit': expected_probe_commit, 'parent_commit': COMMIT, 'bank_sha256': BANK_SHA,
            'plan_file_sha256': PLAN_SHA, 'registered_plan': registered, 'optimizer_updates': 0, 'guidance_scale': 1.,
            'native_steps': 28, 'checkpoint_sha256': result['checkpoint_sha256'],
            'parent_result_sha256': sha(parent / 'train/result.json')}.items():
        if identity.get(key) != value:
            raise ValueError('Unexpected validation binding: ' + key)
    after = parent / 'train/trained'
    if sha(after / 'generations.jsonl') != read(after / 'complete.json')['artifact_hashes']['generations.jsonl']:
        raise ValueError('Parent raw evaluation changed')
    development, _ = phase_summary(jsonl(after / 'generations.jsonl'), bank, 'trained')
    interpretation = 'fresh_confirmation' if development['correct_eos'] == 1510 else 'diagnostic_after_development_failure'
    if identity['interpretation'] != interpretation or identity['development_correct_eos'] != development['correct_eos']:
        raise ValueError('Parent development outcome or validation interpretation changed')
    rows = jsonl(run / 'generations.jsonl')
    summary, artifacts, cases = summarize_rows(rows, registered, bank, identity['mode'], identity['prefix_lane'])
    if (identity['selected_cases'] != cases or complete['cells'] != summary['cells']
            or complete['all_generated_correct_eos'] != summary['all_generated_correct_eos']
            or set(complete['artifact_hashes']) != artifacts):
        raise ValueError('Sealed coverage, artifacts or outcomes differ')
    omitted = []
    def verify(path, digest, label):
        if text_only and path.suffix == '.pt' and not path.exists():
            omitted.append(label)
        elif sha(path) != digest:
            raise ValueError('Changed artifact: ' + label)
    verify(parent / 'train/checkpoint-final.pt', result['checkpoint_sha256'], 'parent/train/checkpoint-final.pt')
    for name, digest in complete['artifact_hashes'].items():
        if Path(name).name != name:
            raise ValueError('Invalid artifact path')
        verify(run / name, digest, name)
    for row in rows:
        if identity['mode'] == 'rgb_chains' and sha(run / row['image_artifact']) != row['image_file_sha256']:
            raise ValueError('Stored chain PNG differs from the generation record')
    checked = 0
    if not text_only:
        from scripts.train.train_latent_bank_unet import source_hashes
        if identity['source_hashes'] != source_hashes() or identity['probe_file_sha256'] != sha(ROOT / 'scripts/probes/official_broader_confirmation.py'):
            raise ValueError('Validation source differs from the inspected checkout')
        checked = verify_tensors(run, rows, identity['mode'])
    return {**summary, 'identity': identity, 'complete_sha256': sha(run / 'complete.json'), 'interpretation': interpretation,
        'artifacts_omitted_locally': omitted, 'all_files_verified_here': not omitted,
        'pixel_noise_trajectory_tensors_checked': checked}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'parent', 'bank', 'output-prefix'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-probe-commit', required=True)
    parser.add_argument('--text-only', action='store_true')
    a = parser.parse_args()
    summary = collect(a.run, a.parent, a.bank, a.expected_probe_commit, text_only=a.text_only)
    output = Path(str(a.output_prefix) + '-summary.json')
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    if not a.text_only:
        with tarfile.open(str(a.output_prefix) + '-evidence.tgz', 'w:gz') as archive:
            for path in sorted(a.run.iterdir()):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.png'):
                    archive.add(path, arcname=path.name)
            archive.add(output, arcname='verified-summary.json')
            archive.add(a.parent / 'preregistered-experiment.json', arcname='preregistered-plan.json')
    print(json.dumps({key: value for key, value in summary.items() if key not in ('identity', 'groups', 'cells', 'artifacts_omitted_locally')}))
