"""Verify sealed CFG1 validation evidence against its preregistered plan."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

PROMPTS = ('original_open', 'paraphrase_1', 'paraphrase_2', 'paraphrase_3', 'paraphrase_4')
PARENT_SHA = '4dfb56f942d7dea62ac1d52ec91a0a814df30d4f6accc1238f9310dd822dff14'
CHECKPOINT_SHA = '3c4b0679f16dd7714a662cfaddcd7716f7d3d43a49920898ab19a522d77af38d'
PROBE_COMMIT = '09b324dc7574ed33c0236ee09a4f5c3526fb85bb'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def collect(run, plan_path, kind, *, text_only=False):
    complete = json.loads((run / 'complete.json').read_text())
    identity = complete['identity']
    plan = json.loads(plan_path.read_text())
    require(identity['plan'] == plan, 'Executed plan differs from preregistration')
    for key, value in {'parent_result_sha256': PARENT_SHA, 'checkpoint_sha256': CHECKPOINT_SHA,
                       'probe_commit': PROBE_COMMIT, 'optimizer_updates': 0,
                       'guidance_scale': 1.0, 'image_guidance_scale': 1.0}.items():
        require(identity.get(key) == value, 'Unexpected identity: ' + key)
    require(identity.get('inference_steps', identity.get('native_steps')) == 28, 'Not native 28 steps')
    verified, remote_only = [], []
    for name, digest in complete['artifact_hashes'].items():
        require(Path(name).name == name, 'Artifact path must be a basename')
        path = run / name
        if text_only and path.suffix == '.pt' and not path.exists():
            remote_only.append(name)
            continue
        require(sha(path) == digest, 'Artifact hash mismatch: ' + name)
        verified.append(name)
    require(json.loads((run / 'identity.json').read_text()) == identity, 'Identity file differs')
    rows = [json.loads(line) for line in (run / 'generations.jsonl').read_text().splitlines()]
    expected, seen, groups, cells = {}, set(), {}, {}
    if kind == 'confirmation':
        for case in plan:
            for seed in case['seeds']:
                for prompt in PROMPTS:
                    expected[case['case'], seed, prompt] = (case['state'], case['gold'], case['event'])
        for state, gold in (('ambient', 'ambient'), ('jazz', 'jazz'), ('clear', 'no active preference')):
            for control in ('blank', 'donor'):
                for prompt in PROMPTS:
                    expected[state + '_' + control, None, prompt] = (state, gold, None)
    else:
        for sequence in plan:
            for step in sequence['steps']:
                for prompt in PROMPTS:
                    expected[sequence['sequence'], sequence['repetition'], step['step'], prompt] = step
    image_passes = {}
    for row in rows:
        if kind == 'confirmation':
            key = row['case'], row['noise_seed'], row['prompt_id']
            require(key in expected and key not in seen, 'Unexpected/duplicate confirmation row')
            require((row['state'], row['gold'], row['event_text']) == expected[key], 'Confirmation condition changed')
            category = row['case']
            cell_key = category + '/' + row['prompt_id']
            image_key = key[:-1]
        else:
            key = row['sequence'], row['repetition'], row['step'], row['prompt_id']
            require(key in expected and key not in seen, 'Unexpected/duplicate chain row')
            require(all(row[k] == v for k, v in expected[key].items()), 'Chain operation differs from plan')
            label = f"sequence-{row['sequence']}-rep-{row['repetition']}"
            prior = None if row['step'] == 0 else label + f"-step-{row['step'] - 1}.png"
            require(row['source_artifact'] == prior, 'Chain was reset or its source is mislinked')
            require(row['image_artifact'] == label + f"-step-{row['step']}.png", 'Unexpected generated image')
            require(sha(run / row['image_artifact']) == row['image_file_sha256'], 'PNG differs from raw row')
            category = ('initial_write' if row['step'] == 0 else row['operation']) + '/' + row['expected_state']
            cell_key = f"sequence-{row['sequence']}/step-{row['step']}/{row['prompt_id']}"
            image_key = key[:-1]
        seen.add(key)
        score = row['scorer']
        passed = bool(score['strict_correct'] and score['answer_followed_immediately_by_eos'])
        exact_tokens = row['generated_token_ids'] == score['gold_token_ids'] + [151645]
        require(passed == exact_tokens, 'Raw tokens disagree with strict answer/EOS score')
        group = groups.setdefault(category, {'n': 0, 'correct_eos': 0, 'raw': Counter()})
        group['n'] += 1
        group['correct_eos'] += int(passed)
        group['raw'][row['raw']] += 1
        count_key = 'answer_eos' if kind == 'confirmation' else 'correct_eos'
        cell = cells.setdefault(cell_key, {'n': 0, count_key: 0})
        cell['n'] += 1
        cell[count_key] += int(passed)
        if kind == 'confirmation':
            cell.setdefault('exact_match', 0)
            cell['exact_match'] += int(score['strict_correct'])
        image_passes.setdefault(image_key, []).append(passed)
    require(seen == set(expected), 'Missing rows from the registered plan')
    require(cells == complete['cells'], 'Sealed cells disagree with raw recomputation')
    matched = {k: v for k, v in groups.items() if not k.endswith(('_blank', '_donor'))}
    all_pass = all(v['correct_eos'] == v['n'] for v in matched.values())
    require(complete['all_generated_answer_eos' if kind == 'confirmation' else 'all_correct_eos'] == all_pass,
            'Sealed aggregate disagrees with raw recomputation')
    matched_images = {k: v for k, v in image_passes.items()
                      if kind == 'chains' or not k[0].endswith(('_blank', '_donor'))}
    summary = {'kind': kind, 'identity': identity, 'complete_sha256': sha(run / 'complete.json'),
               'preregistered_plan_sha256': sha(plan_path), 'verified_files': verified,
               'pt_files_omitted_locally': remote_only, 'all_remote_files_verified_here': not remote_only,
               'raw_rows': len(rows), 'groups': groups, 'matched_rows': sum(v['n'] for v in matched.values()),
               'matched_correct_eos': sum(v['correct_eos'] for v in matched.values()),
               'all_five_prompt_images': sum(len(v) == 5 and all(v) for v in matched_images.values()),
               'generated_images': len(matched_images), 'all_correct_eos': all_pass}
    if kind == 'chains':
        chain_passes = {}
        for key, values in image_passes.items():
            chain_passes.setdefault(key[:2], []).extend(values)
        summary['complete_chains_correct_eos'] = sum(len(v) == 30 and all(v) for v in chain_passes.values())
        summary['chain_count'] = len(chain_passes)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--kind', choices=('confirmation', 'chains'), required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--text-only', action='store_true', help='Explicitly omit absent remote PT files from local verification')
    a = p.parse_args()
    summary = collect(a.run, a.plan, a.kind, text_only=a.text_only)
    out = Path(str(a.output_prefix) + '-summary.json')
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    if not a.text_only:
        with tarfile.open(str(a.output_prefix) + '-evidence.tgz', 'w:gz') as archive:
            for path in sorted(a.run.iterdir()):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.png'):
                    archive.add(path, arcname=path.name)
            archive.add(out, arcname='verified-summary.json')
            archive.add(a.plan, arcname='preregistered-plan.json')
            log = Path(str(a.run) + '.log')
            if log.exists():
                archive.add(log, arcname='worker.log')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('identity', 'verified_files')}))


if __name__ == '__main__':
    main()
