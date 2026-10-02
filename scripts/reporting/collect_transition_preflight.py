"""Seal the completed45-condition baseline and source/teacher preflight during training."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', type=Path, required=True)
p.add_argument('--bank', type=Path, required=True)
p.add_argument('--output-prefix', type=Path, required=True)
a = p.parse_args()
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
if sha(a.bank) != '962f02846ed1a1933e6c219604bc22ee520e28f2dfe2721e26f111dc36ea122e':
    raise ValueError('Unexpected45-condition bank')
bank = json.loads(a.bank.read_text())
groups = {g['question_id']: g for g in bank['groups']}
runtime = json.loads((a.run / 'train/runtime.json').read_text())
identity = json.loads((a.run / 'train/identity.json').read_text())
if (len(groups) != 45 or identity['steps'] != 2880 or identity['eval_seeds'] != 4
        or identity['bank_manifest_sha256'] != sha(a.bank) or identity['trainable_scope'] != 'full_unet'):
    raise ValueError('Wrong training design')
sources = runtime['additional_protocol_binding']['source_bindings']
if set(sources) != set(groups):
    raise ValueError('Missing source bindings')
for qid, group in groups.items():
    if group['source_kind'] == 'sealed_rgb_1024':
        if (sources[qid]['rms_difference'] != 0 or sources[qid]['official_source_sha256'] != group['source_latent_sha256']
                or sources[qid]['source_image_file_sha256'] != group['source_image_file_sha256']):
            raise ValueError('Non-gray source differs from the verified PNG encoding')
teacher = json.loads((a.run / 'train/teacher-readback.json').read_text())['rows']
if len(teacher) != 45 or len({r['question_id'] for r in teacher}) != 45:
    raise ValueError('Missing teacher readbacks')
for row in teacher:
    if not row['scorer']['strict_correct'] or not row['scorer']['answer_followed_immediately_by_eos']:
        raise ValueError('Direct positive control failed')
phase = a.run / 'train/baseline'
complete = json.loads((phase / 'complete.json').read_text())
for name, digest in complete['artifact_hashes'].items():
    if Path(name).name != name or sha(phase / name) != digest:
        raise ValueError('Baseline artifact changed')
rows = [json.loads(line) for line in (phase / 'generations.jsonl').read_text().splitlines()]
if len(rows) != 1350 or len({(r['question_id'], r['condition'], r['noise_seed'], r['prompt_id']) for r in rows}) != 1350:
    raise ValueError('Incomplete or duplicated baseline coverage')
cells = {}
for row in rows:
    group = groups[row['question_id']]
    if row['query'] != group['question_variants'][row['prompt_id']] or row['gold'] != group['answer']:
        raise ValueError('Baseline query/gold changed')
    passed = row['generated_token_ids'] == row['scorer']['gold_token_ids'] + [151645]
    if passed != bool(row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']):
        raise ValueError('Raw tokens disagree with baseline score')
    if row['condition'] == 'matched':
        key = group['source_state'] + '/' + group['operation'] + '/' + group['target_state'] + '/wording-' + str(group['wording_index'])
        cell = cells.setdefault(key, {'n': 0, 'correct_eos': 0})
        cell['n'] += 1
        cell['correct_eos'] += int(passed)
if len(cells) != 45 or any(c['n'] != 20 for c in cells.values()):
    raise ValueError('Missing per-transition matched coverage')
summary = {'bank_sha256': sha(a.bank), 'baseline_complete_sha256': sha(phase / 'complete.json'),
    'phase_files_verified': len(complete['artifact_hashes']), 'teacher_positive_controls': '45/45 original-question exact answer plus immediate EOS',
    'non_gray_source_bindings_equal': sum(g['source_kind'] == 'sealed_rgb_1024' for g in groups.values()),
    'source_groups': dict(Counter(g['source_state'] for g in groups.values())), 'conditional_groups': 45,
    'baseline_matched_rows': 900, 'baseline_correct_eos': sum(c['correct_eos'] for c in cells.values()),
    'baseline_cells': cells, 'identity': identity, 'scope': 'Immutable untrained baseline/source/teacher checks only; ongoing training is not a completed result'}
out = Path(str(a.output_prefix) + '-summary.json')
out.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
with tarfile.open(str(a.output_prefix) + '-evidence.tgz', 'w:gz') as archive:
    for name in ('commands.json', 'dispatch.json', 'train/runtime.json', 'train/identity.json', 'train/teacher-readback.json',
                 'train/baseline/complete.json', 'train/baseline/summary.json', 'train/baseline/generations.jsonl'):
        archive.add(a.run / name, arcname=name)
    archive.add(out, arcname='verified-summary.json')
print(json.dumps({k: v for k, v in summary.items() if k not in ('identity', 'baseline_cells')}))
