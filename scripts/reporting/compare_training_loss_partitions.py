"""Describe all sealed training draws by operation and sigma; not a functional score."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

RUNS = {
    '03': ('9e27050-logical-endpoint-evidence.tgz', '68c8c7f5c0b19703cd1b1b555f9a4d0dfd193e532534a1eb0c046cf80e5c04a4'),
    'b9': ('7b82309-logical-endpoint-evidence.tgz', '5d6e92f5154241b5a03b42e3a267540c4386ceacf84b196f4e64fb7895eb228d'),
    '4f': ('e372f3c-logical-endpoint-evidence.tgz', '3177f04d5cd9be40c2d80c72f4a84ab1ad2f89b80bb22893224a5257f7bbd7b2'),
}
BANK_SHA = 'c27cd65dab809deabb5f2cb08891517d3590244651d08a8c6763c84fea901592'


def describe(values):
    ordered = sorted(values)
    return {'n': len(values), 'mean': statistics.fmean(values), 'median': statistics.median(values),
        'min': ordered[0], 'max': ordered[-1]}


def analyze(root):
    bank_data = (root / 'broader151-bank-manifest.json').read_bytes()
    if hashlib.sha256(bank_data).hexdigest() != BANK_SHA:
        raise ValueError('Changed bank')
    bank = json.loads(bank_data)
    groups = {g['question_id']: g for g in bank['groups']}
    result, draw_reference = {}, None
    for label, (name, digest) in RUNS.items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Archive differs from independently verified source: ' + name)
        with tarfile.open(path) as archive:
            rows = [json.loads(line) for line in archive.extractfile('train/training.jsonl')]
        if len(rows) != 4832:
            raise ValueError('Incomplete fixed training budget')
        losses = defaultdict(list)
        by_sigma = defaultdict(lambda: defaultdict(list))
        by_quarter = defaultdict(lambda: defaultdict(list))
        counts, high_sigma, draws = Counter(), Counter(), []
        for step, row in enumerate(rows, 1):
            if row['optimizer_step'] != step or len(row['microbatches']) != 4:
                raise ValueError('Missing/reordered optimizer step or microbatch')
            actual_mean = statistics.fmean(m['flow_matching_mse'] for m in row['microbatches'])
            if not math.isclose(actual_mean, row['flow_matching_mse'], rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError('Recorded global mean differs from actual four microbatches')
            for micro in row['microbatches']:
                group = groups[micro['question_id']]
                sigma, loss = micro['effective_sigma'], micro['flow_matching_mse']
                if not 0 <= sigma < 1 or not math.isfinite(loss) or loss < 0:
                    raise ValueError('Invalid actual sigma or loss')
                partition = ('historical_clear' if group['answer'] == 'no active preference' else 'historical_value') if 'historical_target_index' in group else 'music_' + group['operation']
                counts[partition] += 1
                high_sigma[partition] += sigma > .5
                losses[partition].append(loss)
                by_sigma[partition][str(min(int(sigma * 4), 3))].append(loss)
                by_quarter[partition][str((step - 1) // 1208)].append(loss)
                draws.append((micro['question_id'], micro['teacher_id'], micro['noise_seed'], sigma))
        if len(draws) != 19328:
            raise ValueError('Incomplete actual draw stream')
        if draw_reference is None:
            draw_reference = draws
        elif draws != draw_reference:
            raise ValueError('Source/teacher/Gaussian/sigma streams differ across experiments')
        result[label] = {'archive': name, 'sha256': digest, 'steps': len(rows), 'draws': len(draws),
            'partitions': {key: {'loss': describe(values), 'sigma_above_half': high_sigma[key],
                'sigma_quarters': {q: describe(v) for q,v in by_sigma[key].items()},
                'update_quarters': {q: describe(v) for q,v in by_quarter[key].items()}} for key,values in losses.items()}}
    return {'runs': result, 'same_actual_condition_teacher_noise_sigma_stream': True,
        'sigma_quarters': {'0': '[0,.25)', '1': '[.25,.5)', '2': '[.5,.75)', '3': '[.75,1)'},
        'update_quarters': 'Four consecutive 1208-step quarters; different draws within each quarter, not a fixed held-out loss curve.',
        'scope': 'Descriptive post-hoc analysis of all 19328 actual training microbatches per run. MSE is a velocity-space fit metric, not a semantic or functional success score. Historical text augmentation changes between 03 and b9; initialization and learning rate change for 4f. No single-factor causal attribution.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.root)
    args.output.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({label: {part: {'draws': item['loss']['n'], 'mean_mse': item['loss']['mean'],
        'sigma_above_half': item['sigma_above_half'], 'first_quarter_mse': item['update_quarters']['0']['mean'],
        'last_quarter_mse': item['update_quarters']['3']['mean']} for part,item in run['partitions'].items()}
        for label,run in value['runs'].items()}, ensure_ascii=False))
