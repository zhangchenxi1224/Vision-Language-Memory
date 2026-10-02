"""Describe complete sealed training logs; online loss is not a functional metric."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from vision_memory.training.latent_bank_unet import logical_condition_strata


def diagnose(archive, expected_sha):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Archive differs from independently verified endpoint evidence')
    with tarfile.open(archive) as source:
        bank = json.load(source.extractfile('bank/manifest.json'))
        records = [json.loads(line) for line in source.extractfile('train/training.jsonl')]
        identity = json.load(source.extractfile('train/identity.json'))
    groups = bank['groups']
    strata = logical_condition_strata(groups)
    membership = {groups[index]['question_id']: key for key, indices in strata.items() for index in indices}
    if len(records) != 4832 or len(strata) != 31:
        raise ValueError('This diagnostic requires the complete two registered4832-step runs')
    buckets = defaultdict(list)
    clips = defaultdict(list)
    draws = 0
    for step, row in enumerate(records, 1):
        if row['optimizer_step'] != step or len(row['microbatches']) != 4:
            raise ValueError('Missing or changed global4 optimization row')
        quarter = (step - 1) // 1208
        values = row['microbatches']
        if not math.isclose(statistics.mean(v['flow_matching_mse'] for v in values), row['flow_matching_mse'], rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError('Global mean differs from recorded per-draw losses')
        clips[quarter].append(row['unet_grad_norm_before_clip'])
        for value in values:
            sigma, loss = value['effective_sigma'], value['flow_matching_mse']
            if not 0 <= sigma < 1 or not math.isfinite(loss) or loss < 0:
                raise ValueError('Invalid official training draw or loss')
            key = membership[value['question_id']]
            for sigma_bin in ('all', str(min(3, int(sigma * 4)))):
                buckets[(key, quarter, sigma_bin)].append(loss)
            draws += 1
    def stats(values):
        return {'draws': len(values), 'mean': statistics.mean(values), 'median': statistics.median(values), 'maximum': max(values)}
    return {'archive_sha256': expected_sha, 'sampling_strategy': identity.get('sampling', {}).get('strategy', 'condition'),
        'optimizer_steps': len(records), 'draws': draws,
        'sigma_bins': {'0': '[0,.25)', '1': '[.25,.5)', '2': '[.5,.75)', '3': '[.75,1)', 'all': '[0,1)'},
        'quarter_steps': 1208,
        'strata': {key: {'question_ids': [groups[index]['question_id'] for index in strata[key]],
            'quarters': {str(q): {b: stats(buckets[(key, q, b)]) for b in ('all','0','1','2','3') if buckets[(key,q,b)]}
                         for q in range(4)}} for key in strata},
        'gradient_clipping_by_quarter': {str(q): {'updates':len(v), 'norm_above_one':sum(x>1 for x in v),
            'mean_norm':statistics.mean(v), 'max_norm':max(v)} for q,v in clips.items()},
        'scope': 'Descriptive online losses at changing model parameters and different condition/time assignments. These are not fixed-model held-out FM measurements, causal gradient-conflict estimates, or Reader accuracy. All draws retained.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = diagnose(a.archive, a.sha256)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('archive_sha256', 'optimizer_steps', 'draws', 'gradient_clipping_by_quarter')}))
