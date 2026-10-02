"""Audit completed paired train64 readbacks; no model selection on holdout data."""
import hashlib
import json
import random
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-multitarget-20260927')
OLD = ROOT/'pilot64'
NEW = ROOT/'round2/recovery-20260928-0045'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, p):
    x = sorted(values)
    at = (len(x)-1)*p
    lo = int(at)
    return x[lo] + (x[min(lo+1, len(x)-1)]-x[lo])*(at-lo)


def main():
    assert json.loads((NEW/'complete.json').read_text())['results_sha256'] == sha(NEW/'results.json')
    ids = json.loads((OLD/'ids.json').read_text())['ids']
    result = {'scope': 'Same previously trained 64 preferences; all V0/V1 paired eight seeds, no best-of-8.',
              'controller': json.loads((NEW/'controller.json').read_text()),
              'source_results_sha256': sha(NEW/'results.json'), 'arms': {}, 'paired_bootstrap': {}}
    for arm in ['B0', 'F1', 'F8', 'S8', 'S1', 'C8']:
        base = NEW if arm in ['S1', 'C8'] else OLD
        totals, per, images, hashes = defaultdict(lambda: [0, 0]), defaultdict(lambda: defaultdict(list)), defaultdict(list), {}
        parse = 0
        for v in range(2):
            files = sorted((base/arm/f'read-V{v}').glob('readback-*.jsonl'))
            assert len(files) == (4 if base == NEW else 1)
            seen = set()
            for path in files:
                hashes[str(path)] = sha(path)
                for line in path.read_text().splitlines():
                    r = json.loads(line)
                    key = tuple(r[k] for k in ['pair_id', 'chain', 'prefix', 'control', 'family', 'task'])
                    assert key not in seen and r['pair_id'] in ids and r['prefix'] == 0 and r['task'] == 'mcq'
                    seen.add(key)
                    name = r['control']+'/'+r['family']
                    totals[name][0] += int(r['correct']); totals[name][1] += 1
                    per[r['pair_id']][name].append(int(r['correct']))
                    if r['control'] == 'memory':
                        images[r['pair_id'], v, r['chain']].append(int(r['correct']))
                    parse += int(r['parse_failure'])
            assert len(seen) == 3456
        assert len(images) == 1024 and all(len(x) == 3 for x in images.values())
        assert all(v[1] == (1024 if k.split('/')[0] in ['memory','mismatch'] else 128) for k,v in totals.items())
        summaries = {}
        for pid in ids:
            p = {k: sum(v)/len(v) for k,v in per[pid].items()}
            p['all_three'] = sum(all(images[pid, v, c]) for v in range(2) for c in range(8))/16
            p['T1_match_gain'] = p['memory/T1'] - p['mismatch/T1']
            summaries[pid] = p
        result['arms'][arm] = {'correct_total': dict(totals), 'all_three_correct': sum(all(x) for x in images.values()),
            'images': 1024, 'parse_failures': parse, 'per_preference': summaries, 'readback_sha256': hashes,
            'all16_T1_correct_preferences': sum(x['memory/T1'] == 1 for x in summaries.values()),
            'all16_three_correct_preferences': sum(x['all_three'] == 1 for x in summaries.values()),
            'all16_T1_wrong_preferences': sum(x['memory/T1'] == 0 for x in summaries.values())}
    for arm in result['arms']:
        for name in ['blank/T1','blank/T2','blank/T3','text/T1','text/T2','text/T3']:
            assert result['arms'][arm]['correct_total'][name] == result['arms']['B0']['correct_total'][name]
    for left, right in [('S1','S8'),('S1','F1'),('C8','F1'),('C8','F8'),('C8','S1')]:
        for metric in ['memory/T1', 'all_three', 'T1_match_gain']:
            differences = [result['arms'][left]['per_preference'][p][metric] - result['arms'][right]['per_preference'][p][metric] for p in ids]
            rng = random.Random(20260927)
            samples = [sum(rng.choices(differences, k=64))/64 for _ in range(10000)]
            result['paired_bootstrap'][f'{left}-{right}/{metric}'] = {'difference_pp': 100*statistics.mean(differences),
                'ci95_pp': [100*quantile(samples, .025),100*quantile(samples, .975)],
                'unit': 'preference', 'replicates': 10000, 'seed': 20260927}
    (NEW/'analysis_results.json').write_text(json.dumps(result, indent=2)+'\n')
    files = set(p for p in NEW.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log','.py'])
    orig = ROOT/'round2/pilot64'
    files.update(p for p in orig.glob('*.json'))
    for arm in ['S1','C8']:
        files.update(p for p in (orig/arm/'train').glob('*') if p.suffix in ['.json','.jsonl'])
    archive = ROOT/'round2-final-evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in sorted(files): tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
    (NEW/'evidence_archive.json').write_text(json.dumps({'path': str(archive), 'sha256': sha(archive), 'files': len(files), 'bytes': archive.stat().st_size}, indent=2)+'\n')
    print(json.dumps({'arms': {a:{k:v for k,v in x.items() if k not in ['per_preference','readback_sha256']} for a,x in result['arms'].items()}, 'paired_bootstrap':result['paired_bootstrap']}))


if __name__ == '__main__':
    main()
