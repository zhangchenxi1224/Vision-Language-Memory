"""Audit completed round3 against the unchanged C8; write only a new analysis directory."""
import hashlib
import json
import random
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-multitarget-20260927')
OLD = ROOT/'round2/recovery-20260928-0045'
NEW = ROOT/'round3/pilot64'
OUT = ROOT/'round3/analysis-20260928'
WEIGHTS = {
    'C8': ('round2/pilot64/C8/train/checkpoint-final.pt', 'efd7b4a22a40508e9ae5a9d7ca72305f5a1c8b3130b3e22c5e2b72aa7bb36978'),
    'C8R': ('round3/pilot64/C8R/train/checkpoint-final.pt', '8a7092d98a44b2f05d60b310020bedd00af927f0e71f59bb0f8119c0df02b00d'),
    'U16': ('round3/pilot64/U16/train/checkpoint-final.pt', '7598446d8919efd31a81301036a4b0b904643f175cf6552b04bafe355f9148f5'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def quantile(values, p):
    x = sorted(values)
    at = (len(x)-1)*p
    lo = int(at)
    return x[lo] + (x[min(lo+1, len(x)-1)]-x[lo])*(at-lo)


def main():
    ids = json.loads((ROOT/'pilot64/ids.json').read_text())['ids']
    assert len(ids) == len(set(ids)) == 64
    for base in [OLD, NEW]:
        assert json.loads((base/'complete.json').read_text())['results_sha256'] == sha(base/'results.json')
    original = json.loads((NEW/'results.json').read_text())
    result = {'scope': 'Previously trained train64, V0/V1 and all eight paired seeds; no best-of-8.',
              'controller': json.loads((NEW/'controller.json').read_text()),
              'source_results_sha256': sha(NEW/'results.json'),
              'audit_script_sha256': sha(Path(__file__)),
              'arms': {}, 'paired_bootstrap': {}, 'verified_png_sha256': {}}
    baselines = {}
    donors = {}
    for arm in WEIGHTS:
        checkpoint, expected = WEIGHTS[arm]
        assert sha(ROOT/checkpoint) == expected
        base = OLD if arm == 'C8' else NEW
        totals = defaultdict(lambda: [0, 0])
        per = defaultdict(lambda: defaultdict(list))
        images = defaultdict(dict)
        hashes, failures = {}, []
        parse = 0
        for v in range(2):
            seen = set()
            for shard in range(4):
                path = base/arm/f'read-V{v}/readback-{shard}.jsonl'
                hashes[str(path)] = sha(path)
                lines = path.read_text().splitlines()
                assert len(lines) == 864
                for line in lines:
                    r = json.loads(line)
                    key = tuple(r[k] for k in ['pair_id', 'chain', 'prefix', 'control', 'family', 'task'])
                    assert key not in seen and r['pair_id'] in ids and r['prefix'] == 0 and r['task'] == 'mcq'
                    assert r['family'] in ['T1','T2','T3'] and r['split'] == 'train'
                    assert r['correct'] == (r['predicted_letter'] == r['correct_letter'])
                    seen.add(key)
                    name = r['control']+'/'+r['family']
                    totals[name][0] += int(r['correct'])
                    totals[name][1] += 1
                    per[r['pair_id']][name].append(int(r['correct']))
                    parse += int(r['parse_failure'])
                    if r['control'] in ['memory','mismatch']:
                        assert 0 <= r['chain'] < 8
                        png = Path(r['png_path'])
                        if str(png) not in result['verified_png_sha256']:
                            actual = sha(png)
                            complete = json.loads((png.parent/'complete.json').read_text())
                            assert complete['png_hashes'][png.name] == actual
                            assert complete['binding']['checkpoint_sha256'] == expected
                            assert complete['binding']['steps'] == 28 and complete['binding']['cfg'] == 1
                            assert complete['binding']['noise_domain'] == 'mt8-eval'
                            assert complete['binding']['initial_variant'] == v
                            result['verified_png_sha256'][str(png)] = actual
                        assert result['verified_png_sha256'][str(png)] == r['png_sha256']
                        source = r['pair_id'] if r['control'] == 'memory' else r['donor_pair_id']
                        assert png.parent.name == f"seed-{r['chain']}"
                        assert png.parent.parent.name == source.replace(':','_')
                        if r['control'] == 'mismatch':
                            assert source != r['pair_id'] and source.split(':')[0] == r['pair_id'].split(':')[0]
                            dk = (v, key)
                            if arm == 'C8': donors[dk] = source
                            else: assert donors[dk] == source
                    else:
                        assert r['control'] in ['blank','text'] and r['chain'] == 0 and r['png_path'] is None
                        bk = (v, key)
                        val = {k:r[k] for k in ['predicted_letter','correct_letter','reader_query','preference']}
                        if arm == 'C8': baselines[bk] = val
                        else: assert baselines[bk] == val
                    if r['control'] == 'memory':
                        images[r['pair_id'],v,r['chain']][r['family']] = bool(r['correct'])
                        if not r['correct']:
                            failures.append({k:r[k] for k in ['pair_id','chain','family','correct_letter','predicted_letter','parse_failure','png_sha256']} | {'variant':v})
            expected_keys = {(pid,c,0,control,f,'mcq') for pid in ids
                             for control in ['memory','mismatch','blank','text']
                             for c in (range(8) if control in ['memory','mismatch'] else [0])
                             for f in ['T1','T2','T3']}
            assert seen == expected_keys
        assert len(images) == 1024 and all(set(x) == {'T1','T2','T3'} for x in images.values())
        summaries = {}
        for pid in ids:
            p = {k:sum(x)/len(x) for k,x in per[pid].items()}
            p['all_three'] = sum(all(images[pid,v,c].values()) for v in range(2) for c in range(8))/16
            p['T1_match_gain'] = p['memory/T1']-p['mismatch/T1']
            summaries[pid] = p
        all3 = sum(all(x.values()) for x in images.values())
        if arm != 'C8':
            assert original[arm]['all_three_forms_correct'] == all3
            assert original[arm]['readback_sha256'] == hashes
        result['arms'][arm] = {'correct_total':dict(totals),'all_three_correct':all3,'images':1024,
                              'unique_readbacks':6912,'parse_failures':parse,'per_preference':summaries,
                              'all16_T1_correct_preferences':sum(x['memory/T1']==1 for x in summaries.values()),
                              'all16_three_correct_preferences':sum(x['all_three']==1 for x in summaries.values()),
                              'memory_failures':failures,'readback_sha256':hashes,'checkpoint_sha256':expected}
    assert len(result['verified_png_sha256']) == 3072
    for left,right in [('C8R','C8'),('U16','C8R'),('U16','C8')]:
        for metric in ['memory/T1','all_three','T1_match_gain']:
            diff = [result['arms'][left]['per_preference'][p][metric]-result['arms'][right]['per_preference'][p][metric] for p in ids]
            rng = random.Random(20260928)
            samples = [sum(rng.choices(diff,k=64))/64 for _ in range(10000)]
            result['paired_bootstrap'][f'{left}-{right}/{metric}'] = {
                'difference_pp':100*statistics.mean(diff),
                'ci95_pp':[100*quantile(samples,.025),100*quantile(samples,.975)],
                'unit':'preference','replicates':10000,'seed':20260928,
                'better_equal_worse':[sum(x>0 for x in diff),sum(x==0 for x in diff),sum(x<0 for x in diff)]}
    result['interpretation'] = {
        'refresh': 'No added benefit observed for U16 over equal-exposure C8R; retain the non-positive result. Small clustered errors do not establish a general harm claim.',
        'exposure': 'C8R and U16 each add 2048 FM steps; comparison to original C8 includes this extra exposure. Only U16 vs C8R isolates this refresh-and-replay intervention at equal added budget.',
        'next': 'Run fixed original C8 and B0 on all730 as user requested, in parallel with the unchanged long10 evaluation. Do not select a different model from this ranking.',
        'limits': 'Train64 learnability only; all730 previously trained by B0. No new-preference generalization, semantic-mode count, or retention-training claim.'}
    OUT.mkdir(parents=True,exist_ok=True)
    dest = OUT/'analysis_results.json'
    dest.write_text(json.dumps(result,indent=2)+'\n')
    files = set(NEW.glob('*.json'))
    for arm in ['C8R','U16']:
        for sub in ['train','eval-V0','eval-V1','read-V0','read-V1']:
            files.update(p for p in (NEW/arm/sub).rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log'])
    files.update(NEW.glob('*.log'))
    files.update([ROOT/'round3/launch.json',ROOT/'round3-pipeline.log',dest])
    evidence = {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}
    (OUT/'source-evidence-sha256.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (OUT/'audit-source.py').write_bytes(Path(__file__).read_bytes())
    files.update([OUT/'source-evidence-sha256.json',OUT/'audit-source.py'])
    archive = ROOT/'round3-final-evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files): tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    receipt = {'path':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files)}
    (OUT/'evidence_archive.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'arms':{a:{k:v for k,v in x.items() if k not in ['per_preference','readback_sha256']} for a,x in result['arms'].items()},'paired_bootstrap':result['paired_bootstrap'],'archive':receipt}))


if __name__ == '__main__':
    main()

