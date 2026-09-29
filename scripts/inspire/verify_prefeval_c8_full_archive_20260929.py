"""Independently recount the C8 stage from archived raw reads, without GPU access."""
import hashlib
import json
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    report_dir = Path(sys.argv[1])
    receipt = json.loads((report_dir/'C8_FULL_STAGE_ARCHIVE_20260929.json').read_text())
    archive = report_dir/'c8-full-stage-evidence-20260929.tar.gz'
    assert sha(archive.read_bytes()) == receipt['sha256']
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len({m.name for m in members}) == receipt['files']
        files = {m.name: tar.extractfile(m).read() for m in members}
    analysis = 'eval730/analysis-c8-full-20260929-1725'
    manifest = json.loads(files[analysis+'/evidence-manifest.json'])
    assert set(manifest) == set(files)-{analysis+'/evidence-manifest.json'}
    for name, expected in manifest.items():
        assert sha(files[name]) == expected, name
    report_bytes = files[analysis+'/C8_FULL_STAGE_RESULTS.json']
    assert report_bytes == (report_dir/'C8_FULL_STAGE_RESULTS_20260929.json').read_bytes()
    report = json.loads(report_bytes)
    source = 'eval730/recovery-20260929-1325'
    protocol = json.loads(files[source+'/protocol.json'])
    ids, pilot = protocol['ids'], set(protocol['pilot64_ids'])
    counts = defaultdict(lambda: [0, 0])
    images = defaultdict(dict)
    keys, readcounts, failures = set(), Counter(), 0
    for v in range(2):
        for s in range(4):
            name = f'{source}/C8/read-V{v}/readback-{s}.jsonl'
            for raw in files[name].splitlines():
                r = json.loads(raw)
                pid, chain, control, family = [r[k] for k in ['pair_id', 'chain', 'control', 'family']]
                key = (v, pid, chain, r['prefix'], control, family, r['task'])
                assert key not in keys and pid in ids[s::4]
                keys.add(key)
                readcounts[f'V{v}/{s}'] += 1
                assert r['correct'] == (r['correct_letter'] == r['predicted_letter'])
                counts[pid, v, control, family][0] += int(r['correct'])
                counts[pid, v, control, family][1] += 1
                failures += int(r['parse_failure'])
                if control == 'memory':
                    images[pid, v, chain][family] = bool(r['correct'])
    expected = {(v, pid, c, 0, control, family, 'mcq') for v in range(2) for pid in ids
                for control in protocol['controls'] for c in (range(8) if control in ['memory', 'mismatch'] else [0])
                for family in protocol['families']}
    assert keys == expected and len(keys) == 78840 and len(images) == 11680
    assert dict(readcounts) == report['readback_counts'] and failures == report['parse_failures']
    all3 = Counter()
    for (pid, v, c), values in images.items():
        assert set(values) == {'T1', 'T2', 'T3'}
        all3[pid, v] += int(all(values.values()))
    for pid in ids:
        entry = report['per_preference'][pid]
        for v in range(2):
            for ctrl in protocol['controls']:
                for family in protocol['families']:
                    assert entry['counts'][f'V{v}/{ctrl}/{family}'] == counts[pid, v, ctrl, family]
            assert entry['all_three_by_variant'][f'V{v}'] == [all3[pid, v], 8]
        assert entry['all_three'] == [all3[pid, 0]+all3[pid, 1], 16]
    for scope, reported in report['strata'].items():
        subset = ids if scope == 'all730' else [i for i in ids if ((i in pilot) if scope == 'adapted64' else (i not in pilot))] if not scope.startswith('topic/') else [i for i in ids if i.split(':')[0] == scope.split('/')[1]]
        assert len(subset) == reported['preferences']
        for ctrl in protocol['controls']:
            for family in protocol['families']:
                for v in range(2):
                    total = [sum(counts[i, v, ctrl, family][j] for i in subset) for j in range(2)]
                    assert total == reported['by_variant'][f'V{v}']['counts'][f'{ctrl}/{family}']
                total = [sum(counts[i, v, ctrl, family][j] for i in subset for v in range(2)) for j in range(2)]
                assert total == reported['counts'][f'{ctrl}/{family}']
        assert reported['all_three'] == [sum(all3[i, v] for i in subset for v in range(2)), len(subset)*16]
        assert reported['all16_T1_correct_preferences'] == sum(sum(counts[i, v, 'memory', 'T1'][0] for v in range(2)) == 16 for i in subset)
        assert reported['all16_three_correct_preferences'] == sum(all3[i, 0]+all3[i, 1] == 16 for i in subset)
        gain = sum(counts[i, v, 'memory', 'T1'][0]-counts[i, v, 'mismatch', 'T1'][0] for i in subset for v in range(2))*100/(len(subset)*16)
        assert abs(gain-reported['T1_matching_gain_pp']) < 1e-10
    verification = {'utc': datetime.now(timezone.utc).isoformat(), 'archive_sha256': receipt['sha256'],
        'archive_files_checked': len(files), 'unique_readbacks_recounted': len(keys), 'images_recounted': len(images),
        'all_scopes_variants_preferences_controls_families_match': True,
        'same_image_all_three_and_all16_stability_match': True, 'parse_failures': failures,
        'scope': 'Independent local raw-read recount and full archive hash verification; PNG bytes rehashed by remote audit.',
        'script_sha256': sha(Path(__file__).read_bytes())}
    (report_dir/'C8_FULL_STAGE_LOCAL_VERIFICATION_20260929.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
