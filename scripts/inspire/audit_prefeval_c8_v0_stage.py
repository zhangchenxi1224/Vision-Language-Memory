"""Read-only audit of the completed, preplanned C8/V0 slice; not the final C8/B0 result."""
import hashlib
import json
import sys
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
SOURCE = TASK/'eval730/recovery-20260928-1920'
OUTPUT = TASK/'eval730/analysis-c8-v0-20260929-1225'
FROZEN = PROJECT/'repos/prefeval-multitarget-round2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not OUTPUT.exists(), 'Use a new independent analysis directory'
    assert sha(SOURCE/'protocol.json') == 'c670e3db2d7d54bf449999cf693d7544fb54bbc7040420b641c31484e6f641dc'
    p = json.loads((SOURCE/'protocol.json').read_text())
    assert p['noise_chains'] == 8 and p['inter_turns'] == 0 and p['teacher_updates'] == p['fm_updates'] == 0
    assert sha(p['checkpoints']['C8']['path']) == p['checkpoints']['C8']['sha256']
    for path, expected in p['data_sha256'].items():
        assert sha(path) == expected
    sys.path[:0] = [str(FROZEN), str(FROZEN/'src')]
    from scripts.experiments.prefeval_k1_data import load_records, event_text
    from scripts.experiments.prefeval_k1_variants import load_variants, apply_variant
    from scripts.experiments.prefeval_k1_source_bank import noise_namespace
    from vision_memory.training.latent_bank_unet import stable_seed
    rows = load_records('train')
    variants_path = PROJECT/'runs/prefeval-b-mcq-20260925/variants-train.json'
    variants = load_variants(variants_path, rows)
    rowmap = {r['base_pair_id']: apply_variant(r, variants, 0) for r in rows}
    ids, pilot = p['ids'], set(p['pilot64_ids'])
    assert len(ids) == len(set(ids)) == len(rowmap) == 730 and len(pilot) == 64
    binding = {'checkpoint_sha256': p['checkpoints']['C8']['sha256'], 'split': 'train',
               'steps': 28, 'cfg': 1, 'noise_chains': 8, 'inter_turns': 0,
               'state': 'only reopened uint8 RGB PNG; fresh Gaussian each write',
               'noise_domain': 'mt8-eval', 'initial_variants_sha256': sha(variants_path), 'initial_variant': 0}
    png_hashes, evidence = {}, [SOURCE/'protocol.json', SOURCE/'ids.json']
    for pid in ids:
        for chain in range(8):
            folder = SOURCE/'C8/eval-V0'/pid.replace(':', '_')/f'seed-{chain}'
            donepath, writespath, png = folder/'complete.json', folder/'writes.jsonl', folder/'prefix-00.png'
            done = json.loads(donepath.read_text())
            assert done['binding'] == binding and set(done['png_hashes']) == {'prefix-00.png'}
            actual = sha(png)
            assert actual == done['png_hashes']['prefix-00.png']
            writes = [json.loads(line) for line in writespath.read_text().splitlines()]
            assert len(writes) == 1
            w = writes[0]
            assert w['position'] == 0 and w['source_png_sha256'] is None and w['output_png_sha256'] == actual
            assert w['noise_seed'] == stable_seed(20260924, noise_namespace(pid, chain, 'mt8-eval'), 0)
            assert w['event'] == event_text(rowmap[pid]['history'][:2])
            png_hashes[str(png)] = actual
            evidence += [donepath, writespath]
    per_pref = {pid: {'counts': defaultdict(lambda: [0, 0]), 'all_three': [0, 8]} for pid in ids}
    seen, images, finished, counts, readhash = set(), defaultdict(dict), {}, {}, {}
    parse_failures = 0
    for shard in range(4):
        read = SOURCE/'C8/read-V0'/f'readback-{shard}.jsonl'
        done = read.with_name(f'finished-{shard}.json')
        finished[str(shard)] = json.loads(done.read_text())
        readhash[str(read)] = sha(read)
        count = 0
        for line in read.read_text().splitlines():
            r = json.loads(line)
            pid, chain, control, family = (r[k] for k in ['pair_id', 'chain', 'control', 'family'])
            key = (pid, chain, r['prefix'], control, family, r['task'])
            assert key not in seen and pid in ids[shard::4]
            assert r['prefix'] == 0 and r['task'] == 'mcq' and family in p['families'] and control in p['controls']
            assert chain in (range(8) if control in ['memory', 'mismatch'] else [0])
            assert r['split'] == 'train' and r['endpoint_kind'] == 'student'
            assert r['correct'] == (r['predicted_letter'] == r['correct_letter'])
            if control in ['memory', 'mismatch']:
                donor = pid if control == 'memory' else p['donors'][pid]
                if control == 'mismatch':
                    assert r['donor_pair_id'] == donor
                expected_path = SOURCE/'C8/eval-V0'/donor.replace(':', '_')/f'seed-{chain}'/'prefix-00.png'
                assert r['png_path'] == str(expected_path)
                assert r['png_sha256'] == png_hashes[str(expected_path)]
            seen.add(key)
            metric = per_pref[pid]['counts'][f'{control}/{family}']
            metric[0] += int(r['correct'])
            metric[1] += 1
            parse_failures += int(r['parse_failure'])
            if control == 'memory':
                images[pid, chain][family] = bool(r['correct'])
            count += 1
        assert count == len(ids[shard::4])*54
        counts[str(shard)] = count
        assert sha(read) == readhash[str(read)], 'Completed readback changed during audit'
        evidence += [read, done]
    expected = {(pid, chain, 0, control, family, 'mcq') for pid in ids for control in p['controls']
                for chain in (range(8) if control in ['memory', 'mismatch'] else [0]) for family in p['families']}
    assert seen == expected and len(seen) == 39420 and len(images) == len(png_hashes) == 5840
    for (pid, chain), values in images.items():
        assert set(values) == set(p['families'])
        per_pref[pid]['all_three'][0] += int(all(values.values()))
    scopes = [('all730', ids), ('adapted64', [i for i in ids if i in pilot]), ('other666', [i for i in ids if i not in pilot])]
    scopes += [(f'topic/{topic}', [i for i in ids if i.split(':')[0] == topic]) for topic in p['topics']]
    strata = {}
    for label, subset in scopes:
        totals = {key: [sum(per_pref[i]['counts'][key][j] for i in subset) for j in range(2)]
                  for key in per_pref[ids[0]]['counts']}
        all_three = [sum(per_pref[i]['all_three'][j] for i in subset) for j in range(2)]
        strata[label] = {'preferences': len(subset), 'counts': totals, 'all_three': all_three,
                         'all8_T1_correct_preferences': sum(per_pref[i]['counts']['memory/T1'][0] == 8 for i in subset),
                         'all8_three_correct_preferences': sum(per_pref[i]['all_three'][0] == 8 for i in subset),
                         'T1_matching_gain_pp': 100*(totals['memory/T1'][0]-totals['mismatch/T1'][0])/totals['memory/T1'][1]}
    OUTPUT.mkdir()
    report = {'status': 'completed_preplanned_C8_V0_slice_only_not_final_comparison',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
              'scope': 'All train730, V0, eight prespecified seeds; V1 and B0 not yet included. No model selection or stopping based on these results.',
              'limitations': ['All 730 were in B0 training; other666 were not in C8 additional adaptation.',
                              'No C8 minus B0 conclusion until paired B0 and V1 finish.',
                              'Donors drawn from complete730; pilot64 mismatch gains are not directly comparable.'],
              'checkpoint': p['checkpoints']['C8'], 'protocol_sha256': sha(SOURCE/'protocol.json'),
              'readback_sha256': readhash, 'readback_counts': counts, 'finished_receipts': finished,
              'verified_png_sha256': png_hashes, 'parse_failures': parse_failures,
              'strata': strata, 'per_preference': per_pref, 'audit_script_sha256': sha(Path(__file__))}
    (OUTPUT/'C8_V0_STAGE_RESULTS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    evidence += [OUTPUT/'C8_V0_STAGE_RESULTS.json', Path(__file__)]
    filehashes = {str(f.relative_to(TASK)): sha(f) for f in evidence}
    (OUTPUT/'evidence-manifest.json').write_text(json.dumps(filehashes, indent=2)+'\n')
    archive = TASK/'c8-v0-stage-evidence-20260929.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for f in evidence + [OUTPUT/'evidence-manifest.json']:
            tar.add(f, arcname=str(f.relative_to(TASK)), recursive=False)
    receipt = {'path': str(archive), 'sha256': sha(archive), 'files': len(evidence)+1,
               'png_policy': 'All 5840 actual PNGs rehashed; archive retains their hashes and original complete/writes metadata, PNG bytes remain on shared storage.'}
    (OUTPUT/'archive.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'strata': {k: strata[k] for k in ['all730', 'adapted64', 'other666']},
                      'parse_failures': parse_failures, 'readback_counts': counts, 'archive': receipt}, ensure_ascii=False))


if __name__ == '__main__':
    main()
