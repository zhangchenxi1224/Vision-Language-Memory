"""Read-only audit of complete C8 V0/V1, before its paired B0 evaluation finishes."""
import hashlib
import importlib.util
import json
import sys
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
SOURCE = TASK/'eval730/recovery-20260929-1325'
OLD = TASK/'eval730/recovery-20260928-1920'
OUTPUT = TASK/'eval730/analysis-c8-full-20260929-1725'
FROZEN = PROJECT/'repos/prefeval-multitarget-round2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert not OUTPUT.exists(), 'Use a new independent analysis directory'
    assert sha(SOURCE/'protocol.json') == 'c670e3db2d7d54bf449999cf693d7544fb54bbc7040420b641c31484e6f641dc'
    p = json.loads((SOURCE/'protocol.json').read_text())
    assert p['initial_variants'] == [0, 1] and p['noise_chains'] == 8
    assert p['inter_turns'] == p['teacher_updates'] == p['fm_updates'] == 0
    code = PROJECT/'repos/prefeval-c8-eval730-20260928/scripts/inspire/run_prefeval_c8_eval730.py'
    assert sha(code) == 'ab4d702b41115bcfd888d5ef7cb561577eaf8af88b035830395ebbe45be2a8cf'
    spec = importlib.util.spec_from_file_location('frozen_eval730', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.git_head(FROZEN) == '224cc77d790cf3967b5a56ce2e77c364959435a2'
    for checkpoint in p['checkpoints'].values():
        assert sha(checkpoint['path']) == checkpoint['sha256']
    for path, expected in p['data_sha256'].items():
        assert sha(path) == expected
    deps = TASK/'long10/audit-deps-20260929/dependency-manifest.json'
    dependency_manifest = json.loads(deps.read_text())
    sys.path[:0] = [str(deps.parent), str(FROZEN), str(FROZEN/'src')]
    import bs4
    assert bs4.__version__ == dependency_manifest['bs4_version']
    for path, expected in dependency_manifest['copied_sha256'].items():
        assert sha(deps.parent/path) == expected
    from scripts.experiments.prefeval_k1_data import load_records, event_text, official_mcq, option_order
    from scripts.experiments.prefeval_k1_variants import load_variants, apply_variant
    from scripts.experiments.prefeval_k1_source_bank import noise_namespace
    from vision_memory.training.latent_bank_unet import stable_seed
    records = load_records('train')
    variants = load_variants(module.VARIANTS, records)
    rows = {(r['base_pair_id'], v): apply_variant(r, variants, v) for r in records for v in range(2)}
    ids, pilot = p['ids'], set(p['pilot64_ids'])
    assert len(ids) == len(set(ids)) == 730 and len(pilot) == 64
    groups = defaultdict(list)
    for r in records:
        groups[r['topic']].append(r['base_pair_id'])
    assert p['donors'] == {pid: peers[(i+1) % len(peers)] for peers in groups.values() for i, pid in enumerate(peers)}
    recovery_path = SOURCE/'recovery-preparation.json'
    recovery = json.loads(recovery_path.read_text())
    assert recovery['source'] == str(OLD) and recovery['destination'] == str(SOURCE)
    previous_path = TASK/'eval730/analysis-c8-v0-20260929-1225/C8_V0_STAGE_RESULTS.json'
    previous = json.loads(previous_path.read_text())
    evidence = [SOURCE/'protocol.json', SOURCE/'ids.json', recovery_path, previous_path]
    pngs, readhashes, readcounts, receipts = {}, {}, {}, {}
    per_pref = {pid: {'counts': defaultdict(lambda: [0, 0]), 'all_three': [0, 16],
                     'all_three_by_variant': {'V0': [0, 8], 'V1': [0, 8]}} for pid in ids}
    all_images, all_keys = {}, set()
    parse_failures = 0
    mcq = official_mcq(FROZEN/'third_party/prefeval_reference')
    for v in range(2):
        images = SOURCE/'C8'/f'eval-V{v}'
        binding = {'checkpoint_sha256': p['checkpoints']['C8']['sha256'], 'split': 'train',
                   'steps': 28, 'cfg': 1, 'noise_chains': 8, 'inter_turns': 0,
                   'state': 'only reopened uint8 RGB PNG; fresh Gaussian each write',
                   'noise_domain': 'mt8-eval', 'initial_variants_sha256': sha(module.VARIANTS), 'initial_variant': v}
        for filename in ['manifest.json'] + [f'manifest-shard-{s}.json' for s in range(4)]:
            f = images/filename
            assert json.loads(f.read_text()) == binding
            evidence.append(f)
        for pid in ids:
            for c in range(8):
                folder = images/pid.replace(':', '_')/f'seed-{c}'
                donepath, writespath, png = folder/'complete.json', folder/'writes.jsonl', folder/'prefix-00.png'
                done = json.loads(donepath.read_text())
                actual = sha(png)
                assert done['binding'] == binding and done['png_hashes'] == {'prefix-00.png': actual}
                writes = [json.loads(line) for line in writespath.read_text().splitlines()]
                assert len(writes) == 1
                w = writes[0]
                assert w['position'] == 0 and w['source_png_sha256'] is None and w['output_png_sha256'] == actual
                assert w['noise_seed'] == stable_seed(20260924, noise_namespace(pid, c, 'mt8-eval'), 0)
                assert w['event'] == event_text(rows[pid, v]['history'][:2])
                pngs[str(png)] = actual
                if v == 0:
                    relative = png.relative_to(SOURCE)
                    assert recovery['copied_sha256'][str(relative)] == actual
                    assert previous['verified_png_sha256'][str(OLD/relative)] == actual
                evidence += [donepath, writespath]
        print(json.dumps({'verified_variant': v, 'actual_pngs': 5840}), flush=True)
        seen = set()
        for s in range(4):
            read = SOURCE/'C8'/f'read-V{v}'/f'readback-{s}.jsonl'
            finished = read.with_name(f'finished-{s}.json')
            h = sha(read)
            readhashes[str(read)] = h
            receipts[f'V{v}/{s}'] = json.loads(finished.read_text())
            lines = read.read_text().splitlines()
            assert len(lines) == len(ids[s::4])*54 == receipts[f'V{v}/{s}']['items']
            if v == 0:
                assert h == previous['readback_sha256'][str(OLD/read.relative_to(SOURCE))]
                assert h == recovery['copied_sha256'][str(read.relative_to(SOURCE))]
            for line in lines:
                r = json.loads(line)
                pid, c, depth, control, family, task = [r[k] for k in ['pair_id', 'chain', 'prefix', 'control', 'family', 'task']]
                key = (pid, c, depth, control, family, task)
                assert key not in seen and pid in ids[s::4]
                seen.add(key)
                assert depth == 0 and task == 'mcq' and family in p['families'] and control in p['controls']
                assert c in (range(8) if control in ['memory', 'mismatch'] else [0])
                assert r['split'] == 'train' and r['endpoint_kind'] == 'student'
                step = int.from_bytes(hashlib.sha256(f'eval:{pid}:{family}'.encode()).digest()[:4], 'big')
                order, correct = option_order(pid, step)
                predicted = mcq['extract_choice'](r['generated']['raw'])
                assert r['option_order'] == order and r['correct_letter'] == 'ABCD'[correct]
                assert r['predicted_letter'] == predicted and r['correct'] == (predicted == 'ABCD'[correct])
                assert r['parse_failure'] == (predicted is None) and r['question'] == rows[pid, v]['forms'][family]
                if control in ['memory', 'mismatch']:
                    donor = pid if control == 'memory' else p['donors'][pid]
                    expected = images/donor.replace(':', '_')/f'seed-{c}'/'prefix-00.png'
                    raw_path = OLD/expected.relative_to(SOURCE) if v == 0 else expected
                    assert r['png_path'] == str(raw_path) and r['png_sha256'] == pngs[str(expected)]
                    if control == 'mismatch':
                        assert r['donor_pair_id'] == donor
                else:
                    assert r['png_path'] is None and r['png_sha256'] is None
                counts = per_pref[pid]['counts'][f'V{v}/{control}/{family}']
                counts[0] += int(r['correct'])
                counts[1] += 1
                parse_failures += int(r['parse_failure'])
                if control == 'memory':
                    all_images.setdefault((pid, v, c), {})[family] = bool(r['correct'])
                all_keys.add((v, *key))
            assert sha(read) == h, 'Completed readback changed during audit'
            readcounts[f'V{v}/{s}'] = len(lines)
            evidence += [read, finished]
        expected_keys = {(pid, c, 0, control, family, 'mcq') for pid in ids for control in p['controls']
                         for c in (range(8) if control in ['memory', 'mismatch'] else [0]) for family in p['families']}
        assert seen == expected_keys and len(seen) == 39420
    assert len(all_keys) == 78840 and len(all_images) == len(pngs) == 11680
    for (pid, v, c), values in all_images.items():
        assert set(values) == set(p['families'])
        correct = int(all(values.values()))
        per_pref[pid]['all_three'][0] += correct
        per_pref[pid]['all_three_by_variant'][f'V{v}'][0] += correct
    strata = {}
    scopes = [('all730', ids), ('adapted64', [i for i in ids if i in pilot]), ('other666', [i for i in ids if i not in pilot])]
    scopes += [(f'topic/{topic}', [i for i in ids if i.split(':')[0] == topic]) for topic in p['topics']]
    for label, subset in scopes:
        by_variant = {}
        for v in range(2):
            counts = {f'{ctrl}/{fam}': [sum(per_pref[i]['counts'][f'V{v}/{ctrl}/{fam}'][j] for i in subset) for j in range(2)]
                      for ctrl in p['controls'] for fam in p['families']}
            all_three = [sum(per_pref[i]['all_three_by_variant'][f'V{v}'][j] for i in subset) for j in range(2)]
            by_variant[f'V{v}'] = {'counts': counts, 'all_three': all_three}
        totals = {key: [sum(by_variant[f'V{v}']['counts'][key][j] for v in range(2)) for j in range(2)] for key in counts}
        all_three = [sum(per_pref[i]['all_three'][j] for i in subset) for j in range(2)]
        strata[label] = {'preferences': len(subset), 'counts': totals, 'all_three': all_three, 'by_variant': by_variant,
            'all16_T1_correct_preferences': sum(sum(per_pref[i]['counts'][f'V{v}/memory/T1'][0] for v in range(2)) == 16 for i in subset),
            'all16_three_correct_preferences': sum(per_pref[i]['all_three'][0] == 16 for i in subset),
            'T1_matching_gain_pp': 100*(totals['memory/T1'][0]-totals['mismatch/T1'][0])/totals['memory/T1'][1]}
    import numpy as np
    intervals = {}
    for label, subset in scopes[:3]:
        values = np.array([[(sum(per_pref[i]['counts'][f'V{v}/memory/T1'][0]-per_pref[i]['counts'][f'V{v}/mismatch/T1'][0] for v in range(2)))/16,
                            (per_pref[i]['counts']['V1/memory/T1'][0]-per_pref[i]['counts']['V0/memory/T1'][0])/8,
                            (per_pref[i]['all_three_by_variant']['V1'][0]-per_pref[i]['all_three_by_variant']['V0'][0])/8] for i in subset])
        rng = np.random.default_rng(20260928)
        bootstrap = np.array([values[rng.integers(len(values), size=len(values))].mean(axis=0) for _ in range(10000)])
        intervals[label] = {name: {'difference_pp': float(values[:, j].mean()*100), 'paired_preference_bootstrap_95ci_pp': (np.percentile(bootstrap[:, j], [2.5, 97.5])*100).tolist()}
                           for j, name in enumerate(['T1_memory_minus_mismatch', 'V1_minus_V0_T1', 'V1_minus_V0_all_three'])}
    report = {'status': 'complete_prespecified_C8_V0_V1_only_B0_comparison_pending',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE),
        'scope': 'Fixed C8, all730, both V0/V1 and eight prespecified seeds; B0 not included; no budget/model selection.',
        'limitations': ['All730 were in B0 training; only64 in C8 additional adaptation.', 'No C8 minus B0, forgetting or unseen-preference generalization claim.'],
        'protocol_sha256': sha(SOURCE/'protocol.json'), 'checkpoint': p['checkpoints']['C8'],
        'verified_png_sha256': pngs, 'readback_sha256': readhashes, 'readback_counts': readcounts,
        'finished_receipts': receipts, 'parse_failures': parse_failures, 'strata': strata, 'per_preference': per_pref,
        'bootstrap_replicates': 10000, 'bootstrap_seed': 20260928, 'paired_preference_intervals': intervals,
        'raw_png_path_provenance': {'V0': str(OLD), 'V1': str(SOURCE), 'recovery_receipt_sha256': sha(recovery_path), 'prior_V0_report_sha256': sha(previous_path)},
        'script_sha256': sha(Path(__file__))}
    OUTPUT.mkdir()
    result = OUTPUT/'C8_FULL_STAGE_RESULTS.json'
    result.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    evidence += [result, Path(__file__), deps]
    manifest = OUTPUT/'evidence-manifest.json'
    manifest.write_text(json.dumps({str(f.relative_to(TASK)): sha(f) for f in evidence}, indent=2)+'\n')
    archive = TASK/'c8-full-stage-evidence-20260929.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for f in evidence + [manifest]:
            tar.add(f, arcname=str(f.relative_to(TASK)), recursive=False)
    receipt = {'path': str(archive), 'sha256': sha(archive), 'files': len(evidence)+1,
               'png_policy': 'All11680 actual PNGs rehashed; all complete/writes and hashes archived; PNG bytes remain on shared storage.'}
    (OUTPUT/'archive.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'strata': {k: strata[k] for k in ['all730', 'adapted64', 'other666']}, 'intervals': intervals,
                      'parse_failures': parse_failures, 'archive': receipt}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
