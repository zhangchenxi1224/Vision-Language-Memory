"""Recover the stopped two-GPU long10 run and retain its completed raw reads."""
import importlib.util
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
OLD = TASK/'long10/recovery-20260929-1425-2gpu'
NEW = TASK/'long10/recovery-20260930-1210-2gpu'
CODE = PROJECT/'repos/prefeval-c8-long10-20260928'
SCHEDULER = TASK/'long10/resume-two-gpu-20260929.py'


def preserve_completed_reads(old, new, protocol, receipt, copy, digest, long_run):
    """Validate complete read keys and documented old PNG provenance without rewriting rows."""
    import hashlib
    import sys
    frozen = PROJECT/'repos/prefeval-multitarget-round2'
    deps = TASK/'long10/audit-deps-20260929'
    dependency = json.loads((deps/'dependency-manifest.json').read_text())
    for path, expected in dependency['copied_sha256'].items():
        assert digest(deps/path) == expected
    sys.path[:0] = [str(deps), str(frozen), str(frozen/'src')]
    from scripts.experiments.prefeval_k1_data import load_records, official_mcq, option_order
    from scripts.experiments.prefeval_k1_variants import load_variants, apply_variant
    records = load_records('train')
    variants = load_variants(PROJECT/'runs/prefeval-b-mcq-20260925/variants-train.json', records)
    rows = {(r['base_pair_id'], v): apply_variant(r, variants, v) for r in records for v in range(2)}
    mcq = official_mcq(frozen/'third_party/prefeval_reference')
    prior_receipt = json.loads((old/'recovery-preparation.json').read_text())
    if long_run:
        historical = TASK/'long10/recovery-20260929-0005-2gpu'
        audit = json.loads((TASK/'long10/analysis-c8-v0-20260929-1330/C8_LONG10_V0_STAGE_RESULTS.json').read_text())
        expected_groups = {('C8', 0)}
    else:
        historical = TASK/'eval730/recovery-20260928-1920'
        audit = json.loads((TASK/'eval730/analysis-c8-full-20260929-1725/C8_FULL_STAGE_RESULTS.json').read_text())
        expected_groups = {('C8', 0), ('C8', 1), ('B0', 0)}
    readfiles = sorted(old.rglob('readback-*.jsonl'))
    assert len(readfiles) == 4*len(expected_groups)
    assert {(f.relative_to(old).parts[0], int(f.parent.name[-1])) for f in readfiles} == expected_groups
    receipt['readback_verification'] = {}
    cached = {}
    for arm, v in sorted(expected_groups):
        assert receipt['complete_chains'][f'{arm}/V{v}'] == 730*protocol['noise_chains']
        for s in range(4):
            source = old/arm/f'read-V{v}'/f'readback-{s}.jsonl'
            finished = source.with_name(f'finished-{s}.json')
            before = digest(source)
            done = json.loads(finished.read_text())
            ids = protocol['ids'][s::4]
            depths = [0, 5, 10] if long_run else [0]
            expected = {(pid,c,d,ctrl,fam,'mcq') for pid in ids for d in depths
                for ctrl in protocol['controls'] if ctrl != 'blank' or d == 0
                for c in (range(protocol['noise_chains']) if ctrl in ['memory','mismatch'] else [0])
                for fam in protocol['families']}
            seen = set()
            parse_failures = 0
            provenance = set()
            for line in source.read_text().splitlines():
                r = json.loads(line)
                pid,c,d,ctrl,fam,task = [r[k] for k in ['pair_id','chain','prefix','control','family','task']]
                key = (pid,c,d,ctrl,fam,task)
                assert key in expected and key not in seen
                seen.add(key)
                assert r['split'] == 'train' and r['endpoint_kind'] == 'student'
                step = int.from_bytes(hashlib.sha256(f'eval:{pid}:{fam}'.encode()).digest()[:4], 'big')
                order, correct = option_order(pid, step)
                prediction = mcq['extract_choice'](r['generated']['raw'])
                assert r['option_order'] == order and r['correct_letter'] == 'ABCD'[correct]
                assert r['predicted_letter'] == prediction and r['correct'] == (prediction == 'ABCD'[correct])
                assert r['parse_failure'] == (prediction is None)
                assert r['question'] == rows[pid,v]['forms'][fam]
                parse_failures += int(r['parse_failure'])
                if ctrl in ['memory','mismatch']:
                    donor = pid if ctrl == 'memory' else protocol['donors'][pid]
                    relative = Path(arm)/f'eval-V{v}'/donor.replace(':','_')/f'seed-{c}'/f'prefix-{d:02d}.png'
                    raw = Path(r['png_path'])
                    allowed = historical/relative if arm == 'C8' and v == 0 else old/relative
                    assert raw == allowed
                    value = receipt['copied_sha256'][str(relative)]
                    assert r['png_sha256'] == value
                    if raw not in cached:
                        cached[raw] = digest(raw)
                    assert cached[raw] == value
                    if raw != old/relative:
                        assert prior_receipt['copied_sha256'][str(relative)] == value
                    if ctrl == 'mismatch':
                        assert r['donor_pair_id'] == donor
                    provenance.add(str(allowed.parents[4]))
                else:
                    assert r['png_path'] is None and r['png_sha256'] is None
            assert seen == expected and done['items'] == len(expected)
            assert digest(source) == before
            if arm == 'C8':
                audited_source = historical/source.relative_to(old) if long_run else source
                assert audit['readback_sha256'][str(audited_source)] == before
            for original in [source, finished]:
                copy(original, new/original.relative_to(old))
            relative_read = str(source.relative_to(old))
            receipt['readbacks'][relative_read] = len(expected)
            receipt['readback_verification'][relative_read] = {'sha256': before, 'unique_rows': len(seen),
                'parse_failures': parse_failures, 'raw_png_roots': sorted(provenance)}
    receipt['prior_recovery_receipt_sha256'] = digest(old/'recovery-preparation.json')
    print(json.dumps({'preserved_unique_readbacks': sum(receipt['readbacks'].values())}), flush=True)


def main():
    assert not NEW.exists()
    helper = TASK/'long10/prepare-recovery-20260928-2135.py'
    spec = importlib.util.spec_from_file_location('frozen_recovery_validation', helper)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    assert h.sha(helper) == '76bb631064b256343b48c8c837227514e8a105d5ce802937ef370fb846e8bc9b'
    stop = json.loads((TASK/'long10/platform-stopped-20260930-1205.json').read_text())
    assert stop['data']['name'] == 'prefeval-b-read-h200x2-20260925' and stop['data']['status'] == 'STOPPED'
    assert not (OLD/'complete.json').exists()
    assert h.sha(SCHEDULER) == '392638990e1327d67283a8b07da53a849777d176ae1446fbccdceeb62567c6d1'
    code = CODE/'scripts/inspire/run_prefeval_c8_long10.py'
    assert h.sha(code) == h.CONTROLLER_SHA and h.sha(OLD/'protocol.json') == h.PROTOCOL_SHA
    spec = importlib.util.spec_from_file_location('long10_frozen', code)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    assert frozen.base.git_head(CODE) == h.COMMIT
    protocol = frozen.prepare(NEW, False)
    assert h.sha(NEW/'protocol.json') == h.PROTOCOL_SHA
    assert h.sha(NEW/'ids.json') == h.sha(OLD/'ids.json')
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(OLD),
        'destination': str(NEW), 'mode': 'same_user_instance_after_platform_recycle', 'stop_evidence': stop,
        'protocol_sha256': h.PROTOCOL_SHA, 'checkpoints': protocol['checkpoints'],
        'complete_chains': {}, 'partial_chains_not_copied': [], 'copied_sha256': {}, 'readbacks': {},
        'script_sha256': h.sha(Path(__file__)), 'frozen_validation_sha256': h.sha(helper),
        'scheduler_sha256': h.sha(SCHEDULER), 'gpu_launch_performed': False,
        'policy': 'Only fully validated 11-PNG chains are copied byte-for-byte. Rebuild partial chains without old writes. Raw completed Reader rows preserve old PNG paths and bytes; their corresponding new PNGs have identical verified hashes. All four logical shards remain on two GPUs.'}

    def copy(original, target):
        digest = h.sha(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        assert h.sha(target) == digest
        receipt['copied_sha256'][str(target.relative_to(NEW))] = digest

    folders = {pid.replace(':', '_'): pid for pid in protocol['ids']}
    for arm in protocol['arms']:
        for v in protocol['initial_variants']:
            source = OLD/arm/f'eval-V{v}'
            if not source.exists():
                continue
            binding = {'checkpoint_sha256': protocol['checkpoints'][arm]['sha256'], 'split': 'train',
                'steps': 28, 'cfg': 1, 'noise_chains': 2, 'inter_turns': 10,
                'state': 'only reopened uint8 RGB PNG; fresh Gaussian each write', 'noise_domain': 'mt8-eval',
                'initial_variants_sha256': h.sha(frozen.base.VARIANTS), 'initial_variant': v}
            for manifest in source.glob('manifest*.json'):
                assert json.loads(manifest.read_text()) == binding
                copy(manifest, NEW/manifest.relative_to(OLD))
            count = 0
            for directory in sorted(source.glob('*/seed-*')):
                if not (directory/'complete.json').exists():
                    receipt['partial_chains_not_copied'].append(str(directory))
                    continue
                pid, chain = folders[directory.parent.name], int(directory.name.split('-')[1])
                assert chain in range(2)
                done = json.loads((directory/'complete.json').read_text())
                assert done['binding'] == binding
                actual = {f'prefix-{i:02d}.png': h.sha(directory/f'prefix-{i:02d}.png') for i in range(11)}
                assert done['png_hashes'] == actual and {p.name for p in directory.glob('prefix-*.png')} == set(actual)
                writes = [json.loads(line) for line in (directory/'writes.jsonl').read_text().splitlines()]
                h.validate_writes(writes, actual, protocol['current_exchange_sha256'][f'V{v}'][pid], pid, chain)
                for name in [*actual, 'complete.json', 'writes.jsonl']:
                    copy(directory/name, NEW/(directory/name).relative_to(OLD))
                count += 1
            receipt['complete_chains'][f'{arm}/V{v}'] = count
            print(json.dumps({'verified_copied': f'{arm}/V{v}', 'chains': count}), flush=True)
    assert receipt['complete_chains'] == {'C8/V0': 1460, 'C8/V1': 1120}
    preserve_completed_reads(OLD, NEW, protocol, receipt, copy, h.sha, True)
    launcher = (OLD/'launch-two-gpu.sh').read_text().replace(str(OLD), str(NEW))
    (NEW/'launch-two-gpu.sh').write_text(launcher)
    receipt['launcher_sha256'] = h.sha(NEW/'launch-two-gpu.sh')
    # Never copy old host/PID-bound topology or process/sentinel files.
    evidence = [p for p in OLD.rglob('*') if p.is_file() and p.suffix in ['.json', '.jsonl', '.log']]
    evidence += [TASK/'long10/platform-stopped-20260930-1205.json', TASK/'long10/platform-events-20260930-1205.json', Path(__file__)]
    manifest = NEW/'interruption-manifest.json'
    manifest.write_text(json.dumps({str(p.relative_to(TASK)): h.sha(p) for p in evidence}, indent=2)+'\n')
    archive = TASK/'long10-interruption-20260930-0826.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for path in evidence + [manifest]:
            tar.add(path, arcname=str(path.relative_to(TASK)), recursive=False)
    receipt['archive'] = {'path': str(archive), 'sha256': h.sha(archive), 'files': len(evidence)+1}
    (NEW/'recovery-preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    # Required by the unchanged two-GPU scheduler; content explicitly describes recovery.
    (NEW/'migration-preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'copied_sha256'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()


