"""Recover the stopped all730 evaluation, retaining audited completed readbacks."""
import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
OLD = TASK/'eval730/recovery-20260929-1325'
NEW = TASK/'eval730/recovery-20260930-1210'
CONTROLLER = PROJECT/'repos/prefeval-c8-eval730-20260928'
PROTOCOL_SHA = 'c670e3db2d7d54bf449999cf693d7544fb54bbc7040420b641c31484e6f641dc'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


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
    assert not NEW.exists(), 'Destination must be new'
    stopped = json.loads((TASK/'eval730/platform-stopped-20260930-1205.json').read_text())
    assert stopped['data']['name'] == 'prefeval-mt-eval-h200x2-20260928'
    assert stopped['data']['status'] == 'STOPPED'
    assert not (OLD/'complete.json').exists()
    code = CONTROLLER/'scripts/inspire/run_prefeval_c8_eval730.py'
    assert sha(code) == 'ab4d702b41115bcfd888d5ef7cb561577eaf8af88b035830395ebbe45be2a8cf'
    assert sha(OLD/'protocol.json') == PROTOCOL_SHA
    spec = importlib.util.spec_from_file_location('frozen_eval730', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = module.prepare(NEW)
    assert sha(NEW/'protocol.json') == PROTOCOL_SHA
    assert sha(NEW/'ids.json') == sha(OLD/'ids.json')
    frozen = PROJECT/'repos/prefeval-multitarget-round2'
    sys.path[:0] = [str(frozen), str(frozen/'src')]
    from scripts.experiments.prefeval_k1_data import load_records, event_text
    from scripts.experiments.prefeval_k1_variants import load_variants, apply_variant
    from scripts.experiments.prefeval_k1_source_bank import noise_namespace
    from vision_memory.training.latent_bank_unet import stable_seed
    rows = load_records('train')
    variants = load_variants(module.VARIANTS, rows)
    rowmap = {(r['base_pair_id'], v): apply_variant(r, variants, v) for r in rows for v in range(2)}
    foldermap = {x.replace(':', '_'): x for x in protocol['ids']}
    receipt = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source': str(OLD),
        'destination': str(NEW), 'protocol_sha256': PROTOCOL_SHA, 'checkpoints': protocol['checkpoints'],
        'copied_sha256': {}, 'complete_chains': {}, 'partial_chains_not_copied': [],
        'readbacks': {}, 'script_sha256': sha(Path(__file__)),
        'policy': 'Rehash every complete chain and copy byte-for-byte. Incomplete chains are rebuilt in new directories. All completed readbacks retain their original PNG paths and bytes; matching copied PNGs are rehashed. Frozen workers skip completed keys; no additional inference for reused keys.'}

    def copy(original, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        actual = sha(original)
        shutil.copyfile(original, destination)
        assert sha(destination) == actual
        receipt['copied_sha256'][str(destination.relative_to(NEW))] = actual

    for arm in ['C8', 'B0']:
        for v in range(2):
            source, target = OLD/arm/f'eval-V{v}', NEW/arm/f'eval-V{v}'
            if not source.exists():
                continue
            binding = {'checkpoint_sha256': protocol['checkpoints'][arm]['sha256'], 'split': 'train',
                'steps': 28, 'cfg': 1, 'noise_chains': 8, 'inter_turns': 0,
                'state': 'only reopened uint8 RGB PNG; fresh Gaussian each write',
                'noise_domain': 'mt8-eval', 'initial_variants_sha256': sha(module.VARIANTS), 'initial_variant': v}
            for manifest in source.glob('manifest*.json'):
                assert json.loads(manifest.read_text()) == binding
                copy(manifest, target/manifest.name)
            count = 0
            for chain_folder in sorted(source.glob('*/seed-*')):
                if not (chain_folder/'complete.json').exists():
                    receipt['partial_chains_not_copied'].append(str(chain_folder))
                    continue
                pid = foldermap[chain_folder.parent.name]
                chain = int(chain_folder.name.split('-')[1])
                assert 0 <= chain < 8
                done = json.loads((chain_folder/'complete.json').read_text())
                assert done['binding'] == binding and set(done['png_hashes']) == {'prefix-00.png'}
                actual = sha(chain_folder/'prefix-00.png')
                assert actual == done['png_hashes']['prefix-00.png']
                writes = [json.loads(x) for x in (chain_folder/'writes.jsonl').read_text().splitlines()]
                assert len(writes) == 1
                w = writes[0]
                assert w['position'] == 0 and w['source_png_sha256'] is None
                assert w['output_png_sha256'] == actual
                assert w['event'] == event_text(rowmap[pid, v]['history'][:2])
                assert w['noise_seed'] == stable_seed(20260924, noise_namespace(pid, chain, 'mt8-eval'), 0)
                for name in ['prefix-00.png', 'complete.json', 'writes.jsonl']:
                    copy(chain_folder/name, target/chain_folder.relative_to(source)/name)
                count += 1
            receipt['complete_chains'][f'{arm}/V{v}'] = count
            print(json.dumps({'verified_copied': f'{arm}/V{v}', 'chains': count}), flush=True)
    assert receipt['complete_chains'] == {'C8/V0': 5840, 'C8/V1': 5840, 'B0/V0': 5840, 'B0/V1': 2852}
    preserve_completed_reads(OLD, NEW, protocol, receipt, copy, sha, False)
    launcher = (OLD/'launch-recovery.sh').read_text().replace(str(OLD), str(NEW))
    (NEW/'launch-recovery.sh').write_text(launcher)
    receipt['launcher_sha256'] = sha(NEW/'launch-recovery.sh')
    evidence = [p for p in OLD.rglob('*') if p.is_file() and p.suffix in ['.json', '.jsonl', '.log']]
    evidence += [TASK/'eval730/platform-stopped-20260930-1205.json',
                 TASK/'eval730/platform-events-20260930-1205.json', Path(__file__)]
    filehashes = {str(p.relative_to(TASK)): sha(p) for p in evidence}
    manifest = NEW/'interruption-manifest.json'
    manifest.write_text(json.dumps(filehashes, indent=2)+'\n')
    archive = TASK/'eval730-interruption-20260930-0732.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive, 'w:gz') as tar:
        for path in evidence + [manifest]:
            tar.add(path, arcname=str(path.relative_to(TASK)), recursive=False)
    receipt['archive'] = {'path': str(archive), 'sha256': sha(archive), 'files': len(evidence)+1}
    (NEW/'recovery-preparation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'copied_sha256'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()


