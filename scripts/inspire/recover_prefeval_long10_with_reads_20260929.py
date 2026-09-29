"""Recover the stopped two-GPU long10 run and retain its completed raw reads."""
import importlib.util
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
OLD = TASK/'long10/recovery-20260929-0005-2gpu'
NEW = TASK/'long10/recovery-20260929-1425-2gpu'
CODE = PROJECT/'repos/prefeval-c8-long10-20260928'
SCHEDULER = TASK/'long10/resume-two-gpu-20260929.py'


def main():
    assert not NEW.exists()
    helper = TASK/'long10/prepare-recovery-20260928-2135.py'
    spec = importlib.util.spec_from_file_location('frozen_recovery_validation', helper)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    assert h.sha(helper) == '76bb631064b256343b48c8c837227514e8a105d5ce802937ef370fb846e8bc9b'
    stop = json.loads((TASK/'long10/platform-stopped-20260929-1419.json').read_text())
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
    assert receipt['complete_chains'] == {'C8/V0': 1460, 'C8/V1': 26}
    audit = json.loads((TASK/'long10/analysis-c8-v0-20260929-1330/C8_LONG10_V0_STAGE_RESULTS.json').read_text())
    reads = sorted(OLD.rglob('readback-*.jsonl'))
    assert len(reads) == 4 and all(p.parent == OLD/'C8/read-V0' for p in reads)
    for source in reads:
        assert h.sha(source) == audit['readback_sha256'][str(source)]
        s = int(source.stem.split('-')[-1])
        done = source.with_name(f'finished-{s}.json')
        assert json.loads(done.read_text()) == audit['finished_receipts'][str(s)]
        assert json.loads(done.read_text())['items'] == len(protocol['ids'][s::4])*48
        for line in source.read_text().splitlines():
            row = json.loads(line)
            if row['control'] in ['memory', 'mismatch']:
                relative = str(Path(row['png_path']).relative_to(OLD))
                assert receipt['copied_sha256'][relative] == row['png_sha256']
        for original in [source, done]:
            copy(original, NEW/original.relative_to(OLD))
        receipt['readbacks'][str(source.relative_to(OLD))] = json.loads(done.read_text())['items']
    launcher = (OLD/'launch-two-gpu.sh').read_text().replace(str(OLD), str(NEW))
    (NEW/'launch-two-gpu.sh').write_text(launcher)
    receipt['launcher_sha256'] = h.sha(NEW/'launch-two-gpu.sh')
    # Never copy old host/PID-bound topology or process/sentinel files.
    evidence = [p for p in OLD.rglob('*') if p.is_file() and p.suffix in ['.json', '.jsonl', '.log']]
    evidence += [TASK/'long10/platform-stopped-20260929-1419.json', TASK/'long10/platform-events-20260929-1346.json', Path(__file__)]
    manifest = NEW/'interruption-manifest.json'
    manifest.write_text(json.dumps({str(p.relative_to(TASK)): h.sha(p) for p in evidence}, indent=2)+'\n')
    archive = TASK/'long10-interruption-20260929-1346.tar.gz'
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
