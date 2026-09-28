"""Audit live completed PNG chains; copy them only after documented source shutdown.

This helper never launches GPU work or edits the source. Run --audit-only while
the source is live. After STOPPED, refresh platform status, use a new --output
with --stopped-status, and check the actual target host before running its shell.
Generation-only recovery deliberately refuses any existing Reader outputs.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from datetime import datetime, timezone

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
TASK = PROJECT/'runs/prefeval-multitarget-20260927'
SOURCE = TASK/'long10/fixed-c8-b0-730-20260928'
CODE = PROJECT/'repos/prefeval-c8-long10-20260928'
COMMIT = '6563d16322c792d6ad0007c17cd6be9c6ca279e3'
PROTOCOL_SHA = 'b43fd72a86e8a50547b359ac1ad7b0bf97bcdcbc90cb542f884206980caaae33'
CONTROLLER_SHA = 'c5a9051c1d8426f45e017b69c960188c0eacf2c2dc507a782fc264a63783e4ce'
NOTEBOOK = 'dl-clear-retain-h200x4-20260914'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_writes(writes, actual, event_hashes, pid, chain):
    assert len(writes) == 11 and [r['position'] for r in writes] == list(range(11))
    for i, r in enumerate(writes):
        assert r['source_png_sha256'] == (actual[f'prefix-{i-1:02d}.png'] if i else None)
        assert r['output_png_sha256'] == actual[f'prefix-{i:02d}.png']
        assert hashlib.sha256(r['event'].encode()).hexdigest() == event_hashes[i]
        seed = int.from_bytes(hashlib.sha256(f'20260924:mt8-eval:rollout:{pid}:{chain}:{i}'.encode()).digest()[:8], 'big') % (2**63-1)
        assert r['noise_seed'] == seed


def main(args):
    output = args.output.resolve()
    assert output.parent == (TASK/'long10').resolve() and not output.exists()
    assert not (SOURCE/'complete.json').exists(), 'Already complete; inspect results instead'
    assert not list(SOURCE.rglob('readback-*.jsonl')), 'Generation-only recovery; Reader evidence needs separate audit'
    controller = CODE/'scripts/inspire/run_prefeval_c8_long10.py'
    assert sha(controller) == CONTROLLER_SHA and sha(SOURCE/'protocol.json') == PROTOCOL_SHA
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip() == COMMIT
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=CODE, text=True).strip()
    stop = None
    if not args.audit_only:
        assert args.stopped_status, 'Fresh platform STOPPED evidence is required before copying'
        assert 0 <= time.time()-args.stopped_status.stat().st_mtime < 300
        stop = json.loads(args.stopped_status.read_text())
        assert stop['success'] and stop['data']['name'] == NOTEBOOK and stop['data']['status'] == 'STOPPED'
    spec = importlib.util.spec_from_file_location('long10_frozen', controller)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Frozen preparation rehashes both checkpoints, source data and worker code.
    protocol = module.prepare(output, False)
    assert sha(output/'protocol.json') == PROTOCOL_SHA
    assert sha(output/'ids.json') == sha(SOURCE/'ids.json')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'audit_host': os.uname().nodename,
        'mode': 'live_readonly_readiness' if args.audit_only else 'stopped_source_recovery_copy',
        'source': str(SOURCE), 'destination': str(output), 'protocol_sha256': PROTOCOL_SHA,
        'controller_commit': COMMIT, 'controller_sha256': CONTROLLER_SHA,
        'helper_sha256': sha(Path(__file__)), 'checkpoints': protocol['checkpoints'],
        'complete_chains': {}, 'partial_snapshot_not_copied': [], 'verified_files': {},
        'copied_files': 0, 'copied_pngs': 0, 'negative_checks': [],
        'stop_evidence': stop, 'gpu_launch_performed': False,
        'policy': 'Only complete 11-PNG trajectories are reusable, byte-for-byte; partial trajectories rebuild in a new directory with no old writes appended.'}
    folders = {pid.replace(':', '_'): pid for pid in protocol['ids']}
    metadata = [SOURCE/'ids.json', SOURCE/'protocol.json']
    variants_sha = sha(module.base.VARIANTS)
    checked_negative = False
    for arm in protocol['arms']:
        for variant in protocol['initial_variants']:
            source = SOURCE/arm/f'eval-V{variant}'
            if not source.exists():
                continue
            binding = {'checkpoint_sha256': protocol['checkpoints'][arm]['sha256'], 'split': 'train',
                'steps': 28, 'cfg': 1, 'noise_chains': 2, 'inter_turns': 10,
                'state': 'only reopened uint8 RGB PNG; fresh Gaussian each write',
                'noise_domain': 'mt8-eval', 'initial_variants_sha256': variants_sha, 'initial_variant': variant}
            for manifest in source.glob('manifest*.json'):
                assert json.loads(manifest.read_text()) == binding
                metadata.append(manifest)
                if not args.audit_only:
                    target = output/manifest.relative_to(SOURCE)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(manifest, target)
                    assert sha(target) == sha(manifest)
            # Freeze completion membership; a partial chain finishing later is not
            # accidentally archived while still being written.
            chain_dirs = sorted(source.glob('*/seed-*'))
            complete_dirs = [d for d in chain_dirs if (d/'complete.json').exists()]
            report['partial_snapshot_not_copied'] += [str(d) for d in chain_dirs if d not in complete_dirs]
            count = 0
            for directory in complete_dirs:
                pid, chain = folders[directory.parent.name], int(directory.name.split('-')[1])
                assert chain in range(2)
                done = json.loads((directory/'complete.json').read_text())
                assert done['binding'] == binding
                actual = {f'prefix-{i:02d}.png': sha(directory/f'prefix-{i:02d}.png') for i in range(11)}
                assert done['png_hashes'] == actual
                assert {p.name for p in directory.glob('prefix-*.png')} == set(actual)
                writes = [json.loads(x) for x in (directory/'writes.jsonl').read_text().splitlines()]
                events = protocol['current_exchange_sha256'][f'V{variant}'][pid]
                validate_writes(writes, actual, events, pid, chain)
                if not checked_negative:
                    for label, field, value in [('wrong_seed', 'noise_seed', -1), ('broken_previous_png', 'source_png_sha256', 'wrong'), ('wrong_exchange', 'event', 'wrong')]:
                        bad = copy.deepcopy(writes)
                        bad[5][field] = value
                        try:
                            validate_writes(bad, actual, events, pid, chain)
                        except AssertionError:
                            report['negative_checks'].append(label)
                        else:
                            raise AssertionError(f'Corruption was not rejected: {label}')
                    checked_negative = True
                metadata += [directory/'complete.json', directory/'writes.jsonl']
                for name in [*actual, 'complete.json', 'writes.jsonl']:
                    original = directory/name
                    digest = actual.get(name) or sha(original)
                    relative = str(original.relative_to(SOURCE))
                    report['verified_files'][relative] = digest
                    if not args.audit_only:
                        target = output/relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, target)
                        assert sha(target) == digest
                        report['copied_files'] += 1
                        report['copied_pngs'] += int(name.endswith('.png'))
                count += 1
            report['complete_chains'][f'{arm}/V{variant}'] = count
    assert checked_negative
    if not args.audit_only:
        # Formal stage only: never rerun the completed smoke.
        shell = '\n'.join(['#!/usr/bin/env bash', 'set -euo pipefail',
            f'exec 9>"{TASK}/long10/pipeline.lock"', 'flock -n 9', f'cd "{CODE}"',
            'export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1',
            f'exec "{module.PYTHON}" scripts/inspire/run_prefeval_c8_long10.py --output "{output}"', ''])
        (output/'launch-recovery.sh').write_text(shell)
        report['launcher_sha256'] = sha(output/'launch-recovery.sh')
    archive = output/'validated-source-metadata.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for path in sorted(set(metadata)):
            tar.add(path, arcname=str(path.relative_to(SOURCE)), recursive=False)
    report['metadata_archive'] = {'path': str(archive), 'sha256': sha(archive), 'files': len(set(metadata))}
    (output/'recovery-preparation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'verified_files'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--stopped-status', type=Path)
    main(parser.parse_args())
