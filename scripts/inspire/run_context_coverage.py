"""Bounded two-GPU coverage ablation; leaves existing experiments read-only."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.inspire.run_prompt_matching_parallel import run_job, write_json

MODES = ('history_hard', 'prompt_matching')


def build_plan(args):
    root = args.output.resolve()
    smoke = args.phase == 'smoke'
    groups = []
    group = []
    for gpu, mode in enumerate(MODES):
        command = [sys.executable, 'scripts/experiments/prefeval_k1_teacher.py', '--arm', 'B',
            '--supervision', mode, '--context-suite', 'diverse-v1', '--split', 'pilot',
            '--ids-file', str(args.ids_file.resolve()), '--shards', '1', '--shard', '0',
            '--base', str(args.base), '--reader', str(args.reader), '--prefeval', str(args.prefeval),
            '--output', str(root / mode / 'teachers'), '--teacher-cache', str(root / 'targets'),
            '--steps', '24' if smoke else '288', '--snapshot-steps', '24' if smoke else '72,144,216,288']
        if smoke:
            command += ['--limit', '1']
        group.append({'name': 'teacher-' + mode, 'gpu': gpu, 'command': command})
    groups.append(group)
    if not smoke:
        groups.append([{'name': 'select-' + mode, 'gpu': gpu, 'command': [sys.executable,
            'scripts/experiments/prefeval_context_select.py', '--teachers', str(root / mode / 'teachers'),
            '--reader', str(args.reader), '--cache', str(root / 'validation-targets'),
            '--output', str(root / mode / 'selected'), '--ids-file', str(args.ids_file.resolve())]}
            for gpu, mode in enumerate(MODES)])
        for endpoint in ('teachers', 'selected'):
            group = []
            for gpu, mode in enumerate(MODES):
                command = [sys.executable, 'scripts/experiments/prefeval_k1_evaluate.py',
                    '--kind', 'teacher', '--split', 'pilot', '--reader', str(args.reader),
                    '--images', str(root / mode / endpoint), '--output', str(root / mode / ('readback-' + endpoint)),
                    '--ids-file', str(args.ids_file.resolve()), '--shards', '1', '--shard', '0',
                    '--controls', 'memory,blank,mismatch,text', '--tasks', 'mcq',
                    '--families', 'T1,T2,T3,O1,O2', '--prefixes', '0', '--noise-chains', '1']
                if endpoint == 'selected':
                    command += ['--selected-teacher']
                group.append({'name': f'readback-{mode}-{endpoint}', 'gpu': gpu, 'command': command})
            groups.append(group)
        groups.append([{'name': 'reference-mismatch-' + mode, 'gpu': gpu, 'command': [sys.executable,
            'scripts/experiments/prefeval_k1_evaluate.py', '--kind', 'teacher', '--split', 'pilot',
            '--reader', str(args.reader), '--images', str(args.reference / mode / 'teachers'),
            '--output', str(root / mode / 'reference-mismatch'), '--ids-file', str(args.ids_file.resolve()),
            '--shards', '1', '--shard', '0', '--controls', 'mismatch', '--tasks', 'mcq',
            '--families', 'T1,T2,T3,O1,O2', '--prefixes', '0', '--noise-chains', '1']}
            for gpu, mode in enumerate(MODES)])
    return {'schema': 'dreamlite.context-coverage-run.v1', 'phase': args.phase,
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'ids_sha256': hashlib.sha256(args.ids_file.read_bytes()).hexdigest(),
            'groups': groups, 'new_writer_training': False,
            'selection_scope': 'held-out queries on training histories; no dev or official histories'}


def main(args):
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not 0 < args.max_hours <= 6:
        raise ValueError('Each invocation is bounded to at most six hours')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('A clean frozen checkout is required')
    args.output.mkdir(parents=True, exist_ok=True)
    claim = args.output / 'active-owner'
    claim.mkdir()  # Shared-filesystem exclusive creation; never steal another host's claim.
    try:
        write_json(claim / 'owner.json', {'hostname': socket.gethostname(), 'pid': os.getpid(),
            'started': time.time(), 'commit': plan['commit'], 'output': str(args.output.resolve())})
        manifest = args.output / 'plan.json'
        if manifest.exists() and json.loads(manifest.read_text()) != plan:
            raise ValueError('Immutable run plan changed')
        write_json(manifest, plan)
        for d in ('receipts', 'logs', 'attempts'):
            (args.output / d).mkdir(exist_ok=True)
        # Preserve failed attempts before retry; GPU-time accounting must not reset.
        for p in (args.output / 'receipts').glob('*.json'):
            value = json.loads(p.read_text())
            if value['exit_code'] != 0:
                archive = args.output / 'attempts' / (p.stem + '-' + str(value['started']) + '.json')
                if not archive.exists():
                    write_json(archive, value)
                p.unlink()
        used = sum(max(0, v['finished']-v['started']) for d in ('receipts', 'attempts')
                   for p in (args.output/d).glob('*.json') for v in [json.loads(p.read_text())])
        remaining = (1 if args.phase == 'smoke' else 16) * 3600 - used
        if remaining <= 0:
            raise RuntimeError('Campaign GPU-hour budget exhausted')
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('Instance has existing GPU processes; do not disturb them')
        # At most two one-GPU workers: wall-clock remainder conservatively halves GPU budget.
        deadline = time.monotonic() + min(args.max_hours*3600, remaining/2)
        write_json(args.output/'status.json', {'status': 'running', 'plan': plan['commit'],
                                              'previous_gpu_seconds': used, 'started': time.time()})
        for group in plan['groups']:
            if time.monotonic() >= deadline:
                raise TimeoutError('Invocation budget exhausted; preserve resumable artifacts')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(run_job, job, args.output, deadline) for job in group]
                for future in futures:
                    print(json.dumps(future.result()), flush=True)
        if args.phase == 'pilot':
            subprocess.run([sys.executable, 'scripts/reporting/context_coverage_report.py',
                '--run', str(args.output), '--reference', str(args.reference),
                '--ids-file', str(args.ids_file), '--output', str(args.output/'comparison.json')], cwd=ROOT, check=True)
        write_json(args.output/'status.json', {'status': 'completed', 'commit': plan['commit'],
            'promotion': 'requires paired report and independent Writer evidence', 'finished': time.time()})
    except BaseException as error:
        write_json(args.output/'status.json', {'status': 'failed', 'error': str(error), 'time': time.time()})
        raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True)
        claim.rmdir()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['smoke', 'pilot'], required=True)
    for name in ('output', 'base', 'reader', 'prefeval', 'ids-file', 'reference'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--max-hours', type=float, default=6)
    p.add_argument('--dry-run', action='store_true')
    main(p.parse_args())
