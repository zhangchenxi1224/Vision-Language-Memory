"""Bounded, resumable K1 hard/history-hard/soft comparison on two existing GPUs."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MODES = ('hard_ce', 'history_hard', 'prompt_matching')


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def build_plan(args):
    smoke = args.phase == 'smoke'
    steps, shards = (12, 1) if smoke else (288, 2)
    root = args.output.resolve()
    common = ['--arm', args.arm, '--base', str(args.base), '--split', 'pilot']
    groups = []
    for mode in MODES:
        jobs = []
        for shard in range(shards):
            command = [sys.executable, 'scripts/experiments/prefeval_k1_teacher.py', *common,
                       '--reader', str(args.reader), '--prefeval', str(args.prefeval),
                       '--output', str(root / mode / 'teachers'), '--supervision', mode,
                       '--steps', str(steps), '--shard', str(shard), '--shards', str(shards)]
            if mode != 'hard_ce':
                command += ['--teacher-cache', str(root / 'shared-teacher-targets'),
                            '--teacher-max-new-tokens', str(args.teacher_max_new_tokens)]
            if smoke:
                command += ['--limit', '1']
            jobs.append({'name': f'teacher-{mode}-{shard}', 'gpu': shard, 'command': command})
        groups.append(jobs)
    if smoke:
        groups.append([{'name': 'technical-png-readback', 'gpu': 0,
                        'command': [sys.executable, 'scripts/probes/prompt_matching_smoke_readback.py',
                                    '--run', str(root), '--reader', str(args.reader), '--arm', args.arm]}])
    else:
        # PNG reopening uses the unchanged evaluator. Smoke controls exclude mismatch:
        # a one-example donor would equal the original memory and is not a valid control.
        for mode in MODES:
            jobs = []
            for shard in range(shards):
                command = [sys.executable, 'scripts/experiments/prefeval_k1_evaluate.py',
                           '--reader', str(args.reader), '--images', str(root / mode / 'teachers'),
                           '--output', str(root / mode / 'teacher-readback'), '--kind', 'teacher',
                           '--split', 'pilot', '--tasks', 'mcq', '--prefixes', '0',
                           '--families', 'T1,T2,T3,O1,O2', '--noise-chains', '1',
                           '--controls', 'memory,blank,text' if smoke else 'memory,blank,mismatch,text',
                           '--shard', str(shard), '--shards', str(shards)]
                if smoke:
                    command += ['--limit', '1']
                jobs.append({'name': f'teacher-readback-{mode}-{shard}', 'gpu': shard, 'command': command})
            groups.append(jobs)
    if not smoke:
        writer_jobs = []
        for i, mode in enumerate(MODES):
            command = [sys.executable, 'scripts/experiments/prefeval_k1_writer.py', 'train', *common,
                       '--official-source', str(args.official_source), '--checkpoint', str(args.checkpoint),
                       '--teachers', str(root / mode / 'teachers'), '--teacher-supervision', mode,
                       '--output', str(root / mode / 'writer'), '--steps', '2048', '--stage', 'write']
            writer_jobs.append({'name': f'writer-{mode}', 'gpu': i % 2, 'command': command})
        groups += [writer_jobs[:2], writer_jobs[2:]]
        for split in ('pilot', 'dev'):
            jobs = []
            for i, mode in enumerate(MODES):
                command = [sys.executable, 'scripts/experiments/prefeval_k1_writer.py', 'rollout',
                           '--arm', args.arm, '--base', str(args.base), '--official-source', str(args.official_source),
                           '--checkpoint', str(root / mode / 'writer' / 'checkpoint-final.pt'),
                           '--output', str(root / mode / f'student-{split}'), '--split', split,
                           '--inter-turns', '0', '--noise-chains', '2']
                jobs.append({'name': f'rollout-{mode}-{split}', 'gpu': i % 2, 'command': command})
            groups += [jobs[:2], jobs[2:]]
            for mode in MODES:
                jobs = []
                for shard in range(2):
                    command = [sys.executable, 'scripts/experiments/prefeval_k1_evaluate.py',
                               '--reader', str(args.reader), '--images', str(root / mode / f'student-{split}'),
                               '--output', str(root / mode / f'student-readback-{split}'), '--kind', 'student',
                               '--split', split, '--tasks', 'mcq', '--prefixes', '0', '--noise-chains', '2',
                               '--families', 'T1,T2,T3,O1,O2', '--controls', 'memory,blank,mismatch,text',
                               '--shard', str(shard), '--shards', '2']
                    jobs.append({'name': f'student-readback-{mode}-{split}-{shard}', 'gpu': shard, 'command': command})
                groups.append(jobs)
    return {'schema': 'vision_memory.prompt-matching-run.v1',
            'phase': args.phase, 'arm': args.arm, 'teacher_steps': steps,
            'writer_steps': None if smoke else 2048,
            'modes': list(MODES), 'teacher_reference': 'gray_128/255',
            'teacher_history': 'initial observed exchange only',
            'temperature': 1.0, 'groups': groups,
            'scope': 'technical_only' if smoke else 'pilot64_and_exposed_internal_dev90_not_sealed_OOD',
            'promotion': 'never automatic from this pilot; free-response, long-chain and held-out confirmation required'}


def run_job(job, output, deadline):
    result = output / 'receipts' / (job['name'] + '.json')
    command_hash = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    if result.exists():
        old = json.loads(result.read_text())
        if old['command_sha256'] != command_hash:
            raise ValueError('Attempt to resume a different command')
        if old['exit_code'] == 0:
            return old
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']), PYTHONUNBUFFERED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONHASHSEED='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    started = time.time()
    with (output / 'logs' / (job['name'] + '.log')).open('a') as log:
        process = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=max(1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code = 124
    receipt = dict(name=job['name'], command_sha256=command_hash, exit_code=code,
                   started=started, finished=time.time())
    write_json(result, receipt)
    if code:
        raise RuntimeError(f"{job['name']} failed with {code}; see its log")
    return receipt


def main(args):
    if args.max_hours <= 0:
        raise ValueError('A positive wall-clock limit is required')
    plan = build_plan(args)
    plan['commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Deploy a clean committed checkout before training')
    args.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.output / 'launcher.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.output / 'plan.json'
        if manifest.exists() and json.loads(manifest.read_text()) != plan:
            raise ValueError('Output directory belongs to a different experiment')
        write_json(manifest, plan)
        for name in ('receipts', 'logs'):
            (args.output / name).mkdir(exist_ok=True)
        deadline = time.monotonic() + args.max_hours * 3600
        write_json(args.output / 'status.json', {'status': 'running', 'commit': plan['commit']})
        try:
            for group in plan['groups']:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Experiment wall-clock budget exhausted')
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(run_job, job, args.output, deadline) for job in group]
                    for future in futures:
                        print(json.dumps(future.result()), flush=True)
            write_json(args.output / 'status.json', {'status': 'completed', 'commit': plan['commit'],
                                                    'promotion': 'pending independent evidence review'})
        except BaseException as error:
            write_json(args.output / 'status.json', {'status': 'failed', 'error': str(error), 'commit': plan['commit']})
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=['smoke', 'pilot'], required=True)
    p.add_argument('--arm', choices=['A', 'B'], default='B')
    for name in ('base', 'reader', 'prefeval', 'official-source', 'checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--teacher-max-new-tokens', type=int, default=512)
    p.add_argument('--max-hours', type=float, default=8)
    p.add_argument('--dry-run', action='store_true')
    main(p.parse_args())
