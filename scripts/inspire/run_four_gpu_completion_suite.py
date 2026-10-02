"""Wait for the fixed four-GPU endpoint, verify it, and run fresh validation."""
import argparse
import fcntl
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUNS = PROJECT / 'runs/dreamlite-official-alignment'
MODELS = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
VALIDATION_COMMIT = '1201efe53f0a8886ff854d03a42432f84bb7fb07'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deadline-unix', type=float, required=True)
    a = p.parse_args()
    if not math.isfinite(a.deadline_unix):
        raise ValueError('A finite validation deadline is required')
    source = PROJECT / 'repos/dreamlite-four-gpu-validation-20260913'
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != VALIDATION_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()):
        raise ValueError('Validation source must be the locked clean commit')
    parent = RUNS / '046c1f1-transition-warm2880-h200x4-seed20260914'
    status = RUNS / 'four-gpu-completion-suite-status.json'
    lock = (RUNS / 'four-gpu-completion-suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status.exists():
        raise ValueError('Suite already has evidence; inspect before resuming any stage')
    def record(stage, state, **extra):
        value = {'stage': stage, 'state': state, 'time_unix': time.time(), 'deadline_unix': a.deadline_unix,
                 'validation_commit': VALIDATION_COMMIT, 'parent': str(parent), **extra}
        temporary = status.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        temporary.replace(status)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    cpu_env = {**env, 'CUDA_VISIBLE_DEVICES': ''}
    def stop_process_group(child):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    def execute(script, arguments, label, *, cpu=True):
        remaining = a.deadline_unix - time.time()
        if remaining <= 0:
            raise TimeoutError('Suite deadline reached')
        command = [sys.executable, '-u', str(source / script), *map(str, arguments)]
        record(label, 'running', command=command)
        with (RUNS / ('four-gpu-' + label + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=source, env=cpu_env if cpu else env,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = child.wait(timeout=remaining)
            except BaseException:
                stop_process_group(child)
                raise
        if returncode:
            stop_process_group(child)
            raise RuntimeError(label + ' failed, return code ' + str(returncode))
    single = RUNS / '1201efe-four-gpu-warm-confirmation'
    chains = RUNS / '1201efe-four-gpu-warm-chains'
    package = RUNS / '1201efe-four-gpu-warm-package'
    prepared = RUNS / '1201efe-four-gpu-warm-parity'
    inference = RUNS / '1201efe-four-gpu-warm-inference'
    try:
        record('waiting_for_fixed_endpoint', 'waiting')
        while not (parent / 'terminal.json').exists():
            if time.time() >= a.deadline_unix:
                raise TimeoutError('Parent did not complete before the suite deadline')
            time.sleep(15)
        terminal = json.loads((parent / 'terminal.json').read_bytes())
        if terminal.get('state') != 'completed':
            record('parent_not_completed', 'needs_attention', parent_terminal=terminal)
            return 75
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            if time.time() >= a.deadline_unix:
                raise TimeoutError('GPUs did not become available; no work was evicted')
            time.sleep(10)
        if a.deadline_unix - time.time() < 45 * 60:
            raise TimeoutError('Require forty-five minutes for the complete fresh validation and replay')
        if any(path.exists() for path in (single, chains, package, prepared, inference)):
            raise ValueError('Validation output exists; refuse duplicate execution')
        bank = RUNS / '9628d71-transition-wording-bank/manifest.json'
        execute('scripts/reporting/collect_transition_endpoint.py', ['--run', parent, '--bank', bank,
                '--output-prefix', RUNS / 'four-gpu-warm-endpoint', '--four-gpu-warm-start'], 'endpoint-collection')
        # Two independent native batch-one jobs on disjoint devices. Each job
        # preserves its full original single-write or six-write-chain matrix.
        children, logs = [], []
        try:
            for mode, output, device in (('single_writes', single, 0), ('rgb_chains', chains, 1)):
                command = [sys.executable, '-u', str(source / 'scripts/probes/official_transition_confirmation.py'),
                    '--parent-run', str(parent), '--output', str(output), '--mode', mode,
                    '--deadline-unix', str(a.deadline_unix), '--four-gpu-warm-start', '--diagnostic', '--device', str(device)]
                log = (RUNS / ('four-gpu-' + mode + '.log')).open('w')
                logs.append(log)
                children.append(subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True))
            record('independent_single_writes_and_rgb_chains', 'running', worker_pids=[child.pid for child in children])
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('An independent validation worker failed; retain both partial outputs')
                if time.time() >= a.deadline_unix:
                    raise TimeoutError('Independent validation reached its deadline')
                time.sleep(10)
            if any(child.returncode for child in children):
                raise RuntimeError('Independent validation failed')
        finally:
            for child in children:
                if child.poll() is None or child.returncode:
                    stop_process_group(child)
            for log in logs:
                log.close()
        for label, output in (('confirmation', single), ('chains', chains)):
            execute('scripts/reporting/collect_transition_validation.py', ['--run', output, '--parent', parent,
                '--bank', bank, '--plan', source / 'reports/official-transition-warm-start-plan-20260913.json',
                '--output-prefix', RUNS / ('four-gpu-warm-' + label), '--four-gpu-warm-start',
                '--expected-probe-commit', VALIDATION_COMMIT], label + '-collection')
        execute('scripts/reporting/verify_rgb_chain_tensors.py', ['--run', chains,
            '--output', RUNS / 'four-gpu-warm-chains-tensor-verification.json'], 'chain-tensor-verification')
        execute('scripts/inference/export_rgb_writer.py', ['--parent-run', parent, '--output', package], 'package-export')
        execute('scripts/probes/rgb_package_parity.py', ['prepare', '--reference', chains, '--package', package,
            '--output', prepared, '--four-gpu-warm-start'], 'package-prepare')
        execute('scripts/inference/rgb_memory.py', ['--package', package,
            '--base-model', MODELS / 'DreamLite-base-a9a0f15-20260907',
            '--official-source', PROJECT / 'Vision-Language-Memory/third_party/DreamLite',
            '--reader-model', MODELS / 'Qwen3-VL-4B-Instruct', '--commands', prepared / 'commands.jsonl', '--output', inference],
            'independent-package-inference', cpu=False)
        execute('scripts/probes/rgb_package_parity.py', ['verify', '--prepared', prepared, '--inference', inference], 'package-parity')
        record('all_registered_workloads_finished', 'completed', functional_success_requires_raw_review=True)
        return 0
    except BaseException as error:
        record('suite_error', 'failed', error=str(error), functional_success=False)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
