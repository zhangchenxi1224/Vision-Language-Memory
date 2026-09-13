"""Finish the fixed4832 endpoint and run four disjoint validation lanes plus CLI replay."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUNS = PROJECT / 'runs/dreamlite-official-alignment'
MODELS = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--expected-commit', required=True)
    a = parser.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix <= time.time():
        raise ValueError('A finite future suite deadline is required')
    if (len(a.expected_commit) != 40 or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != a.expected_commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()):
        raise ValueError('Require the explicitly locked clean validation source')
    parent = RUNS / '84cdfdb-broader151-full4832'
    prefix = a.expected_commit[:7] + '-broader'
    status = RUNS / 'broader151-completion-suite-status.json'
    lock = (RUNS / 'broader151-completion-suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status.exists():
        raise ValueError('Suite already has evidence; inspect before resuming any stage')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    cpu_env = {**env, 'CUDA_VISIBLE_DEVICES': ''}
    def record(stage, state, **extra):
        temporary = status.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'stage': stage, 'state': state, 'time_unix': time.time(),
            'deadline_unix': a.deadline_unix, 'validation_commit': a.expected_commit, 'parent': str(parent), **extra}, indent=2) + '\n')
        temporary.replace(status)
    def stop(child):
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
        command = [sys.executable, '-u', str(ROOT / script), *map(str, arguments)]
        record(label, 'running', command=command)
        with (RUNS / (prefix + '-' + label + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, env=cpu_env if cpu else env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = child.wait(timeout=remaining)
            except BaseException:
                stop(child)
                raise
        if returncode:
            stop(child)
            raise RuntimeError(label + ' failed with code ' + str(returncode))
    lanes = [('single_writes', 'confirmation', 0, None), ('rgb_chains', 'chains', 1, None),
             ('historical_prefixes', 'prefix0', 2, 0), ('historical_prefixes', 'prefix1', 3, 1)]
    outputs = {label: RUNS / (prefix + '-' + label) for _, label, _, _ in lanes}
    package, prepared, inference = [RUNS / (prefix + '-' + label) for label in ('package', 'parity', 'inference')]
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
                raise TimeoutError('GPUs remain occupied; no work was evicted')
            time.sleep(10)
        if a.deadline_unix - time.time() < 60 * 60:
            raise TimeoutError('Require one hour for the complete four-lane validation and package replay')
        if any(path.exists() for path in (*outputs.values(), package, prepared, inference)):
            raise ValueError('Output exists; refuse duplicate execution')
        bank = parent / 'bank/manifest.json'
        execute('scripts/reporting/collect_broader_endpoint.py', ['--run', parent, '--bank', bank,
            '--output-prefix', RUNS / (prefix + '-endpoint')], 'endpoint-collection')
        children, logs = [], []
        try:
            for mode, label, device, lane in lanes:
                command = [sys.executable, '-u', str(ROOT / 'scripts/probes/official_broader_confirmation.py'),
                    '--parent-run', str(parent), '--output', str(outputs[label]), '--mode', mode,
                    '--device', str(device), '--deadline-unix', str(a.deadline_unix), '--diagnostic']
                if lane is not None:
                    command += ['--prefix-lane', str(lane)]
                log = (RUNS / (prefix + '-' + label + '.log')).open('w')
                logs.append(log)
                children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            record('four_independent_validation_lanes', 'running', worker_pids=[child.pid for child in children],
                outputs={key: str(value) for key, value in outputs.items()})
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError('An independent lane failed; retain all partial evidence')
                if time.time() >= a.deadline_unix:
                    raise TimeoutError('Independent validation reached its deadline')
                time.sleep(10)
            if any(child.returncode for child in children):
                raise RuntimeError('Independent validation failed')
        finally:
            for child in children:
                if child.poll() is None or child.returncode:
                    stop(child)
            for log in logs:
                log.close()
        summaries = {}
        for label, output in outputs.items():
            execute('scripts/reporting/collect_broader_validation.py', ['--run', output, '--parent', parent,
                '--bank', bank, '--output-prefix', output, '--expected-probe-commit', a.expected_commit], label + '-collection')
            summaries[label] = json.loads(Path(str(output) + '-summary.json').read_bytes())
        execute('scripts/inference/export_rgb_writer.py', ['--parent-run', parent, '--output', package], 'package-export')
        execute('scripts/probes/rgb_package_parity.py', ['prepare', '--reference', outputs['chains'], '--package', package,
            '--output', prepared, '--broader'], 'package-prepare')
        execute('scripts/inference/rgb_memory.py', ['--package', package,
            '--base-model', MODELS / 'DreamLite-base-a9a0f15-20260907',
            '--official-source', PROJECT / 'Vision-Language-Memory/third_party/DreamLite',
            '--reader-model', MODELS / 'Qwen3-VL-4B-Instruct', '--commands', prepared / 'commands.jsonl', '--output', inference],
            'independent-package-inference', cpu=False)
        execute('scripts/probes/rgb_package_parity.py', ['verify', '--prepared', prepared, '--inference', inference], 'package-parity')
        record('all_registered_workloads_finished', 'completed',
            functional_all_registered_correct=all(value['all_generated_correct_eos'] for value in summaries.values()),
            matched_results={label: [value['matched_correct_eos'], value['matched_rows']] for label, value in summaries.items()},
            scope='Seen questions; fresh transition expressions and noise, historical original/reworded full prefixes. Not unseen entities or simultaneous multi-fact retention.')
        return 0
    except BaseException as error:
        record('suite_error', 'failed', error=str(error), functional_success=False)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
