"""Run the entire fixed logical comparison matrix sequentially on one idle H200.

Only concurrency/device placement changes. Probe and collector source is an
explicit immutable checkout; all original rows, cases, seeds and failures remain.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

P = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUNS = P / 'runs/dreamlite-official-alignment'
M = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
COMMIT = '16bc3d077fa7604a0d84c3b008a80baf327b265b'
TRAINING_COMMIT = 'bb34092ab0d1292c87d16d9632716b218f54054b'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    a = parser.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix - time.time() < 120 * 60:
        raise ValueError('Reserve at least two hours for the complete sequential validation')
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.source, text=True).strip() != COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=a.source, text=True).strip()):
        raise ValueError('Require the exact clean16bc3d0 validation source')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True)
    rows = [row.split(',') for row in gpu.strip().splitlines()]
    if len(rows) != 1 or 'H200' not in rows[0][0] or int(rows[0][1]) < 140000 or int(rows[0][2]) > 100:
        raise ValueError('Require one idle full-memory H200; never evict another workload')
    parent = RUNS / 'bb34092-logical31-full4832'
    status = RUNS / '16bc3d0-logical-serial-suite-status.json'
    lock = (RUNS / '16bc3d0-logical-completion-suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    names = ('confirmation', 'chains', 'prefix0', 'prefix1', 'package', 'parity', 'inference')
    outputs = {name: RUNS / ('16bc3d0-logical-' + name) for name in names}
    if status.exists() or (RUNS / '16bc3d0-logical-completion-suite-status.json').exists() or any(path.exists() for path in outputs.values()):
        raise ValueError('Existing suite evidence: inspect it, do not duplicate or overwrite validation')
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
    cpu_env = {**env, 'CUDA_VISIBLE_DEVICES': ''}
    bank = parent / 'bank/manifest.json'
    protocol = ['--logical-sampling-commit', TRAINING_COMMIT]
    child = None
    def record(stage, state, **extra):
        value = {'stage': stage, 'state': state, 'time_unix': time.time(), 'deadline_unix': a.deadline_unix,
            'training_commit': TRAINING_COMMIT, 'validation_commit': COMMIT, 'parent': str(parent),
            'orchestrator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'execution': 'all four original matrices, sequential device0; zero optimization', **extra}
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2) + '\n')
        temporary.replace(status)
    def stop():
        if child is not None and child.poll() != 0:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    def terminate(signum, frame):
        raise TimeoutError('Sequential validation interrupted; preserve all evidence')
    signal.signal(signal.SIGTERM, terminate)
    def execute(script, arguments, label, *, cpu=True):
        nonlocal child
        if a.deadline_unix <= time.time():
            raise TimeoutError('Validation deadline reached')
        command = [sys.executable, '-u', str(a.source / script), *map(str, arguments)]
        with (RUNS / ('16bc3d0-logical-' + label + '.log')).open('w') as log:
            child = subprocess.Popen(command, cwd=a.source, env=cpu_env if cpu else env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record(label, 'running', worker_pid=child.pid, command=command)
            code = child.wait(timeout=a.deadline_unix - time.time())
        if code:
            raise RuntimeError(label + ' failed: ' + str(code))
    try:
        execute('scripts/reporting/collect_broader_endpoint.py', ['--run', parent, '--bank', bank,
            '--output-prefix', RUNS / '16bc3d0-logical-endpoint', *protocol], 'endpoint-collection')
        summaries = {}
        for mode, label, lane in [('single_writes', 'confirmation', None), ('rgb_chains', 'chains', None),
                ('historical_prefixes', 'prefix0', 0), ('historical_prefixes', 'prefix1', 1)]:
            arguments = ['--parent-run', parent, '--output', outputs[label], '--mode', mode,
                '--device', '0', '--deadline-unix', a.deadline_unix, '--diagnostic', *protocol]
            if lane is not None:
                arguments += ['--prefix-lane', lane]
            execute('scripts/probes/official_broader_confirmation.py', arguments, label, cpu=False)
            execute('scripts/reporting/collect_broader_validation.py', ['--run', outputs[label], '--parent', parent,
                '--bank', bank, '--output-prefix', outputs[label], '--expected-probe-commit', COMMIT, *protocol], label + '-collection')
            summaries[label] = json.loads(Path(str(outputs[label]) + '-summary.json').read_bytes())
        execute('scripts/inference/export_rgb_writer.py', ['--parent-run', parent, '--output', outputs['package']], 'package-export')
        execute('scripts/probes/rgb_package_parity.py', ['prepare', '--reference', outputs['chains'], '--package', outputs['package'],
            '--output', outputs['parity'], '--broader', *protocol], 'package-prepare')
        execute('scripts/inference/rgb_memory.py', ['--package', outputs['package'], '--base-model', M / 'DreamLite-base-a9a0f15-20260907',
            '--official-source', P / 'Vision-Language-Memory/third_party/DreamLite', '--reader-model', M / 'Qwen3-VL-4B-Instruct',
            '--commands', outputs['parity'] / 'commands.jsonl', '--output', outputs['inference']], 'package-inference', cpu=False)
        execute('scripts/probes/rgb_package_parity.py', ['verify', '--prepared', outputs['parity'], '--inference', outputs['inference']], 'package-parity')
        record('all_registered_workloads_finished', 'completed',
            functional_all_registered_correct=all(value['all_generated_correct_eos'] for value in summaries.values()),
            matched_results={label: [value['matched_correct_eos'], value['matched_rows']] for label, value in summaries.items()},
            scope='Observed cases reused as a paired diagnostic; not fresh holdouts or a usability certification.')
    except BaseException as error:
        record('suite_error', 'failed', error=str(error), functional_success=False)
        raise
    finally:
        stop()


if __name__ == '__main__':
    main()
