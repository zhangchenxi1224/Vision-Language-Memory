"""Resume only collection/CLI after the sealed e372 continuation-label failure."""
import argparse
import hashlib
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
GENERATION = 'e372f3cf7330ffbfbfe6570a75cd9b80c6011dbc'
PARENT_COMMIT = '4fbc85725d78427235757ace2661d086b896a97f'
PARENT = RUNS / '4fbc857-clear-retention-full4832'
SOURCE = PROJECT / 'repos/dreamlite-clear-retention-validation-20260914'
PREFIX = 'e372f3c-fresh-wording'


def read(path):
    return json.loads(path.read_bytes())


def clean(root, commit):
    if (len(commit) != 40 or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() != commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip()):
        raise ValueError('Require the exact clean source checkout')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    a = parser.parse_args()
    clean(ROOT, a.expected_commit)
    clean(SOURCE, GENERATION)
    if not math.isfinite(a.deadline_unix) or time.time() >= a.deadline_unix:
        raise ValueError('Require a finite future deadline')
    status = RUNS / (PREFIX + '-completion-suite-status.json')
    import fcntl
    lock = (RUNS / (PREFIX + '-completion-suite.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = status.read_bytes()
    failure = json.loads(previous)
    if (failure['state'] != 'failed' or failure['stage'] != 'suite_error'
            or failure['error'] != 'confirmation-collection failed with code 1'
            or failure['validation_commit'] != GENERATION or failure['parent'] != str(PARENT)):
        raise ValueError('This recovery only handles the inspected fixed collection failure')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('Require actually idle GPUs after the failed drivers exited')
    outputs = {lane: RUNS / (PREFIX + '-' + lane) for lane in ('confirmation', 'chains', 'prefix0', 'prefix1')}
    checkpoint = read(PARENT / 'train/result.json')['checkpoint_sha256']
    for output in outputs.values():
        value = read(output / 'complete.json')
        identity = value['identity']
        if (identity['probe_commit'] != GENERATION or identity['parent_commit'] != PARENT_COMMIT
                or identity['checkpoint_sha256'] != checkpoint
                or identity['interpretation'] != 'observed_wording_regression_seen_semantic_questions'
                or read(output / 'identity.json') != identity):
            raise ValueError('Existing generated evidence differs from the inspected recovery target')
        if Path(str(output) + '-summary.json').exists() or Path(str(output) + '-evidence.tgz').exists():
            raise ValueError('A collector already produced evidence; inspect instead of overwriting')
    package, prepared, inference = [RUNS / (PREFIX + '-' + label) for label in ('package', 'parity', 'inference')]
    if any(path.exists() for path in (package, prepared, inference)):
        raise ValueError('CLI evidence already exists')
    if (RUNS / (a.expected_commit[:7] + '-png-readback-status.json')).exists():
        raise ValueError('Recovery PNG suite already exists')
    preserved = RUNS / (PREFIX + '-initial-failure.json')
    with preserved.open('xb') as stream:
        stream.write(previous)
    recovery = {'collector_commit': a.expected_commit, 'generation_commit': GENERATION,
        'generation_source_root': str(SOURCE), 'preserved_failure': str(preserved),
        'preserved_failure_sha256': hashlib.sha256(previous).hexdigest(),
        'policy': 'Reuse all sealed generated rows and images; no regeneration, scoring change or checkpoint selection.'}
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
        'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}

    def record(stage, state, **extra):
        temporary = status.with_suffix('.recovery.tmp')
        temporary.write_text(json.dumps({'stage': stage, 'state': state, 'time_unix': time.time(),
            'deadline_unix': a.deadline_unix, 'validation_commit': GENERATION, 'parent': str(PARENT),
            'recovery': recovery, **extra}, indent=2) + '\n')
        temporary.replace(status)

    def execute(root, script, arguments, label, gpu=False):
        remaining = a.deadline_unix - time.time()
        if remaining <= 0:
            raise TimeoutError('Recovery deadline reached')
        command = [sys.executable, '-u', str(root / script), *map(str, arguments)]
        record(label, 'running')
        log_path = RUNS / (a.expected_commit[:7] + '-recovery-' + label + '.log')
        with log_path.open('x') as log:
            child = subprocess.Popen(command, cwd=root, env=env if gpu else {**env, 'CUDA_VISIBLE_DEVICES': ''},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=remaining)
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                raise
        if code:
            raise RuntimeError(label + ' failed with code ' + str(code))

    try:
        summaries = {}
        for lane, output in outputs.items():
            execute(ROOT, 'scripts/reporting/collect_broader_validation.py', ['--run', output,
                '--parent', PARENT, '--bank', PARENT / 'bank/manifest.json', '--output-prefix', output,
                '--expected-probe-commit', GENERATION, '--logical-sampling-commit', PARENT_COMMIT,
                '--validation-set', 'fresh_wording_v1', '--generation-source-root', SOURCE], lane + '-collection')
            summaries[lane] = read(Path(str(output) + '-summary.json'))
        execute(SOURCE, 'scripts/inference/export_rgb_writer.py', ['--parent-run', PARENT, '--output', package], 'package-export')
        execute(SOURCE, 'scripts/probes/rgb_package_parity.py', ['prepare', '--reference', outputs['chains'],
            '--package', package, '--output', prepared, '--broader', '--logical-sampling-commit', PARENT_COMMIT], 'package-prepare')
        execute(SOURCE, 'scripts/inference/rgb_memory.py', ['--package', package,
            '--base-model', MODELS / 'DreamLite-base-a9a0f15-20260907',
            '--official-source', PROJECT / 'Vision-Language-Memory/third_party/DreamLite',
            '--reader-model', MODELS / 'Qwen3-VL-4B-Instruct', '--commands', prepared / 'commands.jsonl',
            '--output', inference], 'independent-package-inference', gpu=True)
        execute(SOURCE, 'scripts/probes/rgb_package_parity.py', ['verify', '--prepared', prepared,
            '--inference', inference], 'package-parity')
        record('all_registered_workloads_finished', 'completed',
            functional_all_registered_correct=all(x['all_generated_correct_eos'] for x in summaries.values()),
            matched_results={k: [v['matched_correct_eos'], v['matched_rows']] for k, v in summaries.items()},
            scope='Previously observed complete expression regression, recovered collection plus real original-source CLI replay.')
    except BaseException as error:
        record('recovery_error', 'failed', error=repr(error), functional_success=False)
        raise
    # A distinct new PNG run preserves the original failed PNG status unchanged.
    command = [sys.executable, '-u', str(ROOT / 'scripts/inspire/run_png_readback_suite.py'),
        '--runs', str(RUNS), '--reader-model', str(MODELS / 'Qwen3-VL-4B-Instruct'),
        '--expected-commit', a.expected_commit, '--continuation-validation-commit', GENERATION,
        '--deadline-unix', str(a.deadline_unix)]
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())
