"""Launch the preregistered same-baseline logical-condition sampling comparison."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.refine_historical_writer_targets import P, M, clean_source, write
from scripts.experiments.logical_sampling_protocol import plan, BANK_SHA, REFERENCE_RESULT
from scripts.experiments.broader_writer_protocol import PARENT_PACKAGE, PARENT_CHECKPOINT, SEED
from scripts.reporting.collect_transition_endpoint import sha, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    a = parser.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix - time.time() < 150 * 60:
        raise ValueError('Reserve150 minutes for the complete paired training run')
    clean_source(a.expected_commit)
    runs = P / 'runs/dreamlite-official-alignment'
    reference = runs / '84cdfdb-broader151-full4832'
    bank_path = reference / 'bank/manifest.json'
    package = runs / '1201efe-four-gpu-warm-package'
    if (sha(bank_path) != BANK_SHA or sha(reference / 'train/result.json') != REFERENCE_RESULT
            or read(reference / 'terminal.json')['state'] != 'completed'
            or sha(package / 'manifest.json') != PARENT_PACKAGE
            or read(package / 'manifest.json')['parent_checkpoint_sha256'] != PARENT_CHECKPOINT):
        raise ValueError('The sealed paired bank, reference or original initialization differs')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPUs are occupied; inspect existing work before dispatch')
    registered = plan(read(bank_path), a.expected_commit)
    a.output.mkdir(parents=True, exist_ok=False)
    # Preserve actual input bytes in the evidence archive while training reads
    # the exact same source path as the reference for runtime identity equality.
    (a.output / 'bank').mkdir()
    for name in ('manifest.json', 'complete.json'):
        (a.output / 'bank' / name).write_bytes((bank_path.parent / name).read_bytes())
    write(a.output / 'preregistered-experiment.json', registered)
    command = [sys.executable, '-u', str(ROOT / 'scripts/inspire/run_official_alignment_pilot.py'),
        '--bank-manifest', str(bank_path), '--bank-sha256', BANK_SHA,
        '--dreamlite', str(M / 'DreamLite-base-a9a0f15-20260907'), '--model-variant', 'base',
        '--teacher-dreamlite', str(M / 'DreamLite-mobile'), '--official-source', str(P / 'Vision-Language-Memory/third_party/DreamLite'),
        '--base-manifest', str(runs / 'base-complete-snapshot-seal.json'), '--base-guidance-scale', '1',
        '--reader', str(M / 'Qwen3-VL-4B-Instruct'), '--output-dir', str(a.output), '--expected-commit', a.expected_commit,
        '--target-mode', 'single', '--steps', '4832', '--eval-seeds', '2', '--trainable-scope', 'full_unet',
        '--checkpoint-interval', '16', '--colocate-models', '--data-parallel-world-size', '4', '--seed', str(SEED),
        '--deadline-unix', str(a.deadline_unix), '--initial-writer-package', str(package),
        '--initial-writer-package-sha256', PARENT_PACKAGE, '--sampling-strategy', 'logical_condition',
        '--initial-baseline-match', str(reference), '--initial-baseline-match-result-sha256', REFERENCE_RESULT]
    def record(state, **extra):
        write(a.output / 'comparison-driver-status.json', {'state': state, 'time_unix': time.time(),
            'deadline_unix': a.deadline_unix, 'commit': a.expected_commit, **extra})
    def terminate(signum, frame):
        raise TimeoutError('Comparison terminated; preserve all evidence')
    signal.signal(signal.SIGTERM, terminate)
    child = None
    try:
        clean_source(a.expected_commit)
        with (a.output / 'pilot.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record('running', pilot_pid=child.pid, command=command, plan_sha256=sha(a.output / 'preregistered-experiment.json'))
            code = child.wait(timeout=a.deadline_unix - time.time())
        if code or read(a.output / 'terminal.json')['state'] != 'completed':
            raise RuntimeError('Fixed comparison did not complete; inspect baseline gate and retained evidence')
        record('completed', functional_success_requires_validation=True)
    except BaseException as error:
        record('failed', error=str(error), functional_success=False)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    main()
