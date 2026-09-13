"""Wait for all16 qualified targets, seal expanded conditions and train four GPUs."""
import argparse
import json
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
from scripts.experiments.build_broader_writer_bank import build
from scripts.experiments.broader_writer_protocol import training_plan, PARENT_CHECKPOINT, PARENT_PACKAGE, SEED
from scripts.reporting.collect_transition_endpoint import sha, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    a = parser.parse_args()
    if not math.isfinite(a.deadline_unix) or a.deadline_unix - time.time() < 150 * 60:
        raise ValueError('Reserve at least150 minutes for the complete broader experiment')
    clean_source(a.expected_commit)
    runs = P / 'runs/dreamlite-official-alignment'
    refinement = runs / '918e7d2-five-query-target-refinement'
    package = runs / '1201efe-four-gpu-warm-package'
    parity = runs / '1201efe-four-gpu-warm-parity/parity-result.json'
    if (sha(package / 'manifest.json') != PARENT_PACKAGE
            or read(package / 'manifest.json')['parent_checkpoint_sha256'] != PARENT_CHECKPOINT
            or sha(parity) != 'f3c0e76a6360c6f0d46d088d75599b5b44af252274c7b75ff115a7357256275f'
            or read(parity)['parity_pass'] is not True):
        raise ValueError('Require the actual sealed warm endpoint and its independent package replay')
    a.output.mkdir(parents=True, exist_ok=False)
    def status(state, **extra):
        write(a.output / 'broader-driver-status.json', {'state': state, 'time_unix': time.time(),
            'deadline_unix': a.deadline_unix, 'commit': a.expected_commit, **extra})
    status('waiting_for_complete_teacher_qualification')
    child = None
    def terminate(signum, frame):
        raise TimeoutError('Broader driver received termination; preserve all evidence')
    signal.signal(signal.SIGTERM, terminate)
    try:
        while not (refinement / 'complete.json').exists():
            if (refinement / 'status.json').exists() and read(refinement / 'status.json')['state'] == 'failed':
                raise RuntimeError('Target refinement failed; no Writer optimization started')
            if time.time() >= a.deadline_unix - 150 * 60:
                raise TimeoutError('Insufficient remaining budget for the complete fixed Writer experiment')
            time.sleep(10)
        qualified = read(refinement / 'complete.json')
        if qualified['bank_sealed'] is not True or qualified['correct_eos'] != 160:
            raise ValueError('All16 target images must pass allfive queries in both forms')
        # The actual target workers must release their CUDA contexts first.
        while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
            if time.time() >= a.deadline_unix - 150 * 60:
                raise TimeoutError('GPUs are occupied; no existing work was evicted')
            time.sleep(10)
        seal = build(runs / '9628d71-transition-wording-bank/manifest.json',
            runs / '7ee3a92-historical-writer-bank/manifest.json', refinement, a.output / 'bank',
            a.expected_commit, expand_wordings=True)
        registered = training_plan(seal['manifest_sha256'], a.expected_commit)
        write(a.output / 'preregistered-experiment.json', registered)
        write(a.output / 'parent-evidence.json', {'package_manifest_sha256': PARENT_PACKAGE,
            'parent_checkpoint_sha256': PARENT_CHECKPOINT, 'package_parity_sha256': sha(parity),
            'parent_independent_functional_success': False,
            'parent_failures': 'new event expressions; retained as previous independent failures',
            'refinement_complete_sha256': sha(refinement / 'complete.json')})
        command = [sys.executable, '-u', str(ROOT / 'scripts/inspire/run_official_alignment_pilot.py'),
            '--bank-manifest', str(a.output / 'bank/manifest.json'), '--bank-sha256', seal['manifest_sha256'],
            '--dreamlite', str(M / 'DreamLite-base-a9a0f15-20260907'), '--model-variant', 'base',
            '--teacher-dreamlite', str(M / 'DreamLite-mobile'), '--official-source', str(P / 'Vision-Language-Memory/third_party/DreamLite'),
            '--base-manifest', str(runs / 'base-complete-snapshot-seal.json'), '--base-guidance-scale', '1',
            '--reader', str(M / 'Qwen3-VL-4B-Instruct'), '--output-dir', str(a.output),
            '--expected-commit', a.expected_commit, '--target-mode', 'single', '--steps', str(registered['optimizer_steps']),
            '--eval-seeds', str(registered['eval_seeds']), '--trainable-scope', 'full_unet', '--checkpoint-interval', '16',
            '--colocate-models', '--data-parallel-world-size', '4', '--seed', str(SEED), '--deadline-unix', str(a.deadline_unix),
            '--initial-writer-package', str(package), '--initial-writer-package-sha256', PARENT_PACKAGE]
        clean_source(a.expected_commit)
        with (a.output / 'pilot.log').open('w') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            status('running_fixed_writer_experiment', pilot_pid=child.pid, command=command,
                plan_sha256=sha(a.output / 'preregistered-experiment.json'), bank_sha256=seal['manifest_sha256'])
            returncode = child.wait(timeout=a.deadline_unix - time.time())
        if returncode or read(a.output / 'terminal.json')['state'] != 'completed':
            raise RuntimeError('Fixed Writer experiment did not complete; preserve terminal/checkpoints before any recovery')
        status('completed', terminal_sha256=sha(a.output / 'terminal.json'),
            functional_success_requires_complete_raw_and_independent_validation=True)
        return 0
    except BaseException as error:
        status('failed', error=str(error), functional_success=False)
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
    raise SystemExit(main())
