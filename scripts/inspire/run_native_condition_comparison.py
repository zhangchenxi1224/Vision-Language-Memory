"""Launch the same-budget native Base training-condition comparison."""
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
from scripts.experiments.native_condition_protocol import plan, BANK_SHA, REFERENCE_RESULT, FIRST_STEP_EVIDENCE
from scripts.experiments.broader_writer_protocol import PARENT_PACKAGE, PARENT_CHECKPOINT, SEED
from scripts.reporting.collect_transition_endpoint import sha, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--deadline-unix', type=float, required=True)
    parser.add_argument('--historical-wording-augmentation', action='store_true')
    parser.add_argument('--clear-retention-continuation', action='store_true')
    parser.add_argument('--generated-source-continuation', action='store_true')
    a = parser.parse_args()
    if a.clear_retention_continuation and not a.historical_wording_augmentation:
        raise ValueError('Clear retention continuation requires the registered historical augmentation')
    if a.generated_source_continuation and (not a.historical_wording_augmentation or a.clear_retention_continuation):
        raise ValueError('Generated-source continuation requires historical augmentation and its separate4f initialization')
    if not math.isfinite(a.deadline_unix) or a.deadline_unix - time.time() < 150 * 60:
        raise ValueError('Reserve150 minutes for the complete paired training run')
    clean_source(a.expected_commit)
    runs = P / 'runs/dreamlite-official-alignment'
    reference = runs / 'bb34092-logical31-full4832'
    reference_result, make_plan = REFERENCE_RESULT, plan
    if a.historical_wording_augmentation:
        from scripts.experiments.historical_wording_protocol import plan as wording_plan, REFERENCE_RESULT as wording_reference_result
        reference = runs / '03f8467-native-condition-full4832'
        reference_result, make_plan = wording_reference_result, wording_plan
        prior = read(runs / '5ef8aa8-raw-condition-completion-suite-status.json')
        if prior['state'] != 'completed' or prior['stage'] != 'all_registered_workloads_finished':
            raise ValueError('Complete the full raw-condition experiment before the wording comparison')
    bank_path = runs / '84cdfdb-broader151-full4832/bank/manifest.json'
    if sha(runs / '17f35be-first-step-verified-evidence.tgz') != FIRST_STEP_EVIDENCE:
        raise ValueError('Require the complete CPU-verified first-step diagnostic')
    package = runs / '1201efe-four-gpu-warm-package'
    package_sha, checkpoint_sha, learning_rate = PARENT_PACKAGE, PARENT_CHECKPOINT, 5e-5
    if a.clear_retention_continuation:
        from scripts.experiments.clear_retention_protocol import (plan as retention_plan,
            PARENT_PACKAGE as retention_package, PARENT_CHECKPOINT as retention_checkpoint,
            SOURCE_SWAP_RESULT, LEARNING_RATE)
        diagnostic = runs / 'clear-source-swap-20260914/outputs/result.json'
        if sha(diagnostic) != SOURCE_SWAP_RESULT or read(diagnostic)['diagonal_parity_pass'] is not True:
            raise ValueError('Require the completed source-swap diagnostic before continuation')
        package = runs / '9e27050-logical-package'
        package_sha, checkpoint_sha, learning_rate = retention_package, retention_checkpoint, LEARNING_RATE
        make_plan = retention_plan
    source_pool = None
    if a.generated_source_continuation:
        from scripts.experiments.generated_source_training_protocol import (plan as generated_plan,
            POOL_MANIFEST_SHA, REFERENCE_RESULT as generated_reference_result)
        from scripts.experiments.generated_source_pool_protocol import PACKAGE_SHA, CHECKPOINT_SHA
        source_pool = runs / '90b41a2-generated-source-pool/manifest.json'
        if sha(source_pool) != POOL_MANIFEST_SHA or read(source_pool).get('qualified_for_training') is not True:
            raise ValueError('Require the entire qualified generated-source pool')
        reference = runs / '4fbc857-clear-retention-full4832'
        reference_result = generated_reference_result
        package = runs / 'e372f3c-logical-package'
        package_sha, checkpoint_sha, learning_rate = PACKAGE_SHA, CHECKPOINT_SHA, 1e-5
        make_plan = generated_plan
    if (sha(bank_path) != BANK_SHA or sha(reference / 'train/result.json') != reference_result
            or read(reference / 'terminal.json')['state'] != 'completed'
            or sha(package / 'manifest.json') != package_sha
            or read(package / 'manifest.json')['parent_checkpoint_sha256'] != checkpoint_sha):
        raise ValueError('The sealed paired bank, reference or original initialization differs')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPUs are occupied; inspect existing work before dispatch')
    registered = make_plan(read(bank_path), a.expected_commit)
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
        '--initial-writer-package-sha256', package_sha, '--sampling-strategy', 'logical_condition', '--lr', str(learning_rate),
        '--prompt-style', 'native_base',
        '--initial-baseline-match', str(reference), '--initial-baseline-match-result-sha256', reference_result]
    command.append('--historical-wording-augmentation' if a.historical_wording_augmentation else '--native-condition-baseline-control')
    if a.clear_retention_continuation or a.generated_source_continuation:
        command.extend(['--initial-baseline-reference-phase', 'trained'])
    if source_pool is not None:
        command.extend(['--generated-source-pool', str(source_pool), '--generated-source-pool-sha256', POOL_MANIFEST_SHA])
    def record(state, **extra):
        write(a.output / 'native-condition-driver-status.json', {'state': state, 'time_unix': time.time(),
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
