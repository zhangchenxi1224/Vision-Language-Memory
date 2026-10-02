"""Dispatch the fixed additional2880-update experiment from the sealed endpoint."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deadline-unix', type=float, required=True)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not math.isfinite(args.deadline_unix) or args.deadline_unix <= 0:
        raise ValueError('An explicit finite positive lease deadline is required')
    from scripts.experiments.transition_warm_start_plan import plan, TRAINING_COMMIT
    from vision_memory.dreamlite.writer_package import inspect_package, file_sha
    registered = ROOT / 'reports/official-transition-warm-start-plan-20260913.json'
    design = plan()
    if json.loads(registered.read_bytes()) != design:
        raise ValueError('Preregistered experiment changed')
    plan_sha = file_sha(registered)
    project = Path('/inspire/ssd/project/exploration-topic/czxs26210936')
    models = Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')
    runs = project / 'runs/dreamlite-official-alignment'
    source = project / 'repos/dreamlite-writer-warm-start-20260913'
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != TRAINING_COMMIT
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True).strip()):
        raise ValueError('Require the fixed clean warm-start training implementation')
    package = runs / '281b653-transition-writer-package'
    manifest = inspect_package(package)
    for key, expected in {'parent_checkpoint_sha256': design['parent_checkpoint_sha256'],
        'parent_result_sha256': design['parent_result_sha256'], 'optimizer_steps': 2880,
        'training_bank_sha256': design['bank_sha256'], 'conditional_group_count': 45,
        'semantic_question_count': 1, 'guidance_scale': 1.}.items():
        if manifest.get(key) != expected:
            raise ValueError('Initial package is not the registered failed endpoint: ' + key)
    manifest_sha = file_sha(package / 'manifest.json')
    parity = json.loads((runs / '281b653-transition-package-parity/parity-result.json').read_bytes())
    if (parity.get('parity_pass') is not True or parity.get('package_manifest_sha256') != manifest_sha
            or parity.get('writes_compared') != 6 or parity.get('reads_compared') != 30):
        raise ValueError('Require actual independent package replay parity before initializing another experiment')
    if file_sha(runs / '9628d71-transition-wording-bank/manifest.json') != design['bank_sha256']:
        raise ValueError('Preregistered training bank changed')
    output = runs / 'd9a1a11-transition-warm2880-seed20260914'
    command = [sys.executable, '-u', str(source / 'scripts/inspire/run_official_alignment_pilot.py'),
        '--bank-manifest', str(runs / '9628d71-transition-wording-bank/manifest.json'), '--bank-sha256', design['bank_sha256'],
        '--dreamlite', str(models / 'DreamLite-base-a9a0f15-20260907'), '--model-variant', 'base',
        '--teacher-dreamlite', str(models / 'DreamLite-mobile'), '--official-source', str(project / 'Vision-Language-Memory/third_party/DreamLite'),
        '--base-manifest', str(runs / 'base-complete-snapshot-seal.json'), '--base-guidance-scale', '1',
        '--reader', str(models / 'Qwen3-VL-4B-Instruct'), '--output-dir', str(output), '--expected-commit', TRAINING_COMMIT,
        '--target-mode', 'single', '--steps', '2880', '--eval-seeds', '4', '--trainable-scope', 'full_unet',
        '--checkpoint-interval', '16', '--colocate-models', '--seed', '20260914', '--deadline-unix', str(args.deadline_unix),
        '--initial-writer-package', str(package), '--initial-writer-package-sha256', manifest_sha]
    binding = {'plan_file_sha256': plan_sha, 'initial_package_manifest_sha256': manifest_sha,
        'parent_checkpoint_sha256': manifest['parent_checkpoint_sha256'],
        'initial_package_parity_sha256': file_sha(runs / '281b653-transition-package-parity/parity-result.json'), 'command': command}
    print(json.dumps(binding), flush=True)
    if args.dry_run:
        return 0
    if args.deadline_unix - time.time() < 200 * 60:
        raise ValueError('Require at least200min within a fresh idle H200 lease for measured baseline, training and final')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'preregistered-experiment.json').write_bytes(registered.read_bytes())
    (output / 'initialization-launch.json').write_text(json.dumps(binding, indent=2, sort_keys=True) + '\n')
    result = subprocess.run(command, cwd=source)
    if file_sha(registered) != plan_sha or file_sha(package / 'manifest.json') != manifest_sha:
        raise RuntimeError('Experiment plan or initial package changed during execution')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
