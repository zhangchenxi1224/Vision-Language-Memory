"""Recount a complete GPU source-runtime preflight archive without claiming model acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def verify(archive, expected_sha, probe_commit, training_commit, run_name=None):
    from scripts.reporting.verify_generated_source_draws import verify_seal, verify_draw, identity_binding
    from scripts.probes.generated_source_training_preflight import selected_draws, PARAMETERS_SHA, RUNTIME_SHA
    from vision_memory.training.latent_bank_unet import balanced_draw
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise ValueError('Archive differs from the independently observed remote digest')
    sources = {'b82228e4689f08be925452700d12b1da947f4e9a': 'ef163b26e33f62c496ed0da8744ebb7bf1163873',
        'c83aca055df8bb71b11e28ac242f0471f516d8da': 'b62ec027ad725aeb6ecc772aa85e7a3ff6e49b36'}
    if sources.get(probe_commit) != training_commit:
        raise ValueError('Unregistered probe/training source pair')
    base_name = probe_commit[:7]+'-source-runtime-preflight'
    run_name = run_name or base_name
    if run_name not in (base_name, base_name+'-r2'):
        raise ValueError('Unregistered retry output name')
    read = lambda path: json.loads(path.read_bytes())
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp).resolve()
        with tarfile.open(archive) as tar:
            seen = set()
            for member in tar.getmembers():
                path = (directory/member.name).resolve()
                if not path.is_relative_to(directory) or path in seen:
                    raise ValueError('Unsafe or duplicate archive member')
                seen.add(path)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile() or path.suffix not in ('.json', '.log'):
                    raise ValueError('Unexpected evidence member type')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(tar.extractfile(member).read())
        run = directory/run_name
        status = read(run/'status.json')
        if status['state'] not in ('completed', 'failed') or status['optimizer_updates'] != 0:
            raise ValueError('Require terminal zero-update preflight evidence')
        if not all((run/name).exists() for name in ('runtime.json', 'training-source-augmentation.json', 'gradient-draws.json')):
            if status['state'] != 'failed' or (run/'result.json').exists():
                raise ValueError('Incomplete runtime evidence cannot support completion')
            return {'archive_sha256': expected_sha, 'archive_bytes': archive.stat().st_size,
                'expected_training_commit': training_commit, 'expected_probe_commit': probe_commit,
                'run_name': run_name, 'execution_status': status, 'preflight_passed': False,
                'artifacts': sorted(p.name for p in run.iterdir()),
                'scope': 'Preserved failed execution before complete runtime/gradient evidence. No source-encoding, gradient, optimization or functional success claim.'}
        results = ROOT/'reports/official-alignment-results-20260913'
        bank = read(results/'broader151-bank-manifest.json')
        name = ('ef163b2-superseded-' if training_commit.startswith('ef163b2') else '')+'generated-source-training-preregistered.json'
        registered = read(results/name)
        if registered['training_commit'] != training_commit:
            raise ValueError('Wrong independent training registration')
        runtime, seal = read(run/'runtime.json'), read(run/'training-source-augmentation.json')
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        if digest(run/'runtime.json') != RUNTIME_SHA:
            raise ValueError('Canonical runtime changed')
        # Expected pool identity is supplied here; no synthetic training identity is claimed as observed.
        seal_proof = verify_seal(bank, {'generated_source_pool': identity_binding()}, runtime, seal, registered)
        draws, indices = read(run/'gradient-draws.json'), selected_draws(bank)
        if ([d['draw_index'] for d in draws] != indices or read(run/'selected-draws.json')['indices'] != indices or len(draws) != 27):
            raise ValueError('Incomplete or changed source-state/choice probe coverage')
        for draw in draws:
            group, teacher, noise, sigma = balanced_draw(bank['groups'], 20260915, draw['draw_index'], sampling_strategy='logical_condition')
            if ((draw['question_id'], draw['teacher_id'], draw['noise_seed'], draw['effective_sigma']) != (group['question_id'], teacher, noise, sigma)
                    or not math.isfinite(draw['flow_matching_mse'])):
                raise ValueError('Changed draw or nonfinite actual loss')
            verify_draw(group, draw, draw['draw_index'], seal)
        passed = status['state'] == 'completed'
        if passed:
            result = read(run/'result.json')
            expected = {'status': 'preflight_passed', 'optimizer_updates': 0, 'actual_gradient_draws': 27,
                'source_conditions': 108, 'source_condition_pairs': 972, 'canonical_runtime_sha256': RUNTIME_SHA,
                'training_commit': training_commit, 'probe_commit': probe_commit, 'parameters_before_after_sha256': PARAMETERS_SHA,
                'source_seal_sha256': digest(run/'training-source-augmentation.json'), 'gradient_draws_sha256': digest(run/'gradient-draws.json')}
            if any(result.get(k) != v or status.get(k) != v for k, v in expected.items()):
                raise ValueError('Completion, parameter identity or executed source bindings differ')
        elif (run/'result.json').exists():
            raise ValueError('Failed preflight has a contradictory success result')
        return {'archive_sha256': expected_sha, 'archive_bytes': archive.stat().st_size,
            'expected_training_commit': training_commit, 'expected_probe_commit': probe_commit,
            'execution_status': status, 'preflight_passed': passed, 'actual_draws_recounted': 27,
            'canonical_runtime_sha256': RUNTIME_SHA, 'source_seal': seal_proof,
            'scope': 'Complete local metadata recount: all27 draws and972 source/condition bindings. Native tensors, gradients and unchanged parameters were checked by the GPU probe; they are not independently regenerated by this CPU verifier. No optimizer or functional acceptance claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--probe-commit', required=True)
    parser.add_argument('--training-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-name')
    args = parser.parse_args()
    result = verify(args.archive, args.sha256, args.probe_commit, args.training_commit, args.run_name)
    args.output.write_bytes((json.dumps(result, indent=2, sort_keys=True)+'\n').encode())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
