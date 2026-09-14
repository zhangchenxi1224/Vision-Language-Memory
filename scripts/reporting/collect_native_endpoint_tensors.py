"""Independently inspect the actual final native-condition development tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

SOURCE = '9e27050ea3fe1e7d54fe81714244f93ae07bac81'
RUNTIME = 'b97bf55f679cb94805ef769b9e748f09a55182a8f36fb9dcb48c6fcfb7bdd1a0'


def check_payload(payload, question, seed, image_sha):
    import torch
    from vision_memory.repro import canonical_tensor_sha256
    if (payload['question_id'], payload['noise_seed']) != (question, seed):
        raise ValueError('Saved condition/noise identity differs')
    trajectory, latent, pixels = payload['trajectory'], payload['latent'], payload['image']
    if (len(trajectory) != 29 or latent.shape != (1, 4, 128, 128)
            or pixels.shape != (1, 3, 1024, 1024)
            or any(value.dtype != torch.float32 or not torch.isfinite(value).all()
                for value in [latent, pixels, *trajectory])
            or any(value.shape != latent.shape for value in trajectory)):
        raise ValueError('Expected all 29 finite FP32 native states and decoded pixels')
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    if (not torch.equal(trajectory[0], noise) or not torch.equal(trajectory[-1], latent)
            or canonical_tensor_sha256(pixels) != image_sha or pixels.min() < 0 or pixels.max() > 1):
        raise ValueError('Actual Gaussian start, endpoint or Reader pixels differ')
    return {'question_id': question, 'noise_seed': seed, 'image_sha256': image_sha,
        'initial_noise_sha256': canonical_tensor_sha256(noise),
        'final_latent_sha256': canonical_tensor_sha256(latent), 'finite_fp32_states': 29}


def inspect(run, *, text_only=False, recorded_cells=None, logical_sampling_commit=None):
    from scripts.reporting.collect_transition_endpoint import read, jsonl, sha
    from scripts.reporting.collect_broader_endpoint import parent_binding, phase_summary, NATIVE_CONDITION_COMMIT, HISTORICAL_WORDING_COMMIT, CLEAR_RETENTION_COMMIT
    from scripts.experiments.broader_writer_protocol import SEED
    from vision_memory.training.latent_bank_unet import stable_seed
    logical_sampling_commit = logical_sampling_commit or NATIVE_CONDITION_COMMIT
    if logical_sampling_commit not in (NATIVE_CONDITION_COMMIT, HISTORICAL_WORDING_COMMIT, CLEAR_RETENTION_COMMIT):
        raise ValueError('Require a registered native-condition training run')
    bank, identity, result = parent_binding(run, run/'bank/manifest.json', logical_sampling_commit=logical_sampling_commit)
    if sha(run/'train/runtime.json') != RUNTIME:
        raise ValueError('Actual native training runtime changed')
    phase = run/'train/trained'
    complete = read(phase/'complete.json')
    expected = {hashlib.sha256(group['question_id'].encode()).hexdigest()[:16]+f'-seed-{index:02d}.pt':
        (group['question_id'], stable_seed(SEED, 'heldout-evaluation-noise', index))
        for group in bank['groups'] for index in range(2)}
    if len(expected) != 302 or set(complete['artifact_hashes']) != set(expected)|{'generations.jsonl', 'summary.json'}:
        raise ValueError('Incomplete fixed native development artifacts')
    for name in ('generations.jsonl', 'summary.json'):
        if sha(phase/name) != complete['artifact_hashes'][name]:
            raise ValueError('Changed final development text artifact')
    summary, pairs = phase_summary(jsonl(phase/'generations.jsonl'), bank, 'trained')
    if complete['generation_rows'] != 3020:
        raise ValueError('Incomplete final development readback')
    cells = []
    if not text_only:
        import torch
        if sha(run/'train/checkpoint-final.pt') != result['checkpoint_sha256']:
            raise ValueError('Actual final checkpoint differs from the fixed endpoint')
    elif recorded_cells is None or len(recorded_cells) != 302:
        raise ValueError('Missing independent full tensor audit cells')
    recorded = {cell['artifact']: cell for cell in recorded_cells or []}
    if text_only and set(recorded) != set(expected):
        raise ValueError('Duplicate or changed tensor audit cell')
    for name, (question, seed) in expected.items():
        image_hashes = {value['image_sha256'] for key, value in pairs.items()
            if key[:3] == (question, 'matched', seed)}
        if len(image_hashes) != 1:
            raise ValueError('The five Reader questions must see the same image')
        image_sha = next(iter(image_hashes))
        digest = complete['artifact_hashes'][name]
        if not text_only:
            if sha(phase/name) != digest:
                raise ValueError('Changed final native tensor artifact')
            payload = torch.load(phase/name, map_location='cpu', weights_only=True)
            cell = {**check_payload(payload, question, seed, image_sha), 'artifact': name, 'artifact_sha256': digest}
        else:
            cell = recorded[name]
            for key, value in {'question_id': question, 'noise_seed': seed, 'image_sha256': image_sha,
                    'artifact_sha256': digest, 'finite_fp32_states': 29}.items():
                if cell[key] != value:
                    raise ValueError('Portable tensor audit binding differs')
            for key in ('initial_noise_sha256', 'final_latent_sha256'):
                if len(cell[key]) != 64 or any(value not in '0123456789abcdef' for value in cell[key]):
                    raise ValueError('Missing actual tensor digest')
        cells.append(cell)
    return {'training_commit': identity['git_commit'], 'runtime_sha256': RUNTIME,
        'result_sha256': sha(run/'train/result.json'), 'checkpoint_sha256': result['checkpoint_sha256'],
        'phase_complete_sha256': sha(phase/'complete.json'), 'development': summary, 'cells': cells,
        'raw_rows_recounted': 3020, 'generated_tensors': 302, 'trajectory_states_per_tensor': 29,
        'scope': 'Independent CPU inspection of every final native development PT, actual Gaussian start, finite FP32 trajectory, final latent and Reader pixels. Large PTs and checkpoint remain remote; development does not establish independent functional success.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output-prefix', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--logical-sampling-commit')
    parser.add_argument('--expected-source-commit', default=SOURCE)
    args = parser.parse_args()
    sys.path[:0] = [str(args.source_root), str(args.source_root/'src')]
    from scripts.reporting.collect_transition_endpoint import read, sha
    if args.archive:
        from scripts.reporting.verify_broader_outputs_local import unpack, compare_recount
        with tempfile.TemporaryDirectory(prefix='native-final-tensors-') as temporary:
            root = Path(temporary)
            unpack(args.archive, args.sha256, root)
            recorded = read(root/'verified-summary.json')
            for name, digest in recorded['portable_sha256'].items():
                path = (root/name).resolve()
                if not path.is_relative_to(root.resolve()) or sha(path) != digest:
                    raise ValueError('Portable final native evidence seal differs')
            summary = inspect(root, text_only=True, recorded_cells=recorded['cells'], logical_sampling_commit=args.logical_sampling_commit)
            compare_recount(recorded, summary, {'portable_sha256'})
        summary = {'archive_sha256': args.sha256, 'raw_rows_recounted': 3020, 'tensor_cells_recounted': 302,
            'correct_eos': summary['development']['correct_eos'],
            'scope': 'All original raw answers and 302 tensor-audit bindings recounted locally. Actual PTs/checkpoint checked on remote CPU; tensors are not contained in this portable archive.'}
    else:
        if args.run is None:
            parser.error('Remote inspection requires --run')
        if (len(args.expected_source_commit) != 40
                or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.source_root, text=True).strip() != args.expected_source_commit
                or subprocess.check_output(['git', 'status', '--porcelain'], cwd=args.source_root, text=True).strip()):
            raise ValueError('Require the fixed clean native validation source')
        import torch
        torch.set_num_threads(1)
        summary = inspect(args.run, logical_sampling_commit=args.logical_sampling_commit)
        names = ['bank/manifest.json', 'preregistered-experiment.json', 'terminal.json', 'train/identity.json',
            'train/runtime.json', 'train/result.json', 'train/baseline-reference-check.json',
            'train/trained/complete.json', 'train/trained/summary.json', 'train/trained/generations.jsonl']
        from scripts.reporting.collect_broader_endpoint import HISTORICAL_WORDING_COMMIT, CLEAR_RETENTION_COMMIT
        if args.logical_sampling_commit in (HISTORICAL_WORDING_COMMIT, CLEAR_RETENTION_COMMIT):
            names.append('train/training-condition-augmentation.json')
        summary['portable_sha256'] = {name: sha(args.run/name) for name in names}
    output = Path(str(args.output_prefix)+'-summary.json')
    archive_path = Path(str(args.output_prefix)+'-evidence.tgz')
    if output.exists() or (not args.archive and archive_path.exists()):
        raise ValueError('Evidence output already exists')
    output.write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    if not args.archive:
        with tarfile.open(archive_path, 'w:gz') as archive:
            for name in names:
                if sha(args.run/name) != summary['portable_sha256'][name]:
                    raise ValueError('Immutable endpoint text changed while collecting')
                archive.add(args.run/name, arcname=name)
            archive.add(output, arcname='verified-summary.json')
        print(json.dumps({'archive_sha256': sha(archive_path), 'summary_sha256': sha(output),
            'correct_eos': summary['development']['correct_eos'], 'actual_tensors_checked': 302}))
    else:
        print(json.dumps(summary))


if __name__ == '__main__':
    main()
