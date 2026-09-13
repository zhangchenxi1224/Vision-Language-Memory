"""Read-only endpoint geometry: intended-target error and nearest sealed target.

This is post-hoc diagnosis, not an accuracy metric, target selector or new loss.
It uses CPU tensors only and preserves every condition and both evaluation seeds.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def diagnose(run, bank_path, expected_bank_sha, expected_result_sha):
    if sha(bank_path) != expected_bank_sha or sha(run / 'train/result.json') != expected_result_sha:
        raise ValueError('Sealed bank or endpoint changed')
    bank, result, terminal = read(bank_path), read(run / 'train/result.json'), read(run / 'terminal.json')
    if terminal['state'] != 'completed' or terminal['training_result_sha256'] != expected_result_sha:
        raise ValueError('Require the completed endpoint')
    targets, teachers = {}, {teacher['teacher_id']: teacher for teacher in bank['teachers']}
    for teacher in teachers.values():
        digest = teacher['latent_sha256']
        if digest in targets:
            continue
        path = Path(teacher['latent_path'])
        if not path.is_absolute():
            path = bank_path.parent / path
        if sha(path) != teacher['latent_file_sha256']:
            raise ValueError('A sealed teacher tensor changed')
        tensor = torch.load(path, map_location='cpu', weights_only=True)
        if teacher.get('latent_tensor_key'):
            tensor = tensor[teacher['latent_tensor_key']]
        if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
            raise ValueError('Invalid teacher tensor')
        targets[digest] = {'tensor': tensor.double().reshape(-1), 'teacher_id': teacher['teacher_id'], 'gold': teacher['answer']}
    keys = sorted(targets)
    matrix = torch.stack([targets[key]['tensor'] for key in keys])
    geometry = {key: {'teacher_id': targets[key]['teacher_id'], 'gold': targets[key]['gold'],
        'mean': float(matrix[i].mean()), 'rms': float(matrix[i].square().mean().sqrt()),
        'expected_unit_gaussian_velocity_mse': float(1 + matrix[i].square().mean()),
        'pairwise_rms': {other: float((matrix[i] - matrix[j]).square().mean().sqrt()) for j, other in enumerate(keys)}}
        for i, key in enumerate(keys)}
    phases = {}
    for phase in ('baseline', 'trained'):
        directory = run / 'train' / phase
        seal = read(directory / 'complete.json')
        generations = directory / 'generations.jsonl'
        if sha(generations) != seal['artifact_hashes']['generations.jsonl']:
            raise ValueError('Sealed raw Reader records changed')
        rows = [json.loads(line) for line in generations.read_text().splitlines()]
        cells, counts = [], Counter()
        for group in bank['groups']:
            if len(group['teacher_ids']) != 1:
                raise ValueError('This diagnosis requires the actual single-target151 bank')
            intended = teachers[group['teacher_ids'][0]]['latent_sha256']
            intended_index = keys.index(intended)
            for seed_index in range(2):
                name = hashlib.sha256(group['question_id'].encode()).hexdigest()[:16] + f'-seed-{seed_index:02d}.pt'
                path = directory / name
                if sha(path) != seal['artifact_hashes'][name]:
                    raise ValueError('Sealed generated latent changed')
                payload = torch.load(path, map_location='cpu', weights_only=True)
                latent = payload['latent'].double().reshape(-1)
                if latent.shape != matrix[0].shape or not torch.isfinite(latent).all():
                    raise ValueError('Unexpected generated latent')
                errors = (matrix - latent).square().mean(dim=1)
                nearest = keys[int(errors.argmin())]
                selected = [row for row in rows if row['question_id'] == group['question_id'] and row['condition'] == 'matched'
                    and row['noise_seed'] == payload['noise_seed']]
                if len(selected) != 5 or len({row['prompt_id'] for row in selected}) != 5:
                    raise ValueError('Incomplete five-query cell')
                passed = sum(row['generated_token_ids'] == row['scorer']['gold_token_ids'] + [151645] for row in selected)
                cells.append({'question_id': group['question_id'], 'historical_target_index': group.get('historical_target_index'),
                    'noise_seed': payload['noise_seed'], 'gold': group['answer'], 'correct_eos_out_of_five': passed,
                    'intended_target_sha256': intended, 'nearest_target_sha256': nearest,
                    'nearest_is_intended': nearest == intended,
                    'intended_rms_error': float(errors[intended_index].sqrt()), 'nearest_rms_error': float(errors.min().sqrt()),
                    'artifact_file_sha256': seal['artifact_hashes'][name]})
                counts['images'] += 1
                counts['nearest_is_intended'] += nearest == intended
                counts['correct_eos_rows'] += passed
                del payload
        phases[phase] = {'counts': dict(counts), 'cells': cells}
    return {'script_sha256': sha(__file__), 'torch_version': torch.__version__, 'device': 'cpu',
        'bank_sha256': expected_bank_sha, 'result_sha256': expected_result_sha, 'checkpoint_sha256': result['checkpoint_sha256'],
        'unique_targets': len(targets), 'geometry': geometry, 'phases': phases,
        'scope': 'Post-hoc CPU distance diagnosis of every sealed generated latent; does not establish perceptual similarity, Reader robustness or causal attribution.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bank', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--bank-sha256', required=True)
    parser.add_argument('--result-sha256', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():
        raise ValueError('Do not overwrite a diagnosis')
    value = diagnose(args.run, args.bank, args.bank_sha256, args.result_sha256)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'unique_targets': value['unique_targets'], 'phases': {key: data['counts'] for key, data in value['phases'].items()}}))
