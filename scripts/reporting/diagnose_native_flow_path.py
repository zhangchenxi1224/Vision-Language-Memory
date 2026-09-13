"""Infer Euler clean-target estimates from sealed native trajectories, on CPU.

For dx/dsigma=v, an Euler step gives v~=(x_next-x)/delta_sigma and
y~=x-sigma*v~. These are finite-difference estimates, not saved model outputs.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
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


def clean_estimates(trajectory, sigmas):
    if (len(trajectory) != len(sigmas) + 1 or not sigmas or sigmas[0] != 1.
            or not all(math.isfinite(sigma) and 0 < sigma <= 1 for sigma in sigmas)
            or any(a <= b for a, b in zip(sigmas, sigmas[1:]))):
        raise ValueError('Require a complete decreasing noise-to-zero Euler path')
    stops = [*sigmas, 0.]
    for index, (sigma, next_sigma) in enumerate(zip(stops, stops[1:])):
        current, following = trajectory[index].double(), trajectory[index + 1].double()
        if current.shape != following.shape or not torch.isfinite(current).all() or not torch.isfinite(following).all():
            raise ValueError('Invalid trajectory states')
        yield current - sigma * (following - current) / (next_sigma - sigma)


def diagnose(run, bank_path, geometry_path, geometry_sha):
    if sha(geometry_path) != geometry_sha:
        raise ValueError('Geometry evidence changed')
    geometry, bank, runtime = read(geometry_path), read(bank_path), read(run / 'train/runtime.json')
    if sha(bank_path) != geometry['bank_sha256'] or sha(run / 'train/result.json') != geometry['result_sha256']:
        raise ValueError('Wrong sealed bank or endpoint')
    config = runtime['scheduler_config']
    if config['_class_name'] != 'FlowMatchEulerDiscreteScheduler' or any(config[key] for key in (
            'stochastic_sampling', 'invert_sigmas', 'use_beta_sigmas', 'use_exponential_sigmas', 'use_karras_sigmas')):
        raise ValueError('Finite differences require the inspected deterministic Euler schedule')
    teachers = {teacher['teacher_id']: teacher for teacher in bank['teachers']}
    targets = {}
    for teacher in teachers.values():
        key = teacher['latent_sha256']
        if key in targets:
            continue
        path = Path(teacher['latent_path'])
        if not path.is_absolute():
            path = bank_path.parent / path
        if sha(path) != teacher['latent_file_sha256']:
            raise ValueError('Teacher tensor changed')
        tensor = torch.load(path, map_location='cpu', weights_only=True)
        if teacher.get('latent_tensor_key'):
            tensor = tensor[teacher['latent_tensor_key']]
        targets[key] = tensor.double().reshape(-1)
    keys = sorted(targets)
    matrix = torch.stack([targets[key] for key in keys])
    directory = run / 'train/trained'
    seal = read(directory / 'complete.json')
    cells, counts = [], Counter()
    for prior in geometry['phases']['trained']['cells']:
        qid = prior['question_id']
        matching = [group for group in bank['groups'] if group['question_id'] == qid]
        if len(matching) != 1 or len(matching[0]['teacher_ids']) != 1:
            raise ValueError('Missing unique single-target condition')
        intended = teachers[matching[0]['teacher_ids'][0]]['latent_sha256']
        if intended != prior['intended_target_sha256']:
            raise ValueError('Intended teacher changed')
        # The prior report's artifact SHA identifies the exact saved seed file.
        names = [name for name, digest in seal['artifact_hashes'].items() if name.endswith('.pt') and digest == prior['artifact_file_sha256']]
        if len(names) != 1 or sha(directory / names[0]) != prior['artifact_file_sha256']:
            raise ValueError('Generated trajectory identity changed')
        payload = torch.load(directory / names[0], map_location='cpu', weights_only=True)
        path = payload['trajectory']
        expected_noise = torch.randn(path[0].shape, generator=torch.Generator().manual_seed(prior['noise_seed']), dtype=torch.float32)
        if (payload['question_id'] != qid or payload['noise_seed'] != prior['noise_seed']
                or not torch.equal(path[0], expected_noise) or not torch.equal(path[-1], payload['latent'])):
            raise ValueError('Pure Gaussian initial state or endpoint does not match')
        sigmas = runtime['effective_inference_sigmas'][qid]
        if len(sigmas) != 28:
            raise ValueError('Native28-step schedule changed')
        estimates, intended_index = [], keys.index(intended)
        for index, clean in enumerate(clean_estimates(path, sigmas)):
            error = (matrix - clean.reshape(-1)).square().mean(dim=1)
            nearest = keys[int(error.argmin())]
            estimates.append({'step': index, 'sigma': sigmas[index], 'nearest_target_sha256': nearest,
                'nearest_is_intended': nearest == intended, 'intended_rms_error': float(error[intended_index].sqrt()),
                'nearest_rms_error': float(error.min().sqrt())})
        if estimates[-1]['nearest_target_sha256'] != prior['nearest_target_sha256']:
            raise ValueError('Final-step clean estimate differs from the sealed endpoint geometry')
        cells.append({**prior, 'step_estimates': estimates})
        counts['images'] += 1
        counts['gaussian_initial_states_verified'] += 1
        counts['native_steps_analyzed'] += len(estimates)
    return {'script_sha256': sha(__file__), 'geometry_sha256': geometry_sha, 'runtime_sha256': sha(run / 'train/runtime.json'),
        'bank_sha256': geometry['bank_sha256'], 'result_sha256': geometry['result_sha256'], 'counts': dict(counts),
        'cells': cells, 'scope': 'Post-hoc CPU finite-difference clean estimates from every trained native path. Float32 step rounding is included; not measured raw model velocities or an accuracy substitute.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'bank', 'geometry', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--geometry-sha256', required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise ValueError('Do not overwrite path evidence')
    torch.set_num_threads(1)
    value = diagnose(a.run, a.bank, a.geometry, a.geometry_sha256)
    a.output.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(value['counts']))
