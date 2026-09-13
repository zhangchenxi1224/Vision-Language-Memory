"""Check downloaded native-path diagnostics against the sealed endpoint geometry."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path, expected_sha, geometry, runtime):
    if sha(path) != expected_sha:
        raise ValueError('Downloaded diagnostic differs from the observed remote SHA')
    data = json.loads(path.read_bytes())
    if data['geometry_sha256'] != sha(geometry) or data['runtime_sha256'] != sha(runtime):
        raise ValueError('Geometry or scheduler binding changed')
    previous, schedule = json.loads(geometry.read_bytes()), json.loads(runtime.read_bytes())
    expected = {(cell['question_id'], cell['noise_seed']): cell for cell in previous['phases']['trained']['cells']}
    seen, patterns = set(), Counter()
    for cell in data['cells']:
        key = cell['question_id'], cell['noise_seed']
        if key not in expected or key in seen:
            raise ValueError('Missing or duplicated prior cell')
        seen.add(key)
        if {name: value for name, value in cell.items() if name != 'step_estimates'} != expected[key]:
            raise ValueError('Prior endpoint evidence changed')
        steps = cell['step_estimates']
        if (len(steps) != 28 or [step['step'] for step in steps] != list(range(28))
                or [step['sigma'] for step in steps] != schedule['effective_inference_sigmas'][key[0]]):
            raise ValueError('Actual native28 schedule changed')
        for step in steps:
            if step['nearest_target_sha256'] not in previous['geometry']:
                raise ValueError('Unknown nearest target')
            if step['nearest_is_intended'] != (step['nearest_target_sha256'] == cell['intended_target_sha256']):
                raise ValueError('Nearest-target label changed')
        if (steps[-1]['nearest_target_sha256'] != cell['nearest_target_sha256']
                or abs(steps[-1]['intended_rms_error'] - cell['intended_rms_error']) > 1e-12):
            raise ValueError('Last Euler estimate does not match the sealed endpoint')
        patterns[(cell['correct_eos_out_of_five'] == 5, steps[0]['nearest_is_intended'],
            any(step['nearest_is_intended'] for step in steps), steps[-1]['nearest_is_intended'])] += 1
    if seen != set(expected) or len(seen) != 302 or data['counts'] != {
            'images': 302, 'gaussian_initial_states_verified': 302, 'native_steps_analyzed': 8456}:
        raise ValueError('Incomplete diagnostic matrix')
    return {'artifact_sha256': expected_sha, 'geometry_sha256': sha(geometry), 'runtime_sha256': sha(runtime),
        'cells_checked': len(seen), 'steps_checked': 8456,
        'patterns': [{'all_five_reads_correct': key[0], 'initial_nearest_correct': key[1],
            'ever_nearest_correct': key[2], 'final_nearest_correct': key[3], 'images': count} for key, count in sorted(patterns.items())],
        'scope': 'Local checksum, complete-cell and bound-scheduler verification only; actual Gaussian tensors and finite differences were checked remotely on CPU.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('artifact', 'geometry', 'runtime', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    a = parser.parse_args()
    value = verify(a.artifact, a.sha256, a.geometry, a.runtime)
    a.output.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    print(json.dumps(value))
