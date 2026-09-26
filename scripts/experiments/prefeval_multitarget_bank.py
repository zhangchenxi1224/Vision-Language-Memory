"""Training-only empirical target distributions; no Reader/model imports."""
import hashlib
import json
from pathlib import Path


def target_index(pair_id, draw, count):
    if count < 1:
        raise ValueError('Empty qualified target set')
    key = f'mt8-target-20260927:{pair_id}:{draw}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % count


def select_rows(rows, ids_file):
    ids = json.loads(Path(ids_file).read_text())['ids']
    by_id = {r['base_pair_id']: r for r in rows}
    if len(ids) != len(set(ids)) or not set(ids) <= set(by_id):
        raise ValueError('Duplicate IDs or IDs outside requested split')
    return [by_id[pid] for pid in ids]


def freeze_bank(root, ids, arm):
    """Require a nonempty qualified set for EVERY registered preference."""
    root = Path(root)
    targets, coverage = {}, {}
    source_arm = 'F8' if arm == 'F1' else arm
    for pid in ids:
        candidates = []
        attempts = sorted((root / 'teachers' / source_arm / pid.replace(':', '_')).glob('target-*/complete.json'))
        for path in attempts:
            result = json.loads(path.read_text())
            if result['qualified']:
                candidates.append({'latent': str(path.parent / 'latent.pt'),
                    'latent_sha256': result['latent_sha256'],
                    'png': str(path.parent / 'memory.png'),
                    'png_sha256': result['png_sha256'], 'target_index': result['target_index']})
        coverage[pid] = {'attempted': len(attempts), 'qualified': len(candidates)}
        if not candidates:
            continue
        if arm == 'F1':
            candidates = [candidates[target_index(pid, -1, len(candidates))]]
        targets[pid] = candidates
    missing = [pid for pid in ids if pid not in targets]
    return {'schema': 'prefeval-multitarget-v1', 'arm': arm,
        'scope': 'official_training_side_only', 'targets': targets, 'coverage': coverage,
        'missing_preferences': missing, 'ready': not missing,
        'target_representation': 'optimized_model_latent; original teacher entry retained',
        'qualification': 'saved PNG; T1/T2/T3 x all four correct-option positions',
        'sampling': 'uniform preference; uniform within its qualified target set'}
