"""Freeze training-only generated RGB contexts before observing their outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'reports/official-alignment-results-20260913'
BANK_SHA = 'c27cd65dab809deabb5f2cb08891517d3590244651d08a8c6763c84fea901592'
PACKAGE_SHA = '267152e0c1c059f1f1c3a3e041839a68ba0b8f96a6ca4d782a7ef90ba5ccf05f'
CHECKPOINT_SHA = '7294684170578dfc617b4fafcea97e6480c08642ca4f1e5967ff8966aa103182'
GOLD = {'ambient': [59614, 151645], 'jazz': [73, 9802, 151645], 'clear': [2152, 4541, 21933, 151645]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed(namespace, index):
    return int.from_bytes(hashlib.sha256(f'20260915:{namespace}:{index}'.encode()).digest()[:8], 'big') % (2**63 - 1)


def noise_seeds(value):
    """Only explicit noise seed fields; no guessing from repetition indices."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ('noise_seed', 'seed') and type(item) is int:
                yield item
            elif key in ('noise_seeds', 'heldout_noise_seeds') and isinstance(item, list):
                yield from (n for n in item if type(n) is int)
            yield from noise_seeds(item)
    elif isinstance(value, list):
        for item in value:
            yield from noise_seeds(item)


def plan():
    bank_path = RESULTS / 'broader151-bank-manifest.json'
    if digest(bank_path) != BANK_SHA:
        raise ValueError('Changed original151 training bank')
    bank = json.loads(bank_path.read_bytes())
    originals = {g['target_state']: g for g in bank['groups']
                 if g.get('source_state') == 'gray' and g.get('wording_index') == 0}
    if set(originals) != set(GOLD):
        raise ValueError('Require exactly the original three gray-to-state training events')
    refs = {}
    excluded = {seed('training-noise', i) for i in range(19328)}
    excluded |= {seed('heldout-evaluation-noise', i) for i in range(2)}
    for name in ('broader151-preregistered-plan.json', '03-observed-wording-baseline-preregistered.json'):
        path = RESULTS / name
        refs[name] = digest(path)
        excluded.update(noise_seeds(json.loads(path.read_bytes())))
    jobs = []
    for state in ('ambient', 'jazz', 'clear'):
        group = originals[state]
        if len(group['question_variants']) != 5:
            raise ValueError('Require all five original training queries for qualification')
        for repetition in range(8):
            n = seed('source-pool-training-only-v1:' + state, repetition)
            if n in excluded:
                raise ValueError('Training source seed overlaps an existing training/evaluation noise seed')
            jobs.append({'id': f'{state}-{repetition:02d}', 'state': state, 'repetition': repetition,
                'lane': repetition % 4, 'seed': n, 'event': group['event_text'],
                'queries': group['question_variants'], 'expected_token_ids': GOLD[state]})
    if len({j['seed'] for j in jobs}) != 24:
        raise ValueError('Duplicate source generation noise')
    return {'schema': 'generated-training-source-pool/v1', 'bank_sha256': BANK_SHA,
        'parent_package_sha256': PACKAGE_SHA, 'parent_checkpoint_sha256': CHECKPOINT_SHA,
        'parent_commit': '4fbc85725d78427235757ace2661d086b896a97f',
        'seed_namespace': 'source-pool-training-only-v1:<state>', 'excluded_plan_hashes': refs,
        'source': 'Reset to official RGB gray128 before each independent write; original training event only.',
        'protocol': {'native_steps': 28, 'guidance_scale': 1.0, 'writer_dtype': 'float32',
                     'reader_dtype': 'bfloat16', 'optimization_steps': 0},
        'qualification': 'Preserve all24 outputs and120 raw reads. All five original queries on every PNG must match original token IDs plus immediate EOS. Any failed case rejects the entire pool for training; no replacement seed or best-image selection.',
        'scope': 'Training material generation and qualification only; not a holdout, new training endpoint, or functional acceptance score. No validation PNG is a training input.',
        'future_training_contract': 'If qualified, add these PNG sources and their official VAE latents only on the condition side. Re-encode native event condition on the selected actual PNG. Keep canonical contexts for the complete baseline and evaluation. Official target-noise FM and target bank stay unchanged. A separate training registration is required.',
        'jobs': jobs}
