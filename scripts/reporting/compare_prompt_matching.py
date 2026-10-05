"""Strict paired K1 comparison; report review evidence, never change training defaults.

Consumes the existing readback JSONL and judge JSON schemas. See
docs/PROMPT_MATCHING.md for the required, frozen per-run manifest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.prefeval_k1_data import official_eval_disclosures, official_mcq  # noqa: E402

DEFAULT_PROTOCOL = ROOT / 'configs/experiments/prompt_matching_parallel.json'
IDENTITY = ('pair_id', 'chain', 'prefix', 'control', 'family', 'task')
JUDGE_FIELDS = ('judge_model', 'model_label', 'provider', 'max_tokens', 'enable_thinking')
COMMON = ('format_arm', 'split', 'endpoint_kind', 'training_stage', 'evaluation_phase',
          'expected_ids', 'training_ids', 'selection_ids', 'reader_revision', 'tokenizer_revision',
          'chat_template_sha256', 'dataset_sha256', 'history_sha256', 'history_protocol',
          'writer_base_sha256', 'training', 'protocol_sha256', 'judge')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def registered_ids():
    plan = json.loads((ROOT / 'reports/prefeval-official-alignment-20260923/'
                      'WRITER_IMPLEMENTATION_SPLIT.json').read_text(encoding='utf-8'))
    return {'pilot': set(plan['pilot_train_ids']), 'train': set(plan['train_ids']),
            'dev': set(plan['internal_dev_ids']),
            'official': {r['base_pair_id'] for r in official_eval_disclosures()}}


def unique_ids(value, field):
    require(isinstance(value, list) and all(isinstance(x, str) and x for x in value),
            f'{field} must be a list of sample IDs')
    require(len(value) == len(set(value)), f'Duplicate IDs in {field}')
    return set(value)


def validate_manifest(manifest, protocol, protocol_sha, registry):
    require(manifest.get('schema') == 'vision_memory.prompt_matching_readback.v1', 'Unknown manifest schema')
    for key in COMMON + ('label', 'supervision', 'checkpoint_sha256', 'readback_files', 'judge_dirs'):
        require(key in manifest, f'Missing manifest field: {key}')
    require(manifest['supervision'] in protocol['supervision_modes'], 'Unknown supervision mode')
    require(manifest['format_arm'] in protocol['format_arms'], 'Unknown format arm')
    require(manifest['protocol_sha256'] == protocol_sha, 'Manifest does not bind the supplied protocol hash')
    for key in ('chat_template_sha256', 'dataset_sha256', 'history_sha256', 'writer_base_sha256',
                'checkpoint_sha256'):
        value = manifest[key]
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value),
                f'{key} must be a SHA256 digest, not a mutable path or placeholder')
    for key in ('reader_revision', 'tokenizer_revision', 'history_protocol', 'label'):
        require(isinstance(manifest[key], str) and manifest[key].strip(), f'Empty {key}')
    split = manifest['split']
    require(split in registry, 'Unknown split')
    expected = unique_ids(manifest['expected_ids'], 'expected_ids')
    training = unique_ids(manifest['training_ids'], 'training_ids')
    selection = unique_ids(manifest['selection_ids'], 'selection_ids')
    require(expected and expected == registry[split], 'expected_ids must cover the complete registered split')
    require(manifest['training_stage'] in {'pilot', 'train'}, 'Unknown training stage')
    require(training == registry[manifest['training_stage']], 'training_ids do not match the registered stage')
    require(selection <= registry['train'], 'Selection IDs must stay on the training side')
    if split in {'dev', 'official'}:
        require(not expected & (training | selection), 'Held-out IDs overlap training or selection IDs')
    require(manifest['endpoint_kind'] in {'teacher', 'student'}, 'Unknown endpoint kind')
    require(manifest['evaluation_phase'] in protocol['evaluation']['phases'], 'Unknown evaluation phase')
    if manifest['endpoint_kind'] == 'teacher':
        require(split in {'pilot', 'train'} and manifest['evaluation_phase'] == 'initial',
                'Teachers may only be evaluated on training rows at initial prefix')
    require(manifest['training'] == protocol['training'], 'Training budget/settings differ from prospective protocol')
    require(isinstance(manifest['readback_files'], list) and manifest['readback_files'], 'No readback files')
    require(isinstance(manifest['judge_dirs'], list) and manifest['judge_dirs'], 'No judge directories')
    require(isinstance(manifest['judge'], dict) and set(manifest['judge']) == set(JUDGE_FIELDS),
            'Judge identity must include its model, label, provider, budget and thinking mode')


def expected_keys(manifest, protocol):
    evaluation = protocol['evaluation']
    prefixes = evaluation['phases'][manifest['evaluation_phase']]
    chains = 1 if manifest['endpoint_kind'] == 'teacher' else evaluation['noise_chains']
    keys = set()
    for pid in manifest['expected_ids']:
        for prefix in prefixes:
            for chain in range(chains):
                for control in evaluation['controls']:
                    if control == 'blank' and (prefix != 0 or chain != 0):
                        continue
                    if control == 'text' and chain != 0:
                        continue
                    for family in evaluation['families']:
                        for task in evaluation['tasks']:
                            keys.add((pid, chain, prefix, control, family, task))
    return keys


def load_scores(manifest, base, protocol):
    rows, pngs, files = {}, {}, []
    expected = expected_keys(manifest, protocol)
    for source in manifest['readback_files']:
        path = base / source
        files.append({'path': str(path), 'sha256': digest(path)})
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = tuple(row[k] for k in IDENTITY)
            require(key in expected, f'Unexpected readback identity: {key}')
            require(row['split'] == manifest['split'], 'Mixed or mismatched readback splits')
            require(row['endpoint_kind'] == manifest['endpoint_kind'], 'Mixed endpoint kinds')
            require(row['history_protocol'] == manifest['history_protocol'], 'Mixed history protocols')
            require(row['max_new_tokens'] == protocol['evaluation']['max_new_tokens'][row['task']],
                    'Mixed generation budgets')
            require(type(row['generated']['truncated']) is bool, 'Missing generation truncation flag')
            if row['control'] in {'memory', 'mismatch'}:
                require(bool(row['png_sha256']), 'Missing PNG hash')
                endpoint = key[:4]
                require(endpoint not in pngs or pngs[endpoint] == row['png_sha256'],
                        'Different questions did not read the same frozen PNG')
                pngs[endpoint] = row['png_sha256']
            require(key not in rows or canonical(rows[key]) == canonical(row), 'Conflicting duplicate readback')
            rows[key] = row
    missing = expected - rows.keys()
    require(not missing, f'Missing paired readbacks: {len(missing)} of {len(expected)}')
    needed = {canonical(row) for row in rows.values() if row['task'] == 'free'}
    judgments = {}
    for directory in manifest['judge_dirs']:
        for path in sorted((base / directory).glob('*.json')):
            record = json.loads(path.read_text(encoding='utf-8'))
            source = canonical(record.get('input'))
            if source not in needed:
                continue
            require({k: record.get(k) for k in JUDGE_FIELDS} == manifest['judge'], 'Mixed or mismatched judge settings')
            require(record.get('status') == 'complete' and type(record.get('correct')) is bool,
                    'Missing, failed or pending free-response judgment')
            require(source not in judgments or judgments[source] == record['correct'], 'Conflicting judge scores')
            judgments[source] = record['correct']
            files.append({'path': str(path), 'sha256': digest(path)})
    require(set(judgments) == needed, 'Missing complete judgments; no observed-only accuracy is allowed')
    parser = official_mcq(ROOT / 'third_party/prefeval_reference')['extract_choice']
    scores = {}
    for key, row in rows.items():
        if row['task'] == 'free':
            scores[key] = judgments[canonical(row)]
            continue
        require(sorted(row['option_order']) == [0, 1, 2, 3], 'Invalid MCQ option order')
        letter = 'ABCD'[row['option_order'].index(0)]
        parsed = parser(row['generated']['raw'])
        correct = parsed == letter
        require(row['correct_letter'] == letter and type(row['correct']) is bool and row['correct'] == correct,
                'MCQ score does not match the official parser')
        require(row['parse_failure'] == (parsed is None), 'MCQ parse-failure metadata mismatch')
        scores[key] = correct
    return rows, scores, files


def paired_interval(before, after, *, samples, seed, confidence):
    require(len(before) == len(after) and bool(before), 'A complete nonempty pair set is required')
    differences = [b - a for a, b in zip(before, after)]
    n = len(differences)
    rng = random.Random(seed)
    draws = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples))
    alpha = (1 - confidence) / 2
    def quantile(p):
        offset = p * (samples - 1)
        lo = int(offset)
        hi = min(lo + 1, samples - 1)
        return draws[lo] + (draws[hi] - draws[lo]) * (offset - lo)
    return {'preferences': n, 'baseline': sum(before) / n, 'candidate': sum(after) / n,
            'delta': sum(differences) / n, 'interval': [quantile(alpha), quantile(1 - alpha)]}


def joint_vector(scores, manifest, protocol, prefix, task, control):
    chains = 1 if manifest['endpoint_kind'] == 'teacher' else protocol['evaluation']['noise_chains']
    families = protocol['evaluation']['joint_families']
    result = []
    for pid in sorted(manifest['expected_ids']):
        selected = range(chains) if control in {'memory', 'mismatch'} else [0]
        selected_prefix = 0 if control == 'blank' else prefix
        values = [all(scores[(pid, chain, selected_prefix, control, family, task)] for family in families)
                  for chain in selected]
        result.append(sum(values) / len(values))
    return result


def compare(baseline_path, candidate_path, protocol_path=DEFAULT_PROTOCOL):
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    manifests = [json.loads(Path(p).read_text(encoding='utf-8')) for p in (baseline_path, candidate_path)]
    registry = registered_ids()
    for manifest in manifests:
        validate_manifest(manifest, protocol, digest(protocol_path), registry)
    baseline, candidate = manifests
    for key in COMMON:
        left, right = baseline[key], candidate[key]
        if key.endswith('_ids'):
            left, right = sorted(left), sorted(right)
        require(left == right, f'Unpaired experiment manifest field: {key}')
    require(baseline['supervision'] != candidate['supervision'], 'Supervision modes must differ')
    loaded = [load_scores(manifest, Path(path).parent, protocol)
              for manifest, path in zip(manifests, (baseline_path, candidate_path))]
    left_rows, left_scores, left_sources = loaded[0]
    right_rows, right_scores, right_sources = loaded[1]
    for key, left in left_rows.items():
        right = right_rows[key]
        for field in ('question', 'reader_query', 'preference', 'option_order', 'correct_letter', 'donor_pair_id'):
            require(left.get(field) == right.get(field), f'Unpaired evaluation input: {field}/{key}')
    settings = protocol['comparison']
    kwargs = {'samples': settings['bootstrap_samples'], 'seed': settings['bootstrap_seed'],
              'confidence': settings['confidence']}
    metrics, specificity = [], []
    prefixes = protocol['evaluation']['phases'][baseline['evaluation_phase']]
    for prefix in prefixes:
        for task in protocol['evaluation']['tasks']:
            vectors = {}
            for control in protocol['evaluation']['controls']:
                a = joint_vector(left_scores, baseline, protocol, prefix, task, control)
                b = joint_vector(right_scores, candidate, protocol, prefix, task, control)
                vectors[control] = b
                metrics.append({'prefix': prefix, 'task': task, 'control': control,
                                **paired_interval(a, b, **kwargs)})
            for control in ('blank', 'mismatch'):
                specificity.append({'prefix': prefix, 'task': task, 'reference': control,
                                    **paired_interval(vectors[control], vectors['memory'], **kwargs)})
    primary = next(m for m in metrics if m['control'] == 'memory'
                   and m['prefix'] == settings['primary_prefix'] and m['task'] == settings['primary_task'])
    guards = [m for m in metrics if m['control'] == 'memory' and m is not primary]
    required_specificity = [m for m in specificity if m['task'] == settings['primary_task']
                            and m['prefix'] in {0, 10}]
    gates = {
        'heldout_student_retention': baseline['split'] in settings['promotion_requires_split']
            and baseline['endpoint_kind'] == settings['promotion_requires_endpoint']
            and baseline['evaluation_phase'] == settings['promotion_requires_phase'],
        'candidate_is_prompt_matching': candidate['supervision'] == 'prompt_matching',
        'primary_positive_interval': primary['interval'][0] > 0,
        'secondary_noninferiority': all(m['interval'][0] >= -settings['noninferiority_margin'] for m in guards),
        'candidate_uses_memory': all(m['interval'][0] > 0 for m in required_specificity),
    }
    result = {
        'schema': 'vision_memory.prompt_matching_comparison.v1',
        'status': 'eligible_for_promotion_review' if all(gates.values()) else 'diagnostic_only_not_promoted',
        'automatic_default_switch': False, 'gates': gates,
        'baseline': {'label': baseline['label'], 'supervision': baseline['supervision']},
        'candidate': {'label': candidate['label'], 'supervision': candidate['supervision']},
        'split': baseline['split'], 'endpoint_kind': baseline['endpoint_kind'],
        'definition': 'O1 AND O2 correct per PNG, averaged over noise chains within each preference ID',
        'primary': primary, 'metrics': metrics, 'candidate_memory_specificity': specificity,
        'bootstrap': kwargs, 'source_files': left_sources + right_sources,
        'manifest_sha256': [digest(baseline_path), digest(candidate_path)],
        'protocol_sha256': digest(protocol_path),
        'generation_truncations': {m['label']: sum(r['generated']['truncated'] for r in rows.values())
                                   for m, (rows, _, _) in zip(manifests, loaded)},
        'disclosures': protocol['disclosures'] + [
            'Manifest provenance is declared and hash-bound, not independently attested training execution.',
            'Use history_hard versus prompt_matching to isolate soft labels; hard_ce changes target provenance too.',
            'No final-default change follows automatically, including when every descriptive gate passes.'
        ]
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--candidate-manifest', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline_manifest, args.candidate_manifest, args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
