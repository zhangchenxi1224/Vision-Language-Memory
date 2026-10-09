"""Verify complete AUG4 readback shards and summarize at preference-group level."""
# ruff: noqa: E402 -- establish repository imports before importing runtime modules.
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.experiments.prefeval_aug4_evaluate import (
    KEY_FIELDS, binding_digest, eval_order, mismatch_donors, planned_read_keys, question_suites,
)
from scripts.experiments.prefeval_aug4_teacher import save_json
from scripts.experiments.prefeval_k1_data import official_mcq, sha

BOOTSTRAP_SAMPLES = 10000
BOOTSTRAP_SEED = 20261009


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bootstrap_mean(values):
    """Percentile interval over preference means; never over individual answers."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError('Bootstrap requires nonempty finite preference-level values')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for first in range(0, BOOTSTRAP_SAMPLES, 256):
        count = min(256, BOOTSTRAP_SAMPLES - first)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        draws[first:first + count] = values[indices].mean(axis=1)
    interval = np.quantile(draws, [.025, .975])
    return {'preference_count': len(values), 'mean': float(values.mean()),
            'ci95': [float(x) for x in interval], 'method': 'preference-group paired percentile bootstrap',
            'bootstrap_samples': BOOTSTRAP_SAMPLES, 'bootstrap_seed': BOOTSTRAP_SEED}


def _metrics(records):
    """Pool repeated measurements within preference before statistical inference."""
    controls = {}
    rates = defaultdict(dict)
    for control in sorted({r['control'] for r in records}):
        subset = [r for r in records if r['control'] == control]
        by_preference = defaultdict(list)
        for r in subset:
            by_preference[r['pair_id']].append(int(r['correct']))
        means = {pid: sum(values) / len(values) for pid, values in by_preference.items()}
        for pid, value in means.items():
            rates[pid][control] = value
        correct = sum(int(r['correct']) for r in subset)
        controls[control] = {'correct': correct, 'total': len(subset), 'rate': correct / len(subset),
                             'parse_failures': sum(int(r['parse_failure']) for r in subset),
                             'preference_count': len(means),
                             'preference_mean_rate': sum(means.values()) / len(means)}
    gains = {}
    if 'memory' in controls:
        for control in ('mismatch', 'blank'):
            if control not in controls:
                continue
            if any('memory' not in v or control not in v for v in rates.values()):
                raise ValueError('Paired control is missing for a preference')
            gain = bootstrap_mean([rates[pid]['memory'] - rates[pid][control] for pid in sorted(rates)])
            gain['difference_percentage_points'] = 100 * gain['mean']
            gain['ci95_percentage_points'] = [100 * x for x in gain['ci95']]
            gain['comparison'] = 'memory minus ' + control
            gains[control] = gain
    return {'controls': controls, 'paired_memory_gain': gains,
            'preference_control_rates': dict(sorted(rates.items()))}


def _summarize_records(records):
    metrics = _metrics(records)
    metrics['suites'] = {}
    for writer, question in sorted({(r['writer_family'], r['question_family']) for r in records}):
        subset = [r for r in records if r['writer_family'] == writer and r['question_family'] == question]
        metrics['suites'][writer + '/' + question] = _metrics(subset)
    metrics['measurement_count'] = len(records)
    metrics['preference_count'] = len({r['pair_id'] for r in records})
    return metrics


def _receipt_counts(records):
    counts = defaultdict(lambda: {'correct': 0, 'total': 0, 'parse_failures': 0})
    for record in records:
        for key in (record['control'], '/'.join(record[k] for k in ('writer_family', 'question_family', 'control'))):
            counts[key]['correct'] += int(record['correct'])
            counts[key]['total'] += 1
            counts[key]['parse_failures'] += int(record['parse_failure'])
    return dict(counts)


def _official_exclusions(data):
    official = {r['base_pair_id']: r for r in data['official']}
    train = {r['base_pair_id']: r for r in data['train']}
    ids = set(official)
    marked = {r['base_pair_id'] for r in data['official'] if r.get('official_duplicate_of_train')}
    metadata = {r['official_eval_base_pair_id'] for r in data.get('metadata', {}).get('official_overlap_pairs', [])}
    if not marked <= ids or not metadata <= ids:
        raise ValueError('Official duplicate disclosure names an unknown official preference')
    if marked and metadata and marked != metadata:
        raise ValueError('Official duplicate row flags and metadata disagree')
    excluded = marked | metadata
    if len(ids) != 180 or len(ids - excluded) != 177:
        raise ValueError('The registered official sensitivity analysis requires exactly 180/177 preferences')
    def content(row):
        return tuple(' '.join(row[field][form].casefold().split())
                     for field, form in [('writer_user_forms', 'W0'), ('teacher_question_forms', 'T1')])
    train_content = {content(r) for r in train.values()}
    if any(content(official[pid]) not in train_content for pid in excluded):
        raise ValueError('Disclosed official overlap is not supported by preference and question content')
    for pair in data.get('metadata', {}).get('official_overlap_pairs', []):
        original = train.get(pair['train_base_pair_id'])
        if original is None or content(original) != content(official[pair['official_eval_base_pair_id']]):
            raise ValueError('Official overlap metadata does not match the named training record')
    return excluded


def summarize(data_path, output_root):
    """Return a verified summary; the CLI (or controller) chooses where to save it."""
    data_path, output_root = Path(data_path), Path(output_root)
    data, data_sha = read_json(data_path), sha(data_path)
    files = list(output_root.glob('read-finished-*.json'))
    if not files:
        raise ValueError('No completed readback shards')
    first = read_json(files[0])
    shared = dict(first['binding'])
    shared.pop('assignment')
    shared.pop('shard')
    split, kind, nshards = shared['split'], shared['kind'], int(shared['shards'])
    if nshards < 1 or shared['schema'] != 'prefeval-aug4-evaluation/v1' or shared['data_sha256'] != data_sha:
        raise ValueError('Invalid evaluation schema, shard count, or dataset binding')
    expected_files = {f'read-finished-{i}.json' for i in range(nshards)}
    if {p.name for p in files} != expected_files:
        raise ValueError('Readback shards are missing or unexpected')
    rows = data[split]
    by_id = {r['base_pair_id']: r for r in rows}
    if len(by_id) != len(rows) or not rows:
        raise ValueError('Duplicate or empty dataset preference identifiers')
    expected_suites = [list(x) for x in question_suites(kind, split)]
    if shared['suites'] != expected_suites or shared['seeds'] != [0, 1] or shared['correct_positions'] != [0, 1, 2, 3]:
        raise ValueError('Evaluation conditions differ from the registered suites/seeds/positions')
    controls = shared['controls']
    if len(set(controls)) != len(controls) or not {'memory', 'mismatch', 'blank'} <= set(controls) or not set(controls) <= {'memory', 'mismatch', 'blank', 'text'}:
        raise ValueError('Required memory, mismatch, and blank controls are missing or invalid')
    if kind == 'student' and (not shared.get('checkpoint') or
       (split == 'official' and shared['checkpoint']['optimizer_step'] != 93440)):
        raise ValueError('Missing checkpoint identity or non-final official evaluation')
    mcq_source = ROOT / 'third_party/prefeval_reference/utils/utils_mcq.py'
    if shared['mcq_source_sha256'] != sha(mcq_source):
        raise ValueError('Official question/scoring source differs from the evaluation binding')
    mcq = official_mcq(mcq_source.parents[1])
    donors = mismatch_donors(rows, split)
    args = SimpleNamespace(kind=kind, split=split, seeds=shared['seeds'], controls=controls)
    records, receipts, all_keys = [], [], set()
    for shard in range(nshards):
        finished_path = output_root / f'read-finished-{shard}.json'
        finished, identity = read_json(finished_path), read_json(output_root / f'identity-{shard}.json')
        binding = finished['binding']
        remainder = {k: v for k, v in binding.items() if k not in {'shard', 'assignment'}}
        selected = rows[shard::nshards]
        if (finished['status'] != 'completed' or binding != identity or remainder != shared or
                binding['shard'] != shard or binding['assignment'] != [r['base_pair_id'] for r in selected]):
            raise ValueError(f'Shard {shard} identity, assignment, or completion status differs')
        log_path = output_root / f'readback-{shard}.jsonl'
        if sha(log_path) != finished['readback_sha256']:
            raise ValueError(f'Shard {shard} log checksum differs')
        current = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
        keys = [tuple(r[k] for k in KEY_FIELDS) for r in current]
        expected = set(planned_read_keys(selected, args))
        if len(keys) != len(set(keys)) or set(keys) != expected or all_keys.intersection(keys):
            raise ValueError(f'Shard {shard} has missing, duplicate, or unexpected readback keys')
        if finished['items'] != len(current) or finished['expected_items'] != len(expected):
            raise ValueError(f'Shard {shard} completion counts differ')
        for record in current:
            row = by_id[record['pair_id']]
            if record['binding_sha256'] != binding_digest(binding) or record['split'] != split or record['kind'] != kind:
                raise ValueError('Readback row belongs to another evaluation')
            order, position = eval_order(record['pair_id'], record['question_family'], record['correct_position'])
            query = row['teacher_question_forms'][record['question_family']] + mcq['get_mcq_question_format']([row['options'][i] for i in order])
            predicted = mcq['extract_choice'](record['generated']['raw'])
            if (record['option_order'] != order or record['correct_letter'] != 'ABCD'[position] or
                    record['reader_query'] != query or record['predicted_letter'] != predicted or
                    type(record['correct']) is not bool or record['correct'] != (predicted == 'ABCD'[position]) or
                    type(record['parse_failure']) is not bool or record['parse_failure'] != (predicted is None)):
                raise ValueError('Readback query, option mapping, or stored score is inconsistent')
            expected_donor = donors[record['pair_id']] if record['control'] == 'mismatch' else None
            if record.get('donor_pair_id') != expected_donor:
                raise ValueError('Readback mismatch donor differs from the registered mapping')
        if finished['counts'] != _receipt_counts(current):
            raise ValueError(f'Shard {shard} receipt metric counts differ from raw readback')
        all_keys.update(keys)
        records.extend(current)
        receipts.append({'shard': shard, 'receipt_sha256': sha(finished_path), 'readback_sha256': sha(log_path), 'items': len(current)})
    if all_keys != set(planned_read_keys(rows, args)):
        raise ValueError('Combined shards do not cover every registered preference/measurement')
    result = {'schema': 'prefeval-aug4-summary/v1', 'status': 'verified_complete', 'split': split, 'kind': kind,
              'binding': shared, 'source_receipts': receipts, 'summarizer_source_sha256': sha(Path(__file__)),
              **_summarize_records(records),
              'statistical_unit': 'base_pair_id; average seeds, positions and applicable question suites within preference first',
              'interval_scope': 'preference sampling uncertainty for this paired training seed, not between-training-seed uncertainty',
              'score_policy': 'unchanged upstream extract_choice; parse failures count as incorrect and remain in denominators'}
    if split == 'official':
        excluded = _official_exclusions(data)
        clean = [r for r in records if r['pair_id'] not in excluded]
        result['official177'] = {'excluded_preference_ids': sorted(excluded), **_summarize_records(clean)}
    if split == 'opposites':
        result['counterfactual_diagnostic'] = {
            'scope': 'counterfactual preferences with disambiguating additions; not a minimal-negation causal experiment',
            'mismatch_meaning': 'the corresponding original-preference PNG read under the counterfactual answer label',
            'changed_preference_accuracy': result['controls']['memory']['preference_mean_rate'],
            'memory_gain_over_original_png': result['paired_memory_gain']['mismatch'],
            'original_pairs': {r['base_pair_id']: {'origin_id': r['origin_id'],
                'new_correct_option_original_index': r['new_correct_option_original_index']} for r in rows},
            'limitation': 'does not on its own measure joint success on both original and counterfactual inputs'}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary = summarize(args.data, args.input)
    destination = args.output or args.input / 'summary.json'
    save_json(destination, summary)
    print(json.dumps({'output': str(destination), 'status': summary['status'],
                      'preferences': summary['preference_count'], 'measurements': summary['measurement_count']}))
