import json
from types import SimpleNamespace

import pytest

from scripts.experiments.prefeval_aug4_evaluate import KEY_FIELDS, binding_digest, eval_order, planned_read_keys
from scripts.experiments.prefeval_aug4_teacher import save_json
from scripts.experiments.prefeval_k1_data import official_mcq, sha
from scripts.reporting import summarize_prefeval_aug4 as report


def fixture(tmp_path):
    rows = [{'base_pair_id': f'topic:{i}', 'topic': 'topic',
             'writer_user_forms': {'W0': f'Preference {i}'},
             'teacher_question_forms': {'T1': f'Question {i}?'},
             'options': ['correct', 'wrong one', 'wrong two', 'wrong three']} for i in range(2)]
    data = tmp_path / 'data.json'
    save_json(data, {'dev': rows})
    out = tmp_path / 'eval'
    out.mkdir()
    mcq_path = report.ROOT / 'third_party/prefeval_reference/utils/utils_mcq.py'
    mcq = official_mcq(mcq_path.parents[1])
    binding = {'schema': 'prefeval-aug4-evaluation/v1', 'data_sha256': sha(data),
               'evaluation_source_sha256': 'fixture', 'kind': 'student', 'split': 'dev',
               'checkpoint': {'checkpoint_sha256': 'frozen', 'optimizer_step': 5840},
               'seeds': [0, 1], 'controls': ['memory', 'mismatch', 'blank', 'text'],
               'suites': [['W0', 'T1']], 'correct_positions': [0, 1, 2, 3],
               'shard': 0, 'shards': 1, 'assignment': [r['base_pair_id'] for r in rows],
               'reader': 'frozen', 'canonical_images': None, 'mcq_source_sha256': sha(mcq_path)}
    save_json(out / 'identity-0.json', binding)
    args = SimpleNamespace(kind='student', split='dev', seeds=[0, 1], controls=binding['controls'])
    records = []
    for key in planned_read_keys(rows, args):
        pid, seed, w, q, control, position = key
        row = rows[int(pid[-1])]
        order, correct = eval_order(pid, q, position)
        is_correct = (control == 'memory' and pid == 'topic:0') or (control == 'mismatch' and pid == 'topic:1')
        predicted = 'ABCD'[correct if is_correct else (correct + 1) % 4]
        if pid == 'topic:0' and control == 'memory' and seed == 0 and position == 0:
            predicted = None
        record = dict(zip(KEY_FIELDS, key))
        record.update({'binding_sha256': binding_digest(binding), 'kind': 'student', 'split': 'dev',
                       'option_order': order, 'correct_letter': 'ABCD'[correct],
                       'reader_query': row['teacher_question_forms'][q] + mcq['get_mcq_question_format']([row['options'][i] for i in order]),
                       'generated': {'raw': f'<choice>{predicted}</choice>' if predicted else 'No answer.'},
                       'predicted_letter': predicted, 'correct': predicted == 'ABCD'[correct],
                       'parse_failure': predicted is None,
                       'donor_pair_id': f'topic:{1 - int(pid[-1])}' if control == 'mismatch' else None})
        records.append(record)
    log = out / 'readback-0.jsonl'
    log.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    save_json(out / 'read-finished-0.json', {'binding': binding, 'status': 'completed', 'items': len(records),
        'expected_items': len(records), 'readback_sha256': sha(log), 'counts': report._receipt_counts(records)})
    return data, out, records


def test_real_summary_keeps_parse_failure_and_bootstraps_two_preferences(tmp_path):
    data, out, records = fixture(tmp_path)
    result = report.summarize(data, out)
    assert result['measurement_count'] == 48
    assert result['controls']['memory']['total'] == 16
    assert result['controls']['memory']['correct'] == 7
    assert result['controls']['memory']['parse_failures'] == 1
    assert result['paired_memory_gain']['mismatch']['preference_count'] == 2
    assert result['paired_memory_gain']['mismatch']['mean'] == pytest.approx(-.0625)
    assert result['paired_memory_gain']['blank']['mean'] == pytest.approx(.4375)
    assert result['paired_memory_gain']['mismatch']['bootstrap_samples'] == 10000
    assert result['paired_memory_gain']['mismatch']['ci95'] == [-1.0, .875]


def test_recomputed_checksums_cannot_hide_omitted_observations(tmp_path):
    data, out, records = fixture(tmp_path)
    records = records[1:]
    log = out / 'readback-0.jsonl'
    log.write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
    receipt = report.read_json(out / 'read-finished-0.json')
    receipt.update(items=len(records), expected_items=len(records), readback_sha256=sha(log),
                   counts=report._receipt_counts(records))
    save_json(out / 'read-finished-0.json', receipt)
    with pytest.raises(ValueError, match='missing, duplicate, or unexpected'):
        report.summarize(data, out)
