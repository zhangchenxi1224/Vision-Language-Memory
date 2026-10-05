"""No GPU or historical scores: synthetic examples exercise comparison failure modes."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reporting import compare_prompt_matching as comparison  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def fixture_runs(tmp_path, monkeypatch):
    registry = {'pilot': {'train-1', 'train-2'}, 'train': {'train-1', 'train-2', 'train-3'},
                'dev': {'dev-1', 'dev-2', 'dev-3'}, 'official': {'test-1', 'test-2'}}
    monkeypatch.setattr(comparison, 'registered_ids', lambda: registry)
    protocol = json.loads(comparison.DEFAULT_PROTOCOL.read_text(encoding='utf-8'))
    protocol['comparison']['bootstrap_samples'] = 100
    protocol_path = tmp_path / 'protocol.json'
    save(protocol_path, protocol)

    def make(label, mode, split='dev', kind='student', phase='retention', correct=False):
        directory = tmp_path / label
        directory.mkdir()
        judges = directory / 'judges'
        judges.mkdir()
        manifest = {
            'schema': 'vision_memory.prompt_matching_readback.v1', 'label': label, 'supervision': mode,
            'format_arm': 'A', 'split': split, 'endpoint_kind': kind, 'training_stage': 'pilot',
            'evaluation_phase': phase, 'expected_ids': sorted(registry[split]),
            'training_ids': sorted(registry['pilot']), 'selection_ids': [],
            'reader_revision': 'reader-commit', 'tokenizer_revision': 'tokenizer-commit',
            'chat_template_sha256': '0' * 64, 'dataset_sha256': '1' * 64, 'history_sha256': '2' * 64,
            'history_protocol': 'synthetic fixture history', 'writer_base_sha256': '3' * 64,
            'checkpoint_sha256': '4' * 64, 'training': protocol['training'],
            'protocol_sha256': comparison.digest(protocol_path), 'readback_files': ['readback-0.jsonl'],
            'judge_dirs': ['judges'], 'judge': {'judge_model': 'fixed-judge', 'model_label': 'substitute_judge',
                                             'provider': 'openai-compatible', 'max_tokens': 100,
                                             'enable_thinking': False},
        }
        rows = []
        for i, key in enumerate(sorted(comparison.expected_keys(manifest, protocol))):
            row = dict(zip(comparison.IDENTITY, key))
            success = correct and row['control'] in {'memory', 'text'}
            row.update(split=split, endpoint_kind=kind, history_protocol=manifest['history_protocol'],
                       max_new_tokens=protocol['evaluation']['max_new_tokens'][row['task']],
                       png_sha256='5' * 64 if row['control'] in {'memory', 'mismatch'} else None,
                       question='Question ' + row['family'], reader_query='Query ' + row['family'],
                       preference='Preference ' + row['pair_id'],
                       donor_pair_id='donor' if row['control'] == 'mismatch' else None,
                       generated={'raw': '<choice>A</choice>' if success else 'unparseable answer',
                                  'truncated': False})
            if row['task'] == 'mcq':
                row.update(option_order=[0, 1, 2, 3], correct_letter='A', correct=success,
                           parse_failure=not success, predicted_letter='A' if success else None)
            else:
                row['official_judge_status'] = 'pending'
                save(judges / f'{i}.json', {'input': row, 'status': 'complete', 'correct': success,
                                          **manifest['judge']})
            rows.append(row)
        readback = directory / 'readback-0.jsonl'
        readback.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        path = directory / 'manifest.json'
        save(path, manifest)
        return path, readback, judges

    return make, protocol_path


def mutate_json(path, **updates):
    value = json.loads(path.read_text(encoding='utf-8'))
    value.update(updates)
    save(path, value)


def mutate_row(path, change):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    change(rows)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def test_complete_paired_evidence_never_switches_default(fixture_runs):
    make, protocol = fixture_runs
    a, _, _ = make('a', 'history_hard')
    b, _, _ = make('b', 'prompt_matching', correct=True)
    result = comparison.compare(a, b, protocol)
    assert result['status'] == 'eligible_for_promotion_review'
    assert result['automatic_default_switch'] is False
    assert result['primary']['preferences'] == 3  # Not 3 preferences x 2 chains x 2 wordings.
    assert result['primary']['delta'] == 1
    assert result['primary']['interval'] == [1, 1]
    assert len(result['metrics']) == 3 * 2 * 4


@pytest.mark.parametrize('kind,phase', [('student', 'retention'), ('teacher', 'initial')])
def test_training_pilot_and_teacher_cannot_promote(fixture_runs, kind, phase):
    make, protocol = fixture_runs
    a, _, _ = make('a', 'hard_ce', split='pilot', kind=kind, phase=phase)
    b, _, _ = make('b', 'prompt_matching', split='pilot', kind=kind, phase=phase, correct=True)
    result = comparison.compare(a, b, protocol)
    assert result['status'] == 'diagnostic_only_not_promoted'
    assert result['gates']['heldout_student_retention'] is False


@pytest.mark.parametrize('field,value,match', [
    ('split', 'official', 'complete registered split'),
    ('training_ids', ['train-1'], 'registered stage'),
    ('selection_ids', ['dev-1'], 'training side'),
    ('reader_revision', 'another-reader', 'Unpaired experiment'),
    ('protocol_sha256', '9' * 64, 'protocol hash'),
])
def test_unpaired_or_leaking_manifests_fail(fixture_runs, field, value, match):
    make, protocol = fixture_runs
    a, _, _ = make('a', 'hard_ce')
    b, _, _ = make('b', 'prompt_matching')
    mutate_json(b, **{field: value})
    with pytest.raises(ValueError, match=match):
        comparison.compare(a, b, protocol)


@pytest.mark.parametrize('case,match', [
    ('missing', 'Missing paired readbacks'),
    ('split', 'readback splits'),
    ('duplicate', 'Conflicting duplicate'),
    ('png', 'same frozen PNG'),
    ('mcq', 'official parser'),
])
def test_readback_integrity_failures(fixture_runs, case, match):
    make, protocol = fixture_runs
    a, _, _ = make('a', 'hard_ce')
    b, readback, _ = make('b', 'prompt_matching')
    def change(rows):
        index = next(i for i, row in enumerate(rows) if row['control'] == 'memory' and row['task'] == 'mcq')
        if case == 'missing':
            rows.pop(index)
        elif case == 'split':
            rows[index]['split'] = 'official'
        elif case == 'duplicate':
            duplicate = copy.deepcopy(rows[index])
            duplicate['generated']['raw'] = 'conflicting duplicate'
            rows.append(duplicate)
        elif case == 'png':
            rows[index]['png_sha256'] = '6' * 64
        else:
            rows[index]['correct'] = True
    mutate_row(readback, change)
    with pytest.raises(ValueError, match=match):
        comparison.compare(a, b, protocol)


@pytest.mark.parametrize('case,match', [('missing', 'Missing complete judgments'),
                                       ('pending', 'pending free-response'), ('model', 'judge settings')])
def test_missing_and_mixed_judges_fail_closed(fixture_runs, case, match):
    make, protocol = fixture_runs
    a, _, _ = make('a', 'hard_ce')
    b, _, judges = make('b', 'prompt_matching')
    path = next(judges.glob('*.json'))
    if case == 'missing':
        path.unlink()
    elif case == 'pending':
        mutate_json(path, status='judge_parse_failure')
    else:
        mutate_json(path, judge_model='another-judge')
    with pytest.raises(ValueError, match=match):
        comparison.compare(a, b, protocol)


def test_paired_bootstrap_is_reproducible_and_clustered():
    kwargs = {'samples': 1000, 'seed': 20261005, 'confidence': .95}
    first = comparison.paired_interval([0, .5, 1], [.5, 1, 0], **kwargs)
    assert first == comparison.paired_interval([0, .5, 1], [.5, 1, 0], **kwargs)
    assert first['preferences'] == 3 and first['delta'] == 0
    assert first['interval'][0] < 0 < first['interval'][1]
