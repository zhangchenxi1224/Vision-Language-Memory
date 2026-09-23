from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.prefeval_k1_data import load_records, option_order, official_mcq, event_text, official_eval_disclosures, ALIGN, sha

def test_pairing_and_no_query_in_writer():
    rows = load_records()
    assert len(rows) == 64
    for row in rows:
        assert set(row['forms']) == {'T1', 'T2', 'T3', 'O1', 'O2'}
        assert row['forms']['T1'] == row['query']['content']
        assert len(row['history']) == 22
        assert len(row['options']) == 4
        text = event_text(row['history'][:2])
        assert text == 'user: ' + row['history'][0]['content'] + '\nassistant: ' + row['history'][1]['content']
        assert row['query']['content'] not in text
        assert row['target']['content'] not in text

def test_position_and_form_balance():
    for row in load_records():
        counts = Counter()
        for step in range(288):
            order, correct = option_order(row['base_pair_id'], step)
            assert sorted(order) == [0, 1, 2, 3] and order[correct] == 0
            counts[step % 3, correct] += 1
        assert len(counts) == 12 and set(counts.values()) == {24}


def test_full_train_forms_preserve_pilot_and_exclude_heldout():
    rows=load_records('train')
    by_id={r['base_pair_id']:r for r in rows}
    assert len(by_id)==len(rows)==730
    assert set(by_id).isdisjoint(r['base_pair_id'] for r in load_records('dev'))
    assert set(by_id).isdisjoint(r['base_pair_id'] for r in official_eval_disclosures())
    for row in load_records('pilot'):
        assert by_id[row['base_pair_id']]['forms']==row['forms']
    for row in rows:
        assert set(row['forms'])=={'T1','T2','T3','O1','O2'}
        assert row['forms']['T1']==row['query']['content']
        assert len(set(row['forms'].values()))==5


def test_expanded_teacher_partition_and_training_side_only():
    import pytest
    from scripts.experiments.prefeval_k1_data import load_training_records
    pilot={r['base_pair_id'] for r in load_training_records()}
    remaining={r['base_pair_id'] for r in load_training_records('train', exclude_pilot=True)}
    full={r['base_pair_id'] for r in load_training_records('train')}
    assert len(pilot)==64 and len(remaining)==666 and len(full)==730
    assert pilot.isdisjoint(remaining) and pilot|remaining==full
    for split in ['dev','official']:
        with pytest.raises(ValueError):
            load_training_records(split)
    with pytest.raises(ValueError):
        load_training_records('pilot', exclude_pilot=True)

def test_upstream_prompt_and_parser():
    funcs = official_mcq(ROOT / 'third_party/prefeval_reference')
    assert 'A. a\nB. b\nC. c\nD. d' in funcs['get_mcq_question_format'](['a','b','c','d'])
    assert funcs['extract_choice']('<choice>C</choice>') == 'C'
    assert funcs['extract_choice']('C') is None


def test_official_history_uses_tested_reader_ack_and_benchmark_context(tmp_path):
    disclosures=official_eval_disclosures()
    path=tmp_path/'acknowledgments.json'
    path.write_text(json.dumps({'binding':{'benchmark_sha256':sha(ALIGN/'data/benchmark-disclosures.jsonl.gz')},
        'acknowledgments':{r['base_pair_id']:{'preference':r['input']['disclosure'][0]['content'],
            'generated':{'raw':'TESTED_READER_ACK '+r['base_pair_id']}} for r in disclosures}}),encoding='utf-8')
    rows=load_records('official',history_file=path)
    contexts=json.loads((ALIGN/'data/context-pools.json').read_text(encoding='utf-8'))
    assert len(rows)==180
    assert {r['topic'] for r in rows}.isdisjoint({r['topic'] for r in load_records('train')})
    for row,bench in zip(rows,disclosures):
        assert row['history'][0]==bench['input']['disclosure'][0]
        assert row['history'][1]['content']=='TESTED_READER_ACK '+row['base_pair_id']
        assert row['history'][2:]==contexts['benchmark'][:20]
        assert row['query']==bench['input']['query']
        assert 'target' not in row
        assert len(row['history'])==22
