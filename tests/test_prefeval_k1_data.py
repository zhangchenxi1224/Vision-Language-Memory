from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.experiments.prefeval_k1_data import load_records, option_order, official_mcq, event_text

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

def test_upstream_prompt_and_parser():
    funcs = official_mcq(ROOT / 'third_party/prefeval_reference')
    assert 'A. a\nB. b\nC. c\nD. d' in funcs['get_mcq_question_format'](['a','b','c','d'])
    assert funcs['extract_choice']('<choice>C</choice>') == 'C'
    assert funcs['extract_choice']('C') is None
