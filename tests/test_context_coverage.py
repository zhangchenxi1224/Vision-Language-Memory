from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiments.prefeval_context_coverage import (
    training_context, context_manifest, choose_checkpoint, family_mean, VALIDATION,
)


def example():
    return {'base_pair_id': 'topic:1', 'forms': {**{f'T{i}': f'train {i}' for i in range(1, 4)},
        'O1': 'HELDOUT', 'O2': 'HELDOUT2'}, 'options': ['a', 'b', 'c', 'd'],
        'target': 'GOLD', 'history': ['past', 'past', 'FUTURE']}


MCQ = {'get_mcq_question_format': lambda values: ' choices=' + repr(values)}


def test_coverage_and_mcq_permutations_are_balanced():
    values = [training_context(example(), s, MCQ) for s in range(288)]
    assert Counter(f for _, f, _ in values) == {'recall': 96, 'application_mcq': 96, 'application_open': 48, 'neutral': 48}
    crosses = Counter((q[:7], order.index(0)) for q, f, order in values if f == 'application_mcq')
    assert len(crosses) == 12 and set(crosses.values()) == {8}


def test_future_answers_and_heldout_questions_cannot_change_contexts():
    row = example()
    before = context_manifest([row], 288, MCQ)
    changed = deepcopy(row)
    changed.update(target='DIFFERENT', history=['anything'], query='FUTURE QUERY')
    changed['forms'].update(O1='changed', O2='also changed')
    assert context_manifest([changed], 288, MCQ) == before
    text = str(before)
    assert 'HELDOUT' not in text and 'FUTURE' not in text and 'GOLD' not in text
    assert {q for q, _, _ in before['schedule'][row['base_pair_id']]}.isdisjoint(q for _, q in VALIDATION)


def test_selection_requires_complete_denominator_and_breaks_ties_by_step():
    assert choose_checkpoint([{'step': 144, 'score': .2}, {'step': 72, 'score': .2}], [72,144])['step'] == 72
    for rows in ([{'step':72,'score':.1}], [{'step':72,'score':.1}]*2,
                 [{'step':72,'score':float('nan')},{'step':144,'score':.2}]):
        with pytest.raises(ValueError):
            choose_checkpoint(rows,[72,144])
    assert family_mean([1,1,2,2,3,3]) == 2
    with pytest.raises(ValueError):
        family_mean([0,0])


def test_bounded_plan_does_not_retrain_originals_or_writer(tmp_path):
    from types import SimpleNamespace
    from scripts.inspire.run_context_coverage import build_plan
    ids=tmp_path/'ids.json'
    ids.write_text('{"ids":["topic:1"]}')
    args=SimpleNamespace(output=tmp_path/'out', phase='pilot', ids_file=ids,
                         base=tmp_path/'base', reader=tmp_path/'reader', prefeval=tmp_path/'prefeval', reference=tmp_path/'reference')
    plan=build_plan(args)
    jobs=[j for g in plan['groups'] for j in g]
    assert len(jobs)==10 and not plan['new_writer_training']
    assert all(j['gpu'] in (0,1) for j in jobs)
    assert all('prefeval_k1_writer.py' not in ' '.join(j['command']) for j in jobs)
    assert sum('--selected-teacher' in j['command'] for j in jobs)==2
