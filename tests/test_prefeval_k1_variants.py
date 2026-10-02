import collections
import json
import pytest
from scripts.experiments.prefeval_k1_variants import load_variants,apply_variant,training_variant,select_fm_target


def test_full_budget_balances_both_training_expressions_and_excludes_v2():
    counts=collections.Counter()
    for draw in range(23360*4):
        cycle,offset=divmod(draw,730)
        counts[offset,training_variant(cycle,True)]+=1
    assert len(counts)==1460
    assert set(counts.values())=={64}
    assert {key[1] for key in counts}=={0,1}


def test_variants_preserve_user_and_future_history_and_reject_preference_change(tmp_path):
    rows=[{'base_pair_id':'p:0','history':[{'role':'user','content':'Avoid spicy food.'},
        {'role':'assistant','content':'original'},{'role':'user','content':'unrelated'}]}]
    data={'review_complete':True,'items':{'p:0':{'preference':'Avoid spicy food.',
                                             'variants':['original','noted','thank you']}}}
    path=tmp_path/'variants.json'
    path.write_text(json.dumps(data))
    variants=load_variants(path,rows)
    changed=apply_variant(rows[0],variants,1)
    assert changed['history'][0]==rows[0]['history'][0]
    assert changed['history'][2:]==rows[0]['history'][2:]
    assert rows[0]['history'][1]['content']=='original'
    data['items']['p:0']['preference']='Likes spicy food.'
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):load_variants(path,rows)


def test_identity_target_changes_only_retention_not_initial_writes():
    teacher,source=object(),object()
    for position in range(11):
        assert select_fm_target(teacher,source,position,False) is teacher
        assert select_fm_target(teacher,source,position,True) is (teacher if position==0 else source)
