import json
from scripts.experiments.prefeval_multitarget_bank import freeze_bank, select_rows, target_index, fixed_single_bank


def test_empty_preferences_cannot_be_silently_dropped(tmp_path):
    bank=freeze_bank(tmp_path,['a','b'],'F8')
    assert not bank['ready'] and bank['missing_preferences']==['a','b']


def test_reject_failed_targets_and_f1_is_fixed_subset(tmp_path):
    for k,qualified in enumerate([True,False,True]):
        d=tmp_path/'teachers/F8/a'/f'target-{k}'
        d.mkdir(parents=True)
        (d/'complete.json').write_text(json.dumps({'qualified':qualified,'latent_sha256':str(k),
            'png_sha256':str(k),'target_index':k}))
    f8=freeze_bank(tmp_path,['a'],'F8')
    f1=freeze_bank(tmp_path,['a'],'F1')
    assert f8['ready'] and [x['target_index'] for x in f8['targets']['a']]==[0,2]
    assert len(f1['targets']['a'])==1 and f1['targets']['a'][0] in f8['targets']['a']
    assert f1==freeze_bank(tmp_path,['a'],'F1')


def test_ids_cannot_cross_split(tmp_path):
    import pytest
    path=tmp_path/'ids.json'
    path.write_text(json.dumps({'ids':['heldout']}))
    with pytest.raises(ValueError): select_rows([{'base_pair_id':'train'}],path)


def test_target_draw_has_coverage_and_is_reproducible():
    draws=[target_index('pref',i,8) for i in range(256)]
    assert set(draws)==set(range(8))
    assert draws==[target_index('pref',i,8) for i in range(256)]


def test_s1_preserves_preferences_and_does_not_mutate_source():
    bank={'ready':True,'missing_preferences':[],'arm':'S8',
          'targets':{'a':[{'target_index':1},{'target_index':3}],'b':[{'target_index':2}]},
          'coverage':{'a':{'qualified':2},'b':{'qualified':1}}}
    result=fixed_single_bank(bank,'S1')
    assert set(result['targets'])=={'a','b'}
    assert all(len(v)==1 for v in result['targets'].values())
    assert len(bank['targets']['a'])==2 and result['coverage']==bank['coverage']
    assert result==fixed_single_bank(bank,'S1')


def test_s1_rejects_empty_qualified_pool():
    import pytest
    with pytest.raises(ValueError):
        fixed_single_bank({'ready':True,'missing_preferences':[],'targets':{'a':[]}},'S1')
