import json
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire import run_context_layout_aug_readout as m


def fixture_rows():
    spec=m.read(m.p.PROTOCOL);ids=[r['base_pair_id'] for r in m.p.population()[0]]
    def rows(keys):
        result=[]
        for key in sorted(keys):
            pid,qid,e,c,n=key;r=dict(zip(m.KEYS,key))
            value=2.+(c=='mismatch')
            if e.startswith('augmented-') and c=='memory':value-=.25
            r.update(kl=value,teacher_target=pid+qid,target_ids=[1,2],teacher_logits_sha256=pid+qid);result.append(r)
        return result
    return spec,rows(m.old_keys(ids,spec)),rows(m.new_keys(ids,spec,m.ENDPOINTS))


def test_all_formats_strata_denominator_and_paired_direction():
    spec,old,new=fixture_rows();v=m.summarize(old,new,spec)
    assert (v['combined_rows'],v['new_rows'],v['reused_rows'])==(16128,9216,6912)
    for group in v['strata'].values():
        assert group['independent_n']==16
        for f in group['families'].values():
            assert set(f['formats'])=={'canonical','markdown','xml'}
            for style,g in f['formats'].items():
                assert g['control_minus_augmented_memory']['mean']==.25
                assert g['augmented_minus_control_specificity']['mean']==.25
                assert g['endpoints']['augmented']['mismatch_minus_memory']['mean']==1.25
                assert ('parent256_memory_kl' in g)==(style!='xml')


@pytest.mark.parametrize('bad',['missing','duplicate','prefix','foreign','nan'])
def test_corrupt_new_readout_refused(bad):
    spec,old,new=fixture_rows()
    if bad=='missing':new.pop()
    if bad=='duplicate':new.append(new[0])
    if bad=='prefix':new[0]=dict(new[0],target_ids=[9])
    if bad=='foreign':new[0]=dict(new[0],pair_id='not-trained')
    if bad=='nan':new[0]=dict(new[0],kl=float('nan'))
    with pytest.raises(ValueError):m.summarize(old,new,spec)


def test_parent_rows_cannot_be_counted_twice():
    spec,old,new=fixture_rows()
    with pytest.raises(ValueError,match='Duplicate'):m.summarize(old+old[:1],new,spec)


def test_split_stage_caps_include_failed_training_and_readout_once(tmp_path,monkeypatch):
    monkeypatch.setattr(m.a,'budget',lambda _:(0,28000,2500))
    p=tmp_path/'attempts';p.mkdir();(p/'failed.json').write_text(json.dumps(dict(started=1,finished=101,exit_code=124)))
    assert m.remaining_seconds(tmp_path)==(8000,30500,100)
    monkeypatch.setattr(m.a,'budget',lambda _:(0,28000,3000))
    assert m.remaining_seconds(tmp_path)==(7700,31000,100)
    (p/'active.json').write_text(json.dumps(dict(started=1)))
    with pytest.raises(ValueError,match='Unsettled'):m.remaining_seconds(tmp_path)
