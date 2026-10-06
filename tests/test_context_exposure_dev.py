import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire import run_context_exposure_dev as m
from scripts.experiments.prefeval_route_functional import old_keys


def fixture_rows():
    spec=m.read(m.PROTOCOL)
    ids=[r['base_pair_id'] for r in m.cohort_rows()['dev']]
    def rows(keys):
        values=[]
        for key in sorted(keys):
            pid,qid,endpoint,control,chain=key
            row=dict(zip(m.KEYS,key))
            row.update(kl=(1. if endpoint==m.ENDPOINT else 2.)+(control=='mismatch'),
                teacher_target=pid+qid,target_ids=[1,2],teacher_logits_sha256=pid+qid)
            values.append(row)
        return values
    return spec,rows(old_keys(ids,spec['queries'])),rows(m.new_keys(ids,spec,m.d.ENDPOINTS)),rows(m.new_keys(ids,spec,(m.ENDPOINT,)))


def test_full_denominator_and_preference_pairing():
    spec,old,prior,new=fixture_rows()
    result=m.summarize(old,prior,new,spec)
    assert (result['new_rows'],result['reused_rows'],result['combined_rows'],result['independent_n'])==(4320,23760,28080,90)
    for family in result['families'].values():
        value=family['endpoints'][m.ENDPOINT]
        assert value['paired_reference']=='context-diverse'
        assert value['paired_reference_minus_memory']['mean']==1
        assert value['mismatch_minus_memory']['mean']==1


@pytest.mark.parametrize('corruption',['missing','duplicate','prefix','foreign'])
def test_invalid_or_unpaired_readout_rejected(corruption):
    spec,old,prior,new=fixture_rows()
    if corruption=='missing':new.pop()
    if corruption=='duplicate':new.append(new[0])
    if corruption=='prefix':new[0]=dict(new[0],target_ids=[99])
    if corruption=='foreign':new[0]=dict(new[0],pair_id='not-dev')
    with pytest.raises(ValueError):m.summarize(old,prior,new,spec)


def test_budget_includes_finished_exposure_and_failed_new_attempt_once(tmp_path,monkeypatch):
    monkeypatch.setattr(m.e,'budget',lambda _: (0,20000,1000))
    attempts=tmp_path/'attempts';attempts.mkdir()
    (attempts/'failed.json').write_text(json.dumps(dict(started=1,finished=301,exit_code=124)))
    receipts=tmp_path/'receipts';receipts.mkdir()
    (receipts/'duplicate.json').write_text(json.dumps(dict(started=1,finished=301)))
    assert m.remaining_seconds(tmp_path)==(6900,21000,300)
    (attempts/'unfinished.json').write_text(json.dumps(dict(started=500)))
    with pytest.raises(ValueError,match='Unsettled'):m.remaining_seconds(tmp_path)


def test_frozen_endpoint_has_no_training_or_selection_and_dev_is_disjoint():
    args=SimpleNamespace(output=Path('/out'),base=Path('/base'),official_source=Path('/official'))
    job=m.rollout_job(args);cmd=job['command']
    assert 'train' not in cmd and '--steps' not in cmd and '--snapshot-steps' not in cmd
    assert cmd[cmd.index('--checkpoint')+1]==str(m.SOURCE/'train/checkpoint-final.pt')
    assert cmd[cmd.index('--split')+1]=='dev'
    ids={r['base_pair_id'] for r in m.cohort_rows()['dev']}
    assert len(ids)==90 and not ids & {r['base_pair_id'] for r in m.p.population()[0]}
