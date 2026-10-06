import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire.run_context_dev import ENDPOINTS,build_jobs,remaining_seconds,verify_source
from scripts.experiments.prefeval_route_functional import new_keys,old_keys,summarize,cohort_rows
SPEC=json.loads((ROOT/'configs/experiments/context_readout_audit.json').read_text())


def synthetic(keys):
    return [dict(pair_id=p,query_id=q,endpoint=e,control=c,chain=n,kl=4.,
        target_ids=[7,9],teacher_logits_sha256='same',teacher_target=p+'-'+q)
        for p,q,e,c,n in keys]


def test_dev_denominator_scope_and_paired_statistics():
    ids=[r['base_pair_id'] for r in cohort_rows()['dev']]
    assert len(new_keys(ids,SPEC,ENDPOINTS))==8640
    assert len(old_keys(ids,SPEC['queries']))==15120
    ids=['a','b'];old=synthetic(old_keys(ids,SPEC['queries']));new=synthetic(new_keys(ids,SPEC,ENDPOINTS))
    for r in new:
        if r['endpoint']=='context-diverse' and r['control']=='memory':r['kl']=1+2*r['chain']
    value=summarize(old,new,ids,SPEC,ENDPOINTS,{'context-narrow':'b730','context-diverse':'context-narrow'},cohort='dev')
    assert 'exploratory internal dev' in value['metric']
    assert value['families']['recall']['endpoints']['context-diverse']['paired_reference_minus_memory']['mean']==2
    assert value['independent_n']==2
    with pytest.raises(ValueError,match='denominator'):summarize(old,new[:-1],ids,SPEC,ENDPOINTS,{},cohort='dev')


def test_no_training_no_selected_weights_no_official_data(tmp_path):
    args=SimpleNamespace(output=tmp_path/'out',source=tmp_path/'fit',base=Path('/base'),official_source=Path('/official'))
    for j,e in zip(build_jobs(args),ENDPOINTS):
        cmd=j['command']
        assert cmd[2]=='rollout' and cmd[cmd.index('--split')+1]=='dev'
        assert cmd[cmd.index('--checkpoint')+1]==str(args.source/e/'train/checkpoint-final.pt')
        assert '--ids-file' not in cmd and '--teachers' not in cmd


def test_cost_includes_completed_fit_failures_and_prior_routes(tmp_path):
    for name,seconds in [('context-fit-v1',300),('context-dev-v1',200),('writer-readout-v1',400),('route-functional-v1',100)]:
        folder=tmp_path/name/'attempts';folder.mkdir(parents=True)
        (folder/'a.json').write_text(json.dumps(dict(started=0,finished=seconds,exit_code=124)))
    assert remaining_seconds(tmp_path/'context-dev-v1')==(7000,800,200)
    p=tmp_path/'context-dev-v1/attempts/live.json';p.write_text(json.dumps(dict(started=1)))
    with pytest.raises(ValueError,match='Unsettled'):remaining_seconds(tmp_path/'context-dev-v1')


def test_partial_fit_cannot_be_consumed(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps(dict(status='rollout')))
    with pytest.raises(ValueError,match='incomplete'):verify_source(tmp_path)
