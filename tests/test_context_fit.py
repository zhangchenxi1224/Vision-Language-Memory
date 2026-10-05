import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire.run_context_fit import ENDPOINTS,build_jobs,remaining_seconds,verify_training,BANKS,PARENT_SHA
from scripts.experiments.prefeval_route_functional import new_keys,old_keys,summarize,sha,cohort_rows
SPEC=json.loads((ROOT/'configs/experiments/context_readout_audit.json').read_text())


def synthetic(keys):
    return [dict(pair_id=p,query_id=q,endpoint=e,control=c,chain=n,kl=4.,
        target_ids=[7,9],teacher_logits_sha256='same',teacher_target=p+'-'+q)
        for p,q,e,c,n in keys]


def test_fit_denominator_and_paired_context_contrast():
    ids=[r['base_pair_id'] for r in cohort_rows()['pilot']]
    assert len(new_keys(ids,SPEC,ENDPOINTS))==1536
    assert len(old_keys(ids,SPEC['queries']))==2688
    ids=['a','b'];old=synthetic(old_keys(ids,SPEC['queries']));new=synthetic(new_keys(ids,SPEC,ENDPOINTS))
    for r in new:
        if r['endpoint']=='context-diverse' and r['control']=='memory':r['kl']=1+2*r['chain']
    result=summarize(old,new,ids,SPEC,ENDPOINTS,{'context-narrow':'b730','context-diverse':'context-narrow'})
    d=result['families']['recall']['endpoints']['context-diverse']
    assert d['memory_kl']==2 and d['paired_reference_minus_memory']['mean']==2
    assert d['paired_reference_minus_memory']['independent_n']==2


def test_jobs_only_pilot_and_same_parent_train_budget(tmp_path):
    args=SimpleNamespace(output=tmp_path,base=Path('/model'),official_source=Path('/official'))
    for phase in ('train','rollout'):
        jobs=build_jobs(args,phase)
        assert [j['gpu'] for j in jobs]==[0,1]
        for j in jobs:
            cmd=j['command']
            assert cmd[cmd.index('--split')+1]=='pilot'
            assert '--ids-file' in cmd
            if phase=='train':
                assert cmd[cmd.index('--steps')+1]=='128'
                assert cmd[cmd.index('--teacher-supervision')+1]=='prompt_matching'
            else:assert cmd[cmd.index('--inter-turns')+1]=='0'


def test_fit_budget_includes_route_and_failed_current(tmp_path):
    for name,seconds in [('route-functional-v1',120),('writer-readout-v1',240),('context-fit-v1',300)]:
        folder=tmp_path/name/'attempts';folder.mkdir(parents=True)
        (folder/'a.json').write_text(json.dumps(dict(started=0,finished=seconds,exit_code=124)))
    assert remaining_seconds(tmp_path/'context-fit-v1')==(3300,360)


def test_training_audit_rejects_unpaired_draw_and_zero_gradient(tmp_path):
    ids=['a'];banks={'ids':ids,'files':{},'teacher_bindings':{}}
    for e in ENDPOINTS:
        banks['files'][str(BANKS[e]/'a/latent.pt')]='target-'+e
        banks['teacher_bindings'][e+'|a']='binding-'+e
        folder=tmp_path/e/'train';folder.mkdir(parents=True)
        (folder/'checkpoint-final.pt').write_bytes(e.encode())
        (folder/'complete.json').write_text(json.dumps(dict(steps=128,checkpoint_sha256=sha(folder/'checkpoint-final.pt'))))
        (folder/'manifest.json').write_text(json.dumps(dict(arm='B',stage='write',steps=128,effective_batch=4,
            split='pilot',parent_sha256=PARENT_SHA,seed=20260924,teacher_supervision='prompt_matching',teacher_steps=288,
            targets={'a':'target-'+e},teacher_binding_hashes={'a':'binding-'+e})))
        values=[dict(step=i,grad_norm=1.,draws=[dict(pair_id='a',position=0,sigma=.5,mse=.1)]*4) for i in range(1,129)]
        (folder/'optimization.jsonl').write_text('\n'.join(json.dumps(v) for v in values))
    assert verify_training(tmp_path,banks)['paired_draws']
    p=tmp_path/ENDPOINTS[1]/'train/optimization.jsonl';values=[json.loads(s) for s in p.read_text().splitlines()]
    values[0]['draws'][0]['sigma']=.25;p.write_text('\n'.join(json.dumps(v) for v in values))
    with pytest.raises(ValueError,match='draw order'):verify_training(tmp_path,banks)
    values[0]['grad_norm']=0;p.write_text('\n'.join(json.dumps(v) for v in values))
    with pytest.raises(ValueError,match='gradient'):verify_training(tmp_path,banks)
