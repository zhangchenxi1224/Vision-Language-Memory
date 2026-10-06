from collections import Counter
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire import run_context_layout_aug as m


def source_payload():
    return dict(schema_version=1,optimizer_step=256,episode_cursor=1024,epoch=0,
        manifest=dict(steps=256,implementation_sha256=m.sha(ROOT/'scripts/experiments/prefeval_k1_writer.py'),targets={'p':'frozen'}),
        trainable_state={'p':torch.tensor([1.,2.])},optimizer={'state':{0:{'step':torch.tensor(256.),'exp_avg':torch.tensor([.3])}}},
        rng_state=dict(python=(1,2),numpy=[1],torch_cpu=torch.tensor([2],dtype=torch.uint8),torch_cuda=[torch.tensor([3],dtype=torch.uint8)]))


def test_resume_migration_changes_only_manifest_and_does_not_mutate_source():
    original=source_payload();before=copy.deepcopy(original)
    for arm in ('native',*m.ARMS):
        new=m.migrated_payload(original,arm)
        for key in original:
            if key!='manifest':assert m.e.equivalent(original[key],new[key])
        assert new['manifest']['steps']==(258 if arm=='native' else 384)
    assert m.e.equivalent(before,original)


@pytest.mark.parametrize('key,value',[('optimizer_step',255),('episode_cursor',1023),('rng_state',{}),('optimizer',{'state':{0:{'step':255}}})])
def test_incomplete_state_refused(key,value):
    source=source_payload();source[key]=value
    with pytest.raises(ValueError):m.migrated_payload(source,'canonical')


def test_per_history_balanced_draws_and_heldout_xml_exclusion():
    rows=m.p.population()[0]
    from vision_memory.training.latent_bank_unet import stable_seed
    for arm in m.ARMS:
        counts=Counter()
        for draw in range(1024,1536):
            cycle,offset=divmod(draw,32)
            order=torch.randperm(32,generator=torch.Generator().manual_seed(stable_seed(20260924,'order',cycle))).tolist()
            counts[rows[order[offset]]['base_pair_id'],m.training_layout(cycle,arm)]+=1
        assert len(counts)==(32 if arm=='canonical' else 64)
        assert set(counts.values())==({16} if arm=='canonical' else {8})
    assert m.training_layout(32,'canonical')==m.training_layout(32,'augmented')==0
    with pytest.raises(ValueError):m.training_layout(32,'xml')


def test_three_formats_preserve_original_contents_and_roles():
    for row in m.p.population()[0]:
        ex=row['history'][:2];copy_before=copy.deepcopy(ex)
        for style in ('canonical','markdown','xml'):
            text=m.evaluation_text(ex,style)
            assert all(msg['content'] in text for msg in ex)
            assert ex==copy_before
        assert m.evaluation_text(ex,'xml')=='<user>\n'+ex[0]['content']+'\n</user>\n<assistant>\n'+ex[1]['content']+'\n</assistant>'


@pytest.mark.parametrize('corrupt',[False,True])
def test_exact_parity_gate_rejects_optimizer_drift(tmp_path,corrupt):
    state=source_payload();state['optimizer_step']=258;state['episode_cursor']=1032
    trace=[dict(step=257+i,grad_norm=1.,draws=[dict(pair_id='p',position=0,sigma=.2,mse=.3)]) for i in range(2)]
    for arm in ('native',*m.ARMS):
        folder=tmp_path/arm;folder.mkdir()
        payload=copy.deepcopy(state)
        if corrupt and arm=='augmented':payload['optimizer']['state'][0]['exp_avg']+=.1
        torch.save(payload,folder/('resume.pt' if arm=='native' else 'resume-step258.pt'))
        rows=copy.deepcopy(trace)
        if arm!='native':
            for row in rows:row['draws'][0]['layout']=0
        (folder/'optimization.jsonl').write_text('\n'.join(json.dumps(x) for x in ([{}]*256+rows)),encoding='utf8')
    if corrupt:
        with pytest.raises(ValueError,match='parity'):m.verify_parity(tmp_path)
    else:assert m.verify_parity(tmp_path)['arms']['augmented']['exact_parameters_optimizer_rng']


def test_attempt_budget_and_job_cursors(tmp_path,monkeypatch):
    monkeypatch.setattr(m.l,'remaining_seconds',lambda _:(0,27000,500))
    a=tmp_path/'attempts';a.mkdir();(a/'failed.json').write_text(json.dumps(dict(started=1,finished=101,exit_code=124)))
    assert m.budget(tmp_path)==(2600,27500,100)
    (a/'active.json').write_text(json.dumps(dict(started=1)))
    with pytest.raises(ValueError,match='Unsettled'):m.budget(tmp_path)
    args=SimpleNamespace(output=tmp_path,base=Path('/base'),reader=Path('/reader'),official_source=Path('/official'))
    jobs=m.jobs(args,'full')
    assert [j['gpu'] for j in jobs]==[0,1]
    assert all(j['command'][-2:]==['--stop-step','384'] for j in jobs)
    assert m.jobs(args,'probe')[0]['command'][-2:]==['--steps','258']


def test_parent_audit_uses_original_checkout_and_stable_source_paths(monkeypatch):
    calls=[]
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:'9cf6ea72da0e4b17b3e0bf27232d55ae74db6c17\n')
    monkeypatch.setattr(m.subprocess,'run',lambda *args,**kwargs:calls.append((args,kwargs)))
    def checksum(path):
        if path==m.SOURCE/'resume.pt':return m.RESUME_SHA
        if path==m.SOURCE/'checkpoint-final.pt':return m.l.CHECKPOINT_SHA
        return 'stable-content'
    monkeypatch.setattr(m,'sha',checksum)
    def read(path):
        if path==m.PREVIOUS/'status.json':return {'status':'completed'}
        if path==m.PREVIOUS/'audit-0650.json':return {'exact_report_recomputation':True}
        if path==m.OUTPUT/'plan.json':return dict(plan_sha256='stable-content',ids_sha256='stable-content',implementation_sha256='stable-content')
        raise AssertionError(path)
    monkeypatch.setattr(m,'read',read)
    files=m.verify_source()
    assert len(calls)==1 and calls[0][1]['cwd']==m.LAYOUT_REPO
    assert str(m.OUTPUT/'plan.json') in files
    assert str(m.PLAN) not in files and str(m.p.ALL_IDS) not in files
