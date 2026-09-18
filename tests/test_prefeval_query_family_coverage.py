import inspect
from collections import Counter
import torch
from scripts.experiments import prefeval_query_family_coverage as q
s=q.s

def test_exact_schedule_weights_and_compute():
    _,p=q.j.load();counts=Counter();coverage=Counter();formats=Counter()
    for sid,t in p['targets'].items():
        for step in range(64):
            for arm in ('U','V'):
                rec,app=q.jobs(t,step,arm);assert len(rec)==3*len(t['state'])
                assert abs(sum(w for _,w in rec)-1)<1e-12
                assert len(app)==sum(v is not None for v in t['state'].values())
                counts[arm]+=len(rec)+4*len(app)
                for item,w in rec:
                    if arm=='V' and item.get('bank'):coverage[sid,item['scope'],item['bank'],item['form']]+=1
                for item,w,ch,gold in app:
                    assert ch[gold]==item['target'];formats[arm,item['kind']]+=1
        for scope in t['state']:
            for bank in ('question','instruction'):
                assert {coverage[sid,scope,bank,i] for i in range(4)}=={16}
    assert counts=={'U':38400,'V':38400}
    assert 40*2*64==5120 and sum(counts.values())==76800
    assert formats['V','application']==formats['V','derived_xml_application']

def test_xml_mapping_and_query_isolation():
    _,p=q.j.load();case=next(iter(next(iter(p['targets'].values()))['applications']))
    seen=[]
    for rot in range(4):
        item,choices,gold=q.xml_candidates(case,rot);seen.append(gold)
        assert choices[gold]==item['target'] and item['target']==f'<choice>{chr(65+gold)}</choice>'
        assert case['target'] not in item['query'].split('here are 4 options')[0]
        assert '<choice>' not in item['query'].split('Answer example:')[0]
    assert sorted(seen)==[0,1,2,3]
    code=inspect.getsource(q.train)
    for forbidden in ('evaluation-payload','reserved-scenarios','qualification'):
        assert forbidden not in code

def test_joint_objective_uses_one_image_and_one_step(monkeypatch):
    _,p=q.j.load();t=next(iter(p['targets'].values()));z=torch.full((1,3,2,2),.4,requires_grad=True)
    for arm in ('U','V'):
        image=z.sigmoid();rec,app=q.jobs(t,0,arm)
        loss=sum(w*(image-.8).square().mean() for _,w in rec)+sum(w*(image-.2).square().mean() for _,w,_,_ in app)
        grad=torch.autograd.grad(loss,z,retain_graph=True)[0]
        explicit=torch.zeros_like(z)
        for _,w in rec:explicit+=torch.autograd.grad(w*(image-.8).square().mean(),z,retain_graph=True)[0]
        for _,w,_,_ in app:explicit+=torch.autograd.grad(w*(image-.2).square().mean(),z,retain_graph=True)[0]
        assert torch.allclose(grad,explicit,atol=1e-7,rtol=1e-6)
