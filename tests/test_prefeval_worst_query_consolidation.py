from collections import Counter
import torch
import pytest
from scripts.experiments import prefeval_worst_query_consolidation as x
from scripts.reporting.verify_prefeval_compositional_evidence import expected


@pytest.mark.parametrize('arm',['M','W'])
def test_streamed_gradient_and_single_adam_step_match_explicit_objective(arm):
    z=torch.tensor([.2,-.4,.7],dtype=torch.float64,requires_grad=True)
    reference=z.detach().clone().requires_grad_(True)
    weights=[.15,.25,.6]
    def losses(v):return [(v[0]+2*v[1])**2+.3, (v[2]-v[1])**2+.2, (v[0]-v[2])**2+.4]
    acc=x.RecoveryAccumulator(z)
    vals=losses(z)
    for i in (2,0,1):
        acc.add(('scope',str(i)),float(vals[i].detach()),weights[i],torch.autograd.grad(vals[i],z,retain_graph=True)[0])
    actual,value=acc.finish(arm)
    application=.25*(z.sum()+.1)**2
    actual=actual+torch.autograd.grad(application,z)[0]
    ref=losses(reference);mean=sum(w*v for w,v in zip(weights,ref))
    ix=max(range(3),key=lambda i:(float(ref[i].detach()),-i))
    objective=mean if arm=='M' else .5*mean+.5*ref[ix]
    assert value==pytest.approx(float(objective.detach()))
    objective=objective+.25*(reference.sum()+.1)**2
    assert torch.allclose(actual,torch.autograd.grad(objective,reference)[0],atol=1e-12,rtol=1e-12)
    opt=torch.optim.Adam([z],lr=.01);other=torch.optim.Adam([reference],lr=.01)
    z.grad=actual;reference.grad=actual.clone();opt.step();other.step()
    assert torch.equal(z,reference) and opt.state[z]['step']==other.state[reference]['step']==1


def test_ties_are_lexicographic_and_zero_gradients_are_valid():
    z=torch.zeros(2);acc=x.RecoveryAccumulator(z)
    acc.add(('b','2'),1.,.5,torch.tensor([3.,4.]))
    acc.add(('a','1'),1.,.5,torch.tensor([1.,2.]))
    assert acc.worst_identity==('a','1')
    assert torch.equal(acc.finish('W')[0],torch.tensor([1.5,2.5]))
    zero=x.RecoveryAccumulator(z);zero.add(('a','1'),0.,1.,torch.zeros(2))
    assert torch.equal(zero.finish('W')[0],torch.zeros(2))


def test_complete_recovery_bank_excludes_qualification_and_budgets_match():
    reg,p,cases=x.e.load();ev=x.s.load_json(x.s.DATA/'evaluation-payload.json')
    calls=Counter();draws=[x.schedule(t) for t in range(32)]
    assert len({(d['case'],d['bank'],d['rotation']) for d in draws})==32
    for case in range(4):
        for bank in range(2):assert Counter(d['format'] for d in draws if d['case']==case and d['bank']==bank)=={'xml':2,'full-action':2}
    for sid,target in p['targets'].items():
        bank=x.recovery_bank(target)
        assert len(bank)==11*len(target['state']) and len({q['query'] for q,w in bank})==len(bank)
        excluded={q['query'] for q in ev['targets'][sid]['qualification']}
        assert not excluded.intersection(q['query'] for q,w in bank)
        for scope in target['state']:
            assert sum(q['scope']==scope for q,w in bank)==11
        for step in range(32):
            apps=x.applications(target,step,cases)
            assert len(apps)==sum(v is not None for v in target['state'].values())
            calls['recovery']+=len(bank);calls['application']+=4*len(apps)
    assert calls=={'recovery':30976,'application':10752}
    g,r=expected(reg,p,cases,('M','W'),('M','W'))
    assert len(g)==5320 and len(r)==2256
    ce=[v for k,v in g.items() if k[2] in ('recovery_training','recovery_coverage') or (k[2]=='qualification' and v['kind']=='recovery')]
    assert len(ce)==2288
