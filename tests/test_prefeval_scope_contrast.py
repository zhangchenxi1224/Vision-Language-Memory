from collections import Counter
from types import SimpleNamespace
import torch
import pytest
from scripts.experiments import prefeval_scope_contrast as x
from scripts.reporting.verify_prefeval_scope_contrast import verify_contrast
from scripts.reporting.verify_prefeval_compositional_evidence import expected


@pytest.mark.parametrize('gold,rivals,expected',[
    ([1,2,9],{'a':[3,2,9]},[(0,3)]),
    ([1,2,3,4,9],{'a':[1,2,3,5,9],'b':[1,2,3,5,6,9]},[(3,5)]),
    ([1,2,9],{'long':[1,2,3,9]},[(2,3)]),
    ([1,2,3,9],{'short':[1,2,9]},[(2,9)]),
    ([1,9],{},[]),
    ([1,9],{'equal':[1,9]},[]),
])
def test_first_divergence_deduplication_and_eos(gold,rivals,expected):
    actual=x.first_branches(gold,rivals)
    assert [(v['position'],v['rival_id']) for v in actual]==expected
    for row in actual:
        for name in row['rivals']:assert rivals[name][:row['position']]==gold[:row['position']]


def test_contextual_tokenization_quoted_values_and_prompt_isolation(monkeypatch):
    seen=[]
    class Tokenizer:
        def __call__(self,text,**kwargs):
            seen.append(text)
            # Continuations have different IDs in context versus isolated strings.
            split=text.find('|')+1
            ids=[ord(c) for c in text[:split]]+[ord(c)+256 for c in text[split:]]
            return {'input_ids':torch.tensor([ids])}
    processor=SimpleNamespace(tokenizer=Tokenizer())
    monkeypatch.setattr(x.s,'query_prompt',lambda processor,query:'chat:'+query+'|')
    item={'query':'Which scope?','scope':'a','target':'"same prefix one"'}
    target={'state':{'a':item['target'],'b':'"same prefix two"'}}
    mapping=x.contextual_map(processor,item,target,{'assistant_end_token_text':'~','assistant_end_token_id':382})
    assert mapping['gold_ids']==[ord(c)+256 for c in item['target']+'~']
    assert all(t=='chat:Which scope?|' or t.startswith('chat:Which scope?|') for t in seen)
    assert len([r for r in mapping['branches'] if r['position']>0])==1


@pytest.mark.parametrize('arm',x.ARMS)
def test_streamed_combined_gradient_and_single_fresh_adam_step(arm):
    z=torch.tensor([.2,-.3],requires_grad=True);ref=z.detach().clone().requires_grad_(True)
    mapping={'gold_ids':[0,1],'rival_ids':{'other':[1,1]},'branches':x.first_branches([0,1],{'other':[1,1]})}
    def objectives(v):
        results=[]
        for scale in (1.,2.,3.):
            logits=torch.stack([v*scale,v]).unsqueeze(0)
            ce=(v.sum()+scale)**2
            contrast,_=x.contrast_loss(logits,mapping)
            results.append(ce+.25*contrast if arm=='S13' else ce)
        return results
    acc=torch.zeros_like(z)
    for weight,obj in zip((.2,.3,.5),objectives(z)):
        acc+=weight*torch.autograd.grad(obj,z)[0]
    explicit=sum(w*o for w,o in zip((.2,.3,.5),objectives(ref)))
    expected_grad=torch.autograd.grad(explicit,ref)[0]
    assert torch.allclose(acc,expected_grad,atol=1e-6)
    a=torch.optim.Adam([z],lr=.01);b=torch.optim.Adam([ref],lr=.01)
    assert not a.state and not b.state
    z.grad=acc;ref.grad=expected_grad;a.step();b.step()
    assert torch.allclose(z,ref,atol=1e-7) and a.state[z]['step']==1


def test_zero_hinge_empty_rivals_and_verifier_detects_bad_margin():
    vals=torch.tensor([[[5.,0.],[0.,0.]]],requires_grad=True)
    mapping=dict(gold_ids=[0,1],rival_ids={'no active preference':[1,1]},branches=x.first_branches([0,1],{'no active preference':[1,1]}))
    loss,record=x.contrast_loss(vals,mapping)
    assert loss==0 and torch.equal(torch.autograd.grad(loss,vals)[0],torch.zeros_like(vals))
    target={'state':{'a':'pref'}};query={'scope':'a','target':'pref'}
    verify_contrast(record,target,query,{'assistant_end_token_id':1})
    record['comparisons'][0]['margin']=4.
    with pytest.raises(AssertionError):verify_contrast(record,target,query,{'assistant_end_token_id':1})
    loss,_=x.contrast_loss(vals,dict(gold_ids=[0,1],rival_ids={},branches=[]))
    assert loss==0 and torch.equal(torch.autograd.grad(loss,vals)[0],torch.zeros_like(vals))


def test_rivals_and_exact_coverage_budget_exclude_qualification():
    assert x.rivals({'state':{'a':'same','b':'SAME','c':'different','d':'different'}},'a')==['different','no active preference']
    assert x.rivals({'state':{'a':None}},'a')==[]
    reg,p,cases=x.x.load();ev=x.s.load_json(x.s.DATA/'evaluation-payload.json');counts=Counter()
    for sid,t in p['targets'].items():
        bank=x.recovery_bank(t);assert bank==x.x.recovery_bank(t)
        assert not {q['query'] for q,w in bank}.intersection(q['query'] for q in ev['targets'][sid]['qualification'])
        for q,w in bank:
            valid={v for v in t['state'].values() if v is not None}|{'no active preference'}
            assert set(x.rivals(t,q['scope']))<=valid
        for step in range(32):
            apps=x.applications(t,step,cases);assert apps==x.x.applications(t,step,cases)
            counts['recovery']+=len(bank);counts['application']+=4*len(apps)
    assert counts=={'recovery':30976,'application':10752}
    g,r=expected(reg,p,cases,x.ARMS,x.ARMS)
    assert len(g)==5320 and len(r)==2256
