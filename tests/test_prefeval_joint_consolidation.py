import json,inspect
from collections import Counter
from types import SimpleNamespace
import torch
import numpy as np
from PIL import Image
from scripts.experiments import prefeval_joint_consolidation as j
s=j.s

def test_population_schedule_weights_and_exact_budget():
    _,p=s.trial_load();counts=Counter();rec_counts=Counter()
    for t in p['targets'].values():
        seen={scope:Counter() for scope,value in t['state'].items() if value is not None}
        for step in range(64):
            for arm in ('C','J'):
                rec,app=j.jobs(t,step,arm)
                assert len(rec)==3*len(t['state']) and abs(sum(w for _,w in rec)-1)<1e-12
                for scope in t['state']:
                    assert sorted(q['form'] for q,w in rec if q['scope']==scope)==[0,1,2]
                    base=.5 if len(t['state'])>1 and scope==t['changed_scope'] else .5/(len(t['state'])-1) if len(t['state'])>1 else 1
                    assert all(w==base/3 for q,w in rec if q['scope']==scope)
                    assert len([q for q,w,ch,gi in app if q['scope']==scope])==int(arm=='J' and t['state'][scope] is not None)
                counts[arm]+=len(rec)+4*len(app);rec_counts[arm]+=len(rec)
                for q,w,ch,gi in app:
                    assert ch[gi]==q['target'];seen[q['scope']][q['scenario'],q['rotation']]+=1
                    assert w==next(rw for rq,rw in rec if rq['scope']==q['scope'])*3*.25
        for c in seen.values():assert len(c)==16 and set(c.values())=={4}
    assert counts=={'C':16896,'J':38400} and rec_counts=={'C':16896,'J':16896}
    assert 40*2*64==5120 and sum(counts.values())==55296

def test_evaluation_excludes_training_until_frozen_and_budget():
    _,p=s.trial_load();e=s.load_json(s.DATA/'evaluation-payload.json');reg=s.load_json(s.TRIAL_DATA/'registration.json')
    perarm=sum(sum(len(t[pan]) for pan in ('recovery_training','qualification','application_training','mcq','application_reserved')) for t in e['targets'].values())
    ranks=sum(len(t['application_training'])+len(t['application_reserved'])+3*sum(q['value_id'] in reg['contrast_values'] for q in t['application_training']) for t in e['targets'].values())
    assert perarm*2==2568 and ranks*2==1584 and ranks*8==6336
    assert sum(len(t['recovery']) for t in p['targets'].values())*2==528
    code=inspect.getsource(j.train);assert 'evaluation-payload' not in code and 'reserved-scenarios' not in code
    evaluation=inspect.getsource(j.evaluate)
    assert evaluation.index("done['additional_updates']==64")<evaluation.index("evaluation-payload.json")

def test_real_training_loop_accumulates_same_image_before_single_adam_step(tmp_path,monkeypatch):
    _,p=s.trial_load();sid=sorted(p['targets'])[0];t=p['targets'][sid];p=dict(targets={sid:t})
    source=tmp_path/'old';folder=source/'training/B'/sid;folder.mkdir(parents=True)
    z=torch.full((1,3,2,2),.4);torch.save(z,folder/'latent.pt')
    receipt=dict(final_tensor_sha=j.tensor_sha(z),artifacts={'latent.pt':s.file_sha(folder/'latent.pt')})
    (folder/'complete.json').write_text(json.dumps(receipt))
    reg=dict(endpoint_receipts={sid:receipt},endpoint_receipt_hashes={sid:s.file_sha(folder/'complete.json')},optimizer=dict(lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0.))
    monkeypatch.setattr(j,'load',lambda:(reg,p));monkeypatch.setattr(j,'guard',lambda *a:None);monkeypatch.setattr(j,'actual_host',lambda:{})
    processor=SimpleNamespace(total_calls=0,total_input_tokens=0);events=[];optimizers=[];original_adam=torch.optim.Adam
    class SpyAdam(original_adam):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);self.steps=0;self.first_grad=None;optimizers.append(self)
        def step(self,*a,**kw):
            if self.steps==0:self.first_grad=self.param_groups[0]['params'][0].grad.detach().clone()
            events.append(('step',len(optimizers)-1,self.steps,None));self.steps+=1;return super().step(*a,**kw)
    class Oracle:
        def __init__(self,initial_latent,**kw):self.latent_fp32=torch.nn.Parameter(initial_latent.clone())
        def image(self):return self.latent_fp32.sigmoid()
    def observe(image,calls):
        processor.total_calls+=calls;processor.total_input_tokens+=10*calls
        events.append(('loss',len(optimizers)-1,optimizers[-1].steps,image.detach().clone()))
    def rec(image,**kw):
        observe(image,1);loss=(image-.8).square().mean();return SimpleNamespace(loss=loss,answer_loss=loss,eos_loss=loss*0)
    def app(image,**kw):
        observe(image,4);loss=(image-.2).square().mean();return SimpleNamespace(loss=loss,choice_logits=torch.zeros(4),choice_token_counts=(2,2,2,2))
    monkeypatch.setattr(torch.optim,'Adam',SpyAdam);monkeypatch.setattr(s,'VAELatentOracle',Oracle)
    monkeypatch.setattr(s,'models',lambda *a:(processor,None,None,[],{}));monkeypatch.setattr(s,'assistant_termination_contract',lambda *a:{})
    monkeypatch.setattr(s,'qwen3vl_answer_eos_ce',rec);monkeypatch.setattr(s,'qwen3vl_listwise_choice_ce',app)
    monkeypatch.setattr(torch.cuda,'get_rng_state',lambda:torch.get_rng_state())
    monkeypatch.setattr(s,'read_png',lambda path:torch.from_numpy(np.array(Image.open(path))).permute(2,0,1).float()/255)
    out=tmp_path/'new';out.mkdir();j.train(SimpleNamespace(source=source,output=out,shard=0,shards=1,device='cpu',mode='train'))
    assert len(optimizers)==2 and all(o.steps==64 for o in optimizers)
    for index,arm in enumerate(('C','J')):
        variable=z.clone().requires_grad_();image=variable.sigmoid();recjobs,appjobs=j.jobs(t,0,arm)
        explicit=sum(w*(image-.8).square().mean() for _,w in recjobs)+sum(w*(image-.2).square().mean() for _,w,_,_ in appjobs)
        expected=torch.autograd.grad(explicit,variable)[0];assert torch.allclose(optimizers[index].first_grad,expected,atol=1e-7,rtol=1e-6)
        for step in range(64):
            group=[e for e in events if e[1:3]==(index,step)]
            assert group[-1][0]=='step' and sum(e[0]=='step' for e in group)==1
            assert len(group)==len(j.jobs(t,step,arm)[0])+len(j.jobs(t,step,arm)[1])+1
            assert all(torch.equal(e[3],group[0][3]) for e in group[:-1])
        saved=torch.load(out/'training'/arm/sid/'initial-latent.pt',weights_only=True);assert torch.equal(saved,z)
