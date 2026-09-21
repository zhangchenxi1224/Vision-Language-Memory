"""Plan13: paired fixed-budget scope contrast using existing gold-forward logits."""
from __future__ import annotations
import argparse, hashlib, inspect, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from PIL import Image
from scripts.experiments import prefeval_worst_query_consolidation as x
from vision_memory.reader.qwen3vl import _joint_prompt_target_tokenization
from vision_memory.reader.open_answer import score_short_answer
s=x.s;j=x.j;q=x.q;digest=x.digest;tensor_sha=x.tensor_sha
DATA=s.REPORT/'scope-contrast-v1'
INSTANCE=x.INSTANCE
ARMS=('C13','S13')
schedule=x.schedule;recovery_bank=x.recovery_bank;applications=x.applications


def rivals(target, scope):
    gold=target['state'][scope] or 'no active preference'
    values={v or 'no active preference' for k,v in target['state'].items() if k!=scope}
    values.add('no active preference')
    return sorted(v for v in values if not score_short_answer(v,gold)['strict_correct'])


def first_branches(gold_ids, rival_ids):
    branches={}
    for name,ids in rival_ids.items():
        differing=next((i for i,(a,b) in enumerate(zip(gold_ids,ids)) if a!=b),None)
        if differing is None:
            if gold_ids!=ids:raise ValueError('Terminated sequences must diverge before either ends')
            continue
        key=(differing,ids[differing])
        branches.setdefault(key,[]).append(name)
    return [dict(position=d,gold_id=gold_ids[d],rival_id=r,rivals=sorted(names))
            for (d,r),names in sorted(branches.items())]


def contextual_map(processor, item, target, termination):
    prompt=s.query_prompt(processor,item['query'])
    def ids(text):
        _,v=_joint_prompt_target_tokenization(processor,prompt,text+termination['assistant_end_token_text'])
        result=v[0].tolist();assert result[-1]==termination['assistant_end_token_id'];return result
    gold=ids(item['target']);others={r:ids(r) for r in rivals(target,item['scope'])}
    return dict(gold_ids=gold,rival_ids=others,branches=first_branches(gold,others))


def contrast_loss(logits, mapping):
    values=logits[0].float();terms=[];records=[]
    for branch in mapping['branches']:
        d=branch['position'];g=values[d,branch['gold_id']];r=values[d,branch['rival_id']]
        margin=g-r;hinge=torch.relu(2.-margin);terms.append(hinge)
        records.append(dict(**branch,gold_logit=float(g.detach()),rival_logit=float(r.detach()),
                            margin=float(margin.detach()),hinge=float(hinge.detach())))
    loss=torch.stack(terms).mean() if terms else values[0,0]*0.
    return loss,dict(**mapping,comparisons=records,contrast_loss=float(loss.detach()),
                    active=sum(r['hinge']>0 for r in records))


def functions():
    return {f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
            for f in (rivals,first_branches,contextual_map,contrast_loss)}


def prepare():
    parent,p,cases=x.load()
    DATA.mkdir(parents=True,exist_ok=True)
    bank={sid:{scope:rivals(t,scope) for scope in t['state']} for sid,t in p['targets'].items()}
    s.write_frozen(DATA/'rivals.json',bank)
    return parent,p,cases,bank


def load():
    reg=s.load_json(DATA/'registration.json');parent,p,cases=x.load()
    assert digest(parent)==reg['parent_registration_digest']=='cb325203bbe45020bd68b68e4a62ef7ec2037989da7f2dd4cc3d126c7e12774a'
    assert reg['functions']==functions()
    bank={sid:{scope:rivals(t,scope) for scope in t['state']} for sid,t in p['targets'].items()}
    assert bank==s.load_json(DATA/'rivals.json') and digest(bank)==reg['rival_bank_digest']
    return reg,p,cases


def register(a):
    parent,p,cases,bank=prepare();runtime=s.load_json(a.runtime_receipt)
    assert runtime['instance']==INSTANCE and len(runtime['actual']['gpus'])==4
    assert INSTANCE in runtime['platform_status'] and 'RUNNING' in runtime['platform_status']
    endpoints={sid:s.load_json(a.source/'training/M'/sid/'complete.json') for sid in p['targets']}
    for sid,done in endpoints.items():
        for name,h in done['artifacts'].items():assert s.file_sha(a.source/'training/M'/sid/name)==h
    reg=dict(plan='prefeval-rgb-scope-contrast-13',parent_registration_digest=digest(parent),
        parent_verified=s.file_sha(a.source/'final-verified.json'),instance=INSTANCE,runtime=runtime,
        training_digest=digest(p),evaluation_digest=parent['evaluation_digest'],contrast_values=parent['contrast_values'],
        cases_digest=digest(cases),rival_bank_digest=digest(bank),functions=functions(),
        implementation={**parent['implementation'],str(Path(__file__).relative_to(ROOT)).replace('\\','/'):s.file_sha(Path(__file__)),
            'src/vision_memory/reader/open_eos.py':s.file_sha(ROOT/'src/vision_memory/reader/open_eos.py'),
            'src/vision_memory/reader/qwen3vl.py':s.file_sha(ROOT/'src/vision_memory/reader/qwen3vl.py')},
        endpoint_receipts=endpoints,endpoint_receipt_hashes={sid:s.file_sha(a.source/'training/M'/sid/'complete.json') for sid in p['targets']},
        arms=list(ARMS),steps=32,inherited_updates=[256,128,64,64,64,32],optimizer=parent['optimizer'],
        schedule=parent['schedule'],progression=parent['progression'],margin=2.,coefficient=.25,
        objectives={'C13':'weighted mean recovery + .25 slot-weighted application','S13':'C13 + .25 weighted mean first-divergence hinge'},
        budget={**parent['budget'],'gradient_forwards':{'C13':41728,'S13':41728,'total':83456}},writer_updates=0,
        limitation='Observed development endpoints; independent latent teachers, not recurrent Writer transitions. Historical gates unchanged.')
    s.write_frozen(DATA/'registration.json',reg);print(json.dumps(dict(registration_digest=digest(reg),budget=reg['budget'])))


def guard(a,reg):
    assert j.actual_host()==reg['runtime']['actual']
    assert s.file_sha(a.source/'final-verified.json')==reg['parent_verified']
    for name,sha in reg['implementation'].items():assert s.file_sha(ROOT/name)==sha


def train(a):
    reg,p,cases=load();guard(a,reg);processor,reader,vae,versions,bindings=s.models(a,True)
    termination=s.assistant_termination_contract(reader,processor);assignment=sorted(p['targets'])[a.shard::a.shards]
    s.write_frozen(a.output/f'train-identity-{a.shard}.json',dict(**s.identity(a,bindings,reg,assignment),actual_host=j.actual_host(),
        prompt_frame=s.query_prompt(processor,'__REGISTERED_QUERY__'),termination=termination))
    for sid in assignment:
        target=p['targets'][sid];src=a.source/'training/M'/sid
        assert s.file_sha(src/'complete.json')==reg['endpoint_receipt_hashes'][sid]
        z=torch.load(src/'latent.pt',map_location=a.device,weights_only=True)
        assert tensor_sha(z)==reg['endpoint_receipts'][sid]['final_tensor_sha']
        bank=recovery_bank(target);maps={item['id']:contextual_map(processor,item,target,termination) for item,w in bank}
        for arm in (ARMS if sorted(p['targets']).index(sid)%2==0 else ARMS[::-1]):
            out=a.output/'training'/arm/sid;out.mkdir(parents=True,exist_ok=False)
            oracle=s.VAELatentOracle(vae=vae,initial_latent=z,compute_dtype=torch.float32)
            opt=torch.optim.Adam([oracle.latent_fp32],**reg['optimizer']);assert not opt.state
            torch.save(z.detach().cpu(),out/'initial-latent.pt')
            started=time.monotonic();forwards=tokens=0
            for step in range(32):
                opt.zero_grad(set_to_none=True);pixels=oracle.image();before=oracle.latent_fp32.detach().clone()
                total_gradient=torch.zeros_like(oracle.latent_fp32);losses=[];apps=applications(target,step,cases)
                recovery_objective=contrast_objective=application_objective=0.
                for ix,(item,w) in enumerate(bank):
                    c0,t0=processor.total_calls,processor.total_input_tokens;processor.begin_capture()
                    ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=pixels[0],query=item['query'],target=item['target'],device=a.device,
                        termination=termination,lambda_eos=1.,require_image_grad=True,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    proof=processor.end_capture();assert processor.total_calls-c0==1 and len(proof)==1
                    assert ce.target_ids[0].tolist()==maps[item['id']]['gold_ids']
                    branch,record=contrast_loss(ce.target_logits,maps[item['id']])
                    objective=ce.loss+.25*branch if arm=='S13' else ce.loss
                    gradient=torch.autograd.grad(objective,oracle.latent_fp32,retain_graph=bool(apps) or ix+1<len(bank))[0]
                    assert torch.isfinite(gradient).all();total_gradient.add_(gradient,alpha=w)
                    value=float(ce.loss.detach());recovery_objective+=w*value;contrast_objective+=w*record['contrast_loss']
                    used=processor.total_input_tokens-t0;forwards+=1;tokens+=used
                    losses.append(dict(kind='recovery',query=item,mean_weight=w,loss=value,answer_ce=float(ce.answer_loss.detach()),
                        eos_ce=float(ce.eos_loss.detach()),answer_tokens=ce.answer_token_count,reader_forwards=1,processed_input_tokens=used,
                        processor_input=proof[0],gradient_norm=float(gradient.norm()),contrast=record))
                    del ce,branch,objective,gradient
                for ix,(item,w,choices,gold) in enumerate(apps):
                    c0,t0=processor.total_calls,processor.total_input_tokens;processor.begin_capture()
                    ce=s.qwen3vl_listwise_choice_ce(model=reader,processor=processor,image=pixels[0],query=item['query'],choices=choices,target_index=gold,
                        device=a.device,require_image_grad=True,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    proof=processor.end_capture();assert processor.total_calls-c0==4
                    gradient=torch.autograd.grad(ce.loss,oracle.latent_fp32,retain_graph=ix+1<len(apps))[0]
                    assert torch.isfinite(gradient).all();total_gradient.add_(gradient,alpha=w)
                    value=float(ce.loss.detach());application_objective+=w*value;used=processor.total_input_tokens-t0;forwards+=4;tokens+=used
                    losses.append(dict(kind='application',query=item,weight=w,loss=value,choices=choices,gold_index=gold,
                        scores=ce.choice_logits.detach().tolist(),candidate_token_counts=ce.choice_token_counts,
                        reader_forwards=4,processed_input_tokens=used,processor_inputs=proof,gradient_norm=float(gradient.norm())))
                    del ce,gradient
                assert torch.isfinite(total_gradient).all();oracle.latent_fp32.grad=total_gradient;opt.step()
                assert torch.isfinite(oracle.latent_fp32).all()
                s.append(out/'optimization.jsonl',dict(step=step,schedule=schedule(step),losses=losses,
                    recovery_objective=recovery_objective,contrast_objective=contrast_objective,application_objective=application_objective,
                    weighted_objective=recovery_objective+application_objective+(.25*contrast_objective if arm=='S13' else 0.),
                    gradient=dict(norm=float(total_gradient.norm()),displacement_norm=float((oracle.latent_fp32.detach()-before).norm()))))
                if (step+1)%16==0:torch.save(dict(next_step=step+1,latent=oracle.latent_fp32.detach().cpu(),optimizer=opt.state_dict(),registration_digest=digest(reg)),out/'checkpoint.pt')
            torch.save(oracle.latent_fp32.detach().cpu(),out/'latent.pt');torch.save(opt.state_dict(),out/'optimizer.pt')
            with torch.no_grad():array=oracle.image()[0].detach().cpu().mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy()
            Image.fromarray(array).save(out/'memory.png');s.frozen(versions)
            s.write_frozen(out/'complete.json',dict(target=sid,arm=arm,inherited_updates=reg['inherited_updates'],additional_updates=32,
                initial_tensor_sha=tensor_sha(z),final_tensor_sha=tensor_sha(oracle.latent_fp32),reader_forwards=forwards,processed_input_tokens=tokens,
                seconds=time.monotonic()-started,artifacts={x:s.file_sha(out/x) for x in ('initial-latent.pt','latent.pt','optimizer.pt','checkpoint.pt','optimization.jsonl','memory.png')}))
            del opt,oracle
    s.write_frozen(a.output/f'train-complete-{a.shard}.json',dict(targets=assignment))


def measure_ce(a,reader,processor,image,item,termination,target):
    processor.begin_capture()
    with torch.no_grad():
        ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=image,query=item['query'],target=item['target'],device=a.device,
            termination=termination,lambda_eos=1.,require_image_grad=False,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
    proof=processor.end_capture();mapping=contextual_map(processor,item,target,termination)
    assert ce.target_ids[0].tolist()==mapping['gold_ids']
    _,branch_record=contrast_loss(ce.target_logits,mapping)
    vals=ce.target_logits.float();gold=vals.gather(-1,ce.target_ids.unsqueeze(-1)).squeeze(-1)
    nll=torch.logsumexp(vals,-1)-gold;other=vals.clone();other.scatter_(-1,ce.target_ids.unsqueeze(-1),float('-inf'))
    return dict(contrast=branch_record,answer_ce=float(ce.answer_loss),eos_ce=float(ce.eos_loss),loss=float(ce.loss),answer_tokens=ce.answer_token_count,
        target_ids=ce.target_ids[0].tolist(),token_nll=nll[0].tolist(),gold_token_margins=(gold-other.amax(-1))[0].tolist(),
        teacher_forced_correct=(vals.argmax(-1)==ce.target_ids)[0].tolist(),processor_input=proof[0])


def evaluate(a):
    reg,p,new_cases=load();guard(a,reg);e=s.load_json(s.DATA/'evaluation-payload.json');old_cases={}
    for fn in ('training-scenarios.json','reserved-scenarios.json'):
        for cc in s.load_json(s.DATA/fn).values():
            for c in cc:old_cases[(c['id'],c['scope'],c['value_id'])]=c
    endpoints={f'{arm}/{sid}':s.file_sha(a.output/'training'/arm/sid/'complete.json') for sid in p['targets'] for arm in ARMS}
    processor,reader,_,versions,bindings=s.models(a);termination=s.assistant_termination_contract(reader,processor)
    assignment=sorted(p['targets'])[a.shard::a.shards];out=a.output/'evaluation'/f'shard-{a.shard}';out.mkdir(parents=True,exist_ok=False)
    frame=s.query_prompt(processor,'__REGISTERED_QUERY__');s.write_frozen(out/'identity.json',dict(**s.identity(a,bindings,reg,assignment),actual_host=j.actual_host(),frozen_endpoints=endpoints,prompt_frame=frame,termination=termination))
    started=time.monotonic();n=rn=cf=cen=0
    for sid in assignment:
        t=e['targets'][sid];train=p['targets'][sid]
        for arm in ARMS:
            path=a.output/'training'/arm/sid/'memory.png';sha=s.file_sha(path);image=s.read_png(path)
            for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
                for item in t[panel]:
                    processor.begin_capture();g=s.generate(reader,processor,image,item['query'],a.device);proof=processor.end_capture()
                    s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel=panel,query=item,png_sha=sha,generation=g,score=s.score_generation(g,item,panel=='mcq'),reader_query=item['query'],processor_input=proof[0]));n+=1
                    if panel=='recovery_training' or (panel=='qualification' and item['kind']=='recovery'):
                        measured=measure_ce(a,reader,processor,image,item,termination,train)
                        s.append(out/'recovery-ce.jsonl',dict(target=sid,condition=arm,panel=panel,query=item,png_sha=sha,**measured));cen+=1
            for scope,value in sorted(train['state'].items()):
                for bank in ('question','instruction'):
                    for ix in range(4):
                        processor.begin_capture();item=q.recovery_form(scope,value,bank,ix);g=s.generate(reader,processor,image,item['query'],a.device);proof=processor.end_capture()
                        s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='recovery_coverage',query=item,png_sha=sha,generation=g,score=s.score_generation(g,item),reader_query=item['query'],processor_input=proof[0]));n+=1
                        measured=measure_ce(a,reader,processor,image,item,termination,train)
                        s.append(out/'recovery-ce.jsonl',dict(target=sid,condition=arm,panel='recovery_coverage',query=item,png_sha=sha,**measured));cen+=1
            for item0 in t['application_training']:
                processor.begin_capture();case=old_cases[(item0['case_id'],item0['scope'],item0['value_id'])];item,_,gold=q.xml_candidates(case,0);g=s.generate(reader,processor,image,item['query'],a.device);proof=processor.end_capture()
                s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='application_xml',query=item,png_sha=sha,generation=g,score=s.score_generation(g,{**item,'target_index':gold},True),reader_query=item['query'],processor_input=proof[0]));n+=1
            for panel in ('application_training','application_reserved'):
                for item0 in t[panel]:
                    case=old_cases[(item0['case_id'],item0['scope'],item0['value_id'])]
                    for rot in (range(4) if panel=='application_training' and item0['value_id'] in reg['contrast_values'] else (0,)):
                        item,ch,gold=s.ranking_candidates(case,rot);score=s.ranking_read(reader,processor,image,item['query'],ch,a.device);cf+=4
                        margin=score['scores'][gold]-max(x for k,x in enumerate(score['scores']) if k!=gold)
                        s.append(out/'ranking.jsonl',dict(target=sid,condition=arm,panel='application_training_rotations' if rot else panel,query=item,png_sha=sha,choices=ch,gold_index=gold,score=score,gold_margin=margin,unique_correct=margin>0));rn+=1
            for vid in sorted({item['value_id'] for item in t['application_training']}):
                for case in new_cases[vid]:
                    processor.begin_capture();item,_,gold=q.xml_candidates(case,0);g=s.generate(reader,processor,image,item['query'],a.device);proof=processor.end_capture()
                    s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='attribute_xml',query=item,png_sha=sha,generation=g,score=s.score_generation(g,{**item,'target_index':gold},True),reader_query=item['query'],processor_input=proof[0]));n+=1
                    item,ch,gold=s.ranking_candidates(case,0);score=s.ranking_read(reader,processor,image,item['query'],ch,a.device);cf+=4
                    margin=score['scores'][gold]-max(x for k,x in enumerate(score['scores']) if k!=gold)
                    s.append(out/'ranking.jsonl',dict(target=sid,condition=arm,panel='attribute_full_action',query=item,png_sha=sha,choices=ch,gold_index=gold,score=score,gold_margin=margin,unique_correct=margin>0));rn+=1
    s.frozen(versions);s.write_frozen(out/'complete.json',dict(generations=n,rankings=rn,ranking_candidate_forwards=cf,recovery_ce_forwards=cen,processed_input_tokens=processor.total_input_tokens,
        seconds=time.monotonic()-started,artifacts={x:s.file_sha(out/x) for x in ('reads.jsonl','ranking.jsonl','recovery-ce.jsonl','identity.json')}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['prepare','register','train','evaluate'],required=True)
    for n in ('source','output','reader','base','runtime-receipt'):ap.add_argument('--'+n,type=Path)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if a.mode=='prepare':
        _,p,_,bank=prepare()
        print(json.dumps(dict(states=len(p['targets']),addressed_slots=sum(len(t['state']) for t in p['targets'].values()),
            rivals=sum(len(r) for t in bank.values() for r in t.values()),rival_bank_digest=digest(bank),gpu_started=False)))
    elif a.mode=='register':register(a)
    else:a.output.mkdir(parents=True,exist_ok=True);{'train':train,'evaluate':evaluate}[a.mode](a)
