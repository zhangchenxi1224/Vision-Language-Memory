"""Plan11: one compositional-evidence treatment with archived R2 control."""
from __future__ import annotations
import argparse,hashlib,inspect,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from PIL import Image
from scripts.experiments import prefeval_query_family_coverage as q
from vision_memory.prefeval.rgb_protocol import digest

s=q.s;j=q.j;tensor_sha=q.tensor_sha
DATA=s.REPORT/'compositional-evidence-v1'
INSTANCE='dl-clear-retain-h200x4-20260914'

def schedule(step):
    return dict(step=step,case=step%4,bank=(step//4)%2,
        format='xml' if (step//8)%2 else 'full-action',rotation=(step//16)%4)

def application(case,fmt,rotation):
    return q.xml_candidates(case,rotation) if fmt=='xml' else s.ranking_candidates(case,rotation)

def jobs(t,step,arm,new_cases):
    spec=schedule(step);weights={x['scope']:w for x,w in s.training_jobs(t,dict(recovery_form=0,mixture_application=False),'A')}
    recovery=[]
    for scope,value in sorted(t['state'].items()):
        qs=[next(x for x in t['recovery'] if x['scope']==scope and x['form']==step%3),
            q.recovery_form(scope,value,'question',step%4),q.recovery_form(scope,value,'instruction',(step//4)%4)]
        recovery.extend((x,weights[scope]/3) for x in qs)
    apps=[]
    for scope,value in sorted(t['state'].items()):
        if value is None:continue
        old=next(x for x in t['applications'] if x['scope']==scope and x['scenario']==spec['case'])
        case=new_cases[old['value_id']][spec['case']] if arm=='E' and spec['bank'] else old
        item,choices,gold=application(case,spec['format'],spec['rotation'])
        apps.append((item,.25*weights[scope],choices,gold))
    return recovery,apps

def functions():
    return {x.__name__:hashlib.sha256(inspect.getsource(x).encode()).hexdigest() for x in (schedule,application,jobs)}

def load():
    reg=s.load_json(DATA/'registration.json');parent,p=q.load()
    assert digest(parent)==reg['parent_registration_digest']=='64406da59ad3f1a885720969b80943e7f8b2a24d68e6ca236b7a6f3c66d1d0af'
    cases=s.load_json(DATA/'scenarios.json');assert digest(cases)==reg['attribute_scenarios_digest']
    assert functions()==reg['functions'];return reg,p,cases

def register(a):
    parent,p=q.load();assert digest(parent)=='64406da59ad3f1a885720969b80943e7f8b2a24d68e6ca236b7a6f3c66d1d0af'
    runtime=s.load_json(a.runtime_receipt);assert runtime['instance']==INSTANCE and len(runtime['actual']['gpus'])==4
    assert INSTANCE in runtime['platform_status'] and 'RUNNING' in runtime['platform_status']
    cases=s.load_json(DATA/'scenarios.json');assert sum(map(len,cases.values()))==112
    endpoints={sid:s.load_json(a.source/'training/V'/sid/'complete.json') for sid in p['targets']}
    for sid,done in endpoints.items():
        for name,h in done['artifacts'].items():assert s.file_sha(a.source/'training/V'/sid/name)==h
    reg=dict(plan='prefeval-rgb-compositional-evidence-11',parent_registration_digest=digest(parent),
        parent_verified=s.file_sha(a.source/'final-verified.json'),instance=INSTANCE,runtime=runtime,
        training_digest=digest(p),evaluation_digest=parent['evaluation_digest'],contrast_values=parent['contrast_values'],
        attribute_scenarios_digest=digest(cases),attribute_audit_sha=s.file_sha(DATA/'case-audits.json'),
        implementation={str(Path(name).relative_to(ROOT)).replace('\\','/'):s.file_sha(Path(name)) for name in (__file__,q.__file__,s.__file__,j.__file__)},
        endpoint_receipts=endpoints,endpoint_receipt_hashes={sid:s.file_sha(a.source/'training/V'/sid/'complete.json') for sid in p['targets']},
        arms=['E'],control='archived R2-R',control_registration=s.file_sha(s.REPORT/'attribute-generalization-v1/registration.json'),control_verified=s.file_sha(a.control/'final-verified.json'),steps=64,inherited_updates=[256,128,64,64],optimizer=dict(lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        schedule=[schedule(x) for x in range(64)],functions=functions(),progression=parent['progression'],
        comparative_target=dict(metric='semantic-group macro original MCQ accuracy',minimum_gain=0.10,comparison='E-archived-R'),
        budget=dict(updates=2560,gradient_forwards={'E':38400,'total':38400},generations=2996,
            ranking_decisions=1464,max_candidate_forwards=5856,endpoint_recovery_forwards=264),writer_updates=0,
        limitation='Fixed 40 observed endpoints; compositional cases train E and are new-content comparisons for archived R; not a fresh semantic holdout')
    s.write_frozen(DATA/'registration.json',reg);print(json.dumps(dict(registration_digest=digest(reg),budget=reg['budget'])))

def guard(a,reg):
    assert j.actual_host()==reg['runtime']['actual'];assert s.file_sha(a.source/'final-verified.json')==reg['parent_verified'];assert s.file_sha(a.control/'final-verified.json')==reg['control_verified']
    for path,sha in reg['implementation'].items():assert s.file_sha(ROOT/path)==sha

def train(a):
    reg,p,cases=load();guard(a,reg);processor,reader,vae,versions,bindings=s.models(a,True)
    termination=s.assistant_termination_contract(reader,processor);assignment=sorted(p['targets'])[a.shard::a.shards]
    s.write_frozen(a.output/f'train-identity-{a.shard}.json',dict(**s.identity(a,bindings,reg,assignment),actual_host=j.actual_host()))
    for sid in assignment:
        t=p['targets'][sid];src=a.source/'training/V'/sid;initial=reg['endpoint_receipts'][sid]
        assert s.file_sha(src/'complete.json')==reg['endpoint_receipt_hashes'][sid]
        z=torch.load(src/'latent.pt',map_location=a.device,weights_only=True);assert tensor_sha(z)==initial['final_tensor_sha']
        for arm in ('E',):
            out=a.output/'training'/arm/sid;out.mkdir(parents=True,exist_ok=False)
            oracle=s.VAELatentOracle(vae=vae,initial_latent=z,compute_dtype=torch.float32);opt=torch.optim.Adam([oracle.latent_fp32],**reg['optimizer'])
            torch.save(z.detach().cpu(),out/'initial-latent.pt');started=time.monotonic();forwards=tokens=0
            for step in range(64):
                opt.zero_grad(set_to_none=True);pixels=oracle.image();rec,apps=jobs(t,step,arm,cases);losses=[]
                for item,w in rec:
                    c0,t0=processor.total_calls,processor.total_input_tokens
                    ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=pixels[0],query=item['query'],target=item['target'],device=a.device,
                        termination=termination,lambda_eos=1.,require_image_grad=True,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    (w*ce.loss).backward(retain_graph=True);used=processor.total_input_tokens-t0;assert processor.total_calls-c0==1
                    forwards+=1;tokens+=used;losses.append(dict(query=item,weight=w,loss=float(ce.loss.detach()),reader_forwards=1,processed_input_tokens=used))
                for n,(item,w,choices,gold) in enumerate(apps):
                    c0,t0=processor.total_calls,processor.total_input_tokens
                    ce=s.qwen3vl_listwise_choice_ce(model=reader,processor=processor,image=pixels[0],query=item['query'],choices=choices,target_index=gold,
                        device=a.device,require_image_grad=True,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    (w*ce.loss).backward(retain_graph=n+1<len(apps));used=processor.total_input_tokens-t0;assert processor.total_calls-c0==4
                    forwards+=4;tokens+=used;losses.append(dict(query=item,weight=w,loss=float(ce.loss.detach()),choices=choices,gold_index=gold,
                        scores=ce.choice_logits.detach().tolist(),reader_forwards=4,processed_input_tokens=used))
                assert torch.isfinite(oracle.latent_fp32.grad).all();opt.step();s.append(out/'optimization.jsonl',dict(step=step,schedule=schedule(step),losses=losses,weighted_objective=sum(x['weight']*x['loss'] for x in losses)))
                if (step+1)%16==0:torch.save(dict(next_step=step+1,latent=oracle.latent_fp32.detach().cpu(),optimizer=opt.state_dict(),registration_digest=digest(reg)),out/'checkpoint.pt')
            torch.save(oracle.latent_fp32.detach().cpu(),out/'latent.pt');torch.save(opt.state_dict(),out/'optimizer.pt')
            with torch.no_grad():array=oracle.image()[0].detach().cpu().mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy()
            Image.fromarray(array).save(out/'memory.png');s.frozen(versions)
            s.write_frozen(out/'complete.json',dict(target=sid,arm=arm,inherited_updates=reg['inherited_updates'],additional_updates=64,
                initial_tensor_sha=tensor_sha(z),final_tensor_sha=tensor_sha(oracle.latent_fp32),reader_forwards=forwards,processed_input_tokens=tokens,
                seconds=time.monotonic()-started,artifacts={x:s.file_sha(out/x) for x in ('initial-latent.pt','latent.pt','optimizer.pt','checkpoint.pt','optimization.jsonl','memory.png')}))
            del opt,oracle
    s.write_frozen(a.output/f'train-complete-{a.shard}.json',dict(targets=assignment))

def evaluate(a):
    reg,p,new_cases=load();guard(a,reg);e=s.load_json(s.DATA/'evaluation-payload.json');old_cases={}
    for fn in ('training-scenarios.json','reserved-scenarios.json'):
        for cc in s.load_json(s.DATA/fn).values():
            for c in cc:old_cases[(c['id'],c['scope'],c['value_id'])]=c
    endpoints={f'{arm}/{sid}':s.file_sha((a.control if arm=='R' else a.output)/'training'/arm/sid/'complete.json') for sid in p['targets'] for arm in ('R','E')}
    processor,reader,_,versions,bindings=s.models(a);termination=s.assistant_termination_contract(reader,processor)
    assignment=sorted(p['targets'])[a.shard::a.shards];out=a.output/'evaluation'/f'shard-{a.shard}';out.mkdir(parents=True,exist_ok=False)
    frame=s.query_prompt(processor,'__REGISTERED_QUERY__');s.write_frozen(out/'identity.json',dict(**s.identity(a,bindings,reg,assignment),actual_host=j.actual_host(),frozen_endpoints=endpoints,prompt_frame=frame,termination=termination))
    started=time.monotonic();n=rn=cf=cen=0
    for sid in assignment:
        t=e['targets'][sid];train=p['targets'][sid]
        for arm in ('R','E'):
            path=(a.control if arm=='R' else a.output)/'training'/arm/sid/'memory.png';sha=s.file_sha(path);image=s.read_png(path)
            if arm=='E':
                for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
                    for item in t[panel]:
                        processor.begin_capture();g=s.generate(reader,processor,image,item['query'],a.device);proof=processor.end_capture()
                        s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel=panel,query=item,png_sha=sha,generation=g,score=s.score_generation(g,item,panel=='mcq'),reader_query=item['query'],processor_input=proof[0]));n+=1
                        if panel=='recovery_training':
                            with torch.no_grad():ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=image,query=item['query'],target=item['target'],device=a.device,
                                termination=termination,lambda_eos=1.,require_image_grad=False,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                            vals=ce.target_logits.float();nll=torch.logsumexp(vals,-1)-vals.gather(-1,ce.target_ids.unsqueeze(-1)).squeeze(-1)
                            s.append(out/'recovery-ce.jsonl',dict(target=sid,condition=arm,query=item,png_sha=sha,answer_ce=float(ce.answer_loss),eos_ce=float(ce.eos_loss),loss=float(ce.loss),answer_tokens=ce.answer_token_count,target_ids=ce.target_ids[0].tolist(),token_nll=nll[0].tolist()));cen+=1
                for scope,value in sorted(train['state'].items()):
                    for bank in ('question','instruction'):
                        for ix in range(4):
                            item=q.recovery_form(scope,value,bank,ix);g=s.generate(reader,processor,image,item['query'],a.device)
                            s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='recovery_coverage',query=item,png_sha=sha,generation=g,score=s.score_generation(g,item),reader_query=item['query']));n+=1
                for item0 in t['application_training']:
                    case=old_cases[(item0['case_id'],item0['scope'],item0['value_id'])];item,_,gold=q.xml_candidates(case,0);g=s.generate(reader,processor,image,item['query'],a.device)
                    s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='application_xml',query=item,png_sha=sha,generation=g,score=s.score_generation(g,{**item,'target_index':gold},True),reader_query=item['query']));n+=1
                for panel in ('application_training','application_reserved'):
                    for item0 in t[panel]:
                        case=old_cases[(item0['case_id'],item0['scope'],item0['value_id'])]
                        for rot in (range(4) if panel=='application_training' and item0['value_id'] in reg['contrast_values'] else (0,)):
                            item,ch,gold=s.ranking_candidates(case,rot);score=s.ranking_read(reader,processor,image,item['query'],ch,a.device);cf+=4
                            margin=score['scores'][gold]-max(x for k,x in enumerate(score['scores']) if k!=gold)
                            s.append(out/'ranking.jsonl',dict(target=sid,condition=arm,panel='application_training_rotations' if rot else panel,query=item,png_sha=sha,choices=ch,gold_index=gold,score=score,gold_margin=margin,unique_correct=margin>0));rn+=1
            for vid in sorted({item['value_id'] for item in t['application_training']}):
                for case in new_cases[vid]:
                    item,_,gold=q.xml_candidates(case,0);g=s.generate(reader,processor,image,item['query'],a.device)
                    s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='attribute_xml',query=item,png_sha=sha,generation=g,score=s.score_generation(g,{**item,'target_index':gold},True),reader_query=item['query']));n+=1
                    item,ch,gold=s.ranking_candidates(case,0);score=s.ranking_read(reader,processor,image,item['query'],ch,a.device);cf+=4
                    margin=score['scores'][gold]-max(x for k,x in enumerate(score['scores']) if k!=gold)
                    s.append(out/'ranking.jsonl',dict(target=sid,condition=arm,panel='attribute_full_action',query=item,png_sha=sha,choices=ch,gold_index=gold,score=score,gold_margin=margin,unique_correct=margin>0));rn+=1
    s.frozen(versions);s.write_frozen(out/'complete.json',dict(generations=n,rankings=rn,ranking_candidate_forwards=cf,recovery_ce_forwards=cen,processed_input_tokens=processor.total_input_tokens,
        seconds=time.monotonic()-started,artifacts={x:s.file_sha(out/x) for x in ('reads.jsonl','ranking.jsonl','recovery-ce.jsonl','identity.json')}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','train','evaluate'],required=True)
    for n in ('source','control','output','reader','base','runtime-receipt'):ap.add_argument('--'+n,type=Path)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if a.mode=='register':register(a)
    else:a.output.mkdir(parents=True,exist_ok=True);{'train':train,'evaluate':evaluate}[a.mode](a)
