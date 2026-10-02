"""Plan08: simultaneous all-form recovery and application consolidation."""
from __future__ import annotations
import argparse,json,sys,socket,subprocess,time,inspect,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from PIL import Image
from scripts.experiments import prefeval_semantic_transfer as s
from vision_memory.prefeval.rgb_protocol import digest
from vision_memory.repro import canonical_tensor_sha256 as tensor_sha

DATA=s.REPORT/'joint-consolidation-v1'
OLD=s.REPORT/'ranking-learning-trial-v1-run'
INSTANCE='dl-clear-retain-h200x4-20260914'

def jobs(t,step,arm):
    weights={q['scope']:w for q,w in s.training_jobs(t,dict(recovery_form=0,mixture_application=False), 'A')}
    recovery=[(q,weights[q['scope']]/3) for scope in sorted(t['state']) for q in sorted(t['recovery'],key=lambda q:q['form']) if q['scope']==scope]
    application=[]
    if arm=='J':
        for scope,value in sorted(t['state'].items()):
            if value is None:continue
            case=next(c for c in t['applications'] if c['scope']==scope and c['scenario']==step%4)
            q,choices,gold=s.ranking_candidates(case,(step//4)%4)
            application.append((q,.25*weights[scope],choices,gold))
    return recovery,application

def actual_host():
    return dict(hostname=socket.gethostname(),gpus=subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True).strip().splitlines())

def register(a):
    previous,p=s.trial_load();assert digest(previous)=='a719bef64546ba85893b933f7ec5356910de06a03951ef621b165b63ae7c1f9b'
    runtime=s.load_json(a.runtime_receipt);assert runtime['instance']==INSTANCE and len(runtime['actual']['gpus'])==4
    assert INSTANCE in runtime['platform_status'] and 'Status: RUNNING' in runtime['platform_status']
    endpoints={sid:s.load_json(OLD/'training/B'/sid/'complete.json') for sid in p['targets']}
    for sid,receipt in endpoints.items():
        for name,h in receipt['artifacts'].items():assert s.file_sha(OLD/'training/B'/sid/name)==h
    DATA.mkdir(exist_ok=True)
    reg=dict(plan='prefeval-rgb-joint-consolidation-08',parent_registration_digest=digest(previous),
        parent_verified={name:s.file_sha(OLD/name) for name in ('generation-verified.json','final-verified.json')},
        parent_evaluation_files={f'evaluation/shard-{i}/{name}':s.file_sha(OLD/'evaluation'/f'shard-{i}'/name)
            for i in range(4) for name in ('identity.json','complete.json','reads.jsonl','ranking.jsonl')},
        endpoint_receipts=endpoints,endpoint_receipt_hashes={sid:s.file_sha(OLD/'training/B'/sid/'complete.json') for sid in p['targets']},
        training_digest=digest(p),evaluation_digest=previous['evaluation_digest'],contrast_values=previous['contrast_values'],
        runtime=runtime,instance=INSTANCE,steps=64,arms=['C','J'],inherited_updates=[256,128],
        optimizer=dict(lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        objective='C: R; J: R+0.25P, all three recovery forms every update, same latent before one Adam step',
        schedule=[dict(step=i,scenario=i%4,rotation=(i//4)%4,recovery_forms=[0,1,2]) for i in range(64)],
        functions={**s.ranking_functions(),'jobs':hashlib.sha256(inspect.getsource(jobs).encode()).hexdigest()},
        budget=dict(updates=5120,gradient_forwards={'C':16896,'J':38400,'total':55296},
            generations=2568,ranking_decisions=1584,max_candidate_forwards=6336,endpoint_recovery_forwards=528),
        diagnostic_status='Previously reserved scenarios are observed gradient-excluded transfer diagnostics; no fresh holdout claim',
        progression=previous['progression'],writer_updates=0)
    s.write_frozen(DATA/'registration.json',reg);print(json.dumps(dict(registration_digest=digest(reg),budget=reg['budget'])))

def load():
    reg=s.load_json(DATA/'registration.json');previous,p=s.trial_load()
    assert digest(previous)==reg['parent_registration_digest'] and digest(p)==reg['training_digest']
    assert reg['functions']=={**s.ranking_functions(),'jobs':hashlib.sha256(inspect.getsource(jobs).encode()).hexdigest()}
    assert reg['instance']==INSTANCE and reg['steps']==64
    return reg,p

def guard(a,reg):
    assert actual_host()==reg['runtime']['actual']
    for name,h in reg['parent_verified'].items():assert s.file_sha(a.source/name)==h

def train(a):
    reg,p=load();guard(a,reg);processor,reader,vae,versions,bindings=s.models(a,True)
    termination=s.assistant_termination_contract(reader,processor);assignment=sorted(p['targets'])[a.shard::a.shards]
    s.write_frozen(a.output/f'train-identity-{a.shard}.json',dict(**s.identity(a,bindings,reg,assignment),actual_host=actual_host()))
    for sid in assignment:
        t=p['targets'][sid];src=a.source/'training/B'/sid;initial=reg['endpoint_receipts'][sid]
        assert s.file_sha(src/'complete.json')==reg['endpoint_receipt_hashes'][sid]
        for name,h in initial['artifacts'].items():assert s.file_sha(src/name)==h
        z=torch.load(src/'latent.pt',map_location=a.device,weights_only=True);assert tensor_sha(z)==initial['final_tensor_sha']
        for arm in (('C','J') if sorted(p['targets']).index(sid)%2==0 else ('J','C')):
            out=a.output/'training'/arm/sid;out.mkdir(parents=True,exist_ok=False)
            oracle=s.VAELatentOracle(vae=vae,initial_latent=z,compute_dtype=torch.float32)
            opt=torch.optim.Adam([oracle.latent_fp32],**reg['optimizer']);assert not opt.state and tensor_sha(oracle.latent_fp32)==tensor_sha(z)
            torch.save(z.detach().cpu(),out/'initial-latent.pt');started=time.monotonic();forwards=tokens=0
            for step in range(64):
                opt.zero_grad(set_to_none=True);pixels=oracle.image();rec,app=jobs(t,step,arm);losses=[];before=oracle.latent_fp32.detach().clone()
                for j,(q,w) in enumerate(rec):
                    calls0=processor.total_calls;tokens0=processor.total_input_tokens
                    ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=pixels[0],query=q['query'],target=q['target'],
                        device=a.device,termination=termination,lambda_eos=1.,require_image_grad=True,
                        reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    assert torch.isfinite(ce.loss);(w*ce.loss).backward(retain_graph=bool(app) or j+1<len(rec))
                    assert processor.total_calls-calls0==1;n=processor.total_input_tokens-tokens0;forwards+=1;tokens+=n
                    losses.append(dict(query=q,weight=w,loss=float(ce.loss.detach()),answer_ce=float(ce.answer_loss.detach()),
                        eos_ce=float(ce.eos_loss.detach()),reader_forwards=1,processed_input_tokens=n))
                gr=oracle.latent_fp32.grad.detach().clone()
                for j,(q,w,choices,gold) in enumerate(app):
                    calls0=processor.total_calls;tokens0=processor.total_input_tokens
                    ce=s.qwen3vl_listwise_choice_ce(model=reader,processor=processor,image=pixels[0],query=q['query'],choices=choices,
                        target_index=gold,device=a.device,require_image_grad=True,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    assert torch.isfinite(ce.loss);(w*ce.loss).backward(retain_graph=j+1<len(app))
                    assert processor.total_calls-calls0==4;n=processor.total_input_tokens-tokens0;forwards+=4;tokens+=n
                    losses.append(dict(query=q,weight=w,loss=float(ce.loss.detach()),choices=choices,gold_index=gold,
                        scores=ce.choice_logits.detach().tolist(),candidate_token_counts=list(ce.choice_token_counts),reader_forwards=4,processed_input_tokens=n))
                total=oracle.latent_fp32.grad;assert total is not None and torch.isfinite(total).all()
                if step==0:assert torch.any(total!=0)
                gp=total.detach()-gr;nr=float(gr.norm());np=float(gp.norm());dot=float((gr*gp).sum())
                opt.step();assert torch.isfinite(oracle.latent_fp32).all();delta=oracle.latent_fp32.detach()-before
                diagnostics=dict(recovery_norm=nr,weighted_application_norm=np,dot=dot,cosine=dot/(nr*np) if nr*np>0 else None,
                    recovery_dot_displacement=float((gr*delta).sum()),weighted_application_dot_displacement=float((gp*delta).sum()),displacement_norm=float(delta.norm()))
                row=dict(step=step,measurement='pre-update losses; one optimizer step after all components',losses=losses,
                    weighted_objective=sum(x['weight']*x['loss'] for x in losses),gradient=diagnostics,seconds=time.monotonic()-started)
                s.append(out/'optimization.jsonl',row)
                if (step+1)%16==0:
                    torch.save(dict(next_step=step+1,latent=oracle.latent_fp32.detach().cpu(),optimizer=opt.state_dict(),
                        rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),registration_digest=digest(reg)),out/'checkpoint.tmp')
                    (out/'checkpoint.tmp').replace(out/'checkpoint.pt')
                    print(json.dumps(dict(target=sid,arm=arm,step=step+1,loss=row['weighted_objective'])),flush=True)
            torch.save(oracle.latent_fp32.detach().cpu(),out/'latent.pt');torch.save(opt.state_dict(),out/'optimizer.pt')
            with torch.no_grad():array=oracle.image()[0].detach().cpu().mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy()
            Image.fromarray(array).save(out/'memory.png');assert torch.equal(s.read_png(out/'memory.png').mul(255).round().byte().permute(1,2,0),torch.from_numpy(array))
            s.frozen(versions)
            for name,h in initial['artifacts'].items():assert s.file_sha(src/name)==h
            s.write_frozen(out/'complete.json',dict(target=sid,arm=arm,inherited_updates=[256,128],additional_updates=64,
                initial_tensor_sha=tensor_sha(z),final_tensor_sha=tensor_sha(oracle.latent_fp32),reader_forwards=forwards,
                processed_input_tokens=tokens,seconds=time.monotonic()-started,
                artifacts={name:s.file_sha(out/name) for name in ('initial-latent.pt','latent.pt','optimizer.pt','checkpoint.pt','optimization.jsonl','memory.png')}))
            del opt,oracle
    s.write_frozen(a.output/f'train-complete-{a.shard}.json',dict(targets=assignment))

def evaluate(a):
    reg,p=load();guard(a,reg);endpoints={}
    for sid in p['targets']:
        for arm in ('C','J'):
            folder=a.output/'training'/arm/sid;done=s.load_json(folder/'complete.json');assert done['additional_updates']==64
            assert s.file_sha(folder/'memory.png')==done['artifacts']['memory.png'];endpoints[f'{arm}/{sid}']=s.file_sha(folder/'complete.json')
    e=s.load_json(s.DATA/'evaluation-payload.json');assert digest(e)==reg['evaluation_digest']
    cases={}
    for filename in ('training-scenarios.json','reserved-scenarios.json'):
        for cc in s.load_json(s.DATA/filename).values():
            for c in cc:cases[(c['id'],c['scope'],c['value_id'])]=c
    processor,reader,_,versions,bindings=s.models(a);termination=s.assistant_termination_contract(reader,processor)
    assignment=sorted(p['targets'])[a.shard::a.shards];out=a.output/'evaluation'/f'shard-{a.shard}';out.mkdir(parents=True,exist_ok=False)
    frame=s.query_prompt(processor,'__REGISTERED_QUERY__')
    s.write_frozen(out/'identity.json',dict(**s.identity(a,bindings,reg,assignment),actual_host=actual_host(),frozen_endpoints=endpoints,
        prompt_frame=frame,chat_template_digest=digest(processor.chat_template),termination=termination))
    started=time.monotonic();n=rn=cf=cen=0;cache={}
    for sid in assignment:
        t=e['targets'][sid]
        for arm in ('C','J'):
            path=a.output/'training'/arm/sid/'memory.png';sha=s.file_sha(path);image=s.read_png(path)
            for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
                for q in t[panel]:
                    processor.begin_capture();g=s.generate(reader,processor,image,q['query'],a.device);proof=processor.end_capture()
                    assert len(proof)==1 and proof[0]['text']==frame.replace('__REGISTERED_QUERY__',q['query'])
                    assert proof[0]['input_ids_digest']==digest([g['input_token_ids']])
                    s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel=panel,query=q,png_sha=sha,generation=g,
                        score=s.score_generation(g,q,panel=='mcq'),reader_query=q['query'],processor_input=proof[0]));n+=1
                    if panel=='recovery_training':
                        processor.begin_capture()
                        with torch.no_grad():ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=image,query=q['query'],target=q['target'],
                            device=a.device,termination=termination,lambda_eos=1.,require_image_grad=False,
                            reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                        proof=processor.end_capture();assert len(proof)==1
                        assert proof[0]['text']==frame.replace('__REGISTERED_QUERY__',q['query'])+q['target']+termination['assistant_end_token_text']
                        values=ce.target_logits.float();nll=torch.logsumexp(values,-1)-values.gather(-1,ce.target_ids.unsqueeze(-1)).squeeze(-1)
                        s.append(out/'recovery-ce.jsonl',dict(target=sid,condition=arm,query=q,png_sha=sha,answer_ce=float(ce.answer_loss),
                            eos_ce=float(ce.eos_loss),loss=float(ce.loss),answer_tokens=ce.answer_token_count,processor_input=proof[0],
                            target_ids=ce.target_ids[0].tolist(),token_nll=nll[0].tolist()));cen+=1
            for panel in ('application_training','application_reserved'):
                for q0 in t[panel]:
                    case=cases[(q0['case_id'],q0['scope'],q0['value_id'])]
                    for rotation in (range(4) if panel=='application_training' and q0['value_id'] in reg['contrast_values'] else (0,)):
                        q,choices,gold=s.ranking_candidates(case,rotation);actual_panel='application_training_rotations' if rotation else panel
                        key=digest(dict(query=q['query'],choices=choices,png=sha));reuse=key in cache
                        if not reuse:cache[key]=s.ranking_read(reader,processor,image,q['query'],choices,a.device);cf+=4
                        score=cache[key];margin=score['scores'][gold]-max(x for j,x in enumerate(score['scores']) if j!=gold)
                        s.append(out/'ranking.jsonl',dict(target=sid,condition=arm,panel=actual_panel,query=q,png_sha=sha,reader_query=q['query'],
                            choices=choices,gold_index=gold,score=score,gold_margin=margin,unique_correct=margin>0,input_identity=key,
                            reused=reuse,actual_candidate_forwards=0 if reuse else 4));rn+=1
            print(json.dumps(dict(target=sid,arm=arm,generations=n,rankings=rn,recovery_ce=cen)),flush=True)
    s.frozen(versions)
    s.write_frozen(out/'complete.json',dict(generations=n,rankings=rn,ranking_candidate_forwards=cf,recovery_ce_forwards=cen,
        processed_input_tokens=processor.total_input_tokens,seconds=time.monotonic()-started,
        artifacts={name:s.file_sha(out/name) for name in ('reads.jsonl','ranking.jsonl','recovery-ce.jsonl','identity.json')}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','train','evaluate'],required=True)
    for name in ('source','output','reader','base','runtime-receipt'):ap.add_argument('--'+name,type=Path)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args()
    if a.mode=='register':register(a)
    else:a.output.mkdir(parents=True,exist_ok=True);{'train':train,'evaluate':evaluate}[a.mode](a)
