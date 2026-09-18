"""Plan09: paired recovery-query and application-response interface coverage."""
from __future__ import annotations
import argparse,hashlib,inspect,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from PIL import Image
from scripts.experiments import prefeval_joint_consolidation as j
from vision_memory.prefeval.rgb_protocol import digest,scope_name

s=j.s; tensor_sha=j.tensor_sha
DATA=s.REPORT/'query-family-coverage-v1'; OLD=s.REPORT/'joint-consolidation-v1-run'
INSTANCE=j.INSTANCE
QUESTION_STEMS=(
    'What preference has been recorded about {scope}?',
    'For {scope}, what preference does the memory contain?',
    'What does the record say my preference is regarding {scope}?',
    'What preference information for {scope} is present in memory?',)
INSTRUCTION_STEMS=(
    'Report the preference entry associated with {scope}.',
    'Copy my recorded preference concerning {scope}.',
    'Give the recorded preference for {scope}, including its conditions.',
    'Return the preference text associated with {scope} in the memory.',)
RECOVERY_CONTRACT='\nReturn only the complete stored preference value for the requested scope. The scope label is not part of the value. Preserve every word and any quotation marks already present in the stored value. Do not add a label, quotation marks, tags, or explanation. For an inactive preference, return exactly: no active preference'
MCQ_SUFFIX='''\n    I'm trying to decide on this and here are 4 options for my query: \n{options}\nNow, I'd like you to pick one of them as your top recommendation for me.\n    Important instructions for your response:\n    1. Choose only one option (A, B, C, or D) that best matches my preferences.\n    2. Your answer must be one of these options.\n    3. Don't say things like "I can't choose" or suggest alternatives not listed.\n    4. Answer example: <choice>B</choice>. Give me your answer in this exact format, without any additional explanation:\n       <choice>[A/B/C/D]</choice>\n    '''

def recovery_form(scope,value,bank,index):
    stems=QUESTION_STEMS if bank=='question' else INSTRUCTION_STEMS
    query=stems[index].format(scope=scope_name(scope))+RECOVERY_CONTRACT
    return dict(id='coverage:'+digest([scope,bank,index,query]),scope=scope,kind='recovery_coverage',bank=bank,
        form=index,query=query,target=value if value is not None else 'no active preference')

def xml_candidates(case,rotation=0):
    offset=(case['base_rotation']+rotation)%4;options=case['proposals'][offset:]+case['proposals'][:offset]
    gold=options.index(case['target']);formatted='\n'.join(f'{chr(65+i)}. {x}' for i,x in enumerate(options))
    query=dict(id='coverage-xml:'+digest([case['id'],rotation,options]),scope=case['scope'],value_id=case['value_id'],
        case_id=case['id'],scenario=case['scenario'],rotation=rotation,kind='derived_xml_application',
        query=case['situation']+MCQ_SUFFIX.format(options=formatted),target=f'<choice>{chr(65+gold)}</choice>')
    choices=[f'<choice>{x}</choice>' for x in 'ABCD'];return query,choices,gold

def jobs(t,step,arm):
    weights={q['scope']:w for q,w in s.training_jobs(t,dict(recovery_form=0,mixture_application=False),'A')}
    rec=[]
    for scope,value in sorted(t['state'].items()):
        if arm=='U':qs=[q for q in sorted(t['recovery'],key=lambda q:q['form']) if q['scope']==scope]
        else:qs=[next(q for q in t['recovery'] if q['scope']==scope and q['form']==step%3),
                 recovery_form(scope,value,'question',step%4),recovery_form(scope,value,'instruction',(step//4)%4)]
        rec.extend((q,weights[scope]/3) for q in qs)
    app=[]
    for scope,value in sorted(t['state'].items()):
        if value is None:continue
        case=next(c for c in t['applications'] if c['scope']==scope and c['scenario']==step%4);rotation=(step//4)%4
        q,ch,gold=(xml_candidates(case,rotation) if arm=='V' and (step//16)%2 else s.ranking_candidates(case,rotation))
        app.append((q,.25*weights[scope],ch,gold))
    return rec,app

def functions():
    return {f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest() for f in (recovery_form,xml_candidates,jobs)}

def register(a):
    previous,p=j.load();assert digest(previous)=='f41aa88a39fd498813e672c62a036eaf9683e3c25ced1623c1fe3198fd28875e'
    runtime=s.load_json(a.runtime_receipt);assert runtime['instance']==INSTANCE and len(runtime['actual']['gpus'])==4
    assert INSTANCE in runtime['platform_status'] and 'Status: RUNNING' in runtime['platform_status']
    endpoints={sid:s.load_json(OLD/'training/J'/sid/'complete.json') for sid in p['targets']}
    for sid,r in endpoints.items():
        for name,h in r['artifacts'].items():assert s.file_sha(OLD/'training/J'/sid/name)==h
    DATA.mkdir(exist_ok=True)
    training=dict(question_stems=list(QUESTION_STEMS),instruction_stems=list(INSTRUCTION_STEMS),contract=RECOVERY_CONTRACT,
        mcq_suffix=MCQ_SUFFIX,mcq_source=s.load_json(s.REPORT/'reader-format-v2.json')['upstream'],
        schedule=[dict(step=x,original=x%3,question=x%4,instruction=(x//4)%4,
            scenario=x%4,rotation=(x//4)%4,format='xml' if (x//16)%2 else 'full-action') for x in range(64)])
    reg=dict(plan='prefeval-rgb-query-family-coverage-09',parent_registration_digest=digest(previous),
        parent_verified=s.file_sha(OLD/'final-verified.json'),endpoint_receipts=endpoints,
        endpoint_receipt_hashes={sid:s.file_sha(OLD/'training/J'/sid/'complete.json') for sid in p['targets']},
        training_digest=digest(p),evaluation_digest=previous['evaluation_digest'],contrast_values=previous['contrast_values'],
        runtime=runtime,instance=INSTANCE,steps=64,arms=['U','V'],inherited_updates=[256,128,64],
        optimizer=dict(lr=.01,betas=[.9,.999],eps=1e-8,weight_decay=0.),training_interface=training,
        functions=functions(),budget=dict(updates=5120,gradient_forwards={'U':38400,'V':38400,'total':76800},
        generations=4648,ranking_decisions=1584,max_candidate_forwards=6336,endpoint_recovery_forwards=528),
        progression=previous['progression'],writer_updates=0,
        limitation='Evaluation-informed development on independent endpoints; no recurrent Writer or fresh holdout claim')
    s.write_frozen(DATA/'training-payload.json',dict(parent_training_digest=digest(p),interfaces=training))
    s.write_frozen(DATA/'registration.json',reg);print(json.dumps(dict(registration_digest=digest(reg),budget=reg['budget'])))

def load():
    reg=s.load_json(DATA/'registration.json');previous,p=j.load()
    assert digest(previous)==reg['parent_registration_digest'] and digest(p)==reg['training_digest']
    assert functions()==reg['functions'] and reg['instance']==INSTANCE and reg['steps']==64
    assert digest(s.load_json(DATA/'training-payload.json')['interfaces'])==digest(reg['training_interface'])
    return reg,p

def guard(a,reg):
    assert j.actual_host()==reg['runtime']['actual'] and s.file_sha(a.source/'final-verified.json')==reg['parent_verified']

def train(a):
    reg,p=load();guard(a,reg);processor,reader,vae,versions,bindings=s.models(a,True)
    termination=s.assistant_termination_contract(reader,processor);assignment=sorted(p['targets'])[a.shard::a.shards]
    s.write_frozen(a.output/f'train-identity-{a.shard}.json',dict(**s.identity(a,bindings,reg,assignment),actual_host=j.actual_host()))
    for sid in assignment:
        t=p['targets'][sid];src=a.source/'training/J'/sid;initial=reg['endpoint_receipts'][sid]
        assert s.file_sha(src/'complete.json')==reg['endpoint_receipt_hashes'][sid]
        for name,h in initial['artifacts'].items():assert s.file_sha(src/name)==h
        z=torch.load(src/'latent.pt',map_location=a.device,weights_only=True);assert tensor_sha(z)==initial['final_tensor_sha']
        for arm in (('U','V') if sorted(p['targets']).index(sid)%2==0 else ('V','U')):
            out=a.output/'training'/arm/sid;out.mkdir(parents=True,exist_ok=False)
            oracle=s.VAELatentOracle(vae=vae,initial_latent=z,compute_dtype=torch.float32)
            opt=torch.optim.Adam([oracle.latent_fp32],**reg['optimizer']);assert not opt.state and tensor_sha(oracle.latent_fp32)==tensor_sha(z)
            torch.save(z.detach().cpu(),out/'initial-latent.pt');started=time.monotonic();forwards=tokens=0
            for step in range(64):
                opt.zero_grad(set_to_none=True);pixels=oracle.image();rec,app=jobs(t,step,arm);losses=[];before=oracle.latent_fp32.detach().clone()
                for n,(q,w) in enumerate(rec):
                    c0,t0=processor.total_calls,processor.total_input_tokens
                    ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=pixels[0],query=q['query'],target=q['target'],device=a.device,
                        termination=termination,lambda_eos=1.,require_image_grad=True,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    assert torch.isfinite(ce.loss);(w*ce.loss).backward(retain_graph=True);assert processor.total_calls-c0==1
                    used=processor.total_input_tokens-t0;forwards+=1;tokens+=used;losses.append(dict(query=q,weight=w,loss=float(ce.loss.detach()),
                        answer_ce=float(ce.answer_loss.detach()),eos_ce=float(ce.eos_loss.detach()),reader_forwards=1,processed_input_tokens=used))
                gr=oracle.latent_fp32.grad.detach().clone()
                for n,(q,w,ch,gold) in enumerate(app):
                    c0,t0=processor.total_calls,processor.total_input_tokens
                    ce=s.qwen3vl_listwise_choice_ce(model=reader,processor=processor,image=pixels[0],query=q['query'],choices=ch,target_index=gold,
                        device=a.device,require_image_grad=True,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    assert torch.isfinite(ce.loss);(w*ce.loss).backward(retain_graph=n+1<len(app));assert processor.total_calls-c0==4
                    used=processor.total_input_tokens-t0;forwards+=4;tokens+=used;losses.append(dict(query=q,weight=w,loss=float(ce.loss.detach()),
                        choices=ch,gold_index=gold,scores=ce.choice_logits.detach().tolist(),candidate_token_counts=list(ce.choice_token_counts),
                        reader_forwards=4,processed_input_tokens=used))
                total=oracle.latent_fp32.grad;assert total is not None and torch.isfinite(total).all()
                if step==0:assert torch.any(total!=0)
                gp=total.detach()-gr;nr,np=float(gr.norm()),float(gp.norm());dot=float((gr*gp).sum());opt.step();delta=oracle.latent_fp32.detach()-before
                row=dict(step=step,measurement='pre-update losses; one optimizer step after all components',losses=losses,
                    weighted_objective=sum(x['weight']*x['loss'] for x in losses),gradient=dict(recovery_norm=nr,weighted_application_norm=np,dot=dot,
                    cosine=dot/(nr*np) if nr*np else None,recovery_dot_displacement=float((gr*delta).sum()),
                    weighted_application_dot_displacement=float((gp*delta).sum()),displacement_norm=float(delta.norm())),seconds=time.monotonic()-started)
                s.append(out/'optimization.jsonl',row)
                if (step+1)%16==0:
                    torch.save(dict(next_step=step+1,latent=oracle.latent_fp32.detach().cpu(),optimizer=opt.state_dict(),rng_cpu=torch.get_rng_state(),
                        rng_cuda=torch.cuda.get_rng_state(),registration_digest=digest(reg)),out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
                    print(json.dumps(dict(target=sid,arm=arm,step=step+1,loss=row['weighted_objective'])),flush=True)
            torch.save(oracle.latent_fp32.detach().cpu(),out/'latent.pt');torch.save(opt.state_dict(),out/'optimizer.pt')
            with torch.no_grad():array=oracle.image()[0].detach().cpu().mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy()
            Image.fromarray(array).save(out/'memory.png');s.frozen(versions)
            s.write_frozen(out/'complete.json',dict(target=sid,arm=arm,inherited_updates=reg['inherited_updates'],additional_updates=64,
                initial_tensor_sha=tensor_sha(z),final_tensor_sha=tensor_sha(oracle.latent_fp32),reader_forwards=forwards,processed_input_tokens=tokens,
                seconds=time.monotonic()-started,artifacts={x:s.file_sha(out/x) for x in ('initial-latent.pt','latent.pt','optimizer.pt','checkpoint.pt','optimization.jsonl','memory.png')}))
            del opt,oracle
    s.write_frozen(a.output/f'train-complete-{a.shard}.json',dict(targets=assignment))

def evaluate(a):
    reg,p=load();guard(a,reg);endpoints={}
    for sid in p['targets']:
        for arm in ('U','V'):
            f=a.output/'training'/arm/sid;done=s.load_json(f/'complete.json');assert done['additional_updates']==64
            endpoints[f'{arm}/{sid}']=s.file_sha(f/'complete.json')
    e=s.load_json(s.DATA/'evaluation-payload.json');assert digest(e)==reg['evaluation_digest'];cases={}
    for fn in ('training-scenarios.json','reserved-scenarios.json'):
        for cc in s.load_json(s.DATA/fn).values():
            for c in cc:cases[(c['id'],c['scope'],c['value_id'])]=c
    processor,reader,_,versions,bindings=s.models(a);termination=s.assistant_termination_contract(reader,processor)
    assignment=sorted(p['targets'])[a.shard::a.shards];out=a.output/'evaluation'/f'shard-{a.shard}';out.mkdir(parents=True,exist_ok=False)
    frame=s.query_prompt(processor,'__REGISTERED_QUERY__');s.write_frozen(out/'identity.json',dict(**s.identity(a,bindings,reg,assignment),
        actual_host=j.actual_host(),frozen_endpoints=endpoints,prompt_frame=frame,chat_template_digest=digest(processor.chat_template),termination=termination))
    started=time.monotonic();n=rn=cf=cen=0;cache={}
    for sid in assignment:
        t=e['targets'][sid];train=p['targets'][sid]
        for arm in ('U','V'):
            path=a.output/'training'/arm/sid/'memory.png';sha=s.file_sha(path);image=s.read_png(path)
            for panel in ('recovery_training','qualification','application_training','mcq','application_reserved'):
                for q in t[panel]:
                    processor.begin_capture();g=s.generate(reader,processor,image,q['query'],a.device);proof=processor.end_capture();assert len(proof)==1
                    s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel=panel,query=q,png_sha=sha,generation=g,
                        score=s.score_generation(g,q,panel=='mcq'),reader_query=q['query'],processor_input=proof[0]));n+=1
                    if panel=='recovery_training':
                        processor.begin_capture()
                        with torch.no_grad():ce=s.qwen3vl_answer_eos_ce(model=reader,processor=processor,image=image,query=q['query'],target=q['target'],device=a.device,
                            termination=termination,lambda_eos=1.,require_image_grad=False,reader_resize_contract=s.R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                        pp=processor.end_capture();vals=ce.target_logits.float();nll=torch.logsumexp(vals,-1)-vals.gather(-1,ce.target_ids.unsqueeze(-1)).squeeze(-1)
                        s.append(out/'recovery-ce.jsonl',dict(target=sid,condition=arm,query=q,png_sha=sha,answer_ce=float(ce.answer_loss),eos_ce=float(ce.eos_loss),
                            loss=float(ce.loss),answer_tokens=ce.answer_token_count,processor_input=pp[0],target_ids=ce.target_ids[0].tolist(),token_nll=nll[0].tolist()));cen+=1
            for scope,value in sorted(train['state'].items()):
                for bank in ('question','instruction'):
                    for ix in range(4):
                        q=recovery_form(scope,value,bank,ix);processor.begin_capture();g=s.generate(reader,processor,image,q['query'],a.device);pp=processor.end_capture()
                        s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='recovery_coverage',query=q,png_sha=sha,generation=g,
                            score=s.score_generation(g,q),reader_query=q['query'],processor_input=pp[0]));n+=1
            for q0 in t['application_training']:
                case=cases[(q0['case_id'],q0['scope'],q0['value_id'])];q,_,_=xml_candidates(case,0)
                processor.begin_capture();g=s.generate(reader,processor,image,q['query'],a.device);pp=processor.end_capture()
                s.append(out/'reads.jsonl',dict(target=sid,condition=arm,panel='application_xml',query=q,png_sha=sha,generation=g,
                    score=s.score_generation(g,q,True),reader_query=q['query'],processor_input=pp[0]));n+=1
            for panel in ('application_training','application_reserved'):
                for q0 in t[panel]:
                    case=cases[(q0['case_id'],q0['scope'],q0['value_id'])]
                    for rot in (range(4) if panel=='application_training' and q0['value_id'] in reg['contrast_values'] else (0,)):
                        q,ch,gold=s.ranking_candidates(case,rot);pan='application_training_rotations' if rot else panel;key=digest(dict(query=q['query'],choices=ch,png=sha));reuse=key in cache
                        if not reuse:cache[key]=s.ranking_read(reader,processor,image,q['query'],ch,a.device);cf+=4
                        score=cache[key];margin=score['scores'][gold]-max(x for k,x in enumerate(score['scores']) if k!=gold)
                        s.append(out/'ranking.jsonl',dict(target=sid,condition=arm,panel=pan,query=q,png_sha=sha,reader_query=q['query'],choices=ch,gold_index=gold,
                            score=score,gold_margin=margin,unique_correct=margin>0,input_identity=key,reused=reuse,actual_candidate_forwards=0 if reuse else 4));rn+=1
            print(json.dumps(dict(target=sid,arm=arm,generations=n,rankings=rn,recovery_ce=cen)),flush=True)
    s.frozen(versions);s.write_frozen(out/'complete.json',dict(generations=n,rankings=rn,ranking_candidate_forwards=cf,recovery_ce_forwards=cen,
        processed_input_tokens=processor.total_input_tokens,seconds=time.monotonic()-started,
        artifacts={x:s.file_sha(out/x) for x in ('reads.jsonl','ranking.jsonl','recovery-ce.jsonl','identity.json')}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','train','evaluate'],required=True)
    for n in ('source','output','reader','base','runtime-receipt'):ap.add_argument('--'+n,type=Path)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    if a.mode=='register':register(a)
    else:a.output.mkdir(parents=True,exist_ok=True);{'train':train,'evaluate':evaluate}[a.mode](a)
