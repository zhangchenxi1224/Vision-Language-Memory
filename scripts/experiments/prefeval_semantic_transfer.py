"""A fixed semantic-query mixture versus recovery-only teacher continuation."""
from __future__ import annotations
import argparse,json,hashlib,time,sys,subprocess,inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from PIL import Image
from scripts.eval.prefeval_rgb import load_reader,read_png,append,load_overlay
from scripts.probes.prefeval_rgb_endpoint_diagnostic import load_json,file_sha,score_generation,generate
from scripts.train.latent_r11_vae_oracle import VAELatentOracle
from scripts.experiments.build_prefeval_rgb_teachers import validate_vae
from vision_memory.prefeval.rgb_protocol import digest,text_prefix_v2,recovery_summary
from vision_memory.repro import canonical_tensor_sha256 as tensor_sha,configure_strict_cuda_determinism
from vision_memory.reader.open_eos import assistant_termination_contract,qwen3vl_answer_eos_ce
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.reader.qwen3vl import qwen3vl_choice_nll,qwen3vl_listwise_choice_ce,_joint_prompt_target_tokenization

REPORT=ROOT/'reports/prefeval-rgb-20260917';DATA=REPORT/'semantic-transfer-v1'
RANK_DATA=REPORT/'semantic-ranking-v1'
TRIAL_DATA=REPORT/'ranking-learning-trial-v1'

def trial_register():
    previous,p=ranking_load();old=load_json(DATA/'registration.json')
    calibration=REPORT/'semantic-ranking-v1-run/calibration-verified.json'
    failed=load_json(calibration);assert not failed['passed'] and failed['decisions']==2688
    metadata=load_json(DATA/'authoring-source.json')
    values=sorted({c[k] for c in metadata['overwrite_contrasts'] for k in ('before','after')})
    assert len(values)==8
    rows=[]
    for shard in range(4):
        path=calibration.parent/'calibration'/f'shard-{shard}'/'scores.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            if r['query']['value_id'] not in values:continue
            t=p['targets'][r['target']]
            rows.append(dict(target=r['target'],capacity=len(t['state']),value_id=r['query']['value_id'],
                condition=r['condition'],case_id=r['query']['case_id'],rotation=r['query']['rotation'],
                context_identity=digest(t['state']),reader_query=r['reader_query'],input_identity=r['input_identity'],
                candidate_ids=[digest(x) for x in r['choices']],choices=r['choices'],scores=r['score']['scores'],
                gold_index=r['gold_index'],gold_margin=r['gold_margin'],correct=r['unique_correct']))
    assert len(rows)==768 and len({(r['target'],r['value_id']) for r in rows})==24
    paired=[]
    for r in rows:
        if r['capacity']==1:continue
        singleton=[x for x in rows if x['capacity']==1 and all(x[k]==r[k] for k in ('value_id','condition','case_id','rotation'))]
        assert len(singleton)==1;s=singleton[0]
        paired.append(dict(singleton=s['target'],context=r['target'],value_id=r['value_id'],condition=r['condition'],
            case_id=r['case_id'],rotation=r['rotation'],change=f"{int(s['correct'])}->{int(r['correct'])}"))
    TRIAL_DATA.mkdir(exist_ok=True)
    attribution=dict(source_verified_sha=file_sha(calibration),model_calls=0,rows=rows,
        singleton_to_context=paired,overwrite_contrasts=failed['contrasts'])
    write_frozen(TRIAL_DATA/'calibration-attribution.json',attribution)
    reg=dict(plan='prefeval-rgb-ranking-learning-trial-07',allocation='exploratory learning trial; historical gates remain failed',
        instance='dl-clear-retain-h200x4-20260914',parent_registration_digest=digest(previous),
        calibration_verified_sha=file_sha(calibration),calibration_archive_sha=file_sha(REPORT/'semantic-ranking-calibration-v1.tgz'),
        attribution_digest=digest(attribution),training_digest=old['training_digest'],evaluation_digest=old['evaluation_digest'],
        manifest_digest=old['manifest_digest'],endpoints=previous['endpoints'],functions=ranking_functions(),
        steps=128,schedule_digest=digest(p['schedule'][:128]),optimizer=p['optimizer'],contrast_values=values,
        additional_updates=10240,logical_slot_jobs=22528,gradient_forwards={'A':11264,'B':27392,'total':38656},
        final={'generations':3408,'ranking_decisions':2712,'max_candidate_forwards':10848},
        progression=previous['progression'],historical_gates={'Plan05':False,'Plan06':False},
        comparison='fixed updates; unequal compute; no efficiency claim',writer_updates=0)
    write_frozen(TRIAL_DATA/'registration.json',reg)
    print(json.dumps(dict(registration_digest=digest(reg),updates=10240,gradient_forwards=38656)))

def trial_load():
    reg=load_json(TRIAL_DATA/'registration.json');previous,p=ranking_load()
    assert digest(previous)==reg['parent_registration_digest'] and ranking_functions()==reg['functions']
    assert digest(p)==reg['training_digest'] and digest(p['schedule'][:128])==reg['schedule_digest']
    assert reg['steps']==128 and reg['instance']=='dl-clear-retain-h200x4-20260914'
    return reg,p

def value_id(scope,value):return hashlib.sha256((scope+'\n'+value).encode()).hexdigest()[:12]

def application_view(case,rotation=0):
    offset=(case['base_rotation']+rotation)%4
    options=case['proposals'][offset:]+case['proposals'][:offset]
    query=case['situation']+'\n'+ '\n'.join(f'{label}. {text}' for label,text in zip('ABCD',options))+'\n'+case['instruction']
    return dict(id=f'application:{case["id"]}:{rotation}:{case["scope"]}',scope=case['scope'],
        query=query,target=case['target'],kind='application',scenario=case['scenario'],rotation=rotation,
        case_id=case['id'],value_id=case['value_id'])

def write_frozen(path,obj):
    if path.exists():assert load_json(path)==obj
    else:path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')

def register():
    m=load_json(REPORT/'registered/manifest.json');o=load_overlay(REPORT/'reader-format-v2.json',m)
    parent=load_json(REPORT/'endpoint-diagnostic-v1.json');training=load_json(DATA/'training-scenarios.json')
    reserved=load_json(DATA/'reserved-scenarios.json');source=load_json(DATA/'authoring-source.json')
    train_targets={};eval_targets={}
    for sid in sorted(m['sentinel_targets']):
        t=m['targets'][sid];applications=[];probes=[]
        for scope,value in sorted(t['state'].items()):
            if value is None:continue
            vid=value_id(scope,value)
            applications.extend(training[vid]);probes.extend(application_view(c) for c in reserved[vid])
        recoveries=[dict(q,id=f'recovery:{q["scope"]}:{q["form"]}') for q in o['teachers'][sid]['training']]
        train_targets[sid]=dict(state=t['state'],changed_scope=t['changed_scope'],recovery=recoveries,
            applications=applications,initial=parent['targets'][sid],text_prefix=text_prefix_v2(t['state']))
        # Strip evaluation fields from the parent endpoint binding before training.
        train_targets[sid]['initial']={k:v for k,v in parent['targets'][sid].items() if k not in ('mcq','recovery')}
        eval_targets[sid]=dict(state=t['state'],changed_scope=t['changed_scope'],predecessor=t['predecessor'],
            qualification=[dict(q,id=f'qualification:{q["kind"]}:{q["scope"]}:{q.get("form",0)}') for q in o['teachers'][sid]['qualification']],
            recovery_training=recoveries,application_training=[application_view(c) for c in applications],
            application_reserved=probes,mcq=[dict(q,id='mcq:'+q['scope'],kind='official_mcq') for q in parent['targets'][sid]['mcq']],
            text_prefix=text_prefix_v2(t['state']),parent=parent['targets'][sid])
    schedule=[dict(step=s,recovery_form=s%3,mixture_application=bool(s%2),
        application_scenario=(s//2)%4,rotation=((s//2)//4)%4) for s in range(256)]
    train=dict(targets=train_targets,schedule=schedule,arms=['A','B'],
        optimizer=dict(name='Adam',lr=.05,betas=[.9,.999],eps=1e-8,weight_decay=0.),
        training_side='A recovery only; B odd steps application on active slots; all other units recovery',max_new_tokens=128)
    evaluation=dict(targets=eval_targets,overwrite_contrasts=source['overwrite_contrasts'],max_new_tokens=128)
    write_frozen(DATA/'training-payload.json',train);write_frozen(DATA/'evaluation-payload.json',evaluation)
    registration=dict(plan='prefeval-rgb-semantic-query-mixture-05',parent_endpoint_digest=digest(parent),
        manifest_digest=digest(m),overlay_digest=digest(o),training_digest=digest(train),evaluation_digest=digest(evaluation),
        files={name:digest(load_json(DATA/name)) for name in ('authoring-source.json','training-scenarios.json','reserved-scenarios.json','clause-option-audit.json')},
        budgets=dict(feasibility=672,additional_updates=20480,gradient_slot_forwards=45056,final_reads=3408,total_generations=4080),
        feasibility_min_correct=303,feasibility_total=336,
        progression=dict(recovery={'K1':[18,20],'K2':[3,4],'K4':[9,12]},original_mcq=[68,84],reserved_application=[135,168],macro_gain_over_control=.10),
        inherited_steps_per_state=256,added_steps_per_state_per_arm=256,
        arm_order='A then B for even lexicographic target index, B then A for odd; each target pair on one GPU',
        compute_matching='same updates and slot-forward counts; token counts/FLOPs/wall-time may differ',
        probe_limitation='reserved scenarios use distinct action contexts from the same authored attribute schemas; original MCQ is the separate application diagnostic')
    write_frozen(DATA/'registration.json',registration)
    print(json.dumps(dict(registration_digest=digest(registration),budgets=registration['budgets'])))

def training_jobs(t,draw,arm):
    jobs=[]
    for scope,value in sorted(t['state'].items()):
        if arm=='B' and draw['mixture_application'] and value is not None:
            case=next(c for c in t['applications'] if c['scope']==scope and c['scenario']==draw['application_scenario'])
            q=application_view(case,draw['rotation'])
        else:q=next(q for q in t['recovery'] if q['scope']==scope and q['form']==draw['recovery_form'])
        jobs.append(q)
    changed=[q for q in jobs if q['scope']==t['changed_scope']];untouched=[q for q in jobs if q['scope']!=t['changed_scope']]
    groups=[g for g in (changed,untouched) if g]
    return [(q,1/(len(groups)*len(g))) for g in groups for q in g]

class TokenCountProcessor:
    def __init__(self,processor):
        self.processor=processor;self.last_input_tokens=0;self.total_input_tokens=0;self.total_calls=0;self.capture=None
    def __getattr__(self,name):return getattr(self.processor,name)
    def begin_capture(self):self.capture=[]
    def end_capture(self):
        result=self.capture;self.capture=None;return result
    def __call__(self,**kwargs):
        batch=self.processor(**kwargs);self.last_input_tokens=int(batch['input_ids'].numel())
        self.total_calls+=1;self.total_input_tokens+=self.last_input_tokens
        if self.capture is not None:self.capture.append(dict(text=kwargs['text'][0],input_tokens=self.last_input_tokens,
            input_ids_digest=digest(batch['input_ids'].tolist())))
        return batch

def ranking_candidates(case,rotation=0):
    q=application_view(case,rotation);offset=(case['base_rotation']+rotation)%4
    choices=case['proposals'][offset:]+case['proposals'][:offset]
    assert len(set(choices))==4 and choices.count(q['target'])==1
    return q,choices,choices.index(q['target'])

def ranking_functions():
    return {f.__name__:hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
            for f in (qwen3vl_choice_nll,qwen3vl_listwise_choice_ce,ranking_candidates,training_jobs)}

def ranking_register():
    old=load_json(DATA/'registration.json');p=load_json(DATA/'training-payload.json')
    assert digest(old)=='df20224bf3c3ba231e3f90be4dea03154c5c9f39020a17a98f01ff435f15c0ac'
    assert digest(p)==old['training_digest'];RANK_DATA.mkdir(exist_ok=True)
    reg=dict(plan='prefeval-rgb-semantic-ranking-supervision-06',parent_registration_digest=digest(old),
        training_digest=old['training_digest'],evaluation_digest=old['evaluation_digest'],source_files=old['files'],
        endpoints={sid:t['initial'] for sid,t in p['targets'].items()},functions=ranking_functions(),
        schedule_digest=digest(p['schedule']),optimizer=p['optimizer'],additional_updates=20480,
        gradient_forwards={'A':22528,'B':54784,'total':77312},logical_slot_jobs=45056,
        calibration=dict(decisions=2688,max_candidate_forwards=10752,text_min=1210,text_total=1344,
            macro_gain_min=.10,each_contrast_pairs_min=12,each_contrast_pairs_total=16),
        final=dict(generations=3408,ranking_decisions=1848,max_candidate_forwards=7392),
        progression=dict(recovery={'K1':[18,20],'K2':[3,4],'K4':[9,12]},mcq_min=68,reserved_ranking_min=135,
                         mcq_macro_gain=.10,reserved_ranking_macro_gain=.10,contrast_pairs_min=6,each_contrast_min=1),
        objective='temperature1 listwise CE on negative mean full-action token NLL; no EOS on application; recovery answer+EOS unchanged',
        historical_status='Alternative objective proposed after Plan05 generative feasibility failure; Plan05 stays failed',
        no_new_prompts=True,no_reserved_calls_before_all_endpoints=True,comparison='fixed updates; unequal compute; no efficiency claim')
    write_frozen(RANK_DATA/'registration.json',reg);print(json.dumps(dict(registration_digest=digest(reg))))

def ranking_load():
    reg=load_json(RANK_DATA/'registration.json');old=load_json(DATA/'registration.json')
    assert digest(old)==reg['parent_registration_digest'] and ranking_functions()==reg['functions']
    p=load_json(DATA/'training-payload.json');assert digest(p)==reg['training_digest']
    return reg,p

def query_prompt(processor,query):
    return processor.apply_chat_template([{'role':'user','content':[{'type':'image'},{'type':'text','text':query}]}],
                                         tokenize=False,add_generation_prompt=True)

def ranking_read(reader,processor,image,query,choices,device):
    prompt=query_prompt(processor,query);target_counts=[]
    for choice in choices:
        _,ids=_joint_prompt_target_tokenization(processor,prompt,choice);target_counts.append(int(ids.numel()))
    processor.begin_capture()
    result=qwen3vl_choice_nll(model=reader,processor=processor,image=image,query=query,choices=choices,
        device=device,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
    calls=processor.end_capture();assert len(calls)==4
    for call,choice in zip(calls,choices):assert call['text']==prompt+choice
    scores=[-x for x in result.mean_nll];assert all(torch.isfinite(torch.tensor(scores)))
    return dict(scores=scores,candidate_token_counts=target_counts,model_prompt=prompt,processor_calls=calls)

def ranking_calibrate(a):
    reg,p=ranking_load();processor,reader,_,versions,bindings=models(a)
    assignment=sorted(p['targets'])[a.shard::a.shards];out=a.output/'calibration'/f'shard-{a.shard}'
    out.mkdir(parents=True,exist_ok=False)
    frame=query_prompt(processor,'__REGISTERED_QUERY__')
    write_frozen(out/'identity.json',dict(**identity(a,bindings,reg,assignment),prompt_frame=frame,
        chat_template_digest=digest(processor.chat_template)))
    Image.new('RGB',(1024,1024),(128,128,128)).save(out/'blank.png');blank=read_png(out/'blank.png')
    started=time.monotonic();cache={};n=0;actual=0
    for sid in assignment:
        t=p['targets'][sid]
        for case in t['applications']:
            for rotation in range(4):
                q,choices,gold=ranking_candidates(case,rotation)
                for condition in ('text','blank'):
                    query=t['text_prefix']+q['query'] if condition=='text' else q['query']
                    key=digest(dict(query=query,choices=choices,png=file_sha(out/'blank.png')))
                    reuse=key in cache
                    if not reuse:
                        cache[key]=ranking_read(reader,processor,blank,query,choices,a.device);actual+=4
                    scored=cache[key];assert scored['model_prompt']==frame.replace('__REGISTERED_QUERY__',query)
                    margin=scored['scores'][gold]-max(s for j,s in enumerate(scored['scores']) if j!=gold)
                    append(out/'scores.jsonl',dict(target=sid,condition=condition,query=q,reader_query=query,
                        choices=choices,gold_index=gold,input_identity=key,reused=reuse,score=scored,
                        gold_margin=margin,unique_correct=margin>0,actual_candidate_forwards=0 if reuse else 4))
                    n+=1
        print(json.dumps(dict(target=sid,decisions=n,actual_candidate_forwards=actual)),flush=True)
    frozen(versions)
    write_frozen(out/'complete.json',dict(decisions=n,candidate_forwards=actual,seconds=time.monotonic()-started,
        optimizer_updates=0,generated_answers=0,processed_input_tokens=processor.total_input_tokens,
        blank_sha=file_sha(out/'blank.png'),scores_sha=file_sha(out/'scores.jsonl')))

def models(a,with_vae=False):
    configure_strict_cuda_determinism(0)
    from scripts.experiments.refine_historical_writer_targets import snapshot_bindings,M
    assert a.reader.resolve()==M/'Qwen3-VL-4B-Instruct' and a.base.resolve()==M/'DreamLite-base-a9a0f15-20260907'
    bindings=snapshot_bindings();processor,reader=load_reader(a.reader,a.device);processor=TokenCountProcessor(processor);vae=None
    if with_vae:
        from diffusers import AutoencoderTiny
        vae=AutoencoderTiny.from_pretrained(str(a.base),subfolder='vae',local_files_only=True,torch_dtype=torch.float32).to(a.device)
        vae.eval().requires_grad_(False);validate_vae(vae)
    versions=[(p,int(p._version)) for module in ((reader,vae) if vae is not None else (reader,)) for p in module.parameters()]
    return processor,reader,vae,versions,bindings

def frozen(versions):
    for p,v in versions:assert int(p._version)==v and not p.requires_grad and p.grad is None

def identity(a,bindings,registration,assignment):
    return dict(code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        registration_digest=digest(registration),mode=a.mode,assignment=assignment,snapshots=bindings)

def feasibility(a,reg):
    # Intentionally loads no evaluation payload or reserved scenario file.
    payload=load_json(DATA/'training-payload.json');assert digest(payload)==reg['training_digest']
    processor,reader,_,versions,bindings=models(a);assignment=sorted(payload['targets'])[a.shard::a.shards]
    out=a.output/'feasibility'/f'shard-{a.shard}';out.mkdir(parents=True,exist_ok=False)
    (out/'identity.json').write_text(json.dumps(identity(a,bindings,reg,assignment),indent=2))
    Image.new('RGB',(1024,1024),(128,128,128)).save(out/'blank.png')
    blank=read_png(out/'blank.png');started=time.monotonic();n=0
    for sid in assignment:
        t=payload['targets'][sid]
        for case in t['applications']:
            q=application_view(case)
            if len(processor.tokenizer.encode(q['target'],add_special_tokens=False))+8>=128:raise ValueError('Application target exceeds fixed generation budget')
            for condition in ('text','blank'):
                query=t['text_prefix']+q['query'] if condition=='text' else q['query']
                g=generate(reader,processor,blank,query,a.device)
                append(out/'reads.jsonl',dict(target=sid,condition=condition,query=q,generation=g,score=score_generation(g,q)))
                n+=1
        print(json.dumps(dict(target=sid,reads=n)),flush=True)
    frozen(versions);(out/'complete.json').write_text(json.dumps(dict(reads=n,seconds=time.monotonic()-started,optimizer_updates=0,blank_sha=file_sha(out/'blank.png'))))

def train(a,reg):
    payload=load_json(DATA/'training-payload.json');assert digest(payload)==reg['training_digest']
    gate=load_json(a.output/'feasibility-verified.json')
    assert gate['registration_digest']==digest(reg) and gate['passed'] and gate['text'][1]==336 and gate['text'][0]>=303
    processor,reader,vae,versions,bindings=models(a,True);termination=assistant_termination_contract(reader,processor)
    all_targets=sorted(payload['targets']);assignment=all_targets[a.shard::a.shards]
    write_frozen(a.output/f'train-identity-{a.shard}.json',identity(a,bindings,reg,assignment))
    for sid in assignment:
        t=payload['targets'][sid];src=a.source/sid
        for name,h in t['initial']['artifacts'].items():assert file_sha(src/name)==h
        init=torch.load(src/'latent.pt',map_location=a.device,weights_only=True)
        assert tensor_sha(init)==t['initial']['latent_tensor_sha']
        order=('A','B') if all_targets.index(sid)%2==0 else ('B','A')
        for arm in order:
            out=a.output/'training'/arm/sid;out.mkdir(parents=True,exist_ok=False)
            oracle=VAELatentOracle(vae=vae,initial_latent=init,compute_dtype=torch.float32)
            opt=torch.optim.Adam([oracle.latent_fp32],lr=.05,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            assert not opt.state and tensor_sha(oracle.latent_fp32)==t['initial']['latent_tensor_sha']
            torch.save(init.detach().cpu(),out/'initial-latent.pt');started=time.monotonic();forwards=tokens=0
            for draw in payload['schedule']:
                opt.zero_grad(set_to_none=True);pixels=oracle.image();jobs=training_jobs(t,draw,arm);losses=[]
                for j,(q,weight) in enumerate(jobs):
                    ce=qwen3vl_answer_eos_ce(model=reader,processor=processor,image=pixels[0],query=q['query'],target=q['target'],
                        device=a.device,termination=termination,lambda_eos=1.,require_image_grad=True,
                        reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                    loss=ce.loss*weight
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward(retain_graph=j+1<len(jobs));forwards+=1
                    prompt_tokens=len(processor.tokenizer.encode(q['query'],add_special_tokens=False))
                    tokens+=processor.last_input_tokens
                    losses.append(dict(query_id=q['id'],query_digest=digest(q),scope=q['scope'],kind=q['kind'],weight=weight,
                        answer_ce=float(ce.answer_loss.detach()),eos_ce=float(ce.eos_loss.detach()),answer_tokens=ce.answer_token_count,
                        query_text_tokens=prompt_tokens,processed_input_tokens=processor.last_input_tokens))
                grad=oracle.latent_fp32.grad
                assert grad is not None and torch.isfinite(grad).all() and torch.any(grad!=0)
                opt.step();assert torch.isfinite(oracle.latent_fp32).all()
                append(out/'optimization.jsonl',dict(step=draw['step'],losses=losses,seconds=time.monotonic()-started))
                if (draw['step']+1)%64==0:print(json.dumps(dict(target=sid,arm=arm,step=draw['step']+1,loss=sum(x['weight']*(x['answer_ce']+x['eos_ce']) for x in losses))),flush=True)
            torch.save(oracle.latent_fp32.detach().cpu(),out/'latent.pt');torch.save(opt.state_dict(),out/'optimizer.pt')
            with torch.no_grad():image=oracle.image().detach().cpu()[0]
            array=image.mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy();Image.fromarray(array).save(out/'memory.png')
            png=read_png(out/'memory.png');assert torch.equal(png.mul(255).round().byte().permute(1,2,0),torch.from_numpy(array))
            frozen(versions)
            for name,h in t['initial']['artifacts'].items():assert file_sha(src/name)==h
            receipt=dict(target=sid,arm=arm,additional_updates=256,inherited_updates=256,slot_forwards=forwards,
                processed_input_tokens_including_template_vision_and_target=tokens,seconds=time.monotonic()-started,
                initial_tensor_sha=tensor_sha(init),final_tensor_sha=tensor_sha(oracle.latent_fp32),
                artifacts={f:file_sha(out/f) for f in ('initial-latent.pt','latent.pt','optimizer.pt','optimization.jsonl','memory.png')})
            write_frozen(out/'complete.json',receipt);print(json.dumps(receipt),flush=True)
            del opt,oracle
    write_frozen(a.output/f'train-complete-{a.shard}.json',dict(targets=assignment))

def evaluate(a,reg):
    # Load probes only after every endpoint in both arms has been frozen.
    train_payload=load_json(DATA/'training-payload.json');assert digest(train_payload)==reg['training_digest']
    for sid in train_payload['targets']:
        for arm in ('A','B'):
            done=load_json(a.output/'training'/arm/sid/'complete.json');assert done['additional_updates']==256
    payload=load_json(DATA/'evaluation-payload.json');assert digest(payload)==reg['evaluation_digest']
    processor,reader,_,versions,bindings=models(a);assignment=sorted(payload['targets'])[a.shard::a.shards]
    out=a.output/'evaluation'/f'shard-{a.shard}';out.mkdir(parents=True,exist_ok=False)
    write_frozen(out/'identity.json',identity(a,bindings,reg,assignment));started=time.monotonic();n=0
    blank=torch.full((3,1024,1024),128/255,dtype=torch.float32)
    for sid in assignment:
        t=payload['targets'][sid]
        for condition in ('A','B','parent','text','blank'):
            if condition in ('A','B'):
                path=a.output/'training'/condition/sid/'memory.png';receipt=load_json(path.parent/'complete.json')
                assert file_sha(path)==receipt['artifacts']['memory.png'];image=read_png(path)
                panels=['recovery_training','qualification','application_training','mcq','application_reserved']
            elif condition=='parent':
                path=a.source/sid/'memory.png';assert file_sha(path)==t['parent']['artifacts']['memory.png'];image=read_png(path)
                panels=['application_training','application_reserved']
            else:image=blank;path=None;panels=['application_reserved']
            for panel in panels:
                for q in t[panel]:
                    query=t['text_prefix']+q['query'] if condition=='text' else q['query']
                    g=generate(reader,processor,image,query,a.device)
                    append(out/'reads.jsonl',dict(target=sid,capacity=len(t['state']),condition=condition,panel=panel,
                        query=q,generation=g,score=score_generation(g,q,panel=='mcq'),png_sha=file_sha(path) if path else None))
                    n+=1
            print(json.dumps(dict(target=sid,condition=condition,reads=n)),flush=True)
        frozen(versions)
    write_frozen(out/'complete.json',dict(reads=n,seconds=time.monotonic()-started))

def trial_train(a,reg):
    ranking=True;steps=reg["steps"]
    payload=load_json(DATA/'training-payload.json');assert digest(payload)==reg['training_digest']
    gate_path=a.source.parent.parent/'semantic-ranking-v1-run/calibration-verified.json'
    assert file_sha(gate_path)==reg['calibration_verified_sha']
    assert not load_json(gate_path)['passed'] and reg['allocation']=='exploratory learning trial; historical gates remain failed'
    processor,reader,vae,versions,bindings=models(a,True);termination=assistant_termination_contract(reader,processor)
    all_targets=sorted(payload['targets']);assignment=all_targets[a.shard::a.shards]
    write_frozen(a.output/f'train-identity-{a.shard}.json',dict(**identity(a,bindings,reg,assignment),gate_sha=file_sha(gate_path)))
    for sid in assignment:
        t=payload['targets'][sid];src=a.source/sid
        for name,h in t['initial']['artifacts'].items():assert file_sha(src/name)==h
        init=torch.load(src/'latent.pt',map_location=a.device,weights_only=True)
        assert tensor_sha(init)==t['initial']['latent_tensor_sha']
        order=('A','B') if all_targets.index(sid)%2==0 else ('B','A')
        for arm in order:
            out=a.output/'training'/arm/sid;out.mkdir(parents=True,exist_ok=False)
            oracle=VAELatentOracle(vae=vae,initial_latent=init,compute_dtype=torch.float32)
            opt=torch.optim.Adam([oracle.latent_fp32],lr=.05,betas=(.9,.999),eps=1e-8,weight_decay=0.)
            assert not opt.state and tensor_sha(oracle.latent_fp32)==t['initial']['latent_tensor_sha']
            torch.save(init.detach().cpu(),out/'initial-latent.pt');started=time.monotonic();forwards=tokens=reader_forwards=0
            for draw in payload['schedule'][:steps]:
                opt.zero_grad(set_to_none=True);pixels=oracle.image();jobs=training_jobs(t,draw,arm);losses=[]
                for j,(q,weight) in enumerate(jobs):
                    calls_before=processor.total_calls;tokens_before=processor.total_input_tokens
                    if ranking and q['kind']=='application':
                        case=next(c for c in t['applications'] if c['scope']==q['scope'] and c['scenario']==q['scenario'])
                        rq,choices,gold=ranking_candidates(case,q['rotation']);assert rq==q
                        ce=qwen3vl_listwise_choice_ce(model=reader,processor=processor,image=pixels[0],query=q['query'],
                            choices=choices,target_index=gold,device=a.device,require_image_grad=True,
                            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                        details=dict(loss_family='listwise_action_ranking',candidate_scores=ce.choice_logits.detach().tolist(),
                            candidate_token_counts=list(ce.choice_token_counts),choices=choices,gold_index=gold)
                    else:
                        ce=qwen3vl_answer_eos_ce(model=reader,processor=processor,image=pixels[0],query=q['query'],target=q['target'],
                            device=a.device,termination=termination,lambda_eos=1.,require_image_grad=True,
                            reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,deterministic_ce=True)
                        details=dict(loss_family='answer_plus_eos',answer_ce=float(ce.answer_loss.detach()),
                                     eos_ce=float(ce.eos_loss.detach()),answer_tokens=ce.answer_token_count)
                    loss=ce.loss*weight
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward(retain_graph=j+1<len(jobs));forwards+=1
                    prompt_tokens=len(processor.tokenizer.encode(q['query'],add_special_tokens=False))
                    actual_calls=processor.total_calls-calls_before;actual_tokens=processor.total_input_tokens-tokens_before
                    assert actual_calls==(4 if ranking and q['kind']=='application' else 1)
                    reader_forwards+=actual_calls;tokens+=actual_tokens
                    losses.append(dict(query_id=q['id'],query_digest=digest(q),scope=q['scope'],kind=q['kind'],weight=weight,
                        slot_loss=float(ce.loss.detach()),reader_forwards=actual_calls,
                        query_text_tokens=prompt_tokens,processed_input_tokens=actual_tokens,**details))
                grad=oracle.latent_fp32.grad
                assert grad is not None and torch.isfinite(grad).all() and torch.any(grad!=0)
                opt.step();assert torch.isfinite(oracle.latent_fp32).all()
                if (draw['step']+1)%32==0:
                    checkpoint=dict(next_step=draw['step']+1,latent=oracle.latent_fp32.detach().cpu(),optimizer=opt.state_dict(),
                                    rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(),registration_digest=digest(reg))
                    torch.save(checkpoint,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(out/'checkpoint.pt')
                append(out/'optimization.jsonl',dict(step=draw['step'],losses=losses,seconds=time.monotonic()-started))
                if (draw['step']+1)%64==0:print(json.dumps(dict(target=sid,arm=arm,step=draw['step']+1,loss=sum(x['weight']*x['slot_loss'] for x in losses))),flush=True)
            torch.save(oracle.latent_fp32.detach().cpu(),out/'latent.pt');torch.save(opt.state_dict(),out/'optimizer.pt')
            with torch.no_grad():image=oracle.image().detach().cpu()[0]
            array=image.mul(255).round().clamp(0,255).byte().permute(1,2,0).numpy();Image.fromarray(array).save(out/'memory.png')
            png=read_png(out/'memory.png');assert torch.equal(png.mul(255).round().byte().permute(1,2,0),torch.from_numpy(array))
            frozen(versions)
            for name,h in t['initial']['artifacts'].items():assert file_sha(src/name)==h
            receipt=dict(target=sid,arm=arm,additional_updates=steps,inherited_updates=256,slot_forwards=forwards,
                reader_forwards=reader_forwards,protocol='ranking-learning-trial-v1',gate_sha=file_sha(gate_path),
                processed_input_tokens_including_template_vision_and_target=tokens,seconds=time.monotonic()-started,
                initial_tensor_sha=tensor_sha(init),final_tensor_sha=tensor_sha(oracle.latent_fp32),
                artifacts={f:file_sha(out/f) for f in ('initial-latent.pt','latent.pt','optimizer.pt','optimization.jsonl','memory.png','checkpoint.pt')})
            write_frozen(out/'complete.json',receipt);print(json.dumps(receipt),flush=True)
            del opt,oracle
    write_frozen(a.output/f'train-complete-{a.shard}.json',dict(targets=assignment))

def trial_evaluate(a,reg):
    ranking=True
    # Load probes only after every endpoint in both arms has been frozen.
    train_payload=load_json(DATA/'training-payload.json');assert digest(train_payload)==reg['training_digest']
    endpoint_bindings={}
    for sid in train_payload['targets']:
        for arm in ('A','B'):
            folder=a.output/'training'/arm/sid
            done=load_json(folder/'complete.json');assert done['additional_updates']==reg['steps']
            assert file_sha(folder/'memory.png')==done['artifacts']['memory.png']
            endpoint_bindings[f'{arm}/{sid}']=file_sha(folder/'complete.json')
    payload=load_json(DATA/'evaluation-payload.json');assert digest(payload)==reg['evaluation_digest']
    processor,reader,_,versions,bindings=models(a);assignment=sorted(payload['targets'])[a.shard::a.shards]
    out=a.output/'evaluation'/f'shard-{a.shard}';out.mkdir(parents=True,exist_ok=False)
    extra=dict(prompt_frame=query_prompt(processor,'__REGISTERED_QUERY__'),chat_template_digest=digest(processor.chat_template),
               frozen_endpoints=endpoint_bindings) if ranking else {}
    write_frozen(out/'identity.json',dict(**identity(a,bindings,reg,assignment),**extra));started=time.monotonic();n=rank_n=rank_forwards=0
    blank=torch.full((3,1024,1024),128/255,dtype=torch.float32)
    if ranking:
        Image.new('RGB',(1024,1024),(128,128,128)).save(out/'blank.png');blank=read_png(out/'blank.png')
        scenarios={}
        for filename in ('training-scenarios.json','reserved-scenarios.json'):
            for cases in load_json(DATA/filename).values():
                for c in cases:scenarios[(c['id'],c['scope'],c['value_id'])]=c
        rank_cache={}
    for sid in assignment:
        t=payload['targets'][sid]
        for condition in ('A','B','parent','text','blank'):
            if condition in ('A','B'):
                path=a.output/'training'/condition/sid/'memory.png';receipt=load_json(path.parent/'complete.json')
                assert file_sha(path)==receipt['artifacts']['memory.png'];image=read_png(path)
                panels=['recovery_training','qualification','application_training','mcq','application_reserved']
            elif condition=='parent':
                path=a.source/sid/'memory.png';assert file_sha(path)==t['parent']['artifacts']['memory.png'];image=read_png(path)
                panels=['application_training','application_reserved']
            else:image=blank;path=None;panels=['application_reserved']
            for panel in panels:
                for q in t[panel]:
                    query=t['text_prefix']+q['query'] if condition=='text' else q['query']
                    if ranking:processor.begin_capture()
                    g=generate(reader,processor,image,query,a.device)
                    proof={}
                    if ranking:
                        captured=processor.end_capture();assert len(captured)==1
                        assert captured[0]['text']==extra['prompt_frame'].replace('__REGISTERED_QUERY__',query)
                        assert captured[0]['input_ids_digest']==digest([g['input_token_ids']])
                        proof=dict(reader_query=query,processor_input=captured[0])
                    append(out/'reads.jsonl',dict(target=sid,capacity=len(t['state']),condition=condition,panel=panel,
                        query=q,generation=g,score=score_generation(g,q,panel=='mcq'),png_sha=file_sha(path) if path else None,**proof))
                    n+=1
                    if ranking and panel in ('application_training','application_reserved'):
                        case=scenarios[(q['case_id'],q['scope'],q['value_id'])];rq,choices,gold=ranking_candidates(case,q['rotation']);assert rq==q
                        png_sha=file_sha(path if path else out/'blank.png');key=digest(dict(query=query,choices=choices,png=png_sha))
                        reuse=key in rank_cache
                        if not reuse:rank_cache[key]=ranking_read(reader,processor,image,query,choices,a.device);rank_forwards+=4
                        scored=rank_cache[key];margin=scored['scores'][gold]-max(s for j,s in enumerate(scored['scores']) if j!=gold)
                        append(out/'ranking.jsonl',dict(target=sid,condition=condition,panel=panel,query=q,reader_query=query,
                            choices=choices,gold_index=gold,input_identity=key,reused=reuse,score=scored,png_sha=png_sha,
                            gold_margin=margin,unique_correct=margin>0,actual_candidate_forwards=0 if reuse else 4));rank_n+=1
            print(json.dumps(dict(target=sid,condition=condition,reads=n)),flush=True)
        for condition in ('A','B','parent'):
            path=(a.output/'training'/condition/sid/'memory.png' if condition in ('A','B') else a.source/sid/'memory.png')
            image=read_png(path);png_sha=file_sha(path)
            for q0 in t['application_training']:
                if q0['value_id'] not in reg['contrast_values']:continue
                case=scenarios[(q0['case_id'],q0['scope'],q0['value_id'])]
                for rotation in (1,2,3):
                    q,choices,gold=ranking_candidates(case,rotation);query=q['query']
                    key=digest(dict(query=query,choices=choices,png=png_sha));reuse=key in rank_cache
                    if not reuse:rank_cache[key]=ranking_read(reader,processor,image,query,choices,a.device);rank_forwards+=4
                    scored=rank_cache[key];margin=scored['scores'][gold]-max(s for j,s in enumerate(scored['scores']) if j!=gold)
                    append(out/'ranking.jsonl',dict(target=sid,condition=condition,panel='application_training_rotations',query=q,
                        reader_query=query,choices=choices,gold_index=gold,input_identity=key,reused=reuse,score=scored,png_sha=png_sha,
                        gold_margin=margin,unique_correct=margin>0,actual_candidate_forwards=0 if reuse else 4));rank_n+=1
        frozen(versions)
    write_frozen(out/'complete.json',dict(reads=n,seconds=time.monotonic()-started,
        ranking_decisions=rank_n,ranking_candidate_forwards=rank_forwards,processed_input_tokens=processor.total_input_tokens,
        reads_sha=file_sha(out/'reads.jsonl'),ranking_sha=file_sha(out/'ranking.jsonl'),blank_sha=file_sha(out/'blank.png')))

def main(a):
    if a.mode=='trial-register':return trial_register()
    if a.mode in ('trial-train','trial-evaluate'):
        reg,_=trial_load();a.output.mkdir(parents=True,exist_ok=True)
        return {'trial-train':trial_train,'trial-evaluate':trial_evaluate}[a.mode](a,reg)
    if a.mode=='register':return register()
    if a.mode=='ranking-register':return ranking_register()
    if a.mode=='ranking-calibrate':return ranking_calibrate(a)
    reg=load_json(DATA/'registration.json')
    a.output.mkdir(parents=True,exist_ok=True)
    return {'feasibility':feasibility,'train':train,'evaluate':evaluate}[a.mode](a,reg)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','feasibility','train','evaluate','ranking-register','ranking-calibrate','trial-register','trial-train','trial-evaluate'],required=True)
    ap.add_argument('--source',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--base',type=Path);ap.add_argument('--reader',type=Path)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--device',default='cuda:0')
    main(ap.parse_args())
