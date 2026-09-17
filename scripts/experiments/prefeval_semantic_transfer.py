"""A fixed semantic-query mixture versus recovery-only teacher continuation."""
from __future__ import annotations
import argparse,json,hashlib,time,sys,subprocess
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

REPORT=ROOT/'reports/prefeval-rgb-20260917';DATA=REPORT/'semantic-transfer-v1'

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
    def __init__(self,processor):self.processor=processor;self.last_input_tokens=0
    def __getattr__(self,name):return getattr(self.processor,name)
    def __call__(self,**kwargs):
        batch=self.processor(**kwargs);self.last_input_tokens=int(batch['input_ids'].numel());return batch

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

def main(a):
    if a.mode=='register':return register()
    reg=load_json(DATA/'registration.json')
    a.output.mkdir(parents=True,exist_ok=True)
    return {'feasibility':feasibility,'train':train,'evaluate':evaluate}[a.mode](a,reg)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['register','feasibility','train','evaluate'],required=True)
    ap.add_argument('--source',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--base',type=Path);ap.add_argument('--reader',type=Path)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);ap.add_argument('--device',default='cuda:0')
    main(ap.parse_args())
