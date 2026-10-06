"""Paired native-FM continuation with a single role-layout augmentation factor."""
import argparse
from collections import Counter
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.inspire import run_context_layout as l
from scripts.inspire.run_writer_readout import execute,accrued_seconds
from scripts.experiments.prefeval_route_functional import read,read_lines,sha,save_once
from scripts.inspire.run_prompt_matching_parallel import write_json
p=l.p;e=l.e;RUN=l.RUN
PREVIOUS=RUN/'context-layout-v1'
SOURCE=l.SOURCE/'train'
RESUME_SHA='8f5bd84c06c91611a3a38aee7967b05ceaba136e8f331dfca7b74c5f1636e5f3'
PLAN=ROOT/'reports/context-coverage-20261006/CONTEXT_LAYOUT_AUG_PLAN.md'
ARMS=('canonical','augmented')


def training_layout(cycle,arm):
    if arm not in ARMS:raise ValueError('Foreign arm')
    return cycle%2 if arm=='augmented' else 0


def evaluation_text(exchange,style):
    # Validate role order/content even for the canonical native formatter.
    l.layout_text(exchange)
    if style=='canonical':return p.event_text(exchange)
    if style=='markdown':return l.layout_text(exchange)
    if style=='xml':return '\n'.join('<'+m['role']+'>\n'+m['content']+'\n</'+m['role']+'>' for m in exchange)
    raise ValueError('Foreign evaluation format')


def continuation_manifest(original,arm):
    if original['steps']!=256 or original['implementation_sha256']!=sha(ROOT/'scripts/experiments/prefeval_k1_writer.py'):
        raise ValueError('Native256 source required')
    training_layout(32,arm)
    return dict(original,steps=384,implementation_sha256=sha(Path(__file__)),
        layout_source_sha256=sha(ROOT/'scripts/inspire/run_context_layout.py'),
        continuation_resume_sha256=RESUME_SHA,layout_arm=arm,
        layout_policy='cycle parity canonical/markdown; XML excluded' if arm=='augmented' else 'canonical only',
        layout_plan_sha256=sha(PLAN))


def migrated_payload(source,arm):
    if source['schema_version']!=1 or source['optimizer_step']!=256 or source['episode_cursor']!=1024 or source['epoch']!=0:
        raise ValueError('Incomplete source cursor')
    if not {'python','numpy','torch_cpu','torch_cuda'}<=source['rng_state'].keys() or not source['optimizer']['state']:
        raise ValueError('Incomplete optimizer/RNG state')
    if {int(x['step']) for x in source['optimizer']['state'].values()}!={256}:raise ValueError('Adam cursor mismatch')
    manifest=dict(source['manifest'],steps=258) if arm=='native' else continuation_manifest(source['manifest'],arm)
    return dict(source,manifest=manifest)


def budget(output):
    _,prior,layout=l.remaining_seconds(PREVIOUS)
    previous=prior+layout;current=accrued_seconds(output)
    return min(.75*3600-current,16*3600-previous-current),previous,current


def verify_source():
    if read(PREVIOUS/'status.json')['status']!='completed' or not read(PREVIOUS/'audit-0650.json')['exact_report_recomputation']:
        raise ValueError('Require audited completed layout result')
    l.report(PREVIOUS)
    if sha(SOURCE/'resume.pt')!=RESUME_SHA or sha(SOURCE/'checkpoint-final.pt')!=l.CHECKPOINT_SHA:raise ValueError('Source weights changed')
    return {str(path):sha(path) for path in (SOURCE/'resume.pt',SOURCE/'checkpoint-final.pt',SOURCE/'manifest.json',SOURCE/'optimization.jsonl',PREVIOUS/'audit-0650.json',PREVIOUS/'comparison.json',PLAN,p.ALL_IDS)}


def prepare(output):
    import torch
    from scripts.experiments.prefeval_k1_teacher import atomic_save
    source=torch.load(SOURCE/'resume.pt',map_location='cpu',weights_only=False)
    final=torch.load(SOURCE/'checkpoint-final.pt',map_location='cpu',weights_only=True)
    if not e.equivalent(source['trainable_state'],final['trainable_state']) or source['manifest']!=read(SOURCE/'manifest.json'):raise ValueError('Source resume/final mismatch')
    for arm in ('native',*ARMS):
        dest=output/arm;dest.mkdir(exist_ok=True)
        payload=migrated_payload(source,arm);initial=dest/'resume-initial256.pt'
        if initial.exists():
            if not e.equivalent(torch.load(initial,map_location='cpu',weights_only=False),payload):raise ValueError('Initial state changed')
        else:
            if (dest/'resume.pt').exists() or (dest/'optimization.jsonl').exists():raise ValueError('Unclaimed partial training')
            atomic_save(initial,payload)
            if not e.equivalent(torch.load(initial,map_location='cpu',weights_only=False),payload):raise ValueError('Migration changed state')
            shutil.copyfile(initial,dest/'resume.pt');shutil.copyfile(SOURCE/'optimization.jsonl',dest/'optimization.jsonl')
        save_once(dest/'migration.json',dict(source_resume_sha256=RESUME_SHA,initial_resume_sha256=sha(initial),
            source_log_sha256=sha(SOURCE/'optimization.jsonl'),original_manifest=source['manifest'],new_manifest=payload['manifest'],
            allowed_change='Only manifest updates; all weights, Adam, RNG and cursor unchanged'))
    return {arm:sha(output/arm/'resume-initial256.pt') for arm in ('native',*ARMS)}


def train(args):
    import torch
    from scripts.experiments import prefeval_k1_writer as n
    from vision_memory.training.checkpoint import load_training_checkpoint,save_training_checkpoint
    from vision_memory.repro import configure_strict_cuda_determinism
    configure_strict_cuda_determinism(0);torch.set_num_threads(1)
    args.device='cuda:0';args.checkpoint=SOURCE/'checkpoint-final.pt'
    frozen=read(args.output/'source.json')
    for path,h in frozen.items():
        if sha(Path(path))!=h:raise ValueError('Source identity changed')
    dest=args.output/args.arm;manifest=continuation_manifest(read(SOURCE/'manifest.json'),args.arm)
    save_once(dest/'manifest.json',manifest)
    pipe=n.load_pipe(args);rows=p.population()[0];targets={};cache={}
    for row in rows:
        pid=row['base_pair_id'];folder=p.SOURCE/'bank'/pid.replace(':','_');done=read(folder/'complete.json')
        binding=n.validate_teacher_manifest(done,arm='B',supervision='prompt_matching',steps=288)
        if binding!=manifest['teacher_binding_hashes'][pid] or sha(folder/'latent.pt')!=manifest['targets'][pid]:raise ValueError('Teacher changed')
        targets[pid]=torch.load(folder/'latent.pt',map_location='cpu',weights_only=True)
        for style in (0,1):cache[pid,style]=n.cache_condition(pipe,None,evaluation_text(row['history'][:2],('canonical','markdown')[style]),args.device)
    predictor=n.DifferentiableDreamLiteMobileSampler.from_pipeline(pipe,checkpoint_unet=False)
    optimizer=torch.optim.AdamW(pipe.unet.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    saved=load_training_checkpoint(dest/'resume.pt',trainable_module=pipe.unet,optimizer=optimizer,expected_manifest=manifest)
    first=saved['optimizer_step']
    logs=read_lines(dest/'optimization.jsonl')
    if len(logs)!=first or [x['step'] for x in logs]!=list(range(1,first+1)):raise ValueError('Uncommitted log tail; recovery audit required')
    if first not in (256,258) or args.stop_step not in (258,384) or args.stop_step<=first:raise ValueError('Unexpected production cursor')
    started=time.monotonic()
    for step in range(first,args.stop_step):
        optimizer.zero_grad(set_to_none=True);values=[]
        for micro in range(4):
            draw=step*4+micro;cycle,offset=divmod(draw,len(rows))
            order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(n.stable_seed(20260924,'order',cycle))).tolist()
            row=rows[order[offset]];pid=row['base_pair_id'];style=training_layout(cycle,args.arm);c=cache[pid,style]
            source=c['source'].to(args.device);target=targets[pid].to(args.device)
            sigma=float(torch.rand((),generator=torch.Generator().manual_seed(n.stable_seed(20260924,'sigma',draw))))
            generator=torch.Generator(device=args.device).manual_seed(n.stable_seed(20260924,'noise',draw))
            noise=torch.randn(target.shape,generator=generator,device=args.device,dtype=target.dtype)
            state,velocity=n.official_flow_bridge(noise,target,sigma)
            pred=n.predict_velocity(predictor,state,source,sigma,c['embeds'].to(args.device),c['mask'].to(args.device),integer_timestep=True)
            loss=(pred.float()-velocity.float()).square().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite FM loss')
            (loss/4).backward();values.append(dict(pair_id=pid,position=0,sigma=sigma,mse=float(loss.detach()),layout=style))
        norm=torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(),1.,error_if_nonfinite=True)
        if not math.isfinite(float(norm)) or float(norm)<=0:raise ValueError('Invalid gradient')
        optimizer.step();n.append(dest/'optimization.jsonl',dict(step=step+1,draws=values,grad_norm=float(norm),seconds=time.monotonic()-started))
        if (step+1)%32==0 or step+1==args.stop_step:
            save_training_checkpoint(dest/'resume.pt',trainable_module=pipe.unet,optimizer=optimizer,epoch=0,
                episode_cursor=(step+1)*4,optimizer_step=step+1,manifest=manifest)
            print(json.dumps(dict(arm=args.arm,step=step+1,seconds=time.monotonic()-started)),flush=True)
    if args.stop_step==384:
        n.save_inference_checkpoint(dest/'checkpoint-final.pt',pipe,384,manifest)
        save_once(dest/'complete.json',dict(steps=384,checkpoint_sha256=sha(dest/'checkpoint-final.pt')))
    else:
        shutil.copyfile(dest/'resume.pt',dest/'resume-step258.pt')


def verify_parity(output):
    import torch
    reference=torch.load(output/'native/resume.pt',map_location='cpu',weights_only=False)
    if reference['optimizer_step']!=258:raise ValueError('Native probe incomplete')
    logs=read_lines(output/'native/optimization.jsonl')[256:]
    if len(logs)!=2:raise ValueError('Native probe denominator')
    result={}
    for arm in ARMS:
        path=output/arm/'resume-step258.pt';value=torch.load(path,map_location='cpu',weights_only=False)
        for key in reference:
            if key!='manifest' and not e.equivalent(reference[key],value[key]):raise ValueError('Native exact parity failed: '+arm+' '+key)
        new=read_lines(output/arm/'optimization.jsonl')[256:258]
        if len(new)!=2:raise ValueError('New probe denominator')
        for a,b in zip(logs,new):
            if a['step']!=b['step'] or a['grad_norm']!=b['grad_norm'] or a['draws']!=[{k:v for k,v in x.items() if k!='layout'} for x in b['draws']]:raise ValueError('Native update trace differs')
        result[arm]=dict(resume_step258_sha256=sha(path),exact_parameters_optimizer_rng=True)
    return dict(native_resume_sha256=sha(output/'native/resume.pt'),steps=[257,258],arms=result)


def verify_training(output,arm):
    import torch
    from vision_memory.training.latent_bank_unet import stable_seed
    folder=output/arm;m=read(folder/'manifest.json');expected=continuation_manifest(read(SOURCE/'manifest.json'),arm)
    if m!=expected or read(folder/'migration.json')['initial_resume_sha256']!=sha(folder/'resume-initial256.pt'):raise ValueError('Manifest/migration changed')
    initial=torch.load(folder/'resume-initial256.pt',map_location='cpu',weights_only=False)
    source=torch.load(SOURCE/'resume.pt',map_location='cpu',weights_only=False)
    if not e.equivalent(initial,migrated_payload(source,arm)):raise ValueError('Initial state no longer exact')
    if not (folder/'optimization.jsonl').read_bytes().startswith((SOURCE/'optimization.jsonl').read_bytes()):raise ValueError('Source256 prefix changed')
    logs=read_lines(folder/'optimization.jsonl');rows=p.population()[0];counts=Counter();formats=Counter()
    if [x['step'] for x in logs]!=list(range(1,385)):raise ValueError('Missing/duplicate updates')
    for row in logs[256:]:
        if not math.isfinite(row['grad_norm']) or row['grad_norm']<=0 or len(row['draws'])!=4:raise ValueError('Bad gradient/update')
        for micro,d in enumerate(row['draws']):
            draw=(row['step']-1)*4+micro;cycle,offset=divmod(draw,32)
            order=torch.randperm(32,generator=torch.Generator().manual_seed(stable_seed(20260924,'order',cycle))).tolist()
            pid=rows[order[offset]]['base_pair_id'];style=training_layout(cycle,arm)
            sigma=float(torch.rand((),generator=torch.Generator().manual_seed(stable_seed(20260924,'sigma',draw))))
            if d['pair_id']!=pid or d['layout']!=style or d['sigma']!=sigma or d['position']!=0 or not math.isfinite(d['mse']):raise ValueError('Draw/noise/layout schedule changed')
            counts[pid]+=1;formats[pid,style]+=1
    if set(counts.values())!={16} or len(counts)!=32:raise ValueError('Unbalanced preference exposure')
    if (arm=='canonical' and (set(formats.values())!={16} or len(formats)!=32)) or (arm=='augmented' and (set(formats.values())!={8} or len(formats)!=64)):raise ValueError('Unbalanced format exposure')
    resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False);final=torch.load(folder/'checkpoint-final.pt',map_location='cpu',weights_only=True)
    if resume['manifest']!=m or final['manifest']!=m or resume['optimizer_step']!=384 or final['optimizer_step']!=384 or resume['episode_cursor']!=1536:raise ValueError('Final cursor mismatch')
    if not e.equivalent(resume['trainable_state'],final['trainable_state']) or {int(v['step']) for v in resume['optimizer']['state'].values()}!={384}:raise ValueError('Final weights/Adam mismatch')
    if read(folder/'complete.json')!=dict(steps=384,checkpoint_sha256=sha(folder/'checkpoint-final.pt')):raise ValueError('Final receipt mismatch')
    return dict(arm=arm,total_steps=384,new_steps=128,new_draws=512,new_draws_per_history=16,format_draws_per_history={'canonical':16} if arm=='canonical' else {'canonical':8,'markdown':8},finite_nonzero_gradients=True,source256_prefix_exact=True,checkpoint_sha256=sha(folder/'checkpoint-final.pt'),resume_sha256=sha(folder/'resume.pt'))


def jobs(args,phase):
    base=[sys.executable,str(Path(__file__)),'train','--output',str(args.output),'--base',str(args.base),'--reader',str(args.reader),'--official-source',str(args.official_source)]
    def arm_job(arm,gpu,stop):return dict(name=f'{arm}-{stop}',gpu=gpu,command=base+['--arm',arm,'--stop-step',str(stop)])
    if phase=='probe':
        native=[sys.executable,'scripts/experiments/prefeval_k1_writer.py','train','--arm','B','--split','pilot','--ids-file',str(p.ALL_IDS),'--base',str(args.base),'--official-source',str(args.official_source),'--checkpoint',str(e.OLD/'train/checkpoint-final.pt'),'--output',str(args.output/'native'),'--teachers',str(p.SOURCE/'bank'),'--teacher-supervision','prompt_matching','--teacher-steps','288','--steps','258']
        return [dict(name='native-258',gpu=0,command=native),arm_job('canonical',1,258)]
    if phase=='probe-aug':return [arm_job('augmented',0,258)]
    if phase=='full':return [arm_job(arm,i,384) for i,arm in enumerate(ARMS)]
    raise ValueError('Wrong phase')


def main(args):
    if args.output.resolve()!=RUN/'context-layout-aug-v1':raise ValueError('Wrong isolated output')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Dirty source')
    args.output.mkdir(parents=True,exist_ok=True);claim=args.output/'active-owner';claim.mkdir()
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(claim/'owner.json',dict(pid=os.getpid(),host=socket.gethostname(),started=time.time(),commit=commit))
        for name in ('attempts','receipts','logs'):(args.output/name).mkdir(exist_ok=True)
        save_once(args.output/'plan.json',dict(commit=commit,plan_sha256=sha(PLAN),source_resume_sha256=RESUME_SHA,base=str(args.base),reader=str(args.reader),official_source=str(args.official_source),train_gpu_hours_cap=.75,iteration_gpu_hours_cap=3,campaign_gpu_hours_cap=16,steps=384,new_steps=128,arms=list(ARMS),readout_new_pngs=384,readout_new_rows=9216,readout_combined_rows=16128,readout_formats=['canonical','markdown','xml']))
        write_json(args.output/'status.json',dict(status='source_audit',time=time.time(),commit=commit))
        save_once(args.output/'source.json',verify_source());save_once(args.output/'initials.json',prepare(args.output))
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():raise ValueError('GPU occupied')
        if len(subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines())!=2:raise ValueError('Require two GPUs')
        if [x for x in RUN.glob('*/active-owner/owner.json') if x!=claim/'owner.json']:raise ValueError('Another owner')
        left,previous,_=budget(args.output)
        if left<=60:raise ValueError('Budget exhausted')
        deadline=time.monotonic()+min(6*3600,(left-60)/2)
        def group(items):
            if time.monotonic()+30>=deadline:raise ValueError('No next-stage time')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                for f in [pool.submit(execute,j,args.output,deadline) for j in items]:f.result()
        write_json(args.output/'status.json',dict(status='native_parity',time=time.time(),commit=commit))
        group(jobs(args,'probe'));group(jobs(args,'probe-aug'))
        save_once(args.output/'native-parity.json',verify_parity(args.output))
        write_json(args.output/'status.json',dict(status='training',time=time.time(),commit=commit))
        group(jobs(args,'full'));save_once(args.output/'training-audit.json',{arm:verify_training(args.output,arm) for arm in ARMS})
        cost=accrued_seconds(args.output)
        write_json(args.output/'status.json',dict(status='ready_for_frozen_readout',time=time.time(),gpu_hours=cost/3600,campaign_gpu_hours=(previous+cost)/3600,readout_new_rows=9216,readout_combined_rows=16128))
    except BaseException as exc:
        write_json(args.output/'status.json',dict(status='failed',time=time.time(),error=str(exc)));raise
    finally:
        (claim/'owner.json').unlink(missing_ok=True);claim.rmdir()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('run','train'))
    for name in ('output','base','reader','official-source'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--stop-step',type=int,choices=(258,384))
    args=parser.parse_args();(main if args.phase=='run' else train)(args)
