"""Isolated, gated actual-rollout U-Net learnability experiment on a sealed bank."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from scripts.train import train_latent_bank_unet as training
from scripts.inspire.run_oracle_to_unet_pipeline import (gpu_snapshot, own_source,
    parent_death_signal, snapshot_environment, source_unchanged)
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.open_eos import generation_diagnostics, qwen3vl_answer_eos_ce
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism, lora_trainable_parameters
from vision_memory.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from vision_memory.training.latent_bank_unet import file_sha256, load_teacher_bank
from vision_memory.training.learnability import evaluation_seed, gate, teacher_order, training_pair


def rms(x):
    return float(x.detach().double().square().mean().sqrt())


def read(path):
    return json.loads(Path(path).read_text())


def verify_bank(config):
    path=Path(config['bank_manifest'])
    if file_sha256(path)!=config['bank_sha256']:
        raise RuntimeError('Bound data bank changed')
    bank,teachers=load_teacher_bank(path)
    assert len(teachers)==config['teacher_count']==96 and len(bank['groups'])==config['question_count']==1
    assert bank['provenance']['oracle_commit']==config['oracle_commit']
    assert bank['provenance']['oracle_root']==config['oracle_root']
    terminal=read(Path(config['oracle_root'])/'terminal.json')
    assert terminal['status']=='completed' and terminal['completed']==terminal['planned']==96
    seal=read(path.parent/'seal.json')
    assert seal['manifest_sha256']==config['bank_sha256']
    assert file_sha256(path.parent/'oracle_audit.json')==bank['oracle_audit_sha256']
    assert bank['provenance']['oracle_prompt_protocol']['training_prompts']==['original_open','paraphrase_1','paraphrase_2']
    return bank,teachers


def stage_worker(args,config):
    output=Path(args.output)/f'{args.stage}-rank{args.rank}'
    output.mkdir(parents=True,exist_ok=True)
    sources=own_source(args.expected_commit)
    bank,teachers=verify_bank(config)
    ids=teacher_order(bank,config['anchor_run'])
    os.environ.update(snapshot_environment(bank))
    configure_strict_cuda_determinism(config['seed'])
    runtime_args=SimpleNamespace(bank_manifest=Path(config['bank_manifest']),dreamlite=Path(config['dreamlite']),
        reader_model=Path(config['reader_model']),seed=config['seed'],lora_rank=args.rank,
        dreamlite_device='cuda:0',reader_device='cuda:1')
    rt=training.load_runtime(runtime_args,bank)
    rt['sampler'].checkpoint_unet=True
    group=bank['groups'][0]
    ctx=rt['contexts'][group['question_id']]
    source,condition=ctx['source'],ctx['condition']
    pipe,reader=rt['pipe'],rt['reader']
    parameters=lora_trainable_parameters(pipe.unet)
    initial={n:p.detach().cpu().clone() for n,p in pipe.unet.named_parameters() if p.requires_grad}
    versions=training.frozen_versions(pipe,reader)
    device_teachers={tid:teachers[tid].to(source.device) for tid in ids}
    target_stack=torch.cat([device_teachers[tid] for tid in ids])
    binding={'schema':'unet-learnability-stage/v1','commit':args.expected_commit,
        'config_sha256':file_sha256(args.config),'bank_sha256':config['bank_sha256'],'stage':args.stage,
        'rank':args.rank,'seed':config['seed'],'lr':config['lr'],'anchor_id':ids[0],
        'source_hashes':sources,'teacher_ids':ids,'loss':config['loss']}
    identity=output/'identity.json'
    if identity.exists():
        assert read(identity)==binding
    else:
        training.write_json(identity,binding)
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def deadline():
        return stopped[0] or time.time()>=args.deadline-120
    def noise(seed):
        return torch.randn(source.shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float32).to(source.device)
    def forward(seed):
        return rt['sampler'](source_latents=source,noise_latents=noise(seed),
            prompt_embeds=condition.prompt_embeds,prompt_attention_mask=condition.attention_mask,
            num_steps=4,edit_start_sigma=.5,return_trajectory=False).latents
    @torch.no_grad()
    def qa(pixels,prompt_id):
        query=group['question_variants'][prompt_id]
        loss=qwen3vl_answer_eos_ce(model=reader,processor=rt['processor'],
            image=pixels[0].to(rt['reader_device']),device=rt['reader_device'],query=query,
            target=group['answer'],termination=rt['termination'],lambda_eos=1.,
            require_image_grad=False,deterministic_ce=True,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
        generated=generate_short_answer(model=reader,processor=rt['processor'],
            image=pixels.to(rt['reader_device']),query=query,device=rt['reader_device'],**config['generation'])
        score=generation_diagnostics(generated,group['answer'],loss.target_ids[0,:loss.answer_token_count].cpu().tolist())
        return {'question_id':group['question_id'],'prompt_id':prompt_id,'raw':generated['raw'],
                'generation':generated,'scorer':score,'answer_ce':float(loss.answer_loss),'eos_ce':float(loss.eos_loss)}
    if args.stage=='preflight':
        rows=[]
        for tid in ids:
            if deadline():
                raise RuntimeError('Resource lease expired during teacher audit')
            z=device_teachers[tid]
            image=decode_model_latents_unit_interval(pipe.vae,z,clamp=True)
            record=next(r for r in bank['teachers'] if r['teacher_id']==tid)
            image_hash=canonical_tensor_sha256(image)
            assert image_hash==record['qa']['image_sha256']['matched']
            row={'teacher_id':tid,'image_sha256':image_hash,**qa(image,'original_open')}
            rows.append(row)
            assert row['scorer']['strict_correct'] and row['scorer']['answer_followed_immediately_by_eos']
            if len(rows)%16==0:
                print(json.dumps({'stage':'teacher_preflight','verified':len(rows),'total':len(ids)}),flush=True)
        result={'status':'completed','optimizer_steps':0,'teachers':rows,
            'source_bank_sha256':config['bank_sha256'],'anchor_id':ids[0],'completed_epoch':time.time()}
        training.write_json(output/'result.json',result)
        training.write_json(output/'result-0000.json',result)
        return 0
    train_count=1 if args.stage=='single' else len(ids)
    cases=[{'split':'train','index':i,'noise_seed':training_pair(args.stage,i,ids,config['seed'])[0],
            'target_id':training_pair(args.stage,i,ids,config['seed'])[1]} for i in range(train_count)]
    for split,count in [('validation',config['validation_noises']),('test',config['test_noises'])]:
        cases += [{'split':split,'index':i,'noise_seed':evaluation_seed(config['seed'],split,i),
                   'target_id':ids[0] if args.stage!='set' else None} for i in range(count)]
    assert len({c['noise_seed'] for c in cases})==len(cases)
    training.write_json(output/'cases.json',{'cases':cases,'training_count':train_count,'scope':config['scope']})
    baseline_path=output/'baseline-rms.json'
    if baseline_path.exists():
        baseline=read(baseline_path)
    else:
        baseline={}
        with torch.no_grad():
            for c in cases:
                if c['target_id']:
                    baseline[str(c['noise_seed'])]=rms(forward(c['noise_seed'])-device_teachers[c['target_id']])
        training.write_json(baseline_path,baseline)
    optimizer=torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=0.)
    checkpoint=output/'checkpoint-latest.pt'
    step=0
    if checkpoint.exists():
        saved=load_training_checkpoint(checkpoint,trainable_module=pipe.unet,optimizer=optimizer,expected_manifest=binding)
        step=saved['optimizer_step']
    def save():
        save_training_checkpoint(checkpoint,trainable_module=pipe.unet,optimizer=optimizer,
            epoch=0,episode_cursor=step,optimizer_step=step,manifest=binding)
    @torch.no_grad()
    def evaluate(full):
        rows=[]
        subset=cases if full else cases[:1]
        for c in subset:
            if deadline():
                raise training.TrainingPaused('Lease or stop signal during evaluation')
            z=forward(c['noise_seed'])
            image=decode_model_latents_unit_interval(pipe.vae,z,clamp=True)
            nearest=((target_stack-z).square().flatten(1).mean(1)).sqrt()
            error=rms(z-device_teachers[c['target_id']]) if c['target_id'] else None
            relative=error/max(baseline[str(c['noise_seed'])],1e-12) if error is not None else None
            prompts=list(group['question_variants']) if c['split']=='test' else ['original_open']
            for pid in prompts:
                rows.append({**c,'latent_rms':error,'relative_rms':relative,
                    'nearest_teacher_rms':float(nearest.min()),'nearest_teacher_id':ids[int(nearest.argmin())],
                    **qa(image,pid)})
            if full:
                payload={'latent':z.cpu(),'noise_seed':c['noise_seed'],'target_id':c['target_id']}
                if c['split']=='test' or c['index']==0:
                    payload['image']=image.cpu()
                training.atomic_tensor(output/'outputs'/f'step{step:04d}-{c["split"]}-{c["index"]:03d}.pt',payload)
        if full:
            for split,pixels in [('blank',ctx['blank']),('donor',ctx['donor'])]:
                for pid in group['question_variants']:
                    rows.append({'split':split,'index':0,'noise_seed':None,'relative_rms':None,**qa(pixels,pid)})
        result={'optimizer_step':step,'full':full,'rows':rows}
        training.write_json(output/'evaluation'/f'step-{step:04d}-{"full" if full else "anchor"}.json',result)
        if not full:
            print(json.dumps({'stage':args.stage,'rank':args.rank,'evaluation_step':step,
                              'raw':rows[0]['raw'],'relative_rms':rows[0]['relative_rms']}),flush=True)
        return rows
    if step==0:
        save()
        evaluate(False)
    started=time.monotonic()
    while step<args.steps:
        if deadline():
            save()
            training.write_json(output/'result.json',{'status':'paused','optimizer_steps':step,'reason':'lease_or_signal'})
            return 75
        ns,tid=training_pair(args.stage,step,ids,config['seed'])
        optimizer.zero_grad(set_to_none=True)
        pred=forward(ns)
        loss=(pred-device_teachers[tid]).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for p in parameters if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any((g!=0).any() for g in grads)
        grad_norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert all(torch.isfinite(p).all() for p in parameters)
        step+=1
        row={'stage':args.stage,'rank':args.rank,'optimizer_step':step,'target_id':tid,'noise_seed':ns,
             'endpoint_mse':float(loss.detach()),'gradient_norm_before_clip':grad_norm,
             'elapsed_since_resume':time.monotonic()-started,'epoch':time.time()}
        del pred,loss,grads
        training.write_json(output/'metrics'/f'step-{step:04d}.json',row)
        training.write_json(output/'status.json',{'state':'training',**row})
        if step%16==0 or step==args.steps:
            save()
            training.frozen_audit(pipe,reader,versions)
            print(json.dumps(row),flush=True)
        if step in (16,64,128,256,512,1024,1536,2048) and step<args.steps:
            evaluate(False)
    save()
    training.atomic_tensor(output/f'checkpoint-{step:04d}.pt',torch.load(checkpoint,map_location='cpu',weights_only=False))
    rows=evaluate(True)
    decision=gate(rows,args.stage,config['relative_rms_gate'])
    training.frozen_audit(pipe,reader,versions)
    source_unchanged(sources)
    verify_bank(config)
    delta=sum(float((p.detach().cpu()-initial[n]).double().square().sum()) for n,p in pipe.unet.named_parameters() if p.requires_grad)**.5
    result={'status':'completed','stage':args.stage,'rank':args.rank,'optimizer_steps':step,
        'gate':decision,'adapter_delta_l2':delta,'frozen_audit_passed':True,'scope':config['scope'],
        'training_noise_count':train_count,'training_teacher_count':len(ids) if args.stage=='set' else 1,
        'checkpoint_sha256':file_sha256(checkpoint),'completed_epoch':time.time()}
    for split in ('train','validation','test'):
        selected=[r for r in rows if r['split']==split and r['prompt_id']=='original_open']
        result[split]={'n':len(selected),'exact_match':sum(r['scorer']['strict_correct'] for r in selected),
            'answer_and_eos':sum(r['scorer']['answer_followed_immediately_by_eos'] for r in selected),
            'mean_latent_rms':sum(r['latent_rms'] for r in selected)/len(selected) if selected and selected[0]['latent_rms'] is not None else None}
    result['prompt_cells']={}
    for row in rows:
        key=row['split']+'/'+row['prompt_id']
        cell=result['prompt_cells'].setdefault(key,{'n':0,'exact_match':0,'answer_prefix':0,'overgeneration':0})
        cell['n']+=1
        for field,metric in [('exact_match','strict_correct'),('answer_prefix','answer_prefix_token_exact'),('overgeneration','overgeneration')]:
            cell[field]+=int(row['scorer'][metric])
    training.write_json(output/f'result-{step:04d}.json',result)
    training.write_json(output/'result.json',result)
    return 0


def campaign(args,config):
    import fcntl
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    sources=own_source(args.expected_commit)
    assert socket.gethostname()==args.expected_hostname
    devices,active=gpu_snapshot()
    assert set(devices)=={0,1},'Use the explicitly allocated two-GPU notebook only'
    uuids=set(devices.values())
    assert not any(p['uuid'] in uuids for p in active),'GPUs are occupied; do not disturb the owner'
    stopped=[False]
    child=[None]
    def stop(*_):
        stopped[0]=True
        if child[0] is not None and child[0].poll() is None:
            child[0].terminate()
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,stop)
    with ExitStack() as locks:
        for name in [output/'.campaign.lock']+[Path('/tmp')/f'vlm-oracle-writer-{u}.lock' for u in sorted(uuids)]:
            handle=locks.enter_context(name.open('a+'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _,active=gpu_snapshot()
        assert not any(p['uuid'] in uuids for p in active),'GPU became occupied before launch'
        verify_bank(config)
        training.write_json(output/'resource-lease.json',{'hostname':socket.gethostname(),'gpu_uuids':sorted(uuids),
            'deadline_epoch':args.deadline,'commit':args.expected_commit,'config_sha256':file_sha256(args.config)})
        def execute(stage,rank,steps):
            source_unchanged(sources)
            if stopped[0] or time.time()>args.deadline-600:
                raise training.TrainingPaused('Insufficient remaining lease; preserving checkpoints')
            destination=output/f'{stage}-rank{rank}'
            previous=destination/f'result-{steps:04d}.json'
            if previous.exists() and read(previous)['status']=='completed':
                return read(previous)
            command=[sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(args.config),
                '--output',str(output),'--stage',stage,'--rank',str(rank),'--steps',str(steps),
                '--expected-commit',args.expected_commit,'--deadline',str(args.deadline)]
            with (output/f'{stage}-rank{rank}-steps{steps}.log').open('ab',buffering=0) as log:
                child[0]=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,preexec_fn=parent_death_signal)
                training.write_json(output/'status.json',{'state':'running','stage':stage,'rank':rank,
                    'target_steps':steps,'pid':child[0].pid,'command':command,'epoch':time.time()})
                code=child[0].wait()
            if code==75:
                raise training.TrainingPaused('Stage paused at checkpoint')
            if code:
                raise RuntimeError(f'{stage} rank{rank} failed with exit code {code}; no automatic retry')
            return read(destination/'result.json')
        execute('preflight',config['base_rank'],0)
        primary=execute('single',config['base_rank'],config['base_steps'])
        rank,budget=config['base_rank'],config['base_steps']
        if not primary['gate']['passed']:
            # Capacity control uses the same base budget, then budget control
            # changes only rank-4 duration and preserves optimizer/RNG exactly.
            capacity=execute('single',config['capacity_rank'],config['base_steps'])
            extended=execute('single',config['base_rank'],config['extended_steps'])
            if extended['gate']['passed']:
                rank,budget=config['base_rank'],config['extended_steps']
            elif capacity['gate']['passed']:
                rank,budget=config['capacity_rank'],config['base_steps']
            else:
                training.write_json(output/'terminal.json',{'status':'completed_gate_not_met','stage':'single',
                    'reason':'Single-input learnability did not pass; noise/set stages not launched',
                    'base':primary,'capacity':capacity,'budget':extended,'completed_epoch':time.time()})
                return
        multi_noise=execute('noise',rank,budget)
        if not multi_noise['gate']['passed']:
            training.write_json(output/'terminal.json',{'status':'completed_gate_not_met','stage':'noise',
                'result':multi_noise,'reason':'Multiple-noise gate did not pass; set stage not launched','completed_epoch':time.time()})
            return
        multi_target=execute('set',rank,budget)
        training.write_json(output/'terminal.json',{'status':'completed','final_stage':'set','result':multi_target,
            'multi_question_stage':'not_run_missing_multi_question_bank','completed_epoch':time.time()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--expected-hostname')
    p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--stage',choices=['preflight','single','noise','set'])
    p.add_argument('--rank',type=int,default=4)
    p.add_argument('--steps',type=int,default=512)
    args=p.parse_args()
    config=read(args.config)
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        if args.stage:
            return stage_worker(args,config)
        campaign(args,config)
        training.write_json(args.output/'status.json',{'state':'finished',**read(args.output/'terminal.json')})
        return 0
    except training.TrainingPaused as error:
        target=args.output/f'{args.stage}-rank{args.rank}' if args.stage else args.output
        training.write_json(target/'terminal.json',{'status':'paused','reason':str(error),'epoch':time.time()})
        return 75
    except BaseException:
        target=args.output/f'{args.stage}-rank{args.rank}' if args.stage else args.output
        training.write_json(target/'failure.json',{'error':traceback.format_exc(),'epoch':time.time()})
        raise


if __name__=='__main__':
    raise SystemExit(main())
