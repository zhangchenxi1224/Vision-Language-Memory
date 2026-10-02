"""Train fresh U-Net LoRA on actual four-step reference paths from a sealed bank.

Full student rollout gradients. L = endpoint MSE + mean intermediate-state MSE.
All 96 multiprompt Direct teachers participate in the reference field. There is
no QA gradient in this first distillation arm and no teacher-bank access in the
student forward. Raw QA/EOS and fresh-noise geometry are independently evaluated.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from scripts.train import train_latent_bank_unet as training
from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
from vision_memory.dreamlite.differentiable_mobile import DreamLiteSamplerOutput
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.open_eos import generation_diagnostics,qwen3vl_answer_eos_ce
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import canonical_tensor_sha256,configure_strict_cuda_determinism,lora_trainable_parameters
from vision_memory.training.checkpoint import load_training_checkpoint,save_training_checkpoint
from vision_memory.training.empirical_bank_flow import EmpiricalBankFlow
from vision_memory.training.latent_bank_unet import EFFECTIVE_SIGMAS,file_sha256,load_teacher_bank,stable_seed
from vision_memory.training.reference_distillation import reference_trajectory,rollout_distillation_loss

ORACLE_COMMIT='46cd36b1eb421471dafa7865fbab4dfe02336336'
ORACLE_SUFFIX='runs/direct-multiprompt-eos/46cd36b-20260909-r01'


class ReferenceSampler:
    def __init__(self,flow,sigmas): self.flow,self.sigmas=flow,tuple(sigmas)
    @torch.no_grad()
    def __call__(self,*,source_latents,noise_latents,**kwargs):
        path=reference_trajectory(self.flow,source_latents,noise_latents,sigmas=self.sigmas)
        return DreamLiteSamplerOutput(path[-1],path,self.sigmas)


def validate_bank(args):
    if file_sha256(args.bank_manifest)!=args.bank_sha256:
        raise ValueError('Bank SHA differs from deployment binding')
    bank,teachers=load_teacher_bank(args.bank_manifest)
    provenance=bank['provenance']
    if (bank['route']!='direct' or len(teachers)!=96 or len(bank['groups'])!=1
            or provenance['oracle_commit']!=ORACLE_COMMIT
            or not provenance['oracle_root'].endswith(ORACLE_SUFFIX)):
        raise ValueError('Require the user-selected fresh 96/96 multiprompt Direct bank')
    protocol=provenance['oracle_prompt_protocol']
    if protocol['training_prompts']!=['original_open','paraphrase_1','paraphrase_2'] or protocol['heldout_prompts']!=['paraphrase_3','paraphrase_4']:
        raise ValueError('Wrong oracle prompt protocol')
    root=Path(provenance['oracle_root'])
    terminal=json.loads((root/'terminal.json').read_text())
    if any(terminal.get(k)!=v for k,v in {'status':'completed','planned':96,'completed':96,'success':96}.items()):
        raise ValueError('96/96 source campaign not complete and successful')
    seal=json.loads((args.bank_manifest.parent/'seal.json').read_text())
    if seal['manifest_sha256']!=args.bank_sha256:
        raise ValueError('Bank seal invalid')
    return bank,teachers


def run(args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if actual!=args.expected_commit or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Exact clean source commit required')
    bank,teachers=validate_bank(args)
    binding=dict(schema='reference-distillation/v1',source_commit=actual,source_hashes=training.source_hashes(),
        bank_sha256=args.bank_sha256,teacher_ids=bank['groups'][0]['teacher_ids'],teacher_split='all',
        teacher_count=96,question_count=1,seed=args.seed,steps=args.steps,lr=args.lr,rank=args.lora_rank,
        init='fresh pretrained DreamLite with new zero-B LoRA; no old adapter',
        objective='actual endpoint MSE + mean actual intermediate-state MSE',
        trajectory_sigmas=list(EFFECTIVE_SIGMAS),qa_training_weight=0.,eval_seeds=args.eval_seeds,
        teacher_noise_namespace='reference-distillation-training',eval_noise_namespace='heldout-evaluation-noise',
        generalization_scope='single ambient question, fresh noise and two oracle-heldout paraphrases; no heldout target split',
        reference_failure_policy='pause before optimizer update; preserve failed reference, never relabel or silently drop')
    identity=args.output_dir/'identity.json'
    if identity.exists() and json.loads(identity.read_text())!=binding:
        raise ValueError('Output belongs to a different experiment')
    training.write_json(identity,binding)
    os.environ.update(snapshot_environment(bank))
    configure_strict_cuda_determinism(args.seed)
    runtime=training.load_runtime(args,bank)
    runtime['sampler'].checkpoint_unet=True
    pipe,reader=runtime['pipe'],runtime['reader']
    frozen=training.frozen_versions(pipe,reader)
    group=bank['groups'][0]
    ctx=runtime['contexts'][group['question_id']]
    source,cond=ctx['source'],ctx['condition']
    ids=group['teacher_ids']
    targets=torch.stack([teachers[t] for t in ids]).to(source.device)
    _,actual_sigmas=runtime['sampler']._prepare_timesteps(source,4,EFFECTIVE_SIGMAS,sigmas_are_effective=True)
    flow=EmpiricalBankFlow(source,targets,start_sigma=actual_sigmas[0])
    reference_runtime={**runtime,'sampler':ReferenceSampler(flow,actual_sigmas)}
    parameters=lora_trainable_parameters(pipe.unet)
    initial={n:p.detach().cpu().clone() for n,p in pipe.unet.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW(parameters,lr=args.lr,weight_decay=0.)
    checkpoint=args.output_dir/'checkpoint-latest.pt'
    stop=[False]
    def signal_stop(signum,frame): stop[0]=True
    signal.signal(signal.SIGTERM,signal_stop)
    signal.signal(signal.SIGINT,signal_stop)
    def should_stop(): return stop[0] or time.time()>=args.deadline_unix-120
    runtime['should_pause']=should_stop
    reference_runtime['should_pause']=should_stop
    def status(state,**kw):
        training.write_json(args.output_dir/'status.json',dict(state=state,epoch=time.time(),hostname=socket.gethostname(),**kw))
    status('reference_preflight',optimizer_steps=0)
    reference=training.evaluate(args,reference_runtime,bank,teachers,'reference')
    if reference['cells']['matched/original_open']['exact_match']!=args.eval_seeds:
        raise RuntimeError('Reference algorithm failed original-open preflight; not teaching failed targets')
    status('baseline_evaluation',optimizer_steps=0)
    baseline=training.evaluate(args,runtime,bank,teachers,'baseline')
    step=0
    if checkpoint.exists():
        if not args.resume: raise ValueError('Checkpoint exists; explicit resume required')
        payload=load_training_checkpoint(checkpoint,trainable_module=pipe.unet,optimizer=optimizer,expected_manifest=binding)
        step=payload['optimizer_step']
        training.write_json(args.output_dir/'metrics'/f'step-{step:06d}.json',payload['trainer_state']['metrics_row'])
    elif args.resume:
        raise ValueError('Resume requested but checkpoint absent')
    metrics=[json.loads((args.output_dir/'metrics'/f'step-{i:06d}.json').read_text()) for i in range(1,step+1)]
    training.write_json(args.output_dir/'runtime.json',dict(snapshots=runtime['snapshots'],termination=runtime['termination'],
        trainable_parameters=sum(p.numel() for p in parameters),device_count=torch.cuda.device_count(),actual_sigmas=actual_sigmas,
        physical_gpu_indices=args.physical_gpus,process_pid=os.getpid(),hostname=socket.gethostname()))

    def verify_reference(ref,noise_seed):
        endpoint=ref[-1]
        distances=(targets-endpoint).flatten(1).square().mean(1)
        index=int(distances.argmin())
        exact=torch.equal(endpoint,targets[index])
        receipt=dict(noise_seed=noise_seed,nearest_teacher_id=ids[index],nearest_rms=float(distances[index].sqrt()),
            exact_verified_member=exact,endpoint_sha256=canonical_tensor_sha256(endpoint))
        if not exact:
            pixels=decode_model_latents_unit_interval(pipe.vae,endpoint,clamp=True)
            q=group['question_variants']['original_open']
            loss=qwen3vl_answer_eos_ce(model=reader,processor=runtime['processor'],image=pixels[0].to(runtime['reader_device']),
                device=runtime['reader_device'],query=q,target=group['answer'],termination=runtime['termination'],lambda_eos=1.,
                require_image_grad=False,deterministic_ce=True,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            gen=generate_short_answer(model=reader,processor=runtime['processor'],image=pixels.to(runtime['reader_device']),
                query=q,device=runtime['reader_device'],max_new_tokens=32,do_sample=False)
            score=generation_diagnostics(gen,group['answer'],loss.target_ids[0,:loss.answer_token_count].cpu().tolist())
            receipt.update(generation=gen,score=score)
            if not(score['strict_correct'] and score['answer_followed_immediately_by_eos']):
                training.atomic_tensor(args.output_dir/'failed-reference.pt',dict(path=[t.cpu() for t in ref],receipt=receipt))
                raise RuntimeError('Unverified reference endpoint failed raw answer+EOS')
        return receipt

    started=time.monotonic()
    while step<args.steps:
        if should_stop():
            status('paused',optimizer_steps=step,reason='signal or lease deadline',checkpoint=str(checkpoint))
            training.write_json(args.output_dir/'terminal.json',dict(status='paused',optimizer_steps=step,checkpoint=str(checkpoint)))
            return 75
        noise_seed=stable_seed(args.seed,'reference-distillation-training',step)
        noise=torch.randn(source.shape,generator=torch.Generator().manual_seed(noise_seed)).to(source.device)
        with torch.no_grad():
            ref=reference_trajectory(flow,source,noise,sigmas=actual_sigmas)
            receipt=verify_reference(ref,noise_seed)
        optimizer.zero_grad(set_to_none=True)
        output=runtime['sampler'](source_latents=source,noise_latents=noise,prompt_embeds=cond.prompt_embeds,
            prompt_attention_mask=cond.attention_mask,num_steps=4,edit_start_sigma=.5,return_trajectory=True)
        if not torch.equal(output.trajectory[0],ref[0]): raise RuntimeError('Teacher/student starting states differ')
        if output.effective_sigmas!=actual_sigmas: raise RuntimeError('Teacher/student schedules differ')
        loss,end,path=rollout_distillation_loss(output.trajectory,ref)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite distillation loss')
        loss.backward()
        grads=[p.grad for p in parameters if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads) or not any((g!=0).any() for g in grads):
            raise RuntimeError('Missing finite nonzero LoRA gradients')
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        training.frozen_audit(pipe,reader,frozen)
        step+=1
        row=dict(optimizer_step=step,noise_seed=noise_seed,loss=float(loss.detach()),endpoint_mse=float(end.detach()),
            intermediate_mse=float(path.detach()),grad_norm_before_clip=norm,elapsed_seconds=time.monotonic()-started,reference=receipt)
        save_training_checkpoint(checkpoint,trainable_module=pipe.unet,optimizer=optimizer,epoch=0,episode_cursor=step,
            optimizer_step=step,manifest=binding,trainer_state={'metrics_row':row})
        training.write_json(args.output_dir/'metrics'/f'step-{step:06d}.json',row)
        training.append_jsonl(args.output_dir/'training.jsonl',row)
        metrics.append(row)
        status('training',optimizer_steps=step,total_steps=args.steps,latest=row)
        if step==1 or step%16==0: print(json.dumps({'stage':'distillation',**row}),flush=True)
        del loss,end,path,output,ref,grads
        if step in [64,256] and step<args.steps:
            status('intermediate_evaluation',optimizer_steps=step)
            training.evaluate(args,runtime,bank,teachers,f'step-{step:06d}')
    training.atomic_tensor(args.output_dir/'checkpoint-final.pt',torch.load(checkpoint,map_location='cpu',weights_only=False))
    status('final_evaluation',optimizer_steps=step)
    trained=training.evaluate(args,runtime,bank,teachers,'trained')
    training.frozen_audit(pipe,reader,frozen)
    validate_bank(args)
    if training.source_hashes()!=binding['source_hashes']: raise RuntimeError('Source changed during training')
    delta=sum(float((p.detach().cpu()-initial[n]).double().square().sum()) for n,p in pipe.unet.named_parameters() if p.requires_grad)**.5
    if not delta>0: raise RuntimeError('No parameter update')
    result=dict(status='completed',optimizer_steps=step,teacher_count=96,teacher_split='all',heldout_teacher_count=0,
        reference=reference,baseline=baseline,trained=trained,adapter_delta_l2=delta,frozen_audit_passed=True,
        target_visits=dict(Counter(m['reference']['nearest_teacher_id'] for m in metrics)),
        nonmember_reference_count=sum(not m['reference']['exact_verified_member'] for m in metrics),
        distinct_training_noise_count=len({m['noise_seed'] for m in metrics}),checkpoint_sha256=file_sha256(args.output_dir/'checkpoint-final.pt'),
        scientific_success=trained['cells']['matched/original_open']['exact_match']==args.eval_seeds,
        limits='Training completion != functional success; single question, all 96 reference members, fresh noise; no unseen-question claim')
    training.write_json(args.output_dir/'result.json',result)
    training.write_json(args.output_dir/'terminal.json',dict(status='completed',optimizer_steps=step,result_sha256=file_sha256(args.output_dir/'result.json')))
    status('completed',optimizer_steps=step,scientific_success=result['scientific_success'])
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['bank-manifest','output-dir','dreamlite','reader-model']: p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--bank-sha256',required=True)
    p.add_argument('--expected-commit',required=True)
    p.add_argument('--steps',type=int,default=512)
    p.add_argument('--seed',type=int,default=20260908)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--lora-rank',type=int,default=4)
    p.add_argument('--eval-seeds',type=int,default=8)
    p.add_argument('--dreamlite-device',default='cuda:0')
    p.add_argument('--reader-device',default='cuda:1')
    p.add_argument('--physical-gpus',default='2,3')
    p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args();args.teacher_split='all'
    if args.steps<1 or args.eval_seeds<2 or args.lr<=0: raise ValueError('Invalid budget')
    if args.validate_only:
        bank,teachers=validate_bank(args);print(json.dumps(dict(status='validated',teachers=len(teachers),questions=len(bank['groups']))));return 0
    args.output_dir.mkdir(parents=True,exist_ok=True)
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
        devices={int(x.split(',')[0]):x.split(',')[1].strip() for x in raw.splitlines()}
        selected={devices[int(i)] for i in args.physical_gpus.split(',')}
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.physical_gpus or len(selected)!=2:
            raise ValueError('Exactly two explicitly isolated physical GPUs required')
        with ExitStack() as locks:
            owner=locks.enter_context((args.output_dir/'.training.lock').open('a+'));fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for uuid in sorted(selected):
                handle=locks.enter_context((Path('/tmp')/f'vlm-oracle-writer-{uuid}.lock').open('a+'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
            if selected & set(active.splitlines()): raise RuntimeError('Selected GPUs are occupied; no models were loaded')
            return run(args)
    except training.TrainingPaused as error:
        training.write_json(args.output_dir/'terminal.json',dict(status='paused',reason=str(error),checkpoint=str(args.output_dir/'checkpoint-latest.pt')))
        return 75
    except BaseException:
        training.write_json(args.output_dir/'failure.json',dict(status='failed',epoch=time.time(),traceback=traceback.format_exc()))
        training.write_json(args.output_dir/'status.json',dict(state='failed',epoch=time.time(),failure='failure.json'))
        raise


if __name__=='__main__': raise SystemExit(main())
