"""Shared PrefEval Writer: reuse official Base and the existing FM primitives."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
from vision_memory.prefeval.official_ab import records, seed_for, writer_event, sha
from scripts.experiments.prefeval_official_ab import write, append, atomic_tensor

P=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
M=Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory')

def load_pipe(device, package=None):
    import torch
    from vision_memory.dreamlite.writer_package import inspect_package, load_parameter_export, UPSTREAM
    official=P/'Vision-Language-Memory/third_party/DreamLite'
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=official,text=True).strip()!=UPSTREAM:
        raise RuntimeError('Official DreamLite revision changed')
    sys.path.insert(0,str(official))
    from dreamlite import DreamLitePipelineLoRA
    pipe=DreamLitePipelineLoRA.from_pretrained(M/'DreamLite-base-a9a0f15-20260907',
        local_files_only=True,torch_dtype=torch.float32).to(device)
    for module in (pipe.vae,pipe.text_encoder,pipe.unet): module.eval().requires_grad_(False)
    pipe.set_progress_bar_config(disable=True)
    package=package or P/'runs/dreamlite-official-alignment/e372f3c-logical-package'
    manifest=inspect_package(package)
    if manifest['parent_checkpoint_sha256']!='7294684170578dfc617b4fafcea97e6480c08642ca4f1e5967ff8966aa103182':
        raise RuntimeError('Initialization is not the completed 4f Writer')
    load_parameter_export(pipe.unet,package,manifest)
    return pipe, manifest

def run(a):
    import socket
    import torch
    from PIL import Image
    from vision_memory.dreamlite import DifferentiableDreamLiteMobileSampler
    from vision_memory.dreamlite.conditioning import encode_native_base_edit_condition
    from vision_memory.training.latent_bank_unet import official_flow_bridge, predict_velocity
    if not socket.gethostname().startswith('dl-clear-retain-h200x4-20260914'):
        raise RuntimeError('Wrong notebook')
    selected=[r for r in records(a.report) if r['split']=='train']
    out=a.output/'writers'/a.arm/a.stage
    out.mkdir(parents=True,exist_ok=True)
    targets={};target_hashes={}
    for r in selected:
        folder=a.output/'teachers'/a.arm/r['id'].replace(':','-')
        done=json.loads((folder/'complete.json').read_text())
        if done['binding']['steps']!=288 or done['latent_sha256']!=sha(folder/'latent.pt'):
            raise RuntimeError('Require all fixed teacher endpoints, including imperfect ones')
        targets[r['id']]=torch.load(folder/'latent.pt',map_location=a.device,weights_only=True)
        target_hashes[r['id']]=done['latent_sha256']
    pipe,parent=load_pipe(a.device)
    if a.stage=='retain':
        initial=torch.load(a.output/'writers'/a.arm/'write/checkpoint-final.pt',map_location='cpu',weights_only=False)
        pipe.unet.load_state_dict(initial['unet'])
    pipe.unet.requires_grad_(True).eval()
    sampler=DifferentiableDreamLiteMobileSampler.from_pipeline(pipe,checkpoint_unet=False)
    optimizer=torch.optim.AdamW(pipe.unet.parameters(),lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=1e-4)
    binding=dict(arm=a.arm,stage=a.stage,steps=a.steps,batch=4,lr=5e-5,targets=target_hashes,
        parent_checkpoint=parent['parent_checkpoint_sha256'],source='previous real RGB only',
        flow='official_flow_bridge/predict_velocity; native Base condition; full UNet',
        inference=dict(steps=28,cfg=1,initial='pure Gaussian'))
    start=0
    if (out/'checkpoint-latest.pt').exists():
        ck=torch.load(out/'checkpoint-latest.pt',map_location='cpu',weights_only=False)
        if ck['binding']!=binding: raise RuntimeError('Resume configuration changed')
        pipe.unet.load_state_dict(ck['unet']);optimizer.load_state_dict(ck['optimizer']);start=ck['step']
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())['binding']!=binding: raise RuntimeError('Endpoint changed')
        return
    write(out/'identity.json',dict(binding=binding,host=socket.gethostname(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    gray=Image.new('RGB',(1024,1024),(128,128,128));contexts={}
    def context(record,position):
        key=(record['id'],position)
        if key not in contexts:
            image=gray
            if position:
                path=a.output/'rollouts'/a.arm/'training'/record['id'].replace(':','-')/f'memory-{position-1:02d}.png'
                image=Image.open(path).convert('RGB')
            with torch.no_grad():
                source=pipe.prepare_image_latents(pipe.image_processor.preprocess(image),dtype=torch.float32,device=a.device)
                cond=encode_native_base_edit_condition(pipe,image,writer_event(record,position),device=a.device,dtype=torch.float32)
            # Many retain contexts can be large: keep detached conditioning on CPU.
            contexts[key]=(source.cpu(),cond.prompt_embeds.cpu(),cond.attention_mask.cpu())
        return tuple(x.to(a.device) for x in contexts[key])
    attempt=str(time.time_ns())
    for step in range(start,a.steps):
        began=time.monotonic();optimizer.zero_grad(set_to_none=True);losses=[]
        for micro in range(4):
            draw=step*4+micro;epoch,offset=divmod(draw,len(selected))
            order=list(range(len(selected)));random.Random(seed_for('fm-order',epoch)).shuffle(order)
            r=selected[order[offset]]
            # Half initial writes / half generated-source retains; balanced per visit.
            pos=0 if a.stage=='write' or epoch%2==0 else 1+(epoch//2)%10
            source,embeds,mask=context(r,pos)
            seed=seed_for('fm-noise',a.stage,draw)
            noise=torch.randn(source.shape,generator=torch.Generator().manual_seed(seed),dtype=torch.float32).to(a.device)
            sigma=random.Random(seed_for('fm-sigma',a.stage,draw)).random()
            state,target_velocity=official_flow_bridge(noise,targets[r['id']],sigma)
            prediction=predict_velocity(sampler,state,source,sigma,embeds,mask,integer_timestep=True)
            loss=(prediction.float()-target_velocity).square().mean()
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite FM loss')
            (loss/4).backward()
            losses.append(dict(id=r['id'],position=pos,sigma=sigma,noise_seed=seed,mse=float(loss.detach())))
        norm=torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite U-Net gradient')
        optimizer.step()
        row=dict(step=step+1,attempt=attempt,draws=losses,grad_norm=float(norm),seconds=time.monotonic()-began)
        append(out/'optimization.jsonl',row)
        if (step+1)%16==0: print(json.dumps(dict(arm=a.arm,stage=a.stage,step=step+1,mse=sum(x['mse'] for x in losses)/4)),flush=True)
        if (step+1)%64==0 or step+1==a.steps:
            atomic_tensor(out/'checkpoint-latest.pt',dict(binding=binding,step=step+1,
                unet=pipe.unet.state_dict(),optimizer=optimizer.state_dict()))
    # A final parameter-only checkpoint suffices for inference; resume keeps Adam.
    atomic_tensor(out/'checkpoint-final.pt',dict(binding=binding,step=a.steps,unet=pipe.unet.state_dict()))
    write(out/'complete.json',dict(binding=binding,checkpoint_sha256=sha(out/'checkpoint-final.pt')))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,default=ROOT/'reports/prefeval-official-alignment-20260923')
    p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--stage',choices=['write','retain'],default='write')
    p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=2048)
    a=p.parse_args()
    try:run(a)
    except BaseException:
        import traceback
        write(a.output/f'failure-fm-{a.arm}-{a.stage}-{int(time.time())}.json',dict(error=traceback.format_exc()))
        raise

if __name__=='__main__':main()
