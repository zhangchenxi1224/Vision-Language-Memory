"""Zero-update empirical Bayes flow control for the failed four-step Writer."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
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
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.open_eos import generation_diagnostics, qwen3vl_answer_eos_ce
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import configure_strict_cuda_determinism
from vision_memory.training.checkpoint import load_trainable_weights
from vision_memory.training.empirical_bank_flow import EmpiricalBankFlow
from vision_memory.training.latent_bank_unet import file_sha256, load_teacher_bank, member_split, predict_velocity, stable_seed


@torch.no_grad()
def run(args):
    bank,teachers=load_teacher_bank(args.bank_manifest)
    assert len(bank['groups'])==1
    os.environ.update(snapshot_environment(bank))
    configure_strict_cuda_determinism(args.seed)
    runtime=training.load_runtime(args,bank)
    load_trainable_weights(args.checkpoint,trainable_module=runtime['pipe'].unet)
    group=bank['groups'][0]
    ctx=runtime['contexts'][group['question_id']]
    source,cond=ctx['source'],ctx['condition']
    ids,held=member_split(group['teacher_ids'])
    targets=torch.stack([teachers[t] for t in ids]).to(source.device)
    flow=EmpiricalBankFlow(source,targets)
    report=dict(status='running',started_epoch=time.time(),hostname=socket.gethostname(),optimizer_updates=0,
        checkpoint_sha256=file_sha256(args.checkpoint),bank_sha256=file_sha256(args.bank_manifest),
        script_sha256=file_sha256(Path(__file__)),helper_sha256=file_sha256(ROOT/'src/vision_memory/training/empirical_bank_flow.py'),
        training_teacher_count=len(ids),heldout_teacher_count=len(held),seed=args.seed,rows=[],
        interpretation='Uses complete TRAIN bank at inference; diagnostic control, not a learned U-Net or evidence of generalization',
        start_irreducible_velocity_mse=float(4*(targets.double()-flow.mean).square().mean()))
    def save(): training.write_json(args.output_dir/'result.json',report)
    save()
    for i in range(8):
        seed=stable_seed(args.seed,'heldout-evaluation-noise',i)
        noise=torch.randn(source.shape,generator=torch.Generator().manual_seed(seed)).to(source.device)
        start=.5*source+.5*noise
        exact=start.clone()
        learned=start.clone()
        path=[]
        for sigma in [.5,.375,.25,.125]:
            exact_v,weights,var=flow.evaluate(exact,sigma)
            learned_v=predict_velocity(runtime['sampler'],learned,source,sigma,cond.prompt_embeds,cond.attention_mask)
            bayes_on_learned,wl,vl=flow.evaluate(learned,sigma)
            learned_on_exact=predict_velocity(runtime['sampler'],exact,source,sigma,cond.prompt_embeds,cond.attention_mask)
            path.append(dict(sigma=sigma,exact_mode_id=ids[int(weights.argmax())],exact_max_posterior=float(weights.max()),
                exact_entropy=float(-(weights*weights.clamp_min(1e-300).log()).sum()),irreducible_mse_on_exact=float(var),
                learned_vs_bayes_mse_on_exact=float((learned_on_exact.double()-exact_v).square().mean()),
                learned_vs_bayes_mse_on_learned=float((learned_v.double()-bayes_on_learned).square().mean()),
                irreducible_mse_on_learned=float(vl),learned_max_posterior=float(wl.max())))
            exact=(exact.double()-.125*exact_v).float()
            learned=learned-.125*learned_v
        distances=(targets.double()-exact.double()).flatten(1).square().mean(1).sqrt()
        nearest=int(distances.argmin())
        pixels=decode_model_latents_unit_interval(runtime['pipe'].vae,exact,clamp=True)
        loss=qwen3vl_answer_eos_ce(model=runtime['reader'],processor=runtime['processor'],
            image=pixels[0].to(runtime['reader_device']),device=runtime['reader_device'],
            query=group['question_variants']['original_open'],target=group['answer'],termination=runtime['termination'],
            lambda_eos=1.,require_image_grad=False,deterministic_ce=True,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
        gen=generate_short_answer(model=runtime['reader'],processor=runtime['processor'],
            image=pixels.to(runtime['reader_device']),query=group['question_variants']['original_open'],
            device=runtime['reader_device'],max_new_tokens=32,do_sample=False)
        score=generation_diagnostics(gen,group['answer'],loss.target_ids[0,:loss.answer_token_count].cpu().tolist())
        row=dict(noise_seed=seed,path=path,raw=gen['raw'],score=score,answer_ce=float(loss.answer_loss),
            eos_ce=float(loss.eos_loss),nearest_teacher_id=ids[nearest],nearest_teacher_rms=float(distances[nearest]))
        report['rows'].append(row)
        training.atomic_tensor(args.output_dir/f'noise-{i:02d}.pt',dict(noise_seed=seed,exact_latent=exact.cpu(),learned_latent=learned.cpu(),image=pixels.cpu()))
        save()
        print(json.dumps({k:v for k,v in row.items() if k!='path'}),flush=True)
    report.update(status='completed',completed_epoch=time.time(),exact_match=sum(r['score']['strict_correct'] for r in report['rows']),
        n=8,unique_nearest_teachers=len({r['nearest_teacher_id'] for r in report['rows']}))
    save()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['bank-manifest','checkpoint','output-dir','dreamlite','reader-model']:
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260908)
    p.add_argument('--lora-rank',type=int,default=4)
    p.add_argument('--dreamlite-device',default='cuda:0')
    p.add_argument('--reader-device',default='cuda:1')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    devices={int(x.split(',')[0]):x.split(',')[1].strip() for x in raw.splitlines()}
    selected={devices[int(d.split(':')[1])] for d in [args.dreamlite_device,args.reader_device]}
    try:
        with ExitStack() as locks:
            for uuid in sorted(selected):
                handle=locks.enter_context((Path('/tmp')/f'vlm-oracle-writer-{uuid}.lock').open('a+'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True)
            assert not(selected & set(active.splitlines())),'Diagnostic GPUs occupied'
            run(args)
    except BaseException:
        training.write_json(args.output_dir/'failure.json',dict(epoch=time.time(),traceback=traceback.format_exc()))
        raise


if __name__=='__main__': main()
