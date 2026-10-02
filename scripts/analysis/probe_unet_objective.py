"""Bounded paired continuation: flow matching versus actual rollout answer+EOS.

Both arms reload exactly the same adapter, reset AdamW identically, and use the
same fresh training noises. This is a single-question diagnostic, not a formal
multi-question result or a replacement for the oracle-to-Writer experiment.
"""
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

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import torch
from scripts.analysis.diagnose_latent_bank_unet import rms
from scripts.train import train_latent_bank_unet as training
from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer
from vision_memory.reader.open_eos import generation_diagnostics, qwen3vl_answer_eos_ce
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT
from vision_memory.repro import configure_strict_cuda_determinism, lora_trainable_parameters
from vision_memory.training.checkpoint import load_trainable_weights, save_training_checkpoint
from vision_memory.training.latent_bank_unet import anchored_flow_bridge, balanced_draw, file_sha256, load_teacher_bank, predict_velocity, stable_seed


def run(args):
    bank, teachers = load_teacher_bank(args.bank_manifest)
    assert len(bank["groups"]) == 1
    os.environ.update(snapshot_environment(bank))
    configure_strict_cuda_determinism(args.seed)
    runtime = training.load_runtime(args, bank)
    group = bank["groups"][0]
    ctx = runtime["contexts"][group["question_id"]]
    source, cond = ctx["source"], ctx["condition"]
    pipe, reader = runtime["pipe"], runtime["reader"]
    runtime["sampler"].checkpoint_unet = True
    saved = load_trainable_weights(args.checkpoint, trainable_module=pipe.unet)
    assert saved["optimizer_step"] == 512
    parameters = lora_trainable_parameters(pipe.unet)
    initial = {n:p.detach().cpu().clone() for n,p in pipe.unet.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
    versions = training.frozen_versions(pipe, reader)
    report = {"status":"running", "arm":args.arm, "hostname":socket.gethostname(),
        "started_epoch":time.time(), "starting_optimizer_step":512, "additional_updates":0,
        "optimizer_policy":"fresh identical AdamW state in both arms; not an exact optimizer resume",
        "lr":args.lr, "steps":args.steps, "seed":args.seed, "question_count":1,
        "bank_sha256":file_sha256(args.bank_manifest), "checkpoint_sha256":file_sha256(args.checkpoint),
        "script_sha256":file_sha256(Path(__file__)), "evaluations":[], "losses":[],
        "training_prompt_ids":["original_open"], "heldout_scope":"new noise and four paraphrases; same question",
        "generation":{"do_sample":False,"max_new_tokens":32}, "deadline_unix":args.deadline_unix}
    def write():
        training.write_json(args.output_dir / "result.json", report)
    def generate_latent(noise):
        return runtime["sampler"](source_latents=source, noise_latents=noise,
            prompt_embeds=cond.prompt_embeds, prompt_attention_mask=cond.attention_mask,
            num_steps=4, edit_start_sigma=.5, return_trajectory=False).latents
    def qa_loss(pixels, query, grad):
        return qwen3vl_answer_eos_ce(model=reader, processor=runtime["processor"],
            image=pixels[0].to(runtime["reader_device"]),device=runtime["reader_device"],
            query=query,target=group["answer"],termination=runtime["termination"],lambda_eos=1.,
            require_image_grad=grad, deterministic_ce=True,reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
    @torch.no_grad()
    def evaluate(step, full=False):
        rows=[]
        images=[]
        for i in range(8 if full else 2):
            seed=stable_seed(args.seed,"heldout-evaluation-noise",i)
            noise=torch.randn(source.shape,generator=torch.Generator().manual_seed(seed)).to(source.device)
            latent=generate_latent(noise)
            pixels=decode_model_latents_unit_interval(pipe.vae,latent,clamp=True)
            images.append(("matched",seed,pixels.cpu()))
            if full:
                training.atomic_tensor(args.output_dir/f"final-noise-{i:02d}.pt",{"noise_seed":seed,"latent":latent.cpu(),"image":pixels.cpu()})
        if full:
            images += [("blank",None,ctx["blank"]),("donor",None,ctx["donor"])]
        prompts=group["question_variants"] if full else {"original_open":group["question_variants"]["original_open"]}
        for kind, seed, pixels in images:
            for pid, query in prompts.items():
                scores=qa_loss(pixels.to(source.device),query,False)
                gen=generate_short_answer(model=reader,processor=runtime["processor"],
                    image=pixels.to(runtime["reader_device"]),query=query,device=runtime["reader_device"],
                    max_new_tokens=32,do_sample=False)
                score=generation_diagnostics(gen,group["answer"],scores.target_ids[0,:scores.answer_token_count].cpu().tolist())
                rows.append({"question_id":group["question_id"],"condition":kind,"noise_seed":seed,"prompt_id":pid,"raw":gen["raw"],
                    "scorer":score,"answer_ce":float(scores.answer_loss),"eos_ce":float(scores.eos_loss)})
        evaluation={"additional_step":step,"rows":rows,"cells":training.summarize_evaluation(rows,{})["cells"]}
        report["evaluations"].append(evaluation)
        write()
        print(json.dumps({"stage":"evaluation","arm":args.arm,"step":step,"cells":evaluation["cells"]}),flush=True)
    write()
    evaluate(0)
    started=time.monotonic()
    for step in range(args.steps):
        if time.time()>=args.deadline_unix-120:
            report.update(status="paused",reason="bounded diagnostic deadline")
            break
        _,tid,noise_seed,sigma=balanced_draw(bank["groups"],args.seed,512+step)
        noise=torch.randn(source.shape,generator=torch.Generator().manual_seed(noise_seed)).to(source.device)
        optimizer.zero_grad(set_to_none=True)
        details={}
        if args.arm=="flow":
            target=teachers[tid].to(source.device)
            state,velocity=anchored_flow_bridge(source,noise,target,sigma)
            pred=predict_velocity(runtime["sampler"],state,source,sigma,cond.prompt_embeds,cond.attention_mask)
            loss=(pred-velocity).square().mean()
            details={"teacher_id":tid,"sigma":sigma}
        else:
            latent=generate_latent(noise)
            pixels=decode_model_latents_unit_interval(pipe.vae,latent,clamp=True)
            values=qa_loss(pixels,group["question_variants"]["original_open"],True)
            loss=values.loss
            details={"answer_ce":float(values.answer_loss.detach()),"eos_ce":float(values.eos_loss.detach())}
        assert torch.isfinite(loss)
        loss.backward()
        grads=[p.grad for p in parameters if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any((g!=0).any() for g in grads)
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        training.frozen_audit(pipe,reader,versions)
        row={"step":step+1,"noise_seed":noise_seed,"loss":float(loss.detach()),"grad_norm_before_clip":norm,
             "elapsed_seconds":time.monotonic()-started,**details}
        report["losses"].append(row)
        report["additional_updates"]=step+1
        if args.arm=="flow":
            del pred, target, state, velocity
        else:
            del latent, pixels, values
        del loss, grads
        write()
        if step==0 or (step+1)%8==0:
            print(json.dumps({"stage":"training","arm":args.arm,**row}),flush=True)
        if (step+1)%16==0 or step+1==args.steps:
            save_training_checkpoint(args.output_dir/"checkpoint-latest.pt",trainable_module=pipe.unet,
                optimizer=optimizer,epoch=0,episode_cursor=step+1,optimizer_step=512+step+1,
                manifest={k:report[k] for k in ["arm","checkpoint_sha256","bank_sha256","script_sha256","seed","lr"]})
        if step+1 in (16,32) and step+1<args.steps:
            evaluate(step+1)
    evaluate(report["additional_updates"],full=True)
    training.frozen_audit(pipe,reader,versions)
    report["adapter_delta_l2"]=sum(float((p.detach().cpu()-initial[n]).double().square().sum()) for n,p in pipe.unet.named_parameters() if p.requires_grad)**.5
    report["frozen_audit_passed"]=True
    report["completed_epoch"]=time.time()
    if report["additional_updates"]==args.steps:
        report["status"]="completed"
    write()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["bank-manifest","checkpoint","output-dir","dreamlite","reader-model"]:
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--arm",choices=["flow","qa_eos"],required=True)
    p.add_argument("--seed",type=int,default=20260908)
    p.add_argument("--lora-rank",type=int,default=4)
    p.add_argument("--steps",type=int,default=64)
    p.add_argument("--lr",type=float,default=1e-4)
    p.add_argument("--dreamlite-device",default="cuda:0")
    p.add_argument("--reader-device",default="cuda:1")
    p.add_argument("--deadline-unix",type=float,required=True)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    raw=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"],text=True)
    devices={int(x.split(',')[0]):x.split(',')[1].strip() for x in raw.splitlines()}
    selected={devices[int(d.split(':')[1])] for d in [args.dreamlite_device,args.reader_device]}
    try:
        with ExitStack() as locks:
            for uuid in sorted(selected):
                handle=locks.enter_context((Path('/tmp')/f'vlm-oracle-writer-{uuid}.lock').open('a+'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            active=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"],text=True)
            assert not(selected & set(active.splitlines())),"Diagnostic GPUs already occupied"
            run(args)
    except BaseException:
        training.write_json(args.output_dir/"failure.json",{"traceback":traceback.format_exc(),"epoch":time.time()})
        raise


if __name__=="__main__":
    main()
