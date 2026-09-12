"""Run one bound official-FM pilot on an already allocated H200 instance."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.inspire.run_oracle_to_unet_pipeline import snapshot_environment
from scripts.train.train_latent_bank_unet import write_json
from vision_memory.training.latent_bank_unet import load_teacher_bank, file_sha256
from vision_memory.repro.determinism import REQUIRED_DETERMINISM_ENV


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-manifest", type=Path, required=True)
    p.add_argument("--bank-sha256", required=True)
    p.add_argument("--dreamlite", type=Path, required=True)
    p.add_argument("--model-variant",choices=("mobile","base"),default="mobile")
    p.add_argument("--teacher-dreamlite",type=Path)
    p.add_argument("--official-source",type=Path)
    p.add_argument("--base-manifest",type=Path)
    p.add_argument("--base-guidance-scale",type=float,default=7.5)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--target-mode", choices=("single","bank"), default="single")
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--trainable-scope",choices=("lora","full_unet"),default="lora")
    p.add_argument("--checkpoint-interval",type=int,default=1)
    p.add_argument("--colocate-models",action="store_true")
    p.add_argument("--baseline-reference",type=Path)
    p.add_argument("--baseline-reference-result-sha256")
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--resume", action="store_true")
    a=p.parse_args()
    if not 1.0<=a.base_guidance_scale<=100.0 or (a.model_variant!="base" and a.base_guidance_scale!=7.5):
        raise ValueError("Base guidance must be in [1,100] and applies only to Base")
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=a.expected_commit:
        raise RuntimeError("Wrong checkout")
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise RuntimeError("Dirty checkout")
    if file_sha256(a.bank_manifest)!=a.bank_sha256:
        raise RuntimeError("Wrong bank")
    bank,_=load_teacher_bank(a.bank_manifest)
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.used","--format=csv,noheader,nounits"],text=True)
    rows=[r.split(",") for r in gpu.strip().splitlines()]
    expected_gpus=1 if a.colocate_models else 2
    if len(rows)!=expected_gpus or any("H200" not in r[0] or int(r[1])<140000 or int(r[2])>100 for r in rows):
        raise RuntimeError(f"Need {expected_gpus} idle full-memory H200s: {gpu}")
    env={**os.environ, **snapshot_environment(bank), **REQUIRED_DETERMINISM_ENV,
         "PYTHONUNBUFFERED":"1", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    import torch, diffusers, transformers
    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        raise RuntimeError("CUDA 12.8 runtime required")
    binding={"commit":a.expected_commit,"bank_sha256":a.bank_sha256,"torch":torch.__version__,
        "diffusers":diffusers.__version__,"transformers":transformers.__version__,"gpu":gpu,
        "cuda":torch.version.cuda,"target_mode":a.target_mode,"steps":a.steps,"seed":a.seed,
        "trainable_scope":a.trainable_scope,"checkpoint_interval":a.checkpoint_interval,
        "colocate_models":a.colocate_models,"base_guidance_scale":a.base_guidance_scale if a.model_variant=="base" else None,
        "created_unix":time.time(),"deadline_unix":a.deadline_unix}
    write_json(a.output_dir/"dispatch.json",binding)
    parity=a.output_dir/"parity.json"
    commands=[]
    if a.model_variant=="base":
        if not all((a.teacher_dreamlite,a.official_source,a.base_manifest)):
            raise ValueError("Base needs its sealed snapshot, teacher decoder and official source")
        # Base evaluation executes the upstream pipeline itself; there is no
        # copied denoising core whose numerical parity needs to be established.
    elif not parity.exists():
        commands.append([sys.executable,"-u",str(ROOT/"scripts/probes/dreamlite_parity.py"),
            "--model",str(a.dreamlite),"--dtype","float32","--strict-determinism",
            "--atol","0","--rtol","0","--output-json",str(parity)])
    else:
        previous=json.loads(parity.read_text())
        if previous.get("allclose") is not True:
            raise RuntimeError("Previous parity failed")
    train=[sys.executable,"-u",str(ROOT/"scripts/train/train_latent_bank_unet.py"),
        "--bank-manifest",str(a.bank_manifest),"--output-dir",str(a.output_dir/"train"),
        "--dreamlite",str(a.dreamlite),"--reader-model",str(a.reader),"--expected-commit",a.expected_commit,
        "--flow-protocol","official","--prompt-style","official_raw","--target-mode",a.target_mode,
        "--steps",str(a.steps),"--seed",str(a.seed),"--lora-rank","16","--lr","5e-5",
        "--gradient-accumulation-steps","4","--weight-decay","1e-4","--eval-seeds","8",
        "--deadline-unix",str(a.deadline_unix)]
    train.extend(["--model-variant",a.model_variant])
    train.extend(["--trainable-scope",a.trainable_scope,"--checkpoint-interval",str(a.checkpoint_interval)])
    if a.colocate_models:
        train.extend(["--colocate-models","--reader-device","cuda:0"])
    if a.baseline_reference:
        if not a.baseline_reference_result_sha256:
            raise ValueError("Baseline reference requires its verified result SHA")
        train.extend(["--baseline-reference",str(a.baseline_reference),
                      "--baseline-reference-result-sha256",a.baseline_reference_result_sha256])
    if a.model_variant=="base":
        train.extend(["--teacher-dreamlite",str(a.teacher_dreamlite),"--official-source",str(a.official_source),
                      "--base-manifest",str(a.base_manifest),"--base-guidance-scale",str(a.base_guidance_scale)])
    if a.resume: train.append("--resume")
    commands.append(train)
    write_json(a.output_dir/"commands.json",{"commands":commands})
    for i,command in enumerate(commands):
        write_json(a.output_dir/"status.json",{"state":"running","command":command,"time_unix":time.time()})
        with (a.output_dir/f"stage-{i}-{time.time_ns()}.log").open("w") as log:
            completed=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if completed.returncode:
            write_json(a.output_dir/"terminal.json",{"state":"paused" if completed.returncode==75 else "failed",
                "returncode":completed.returncode,"command":command,"time_unix":time.time()})
            return completed.returncode
    result=json.loads((a.output_dir/"train/result.json").read_text())
    write_json(a.output_dir/"terminal.json",{"state":"completed","training_result_sha256":file_sha256(a.output_dir/"train/result.json"),
        "matched_all_five_prompts_correct":result["trained"]["matched_all_five_prompts_correct"],
        "matched_all_five_prompts_answer_eos":result["trained"]["matched_all_five_prompts_answer_eos"],
        "matched_question_noise_pairs":result["trained"]["matched_question_noise_pairs"],
        "scope":"single-question regression evaluation; not evidence of event-conditioned generalization", "time_unix":time.time()})
    return 0


if __name__=="__main__":
    raise SystemExit(main())
