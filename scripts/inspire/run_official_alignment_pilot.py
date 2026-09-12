"""Run one bound official-FM pilot on an already allocated two-H200 instance."""
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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-manifest", type=Path, required=True)
    p.add_argument("--bank-sha256", required=True)
    p.add_argument("--dreamlite", type=Path, required=True)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--target-mode", choices=("single","bank"), default="single")
    p.add_argument("--steps", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260913)
    p.add_argument("--deadline-unix", type=float, required=True)
    p.add_argument("--resume", action="store_true")
    a=p.parse_args()
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()!=a.expected_commit:
        raise RuntimeError("Wrong checkout")
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise RuntimeError("Dirty checkout")
    if file_sha256(a.bank_manifest)!=a.bank_sha256:
        raise RuntimeError("Wrong bank")
    bank,_=load_teacher_bank(a.bank_manifest)
    gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.total,memory.used","--format=csv,noheader,nounits"],text=True)
    rows=[r.split(",") for r in gpu.strip().splitlines()]
    if len(rows)!=2 or any("H200" not in r[0] or int(r[1])<140000 or int(r[2])>100 for r in rows):
        raise RuntimeError(f"Need two idle full-memory H200s: {gpu}")
    env={**os.environ, **snapshot_environment(bank), "PYTHONUNBUFFERED":"1",
         "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1", "CUBLAS_WORKSPACE_CONFIG":":4096:8"}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    import torch, diffusers, transformers
    if not torch.cuda.is_available() or torch.version.cuda != "12.8":
        raise RuntimeError("CUDA 12.8 runtime required")
    binding={"commit":a.expected_commit,"bank_sha256":a.bank_sha256,"torch":torch.__version__,
        "diffusers":diffusers.__version__,"transformers":transformers.__version__,"gpu":gpu,
        "cuda":torch.version.cuda,"target_mode":a.target_mode,"steps":a.steps,"seed":a.seed,
        "created_unix":time.time(),"deadline_unix":a.deadline_unix}
    write_json(a.output_dir/"dispatch.json",binding)
    parity=a.output_dir/"parity.json"
    commands=[]
    if not parity.exists():
        commands.append([sys.executable,"-u",str(ROOT/"scripts/probes/dreamlite_parity.py"),
            "--model",str(a.dreamlite),"--dtype","float32","--atol","0","--rtol","0","--output-json",str(parity)])
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
        "matched_question_noise_pairs":result["trained"]["matched_question_noise_pairs"],
        "scope":"single-question regression evaluation; not evidence of event-conditioned generalization", "time_unix":time.time()})
    return 0


if __name__=="__main__":
    raise SystemExit(main())
