"""Independent CPU sidecar: fresh oracle completion -> audited set -> U-Net arm.

One process is attached to one route and its current host allocation. It does
not modify an active oracle checkout or take a GPU before oracle completion.
Resource extensions are explicit lease-file updates; expired allocations never
cause this process to start another platform instance.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import ctypes
import hashlib
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
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from vision_memory.training.latent_teacher_bank import (AuditError, build_bank, campaign_progress,
    file_sha256, planned_runs, read_json, require, write_json)


def command_output(command):
    return subprocess.check_output(command,cwd=ROOT,text=True).strip()


def own_source(expected_commit):
    require(command_output(["git","rev-parse","HEAD"]) == expected_commit,"Sidecar/trainer checkout commit changed")
    require(not command_output(["git","status","--porcelain"]),"Sidecar/trainer checkout is dirty")
    return {p.relative_to(ROOT).as_posix():file_sha256(p) for base in (ROOT/"src",ROOT/"scripts") for p in sorted(base.rglob("*.py"))}


def source_unchanged(expected):
    for relative,sha in expected.items():
        require(file_sha256(ROOT/relative) == sha,"Sidecar/trainer source changed during operation")


def gpu_snapshot():
    raw=command_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader,nounits"])
    devices={int(line.split(",",1)[0].strip()):line.split(",",1)[1].strip() for line in raw.splitlines() if line.strip()}
    raw=command_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"])
    active=[]
    for line in raw.splitlines():
        if line.strip():
            uuid,pid=(part.strip() for part in line.split(",",1))
            active.append({"uuid":uuid,"pid":int(pid)})
    return devices,active


def validate_allocation(runtime,devices):
    require(socket.gethostname() == runtime["expected_hostname"],"Sidecar is on a different host allocation")
    selected=runtime["gpu_indices"]
    require(len(selected) == 2 and len(set(selected)) == 2,"One Writer arm requires exactly two distinct GPUs")
    require([devices.get(index) for index in selected] == runtime["gpu_uuids"],"GPU UUID/host ownership changed")
    return set(runtime["gpu_uuids"])


def lease_remaining(config):
    lease=read_json(Path(config["resource_lease_path"]))
    require(lease["expected_hostname"] == config["runtime"]["expected_hostname"],"Resource lease belongs to another host")
    require(lease["gpu_uuids"] == config["runtime"]["gpu_uuids"],"Resource lease GPU identity changed")
    deadline=float(lease["deadline_epoch"])
    require(deadline > 0,"Resource deadline must be explicit and positive")
    return deadline-time.time(),lease


def trainer_command(config,bank_path,training_dir,deadline):
    t=config["trainer"]
    command=[sys.executable,str(ROOT/"scripts/train/train_latent_bank_unet.py"),
        "--bank-manifest",str(bank_path),"--output-dir",str(training_dir),
        "--dreamlite",t["dreamlite"],"--reader-model",t["reader_model"],
        "--dreamlite-device","cuda:0","--reader-device","cuda:1",
        "--expected-commit",config["expected_commit"],"--steps",str(t.get("steps",512)),
        "--seed",str(t.get("seed",20260908)),"--deadline-unix",str(deadline)]
    if (training_dir/"checkpoint-latest.pt").exists():
        command.append("--resume")
    return command


def snapshot_environment(bank):
    bindings=bank["models"]
    names={"dreamlite_mobile":"VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256",
           "qwen_reader":"VLM_READER_SNAPSHOT_MANIFEST_SHA256"}
    result={}
    for model,name in names.items():
        binding=bindings[model]
        sha=binding["manifest_sha256"]
        require(binding.get("passed") is True and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),"Missing verified model-manifest SHA")
        result[name]=sha
    return result


def resume_direct_oracle(config_path,deadline):
    """Explicit lifecycle action after the user extends the same allocation.

    A lease-file update alone does not resume an oracle. This entry point must
    be called with a later deadline; it invokes the unchanged original commit.
    """
    import fcntl
    config=read_json(config_path)
    require(config["route"] == "direct","This continuation helper is for the Direct two-lane campaign")
    own_source(config["expected_commit"])
    oracle=config["oracle"]
    repo,root=Path(oracle["repo"]),Path(oracle["root"])
    require(socket.gethostname() == config["runtime"]["expected_hostname"],"Direct allocation host changed")
    require(subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip() == oracle["commit"],"Original oracle source commit changed")
    require(not subprocess.check_output(["git","status","--porcelain"],cwd=repo,text=True).strip(),"Original oracle checkout is dirty")
    require(read_json(root/"terminal.json")["status"] == "paused_at_run_boundary","Only a clean paused Direct campaign can continue")
    old_deadline=read_json(root/"launch.json")["deadline_epoch"]
    _,lease=lease_remaining(config)
    require(deadline > old_deadline and deadline <= lease["deadline_epoch"] and deadline-time.time() > 900,"Continuation deadline must be later, within explicitly extended lease, and leave >=900 seconds")
    for lane in range(2):
        require(read_json(root/f"lane-{lane}"/"terminal.json")["status"] in ("completed","paused_at_run_boundary"),"A failed or interrupted lane cannot be silently continued")
        identity=read_json(root/f"lane-{lane}"/"identity.json")
        require(identity["commit"] == oracle["commit"] and identity["config_sha256"] == file_sha256(Path(oracle["config"])),"Direct lane identity drifted")
    devices,active=gpu_snapshot()
    indices=oracle["resume_gpu_indices"]
    uuids=oracle["resume_gpu_uuids"]
    require(indices == [0,1,2,3] and len(uuids) == 4 and len(set(uuids)) == 4 and [devices.get(i) for i in indices] == uuids,"Direct resume requires the same four explicitly bound GPUs")
    require(not any(p["uuid"] in uuids for p in active),"Direct resume GPUs are occupied")
    with ExitStack() as locks:
        for uuid in sorted(uuids):
            handle=locks.enter_context((Path("/tmp")/f"vlm-oracle-writer-{uuid}.lock").open("a+"))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Check the real campaign owner is absent, then let the original launcher
        # take its own lock. Concurrent continuation commands cannot pass the GPU locks.
        with (root/".campaign.lock").open("a+") as campaign:
            fcntl.flock(campaign,fcntl.LOCK_EX|fcntl.LOCK_NB)
        command=[sys.executable,"-u",str(repo/"scripts/inspire/run_direct_geometry_h200x4.py"),
                 "--output-root",str(root),"--expected-commit",oracle["commit"],"--deadline",str(deadline)]
        write_json(Path(config["output_root"])/"oracle_resume_command.json",{"command":command,"lease":lease,"source_commit_unchanged":oracle["commit"]})
        env=os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES",None)
        return subprocess.call(command,cwd=repo,env=env)


def parent_death_signal():
    # Ensure a killed sidecar cannot leave an untracked GPU child. The source is
    # Linux-only and this hook runs in a single-threaded process before exec.
    parent=os.getppid()
    libc=ctypes.CDLL(None)
    if libc.prctl(1,signal.SIGTERM) != 0 or os.getppid() != parent:
        os._exit(126)


def run_pipeline(config_path,*,once=False,poll_seconds=30):
    import fcntl
    require(1 <= poll_seconds <= 60,"Poll interval must be 1..60 seconds")
    config=read_json(config_path)
    require(config.get("schema") == "vision_memory.oracle-to-unet-pipeline.v1","Wrong pipeline configuration schema")
    output=Path(config["output_root"])
    output.mkdir(parents=True,exist_ok=True)
    identity={"config_sha256":file_sha256(config_path),"commit":config["expected_commit"],"route":config["route"],
              "expected_hostname":config["runtime"]["expected_hostname"]}
    if (output/"identity.json").exists():
        require(read_json(output/"identity.json") == identity,"Pipeline output belongs to another immutable configuration")
    else:
        write_json(output/"identity.json",identity)
    with (output/".pipeline.lock").open("a+") as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        sources=own_source(config["expected_commit"])
        stop=False
        child=None
        def requested_stop(signum,frame):
            nonlocal stop
            stop=True
            if child is not None and child.poll() is None:
                child.terminate()
        old_handlers={signum:signal.signal(signum,requested_stop) for signum in (signal.SIGTERM,signal.SIGINT)}
        try:
            oracle=config["oracle"]
            bank_dir=output/"bank"
            training_dir=output/"unet"
            specs=planned_runs(read_json(Path(oracle["config"])),read_json(Path(oracle["planned_manifest"])) if oracle.get("planned_manifest") else None,config["route"])
            while not stop:
                require(file_sha256(config_path) == identity["config_sha256"],"Immutable pipeline config changed")
                source_unchanged(sources)
                progress=campaign_progress(Path(oracle["root"]),config["route"],specs)
                if progress["failures"]:
                    raise AuditError("Oracle failed; preserving artifacts: "+str(progress["failures"]))
                if not progress["complete"]:
                    write_json(output/"status.json",{**identity,"state":"waiting_for_oracle","oracle_progress":progress,"unet_training_started":False})
                    if once:
                        return 11
                    time.sleep(poll_seconds)
                    continue
                write_json(output/"status.json",{**identity,"state":"auditing_and_analyzing_bank","oracle_progress":progress,"unet_training_started":False})
                bank=build_bank(route=config["route"],oracle_root=Path(oracle["root"]),oracle_repo=Path(oracle["repo"]),
                    oracle_config=Path(oracle["config"]),oracle_commit=oracle["commit"],output_dir=bank_dir,
                    planned_manifest=Path(oracle["planned_manifest"]) if oracle.get("planned_manifest") else None)
                if bank.get("status") == "blocked_no_success":
                    write_json(output/"status.json",{**identity,"state":"blocked_no_success","reason":bank["reason"],"unet_training_started":False})
                    return 10
                require(bank.get("status") == "sealed","Only a sealed completed bank can train U-Net")
                if (training_dir/"terminal.json").exists():
                    terminal=read_json(training_dir/"terminal.json")
                    if terminal.get("status") == "completed":
                        write_json(output/"status.json",{**identity,"state":"completed","unet_training_started":True,"trainer_terminal":terminal})
                        return 0
                    require(terminal.get("status") in ("paused","interrupted"),"Previous trainer failure requires investigation; not silently retried")
                remaining,lease=lease_remaining(config)
                if remaining < config.get("minimum_training_seconds",900):
                    write_json(output/"status.json",{**identity,"state":"waiting_for_resource_extension","lease":lease,"bank_sealed":True,"unet_training_started":False})
                    if once:
                        return 12
                    time.sleep(poll_seconds)
                    continue
                devices,active=gpu_snapshot()
                selected=validate_allocation(config["runtime"],devices)
                busy=[process for process in active if process["uuid"] in selected]
                if busy:
                    write_json(output/"status.json",{**identity,"state":"waiting_for_gpu_release","active_processes":busy,"bank_sealed":True,"unet_training_started":False})
                    if once:
                        return 13
                    time.sleep(poll_seconds)
                    continue
                with ExitStack() as locks:
                    for uuid in sorted(selected):
                        require(all(character.isalnum() or character in "-_" for character in uuid),"Invalid GPU UUID")
                        handle=locks.enter_context((Path("/tmp")/f"vlm-oracle-writer-{uuid}.lock").open("a+"))
                        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    # Recheck after taking cooperative locks, before allocating models.
                    devices,active=gpu_snapshot()
                    validate_allocation(config["runtime"],devices)
                    require(not any(process["uuid"] in selected for process in active),"GPU occupied during launch race; no child was started")
                    command=trainer_command(config,bank_dir/"manifest.json",training_dir,lease["deadline_epoch"])
                    env=os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=",".join(str(index) for index in config["runtime"]["gpu_indices"]),
                               PYTHONHASHSEED="0",CUBLAS_WORKSPACE_CONFIG=":4096:8",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
                               TOKENIZERS_PARALLELISM="false",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
                    env.update(snapshot_environment(bank))
                    with (output/"unet.log").open("a",encoding="utf-8") as log:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=parent_death_signal)
                        write_json(output/"status.json",{**identity,"state":"training_unet","pid":child.pid,"command":command,"unet_training_started":True,"bank_manifest_sha256":file_sha256(bank_dir/"manifest.json")})
                        while child.poll() is None and not stop:
                            left,_=lease_remaining(config)
                            if left < 120:
                                child.terminate()
                                stop=True
                                break
                            time.sleep(min(poll_seconds,10))
                        try:
                            code=child.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            code=child.wait()
                        child=None
                    source_unchanged(sources)
                    if stop:
                        write_json(output/"status.json",{**identity,"state":"paused","reason":"Host lease ending or stop signal; checkpoint required before resume","returncode":code,"unet_training_started":True})
                        return 14
                    terminal=read_json(training_dir/"terminal.json")
                    if code == 75 and terminal.get("status") == "paused":
                        write_json(output/"status.json",{**identity,"state":"paused","reason":"Trainer reached explicit resource deadline","unet_training_started":True,"trainer_terminal":terminal})
                        return 14
                    require(code == 0,"U-Net training process failed; see unet.log")
                    require(terminal.get("status") == "completed","Trainer exited without verified completed terminal")
                    write_json(output/"status.json",{**identity,"state":"completed","unet_training_started":True,"trainer_terminal":terminal})
                    return 0
            write_json(output/"status.json",{**identity,"state":"paused","reason":"Stop signal before U-Net launch","unet_training_started":False})
            return 14
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.terminate()
            write_json(output/"status.json",{**identity,"state":"failed","error":str(error),"traceback":traceback.format_exc()})
            raise
        finally:
            for signum,handler in old_handlers.items():
                signal.signal(signum,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--once",action="store_true",help="Inspect/advance once; waiting is a nonzero status")
    parser.add_argument("--poll-seconds",type=int,default=30)
    parser.add_argument("--resume-direct-oracle",action="store_true",help="Explicitly continue the unchanged paused oracle after resource extension")
    parser.add_argument("--deadline",type=float,help="Later oracle deadline, within the explicit host lease")
    args=parser.parse_args()
    if args.resume_direct_oracle:
        parser.error("--resume-direct-oracle requires --deadline") if args.deadline is None else None
        return resume_direct_oracle(args.config,args.deadline)
    return run_pipeline(args.config,once=args.once,poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
