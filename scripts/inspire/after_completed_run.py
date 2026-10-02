"""Start a serial experiment only after its predecessor exits successfully."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def write(path,value):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--deadline-unix",type=float,required=True)
    p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args()
    command=a.command[1:] if a.command[:1]==["--"] else a.command
    if not command: raise ValueError("Missing next experiment command")
    a.output.mkdir(parents=True,exist_ok=True)
    write(a.output/"identity.json",{"parent":str(a.parent),"command":command,"deadline_unix":a.deadline_unix})
    while not (a.parent/"terminal.json").exists():
        write(a.output/"status.json",{"state":"waiting_for_predecessor","time_unix":time.time()})
        if time.time()>=a.deadline_unix-300: return 75
        time.sleep(30)
    parent=json.loads((a.parent/"terminal.json").read_text())
    if parent.get("state")!="completed":
        write(a.output/"terminal.json",{"state":"predecessor_not_completed","parent_terminal":parent})
        return 2
    result=a.parent/"train/result.json"
    if hashlib.sha256(result.read_bytes()).hexdigest()!=parent["training_result_sha256"]:
        raise ValueError("Predecessor result hash mismatch")
    if time.time()>=a.deadline_unix-300: return 75
    # Let the exited process release its CUDA contexts before checking idle GPUs.
    time.sleep(5)
    write(a.output/"status.json",{"state":"running_next_experiment","time_unix":time.time()})
    with (a.output/"child.log").open("w") as log:
        code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
    write(a.output/"terminal.json",{"state":"completed" if code==0 else "failed","returncode":code,"time_unix":time.time()})
    return code


if __name__=="__main__":
    raise SystemExit(main())
