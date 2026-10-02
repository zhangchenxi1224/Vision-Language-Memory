from pathlib import Path
import os,json,datetime,subprocess,fcntl,hashlib
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936")
r=p/"runs/prefeval-b-mcq-20260925"
out=r/"normal-migration-20260927"
repo=p/"repos/prefeval-b-refresh-5559681"
expected="55596810cee2062022b894ef93feb1ab4cd6b07b"
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()==expected
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=p/"repos/prefeval-b-deploy-52d98d8",text=True).strip().startswith("52d98d8")
assert (out/"stopped.json").exists()
assert not (out/"launch.json").exists(),"Already launched; inspect live processes before any retry"
assert any(os.uname().nodename.startswith(n+"--") for n in ["b-cr4-normal-0927","prefeval-b-cr-h200x4-20260926"])
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
for d in Path("/proc").iterdir():
 if not d.name.isdigit() or int(d.name)==os.getpid():continue
 try:cmd=(d/"cmdline").read_bytes().split(bytes([0]))
 except OSError:continue
 assert not (len(cmd)>1 and b"prefeval" in cmd[1]),"Another PrefEval process is alive"
for arm in ["C","R"]:
 with (r/("retain730-"+arm)/"refresh-driver.lock").open("a") as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
 return h.hexdigest()
assert sha(r/"robust730/train/checkpoint-final.pt")=="6c8eb92f629ccdf6932407124190d651e8231058ef6f232a0a59380370d44c8d"
assert sha(r/"sourcebank730/bank.json")=="fc0b8c6d79949c765867ffeccfc328422be4bfd46ea994df63aecb9d5fe2ddd4"

assert sha(r/"retain730-R/train/checkpoint-step-011680.pt")=="8300e2eb389445a7819e8540f35b6b5d0d768e3db52b19f9d93a31592c4da1d3"
assert sha(r/"retain730-R/sources/round-1/bank.json")=="1fe8cf804b8bf12afcf2240f1d4eb65eb61c0cadd19dc542b3fb511cf66c74f8"
import torch
resume=torch.load(r/"retain730-R/train/resume.pt",map_location="cpu",mmap=True,weights_only=False)
rs=resume["optimizer_step"]
assert rs == 11680
assert resume["episode_cursor"]==4*rs
osteps=sorted({int(v["step"].item()) for v in resume["optimizer"]["state"].values()})
assert osteps==[rs]
assert resume["trainer_state"]["refresh_round"]==1
assert sha(r/"retain730-C/train/checkpoint-final.pt")=="3d67d91f4e26bebc503f95e697d61b9d13453738be5d8a4e90e898cef4372ced"
audit={"step":rs,"cursor":resume["episode_cursor"],"optimizer_steps":osteps,"optimizer_parameters":len(resume["optimizer"]["state"]),"rng_keys":list(resume["rng_state"]),"trainer_state":resume["trainer_state"],"sha256":sha(r/"retain730-R/train/resume.pt")}
(out/"resume-before.json").write_text(json.dumps(audit,indent=2)+"\n")
receipts=[]
for arm,gpus in [("C",[1]),("R",[2,3])]:
 command=[str(p/"envs/vlm-r3-ngc2502/bin/python"),str(repo/"scripts/inspire/run_prefeval_k1_refresh730.py"),"--arm",arm,"--gpus"]+list(map(str,gpus))
 log=r/("retain730-"+arm)/"driver-normal-migration-20260927.log"
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
 with log.open("a") as stream:
  proc=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record={"arm":arm,"pid":proc.pid,"host":os.uname().nodename,"command":command,"gpus":gpus,"code_root":str(repo),"commit":expected,"log":str(log),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"reason":"user requested NORMAL priority migration; preserve completed stages and continue remaining frozen budget"}
 (r/("retain730-"+arm)/"driver-launch.json").write_text(json.dumps(record,indent=2)+"\n")
 receipts.append(record)
 (out/"launch.json").write_text(json.dumps(receipts,indent=2)+"\n")
print(json.dumps(receipts,indent=2))
