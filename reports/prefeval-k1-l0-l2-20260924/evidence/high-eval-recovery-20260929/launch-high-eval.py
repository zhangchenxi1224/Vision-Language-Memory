from pathlib import Path
import os,json,datetime,subprocess,fcntl,hashlib
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936");r=p/"runs/prefeval-b-mcq-20260925"
out=r/"high-eval-recovery-20260929";before=r/"interruption-20260929-0915";repo=p/"repos/prefeval-b-refresh-5559681"
expected="55596810cee2062022b894ef93feb1ab4cd6b07b"
assert os.uname().nodename=="prefeval-k1-h200x4-high-20260924--506d5fcf84c1-c7a2kxozom"
assert not (out/"launch.json").exists(),"Already launched"
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()==expected
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=p/"repos/prefeval-b-deploy-52d98d8",text=True).strip()=="52d98d85f84e1e4910e922e34f59f5f14bbabf5c"
assert json.loads((before/"archive-sha256.json").read_text())["sha256"]=="5b5d5e32f642957afa235c9589a8e6c30a660cae59ac961ff901085bf18c7eac"
audit=json.loads((out/"resume-before.json").read_text())
assert audit["step"]==23360 and audit["cursor"]==93440 and audit["optimizer_steps"]==[23360]
def sha(f):
 h=hashlib.sha256()
 with f.open("rb") as s:
  for b in iter(lambda:s.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
summary=json.loads((before/"summary.json").read_text())
for n,h in summary["hashes"].items():assert sha(r/n)==h,n
for item in json.loads((before/"png-inventory.json").read_text()):
 f=r/item["path"];assert f.stat().st_size==item["bytes"] and f.stat().st_mtime_ns==item["mtime_ns"],str(f)
for name,item in summary["readbacks"].items():assert sha(r/name)==item["sha256"],name
assert json.loads((r/"retain730-R/train/complete.json").read_text())=={"steps":23360,"checkpoint_sha256":summary["hashes"]["retain730-R/train/checkpoint-final.pt"]}
for k in range(4):
 done=json.loads((r/("retain730-R/train/round-"+str(k)+"-complete.json")).read_text())
 assert done["step"]==(k+1)*5840
 assert sha(r/("retain730-R/train/checkpoint-step-%06d.pt"%done["step"]))==done["checkpoint_sha256"]
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(),"GPU occupied"
for d in Path("/proc").iterdir():
 if not d.name.isdigit() or int(d.name)==os.getpid():continue
 try:cmd=(d/"cmdline").read_bytes().split(bytes([0]))
 except OSError:continue
 assert not (len(cmd)>1 and any(b"prefeval" in a for a in cmd[1:]) and b"-c" not in cmd),"Another PrefEval process"
for path in ["retain730-C/refresh-driver.lock","retain730-C/launcher.lock","retain730-R/refresh-driver.lock"]:
 with (r/path).open("a") as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
receipts=[]
for arm,gpus in [("C",[1]),("R",[2,3])]:
 cmd=[str(p/"envs/vlm-r3-ngc2502/bin/python"),str(repo/"scripts/inspire/run_prefeval_k1_refresh730.py"),"--arm",arm,"--gpus"]+list(map(str,gpus))
 log=r/("retain730-"+arm)/"driver-high-eval-recovery-20260929.log"
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
 with log.open("a") as stream:
  proc=subprocess.Popen(cmd,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record={"arm":arm,"pid":proc.pid,"host":os.uname().nodename,"command":cmd,"gpus":gpus,"code_root":str(repo),"commit":expected,"log":str(log),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"reason":"Resume existing fixed endpoint evaluations after platform auto-recycle. All GPUs idle and all old task locks free. R training is complete at 23360; frozen completion checks skip all four training rounds, rollout/readback/position resume by completed keys."}
 (r/("retain730-"+arm)/"driver-launch.json").write_text(json.dumps(record,indent=2)+"\n")
 receipts.append(record);(out/"launch.json").write_text(json.dumps(receipts,indent=2)+"\n")
print(json.dumps(receipts))
