from pathlib import Path
import os,json,datetime,subprocess,fcntl,hashlib
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936")
r=p/"runs/prefeval-b-mcq-20260925"
out=r/"l012-recovery-20260928"
repo=p/"repos/prefeval-b-refresh-5559681"
expected="55596810cee2062022b894ef93feb1ab4cd6b07b"
assert os.uname().nodename=="prefeval-k1-l012-h200x4-20260924--dacf2dea6e9f-6rkv3yumj5"
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()==expected
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=p/"repos/prefeval-b-deploy-52d98d8",text=True).strip()=="52d98d85f84e1e4910e922e34f59f5f14bbabf5c"
assert not (out/"launch.json").exists(),"Already launched; inspect actual processes"
audit=json.loads((out/"resume-before.json").read_text())
assert audit["step"]==17520 and audit["cursor"]==70080
assert audit["png_inventory_verified"]=={"C":4307,"R":2783}
assert audit["host"]==os.uname().nodename
assert not (r/"retain730-R/sources/round-3/bank.json").exists()
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(),"GPU occupied"
for d in Path("/proc").iterdir():
 if not d.name.isdigit() or int(d.name)==os.getpid():continue
 try:cmd=(d/"cmdline").read_bytes().split(bytes([0]))
 except OSError:continue
 assert not (len(cmd)>1 and b"prefeval" in cmd[1]),"Another PrefEval process is alive"
for path in ["retain730-C/refresh-driver.lock","retain730-C/launcher.lock","retain730-R/refresh-driver.lock"]:
 with (r/path).open("a") as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
receipts=[]
for arm,gpus in [("C",[1]),("R",[2,3])]:
 command=[str(p/"envs/vlm-r3-ngc2502/bin/python"),str(repo/"scripts/inspire/run_prefeval_k1_refresh730.py"),"--arm",arm,"--gpus"]+list(map(str,gpus))
 log=r/("retain730-"+arm)/"driver-l012-recovery-20260928.log"
 env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
 with log.open("a") as stream:
  proc=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 record={"arm":arm,"pid":proc.pid,"host":os.uname().nodename,"command":command,"gpus":gpus,"code_root":str(repo),"commit":expected,"log":str(log),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"reason":"User explicitly authorized any idle instance, preferring four GPUs, without competing with other tasks; this instance was verified entirely idle. Resume R17520/round3 and C remaining train evaluation with frozen code and budget."}
 (r/("retain730-"+arm)/"driver-launch.json").write_text(json.dumps(record,indent=2)+"\n")
 receipts.append(record)
 (out/"launch.json").write_text(json.dumps(receipts,indent=2)+"\n")
print(json.dumps(receipts,indent=2))
