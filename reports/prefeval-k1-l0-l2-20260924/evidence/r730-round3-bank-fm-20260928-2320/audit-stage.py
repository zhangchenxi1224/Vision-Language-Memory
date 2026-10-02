from pathlib import Path
import json,hashlib,torch,os,datetime,subprocess,shutil,sys,tarfile
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936")
r=p/"runs/prefeval-b-mcq-20260925"
repo=p/"repos/prefeval-b-refresh-5559681"
sys.path[:0]=[str(repo),str(repo/"src")]
out=r/"r730-round3-bank-fm-20260928-2320"
out.mkdir(exist_ok=True)
assert not (out/"raw.tar.gz").exists()
def sha(f):
 h=hashlib.sha256()
 with f.open("rb") as s:
  for b in iter(lambda:s.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def save(name,value):
 (out/name).write_text(json.dumps(value,indent=2)+"\n")
def copy(f,name):
 dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()=="55596810cee2062022b894ef93feb1ab4cd6b07b"
from scripts.experiments.prefeval_k1_data import load_training_records
from scripts.experiments.prefeval_k1_refresh_bank import freeze_refresh_bank
bank=r/"retain730-R/sources/round-3"
checkpoint=r/"retain730-R/train/checkpoint-step-017520.pt"
assert sha(checkpoint)=="3da185795c65c8a3425086040f83cab18d5c334ef68055b2e4e8603025e52998"
bankhash=sha(bank/"bank.json")
artifact=freeze_refresh_bank(bank,load_training_records("train"),checkpoint,r/"variants-train.json",3)
assert sha(bank/"bank.json")==bankhash and len(artifact["items"])==730
duplicates=[]
for f in bank.rglob("*"):
 if f.is_file() and f.suffix in [".json",".jsonl"] and "fm-conditions" not in f.parts:
  copy(f,"sources/"+str(f.relative_to(bank)))
  if f.name=="writes.jsonl":
   lines=[json.loads(x) for x in f.read_text().splitlines()]
   if len(lines)>10:
    extra=lines[:-10];valid=lines[-10:]
    assert extra==valid[:len(extra)]
    duplicates.append({"path":str(f.relative_to(bank)),"preserved_prefix_rows":len(extra)})
assert [len(list((bank/("shard-"+str(i))).rglob("complete.json"))) for i in range(2)]==[365,365]
assert len(list(bank.rglob("*.png")))==7300
receipt=json.loads((r/"retain730-R/round-3-current-student-rollout-exit.json").read_text())
assert receipt["exit_codes"]==[0,0]
with (r/"retain730-R/train/resume.pt").open("rb") as f:
 h=hashlib.sha256()
 for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 digest=h.hexdigest();f.seek(0);state=torch.load(f,map_location="cpu",weights_only=False)
step=state["optimizer_step"]
steps=sorted({int(v["step"]) for v in state["optimizer"]["state"].values()})
assert 17520<step<=23360 and state["episode_cursor"]==step*4
assert steps==[step] and len(state["optimizer"]["state"])==1075
assert state["trainer_state"]=={"refresh_round":3,"source_bank_sha256":bankhash}
assert all(k in state["rng_state"] for k in ["python","numpy","torch_cpu","torch_cuda"])
resume={"step":step,"cursor":state["episode_cursor"],"optimizer_steps":steps,"optimizer_parameters":1075,"rng_keys":list(state["rng_state"]),"trainer_state":state["trainer_state"],"sha256":digest,"method":"single open file descriptor hash then seek/load CPU"}
save("resume-summary.json",resume);del state
for arm in ["C","R"]:
 base=r/("retain730-"+arm)
 for f in base.iterdir():
  if f.is_file() and f.suffix in [".json",".jsonl",".log"]:copy(f,"snapshots/"+str(f.relative_to(r)))
for f in (r/"retain730-R/train").iterdir():
 if f.is_file() and f.suffix in [".json",".jsonl"]:copy(f,"snapshots/"+str(f.relative_to(r)))
opt=out/"snapshots/retain730-R/train/optimization.jsonl"
raw=opt.read_bytes();lines=raw.splitlines()
if raw and not raw.endswith(b"\n"):
 save("inflight-log-tail.json",{"trailing_partial_hex":lines[-1].hex()});lines=lines[:-1]
rows=[json.loads(x) for x in lines]
assert [x["step"] for x in rows]==list(range(1,len(rows)+1))
assert len(rows)>=step and len(rows)<=23360
for row in rows:
 assert len(row["draws"])==4
 assert sum(x["position"]==0 for x in row["draws"])==2
 for draw in row["draws"]:
  if draw["position"]>0:
   assert draw["source_round"]==(row["step"]-1)//5840
   assert draw["source_index"]==draw["position"]-1
save("first-step-17521.json",rows[17520])
copy(r/"high-recovery-20260928/runtime.json","runtime.json")
copy(r/"high-recovery-20260928/resume-before.json","resume-before-17520.json")
for name in ["scripts/experiments/prefeval_k1_refresh_bank.py","scripts/experiments/prefeval_k1_writer.py","scripts/inspire/run_prefeval_k1_refresh730.py"]:
 copy(repo/name,"frozen-code/"+name)
summary={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"host":os.uname().nodename,"bank_sha256":bankhash,"source_checkpoint_sha256":sha(checkpoint),"bank_mtime_utc":datetime.datetime.fromtimestamp((bank/"bank.json").stat().st_mtime,datetime.timezone.utc).isoformat(),"preferences":730,"png_verified":7300,"shard_complete":[365,365],"rollout_exit_codes":[0,0],"preserved_duplicate_prefixes":duplicates,"optimization_first":1,"optimization_last":len(rows),"draw_count":len(rows)*4,"resume":resume,"claims":"Workflow and continuation evidence only, not memory performance"}
save("stage-summary.json",summary)
files=[]
for f in sorted(out.rglob("*")):
 if f.is_file():files.append({"path":str(f.relative_to(out)),"bytes":f.stat().st_size,"sha256":sha(f)})
save("snapshot-files.json",files)
with tarfile.open(out/"raw.tar.gz","w:gz") as tar:
 for f in sorted(out.rglob("*")):
  if f.is_file() and f.name!="raw.tar.gz":tar.add(f,arcname=str(f.relative_to(out)))
result={"sha256":sha(out/"raw.tar.gz"),"bytes":(out/"raw.tar.gz").stat().st_size,"files":len(files)}
save("archive-sha256.json",result)
print(json.dumps({"summary":summary,"archive":result}))
