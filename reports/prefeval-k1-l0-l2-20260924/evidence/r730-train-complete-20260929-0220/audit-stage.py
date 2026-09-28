from pathlib import Path
import json,hashlib,torch,os,datetime,subprocess,shutil,sys,tarfile,collections
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936")
r=p/"runs/prefeval-b-mcq-20260925"
repo=p/"repos/prefeval-b-refresh-5559681"
sys.path[:0]=[str(repo),str(repo/"src")]
out=r/"r730-train-complete-20260929-0220"
out.mkdir(exist_ok=True)
assert not (out/"raw.tar.gz").exists()
def sha(f):
 h=hashlib.sha256()
 with f.open("rb") as s:
  for b in iter(lambda:s.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def save(name,value):(out/name).write_text(json.dumps(value,indent=2)+"\n")
def copy(f,name):
 dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip()=="55596810cee2062022b894ef93feb1ab4cd6b07b"
train=r/"retain730-R/train"
finalhash=sha(train/"checkpoint-final.pt")
stephash=sha(train/"checkpoint-step-023360.pt")
assert json.loads((train/"complete.json").read_text())=={"steps":23360,"checkpoint_sha256":finalhash}
bankhash=sha(r/"retain730-R/sources/round-3/bank.json")
assert json.loads((train/"round-3-complete.json").read_text())=={"step":23360,"checkpoint_sha256":stephash,"source_bank_sha256":bankhash}
receipt=json.loads((r/"retain730-R/round-3-fm-17520-23360-exit.json").read_text())
assert receipt["exit_codes"]==[0]
with (train/"resume.pt").open("rb") as f:
 h=hashlib.sha256()
 for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 digest=h.hexdigest();f.seek(0);state=torch.load(f,map_location="cpu",weights_only=False)
step=state["optimizer_step"]
steps=sorted({int(v["step"]) for v in state["optimizer"]["state"].values()})
assert step==23360 and state["episode_cursor"]==93440
assert steps==[23360] and len(state["optimizer"]["state"])==1075
assert state["trainer_state"]=={"refresh_round":3,"source_bank_sha256":bankhash}
assert all(k in state["rng_state"] for k in ["python","numpy","torch_cpu","torch_cuda"])
resume={"step":step,"cursor":state["episode_cursor"],"optimizer_steps":steps,"optimizer_parameters":1075,"rng_keys":list(state["rng_state"]),"trainer_state":state["trainer_state"],"sha256":digest,"method":"single file descriptor hash then seek/load CPU"}
save("resume-summary.json",resume);del state
a=torch.load(train/"checkpoint-final.pt",map_location="cpu",weights_only=False)
b=torch.load(train/"checkpoint-step-023360.pt",map_location="cpu",weights_only=False)
assert a["optimizer_step"]==b["optimizer_step"]==23360
assert a["manifest"]==b["manifest"]
assert a["trainable_state"].keys()==b["trainable_state"].keys()
assert all(torch.equal(v,b["trainable_state"][k]) for k,v in a["trainable_state"].items())
tensorcount=len(a["trainable_state"]);del a,b
for arm in ["C","R"]:
 base=r/("retain730-"+arm)
 for f in base.iterdir():
  if f.is_file() and f.suffix in [".json",".jsonl",".log"]:copy(f,"snapshots/"+str(f.relative_to(r)))
for f in train.iterdir():
 if f.is_file() and f.suffix in [".json",".jsonl"]:copy(f,"snapshots/"+str(f.relative_to(r)))
rows=[json.loads(x) for x in (out/"snapshots/retain730-R/train/optimization.jsonl").read_text().splitlines()]
assert [x["step"] for x in rows]==list(range(1,23361))
from scripts.experiments.prefeval_k1_data import load_training_records
ids={x["base_pair_id"] for x in load_training_records("train")}
assert len(ids)==730
budgets=[]
for k in range(4):
 counts=collections.Counter()
 for row in rows[k*5840:(k+1)*5840]:
  assert len(row["draws"])==4 and sum(d["position"]==0 for d in row["draws"])==2
  for d in row["draws"]:
   counts[d["pair_id"],"write" if d["position"]==0 else "retain"]+=1
   if d["position"]>0:assert d["source_round"]==k and d["source_index"]==d["position"]-1
 assert {x[0] for x in counts}==ids and len(counts)==1460 and set(counts.values())=={16}
 banksrc=r/("retain730-R/sources/round-"+str(k)+"/bank.json")
 copy(banksrc,"banks/round-"+str(k)+".json")
 seg=json.loads((train/("round-"+str(k)+".json")).read_text())
 assert seg["source_bank_sha256"]==sha(banksrc)
 budgets.append({"round":k,"preferences":730,"write_per_preference":16,"retain_per_preference":16,"steps":5840,"bank_sha256":sha(banksrc)})
evalroot=r/"retain730-R/final-dev-V1"
manifest=json.loads((evalroot/"manifest.json").read_text())
assert manifest["checkpoint_sha256"]==finalhash and manifest["noise_chains"]==2 and manifest["inter_turns"]==10
assert manifest["split"]=="dev" and manifest["initial_variant"]==1 and "probe_initial_sources" not in manifest
copy(evalroot/"manifest.json","eval-dev/manifest.json")
complete=list(evalroot.rglob("complete.json"))
for done in complete:
 j=json.loads(done.read_text());assert j["binding"]==manifest
 assert len(j["png_hashes"])==11
 for name,h in j["png_hashes"].items():assert sha(done.parent/name)==h
 copy(done,"eval-dev/"+str(done.relative_to(evalroot)))
 copy(done.parent/"writes.jsonl","eval-dev/"+str((done.parent/"writes.jsonl").relative_to(evalroot)))
copy(r/"high-recovery-20260928/runtime.json","runtime.json")
for name in ["scripts/experiments/prefeval_k1_refresh_bank.py","scripts/experiments/prefeval_k1_writer.py","scripts/inspire/run_prefeval_k1_refresh730.py"]:
 copy(repo/name,"frozen-code/"+name)
summary={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"host":os.uname().nodename,"final_checkpoint_sha256":finalhash,"step23360_checkpoint_sha256":stephash,"endpoint_tensors_equal":tensorcount,"optimizer_exit_codes":[0],"completion_mtime_utc":datetime.datetime.fromtimestamp((train/"complete.json").stat().st_mtime,datetime.timezone.utc).isoformat(),"optimization_first":1,"optimization_last":23360,"draw_count":93440,"segments":budgets,"resume":resume,"eval_dev_verified_complete":len(complete),"eval_dev_png_at_snapshot":len(list(evalroot.rglob("*.png"))),"C_train_complete":len(list((r/"retain730-C/final-train-V1").rglob("complete.json"))),"claims":"Training endpoint and workflow completion only; endpoint evaluation still in progress, no partial performance"}
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
