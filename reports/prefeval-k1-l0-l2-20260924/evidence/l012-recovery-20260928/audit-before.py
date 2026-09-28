from pathlib import Path
import json,hashlib,torch,os,datetime,subprocess,fcntl,shutil
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936")
r=p/"runs/prefeval-b-mcq-20260925"
out=r/"l012-recovery-20260928"
out.mkdir(exist_ok=True)
assert not (out/"launch.json").exists()
def sha(f):
 h=hashlib.sha256()
 with f.open("rb") as s:
  for b in iter(lambda:s.read(8*1024*1024),b""): h.update(b)
 return h.hexdigest()
hashes={
"robust730/train/checkpoint-final.pt":"6c8eb92f629ccdf6932407124190d651e8231058ef6f232a0a59380370d44c8d",
"sourcebank730/bank.json":"fc0b8c6d79949c765867ffeccfc328422be4bfd46ea994df63aecb9d5fe2ddd4",
"retain730-C/train/checkpoint-final.pt":"3d67d91f4e26bebc503f95e697d61b9d13453738be5d8a4e90e898cef4372ced",
"retain730-R/train/checkpoint-step-017520.pt":"3da185795c65c8a3425086040f83cab18d5c334ef68055b2e4e8603025e52998",
"retain730-R/sources/round-2/bank.json":"4ca3bc2fc0d11394a1f6e22119a539e7656039a6b438f7ba2a71058b2b44c234"}
for name,value in hashes.items(): assert sha(r/name)==value,name
with (r/"retain730-R/train/resume.pt").open("rb") as f:
 h=hashlib.sha256()
 for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
 digest=h.hexdigest()
 assert digest=="86f4536f99b6f008b6cc2c81e3b3e349314c30f8aa9d6d4d10d97e8a6e7ce9bb"
 f.seek(0); state=torch.load(f,map_location="cpu",weights_only=False)
assert state["optimizer_step"]==17520 and state["episode_cursor"]==70080
steps=sorted({int(v["step"]) for v in state["optimizer"]["state"].values()})
assert steps==[17520] and len(state["optimizer"]["state"])==1075
assert state["trainer_state"]["refresh_round"]==2
assert state["trainer_state"]["source_bank_sha256"]==hashes["retain730-R/sources/round-2/bank.json"]
assert all(k in state["rng_state"] for k in ["python","numpy","torch_cpu","torch_cuda"])
audit={"step":17520,"cursor":70080,"optimizer_steps":steps,"optimizer_parameters":1075,"rng_keys":list(state["rng_state"]),"trainer_state":state["trainer_state"],"sha256":digest,"hashes":hashes}
del state
counts={}
for arm in ["C","R"]:
 inventory=json.loads((r/("interruption-20260928-1335/"+arm+"-png-inventory.json")).read_text())
 for item in inventory:
  f=r/item["path"]; assert sha(f)==item["sha256"],str(f)
  assert f.stat().st_mtime_ns==item["mtime_ns"],str(f)
 counts[arm]=len(inventory)
assert counts=={"C":4307,"R":2783}
assert not (r/"retain730-R/sources/round-3/bank.json").exists()
for arm in ["C","R"]:
 base=r/("retain730-"+arm)
 for f in base.rglob("*"):
  if f.is_file() and f.suffix in [".json",".jsonl",".log"] and "condition" not in str(f):
   target=out/"before"/f.relative_to(r); target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target)
audit.update({"png_inventory_verified":counts,"host":os.uname().nodename,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()})
(out/"resume-before.json").write_text(json.dumps(audit,indent=2)+"\n")
print(json.dumps(audit,indent=2))
