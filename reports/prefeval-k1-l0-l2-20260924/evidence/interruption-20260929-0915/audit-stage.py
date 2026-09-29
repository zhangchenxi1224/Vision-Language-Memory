from pathlib import Path
import json,hashlib,os,datetime,fcntl,shutil,tarfile
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936")
r=p/"runs/prefeval-b-mcq-20260925"
out=r/"interruption-20260929-0915";out.mkdir(exist_ok=True)
assert not (out/"raw.tar.gz").exists()
def sha(f):
 h=hashlib.sha256()
 with f.open("rb") as s:
  for b in iter(lambda:s.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def save(n,v):(out/n).write_text(json.dumps(v,indent=2)+"\n")
def copy(f,n):
 dst=out/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst)
locks={}
for n in ["retain730-C/refresh-driver.lock","retain730-C/launcher.lock","retain730-R/refresh-driver.lock"]:
 with (r/n).open("a") as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[n]="FREE";fcntl.flock(f,fcntl.LOCK_UN)
  except BlockingIOError:locks[n]="HELD"
assert set(locks.values())=={"FREE"}
for arm in ["C","R"]:
 for f in (r/("retain730-"+arm)).rglob("*"):
  if f.is_file() and f.suffix in [".json",".jsonl",".log"] and "condition" not in str(f):
   copy(f,"snapshots/"+str(f.relative_to(r)))
hashes={"retain730-C/train/checkpoint-final.pt":"3d67d91f4e26bebc503f95e697d61b9d13453738be5d8a4e90e898cef4372ced","retain730-R/train/checkpoint-final.pt":"789a49fc488dfa372917b3646625b6cd62946cb48185763eade82d22e7dbc845","retain730-R/train/resume.pt":"b480a8cb81e751251fd8dcfb158a028ecb6c84a0e21ae18dcb6af49e3484b4a4"}
for n,h in hashes.items():assert sha(r/n)==h,n
save("resume-summary.json",{"sha256":hashes["retain730-R/train/resume.pt"],"cpu_deserialized_this_audit":False,"last_full_audit":"r730-train-complete-20260929-0220"})
inventory=[];counts={};partials=[]
for n in ["retain730-C/final-dev-V1","retain730-C/final-pilot-V1","retain730-C/final-train-V1","retain730-R/final-dev-V1"]:
 b=r/n;done=list(b.rglob("complete.json"));pngs=list(b.rglob("*.png"))
 counts[n]={"complete":len(done),"png":len(pngs)}
 for f in pngs:
  inventory.append({"path":str(f.relative_to(r)),"sha256":sha(f),"mtime_ns":f.stat().st_mtime_ns,"bytes":f.stat().st_size})
  if not (f.parent/"complete.json").exists():copy(f,"partial-png/"+str(f.relative_to(r)));partials.append(str(f.relative_to(r)))
 for f in done:
  j=json.loads(f.read_text())
  for name,h in j["png_hashes"].items():assert sha(f.parent/name)==h
save("png-inventory.json",inventory)
rows=[json.loads(x) for x in (r/"retain730-R/train/optimization.jsonl").read_text().splitlines()]
assert [x["step"] for x in rows]==list(range(1,23361))
readbacks={}
for f in (r/"retain730-R").glob("*final-dev-V1*"):
 if f.is_dir():
  fs=list(f.glob("*.jsonl"))
 elif f.suffix==".jsonl":fs=[f]
 else:continue
 for x in fs:readbacks[str(x.relative_to(r))]={"rows":len(x.read_text().splitlines()),"sha256":sha(x)}
summary={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"audit_host":os.uname().nodename,"stopped_host_not_proc_inspected":True,"locks":locks,"hashes":hashes,"counts":counts,"partial_png":partials,"readbacks":readbacks,"optimization_rows":len(rows)}
save("summary.json",summary)
files=[{"path":str(f.relative_to(out)),"bytes":f.stat().st_size,"sha256":sha(f)} for f in sorted(out.rglob("*")) if f.is_file()]
save("snapshot-files.json",files)
with tarfile.open(out/"raw.tar.gz","w:gz") as t:
 for f in sorted(out.rglob("*")):
  if f.is_file() and f.name!="raw.tar.gz":t.add(f,arcname=str(f.relative_to(out)))
archive={"sha256":sha(out/"raw.tar.gz"),"bytes":(out/"raw.tar.gz").stat().st_size,"files":len(files)}
save("archive-sha256.json",archive)
print(json.dumps({"summary":summary,"archive":archive}))
