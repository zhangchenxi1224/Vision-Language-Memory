from pathlib import Path
import json,hashlib,tarfile,shutil,datetime
r=Path("/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b-mcq-20260925")
out=r/"high-eval-recovery-20260929";before=r/"interruption-20260929-0915"
assert not (out/"raw.tar.gz").exists()
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
for k in range(4):
 assert json.loads((r/("retain730-R/final-dev-V1-evaluation-%d-exit.json"%k)).read_text())["exit_codes"]==[0]
summary=json.loads((out/"R-dev-summary.json").read_text())
assert summary["checks"]=={"rows":9450,"position_rows":1800,"pngs":1980,"chains":180,"parse_failures":0,"truncations":0}
preserved=[]
for rel in ["retain730-R/readback-final-dev-V1/readback-0.jsonl","retain730-R/positions-final-dev-V1-0.jsonl","retain730-R/positions-final-dev-V1-10.jsonl"]:
 old=(before/"snapshots"/rel).read_bytes();new=(r/rel).read_bytes()
 assert new.startswith(old)
 preserved.append({"path":rel,"old_bytes":len(old),"new_bytes":len(new),"old_rows":len(old.splitlines()),"new_rows":len(new.splitlines()),"unchanged":old==new,"new_sha256":hashlib.sha256(new).hexdigest()})
(out/"readback-prefix-preservation.json").write_text(json.dumps(preserved,indent=2)+"\n")
for arm in ["C","R"]:
 for f in (r/("retain730-"+arm)).rglob("*"):
  if f.is_file() and f.suffix in [".json",".jsonl",".log"] and "condition" not in str(f):
   dest=out/"snapshots"/f.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
inventory=[{"path":str(f.relative_to(out)),"bytes":f.stat().st_size,"sha256":sha(f)} for f in sorted(out.rglob("*")) if f.is_file()]
(out/"snapshot-files.json").write_text(json.dumps(inventory,indent=2)+"\n")
with tarfile.open(out/"raw.tar.gz","w:gz") as t:
 for f in sorted(out.rglob("*")):
  if f.is_file() and f.name!="raw.tar.gz":t.add(f,arcname=str(f.relative_to(out)))
result={"sha256":sha(out/"raw.tar.gz"),"bytes":(out/"raw.tar.gz").stat().st_size,"files":len(inventory),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/"archive-summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
