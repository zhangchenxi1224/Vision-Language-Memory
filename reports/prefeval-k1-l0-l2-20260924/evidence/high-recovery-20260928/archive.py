from pathlib import Path
import json,hashlib,tarfile,shutil,datetime
r=Path("/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b-mcq-20260925")
out=r/"high-recovery-20260928";before=r/"l012-recovery-20260928/before"
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
snapshot=out/"snapshot";snapshot.mkdir(exist_ok=True)
for arm in ["C","R"]:
 base=r/("retain730-"+arm)
 for f in base.glob("*"):
  if f.is_file() and f.suffix in [".json",".jsonl",".log"]:
   dest=snapshot/f.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
for source in [r/"interruption-20260928-1335/C-png-inventory.json",r/"interruption-20260928-1335/R-png-inventory.json"]:
 shutil.copy2(source,out/source.name)
files=[]
for f in sorted(before.rglob("*")):
 if f.is_file():files.append((f,"before/"+str(f.relative_to(before))))
for f in sorted(out.rglob("*")):
 if f.is_file() and f.name not in ["raw.tar.gz","raw.sha256","snapshot-files.json"]:files.append((f,str(f.relative_to(out))))
manifest=[{"path":name,"bytes":f.stat().st_size,"sha256":sha(f)} for f,name in files]
(out/"snapshot-files.json").write_text(json.dumps(manifest,indent=2)+"\n")
with tarfile.open(out/"raw.tar.gz","w:gz") as tar:
 for f,name in files:tar.add(f,arcname=name,recursive=False)
 tar.add(out/"snapshot-files.json",arcname="snapshot-files.json")
result={"bytes":(out/"raw.tar.gz").stat().st_size,"sha256":sha(out/"raw.tar.gz"),"files":len(files),"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/"raw.sha256").write_text(result["sha256"]+"  raw.tar.gz\n")
(out/"archive-summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
