from pathlib import Path
import json,hashlib,datetime,os,shutil,tarfile
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936");r=p/"runs/prefeval-b-mcq-20260925"
out=r/"high-eval-recovery-20260929";before=r/"interruption-20260929-0915"
def sha(f):
 h=hashlib.sha256()
 with f.open("rb") as s:
  for b in iter(lambda:s.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
summary=json.loads((before/"summary.json").read_text());partial=set(summary["partial_png"]);checked=0
for item in json.loads((before/"png-inventory.json").read_text()):
 if item["path"] in partial:continue
 f=r/item["path"];assert sha(f)==item["sha256"] and f.stat().st_mtime_ns==item["mtime_ns"],str(f);checked+=1
immutable=[];prefixes=[]
for arm in ["C","R"]:
 b=r/("retain730-"+arm)
 for f in list(b.glob("readback-final-*/*.jsonl"))+list(b.glob("positions-final-*.jsonl")):
  old=before/"snapshots"/f.relative_to(r)
  if not old.exists():continue
  data=old.read_bytes();actual=f.read_bytes()
  assert actual[:len(data)]==data,str(f)
  result={"path":str(f.relative_to(r)),"old_bytes":len(data),"new_bytes":len(actual),"old_sha256":hashlib.sha256(data).hexdigest()}
  (immutable if actual==data else prefixes).append(result)
# Validity of completed row keys is checked before resuming by frozen readers; no partial metrics here.
value={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"host":os.uname().nodename,"old_completed_pngs_hash_mtime_verified":checked,"immutable_readbacks":immutable,"prefix_preserved_readbacks":prefixes,"partial_pngs_archived":len(partial),"no_new_performance_claim":True}
(out/"preservation.json").write_text(json.dumps(value,indent=2)+"\n")
print(json.dumps(value))
