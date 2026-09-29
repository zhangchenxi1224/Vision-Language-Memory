import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
a=root/"raw.tar.gz";expected="053018cd769e347af3396a15cf76c0bbb02f4d232635d9f249cfe7c0f27b2862"
assert a.stat().st_size==1061365 and hashlib.sha256(a.read_bytes()).hexdigest()==expected
with tarfile.open(a) as t:
 inv=json.load(t.extractfile("snapshot-files.json"))
 for item in inv:
  data=t.extractfile(item["path"]).read()
  assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"],item["path"]
 s=json.load(t.extractfile("R-pilot-summary.json"))
 assert s["checks"]=={"rows":6720,"position_rows":1280,"pngs":1408,"chains":128,"parse_failures":0,"truncations":0}
 rows=[json.loads(x) for x in t.extractfile("snapshots/retain730-R/readback-final-pilot-V1/readback-0.jsonl").read().splitlines()]
 idx={(x["pair_id"],x["chain"],x["prefix"],x["control"],x["family"]):x for x in rows}
 assert len(rows)==len(idx)==6720 and len({x["pair_id"] for x in rows})==64
 for m in s["metrics"]:
  group=[x for x in rows if x["prefix"]==m["depth"] and x["family"]==m["family"]]
  for label,control in [("matched","memory"),("mismatch","mismatch"),("text","text")]:
   assert sum(x["correct"] for x in group if x["control"]==control)==m[label]
  assert m["net"]==m["matched"]-m["mismatch"]
  ids=sorted({x["pair_id"] for x in rows})
  def ok(i,c,d,k,f):return idx[i,c,d,k,f]["correct"]
  pairs=[(ok(i,c,m["depth"],"memory",m["family"]),ok(i,c,m["depth"],"mismatch",m["family"])) for i in ids for c in range(2)]
  assert m["repair"]==sum(a and not b for a,b in pairs) and m["regress"]==sum(b and not a for a,b in pairs)
  assert m["gray"]==sum(ok(i,0,0,"blank",m["family"]) for i in ids)
  assert m["both_noise"]==sum(all(ok(i,c,m["depth"],"memory",m["family"]) for c in range(2)) for i in ids)
 for d in [0,1,5,10]:
  assert s["joint"]["ood_by_depth"][str(d)]=={"same_image":sum(all(ok(i,c,d,"memory",f) for f in ["O1","O2"]) for i in ids for c in range(2)),"both_noise":sum(all(ok(i,c,d,"memory",f) for c in range(2) for f in ["O1","O2"]) for i in ids)}
 for f in ["T1","T2","T3","O1","O2"]:
  assert s["joint"]["registered_depths_and_both_noise"][f]==sum(all(ok(i,c,d,"memory",f) for c in range(2) for d in [0,1,5,10]) for i in ids)
 assert s["joint"]["ood_registered_depths_and_both_noise"]==sum(all(ok(i,c,d,"memory",f) for c in range(2) for d in [0,1,5,10] for f in ["O1","O2"]) for i in ids)
 for k in range(4):
  assert json.load(t.extractfile("snapshots/retain730-R/final-pilot-V1-evaluation-%d-exit.json"%k))["exit_codes"]==[0]
 for d in [0,10]:
  pr=[json.loads(x) for x in t.extractfile("snapshots/retain730-R/positions-final-pilot-V1-%d.jsonl"%d).read().splitlines()]
  assert len(pr)==len({(x["pair_id"],x["chain"],x["position"]) for x in pr})==640
  pi={(x["pair_id"],x["chain"],x["position"]):x["correct"] for x in pr}
  assert s["positions"][str(d)]["all_four"]==sum(all(pi[i,c,q] for q in range(4)) for i in ids for c in range(2))
  assert s["positions"][str(d)]["all_four_both_noise"]==sum(all(pi[i,c,q] for q in range(4) for c in range(2)) for i in ids)
  for q,v in s["positions"][str(d)]["per_position"].items():
   assert sum(x["correct"] for x in pr if str(x["position"])==q)==v
 for name in ["R-pilot-summary.json","runtime.json","png-inventory.json","snapshot-files.json"]:
  (root/name).write_bytes(t.extractfile(name).read())
result={"verified_files":len(inv),"raw_rows":8000,"full_metrics_recounted":True,"archive_bytes":a.stat().st_size,"archive_sha256":expected}
(root/"local-verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
