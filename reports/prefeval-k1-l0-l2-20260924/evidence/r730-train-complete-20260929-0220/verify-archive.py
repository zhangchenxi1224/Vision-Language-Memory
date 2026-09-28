import hashlib,json,tarfile,collections
from pathlib import Path
root=Path(__file__).resolve().parent
archive=root/"raw.tar.gz"
expected="5d6fbda192f8c74e439caffea87e634371ca7dc526670cb8b81ee93650d7b8dc"
assert archive.stat().st_size==4745464
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
with tarfile.open(archive) as tar:
 inventory=json.load(tar.extractfile("snapshot-files.json"))
 for item in inventory:
  data=tar.extractfile(item["path"]).read()
  assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"],item["path"]
 summary=json.load(tar.extractfile("stage-summary.json"))
 rows=[json.loads(x) for x in tar.extractfile("snapshots/retain730-R/train/optimization.jsonl").read().splitlines()]
 assert [x["step"] for x in rows]==list(range(1,23361))
 for k in range(4):
  counts=collections.Counter()
  for row in rows[k*5840:(k+1)*5840]:
   assert len(row["draws"])==4 and sum(d["position"]==0 for d in row["draws"])==2
   for d in row["draws"]:
    counts[d["pair_id"],"write" if d["position"]==0 else "retain"]+=1
    if d["position"]>0:assert d["source_round"]==k and d["source_index"]==d["position"]-1
  assert len(counts)==1460 and len({x[0] for x in counts})==730 and set(counts.values())=={16}
 assert summary["resume"]["step"]==23360 and summary["resume"]["cursor"]==93440
 assert summary["resume"]["optimizer_steps"]==[23360] and summary["resume"]["optimizer_parameters"]==1075
 assert set(summary["resume"]["rng_keys"])=={"python","numpy","torch_cpu","torch_cuda"}
 for name in ["stage-summary.json","resume-summary.json","runtime.json","snapshot-files.json"]:
  (root/name).write_bytes(tar.extractfile(name).read())
result={"verified_files":len(inventory),"optimization_rows":len(rows),"four_segments_budget_verified":True,"archive_sha256":expected,"archive_bytes":archive.stat().st_size}
(root/"local-verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
