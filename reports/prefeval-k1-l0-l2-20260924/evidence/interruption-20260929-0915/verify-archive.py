import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
a=root/"raw.tar.gz"
expected="5b5d5e32f642957afa235c9589a8e6c30a660cae59ac961ff901085bf18c7eac"
assert a.stat().st_size==21723818 and hashlib.sha256(a.read_bytes()).hexdigest()==expected
with tarfile.open(a) as t:
 inv=json.load(t.extractfile("snapshot-files.json"))
 for item in inv:
  data=t.extractfile(item["path"]).read()
  assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"],item["path"]
 rows=[json.loads(x) for x in t.extractfile("snapshots/retain730-R/train/optimization.jsonl").read().splitlines()]
 assert [x["step"] for x in rows]==list(range(1,23361))
 for name in ["summary.json","resume-summary.json","png-inventory.json","snapshot-files.json"]:
  (root/name).write_bytes(t.extractfile(name).read())
result={"verified_files":len(inv),"optimization_rows":len(rows),"archive_sha256":expected,"archive_bytes":a.stat().st_size}
(root/"local-verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
