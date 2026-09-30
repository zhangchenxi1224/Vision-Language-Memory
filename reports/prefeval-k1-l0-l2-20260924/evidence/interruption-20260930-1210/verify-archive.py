import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
a=root/"raw.tar.gz"
expected="a9b982af3a1afa319c0593e56fd7fc9fd70f832649761ace48a99f9aa63c5008"
assert a.stat().st_size==27375211 and hashlib.sha256(a.read_bytes()).hexdigest()==expected
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

