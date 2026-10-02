import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
a=root/"raw.tar.gz"
expected="7e4f82c112818fec7d7306b7ce1b9aa2890d45a766b898ceaa99fc75287b70bf"
assert a.stat().st_size==14448844 and hashlib.sha256(a.read_bytes()).hexdigest()==expected
with tarfile.open(a) as t:
 inv=json.load(t.extractfile("snapshot-files.json"))
 for item in inv:
  data=t.extractfile(item["path"]).read()
  assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"],item["path"]
 rows=[json.loads(x) for x in t.extractfile("snapshots/retain730-R/train/optimization.jsonl").read().splitlines()]
 assert [x["step"] for x in rows]==list(range(1,23361))
 for name in ["R-dev-summary.json","resume-before.json","launch.json","runtime.json","preservation.json","readback-prefix-preservation.json","snapshot-files.json"]:
  (root/name).write_bytes(t.extractfile(name).read())
 s=json.loads((root/"R-dev-summary.json").read_text())
 assert s["checks"]=={"rows":9450,"position_rows":1800,"pngs":1980,"chains":180,"parse_failures":0,"truncations":0}
 for k in range(4):
  assert json.load(t.extractfile("snapshots/retain730-R/final-dev-V1-evaluation-%d-exit.json"%k))["exit_codes"]==[0]
result={"verified_files":len(inv),"optimization_rows":len(rows),"dev_rows":11250,"archive_sha256":expected,"archive_bytes":a.stat().st_size}
(root/"local-verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
