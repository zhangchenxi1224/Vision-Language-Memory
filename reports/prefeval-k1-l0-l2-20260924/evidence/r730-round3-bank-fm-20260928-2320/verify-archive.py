import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
archive=root/"raw.tar.gz"
expected="2a0a5e34473c347249db84ede1fea0cad1b34293d2d08589fc598b427e1c5cf1"
assert archive.stat().st_size==3715750
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
with tarfile.open(archive) as tar:
 inventory=json.load(tar.extractfile("snapshot-files.json"))
 for item in inventory:
  data=tar.extractfile(item["path"]).read()
  assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"],item["path"]
 summary=json.load(tar.extractfile("stage-summary.json"))
 rows=[json.loads(x) for x in tar.extractfile("snapshots/retain730-R/train/optimization.jsonl").read().splitlines()]
 assert [x["step"] for x in rows]==list(range(1,17691))
 assert summary["png_verified"]==7300
 assert summary["resume"]["step"]==17664 and summary["resume"]["optimizer_steps"]==[17664]
 assert summary["resume"]["optimizer_parameters"]==1075
 for name in ["stage-summary.json","resume-summary.json","resume-before-17520.json","runtime.json","snapshot-files.json","first-step-17521.json"]:
  (root/name).write_bytes(tar.extractfile(name).read())
result={"verified_files":len(inventory),"optimization_rows":len(rows),"archive_sha256":expected,"archive_bytes":archive.stat().st_size}
(root/"local-verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
