import hashlib,json,tarfile
from pathlib import Path
root=Path(__file__).resolve().parent
archive=root/"raw.tar.gz"
expected="53144ad89c530f29670f8dd9d4672d5ca3b8d3ce5707905f8e51349b4eca5fa0"
assert archive.stat().st_size==12170045
assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
with tarfile.open(archive) as tar:
 inventory=json.load(tar.extractfile("snapshot-files.json"))
 for item in inventory:
  data=tar.extractfile(item["path"]).read()
  assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"],item["path"]
 audit=json.load(tar.extractfile("resume-before.json"))
 assert audit["step"]==17520 and audit["cursor"]==70080
 assert audit["optimizer_parameters"]==1075 and audit["optimizer_steps"]==[17520]
 preserved=json.load(tar.extractfile("preserved-completed.json"))
 assert preserved["completed_chains_unchanged"]["Cpilot"]["chains"]==128
 assert preserved["completed_chains_unchanged"]["Cdev"]["chains"]==180
 rows=[json.loads(x) for x in tar.extractfile("before/retain730-R/train/optimization.jsonl").read().splitlines()]
 assert [x["step"] for x in rows]==list(range(1,17521))
 for name in ["launch.json","resume-before.json","preserved-completed.json","runtime.json","snapshot-files.json"]:
  (root/name).write_bytes(tar.extractfile(name).read())
result={"verified_files":len(inventory),"optimization_rows":len(rows),"archive_sha256":expected,"archive_bytes":archive.stat().st_size}
(root/"local-verification.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
