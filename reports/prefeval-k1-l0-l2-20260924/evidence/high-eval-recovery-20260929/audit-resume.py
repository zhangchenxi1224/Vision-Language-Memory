from pathlib import Path
import json,hashlib,torch,os,subprocess,fcntl,datetime
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936")
r=p/"runs/prefeval-b-mcq-20260925"
out=r/"high-eval-recovery-20260929";out.mkdir(exist_ok=True)
assert not (out/"launch.json").exists()
assert os.uname().nodename=="prefeval-k1-h200x4-high-20260924--506d5fcf84c1-c7a2kxozom"
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
with (r/"retain730-R/train/resume.pt").open("rb") as f:
 h=hashlib.sha256()
 for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 assert h.hexdigest()=="b480a8cb81e751251fd8dcfb158a028ecb6c84a0e21ae18dcb6af49e3484b4a4"
 f.seek(0);s=torch.load(f,map_location="cpu",weights_only=False)
 assert s["optimizer_step"]==23360 and s["episode_cursor"]==93440
 assert len(s["optimizer"]["state"])==1075 and {int(v["step"]) for v in s["optimizer"]["state"].values()}=={23360}
 assert all(x in s["rng_state"] for x in ["python","numpy","torch_cpu","torch_cuda"])
 assert s["trainer_state"]=={"refresh_round":3,"source_bank_sha256":"723350c272dd2bea64b0103b3573ae7185d1203e9002abab274f7a1c2d14b704"}
 summary={"step":23360,"cursor":93440,"optimizer_parameters":1075,"optimizer_steps":[23360],"rng_keys":list(s["rng_state"]),"trainer_state":s["trainer_state"],"sha256":h.hexdigest(),"host":os.uname().nodename,"utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
(out/"resume-before.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary))
