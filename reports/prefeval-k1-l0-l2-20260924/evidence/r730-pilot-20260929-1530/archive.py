from pathlib import Path
import json,os,subprocess,datetime,fcntl
p=Path("/inspire/ssd/project/exploration-topic/czxs26210936")
r=p/"runs/prefeval-b-mcq-20260925"
out=r/"r730-pilot-20260929-1530"
rows=[]
for d in Path("/proc").iterdir():
 if not d.name.isdigit() or int(d.name)==os.getpid():continue
 try:
  args=(d/"cmdline").read_bytes().decode().split(chr(0))
  if len(args)<2 or not any("prefeval" in a for a in args[1:]) or "-c" in args:continue
  rows.append({"pid":int(d.name),"cmd":args,"cwd":str((d/"cwd").resolve()),"cuda":[x for x in (d/"environ").read_bytes().decode().split(chr(0)) if x.startswith("CUDA_VISIBLE_DEVICES")],"locks":[str(x.resolve()) for x in (d/"fd").iterdir() if "lock" in str(x.resolve())]})
 except OSError:pass
states={}
for arm in ["C","R"]:
 base=r/("retain730-"+arm)
 states[arm]={"state":json.loads((base/"driver-state.json").read_text()),"log":(base/"driver-high-eval-recovery-20260929.log").read_text()[-3000:]}
counts={}
for name,base in [("C",r/"retain730-C/final-train-V1"),("Rdev",r/"retain730-R/final-dev-V1"),("Rpilot",r/"retain730-R/final-pilot-V1"),("Rtrain",r/"retain730-R/final-train-V1")]:
 counts[name]={"complete":len(list(base.rglob("complete.json"))),"png":len(list(base.rglob("*.png")))}
locks={}
for name in ["retain730-C/refresh-driver.lock","retain730-C/launcher.lock","retain730-R/refresh-driver.lock"]:
 with (r/name).open("a") as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks[name]="FREE";fcntl.flock(f,fcntl.LOCK_UN)
  except BlockingIOError:locks[name]="HELD"
data={"utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"host":os.uname().nodename,"processes":rows,"states":states,"counts":counts,"locks":locks,"gpus":subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader"],text=True)}
(out/"runtime.json").write_text(json.dumps(data,indent=2)+"\n")

import hashlib,tarfile,shutil,base64
(out/"analyze.py").write_bytes(base64.b64decode("ZnJvbSBwYXRobGliIGltcG9ydCBQYXRoCmltcG9ydCBqc29uLHN5cyxoYXNobGliLGNvbGxlY3Rpb25zLGRhdGV0aW1lLG9zLHNodXRpbCx0YXJmaWxlCnA9UGF0aCgnL2luc3BpcmUvc3NkL3Byb2plY3QvZXhwbG9yYXRpb24tdG9waWMvY3p4czI2MjEwOTM2Jyk7cj1wLydydW5zL3ByZWZldmFsLWItbWNxLTIwMjYwOTI1JztyZXBvPXAvJ3JlcG9zL3ByZWZldmFsLWItcmVmcmVzaC01NTU5NjgxJztvdXQ9ci8ncjczMC1waWxvdC0yMDI2MDkyOS0xNTMwJwpvdXQubWtkaXIoZXhpc3Rfb2s9VHJ1ZSkKc3lzLnBhdGguaW5zZXJ0KDAsc3RyKHJlcG8pKQpmcm9tIHNjcmlwdHMuZXhwZXJpbWVudHMucHJlZmV2YWxfazFfZGF0YSBpbXBvcnQgbG9hZF9yZWNvcmRzLG9mZmljaWFsX21jcSxzaGEsb3B0aW9uX29yZGVyCnBhcnNlPW9mZmljaWFsX21jcShyZXBvLyd0aGlyZF9wYXJ0eS9wcmVmZXZhbF9yZWZlcmVuY2UnKVsnZXh0cmFjdF9jaG9pY2UnXQppZHM9W3hbJ2Jhc2VfcGFpcl9pZCddIGZvciB4IGluIGxvYWRfcmVjb3JkcygncGlsb3QnKV07Rj1bJ1QxJywnVDInLCdUMycsJ08xJywnTzInXTtEPVswLDEsNSwxMF0KYT1yLydyZXRhaW43MzAtUic7cm93cz1banNvbi5sb2Fkcyh4KSBmb3IgeCBpbiAoYS8ncmVhZGJhY2stZmluYWwtcGlsb3QtVjEvcmVhZGJhY2stMC5qc29ubCcpLnJlYWRfdGV4dCgpLnNwbGl0bGluZXMoKV0KaWR4PXsoeFsncGFpcl9pZCddLHhbJ2NoYWluJ10seFsncHJlZml4J10seFsnY29udHJvbCddLHhbJ2ZhbWlseSddKTp4IGZvciB4IGluIHJvd3N9CmV4cGVjdGVkPXsoaSxjLGQsayxmKSBmb3IgaSBpbiBpZHMgZm9yIGMgaW4gcmFuZ2UoMikgZm9yIGQgaW4gRCBmb3IgayBpbiBbJ21lbW9yeScsJ21pc21hdGNoJ10gZm9yIGYgaW4gRn0KZXhwZWN0ZWR8PXsoaSwwLGQsJ3RleHQnLGYpIGZvciBpIGluIGlkcyBmb3IgZCBpbiBEIGZvciBmIGluIEZ9CmV4cGVjdGVkfD17KGksMCwwLCdibGFuaycsZikgZm9yIGkgaW4gaWRzIGZvciBmIGluIEZ9CmFzc2VydCBsZW4oaWR4KT09bGVuKHJvd3MpPT02NzIwIGFuZCBzZXQoaWR4KT09ZXhwZWN0ZWQKY2hlY2tzPXsncm93cyc6NjcyMCwncG9zaXRpb25fcm93cyc6MCwncG5ncyc6MCwnY2hhaW5zJzowLCdwYXJzZV9mYWlsdXJlcyc6MCwndHJ1bmNhdGlvbnMnOjB9CmNwPXNoYShhLyd0cmFpbi9jaGVja3BvaW50LWZpbmFsLnB0JykKYXNzZXJ0IGNwPT0nNzg5YTQ5ZmM0ODhkZmEzNzI5MTdiMzY0NjYyNWI2Y2Q2Mjk0NmNiNDgxODU3NjNlYWRlODJkMjJlN2RiYzg0NScKZm9yIGRvbmUgaW4gKGEvJ2ZpbmFsLXBpbG90LVYxJykuZ2xvYignKi9zZWVkLSovY29tcGxldGUuanNvbicpOgogaW5mbz1qc29uLmxvYWRzKGRvbmUucmVhZF90ZXh0KCkpO3dyaXRlcz1banNvbi5sb2Fkcyh4KSBmb3IgeCBpbiAoZG9uZS5wYXJlbnQvJ3dyaXRlcy5qc29ubCcpLnJlYWRfdGV4dCgpLnNwbGl0bGluZXMoKV0KIGFzc2VydCBpbmZvWydiaW5kaW5nJ11bJ2NoZWNrcG9pbnRfc2hhMjU2J109PWNwIGFuZCBsZW4od3JpdGVzKT09MTEKIGZvciBpLHcgaW4gZW51bWVyYXRlKHdyaXRlcyk6CiAgcG5nPWRvbmUucGFyZW50LygncHJlZml4LSUwMmQucG5nJyVpKQogIGFzc2VydCBzaGEocG5nKT09aW5mb1sncG5nX2hhc2hlcyddW3BuZy5uYW1lXT09d1snb3V0cHV0X3BuZ19zaGEyNTYnXQogIGlmIGk6YXNzZXJ0IHdbJ3NvdXJjZV9wbmdfc2hhMjU2J109PXdyaXRlc1tpLTFdWydvdXRwdXRfcG5nX3NoYTI1NiddCiAgY2hlY2tzWydwbmdzJ10rPTEKIGNoZWNrc1snY2hhaW5zJ10rPTEKYXNzZXJ0IGNoZWNrc1sncG5ncyddPT0xNDA4IGFuZCBjaGVja3NbJ2NoYWlucyddPT0xMjgKZm9yIHggaW4gcm93czoKIHByZWQ9cGFyc2UoeFsnZ2VuZXJhdGVkJ11bJ3JhdyddKTthc3NlcnQgcHJlZD09eFsncHJlZGljdGVkX2xldHRlciddCiBhc3NlcnQgeFsnY29ycmVjdF9sZXR0ZXInXT09J0FCQ0QnW3hbJ29wdGlvbl9vcmRlciddLmluZGV4KDApXQogYXNzZXJ0IHhbJ2NvcnJlY3QnXT09KHByZWQ9PXhbJ2NvcnJlY3RfbGV0dGVyJ10pCiBjaGVja3NbJ3BhcnNlX2ZhaWx1cmVzJ10rPXByZWQgaXMgTm9uZTtjaGVja3NbJ3RydW5jYXRpb25zJ10rPXhbJ2dlbmVyYXRlZCddWyd0cnVuY2F0ZWQnXQogaWYgeFsnY29udHJvbCddIGluIFsnbWVtb3J5JywnbWlzbWF0Y2gnXToKICBzb3VyY2U9eFsncGFpcl9pZCddIGlmIHhbJ2NvbnRyb2wnXT09J21lbW9yeScgZWxzZSB4Wydkb25vcl9wYWlyX2lkJ10KICBhc3NlcnQgc291cmNlIGluIGlkcyBhbmQgKHNvdXJjZSE9eFsncGFpcl9pZCddIGlmIHhbJ2NvbnRyb2wnXT09J21pc21hdGNoJyBlbHNlIFRydWUpCiAgcGF0aD1hLydmaW5hbC1waWxvdC1WMScvc291cmNlLnJlcGxhY2UoJzonLCdfJykvKCdzZWVkLSVkJyV4WydjaGFpbiddKS8oJ3ByZWZpeC0lMDJkLnBuZycleFsncHJlZml4J10pCiAgYXNzZXJ0IHN0cihwYXRoKT09eFsncG5nX3BhdGgnXSBhbmQgc2hhKHBhdGgpPT14Wydwbmdfc2hhMjU2J10KIGRlZiBzdGVwKHBpZCxmKTpyZXR1cm4gaW50LmZyb21fYnl0ZXMoaGFzaGxpYi5zaGEyNTYoKCdldmFsOicrcGlkKyc6JytmKS5lbmNvZGUoKSkuZGlnZXN0KClbOjRdLCdiaWcnKQogYXNzZXJ0IHhbJ29wdGlvbl9vcmRlciddPT1vcHRpb25fb3JkZXIoeFsncGFpcl9pZCddLHN0ZXAoeFsncGFpcl9pZCddLHhbJ2ZhbWlseSddKSlbMF0KZGVmIGNvcnJlY3QoaSxjLGQsayxmKTpyZXR1cm4gaWR4W2ksYyxkLGssZl1bJ2NvcnJlY3QnXQpzdGF0cz1bXQpmb3IgZCBpbiBEOgogZm9yIGYgaW4gRjoKICBwYWlycz1bKGNvcnJlY3QoaSxjLGQsJ21lbW9yeScsZiksY29ycmVjdChpLGMsZCwnbWlzbWF0Y2gnLGYpKSBmb3IgaSBpbiBpZHMgZm9yIGMgaW4gcmFuZ2UoMildCiAgc3RhdHMuYXBwZW5kKHsnZGVwdGgnOmQsJ2ZhbWlseSc6ZiwnbWF0Y2hlZCc6c3VtKG0gZm9yIG0sbiBpbiBwYWlycyksJ21pc21hdGNoJzpzdW0obiBmb3IgbSxuIGluIHBhaXJzKSwnbmV0JzpzdW0obS1uIGZvciBtLG4gaW4gcGFpcnMpLCdncmF5JzpzdW0oY29ycmVjdChpLDAsMCwnYmxhbmsnLGYpIGZvciBpIGluIGlkcyksJ3RleHQnOnN1bShjb3JyZWN0KGksMCxkLCd0ZXh0JyxmKSBmb3IgaSBpbiBpZHMpLCdyZXBhaXInOnN1bShtIGFuZCBub3QgbiBmb3IgbSxuIGluIHBhaXJzKSwncmVncmVzcyc6c3VtKG4gYW5kIG5vdCBtIGZvciBtLG4gaW4gcGFpcnMpLCdib3RoX25vaXNlJzpzdW0oYWxsKGNvcnJlY3QoaSxjLGQsJ21lbW9yeScsZikgZm9yIGMgaW4gcmFuZ2UoMikpIGZvciBpIGluIGlkcyl9KQpqb2ludD17J29vZF9ieV9kZXB0aCc6e2Q6eydzYW1lX2ltYWdlJzpzdW0oYWxsKGNvcnJlY3QoaSxjLGQsJ21lbW9yeScsZikgZm9yIGYgaW4gWydPMScsJ08yJ10pIGZvciBpIGluIGlkcyBmb3IgYyBpbiByYW5nZSgyKSksJ2JvdGhfbm9pc2UnOnN1bShhbGwoY29ycmVjdChpLGMsZCwnbWVtb3J5JyxmKSBmb3IgZiBpbiBbJ08xJywnTzInXSBmb3IgYyBpbiByYW5nZSgyKSkgZm9yIGkgaW4gaWRzKX0gZm9yIGQgaW4gRH0sJ3JlZ2lzdGVyZWRfZGVwdGhzX2FuZF9ib3RoX25vaXNlJzp7ZjpzdW0oYWxsKGNvcnJlY3QoaSxjLGQsJ21lbW9yeScsZikgZm9yIGMgaW4gcmFuZ2UoMikgZm9yIGQgaW4gRCkgZm9yIGkgaW4gaWRzKSBmb3IgZiBpbiBGfSwnb29kX3JlZ2lzdGVyZWRfZGVwdGhzX2FuZF9ib3RoX25vaXNlJzpzdW0oYWxsKGNvcnJlY3QoaSxjLGQsJ21lbW9yeScsZikgZm9yIGMgaW4gcmFuZ2UoMikgZm9yIGQgaW4gRCBmb3IgZiBpbiBbJ08xJywnTzInXSkgZm9yIGkgaW4gaWRzKX0KcG9zPXt9CmZvciBkIGluIFswLDEwXToKIHByPVtqc29uLmxvYWRzKHgpIGZvciB4IGluIChhLygncG9zaXRpb25zLWZpbmFsLXBpbG90LVYxLSVkLmpzb25sJyVkKSkucmVhZF90ZXh0KCkuc3BsaXRsaW5lcygpXQogcGk9eyh4WydwYWlyX2lkJ10seFsnY2hhaW4nXSx4Wydwb3NpdGlvbiddKTp4IGZvciB4IGluIHByfQogYXNzZXJ0IGxlbihwaSk9PWxlbihwcik9PTY0MCBhbmQgc2V0KHBpKT09eyhpLGMscSkgZm9yIGkgaW4gaWRzIGZvciBjIGluIHJhbmdlKDIpIGZvciBxIGluIFsnb2ZmaWNpYWwnLDAsMSwyLDNdfQogZm9yIHggaW4gcHI6CiAgcHJlZD1wYXJzZSh4WydnZW5lcmF0ZWQnXVsncmF3J10pCiAgYXNzZXJ0IHByZWQ9PXhbJ3ByZWRpY3RlZF9sZXR0ZXInXSBhbmQgeFsnY29ycmVjdF9sZXR0ZXInXT09J0FCQ0QnW3hbJ29yZGVyJ10uaW5kZXgoMCldIGFuZCB4Wydjb3JyZWN0J109PShwcmVkPT14Wydjb3JyZWN0X2xldHRlciddKQogIGFzc2VydCB4Wydwbmdfc2hhMjU2J109PWlkeFt4WydwYWlyX2lkJ10seFsnY2hhaW4nXSxkLCdtZW1vcnknLCdUMSddWydwbmdfc2hhMjU2J10KICBjaGVja3NbJ3BhcnNlX2ZhaWx1cmVzJ10rPXByZWQgaXMgTm9uZTtjaGVja3NbJ3RydW5jYXRpb25zJ10rPXhbJ2dlbmVyYXRlZCddWyd0cnVuY2F0ZWQnXQogY2hlY2tzWydwb3NpdGlvbl9yb3dzJ10rPWxlbihwcikKIHBvc1tkXT17J3Blcl9wb3NpdGlvbic6e3N0cihxKTpzdW0ocGlbaSxjLHFdWydjb3JyZWN0J10gZm9yIGkgaW4gaWRzIGZvciBjIGluIHJhbmdlKDIpKSBmb3IgcSBpbiBbJ29mZmljaWFsJywwLDEsMiwzXX0sJ2FsbF9mb3VyJzpzdW0oYWxsKHBpW2ksYyxxXVsnY29ycmVjdCddIGZvciBxIGluIHJhbmdlKDQpKSBmb3IgaSBpbiBpZHMgZm9yIGMgaW4gcmFuZ2UoMikpLCdhbGxfZm91cl9ib3RoX25vaXNlJzpzdW0oYWxsKHBpW2ksYyxxXVsnY29ycmVjdCddIGZvciBxIGluIHJhbmdlKDQpIGZvciBjIGluIHJhbmdlKDIpKSBmb3IgaSBpbiBpZHMpfQpzdW1tYXJ5PXsnY2hlY2tzJzpjaGVja3MsJ21ldHJpY3MnOnN0YXRzLCdqb2ludCc6am9pbnQsJ3Bvc2l0aW9ucyc6cG9zLCdjaGVja3BvaW50X3NoYTI1Nic6Y3B9CihvdXQvJ1ItcGlsb3Qtc3VtbWFyeS5qc29uJykud3JpdGVfdGV4dChqc29uLmR1bXBzKHN1bW1hcnksaW5kZW50PTIpKydcbicpCgpwcmludChqc29uLmR1bXBzKHN1bW1hcnkpKQo="))
a=r/"retain730-R"
for k in range(4):
 assert json.loads((a/("final-pilot-V1-evaluation-%d-exit.json"%k)).read_text())["exit_codes"]==[0]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def copy(f):
 dest=out/"snapshots"/f.relative_to(r);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
for base in [a/"final-pilot-V1",a/"readback-final-pilot-V1"]:
 for f in base.rglob("*"):
  if f.is_file() and f.suffix in [".json",".jsonl",".log"]:copy(f)
for arm in ["C","R"]:
 for f in (r/("retain730-"+arm)).iterdir():
  if f.is_file() and f.suffix in [".json",".jsonl",".log"] and (arm=="R" and ("pilot" in f.name or "train" in f.name) or "driver" in f.name):copy(f)
for f in [a/"train/complete.json",r/"variants-train.json"]:copy(f)
inventory=[]
for f in (a/"final-pilot-V1").rglob("*.png"):
 inventory.append({"path":str(f.relative_to(r)),"sha256":sha(f),"bytes":f.stat().st_size,"mtime_ns":f.stat().st_mtime_ns})
assert len(inventory)==1408
(out/"png-inventory.json").write_text(json.dumps(inventory,indent=2)+"\n")
files=[{"path":str(f.relative_to(out)),"sha256":sha(f),"bytes":f.stat().st_size} for f in sorted(out.rglob("*")) if f.is_file()]
(out/"snapshot-files.json").write_text(json.dumps(files,indent=2)+"\n")
assert not (out/"raw.tar.gz").exists()
with tarfile.open(out/"raw.tar.gz","w:gz") as t:
 for f in sorted(out.rglob("*")):
  if f.is_file() and f.name!="raw.tar.gz":t.add(f,arcname=str(f.relative_to(out)))
result={"sha256":sha(out/"raw.tar.gz"),"bytes":(out/"raw.tar.gz").stat().st_size,"files":len(files)}
(out/"archive-summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
