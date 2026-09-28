"""Read receipts on each actual host without interpreting remote PIDs locally."""
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

run=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
host=socket.gethostname()
role='primary' if host.startswith('prefeval-b-cr-h200x4-20260926--') else 'reader'
assert role=='primary' or host.startswith('prefeval-b-read-h200x2-20260925--')
def read(path):
    return json.loads(path.read_text()) if path.exists() else None
def tail(path):
    if not path.exists(): return None
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-16384))
        for line in reversed(f.read().decode(errors='replace').splitlines()):
            try: return json.loads(line)
            except ValueError: pass
def identity(record):
    if not record: return None
    r=dict(record)
    r['observed_locally']=r.get('host')==host
    r['pid_present']=None
    if r['observed_locally']:
        p=Path('/proc')/str(r['pid'])
        r['pid_present']=p.exists()
        if p.exists():
            try:
                r['actual_cmdline']=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                r['actual_cwd']=str((p/'cwd').resolve())
            except FileNotFoundError:
                r['pid_present']=False
    return r
locks={}
for p in [run/n for n in ['launcher.lock','controller.lock','reader-launcher.lock','reader-controller.lock']]+list((run/'evaluation-locks').glob('*.lock')):
    with p.open('a') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            held=False
        except BlockingIOError:
            held=True
    locks[str(p.relative_to(run))]=held
data=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=host,role=role,
    controllers={r:identity(read(run/name)) for r,name in [('primary','controller.json'),('reader','reader-controller.json')]},
    processes=[identity(read(p)) for p in sorted((run/'processes').glob('*.json'))],
    locks_held=locks,latest_train_log=tail(run/'logs/train.log'),latest_optimization=tail(run/'train/optimization.jsonl'),
    train_complete=read(run/'train/complete.json'),complete=read(run/'complete.json'),failure=read(run/'failure.json'),
    reader_failure=read(run/'reader-failure.json'),reader_complete=read(run/'reader-complete.json'),
    summaries=sorted(str(p.relative_to(run)) for p in (run/'readback').glob('*/summary.json')),
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu,driver_version','--format=csv'],text=True))
data['resume_stat']=dict(bytes=(run/'train/resume.pt').stat().st_size,mtime=(run/'train/resume.pt').stat().st_mtime)
(run/f'runtime-{role}.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k not in ['processes','latest_optimization','summaries','locks_held']},indent=2))
print('latest_step',data['latest_optimization']['step'],'summaries',len(data['summaries']))
print('local_live_processes',[(p['label'],p['pid'],p['gpu']) for p in data['processes'] if p.get('status')=='running' and p['observed_locally'] and p['pid_present']])
