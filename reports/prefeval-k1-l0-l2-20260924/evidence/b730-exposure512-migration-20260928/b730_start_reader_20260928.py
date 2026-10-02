import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

run=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
ops=run/'operations/two-gpu-20260928/code'
out=run/'reader-deployment-20260928'
assert not out.exists(), 'Inspect existing deployment receipt before retry'
assert socket.gethostname().startswith('prefeval-b-read-h200x2-20260925--')
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ops,text=True).strip()=='2bd72308158ca733a38423e125f820b43d3c2a8b'
locks=[]
for name in ['reader-launcher.lock','reader-controller.lock']:
    handle=(run/name).open('a')
    fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locks.append(handle)
processes=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid(): continue
    try: cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except (FileNotFoundError,ProcessLookupError): continue
    assert not any(s in cmd for s in ['prefeval_k1_write_extension.py','prefeval_k1_evaluate.py',
        'run_prefeval_k1_refresh730.py','run_prefeval_b730_exposure512_auxiliary.py',
        'launch_prefeval_b730_exposure512_auxiliary.sh','run_prefeval_b730_exposure512_distributed.py','launch_prefeval_b730_exposure512_distributed.sh']), (p.name,cmd)
    if cmd: processes.append(dict(pid=int(p.name),command=cmd))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu,driver_version','--format=csv'],text=True)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert not apps.strip()
assert len(gpu.strip().splitlines())==3
out.mkdir()
report=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=socket.gethostname(),node='qb-prod-gpu2260',gpu=gpu,
    compute_apps_empty=True,processes=processes,scope='Final step93440 pilot/dev V1 only',
    scheduler_commit='2bd72308158ca733a38423e125f820b43d3c2a8b',frozen_commit='656fdf029c4a7c05e53bccb75483a03cf62d5f54')
(out/'before.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ['run_prefeval_b730_exposure512_distributed.py','launch_prefeval_b730_exposure512_distributed.sh','run_prefeval_b730_exposure512_two_gpu.py']:
    shutil.copy2(ops/'scripts/inspire'/name,out/name)
shutil.copy2(__file__,out/'start-auxiliary.py')
for handle in locks: handle.close()
pid=int(subprocess.check_output(['bash',str(ops/'scripts/inspire/launch_prefeval_b730_exposure512_distributed.sh'),'reader'],text=True).strip())
(out/'launch.json').write_text(json.dumps(dict(launcher_pid=pid,time=time.time()),indent=2)+'\n')
print(json.dumps(dict(host=report['host'],launcher_pid=pid,scope=report['scope'])))
