import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time
run=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
out=run/'priority-switch-20260928/auxiliary'
assert out.exists() and not (out/'stop.json').exists()
assert socket.gethostname().startswith('prefeval-b-cr-h200x4-20260926--')
for pid in [40179,40235]:
    p=Path('/proc')/str(pid)/'cmdline'
    assert not p.exists() or not p.read_bytes(), 'Previous auxiliary still running'
assert (run/'auxiliary-exit-status.txt').read_text().strip()=='143'
locks=['auxiliary-launcher.lock','auxiliary-controller.lock']
for name in locks:
    with (run/name).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
report=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=socket.gethostname(),reason='User prioritizes four-GPU primary and two-GPU reader',stopped_pids=[40179,40235],launcher_exit=143,locks_released=locks,compute_apps_empty=True,notes='Initial immediate lock probe raced launcher exit; subsequent read-only verification confirms complete release')
(out/'stop.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
