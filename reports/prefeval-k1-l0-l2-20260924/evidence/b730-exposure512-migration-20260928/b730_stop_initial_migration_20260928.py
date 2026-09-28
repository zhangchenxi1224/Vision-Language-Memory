"""Stop only the just-created task processes for the user's role reversal."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

run=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
role=sys.argv[1]
assert role in ('primary','auxiliary')
out=run/'priority-switch-20260928'/role
assert not out.exists(), 'Inspect switch receipt before any retry'
expected_host='prefeval-b-read-h200x2-20260925--' if role=='primary' else 'prefeval-b-cr-h200x4-20260926--'
assert socket.gethostname().startswith(expected_host)
def command(pid):
    try: return (Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError: return ''
if role=='primary':
    targets=[(128031,'prefeval_k1_write_extension.py'),(128015,'run_prefeval_b730_exposure512_two_gpu.py'),(133879,'b730_verify_recovery_20260928_2000.py')]
    names=['controller.json','launch.log','logs/train.log','train/optimization.jsonl','processes/train.json','recovery-20260928-2000/verification.log']
    locks=['launcher.lock','controller.lock']
else:
    targets=[(40235,'run_prefeval_b730_exposure512_auxiliary.py')]
    names=['auxiliary-controller.json','auxiliary-launch.log']
    locks=['auxiliary-launcher.lock','auxiliary-controller.lock']
before={str(pid):command(pid) for pid,_ in targets}
for pid,needle in targets:
    assert not before[str(pid)] or needle in before[str(pid)], 'PID identity changed'
out.mkdir(parents=True)
for name in names:
    p=run/name
    if p.exists():
        dest=out/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
for pid,needle in targets:
    if needle in command(pid):
        if needle=='prefeval_k1_write_extension.py':
            assert os.getpgid(pid)==pid
            os.killpg(pid,signal.SIGTERM)
            time.sleep(3)
        else:
            os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+25
while any(command(pid) for pid,_ in targets):
    assert time.monotonic()<deadline, 'Task processes did not exit; inspect before restart'
    time.sleep(1)
for name in locks:
    with (run/name).open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
assert not apps.strip(), 'GPU still occupied; do not start replacement'
report=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=socket.gethostname(),reason='User prioritizes four-GPU primary and two-GPU reader',stopped_processes=before,locks_released=locks,compute_apps_empty=True)
(out/'stop.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
