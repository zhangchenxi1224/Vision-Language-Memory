"""Replace idle evaluators/coordinators while preserving the running FM process."""
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
ops=run/'operations/distributed-20260928/code'
role=sys.argv[1]
assert role in ('primary','reader')
prefix='' if role=='primary' else 'reader-'
out=run/'coordinator-switch-20260928'/role
assert not out.exists(), 'Inspect previous switch before retry'
assert not (run/'train/checkpoint-final.pt').exists(), 'Stop and inspect if final evaluations have begun'
controller=json.loads((run/(prefix+'controller.json')).read_text())
assert controller['host']==socket.gethostname() and controller['role']==role
assert controller['scheduler_commit']=='2bd72308158ca733a38423e125f820b43d3c2a8b'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ops,text=True).strip()=='44d251dc629212b61f7373de4875b74dbc459b76'
pid=controller['pid']
def command(pid):
    p=Path('/proc')/str(pid)/'cmdline'
    try: return p.read_bytes().replace(b'\0',b' ').decode().strip()
    except FileNotFoundError: return ''
assert command(pid).endswith('run_prefeval_b730_exposure512_distributed.py '+role)
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
training=json.loads((run/'processes/train.json').read_text())
if role=='primary':
    assert training['pid']==65159 and training['host']==socket.gethostname() and training['status']=='running'
    assert command(65159)==' '.join(training['command'])
    assert apps==['65159']
    ticks=(Path('/proc/65159/stat')).read_text().rsplit(')',1)[1].split()[19]
else:
    assert not apps
out.mkdir(parents=True)
for name in [prefix+'controller.json',prefix+'launch.log','processes/train.json','train/manifest.json','runtime-'+role+'.json']:
    dest=out/name
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(run/name,dest)
def tail():
    with (run/'train/optimization.jsonl').open('rb') as f:
        f.seek(max(0,(run/'train/optimization.jsonl').stat().st_size-16384))
        for line in reversed(f.read().splitlines()):
            try:return json.loads(line)
            except ValueError:pass
before=tail()
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+20
while command(pid):
    assert time.monotonic()<deadline
    time.sleep(.2)
locks=[]
for name in [prefix+'launcher.lock',prefix+'controller.lock']:
    handle=(run/name).open('a')
    while True:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            assert time.monotonic()<deadline
            time.sleep(.2)
    locks.append(handle)
exit_path=run/(prefix+'launcher-exit-status.txt')
assert exit_path.read_text().strip()=='143'
shutil.move(str(exit_path),str(out/exit_path.name))
if role=='primary':
    assert command(65159)==' '.join(training['command'])
    assert (Path('/proc/65159/stat')).read_text().rsplit(')',1)[1].split()[19]==ticks
env=dict(os.environ)
env.pop('B730_ADOPT_TRAIN_PID',None)
if role=='primary':env['B730_ADOPT_TRAIN_PID']='65159'
for handle in locks:handle.close()
launcher=int(subprocess.check_output(['bash',str(ops/'scripts/inspire/launch_prefeval_b730_exposure512_distributed.sh'),role],env=env,text=True).strip())
report=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=socket.gethostname(),role=role,
    old_controller_pid=pid,new_launcher_pid=launcher,training_pid=training['pid'],training_process_preserved=role=='primary',
    training_start_ticks=ticks if role=='primary' else None,latest_step_before=before['step'],latest_step_after=tail()['step'],
    scheduler_commit='44d251dc629212b61f7373de4875b74dbc459b76',scheduler_directory=str(ops),
    reason='Observed flock scope is local; use fixed disjoint host ownership with no cross-host fallback')
(out/'switch.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(__file__,out/'replace-coordinator.py')
print(json.dumps(report))
