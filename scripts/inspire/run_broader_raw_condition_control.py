"""Use the retained single GPU for the complete frozen raw-condition readback."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

root=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
runs=root/'runs/dreamlite-official-alignment'
source=root/'repos/dreamlite-broader-raw-control-20260914'
commit='1f86d56fd51cfd6b96ba4cba39bcbfc26093d251'
output=runs/'1f86d56-broader-raw-condition'
status=runs/'1f86d56-broader-raw-driver-status.json'
deadline=1789335600
if status.exists() or output.exists():raise ValueError('Output exists; inspect before any new execution')
child=None
def record(state,**extra):
    status.write_text(json.dumps({'state':state,'time_unix':time.time(),'deadline_unix':deadline,
        'probe_commit':commit,'notebook':'dl-logical-val-h200x1-20260914','optimizer_updates':0,**extra},indent=2)+'\n')
def terminate(signum,frame):raise TimeoutError('Raw-condition control interrupted')
signal.signal(signal.SIGTERM,terminate)
try:
    record('waiting_for_complete_branch_audit')
    while True:
        previous=json.loads((runs/'105f521-native-branch-driver-status.json').read_bytes())
        if previous['state']=='failed':raise RuntimeError('Branch audit failed; inspect it before further work')
        if previous['state']=='completed' and not subprocess.check_output(
                ['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():break
        if deadline-time.time()<4500:raise TimeoutError('Insufficient75-minute window for complete readback; do not run a subset')
        time.sleep(10)
    if deadline-time.time()<4500:raise TimeoutError('Insufficient75-minute window for complete readback')
    command=[sys.executable,'-u',str(source/'scripts/probes/broader_raw_condition_control.py'),
        '--parent-run',str(runs/'bb34092-logical31-full4832'),'--output',str(output),
        '--expected-commit',commit,'--deadline-unix',str(deadline)]
    with (runs/'1f86d56-broader-raw-condition.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record('running',probe_pid=child.pid,command=command)
        code=child.wait(timeout=deadline-time.time())
    if code or not (output/'complete.json').exists():raise RuntimeError('Full raw-condition control did not complete; preserve partial evidence')
    complete=json.loads((output/'complete.json').read_bytes())
    record('completed',native_correct_eos=complete['native_summary']['correct_eos'],
        raw_condition_correct_eos=complete['raw_summary']['correct_eos'],
        complete_sha256=hashlib.sha256((output/'complete.json').read_bytes()).hexdigest(),
        independent_functional_validation_still_required=True)
except BaseException as error:
    record('failed',error=str(error))
    raise
finally:
    if child is not None and child.poll() is None:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait()
