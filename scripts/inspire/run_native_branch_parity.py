"""Run the frozen branch audit on the retained idle one-H200 notebook."""
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
source=root/'repos/dreamlite-native-branch-parity-20260914'
commit='105f52140f80f562e8d077422fef480931fdd506'
output=runs/'105f521-native-branch-parity'
status=runs/'105f521-native-branch-driver-status.json'
deadline=1789332600
if status.exists() or output.exists() or deadline-time.time()<1200:
    raise ValueError('Require unused evidence paths and at least20 minutes before the fixed deadline')
child=None
def record(state,**extra):
    status.write_text(json.dumps({'state':state,'time_unix':time.time(),'deadline_unix':deadline,
        'probe_commit':commit,'notebook':'dl-logical-val-h200x1-20260914','optimizer_updates':0,**extra},indent=2)+'\n')
def terminate(signum,frame):raise TimeoutError('Frozen branch audit interrupted')
signal.signal(signal.SIGTERM,terminate)
try:
    command=[sys.executable,'-u',str(source/'scripts/probes/native_training_branch_parity.py'),
        '--parent-run',str(runs/'bb34092-logical31-full4832'),
        '--new-training-run',str(runs/'03f8467-native-condition-full4832'),
        '--first-step-run',str(runs/'17f35be-first-step-condition'),
        '--output',str(output),'--expected-commit',commit,'--deadline-unix',str(deadline)]
    with (runs/'105f521-native-branch-parity.log').open('x') as log:
        child=subprocess.Popen(command,cwd=source,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record('running',probe_pid=child.pid,command=command)
        code=child.wait(timeout=deadline-time.time())
    if code or not (output/'complete.json').exists():raise RuntimeError('Frozen audit did not complete; inspect retained evidence')
    complete=json.loads((output/'complete.json').read_bytes())
    if complete['cells']!=302:raise ValueError('Incomplete audit')
    record('completed',cells=302,velocity_parity_pass=complete['all_velocity_comparisons_within_tolerance'],
        complete_sha256=hashlib.sha256((output/'complete.json').read_bytes()).hexdigest())
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
