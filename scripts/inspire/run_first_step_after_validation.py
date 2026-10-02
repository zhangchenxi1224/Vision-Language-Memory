"""Wait for the existing full suite, then run one frozen first-step diagnostic."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

P=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
RUNS=P/'runs/dreamlite-official-alignment'
COMMIT='17f35be7baf9e63376103b1da5d5f3bc84da2af1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    deadline=1789334100. # 2026-09-14 05:15 CST; below the owned notebook lease.
    status=RUNS/'17f35be-first-step-driver-status.json'
    output=RUNS/'17f35be-first-step-condition'
    lock=(RUNS/'17f35be-first-step-driver.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if status.exists() or output.exists():
        raise ValueError('Retain existing diagnostic evidence; do not launch a duplicate')
    child=None
    def record(state,**extra):
        value={'state':state,'time_unix':time.time(),'deadline_unix':deadline,'probe_commit':COMMIT,
            'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**extra}
        temporary=status.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n')
        temporary.replace(status)
    def terminate(signum,frame):
        raise TimeoutError('Bounded diagnostic interrupted')
    signal.signal(signal.SIGTERM,terminate)
    try:
        if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source,text=True).strip()!=COMMIT
                or subprocess.check_output(['git','status','--porcelain'],cwd=a.source,text=True).strip()):
            raise ValueError('Require the exact clean registered probe checkout')
        env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':'src:.:tests'}
        subprocess.run([sys.executable,'-m','pytest','tests/test_first_step_condition_control.py',
            'tests/test_native_training_condition.py','-q'],cwd=a.source,env={**env,'CUDA_VISIBLE_DEVICES':''},
            check=True,timeout=180)
        record('waiting_for_complete_validation')
        while True:
            if deadline-time.time()<45*60:
                raise TimeoutError('Insufficient remaining lease for a complete302-cell diagnostic')
            prior=json.loads((RUNS/'16bc3d0-logical-serial-suite-status.json').read_bytes())
            if prior['state']=='failed':
                raise RuntimeError('Prior validation failed; inspect its evidence before dispatching anything')
            if prior['state']=='completed':
                if prior['stage']!='all_registered_workloads_finished':
                    raise ValueError('Unexpected previous suite completion')
                break
            time.sleep(15)
        command=[sys.executable,'-u',str(a.source/'scripts/probes/logical_first_step_condition.py'),
            '--parent-run',str(RUNS/'bb34092-logical31-full4832'),'--output',str(output),
            '--expected-commit',COMMIT,'--deadline-unix',str(deadline)]
        with (RUNS/'17f35be-first-step-condition.log').open('w') as log:
            child=subprocess.Popen(command,cwd=a.source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            record('running',worker_pid=child.pid,command=command)
            code=child.wait(timeout=deadline-time.time())
        if code or not (output/'complete.json').is_file():
            raise RuntimeError('First-step diagnostic failed; preserve complete or partial evidence')
        complete=json.loads((output/'complete.json').read_bytes())
        if complete['cells']!=302: raise ValueError('Diagnostic omitted development cells')
        record('completed',cells=302,complete_sha256=hashlib.sha256((output/'complete.json').read_bytes()).hexdigest(),
            functional_success_claimed=False)
    except BaseException as error:
        record('failed',error=str(error),functional_success_claimed=False)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()


if __name__=='__main__':
    main()
