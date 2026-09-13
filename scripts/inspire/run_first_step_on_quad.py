import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
runs=root/'runs/dreamlite-official-alignment'
source=root/'repos/dreamlite-logical-first-step-20260914'
status=runs/'17f35be-first-step-quad-driver-status.json'
output=runs/'17f35be-first-step-condition'
commit='17f35be7baf9e63376103b1da5d5f3bc84da2af1'
deadline=time.time()+2700
assert not status.exists() and not output.exists()
old=json.loads((runs/'17f35be-first-step-driver-status.json').read_bytes())
assert old['state']=='failed' and old.get('error')=='Bounded diagnostic interrupted'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()==commit
assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
child=None
def record(state,**extra):
 value={'state':state,'time_unix':time.time(),'deadline_unix':deadline,'source_commit':commit,
  'notebook':'dl-official-exp-h200x4-20260914','cuda_visible_devices':'0','migration':'Cancel only the old idle waiting driver; retain old notebook and complete validation. Same302-cell diagnostic, zero optimizer updates.',**extra}
 status.write_text(json.dumps(value,indent=2)+'\n')
def terminate(signum,frame):raise TimeoutError('Bounded migrated diagnostic interrupted')
signal.signal(signal.SIGTERM,terminate)
try:
 command=[sys.executable,'-u',str(source/'scripts/probes/logical_first_step_condition.py'),
  '--parent-run',str(runs/'bb34092-logical31-full4832'),'--output',str(output),
  '--expected-commit',commit,'--deadline-unix',str(deadline)]
 with (runs/'17f35be-first-step-condition.log').open('x') as log:
  child=subprocess.Popen(command,cwd=source,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
   stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record('running',worker_pid=child.pid,command=command)
  code=child.wait(timeout=deadline-time.time())
 if code or not (output/'complete.json').exists():raise RuntimeError('Migrated first-step diagnostic failed; retain all evidence')
 complete=json.loads((output/'complete.json').read_bytes())
 assert complete['cells']==302
 record('completed',cells=302,complete_sha256=hashlib.sha256((output/'complete.json').read_bytes()).hexdigest())
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
