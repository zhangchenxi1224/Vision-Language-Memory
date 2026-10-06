from pathlib import Path
import os,json,time,socket,subprocess,hashlib
root=Path('/inspire/ssd/project/exploration-topic/czxs26210936');run=root/'runs/context-coverage-20261006';out=run/'context-layout-aug-v1';repo=root/'repos/context-layout-aug-20261006-v2'
def read(p):return json.loads(p.read_text())
def cmd(*x):return subprocess.check_output(x,text=True)
assert read(out/'storage-recovery-0650.json')['phase']=='ready_to_resume'
assert read(out/'status.json')['status']=='failed'
assert not (out/'relaunch-storage-0650.json').exists()
assert not list(run.glob('*/active-owner/owner.json'))
attempts=[read(p) for p in (out/'attempts').glob('*.json')]
for a in attempts:assert 'finished' in a
for pid in [55309]+[a['pid'] for a in attempts]:
 p=Path('/proc')/str(pid)/'cmdline'
 assert not p.exists() or not any(x in p.read_bytes() for x in [b'run_context_layout_aug',b'prefeval_k1_writer'])
meta=read(run/'resource-owners-recovery-0650.json')
for x in meta['items']:
 assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
 assert socket.gethostname() not in (x['host'],x['instance']) and x['instance']!='dl-context-dev-n2-1006'
meta['checked_at']=time.time();(run/'resource-owners-recovery-0650.json').write_text(json.dumps(meta,indent=2))
assert not cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader').strip()
free=[int(x.split()[0]) for x in cmd('nvidia-smi','--query-gpu=memory.free','--format=csv,noheader').splitlines()]
assert len(free)==2 and min(free)>120000
assert cmd('git','-C',str(repo),'rev-parse','HEAD').strip()=='844dda1819168b551882577a2946548dc8602f2f'
assert not cmd('git','-C',str(repo),'status','--porcelain').strip()
assert not any('prefeval_' in x and 'python -c' not in x for x in cmd('ps','-eo','pid,args').splitlines())
launch=read(out/'launch-0650.json');command=launch['command']
e=dict(time=time.time(),host=socket.gethostname(),command=command,prior_controller_pid=55309,old_worker_pids=[a['pid'] for a in attempts],resource_owners=meta,compute_empty=True,memory=cmd('free','-m'),gpus=cmd('nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader'),storage_recovery=read(out/'storage-recovery-0650.json'))
env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with (out/'controller.log').open('a') as f:p=subprocess.Popen(command,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
e['controller_pid']=p.pid;(out/'relaunch-storage-0650.json').write_text(json.dumps(e,indent=2)+'\n')
print(json.dumps(dict(controller_pid=p.pid,time=e['time'],resume_step=258)))
