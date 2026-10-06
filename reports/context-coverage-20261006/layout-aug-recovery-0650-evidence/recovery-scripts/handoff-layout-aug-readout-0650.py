from pathlib import Path
import os,json,time,socket,subprocess,hashlib,traceback
root=Path('/inspire/ssd/project/exploration-topic/czxs26210936');qb=Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936');run=root/'runs/context-coverage-20261006';out=run/'context-layout-aug-v1'
record=out/'readout-handoff-0650.json'
def read(p):return json.loads(p.read_text())
def save(v):record.write_text(json.dumps(v,indent=2)+'\n')
assert not record.exists(),'Handoff already exists: inspect, never duplicate'
e=dict(pid=os.getpid(),started=time.time(),deadline=1791273600,phase='waiting_for_training',host=socket.gethostname()) # 16:00 CST
save(e)
try:
 while time.time()<e['deadline']:
  state=read(out/'status.json')
  if state['status']=='failed':raise RuntimeError('Training failed; automatic handoff canceled')
  if state['status']=='ready_for_frozen_readout' and not (out/'active-owner').exists():break
  time.sleep(15)
 else:raise RuntimeError('Handoff time budget reached; do not launch a readout too close to notebook stop')
 e['phase']='resource_audit';save(e)
 parent=read(out/'relaunch-storage-0650.json')['controller_pid']
 for pid in [parent]+[read(p)['pid'] for p in (out/'attempts').glob('*.json')]:
  p=Path('/proc')/str(pid)/'cmdline'
  assert not p.exists() or not any(s in p.read_bytes() for s in (b'run_context_layout_aug',b'prefeval_k1_writer'))
 assert not list(run.glob('*/active-owner/owner.json'))
 meta=read(run/'resource-owners-recovery-0650.json');pruned=set(meta['pruned']);items=[]
 for base in (root/'runs',qb/'runs'):
  for folder,dirs,files in os.walk(base,followlinks=False):
   dirs[:]=[d for d in dirs if d not in pruned and not d.startswith('heartbeat-')]
   if 'owner.json' not in files:continue
   p=Path(folder)/'owner.json';raw=p.read_bytes();v=json.loads(raw)
   assert e['host'] not in raw.decode() and 'dl-context-dev-n2-1006' not in raw.decode(),str(p)
   items.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),instance=v.get('instance',v.get('notebook')),host=v.get('host',v.get('hostname'))))
 assert time.time()<e['deadline']
 (run/'resource-owners-readout-0650.json').write_text(json.dumps(dict(checked_at=time.time(),items=items,pruned=sorted(pruned)),indent=2)+'\n')
 python=root/'envs/vlm-r3-ngc2502/bin/python';launcher=qb/'launch-context-layout-aug-readout-0650.py'
 result=subprocess.run([str(python),str(launcher),'dl-context-dev-n2-1006'],capture_output=True,text=True)
 e.update(finished=time.time(),returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
 if result.returncode:raise RuntimeError('Readout preflight/launch failed; preserved diagnostics')
 e['phase']='readout_launched';save(e)
except BaseException:
 e.update(phase='stopped',finished=time.time(),error=traceback.format_exc());save();raise
