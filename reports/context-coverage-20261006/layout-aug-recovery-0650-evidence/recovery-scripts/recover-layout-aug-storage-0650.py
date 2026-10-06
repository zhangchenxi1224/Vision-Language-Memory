from pathlib import Path
import hashlib,json,os,shutil,socket,subprocess,time,traceback,sys
root=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
run=root/'runs/context-coverage-20261006';out=run/'context-layout-aug-v1'
repo=root/'repos/context-layout-aug-20261006-v2'
store=Path('/inspire/qb-ilm/project/exploration-topic/czxs26210936/runs/context-coverage-20261006/checkpoint-store/layout-aug-v1')
store.parent.mkdir(parents=True,exist_ok=True)
record=store.parent/'storage-recovery-0650.json'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def cmd(*a):return subprocess.check_output(a,text=True)
e=dict(started=time.time(),phase='preflight',host=socket.gethostname(),destination=str(store))
def save():record.write_text(json.dumps(e,indent=2)+'\n')
try:
 assert not record.exists(),'Recovery already attempted: inspect before repeating'
 assert read(out/'status.json')['status']=='failed'
 attempts=[read(p) for p in (out/'attempts').glob('*.json')]
 for a in attempts:assert 'finished' in a
 for pid in [55309]+[a['pid'] for a in attempts]:
  p=Path('/proc')/str(pid)/'cmdline'
  assert not p.exists() or not any(x in p.read_bytes() for x in [b'run_context_layout_aug',b'prefeval_k1_writer'])
 assert not list(run.glob('*/active-owner/owner.json'))
 assert not cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader').strip()
 assert not store.exists()
 assert shutil.disk_usage(store.parent.parent.parent.parent.parent).free>100*1024**3
 e.update(failed_status=read(out/'status.json'),attempts=attempts,failed_gpu_hours=sum(a['finished']-a['started'] for a in attempts if a['exit_code'])/3600,stage_gpu_hours_before=sum(a['finished']-a['started'] for a in attempts)/3600,source_disk=shutil.disk_usage(root)._asdict(),destination_disk=shutil.disk_usage(store.parent.parent.parent.parent.parent)._asdict())
 save()
 import torch
 torch.set_num_threads(1)
 sys.path.insert(0,str(repo));os.chdir(repo)
 from scripts.inspire import run_context_layout_aug as m
 parity=m.verify_parity(out);assert parity==read(out/'native-parity.json')
 for arm in m.ARMS:
  assert sha(out/arm/'resume.pt')==parity['arms'][arm]['resume_step258_sha256']
  state=torch.load(out/arm/'resume.pt',map_location='cpu',weights_only=False)
  assert state['optimizer_step']==258 and state['episode_cursor']==1032
  assert {int(x['step']) for x in state['optimizer']['state'].values()}=={258}
  del state
 e['parity']=parity;e['phase']='copy_and_hash';e['copies']={};save()
 store.mkdir(parents=True)
 for arm in ('native','canonical','augmented'):
  src=out/arm;dest=store/arm
  assert src.resolve()==out.resolve()/arm and not src.is_symlink()
  assert src.resolve().is_relative_to(out.resolve()) and dest.resolve().is_relative_to(store.resolve())
  assert not any(p.is_symlink() for p in src.rglob('*'))
  inventory={str(p.relative_to(src)):dict(size=p.stat().st_size,sha256=sha(p)) for p in src.rglob('*') if p.is_file()}
  shutil.copytree(src,dest)
  actual={str(p.relative_to(dest)):dict(size=p.stat().st_size,sha256=sha(p)) for p in dest.rglob('*') if p.is_file()}
  assert inventory==actual
  e['copies'][arm]=dict(source=str(src),destination=str(dest),files=inventory,verified=time.time());save()
  # Both complete trees exist and every byte is verified before replacing this own directory.
  shutil.rmtree(src);src.symlink_to(dest,target_is_directory=True)
  assert src.resolve()==dest.resolve()
  e['copies'][arm]['linked']=time.time();save()
 for arm in m.ARMS:
  folder=out/arm;log=folder/'optimization.jsonl';raw=log.read_bytes();lines=raw.splitlines(keepends=True)
  assert [json.loads(x)['step'] for x in lines]==list(range(1,len(lines)+1))
  assert len(lines)==288
  assert raw.startswith((m.SOURCE/'optimization.jsonl').read_bytes())
  archived=folder/'recovery-ssd-full-0650';archived.mkdir()
  shutil.copy2(log,archived/'optimization-through-failed-step288.jsonl')
  (archived/'uncommitted-steps259-288.jsonl').write_bytes(b''.join(lines[258:]))
  tmp=folder/'resume.pt.tmp'
  if tmp.exists():tmp.rename(archived/tmp.name)
  log.write_bytes(b''.join(lines[:258]))
  assert len(m.read_lines(log))==258
  assert sha(folder/'resume.pt')==parity['arms'][arm]['resume_step258_sha256']
  e['copies'][arm]['rollback']=dict(to_step=258,archived_rows=30,complete_original_log_sha256=sha(archived/'optimization-through-failed-step288.jsonl'),working_prefix_sha256=sha(log))
  save()
 assert m.verify_parity(out)==parity
 e.update(phase='ready_to_resume',finished=time.time(),source_disk_after=shutil.disk_usage(root)._asdict(),destination_disk_after=shutil.disk_usage(store)._asdict());save()
 shutil.copy2(record,out/'storage-recovery-0650.json')
 print(json.dumps({'phase':e['phase'],'stage_gpu_hours':e['stage_gpu_hours_before'],'disk_free_gib':shutil.disk_usage(root).free/1024**3}),flush=True)
except BaseException:
 e.update(phase='recovery_failed',error=traceback.format_exc(),failed=time.time());save();raise
