from pathlib import Path
import os,json,hashlib,datetime,shutil,tarfile,subprocess,torch
p=Path('/inspire/ssd/project/exploration-topic/czxs26210936');r=p/'runs/prefeval-b-mcq-20260925';a=r/'retain730-R';repo=p/'repos/prefeval-b-refresh-5559681'
out=r/'r730-round2-fm-resume-20260928-0640';out.mkdir(exist_ok=True)
assert not (out/'raw.tar.gz').exists()
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for c in iter(lambda:s.read(8388608),b''):h.update(c)
 return h.hexdigest()
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()=='55596810cee2062022b894ef93feb1ab4cd6b07b'
with (a/'train/resume.pt').open('rb') as stream:
 inode=os.fstat(stream.fileno()).st_ino
 h=hashlib.sha256()
 for c in iter(lambda:stream.read(8388608),b''):h.update(c)
 stream.seek(0);payload=torch.load(stream,map_location='cpu',weights_only=False)
s=payload['optimizer_step'];states=payload['optimizer']['state']
summary={'optimizer_step':s,'episode_cursor':payload['episode_cursor'],'trainer_state':payload['trainer_state'],'optimizer_parameter_states':len(states),'optimizer_steps':sorted(set(int(x['step']) for x in states.values())),'rng_keys':list(payload['rng_state']),'sha256':h.hexdigest(),'inode':inode,'snapshot_method':'single open file descriptor; no mutation or copy of resume'}
assert s>11680 and s<17520 and summary['episode_cursor']==s*4
assert len(states)==1075 and summary['optimizer_steps']==[s]
banksha=sha(a/'sources/round-2/bank.json')
assert banksha=='4ca3bc2fc0d11394a1f6e22119a539e7656039a6b438f7ba2a71058b2b44c234'
assert summary['trainer_state']=={'refresh_round':2,'source_bank_sha256':banksha}
assert set(summary['rng_keys'])=={'python','numpy','torch_cpu','torch_cuda'}
assert payload['rng_state']['torch_cpu'].numel()>0 and len(payload['rng_state']['torch_cuda'])==1
raw=(a/'train/optimization.jsonl').read_bytes();assert raw.endswith(b'\n')
opt=[json.loads(x) for x in raw.splitlines()]
assert [x['step'] for x in opt]==list(range(1,len(opt)+1))
for row in opt:
 ds=row['draws'];assert len(ds)==4 and sum(x['position']==0 for x in ds)==2
 for d in ds:
  if d['position']>0:
   assert d['source_round']==(row['step']-1)//5840 and d['source_index']==d['position']-1
assert len(opt)>=s
(out/'resume-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'first-step-11681.json').write_text(json.dumps(opt[11680],indent=2)+'\n')
stage={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'optimization_rows':len(opt),'continuous_unique_steps':True,'four_draws_two_write_two_retain':True,'all_source_rounds_and_positions_verified':True,'bank_sha256':banksha,'resume':summary,'driver_state':json.loads((a/'driver-state.json').read_text()),'first_round2_step':11681}
(out/'stage-summary.json').write_text(json.dumps(stage,indent=2)+'\n')
def copy(f,d):
 d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,d)
copy(r/'trust-recovery-20260927/monitor-20260928-0636.json',out/'runtime.json')
for arm in ['C','R']:
 root=r/('retain730-'+arm)
 for f in root.iterdir():
  if f.is_file() and f.suffix in ['.json','.log']:copy(f,out/'snapshots'/root.name/f.name)
 for f in (root/'train').iterdir():
  if f.is_file() and f.suffix in ['.json','.jsonl']:
   dest=out/'snapshots'/root.name/'train'/f.name
   if arm=='R' and f.name=='optimization.jsonl':
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
   else:copy(f,dest)
for rel in ['scripts/experiments/prefeval_k1_writer.py','src/vision_memory/training/checkpoint.py']:
 copy(repo/rel,out/'frozen-code'/rel)
copy(a/'sources/round-2/bank.json',out/'bank.json')
files=[f for f in out.rglob('*') if f.is_file()]
(out/'snapshot-files.json').write_text(json.dumps([{'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(files)],indent=2)+'\n')
with tarfile.open(out/'raw.tar.gz','w:gz') as tar:
 for f in sorted(out.rglob('*')):
  if f.is_file() and f.name not in ['raw.tar.gz','archive-sha256.txt']:tar.add(f,arcname=str(f.relative_to(out)))
digest=sha(out/'raw.tar.gz');(out/'archive-sha256.txt').write_text(digest+'\n')
print(json.dumps(stage,indent=2))
print('archive_bytes',(out/'raw.tar.gz').stat().st_size)
print('archive_hash_halves',digest[:32],digest[32:])
