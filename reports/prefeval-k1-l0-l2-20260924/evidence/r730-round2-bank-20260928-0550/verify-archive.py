from pathlib import Path
import json,hashlib,datetime,shutil,tarfile,collections,subprocess
p=Path('/inspire/ssd/project/exploration-topic/czxs26210936');r=p/'runs/prefeval-b-mcq-20260925';a=r/'retain730-R';b=a/'sources/round-2'
out=r/'r730-round2-bank-20260928-0550';out.mkdir(exist_ok=True)
assert not (out/'raw.tar.gz').exists()
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
bank=json.loads((b/'bank.json').read_text());variants=json.loads((r/'variants-train.json').read_text())
expected='8300e2eb389445a7819e8540f35b6b5d0d768e3db52b19f9d93a31592c4da1d3'
assert sha(a/'train/checkpoint-step-011680.pt')==expected==bank['source_checkpoint_sha256']
assert bank['round']==2 and bank['split']=='train' and bank['source_optimizer_step']==11680 and bank['filter']=='none'
assert sha(r/'variants-train.json')==bank['variants_sha256']
assert set(bank['items'])==set(variants['items']) and len(bank['items'])==730
manifests={}
for i in range(2):
 m=json.loads((b/f'shard-{i}/manifest.json').read_text());manifests[i]=m
 assert m['checkpoint_sha256']==expected and m['initial_variant']==0 and m['noise_domain']=='refresh-2'
 assert m['steps']==28 and m['cfg']==1 and m['inter_turns']==9 and m['noise_chains']==1 and m['split']=='train'
 assert m['initial_variants_sha256']==bank['variants_sha256']
 assert len(list((b/f'shard-{i}').rglob('complete.json')))==365
 assert len(list((b/f'shard-{i}').rglob('*.png')))==3650
counts=collections.Counter();recovered=[];pngs=[];chains=set()
for key,rows in bank['items'].items():
 assert [x['depth'] for x in rows]==list(range(10))
 d=(b/rows[0]['png']).parent
 assert d.parent.name==key.replace(':','_') and d.name=='seed-0'
 assert d not in chains;chains.add(d)
 complete=json.loads((d/'complete.json').read_text())
 m=manifests[int(d.relative_to(b).parts[0].split('-')[-1])]
 assert complete['binding']==m
 assert set(complete['png_hashes'])=={f'prefix-{i:02d}.png' for i in range(10)}
 writes=[json.loads(x) for x in (d/'writes.jsonl').read_text().splitlines() if x.strip()]
 counts[len(writes)]+=1
 assert len(writes)>=10
 valid=writes[-10:];old=writes[:-10]
 assert [x['position'] for x in valid]==list(range(10))
 for x in old:
  assert x==valid[x['position']]
 if old:recovered.append({'key':key,'historical_rows':len(old),'prefix_positions':[x['position'] for x in old],'all_fields_equal':True})
 prior=None
 for i,row in enumerate(rows):
  f=b/row['png'];assert f.parent==d
  actual=sha(f);assert actual==row['sha256']==complete['png_hashes'][f.name]==valid[i]['output_png_sha256']
  assert valid[i]['source_png_sha256']==prior
  prior=actual;pngs.append({'path':str(f.relative_to(b)),'sha256':actual})
assert len(pngs)==7300 and len(chains)==730
exits=json.loads((a/'round-2-current-student-rollout-exit.json').read_text());assert exits['exit_codes']==[0,0]
for c in exits['children']:assert not Path('/proc',str(c['pid'])).exists()
resume_sha=sha(a/'train/resume.pt');assert resume_sha=='1c0886430cb5daaa225e2784a8eeefe30cef8d019ca94078acf3f48c27d6e272'
opt=[json.loads(x) for x in (a/'train/optimization.jsonl').read_text().splitlines() if x.strip()]
print('optimization keys',list(opt[-1]))
steps=[x['step'] for x in opt];assert steps==list(range(1,11681))
cache=[json.loads(x) for x in (a/'round-2-fm-11680-17520-0.log').read_text().splitlines() if x.startswith('{"cached"')]
summary={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bank_sha256':sha(b/'bank.json'),'bank_mtime_utc':datetime.datetime.fromtimestamp((b/'bank.json').stat().st_mtime,datetime.timezone.utc).isoformat(),'source_checkpoint_sha256':expected,'preferences':730,'png_files_verified':7300,'all_recursive_bindings_verified':True,'source_variant':0,'noise_domain':'refresh-2','writes_row_counts':dict(counts),'recovered_chains':recovered,'exits':exits,'resume_sha256':resume_sha,'resume_matches_archived_11680':True,'optimization_rows':len(opt),'optimization_last_step':steps[-1],'current_cache':cache[-1] if cache else None,'driver_state':json.loads((a/'driver-state.json').read_text()),'conclusion':'Training source bank completed; cache preparation only, no new memory performance or step11681 gradient verified.'}
(out/'stage-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'png-hashes.json').write_text(json.dumps(pngs,indent=2)+'\n')
def copy(src,dst):
 dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
copy(r/'trust-recovery-20260927/monitor-20260928-0546.json',out/'runtime.json')
for name in ['retain730-C','retain730-R']:
 root=r/name
 for f in root.iterdir():
  if f.is_file() and f.suffix in ['.json','.log']:copy(f,out/'snapshots'/name/f.name)
 for f in (root/'train').iterdir():
  if f.is_file() and f.suffix in ['.json','.jsonl']:copy(f,out/'snapshots'/name/'train'/f.name)
for f in b.glob('shard-*/*/seed-0/*.json*'):copy(f,out/'sources'/f.relative_to(b))
for f in b.glob('shard-*/manifest.json'):copy(f,out/'sources'/f.relative_to(b))
copy(b/'bank.json',out/'sources/bank.json')
copy(r/'variants-train.json',out/'snapshots/variants-train.json')
files=[f for f in out.rglob('*') if f.is_file() and f.name not in ['snapshot-files.json','archive-sha256.txt','raw.tar.gz']]
manifest=[{'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(files)]
(out/'snapshot-files.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(out/'raw.tar.gz','w:gz') as tar:
 for f in sorted(out.rglob('*')):
  if f.is_file() and f.name not in ['raw.tar.gz','archive-sha256.txt']:tar.add(f,arcname=str(f.relative_to(out)))
digest=sha(out/'raw.tar.gz');(out/'archive-sha256.txt').write_text(digest+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['exits','driver_state']},indent=2))
print(json.dumps({'archive_bytes':(out/'raw.tar.gz').stat().st_size,'archive_sha256':digest,'snapshot_files':len(manifest)}))
