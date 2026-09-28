from pathlib import Path
import tarfile,json,hashlib,collections
root=Path(__file__).resolve().parent
archive=root/'raw.tar.gz'
assert archive.stat().st_size==5702607
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='8c394ed8a59c59bb99a217bbadd9004b98058f37c92c2ad3a5fd544368503db7'
with tarfile.open(archive) as tar:
 def read(name):return tar.extractfile(name).read()
 files=json.loads(read('snapshot-files.json'))
 for f in files:
  data=read(f['path'])
  assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'],f['path']
 rows=[json.loads(x) for x in read('snapshots/retain730-R/train/optimization.jsonl').splitlines()]
 assert [x['step'] for x in rows]==list(range(1,17521))
 for k in range(3):
  counts=collections.Counter()
  for row in rows[k*5840:(k+1)*5840]:
   assert len(row['draws'])==4 and sum(d['position']==0 for d in row['draws'])==2
   for d in row['draws']:
    counts[d['pair_id'],'write' if d['position']==0 else 'retain']+=1
    if d['position']>0:assert d['source_round']==k and d['source_index']==d['position']-1
  assert len(counts)==1460 and len({x[0] for x in counts})==730 and set(counts.values())=={16}
 s=json.loads(read('resume-summary.json'))
 assert s['optimizer_step']==17520 and s['episode_cursor']==70080
 assert s['optimizer_parameter_states']==1075 and s['optimizer_steps']==[17520]
 assert set(s['rng_keys'])=={'python','numpy','torch_cpu','torch_cuda'}
 print(json.dumps({'verified_files':len(files),'optimization_rows':len(rows),'three_segments_budget_verified':True,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()},indent=2))
