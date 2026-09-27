import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import torch

root=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
out=root/'evidence/loss-curve-20260927'
assert not out.exists(), 'Use a new export directory for a new snapshot'
out.mkdir(parents=True)
with (root/'train/resume.pt').open('rb') as f:
    saved=torch.load(f,map_location='cpu',weights_only=False)
    f.seek(0)
    checkpoint_hash=hashlib.file_digest(f,'sha256').hexdigest()
step=saved['optimizer_step']
assert saved['episode_cursor']==step*4 and len(saved['optimizer']['state'])==1075
assert {int(v['step']) for v in saved['optimizer']['state'].values()}=={step}
assert saved['manifest']['steps']==93440
raw=(root/'train/optimization.jsonl').read_bytes()
lines=raw.splitlines(keepends=True)
if lines and not lines[-1].endswith(b'\n'):
    lines.pop()
records=[]
kept=[]
last_observed=0
for line in lines:
    row=json.loads(line)
    last_observed=row['step']
    if row['step']>step:
        continue
    assert len(row['draws'])==4
    assert all(math.isfinite(d['mse']) and d['mse']>=0 for d in row['draws'])
    records.append(row)
    kept.append(line)
assert [r['step'] for r in records]==list(range(1,step+1))
payload=b''.join(kept)
with gzip.open(out/'optimization-committed.jsonl.gz','wb') as f:
    f.write(payload)
archive=out/'optimization-committed.jsonl.gz'
meta=dict(captured_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source=str(root/'train/optimization.jsonl'),source_resume=str(root/'train/resume.pt'),checkpoint_sha256=checkpoint_hash,committed_step=step,observed_log_step=last_observed,uncommitted_tail_omitted=max(0,last_observed-step),rows=len(records),first_step=1,contiguous_unique=True,effective_batch=4,metric='Unweighted arithmetic mean of the four logged microbatch official flow-matching velocity MSEs, equal to the loss averaged for the optimizer update',original_endpoint=23360,fixed_endpoints=[46720,70080,93440],frozen_commit='656fdf029c4a7c05e53bccb75483a03cf62d5f54',optimizer_states=1075,checkpoint_state_steps_verified=True,raw_jsonl_sha256=hashlib.sha256(payload).hexdigest(),gzip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),gzip_bytes=archive.stat().st_size)
(out/'source.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
