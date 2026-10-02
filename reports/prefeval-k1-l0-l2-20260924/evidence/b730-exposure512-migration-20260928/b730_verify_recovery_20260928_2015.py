import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import torch

root=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
recovery=root/'recovery-20260928-2015'
before_state=json.loads((recovery/'recovery.json').read_text())
assert not (recovery/'recovery-verification.json').exists(), 'Already verified; inspect before retry'
deadline=time.monotonic()+900
while (root/'train/resume.pt').stat().st_mtime <= (recovery/'resume-before.pt').stat().st_mtime:
    if (root/'failure.json').exists():
        raise RuntimeError('Controller failure while waiting for the new checkpoint; inspect without restarting')
    if time.monotonic()>=deadline:
        raise TimeoutError('No new full checkpoint after 15 minutes; inspect without restarting')
    time.sleep(10)
with (root/'train/resume.pt').open('rb') as f:
    saved=torch.load(f,map_location='cpu',weights_only=False)
    f.seek(0)
    digest=hashlib.file_digest(f,'sha256').hexdigest()
step=saved['optimizer_step']
assert step>=89856 and saved['episode_cursor']==step*4
assert len(saved['optimizer']['state'])==1075
assert {int(v['step']) for v in saved['optimizer']['state'].values()}=={step}
assert saved['manifest']==json.loads((root/'train/manifest.json').read_text())
assert saved['manifest']['steps']==93440 and saved['manifest']['snapshot_steps']==[46720,70080]
assert len(saved['rng_state']['torch_cuda'])==1

def rows(path):
    out=[]
    for line in path.read_text().splitlines():
        try: out.append(json.loads(line))
        except ValueError: pass
    return out

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

current=[r for r in rows(root/'train/optimization.jsonl') if r['step']<=step]
assert [r['step'] for r in current]==list(range(1,step+1))
before={r['step']:r for r in rows(recovery/'train/optimization.jsonl')}
tail_steps=list(range(before_state['restored_step']+1,before_state['last_uncommitted_log_step']+1))
assert len(tail_steps)==69
for s in tail_steps:
    a,b=before[s],current[s-1]
    assert a['draws']==b['draws'] and a['grad_norm']==b['grad_norm'], f'Replay differs at {s}'
archived=sorted((root/'train').glob('optimization-uncommitted-*.jsonl'))
assert any([r['step'] for r in rows(p)]==tail_steps for p in archived)
for p in archived:
    if [r['step'] for r in rows(p)]==tail_steps: shutil.copy2(p,recovery/p.name)
for name,digest_before in before_state['completed_summary_hashes'].items():
    assert sha(root/name)==digest_before, 'Completed summary changed'
preserved_png_metadata=0
preserved_readback_rows=0
for label,evidence in before_state['evaluation_evidence'].items():
    for name,digest_before in evidence['completed_png_metadata'].items():
        assert sha(root/name)==digest_before
        preserved_png_metadata+=1
    for name,prefix in evidence['readback_prefix'].items():
        with (root/name).open('rb') as f: body=f.read(prefix['bytes'])
        assert hashlib.sha256(body).hexdigest()==prefix['sha256']
        records=rows(root/name)
        assert len(records)==len({tuple(r[k] for k in ['pair_id','chain','prefix','control','family','task']) for r in records})
        preserved_readback_rows+=prefix['rows']
subprocess.check_call(['python3',str(root.parent.parent/'b730_observe_20260927.py')],stdout=subprocess.DEVNULL)
runtime=json.loads((root/'latest-runtime.json').read_text())
assert runtime['failure'] is None and runtime['complete'] is None
assert runtime['controller']['physical_gpus']==[0,1,2,3]
assert runtime['controller']['scheduler_commit']=='2bd72308158ca733a38423e125f820b43d3c2a8b'
assert runtime['controller']['commit']=='656fdf029c4a7c05e53bccb75483a03cf62d5f54'
active=[p for p in runtime['processes'] if p['status']=='running']
assert any(p['label']=='train' and p['gpu']==0 for p in active)
assert len({p['gpu'] for p in active})==len(active)
for p in active:
    assert p['pid_present'] and p['actual_cmdline'].strip()==' '.join(p['command'])
    assert p['actual_cwd']==runtime['controller']['repo']
    if p['label']!='train': assert p['label'].startswith('step-093440-')
shutil.copy2(root/'latest-runtime.json',recovery/'runtime-after.json')
shutil.copy2(root/'train/manifest.json',recovery/'manifest-after.json')
shutil.copy2(root/'launch.log',recovery/'launch-after.log')
(recovery/'optimization-committed-after.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in current))
report=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),restored_from_step=89728,
    verified_new_checkpoint_step=step,optimizer_states=1075,all_optimizer_states_same_step=True,
    full_log_contiguous_unique=True,replayed_updates=len(tail_steps),replayed_loss_grad_and_draws_exact_updates=len(tail_steps),
    max_replayed_mse_delta=0,uncommitted_tail_archived=True,completed_summaries_preserved=13,
    completed_png_metadata_preserved=preserved_png_metadata,readback_prefix_rows_preserved=preserved_readback_rows,
    resume_sha256=digest,fixed_final_step=93440,live_processes=[dict(label=p['label'],pid=p['pid'],gpu=p['gpu']) for p in active])
(recovery/'recovery-verification.json').write_text(json.dumps(report,indent=2)+'\n')
shutil.copy2(__file__,recovery/'verify-recovery.py')
shutil.copy2(root.parent.parent/'b730_recover_20260928_2015.py',recovery/'recover.py')
archive=recovery/'raw.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(recovery.rglob('*')):
        if p.is_file() and p!=archive and p.suffix!='.pt':
            tar.add(p,arcname=str(p.relative_to(recovery)),recursive=False)
receipt=dict(archive=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)
(recovery/'archive-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(verification=report,archive=receipt),indent=2))
