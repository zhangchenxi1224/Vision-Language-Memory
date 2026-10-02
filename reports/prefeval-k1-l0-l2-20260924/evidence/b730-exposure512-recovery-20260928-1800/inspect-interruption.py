from pathlib import Path
import json,hashlib,fcntl,time,shutil
root=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
out=root/'recovery-20260928-1800'
locks={}
for name in ['launcher.lock','controller.lock']:
    with (root/name).open('a') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); locks[name]='FREE'; fcntl.flock(f,fcntl.LOCK_UN)
        except BlockingIOError: locks[name]='HELD'
h=hashlib.sha256()
with (root/'train/resume.pt').open('rb') as f:
    for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
prior=json.loads((out/'recovery.json').read_text())
rows=[json.loads(line) for line in (root/'train/optimization.jsonl').read_text().splitlines() if line.strip()]
before={r['step']:r for r in map(json.loads,(out/'train/optimization.jsonl').read_text().splitlines())}
assert [r['step'] for r in rows]==list(range(1,rows[-1]['step']+1))
assert h.hexdigest()==prior['resume_sha256'] and locks=={'launcher.lock':'FREE','controller.lock':'FREE'}
steps=list(range(89729,89793))
assert rows[-1]['step']>=89792
for s in steps:
    assert before[s]['draws']==rows[s-1]['draws'] and before[s]['grad_norm']==rows[s-1]['grad_norm']
(out/'interruption-replayed-tail.json').write_text(json.dumps([r for r in rows if r['step']>89728],indent=2)+'\n')
shutil.copy2(root/'logs/train.log',out/'interruption-train.log')
data=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),platform_status_observed='PENDING',inspection_host='CPU shared-storage read; original GPU /proc unavailable',locks=locks,resume_step_bound_by_verified_hash=89728,resume_sha256=h.hexdigest(),resume_bytes=(root/'train/resume.pt').stat().st_size,last_optimization_step=rows[-1]['step'],uncommitted_updates=rows[-1]['step']-89728,prior_64_updates_replayed_exact=True,new_full_checkpoint=False,completed_summaries=len(list((root/'readback').glob('*/summary.json'))),verification_log=(out/'verification.log').read_text(),recovery_verified=(out/'recovery-verification.json').exists(),archive_ready=(out/'archive-receipt.json').exists(),training_complete=(root/'train/complete.json').exists(),root_complete=(root/'complete.json').exists(),train_log_tail=(root/'logs/train.log').read_text().splitlines()[-5:])
assert not data['recovery_verified'] and not data['archive_ready'] and not data['training_complete']
(out/'interruption-shared-audit.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
