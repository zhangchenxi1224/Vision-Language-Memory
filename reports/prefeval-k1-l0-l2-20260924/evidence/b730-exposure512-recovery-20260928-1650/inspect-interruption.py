import fcntl,hashlib,json,time
from pathlib import Path
root=Path('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-b730-exposure512-20260927')
out=root/'recovery-20260928-1650'
locks={}
for name in ['launcher.lock','controller.lock']:
    with (root/name).open('a') as f:
        try:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            locks[name]='FREE'
            fcntl.flock(f,fcntl.LOCK_UN)
        except BlockingIOError: locks[name]='HELD'
resume=root/'train/resume.pt'
hasher=hashlib.sha256()
with resume.open('rb') as f:
    for chunk in iter(lambda: f.read(8*1024*1024),b''): hasher.update(chunk)
digest=hasher.hexdigest()
before=json.loads((out/'recovery.json').read_text())
with (root/'train/optimization.jsonl').open('rb') as f:
    f.seek(max(0,f.seek(0,2)-16384)); lines=f.read().splitlines()
last=json.loads(lines[-1])
with (root/'logs/train.log').open('rb') as f:
    f.seek(max(0,f.seek(0,2)-8192)); train_tail=f.read().decode(errors='replace').splitlines()[-5:]
data=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),inspection_host='shared storage via CPU notebook; original GPU host not directly inspected',platform_status_observed='PENDING',locks=locks,resume_sha256=digest,resume_bytes=resume.stat().st_size,resume_matches_recovery_before=digest==before['resume_sha256'],last_optimization_step=last['step'],train_log_tail=train_tail,completed_summaries=len(list((root/'readback').glob('*/summary.json'))),verification_log=(out/'verification.log').read_text(),recovery_verified=(out/'recovery-verification.json').exists(),archive_ready=(out/'archive-receipt.json').exists(),training_complete=(root/'train/complete.json').exists(),root_complete=(root/'complete.json').exists())
assert digest==before['resume_sha256'] and last['step']==89792
assert locks=={'launcher.lock':'FREE','controller.lock':'FREE'}
assert not data['recovery_verified'] and not data['archive_ready'] and not data['training_complete']
(out/'interruption-shared-audit.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(data,indent=2))
