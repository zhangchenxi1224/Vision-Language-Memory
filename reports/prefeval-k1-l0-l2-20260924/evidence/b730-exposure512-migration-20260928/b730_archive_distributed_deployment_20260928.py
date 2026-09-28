"""Archive deployment/role-switch evidence after first checkpoint verification."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

root=Path('/inspire/ssd/project/exploration-topic/czxs26210936')
run=root/'runs/prefeval-b730-exposure512-20260927'
out=run/'evidence/distributed-deployment-20260928-final'
assert not out.exists()
verification=json.loads((run/'recovery-20260928-2015/recovery-verification.json').read_text())
assert verification['verified_new_checkpoint_step']>=89856
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
assert sha(run/'recovery-20260928-2000/train/optimization.jsonl')==sha(run/'recovery-20260928-2015/train/optimization.jsonl')
out.mkdir(parents=True)
names=['train-adoption.json','coordinator-switch-20260928/primary/switch.json','coordinator-switch-20260928/reader/switch.json','controller.json','reader-controller.json','runtime-primary.json','runtime-reader.json','preflight.json',
    'launch.log','reader-launch.log','train/manifest.json','train/continuation.json',
    'recovery-20260928-2000/recovery.json','recovery-20260928-2000/launch.json',
    'recovery-20260928-2015/recovery.json','recovery-20260928-2015/launch.json',
    'recovery-20260928-2015/recovery-verification.json','recovery-20260928-2015/archive-receipt.json',
    'reader-deployment-20260928/before.json','reader-deployment-20260928/launch.json',
    'priority-switch-20260928/primary/stop.json','priority-switch-20260928/auxiliary/stop.json']
for name in names:
    destination=out/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(run/name,destination)
runtime={role:json.loads((out/f'runtime-{role}.json').read_text()) for role in ['primary','reader']}
for role,record in runtime.items():
    controller=record['controllers'][role]
    assert controller['observed_locally'] and controller['pid_present']
    assert controller['commit']=='656fdf029c4a7c05e53bccb75483a03cf62d5f54'
    assert controller['scheduler_commit']=='44d251dc629212b61f7373de4875b74dbc459b76'
    assert controller['actual_cwd']==str(run/'code')
    assert controller['actual_cmdline'].strip().endswith('run_prefeval_b730_exposure512_distributed.py '+role)
    assert record['failure'] is None and record['reader_failure'] is None
    for name in (['launcher.lock','controller.lock'] if role=='primary' else ['reader-launcher.lock','reader-controller.lock']):
        assert record['locks_held'][name]
assert len(runtime['primary']['summaries'])==13
adoption=json.loads((run/'train-adoption.json').read_text())
switch=json.loads((run/'coordinator-switch-20260928/primary/switch.json').read_text())
assert adoption['pid']==switch['training_pid']==65159
assert adoption['process_start_ticks']==switch['training_start_ticks']
active=[p for p in runtime['primary']['processes'] if p['label']=='train']
assert len(active)==1 and active[0]['pid']==65159 and active[0]['pid_present']
assert runtime['primary']['latest_optimization']['step']>switch['latest_step_before']
ops=run/'operations/distributed-20260928/code'
for name in ['scripts/inspire/run_prefeval_b730_exposure512_distributed.py','scripts/inspire/launch_prefeval_b730_exposure512_distributed.sh','scripts/inspire/run_prefeval_b730_exposure512_two_gpu.py','tests/test_prefeval_b730_two_gpu_schedule.py']:
    destination=out/'scheduler-source'/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ops/name,destination)
shutil.copy2(__file__,out/'archive-deployment.py')
proof=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    initial_two_gpu_deployment_added_no_optimizer_updates=True,initial_and_final_before_log_sha256=sha(run/'recovery-20260928-2015/train/optimization.jsonl'),
    frozen_commit='656fdf029c4a7c05e53bccb75483a03cf62d5f54',scheduler_commit='44d251dc629212b61f7373de4875b74dbc459b76',
    hosts={role:record['host'] for role,record in runtime.items()},local_locks_only=True,disjoint_host_ownership=True,training_pid_preserved=65159,training_start_ticks=adoption['process_start_ticks'],latest_step_after_coordinator_replacement=runtime['primary']['latest_optimization']['step'],first_verified_checkpoint=verification['verified_new_checkpoint_step'],
    restored_step=89728,replayed_updates=69,registered_summaries_preserved=13,fixed_final_step=93440,
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
(out/'deployment-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
archive=out/'raw.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(out.rglob('*')):
        if p.is_file() and p!=archive: tar.add(p,arcname=str(p.relative_to(out)),recursive=False)
receipt=dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive))
(out/'archive-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
