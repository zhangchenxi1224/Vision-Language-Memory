import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent/'deployment'
receipt=json.loads((root/'archive-receipt.json').read_text())
assert (root/'raw.tar.gz').stat().st_size==receipt['bytes']
assert hashlib.sha256((root/'raw.tar.gz').read_bytes()).hexdigest()==receipt['sha256']
with tarfile.open(root/'raw.tar.gz') as tar:
    content={m.name:tar.extractfile(m).read() for m in tar if m.isfile()}
proof=json.loads(content['deployment-proof.json'])
for name,digest in proof['files'].items():
    assert hashlib.sha256(content[name]).hexdigest()==digest, name
for role in ['primary','reader']:
    runtime=json.loads(content[f'runtime-{role}.json'])
    controller=runtime['controllers'][role]
    assert controller['observed_locally'] and controller['pid_present']
    assert controller['host']==proof['hosts'][role]==runtime['host']
    assert controller['scheduler_commit']==proof['scheduler_commit']
    assert controller['commit']==proof['frozen_commit']
    assert controller['actual_cmdline'].strip().endswith('run_prefeval_b730_exposure512_distributed.py '+role)
    assert all(runtime['locks_held'][name] for name in (['launcher.lock','controller.lock'] if role=='primary' else ['reader-launcher.lock','reader-controller.lock']))
    (root/f'runtime-{role}.json').write_bytes(content[f'runtime-{role}.json'])
assert proof['initial_two_gpu_deployment_added_no_optimizer_updates']
assert proof['training_pid_preserved']==65159 and proof['disjoint_host_ownership']
assert proof['scheduler_commit']=='44d251dc629212b61f7373de4875b74dbc459b76'
assert proof['first_verified_checkpoint']>=89856 and proof['replayed_updates']==69
for name in ['deployment-proof.json','preflight.json','reader-deployment-20260928/before.json',
             'priority-switch-20260928/primary/stop.json','priority-switch-20260928/auxiliary/stop.json']:
    destination=root/name
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(content[name])
out=dict(archive_sha256=receipt['sha256'],files_verified=len(proof['files']),
    both_hosts_verified=True,first_verified_checkpoint=proof['first_verified_checkpoint'],
    fixed_final_step=proof['fixed_final_step'],no_extra_optimization_from_initial_deployment=True)
(root/'local-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
