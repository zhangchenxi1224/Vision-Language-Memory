import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
receipt=json.loads((root/'archive-receipt.json').read_text())
assert hashlib.sha256((root/'raw.tar.gz').read_bytes()).hexdigest()==receipt['sha256']
names=['train/optimization.jsonl','optimization-committed-after.jsonl','recovery-verification.json',
       'recovery.json','runtime-after.json','launch.json']
with tarfile.open(root/'raw.tar.gz') as tar:
    content={member.name:tar.extractfile(member).read() for member in tar if member.name in names}
before={r['step']:r for line in content['train/optimization.jsonl'].splitlines() if (r:=json.loads(line))}
after=[json.loads(line) for line in content['optimization-committed-after.jsonl'].splitlines()]
verification=json.loads(content['recovery-verification.json'])
recovery=json.loads(content['recovery.json'])
assert [r['step'] for r in after]==list(range(1,verification['verified_new_checkpoint_step']+1))
assert recovery['restored_step']==74624 and recovery['last_uncommitted_log_step']==74641
for step in range(74625,74642):
    assert before[step]['draws']==after[step-1]['draws']
    assert before[step]['grad_norm']==after[step-1]['grad_norm']
runtime=json.loads(content['runtime-after.json'])
assert runtime['controller']['commit']=='656fdf029c4a7c05e53bccb75483a03cf62d5f54'
running=[p for p in runtime['processes'] if p['status']=='running']
assert any(p['label']=='train' and p['gpu']==0 for p in running)
assert len({p['gpu'] for p in running})==len(running)
for p in running:
    assert p['pid_present'] and p['actual_cwd']==runtime['controller']['repo']
    assert p['actual_cmdline'].strip()==' '.join(p['command'])
assert len(recovery['completed_summary_hashes'])==verification['completed_summaries_preserved']==9
assert sum(len(v['completed_png_metadata']) for v in recovery['evaluation_evidence'].values())==verification['completed_png_metadata_preserved']
assert sum(p['rows'] for v in recovery['evaluation_evidence'].values() for p in v['readback_prefix'].values())==verification['readback_prefix_rows_preserved']
for name in ['recovery-verification.json','runtime-after.json','launch.json']:
    (root/name).write_bytes(content[name])
out=dict(archive_sha256=receipt['sha256'],replayed_updates_exact=17,
    committed_step=verification['verified_new_checkpoint_step'],log_contiguous_unique=True,
    runtime_identity_verified=True,completed_summaries_preserved=9,
    preserved_png_metadata_count=verification['completed_png_metadata_preserved'],
    preserved_readback_row_count=verification['readback_prefix_rows_preserved'])
(root/'local-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
