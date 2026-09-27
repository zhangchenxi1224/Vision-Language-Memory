import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
receipt=json.loads((root/'archive-receipt.json').read_text())
assert hashlib.sha256((root/'raw.tar.gz').read_bytes()).hexdigest()==receipt['sha256']
with tarfile.open(root/'raw.tar.gz') as tar:
    def data(name):
        return tar.extractfile(name).read()
    before={r['step']:r for line in data('train/optimization.jsonl').splitlines() if (r:=json.loads(line))}
    after=[json.loads(line) for line in data('optimization-committed-after.jsonl').splitlines()]
    verification=json.loads(data('recovery-verification.json'))
    assert [r['step'] for r in after]==list(range(1,verification['verified_new_checkpoint_step']+1))
    for step in range(57985,58034):
        assert before[step]['draws']==after[step-1]['draws']
        assert before[step]['grad_norm']==after[step-1]['grad_norm']
    runtime=json.loads(data('runtime-after.json'))
    assert runtime['controller']['commit']=='656fdf029c4a7c05e53bccb75483a03cf62d5f54'
    running=[p for p in runtime['processes'] if p['status']=='running']
    assert len(running)==1 and running[0]['label']=='train'
    for p in running:
        assert p['pid_present'] and p['actual_cwd']==runtime['controller']['repo']
        assert p['actual_cmdline'].strip()==' '.join(p['command'])
out=dict(archive_sha256=receipt['sha256'],replayed_updates_exact=49,committed_step=verification['verified_new_checkpoint_step'],log_contiguous_unique=True,runtime_identity_verified=True)
(root/'local-verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
