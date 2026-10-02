"""Verify the complete downloaded historical readback archive without remote PTs."""
import json
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_historical_readback import collect
from scripts.reporting.collect_transition_endpoint import sha

directory = ROOT / 'reports/official-alignment-results-20260913'
archive = directory / 'historical-fp32-readback-evidence.tgz'
summary = directory / 'historical-fp32-readback-summary.json'
if archive.stat().st_size != 61365808 or sha(archive) != 'b002f07d9ab7ca37f7dadebf8282a0ac0fb312f8a38d37b21bcfbe04b5ddc5c5':
    raise ValueError('Historical readback archive is incomplete or changed')
if sha(summary) != '3ea31c28bcf985c914af17bc5bd11c182feb7ebf06365fd4a9d9448cf5f630eb':
    raise ValueError('Historical readback summary changed')
remote = json.loads(summary.read_bytes())
with tempfile.TemporaryDirectory(prefix='historical-readback-') as temporary:
    root = Path(temporary)
    with tarfile.open(archive) as stream:
        stream.extractall(root, filter='data')
    if (root / 'verified-summary.json').read_bytes() != summary.read_bytes():
        raise ValueError('Archived summary changed')
    local = collect(root, root / 'preregistered-panel.json', text_only=True)
    for key in remote:
        if key not in ('artifacts_omitted_locally', 'actual_tensor_reverification') and local[key] != remote[key]:
            raise ValueError('Local readback audit differs: ' + key)
    if len(local['artifacts_omitted_locally']) != 64:
        raise ValueError('Expected explicit omission of the64 remote PTs')
local.update(archive_sha256=sha(archive), summary_sha256=sha(summary),
    scope='All65 PNGs,720 raw records, original panel and coverage reverified locally;64 PTs remain remote and are explicitly omitted here.')
(directory / 'historical-fp32-readback-local-verification.json').write_text(json.dumps(local, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: v for k, v in local.items() if k not in ('checked_members', 'artifacts_omitted_locally')}))
