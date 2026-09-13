"""Reverify the complete downloaded four-GPU warm endpoint and all draws."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_transition_endpoint import collect, sha

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--summary-sha256', required=True)
p.add_argument('--archive-sha256', required=True)
a = p.parse_args()
directory = ROOT / 'reports/official-alignment-results-20260913'
summary_path = directory / 'four-gpu-warm-endpoint-summary.json'
archive_path = directory / 'four-gpu-warm-endpoint-evidence.tgz'
if sha(summary_path) != a.summary_sha256 or sha(archive_path) != a.archive_sha256:
    raise ValueError('Downloaded endpoint differs from the remote digest')
remote = json.loads(summary_path.read_bytes())
with tempfile.TemporaryDirectory(prefix='dreamlite-four-gpu-endpoint-') as temporary:
    root = Path(temporary)
    with tarfile.open(archive_path) as stream:
        stream.extractall(root, filter='data')
    if (root / 'verified-summary.json').read_bytes() != summary_path.read_bytes():
        raise ValueError('Archived summary differs')
    local = collect(root, directory / 'transition-wording-bank-manifest.json',
                    text_only=True, four_gpu_warm_start=True)
    for key in remote:
        if key not in ('artifacts_omitted_locally', 'all_remote_artifacts_verified_here') and local[key] != remote[key]:
            raise ValueError('Local raw/draw/parallel verification differs: ' + key)
    result = {key: local[key] for key in (
        'result_sha256', 'checkpoint_sha256', 'optimizer_steps', 'exact_draws_replayed',
        'matched_pairs', 'sigma_min', 'sigma_max', 'sigma_above_half',
        'development_all_correct_eos', 'parallel_evidence_sha256', 'artifacts_omitted_locally')}
    result.update(summary_sha256=a.summary_sha256, archive_sha256=a.archive_sha256,
        raw_rows_verified=sum(phase['raw_rows'] for phase in local['phases'].values()),
        baseline_correct_eos=local['phases']['baseline']['correct_eos'],
        trained_correct_eos=local['phases']['trained']['correct_eos'],
        images_passing_all_five=local['phases']['trained']['images_passing_all_five'],
        scope='All downloaded raw text, deterministic draws, original plan and four-rank proofs reverified locally. Large checkpoint/PT payloads remain remote; their local omissions are explicit. Development success is not independent validation.')
(directory / 'four-gpu-warm-endpoint-local-verification.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({key: value for key, value in result.items() if key != 'artifacts_omitted_locally'}))
