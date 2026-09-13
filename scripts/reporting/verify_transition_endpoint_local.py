"""Reverify downloaded endpoint text and all deterministic draws on local CPU."""
import json
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_transition_endpoint import collect, sha

directory = ROOT / 'reports/official-alignment-results-20260913'
expected = {
    'transition-wording-endpoint-summary.json': 'ed3355a203129562aced3fcfde5eb8ea170569cf376bc67e09dd09a1f1c1e27c',
    'transition-wording-endpoint-evidence.tgz': '18d4e8f71b123853dffb829ef2daeadeadf17c9915eef1d0a6517960b62b1ee5',
    'transition-wording-state-geometry.json': '40cbc010b5f51bfa8c587f667b72ba32a645c5a075c5686c2c7f0d98b4c129f5',
}
for name, digest in expected.items():
    if sha(directory / name) != digest:
        raise ValueError('Downloaded bytes differ from remote seal: ' + name)
remote = json.loads((directory / 'transition-wording-endpoint-summary.json').read_bytes())
geometry = json.loads((directory / 'transition-wording-state-geometry.json').read_bytes())
with tempfile.TemporaryDirectory(prefix='dreamlite-endpoint-') as temporary:
    root = Path(temporary)
    with tarfile.open(directory / 'transition-wording-endpoint-evidence.tgz') as stream:
        stream.extractall(root, filter='data')
    if (root / 'verified-summary.json').read_bytes() != (directory / 'transition-wording-endpoint-summary.json').read_bytes():
        raise ValueError('Archived summary differs')
    local = collect(root, directory / 'transition-wording-bank-manifest.json', text_only=True)
    for key in remote:
        if key not in ('artifacts_omitted_locally', 'all_remote_artifacts_verified_here') and local[key] != remote[key]:
            raise ValueError('Local raw/draw verification differs: ' + key)
    if (geometry['result_sha256'] != local['result_sha256']
            or geometry['checkpoint_sha256'] != local['checkpoint_sha256']
            or geometry['phase_complete_sha256'] != local['phases']['trained']['complete_sha256']
            or geometry['summary']['correct_eos'] != local['phases']['trained']['correct_eos']):
        raise ValueError('Geometry evidence belongs to another endpoint')
    result = {'download_sha256': expected, 'result_sha256': local['result_sha256'],
        'checkpoint_sha256': local['checkpoint_sha256'], 'raw_rows_verified': 2700,
        'optimizer_steps': local['optimizer_steps'], 'exact_draws_replayed': local['exact_draws_replayed'],
        'matched_rows_per_phase': 900, 'baseline_correct_eos': local['phases']['baseline']['correct_eos'],
        'trained_correct_eos': local['phases']['trained']['correct_eos'],
        'images_passing_all_five': local['phases']['trained']['images_passing_all_five'],
        'development_all_correct_eos': local['development_all_correct_eos'],
        'artifacts_verified_remotely_only': local['artifacts_omitted_locally'],
        'scope': 'Local verification of all downloaded text and deterministic training draws. Checkpoint/PT bytes and numerical latent distances were verified on the remote host; those large tensors were not downloaded.'}
(directory / 'transition-wording-endpoint-local-verification.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'artifacts_verified_remotely_only'}))
