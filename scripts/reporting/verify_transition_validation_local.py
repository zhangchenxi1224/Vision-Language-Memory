"""Reverify full downloaded validation text/PNGs against its original plan bytes."""
import argparse
import json
from pathlib import Path
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_transition_validation import collect
from scripts.reporting.collect_transition_endpoint import sha

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--prefix', choices=('transition-confirmation', 'transition-chains'), required=True)
p.add_argument('--summary-sha256', required=True)
p.add_argument('--archive-sha256', required=True)
a = p.parse_args()
directory = ROOT / 'reports/official-alignment-results-20260913'
summary_path = directory / (a.prefix + '-summary.json')
archive_path = directory / (a.prefix + '-evidence.tgz')
parent_archive = directory / 'transition-wording-endpoint-evidence.tgz'
if sha(summary_path) != a.summary_sha256 or sha(archive_path) != a.archive_sha256:
    raise ValueError('Downloaded validation differs from its remote digest')
parent_sha = '18d4e8f71b123853dffb829ef2daeadeadf17c9915eef1d0a6517960b62b1ee5'
if sha(parent_archive) != parent_sha:
    raise ValueError('Downloaded parent evidence changed')
remote = json.loads(summary_path.read_bytes())
with tempfile.TemporaryDirectory(prefix='dreamlite-validation-') as temporary:
    root = Path(temporary)
    parent, run = root / 'parent', root / 'validation'
    for path, destination in ((parent_archive, parent), (archive_path, run)):
        with tarfile.open(path) as stream:
            stream.extractall(destination, filter='data')
    if (run / 'verified-summary.json').read_bytes() != summary_path.read_bytes():
        raise ValueError('Archived validation summary differs')
    local = collect(run, parent, directory / 'transition-wording-bank-manifest.json',
                    run / 'preregistered-plan.json', text_only=True)
    for key in remote:
        if key not in ('all_files_verified_here', 'verified_files', 'artifacts_omitted_locally') and local[key] != remote[key]:
            raise ValueError('Local validation differs: ' + key)
    result = {key: value for key, value in local.items() if key not in ('identity', 'cells', 'groups', 'verified_files')}
    result.update(summary_sha256=a.summary_sha256, archive_sha256=a.archive_sha256,
                  parent_archive_sha256=parent_sha, png_files_verified=len(list(run.glob('*.png'))),
                  scope='All downloaded PNGs, raw answers, coverage, original plan bytes and parent text reverified locally; missing PT/checkpoint files are explicit, with their full verification retained on the remote host.')
(directory / (a.prefix + '-local-verification.json')).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'artifacts_omitted_locally'}))
