"""Package the audited partial B730 snapshot, preserving original file bytes."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/b730-latest-eval-20261013'
SNAPSHOT = REPORT / 'evidence/partial-summaries-20261014-021530'
TAG = 'b730-latest-eval-partial-20261014'
ASSET = 'b730-latest-eval-partial-scientific-evidence-20261014.tar.gz'
RELEASE = 'https://github.com/zhangchenxi1224/Vision-Language-Memory/releases/tag/' + TAG


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/b730-publication-20261014')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = SNAPSHOT / 'LOCAL-ARCHIVE-RECEIPT.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    assert receipt['status'] == 'verified' and receipt['file_count'] == 509
    assert not receipt['full_experiment_complete']
    members = {r['path']: {'sha256': r['sha256'], 'bytes': r['bytes']} for r in receipt['files']}
    members[receipt_path.name] = {'sha256': sha(receipt_path.read_bytes()), 'bytes': receipt_path.stat().st_size}
    secrets = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')
    asset_path = args.output_dir / ASSET
    with asset_path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode='w|') as target:
                for name, expected in sorted(members.items()):
                    path = SNAPSHOT / name
                    assert path.resolve().is_relative_to(SNAPSHOT.resolve())
                    data = path.read_bytes()
                    if len(data) != expected['bytes'] or sha(data) != expected['sha256']:
                        raise ValueError('Source bytes changed: ' + name)
                    if secrets.search(data):
                        raise ValueError('Potential credential in: ' + name)
                    info = tarfile.TarInfo(SNAPSHOT.name + '/' + name)
                    info.size, info.mode, info.mtime = len(data), 0o644, 0
                    target.addfile(info, io.BytesIO(data))
    # Re-open the produced archive and verify every member before publishing.
    with tarfile.open(asset_path, 'r:gz') as package:
        actual = {}
        for member in package:
            assert member.isfile()
            name = member.name.removeprefix(SNAPSHOT.name + '/')
            data = package.extractfile(member).read()
            actual[name] = {'bytes': len(data), 'sha256': sha(data)}
        assert actual == members
    manifest = {'schema': 'b730-partial-publication/v1', 'status': 'verified_stage_snapshot',
                'snapshot_time_cst': '2026-10-14 02:15:30+08:00', 'full_experiment_complete': False,
                'verified_main_summaries': 12, 'required_main_summaries': 17,
                'verified_new_gpu_readback_rows': 52224, 'verified_logical_row_references': 78336,
                'release_url': RELEASE, 'asset': ASSET, 'asset_bytes': asset_path.stat().st_size,
                'asset_sha256': sha(asset_path.read_bytes()), 'archive_root': SNAPSHOT.name,
                'member_count': len(members), 'uncompressed_member_bytes': sum(x['bytes'] for x in members.values()),
                'members': members,
                'scope': 'Completed source readbacks, identity/completion receipts, data, gray controls, frozen parser/evaluator/summary code, checkpoint and PNG path/hash provenance, published comparisons.',
                'not_uploaded': ['model tensor checkpoints', 'generated memory PNG pixels', 'live owner and queue/session records'],
                'scientific_bytes_modified': False,
                'verification_entrypoint': 'scripts/reporting/verify_b730_supplement_publication.py'}
    (REPORT / 'PUBLICATION-MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    (args.output_dir / 'SHA256SUMS').write_text(manifest['asset_sha256'] + '  ' + ASSET + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('asset', 'asset_bytes', 'asset_sha256', 'member_count')}))


if __name__ == '__main__':
    main()
