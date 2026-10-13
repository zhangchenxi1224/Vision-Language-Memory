"""Verify a downloaded scientific evidence asset and rescore all 556,800 rows.

Dependencies: numpy, beautifulsoup4. No torch/CUDA or model download needed.
Execute only the frozen code from a release whose SHA matches the Git manifest.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/prefeval-aug4x3-init-ablation-exposure512-20261009'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def same(actual, expected, location='root'):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), location + ': keys'
        for key in expected:
            same(actual[key], expected[key], location + '/' + key)
    elif isinstance(expected, list):
        assert len(actual) == len(expected), location + ': length'
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, location + '/' + str(i))
    elif isinstance(expected, float):
        assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), location + ': float'
    else:
        assert actual == expected, location + ': value'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--asset', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True, help='Extraction directory (dedicated, reusable)')
    ap.add_argument('--receipt', type=Path, default=REPORT / 'OFFLINE-VERIFICATION-20261013.json')
    args = ap.parse_args()
    publication = json.loads((REPORT / 'PUBLICATION-MANIFEST.json').read_text(encoding='utf-8'))
    assert sha(args.asset) == publication['asset_sha256'], 'Asset checksum mismatch'
    args.work.mkdir(parents=True, exist_ok=True)
    extracted = set()
    with tarfile.open(args.asset, 'r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or ':' in member.name:
                raise ValueError('Unsafe archive member: ' + member.name)
            assert member.name not in extracted, 'Duplicate member'
            extracted.add(member.name)
            path = args.work / member.name
            # Do not follow preexisting links in a reused verification directory.
            assert not path.is_symlink() and all(not p.is_symlink() for p in path.parents), 'Symlink in extraction path'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    file_manifest = args.work / 'PUBLIC-SHA256-MANIFEST.json'
    assert sha(file_manifest) == publication['member_manifest_sha256']
    files = json.loads(file_manifest.read_text())
    assert extracted == set(files) | {'PUBLIC-SHA256-MANIFEST.json'}
    for name, expected in files.items():
        path = args.work / name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
    sys.path.insert(0, str(args.work / 'frozen-repo'))
    source = args.work / 'frozen-repo/scripts/reporting/summarize_prefeval_aug4.py'
    spec = importlib.util.spec_from_file_location('frozen_aug4_summary', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summaries = sorted((args.work / 'run/evaluation').glob('*/summary.json'))
    assert len(summaries) == 84
    rows, verifications = 0, []
    for i, summary_path in enumerate(summaries):
        expected = json.loads(summary_path.read_text(encoding='utf-8'))
        actual = module.summarize(args.work / 'frozen-data/runtime_data.json', summary_path.parent)
        same(actual, expected, summary_path.parent.name)
        committed = REPORT / 'evidence/summaries' / summary_path.parent.name / 'summary.json'
        assert sha(committed) == sha(summary_path)
        rows += actual['measurement_count']
        verifications.append({'evaluation': summary_path.parent.name, 'sha256': sha(summary_path),
                              'raw_rows': actual['measurement_count'], 'full_summary_recomputed': True})
        print(f'{i+1}/84 {summary_path.parent.name}: {actual["measurement_count"]} raw rows verified', flush=True)
    assert rows == 556800
    subprocess.run([sys.executable, str(ROOT / 'scripts/reporting/build_prefeval_aug4_publication.py'), '--check'], check=True)
    receipt = {'schema': 'prefeval-offline-publication-verification/v1', 'status': 'passed',
               'asset_sha256': publication['asset_sha256'], 'member_files_verified': len(files),
               'summary_count': len(summaries), 'raw_student_rows_reparsed': rows,
               'official177': 'recomputed from the same official180 raw rows after frozen exclusions',
               'checks': ['all asset member SHA256 and byte counts', 'raw shard SHA and receipt counts',
                          'exact seed/position/suite/preference coverage and no duplicate keys',
                          'original official MCQ parser, option order, query and stored score',
                          'frozen mismatch donor mapping', 'all controls, suites and paired bootstrap intervals',
                          'all current and old B730 derived tables and source digests'],
               'not_checked': ['GPU inference rerun', 'training rerun', 'remote model or PNG bytes newly fetched',
                               'old B730 raw outputs reparsed; this publication verifies its archived summaries'],
               'summarizer_sha256': sha(source), 'evaluations': verifications}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('PASS: 84 exact summaries / 556800 raw rows; receipt ' + str(args.receipt))


if __name__ == '__main__':
    main()
