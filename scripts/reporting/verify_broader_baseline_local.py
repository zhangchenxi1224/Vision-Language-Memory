"""Recount all3020 real baseline rows from the complete immutable text archive."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts.reporting.collect_broader_endpoint import phase_summary, BANK_SHA, PLAN_SHA, COMMIT


def main():
    results = ROOT / 'reports/official-alignment-results-20260913'
    archive_path = results / 'broader151-baseline-text.tgz'
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != 'a7cae94094b70e7ef3dd0e9f131e1cf0b3a77235babed187d597d923f8b66aa2':
        raise ValueError('Baseline archive changed')
    with tarfile.open(archive_path) as archive:
        files = {member.name: archive.extractfile(member).read() for member in archive.getmembers() if member.isfile()}
    complete = json.loads(files['train/baseline/complete.json'])
    for name in ('generations.jsonl', 'summary.json'):
        if hashlib.sha256(files['train/baseline/' + name]).hexdigest() != complete['artifact_hashes'][name]:
            raise ValueError('Baseline text artifact changed')
    if hashlib.sha256(files['preregistered-experiment.json']).hexdigest() != PLAN_SHA:
        raise ValueError('Original preregistration bytes changed')
    bank_path = results / 'broader151-bank-manifest.json'
    if hashlib.sha256(bank_path.read_bytes()).hexdigest() != BANK_SHA:
        raise ValueError('Real bank changed')
    identity = json.loads(files['train/identity.json'])
    if identity['git_commit'] != COMMIT or identity['bank_manifest_sha256'] != BANK_SHA:
        raise ValueError('Baseline does not belong to the fixed151 run')
    rows = [json.loads(line) for line in files['train/baseline/generations.jsonl'].splitlines()]
    summary, _ = phase_summary(rows, json.loads(bank_path.read_bytes()), 'baseline')
    summary.update(archive_sha256=digest, actual_raw_rows_recounted=True,
        scope='Baseline before optimization;302 tensor payloads remain remote and were not rechecked locally.')
    (results / 'broader151-baseline-local-verification.json').write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in summary.items() if key != 'cells'}))


if __name__ == '__main__':
    main()
