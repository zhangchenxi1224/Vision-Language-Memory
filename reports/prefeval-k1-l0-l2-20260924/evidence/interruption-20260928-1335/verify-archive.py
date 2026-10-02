import hashlib, json, tarfile
from pathlib import Path

root = Path(__file__).resolve().parent
archive = root / 'raw.tar.gz'
expected = 'c513f49f02d00aaeadc960cf8593a1a56795c1c4d2a9b4b0bfe20578e01235c9'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
assert archive.stat().st_size == 14638334
with tarfile.open(archive) as tar:
    inventory = json.load(tar.extractfile('snapshot-files.json'))
    for name, info in inventory.items():
        data = tar.extractfile(name).read()
        assert len(data) == info['bytes'], name
        assert hashlib.sha256(data).hexdigest() == info['sha256'], name
    summary = json.load(tar.extractfile('summary.json'))
    assert summary['optimization_rows'] == 17520
    assert set(summary['locks'].values()) == {'FREE'}
    assert summary['arms']['C']['complete'] == 391
    assert summary['arms']['R']['complete'] == 278
    assert not summary['round3_bank_exists']
    rows = [json.loads(x) for x in tar.extractfile('snapshot/retain730-R/train/optimization.jsonl').read().splitlines()]
    assert [r['step'] for r in rows] == list(range(1, 17521))
    for arm in ['C', 'R']:
        for chain in summary['arms'][arm]['partial']:
            for item in chain:
                data = tar.extractfile('snapshot/' + item['path']).read()
                assert hashlib.sha256(data).hexdigest() == item['sha256']
print(json.dumps({'verified_files': len(inventory), 'optimization_steps': len(rows), 'partial_pngs': 9, 'archive_sha256': expected}))
