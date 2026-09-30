"""Verify a downloaded recovery evidence archive without extracting or modifying it."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(8 << 20), b''):
        h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    archive = args.receipt.parent/Path(receipt['archive']['path']).name
    with archive.open('rb') as stream:
        actual = digest(stream)
    assert actual == receipt['archive']['sha256']
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == receipt['archive']['files']
        names = [m.name for m in members]
        assert len(names) == len(set(names)) and all(m.isfile() for m in members)
        manifests = [m for m in members if m.name.endswith('/interruption-manifest.json')]
        destination = Path(receipt['destination'])
        relative = destination.relative_to('/inspire/ssd/project/exploration-topic/czxs26210936/runs/prefeval-multitarget-20260927')
        current = tar.getmember(str(relative/'interruption-manifest.json'))
        expected = json.load(tar.extractfile(current))
        assert set(names) == set(expected) | {current.name}
        for name, sha in expected.items():
            assert digest(tar.extractfile(name)) == sha, name
    result = {'archive': str(archive), 'sha256': actual, 'verified_files': len(members),
              'verified_manifest_entries': len(expected), 'complete_chains': receipt['complete_chains'],
              'preserved_unique_readbacks': sum(receipt['readbacks'].values()),
              'note': 'All archived evidence bytes verified locally. Actual PNG hashes were rechecked on the shared project before byte-for-byte copying; PNG bytes remain on that project.'}
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
