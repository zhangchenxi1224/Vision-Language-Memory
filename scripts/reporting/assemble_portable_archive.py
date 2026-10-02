"""Reconstruct an entire evidence archive from independently sealed file chunks."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def assemble(manifest_path, manifest_sha256, output):
    if sha(manifest_path) != manifest_sha256:
        raise ValueError('Manifest differs from the independently observed digest')
    manifest = json.loads(manifest_path.read_bytes())
    root = manifest_path.parent.resolve()
    seen = set()
    total = 0
    paths = []
    for chunk in manifest['chunks']:
        path = (root / chunk['file']).resolve()
        if path.parent != root or path in seen or path.suffix != '.part':
            raise ValueError('Invalid or duplicated chunk path')
        if path.stat().st_size != chunk['bytes'] or sha(path) != chunk['sha256']:
            raise ValueError('Changed or incomplete evidence chunk')
        seen.add(path)
        paths.append(path)
        total += chunk['bytes']
    if not paths or total != manifest['bytes']:
        raise ValueError('Incomplete total chunk coverage')
    if output.exists():
        if output.stat().st_size != total or sha(output) != manifest['sha256']:
            raise ValueError('Output exists with different or incomplete bytes')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + '.assembling')
        digest = hashlib.sha256()
        with temporary.open('xb') as stream:
            for path in paths:
                with path.open('rb') as source:
                    for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                        digest.update(block)
                        stream.write(block)
        if temporary.stat().st_size != total or digest.hexdigest() != manifest['sha256']:
            raise ValueError('Assembled archive differs from the original remote archive')
        temporary.rename(output)
    return {'file': str(output), 'bytes': total, 'sha256': manifest['sha256'],
        'chunks_verified': len(paths), 'manifest_sha256': manifest_sha256,
        'scope': 'All archive bytes reconstructed; no tar member removed or transformed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.manifest, args.manifest_sha256, args.output)))
