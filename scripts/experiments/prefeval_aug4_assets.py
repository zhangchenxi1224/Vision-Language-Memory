"""Verify the immutable, pre-hashed model asset inventory at worker startup."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _under(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def verify_asset_seal(args=None, *, base=None, reader=None):
    """Return the seal's exact byte hash, or None when no seal is configured.

    Deployment performs a complete SHA-256 verification on both hosts first.
    Workers compare every asset's size/mtime against that frozen inventory;
    metadata changes trigger full content hashing before any model is loaded.
    AUG4_ASSET_SEAL_SHA256 optionally pins the inventory itself. The production
    controller must configure both environment variables; omission is useful
    only for isolated CPU unit tests and explicitly unsealed technical work.
    """
    source = os.environ.get('AUG4_ASSET_SEAL')
    if not source:
        if os.environ.get('AUG4_ASSET_SEAL_SHA256'):
            raise ValueError('AUG4_ASSET_SEAL_SHA256 is set without AUG4_ASSET_SEAL')
        return None
    path = Path(source).resolve(strict=True)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = os.environ.get('AUG4_ASSET_SEAL_SHA256')
    if expected and digest != expected:
        raise ValueError('Model asset seal content changed')
    seal = json.loads(raw)
    if seal.get('schema') != 'model-assets/v1':
        raise ValueError('Unsupported model asset seal schema')
    declared_roots = seal.get('roots', seal.get('model_roots'))
    if not isinstance(declared_roots, dict) or not {'base', 'reader'} <= set(declared_roots):
        raise ValueError('Model asset seal requires base and reader roots')
    roots = {name: Path(declared_roots[name]).resolve(strict=True) for name in ('base', 'reader')}
    if any(not root.is_dir() for root in roots.values()) or roots['base'] == roots['reader']:
        raise ValueError('Model asset roots must be distinct directories')
    requested = {'base': base if base is not None else getattr(args, 'base', None),
                 'reader': reader if reader is not None else getattr(args, 'reader', None)}
    for name, value in requested.items():
        if value is not None and Path(value).resolve(strict=True) != roots[name]:
            raise ValueError(f'{name} model argument differs from sealed root')
    files = seal.get('files')
    if not isinstance(files, list) or not files:
        raise ValueError('Model asset seal has no file inventory')
    seen, represented = set(), set()
    for record in files:
        declared_path = Path(record['path'])
        if not declared_path.is_absolute():
            raise ValueError('Sealed asset paths must be absolute')
        asset = declared_path.resolve(strict=True)
        owners = [name for name, root in roots.items() if _under(asset, root)]
        if len(owners) != 1 or not asset.is_file() or asset in seen:
            raise ValueError(f'Asset is outside model roots, duplicated, or not a file: {asset}')
        seen.add(asset)
        represented.add(owners[0])
        if not re.fullmatch(r'[0-9a-f]{64}', record['sha256']):
            raise ValueError('Invalid asset SHA-256')
        recorded_size, recorded_mtime = int(record['size']), int(record['mtime_ns'])
        if recorded_size < 0 or recorded_mtime < 0:
            raise ValueError('Invalid asset metadata')
        stat = asset.stat()
        if stat.st_size != recorded_size or stat.st_mtime_ns != recorded_mtime:
            actual = file_sha256(asset)
            after = asset.stat()
            if actual != record['sha256']:
                raise ValueError(f'Model asset content changed: {asset}')
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f'Model asset mutated during verification: {asset}')
    if represented != {'base', 'reader'}:
        raise ValueError('Seal must inventory files for both model roots')
    return digest
