import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.experiments import prefeval_aug4_assets as assets


def sealed(tmp_path, monkeypatch):
    roots, records = {}, []
    for name in ('base', 'reader'):
        root = tmp_path / name
        root.mkdir()
        path = root / 'weights.bin'
        path.write_bytes((name + '-weights').encode())
        stat = path.stat()
        roots[name] = str(root)
        records.append({'path': str(path), 'sha256': assets.file_sha256(path),
                        'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    seal = tmp_path / 'seal.json'
    seal.write_text(json.dumps({'schema': 'model-assets/v1', 'roots': roots, 'files': records}))
    digest = assets.file_sha256(seal)
    monkeypatch.setenv('AUG4_ASSET_SEAL', str(seal))
    monkeypatch.setenv('AUG4_ASSET_SEAL_SHA256', digest)
    return seal, roots, records, digest


def test_unconfigured_cpu_test_can_run_without_assets(monkeypatch):
    monkeypatch.delenv('AUG4_ASSET_SEAL', raising=False)
    monkeypatch.delenv('AUG4_ASSET_SEAL_SHA256', raising=False)
    assert assets.verify_asset_seal() is None


def test_unchanged_inventory_uses_metadata_without_rehashing_every_weight(tmp_path, monkeypatch):
    _, roots, _, digest = sealed(tmp_path, monkeypatch)
    def forbidden(path):
        raise AssertionError('Unchanged files should not be rehashed by every worker')
    monkeypatch.setattr(assets, 'file_sha256', forbidden)
    assert assets.verify_asset_seal(SimpleNamespace(base=roots['base'], reader=roots['reader'])) == digest


def test_changed_metadata_rehashes_and_accepts_identical_content(tmp_path, monkeypatch):
    _, _, records, digest = sealed(tmp_path, monkeypatch)
    path = Path(records[0]['path'])
    os.utime(path, ns=(path.stat().st_atime_ns, records[0]['mtime_ns'] + 1_000_000_000))
    seen, original = [], assets.file_sha256
    def tracked(path):
        seen.append(Path(path))
        return original(path)
    monkeypatch.setattr(assets, 'file_sha256', tracked)
    assert assets.verify_asset_seal() == digest
    assert seen == [path]


def test_changed_model_content_is_rejected(tmp_path, monkeypatch):
    _, _, records, _ = sealed(tmp_path, monkeypatch)
    Path(records[1]['path']).write_bytes(b'replaced weights')
    with pytest.raises(ValueError, match='asset content changed'):
        assets.verify_asset_seal()


def test_changed_inventory_is_rejected_before_loading_model(tmp_path, monkeypatch):
    seal, _, _, _ = sealed(tmp_path, monkeypatch)
    seal.write_bytes(seal.read_bytes() + b' ')
    with pytest.raises(ValueError, match='seal content changed'):
        assets.verify_asset_seal()


def test_arguments_must_match_sealed_roots(tmp_path, monkeypatch):
    _, _, _, _ = sealed(tmp_path, monkeypatch)
    other = tmp_path / 'wrong-base'
    other.mkdir()
    with pytest.raises(ValueError, match='differs from sealed root'):
        assets.verify_asset_seal(base=other)


def test_inventory_cannot_escape_model_roots(tmp_path, monkeypatch):
    seal, _, _, _ = sealed(tmp_path, monkeypatch)
    artifact = json.loads(seal.read_text())
    outside = tmp_path / 'unrelated.bin'
    outside.write_bytes(b'secret')
    artifact['files'][0]['path'] = str(outside)
    seal.write_text(json.dumps(artifact))
    monkeypatch.setenv('AUG4_ASSET_SEAL_SHA256', assets.file_sha256(seal))
    with pytest.raises(ValueError, match='outside model roots'):
        assets.verify_asset_seal()
