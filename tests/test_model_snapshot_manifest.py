from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "inspire"))

from model_snapshot_manifest import (  # noqa: E402
    SNAPSHOT_MANIFEST_NAME,
    create_snapshot_manifest,
    verify_snapshot_binding,
    verify_snapshot_manifest,
)


REVISION = "a" * 40


def model_fixture(tmp_path: Path) -> Path:
    model = tmp_path / "Qwen3-VL-4B-Instruct"
    model.mkdir()
    (model / ".locked_revision").write_text(REVISION + "\n", encoding="utf-8")
    (model / ".snapshot_complete").write_text(REVISION + "\n", encoding="utf-8")
    (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (model / "tokenizer.json").write_text('{"tokens":[]}\n', encoding="utf-8")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"fixture-weights")
    cache = model / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "download-metadata.json").write_text("transient\n", encoding="utf-8")
    return model


def test_snapshot_manifest_covers_model_files_and_verifies_binding(tmp_path: Path) -> None:
    model = model_fixture(tmp_path)
    create_snapshot_manifest(model_dir=model, repo_id="fixture/qwen", revision=REVISION)
    binding = verify_snapshot_manifest(
        manifest_path=model / SNAPSHOT_MANIFEST_NAME,
        model_dir=model,
        expected_repo_id="fixture/qwen",
        expected_revision=REVISION,
    )
    assert binding["passed"] is True
    assert binding["file_count"] == 3
    assert verify_snapshot_binding(binding) == binding


def test_snapshot_manifest_rejects_tamper_and_unexpected_file(tmp_path: Path) -> None:
    model = model_fixture(tmp_path)
    create_snapshot_manifest(model_dir=model, repo_id="fixture/qwen", revision=REVISION)
    manifest = model / SNAPSHOT_MANIFEST_NAME
    (model / "config.json").write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 lock"):
        verify_snapshot_manifest(
            manifest_path=manifest,
            model_dir=model,
            expected_repo_id="fixture/qwen",
            expected_revision=REVISION,
        )

    (model / "config.json").write_text('{"model_type":"fixture"}\n', encoding="utf-8")
    (model / "unexpected.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        verify_snapshot_manifest(
            manifest_path=manifest,
            model_dir=model,
            expected_repo_id="fixture/qwen",
            expected_revision=REVISION,
        )


def test_snapshot_manifest_rejects_revision_marker_drift(tmp_path: Path) -> None:
    model = model_fixture(tmp_path)
    create_snapshot_manifest(model_dir=model, repo_id="fixture/qwen", revision=REVISION)
    (model / ".locked_revision").write_text("b" * 40 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="locked_revision"):
        verify_snapshot_manifest(
            manifest_path=model / SNAPSHOT_MANIFEST_NAME,
            model_dir=model,
            expected_repo_id="fixture/qwen",
            expected_revision=REVISION,
        )


def test_snapshot_manifest_rejects_model_directory_symlink(tmp_path: Path) -> None:
    model = model_fixture(tmp_path)
    entry = tmp_path / "model-entry"
    try:
        entry.symlink_to(model, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(ValueError, match="itself must not be a symlink"):
        create_snapshot_manifest(model_dir=entry, repo_id="fixture/qwen", revision=REVISION)
