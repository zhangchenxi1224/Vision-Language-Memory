from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_MANIFEST_PROTOCOL = "vlm.hf-snapshot-sha256.v1"
SNAPSHOT_MANIFEST_NAME = ".snapshot_manifest.json"
EXCLUDED_TOP_LEVEL = {".cache"}
EXCLUDED_NAMES = {
    ".locked_revision",
    ".metadata_complete",
    ".snapshot_complete",
    SNAPSHOT_MANIFEST_NAME,
    f"{SNAPSHOT_MANIFEST_NAME}.sha256",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _snapshot_files(model_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in model_dir.rglob("*"):
        relative = path.relative_to(model_dir)
        if relative.parts and relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        if path.is_symlink():
            raise ValueError(f"Model snapshot contains an unsupported symlink: {relative.as_posix()}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(model_dir).as_posix())


def build_snapshot_manifest(*, model_dir: Path, repo_id: str, revision: str) -> dict[str, Any]:
    model_dir = model_dir.expanduser()
    if model_dir.is_symlink():
        raise ValueError(f"Model snapshot directory itself must not be a symlink: {model_dir}")
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise ValueError(f"Model snapshot directory is missing: {model_dir}")
    for marker_name in (".locked_revision", ".snapshot_complete"):
        marker = model_dir / marker_name
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != revision:
            raise ValueError(f"Model snapshot {marker_name} does not bind revision {revision}")
    records = [
        {
            "path": path.relative_to(model_dir).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _snapshot_files(model_dir)
    ]
    if not records:
        raise ValueError(f"Model snapshot has no immutable files: {model_dir}")
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "model_dir_name": model_dir.name,
        "files": records,
    }
    return {
        "schema_version": 1,
        "protocol": SNAPSHOT_MANIFEST_PROTOCOL,
        **payload,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "snapshot_payload_sha256": canonical_sha256(payload),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temporary = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    sidecar_temporary.write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    sidecar_temporary.replace(sidecar)
    return digest


def create_snapshot_manifest(
    *,
    model_dir: Path,
    repo_id: str,
    revision: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    model_dir = model_dir.expanduser()
    if model_dir.is_symlink():
        raise ValueError(f"Model snapshot directory itself must not be a symlink: {model_dir}")
    model_dir = model_dir.resolve()
    manifest_path = model_dir / SNAPSHOT_MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise ValueError(f"Snapshot manifest already exists: {manifest_path}")
    manifest = build_snapshot_manifest(model_dir=model_dir, repo_id=repo_id, revision=revision)
    manifest_sha256 = _atomic_json(manifest_path, manifest)
    return {
        **verify_snapshot_manifest(
            manifest_path=manifest_path,
            model_dir=model_dir,
            expected_repo_id=repo_id,
            expected_revision=revision,
        ),
        "created": True,
        "manifest_sha256": manifest_sha256,
    }


def verify_snapshot_manifest(
    *,
    manifest_path: Path,
    model_dir: Path,
    expected_repo_id: str,
    expected_revision: str,
) -> dict[str, Any]:
    model_dir = model_dir.expanduser()
    if model_dir.is_symlink():
        raise ValueError(f"Model snapshot directory itself must not be a symlink: {model_dir}")
    model_dir = model_dir.resolve()
    manifest_path = manifest_path.expanduser().resolve()
    for marker_name in (".locked_revision", ".snapshot_complete"):
        marker = model_dir / marker_name
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected_revision:
            raise ValueError(f"Model snapshot {marker_name} does not bind revision {expected_revision}")
    if manifest_path != model_dir / SNAPSHOT_MANIFEST_NAME or not manifest_path.is_file():
        raise ValueError(f"Snapshot manifest is missing from its model directory: {manifest_path}")
    manifest_sha256 = sha256_file(manifest_path)
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != (
        f"{manifest_sha256}  {manifest_path.name}"
    ):
        raise ValueError(f"Snapshot manifest SHA256 sidecar mismatch: {manifest_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Snapshot manifest must be a JSON object")
    expected_keys = {
        "schema_version",
        "protocol",
        "repo_id",
        "revision",
        "model_dir_name",
        "files",
        "file_count",
        "total_bytes",
        "snapshot_payload_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("Snapshot manifest has missing or unexpected fields")
    if (
        value.get("schema_version") != 1
        or value.get("protocol") != SNAPSHOT_MANIFEST_PROTOCOL
        or value.get("repo_id") != expected_repo_id
        or value.get("revision") != expected_revision
        or value.get("model_dir_name") != model_dir.name
    ):
        raise ValueError("Snapshot manifest identity differs from the locked model")
    records = value.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError("Snapshot manifest files must be a non-empty list")
    expected_paths = [path.relative_to(model_dir).as_posix() for path in _snapshot_files(model_dir)]
    observed_paths: list[str] = []
    total_bytes = 0
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
            raise ValueError("Snapshot manifest contains a malformed file record")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Snapshot manifest contains an unsafe relative path")
        path = model_dir / Path(relative)
        try:
            path.resolve().relative_to(model_dir)
        except ValueError as exc:
            raise ValueError("Snapshot manifest file escapes the model directory") from exc
        size = record.get("size")
        digest = record.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
        ):
            raise ValueError(f"Snapshot manifest metadata is malformed for {relative}")
        if not path.is_file() or path.is_symlink() or path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"Snapshot file differs from its SHA256 lock: {relative}")
        observed_paths.append(relative)
        total_bytes += size
    if observed_paths != sorted(observed_paths) or observed_paths != expected_paths:
        raise ValueError("Snapshot manifest does not exactly cover the current model files")
    payload = {
        "repo_id": expected_repo_id,
        "revision": expected_revision,
        "model_dir_name": model_dir.name,
        "files": [dict(record) for record in records],
    }
    if (
        value.get("file_count") != len(records)
        or value.get("total_bytes") != total_bytes
        or value.get("snapshot_payload_sha256") != canonical_sha256(payload)
    ):
        raise ValueError("Snapshot manifest aggregate metadata mismatch")
    return {
        "passed": True,
        "model_dir": str(model_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "repo_id": expected_repo_id,
        "revision": expected_revision,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "snapshot_payload_sha256": value["snapshot_payload_sha256"],
    }


def verify_snapshot_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "passed",
        "model_dir",
        "manifest_path",
        "manifest_sha256",
        "repo_id",
        "revision",
        "file_count",
        "total_bytes",
        "snapshot_payload_sha256",
    }
    if set(binding) != expected_keys or binding.get("passed") is not True:
        raise ValueError("Model snapshot binding has missing or unexpected fields")
    current = verify_snapshot_manifest(
        manifest_path=Path(str(binding["manifest_path"])),
        model_dir=Path(str(binding["model_dir"])),
        expected_repo_id=str(binding["repo_id"]),
        expected_revision=str(binding["revision"]),
    )
    if current != dict(binding):
        raise ValueError("Model snapshot differs from its formal-preflight binding")
    return current


def _model_entries(lock_path: Path, model_root: Path) -> list[tuple[Path, str, str]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return [
        (
            model_root / Path(specification["local_dir"]).name,
            str(specification["repo_id"]),
            str(specification["revision"]),
        )
        for specification in lock["models"].values()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create or verify full SHA256 locks for HF model snapshots")
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    model_root = args.model_root.expanduser().resolve()
    results = []
    for model_dir, repo_id, revision in _model_entries(args.lock.resolve(), model_root):
        if args.action == "create":
            result = create_snapshot_manifest(
                model_dir=model_dir,
                repo_id=repo_id,
                revision=revision,
                overwrite=args.overwrite,
            )
        else:
            result = verify_snapshot_manifest(
                manifest_path=model_dir / SNAPSHOT_MANIFEST_NAME,
                model_dir=model_dir,
                expected_repo_id=repo_id,
                expected_revision=revision,
            )
        results.append(result)
    print(json.dumps({"passed": True, "models": results}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
