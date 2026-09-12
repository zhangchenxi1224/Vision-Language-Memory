"""Seal an existing HF local-dir download without modifying model files."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def inspect_download(model_dir: Path, revision: str) -> dict:
    root = model_dir.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.parts[0] == ".cache" or not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError("Snapshot symlinks are not supported")
        metadata = root / ".cache/huggingface/download" / (rel.as_posix()+".metadata")
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or lines[0] != revision:
            raise ValueError(f"Download revision mismatch: {rel}")
        size = path.stat().st_size
        sha256 = hashlib.sha256()
        blob_sha1 = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b""):
                sha256.update(block)
                blob_sha1.update(block)
        etag = lines[1].strip('"')
        actual_etag = sha256.hexdigest() if len(etag)==64 else blob_sha1.hexdigest()
        if etag != actual_etag:
            raise ValueError(f"Downloaded payload differs from HF content hash: {rel}")
        files.append({"path":rel.as_posix(), "bytes":size, "sha256":sha256.hexdigest(), "hf_etag":etag})
    if not files or not (root/"model_index.json").is_file():
        raise ValueError("Missing pipeline snapshot")
    payload={"revision":revision,"files":files}
    return {"schema":"hf-local-download-seal/v1", "model_dir":str(root), **payload,
            "payload_sha256":hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()}


def verify_download_seal(path: Path, model_dir: Path) -> dict:
    sealed=json.loads(path.read_text(encoding="utf-8"))
    if sealed["model_dir"] != str(model_dir.resolve()):
        raise ValueError("Snapshot seal belongs to another directory")
    actual=inspect_download(model_dir,sealed["revision"])
    if actual != sealed:
        raise ValueError("Model snapshot changed after sealing")
    return actual
