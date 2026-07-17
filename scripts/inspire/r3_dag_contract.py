from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


PLAN_PROTOCOL = "r3-inspire-immutable-dag.v1"
STAGE_SPEC_PROTOCOL = "r3-inspire-bound-stage.v1"
STAGE_EVIDENCE_PROTOCOL = "r3-inspire-stage-evidence.v1"
MICRO_COMMAND_PROTOCOL = "r3-inspire-micro-command.v2"

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload.encode("utf-8"))
    temporary.replace(path)
    digest = sha256_bytes(payload.encode("utf-8"))
    sidecar = path.with_suffix(path.suffix + ".sha256")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    temporary_sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    temporary_sidecar.replace(sidecar)
    return digest


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def verify_sha_sidecar(path: Path, *, expected_sha256: str | None = None) -> str:
    if not path.is_file():
        raise ValueError(f"Required artifact is missing: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"Required SHA256 sidecar is missing: {sidecar}")
    declared = sidecar.read_text(encoding="utf-8").split()[0]
    if SHA256_PATTERN.fullmatch(declared) is None:
        raise ValueError(f"Malformed SHA256 sidecar: {sidecar}")
    actual = sha256_file(path)
    if actual != declared:
        raise ValueError(f"SHA256 sidecar mismatch for {path}: declared {declared}, actual {actual}")
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"Bound SHA256 mismatch for {path}: expected {expected_sha256}, actual {actual}")
    return actual


def git(repo: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def verify_clean_commit(repo: Path, expected_commit: str) -> None:
    if COMMIT_PATTERN.fullmatch(expected_commit) is None:
        raise ValueError("expected_commit must be a lowercase full Git commit")
    actual = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if actual != expected_commit:
        raise ValueError(f"Commit mismatch: expected {expected_commit}, got {actual}")
    if status != "":
        raise ValueError("Formal Inspire DAG materialization requires a clean checkout")


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def require_absolute(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path.resolve()


def require_absolute_executable(path: Path, label: str) -> Path:
    """Normalize an executable path without dereferencing a venv interpreter symlink."""

    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    value = Path(os.path.abspath(expanded))
    if not value.is_file():
        raise ValueError(f"{label} is missing: {value}")
    return value


def require_file_sha(path: Path, expected_sha256: str, label: str) -> dict[str, str]:
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} SHA256 must be a lowercase 64-character digest")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}")
    return {"path": str(path.resolve()), "sha256": actual}


def require_json_values(value: Mapping[str, Any], required: Mapping[str, Any], label: str) -> None:
    for key, expected in required.items():
        actual = value.get(key)
        if actual != expected:
            raise ValueError(f"{label} requires {key}={expected!r}, got {actual!r}")


def verify_bound_artifact(binding: Mapping[str, Any]) -> dict[str, Any] | None:
    label = str(binding.get("label", "artifact"))
    path = Path(str(binding["path"]))
    expected_sha256 = str(binding["sha256"])
    if SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError(f"{label} has a malformed bound SHA256")
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected_sha256}, got {actual}")
    required_values = binding.get("required_values", {})
    if not isinstance(required_values, Mapping):
        raise ValueError(f"{label} required_values must be an object")
    if not required_values:
        return None
    value = load_json_object(path)
    require_json_values(value, required_values, label)
    return value
