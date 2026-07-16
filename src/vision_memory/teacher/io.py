"""Atomic, train-only persistence for full-state teacher caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch import Tensor

from .cache import (
    TRAIN_SPLIT,
    TeacherArtifactRecord,
    TeacherArtifactSpec,
    TeacherCacheManifest,
    TeacherProvider,
    TeacherState,
    TeacherTransitionRecord,
    manifest_json,
    tensor_sha256,
    validate_teacher_sidecar,
)
from .loss import FrozenTeacherLossCalibration
from .state import canonical_json_bytes, require_sha256


TEACHER_TENSOR_SCHEMA = "vision_memory.teacher-tensor.v1"
TEACHER_CALIBRATION_FILE_SCHEMA = "vision_memory.teacher-calibration-file.v1"
MANIFEST_FILENAME = "manifest.json"
SIDECAR_FILENAME = "transitions.jsonl"
CALIBRATION_FILENAME = "calibration.json"
_RESERVED_CACHE_PATHS = frozenset({MANIFEST_FILENAME, SIDECAR_FILENAME, CALIBRATION_FILENAME})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _atomic_bytes(path: Path, payload: bytes) -> None:
    _atomic_write(path, lambda handle: handle.write(payload))


def _verify_expected_file_sha256(path: Path, expected_sha256: str | None) -> str:
    actual = file_sha256(path)
    if expected_sha256 is not None:
        require_sha256(expected_sha256, field="expected_file_sha256")
        if actual != expected_sha256:
            raise ValueError(f"File SHA256 mismatch for {path}: expected {expected_sha256}, got {actual}.")
    return actual


def _validate_tensor_against_spec(tensor: Tensor, specification: TeacherArtifactSpec) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError("Teacher artifact payload must contain a torch.Tensor.")
    materialized = tensor.detach().cpu().contiguous()
    if not materialized.is_floating_point() or not torch.isfinite(materialized).all():
        raise ValueError("Teacher artifact tensor must contain finite floating-point values.")
    if tuple(materialized.shape) != specification.shape:
        raise ValueError(
            f"Teacher artifact shape mismatch: expected {specification.shape}, got {tuple(materialized.shape)}."
        )
    dtype = str(materialized.dtype).removeprefix("torch.")
    if dtype != specification.dtype:
        raise ValueError(f"Teacher artifact dtype mismatch: expected {specification.dtype}, got {dtype}.")
    actual_sha256 = tensor_sha256(materialized)
    if actual_sha256 != specification.sha256:
        raise ValueError(
            f"Teacher artifact tensor SHA256 mismatch: expected {specification.sha256}, got {actual_sha256}."
        )
    return materialized


def save_teacher_tensor(path: Path, tensor: Tensor, *, specification: TeacherArtifactSpec) -> str:
    """Atomically save one train-only tensor, then read it back with full validation."""

    materialized = _validate_tensor_against_spec(tensor, specification)
    payload = {"schema": TEACHER_TENSOR_SCHEMA, "split": TRAIN_SPLIT, "tensor": materialized}
    _atomic_write(path, lambda handle: torch.save(payload, handle))
    load_teacher_tensor(path, specification=specification)
    return file_sha256(path)


def load_teacher_tensor(
    path: Path,
    *,
    specification: TeacherArtifactSpec,
    expected_file_sha256: str | None = None,
) -> Tensor:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Teacher tensor is missing: {source}")
    _verify_expected_file_sha256(source, expected_file_sha256)
    with source.open("rb") as handle:
        payload = torch.load(handle, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "split", "tensor"}:
        raise ValueError("Teacher tensor file fields differ from the locked schema.")
    if payload["schema"] != TEACHER_TENSOR_SCHEMA:
        raise ValueError(f"Unsupported teacher tensor schema: {payload['schema']!r}.")
    if payload["split"] != TRAIN_SPLIT:
        raise ValueError("Teacher tensor files are train-only.")
    return _validate_tensor_against_spec(payload["tensor"], specification)


def save_teacher_manifest(path: Path, manifest: TeacherCacheManifest) -> str:
    if not isinstance(manifest, TeacherCacheManifest):
        raise TypeError("manifest must be TeacherCacheManifest.")
    _atomic_bytes(path, manifest_json(manifest).encode("utf-8"))
    loaded = load_teacher_manifest(path)
    if loaded.to_dict() != manifest.to_dict():
        raise RuntimeError("Atomic teacher manifest round trip changed its payload.")
    return file_sha256(path)


def load_teacher_manifest(path: Path, *, expected_file_sha256: str | None = None) -> TeacherCacheManifest:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Teacher cache manifest is missing: {source}")
    _verify_expected_file_sha256(source, expected_file_sha256)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Teacher cache manifest is invalid: {source}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Teacher cache manifest root must be an object.")
    return TeacherCacheManifest.from_dict(value)


def load_teacher_cache_manifest(
    path: Path, *, expected_file_sha256: str | None = None
) -> TeacherCacheManifest:
    """Trainer-facing name for strict cache-manifest loading."""

    return load_teacher_manifest(path, expected_file_sha256=expected_file_sha256)


def _sidecar_bytes(records: Iterable[TeacherTransitionRecord]) -> bytes:
    return b"".join(canonical_json_bytes(record.to_dict()) + b"\n" for record in records)


def save_teacher_sidecar(
    path: Path,
    records: Iterable[TeacherTransitionRecord | Mapping[str, Any]],
    *,
    manifest: TeacherCacheManifest,
) -> str:
    validated = validate_teacher_sidecar(records, manifest=manifest)
    _atomic_bytes(path, _sidecar_bytes(validated))
    loaded = load_teacher_sidecar(path, manifest=manifest)
    if loaded != validated:
        raise RuntimeError("Atomic teacher sidecar round trip changed its payload.")
    return file_sha256(path)


def load_teacher_sidecar(
    path: Path,
    *,
    manifest: TeacherCacheManifest,
    expected_file_sha256: str | None = None,
) -> tuple[TeacherTransitionRecord, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Teacher transition sidecar is missing: {source}")
    _verify_expected_file_sha256(source, expected_file_sha256)
    parsed: list[TeacherTransitionRecord] = []
    try:
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line:
                raise ValueError(f"Teacher transition sidecar contains a blank line at {line_number}.")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Teacher transition sidecar line {line_number} must be an object.")
            parsed.append(TeacherTransitionRecord.from_dict(value))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Teacher transition sidecar is invalid: {source}") from exc
    return validate_teacher_sidecar(parsed, manifest=manifest)


def load_teacher_transition_sidecar(
    path: Path,
    *,
    manifest: TeacherCacheManifest,
    expected_file_sha256: str | None = None,
) -> tuple[TeacherTransitionRecord, ...]:
    """Trainer-facing name for strict transition-sidecar loading."""

    return load_teacher_sidecar(
        path,
        manifest=manifest,
        expected_file_sha256=expected_file_sha256,
    )


def _calibration_payload(calibration: FrozenTeacherLossCalibration) -> dict[str, Any]:
    return {
        "schema": TEACHER_CALIBRATION_FILE_SCHEMA,
        "split": TRAIN_SPLIT,
        "contract_sha256": calibration.contract_sha256,
        "calibration": calibration.to_dict(),
    }


def save_teacher_calibration(path: Path, calibration: FrozenTeacherLossCalibration) -> str:
    if not isinstance(calibration, FrozenTeacherLossCalibration):
        raise TypeError("calibration must be FrozenTeacherLossCalibration.")
    payload = json.dumps(_calibration_payload(calibration), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _atomic_bytes(path, payload.encode("utf-8"))
    loaded = load_teacher_calibration(path)
    if loaded != calibration:
        raise RuntimeError("Atomic teacher calibration round trip changed its payload.")
    return file_sha256(path)


def load_teacher_calibration(
    path: Path,
    *,
    expected_file_sha256: str | None = None,
    expected_contract_sha256: str | None = None,
) -> FrozenTeacherLossCalibration:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Teacher calibration is missing: {source}")
    _verify_expected_file_sha256(source, expected_file_sha256)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Teacher calibration is invalid: {source}") from exc
    if not isinstance(value, Mapping) or set(value) != {"schema", "split", "contract_sha256", "calibration"}:
        raise ValueError("Teacher calibration file fields differ from the locked schema.")
    if value["schema"] != TEACHER_CALIBRATION_FILE_SCHEMA:
        raise ValueError(f"Unsupported teacher calibration file schema: {value['schema']!r}.")
    if value["split"] != TRAIN_SPLIT:
        raise ValueError("Teacher calibration files are train-only.")
    require_sha256(value["contract_sha256"], field="calibration.contract_sha256")
    calibration_value = value["calibration"]
    if not isinstance(calibration_value, Mapping):
        raise ValueError("Teacher calibration payload must be an object.")
    calibration = FrozenTeacherLossCalibration.from_dict(calibration_value)
    if value["contract_sha256"] != calibration.contract_sha256:
        raise ValueError("Teacher calibration contract SHA256 does not match its payload.")
    if expected_contract_sha256 is not None:
        require_sha256(expected_contract_sha256, field="expected_contract_sha256")
        if calibration.contract_sha256 != expected_contract_sha256:
            raise ValueError("Teacher calibration contract SHA256 differs from the expected lock.")
    return calibration


def _resolved_artifact_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.expanduser().resolve()
    candidate = (resolved_root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:  # pragma: no cover - TeacherArtifactSpec already rejects traversal
        raise ValueError("Teacher artifact path escapes the cache root.") from exc
    return candidate


def _validate_manifest_artifact_paths(manifest: TeacherCacheManifest) -> None:
    paths = [
        specification.relative_path
        for record in manifest.records
        for specification in (record.image, record.latent, record.feature)
    ]
    if len(paths) != len(set(paths)):
        raise ValueError("Teacher cache manifest reuses an artifact path.")
    if any(path in _RESERVED_CACHE_PATHS for path in paths):
        raise ValueError("Teacher artifact path collides with cache metadata.")


def _load_record(root: Path, manifest: TeacherCacheManifest, record: TeacherArtifactRecord) -> TeacherState:
    return TeacherState(
        state_id=record.state_id,
        teacher_key=record.teacher_key,
        semantic_state_sha256=record.semantic_state_sha256,
        teacher_contract_sha256=manifest.teacher_contract_sha256,
        renderer_contract_sha256=manifest.renderer_contract_sha256,
        image=load_teacher_tensor(
            _resolved_artifact_path(root, record.image.relative_path), specification=record.image
        ),
        latent=load_teacher_tensor(
            _resolved_artifact_path(root, record.latent.relative_path), specification=record.latent
        ),
        feature=load_teacher_tensor(
            _resolved_artifact_path(root, record.feature.relative_path), specification=record.feature
        ),
    )


@dataclass(frozen=True)
class TeacherCacheFileHashes:
    manifest: str
    sidecar: str
    calibration: str
    artifacts: Mapping[str, str]

    def __post_init__(self) -> None:
        for field in ("manifest", "sidecar", "calibration"):
            require_sha256(getattr(self, field), field=f"cache_files.{field}")
        for path, digest in self.artifacts.items():
            if not isinstance(path, str) or not path:
                raise ValueError("Artifact file-hash keys must be non-empty paths.")
            require_sha256(digest, field=f"cache_files.artifacts[{path!r}]")


class DiskTeacherCache:
    """Validated disk-backed cache; every access remains train-only and rehashed."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: TeacherCacheManifest,
        sidecar: tuple[TeacherTransitionRecord, ...],
        calibration: FrozenTeacherLossCalibration,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.manifest = manifest
        self.sidecar = sidecar
        self.calibration = calibration
        self._provider = TeacherProvider(manifest, lambda record: _load_record(self.root, manifest, record))

    def get(self, state_id: str, *, split: str) -> TeacherState:
        return self._provider.get(state_id, split=split)

    def verify_all(self) -> None:
        for record in self.manifest.records:
            self.get(record.state_id, split=TRAIN_SPLIT)


def make_disk_teacher_provider(
    manifest_path: Path,
    *,
    cache_root: Path | None = None,
    expected_manifest_file_sha256: str | None = None,
) -> TeacherProvider:
    """Create a lazy, rehashing provider from a manifest path or cache directory."""

    source = Path(manifest_path).expanduser()
    if source.is_dir():
        root = source.resolve(strict=True)
        source = root / MANIFEST_FILENAME
    else:
        source = source.resolve(strict=True)
        root = Path(cache_root).expanduser().resolve(strict=True) if cache_root is not None else source.parent
    manifest = load_teacher_cache_manifest(
        source,
        expected_file_sha256=expected_manifest_file_sha256,
    )
    _validate_manifest_artifact_paths(manifest)
    return TeacherProvider(manifest, lambda record: _load_record(root, manifest, record))


def save_teacher_cache(
    root: Path,
    *,
    manifest: TeacherCacheManifest,
    teacher_states: Iterable[TeacherState],
    sidecar: Iterable[TeacherTransitionRecord | Mapping[str, Any]],
    calibration: FrozenTeacherLossCalibration,
) -> TeacherCacheFileHashes:
    """Write tensors first and atomically publish the manifest only after validation."""

    destination = Path(root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    _validate_manifest_artifact_paths(manifest)
    states = tuple(teacher_states)
    by_state_id = {state.state_id: state for state in states}
    if len(by_state_id) != len(states):
        raise ValueError("teacher_states contains duplicate state_id values.")
    if set(by_state_id) != set(manifest.by_state_id):
        raise ValueError("teacher_states state IDs must exactly match the manifest.")

    provider = TeacherProvider(manifest, lambda record: by_state_id[record.state_id])
    artifacts: dict[str, str] = {}
    for record in manifest.records:
        teacher = provider.get(record.state_id, split=TRAIN_SPLIT)
        for tensor, specification in (
            (teacher.image, record.image),
            (teacher.latent, record.latent),
            (teacher.feature, record.feature),
        ):
            path = _resolved_artifact_path(destination, specification.relative_path)
            artifacts[specification.relative_path] = save_teacher_tensor(
                path, tensor, specification=specification
            )

    validated_sidecar = validate_teacher_sidecar(sidecar, manifest=manifest)
    sidecar_sha256 = save_teacher_sidecar(
        destination / SIDECAR_FILENAME,
        validated_sidecar,
        manifest=manifest,
    )
    calibration_sha256 = save_teacher_calibration(destination / CALIBRATION_FILENAME, calibration)
    # Manifest replacement is the publication/commit point. Readers never accept a
    # partially written set of tensors under the new manifest.
    manifest_sha256 = save_teacher_manifest(destination / MANIFEST_FILENAME, manifest)
    return TeacherCacheFileHashes(
        manifest=manifest_sha256,
        sidecar=sidecar_sha256,
        calibration=calibration_sha256,
        artifacts=dict(sorted(artifacts.items())),
    )


def load_teacher_cache(
    root: Path,
    *,
    expected_manifest_file_sha256: str | None = None,
    expected_sidecar_file_sha256: str | None = None,
    expected_calibration_file_sha256: str | None = None,
    expected_calibration_contract_sha256: str | None = None,
    verify_all: bool = True,
) -> DiskTeacherCache:
    source = Path(root).expanduser().resolve(strict=True)
    manifest = load_teacher_manifest(
        source / MANIFEST_FILENAME,
        expected_file_sha256=expected_manifest_file_sha256,
    )
    _validate_manifest_artifact_paths(manifest)
    sidecar = load_teacher_sidecar(
        source / SIDECAR_FILENAME,
        manifest=manifest,
        expected_file_sha256=expected_sidecar_file_sha256,
    )
    calibration = load_teacher_calibration(
        source / CALIBRATION_FILENAME,
        expected_file_sha256=expected_calibration_file_sha256,
        expected_contract_sha256=expected_calibration_contract_sha256,
    )
    cache = DiskTeacherCache(root=source, manifest=manifest, sidecar=sidecar, calibration=calibration)
    if verify_all:
        cache.verify_all()
    return cache


__all__ = [
    "CALIBRATION_FILENAME",
    "MANIFEST_FILENAME",
    "SIDECAR_FILENAME",
    "TEACHER_CALIBRATION_FILE_SCHEMA",
    "TEACHER_TENSOR_SCHEMA",
    "DiskTeacherCache",
    "TeacherCacheFileHashes",
    "file_sha256",
    "load_teacher_cache",
    "load_teacher_cache_manifest",
    "load_teacher_calibration",
    "load_teacher_manifest",
    "load_teacher_sidecar",
    "load_teacher_transition_sidecar",
    "load_teacher_tensor",
    "make_disk_teacher_provider",
    "save_teacher_cache",
    "save_teacher_calibration",
    "save_teacher_manifest",
    "save_teacher_sidecar",
    "save_teacher_tensor",
]
