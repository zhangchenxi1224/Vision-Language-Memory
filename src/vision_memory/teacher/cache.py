"""Train-only teacher artifact, cache-manifest, and transition-sidecar contracts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import torch
from torch import Tensor

from .state import canonical_json_bytes, reject_supervision_keys, require_sha256


TEACHER_CACHE_SCHEMA = "vision_memory.teacher-cache.v1"
TEACHER_TRANSITION_SCHEMA = "vision_memory.teacher-transition.v1"
TRAIN_SPLIT = "train"
_EVENT_KINDS = frozenset({"set", "overwrite", "clear", "noop"})


def tensor_sha256(tensor: Tensor) -> str:
    """Hash tensor dtype, shape, and canonical contiguous CPU bytes."""

    if not isinstance(tensor, Tensor):
        raise TypeError("tensor_sha256 requires a torch.Tensor.")
    materialized = tensor.detach().cpu().contiguous()
    header = canonical_json_bytes(
        {"dtype": str(materialized.dtype).removeprefix("torch."), "shape": list(materialized.shape)}
    )
    raw = materialized.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256()
    digest.update(b"vision-memory-tensor-v1\0")
    digest.update(header)
    digest.update(b"\0")
    digest.update(raw)
    return digest.hexdigest()


def _validate_teacher_tensor(tensor: Tensor, *, name: str, exact_image: bool = False) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"Teacher {name} must be a torch.Tensor.")
    if tensor.numel() == 0 or tensor.ndim < 2 or tensor.shape[0] != 1:
        raise ValueError(f"Teacher {name} must be a non-empty batched tensor with batch size one.")
    if exact_image and tuple(tensor.shape) != (1, 3, 1024, 1024):
        raise ValueError("Teacher image must have shape [1, 3, 1024, 1024].")
    if not tensor.is_floating_point():
        raise TypeError(f"Teacher {name} must use a floating-point dtype.")
    if tensor.requires_grad or tensor.grad_fn is not None:
        raise ValueError(f"Teacher {name} must be detached and cannot require gradients.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"Teacher {name} contains a non-finite value.")
    if exact_image and (float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0):
        raise ValueError("Teacher image values must lie in [0, 1].")


@dataclass(frozen=True)
class TeacherState:
    """Resolved train-only tensors for one full semantic state."""

    state_id: str
    teacher_key: str
    semantic_state_sha256: str
    teacher_contract_sha256: str
    renderer_contract_sha256: str
    image: Tensor
    latent: Tensor
    feature: Tensor
    split: str = TRAIN_SPLIT

    def __post_init__(self) -> None:
        if self.split != TRAIN_SPLIT:
            raise ValueError("TeacherState is train-only; split must equal 'train'.")
        for field in (
            "state_id",
            "teacher_key",
            "semantic_state_sha256",
            "teacher_contract_sha256",
            "renderer_contract_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        _validate_teacher_tensor(self.image, name="image", exact_image=True)
        _validate_teacher_tensor(self.latent, name="latent")
        _validate_teacher_tensor(self.feature, name="feature")

    @property
    def artifact_sha256(self) -> dict[str, str]:
        return {
            "image": tensor_sha256(self.image),
            "latent": tensor_sha256(self.latent),
            "feature": tensor_sha256(self.feature),
        }


def _relative_cache_path(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a non-empty POSIX relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must remain inside the teacher cache root.")
    return path.as_posix()


@dataclass(frozen=True)
class TeacherArtifactSpec:
    relative_path: str
    sha256: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _relative_cache_path(self.relative_path, field="relative_path"))
        require_sha256(self.sha256, field="artifact.sha256")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("artifact.dtype must be non-empty.")
        shape = tuple(self.shape)
        if not shape or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
            raise ValueError("artifact.shape must contain positive integers.")
        object.__setattr__(self, "shape", shape)

    @classmethod
    def from_tensor(cls, tensor: Tensor, *, relative_path: str) -> "TeacherArtifactSpec":
        return cls(
            relative_path=relative_path,
            sha256=tensor_sha256(tensor),
            dtype=str(tensor.dtype).removeprefix("torch."),
            shape=tuple(tensor.shape),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherArtifactSpec":
        if set(value) != {"relative_path", "sha256", "dtype", "shape"}:
            raise ValueError("Teacher artifact fields differ from the locked cache schema.")
        return cls(
            relative_path=value["relative_path"],
            sha256=value["sha256"],
            dtype=value["dtype"],
            shape=tuple(value["shape"]),
        )


@dataclass(frozen=True)
class TeacherArtifactRecord:
    state_id: str
    teacher_key: str
    semantic_state_sha256: str
    image: TeacherArtifactSpec
    latent: TeacherArtifactSpec
    feature: TeacherArtifactSpec

    def __post_init__(self) -> None:
        require_sha256(self.state_id, field="state_id")
        require_sha256(self.teacher_key, field="teacher_key")
        require_sha256(self.semantic_state_sha256, field="semantic_state_sha256")

    @classmethod
    def from_teacher_state(cls, teacher: TeacherState) -> "TeacherArtifactRecord":
        prefix = f"artifacts/{teacher.teacher_key}"
        return cls(
            state_id=teacher.state_id,
            teacher_key=teacher.teacher_key,
            semantic_state_sha256=teacher.semantic_state_sha256,
            image=TeacherArtifactSpec.from_tensor(teacher.image, relative_path=f"{prefix}/image.pt"),
            latent=TeacherArtifactSpec.from_tensor(teacher.latent, relative_path=f"{prefix}/latent.pt"),
            feature=TeacherArtifactSpec.from_tensor(teacher.feature, relative_path=f"{prefix}/feature.pt"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "teacher_key": self.teacher_key,
            "semantic_state_sha256": self.semantic_state_sha256,
            "artifacts": {
                "image": self.image.to_dict(),
                "latent": self.latent.to_dict(),
                "feature": self.feature.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherArtifactRecord":
        reject_supervision_keys(value, path="teacher_cache.records[]")
        if set(value) != {"state_id", "teacher_key", "semantic_state_sha256", "artifacts"}:
            raise ValueError("Teacher record fields differ from the locked cache schema.")
        artifacts = value["artifacts"]
        if not isinstance(artifacts, Mapping) or set(artifacts) != {"image", "latent", "feature"}:
            raise ValueError("Teacher record requires image, latent, and feature artifacts.")
        return cls(
            state_id=value["state_id"],
            teacher_key=value["teacher_key"],
            semantic_state_sha256=value["semantic_state_sha256"],
            image=TeacherArtifactSpec.from_dict(artifacts["image"]),
            latent=TeacherArtifactSpec.from_dict(artifacts["latent"]),
            feature=TeacherArtifactSpec.from_dict(artifacts["feature"]),
        )


@dataclass(frozen=True)
class TeacherCacheManifest:
    teacher_contract_sha256: str
    renderer_contract_sha256: str
    records: tuple[TeacherArtifactRecord, ...]
    split: str = TRAIN_SPLIT
    schema: str = TEACHER_CACHE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TEACHER_CACHE_SCHEMA:
            raise ValueError(f"Unsupported teacher cache schema: {self.schema!r}.")
        if self.split != TRAIN_SPLIT:
            raise ValueError("Teacher caches are train-only.")
        require_sha256(self.teacher_contract_sha256, field="teacher_contract_sha256")
        require_sha256(self.renderer_contract_sha256, field="renderer_contract_sha256")
        records = tuple(self.records)
        state_ids = [record.state_id for record in records]
        teacher_keys = [record.teacher_key for record in records]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("Teacher cache contains duplicate state_id values.")
        if len(teacher_keys) != len(set(teacher_keys)):
            raise ValueError("Teacher cache contains duplicate teacher_key values.")
        object.__setattr__(self, "records", tuple(sorted(records, key=lambda record: record.state_id)))

    @property
    def by_state_id(self) -> dict[str, TeacherArtifactRecord]:
        return {record.state_id: record for record in self.records}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "split": self.split,
            "teacher_contract_sha256": self.teacher_contract_sha256,
            "renderer_contract_sha256": self.renderer_contract_sha256,
            "records": [record.to_dict() for record in self.records],
        }

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherCacheManifest":
        reject_supervision_keys(value, path="teacher_cache")
        expected = {
            "schema",
            "split",
            "teacher_contract_sha256",
            "renderer_contract_sha256",
            "records",
        }
        if set(value) != expected:
            raise ValueError("Teacher cache manifest fields differ from the locked schema.")
        records = value["records"]
        if not isinstance(records, list):
            raise TypeError("Teacher cache records must be a list.")
        return cls(
            schema=value["schema"],
            split=value["split"],
            teacher_contract_sha256=value["teacher_contract_sha256"],
            renderer_contract_sha256=value["renderer_contract_sha256"],
            records=tuple(TeacherArtifactRecord.from_dict(record) for record in records),
        )


class TeacherProvider:
    """Resolve a manifest entry through an injected loader and verify every tensor."""

    def __init__(self, manifest: TeacherCacheManifest, loader: Callable[[TeacherArtifactRecord], TeacherState]):
        if not callable(loader):
            raise TypeError("TeacherProvider loader must be callable.")
        self.manifest = manifest
        self._loader = loader

    def get(self, state_id: str, *, split: str) -> TeacherState:
        if split != TRAIN_SPLIT:
            raise ValueError("TeacherProvider refuses non-train access.")
        require_sha256(state_id, field="state_id")
        record = self.manifest.by_state_id.get(state_id)
        if record is None:
            raise KeyError(f"Teacher state {state_id} is not present in the locked cache.")
        teacher = self._loader(record)
        if not isinstance(teacher, TeacherState):
            raise TypeError("TeacherProvider loader must return TeacherState.")
        expected = {
            "state_id": record.state_id,
            "teacher_key": record.teacher_key,
            "semantic_state_sha256": record.semantic_state_sha256,
            "teacher_contract_sha256": self.manifest.teacher_contract_sha256,
            "renderer_contract_sha256": self.manifest.renderer_contract_sha256,
        }
        for field, value in expected.items():
            if getattr(teacher, field) != value:
                raise ValueError(f"Loaded TeacherState {field} does not match the manifest.")
        for name, specification in (
            ("image", record.image),
            ("latent", record.latent),
            ("feature", record.feature),
        ):
            tensor = getattr(teacher, name)
            if tensor_sha256(tensor) != specification.sha256:
                raise ValueError(f"Loaded teacher {name} SHA256 does not match the manifest.")
            if str(tensor.dtype).removeprefix("torch.") != specification.dtype or tuple(tensor.shape) != specification.shape:
                raise ValueError(f"Loaded teacher {name} dtype/shape does not match the manifest.")
        return teacher


@dataclass(frozen=True)
class TeacherTransitionRecord:
    episode_id: str
    turn_id: int
    before_state_id: str
    after_state_id: str
    event_kind: str
    teacher_key: str
    split: str = TRAIN_SPLIT
    schema: str = TEACHER_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != TEACHER_TRANSITION_SCHEMA:
            raise ValueError(f"Unsupported teacher transition schema: {self.schema!r}.")
        if self.split != TRAIN_SPLIT:
            raise ValueError("Teacher transition sidecars are train-only.")
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("episode_id must be non-empty.")
        if isinstance(self.turn_id, bool) or not isinstance(self.turn_id, int) or self.turn_id < 0:
            raise ValueError("turn_id must be a non-negative integer.")
        for field in ("before_state_id", "after_state_id", "teacher_key"):
            require_sha256(getattr(self, field), field=field)
        normalized_kind = str(self.event_kind).casefold()
        if normalized_kind not in _EVENT_KINDS:
            raise ValueError(f"event_kind must be one of {sorted(_EVENT_KINDS)}.")
        object.__setattr__(self, "event_kind", normalized_kind)
        if normalized_kind == "noop" and self.before_state_id != self.after_state_id:
            raise ValueError("A no-op transition must preserve state_id.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "split": self.split,
            "episode_id": self.episode_id,
            "turn_id": self.turn_id,
            "before_state_id": self.before_state_id,
            "after_state_id": self.after_state_id,
            "event_kind": self.event_kind,
            "teacher_key": self.teacher_key,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TeacherTransitionRecord":
        reject_supervision_keys(value, path="teacher_transition")
        expected = {
            "schema",
            "split",
            "episode_id",
            "turn_id",
            "before_state_id",
            "after_state_id",
            "event_kind",
            "teacher_key",
        }
        if set(value) != expected:
            raise ValueError("Teacher transition fields differ from the locked sidecar schema.")
        return cls(**value)


def validate_teacher_sidecar(
    records: Iterable[TeacherTransitionRecord | Mapping[str, Any]],
    *,
    manifest: TeacherCacheManifest,
) -> tuple[TeacherTransitionRecord, ...]:
    """Validate continuity, no-op identity, and path-invariant teacher lookup."""

    parsed = tuple(
        record if isinstance(record, TeacherTransitionRecord) else TeacherTransitionRecord.from_dict(record)
        for record in records
    )
    by_state = manifest.by_state_id
    seen_turns: set[tuple[str, int]] = set()
    by_episode: dict[str, list[TeacherTransitionRecord]] = defaultdict(list)
    for record in parsed:
        turn_key = (record.episode_id, record.turn_id)
        if turn_key in seen_turns:
            raise ValueError(f"Duplicate teacher transition sidecar key: {turn_key!r}.")
        seen_turns.add(turn_key)
        before = by_state.get(record.before_state_id)
        after = by_state.get(record.after_state_id)
        if before is None or after is None:
            raise ValueError("Teacher transition references a state outside the train-only manifest.")
        if after.teacher_key != record.teacher_key:
            raise ValueError("Teacher transition teacher_key is not path-invariant for after_state_id.")
        by_episode[record.episode_id].append(record)

    for episode_id, episode_records in by_episode.items():
        ordered = sorted(episode_records, key=lambda record: record.turn_id)
        for previous, current in zip(ordered, ordered[1:]):
            if previous.after_state_id != current.before_state_id:
                raise ValueError(f"Teacher transition state continuity failed for episode {episode_id!r}.")
    return tuple(sorted(parsed, key=lambda record: (record.episode_id, record.turn_id)))


def manifest_json(manifest: TeacherCacheManifest) -> str:
    return json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "TEACHER_CACHE_SCHEMA",
    "TEACHER_TRANSITION_SCHEMA",
    "TRAIN_SPLIT",
    "TeacherArtifactRecord",
    "TeacherArtifactSpec",
    "TeacherCacheManifest",
    "TeacherProvider",
    "TeacherState",
    "TeacherTransitionRecord",
    "manifest_json",
    "tensor_sha256",
    "validate_teacher_sidecar",
]
