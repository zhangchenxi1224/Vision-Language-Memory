"""Prospective input locks for R3 teacher-loss calibration.

The generated calibration scales are meaningful only for the exact student
initialization, train episodes, and teacher cache used to estimate them.  This
module keeps that input identity separate from the generated calibration-file
SHA so callers cannot substitute a valid calibration from another micro suite.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CALIBRATION_SUITES = ("set8", "transition16")
CALIBRATION_SAMPLE_SELECTION = {
    "split": "train",
    "unit": "one-unweighted-sample-per-updater-transition",
    "query_turns_excluded": True,
    "duplicate_semantic_after_states_retained": True,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"R3 preregistration field {field} must be an object.")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"R3 preregistration field {field} must be a lowercase SHA256 digest.")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"R3 preregistration field {field} must be a positive integer.")
    return value


@dataclass(frozen=True)
class TeacherCalibrationInputLock:
    suite: str
    preregistration_sha256: str
    train_sha256: str
    manifest_sha256: str
    sidecar_sha256: str
    transition_count: int

    def __post_init__(self) -> None:
        if self.suite not in CALIBRATION_SUITES:
            raise ValueError(f"Unsupported R3 calibration suite: {self.suite!r}.")
        for field in (
            "preregistration_sha256",
            "train_sha256",
            "manifest_sha256",
            "sidecar_sha256",
        ):
            if _SHA256.fullmatch(str(getattr(self, field))) is None:
                raise ValueError(f"{field} must be a lowercase SHA256 digest.")
        _positive_int(self.transition_count, field="transition_count")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "suite": self.suite,
            "preregistration_sha256": self.preregistration_sha256,
            "train_sha256": self.train_sha256,
            "manifest_sha256": self.manifest_sha256,
            "sidecar_sha256": self.sidecar_sha256,
            "transition_count": self.transition_count,
        }


def load_teacher_calibration_input_lock(
    preregistration_path: Path,
    *,
    suite: str,
) -> TeacherCalibrationInputLock:
    """Load one suite's immutable train/cache calibration identity."""

    if suite not in CALIBRATION_SUITES:
        raise ValueError(f"suite must be one of {CALIBRATION_SUITES}.")
    value = json.loads(preregistration_path.read_text(encoding="utf-8"))
    root = _mapping(value, field="root")
    if root.get("schema") != "vision_memory.r3-preregistration.v1":
        raise ValueError("Teacher calibration requires the locked R3 preregistration schema.")
    micro = _mapping(root.get("micro_data"), field="micro_data")
    suite_data = _mapping(micro.get(suite), field=f"micro_data.{suite}")
    teacher = _mapping(root.get("teacher_contract"), field="teacher_contract")
    manifests = _mapping(
        teacher.get("cache_manifest_sha256"),
        field="teacher_contract.cache_manifest_sha256",
    )
    cache_builds = _mapping(teacher.get("cache_builds"), field="teacher_contract.cache_builds")
    cache_build = _mapping(cache_builds.get(suite), field=f"teacher_contract.cache_builds.{suite}")
    protocol = _mapping(
        teacher.get("calibration_protocol"),
        field="teacher_contract.calibration_protocol",
    )
    expected_protocol = {
        "global_seed": 0,
        "adapter_seed": 0,
        "lora_rank": 4,
        "initial_state": "blank_1024",
        "strict_cuda_determinism": True,
        "sdpa_backend": "math_only",
        "sample_unit": "one_unweighted_sample_per_updater_transition",
        "query_turns_excluded": True,
        "duplicate_semantic_after_states_retained": True,
        "component_reduction": "median",
    }
    for field, expected in expected_protocol.items():
        if protocol.get(field) != expected:
            raise ValueError(f"R3 teacher calibration protocol field {field!r} differs from the lock.")
    transition_field = f"{suite}_transition_samples"
    return TeacherCalibrationInputLock(
        suite=suite,
        preregistration_sha256=_sha256_file(preregistration_path),
        train_sha256=_sha256(suite_data.get("train_sha256"), field=f"micro_data.{suite}.train_sha256"),
        manifest_sha256=_sha256(
            manifests.get(suite),
            field=f"teacher_contract.cache_manifest_sha256.{suite}",
        ),
        sidecar_sha256=_sha256(
            cache_build.get("transitions_sha256"),
            field=f"teacher_contract.cache_builds.{suite}.transitions_sha256",
        ),
        transition_count=_positive_int(
            protocol.get(transition_field),
            field=f"teacher_contract.calibration_protocol.{transition_field}",
        ),
    )


def verify_teacher_calibration_input_files(
    lock: TeacherCalibrationInputLock,
    *,
    train: Path,
    manifest: Path,
    sidecar: Path,
) -> dict[str, str | int]:
    """Hash all producer inputs and fail before GPU work on any substitution."""

    observed = {
        "suite": lock.suite,
        "preregistration_sha256": lock.preregistration_sha256,
        "train_sha256": _sha256_file(train),
        "manifest_sha256": _sha256_file(manifest),
        "sidecar_sha256": _sha256_file(sidecar),
        "transition_count": lock.transition_count,
    }
    if observed != lock.to_dict():
        differences = {
            field: {"expected": lock.to_dict()[field], "observed": observed[field]}
            for field in lock.to_dict()
            if observed[field] != lock.to_dict()[field]
        }
        raise ValueError(f"Teacher calibration inputs differ from preregistration: {differences}.")
    return observed
