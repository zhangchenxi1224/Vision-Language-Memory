from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

from r3_teacher_cache_compatibility import (  # noqa: E402
    PROBE_NAME,
    PROBE_SCHEMA_VERSION,
    load_preregistered_cache_locks,
)
from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_GRID_THW,
    R3_QWEN_READER_PIXEL_VALUES_SHAPE,
    R3_QWEN_READER_RESIZE_CONTRACT,
)
from vision_memory.repro import REQUIRED_DETERMINISM_ENV, canonical_json_sha256  # noqa: E402
from vision_memory.teacher import file_sha256  # noqa: E402


VALIDATION_PROTOCOL = "R3-TC0-cache-forward-compatibility-validation.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ARTIFACT_KINDS = ("image", "latent", "feature")
_PROCESSOR_CHECKS = {
    "fast_tensor_processor",
    "resize_enabled_by_default",
    "min_pixels_locked",
    "max_pixels_locked",
    "patch_size_locked",
    "temporal_patch_size_locked",
    "merge_size_locked",
    "bicubic_resample_locked",
    "callable",
}
_FORWARD_CHECKS = {
    "pixel_values_torch_equal",
    "pixel_values_max_absolute_difference_zero",
    "pixel_values_shape_locked",
    "pixel_values_dtype_locked",
    "pixel_values_finite",
    "grid_torch_equal",
    "grid_values_locked",
    "grid_shape_locked",
    "grid_dtype_locked",
}


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object.")
    return value


def _list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list.")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest.")
    return value


def _full_commit(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase full Git commit.")
    return value


def _valid_tensor_evidence(value: Any, *, shape: list[int], dtype: str) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(
        value.get("schema_version") == "vision_memory.canonical_tensor.v1"
        and value.get("shape") == shape
        and value.get("dtype") == dtype
        and _SHA256.fullmatch(str(value.get("sha256"))) is not None
        and value.get("finite") is True
        and isinstance(value.get("device"), str)
    )


def _validate_forward(
    value: Any,
    *,
    image_tensor_content_sha256: str,
    label: str,
    errors: list[str],
) -> None:
    try:
        forward = _mapping(value, field=label)
        if forward.get("passed") is not True:
            errors.append(f"{label} did not pass")
        if forward.get("reader_resize_contract") != R3_QWEN_READER_RESIZE_CONTRACT:
            errors.append(f"{label} has the wrong Reader resize contract")
        if forward.get("source_teacher_tensor_sha256") != image_tensor_content_sha256:
            errors.append(f"{label} is not bound to the cached image tensor")
        input_evidence = _mapping(forward.get("input"), field=f"{label}.input")
        input_dtype = input_evidence.get("dtype")
        if not _valid_tensor_evidence(
            input_evidence,
            shape=[3, 1024, 1024],
            dtype=str(input_dtype),
        ):
            errors.append(f"{label} input tensor evidence is invalid")
        if input_dtype not in {"float16", "bfloat16", "float32"}:
            errors.append(f"{label} input dtype is outside the locked floating dtypes")
        if forward.get("source_canonical_tensor_sha256") != input_evidence.get("sha256"):
            errors.append(f"{label} device input is not bound to the cached canonical tensor SHA")
        if not _valid_tensor_evidence(forward.get("explicitly_resized"), shape=[3, 256, 256], dtype=str(input_dtype)):
            errors.append(f"{label} explicit-resize tensor evidence is invalid")
        for name in ("legacy_pixel_values", "candidate_pixel_values"):
            if not _valid_tensor_evidence(
                forward.get(name), shape=list(R3_QWEN_READER_PIXEL_VALUES_SHAPE), dtype="float32"
            ):
                errors.append(f"{label} {name} evidence is invalid")
        legacy_pixels = _mapping(forward.get("legacy_pixel_values"), field=f"{label}.legacy_pixel_values")
        candidate_pixels = _mapping(forward.get("candidate_pixel_values"), field=f"{label}.candidate_pixel_values")
        legacy_sha = legacy_pixels.get("sha256")
        candidate_sha = candidate_pixels.get("sha256")
        if legacy_sha != candidate_sha:
            errors.append(f"{label} legacy/candidate pixel SHA differs")
        for name in ("legacy_grid", "candidate_grid"):
            if not _valid_tensor_evidence(forward.get(name), shape=[1, 3], dtype="int64"):
                errors.append(f"{label} {name} evidence is invalid")
        legacy_grid = _mapping(forward.get("legacy_grid"), field=f"{label}.legacy_grid")
        candidate_grid = _mapping(forward.get("candidate_grid"), field=f"{label}.candidate_grid")
        if legacy_grid.get("sha256") != candidate_grid.get("sha256"):
            errors.append(f"{label} legacy/candidate grid SHA differs")
        expected_grid = list(R3_QWEN_READER_GRID_THW)
        if (
            forward.get("legacy_grid_thw") != expected_grid
            or forward.get("candidate_grid_thw") != expected_grid
            or forward.get("expected_grid_thw") != expected_grid
        ):
            errors.append(f"{label} grid values differ from the lock")
        if forward.get("expected_pixel_values_shape") != list(R3_QWEN_READER_PIXEL_VALUES_SHAPE):
            errors.append(f"{label} pixel shape lock is missing")
        if forward.get("pixel_values_max_absolute_difference") != 0.0:
            errors.append(f"{label} maximum pixel difference is not zero")
        checks = _mapping(forward.get("checks"), field=f"{label}.checks")
        if set(checks) != _FORWARD_CHECKS or any(value is not True for value in checks.values()):
            errors.append(f"{label} does not pass every forward check")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"{label} is invalid: {exc}")


def _validated_report_map(value: Any, *, field: str) -> dict[str, str]:
    source = _mapping(value, field=field)
    result: dict[str, str] = {}
    for path, digest in source.items():
        if not isinstance(path, str) or not path.startswith("artifacts/") or not path.endswith(".pt"):
            raise ValueError(f"{field} contains an invalid artifact path.")
        result[path] = _sha256(digest, field=f"{field}[{path!r}]")
    return dict(sorted(result.items()))


def _validate_suite(
    *,
    suite: str,
    value: Any,
    lock: Mapping[str, Any],
    expected_reader_revision: str,
    expected_cache_build_commit: str,
    errors: list[str],
) -> dict[str, int]:
    counts = {"state_count": 0, "artifact_tensor_count": 0, "image_forward_count": 0}
    try:
        report = _mapping(value, field=f"suites.{suite}")
        if report.get("suite") != suite or report.get("passed") is not True:
            errors.append(f"{suite} suite identity/pass flag is invalid")
        if report.get("expected_lock") != lock:
            errors.append(f"{suite} report does not reproduce its complete preregistered lock")
        observed = _mapping(report.get("observed"), field=f"{suite}.observed")
        direct_locks = {
            "manifest_sha256": "manifest_sha256",
            "transitions_sha256": "transitions_sha256",
            "build_report_sha256": "build_report_sha256",
            "artifact_file_sha_map_sha256": "artifact_file_sha_map_sha256",
            "tensor_content_sha_map_sha256": "tensor_content_sha_map_sha256",
            "state_count": "state_count",
            "transition_count": "transition_count",
            "artifact_tensor_count": "artifact_tensor_count",
        }
        for observed_field, lock_field in direct_locks.items():
            if observed.get(observed_field) != lock.get(lock_field):
                errors.append(f"{suite}.{observed_field} differs from preregistration")
        if observed.get("cache_build_commit") != expected_cache_build_commit:
            errors.append(f"{suite} has the wrong cache-build commit")
        if observed.get("reader_revision") != expected_reader_revision:
            errors.append(f"{suite} has the wrong Reader revision")
        for field in ("teacher_contract_sha256", "renderer_contract_sha256"):
            _sha256(observed.get(field), field=f"{suite}.observed.{field}")
        if observed.get("image_forward_count") != lock.get("state_count"):
            errors.append(f"{suite} does not contain one image-forward proof per state")

        maps = _mapping(report.get("maps"), field=f"{suite}.maps")
        file_map = _validated_report_map(maps.get("artifact_file_sha256"), field=f"{suite}.maps.artifact_file_sha256")
        content_map = _validated_report_map(
            maps.get("tensor_content_sha256"), field=f"{suite}.maps.tensor_content_sha256"
        )
        if set(file_map) != set(content_map):
            errors.append(f"{suite} file/content maps enumerate different artifact paths")
        if canonical_json_sha256(file_map) != lock.get("artifact_file_sha_map_sha256"):
            errors.append(f"{suite} artifact file map hash differs from preregistration")
        if canonical_json_sha256(content_map) != lock.get("tensor_content_sha_map_sha256"):
            errors.append(f"{suite} tensor content map hash differs from preregistration")

        states = _list(report.get("states"), field=f"{suite}.states")
        state_ids = [state.get("state_id") if isinstance(state, Mapping) else None for state in states]
        if (
            len(states) != lock.get("state_count")
            or state_ids != sorted(state_ids)
            or len(state_ids) != len(set(state_ids))
        ):
            errors.append(f"{suite} state evidence is incomplete, duplicated, or unsorted")
        evidence_paths: set[str] = set()
        for index, state_value in enumerate(states):
            label = f"{suite}.states[{index}]"
            try:
                state = _mapping(state_value, field=label)
                for identity in ("state_id", "teacher_key", "semantic_state_sha256"):
                    _sha256(state.get(identity), field=f"{label}.{identity}")
                if state.get("passed") is not True:
                    errors.append(f"{label} did not pass")
                artifacts = _mapping(state.get("artifacts"), field=f"{label}.artifacts")
                if set(artifacts) != set(_ARTIFACT_KINDS):
                    errors.append(f"{label} does not contain exactly image/latent/feature")
                    continue
                for kind in _ARTIFACT_KINDS:
                    artifact = _mapping(artifacts[kind], field=f"{label}.artifacts.{kind}")
                    path = artifact.get("relative_path")
                    if not isinstance(path, str) or path in evidence_paths:
                        errors.append(f"{label}.{kind} has an invalid or duplicate path")
                        continue
                    evidence_paths.add(path)
                    expected_path = f"artifacts/{state['teacher_key']}/{kind}.pt"
                    if path != expected_path:
                        errors.append(f"{label}.{kind} path is not bound to teacher_key")
                    if artifact.get("passed") is not True:
                        errors.append(f"{label}.{kind} did not pass")
                    if artifact.get("file_sha256") != file_map.get(path) or artifact.get(
                        "build_report_file_sha256"
                    ) != file_map.get(path):
                        errors.append(f"{label}.{kind} is not bound to the file SHA map")
                    if artifact.get("tensor_content_sha256") != content_map.get(path) or artifact.get(
                        "manifest_tensor_content_sha256"
                    ) != content_map.get(path):
                        errors.append(f"{label}.{kind} is not bound to the tensor-content SHA map")
                    dtype = artifact.get("dtype")
                    shape = artifact.get("shape")
                    if (
                        dtype not in {"float16", "bfloat16", "float32"}
                        or not isinstance(shape, list)
                        or len(shape) < 2
                        or shape[0] != 1
                        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
                    ):
                        errors.append(f"{label}.{kind} lacks dtype/shape evidence")
                    if kind == "image" and shape != [1, 3, 1024, 1024]:
                        errors.append(f"{label}.image shape differs from the locked decoded image")
                image_content_sha = str(artifacts["image"].get("tensor_content_sha256"))
                _validate_forward(
                    state.get("forward"),
                    image_tensor_content_sha256=image_content_sha,
                    label=f"{label}.forward",
                    errors=errors,
                )
                forward_value = state.get("forward")
                forward_input = forward_value.get("input") if isinstance(forward_value, Mapping) else None
                if not isinstance(forward_input, Mapping) or forward_input.get("dtype") != artifacts["image"].get(
                    "dtype"
                ):
                    errors.append(f"{label}.forward input dtype differs from cached image dtype")
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{label} is invalid: {exc}")
        if evidence_paths != set(file_map):
            errors.append(f"{suite} per-state evidence does not cover the complete artifact maps")

        read_only = _mapping(report.get("read_only_integrity"), field=f"{suite}.read_only_integrity")
        before = _mapping(read_only.get("core_file_sha256_before"), field=f"{suite}.core_before")
        after = _mapping(read_only.get("core_file_sha256_after"), field=f"{suite}.core_after")
        expected_core = {
            "manifest.json": lock.get("manifest_sha256"),
            "transitions.jsonl": lock.get("transitions_sha256"),
            "build_report.json": lock.get("build_report_sha256"),
        }
        if before != expected_core or after != expected_core or read_only.get("unchanged") is not True:
            errors.append(f"{suite} does not prove read-only core-file integrity")
        if read_only.get("artifact_file_sha_map_sha256_before") != lock.get(
            "artifact_file_sha_map_sha256"
        ) or read_only.get("artifact_file_sha_map_sha256_after") != lock.get("artifact_file_sha_map_sha256"):
            errors.append(f"{suite} does not prove read-only artifact-map integrity")

        counts = {
            "state_count": len(states),
            "artifact_tensor_count": len(evidence_paths),
            "image_forward_count": len(states),
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"{suite} suite report is invalid: {exc}")
    return counts


def validate_tc0_report(
    report: Mapping[str, Any],
    *,
    locks: Mapping[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    _full_commit(expected_commit, field="expected_commit")
    errors: list[str] = []
    if report.get("schema_version") != PROBE_SCHEMA_VERSION or report.get("probe") != PROBE_NAME:
        errors.append("raw R3-TC0 probe identity is invalid")
    if report.get("passed") is not True or "error" in report:
        errors.append("raw R3-TC0 probe did not pass fail-closed")
    if report.get("reader_revision") != locks.get("reader_revision"):
        errors.append("raw R3-TC0 report has the wrong Reader revision")
    if report.get("reader_resize_contract") != R3_QWEN_READER_RESIZE_CONTRACT:
        errors.append("raw R3-TC0 report has the wrong Reader resize contract")
    if report.get("preregistration") != locks:
        errors.append("raw R3-TC0 report does not bind the complete preregistered cache locks")

    provenance = report.get("provenance")
    git = provenance.get("git") if isinstance(provenance, Mapping) else None
    if not isinstance(git, Mapping) or git.get("commit") != expected_commit or git.get("clean") is not True:
        errors.append("raw R3-TC0 provenance is not the expected clean commit")
    models = provenance.get("models") if isinstance(provenance, Mapping) else None
    model = models.get("reader") if isinstance(models, Mapping) else None
    if (
        not isinstance(model, Mapping)
        or model.get("observed_revision") != locks.get("reader_revision")
        or model.get("expected_revision") != locks.get("reader_revision")
        or model.get("revision_matches_lock") is not True
    ):
        errors.append("raw R3-TC0 Reader model provenance is invalid")

    processor = report.get("processor")
    processor_checks = processor.get("checks") if isinstance(processor, Mapping) else None
    if (
        not isinstance(processor, Mapping)
        or processor.get("passed") is not True
        or not isinstance(processor_checks, Mapping)
        or set(processor_checks) != _PROCESSOR_CHECKS
        or any(value is not True for value in processor_checks.values())
    ):
        errors.append("raw R3-TC0 fast processor audit did not pass")
    determinism = report.get("determinism")
    if (
        not isinstance(determinism, Mapping)
        or determinism.get("deterministic_algorithms") is not True
        or determinism.get("deterministic_warn_only") is not False
        or determinism.get("environment") != dict(sorted(REQUIRED_DETERMINISM_ENV.items()))
    ):
        errors.append("raw R3-TC0 strict determinism contract is invalid")
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device_type") != "cuda":
        errors.append("raw R3-TC0 did not run on CUDA")
    else:
        if str(locks.get("gpu_model")) not in str(runtime.get("device_name")):
            errors.append("raw R3-TC0 did not run on the preregistered GPU model")
        if runtime.get("device_capability") != [9, 0]:
            errors.append("raw R3-TC0 device capability is not NVIDIA Hopper 9.0")
        if runtime.get("torch") != locks.get("torch_version"):
            errors.append("raw R3-TC0 Torch version differs from preregistration")
        if runtime.get("cuda") != locks.get("cuda_version"):
            errors.append("raw R3-TC0 CUDA version differs from preregistration")
        if runtime.get("transformers") != locks.get("transformers_version"):
            errors.append("raw R3-TC0 Transformers version differs from preregistration")
        if runtime.get("torchvision") != locks.get("torchvision_version"):
            errors.append("raw R3-TC0 Torchvision version differs from preregistration")

    suite_locks = locks.get("suites")
    suites = report.get("suites")
    aggregate = {"state_count": 0, "artifact_tensor_count": 0, "image_forward_count": 0}
    if not isinstance(suite_locks, Mapping) or set(suite_locks) != {"set8", "transition16"}:
        raise ValueError("Validator locks must contain exactly Set8 and Transition16.")
    if not isinstance(suites, Mapping) or set(suites) != {"set8", "transition16"}:
        errors.append("raw R3-TC0 report must contain exactly Set8 and Transition16 suites")
    else:
        for suite in ("set8", "transition16"):
            counts = _validate_suite(
                suite=suite,
                value=suites[suite],
                lock=_mapping(suite_locks[suite], field=f"locks.suites.{suite}"),
                expected_reader_revision=str(locks.get("reader_revision")),
                expected_cache_build_commit=str(locks.get("cache_build_commit")),
                errors=errors,
            )
            for field, value in counts.items():
                aggregate[field] += value

    expected_summary = {
        "suite_count": 2,
        "suite_pass_count": 2,
        "state_count": aggregate["state_count"],
        "artifact_tensor_count": aggregate["artifact_tensor_count"],
        "image_forward_count": aggregate["image_forward_count"],
        "image_forward_pass_count": aggregate["image_forward_count"],
        "cache_mutation_count": 0,
    }
    if report.get("summary") != expected_summary:
        errors.append("raw R3-TC0 aggregate summary is inconsistent with complete state evidence")
    scope = report.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("cache_access") != "read-only"
        or "attention-backend" not in str(scope.get("excluded"))
    ):
        errors.append("raw R3-TC0 scope does not preserve the separate feature-backend audit")
    feature_backend = report.get("feature_backend_compatibility")
    if (
        not isinstance(feature_backend, Mapping)
        or feature_backend.get("status") != "not_evaluated_by_r3_tc0"
        or "default SDPA" not in str(feature_backend.get("risk"))
        or "R3-TF0" not in str(feature_backend.get("required_followup"))
        or feature_backend.get("preregistered_thresholds") != {"relative_l2_max": 0.01, "cosine_min": 0.9999}
        or feature_backend.get("teacher_t0_unlocked") is not False
        or feature_backend.get("teacher_calibration_unlocked") is not False
        or feature_backend.get("teacher_assisted_training_unlocked") is not False
    ):
        errors.append("raw R3-TC0 does not fail closed on unresolved cached-feature backend compatibility")

    return {
        "schema_version": 1,
        "protocol": VALIDATION_PROTOCOL,
        "expected_commit": expected_commit,
        "reader_revision": locks.get("reader_revision"),
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "preregistration_sha256": locks.get("preregistration_sha256"),
        "validated_suites": ["set8", "transition16"],
        "validated_state_count": aggregate["state_count"],
        "validated_artifact_tensor_count": aggregate["artifact_tensor_count"],
        "validated_image_forward_count": aggregate["image_forward_count"],
        "cache_forward_compatibility_complete": not errors,
        "feature_backend_compatibility_complete": False,
        "teacher_t0_unlocked": False,
        "teacher_calibration_unlocked": False,
        "teacher_assisted_training_unlocked": False,
        "errors": errors,
        "passed": not errors,
    }


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed validator for raw R3-TC0 evidence")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _sha256(args.report_sha256, field="report_sha256")
        _sha256(args.preregistration_sha256, field="preregistration_sha256")
        _full_commit(args.expected_commit, field="expected_commit")
        report_path = args.report.expanduser().resolve(strict=True)
        preregistration_path = args.preregistration.expanduser().resolve(strict=True)
        if file_sha256(report_path) != args.report_sha256:
            raise ValueError("Raw R3-TC0 report SHA256 mismatch.")
        if file_sha256(preregistration_path) != args.preregistration_sha256:
            raise ValueError("R3 preregistration SHA256 mismatch.")
        locks = load_preregistered_cache_locks(preregistration_path)
        if locks["preregistration_sha256"] != args.preregistration_sha256:
            raise ValueError("Loaded R3 preregistration differs from the supplied SHA256 lock.")
        result = validate_tc0_report(
            _json_object(report_path),
            locks=locks,
            expected_commit=args.expected_commit,
        )
    except Exception as exc:  # noqa: BLE001 - validator must always materialize a fail-closed artifact
        result = {
            "schema_version": 1,
            "protocol": VALIDATION_PROTOCOL,
            "errors": [str(exc)],
            "passed": False,
        }
    result["inputs"] = {
        "report": str(args.report),
        "report_sha256": args.report_sha256,
        "preregistration": str(args.preregistration),
        "preregistration_sha256": args.preregistration_sha256,
        "expected_commit": args.expected_commit,
    }
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
