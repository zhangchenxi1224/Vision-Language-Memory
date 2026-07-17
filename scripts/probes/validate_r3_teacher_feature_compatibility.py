from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

import r3_teacher_feature_compatibility as tf0  # noqa: E402
from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_GRID_THW,
    R3_QWEN_READER_PIXEL_VALUES_SHAPE,
    R3_QWEN_READER_RESIZE_CONTRACT,
)
from vision_memory.repro import REQUIRED_DETERMINISM_ENV, canonical_json_sha256  # noqa: E402
from vision_memory.teacher import file_sha256  # noqa: E402


VALIDATION_PROTOCOL = "R3-TF0-feature-backend-compatibility-validation.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
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
_PREPROCESSING_CHECKS = {
    "pixel_values_shape_locked",
    "pixel_values_dtype_locked",
    "pixel_values_finite",
    "grid_shape_locked",
    "grid_dtype_locked",
    "grid_values_locked",
}
_FEATURE_CHECKS = {
    "shape_equal",
    "dtype_equal",
    "finite",
    "nonzero_norms",
    "metrics_finite",
    "relative_l2_within_gate",
    "cosine_within_gate",
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


def _commit(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase full Git commit.")
    return value


def _canonical_tensor(
    value: Any,
    *,
    field: str,
    dtype: str | None = None,
    shape: list[int] | None = None,
) -> Mapping[str, Any]:
    tensor = _mapping(value, field=field)
    if tensor.get("schema_version") != "vision_memory.canonical_tensor.v1":
        raise ValueError(f"{field} has the wrong canonical tensor schema.")
    _sha256(tensor.get("sha256"), field=f"{field}.sha256")
    actual_shape = tensor.get("shape")
    if (
        not isinstance(actual_shape, list)
        or not actual_shape
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in actual_shape)
    ):
        raise ValueError(f"{field}.shape is invalid.")
    if dtype is not None and tensor.get("dtype") != dtype:
        raise ValueError(f"{field}.dtype differs from {dtype}.")
    if shape is not None and actual_shape != shape:
        raise ValueError(f"{field}.shape differs from {shape}.")
    return tensor


def _artifact_sha_map(value: Any, *, field: str) -> dict[str, str]:
    source = _mapping(value, field=field)
    result: dict[str, str] = {}
    for path, digest in source.items():
        if not isinstance(path, str) or not path.startswith("artifacts/") or not path.endswith(".pt"):
            raise ValueError(f"{field} contains an invalid artifact path.")
        result[path] = _sha256(digest, field=f"{field}[{path!r}]")
    return dict(sorted(result.items()))


def _validate_feature_comparison(
    value: Any,
    *,
    cached_content_sha256: str,
    cached_shape: list[int],
    gate: Mapping[str, Any],
    label: str,
    errors: list[str],
) -> None:
    try:
        comparison = _mapping(value, field=label)
        if comparison.get("passed") is not True:
            errors.append(f"{label} did not pass")
        if comparison.get("reference") != "immutable_cached_default_sdpa_feature":
            errors.append(f"{label} has the wrong reference feature")
        if comparison.get("replacement") != "strict_math_only_query_free_feature":
            errors.append(f"{label} has the wrong replacement feature")
        thresholds = _mapping(comparison.get("thresholds"), field=f"{label}.thresholds")
        if thresholds != {
            "relative_l2_max": gate["relative_l2_max"],
            "cosine_min": gate["cosine_min"],
        }:
            errors.append(f"{label} thresholds differ from preregistration")
        cached = _canonical_tensor(
            comparison.get("cached"),
            field=f"{label}.cached",
            dtype="bfloat16",
            shape=cached_shape,
        )
        replacement = _canonical_tensor(
            comparison.get("replacement_feature"),
            field=f"{label}.replacement_feature",
            dtype="bfloat16",
            shape=cached_shape,
        )
        if comparison.get("cached_teacher_tensor_sha256") != cached_content_sha256:
            errors.append(f"{label} is not bound to the immutable cached feature")
        _sha256(
            comparison.get("replacement_teacher_tensor_sha256"),
            field=f"{label}.replacement_teacher_tensor_sha256",
        )
        if cached.get("sha256") == replacement.get("sha256"):
            # Exact equality is allowed and expected on some runtimes; this branch
            # intentionally has no failure. It documents that hashes were parsed.
            pass
        relative_l2 = comparison.get("relative_l2")
        cosine = comparison.get("cosine")
        numeric = (
            relative_l2,
            cosine,
            comparison.get("reference_norm"),
            comparison.get("replacement_norm"),
            comparison.get("difference_norm"),
            comparison.get("maximum_absolute_difference"),
        )
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
            for item in numeric
        ):
            errors.append(f"{label} contains a missing or non-finite metric")
        else:
            if float(relative_l2) > float(gate["relative_l2_max"]):
                errors.append(f"{label} relative L2 exceeds preregistration")
            if float(cosine) < float(gate["cosine_min"]):
                errors.append(f"{label} cosine is below preregistration")
            if float(comparison["reference_norm"]) <= 0 or float(comparison["replacement_norm"]) <= 0:
                errors.append(f"{label} has a zero feature norm")
            if float(comparison["difference_norm"]) < 0 or float(comparison["maximum_absolute_difference"]) < 0:
                errors.append(f"{label} has an invalid nonnegative distance")
        checks = _mapping(comparison.get("checks"), field=f"{label}.checks")
        if set(checks) != _FEATURE_CHECKS or any(item is not True for item in checks.values()):
            errors.append(f"{label} does not pass every locked feature check")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"{label} is invalid: {exc}")


def _validate_suite(
    *,
    suite: str,
    value: Any,
    lock: Mapping[str, Any],
    gate: Mapping[str, Any],
    expected_reader_revision: str,
    expected_cache_build_commit: str,
    errors: list[str],
) -> dict[str, int]:
    counts = {"state_count": 0, "feature_comparison_count": 0, "feature_pass_count": 0}
    try:
        report = _mapping(value, field=f"suites.{suite}")
        if report.get("suite") != suite or report.get("passed") is not True:
            errors.append(f"{suite} identity/pass flag is invalid")
        if report.get("expected_lock") != lock:
            errors.append(f"{suite} does not reproduce its complete cache lock")
        observed = _mapping(report.get("observed"), field=f"{suite}.observed")
        for field in (
            "manifest_sha256",
            "transitions_sha256",
            "build_report_sha256",
            "artifact_file_sha_map_sha256",
            "tensor_content_sha_map_sha256",
            "state_count",
            "transition_count",
            "artifact_tensor_count",
        ):
            if observed.get(field) != lock.get(field):
                errors.append(f"{suite}.{field} differs from preregistration")
        if observed.get("cache_build_commit") != expected_cache_build_commit:
            errors.append(f"{suite} cache-build commit differs from preregistration")
        if observed.get("reader_revision") != expected_reader_revision:
            errors.append(f"{suite} Reader revision differs from preregistration")
        if observed.get("feature_comparison_count") != lock.get("state_count"):
            errors.append(f"{suite} lacks one feature comparison per state")
        if observed.get("feature_pass_count") != lock.get("state_count"):
            errors.append(f"{suite} does not pass every state feature comparison")

        maps = _mapping(report.get("maps"), field=f"{suite}.maps")
        file_map = _artifact_sha_map(maps.get("artifact_file_sha256"), field=f"{suite}.maps.artifact_file_sha256")
        content_map = _artifact_sha_map(maps.get("tensor_content_sha256"), field=f"{suite}.maps.tensor_content_sha256")
        if set(file_map) != set(content_map):
            errors.append(f"{suite} file/content maps enumerate different artifacts")
        if len(file_map) != lock.get("artifact_tensor_count"):
            errors.append(f"{suite} full artifact maps have the wrong size")
        if canonical_json_sha256(file_map) != lock.get("artifact_file_sha_map_sha256"):
            errors.append(f"{suite} full artifact file map differs from preregistration")
        if canonical_json_sha256(content_map) != lock.get("tensor_content_sha_map_sha256"):
            errors.append(f"{suite} full tensor-content map differs from preregistration")

        states = _list(report.get("states"), field=f"{suite}.states")
        state_ids = [state.get("state_id") if isinstance(state, Mapping) else None for state in states]
        if (
            len(states) != lock.get("state_count")
            or state_ids != sorted(state_ids)
            or len(state_ids) != len(set(state_ids))
        ):
            errors.append(f"{suite} state evidence is incomplete, duplicated, or unsorted")
        for index, state_value in enumerate(states):
            label = f"{suite}.states[{index}]"
            try:
                state = _mapping(state_value, field=label)
                for identity in ("state_id", "teacher_key", "semantic_state_sha256"):
                    _sha256(state.get(identity), field=f"{label}.{identity}")
                if state.get("passed") is not True:
                    errors.append(f"{label} did not pass")
                teacher_key = state["teacher_key"]
                image = _mapping(state.get("image"), field=f"{label}.image")
                cached = _mapping(state.get("cached_feature"), field=f"{label}.cached_feature")
                for name, artifact, expected_name in (
                    ("image", image, "image.pt"),
                    ("cached_feature", cached, "feature.pt"),
                ):
                    expected_path = f"artifacts/{teacher_key}/{expected_name}"
                    if artifact.get("relative_path") != expected_path:
                        errors.append(f"{label}.{name} path is not bound to teacher_key")
                    file_sha = _sha256(artifact.get("file_sha256"), field=f"{label}.{name}.file_sha256")
                    if file_map.get(expected_path) != file_sha:
                        errors.append(f"{label}.{name} is not bound to the complete artifact file map")
                    content_sha = _sha256(
                        artifact.get("tensor_content_sha256"), field=f"{label}.{name}.tensor_content_sha256"
                    )
                    if artifact.get("manifest_tensor_content_sha256") != content_sha:
                        errors.append(f"{label}.{name} content SHA differs from manifest")
                    if content_map.get(expected_path) != content_sha:
                        errors.append(f"{label}.{name} is not bound to the complete tensor-content map")
                    if artifact.get("dtype") != "bfloat16":
                        errors.append(f"{label}.{name} is not BF16")
                    shape = artifact.get("shape")
                    if not isinstance(shape, list) or not shape or shape[0] != 1:
                        errors.append(f"{label}.{name} shape is invalid")
                if image.get("shape") != [1, 3, 1024, 1024]:
                    errors.append(f"{label}.image shape differs from the decoded teacher lock")
                _canonical_tensor(
                    image.get("canonical_unbatched"),
                    field=f"{label}.image.canonical_unbatched",
                    dtype="bfloat16",
                    shape=[3, 1024, 1024],
                )

                preprocessing = _mapping(
                    state.get("replacement_preprocessing"), field=f"{label}.replacement_preprocessing"
                )
                if preprocessing.get("reader_resize_contract") != R3_QWEN_READER_RESIZE_CONTRACT:
                    errors.append(f"{label} preprocessing has the wrong resize contract")
                _canonical_tensor(
                    preprocessing.get("pixel_values"),
                    field=f"{label}.pixel_values",
                    dtype="float32",
                    shape=list(R3_QWEN_READER_PIXEL_VALUES_SHAPE),
                )
                _canonical_tensor(
                    preprocessing.get("image_grid_thw"),
                    field=f"{label}.image_grid_thw",
                    dtype="int64",
                    shape=[1, 3],
                )
                if preprocessing.get("grid_values") != list(R3_QWEN_READER_GRID_THW):
                    errors.append(f"{label} preprocessing grid differs from the lock")
                preprocessing_checks = _mapping(preprocessing.get("checks"), field=f"{label}.preprocessing.checks")
                if set(preprocessing_checks) != _PREPROCESSING_CHECKS or any(
                    item is not True for item in preprocessing_checks.values()
                ):
                    errors.append(f"{label} replacement preprocessing did not pass")
                cached_shape = cached.get("shape")
                if isinstance(cached_shape, list):
                    _validate_feature_comparison(
                        state.get("feature_comparison"),
                        cached_content_sha256=str(cached.get("tensor_content_sha256")),
                        cached_shape=cached_shape,
                        gate=gate,
                        label=f"{label}.feature_comparison",
                        errors=errors,
                    )
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                errors.append(f"{label} is invalid: {exc}")

        read_only = _mapping(report.get("read_only_integrity"), field=f"{suite}.read_only_integrity")
        expected_core = {
            "manifest.json": lock.get("manifest_sha256"),
            "transitions.jsonl": lock.get("transitions_sha256"),
            "build_report.json": lock.get("build_report_sha256"),
        }
        if (
            read_only.get("core_file_sha256_before") != expected_core
            or read_only.get("core_file_sha256_after") != expected_core
            or read_only.get("artifact_file_sha_map_sha256_before") != lock.get("artifact_file_sha_map_sha256")
            or read_only.get("artifact_file_sha_map_sha256_after") != lock.get("artifact_file_sha_map_sha256")
            or read_only.get("unchanged") is not True
        ):
            errors.append(f"{suite} does not prove immutable cache access")
        counts = {
            "state_count": len(states),
            "feature_comparison_count": len(states),
            "feature_pass_count": sum(
                int(isinstance(state, Mapping) and state.get("passed") is True) for state in states
            ),
        }
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"{suite} report is invalid: {exc}")
    return counts


def validate_tf0_report(
    report: Mapping[str, Any],
    *,
    locks: Mapping[str, Any],
    expected_commit: str,
    tc0_validation_sha256: str,
) -> dict[str, Any]:
    _commit(expected_commit, field="expected_commit")
    _sha256(tc0_validation_sha256, field="tc0_validation_sha256")
    errors: list[str] = []
    gate = _mapping(locks.get("feature_backend_gate"), field="locks.feature_backend_gate")
    if report.get("schema_version") != tf0.PROBE_SCHEMA_VERSION or report.get("probe") != tf0.PROBE_NAME:
        errors.append("raw R3-TF0 probe identity is invalid")
    if report.get("passed") is not True or "error" in report:
        errors.append("raw R3-TF0 probe did not pass fail-closed")
    if report.get("reader_revision") != locks.get("reader_revision"):
        errors.append("raw R3-TF0 Reader revision differs from preregistration")
    if report.get("reader_resize_contract") != R3_QWEN_READER_RESIZE_CONTRACT:
        errors.append("raw R3-TF0 resize contract is invalid")
    if report.get("feature_gate") != gate:
        errors.append("raw R3-TF0 feature gate differs from preregistration")
    if report.get("feature_gate_sha256") != canonical_json_sha256(gate):
        errors.append("raw R3-TF0 feature gate SHA differs from preregistration")
    if report.get("preregistration") != locks:
        errors.append("raw R3-TF0 does not bind the complete preregistration")

    tc0_prerequisite = report.get("tc0_prerequisite")
    if (
        not isinstance(tc0_prerequisite, Mapping)
        or tc0_prerequisite.get("sha256") != tc0_validation_sha256
        or tc0_prerequisite.get("protocol") != tf0.TC0_VALIDATION_PROTOCOL
        or tc0_prerequisite.get("expected_commit") != expected_commit
        or tc0_prerequisite.get("passed") is not True
    ):
        errors.append("raw R3-TF0 is not bound to the passing same-commit TC0 validation")
    provenance = report.get("provenance")
    git = provenance.get("git") if isinstance(provenance, Mapping) else None
    if not isinstance(git, Mapping) or git.get("commit") != expected_commit or git.get("clean") is not True:
        errors.append("raw R3-TF0 provenance is not the expected clean commit")
    models = provenance.get("models") if isinstance(provenance, Mapping) else None
    model = models.get("reader") if isinstance(models, Mapping) else None
    if (
        not isinstance(model, Mapping)
        or model.get("observed_revision") != locks.get("reader_revision")
        or model.get("expected_revision") != locks.get("reader_revision")
        or model.get("revision_matches_lock") is not True
    ):
        errors.append("raw R3-TF0 Reader model provenance is invalid")

    processor = report.get("processor")
    processor_checks = processor.get("checks") if isinstance(processor, Mapping) else None
    if (
        not isinstance(processor, Mapping)
        or processor.get("passed") is not True
        or not isinstance(processor_checks, Mapping)
        or set(processor_checks) != _PROCESSOR_CHECKS
        or any(item is not True for item in processor_checks.values())
    ):
        errors.append("raw R3-TF0 fast processor audit did not pass")
    attention = report.get("attention_backend")
    if (
        not isinstance(attention, Mapping)
        or attention.get("requested_implementation") != "sdpa"
        or attention.get("model_config_implementation") != "sdpa"
        or attention.get("sdpa") != {"flash": False, "memory_efficient": False, "cudnn": False, "math": True}
    ):
        errors.append("raw R3-TF0 Reader attention backend is not strict math-only SDPA")
    determinism = report.get("determinism")
    if (
        not isinstance(determinism, Mapping)
        or determinism.get("deterministic_algorithms") is not True
        or determinism.get("deterministic_warn_only") is not False
        or determinism.get("environment") != dict(sorted(REQUIRED_DETERMINISM_ENV.items()))
        or determinism.get("sdpa") != {"flash": False, "memory_efficient": False, "cudnn": False, "math": True}
    ):
        errors.append("raw R3-TF0 strict determinism evidence is invalid")
    frozen = report.get("reader_frozen")
    if (
        not isinstance(frozen, Mapping)
        or frozen.get("passed") is not True
        or frozen.get("training") is not False
        or not isinstance(frozen.get("parameter_tensors"), int)
        or frozen.get("parameter_tensors", 0) <= 0
        or frozen.get("trainable_parameter_tensors") != 0
        or frozen.get("parameter_tensors_with_grad") != 0
    ):
        errors.append("raw R3-TF0 does not prove a frozen eval-mode Reader")

    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("device_type") != "cuda":
        errors.append("raw R3-TF0 did not run on CUDA")
    else:
        if str(locks.get("gpu_model")) not in str(runtime.get("device_name")):
            errors.append("raw R3-TF0 did not run on the preregistered GPU")
        if runtime.get("device_capability") != [9, 0]:
            errors.append("raw R3-TF0 GPU capability is not Hopper 9.0")
        for runtime_field, lock_field in (
            ("torch", "torch_version"),
            ("cuda", "cuda_version"),
            ("transformers", "transformers_version"),
            ("torchvision", "torchvision_version"),
        ):
            if runtime.get(runtime_field) != locks.get(lock_field):
                errors.append(f"raw R3-TF0 {runtime_field} version differs from preregistration")
        for memory_field in ("peak_allocated_gib", "peak_reserved_gib"):
            value = runtime.get(memory_field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                errors.append(f"raw R3-TF0 {memory_field} is invalid")

    suite_locks = locks.get("suites")
    suites = report.get("suites")
    aggregate = {"state_count": 0, "feature_comparison_count": 0, "feature_pass_count": 0}
    if not isinstance(suite_locks, Mapping) or set(suite_locks) != {"set8", "transition16"}:
        raise ValueError("R3-TF0 locks must contain exactly Set8 and Transition16.")
    if not isinstance(suites, Mapping) or set(suites) != {"set8", "transition16"}:
        errors.append("raw R3-TF0 must contain exactly Set8 and Transition16 suites")
    else:
        for suite in ("set8", "transition16"):
            counts = _validate_suite(
                suite=suite,
                value=suites[suite],
                lock=_mapping(suite_locks[suite], field=f"locks.suites.{suite}"),
                gate=gate,
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
        "feature_comparison_count": aggregate["feature_comparison_count"],
        "feature_pass_count": aggregate["feature_pass_count"],
        "cache_mutation_count": 0,
    }
    if report.get("summary") != expected_summary:
        errors.append("raw R3-TF0 summary is inconsistent with complete per-state evidence")
    expected_unlocks = {
        "teacher_t0": True,
        "teacher_calibration": True,
        "teacher_assisted_training": True,
        "qa_only_dependency": False,
    }
    if report.get("unlocks") != expected_unlocks:
        errors.append("raw R3-TF0 does not have the exact preregistered unlock scope")

    passed = not errors
    return {
        "schema_version": 1,
        "protocol": VALIDATION_PROTOCOL,
        "expected_commit": expected_commit,
        "reader_revision": locks.get("reader_revision"),
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "feature_gate_sha256": canonical_json_sha256(gate),
        "preregistration_sha256": locks.get("preregistration_sha256"),
        "tc0_validation_sha256": tc0_validation_sha256,
        "validated_suites": ["set8", "transition16"],
        "validated_state_count": aggregate["state_count"],
        "validated_feature_comparison_count": aggregate["feature_comparison_count"],
        "validated_feature_pass_count": aggregate["feature_pass_count"],
        "teacher_t0_unlocked": passed,
        "teacher_calibration_unlocked": passed,
        "teacher_assisted_training_unlocked": passed,
        "qa_only_dependency": False,
        "errors": errors,
        "passed": passed,
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
    parser = argparse.ArgumentParser(description="Fail-closed validator for raw R3-TF0 evidence")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--tc0-validation-report", type=Path, required=True)
    parser.add_argument("--tc0-validation-report-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _sha256(args.report_sha256, field="report_sha256")
        _sha256(args.tc0_validation_report_sha256, field="tc0_validation_report_sha256")
        _sha256(args.preregistration_sha256, field="preregistration_sha256")
        _commit(args.expected_commit, field="expected_commit")
        report_path = args.report.expanduser().resolve(strict=True)
        tc0_path = args.tc0_validation_report.expanduser().resolve(strict=True)
        preregistration_path = args.preregistration.expanduser().resolve(strict=True)
        if file_sha256(report_path) != args.report_sha256:
            raise ValueError("Raw R3-TF0 report SHA256 mismatch.")
        if file_sha256(tc0_path) != args.tc0_validation_report_sha256:
            raise ValueError("R3-TC0 validation report SHA256 mismatch.")
        if file_sha256(preregistration_path) != args.preregistration_sha256:
            raise ValueError("R3 preregistration SHA256 mismatch.")
        locks = tf0.load_preregistered_feature_lock(preregistration_path)
        if locks["preregistration_sha256"] != args.preregistration_sha256:
            raise ValueError("Loaded R3 preregistration differs from its supplied SHA256 lock.")
        tf0.validate_tc0_prerequisite(
            tc0_path,
            expected_file_sha256=args.tc0_validation_report_sha256,
            expected_commit=args.expected_commit,
            locks=locks,
        )
        result = validate_tf0_report(
            _json_object(report_path),
            locks=locks,
            expected_commit=args.expected_commit,
            tc0_validation_sha256=args.tc0_validation_report_sha256,
        )
    except Exception as exc:  # noqa: BLE001 - validator always writes a fail-closed artifact
        result = {
            "schema_version": 1,
            "protocol": VALIDATION_PROTOCOL,
            "teacher_t0_unlocked": False,
            "teacher_calibration_unlocked": False,
            "teacher_assisted_training_unlocked": False,
            "qa_only_dependency": False,
            "errors": [str(exc)],
            "passed": False,
        }
    result["inputs"] = {
        "report": str(args.report),
        "report_sha256": args.report_sha256,
        "tc0_validation_report": str(args.tc0_validation_report),
        "tc0_validation_report_sha256": args.tc0_validation_report_sha256,
        "preregistration": str(args.preregistration),
        "preregistration_sha256": args.preregistration_sha256,
        "expected_commit": args.expected_commit,
    }
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
