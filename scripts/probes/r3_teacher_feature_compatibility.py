from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

import r3_teacher_cache_compatibility as tc0  # noqa: E402
from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_GRID_THW,
    R3_QWEN_READER_PIXEL_VALUES_SHAPE,
    R3_QWEN_READER_RESIZE_CONTRACT,
    qwen3vl_query_free_visual_features,
)
from vision_memory.repro import (  # noqa: E402
    canonical_json_sha256,
    canonical_tensor_manifest,
    configure_strict_cuda_determinism,
    emit_json_report,
    probe_provenance,
)
from vision_memory.teacher import (  # noqa: E402
    file_sha256,
    load_teacher_cache_manifest,
    load_teacher_tensor,
    load_teacher_transition_sidecar,
    tensor_sha256,
)


PROBE_NAME = "r3_tf0_teacher_feature_backend_compatibility"
PROBE_SCHEMA_VERSION = 1
TC0_VALIDATION_PROTOCOL = "R3-TC0-cache-forward-compatibility-validation.v1"
FEATURE_GATE_FIELDS = {
    "gate",
    "cached_feature_source_backend",
    "replacement_backend",
    "feature_boundary",
    "relative_l2_max",
    "cosine_min",
    "all_states_must_pass",
    "state_count",
    "cache_access",
}
LOCKED_FEATURE_GATE = {
    "gate": "R3-TF0",
    "cached_feature_source_backend": "default_sdpa",
    "replacement_backend": "strict_math_only",
    "feature_boundary": "qwen3vl-post-merger-query-free.v1",
    "relative_l2_max": 0.01,
    "cosine_min": 0.9999,
    "all_states_must_pass": True,
    "state_count": 30,
    "cache_access": "read-only",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "R3-TF0: recompute every cached query-free Qwen merger feature under the replacement "
            "strict math-only backend and compare it with the immutable default-SDPA cache."
        )
    )
    parser.add_argument("--set8-cache", type=Path, required=True)
    parser.add_argument("--transition16-cache", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--tc0-validation-report", type=Path, required=True)
    parser.add_argument("--tc0-validation-report-sha256", required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT / "configs" / "experiments" / "r3_preregistration.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _json_object(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=True)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest.")
    return value


def _commit(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase full Git commit.")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object.")
    return value


def load_preregistered_feature_lock(path: Path) -> dict[str, Any]:
    locks = tc0.load_preregistered_cache_locks(path)
    payload = _json_object(path)
    teacher_t0 = _mapping(payload.get("teacher_t0"), field="teacher_t0")
    feature_gate = _mapping(teacher_t0.get("feature_backend_gate"), field="teacher_t0.feature_backend_gate")
    if set(feature_gate) != FEATURE_GATE_FIELDS or dict(feature_gate) != LOCKED_FEATURE_GATE:
        raise ValueError("R3-TF0 feature-backend gate differs from the prospective fixed contract.")
    prerequisites = teacher_t0.get("prerequisite_gates")
    if not isinstance(prerequisites, list) or prerequisites != [
        "R3-TC0 cache-forward pixel/grid compatibility",
        "R3-TF0 cached-feature strict-math backend compatibility",
    ]:
        raise ValueError("Teacher T0 prerequisite order must explicitly include TC0 then TF0.")
    return {
        **locks,
        "feature_backend_gate": dict(feature_gate),
        "feature_backend_gate_sha256": canonical_json_sha256(feature_gate),
    }


def validate_tc0_prerequisite(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_commit: str,
    locks: Mapping[str, Any],
) -> dict[str, Any]:
    _sha256(expected_file_sha256, field="tc0_validation_report_sha256")
    source = path.expanduser().resolve(strict=True)
    actual_sha256 = file_sha256(source)
    if actual_sha256 != expected_file_sha256:
        raise ValueError("R3-TC0 validation report SHA256 mismatch.")
    report = _json_object(source)
    required = {
        "protocol": TC0_VALIDATION_PROTOCOL,
        "expected_commit": expected_commit,
        "reader_revision": locks["reader_revision"],
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "preregistration_sha256": locks["preregistration_sha256"],
        "validated_suites": ["set8", "transition16"],
        "validated_state_count": 30,
        "validated_artifact_tensor_count": 90,
        "validated_image_forward_count": 30,
        "cache_forward_compatibility_complete": True,
        "feature_backend_compatibility_complete": False,
        "teacher_t0_unlocked": False,
        "teacher_calibration_unlocked": False,
        "teacher_assisted_training_unlocked": False,
        "errors": [],
        "passed": True,
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise ValueError(f"R3-TC0 prerequisite field {field!r} differs from the required value.")
    return {
        "path": str(source),
        "sha256": actual_sha256,
        "protocol": TC0_VALIDATION_PROTOCOL,
        "expected_commit": expected_commit,
        "passed": True,
    }


def feature_distance(
    cached: Tensor,
    replacement: Tensor,
    *,
    relative_l2_max: float,
    cosine_min: float,
) -> dict[str, Any]:
    """Compare complete flattened merger features using the cache as the L2 reference."""

    shape_equal = tuple(cached.shape) == tuple(replacement.shape)
    dtype_equal = cached.dtype == replacement.dtype
    finite = bool(torch.isfinite(cached).all().item() and torch.isfinite(replacement).all().item())
    relative_l2: float | None = None
    cosine: float | None = None
    maximum_absolute_difference: float | None = None
    reference_norm: float | None = None
    replacement_norm: float | None = None
    difference_norm: float | None = None
    if shape_equal and finite:
        cached_flat = cached.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        replacement_flat = replacement.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        reference_norm = float(torch.linalg.vector_norm(cached_flat).item())
        replacement_norm = float(torch.linalg.vector_norm(replacement_flat).item())
        difference = replacement_flat - cached_flat
        difference_norm = float(torch.linalg.vector_norm(difference).item())
        maximum_absolute_difference = float(difference.abs().max().item())
        if reference_norm > 0.0 and replacement_norm > 0.0:
            relative_l2 = difference_norm / reference_norm
            cosine = float(torch.dot(cached_flat, replacement_flat).item()) / (reference_norm * replacement_norm)
    metric_finite = bool(
        relative_l2 is not None and cosine is not None and math.isfinite(relative_l2) and math.isfinite(cosine)
    )
    checks = {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "finite": finite,
        "nonzero_norms": bool(reference_norm and replacement_norm),
        "metrics_finite": metric_finite,
        "relative_l2_within_gate": bool(metric_finite and relative_l2 <= relative_l2_max),
        "cosine_within_gate": bool(metric_finite and cosine >= cosine_min),
    }
    return {
        "passed": all(checks.values()),
        "reference": "immutable_cached_default_sdpa_feature",
        "replacement": "strict_math_only_query_free_feature",
        "thresholds": {
            "relative_l2_max": relative_l2_max,
            "cosine_min": cosine_min,
        },
        "cached": canonical_tensor_manifest(cached),
        "replacement_feature": canonical_tensor_manifest(replacement),
        "cached_teacher_tensor_sha256": tensor_sha256(cached),
        "replacement_teacher_tensor_sha256": tensor_sha256(replacement),
        "relative_l2": relative_l2,
        "cosine": cosine,
        "reference_norm": reference_norm,
        "replacement_norm": replacement_norm,
        "difference_norm": difference_norm,
        "maximum_absolute_difference": maximum_absolute_difference,
        "checks": checks,
    }


def _manifest_content_map(manifest: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for record in manifest.records:
        for specification in (record.image, record.latent, record.feature):
            if specification.relative_path in result:
                raise ValueError("Teacher cache manifest reuses an artifact path.")
            result[specification.relative_path] = specification.sha256
    return dict(sorted(result.items()))


def audit_feature_cache_suite(
    *,
    suite: str,
    cache_root: Path,
    expected_lock: Mapping[str, Any],
    expected_reader_revision: str,
    expected_cache_build_commit: str,
    feature_gate: Mapping[str, Any],
    model: Any,
    processor: Any,
    device: torch.device,
) -> dict[str, Any]:
    if suite not in tc0.EXPECTED_SUITES:
        raise ValueError(f"Unsupported R3-TF0 suite: {suite!r}.")
    root = cache_root.expanduser().resolve(strict=True)
    pre_core = tc0._core_file_hashes(root)
    expected_core = {
        "manifest.json": expected_lock["manifest_sha256"],
        "transitions.jsonl": expected_lock["transitions_sha256"],
        "build_report.json": expected_lock["build_report_sha256"],
    }
    if pre_core != expected_core:
        raise ValueError(f"{suite} core cache files differ from their preregistered SHA256 locks.")
    manifest = load_teacher_cache_manifest(
        root / "manifest.json",
        expected_file_sha256=str(expected_lock["manifest_sha256"]),
    )
    transitions = load_teacher_transition_sidecar(
        root / "transitions.jsonl",
        manifest=manifest,
        expected_file_sha256=str(expected_lock["transitions_sha256"]),
    )
    build_report = tc0._json_object(root / "build_report.json")
    if build_report.get("schema") != tc0.BUILD_REPORT_SCHEMA:
        raise ValueError(f"{suite} build report schema drifted.")
    if build_report.get("git_commit") != expected_cache_build_commit:
        raise ValueError(f"{suite} build report has the wrong historical cache-build commit.")
    revisions = _mapping(build_report.get("model_revisions"), field=f"{suite}.model_revisions")
    if revisions.get("qwen_reader") != expected_reader_revision:
        raise ValueError(f"{suite} build report has the wrong Reader revision.")
    if (
        build_report.get("manifest_sha256") != expected_lock["manifest_sha256"]
        or build_report.get("sidecar_sha256") != expected_lock["transitions_sha256"]
    ):
        raise ValueError(f"{suite} build report does not bind the locked manifest/sidecar.")
    if len(manifest.records) != expected_lock["state_count"] or len(transitions) != expected_lock["transition_count"]:
        raise ValueError(f"{suite} cache counts differ from preregistration.")

    build_file_map = tc0._validated_sha_map(
        build_report.get("artifact_file_sha256"),
        field=f"{suite}.build_report.artifact_file_sha256",
    )
    if (
        len(build_file_map) != expected_lock["artifact_tensor_count"]
        or canonical_json_sha256(build_file_map) != expected_lock["artifact_file_sha_map_sha256"]
    ):
        raise ValueError(f"{suite} build-report artifact file map differs from preregistration.")
    content_map = _manifest_content_map(manifest)
    if (
        len(content_map) != expected_lock["artifact_tensor_count"]
        or canonical_json_sha256(content_map) != expected_lock["tensor_content_sha_map_sha256"]
        or set(content_map) != set(build_file_map)
    ):
        raise ValueError(f"{suite} manifest tensor-content map differs from preregistration.")
    pre_artifacts = tc0._artifact_inventory(root)
    if pre_artifacts != build_file_map:
        raise ValueError(f"{suite} on-disk artifact files differ from the complete locked map.")

    state_reports: list[dict[str, Any]] = []
    for record in manifest.records:
        image_path = tc0._safe_cache_file(root, record.image.relative_path)
        feature_path = tc0._safe_cache_file(root, record.feature.relative_path)
        image = load_teacher_tensor(
            image_path,
            specification=record.image,
            expected_file_sha256=build_file_map[record.image.relative_path],
        )
        cached_feature = load_teacher_tensor(
            feature_path,
            specification=record.feature,
            expected_file_sha256=build_file_map[record.feature.relative_path],
        )
        if image.dtype != torch.bfloat16 or cached_feature.dtype != torch.bfloat16:
            raise ValueError(f"{suite} state {record.state_id} cache is not the locked BF16 teacher cache.")
        with torch.no_grad():
            replacement_output = qwen3vl_query_free_visual_features(
                model=model,
                processor=processor,
                image=image.to(device=device, dtype=torch.bfloat16),
                device=device,
                require_image_grad=False,
                reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
            )
        replacement = replacement_output.features.detach().cpu().contiguous()
        comparison = feature_distance(
            cached_feature,
            replacement,
            relative_l2_max=float(feature_gate["relative_l2_max"]),
            cosine_min=float(feature_gate["cosine_min"]),
        )
        pixels = replacement_output.pixel_values.detach()
        grid = replacement_output.image_grid_thw.detach()
        actual_grid = [int(value) for value in grid.cpu().reshape(-1).tolist()]
        preprocessing_checks = {
            "pixel_values_shape_locked": tuple(pixels.shape) == tuple(R3_QWEN_READER_PIXEL_VALUES_SHAPE),
            "pixel_values_dtype_locked": pixels.dtype == torch.float32,
            "pixel_values_finite": bool(torch.isfinite(pixels).all().item()),
            "grid_shape_locked": tuple(grid.shape) == (1, 3),
            "grid_dtype_locked": grid.dtype == torch.int64,
            "grid_values_locked": actual_grid == list(R3_QWEN_READER_GRID_THW),
        }
        state_reports.append(
            {
                "state_id": record.state_id,
                "teacher_key": record.teacher_key,
                "semantic_state_sha256": record.semantic_state_sha256,
                "image": {
                    "relative_path": record.image.relative_path,
                    "file_sha256": build_file_map[record.image.relative_path],
                    "tensor_content_sha256": tensor_sha256(image),
                    "manifest_tensor_content_sha256": record.image.sha256,
                    "canonical_unbatched": canonical_tensor_manifest(image[0]),
                    "dtype": record.image.dtype,
                    "shape": list(record.image.shape),
                },
                "cached_feature": {
                    "relative_path": record.feature.relative_path,
                    "file_sha256": build_file_map[record.feature.relative_path],
                    "tensor_content_sha256": tensor_sha256(cached_feature),
                    "manifest_tensor_content_sha256": record.feature.sha256,
                    "dtype": record.feature.dtype,
                    "shape": list(record.feature.shape),
                },
                "replacement_preprocessing": {
                    "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
                    "pixel_values": canonical_tensor_manifest(pixels),
                    "image_grid_thw": canonical_tensor_manifest(grid),
                    "grid_values": actual_grid,
                    "checks": preprocessing_checks,
                },
                "feature_comparison": comparison,
                "passed": bool(all(preprocessing_checks.values()) and comparison["passed"]),
            }
        )

    post_core = tc0._core_file_hashes(root)
    post_artifacts = tc0._artifact_inventory(root)
    read_only = pre_core == post_core and pre_artifacts == post_artifacts
    return {
        "suite": suite,
        "cache_root": str(root),
        "expected_lock": dict(expected_lock),
        "observed": {
            "manifest_sha256": pre_core["manifest.json"],
            "transitions_sha256": pre_core["transitions.jsonl"],
            "build_report_sha256": pre_core["build_report.json"],
            "cache_build_commit": build_report["git_commit"],
            "reader_revision": revisions["qwen_reader"],
            "state_count": len(manifest.records),
            "transition_count": len(transitions),
            "artifact_tensor_count": len(build_file_map),
            "feature_comparison_count": len(state_reports),
            "feature_pass_count": sum(int(state["passed"]) for state in state_reports),
            "artifact_file_sha_map_sha256": canonical_json_sha256(pre_artifacts),
            "tensor_content_sha_map_sha256": canonical_json_sha256(content_map),
        },
        "maps": {
            "artifact_file_sha256": pre_artifacts,
            "tensor_content_sha256": content_map,
        },
        "states": state_reports,
        "read_only_integrity": {
            "core_file_sha256_before": pre_core,
            "core_file_sha256_after": post_core,
            "artifact_file_sha_map_sha256_before": canonical_json_sha256(pre_artifacts),
            "artifact_file_sha_map_sha256_after": canonical_json_sha256(post_artifacts),
            "unchanged": read_only,
        },
        "passed": bool(state_reports and all(state["passed"] for state in state_reports) and read_only),
    }


def frozen_reader_evidence(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    with_grad = [parameter for parameter in parameters if parameter.grad is not None]
    return {
        "training": bool(model.training),
        "parameter_tensors": len(parameters),
        "trainable_parameter_tensors": len(trainable),
        "parameter_tensors_with_grad": len(with_grad),
        "passed": bool(parameters and not model.training and not trainable and not with_grad),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-TF0 requires CUDA on the locked Inspire H200 runtime.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("R3-TF0 formal evidence must run on CUDA.")
    tc0._assert_output_outside_caches(args.output_json, (args.set8_cache, args.transition16_cache))
    if args.set8_cache.expanduser().resolve(strict=True) == args.transition16_cache.expanduser().resolve(strict=True):
        raise ValueError("Set8 and Transition16 must reference distinct immutable caches.")

    locks = load_preregistered_feature_lock(args.preregistration)
    provenance = probe_provenance(root=ROOT, arguments=args, models={"reader": args.reader})
    git = provenance.get("git", {})
    commit = _commit(git.get("commit"), field="provenance.git.commit")
    if git.get("clean") is not True:
        raise RuntimeError("R3-TF0 requires an exact clean Git commit.")
    reader_revision = tc0.locked_revision(args.reader)
    if reader_revision != locks["reader_revision"]:
        raise ValueError("Reader snapshot revision differs from preregistration.")
    model_provenance = provenance.get("models", {}).get("reader", {})
    if (
        model_provenance.get("observed_revision") != reader_revision
        or model_provenance.get("expected_revision") != reader_revision
        or model_provenance.get("revision_matches_lock") is not True
    ):
        raise ValueError("Reader model provenance is not bound to the locked revision.")
    tc0_prerequisite = validate_tc0_prerequisite(
        args.tc0_validation_report,
        expected_file_sha256=args.tc0_validation_report_sha256,
        expected_commit=commit,
        locks=locks,
    )
    determinism = configure_strict_cuda_determinism(seed=0)

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=tc0.EXPECTED_MIN_MAX_PIXELS,
        max_pixels=tc0.EXPECTED_MIN_MAX_PIXELS,
    )
    processor_audit = tc0.audit_fast_processor(processor.image_processor)
    if not processor_audit["passed"]:
        raise RuntimeError("Real Qwen fast image processor differs from the locked R3 geometry.")
    reader = Qwen3VLForConditionalGeneration.from_pretrained(
        args.reader,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    reader.requires_grad_(False)
    reader.eval()
    attention = {
        "requested_implementation": "sdpa",
        "model_config_implementation": getattr(reader.config, "_attn_implementation", None),
        "sdpa": determinism["sdpa"],
    }
    if attention["model_config_implementation"] != "sdpa" or attention["sdpa"] != {
        "flash": False,
        "memory_efficient": False,
        "cudnn": False,
        "math": True,
    }:
        raise RuntimeError("R3-TF0 failed to establish the strict math-only SDPA Reader backend.")

    torch.cuda.reset_peak_memory_stats(device)
    suites = {
        "set8": audit_feature_cache_suite(
            suite="set8",
            cache_root=args.set8_cache,
            expected_lock=locks["suites"]["set8"],
            expected_reader_revision=reader_revision,
            expected_cache_build_commit=locks["cache_build_commit"],
            feature_gate=locks["feature_backend_gate"],
            model=reader,
            processor=processor,
            device=device,
        ),
        "transition16": audit_feature_cache_suite(
            suite="transition16",
            cache_root=args.transition16_cache,
            expected_lock=locks["suites"]["transition16"],
            expected_reader_revision=reader_revision,
            expected_cache_build_commit=locks["cache_build_commit"],
            feature_gate=locks["feature_backend_gate"],
            model=reader,
            processor=processor,
            device=device,
        ),
    }
    torch.cuda.synchronize(device)
    reader_frozen = frozen_reader_evidence(reader)
    summary = {
        "suite_count": 2,
        "suite_pass_count": sum(int(suite["passed"]) for suite in suites.values()),
        "state_count": sum(suite["observed"]["state_count"] for suite in suites.values()),
        "feature_comparison_count": sum(suite["observed"]["feature_comparison_count"] for suite in suites.values()),
        "feature_pass_count": sum(suite["observed"]["feature_pass_count"] for suite in suites.values()),
        "cache_mutation_count": sum(int(not suite["read_only_integrity"]["unchanged"]) for suite in suites.values()),
    }
    runtime = {
        "device": str(device),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "torchvision": importlib.metadata.version("torchvision"),
    }
    passed = bool(
        all(suite["passed"] for suite in suites.values())
        and summary
        == {
            "suite_count": 2,
            "suite_pass_count": 2,
            "state_count": 30,
            "feature_comparison_count": 30,
            "feature_pass_count": 30,
            "cache_mutation_count": 0,
        }
        and reader_frozen["passed"]
        and locks["gpu_model"] in runtime["device_name"]
        and runtime["device_capability"] == [9, 0]
        and runtime["torch"] == locks["torch_version"]
        and runtime["cuda"] == locks["cuda_version"]
        and runtime["transformers"] == locks["transformers_version"]
        and runtime["torchvision"] == locks["torchvision_version"]
    )
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe": PROBE_NAME,
        "reader_revision": reader_revision,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "feature_gate": locks["feature_backend_gate"],
        "feature_gate_sha256": locks["feature_backend_gate_sha256"],
        "preregistration": locks,
        "tc0_prerequisite": tc0_prerequisite,
        "processor": processor_audit,
        "attention_backend": attention,
        "determinism": determinism,
        "reader_frozen": reader_frozen,
        "runtime": runtime,
        "suites": suites,
        "summary": summary,
        "unlocks": {
            "teacher_t0": passed,
            "teacher_calibration": passed,
            "teacher_assisted_training": passed,
            "qa_only_dependency": False,
        },
        "provenance": provenance,
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        tc0._assert_output_outside_caches(args.output_json, (args.set8_cache, args.transition16_cache))
    except Exception as exc:  # noqa: BLE001 - never write even failure evidence into an immutable cache
        report = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "probe": PROBE_NAME,
            "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
            "unlocks": {
                "teacher_t0": False,
                "teacher_calibration": False,
                "teacher_assisted_training": False,
                "qa_only_dependency": False,
            },
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "passed": False,
        }
        emit_json_report(report, None)
        return 1
    try:
        report = run_probe(args)
    except Exception as exc:  # noqa: BLE001 - formal probe always emits fail-closed JSON
        report = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "probe": PROBE_NAME,
            "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
            "unlocks": {
                "teacher_t0": False,
                "teacher_calibration": False,
                "teacher_assisted_training": False,
                "qa_only_dependency": False,
            },
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "passed": False,
        }
    emit_json_report(report, args.output_json)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
