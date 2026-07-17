from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_GRID_THW,
    R3_QWEN_READER_PIXEL_VALUES_SHAPE,
    R3_QWEN_READER_RESIZE_CONTRACT,
    deterministic_qwen_reader_resize,
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


PROBE_NAME = "r3_tc0_teacher_cache_forward_compatibility"
PROBE_SCHEMA_VERSION = 1
PREREGISTRATION_SCHEMA = "vision_memory.r3-preregistration.v1"
BUILD_REPORT_SCHEMA = "vision_memory.r3-teacher-cache-build-report.v1"
EXPECTED_MIN_MAX_PIXELS = 256 * 256
EXPECTED_PATCH_SIZE = 16
EXPECTED_TEMPORAL_PATCH_SIZE = 2
EXPECTED_MERGE_SIZE = 2
EXPECTED_BICUBIC_RESAMPLE_VALUE = 3
EXPECTED_SUITES = {
    "set8": {"state_count": 10, "transition_count": 8, "artifact_tensor_count": 30},
    "transition16": {"state_count": 20, "transition_count": 28, "artifact_tensor_count": 60},
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "R3-TC0: verify the immutable Set8/Transition16 teacher caches and prove that every "
            "cached decoded image produces bitwise-identical Qwen pixels/grid through the legacy "
            "default resize and the deterministic explicit-resize path."
        )
    )
    parser.add_argument("--set8-cache", type=Path, required=True)
    parser.add_argument("--transition16-cache", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=ROOT / "configs" / "experiments" / "r3_preregistration.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest.")
    return value


def _require_commit(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase full Git commit.")
    return value


def _require_count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _require_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string.")
    return value


def _json_object(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Expected a valid JSON object in {source}.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return value


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object.")
    return value


def locked_revision(path: Path) -> str:
    marker = path.expanduser().resolve(strict=True) / ".locked_revision"
    if not marker.is_file() or marker.is_symlink():
        raise ValueError(f"Reader snapshot lacks a regular .locked_revision file: {marker}")
    revision = marker.read_text(encoding="utf-8").strip()
    return _require_commit(revision, field="reader .locked_revision")


def load_preregistered_cache_locks(path: Path) -> dict[str, Any]:
    """Load only the prospective TC0 inputs from the immutable R3 preregistration."""

    source = path.expanduser().resolve(strict=True)
    payload = _json_object(source)
    if payload.get("schema") != PREREGISTRATION_SCHEMA:
        raise ValueError("R3-TC0 requires the locked R3 preregistration schema.")
    models = _mapping(payload.get("models"), field="models")
    reader = _mapping(models.get("reader"), field="models.reader")
    reader_revision = _require_commit(reader.get("revision"), field="models.reader.revision")
    teacher_contract = _mapping(payload.get("teacher_contract"), field="teacher_contract")
    cache_build_commit = _require_commit(
        teacher_contract.get("cache_build_commit"), field="teacher_contract.cache_build_commit"
    )
    manifest_locks = _mapping(
        teacher_contract.get("cache_manifest_sha256"), field="teacher_contract.cache_manifest_sha256"
    )
    build_locks = _mapping(teacher_contract.get("cache_builds"), field="teacher_contract.cache_builds")
    if set(manifest_locks) != set(EXPECTED_SUITES) or set(build_locks) != set(EXPECTED_SUITES):
        raise ValueError("R3-TC0 preregistration must lock exactly Set8 and Transition16 caches.")

    teacher_t0 = _mapping(payload.get("teacher_t0"), field="teacher_t0")
    resize_contract = teacher_t0.get("reader_resize_contract")
    if resize_contract != R3_QWEN_READER_RESIZE_CONTRACT:
        raise ValueError("R3-TC0 preregistration has a different Reader resize contract.")

    suites: dict[str, dict[str, Any]] = {}
    for suite, fixed_counts in EXPECTED_SUITES.items():
        lock = _mapping(build_locks[suite], field=f"teacher_contract.cache_builds.{suite}")
        observed_counts = {field: _require_count(lock.get(field), field=f"{suite}.{field}") for field in fixed_counts}
        if observed_counts != fixed_counts:
            raise ValueError(f"R3-TC0 {suite} counts differ from the fixed micro-cache contract.")
        suites[suite] = {
            **observed_counts,
            "manifest_sha256": _require_sha256(manifest_locks[suite], field=f"{suite}.manifest_sha256"),
            "transitions_sha256": _require_sha256(lock.get("transitions_sha256"), field=f"{suite}.transitions_sha256"),
            "build_report_sha256": _require_sha256(
                lock.get("build_report_sha256"), field=f"{suite}.build_report_sha256"
            ),
            "artifact_file_sha_map_sha256": _require_sha256(
                lock.get("artifact_file_sha_map_sha256"),
                field=f"{suite}.artifact_file_sha_map_sha256",
            ),
            "tensor_content_sha_map_sha256": _require_sha256(
                lock.get("tensor_content_sha_map_sha256"),
                field=f"{suite}.tensor_content_sha_map_sha256",
            ),
        }

    cluster = _mapping(payload.get("cluster"), field="cluster")
    r3_r0 = _mapping(
        _mapping(payload.get("technical_gates"), field="technical_gates").get("r3_r0"),
        field="technical_gates.r3_r0",
    )
    runtime = _mapping(r3_r0.get("runtime"), field="technical_gates.r3_r0.runtime")
    return {
        "preregistration_path": str(source),
        "preregistration_sha256": file_sha256(source),
        "preregistration_schema": PREREGISTRATION_SCHEMA,
        "reader_revision": reader_revision,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "cache_build_commit": cache_build_commit,
        "gpu_model": _require_text(cluster.get("gpu_model"), field="cluster.gpu_model"),
        "transformers_version": _require_text(
            runtime.get("transformers"), field="technical_gates.r3_r0.runtime.transformers"
        ),
        "torchvision_version": _require_text(
            runtime.get("torchvision"), field="technical_gates.r3_r0.runtime.torchvision"
        ),
        "torch_version": _require_text(runtime.get("torch"), field="technical_gates.r3_r0.runtime.torch"),
        "cuda_version": _require_text(runtime.get("cuda"), field="technical_gates.r3_r0.runtime.cuda"),
        "suites": suites,
    }


def _size_value(size: Any, key: str) -> int | None:
    value = size.get(key) if isinstance(size, Mapping) else getattr(size, key, None)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _resample_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def audit_fast_processor(image_processor: Any) -> dict[str, Any]:
    name = type(image_processor).__name__
    size = getattr(image_processor, "size", None)
    observed = {
        "class": name,
        "do_resize": getattr(image_processor, "do_resize", None),
        "min_pixels": getattr(image_processor, "min_pixels", None),
        "max_pixels": getattr(image_processor, "max_pixels", None),
        "shortest_edge": _size_value(size, "shortest_edge"),
        "longest_edge": _size_value(size, "longest_edge"),
        "patch_size": getattr(image_processor, "patch_size", None),
        "temporal_patch_size": getattr(image_processor, "temporal_patch_size", None),
        "merge_size": getattr(image_processor, "merge_size", None),
        "resample": str(getattr(image_processor, "resample", None)),
        "resample_value": _resample_value(getattr(image_processor, "resample", None)),
    }
    checks = {
        "fast_tensor_processor": "Fast" in name,
        "resize_enabled_by_default": observed["do_resize"] is True,
        "min_pixels_locked": observed["min_pixels"] == EXPECTED_MIN_MAX_PIXELS,
        "max_pixels_locked": observed["max_pixels"] == EXPECTED_MIN_MAX_PIXELS,
        "patch_size_locked": observed["patch_size"] == EXPECTED_PATCH_SIZE,
        "temporal_patch_size_locked": observed["temporal_patch_size"] == EXPECTED_TEMPORAL_PATCH_SIZE,
        "merge_size_locked": observed["merge_size"] == EXPECTED_MERGE_SIZE,
        "bicubic_resample_locked": observed["resample_value"] == EXPECTED_BICUBIC_RESAMPLE_VALUE,
        "callable": callable(image_processor),
    }
    return {"passed": all(checks.values()), "observed": observed, "checks": checks}


def _processor_output(image_processor: Any, image: Tensor, *, do_resize: bool | None) -> tuple[Tensor, Tensor]:
    kwargs: dict[str, Any] = {
        "images": [image],
        "return_tensors": "pt",
        "do_rescale": False,
    }
    if do_resize is not None:
        kwargs["do_resize"] = do_resize
    batch = image_processor(**kwargs)
    pixel_values = batch.get("pixel_values") if isinstance(batch, Mapping) else getattr(batch, "pixel_values", None)
    grid = batch.get("image_grid_thw") if isinstance(batch, Mapping) else getattr(batch, "image_grid_thw", None)
    if not isinstance(pixel_values, Tensor) or not isinstance(grid, Tensor):
        raise TypeError("Qwen fast image processor must return tensor pixel_values and image_grid_thw.")
    return pixel_values, grid


def _grid_values(grid: Tensor) -> list[int]:
    return [int(value) for value in grid.detach().cpu().reshape(-1).tolist()]


def _tensor_evidence(tensor: Tensor) -> dict[str, Any]:
    return {
        **canonical_tensor_manifest(tensor),
        "device": str(tensor.device),
        "finite": bool(torch.isfinite(tensor).all().item()),
    }


def compare_forward_paths(*, image_processor: Any, image: Tensor) -> dict[str, Any]:
    """Compare old and repaired preprocessing without loading or running Qwen weights."""

    if tuple(image.shape) != (3, 1024, 1024) or not image.is_floating_point():
        raise ValueError("R3-TC0 requires one floating [3,1024,1024] cached teacher image.")
    with torch.no_grad():
        legacy_pixels, legacy_grid = _processor_output(image_processor, image, do_resize=None)
        resized = deterministic_qwen_reader_resize(
            image,
            contract=R3_QWEN_READER_RESIZE_CONTRACT,
        )
        candidate_pixels, candidate_grid = _processor_output(image_processor, resized, do_resize=False)

    pixel_equal = bool(torch.equal(legacy_pixels, candidate_pixels))
    grid_equal = bool(torch.equal(legacy_grid, candidate_grid))
    maximum_difference = float((legacy_pixels.float() - candidate_pixels.float()).abs().max().item())
    expected_grid = list(R3_QWEN_READER_GRID_THW)
    legacy_grid_values = _grid_values(legacy_grid)
    candidate_grid_values = _grid_values(candidate_grid)
    checks = {
        "pixel_values_torch_equal": pixel_equal,
        "pixel_values_max_absolute_difference_zero": maximum_difference == 0.0,
        "pixel_values_shape_locked": tuple(legacy_pixels.shape)
        == tuple(candidate_pixels.shape)
        == tuple(R3_QWEN_READER_PIXEL_VALUES_SHAPE),
        "pixel_values_dtype_locked": legacy_pixels.dtype == candidate_pixels.dtype == torch.float32,
        "pixel_values_finite": bool(
            torch.isfinite(legacy_pixels).all().item() and torch.isfinite(candidate_pixels).all().item()
        ),
        "grid_torch_equal": grid_equal,
        "grid_values_locked": legacy_grid_values == candidate_grid_values == expected_grid,
        "grid_shape_locked": tuple(legacy_grid.shape) == tuple(candidate_grid.shape) == (1, 3),
        "grid_dtype_locked": legacy_grid.dtype == candidate_grid.dtype == torch.int64,
    }
    return {
        "passed": all(checks.values()),
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "input": _tensor_evidence(image),
        "explicitly_resized": _tensor_evidence(resized),
        "legacy_pixel_values": _tensor_evidence(legacy_pixels),
        "candidate_pixel_values": _tensor_evidence(candidate_pixels),
        "legacy_grid": _tensor_evidence(legacy_grid),
        "candidate_grid": _tensor_evidence(candidate_grid),
        "legacy_grid_thw": legacy_grid_values,
        "candidate_grid_thw": candidate_grid_values,
        "expected_grid_thw": expected_grid,
        "expected_pixel_values_shape": list(R3_QWEN_READER_PIXEL_VALUES_SHAPE),
        "pixel_values_max_absolute_difference": maximum_difference,
        "checks": checks,
    }


def _safe_cache_file(root: Path, relative_path: str) -> Path:
    candidate = root / Path(relative_path)
    if candidate.is_symlink():
        raise ValueError(f"R3-TC0 refuses symlinked cache artifacts: {candidate}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Teacher artifact escapes the cache root: {relative_path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Teacher artifact is not a regular file: {resolved}")
    return resolved


def _core_file_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("manifest.json", "transitions.jsonl", "build_report.json"):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"R3-TC0 requires a regular immutable cache file: {path}")
        result[name] = file_sha256(path)
    return result


def _artifact_inventory(root: Path) -> dict[str, str]:
    artifacts_root = root / "artifacts"
    if artifacts_root.is_symlink() or not artifacts_root.is_dir():
        raise ValueError(f"R3-TC0 requires one regular artifacts directory: {artifacts_root}")
    result: dict[str, str] = {}
    for candidate in sorted(artifacts_root.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(f"R3-TC0 refuses symlinks inside the cache: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(root).as_posix()
            result[relative] = file_sha256(candidate)
    return result


def _validated_sha_map(value: Any, *, field: str) -> dict[str, str]:
    source = _mapping(value, field=field)
    result: dict[str, str] = {}
    for path, digest in source.items():
        if not isinstance(path, str) or not path.startswith("artifacts/") or not path.endswith(".pt"):
            raise ValueError(f"{field} contains an invalid artifact path: {path!r}.")
        result[path] = _require_sha256(digest, field=f"{field}[{path!r}]")
    return dict(sorted(result.items()))


def audit_cache_suite(
    *,
    suite: str,
    cache_root: Path,
    expected_lock: Mapping[str, Any],
    expected_reader_revision: str,
    expected_cache_build_commit: str,
    image_processor: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Read and hash every cache artifact, then audit every decoded image forward."""

    if suite not in EXPECTED_SUITES:
        raise ValueError(f"Unsupported R3-TC0 suite: {suite!r}.")
    root = cache_root.expanduser().resolve(strict=True)
    pre_core = _core_file_hashes(root)
    for name, lock_field in (
        ("manifest.json", "manifest_sha256"),
        ("transitions.jsonl", "transitions_sha256"),
        ("build_report.json", "build_report_sha256"),
    ):
        if pre_core[name] != expected_lock[lock_field]:
            raise ValueError(f"{suite} {name} differs from its preregistered SHA256.")

    manifest = load_teacher_cache_manifest(
        root / "manifest.json",
        expected_file_sha256=str(expected_lock["manifest_sha256"]),
    )
    transitions = load_teacher_transition_sidecar(
        root / "transitions.jsonl",
        manifest=manifest,
        expected_file_sha256=str(expected_lock["transitions_sha256"]),
    )
    build_report = _json_object(root / "build_report.json")
    if build_report.get("schema") != BUILD_REPORT_SCHEMA:
        raise ValueError(f"{suite} build report has an unsupported schema.")
    if _require_commit(build_report.get("git_commit"), field=f"{suite}.build_report.git_commit") != (
        expected_cache_build_commit
    ):
        raise ValueError(f"{suite} build report has the wrong cache-build commit.")
    revisions = _mapping(build_report.get("model_revisions"), field=f"{suite}.model_revisions")
    if revisions.get("qwen_reader") != expected_reader_revision:
        raise ValueError(f"{suite} cache was built with a different Reader revision.")
    if build_report.get("manifest_sha256") != expected_lock["manifest_sha256"]:
        raise ValueError(f"{suite} build report does not bind the locked manifest.")
    if build_report.get("sidecar_sha256") != expected_lock["transitions_sha256"]:
        raise ValueError(f"{suite} build report does not bind the locked transition sidecar.")
    if build_report.get("teacher_contract_sha256") != manifest.teacher_contract_sha256:
        raise ValueError(f"{suite} teacher-contract identity differs between report and manifest.")
    if build_report.get("renderer_contract_sha256") != manifest.renderer_contract_sha256:
        raise ValueError(f"{suite} renderer identity differs between report and manifest.")

    expected_state_count = _require_count(expected_lock.get("state_count"), field=f"{suite}.state_count")
    expected_transition_count = _require_count(expected_lock.get("transition_count"), field=f"{suite}.transition_count")
    expected_tensor_count = _require_count(
        expected_lock.get("artifact_tensor_count"), field=f"{suite}.artifact_tensor_count"
    )
    if len(manifest.records) != expected_state_count or build_report.get("state_count") != expected_state_count:
        raise ValueError(f"{suite} state count differs from its preregistered lock.")
    if (
        len(transitions) != expected_transition_count
        or build_report.get("transition_count") != expected_transition_count
    ):
        raise ValueError(f"{suite} transition count differs from its preregistered lock.")

    build_file_map = _validated_sha_map(
        build_report.get("artifact_file_sha256"), field=f"{suite}.build_report.artifact_file_sha256"
    )
    if len(build_file_map) != expected_tensor_count:
        raise ValueError(f"{suite} build report has the wrong number of tensor artifacts.")
    build_file_map_sha = canonical_json_sha256(build_file_map)
    if build_file_map_sha != expected_lock["artifact_file_sha_map_sha256"]:
        raise ValueError(f"{suite} build-report artifact SHA map differs from preregistration.")

    manifest_content_map: dict[str, str] = {}
    manifest_paths: list[str] = []
    for record in manifest.records:
        for specification in (record.image, record.latent, record.feature):
            if specification.relative_path in manifest_content_map:
                raise ValueError(f"{suite} manifest reuses an artifact path.")
            manifest_content_map[specification.relative_path] = specification.sha256
            manifest_paths.append(specification.relative_path)
    manifest_content_map = dict(sorted(manifest_content_map.items()))
    if len(manifest_content_map) != expected_tensor_count:
        raise ValueError(f"{suite} manifest has the wrong number of tensor artifacts.")
    manifest_content_map_sha = canonical_json_sha256(manifest_content_map)
    if manifest_content_map_sha != expected_lock["tensor_content_sha_map_sha256"]:
        raise ValueError(f"{suite} manifest tensor-content SHA map differs from preregistration.")
    if set(build_file_map) != set(manifest_paths):
        raise ValueError(f"{suite} build report and manifest enumerate different artifact paths.")

    pre_artifact_files = _artifact_inventory(root)
    if pre_artifact_files != build_file_map:
        raise ValueError(f"{suite} on-disk artifact files differ from the complete build-report SHA map.")

    loaded_content_map: dict[str, str] = {}
    state_reports: list[dict[str, Any]] = []
    for record in manifest.records:
        artifacts: dict[str, dict[str, Any]] = {}
        image: Tensor | None = None
        for name, specification in (
            ("image", record.image),
            ("latent", record.latent),
            ("feature", record.feature),
        ):
            artifact_path = _safe_cache_file(root, specification.relative_path)
            tensor = load_teacher_tensor(
                artifact_path,
                specification=specification,
                expected_file_sha256=build_file_map[specification.relative_path],
            )
            actual_content_sha = tensor_sha256(tensor)
            loaded_content_map[specification.relative_path] = actual_content_sha
            artifacts[name] = {
                "relative_path": specification.relative_path,
                "file_sha256": pre_artifact_files[specification.relative_path],
                "build_report_file_sha256": build_file_map[specification.relative_path],
                "tensor_content_sha256": actual_content_sha,
                "manifest_tensor_content_sha256": specification.sha256,
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "shape": list(tensor.shape),
                "passed": bool(
                    pre_artifact_files[specification.relative_path] == build_file_map[specification.relative_path]
                    and actual_content_sha == specification.sha256
                    and str(tensor.dtype).removeprefix("torch.") == specification.dtype
                    and tuple(tensor.shape) == specification.shape
                ),
            }
            if name == "image":
                image = tensor
        if image is None:  # pragma: no cover - manifest schema already requires image
            raise RuntimeError(f"{suite} state {record.state_id} has no decoded image.")
        forward = compare_forward_paths(
            image_processor=image_processor,
            image=image[0].to(device=device),
        )
        forward["source_teacher_tensor_sha256"] = artifacts["image"]["tensor_content_sha256"]
        source_canonical_sha256 = canonical_tensor_manifest(image[0])["sha256"]
        forward["source_canonical_tensor_sha256"] = source_canonical_sha256
        if forward["input"]["sha256"] != source_canonical_sha256:
            raise RuntimeError(f"{suite} image device transfer changed logical tensor contents.")
        state_reports.append(
            {
                "state_id": record.state_id,
                "teacher_key": record.teacher_key,
                "semantic_state_sha256": record.semantic_state_sha256,
                "artifacts": artifacts,
                "forward": forward,
                "passed": bool(all(item["passed"] for item in artifacts.values()) and forward["passed"]),
            }
        )

    loaded_content_map = dict(sorted(loaded_content_map.items()))
    if loaded_content_map != manifest_content_map:
        raise ValueError(f"{suite} loaded tensor contents differ from the complete manifest SHA map.")
    post_core = _core_file_hashes(root)
    post_artifact_files = _artifact_inventory(root)
    read_only = pre_core == post_core and pre_artifact_files == post_artifact_files
    observed = {
        "manifest_sha256": pre_core["manifest.json"],
        "transitions_sha256": pre_core["transitions.jsonl"],
        "build_report_sha256": pre_core["build_report.json"],
        "cache_build_commit": build_report["git_commit"],
        "reader_revision": revisions["qwen_reader"],
        "teacher_contract_sha256": manifest.teacher_contract_sha256,
        "renderer_contract_sha256": manifest.renderer_contract_sha256,
        "state_count": len(manifest.records),
        "transition_count": len(transitions),
        "artifact_tensor_count": len(loaded_content_map),
        "image_forward_count": len(state_reports),
        "artifact_file_sha_map_sha256": canonical_json_sha256(pre_artifact_files),
        "tensor_content_sha_map_sha256": canonical_json_sha256(loaded_content_map),
    }
    return {
        "suite": suite,
        "cache_root": str(root),
        "expected_lock": dict(expected_lock),
        "observed": observed,
        "maps": {
            "artifact_file_sha256": pre_artifact_files,
            "tensor_content_sha256": loaded_content_map,
        },
        "states": state_reports,
        "read_only_integrity": {
            "core_file_sha256_before": pre_core,
            "core_file_sha256_after": post_core,
            "artifact_file_sha_map_sha256_before": canonical_json_sha256(pre_artifact_files),
            "artifact_file_sha_map_sha256_after": canonical_json_sha256(post_artifact_files),
            "unchanged": read_only,
        },
        "passed": bool(
            observed["manifest_sha256"] == expected_lock["manifest_sha256"]
            and observed["transitions_sha256"] == expected_lock["transitions_sha256"]
            and observed["build_report_sha256"] == expected_lock["build_report_sha256"]
            and observed["artifact_file_sha_map_sha256"] == expected_lock["artifact_file_sha_map_sha256"]
            and observed["tensor_content_sha_map_sha256"] == expected_lock["tensor_content_sha_map_sha256"]
            and all(state["passed"] for state in state_reports)
            and read_only
        ),
    }


def _assert_output_outside_caches(output: Path, cache_paths: Sequence[Path]) -> None:
    destination = output.expanduser().resolve(strict=False)
    for cache in cache_paths:
        root = cache.expanduser().resolve(strict=False)
        try:
            destination.relative_to(root)
        except ValueError:
            continue
        raise ValueError("R3-TC0 output must remain outside both immutable teacher caches.")


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("R3-TC0 requires CUDA on the locked Inspire H200 runtime.")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("R3-TC0 formal evidence must run on CUDA.")
    _assert_output_outside_caches(
        args.output_json,
        (args.set8_cache, args.transition16_cache),
    )
    if args.set8_cache.expanduser().resolve(strict=True) == args.transition16_cache.expanduser().resolve(strict=True):
        raise ValueError("Set8 and Transition16 must reference distinct immutable caches.")

    locks = load_preregistered_cache_locks(args.preregistration)
    provenance = probe_provenance(
        root=ROOT,
        arguments=args,
        models={"reader": args.reader},
    )
    git = provenance.get("git", {})
    if git.get("clean") is not True or _COMMIT.fullmatch(str(git.get("commit"))) is None:
        raise RuntimeError("R3-TC0 requires an exact clean Git commit.")
    revision = locked_revision(args.reader)
    if revision != locks["reader_revision"]:
        raise ValueError("Reader snapshot revision differs from the preregistered Qwen revision.")
    model_provenance = provenance.get("models", {}).get("reader", {})
    if (
        model_provenance.get("observed_revision") != revision
        or model_provenance.get("expected_revision") != revision
        or model_provenance.get("revision_matches_lock") is not True
    ):
        raise ValueError("Reader provenance does not bind the model lock and snapshot revision.")

    determinism = configure_strict_cuda_determinism(seed=0)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.reader,
        local_files_only=True,
        use_fast=True,
        min_pixels=EXPECTED_MIN_MAX_PIXELS,
        max_pixels=EXPECTED_MIN_MAX_PIXELS,
    )
    image_processor = getattr(processor, "image_processor", None)
    processor_audit = audit_fast_processor(image_processor)
    if not processor_audit["passed"]:
        raise RuntimeError("Real Qwen fast image processor differs from the locked R3 geometry.")

    suites = {
        "set8": audit_cache_suite(
            suite="set8",
            cache_root=args.set8_cache,
            expected_lock=locks["suites"]["set8"],
            expected_reader_revision=revision,
            expected_cache_build_commit=locks["cache_build_commit"],
            image_processor=image_processor,
            device=device,
        ),
        "transition16": audit_cache_suite(
            suite="transition16",
            cache_root=args.transition16_cache,
            expected_lock=locks["suites"]["transition16"],
            expected_reader_revision=revision,
            expected_cache_build_commit=locks["cache_build_commit"],
            image_processor=image_processor,
            device=device,
        ),
    }
    summary = {
        "suite_count": len(suites),
        "suite_pass_count": sum(int(report["passed"]) for report in suites.values()),
        "state_count": sum(report["observed"]["state_count"] for report in suites.values()),
        "artifact_tensor_count": sum(report["observed"]["artifact_tensor_count"] for report in suites.values()),
        "image_forward_count": sum(report["observed"]["image_forward_count"] for report in suites.values()),
        "image_forward_pass_count": sum(
            int(state["forward"]["passed"]) for report in suites.values() for state in report["states"]
        ),
        "cache_mutation_count": sum(int(not report["read_only_integrity"]["unchanged"]) for report in suites.values()),
    }
    runtime = {
        "device": str(device),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": importlib.metadata.version("transformers"),
        "torchvision": importlib.metadata.version("torchvision"),
    }
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe": PROBE_NAME,
        "scope": {
            "included": "cached decoded image -> Qwen fast processor pixel_values/image_grid_thw",
            "excluded": "cached Qwen feature attention-backend compatibility; audited separately before T0/calibration",
            "cache_access": "read-only",
        },
        "feature_backend_compatibility": {
            "status": "not_evaluated_by_r3_tc0",
            "risk": (
                "existing feature.pt tensors were built with the Reader's default SDPA backend, while replacement "
                "training uses strict math-only SDPA"
            ),
            "required_followup": "preregistered R3-TF0 per-state strict-math query-free feature compatibility gate",
            "preregistered_thresholds": {
                "relative_l2_max": 0.01,
                "cosine_min": 0.9999,
            },
            "teacher_t0_unlocked": False,
            "teacher_calibration_unlocked": False,
            "teacher_assisted_training_unlocked": False,
        },
        "reader_revision": revision,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "preregistration": locks,
        "processor": processor_audit,
        "determinism": determinism,
        "runtime": runtime,
        "suites": suites,
        "summary": summary,
        "provenance": provenance,
        "passed": bool(
            all(report["passed"] for report in suites.values())
            and summary
            == {
                "suite_count": 2,
                "suite_pass_count": 2,
                "state_count": 30,
                "artifact_tensor_count": 90,
                "image_forward_count": 30,
                "image_forward_pass_count": 30,
                "cache_mutation_count": 0,
            }
            and locks["gpu_model"] in runtime["device_name"]
            and runtime["device_capability"] == [9, 0]
            and runtime["torch"] == locks["torch_version"]
            and runtime["cuda"] == locks["cuda_version"]
            and runtime["transformers"] == locks["transformers_version"]
            and runtime["torchvision"] == locks["torchvision_version"]
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _assert_output_outside_caches(args.output_json, (args.set8_cache, args.transition16_cache))
    except Exception as exc:  # noqa: BLE001 - never write even failure evidence into an immutable cache
        report = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "probe": PROBE_NAME,
            "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "passed": False,
        }
        emit_json_report(report, None)
        return 1
    try:
        report = run_probe(args)
    except Exception as exc:  # noqa: BLE001 - formal probes must emit fail-closed JSON for every failure
        report = {
            "schema_version": PROBE_SCHEMA_VERSION,
            "probe": PROBE_NAME,
            "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "passed": False,
        }
    emit_json_report(report, args.output_json)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
