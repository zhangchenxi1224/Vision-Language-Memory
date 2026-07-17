from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_GRID_THW,
    R3_QWEN_READER_INPUT_HW,
    R3_QWEN_READER_PIXEL_VALUES_SHAPE,
    R3_QWEN_READER_RESIZE_CONTRACT,
)


FIXTURE_RGB_SHA256 = "c44093f3ad73d6a3d62b5bf9b8ad226f65e65afd7841d5ef3ed80bc7d14a841a"
SET_EVENT = "The user prefers red mugs."
OVERWRITE_EVENT = "The user now prefers blue mugs instead of red mugs."
QUERY = "Which mug color does the user prefer?"
CHOICES = ("red", "blue", "green", "yellow")

GATE_PROTOCOLS: dict[str, dict[str, Any]] = {
    "G4-L": {
        "semantic_operations": ("set",),
        "events": (SET_EVENT,),
        "target_index": 0,
        "detach_between_events": False,
    },
    "G5-L": {
        "semantic_operations": ("set", "overwrite"),
        "events": (SET_EVENT, OVERWRITE_EVENT),
        "target_index": 1,
        "detach_between_events": False,
    },
    "G6-L": {
        "semantic_operations": ("set", "overwrite"),
        "events": (SET_EVENT, OVERWRITE_EVENT),
        "target_index": 1,
        "detach_between_events": True,
    },
}

RESIZE_CONTRACT_GATE = "R3-R0"
SCORER_CONTRACT_GATE = "R3-S0"
GATE_ORDER = (RESIZE_CONTRACT_GATE, "G4-L", "G5-L", "G6-L", "DL-S")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")

_STRICT_DETERMINISM_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
}
_R3_R0_RUNTIME = {
    "torch": "2.7.0a0+ecf3bae40a.nv25.02",
    "cuda_runtime": "12.8",
    "torchvision": "0.22.0a0",
    "transformers": "4.57.3",
    "device_name": "NVIDIA H200",
}
_LEGACY_NATIVE_THRESHOLDS = {
    "float32": {"candidate_relative_l2_max": 1e-5, "candidate_cosine_min": 0.999999},
    "bfloat16": {"candidate_relative_l2_max": 1e-2, "candidate_cosine_min": 0.9999},
}
_DL_S_RUNTIME = {
    "python": "3.12.3",
    "torch": "2.7.0a0+ecf3bae40a.nv25.02",
    "torchvision": "0.22.0a0",
    "cuda_runtime": "12.8",
    "diffusers": "0.39.0",
    "transformers": "4.57.3",
    "peft": "0.18.1",
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, found {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, found {result!r}.")
    return result


def _positive(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0:
        raise ValueError(f"{label} must be greater than zero, found {result!r}.")
    return result


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, found {actual!r}.")


def _validate_provenance(report: Mapping[str, Any]) -> dict[str, str]:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Probe report is missing provenance.")
    git = provenance.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("Probe provenance is missing git metadata.")
    commit = git.get("commit")
    if not isinstance(commit, str) or _HEX_40.fullmatch(commit) is None:
        raise ValueError("Probe provenance must contain a full 40-character git commit.")
    _require_equal(git.get("clean"), True, label="provenance.git.clean")
    runtime = provenance.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("Probe provenance is missing runtime metadata.")
    _require_equal(
        runtime.get("torch"),
        "2.7.0a0+ecf3bae40a.nv25.02",
        label="provenance.runtime.torch",
    )
    _require_equal(runtime.get("cuda_runtime"), "12.8", label="provenance.runtime.cuda_runtime")

    models = provenance.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("Probe provenance is missing model metadata.")
    revisions: dict[str, str] = {}
    for model_name in ("dreamlite", "reader"):
        model = models.get(model_name)
        if not isinstance(model, Mapping):
            raise ValueError(f"Probe provenance is missing {model_name} metadata.")
        _require_equal(
            model.get("revision_matches_lock"),
            True,
            label=f"provenance.models.{model_name}.revision_matches_lock",
        )
        expected_revision = model.get("expected_revision")
        observed_revision = model.get("observed_revision")
        if (
            not isinstance(expected_revision, str)
            or _HEX_40.fullmatch(expected_revision) is None
            or not isinstance(observed_revision, str)
            or _HEX_40.fullmatch(observed_revision) is None
        ):
            raise ValueError(
                f"Probe provenance must record full expected and observed {model_name} revisions."
            )
        _require_equal(
            observed_revision,
            expected_revision,
            label=f"provenance.models.{model_name}.observed_revision",
        )
        revisions[f"{model_name}_revision"] = observed_revision
    return {"git_commit": commit, **revisions}


def _validate_resize_provenance(report: Mapping[str, Any]) -> dict[str, str]:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("R3-R0 is missing provenance.")
    git = provenance.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("R3-R0 provenance is missing git metadata.")
    commit = git.get("commit")
    if not isinstance(commit, str) or _HEX_40.fullmatch(commit) is None:
        raise ValueError("R3-R0 provenance must contain a full 40-character git commit.")
    _require_equal(git.get("clean"), True, label="R3-R0.provenance.git.clean")

    models = provenance.get("models")
    reader = models.get("reader") if isinstance(models, Mapping) else None
    if not isinstance(reader, Mapping):
        raise ValueError("R3-R0 provenance is missing Reader model metadata.")
    _require_equal(
        reader.get("revision_matches_lock"),
        True,
        label="R3-R0.provenance.models.reader.revision_matches_lock",
    )
    expected_revision = reader.get("expected_revision")
    observed_revision = reader.get("observed_revision")
    if (
        not isinstance(expected_revision, str)
        or _HEX_40.fullmatch(expected_revision) is None
        or not isinstance(observed_revision, str)
        or _HEX_40.fullmatch(observed_revision) is None
    ):
        raise ValueError("R3-R0 Reader provenance must record full expected and observed revisions.")
    _require_equal(
        observed_revision,
        expected_revision,
        label="R3-R0.provenance.models.reader.observed_revision",
    )
    return {"git_commit": commit, "reader_revision": observed_revision}


def _validate_strict_determinism(report: Mapping[str, Any], *, label: str = "R3-R0") -> None:
    determinism = report.get("strict_determinism")
    if not isinstance(determinism, Mapping):
        raise ValueError(f"{label} is missing strict_determinism evidence.")
    _require_equal(determinism.get("seed"), 0, label=f"{label}.strict_determinism.seed")
    _require_equal(
        determinism.get("environment"),
        _STRICT_DETERMINISM_ENV,
        label=f"{label}.strict_determinism.environment",
    )
    expected = {
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "sdpa": {"flash": False, "memory_efficient": False, "cudnn": False, "math": True},
    }
    for field, expected_value in expected.items():
        _require_equal(
            determinism.get(field),
            expected_value,
            label=f"{label}.strict_determinism.{field}",
        )


def validate_scorer_contract_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the real-Qwen R3-S0 scorer contract included in the final C0 aggregate."""

    _require_equal(report.get("schema_version"), 1, label="R3-S0.schema_version")
    _require_equal(report.get("probe"), "r3_s0_qwen_scorer_contract", label="R3-S0.probe")
    _require_equal(report.get("passed"), True, label="R3-S0.passed")
    contract = report.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("R3-S0 is missing its scorer contract.")
    _require_equal(contract.get("reader_loss_mode"), "listwise-choice", label="R3-S0.reader_loss_mode")
    _require_equal(
        contract.get("reader_resize_contract"),
        R3_QWEN_READER_RESIZE_CONTRACT,
        label="R3-S0.reader_resize_contract",
    )
    _require_equal(
        report.get("summary"),
        {
            "views_passed": 8,
            "views_required": 8,
            "joint_tokenization_views_passed": 8,
            "train_eval_views_passed": 8,
            "repeat_eval_views_passed": 8,
        },
        label="R3-S0.summary",
    )
    _validate_strict_determinism(report, label="R3-S0")
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("R3-S0 is missing provenance.")
    git = provenance.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("R3-S0 provenance is missing Git metadata.")
    commit = git.get("commit")
    if not isinstance(commit, str) or _HEX_40.fullmatch(commit) is None:
        raise ValueError("R3-S0 provenance must contain a full Git commit.")
    _require_equal(git.get("clean"), True, label="R3-S0.provenance.git.clean")
    runtime = provenance.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("R3-S0 provenance is missing runtime metadata.")
    _require_equal(runtime.get("torch"), _R3_R0_RUNTIME["torch"], label="R3-S0.runtime.torch")
    _require_equal(runtime.get("cuda_runtime"), _R3_R0_RUNTIME["cuda_runtime"], label="R3-S0.runtime.cuda")
    models = provenance.get("models")
    reader = models.get("reader") if isinstance(models, Mapping) else None
    if not isinstance(reader, Mapping):
        raise ValueError("R3-S0 provenance is missing Reader metadata.")
    _require_equal(reader.get("revision_matches_lock"), True, label="R3-S0.reader.revision_matches_lock")
    expected_revision = reader.get("expected_revision")
    observed_revision = reader.get("observed_revision")
    if not isinstance(expected_revision, str) or _HEX_40.fullmatch(expected_revision) is None:
        raise ValueError("R3-S0 Reader revision lock is malformed.")
    _require_equal(observed_revision, expected_revision, label="R3-S0.reader.observed_revision")
    frozen = report.get("frozen_gradients")
    frozen_reader = frozen.get("reader") if isinstance(frozen, Mapping) else None
    if not isinstance(frozen_reader, Mapping):
        raise ValueError("R3-S0 is missing frozen Reader gradient evidence.")
    for field in ("trainable_parameter_tensors", "frozen_tensors_with_grad", "frozen_nonfinite_grad_elements"):
        _require_equal(frozen_reader.get(field), 0, label=f"R3-S0.frozen_gradients.reader.{field}")
    memory = report.get("cuda_peak_memory")
    cuda0 = memory.get("cuda:0") if isinstance(memory, Mapping) else None
    if not isinstance(cuda0, Mapping):
        raise ValueError("R3-S0 is missing cuda:0 peak-memory evidence.")
    _require_equal(cuda0.get("name"), "NVIDIA H200", label="R3-S0.cuda_peak_memory.cuda:0.name")
    for field in ("peak_allocated_gib", "peak_reserved_gib"):
        if _finite(cuda0.get(field), label=f"R3-S0.cuda_peak_memory.cuda:0.{field}") < 0:
            raise ValueError(f"R3-S0.cuda_peak_memory.cuda:0.{field} cannot be negative.")
    return {"valid": True, "git_commit": commit, "reader_revision": expected_revision}


def _validate_r3_r0_runtime(report: Mapping[str, Any]) -> None:
    runtime = report.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("R3-R0 is missing its locked Inspire H200 runtime.")
    packages = runtime.get("packages")
    if not isinstance(packages, Mapping):
        raise ValueError("R3-R0.runtime.packages is missing.")
    observed = {
        "torch": runtime.get("torch"),
        "cuda_runtime": runtime.get("cuda_runtime"),
        "torchvision": packages.get("torchvision"),
        "transformers": packages.get("transformers"),
        "device_name": runtime.get("device_name"),
    }
    _require_equal(observed, _R3_R0_RUNTIME, label="R3-R0.runtime.lock")
    total_memory = runtime.get("device_total_memory_bytes")
    if not isinstance(total_memory, int) or total_memory < 140_000 * 1024**2:
        raise ValueError("R3-R0.runtime must expose at least 140000 MiB on the locked H200.")


def _validate_r3_r0_execution_binding(report: Mapping[str, Any], *, git_commit: str) -> dict[str, str]:
    binding = report.get("execution_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("R3-R0 is missing its Inspire worker/preflight execution binding.")
    for field, expected in {
        "passed": True,
        "stage": "r3-r0",
        "infrastructure_stage": False,
        "git_commit": git_commit,
    }.items():
        _require_equal(binding.get(field), expected, label=f"R3-R0.execution_binding.{field}")
    result: dict[str, str] = {}
    for field in ("worker_input_sha256", "formal_preflight_sha256"):
        digest = binding.get(field)
        if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
            raise ValueError(f"R3-R0.execution_binding.{field} must be a SHA256 digest.")
        result[field] = digest
    for field in ("worker_input_path", "formal_preflight_path"):
        value = binding.get(field)
        if not isinstance(value, str) or not (value.startswith("/") or PureWindowsPath(value).is_absolute()):
            raise ValueError(f"R3-R0.execution_binding.{field} must be an absolute path.")
    return result


def _validate_tensor_summary(
    summary: Any,
    *,
    label: str,
    shape: Sequence[int],
    dtype: str | None = None,
) -> None:
    if not isinstance(summary, Mapping):
        raise ValueError(f"{label} must be a tensor summary.")
    _require_equal(summary.get("shape"), list(shape), label=f"{label}.shape")
    if dtype is not None:
        _require_equal(summary.get("dtype"), dtype, label=f"{label}.dtype")
    _require_equal(summary.get("finite"), True, label=f"{label}.finite")
    digest = summary.get("sha256")
    if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
        raise ValueError(f"{label}.sha256 must be a SHA256 digest.")


def validate_resize_contract_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the independent deterministic Qwen resize forward/backward gate."""

    _require_equal(report.get("schema_version"), 2, label="R3-R0.schema_version")
    _require_equal(
        report.get("probe"),
        "r3_qwen_resize_forward_backward_contract",
        label="R3-R0.probe",
    )
    _require_equal(report.get("passed"), True, label="R3-R0.passed")
    _require_equal(
        report.get("resize_contract"),
        R3_QWEN_READER_RESIZE_CONTRACT,
        label="R3-R0.resize_contract",
    )
    _require_equal(report.get("seed"), 0, label="R3-R0.seed")
    _require_equal(report.get("device"), "cuda:0", label="R3-R0.device")

    processor = report.get("processor")
    if not isinstance(processor, Mapping):
        raise ValueError("R3-R0 is missing the real fast-processor audit.")
    _require_equal(processor.get("passed"), True, label="R3-R0.processor.passed")
    checks = processor.get("checks")
    expected_processor_checks = {
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
    if not isinstance(checks, Mapping) or set(checks) != expected_processor_checks:
        raise ValueError("R3-R0 processor audit has missing or unexpected checks.")
    if not all(value is True for value in checks.values()):
        raise ValueError("R3-R0 processor audit did not pass every locked check.")
    observed_processor = processor.get("observed")
    if not isinstance(observed_processor, Mapping):
        raise ValueError("R3-R0 processor audit is missing observed values.")
    for field, expected in {
        "class": "Qwen2VLImageProcessorFast",
        "do_resize": True,
        "min_pixels": 65536,
        "max_pixels": 65536,
        "shortest_edge": 65536,
        "longest_edge": 16777216,
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "resample_value": 3,
    }.items():
        _require_equal(
            observed_processor.get(field),
            expected,
            label=f"R3-R0.processor.observed.{field}",
        )

    dtypes = report.get("dtypes")
    if not isinstance(dtypes, Mapping) or set(dtypes) != {"float32", "bfloat16"}:
        raise ValueError("R3-R0 must contain exactly float32 and bfloat16 evidence.")
    for dtype_name, tensor_dtype in (
        ("float32", "torch.float32"),
        ("bfloat16", "torch.bfloat16"),
    ):
        dtype_report = dtypes.get(dtype_name)
        if not isinstance(dtype_report, Mapping):
            raise ValueError(f"R3-R0.dtypes.{dtype_name} must be an object.")
        _require_equal(dtype_report.get("passed"), True, label=f"R3-R0.dtypes.{dtype_name}.passed")

        forward = dtype_report.get("forward_equivalence")
        if not isinstance(forward, Mapping):
            raise ValueError(f"R3-R0.dtypes.{dtype_name} is missing forward equivalence.")
        prefix = f"R3-R0.dtypes.{dtype_name}.forward_equivalence"
        _require_equal(forward.get("passed"), True, label=f"{prefix}.passed")
        _require_equal(
            forward.get("pixel_values_torch_equal"),
            True,
            label=f"{prefix}.pixel_values_torch_equal",
        )
        _require_equal(
            forward.get("pixel_values_max_absolute_difference"),
            0.0,
            label=f"{prefix}.pixel_values_max_absolute_difference",
        )
        _require_equal(forward.get("pixel_values_shape_locked"), True, label=f"{prefix}.pixel_values_shape_locked")
        _require_equal(
            forward.get("expected_pixel_values_shape"),
            list(R3_QWEN_READER_PIXEL_VALUES_SHAPE),
            label=f"{prefix}.expected_pixel_values_shape",
        )
        _require_equal(
            forward.get("legacy_grid_thw"),
            list(R3_QWEN_READER_GRID_THW),
            label=f"{prefix}.legacy_grid_thw",
        )
        _require_equal(
            forward.get("candidate_grid_thw"),
            list(R3_QWEN_READER_GRID_THW),
            label=f"{prefix}.candidate_grid_thw",
        )
        _require_equal(forward.get("grid_torch_equal_and_locked"), True, label=f"{prefix}.grid_torch_equal_and_locked")
        _validate_tensor_summary(
            forward.get("input"),
            label=f"{prefix}.input",
            shape=(3, *R3_QWEN_READER_INPUT_HW),
            dtype=tensor_dtype,
        )
        _validate_tensor_summary(
            forward.get("resized"),
            label=f"{prefix}.resized",
            shape=(3, 256, 256),
            dtype=tensor_dtype,
        )
        for path in ("legacy_pixel_values", "candidate_pixel_values"):
            _validate_tensor_summary(
                forward.get(path),
                label=f"{prefix}.{path}",
                shape=R3_QWEN_READER_PIXEL_VALUES_SHAPE,
            )
        legacy_summary = forward["legacy_pixel_values"]
        candidate_summary = forward["candidate_pixel_values"]
        _require_equal(
            candidate_summary.get("sha256"),
            legacy_summary.get("sha256"),
            label=f"{prefix}.candidate_pixel_values.sha256",
        )

        backward = dtype_report.get("strict_backward_repeat")
        if not isinstance(backward, Mapping):
            raise ValueError(f"R3-R0.dtypes.{dtype_name} is missing strict backward evidence.")
        backward_prefix = f"R3-R0.dtypes.{dtype_name}.strict_backward_repeat"
        for field in (
            "passed",
            "gradient_finite",
            "gradient_nonzero",
            "gradient_torch_equal",
            "loss_bitwise_equal",
        ):
            _require_equal(backward.get(field), True, label=f"{backward_prefix}.{field}")
        _require_equal(
            backward.get("gradient_max_absolute_difference"),
            0.0,
            label=f"{backward_prefix}.gradient_max_absolute_difference",
        )
        first = backward.get("first")
        second = backward.get("second")
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise ValueError(f"{backward_prefix} must contain both repeated runs.")
        for run_name, run in (("first", first), ("second", second)):
            _validate_tensor_summary(
                run.get("gradient"),
                label=f"{backward_prefix}.{run_name}.gradient",
                shape=(3, *R3_QWEN_READER_INPUT_HW),
                dtype=tensor_dtype,
            )
            _positive(run.get("gradient_norm"), label=f"{backward_prefix}.{run_name}.gradient_norm")
            _positive(run.get("loss"), label=f"{backward_prefix}.{run_name}.loss")
            if not isinstance(run.get("loss_float_hex"), str):
                raise ValueError(f"{backward_prefix}.{run_name}.loss_float_hex must be a string.")
            digest = run.get("pixel_values_sha256")
            if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
                raise ValueError(f"{backward_prefix}.{run_name}.pixel_values_sha256 must be a SHA256 digest.")
        _require_equal(
            second["gradient"].get("sha256"),
            first["gradient"].get("sha256"),
            label=f"{backward_prefix}.second.gradient.sha256",
        )
        _require_equal(second.get("loss_float_hex"), first.get("loss_float_hex"), label=f"{backward_prefix}.loss")
        _require_equal(
            second.get("pixel_values_sha256"),
            first.get("pixel_values_sha256"),
            label=f"{backward_prefix}.pixel_values_sha256",
        )

        adjoint = dtype_report.get("cpu_adjoint_reference")
        if not isinstance(adjoint, Mapping):
            raise ValueError(f"R3-R0.dtypes.{dtype_name} is missing CPU adjoint reference evidence.")
        adjoint_prefix = f"R3-R0.dtypes.{dtype_name}.cpu_adjoint_reference"
        _require_equal(adjoint.get("passed"), True, label=f"{adjoint_prefix}.passed")
        _require_equal(
            adjoint.get("reference"),
            "native-torchvision-cpu-fp32-autograd",
            label=f"{adjoint_prefix}.reference",
        )
        for field in ("gradient_finite", "gradient_nonzero", "gradient_torch_equal"):
            _require_equal(adjoint.get(field), True, label=f"{adjoint_prefix}.{field}")
        _require_equal(
            adjoint.get("gradient_max_absolute_difference"),
            0.0,
            label=f"{adjoint_prefix}.gradient_max_absolute_difference",
        )
        _positive(adjoint.get("candidate_gradient_norm"), label=f"{adjoint_prefix}.candidate_gradient_norm")
        _positive(adjoint.get("reference_gradient_norm"), label=f"{adjoint_prefix}.reference_gradient_norm")
        _validate_tensor_summary(
            adjoint.get("output_gradient"),
            label=f"{adjoint_prefix}.output_gradient",
            shape=(3, 256, 256),
            dtype=tensor_dtype,
        )
        for path in ("candidate_gradient", "reference_gradient"):
            _validate_tensor_summary(
                adjoint.get(path),
                label=f"{adjoint_prefix}.{path}",
                shape=(3, *R3_QWEN_READER_INPUT_HW),
                dtype=tensor_dtype,
            )
        _require_equal(
            adjoint["candidate_gradient"].get("sha256"),
            adjoint["reference_gradient"].get("sha256"),
            label=f"{adjoint_prefix}.gradient.sha256",
        )

        legacy = dtype_report.get("legacy_native_cuda_reference")
        if not isinstance(legacy, Mapping):
            raise ValueError(f"R3-R0.dtypes.{dtype_name} is missing legacy native CUDA evidence.")
        legacy_prefix = f"R3-R0.dtypes.{dtype_name}.legacy_native_cuda_reference"
        expected_thresholds = _LEGACY_NATIVE_THRESHOLDS[dtype_name]
        for field, expected in {
            "passed": True,
            "replicas": 3,
            "reference_only": True,
            "no_optimizer": True,
            "no_scientific_metric": True,
            "candidate_strict_determinism": True,
            "native_reference_determinism_disabled": True,
            "determinism_restored": True,
            "all_gradients_finite_nonzero": True,
            "thresholds": expected_thresholds,
        }.items():
            _require_equal(legacy.get(field), expected, label=f"{legacy_prefix}.{field}")
        relative_l2 = _finite(
            legacy.get("candidate_relative_l2_max"),
            label=f"{legacy_prefix}.candidate_relative_l2_max",
        )
        cosine = _finite(
            legacy.get("candidate_cosine_min"),
            label=f"{legacy_prefix}.candidate_cosine_min",
        )
        native_repeat = _finite(
            legacy.get("native_repeat_relative_l2_max"),
            label=f"{legacy_prefix}.native_repeat_relative_l2_max",
        )
        if relative_l2 < 0 or relative_l2 > expected_thresholds["candidate_relative_l2_max"]:
            raise ValueError(f"{legacy_prefix}.candidate_relative_l2_max exceeded its locked threshold.")
        if cosine < expected_thresholds["candidate_cosine_min"] or cosine > 1.0:
            raise ValueError(f"{legacy_prefix}.candidate_cosine_min violated its locked threshold.")
        if native_repeat < 0:
            raise ValueError(f"{legacy_prefix}.native_repeat_relative_l2_max cannot be negative.")
        candidate = legacy.get("candidate")
        native_runs = legacy.get("native_runs")
        if not isinstance(candidate, Mapping) or not isinstance(native_runs, list) or len(native_runs) != 3:
            raise ValueError(f"{legacy_prefix} must contain one candidate and three native runs.")
        _validate_tensor_summary(
            candidate.get("gradient"),
            label=f"{legacy_prefix}.candidate.gradient",
            shape=(3, *R3_QWEN_READER_INPUT_HW),
            dtype=tensor_dtype,
        )
        _positive(candidate.get("gradient_norm"), label=f"{legacy_prefix}.candidate.gradient_norm")
        for index, run in enumerate(native_runs):
            if not isinstance(run, Mapping):
                raise ValueError(f"{legacy_prefix}.native_runs[{index}] must be an object.")
            _require_equal(
                run.get("determinism_restored"),
                True,
                label=f"{legacy_prefix}.native_runs[{index}].determinism_restored",
            )
            _validate_tensor_summary(
                run.get("gradient"),
                label=f"{legacy_prefix}.native_runs[{index}].gradient",
                shape=(3, *R3_QWEN_READER_INPUT_HW),
                dtype=tensor_dtype,
            )
            _positive(
                run.get("gradient_norm"),
                label=f"{legacy_prefix}.native_runs[{index}].gradient_norm",
            )
            run_relative = _finite(
                run.get("candidate_relative_l2"),
                label=f"{legacy_prefix}.native_runs[{index}].candidate_relative_l2",
            )
            run_cosine = _finite(
                run.get("candidate_cosine"),
                label=f"{legacy_prefix}.native_runs[{index}].candidate_cosine",
            )
            if run_relative < 0 or run_relative > expected_thresholds["candidate_relative_l2_max"]:
                raise ValueError(f"{legacy_prefix}.native_runs[{index}] exceeded relative-L2 threshold.")
            if run_cosine < expected_thresholds["candidate_cosine_min"] or run_cosine > 1.0:
                raise ValueError(f"{legacy_prefix}.native_runs[{index}] violated cosine threshold.")

    _validate_strict_determinism(report)
    _validate_r3_r0_runtime(report)
    identity = _validate_resize_provenance(report)
    execution = _validate_r3_r0_execution_binding(report, git_commit=identity["git_commit"])
    return {
        "valid": True,
        "resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        **identity,
        **execution,
        "dtypes": ["float32", "bfloat16"],
        "forward_bitwise_equivalent": True,
        "backward_bitwise_repeatable": True,
        "cpu_adjoint_reference_exact": True,
        "legacy_native_cuda_reference_within_locked_tolerance": True,
    }


def _validate_frozen_gradients(report: Mapping[str, Any]) -> None:
    frozen = report.get("frozen_gradients")
    if not isinstance(frozen, Mapping):
        raise ValueError("Probe report is missing frozen_gradients.")
    for module_name in ("base_unet", "vae", "internal_qwen", "reader"):
        module = frozen.get(module_name)
        if not isinstance(module, Mapping):
            raise ValueError(f"frozen_gradients is missing {module_name}.")
        _require_equal(
            module.get("frozen_tensors_with_grad"),
            0,
            label=f"frozen_gradients.{module_name}.frozen_tensors_with_grad",
        )
        _require_equal(
            module.get("frozen_nonfinite_grad_elements"),
            0,
            label=f"frozen_gradients.{module_name}.frozen_nonfinite_grad_elements",
        )
        if module_name in {"vae", "internal_qwen", "reader"}:
            _require_equal(
                module.get("trainable_parameter_tensors"),
                0,
                label=f"frozen_gradients.{module_name}.trainable_parameter_tensors",
            )


def _validate_memory_report(report: Mapping[str, Any]) -> None:
    memory = report.get("cuda_peak_memory")
    if not isinstance(memory, Mapping):
        raise ValueError("Probe report is missing cuda_peak_memory.")
    for device in ("cuda:0", "cuda:1"):
        record = memory.get(device)
        if not isinstance(record, Mapping):
            raise ValueError(f"cuda_peak_memory is missing {device}; both allocated H200s must be recorded.")
        _require_equal(
            record.get("name"),
            "NVIDIA H200",
            label=f"cuda_peak_memory.{device}.name",
        )
        for field in ("peak_allocated_gib", "peak_reserved_gib"):
            value = _finite(record.get(field), label=f"cuda_peak_memory.{device}.{field}")
            if value < 0:
                raise ValueError(f"cuda_peak_memory.{device}.{field} cannot be negative.")


def validate_probe_report(report: Mapping[str, Any], gate: str) -> dict[str, Any]:
    """Validate one R3 listwise technical gate against its locked semantic fixture."""

    if gate not in GATE_PROTOCOLS:
        raise ValueError(f"Unsupported probe gate: {gate}.")
    protocol = GATE_PROTOCOLS[gate]
    events = list(protocol["events"])
    expected_intermediate_count = len(events) - 1

    _require_equal(report.get("probe"), "e2e_episode_grad", label=f"{gate}.probe")
    _require_equal(report.get("events"), len(events), label=f"{gate}.events")
    _require_equal(
        report.get("detach_between_events"),
        protocol["detach_between_events"],
        label=f"{gate}.detach_between_events",
    )
    _require_equal(report.get("reader_loss_mode"), "listwise-choice", label=f"{gate}.reader_loss_mode")
    _require_equal(report.get("updater_device"), "cuda:0", label=f"{gate}.updater_device")
    _require_equal(report.get("reader_device"), "cuda:1", label=f"{gate}.reader_device")
    _require_equal(
        report.get("reader_resize_contract"),
        R3_QWEN_READER_RESIZE_CONTRACT,
        label=f"{gate}.reader_resize_contract",
    )

    metadata = report.get("pair_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{gate} is missing pair_metadata.")
    _require_equal(metadata.get("event"), events, label=f"{gate}.pair_metadata.event")
    _require_equal(metadata.get("query"), QUERY, label=f"{gate}.pair_metadata.query")
    _require_equal(metadata.get("reader_loss_mode"), "listwise-choice", label=f"{gate}.pair_metadata.reader_loss_mode")
    _require_equal(metadata.get("target"), None, label=f"{gate}.pair_metadata.target")
    _require_equal(metadata.get("choices"), list(CHOICES), label=f"{gate}.pair_metadata.choices")
    _require_equal(
        metadata.get("target_index"),
        protocol["target_index"],
        label=f"{gate}.pair_metadata.target_index",
    )
    _require_equal(metadata.get("resolution"), 1024, label=f"{gate}.pair_metadata.resolution")
    _require_equal(metadata.get("adapter_seed"), 0, label=f"{gate}.pair_metadata.adapter_seed")
    _require_equal(
        metadata.get("event_noise_seeds"),
        list(range(len(events))),
        label=f"{gate}.pair_metadata.event_noise_seeds",
    )
    _require_equal(metadata.get("lora_rank"), 4, label=f"{gate}.pair_metadata.lora_rank")
    _require_equal(metadata.get("checkpoint_unet"), True, label=f"{gate}.pair_metadata.checkpoint_unet")
    _require_equal(metadata.get("dreamlite_device"), "cuda:0", label=f"{gate}.pair_metadata.dreamlite_device")
    _require_equal(metadata.get("reader_device"), "cuda:1", label=f"{gate}.pair_metadata.reader_device")
    _require_equal(
        metadata.get("reader_resize_contract"),
        R3_QWEN_READER_RESIZE_CONTRACT,
        label=f"{gate}.pair_metadata.reader_resize_contract",
    )
    _require_equal(
        metadata.get("strict_determinism"),
        report.get("strict_determinism"),
        label=f"{gate}.pair_metadata.strict_determinism",
    )

    source = metadata.get("source_image")
    if not isinstance(source, Mapping):
        raise ValueError(f"{gate}.pair_metadata is missing source_image.")
    _require_equal(source.get("origin"), "deterministic_fixture", label=f"{gate}.source_image.origin")
    _require_equal(source.get("mode"), "RGB", label=f"{gate}.source_image.mode")
    _require_equal(source.get("size"), [1024, 1024], label=f"{gate}.source_image.size")
    _require_equal(source.get("rgb_sha256"), FIXTURE_RGB_SHA256, label=f"{gate}.source_image.rgb_sha256")

    expected_pair_id = _canonical_sha256(dict(metadata))
    _require_equal(report.get("pair_id"), expected_pair_id, label=f"{gate}.pair_id")
    if _HEX_64.fullmatch(expected_pair_id) is None:  # pragma: no cover - hashlib contract
        raise ValueError(f"{gate}.pair_id is not a SHA256 digest.")

    _positive(report.get("loss"), label=f"{gate}.loss")
    choice_nll = report.get("choice_mean_nll")
    if not isinstance(choice_nll, Sequence) or isinstance(choice_nll, (str, bytes)) or len(choice_nll) != 4:
        raise ValueError(f"{gate}.choice_mean_nll must contain exactly four values.")
    for index, value in enumerate(choice_nll):
        _finite(value, label=f"{gate}.choice_mean_nll[{index}]")

    _positive(report.get("lora_grad_norm"), label=f"{gate}.lora_grad_norm")
    tensors_with_grad = report.get("lora_tensors_with_grad")
    if not isinstance(tensors_with_grad, int) or tensors_with_grad <= 0:
        raise ValueError(f"{gate}.lora_tensors_with_grad must be a positive integer.")
    _require_equal(report.get("lora_nonfinite_elements"), 0, label=f"{gate}.lora_nonfinite_elements")
    _positive(report.get("unclamped_image_grad_norm"), label=f"{gate}.unclamped_image_grad_norm")
    final_state_sha256 = report.get("final_state_sha256")
    if not isinstance(final_state_sha256, str) or _HEX_64.fullmatch(final_state_sha256) is None:
        raise ValueError(f"{gate}.final_state_sha256 must be a SHA256 digest.")
    final_state_gradient = report.get("final_state_gradient")
    if not isinstance(final_state_gradient, Mapping):
        raise ValueError(f"{gate}.final_state_gradient is missing.")
    _positive(final_state_gradient.get("norm"), label=f"{gate}.final_state_gradient.norm")
    _require_equal(
        final_state_gradient.get("nonfinite_elements"),
        0,
        label=f"{gate}.final_state_gradient.nonfinite_elements",
    )

    intermediate = report.get("intermediate_gradients")
    if not isinstance(intermediate, list) or len(intermediate) != expected_intermediate_count:
        raise ValueError(f"{gate}.intermediate_gradients must contain {expected_intermediate_count} record(s).")
    for index, record in enumerate(intermediate):
        if not isinstance(record, Mapping):
            raise ValueError(f"{gate}.intermediate_gradients[{index}] must be an object.")
        if protocol["detach_between_events"]:
            _require_equal(record.get("norm"), None, label=f"{gate}.intermediate_gradients[{index}].norm")
            _require_equal(
                record.get("nonfinite_elements"),
                None,
                label=f"{gate}.intermediate_gradients[{index}].nonfinite_elements",
            )
        else:
            _positive(record.get("norm"), label=f"{gate}.intermediate_gradients[{index}].norm")
            _require_equal(
                record.get("nonfinite_elements"),
                0,
                label=f"{gate}.intermediate_gradients[{index}].nonfinite_elements",
            )

    _validate_frozen_gradients(report)
    _validate_memory_report(report)
    _validate_strict_determinism(report, label=gate)
    identity = _validate_provenance(report)
    return {
        "valid": True,
        **identity,
        "semantic_operations": list(protocol["semantic_operations"]),
        "event_count": len(events),
        "target_index": protocol["target_index"],
        "loss": float(report["loss"]),
        "pair_id": report["pair_id"],
    }


def validate_pair(
    positive: Mapping[str, Any],
    detached: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Validate the G5-L/G6-L intervention as a forward-identical detach pair."""

    if atol < 0 or rtol < 0:
        raise ValueError("Pair tolerances must be non-negative.")
    _require_equal(positive.get("detach_between_events"), False, label="G5-L.detach_between_events")
    _require_equal(detached.get("detach_between_events"), True, label="G6-L.detach_between_events")
    _require_equal(detached.get("pair_id"), positive.get("pair_id"), label="G5-L/G6-L.pair_id")
    _require_equal(
        detached.get("pair_metadata"),
        positive.get("pair_metadata"),
        label="G5-L/G6-L.pair_metadata",
    )
    _require_equal(
        detached.get("final_state_sha256"),
        positive.get("final_state_sha256"),
        label="G5-L/G6-L.final_state_sha256",
    )
    positive_loss = _finite(positive.get("loss"), label="G5-L.loss")
    detached_loss = _finite(detached.get("loss"), label="G6-L.loss")
    if not math.isclose(positive_loss, detached_loss, abs_tol=atol, rel_tol=rtol):
        raise ValueError(
            "G5-L/G6-L forward losses differ outside the locked tolerances: "
            f"positive={positive_loss}, detached={detached_loss}, atol={atol}, rtol={rtol}."
        )
    return {
        "valid": True,
        "pair_id": positive.get("pair_id"),
        "positive_loss": positive_loss,
        "detached_loss": detached_loss,
        "absolute_loss_difference": abs(positive_loss - detached_loss),
        "atol": atol,
        "rtol": rtol,
    }


def validate_resume_report(report: Mapping[str, Any]) -> dict[str, Any]:
    _require_equal(report.get("schema_version"), 2, label="DL-S.schema_version")
    _require_equal(
        report.get("protocol"),
        "DL-S-common-prefix-16-vs-8-resume-8-next-step-v2",
        label="DL-S.protocol",
    )
    _require_equal(report.get("passed"), True, label="DL-S.passed")
    git_commit = report.get("git_commit")
    if not isinstance(git_commit, str) or _HEX_40.fullmatch(git_commit) is None:
        raise ValueError("DL-S.git_commit must bind the checkpoint manifest to one full Git commit.")
    _require_equal(
        report.get("reader_resize_contract"),
        R3_QWEN_READER_RESIZE_CONTRACT,
        label="DL-S.reader_resize_contract",
    )
    for field in ("dreamlite_revision", "reader_revision"):
        value = report.get(field)
        if not isinstance(value, str) or _HEX_40.fullmatch(value) is None:
            raise ValueError(f"DL-S.{field} must be a full model revision.")
    _require_equal(
        report.get("runtime_environment"),
        _DL_S_RUNTIME,
        label="DL-S.runtime_environment",
    )
    _require_equal(report.get("exact"), True, label="DL-S.exact")
    _require_equal(report.get("atol"), 0.0, label="DL-S.atol")
    _require_equal(report.get("rtol"), 0.0, label="DL-S.rtol")
    _require_equal(report.get("mismatch_count"), 0, label="DL-S.mismatch_count")
    _require_equal(
        report.get("presentations"),
        {"uninterrupted": 16, "shared_prefix": 8, "resumed_suffix": 8, "next_step": 17},
        label="DL-S.presentations",
    )
    checkpoints = report.get("checkpoint_state")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("DL-S is missing checkpoint_state.")
    _require_equal(
        checkpoints.get("prefix"),
        {"epoch": 0, "episode_cursor": 8, "optimizer_step": 8},
        label="DL-S.checkpoint_state.prefix",
    )
    expected_final = {"epoch": 0, "episode_cursor": 16, "optimizer_step": 16}
    _require_equal(checkpoints.get("reference"), expected_final, label="DL-S.checkpoint_state.reference")
    _require_equal(checkpoints.get("resumed"), expected_final, label="DL-S.checkpoint_state.resumed")
    next_checkpoints = report.get("next_checkpoint_state")
    if not isinstance(next_checkpoints, Mapping):
        raise ValueError("DL-S is missing next_checkpoint_state.")
    expected_next = {"epoch": 1, "episode_cursor": 1, "optimizer_step": 17}
    _require_equal(next_checkpoints.get("reference"), expected_next, label="DL-S.next.reference")
    _require_equal(next_checkpoints.get("resumed"), expected_next, label="DL-S.next.resumed")
    next_metric = report.get("next_step_metric")
    if not isinstance(next_metric, Mapping):
        raise ValueError("DL-S is missing exact next_step_metric evidence.")
    for field in ("raw_gradient_sha256", "clipped_gradient_sha256"):
        digest = next_metric.get(field)
        if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
            raise ValueError(f"DL-S.next_step_metric.{field} must be a SHA256 digest.")
    for field in ("loss_hex", "gradient_norm_hex"):
        if not isinstance(next_metric.get(field), str):
            raise ValueError(f"DL-S.next_step_metric.{field} must be an exact hexadecimal float string.")
    lineage = report.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("DL-S is missing lineage.")
    _require_equal(lineage.get("training_regime"), "qa_only", label="DL-S.lineage.training_regime")
    _require_equal(lineage.get("reader_loss_mode"), "listwise-choice", label="DL-S.lineage.reader_loss_mode")
    _require_equal(lineage.get("qa_supervision"), "listwise-choice", label="DL-S.lineage.qa_supervision")
    _require_equal(lineage.get("choice_view_schedule"), "cyclic4", label="DL-S.lineage.choice_view_schedule")

    paths = report.get("checkpoint_paths")
    if not isinstance(paths, Mapping) or not all(
        isinstance(paths.get(name), str)
        for name in ("prefix", "reference", "resumed", "reference_next", "resumed_next")
    ):
        raise ValueError("DL-S must record prefix/reference/resumed checkpoint paths.")
    prefix_path = Path(str(paths["prefix"]))
    reference_path = Path(str(paths["reference"]))
    resumed_path = Path(str(paths["resumed"]))
    _require_equal(prefix_path.name, "checkpoint-000008.pt", label="DL-S.checkpoint_paths.prefix.name")
    _require_equal(reference_path.name, "checkpoint-000016.pt", label="DL-S.checkpoint_paths.reference.name")
    _require_equal(resumed_path.name, "checkpoint-000016.pt", label="DL-S.checkpoint_paths.resumed.name")
    _require_equal(prefix_path.parent, reference_path.parent, label="DL-S.common_prefix_parent")
    if resumed_path.parent == reference_path.parent:
        raise ValueError("DL-S resumed checkpoint must come from a separate output directory.")
    _require_equal(Path(str(paths["reference_next"])).parent, reference_path.parent, label="DL-S.reference_next.parent")
    _require_equal(Path(str(paths["resumed_next"])).parent, resumed_path.parent, label="DL-S.resumed_next.parent")

    checkpoint_sha256 = report.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, Mapping):
        raise ValueError("DL-S must record checkpoint SHA256 values.")
    for name in ("prefix", "reference", "resumed", "reference_next", "resumed_next"):
        digest = checkpoint_sha256.get(name)
        if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
            raise ValueError(f"DL-S.checkpoint_sha256.{name} must be a SHA256 digest.")
    return {
        "valid": True,
        "exact": True,
        "git_commit": git_commit,
        "dreamlite_revision": str(report["dreamlite_revision"]),
        "reader_revision": str(report["reader_revision"]),
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "presentations": dict(report["presentations"]),
    }


def validate_reports(
    *,
    through: str,
    resize_contract: Mapping[str, Any] | None = None,
    scorer_s0: Mapping[str, Any] | None = None,
    g4: Mapping[str, Any] | None = None,
    g5: Mapping[str, Any] | None = None,
    g6: Mapping[str, Any] | None = None,
    resume: Mapping[str, Any] | None = None,
    pair_atol: float = 1e-5,
    pair_rtol: float = 1e-4,
) -> dict[str, Any]:
    if through not in GATE_ORDER:
        raise ValueError(f"Unknown through gate: {through}.")
    required = GATE_ORDER[: GATE_ORDER.index(through) + 1]
    inputs = {
        RESIZE_CONTRACT_GATE: resize_contract,
        "G4-L": g4,
        "G5-L": g5,
        "G6-L": g6,
        "DL-S": resume,
    }
    checks: dict[str, Any] = {}
    errors: list[str] = []
    probe_commits: set[str] = set()
    reader_revisions: set[str] = set()
    dreamlite_revisions: set[str] = set()

    if through == "DL-S":
        if scorer_s0 is None:
            errors.append("R3-S0: required report was not supplied.")
        else:
            try:
                checks[SCORER_CONTRACT_GATE] = validate_scorer_contract_report(scorer_s0)
                scorer_commit = checks[SCORER_CONTRACT_GATE].get("git_commit")
                scorer_reader_revision = checks[SCORER_CONTRACT_GATE].get("reader_revision")
                if isinstance(scorer_commit, str):
                    probe_commits.add(scorer_commit)
                if isinstance(scorer_reader_revision, str):
                    reader_revisions.add(scorer_reader_revision)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"R3-S0: {exc}")

    for gate in required:
        report = inputs[gate]
        if report is None:
            errors.append(f"{gate}: required report was not supplied.")
            continue
        try:
            if gate == RESIZE_CONTRACT_GATE:
                checks[gate] = validate_resize_contract_report(report)
            elif gate == "DL-S":
                checks[gate] = validate_resume_report(report)
            else:
                checks[gate] = validate_probe_report(report, gate)
            commit = checks[gate].get("git_commit")
            reader_revision = checks[gate].get("reader_revision")
            dreamlite_revision = checks[gate].get("dreamlite_revision")
            if isinstance(commit, str):
                probe_commits.add(commit)
            if isinstance(reader_revision, str):
                reader_revisions.add(reader_revision)
            if isinstance(dreamlite_revision, str):
                dreamlite_revisions.add(dreamlite_revision)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{gate}: {exc}")

    if "G6-L" in required and g5 is not None and g6 is not None:
        try:
            checks["G5-L/G6-L-pair"] = validate_pair(g5, g6, atol=pair_atol, rtol=pair_rtol)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"G5-L/G6-L-pair: {exc}")

    if len(probe_commits) != 1:
        errors.append("technical probe reports do not share one clean Git commit")
    if len(reader_revisions) != 1:
        errors.append("technical probe reports do not share one locked Reader revision")
    if any(gate in required for gate in ("G4-L", "G5-L", "G6-L", "DL-S")) and len(
        dreamlite_revisions
    ) != 1:
        errors.append("technical probe reports do not share one locked DreamLite revision")
    git_commit = next(iter(probe_commits)) if len(probe_commits) == 1 else None
    reported_required = list(required)
    if through == "DL-S":
        reported_required.insert(1, SCORER_CONTRACT_GATE)
    return {
        "schema_version": 2,
        "protocol": "R3-technical-listwise-resize-v2",
        "through": through,
        "required_gates": reported_required,
        "failure_policy": "fail-closed; gates are serial and downstream gates require prior success",
        "pair_atol": pair_atol,
        "pair_rtol": pair_rtol,
        "checks": checks,
        "git_commit": git_commit,
        "errors": errors,
        "passed": not errors and all(gate in checks for gate in reported_required),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed validation for R3 listwise technical gates")
    parser.add_argument("--through", choices=GATE_ORDER, required=True)
    parser.add_argument("--resize-contract", type=Path, required=True)
    parser.add_argument("--scorer-s0", type=Path)
    parser.add_argument("--g4", type=Path)
    parser.add_argument("--g5", type=Path)
    parser.add_argument("--g6", type=Path)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--pair-atol", type=float, default=1e-5)
    parser.add_argument("--pair-rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "resize_contract": args.resize_contract,
        "scorer_s0": args.scorer_s0,
        "g4": args.g4,
        "g5": args.g5,
        "g6": args.g6,
        "resume": args.resume_report,
    }
    loaded: dict[str, dict[str, Any] | None] = {}
    load_errors: list[str] = []
    for name, path in paths.items():
        if path is None:
            loaded[name] = None
            continue
        try:
            loaded[name] = _read_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            loaded[name] = None
            load_errors.append(f"{name}: could not load {path}: {exc}")

    try:
        report = validate_reports(
            through=args.through,
            resize_contract=loaded["resize_contract"],
            scorer_s0=loaded["scorer_s0"],
            g4=loaded["g4"],
            g5=loaded["g5"],
            g6=loaded["g6"],
            resume=loaded["resume"],
            pair_atol=args.pair_atol,
            pair_rtol=args.pair_rtol,
        )
    except (TypeError, ValueError) as exc:
        report = {
            "schema_version": 2,
            "protocol": "R3-technical-listwise-resize-v2",
            "through": args.through,
            "checks": {},
            "errors": [str(exc)],
            "passed": False,
        }
    if load_errors:
        report["errors"] = load_errors + list(report.get("errors", []))
        report["passed"] = False
    report["input_paths"] = {name: None if path is None else str(path.resolve()) for name, path in paths.items()}
    report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
