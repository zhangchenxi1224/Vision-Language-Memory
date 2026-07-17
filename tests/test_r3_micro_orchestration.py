from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "cluster"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from render_r3_micro_gates import (  # noqa: E402
    MicroPaths,
    SuiteSpec,
    build_stages,
    materialize_dry_run,
    render_stage_sbatch,
)
from score_r3_micro import (  # noqa: E402
    build_artifact_provenance,
    scientific_prediction_payload,
    sha256_file,
)
from dreamlite_episode import audit_episode_gradients, gradient_audit_summary  # noqa: E402
from validate_r3_micro_prerequisites import validate_prerequisites  # noqa: E402
from validate_r3_micro_replication import validate_replication  # noqa: E402
from vision_memory.reader import R3_QWEN_READER_RESIZE_CONTRACT  # noqa: E402
from vision_memory.teacher import load_teacher_calibration_input_lock  # noqa: E402


SHA = "a" * 64
COMMIT = "b" * 40


def _technical_report() -> dict:
    return {
        "protocol": "R3-technical-listwise-resize-v2",
        "through": "DL-S",
        "required_gates": ["R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L", "DL-S"],
        "checks": {
            "R3-R0": {
                "valid": True,
                "resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
                "reader_revision": "2" * 40,
            },
            "R3-S0": {"valid": True, "reader_revision": "2" * 40},
            "G4-L": {"valid": True, "reader_revision": "2" * 40, "dreamlite_revision": "1" * 40},
            "G5-L": {"valid": True, "reader_revision": "2" * 40, "dreamlite_revision": "1" * 40},
            "G6-L": {"valid": True, "reader_revision": "2" * 40, "dreamlite_revision": "1" * 40},
            "DL-S": {"valid": True, "reader_revision": "2" * 40, "dreamlite_revision": "1" * 40},
        },
        "errors": [],
        "git_commit": COMMIT,
        "passed": True,
    }


def _resize_tensor(shape: list[int], dtype: str, digest: str) -> dict:
    return {
        "shape": shape,
        "dtype": dtype,
        "device": "cuda:0",
        "sha256": digest,
        "finite": True,
    }


def _resize_contract_report() -> dict:
    dtypes = {}
    for index, (name, dtype) in enumerate(
        (("float32", "torch.float32"), ("bfloat16", "torch.bfloat16")),
        start=1,
    ):
        pixels_sha = str(index + 4) * 64
        gradient_sha = str(index + 6) * 64
        run = {
            "loss": 1.0,
            "loss_float_hex": "0x1.0000000000000p+0",
            "pixel_values_sha256": pixels_sha,
            "gradient": _resize_tensor([3, 1024, 1024], dtype, gradient_sha),
            "gradient_norm": 0.5,
        }
        adjoint_sha = ("b" if index == 1 else "c") * 64
        native_gradient = _resize_tensor([3, 1024, 1024], dtype, adjoint_sha)
        native_run = {
            "gradient": native_gradient,
            "gradient_norm": 0.5,
            "candidate_relative_l2": 0.0,
            "candidate_cosine": 1.0,
            "determinism_restored": True,
        }
        thresholds = (
            {"candidate_relative_l2_max": 1e-5, "candidate_cosine_min": 0.999999}
            if name == "float32"
            else {"candidate_relative_l2_max": 1e-2, "candidate_cosine_min": 0.9999}
        )
        dtypes[name] = {
            "passed": True,
            "forward_equivalence": {
                "passed": True,
                "input": _resize_tensor([3, 1024, 1024], dtype, str(index) * 64),
                "resized": _resize_tensor([3, 256, 256], dtype, str(index + 2) * 64),
                "legacy_pixel_values": _resize_tensor(
                    [256, 1536], "torch.float32", pixels_sha
                ),
                "candidate_pixel_values": _resize_tensor(
                    [256, 1536], "torch.float32", pixels_sha
                ),
                "pixel_values_torch_equal": True,
                "pixel_values_max_absolute_difference": 0.0,
                "expected_pixel_values_shape": [256, 1536],
                "pixel_values_shape_locked": True,
                "legacy_grid_thw": [1, 16, 16],
                "candidate_grid_thw": [1, 16, 16],
                "expected_grid_thw": [1, 16, 16],
                "grid_torch_equal_and_locked": True,
            },
            "strict_backward_repeat": {
                "passed": True,
                "first": dict(run),
                "second": dict(run),
                "gradient_finite": True,
                "gradient_nonzero": True,
                "gradient_torch_equal": True,
                "gradient_max_absolute_difference": 0.0,
                "loss_bitwise_equal": True,
            },
            "cpu_adjoint_reference": {
                "passed": True,
                "reference": "native-torchvision-cpu-fp32-autograd",
                "output_gradient": _resize_tensor(
                    [3, 256, 256], dtype, ("9" if index == 1 else "a") * 64
                ),
                "candidate_gradient": _resize_tensor(
                    [3, 1024, 1024], dtype, adjoint_sha
                ),
                "reference_gradient": _resize_tensor(
                    [3, 1024, 1024], dtype, adjoint_sha
                ),
                "gradient_finite": True,
                "gradient_nonzero": True,
                "gradient_torch_equal": True,
                "gradient_max_absolute_difference": 0.0,
                "candidate_gradient_norm": 0.5,
                "reference_gradient_norm": 0.5,
            },
            "legacy_native_cuda_reference": {
                "passed": True,
                "replicas": 3,
                "reference_only": True,
                "no_optimizer": True,
                "no_scientific_metric": True,
                "candidate_strict_determinism": True,
                "native_reference_determinism_disabled": True,
                "determinism_restored": True,
                "thresholds": thresholds,
                "all_gradients_finite_nonzero": True,
                "candidate_relative_l2_max": 0.0,
                "candidate_cosine_min": 1.0,
                "native_repeat_relative_l2_max": 0.0,
                "candidate": {"gradient": native_gradient, "gradient_norm": 0.5},
                "native_runs": [dict(native_run) for _ in range(3)],
            },
        }
    return {
        "schema_version": 2,
        "probe": "r3_qwen_resize_forward_backward_contract",
        "passed": True,
        "resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "seed": 0,
        "device": "cuda:0",
        "processor": {
            "passed": True,
            "observed": {
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
            },
            "checks": {
                name: True
                for name in (
                    "fast_tensor_processor",
                    "resize_enabled_by_default",
                    "min_pixels_locked",
                    "max_pixels_locked",
                    "patch_size_locked",
                    "temporal_patch_size_locked",
                    "merge_size_locked",
                    "bicubic_resample_locked",
                    "callable",
                )
            },
        },
        "dtypes": dtypes,
        "strict_determinism": {
            "seed": 0,
            "environment": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "sdpa": {"flash": False, "memory_efficient": False, "cudnn": False, "math": True},
        },
        "runtime": {
            "torch": "2.7.0a0+ecf3bae40a.nv25.02",
            "cuda_runtime": "12.8",
            "packages": {"torchvision": "0.22.0a0", "transformers": "4.57.3"},
            "device_name": "NVIDIA H200",
            "device_total_memory_bytes": 143 * 1024**3,
        },
        "execution_binding": {
            "passed": True,
            "stage": "r3-r0",
            "infrastructure_stage": False,
            "git_commit": COMMIT,
            "worker_input_path": "/runs/r3-r0/worker_input.json",
            "worker_input_sha256": "d" * 64,
            "formal_preflight_path": "/runs/preflight/r3_h200_formal.json",
            "formal_preflight_sha256": "e" * 64,
        },
        "provenance": {
            "git": {"commit": COMMIT, "clean": True},
            "runtime": {
                "torch": "2.7.0a0+ecf3bae40a.nv25.02",
                "cuda_runtime": "12.8",
            },
            "models": {
                "reader": {
                    "expected_revision": "2" * 40,
                    "observed_revision": "2" * 40,
                    "revision_matches_lock": True,
                }
            },
        },
    }


def _s0_report() -> dict:
    return {
        "schema_version": 1,
        "probe": "r3_s0_qwen_scorer_contract",
        "passed": True,
        "contract": {
            "reader_loss_mode": "listwise-choice",
            "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        },
        "summary": {
            "views_passed": 8,
            "views_required": 8,
            "joint_tokenization_views_passed": 8,
            "train_eval_views_passed": 8,
            "repeat_eval_views_passed": 8,
        },
        "strict_determinism": {
            "seed": 0,
            "environment": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONHASHSEED": "0",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
            "sdpa": {"flash": False, "memory_efficient": False, "cudnn": False, "math": True},
        },
        "provenance": {
            "git": {"commit": COMMIT, "clean": True},
            "runtime": {
                "torch": "2.7.0a0+ecf3bae40a.nv25.02",
                "cuda_runtime": "12.8",
            },
            "models": {
                "reader": {
                    "expected_revision": "2" * 40,
                    "observed_revision": "2" * 40,
                    "revision_matches_lock": True,
                }
            },
        },
    }


def _teacher_calibration_report() -> dict:
    from vision_memory.teacher import FrozenTeacherLossCalibration

    calibration = FrozenTeacherLossCalibration(1.0, 2.0, 3.0)
    input_lock = load_teacher_calibration_input_lock(
        ROOT / "configs" / "experiments" / "r3_preregistration.json",
        suite="set8",
    )
    models = {
        "reader": {
            "expected_revision": "2" * 40,
            "observed_revision": "2" * 40,
            "revision_matches_lock": True,
        },
        "dreamlite": {
            "expected_revision": "1" * 40,
            "observed_revision": "1" * 40,
            "revision_matches_lock": True,
        },
    }
    return {
        "schema": "vision_memory.r3-teacher-calibration-report.v1",
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        **input_lock.to_dict(),
        "calibration_file_sha256": SHA,
        "calibration_contract_sha256": calibration.contract_sha256,
        "seed": 0,
        "adapter_seed": 0,
        "lora_rank": 4,
        "initial_state": {"origin": "blank_fixture", "mode": "RGB", "size": [1024, 1024]},
        "sample_selection": {
            "split": "train",
            "unit": "one-unweighted-sample-per-updater-transition",
            "query_turns_excluded": True,
            "duplicate_semantic_after_states_retained": True,
        },
        "scales": calibration.to_dict(),
        "raw_component_ranges": {
            "latent": [0.1, 1.0],
            "image": [0.1, 1.0],
            "feature": [0.1, 1.0],
        },
        "strict_determinism": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "sdpa": {"flash": False, "memory_efficient": False, "cudnn": False, "math": True},
        },
        "provenance": {"git": {"commit": COMMIT, "clean": True}, "models": models},
    }


def _teacher_calibration_binding_kwargs() -> dict:
    input_lock = load_teacher_calibration_input_lock(
        ROOT / "configs" / "experiments" / "r3_preregistration.json",
        suite="set8",
    )
    return {
        "teacher_calibration_suite": input_lock.suite,
        "teacher_calibration_preregistration_sha256": input_lock.preregistration_sha256,
        "teacher_calibration_train_sha256": input_lock.train_sha256,
        "teacher_calibration_manifest_sha256": input_lock.manifest_sha256,
        "teacher_calibration_sidecar_sha256": input_lock.sidecar_sha256,
    }


def _teacher_t0_report() -> dict:
    return {
        "schema_version": 1,
        "probe": "teacher_t0_real_qwen_integrity_upper_bound",
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "passed": True,
        "preregistered_inputs": {
            "passed": True,
            "checks": {
                "gate_jsonl_sha256": True,
                "raw_sidecar_sha256": True,
                "teacher_manifest_sha256": True,
                "font_sha256": True,
                "renderer_contract_sha256": True,
                "teacher_contract_sha256": True,
                "reader_revision": True,
                "pillow_version": True,
            },
        },
        "cache_integrity": {"passed": True},
        "cross_split_fail_closed": {"passed": True},
        "upper_bound": {"passed": True},
        "identity_mutations": {"state": {"passed": True}},
        "provenance": {
            "git": {"commit": COMMIT, "clean": True},
            "runtime": {
                "torch": "2.7.0a0+ecf3bae40a.nv25.02",
                "cuda_runtime": "12.8",
            },
            "models": {
                "reader": {
                    "expected_revision": "2" * 40,
                    "observed_revision": "2" * 40,
                    "revision_matches_lock": True,
                }
            },
        },
        "strict_determinism": _s0_report()["strict_determinism"],
        "frozen_gradients": {
            "reader": {
                "trainable_parameter_tensors": 0,
                "frozen_tensors_with_grad": 0,
                "frozen_nonfinite_grad_elements": 0,
            }
        },
        "cuda_peak_memory": {
            "cuda:0": {
                "name": "NVIDIA H200",
                "peak_allocated_gib": 1.0,
                "peak_reserved_gib": 2.0,
            }
        },
    }


def _teacher_tc0_report() -> dict:
    return {
        "schema_version": 1,
        "protocol": "R3-TC0-cache-forward-compatibility-validation.v1",
        "expected_commit": COMMIT,
        "reader_revision": "2" * 40,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "preregistration_sha256": "9" * 64,
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


def _teacher_tf0_report() -> dict:
    return {
        "schema_version": 1,
        "protocol": "R3-TF0-feature-backend-compatibility-validation.v1",
        "expected_commit": COMMIT,
        "reader_revision": "2" * 40,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "feature_gate_sha256": "8" * 64,
        "preregistration_sha256": "9" * 64,
        "tc0_validation_sha256": SHA,
        "validated_suites": ["set8", "transition16"],
        "validated_state_count": 30,
        "validated_feature_comparison_count": 30,
        "validated_feature_pass_count": 30,
        "teacher_t0_unlocked": True,
        "teacher_calibration_unlocked": True,
        "teacher_assisted_training_unlocked": True,
        "qa_only_dependency": False,
        "errors": [],
        "passed": True,
    }


def _paths(root: Path) -> MicroPaths:
    return MicroPaths(
        project=root / "project",
        environment=root / "environment",
        model_root=root / "models",
        run_root=root / "run",
        resize_contract_report=root / "r0.json",
        scorer_s0_report=root / "s0.json",
        technical_report=root / "technical.json",
        teacher_t0_report=root / "teacher_t0.json",
    )


def _suite(root: Path, name: str, episodes: int) -> SuiteSpec:
    return SuiteSpec(
        name=name,
        train=root / "data" / name / "train.jsonl",
        gate=root / "data" / name / "gate.jsonl",
        teacher_cache=root / "teacher" / name,
        train_sha256=SHA,
        gate_sha256=SHA,
        teacher_manifest_sha256=SHA,
        teacher_sidecar_sha256=SHA,
        teacher_calibration_sha256=SHA,
        episodes=episodes,
    )


def test_prerequisites_require_both_complete_reports_and_clean_commit() -> None:
    passed = validate_prerequisites(
        resize_contract=_resize_contract_report(),
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=_teacher_t0_report(),
        teacher_calibration=_teacher_calibration_report(),
        teacher_calibration_file_sha256=SHA,
        teacher_tc0=_teacher_tc0_report(),
        teacher_tc0_file_sha256=SHA,
        teacher_tf0=_teacher_tf0_report(),
        teacher_tf0_file_sha256=SHA,
        training_regime="teacher_assisted",
        expected_commit=COMMIT,
        **_teacher_calibration_binding_kwargs(),
    )
    assert passed["passed"] is True
    failed_t0 = _teacher_t0_report()
    failed_t0["upper_bound"] = {"passed": False}
    failed = validate_prerequisites(
        resize_contract=_resize_contract_report(),
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=failed_t0,
        teacher_calibration=_teacher_calibration_report(),
        teacher_calibration_file_sha256=SHA,
        teacher_tc0=_teacher_tc0_report(),
        teacher_tc0_file_sha256=SHA,
        teacher_tf0=_teacher_tf0_report(),
        teacher_tf0_file_sha256=SHA,
        training_regime="teacher_assisted",
        expected_commit=COMMIT,
        **_teacher_calibration_binding_kwargs(),
    )
    assert failed["passed"] is False
    assert any("upper_bound" in error for error in failed["errors"])
    unbound_t0 = _teacher_t0_report()
    unbound_t0["preregistered_inputs"]["checks"]["teacher_manifest_sha256"] = False
    unbound = validate_prerequisites(
        resize_contract=_resize_contract_report(),
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=unbound_t0,
        teacher_calibration=_teacher_calibration_report(),
        teacher_calibration_file_sha256=SHA,
        teacher_tc0=_teacher_tc0_report(),
        teacher_tc0_file_sha256=SHA,
        teacher_tf0=_teacher_tf0_report(),
        teacher_tf0_file_sha256=SHA,
        training_regime="teacher_assisted",
        expected_commit=COMMIT,
        **_teacher_calibration_binding_kwargs(),
    )
    assert unbound["passed"] is False
    assert any("prospective preregistered input" in error for error in unbound["errors"])
    drifted_calibration = _teacher_calibration_report()
    drifted_calibration["reader_resize_contract"] = "drifted"
    calibration_failed = validate_prerequisites(
        resize_contract=_resize_contract_report(),
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=_teacher_t0_report(),
        teacher_calibration=drifted_calibration,
        teacher_calibration_file_sha256=SHA,
        teacher_tc0=_teacher_tc0_report(),
        teacher_tc0_file_sha256=SHA,
        teacher_tf0=_teacher_tf0_report(),
        teacher_tf0_file_sha256=SHA,
        training_regime="teacher_assisted",
        expected_commit=COMMIT,
        **_teacher_calibration_binding_kwargs(),
    )
    assert calibration_failed["passed"] is False
    assert any("calibration" in error for error in calibration_failed["errors"])
    drifted_tf0 = _teacher_tf0_report()
    drifted_tf0["validated_feature_pass_count"] = 29
    compatibility_failed = validate_prerequisites(
        resize_contract=_resize_contract_report(),
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=_teacher_t0_report(),
        teacher_calibration=_teacher_calibration_report(),
        teacher_calibration_file_sha256=SHA,
        teacher_tc0=_teacher_tc0_report(),
        teacher_tc0_file_sha256=SHA,
        teacher_tf0=drifted_tf0,
        teacher_tf0_file_sha256=SHA,
        training_regime="teacher_assisted",
        expected_commit=COMMIT,
        **_teacher_calibration_binding_kwargs(),
    )
    assert compatibility_failed["passed"] is False
    assert compatibility_failed["teacher_tf0_complete"] is False
    qa_only = validate_prerequisites(
        resize_contract=_resize_contract_report(),
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=None,
        teacher_calibration=None,
        teacher_calibration_file_sha256=None,
        training_regime="qa_only",
        expected_commit=COMMIT,
    )
    assert qa_only["passed"] is True
    assert qa_only["teacher_t0_required"] is False
    assert qa_only["teacher_t0_complete"] is None
    qa_with_teacher = validate_prerequisites(
        resize_contract=_resize_contract_report(),
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=None,
        teacher_calibration=None,
        teacher_calibration_file_sha256=None,
        teacher_tc0=_teacher_tc0_report(),
        teacher_tc0_file_sha256=SHA,
        training_regime="qa_only",
        expected_commit=COMMIT,
    )
    assert qa_with_teacher["passed"] is False
    assert any("qa_only" in error for error in qa_with_teacher["errors"])

    invalid_resize = _resize_contract_report()
    invalid_resize["resize_contract"] = "drifted"
    resize_failed = validate_prerequisites(
        resize_contract=invalid_resize,
        scorer_s0=_s0_report(),
        technical=_technical_report(),
        teacher_t0=None,
        teacher_calibration=None,
        teacher_calibration_file_sha256=None,
        training_regime="qa_only",
        expected_commit=COMMIT,
    )
    assert resize_failed["passed"] is False
    assert resize_failed["resize_r0_complete"] is False


def test_teacher_calibration_rejects_report_and_command_input_substitution() -> None:
    base = {
        "resize_contract": _resize_contract_report(),
        "scorer_s0": _s0_report(),
        "technical": _technical_report(),
        "teacher_t0": _teacher_t0_report(),
        "teacher_calibration_file_sha256": SHA,
        "teacher_tc0": _teacher_tc0_report(),
        "teacher_tc0_file_sha256": SHA,
        "teacher_tf0": _teacher_tf0_report(),
        "teacher_tf0_file_sha256": SHA,
        "training_regime": "teacher_assisted",
        "expected_commit": COMMIT,
        **_teacher_calibration_binding_kwargs(),
    }

    substituted_report = _teacher_calibration_report()
    substituted_report["train_sha256"] = "f" * 64
    report_result = validate_prerequisites(
        **base,
        teacher_calibration=substituted_report,
    )
    assert report_result["passed"] is False
    assert any("expected train_sha256" in error for error in report_result["errors"])

    legacy_typo = _teacher_calibration_report()
    legacy_typo["sample_selection"]["duplicate-semantic-after-states_retained"] = (
        legacy_typo["sample_selection"].pop("duplicate_semantic_after_states_retained")
    )
    typo_result = validate_prerequisites(
        **base,
        teacher_calibration=legacy_typo,
    )
    assert typo_result["passed"] is False
    assert any("sample selection" in error for error in typo_result["errors"])

    substituted_command = dict(base)
    substituted_command["teacher_calibration_train_sha256"] = "f" * 64
    command_result = validate_prerequisites(
        **substituted_command,
        teacher_calibration=_teacher_calibration_report(),
    )
    assert command_result["passed"] is False
    assert any("command binding" in error for error in command_result["errors"])


def test_micro_gradient_audit_requires_finite_positive_state_and_image_gradients() -> None:
    state = torch.tensor([2.0], requires_grad=True)
    image = state.square()
    image.retain_grad()
    loss = image.sum()
    loss.backward()
    accumulator: dict[str, list[float]] = {}
    audit_episode_gradients(
        [("final_state", state), ("query_image", image)],
        accumulator,
    )
    summary = gradient_audit_summary(accumulator, enabled=True, objective_stage="qa")
    assert summary["passed"] is True
    assert summary["categories"]["final_state"]["positive_finite"] == 1

    zero = torch.tensor([1.0], requires_grad=True)
    (zero * 0.0).sum().backward()
    with pytest.raises(RuntimeError, match="non-positive"):
        audit_episode_gradients([("final_state", zero)], {})


def test_score_provenance_binds_rows_report_checkpoint_and_lineage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        checkpoint = root / "checkpoint-000256.pt"
        checkpoint.write_bytes(b"checkpoint")
        (root / "summary.json").write_text(
            json.dumps(
                {
                    "optimizer_steps": 256,
                    "state_gradient_audit": {
                        "schema": "vision_memory.r3-state-gradient-audit.v1",
                        "enabled": True,
                        "objective_stage": "qa",
                        "passed": True,
                    },
                }
            ),
            encoding="utf-8",
        )
        metric_rows = [
            {
                "kind": "train",
                "optimizer_step": step,
                "choice_rotation_counts": [2, 2, 2, 2],
                "gradient_norm": 1.0,
                "state_gradient_audit": {"passed": True},
            }
            for step in range(1, 257)
        ]
        metric_rows.extend({"kind": "dev", "optimizer_step": step, "loss": 1.0} for step in range(64, 257, 32))
        (root / "metrics.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in metric_rows),
            encoding="utf-8",
        )
        for step in range(32, 257, 32):
            (root / f"checkpoint-{step:06d}.pt").write_bytes(b"checkpoint")
        predictions = root / "predictions.jsonl"
        row = {
            "checkpoint": str(checkpoint),
            "training_regime": "teacher_assisted",
            "parent_checkpoint_regime": "teacher_assisted",
            "teacher_control": "correct",
            "recurrence_mode": "direct_latent",
            "initial_state_mode": "blank",
            "seed": 0,
            "diffusion_seed": 0,
            "deterministic_ce": True,
        }
        predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")
        lineage = {
            "schema_version": 2,
            "training_regime": "teacher_assisted",
            "parent_checkpoint_regime": "teacher_assisted",
            "objective_stage": "qa",
            "reader_loss_mode": "listwise-choice",
            "choice_view_schedule": "cyclic4",
            "teacher_control": "correct",
            "teacher_control_sha256": SHA,
            "teacher_manifest_sha256": SHA,
            "teacher_sidecar_sha256": SHA,
            "teacher_calibration_sha256": SHA,
            "presentations_per_state": 512,
            "distill_presentations": 256,
            "qa_presentations": 256,
            "parent_checkpoint_sha256": SHA,
        }
        arguments = {
            "reader_loss_mode": "listwise-choice",
            "choice_view_schedule": "cyclic4",
            "recurrence_mode": "direct_latent",
            "detach_between_events": False,
            "noop_policy": "update",
            "initial_state_mode": "blank",
            "learn_initial_state": False,
            "lora_rank": 4,
            "seed": 0,
            "adapter_seed": 0,
            "strict_determinism": True,
            "audit_state_gradients": True,
            "disable_early_stopping": True,
            "require_mixed_delayed_probe": True,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "gradient_accumulation": 8,
            "gradient_clip": 1.0,
            "resolution": 1024,
            "checkpoint_unet": True,
            "curriculum": "full",
            "max_optimizer_steps": None,
            "max_train_episodes": 8,
            "epochs": 256,
            "presentations_per_state": 256,
            "distill_presentations": 256,
            "qa_presentations": 256,
            "checkpoint_every": 32,
            "eval_start_step": 64,
            "eval_every": 32,
            "eval_limit": 8,
        }
        companion = root / "predictions.jsonl.report.json"
        companion.write_text(
            json.dumps(
                {
                    "output_sha256": sha256_file(predictions),
                    "choice_view_family": "reverse-cyclic4",
                    "conditions": ["standard", "reset", "shuffle"],
                    "noop_policy": "keep",
                    "episodes_sha256": SHA,
                    "deterministic_ce": True,
                    "checkpoint_manifest": {
                        "training_lineage": lineage,
                        "arguments": arguments,
                        "strict_determinism": _s0_report()["strict_determinism"],
                        "model_snapshot_manifests": {
                            "dreamlite_mobile": "8" * 64,
                            "qwen_reader": "9" * 64,
                        },
                        "dev_sha256": SHA,
                    },
                }
            ),
            encoding="utf-8",
        )
        provenance = build_artifact_provenance(
            predictions=predictions,
            rows=[row],
            prediction_report=companion,
            suite="set8",
        )
        assert provenance["checkpoint_sha256"] == sha256_file(checkpoint)
        assert provenance["objective_stage"] == "qa"
        assert provenance["distill_presentations"] == 256


def _gate_report(path: str, *, payload_sha: str = SHA) -> dict:
    return {
        "schema_version": "vlm.r3.set8_gate.v1",
        "suite": "set8",
        "passed": True,
        "scientific_payload_sha256": payload_sha,
        "scientific_prediction_payload": {"sha256": "d" * 64},
        "artifact_provenance": {
            "checkpoint_path": path,
            "checkpoint_sha256": SHA,
            "training_regime": "qa_only",
            "objective_stage": "qa",
            "teacher_control": "none",
        },
    }


def test_replication_requires_two_fresh_paths_and_identical_scientific_payload() -> None:
    passed = validate_replication(
        _gate_report("/run/A/last.pt"),
        _gate_report("/run/B/last.pt"),
        suite="set8",
        training_regime="qa_only",
        teacher_control="none",
    )
    assert passed["passed"] is True
    with pytest.raises(ValueError, match="must be 'none'"):
        validate_replication(
            _gate_report("/run/A/last.pt"),
            _gate_report("/run/B/last.pt"),
            suite="set8",
            training_regime="qa_only",
            teacher_control="correct",
        )
    drift = validate_replication(
        _gate_report("/run/A/last.pt"),
        _gate_report("/run/B/last.pt", payload_sha="c" * 64),
        suite="set8",
        training_regime="qa_only",
        teacher_control="none",
    )
    assert drift["passed"] is False
    assert drift["bitwise_scientific_payload_match"] is False
    prediction_drift_b = _gate_report("/run/B/last.pt")
    prediction_drift_b["scientific_prediction_payload"]["sha256"] = "e" * 64
    prediction_drift = validate_replication(
        _gate_report("/run/A/last.pt"),
        prediction_drift_b,
        suite="set8",
        training_regime="qa_only",
        teacher_control="none",
    )
    assert prediction_drift["passed"] is False


def test_scientific_prediction_hash_excludes_latency_but_includes_choice_scores() -> None:
    row = {
        "episode_id": "r3-set8-r0-v0",
        "query_ordinal": 0,
        "probe_role": "delayed",
        "choice_view_family": "reverse-cyclic4",
        "choice_view_index": 0,
        "condition": "standard",
        "choices": ["a", "b", "c", "d"],
        "target_index": 0,
        "target_text": "a",
        "prediction_index": 0,
        "prediction_text": "a",
        "choice_mean_nll": [1.0, 2.0, 3.0, 4.0],
        "latency_seconds": 1.0,
    }
    first = scientific_prediction_payload([row])
    latency_changed = scientific_prediction_payload([{**row, "latency_seconds": 99.0}])
    score_changed = scientific_prediction_payload([{**row, "choice_mean_nll": [1.1, 2.0, 3.0, 4.0]}])
    assert first["sha256"] == latency_changed["sha256"]
    assert first["sha256"] != score_changed["sha256"]


def test_micro_dag_locks_tracks_controls_budgets_and_unlocks() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = _paths(root)
        set8 = _suite(root, "set8", 8)
        transition16 = _suite(root, "transition16", 16)
        stages = build_stages(paths, set8, transition16)
        by_name = {stage.name: stage for stage in stages}
        assert list(by_name) == [
            "QA8-A",
            "QA8-B",
            "QA16-A",
            "QA16-B",
            "TD8-CORRECT-A",
            "TD8-SHUFFLED-A",
            "TD8-RANDOM-A",
            "TD8-ATTRIBUTION-A",
            "TD8-CORRECT-B",
            "TD16-CORRECT-A",
            "TD16-CORRECT-B",
        ]
        assert by_name["QA8-B"].dependencies == ("QA8-A",)
        assert by_name["QA16-A"].dependencies == ("QA8-B",)
        assert by_name["TD8-ATTRIBUTION-A"].dependencies == (
            "TD8-CORRECT-A",
            "TD8-SHUFFLED-A",
            "TD8-RANDOM-A",
        )
        assert by_name["TD16-CORRECT-A"].dependencies == ("TD8-CORRECT-B",)

        qa8 = "\n".join(by_name["QA8-A"].commands)
        assert "--training-regime qa_only" in qa8
        assert "--epochs 512" in qa8
        assert "--presentations-per-state 512" in qa8
        assert "--eval-start-step 64" in qa8
        assert "--eval-every 32" in qa8
        assert "--choice-view-family reverse-cyclic4" in qa8
        assert "--require-mixed-delayed-probe" in qa8
        assert "--conditions standard reset shuffle" in qa8

        qa16 = "\n".join(by_name["QA16-A"].commands)
        assert "--eval-start-step 128" in qa16
        assert "--eval-every 64" in qa16
        assert "state_swap" in qa16

        correct = "\n".join(by_name["TD8-CORRECT-A"].commands)
        assert "--objective-stage distill" in correct
        assert "--epochs 256" in correct
        assert "--distill-presentations 256" in correct
        assert "--objective-stage qa" in correct
        assert "--qa-presentations 256" in correct
        assert "--initialize-from" in correct
        assert "--expected-teacher-control correct" in correct
        shuffled = "\n".join(by_name["TD8-SHUFFLED-A"].commands)
        random = "\n".join(by_name["TD8-RANDOM-A"].commands)
        assert "--teacher-control shuffled" in shuffled
        assert "--no-fail-on-gate" in shuffled
        assert "--teacher-control random-moment-matched" in random
        attribution = "\n".join(by_name["TD8-ATTRIBUTION-A"].commands)
        assert "score_r3_teacher_attribution.py" in attribution
        assert "--random-qa-gate" in attribution

        text = render_stage_sbatch(
            by_name["QA8-A"],
            paths=paths,
            expected_commit=COMMIT,
            expected_torch="2.7.1+cu118",
            resize_contract_report_sha256=SHA,
            scorer_s0_report_sha256=SHA,
            technical_report_sha256=SHA,
            teacher_t0_report_sha256=SHA,
        )
        assert "#SBATCH --nodes=1" in text
        assert "#SBATCH --gres=gpu:2" in text
        assert "#SBATCH --partition=a800" in text
        assert "validate_r3_micro_prerequisites.py" in text
        assert "--resize-contract-report" in text
        assert "--resize-contract-report-sha256" in text
        assert "--training-regime qa_only" in text
        assert "--teacher-t0-report" not in text
        assert "transitions.jsonl" not in text
        assert "R3_SUBMISSION_SUPPORTED=0" in text

        teacher_text = render_stage_sbatch(
            by_name["TD8-CORRECT-A"],
            paths=paths,
            expected_commit=COMMIT,
            expected_torch="2.7.1+cu118",
            resize_contract_report_sha256=SHA,
            scorer_s0_report_sha256=SHA,
            technical_report_sha256=SHA,
            teacher_t0_report_sha256=SHA,
        )
        assert "--training-regime teacher_assisted" in teacher_text
        assert "--teacher-t0-report" in teacher_text
        assert "transitions.jsonl" in teacher_text


def test_materialized_plan_is_template_only_and_has_separate_unlocks() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paths = _paths(root)
        set8 = _suite(root, "set8", 8)
        transition16 = _suite(root, "transition16", 16)
        stages = build_stages(paths, set8, transition16)
        manifest = materialize_dry_run(
            paths=paths,
            stages=stages,
            set8=set8,
            transition16=transition16,
            expected_commit=COMMIT,
            expected_torch="2.7.1+cu118",
            resize_contract_report_sha256=SHA,
            scorer_s0_report_sha256=SHA,
            technical_report_sha256=SHA,
            teacher_t0_report_sha256=SHA,
        )
        assert manifest["dry_run"] is True
        assert manifest["submission_supported"] is False
        assert manifest["protocol"] == "R3-Set8-Transition16-micro-resize-dry-run-v2"
        assert manifest["unlock_rules"]["cross_track_substitution"] is False
        assert manifest["jobs"]["QA8-A"]["hard_prerequisites"][0] == "R3-R0"
        assert "T0" not in manifest["jobs"]["QA8-A"]["hard_prerequisites"]
        assert "T0" in manifest["jobs"]["TD8-CORRECT-A"]["hard_prerequisites"]
        assert manifest["fixed_protocol"]["qa_only_presentations_per_state"] == 512
        assert manifest["fixed_protocol"]["teacher_distill_presentations_per_state"] == 256
        assert all(job["nodes"] == 1 and job["gpus_per_node"] == 2 for job in manifest["jobs"].values())
        assert len(list(paths.sbatch.glob("*.sbatch"))) == len(stages)
