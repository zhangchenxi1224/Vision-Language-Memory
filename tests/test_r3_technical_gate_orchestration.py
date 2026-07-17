from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "scripts" / "probes"
CLUSTER = ROOT / "scripts" / "cluster"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PROBES))
sys.path.insert(0, str(CLUSTER))

from render_r3_technical_gates import (  # noqa: E402
    R3Paths,
    build_gates,
    materialize_dry_run,
    render_gate_sbatch,
    render_strict_chain_sbatch,
)
from validate_r3_resume_equivalence import (  # noqa: E402
    EXPECTED_ARGUMENTS,
    validate_checkpoint_paths,
    validate_resume_checkpoints,
)
from validate_r3_technical_gates import (  # noqa: E402
    CHOICES,
    FIXTURE_RGB_SHA256,
    GATE_PROTOCOLS,
    OVERWRITE_EVENT,
    QUERY,
    SET_EVENT,
    validate_reports,
)
from vision_memory.reader import R3_QWEN_READER_RESIZE_CONTRACT  # noqa: E402


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tensor_summary(shape: list[int], *, dtype: str, digest: str) -> dict:
    return {
        "shape": shape,
        "dtype": dtype,
        "device": "cuda:0",
        "sha256": digest,
        "finite": True,
    }


def resize_contract_report(*, commit: str = "a" * 40) -> dict:
    dtype_reports = {}
    for index, (name, dtype) in enumerate(
        (("float32", "torch.float32"), ("bfloat16", "torch.bfloat16")),
        start=1,
    ):
        input_sha = str(index) * 64
        resized_sha = str(index + 2) * 64
        pixels_sha = str(index + 4) * 64
        gradient_sha = str(index + 6) * 64
        forward = {
            "passed": True,
            "input": _tensor_summary([3, 1024, 1024], dtype=dtype, digest=input_sha),
            "resized": _tensor_summary([3, 256, 256], dtype=dtype, digest=resized_sha),
            "legacy_pixel_values": _tensor_summary(
                [256, 1536], dtype="torch.float32", digest=pixels_sha
            ),
            "candidate_pixel_values": _tensor_summary(
                [256, 1536], dtype="torch.float32", digest=pixels_sha
            ),
            "pixel_values_torch_equal": True,
            "pixel_values_max_absolute_difference": 0.0,
            "expected_pixel_values_shape": [256, 1536],
            "pixel_values_shape_locked": True,
            "legacy_grid_thw": [1, 16, 16],
            "candidate_grid_thw": [1, 16, 16],
            "expected_grid_thw": [1, 16, 16],
            "grid_torch_equal_and_locked": True,
        }
        run = {
            "loss": 1.0,
            "loss_float_hex": "0x1.0000000000000p+0",
            "pixel_values_sha256": pixels_sha,
            "gradient": _tensor_summary([3, 1024, 1024], dtype=dtype, digest=gradient_sha),
            "gradient_norm": 0.5,
        }
        backward = {
            "passed": True,
            "first": copy.deepcopy(run),
            "second": copy.deepcopy(run),
            "gradient_finite": True,
            "gradient_nonzero": True,
            "gradient_torch_equal": True,
            "gradient_max_absolute_difference": 0.0,
            "loss_bitwise_equal": True,
        }
        output_gradient_sha = ("9" if index == 1 else "a") * 64
        adjoint_gradient_sha = ("b" if index == 1 else "c") * 64
        adjoint = {
            "passed": True,
            "reference": "native-torchvision-cpu-fp32-autograd",
            "output_gradient": _tensor_summary(
                [3, 256, 256], dtype=dtype, digest=output_gradient_sha
            ),
            "candidate_gradient": _tensor_summary(
                [3, 1024, 1024], dtype=dtype, digest=adjoint_gradient_sha
            ),
            "reference_gradient": _tensor_summary(
                [3, 1024, 1024], dtype=dtype, digest=adjoint_gradient_sha
            ),
            "gradient_finite": True,
            "gradient_nonzero": True,
            "gradient_torch_equal": True,
            "gradient_max_absolute_difference": 0.0,
            "candidate_gradient_norm": 0.5,
            "reference_gradient_norm": 0.5,
        }
        native_run = {
            "loss": 1.0,
            "loss_float_hex": "0x1.0000000000000p+0",
            "pixel_values_sha256": pixels_sha,
            "gradient": _tensor_summary(
                [3, 1024, 1024], dtype=dtype, digest=adjoint_gradient_sha
            ),
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
        legacy_native = {
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
            "candidate": {
                "gradient": _tensor_summary(
                    [3, 1024, 1024], dtype=dtype, digest=adjoint_gradient_sha
                ),
                "gradient_norm": 0.5,
            },
            "native_runs": [dict(native_run) for _ in range(3)],
        }
        dtype_reports[name] = {
            "passed": True,
            "forward_equivalence": forward,
            "strict_backward_repeat": backward,
            "cpu_adjoint_reference": adjoint,
            "legacy_native_cuda_reference": legacy_native,
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
                "fast_tensor_processor": True,
                "resize_enabled_by_default": True,
                "min_pixels_locked": True,
                "max_pixels_locked": True,
                "patch_size_locked": True,
                "temporal_patch_size_locked": True,
                "merge_size_locked": True,
                "bicubic_resample_locked": True,
                "callable": True,
            },
        },
        "dtypes": dtype_reports,
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
            "device_total_memory_bytes": 143_167 * 1024**2,
        },
        "execution_binding": {
            "passed": True,
            "stage": "r3-r0",
            "infrastructure_stage": False,
            "git_commit": commit,
            "worker_input_path": "/runs/r3-r0/worker_input.json",
            "worker_input_sha256": "d" * 64,
            "formal_preflight_path": "/runs/preflight/r3_h200_formal.json",
            "formal_preflight_sha256": "e" * 64,
        },
        "provenance": {
            "git": {"commit": commit, "clean": True},
            "models": {
                "reader": {
                    "expected_revision": "2" * 40,
                    "observed_revision": "2" * 40,
                    "revision_matches_lock": True,
                }
            },
        },
    }


def scorer_contract_report(*, commit: str = "a" * 40) -> dict:
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
        "strict_determinism": copy.deepcopy(resize_contract_report()["strict_determinism"]),
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
                "peak_allocated_gib": 8.0,
                "peak_reserved_gib": 9.0,
            }
        },
        "provenance": {
            "git": {"commit": commit, "clean": True},
            "runtime": {"torch": "2.7.0a0+ecf3bae40a.nv25.02", "cuda_runtime": "12.8"},
            "models": {
                "reader": {
                    "expected_revision": "2" * 40,
                    "observed_revision": "2" * 40,
                    "revision_matches_lock": True,
                }
            },
        },
    }


def probe_report(gate: str, *, loss: float = 1.25) -> dict:
    protocol = GATE_PROTOCOLS[gate]
    events = list(protocol["events"])
    source = {
        "origin": "deterministic_fixture",
        "fixture_id": "vision-memory-rgb-blocks-v1-1024",
        "path": None,
        "file_sha256": None,
        "mode": "RGB",
        "size": [1024, 1024],
        "rgb_sha256": FIXTURE_RGB_SHA256,
    }
    git = {"commit": "a" * 40, "clean": True, "status_sha256": hashlib.sha256(b"").hexdigest()}
    models = {
        "dreamlite": {
            "path": "/models/DreamLite-mobile",
            "repo_id": "dreamlite/DreamLite-mobile",
            "expected_revision": "1" * 40,
            "observed_revision": "1" * 40,
            "revision_matches_lock": True,
        },
        "reader": {
            "path": "/models/Qwen3-VL-4B-Instruct",
            "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
            "expected_revision": "2" * 40,
            "observed_revision": "2" * 40,
            "revision_matches_lock": True,
        },
    }
    strict_determinism = {
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
    }
    metadata = {
        "schema_version": 1,
        "git": git,
        "models": models,
        "source_image": source,
        "event": events,
        "query": QUERY,
        "reader_loss_mode": "listwise-choice",
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "target": None,
        "choices": list(CHOICES),
        "target_index": protocol["target_index"],
        "resolution": 1024,
        "adapter_seed": 0,
        "event_noise_seeds": list(range(len(events))),
        "lora_rank": 4,
        "checkpoint_unet": True,
        "dreamlite_device": "cuda:0",
        "reader_device": "cuda:1",
        "updater_dtype": "torch.bfloat16",
        "reader_dtype": "torch.bfloat16",
        "strict_determinism": strict_determinism,
    }
    frozen = {
        name: {
            "parameter_tensors": 10,
            "trainable_parameter_tensors": 2 if name == "base_unet" else 0,
            "frozen_parameter_tensors": 8 if name == "base_unet" else 10,
            "frozen_tensors_with_grad": 0,
            "frozen_nonfinite_grad_elements": 0,
        }
        for name in ("base_unet", "vae", "internal_qwen", "reader")
    }
    intermediate = []
    if len(events) == 2:
        intermediate = [
            {
                "norm": None if protocol["detach_between_events"] else 0.125,
                "nonfinite_elements": None if protocol["detach_between_events"] else 0,
            }
        ]
    return {
        "probe": "e2e_episode_grad",
        "events": len(events),
        "detach_between_events": protocol["detach_between_events"],
        "pair_id": canonical_sha256(metadata),
        "pair_metadata": metadata,
        "loss": loss,
        "reader_loss_mode": "listwise-choice",
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "strict_determinism": strict_determinism,
        "choice_mean_nll": [1.0, 2.0, 3.0, 4.0],
        "final_state_shape": [1, 4, 128, 128],
        "final_state_sha256": "f" * 64,
        "final_state_gradient": {"norm": 0.75, "nonfinite_elements": 0},
        "intermediate_gradients": intermediate,
        "lora_grad_norm": 0.25,
        "lora_tensors_with_grad": 8,
        "lora_nonfinite_elements": 0,
        "unclamped_image_grad_norm": 0.5,
        "updater_device": "cuda:0",
        "reader_device": "cuda:1",
        "frozen_gradients": frozen,
        "cuda_peak_memory": {
            device: {"name": "NVIDIA H200", "peak_allocated_gib": 1.0, "peak_reserved_gib": 2.0}
            for device in ("cuda:0", "cuda:1")
        },
        "provenance": {
            "git": git,
            "models": models,
            "runtime": {"torch": "2.7.0a0+ecf3bae40a.nv25.02", "cuda_runtime": "12.8"},
            "source_image": source,
        },
    }


def resume_report() -> dict:
    return {
        "schema_version": 2,
        "protocol": "DL-S-common-prefix-16-vs-8-resume-8-next-step-v2",
        "git_commit": "a" * 40,
        "dreamlite_revision": "1" * 40,
        "reader_revision": "2" * 40,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "runtime_environment": {
            "python": "3.12.3",
            "torch": "2.7.0a0+ecf3bae40a.nv25.02",
            "torchvision": "0.22.0a0",
            "cuda_runtime": "12.8",
            "diffusers": "0.39.0",
            "transformers": "4.57.3",
            "peft": "0.18.1",
        },
        "presentations": {"uninterrupted": 16, "shared_prefix": 8, "resumed_suffix": 8, "next_step": 17},
        "atol": 0.0,
        "rtol": 0.0,
        "exact": True,
        "mismatch_count": 0,
        "checkpoint_state": {
            "prefix": {"epoch": 0, "episode_cursor": 8, "optimizer_step": 8},
            "reference": {"epoch": 0, "episode_cursor": 16, "optimizer_step": 16},
            "resumed": {"epoch": 0, "episode_cursor": 16, "optimizer_step": 16},
        },
        "next_checkpoint_state": {
            "reference": {"epoch": 1, "episode_cursor": 1, "optimizer_step": 17},
            "resumed": {"epoch": 1, "episode_cursor": 1, "optimizer_step": 17},
        },
        "next_step_metric": {
            "loss_hex": "0x1.0000000000000p+0",
            "gradient_norm_hex": "0x1.0000000000000p+0",
            "raw_gradient_sha256": "4" * 64,
            "clipped_gradient_sha256": "5" * 64,
        },
        "lineage": {
            "training_regime": "qa_only",
            "reader_loss_mode": "listwise-choice",
            "qa_supervision": "listwise-choice",
            "choice_view_schedule": "cyclic4",
        },
        "checkpoint_paths": {
            "prefix": "/runs/reference_16/checkpoint-000008.pt",
            "reference": "/runs/reference_16/checkpoint-000016.pt",
            "resumed": "/runs/resumed_from_8/checkpoint-000016.pt",
            "reference_next": "/runs/reference_16/last.pt",
            "resumed_next": "/runs/resumed_from_8/last.pt",
        },
        "checkpoint_sha256": {
            "prefix": "1" * 64,
            "reference": "2" * 64,
            "resumed": "3" * 64,
            "reference_next": "6" * 64,
            "resumed_next": "7" * 64,
        },
        "passed": True,
    }


def checkpoint(*, cursor: int, weight: float, epoch: int = 0, optimizer_step: int | None = None) -> dict:
    step = cursor if optimizer_step is None else optimizer_step
    manifest = {
        "git_dirty": False,
        "git_commit": "c" * 40,
        "dreamlite_revision": "d" * 40,
        "reader_revision": "e" * 40,
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "training_lineage": {
            "training_regime": "qa_only",
            "reader_loss_mode": "listwise-choice",
            "qa_supervision": "listwise-choice",
            "choice_view_schedule": "cyclic4",
        },
        "arguments": dict(EXPECTED_ARGUMENTS),
        "environment": {
            "python": "3.12.3",
            "torch": "2.7.0a0+ecf3bae40a.nv25.02",
            "torchvision": "0.22.0a0",
            "cuda_runtime": "12.8",
            "diffusers": "0.39.0",
            "transformers": "4.57.3",
            "peft": "0.18.1",
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
        "train_sha256": "a" * 64,
        "dev_sha256": "b" * 64,
        "initial_image": {
            "initial_state_mode": "blank",
            "origin": "blank_fixture",
            "mode": "RGB",
            "size": [1024, 1024],
        },
    }
    return {
        "schema_version": 1,
        "trainable_state": {"weight": torch.tensor([weight])},
        "optimizer": {"state": {0: {"step": torch.tensor(cursor)}}},
        "epoch": epoch,
        "episode_cursor": cursor,
        "optimizer_step": step,
        "rng_state": {"torch": torch.tensor([step], dtype=torch.uint8)},
        "manifest": manifest,
        "trainer_state": {"best_dev": float("inf"), "stale_evals": 0},
    }


class R3TechnicalValidationTest(unittest.TestCase):
    def test_full_listwise_gate_sequence_passes(self):
        g4 = probe_report("G4-L")
        g5 = probe_report("G5-L")
        g6 = probe_report("G6-L")
        report = validate_reports(
            through="DL-S",
            resize_contract=resize_contract_report(),
            scorer_s0=scorer_contract_report(),
            g4=g4,
            g5=g5,
            g6=g6,
            resume=resume_report(),
        )
        self.assertTrue(report["passed"], report["errors"])
        self.assertEqual(
            report["required_gates"],
            ["R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L", "DL-S"],
        )
        self.assertTrue(report["checks"]["R3-R0"]["backward_bitwise_repeatable"])
        self.assertEqual(report["checks"]["G4-L"]["semantic_operations"], ["set"])
        self.assertEqual(report["checks"]["G5-L"]["semantic_operations"], ["set", "overwrite"])
        self.assertTrue(report["checks"]["G5-L/G6-L-pair"]["valid"])

    def test_non_listwise_or_semantically_drifted_gate_fails_closed(self):
        g4 = probe_report("G4-L")
        g4["reader_loss_mode"] = "legacy-target-only"
        report = validate_reports(
            through="G4-L",
            resize_contract=resize_contract_report(),
            g4=g4,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("listwise-choice" in error for error in report["errors"]))

        g5 = probe_report("G5-L")
        g5["pair_metadata"]["event"][1] = "The user also likes blue mugs."
        g5["pair_id"] = canonical_sha256(g5["pair_metadata"])
        report = validate_reports(
            through="G5-L",
            resize_contract=resize_contract_report(),
            g4=probe_report("G4-L"),
            g5=g5,
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("pair_metadata.event" in error for error in report["errors"]))

    def test_resize_contract_is_a_required_fail_closed_first_gate(self):
        missing = validate_reports(through="G4-L", g4=probe_report("G4-L"))
        self.assertFalse(missing["passed"])
        self.assertTrue(any("R3-R0: required report" in error for error in missing["errors"]))

        drifted = resize_contract_report()
        drifted["dtypes"]["bfloat16"]["strict_backward_repeat"]["gradient_torch_equal"] = False
        invalid = validate_reports(
            through="G4-L",
            resize_contract=drifted,
            g4=probe_report("G4-L"),
        )
        self.assertFalse(invalid["passed"])
        self.assertTrue(any("gradient_torch_equal" in error for error in invalid["errors"]))

        drifted_adjoint = resize_contract_report()
        drifted_adjoint["dtypes"]["float32"]["cpu_adjoint_reference"][
            "gradient_torch_equal"
        ] = False
        invalid_adjoint = validate_reports(
            through="R3-R0",
            resize_contract=drifted_adjoint,
        )
        self.assertFalse(invalid_adjoint["passed"])

        drifted_native = resize_contract_report()
        drifted_native["dtypes"]["bfloat16"]["legacy_native_cuda_reference"][
            "candidate_relative_l2_max"
        ] = 0.02
        invalid_native = validate_reports(through="R3-R0", resize_contract=drifted_native)
        self.assertFalse(invalid_native["passed"])

        drifted_runtime = resize_contract_report()
        drifted_runtime["runtime"]["device_name"] = "NVIDIA A800-SXM4-80GB"
        invalid_runtime = validate_reports(through="R3-R0", resize_contract=drifted_runtime)
        self.assertFalse(invalid_runtime["passed"])

        boundary_runtime = resize_contract_report()
        boundary_runtime["runtime"]["device_total_memory_bytes"] = 140_000 * 1024**2
        self.assertTrue(validate_reports(through="R3-R0", resize_contract=boundary_runtime)["passed"])
        too_small_runtime = resize_contract_report()
        too_small_runtime["runtime"]["device_total_memory_bytes"] = 139_999 * 1024**2
        invalid_memory = validate_reports(through="R3-R0", resize_contract=too_small_runtime)
        self.assertFalse(invalid_memory["passed"])
        self.assertTrue(any("140000 MiB" in error for error in invalid_memory["errors"]))

        drifted_binding = resize_contract_report()
        drifted_binding["execution_binding"]["formal_preflight_sha256"] = "z" * 64
        invalid_binding = validate_reports(through="R3-R0", resize_contract=drifted_binding)
        self.assertFalse(invalid_binding["passed"])

        drifted_processor = resize_contract_report()
        drifted_processor["processor"]["observed"]["max_pixels"] = 16777216
        invalid_processor = validate_reports(through="R3-R0", resize_contract=drifted_processor)
        self.assertFalse(invalid_processor["passed"])

        mismatched_commit = validate_reports(
            through="G4-L",
            resize_contract=resize_contract_report(commit="b" * 40),
            g4=probe_report("G4-L"),
        )
        self.assertFalse(mismatched_commit["passed"])
        self.assertIn(
            "technical probe reports do not share one clean Git commit",
            mismatched_commit["errors"],
        )

        mismatched_dl_s = resume_report()
        mismatched_dl_s["git_commit"] = "b" * 40
        invalid_dl_s = validate_reports(
            through="DL-S",
            resize_contract=resize_contract_report(),
            scorer_s0=scorer_contract_report(),
            g4=probe_report("G4-L"),
            g5=probe_report("G5-L"),
            g6=probe_report("G6-L"),
            resume=mismatched_dl_s,
        )
        self.assertFalse(invalid_dl_s["passed"])
        self.assertIn(
            "technical probe reports do not share one clean Git commit",
            invalid_dl_s["errors"],
        )

        mismatched_reader = resume_report()
        mismatched_reader["reader_revision"] = "f" * 40
        invalid_reader = validate_reports(
            through="DL-S",
            resize_contract=resize_contract_report(),
            scorer_s0=scorer_contract_report(),
            g4=probe_report("G4-L"),
            g5=probe_report("G5-L"),
            g6=probe_report("G6-L"),
            resume=mismatched_reader,
        )
        self.assertFalse(invalid_reader["passed"])
        self.assertIn(
            "technical probe reports do not share one locked Reader revision",
            invalid_reader["errors"],
        )

        mismatched_dreamlite = resume_report()
        mismatched_dreamlite["dreamlite_revision"] = "f" * 40
        invalid_dreamlite = validate_reports(
            through="DL-S",
            resize_contract=resize_contract_report(),
            scorer_s0=scorer_contract_report(),
            g4=probe_report("G4-L"),
            g5=probe_report("G5-L"),
            g6=probe_report("G6-L"),
            resume=mismatched_dreamlite,
        )
        self.assertFalse(invalid_dreamlite["passed"])
        self.assertIn(
            "technical probe reports do not share one locked DreamLite revision",
            invalid_dreamlite["errors"],
        )

    def test_resume_equivalence_is_bitwise_and_lineage_locked(self):
        prefix = checkpoint(cursor=8, weight=1.0)
        reference = checkpoint(cursor=16, weight=2.0)
        resumed = copy.deepcopy(reference)
        next_reference = checkpoint(cursor=1, weight=3.0, epoch=1, optimizer_step=17)
        next_resumed = copy.deepcopy(next_reference)
        metric = {
            "loss_hex": "0x1.0p+0",
            "gradient_norm_hex": "0x1.0p+1",
            "raw_gradient_sha256": "a" * 64,
            "clipped_gradient_sha256": "b" * 64,
        }
        report = validate_resume_checkpoints(
            prefix,
            reference,
            resumed,
            next_reference,
            next_resumed,
            metric,
            metric,
        )
        self.assertTrue(report["passed"], report["errors"])
        self.assertTrue(report["exact"])
        self.assertEqual(report["mismatch_count"], 0)

        resumed["trainable_state"]["weight"] += 1.0
        drift = validate_resume_checkpoints(
            prefix,
            reference,
            resumed,
            next_reference,
            next_resumed,
            metric,
            metric,
        )
        self.assertFalse(drift["passed"])
        self.assertFalse(drift["exact"])
        self.assertTrue(any("max_abs_difference" in error for error in drift["errors"]))

    def test_resume_paths_must_encode_the_common_prefix_fork(self):
        root = Path("/runs")
        self.assertEqual(
            validate_checkpoint_paths(
                root / "reference_16" / "checkpoint-000008.pt",
                root / "reference_16" / "checkpoint-000016.pt",
                root / "resumed_from_8" / "checkpoint-000016.pt",
                root / "reference_16" / "last.pt",
                root / "resumed_from_8" / "last.pt",
            ),
            [],
        )
        errors = validate_checkpoint_paths(
            root / "unrelated" / "prefix.pt",
            root / "reference_16" / "checkpoint-000016.pt",
            root / "reference_16" / "checkpoint-000016.pt",
            root / "other" / "next.pt",
            root / "reference_16" / "last.pt",
        )
        self.assertGreaterEqual(len(errors), 3)


class R3TechnicalRenderingTest(unittest.TestCase):
    def paths(self, root: Path) -> R3Paths:
        return R3Paths(
            project=root / "project",
            environment=root / "environment",
            model_root=root / "models",
            train=root / "data" / "train.jsonl",
            dev=root / "data" / "dev.jsonl",
            run_root=root / "runs" / "r3",
        )

    def test_gate_commands_lock_semantics_listwise_resume_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(Path(directory))
            gates = build_gates(
                paths,
                expected_train_sha256="a" * 64,
                expected_dev_sha256="b" * 64,
            )
            self.assertEqual(
                [gate.name for gate in gates],
                ["R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L", "DL-S"],
            )
            self.assertEqual(
                [gate.dependency for gate in gates],
                [None, "R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L"],
            )

            r0_commands = "\n".join(gates[0].commands)
            self.assertIn("qwen_resize_contract.py", r0_commands)
            self.assertIn("--through R3-R0", r0_commands)
            self.assertIn("--resize-contract", r0_commands)
            self.assertIn("qwen_scorer_contract.py", "\n".join(gates[1].commands))
            probe_commands = "\n".join(command for gate in gates[2:5] for command in gate.commands)
            self.assertEqual(probe_commands.count("--reader-loss-mode listwise-choice"), 3)
            self.assertEqual(probe_commands.count("--resize-contract"), 3)
            self.assertNotIn("legacy-target-only", probe_commands)
            self.assertIn(SET_EVENT, probe_commands)
            self.assertIn(OVERWRITE_EVENT, probe_commands)
            self.assertIn("--detach-between-events", "\n".join(gates[4].commands))

            resume_commands = "\n".join(gates[5].commands)
            self.assertIn("--max-train-episodes 16", resume_commands)
            self.assertIn("--max-optimizer-steps 17", resume_commands)
            self.assertIn("--audit-gradient-sha", resume_commands)
            self.assertIn("--strict-determinism", resume_commands)
            self.assertIn("--require-mixed-delayed-probe", resume_commands)
            self.assertIn("--gradient-accumulation 1", resume_commands)
            self.assertIn("checkpoint-000008.pt", resume_commands)
            self.assertIn("--resume", resume_commands)
            self.assertIn("validate_r3_resume_equivalence.py", resume_commands)

    def test_every_template_is_one_node_two_a800_and_chain_is_fail_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(Path(directory))
            gates = build_gates(
                paths,
                expected_train_sha256="a" * 64,
                expected_dev_sha256="b" * 64,
            )
            for gate in gates:
                rendered = render_gate_sbatch(
                    gate,
                    paths=paths,
                    expected_commit="c" * 40,
                    expected_torch="2.7.1+cu118",
                )
                self.assertIn("#SBATCH --nodes=1", rendered)
                self.assertIn("#SBATCH --gres=gpu:2", rendered)
                self.assertIn("#SBATCH --partition=a800", rendered)
                self.assertIn("set -euo pipefail", rendered)
                self.assertIn("--min-gpus 2", rendered)
                self.assertIn("A800", rendered)
                self.assertNotIn("sbatch --parsable", rendered)

            chain = render_strict_chain_sbatch(
                gates,
                paths=paths,
                expected_commit="c" * 40,
                expected_torch="2.7.1+cu118",
            )
            offsets = [
                chain.index(f"# BEGIN {name}")
                for name in ("R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L", "DL-S")
            ]
            self.assertEqual(offsets, sorted(offsets))
            self.assertIn("R3_STRICT_SERIAL_FAIL_STOP=1", chain)
            self.assertIn("export PYTHONHASHSEED=0", chain)
            self.assertIn("export CUBLAS_WORKSPACE_CONFIG=:4096:8", chain)

    def test_materialization_is_template_only_and_never_submits(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.paths(Path(directory))
            gates = build_gates(
                paths,
                expected_train_sha256="a" * 64,
                expected_dev_sha256="b" * 64,
            )
            manifest = materialize_dry_run(
                paths=paths,
                gates=gates,
                expected_commit="d" * 40,
                expected_torch="2.7.1+cu118",
                expected_train_sha256="a" * 64,
                expected_dev_sha256="b" * 64,
            )
            self.assertTrue(manifest["dry_run"])
            self.assertFalse(manifest["submission_supported"])
            self.assertTrue(all(job["job_id"] is None for job in manifest["jobs"].values()))
            self.assertTrue((paths.sbatch / "R3_strict_chain.sbatch").is_file())
            self.assertTrue((paths.run_root / "dry_run_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
