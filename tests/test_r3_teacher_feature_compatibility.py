from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

from scripts.probes import r3_teacher_feature_compatibility as probe  # noqa: E402
from scripts.probes import validate_r3_teacher_feature_compatibility as validator  # noqa: E402
from vision_memory.repro import REQUIRED_DETERMINISM_ENV, canonical_json_sha256  # noqa: E402
from vision_memory.teacher import (  # noqa: E402
    TeacherArtifactRecord,
    TeacherCacheManifest,
    TeacherState,
    TeacherTransitionRecord,
    file_sha256,
    save_teacher_manifest,
    save_teacher_sidecar,
    save_teacher_tensor,
)


READER_REVISION = "e" * 40
CACHE_COMMIT = "d" * 40
SCIENCE_COMMIT = "a" * 40


def _teacher() -> TeacherState:
    return TeacherState(
        state_id="1" * 64,
        teacher_key="2" * 64,
        semantic_state_sha256="3" * 64,
        teacher_contract_sha256="4" * 64,
        renderer_contract_sha256="5" * 64,
        image=torch.zeros(1, 3, 1024, 1024, dtype=torch.bfloat16),
        latent=torch.arange(256, dtype=torch.float32).reshape(1, 4, 8, 8) / 255.0,
        feature=(torch.arange(18, dtype=torch.float32).reshape(1, 3, 6) / 17.0).to(torch.bfloat16),
    )


def _build_cache(root: Path) -> tuple[dict, TeacherState]:
    teacher = _teacher()
    record = TeacherArtifactRecord.from_teacher_state(teacher)
    manifest = TeacherCacheManifest(
        teacher_contract_sha256=teacher.teacher_contract_sha256,
        renderer_contract_sha256=teacher.renderer_contract_sha256,
        records=(record,),
    )
    transition = TeacherTransitionRecord(
        episode_id="episode-0",
        turn_id=0,
        before_state_id=teacher.state_id,
        after_state_id=teacher.state_id,
        event_kind="noop",
        teacher_key=teacher.teacher_key,
    )
    file_map: dict[str, str] = {}
    for tensor, specification in (
        (teacher.image, record.image),
        (teacher.latent, record.latent),
        (teacher.feature, record.feature),
    ):
        file_map[specification.relative_path] = save_teacher_tensor(
            root / specification.relative_path,
            tensor,
            specification=specification,
        )
    transitions_sha = save_teacher_sidecar(root / "transitions.jsonl", (transition,), manifest=manifest)
    manifest_sha = save_teacher_manifest(root / "manifest.json", manifest)
    build_report = {
        "schema": probe.tc0.BUILD_REPORT_SCHEMA,
        "git_commit": CACHE_COMMIT,
        "model_revisions": {"dreamlite": "b" * 40, "qwen_reader": READER_REVISION},
        "state_count": 1,
        "transition_count": 1,
        "manifest_sha256": manifest_sha,
        "sidecar_sha256": transitions_sha,
        "artifact_file_sha256": dict(sorted(file_map.items())),
    }
    build_path = root / "build_report.json"
    build_path.write_text(json.dumps(build_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    content_map = {
        specification.relative_path: specification.sha256
        for specification in (record.image, record.latent, record.feature)
    }
    lock = {
        "state_count": 1,
        "transition_count": 1,
        "artifact_tensor_count": 3,
        "manifest_sha256": manifest_sha,
        "transitions_sha256": transitions_sha,
        "build_report_sha256": file_sha256(build_path),
        "artifact_file_sha_map_sha256": canonical_json_sha256(file_map),
        "tensor_content_sha_map_sha256": canonical_json_sha256(content_map),
    }
    return lock, teacher


def _tc0_validation(path: Path, preregistration_sha256: str) -> str:
    report = {
        "schema_version": 1,
        "protocol": probe.TC0_VALIDATION_PROTOCOL,
        "expected_commit": SCIENCE_COMMIT,
        "reader_revision": READER_REVISION,
        "reader_resize_contract": probe.R3_QWEN_READER_RESIZE_CONTRACT,
        "preregistration_sha256": preregistration_sha256,
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
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_sha256(path)


def _locks(lock: dict) -> dict:
    return {
        "preregistration_path": "/locked/r3_preregistration.json",
        "preregistration_sha256": "6" * 64,
        "preregistration_schema": probe.tc0.PREREGISTRATION_SCHEMA,
        "reader_revision": READER_REVISION,
        "reader_resize_contract": probe.R3_QWEN_READER_RESIZE_CONTRACT,
        "cache_build_commit": CACHE_COMMIT,
        "gpu_model": "MockH200",
        "transformers_version": "4.57.3",
        "torchvision_version": "0.22.0a0",
        "torch_version": "2.7.0a0+mock",
        "cuda_version": "12.8",
        "suites": {"set8": lock, "transition16": lock},
        "feature_backend_gate": probe.LOCKED_FEATURE_GATE,
        "feature_backend_gate_sha256": canonical_json_sha256(probe.LOCKED_FEATURE_GATE),
    }


def _raw_report(suite: dict, locks: dict, tc0_sha: str) -> dict:
    second = copy.deepcopy(suite)
    second["suite"] = "transition16"
    second["cache_root"] = "/immutable/transition16"
    return {
        "schema_version": probe.PROBE_SCHEMA_VERSION,
        "probe": probe.PROBE_NAME,
        "reader_revision": READER_REVISION,
        "reader_resize_contract": probe.R3_QWEN_READER_RESIZE_CONTRACT,
        "feature_gate": probe.LOCKED_FEATURE_GATE,
        "feature_gate_sha256": canonical_json_sha256(probe.LOCKED_FEATURE_GATE),
        "preregistration": locks,
        "tc0_prerequisite": {
            "path": "/evidence/tc0_validation.json",
            "sha256": tc0_sha,
            "protocol": probe.TC0_VALIDATION_PROTOCOL,
            "expected_commit": SCIENCE_COMMIT,
            "passed": True,
        },
        "processor": {"passed": True, "checks": {name: True for name in validator._PROCESSOR_CHECKS}},
        "attention_backend": {
            "requested_implementation": "sdpa",
            "model_config_implementation": "sdpa",
            "sdpa": {"flash": False, "memory_efficient": False, "cudnn": False, "math": True},
        },
        "determinism": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "environment": dict(sorted(REQUIRED_DETERMINISM_ENV.items())),
            "sdpa": {"flash": False, "memory_efficient": False, "cudnn": False, "math": True},
        },
        "reader_frozen": {
            "training": False,
            "parameter_tensors": 10,
            "trainable_parameter_tensors": 0,
            "parameter_tensors_with_grad": 0,
            "passed": True,
        },
        "runtime": {
            "device_type": "cuda",
            "device_name": "NVIDIA MockH200",
            "device_capability": [9, 0],
            "peak_allocated_gib": 9.0,
            "peak_reserved_gib": 10.0,
            "torch": "2.7.0a0+mock",
            "cuda": "12.8",
            "transformers": "4.57.3",
            "torchvision": "0.22.0a0",
        },
        "suites": {"set8": suite, "transition16": second},
        "summary": {
            "suite_count": 2,
            "suite_pass_count": 2,
            "state_count": 2,
            "feature_comparison_count": 2,
            "feature_pass_count": 2,
            "cache_mutation_count": 0,
        },
        "unlocks": {
            "teacher_t0": True,
            "teacher_calibration": True,
            "teacher_assisted_training": True,
            "qa_only_dependency": False,
        },
        "provenance": {
            "git": {"commit": SCIENCE_COMMIT, "clean": True},
            "models": {
                "reader": {
                    "observed_revision": READER_REVISION,
                    "expected_revision": READER_REVISION,
                    "revision_matches_lock": True,
                }
            },
        },
        "passed": True,
    }


class R3TeacherFeatureCompatibilityTest(unittest.TestCase):
    def test_real_preregistration_contains_exact_prospective_feature_gate(self) -> None:
        locks = probe.load_preregistered_feature_lock(ROOT / "configs" / "experiments" / "r3_preregistration.json")
        self.assertEqual(locks["feature_backend_gate"], probe.LOCKED_FEATURE_GATE)
        self.assertEqual(
            locks["feature_backend_gate_sha256"],
            canonical_json_sha256(probe.LOCKED_FEATURE_GATE),
        )

    def test_feature_distance_uses_fixed_l2_and_cosine_thresholds(self) -> None:
        cached = (torch.arange(18).reshape(1, 3, 6) + 1).to(torch.bfloat16)
        exact = probe.feature_distance(
            cached,
            cached.clone(),
            relative_l2_max=0.01,
            cosine_min=0.9999,
        )
        self.assertTrue(exact["passed"])
        self.assertEqual(exact["relative_l2"], 0.0)
        self.assertGreaterEqual(exact["cosine"], 0.9999)

        drifted = probe.feature_distance(
            cached,
            -cached,
            relative_l2_max=0.01,
            cosine_min=0.9999,
        )
        self.assertFalse(drifted["passed"])
        self.assertGreater(drifted["relative_l2"], 0.01)
        self.assertLess(drifted["cosine"], 0.9999)

    def test_tc0_prerequisite_is_same_commit_and_sha_bound(self) -> None:
        locks = _locks({})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tc0.json"
            digest = _tc0_validation(path, locks["preregistration_sha256"])
            evidence = probe.validate_tc0_prerequisite(
                path,
                expected_file_sha256=digest,
                expected_commit=SCIENCE_COMMIT,
                locks=locks,
            )
            self.assertTrue(evidence["passed"])
            self.assertEqual(evidence["sha256"], digest)
            with self.assertRaisesRegex(ValueError, "expected_commit"):
                probe.validate_tc0_prerequisite(
                    path,
                    expected_file_sha256=digest,
                    expected_commit="f" * 40,
                    locks=locks,
                )

    def test_suite_recomputes_feature_and_preserves_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, teacher = _build_cache(root)
            output = SimpleNamespace(
                features=teacher.feature.clone(),
                pixel_values=torch.zeros(256, 1536, dtype=torch.float32),
                image_grid_thw=torch.tensor([[1, 16, 16]], dtype=torch.int64),
            )
            with mock.patch.object(probe, "qwen3vl_query_free_visual_features", return_value=output):
                report = probe.audit_feature_cache_suite(
                    suite="set8",
                    cache_root=root,
                    expected_lock=lock,
                    expected_reader_revision=READER_REVISION,
                    expected_cache_build_commit=CACHE_COMMIT,
                    feature_gate=probe.LOCKED_FEATURE_GATE,
                    model=object(),
                    processor=object(),
                    device=torch.device("cpu"),
                )
        self.assertTrue(report["passed"])
        self.assertEqual(report["observed"]["feature_comparison_count"], 1)
        self.assertEqual(report["observed"]["feature_pass_count"], 1)
        self.assertTrue(report["read_only_integrity"]["unchanged"])
        self.assertEqual(report["states"][0]["feature_comparison"]["relative_l2"], 0.0)

    def test_validator_requires_all_states_thresholds_and_read_only_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, teacher = _build_cache(root)
            output = SimpleNamespace(
                features=teacher.feature.clone(),
                pixel_values=torch.zeros(256, 1536, dtype=torch.float32),
                image_grid_thw=torch.tensor([[1, 16, 16]], dtype=torch.int64),
            )
            with mock.patch.object(probe, "qwen3vl_query_free_visual_features", return_value=output):
                suite = probe.audit_feature_cache_suite(
                    suite="set8",
                    cache_root=root,
                    expected_lock=lock,
                    expected_reader_revision=READER_REVISION,
                    expected_cache_build_commit=CACHE_COMMIT,
                    feature_gate=probe.LOCKED_FEATURE_GATE,
                    model=object(),
                    processor=object(),
                    device=torch.device("cpu"),
                )
        locks = _locks(lock)
        tc0_sha = "7" * 64
        raw = _raw_report(suite, locks, tc0_sha)
        accepted = validator.validate_tf0_report(
            raw,
            locks=locks,
            expected_commit=SCIENCE_COMMIT,
            tc0_validation_sha256=tc0_sha,
        )
        self.assertTrue(accepted["passed"], accepted["errors"])
        self.assertTrue(accepted["teacher_t0_unlocked"])
        self.assertEqual(accepted["validated_feature_pass_count"], 2)

        mutations = []
        wrong_threshold = copy.deepcopy(raw)
        wrong_threshold["suites"]["set8"]["states"][0]["feature_comparison"]["thresholds"]["relative_l2_max"] = 1.0
        mutations.append(wrong_threshold)
        failed_state = copy.deepcopy(raw)
        failed_state["suites"]["set8"]["states"][0]["passed"] = False
        mutations.append(failed_state)
        mutated_cache = copy.deepcopy(raw)
        mutated_cache["suites"]["set8"]["read_only_integrity"]["unchanged"] = False
        mutations.append(mutated_cache)
        wrong_backend = copy.deepcopy(raw)
        wrong_backend["attention_backend"]["sdpa"]["flash"] = True
        mutations.append(wrong_backend)
        for index, candidate in enumerate(mutations):
            with self.subTest(index=index):
                rejected = validator.validate_tf0_report(
                    candidate,
                    locks=locks,
                    expected_commit=SCIENCE_COMMIT,
                    tc0_validation_sha256=tc0_sha,
                )
                self.assertFalse(rejected["passed"])
                self.assertFalse(rejected["teacher_t0_unlocked"])

    def test_probe_main_emits_fail_closed_json_without_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tf0.json"
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                exit_code = probe.main(
                    [
                        "--set8-cache",
                        str(Path(directory) / "set8"),
                        "--transition16-cache",
                        str(Path(directory) / "transition16"),
                        "--reader",
                        str(Path(directory) / "reader"),
                        "--tc0-validation-report",
                        str(Path(directory) / "tc0.json"),
                        "--tc0-validation-report-sha256",
                        "0" * 64,
                        "--output-json",
                        str(output),
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertFalse(report["unlocks"]["teacher_t0"])
        self.assertIn("requires CUDA", report["error"]["message"])

    def test_probe_never_writes_even_failure_report_inside_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            set8 = Path(directory) / "set8"
            transition16 = Path(directory) / "transition16"
            set8.mkdir()
            transition16.mkdir()
            output = transition16 / "forbidden-report.json"
            exit_code = probe.main(
                [
                    "--set8-cache",
                    str(set8),
                    "--transition16-cache",
                    str(transition16),
                    "--reader",
                    str(Path(directory) / "reader"),
                    "--tc0-validation-report",
                    str(Path(directory) / "tc0.json"),
                    "--tc0-validation-report-sha256",
                    "0" * 64,
                    "--output-json",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
