from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

from scripts.probes import r3_teacher_cache_compatibility as probe  # noqa: E402
from scripts.probes import validate_r3_teacher_cache_compatibility as validator  # noqa: E402
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


class MockQwen2VLImageProcessorFast:
    do_resize = True
    min_pixels = 256 * 256
    max_pixels = 256 * 256
    size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256 * 256}
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    resample = 3

    def __init__(self, *, perturb_candidate: bool = False) -> None:
        self.perturb_candidate = perturb_candidate
        self.resize_flags: list[bool] = []

    def __call__(self, *, images, return_tensors, do_rescale, do_resize=True):
        if return_tensors != "pt" or do_rescale is not False or len(images) != 1:
            raise AssertionError("Unexpected mock processor arguments.")
        self.resize_flags.append(bool(do_resize))
        image = images[0]
        if do_resize:
            image = image[:, ::4, ::4]
        if not do_resize and self.perturb_candidate:
            image = image + torch.finfo(image.dtype).eps
        pixels = torch.cat((image, image), dim=0).reshape(256, 1536).float()
        return {
            "pixel_values": pixels * 2.0 - 1.0,
            "image_grid_thw": torch.tensor([[1, 16, 16]], dtype=torch.int64, device=image.device),
        }


def mock_resize(image: torch.Tensor, *, contract: str) -> torch.Tensor:
    if contract != probe.R3_QWEN_READER_RESIZE_CONTRACT:
        raise AssertionError("Unexpected resize contract.")
    return image[:, ::4, ::4]


def _teacher(identity: str = "1") -> TeacherState:
    return TeacherState(
        state_id=identity * 64,
        teacher_key="2" * 64,
        semantic_state_sha256="3" * 64,
        teacher_contract_sha256="4" * 64,
        renderer_contract_sha256="5" * 64,
        image=torch.zeros(1, 3, 1024, 1024, dtype=torch.float16),
        latent=torch.arange(256, dtype=torch.float32).reshape(1, 4, 8, 8) / 255.0,
        feature=torch.arange(18, dtype=torch.float32).reshape(1, 3, 6) / 17.0,
    )


def _build_cache(root: Path) -> tuple[dict, dict]:
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
        "schema": probe.BUILD_REPORT_SCHEMA,
        "git_commit": CACHE_COMMIT,
        "model_revisions": {"dreamlite": "a" * 40, "qwen_reader": READER_REVISION},
        "teacher_contract_sha256": manifest.teacher_contract_sha256,
        "renderer_contract_sha256": manifest.renderer_contract_sha256,
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
    return lock, build_report


def _raw_report(suite_report: dict, lock: dict) -> tuple[dict, dict]:
    locks = {
        "preregistration_path": "/locked/r3_preregistration.json",
        "preregistration_sha256": "6" * 64,
        "preregistration_schema": probe.PREREGISTRATION_SCHEMA,
        "reader_revision": READER_REVISION,
        "reader_resize_contract": probe.R3_QWEN_READER_RESIZE_CONTRACT,
        "cache_build_commit": CACHE_COMMIT,
        "gpu_model": "MockH200",
        "transformers_version": "4.57.3",
        "torchvision_version": "0.22.0a0",
        "torch_version": "2.7.0a0+mock",
        "cuda_version": "12.8",
        "suites": {"set8": lock, "transition16": lock},
    }
    second = copy.deepcopy(suite_report)
    second["suite"] = "transition16"
    second["cache_root"] = "/immutable/transition16"
    report = {
        "schema_version": probe.PROBE_SCHEMA_VERSION,
        "probe": probe.PROBE_NAME,
        "scope": {
            "included": "cached decoded image -> Qwen fast processor pixel_values/image_grid_thw",
            "excluded": "cached Qwen feature attention-backend compatibility; audited separately",
            "cache_access": "read-only",
        },
        "feature_backend_compatibility": {
            "status": "not_evaluated_by_r3_tc0",
            "risk": "existing feature.pt used default SDPA while replacement training uses strict math-only",
            "required_followup": "preregistered R3-TF0 strict-math gate",
            "preregistered_thresholds": {"relative_l2_max": 0.01, "cosine_min": 0.9999},
            "teacher_t0_unlocked": False,
            "teacher_calibration_unlocked": False,
            "teacher_assisted_training_unlocked": False,
        },
        "reader_revision": READER_REVISION,
        "reader_resize_contract": probe.R3_QWEN_READER_RESIZE_CONTRACT,
        "preregistration": locks,
        "processor": {"passed": True, "checks": {name: True for name in validator._PROCESSOR_CHECKS}},
        "determinism": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "environment": dict(sorted(REQUIRED_DETERMINISM_ENV.items())),
        },
        "runtime": {
            "device_type": "cuda",
            "device_name": "NVIDIA MockH200",
            "device_capability": [9, 0],
            "torch": "2.7.0a0+mock",
            "cuda": "12.8",
            "transformers": "4.57.3",
            "torchvision": "0.22.0a0",
        },
        "suites": {"set8": suite_report, "transition16": second},
        "summary": {
            "suite_count": 2,
            "suite_pass_count": 2,
            "state_count": 2,
            "artifact_tensor_count": 6,
            "image_forward_count": 2,
            "image_forward_pass_count": 2,
            "cache_mutation_count": 0,
        },
        "provenance": {
            "git": {"commit": "a" * 40, "clean": True},
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
    return report, locks


class R3TeacherCacheCompatibilityTest(unittest.TestCase):
    def test_parser_exposes_both_locked_caches_and_reader(self) -> None:
        args = probe.parse_args(
            [
                "--set8-cache",
                "set8",
                "--transition16-cache",
                "transition16",
                "--reader",
                "reader",
                "--device",
                "cuda:1",
                "--output-json",
                "tc0.json",
            ]
        )
        self.assertEqual(args.set8_cache, Path("set8"))
        self.assertEqual(args.transition16_cache, Path("transition16"))
        self.assertEqual(args.reader, Path("reader"))
        self.assertEqual(args.device, "cuda:1")

    def test_real_preregistration_locks_complete_existing_cache_evidence(self) -> None:
        locks = probe.load_preregistered_cache_locks(ROOT / "configs" / "experiments" / "r3_preregistration.json")
        self.assertEqual(locks["reader_revision"], "ebb281ec70b05090aa6165b016eac8ec08e71b17")
        self.assertEqual(set(locks["suites"]), {"set8", "transition16"})
        self.assertEqual(locks["suites"]["set8"]["artifact_tensor_count"], 30)
        self.assertEqual(locks["suites"]["transition16"]["artifact_tensor_count"], 60)
        for suite in locks["suites"].values():
            for field, value in suite.items():
                if field.endswith("sha256"):
                    self.assertRegex(value, r"^[0-9a-f]{64}$")

    def test_forward_comparison_requires_bitwise_pixels_and_grid(self) -> None:
        image = torch.linspace(0.0, 1.0, 3 * 1024 * 1024).reshape(3, 1024, 1024)
        processor = MockQwen2VLImageProcessorFast()
        with mock.patch.object(probe, "deterministic_qwen_reader_resize", side_effect=mock_resize):
            result = probe.compare_forward_paths(image_processor=processor, image=image)
        self.assertTrue(result["passed"])
        self.assertEqual(result["pixel_values_max_absolute_difference"], 0.0)
        self.assertEqual(result["legacy_pixel_values"]["sha256"], result["candidate_pixel_values"]["sha256"])
        self.assertEqual(processor.resize_flags, [True, False])

        drifted = MockQwen2VLImageProcessorFast(perturb_candidate=True)
        with mock.patch.object(probe, "deterministic_qwen_reader_resize", side_effect=mock_resize):
            rejected = probe.compare_forward_paths(image_processor=drifted, image=image)
        self.assertFalse(rejected["passed"])
        self.assertGreater(rejected["pixel_values_max_absolute_difference"], 0.0)

    def test_suite_audit_rehashes_all_tensors_and_preserves_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, _build_report = _build_cache(root)
            processor = MockQwen2VLImageProcessorFast()
            with mock.patch.object(probe, "deterministic_qwen_reader_resize", side_effect=mock_resize):
                report = probe.audit_cache_suite(
                    suite="set8",
                    cache_root=root,
                    expected_lock=lock,
                    expected_reader_revision=READER_REVISION,
                    expected_cache_build_commit=CACHE_COMMIT,
                    image_processor=processor,
                    device=torch.device("cpu"),
                )
        self.assertTrue(report["passed"])
        self.assertEqual(report["observed"]["artifact_tensor_count"], 3)
        self.assertEqual(report["observed"]["image_forward_count"], 1)
        self.assertTrue(report["read_only_integrity"]["unchanged"])
        self.assertEqual(set(report["states"][0]["artifacts"]), {"image", "latent", "feature"})
        self.assertEqual(
            report["states"][0]["forward"]["source_teacher_tensor_sha256"],
            report["states"][0]["artifacts"]["image"]["tensor_content_sha256"],
        )

    def test_suite_audit_fails_closed_on_artifact_file_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, build_report = _build_cache(root)
            artifact = root / next(iter(build_report["artifact_file_sha256"]))
            artifact.write_bytes(artifact.read_bytes() + b"drift")
            with self.assertRaisesRegex(ValueError, "on-disk artifact files differ"):
                probe.audit_cache_suite(
                    suite="set8",
                    cache_root=root,
                    expected_lock=lock,
                    expected_reader_revision=READER_REVISION,
                    expected_cache_build_commit=CACHE_COMMIT,
                    image_processor=MockQwen2VLImageProcessorFast(),
                    device=torch.device("cpu"),
                )

    def test_fail_closed_validator_checks_complete_per_state_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, _build_report = _build_cache(root)
            with mock.patch.object(probe, "deterministic_qwen_reader_resize", side_effect=mock_resize):
                suite = probe.audit_cache_suite(
                    suite="set8",
                    cache_root=root,
                    expected_lock=lock,
                    expected_reader_revision=READER_REVISION,
                    expected_cache_build_commit=CACHE_COMMIT,
                    image_processor=MockQwen2VLImageProcessorFast(),
                    device=torch.device("cpu"),
                )
        raw, locks = _raw_report(suite, lock)
        accepted = validator.validate_tc0_report(raw, locks=locks, expected_commit="a" * 40)
        self.assertTrue(accepted["passed"], accepted["errors"])
        self.assertEqual(accepted["validated_state_count"], 2)
        self.assertEqual(accepted["validated_artifact_tensor_count"], 6)
        self.assertTrue(accepted["cache_forward_compatibility_complete"])
        self.assertFalse(accepted["feature_backend_compatibility_complete"])
        self.assertFalse(accepted["teacher_t0_unlocked"])

        mutations = []
        dirty = copy.deepcopy(raw)
        dirty["provenance"]["git"]["clean"] = False
        mutations.append(dirty)
        incomplete = copy.deepcopy(raw)
        incomplete["suites"]["set8"]["states"] = []
        mutations.append(incomplete)
        changed_pixel = copy.deepcopy(raw)
        changed_pixel["suites"]["set8"]["states"][0]["forward"]["candidate_pixel_values"]["sha256"] = "f" * 64
        mutations.append(changed_pixel)
        mutated_cache = copy.deepcopy(raw)
        mutated_cache["suites"]["set8"]["read_only_integrity"]["unchanged"] = False
        mutations.append(mutated_cache)
        for candidate in mutations:
            with self.subTest(mutation=mutations.index(candidate)):
                rejected = validator.validate_tc0_report(
                    candidate,
                    locks=locks,
                    expected_commit="a" * 40,
                )
                self.assertFalse(rejected["passed"])
                self.assertTrue(rejected["errors"])

    def test_probe_main_emits_fail_closed_json_without_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tc0.json"
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                exit_code = probe.main(
                    [
                        "--set8-cache",
                        str(Path(directory) / "set8"),
                        "--transition16-cache",
                        str(Path(directory) / "transition16"),
                        "--reader",
                        str(Path(directory) / "reader"),
                        "--output-json",
                        str(output),
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["error"]["type"], "RuntimeError")
        self.assertIn("requires CUDA", report["error"]["message"])

    def test_probe_never_writes_even_failure_report_inside_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            set8 = Path(directory) / "set8"
            transition16 = Path(directory) / "transition16"
            set8.mkdir()
            transition16.mkdir()
            output = set8 / "forbidden-report.json"
            exit_code = probe.main(
                [
                    "--set8-cache",
                    str(set8),
                    "--transition16-cache",
                    str(transition16),
                    "--reader",
                    str(Path(directory) / "reader"),
                    "--output-json",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
