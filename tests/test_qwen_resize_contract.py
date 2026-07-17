from __future__ import annotations

import hashlib
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

from scripts.probes import qwen_resize_contract as probe  # noqa: E402


class MockQwen2VLImageProcessorFast:
    do_resize = True
    min_pixels = 256 * 256
    max_pixels = 256 * 256
    size = {"shortest_edge": 256 * 256, "longest_edge": 256 * 256 * 256}
    patch_size = 16
    temporal_patch_size = 2
    merge_size = 2
    resample = 3

    def __init__(self, *, perturb_no_resize: bool = False) -> None:
        self.perturb_no_resize = perturb_no_resize
        self.resize_flags: list[bool] = []

    def __call__(self, *, images, return_tensors, do_rescale, do_resize=True):
        if return_tensors != "pt" or do_rescale is not False or len(images) != 1:
            raise AssertionError("Unexpected mock processor arguments.")
        self.resize_flags.append(bool(do_resize))
        image = images[0]
        if do_resize:
            if tuple(image.shape) != (3, 1024, 1024):
                raise AssertionError("The legacy mock path expects 1024x1024 input.")
            image = image[:, ::4, ::4]
        elif tuple(image.shape) != (3, 256, 256):
            raise AssertionError("The no-resize mock path expects prepared 256x256 input.")
        if not do_resize and self.perturb_no_resize:
            image = image + torch.finfo(image.dtype).eps
        # Qwen repeats one image across temporal_patch_size=2 before packing.
        pixel_values = torch.cat((image, image), dim=0).reshape(256, 1536).float()
        pixel_values = pixel_values * 2.0 - 1.0
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 16, 16]], dtype=torch.long),
        }


def mock_locked_resize(image: torch.Tensor, *, contract: str) -> torch.Tensor:
    if contract != probe.R3_QWEN_READER_RESIZE_CONTRACT:
        raise AssertionError("Unexpected resize contract.")
    if tuple(image.shape) != (3, 1024, 1024):
        raise AssertionError("Unexpected resize input shape.")
    return image[:, ::4, ::4]


class QwenResizeContractTest(unittest.TestCase):
    def test_inspire_execution_binding_locks_worker_and_formal_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            preflight = root / "r3_h200_formal.json"
            commit = "a" * 40
            preflight.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "formal_ready": True,
                        "git": {"commit": commit},
                    }
                ),
                encoding="utf-8",
            )
            preflight_sha = hashlib.sha256(preflight.read_bytes()).hexdigest()
            preflight.with_suffix(".json.sha256").write_text(preflight_sha + "\n", encoding="utf-8")
            worker = root / "worker_input.json"
            worker.write_text(
                json.dumps(
                    {
                        "stage": "r3-r0",
                        "infrastructure_stage": False,
                        "expected_commit": commit,
                        "preflight": str(preflight),
                        "preflight_sha256": preflight_sha,
                        "strict_environment": probe.REQUIRED_STAGE_ENVIRONMENT,
                    }
                ),
                encoding="utf-8",
            )
            worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
            environment = {
                "VLM_STAGE_WORKER_INPUT": str(worker),
                "VLM_STAGE_CONFIGURATION_SHA256": worker_sha,
                "VLM_STAGE_PREFLIGHT": str(preflight),
                "VLM_STAGE_PREFLIGHT_SHA256": preflight_sha,
            }
            report = probe.audit_inspire_execution_binding(environment)
            self.assertTrue(report["passed"])
            self.assertEqual(report["git_commit"], commit)
            self.assertEqual(report["worker_input_sha256"], worker_sha)
            self.assertEqual(report["formal_preflight_sha256"], preflight_sha)

    def test_parser_exposes_locked_runtime_inputs(self) -> None:
        args = probe.parse_args(
            [
                "--reader",
                "Qwen3-VL-4B-Instruct",
                "--device",
                "cuda:1",
                "--seed",
                "17",
                "--output-json",
                "report.json",
            ]
        )
        self.assertEqual(args.reader, Path("Qwen3-VL-4B-Instruct"))
        self.assertEqual(args.device, "cuda:1")
        self.assertEqual(args.seed, 17)
        self.assertEqual(args.output_json, Path("report.json"))

    def test_fast_processor_audit_accepts_only_locked_geometry(self) -> None:
        accepted = probe.audit_fast_processor(MockQwen2VLImageProcessorFast())
        self.assertTrue(accepted["passed"])
        self.assertTrue(all(accepted["checks"].values()))

        drifted = MockQwen2VLImageProcessorFast()
        drifted.max_pixels = 1024 * 1024
        rejected = probe.audit_fast_processor(drifted)
        self.assertFalse(rejected["passed"])
        self.assertFalse(rejected["checks"]["max_pixels_locked"])

    def test_forward_report_requires_exact_pixels_sha_shape_and_grid(self) -> None:
        processor = MockQwen2VLImageProcessorFast()
        image = torch.linspace(0.0, 1.0, 3 * 1024 * 1024).reshape(3, 1024, 1024)
        with mock.patch.object(probe, "deterministic_qwen_reader_resize", side_effect=mock_locked_resize) as resize:
            report = probe.compare_forward_paths(image_processor=processor, image=image)

        self.assertTrue(report["passed"])
        self.assertTrue(report["pixel_values_torch_equal"])
        self.assertEqual(report["pixel_values_max_absolute_difference"], 0.0)
        self.assertEqual(report["legacy_grid_thw"], [1, 16, 16])
        self.assertEqual(report["candidate_grid_thw"], [1, 16, 16])
        self.assertEqual(
            report["legacy_pixel_values"]["sha256"],
            report["candidate_pixel_values"]["sha256"],
        )
        self.assertEqual(processor.resize_flags, [True, False])
        resize.assert_called_once()

    def test_forward_report_fails_on_one_path_difference(self) -> None:
        processor = MockQwen2VLImageProcessorFast(perturb_no_resize=True)
        image = torch.rand(3, 1024, 1024)
        with mock.patch.object(probe, "deterministic_qwen_reader_resize", side_effect=mock_locked_resize):
            report = probe.compare_forward_paths(image_processor=processor, image=image)

        self.assertFalse(report["passed"])
        self.assertFalse(report["pixel_values_torch_equal"])
        self.assertGreater(report["pixel_values_max_absolute_difference"], 0.0)

    def test_strict_backward_report_requires_finite_nonzero_bitwise_repeat(self) -> None:
        processor = MockQwen2VLImageProcessorFast()
        image = torch.rand(3, 1024, 1024, dtype=torch.bfloat16)
        with mock.patch.object(probe, "deterministic_qwen_reader_resize", side_effect=mock_locked_resize) as resize:
            report = probe.compare_strict_backwards(image_processor=processor, image=image)

        self.assertTrue(report["passed"])
        self.assertTrue(report["gradient_finite"])
        self.assertTrue(report["gradient_nonzero"])
        self.assertTrue(report["gradient_torch_equal"])
        self.assertTrue(report["loss_bitwise_equal"])
        self.assertEqual(report["gradient_max_absolute_difference"], 0.0)
        self.assertEqual(
            report["first"]["gradient"]["sha256"],
            report["second"]["gradient"]["sha256"],
        )
        self.assertEqual(resize.call_count, 2)

    def test_full_contract_covers_fp32_and_bf16(self) -> None:
        processor = MockQwen2VLImageProcessorFast()
        adjoint = {
            "passed": True,
            "gradient_finite": True,
            "gradient_nonzero": True,
            "gradient_torch_equal": True,
            "gradient_max_absolute_difference": 0.0,
        }
        previous_enabled = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            with (
                mock.patch.object(
                    probe,
                    "deterministic_qwen_reader_resize",
                    side_effect=mock_locked_resize,
                ),
                mock.patch.object(probe, "compare_cpu_adjoint_reference", return_value=adjoint),
            ):
                report = probe.run_contract(
                    image_processor=processor,
                    device=torch.device("cpu"),
                    seed=23,
                )
        finally:
            torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)

        self.assertTrue(report["passed"])
        self.assertEqual(set(report["dtypes"]), {"float32", "bfloat16"})
        self.assertTrue(all(item["passed"] for item in report["dtypes"].values()))
        self.assertTrue(all(item["cpu_adjoint_reference"]["passed"] for item in report["dtypes"].values()))
        self.assertTrue(
            all(item["legacy_native_cuda_reference"]["passed"] for item in report["dtypes"].values())
        )

    def test_cpu_adjoint_reference_is_exact_for_fp32_and_bf16(self) -> None:
        torch.manual_seed(29)
        base = torch.rand(3, 1024, 1024)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                report = probe.compare_cpu_adjoint_reference(image=base.to(dtype))
                self.assertTrue(report["passed"])
                self.assertTrue(report["gradient_torch_equal"])
                self.assertEqual(report["gradient_max_absolute_difference"], 0.0)

    def test_main_emits_fail_closed_json_without_cuda(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                exit_code = probe.main(
                    [
                        "--reader",
                        str(Path(temporary) / "reader"),
                        "--output-json",
                        str(output),
                    ]
                )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["error"]["type"], "RuntimeError")
        self.assertIn("requires CUDA", report["error"]["message"])


if __name__ == "__main__":
    unittest.main()
