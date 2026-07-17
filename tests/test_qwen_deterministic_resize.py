from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as tv_functional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.reader import (  # noqa: E402
    R3_QWEN_READER_OUTPUT_HW,
    R3_QWEN_READER_RESIZE_CONTRACT,
    deterministic_qwen_reader_resize,
)


def reference_resize(image: torch.Tensor) -> torch.Tensor:
    return tv_functional.resize(
        image.unsqueeze(0),
        list(R3_QWEN_READER_OUTPUT_HW),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )[0]


class DeterministicQwenResizeTest(unittest.TestCase):
    def test_forward_is_bitwise_torchvision_bicubic_antialias_for_float32_and_bfloat16(self) -> None:
        torch.manual_seed(101)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                image = torch.rand(3, 1024, 1024, dtype=dtype)
                expected = reference_resize(image)
                actual = deterministic_qwen_reader_resize(image)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(tuple(actual.shape), (3, 256, 256))

    def test_strict_backward_is_finite_and_bitwise_repeatable(self) -> None:
        torch.manual_seed(103)
        source = torch.rand(3, 1024, 1024, dtype=torch.bfloat16)
        output_weight = torch.rand(3, 256, 256, dtype=torch.float32)
        previous_enabled = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        gradients: list[torch.Tensor] = []
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
            for _replica in range(2):
                image = source.clone().requires_grad_(True)
                resized = deterministic_qwen_reader_resize(image)
                (resized.float() * output_weight).sum().backward()
                assert image.grad is not None
                self.assertTrue(torch.isfinite(image.grad).all())
                self.assertGreater(float(image.grad.float().norm()), 0.0)
                self.assertEqual(image.grad.dtype, torch.bfloat16)
                gradients.append(image.grad.detach().clone())
        finally:
            torch.use_deterministic_algorithms(previous_enabled, warn_only=previous_warn_only)

        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)

    def test_cpu_adjoint_matches_native_torchvision_backward_exactly(self) -> None:
        torch.manual_seed(107)
        source = torch.rand(3, 1024, 1024, dtype=torch.float32)
        output_gradient = torch.rand(3, 256, 256, dtype=torch.float32)
        native_source = source.clone().requires_grad_(True)
        custom_source = source.clone().requires_grad_(True)

        native = reference_resize(native_source)
        native.backward(output_gradient)
        custom = deterministic_qwen_reader_resize(custom_source)
        custom.backward(output_gradient)

        assert native_source.grad is not None and custom_source.grad is not None
        torch.testing.assert_close(custom_source.grad, native_source.grad, rtol=0, atol=0)

    def test_contract_fails_closed_on_shape_dtype_and_identifier_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            deterministic_qwen_reader_resize(torch.rand(3, 256, 256))
        with self.assertRaisesRegex(TypeError, "float16"):
            deterministic_qwen_reader_resize(torch.zeros(3, 1024, 1024, dtype=torch.uint8))
        with self.assertRaisesRegex(ValueError, "Unknown"):
            deterministic_qwen_reader_resize(
                torch.rand(3, 1024, 1024),
                contract=R3_QWEN_READER_RESIZE_CONTRACT + "-drift",
            )


if __name__ == "__main__":
    unittest.main()
