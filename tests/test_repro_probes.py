from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.repro import (  # noqa: E402
    DETERMINISTIC_FIXTURE_RGB_SHA256_1024,
    assert_no_frozen_parameter_grads,
    canonical_json_sha256,
    load_source_image,
    lora_trainable_parameters,
    seed_adapter_initialization,
    validate_e2e_pair_reports,
)


def make_pair_report(*, detached: bool) -> dict:
    metadata = {
        "events": ["remember red", "the room has a table"],
        "adapter_seed": 11,
        "event_noise_seeds": [19, 20],
        "source_image": {"rgb_sha256": "abc"},
    }
    return {
        "events": 2,
        "detach_between_events": detached,
        "pair_id": canonical_json_sha256(metadata),
        "pair_metadata": metadata,
        "loss": 1.25,
        "intermediate_gradients": [
            {"norm": None if detached else 0.75, "nonfinite_elements": None if detached else 0}
        ],
        "lora_grad_norm": 2.0,
        "unclamped_image_grad_norm": 3.0,
    }


class ReproProbeContractTest(unittest.TestCase):
    def test_lora_whitelist_rejects_accidentally_trainable_base_weights(self):
        module = nn.Module()
        module.lora_A = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})
        module.base_weight = nn.Parameter(torch.ones(2, 2))
        with self.assertRaisesRegex(RuntimeError, "non-LoRA"):
            lora_trainable_parameters(module)
        module.base_weight.requires_grad_(False)
        self.assertEqual(len(lora_trainable_parameters(module)), 1)

    def test_deterministic_fixture_has_locked_rgb_sha(self):
        first, first_metadata = load_source_image(None, resolution=1024)
        second, second_metadata = load_source_image(None, resolution=1024)

        self.assertEqual(first.size, (1024, 1024))
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["rgb_sha256"], DETERMINISTIC_FIXTURE_RGB_SHA256_1024)

    def test_adapter_seed_is_repeatable_and_explicit(self):
        seed_adapter_initialization(37)
        first = torch.randn(4)
        seed_adapter_initialization(37)
        second = torch.randn(4)
        torch.testing.assert_close(first, second)

    def test_frozen_gradient_contract_reports_and_rejects_gradients(self):
        module = nn.Linear(2, 2).requires_grad_(False)
        report = assert_no_frozen_parameter_grads({"reader": module}, fully_frozen={"reader"})
        self.assertEqual(report["reader"]["frozen_tensors_with_grad"], 0)

        module.weight.grad = torch.ones_like(module.weight)
        with self.assertRaisesRegex(RuntimeError, "reader accumulated gradients"):
            assert_no_frozen_parameter_grads({"reader": module}, fully_frozen={"reader"})

    def test_valid_e2e_positive_detach_pair(self):
        report = validate_e2e_pair_reports(
            make_pair_report(detached=False),
            make_pair_report(detached=True),
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["positive_intermediate_grad_norm"], 0.75)
        self.assertIsNone(report["detached_intermediate_grad_norm"])

    def test_pair_validator_rejects_metadata_drift(self):
        positive = make_pair_report(detached=False)
        detached = make_pair_report(detached=True)
        detached["pair_metadata"] = {**detached["pair_metadata"], "adapter_seed": 12}
        with self.assertRaisesRegex(ValueError, "metadata"):
            validate_e2e_pair_reports(positive, detached)

    def test_pair_validator_rejects_forward_drift(self):
        positive = make_pair_report(detached=False)
        detached = make_pair_report(detached=True)
        detached["loss"] = 1.5
        with self.assertRaisesRegex(ValueError, "Forward losses differ"):
            validate_e2e_pair_reports(positive, detached)


if __name__ == "__main__":
    unittest.main()
