from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.repro import (  # noqa: E402
    DETERMINISTIC_FIXTURE_RGB_SHA256_1024,
    assert_no_frozen_parameter_grads,
    assert_determinism_environment,
    canonical_json_sha256,
    canonical_object_sha256,
    canonical_tensor_sha256,
    compare_bitwise_repro_reports,
    load_initial_image,
    load_source_image,
    lora_trainable_parameters,
    model_optimizer_rng_manifest,
    named_tensors_manifest,
    seed_adapter_initialization,
    validate_e2e_pair_reports,
)
from scripts.probes.lightweight_determinism import validate_qwen_image_grid_contract  # noqa: E402


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
        "intermediate_gradients": [{"norm": None if detached else 0.75, "nonfinite_elements": None if detached else 0}],
        "lora_grad_norm": 2.0,
        "unclamped_image_grad_norm": 3.0,
    }


class ReproProbeContractTest(unittest.TestCase):
    def test_qwen3_image_grid_contract_accepts_256_and_rejects_252(self):
        processor = SimpleNamespace(patch_size=16, temporal_patch_size=2, merge_size=2)

        contract = validate_qwen_image_grid_contract(processor, image_size=256)

        self.assertEqual(contract["spatial_factor"], 32)
        with self.assertRaisesRegex(RuntimeError, "divisible"):
            validate_qwen_image_grid_contract(processor, image_size=252)

    def test_qwen3_image_grid_contract_rejects_configuration_drift(self):
        processor = SimpleNamespace(patch_size=14, temporal_patch_size=2, merge_size=2)
        with self.assertRaisesRegex(RuntimeError, "patch_size drifted"):
            validate_qwen_image_grid_contract(processor, image_size=252)

    def test_canonical_tensor_and_object_hashes_are_bitwise_and_order_stable(self):
        tensor = torch.tensor([[1.0, -0.0], [2.0, 3.0]], dtype=torch.float32)
        clone = tensor.clone()
        changed = tensor.clone()
        changed[1, 1] = torch.nextafter(changed[1, 1], torch.tensor(float("inf")))

        self.assertEqual(canonical_tensor_sha256(tensor), canonical_tensor_sha256(clone))
        self.assertNotEqual(canonical_tensor_sha256(tensor), canonical_tensor_sha256(changed))
        self.assertNotEqual(canonical_tensor_sha256(tensor), canonical_tensor_sha256(tensor.double()))
        self.assertEqual(
            canonical_object_sha256({"b": [tensor], "a": 1}),
            canonical_object_sha256({"a": 1, "b": [clone]}),
        )
        self.assertNotEqual(canonical_object_sha256(0.0), canonical_object_sha256(-0.0))

        manifest = named_tensors_manifest({"second": changed, "first": tensor})
        self.assertEqual(list(manifest["tensors"]), ["first", "second"])
        self.assertEqual(manifest["tensors"]["first"]["sha256"], canonical_tensor_sha256(tensor))

    def test_determinism_environment_fails_closed(self):
        expected = {
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        self.assertEqual(assert_determinism_environment(expected), dict(sorted(expected.items())))
        with self.assertRaisesRegex(RuntimeError, "environment mismatch"):
            assert_determinism_environment({**expected, "OMP_NUM_THREADS": "2"})

    def test_model_manifest_includes_nonpersistent_buffers(self):
        module = nn.Linear(2, 2)
        module.register_buffer("ephemeral", torch.tensor([3.0]), persistent=False)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)

        manifest = model_optimizer_rng_manifest(module, optimizer)

        self.assertIn("parameter:weight", manifest["model"]["tensors"])
        self.assertIn("parameter:bias", manifest["model"]["tensors"])
        self.assertIn("buffer:ephemeral", manifest["model"]["tensors"])

    def test_bitwise_report_comparator_ignores_runtime_but_rejects_payload_drift(self):
        first = {
            "status": "complete",
            "runtime": {"pid": 1},
            "comparison_payload": {"trace": [{"loss": "0x1.0p+0"}], "model": "abc"},
        }
        second = {
            "status": "complete",
            "runtime": {"pid": 2},
            "comparison_payload": {"model": "abc", "trace": [{"loss": "0x1.0p+0"}]},
        }
        self.assertTrue(compare_bitwise_repro_reports(first, second)["valid"])

        second["comparison_payload"]["trace"][0]["loss"] = "0x1.0000000000001p+0"
        comparison = compare_bitwise_repro_reports(first, second)
        self.assertFalse(comparison["valid"])
        self.assertTrue(any("trace[0].loss" in mismatch for mismatch in comparison["mismatches"]))

        second["status"] = "failed"
        self.assertFalse(compare_bitwise_repro_reports(first, second)["valid"])

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

    def test_formal_blank_initial_image_is_uniform_and_fail_closed(self):
        first, first_metadata = load_initial_image("blank", resolution=64)
        second, second_metadata = load_initial_image("blank", resolution=64)

        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["initial_state_mode"], "blank")
        self.assertEqual(first.getextrema(), ((127, 127), (127, 127), (127, 127)))
        with self.assertRaisesRegex(ValueError, "does not accept"):
            load_initial_image("blank", Path("unexpected.png"), resolution=64)
        with self.assertRaisesRegex(ValueError, "requires"):
            load_initial_image("file", resolution=64)

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
