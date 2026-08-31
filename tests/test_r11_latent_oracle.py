from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    path = ROOT / relative
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


trainer = load_script("r11_latent_trainer_under_test", "scripts/train/latent_r11_vae_oracle.py")
controller = load_script("r11_latent_controller_under_test", "scripts/inspire/run_r11_latent_target.py")


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def decode(self, latent, return_dict=False):
        return (latent[:, :3] * self.weight,)

    def encode(self, image, return_dict=True):
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: image))


def write_delivery(path: Path, schema: str, artifact: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "artifacts": [
                    {
                        "path": artifact.name,
                        "bytes": artifact.stat().st_size,
                        "sha256": controller._sha256(artifact),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


class R11LatentOracleTest(unittest.TestCase):
    def test_parser_freezes_preregistered_hyperparameters(self):
        args = trainer.parse_args(
            (
                "--target-index",
                "6",
                "--train",
                "train.jsonl",
                "--dev",
                "dev.jsonl",
                "--dreamlite",
                "dreamlite",
                "--reader",
                "reader",
                "--output-dir",
                "output",
            )
        )
        self.assertEqual(args.target_index, 6)
        self.assertEqual(trainer.R11_OPTIMIZER_STEPS, 256)
        self.assertEqual(trainer.R11_LEARNING_RATE, 0.05)
        self.assertEqual(trainer.R11_CHECKPOINT_STEPS, (0, 64, 128, 192, 256))
        help_text = trainer.build_parser().format_help()
        for forbidden in ("--learning-rate", "--optimizer-steps", "--weight-decay", "--checkpoint-steps"):
            self.assertNotIn(forbidden, help_text)

    def test_oracle_has_exactly_one_fp32_trainable_latent(self):
        vae = FakeVAE()
        initial = torch.zeros((1, 3, 4, 4), dtype=torch.bfloat16)
        oracle = trainer.VAELatentOracle(vae=vae, initial_latent=initial, compute_dtype=torch.bfloat16)
        trainable = [(name, parameter) for name, parameter in oracle.named_parameters() if parameter.requires_grad]
        self.assertEqual([name for name, _parameter in trainable], ["latent_fp32"])
        self.assertEqual(trainable[0][1].dtype, torch.float32)
        self.assertTrue(all(not parameter.requires_grad for parameter in oracle.vae.parameters()))
        loss = oracle.image().float().sum()
        loss.backward()
        self.assertIsNotNone(oracle.latent_fp32.grad)
        self.assertIsNone(oracle.vae.weight.grad)

    def test_technical_gate_requires_exact_views_receipts_and_frozen_models(self):
        oracle = trainer.VAELatentOracle(
            vae=FakeVAE(),
            initial_latent=torch.zeros((1, 3, 4, 4), dtype=torch.float32),
            compute_dtype=torch.float32,
        )
        reader = nn.Linear(2, 2).requires_grad_(False)
        metrics = []
        for step in range(1, 257):
            metrics.append(
                {
                    "kind": "optimizer_step",
                    "optimizer_step": step,
                    "target_segment_id": "target",
                    "forward_cyclic_training_view": (step - 1) % 4,
                    "loss_before_step": 1.0,
                    "gradient_norm": 0.5,
                    "gradient_nonzero_fraction": 1.0,
                    "latent_min_after_step": -1.0,
                    "latent_max_after_step": 1.0,
                    "latent_rms_after_step": 0.5,
                    "latent_delta_norm_after_step": 1.0,
                    "image_min_after_step": 0.0,
                    "image_max_after_step": 1.0,
                }
            )
        gate = trainer._technical_gate(
            metrics,
            target_segment_id="target",
            oracle=oracle,
            reader=reader,
            checkpoint_steps={0, 64, 128, 192, 256},
            png_steps={0, 64, 128, 192, 256},
            snapshots_unchanged=True,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["training_view_counts"], {0: 64, 1: 64, 2: 64, 3: 64})
        self.assertFalse(
            trainer._technical_gate(
                metrics,
                target_segment_id="target",
                oracle=oracle,
                reader=reader,
                checkpoint_steps={0, 64, 128, 192},
                png_steps={0, 64, 128, 192, 256},
                snapshots_unchanged=True,
            )["passed"]
        )

    def test_controller_command_does_not_expose_oracle_hyperparameters(self):
        args = SimpleNamespace(
            target_index=2,
            train=Path("train.jsonl"),
            dev=Path("dev.jsonl"),
            dreamlite=Path("dreamlite"),
            reader=Path("reader"),
            dreamlite_device="cuda:0",
            reader_device="cuda:1",
        )
        command = controller._command(args, Path("run"))
        for forbidden in ("--learning-rate", "--optimizer-steps", "--weight-decay"):
            self.assertNotIn(forbidden, command)
        self.assertIn("--strict-determinism", command)

    def test_machine_readable_preregistration_is_locked(self):
        config = json.loads(
            (ROOT / "configs" / "experiments" / "r11_vae_latent_reachability.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["oracle"]["optimizer_steps"], 256)
        self.assertEqual(config["oracle"]["learning_rate"], 0.05)
        self.assertFalse(config["oracle"]["dreamlite_unet_used"])
        self.assertEqual(config["arm_gate"]["required_target_passes"], 8)
        self.assertTrue(config["success_boundary"]["diagnostic_only"])

    def test_parent_delivery_hash_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison_path = root / "comparison.json"
            comparison_path.write_text("{}", encoding="utf-8")
            delivery_path = root / "DELIVERY_MANIFEST.json"
            write_delivery(delivery_path, r10_schema := controller.r10.DELIVERY_SCHEMA, comparison_path)
            delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
            delivery["artifacts"][0]["sha256"] = "0" * 64
            delivery_path.write_text(json.dumps(delivery), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size/SHA"):
                controller._validate_delivery(
                    delivery_path,
                    schema=r10_schema,
                    required_name="comparison.json",
                )


if __name__ == "__main__":
    unittest.main()
