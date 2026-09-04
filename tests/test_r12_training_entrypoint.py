from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "train" / "r12_shared_event_latent_writer.py"
    specification = importlib.util.spec_from_file_location("r12_trainer_under_test", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


trainer = load_script()


def load_controller():
    path = ROOT / "scripts" / "inspire" / "run_r12_shared_writer_arm.py"
    specification = importlib.util.spec_from_file_location("r12_controller_under_test", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


controller = load_controller()


class R12TrainingEntrypointTest(unittest.TestCase):
    def test_parser_exposes_arm_and_paths_but_not_preregistered_hyperparameters(self):
        args = trainer.parse_args(
            (
                "--arm",
                "conditioned",
                "--train",
                "train.jsonl",
                "--dev",
                "dev.jsonl",
                "--dreamlite",
                "dreamlite",
                "--reader",
                "reader",
                "--r11-root",
                "r11",
                "--r11-comparison",
                "comparison.json",
                "--output-dir",
                "output",
            )
        )
        self.assertEqual(args.arm, "conditioned")
        self.assertEqual(args.seed, 0)
        help_text = trainer.build_parser().format_help()
        for forbidden in (
            "--learning-rate",
            "--optimizer-steps",
            "--epochs",
            "--basis-count",
            "--gradient-accumulation",
            "--checkpoint-steps",
        ):
            self.assertNotIn(forbidden, help_text)

    def test_writer_starts_at_exact_initial_latent_and_constant_arm_ignores_tokens(self):
        generator = torch.Generator().manual_seed(0)
        initial = torch.randn((1, 4, 128, 128), generator=generator)
        basis = torch.randn(
            (trainer.R12_BASIS_COUNT, 4, 128, 128), generator=generator
        )
        writer = trainer.SharedEventLatentWriter(
            initial_latent=initial,
            initial_basis=basis,
        )
        first = torch.randn((1, 5, 2048), generator=generator)
        second = torch.randn((1, 7, 2048), generator=generator)
        first_latent, first_diagnostics = writer(
            first, torch.ones((1, 5), dtype=torch.long), conditioned=False
        )
        second_latent, second_diagnostics = writer(
            second, torch.ones((1, 7), dtype=torch.long), conditioned=False
        )
        self.assertTrue(torch.equal(first_latent, initial))
        self.assertTrue(torch.equal(second_latent, initial))
        self.assertTrue(
            torch.equal(
                first_diagnostics["coefficients"],
                second_diagnostics["coefficients"],
            )
        )
        loss = first_latent.square().mean()
        loss.backward()
        final = writer.coefficient_mlp[-1]
        self.assertIsNotNone(final.weight.grad)
        self.assertGreater(float(final.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(final.bias.grad)
        self.assertGreater(float(final.bias.grad.abs().sum()), 0.0)

    def test_wrong_token_shape_fails_closed(self):
        writer = trainer.SharedEventLatentWriter(
            initial_latent=torch.zeros((1, 4, 128, 128)),
            initial_basis=torch.ones((trainer.R12_BASIS_COUNT, 4, 128, 128)),
        )
        with self.assertRaisesRegex(ValueError, "token states"):
            writer(
                torch.zeros((1, 4, 1024)),
                torch.ones((1, 4), dtype=torch.long),
                conditioned=True,
            )

    def test_controller_command_cannot_override_fixed_training_contract(self):
        args = SimpleNamespace(
            arm="constant-control",
            train=Path("train.jsonl"),
            dev=Path("dev.jsonl"),
            dreamlite=Path("dreamlite"),
            reader=Path("reader"),
            r11_root=Path("r11"),
            r11_comparison=Path("comparison.json"),
            dreamlite_device="cuda:0",
            reader_device="cuda:1",
        )
        command = controller._command(args, Path("run"))
        self.assertIn("constant-control", command)
        self.assertIn("--strict-determinism", command)
        for forbidden in (
            "--learning-rate",
            "--optimizer-steps",
            "--epochs",
            "--basis-count",
            "--gradient-accumulation",
        ):
            self.assertNotIn(forbidden, command)


if __name__ == "__main__":
    unittest.main()
