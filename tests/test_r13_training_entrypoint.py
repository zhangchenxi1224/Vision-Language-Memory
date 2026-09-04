from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.inspire import run_r13_centered_residual as controller  # noqa: E402
from scripts.train import r13_centered_residual_writer as trainer  # noqa: E402


class R13TrainingEntrypointTest(unittest.TestCase):
    def test_parser_exposes_sources_but_not_locked_hyperparameters(self) -> None:
        options = {action.dest for action in trainer.build_parser()._actions}
        self.assertIn("r12_conditioned_root", options)
        self.assertIn("r12_control_root", options)
        self.assertIn("r12_comparison", options)
        self.assertIn("r12_collapse_audit", options)
        for forbidden in (
            "learning_rate",
            "epochs",
            "optimizer_steps",
            "gradient_accumulation",
            "gradient_clip",
            "checkpoint_steps",
        ):
            self.assertNotIn(forbidden, options)

    def test_controller_command_fixes_seed_and_strict_determinism(self) -> None:
        args = argparse.Namespace(
            train=Path("train.jsonl"),
            dev=Path("dev.jsonl"),
            dreamlite=Path("dreamlite"),
            reader=Path("reader"),
            r12_conditioned_root=Path("conditioned"),
            r12_control_root=Path("control"),
            r12_comparison=Path("comparison.json"),
            r12_collapse_audit=Path("collapse.json"),
            dreamlite_device="cuda:0",
            reader_device="cuda:1",
        )
        command = controller._command(args, Path("run"))
        self.assertIn("--strict-determinism", command)
        self.assertEqual(command[command.index("--seed") + 1], "0")
        self.assertNotIn("--allow-dirty", command)
        self.assertNotIn("--learning-rate", command)
        self.assertNotIn("--optimizer-steps", command)


if __name__ == "__main__":
    unittest.main()
