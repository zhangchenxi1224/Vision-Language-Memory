from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.inspire import run_r14_symmetric_donor as controller  # noqa: E402
from scripts.train import r14_symmetric_donor_writer as trainer  # noqa: E402


class R14TrainingEntrypointTest(unittest.TestCase):
    def test_preregistered_config_locks_the_single_intervention(self) -> None:
        config = json.loads(trainer.CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["schema"], "vision_memory.r14-symmetric-donor-ranking-config.v1")
        self.assertEqual(config["status"], "preregistered_before_any_r14_model_outcome")
        self.assertEqual(
            controller._text_sha256_lf(trainer.R14_PREREG_PATH),
            config["preregistration"]["sha256"],
        )
        self.assertEqual(
            trainer._text_sha256_lf(trainer.R13_ANALYSIS_PATH),
            config["activation"]["r13_report_sha256"]["analysis"],
        )
        self.assertEqual(
            trainer._text_sha256_lf(trainer.R13_DELIVERY_MANIFEST_PATH),
            config["activation"]["r13_report_sha256"]["delivery_manifest"],
        )
        self.assertTrue(config["writer"]["r13_architecture_unchanged"])
        self.assertTrue(config["evaluation"]["identical_to_r13_except_endpoint_label"])
        self.assertEqual(config["optimization"]["micro_steps"], 4608)
        self.assertEqual(config["optimization"]["optimizer_steps"], 1152)
        self.assertIsNone(config["optimization"]["gradient_clipping"])
        self.assertEqual(config["optimization"]["reader_calls_per_micro_step"], 2)
        self.assertEqual(config["training_pairing"]["pair_count"], 72)
        self.assertEqual(
            config["training_pairing"]["pairs_sha256"],
            "6d6ae8a27374a505164db1db0be4caeec059f29cf1b95720a20867d30424683c",
        )
        self.assertTrue(config["success_boundary"]["cannot_establish_full_picture_memory_success"])

    def test_parser_exposes_sources_but_not_locked_hyperparameters(self) -> None:
        options = {action.dest for action in trainer.build_parser()._actions}
        self.assertIn("r12_conditioned_root", options)
        self.assertIn("r12_control_root", options)
        self.assertIn("r12_comparison", options)
        self.assertIn("r12_collapse_audit", options)
        self.assertIn("r13_root", options)
        for forbidden in (
            "learning_rate",
            "epochs",
            "optimizer_steps",
            "gradient_accumulation",
            "gradient_clip",
            "checkpoint_steps",
            "pairing_seed",
            "ranking_margin",
            "ranking_weight",
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
            r13_root=Path("r13"),
            dreamlite_device="cuda:0",
            reader_device="cuda:1",
        )
        command = controller._command(args, Path("run"))
        self.assertIn("--strict-determinism", command)
        self.assertEqual(command[command.index("--seed") + 1], "0")
        self.assertEqual(command[command.index("--r13-root") + 1], "r13")
        self.assertNotIn("--allow-dirty", command)
        self.assertNotIn("--learning-rate", command)
        self.assertNotIn("--optimizer-steps", command)


if __name__ == "__main__":
    unittest.main()
