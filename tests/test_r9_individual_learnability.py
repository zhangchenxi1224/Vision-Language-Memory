from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train" / "dreamlite_r9_individual_learnability.py"
SPEC = importlib.util.spec_from_file_location("r9_individual_learnability_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
r9 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r9
SPEC.loader.exec_module(r9)


class R9IndividualLearnabilityTest(unittest.TestCase):
    def test_direct_entrypoint_and_parser_freeze_contract(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT.parent,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--target-index {0,1,2,3,4,5,6,7}", result.stdout)
        for forbidden in ("--learning-rate", "--gradient-clip", "--lora-rank", "--edit-start-sigma"):
            self.assertNotIn(forbidden, result.stdout)
        args = r9.parse_args(
            [
                "--target-index",
                "3",
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
            ]
        )
        self.assertEqual(args.target_index, 3)
        self.assertEqual(args.gradient_aggregation, "single-target-one-eighth")
        self.assertEqual(args.gradient_accumulation, 8)
        self.assertEqual(args.lora_rank, 4)
        self.assertEqual(args.gradient_clip, 10.0)
        self.assertEqual(args.checkpoint_every, 32)
        self.assertEqual(args.max_optimizer_steps, 128)

    def test_target_statistics_use_four_fixed_views_without_fake_bootstrap(self):
        target = "target-segment"
        rows = []
        for checkpoint in ("m0", "ema_step128"):
            for condition in ("normal", "reset", "cross_episode_swap", "temporal_swap"):
                for view_index in range(4):
                    learned = checkpoint == "ema_step128" and condition == "normal"
                    rows.append(
                        {
                            "suite": "train_overfit_hard8",
                            "checkpoint": checkpoint,
                            "condition": condition,
                            "pair_unit": target,
                            "view_index": view_index,
                            "ce": 7.0 + view_index / 10.0 if learned else 10.0 + view_index / 10.0,
                            "correct": learned,
                        }
                    )
        result = r9._target_statistics(rows, target_segment_id=target)
        self.assertEqual(result["improved_choice_views"], 4)
        self.assertLessEqual(result["relative_change"], -0.20)
        self.assertEqual(result["accuracy_delta"], 1.0)
        self.assertLess(result["normal_reset_difference_in_differences"], 0.0)
        self.assertFalse(result["bootstrap_ci_used"])

    def test_single_target_technical_gate_requires_exact_receipts_and_scaling(self):
        target = "target-segment"
        metrics = [
            {
                "kind": "optimizer_step",
                "gradient_aggregation": {
                    "mode": "single-target-one-eighth",
                    "target_segment_id": target,
                    "gradient_coefficient": 0.125,
                    "scale_relative_error": 1e-8,
                    "raw_vs_applied_cosine": 1.0,
                },
            }
            for _ in range(128)
        ]
        gate = r9._aggregation_technical_gate(
            metrics,
            [{"segment_id": target} for _ in range(128)],
            target_segment_id=target,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["micro_records"], 128)
        metrics[-1]["gradient_aggregation"]["scale_relative_error"] = 1e-4
        self.assertFalse(
            r9._aggregation_technical_gate(
                metrics,
                [{"segment_id": target} for _ in range(128)],
                target_segment_id=target,
            )["passed"]
        )


if __name__ == "__main__":
    unittest.main()
