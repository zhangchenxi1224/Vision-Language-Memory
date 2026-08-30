from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train" / "dreamlite_r9_individual_learnability.py"
SPEC = importlib.util.spec_from_file_location("r9_individual_learnability_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
r9 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r9
SPEC.loader.exec_module(r9)

CONTROLLER_SCRIPT = ROOT / "scripts" / "inspire" / "run_r9_individual_target.py"
CONTROLLER_SPEC = importlib.util.spec_from_file_location(
    "r9_individual_target_controller_under_test",
    CONTROLLER_SCRIPT,
)
assert CONTROLLER_SPEC is not None and CONTROLLER_SPEC.loader is not None
controller = importlib.util.module_from_spec(CONTROLLER_SPEC)
sys.modules[CONTROLLER_SPEC.name] = controller
CONTROLLER_SPEC.loader.exec_module(controller)


def make_r8_activation(root: Path, *, projected_passes: bool = False) -> Path:
    arms = {}
    paths = {}
    hashes = {}
    for arm in controller.R8_ARMS:
        summary = {
            "schema": "vision_memory.r8-common-descent-summary.v1",
            "status": "completed",
            "arm": arm,
            "git_commit": controller.EXPECTED_R8_TRAINING_COMMIT,
            "selected_segments_sha256": controller.EXPECTED_SELECTED_SHA,
            "full_success_claim_allowed": False,
            "gates": {
                "technical_gate": True,
                "hard8_overfit_learnability_gate": projected_passes and arm.startswith("common"),
            },
        }
        path = root / f"{arm}.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        arms[arm] = summary
        paths[arm] = str(path.resolve())
        hashes[arm] = controller._sha256(path)
    comparison = {
        "schema": "vision_memory.r8-common-descent-comparison.v1",
        "status": "completed",
        "decision": controller.ACTIVATION_DECISION,
        "formal_success_claim": False,
        "arms": arms,
        "summary_paths": paths,
        "summary_sha256": hashes,
    }
    comparison_path = root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    return comparison_path


class R9IndividualLearnabilityTest(unittest.TestCase):
    def test_positive_scaled_target_gradient_has_unit_cosine(self):
        raw = torch.tensor([1.0, -2.0, 3.0])
        self.assertAlmostEqual(r9._cosine(raw, raw / 8.0), 1.0, places=7)
        with self.assertRaisesRegex(ValueError, "zero vector"):
            r9._cosine(raw, torch.zeros_like(raw))

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

    def test_controller_entrypoint_command_and_activation_lock(self):
        result = subprocess.run(
            [sys.executable, str(CONTROLLER_SCRIPT), "--help"],
            cwd=ROOT.parent,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--parent-comparison", result.stdout)
        args = SimpleNamespace(
            target_index=5,
            train=Path("train.jsonl"),
            dev=Path("dev.jsonl"),
            dreamlite=Path("dreamlite"),
            reader=Path("reader"),
            seed=0,
            dreamlite_device="cuda:0",
            reader_device="cuda:1",
        )
        command = controller._command(args, Path("run"))
        self.assertEqual(command[command.index("--target-index") + 1], "5")
        self.assertIn("--strict-determinism", command)
        for forbidden in ("--learning-rate", "--gradient-clip", "--lora-rank", "--edit-start-sigma"):
            self.assertNotIn(forbidden, command)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison = make_r8_activation(root)
            observed = controller._validate_parent(comparison)
            self.assertEqual(observed["decision"], controller.ACTIVATION_DECISION)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison = make_r8_activation(root, projected_passes=True)
            with self.assertRaisesRegex(ValueError, "did not jointly fail"):
                controller._validate_parent(comparison)

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
