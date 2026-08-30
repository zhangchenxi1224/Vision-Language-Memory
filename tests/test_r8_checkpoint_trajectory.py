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
SCRIPT = ROOT / "scripts" / "eval" / "dreamlite_r8_checkpoint_trajectory.py"
SPEC = importlib.util.spec_from_file_location("r8_checkpoint_trajectory_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
trajectory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = trajectory
SPEC.loader.exec_module(trajectory)

CONTROLLER_SCRIPT = ROOT / "scripts" / "inspire" / "run_r8_checkpoint_trajectory.py"
CONTROLLER_SPEC = importlib.util.spec_from_file_location(
    "r8_checkpoint_trajectory_controller_under_test",
    CONTROLLER_SCRIPT,
)
assert CONTROLLER_SPEC is not None and CONTROLLER_SPEC.loader is not None
controller = importlib.util.module_from_spec(CONTROLLER_SPEC)
sys.modules[CONTROLLER_SPEC.name] = controller
CONTROLLER_SPEC.loader.exec_module(controller)


class R8CheckpointTrajectoryTest(unittest.TestCase):
    def test_direct_entrypoint_help_imports_outside_repository(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT.parent,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--expected-training-commit", result.stdout)
        self.assertIn("common-descent-projected-norm-matched", result.stdout)

    def test_controller_direct_entrypoint_and_frozen_command(self):
        result = subprocess.run(
            [sys.executable, str(CONTROLLER_SCRIPT), "--help"],
            cwd=ROOT.parent,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--expected-analysis-commit", result.stdout)
        args = SimpleNamespace(
            arm="raw-mean-control",
            arm_root=Path("arm"),
            train=Path("train.jsonl"),
            dev=Path("dev.jsonl"),
            dreamlite=Path("dreamlite"),
            reader=Path("reader"),
            expected_training_commit="a" * 40,
            seed=0,
            dreamlite_device="cuda:0",
            reader_device="cuda:1",
        )
        command = controller._command(args, Path("evaluation"))
        self.assertEqual(command[command.index("--bootstrap-iterations") + 1], "10000")
        self.assertEqual(command[command.index("--expected-training-commit") + 1], "a" * 40)
        for forbidden in ("--learning-rate", "--gradient-clip", "--lora-rank", "--edit-start-sigma"):
            self.assertNotIn(forbidden, command)

    def test_controller_output_checks_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "hard8_checkpoint_rows.jsonl"
            artifact.write_text("{}\n", encoding="utf-8")
            summary = {
                "schema": controller.TRAJECTORY_SCHEMA,
                "status": "completed",
                "formal_success_claim": False,
                "arm": "raw-mean-control",
                "gradient_aggregation": "raw-mean",
                "training_commit": "a" * 40,
                "analysis_commit": "b" * 40,
                "selected_segments_sha256": controller.EXPECTED_SELECTED_SHA,
                "rows": 640,
                "endpoint_binding": {"passed": True},
            }
            summary_path = root / "trajectory_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            inventory = {
                "schema": controller.TRAJECTORY_INVENTORY_SCHEMA,
                "artifacts": [
                    {
                        "path": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": controller._sha256(path),
                    }
                    for path in (artifact, summary_path)
                ],
            }
            (root / "artifact_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
            checks = controller._evaluation_checks(
                root,
                arm="raw-mean-control",
                expected_training_commit="a" * 40,
                expected_analysis_commit="b" * 40,
            )
            self.assertTrue(all(checks.values()))
            artifact.write_text("tampered\n", encoding="utf-8")
            checks = controller._evaluation_checks(
                root,
                arm="raw-mean-control",
                expected_training_commit="a" * 40,
                expected_analysis_commit="b" * 40,
            )
            self.assertFalse(checks["artifact_inventory_valid"])

    def test_trajectory_statistics_keep_step128_as_primary_endpoint(self):
        rows = []
        labels = ("m0", "ema_step32", "ema_step64", "ema_step96", "ema_step128")
        for label_index, label in enumerate(labels):
            for unit in range(8):
                for condition in ("normal", "reset"):
                    normal_improvement = 0.5 * label_index if condition == "normal" else 0.0
                    rows.append(
                        {
                            "checkpoint": label,
                            "suite": "train_overfit_hard8",
                            "condition": condition,
                            "pair_unit": f"unit-{unit}",
                            "ce": 10.0 + unit / 10.0 - normal_improvement,
                            "correct": label_index >= 2 and condition == "normal",
                        }
                    )
        result = trajectory._trajectory_statistics(rows, bootstrap_iterations=100)
        self.assertTrue(result["descriptive_only_not_checkpoint_selection"])
        self.assertEqual(result["primary_endpoint_remains"], "ema_step128")
        self.assertEqual(set(result["normal_ce_vs_m0"]), set(labels[1:]))
        self.assertLess(result["normal_ce_vs_m0"]["ema_step128"]["estimate"], 0.0)
        self.assertEqual(result["normal_accuracy"]["ema_step128"], 1.0)

    def test_checkpoint_and_endpoint_binding_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"schema": "manifest", "value": 1}
            checkpoint = root / "step-000128.pt"
            payload = {
                "schema_version": 1,
                "optimizer_step": 128,
                "manifest": manifest,
                "trainer_state": {
                    "schema": trajectory.r5.R5_TRAINER_STATE_SCHEMA,
                    "next_optimizer_step": 128,
                    "ema_state": {"weight": torch.tensor([1.0, 2.0])},
                },
            }
            torch.save(payload, checkpoint)
            observed = trajectory._checkpoint_payload(
                checkpoint,
                expected_manifest=manifest,
                expected_step=128,
            )
            endpoint = root / "endpoint_ema.pt"
            torch.save(
                {"trainable_state": {"weight": torch.tensor([1.0, 2.0])}},
                endpoint,
            )
            binding = trajectory._endpoint_binding(endpoint_path=endpoint, step128_payload=observed)
            self.assertTrue(binding["passed"])
            torch.save(
                {"trainable_state": {"weight": torch.tensor([1.0, 3.0])}},
                endpoint,
            )
            with self.assertRaisesRegex(ValueError, "not the exact step128 EMA"):
                trajectory._endpoint_binding(endpoint_path=endpoint, step128_payload=observed)

    def test_checkpoint_manifest_uses_json_normalized_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "step-000000.pt"
            expected_manifest = {"environment": {"torch": "2.7"}, "counts": {"1": 3}}
            observed_manifest = {
                "environment": {"torch": torch.torch_version.TorchVersion("2.7")},
                "counts": {1: 3},
            }
            payload = {
                "schema_version": 1,
                "optimizer_step": 0,
                "manifest": observed_manifest,
                "trainer_state": {
                    "schema": trajectory.r5.R5_TRAINER_STATE_SCHEMA,
                    "next_optimizer_step": 0,
                    "ema_state": {"weight": torch.tensor([1.0])},
                },
            }
            torch.save(payload, checkpoint)
            trajectory._checkpoint_payload(
                checkpoint,
                expected_manifest=expected_manifest,
                expected_step=0,
            )
            expected_manifest["counts"]["1"] = 4
            with self.assertRaisesRegex(ValueError, "manifest drift"):
                trajectory._checkpoint_payload(
                    checkpoint,
                    expected_manifest=expected_manifest,
                    expected_step=0,
                )

    def test_source_inventory_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.txt"
            artifact.write_text("evidence", encoding="utf-8")
            inventory = {
                "schema": "vision_memory.r8-artifact-inventory.v1",
                "artifacts": [
                    {
                        "path": "artifact.txt",
                        "bytes": artifact.stat().st_size,
                        "sha256": trajectory._sha256(artifact),
                    }
                ],
            }
            (root / "artifact_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
            self.assertEqual(trajectory._validate_source_inventory(root)["records_checked"], 1)
            artifact.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed size/SHA"):
                trajectory._validate_source_inventory(root)


if __name__ == "__main__":
    unittest.main()
