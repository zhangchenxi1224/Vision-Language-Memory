from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probes import run_lightweight_determinism_pair as pair  # noqa: E402
from vision_memory.repro import REQUIRED_DETERMINISM_ENV  # noqa: E402


class DeterminismPairWrapperTest(unittest.TestCase):
    @staticmethod
    def complete_child_result(*, passed: bool) -> dict:
        gate = {
            "applicable": True,
            "passed": passed,
            "step_budget": 2000,
            "optimizer_steps_completed": 2000,
            "final_prediction_count": 128,
            "final_correct": 116 if passed else 115,
        }
        return {
            "returncode": 0,
            "report": {
                "status": "complete",
                "comparison_payload": {"reachability_gate": gate, "trace_sha256": "same"},
                "reachability_gate": gate,
            },
        }

    def test_wrapper_step_choices_include_preregistered_reachability_budget(self):
        self.assertEqual(pair.ALLOWED_STEP_COUNTS, (1, 100, 2000))

    def test_second_replica_runs_after_first_replica_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "pair"
            arguments = argparse.Namespace(
                train=Path(temporary) / "train.jsonl",
                reader=Path(temporary) / "reader",
                output_dir=output_dir,
                steps=1,
                device="cuda:0",
            )
            environment = {
                **REQUIRED_DETERMINISM_ENV,
                "SLURM_JOB_ID": "12345",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            failures = [
                subprocess.CompletedProcess(args=["replica-a"], returncode=17),
                subprocess.CompletedProcess(args=["replica-b"], returncode=19),
            ]
            with (
                mock.patch.object(pair, "parse_args", return_value=arguments),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(pair.subprocess, "run", side_effect=failures) as run,
                mock.patch("builtins.print"),
            ):
                returncode = pair.main()

            self.assertEqual(returncode, 1)
            self.assertEqual(run.call_count, 2)
            report = json.loads((output_dir / "pair_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["valid"])
            self.assertFalse(report["overall_passed"])
            self.assertEqual(report["children"]["a"]["returncode"], 17)
            self.assertEqual(report["children"]["b"]["returncode"], 19)

    def test_pair_reachability_gate_requires_both_complete_children_to_pass(self):
        passed = pair.pair_reachability_gate(
            steps=2000,
            child_results={
                "a": self.complete_child_result(passed=True),
                "b": self.complete_child_result(passed=True),
            },
        )
        failed = pair.pair_reachability_gate(
            steps=2000,
            child_results={
                "a": self.complete_child_result(passed=False),
                "b": self.complete_child_result(passed=False),
            },
        )

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])

    def test_scientific_gate_failure_still_runs_both_replicas_and_preserves_bitwise_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "pair"
            arguments = argparse.Namespace(
                train=Path(temporary) / "train.jsonl",
                reader=Path(temporary) / "reader",
                output_dir=output_dir,
                steps=2000,
                device="cuda:0",
            )
            environment = {
                **REQUIRED_DETERMINISM_ENV,
                "SLURM_JOB_ID": "12346",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            gate = self.complete_child_result(passed=False)["report"]["reachability_gate"]

            def complete_with_failed_scientific_gate(command, **_kwargs):
                replica_dir = Path(command[command.index("--output-dir") + 1])
                replica_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "status": "complete",
                    "comparison_payload": {"reachability_gate": gate, "trace_sha256": "same"},
                    "reachability_gate": gate,
                }
                (replica_dir / "report.json").write_text(json.dumps(report) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(args=command, returncode=0)

            with (
                mock.patch.object(pair, "parse_args", return_value=arguments),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(pair.subprocess, "run", side_effect=complete_with_failed_scientific_gate) as run,
                mock.patch("builtins.print"),
            ):
                returncode = pair.main()

            self.assertEqual(returncode, 1)
            self.assertEqual(run.call_count, 2)
            report = json.loads((output_dir / "pair_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["reproducibility_valid"])
            self.assertTrue(report["valid"])
            self.assertFalse(report["reachability_gate_passed"])
            self.assertFalse(report["overall_passed"])


if __name__ == "__main__":
    unittest.main()
