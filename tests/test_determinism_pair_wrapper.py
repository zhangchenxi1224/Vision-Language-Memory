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

from scripts.probes import lightweight_determinism as probe  # noqa: E402
from scripts.probes import run_lightweight_determinism_pair as pair  # noqa: E402
from vision_memory.repro import REQUIRED_DETERMINISM_ENV  # noqa: E402


class DeterminismPairWrapperTest(unittest.TestCase):
    @staticmethod
    def target_only_predictions(*, passed: bool) -> list[dict]:
        correct_count = 116 if passed else 115
        return [
            {
                "correct": index < correct_count,
                "target_index": index % 4,
                "event_kind": "set" if index % 2 == 0 else "overwrite",
                "distractor_variant": "clean" if index % 2 == 0 else "distractor",
                "turn_type": "mixed" if index >= 104 else "query",
                "topic": "style",
            }
            for index in range(128)
        ]

    @staticmethod
    def listwise_predictions(*, view: str, passed: bool) -> list[dict]:
        predictions: list[dict] = []
        for pair_index in range(64):
            canonical_choices = [f"pair-{pair_index:02d}-choice-{index}" for index in range(4)]
            canonical_target = pair_index % 4
            choices = canonical_choices if view == "canonical" else canonical_choices[1:] + canonical_choices[:1]
            target_index = canonical_target if view == "canonical" else (canonical_target - 1) % 4
            correct = pair_index >= 6
            for variant in ("clean", "distractor"):
                predicted_index = target_index if correct else (target_index + 1) % 4
                predictions.append(
                    {
                        "episode_id": f"episode-{pair_index:02d}-{variant}",
                        "turn_id": 3,
                        "query_ordinal": 0,
                        "comparison_id": f"comparison-{pair_index:02d}",
                        "view": view,
                        "choices": choices,
                        "target_index": target_index,
                        "target_text": choices[target_index],
                        "predicted_index": predicted_index,
                        "predicted_text": choices[predicted_index],
                        "correct": predicted_index == target_index,
                        "event_kind": "noop" if variant == "distractor" else "set",
                        "state_event_kind": "overwrite" if pair_index % 2 else "set",
                        "distractor_variant": variant,
                        "turn_type": "mixed" if pair_index >= 52 else "query",
                        "topic": "style",
                    }
                )
        if not passed:
            record = predictions[12]
            record["predicted_index"] = (record["target_index"] + 1) % 4
            record["predicted_text"] = record["choices"][record["predicted_index"]]
            record["correct"] = False
        return predictions

    @staticmethod
    def listwise_trace(steps: int) -> list[dict]:
        return [
            {
                "optimizer_step": step,
                "gradient_norm_before_clip_float_hex": float(1.0).hex(),
                "listwise_queries": [
                    {
                        "all_values_finite": True,
                        "image_gradient_norm_float_hex": float(1.0).hex(),
                    }
                ],
            }
            for step in range(1, steps + 1)
        ]

    @staticmethod
    def complete_child_result(*, passed: bool, reader_loss_mode: str = "target-only", steps: int = 2000) -> dict:
        objective = pair.reader_objective_contract(reader_loss_mode)
        if reader_loss_mode == "listwise-choice":
            predictions = DeterminismPairWrapperTest.listwise_predictions(view="canonical", passed=passed)
            rotated_predictions = DeterminismPairWrapperTest.listwise_predictions(
                view="left-rotate-one",
                passed=passed,
            )
            trace = DeterminismPairWrapperTest.listwise_trace(steps)
        else:
            predictions = DeterminismPairWrapperTest.target_only_predictions(passed=passed)
            rotated_predictions = []
            trace = []
        reachability_gate = probe.reachability_gate_summary(
            steps=steps,
            optimizer_steps_completed=steps,
            predictions=predictions,
            positive_gradient_steps=steps,
            clipped_steps=0,
            reader_loss_mode=reader_loss_mode,
        )
        r2_gate = probe.r2_gate_summary(
            reader_loss_mode=reader_loss_mode,
            steps=steps,
            optimizer_steps_completed=steps,
            canonical_predictions=predictions,
            rotated_predictions=rotated_predictions,
            trace=trace,
            positive_gradient_steps=steps,
            clipped_steps=0,
        )
        r2a = probe.r2a_autograd_diagnostic(reader_loss_mode, trace)
        payload = {
            "protocol": {
                "schema_version": "vision_memory.lightweight_determinism_protocol.v4",
                "reader_loss_mode": reader_loss_mode,
                "reader_objective": objective,
            },
            "reader_loss_mode": reader_loss_mode,
            "reader_objective": objective,
            "reachability_gate": reachability_gate,
            "r2_gate": r2_gate,
            "r2a_autograd_diagnostic": r2a,
            "trace_sha256": "same",
        }
        return {
            "returncode": 0,
            "report": {
                "schema_version": "vision_memory.lightweight_determinism_report.v3",
                "status": "complete",
                "reader_loss_mode": reader_loss_mode,
                "reader_objective": objective,
                "comparison_payload": payload,
                "comparison_payload_sha256": pair.canonical_object_sha256(payload),
                "reachability_gate": reachability_gate,
                "r2_gate": r2_gate,
                "r2a_autograd_diagnostic": payload["r2a_autograd_diagnostic"],
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
                reader_loss_mode="target-only",
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

        listwise_is_not_r1 = pair.pair_reachability_gate(
            steps=2000,
            reader_loss_mode="listwise-choice",
            child_results={
                "a": self.complete_child_result(passed=True, reader_loss_mode="listwise-choice"),
                "b": self.complete_child_result(passed=True, reader_loss_mode="listwise-choice"),
            },
        )
        self.assertFalse(listwise_is_not_r1["applicable"])
        self.assertIsNone(listwise_is_not_r1["passed"])

    def test_scientific_gate_failure_still_runs_both_replicas_and_preserves_bitwise_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "pair"
            arguments = argparse.Namespace(
                train=Path(temporary) / "train.jsonl",
                reader=Path(temporary) / "reader",
                output_dir=output_dir,
                steps=2000,
                device="cuda:0",
                reader_loss_mode="target-only",
            )
            environment = {
                **REQUIRED_DETERMINISM_ENV,
                "SLURM_JOB_ID": "12346",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            child_report = self.complete_child_result(passed=False)["report"]

            def complete_with_failed_scientific_gate(command, **_kwargs):
                replica_dir = Path(command[command.index("--output-dir") + 1])
                replica_dir.mkdir(parents=True, exist_ok=True)
                (replica_dir / "report.json").write_text(
                    json.dumps(child_report) + "\n",
                    encoding="utf-8",
                )
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
            self.assertEqual(report["reader_loss_mode"], "target-only")
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index("--reader-loss-mode") + 1], "target-only")

    def test_listwise_pair_uses_only_prospective_r2_gate(self):
        children = {
            "a": self.complete_child_result(passed=True, reader_loss_mode="listwise-choice"),
            "b": self.complete_child_result(passed=True, reader_loss_mode="listwise-choice"),
        }

        r1 = pair.pair_reachability_gate(
            steps=2000,
            reader_loss_mode="listwise-choice",
            child_results=children,
        )
        r2 = pair.pair_r2_gate(
            steps=2000,
            reader_loss_mode="listwise-choice",
            child_results=children,
        )
        r2a = pair.pair_r2a_autograd_diagnostic(
            steps=2000,
            reader_loss_mode="listwise-choice",
            child_results=children,
        )

        self.assertFalse(r1["applicable"])
        self.assertIsNone(r1["passed"])
        self.assertTrue(r2["applicable"])
        self.assertTrue(r2["passed"])
        self.assertTrue(r2a["applicable"])
        self.assertTrue(r2a["passed"])

    def test_child_contract_fails_closed_on_mode_or_gate_drift(self):
        child = self.complete_child_result(passed=True, reader_loss_mode="listwise-choice")
        valid = pair.child_canonical_contract(
            reader_loss_mode="listwise-choice",
            steps=2000,
            child_result=child,
        )
        child["report"]["comparison_payload"]["protocol"]["reader_loss_mode"] = "target-only"
        drifted = pair.child_canonical_contract(
            reader_loss_mode="listwise-choice",
            steps=2000,
            child_result=child,
        )

        self.assertTrue(valid["valid"])
        self.assertFalse(drifted["valid"])
        self.assertFalse(drifted["checks"]["protocol_mode_matches"])

    def test_child_contract_rejects_truncated_passed_r2_gate(self):
        child = self.complete_child_result(passed=True, reader_loss_mode="listwise-choice")
        truncated = {
            "applicable": True,
            "passed": True,
            "reader_loss_mode": "listwise-choice",
            "step_budget": 2000,
            "optimizer_steps_completed": 2000,
        }
        child["report"]["r2_gate"] = truncated
        child["report"]["comparison_payload"]["r2_gate"] = truncated
        child["report"]["comparison_payload_sha256"] = pair.canonical_object_sha256(
            child["report"]["comparison_payload"]
        )

        contract = pair.child_canonical_contract(
            reader_loss_mode="listwise-choice",
            steps=2000,
            child_result=child,
        )

        self.assertFalse(contract["valid"])
        self.assertFalse(contract["checks"]["r2_gate_semantics_valid"])

    def test_listwise_one_step_failed_r2a_is_a_technical_failure(self):
        child = self.complete_child_result(
            passed=True,
            reader_loss_mode="listwise-choice",
            steps=1,
        )
        failed_r2a = dict(child["report"]["r2a_autograd_diagnostic"])
        failed_r2a.update(
            {
                "passed": False,
                "positive_image_gradient_query_count": 0,
                "all_query_image_gradients_positive": False,
                "steps_with_positive_image_gradient": 0,
                "every_step_has_positive_image_gradient": False,
            }
        )
        child["report"]["r2a_autograd_diagnostic"] = failed_r2a
        child["report"]["comparison_payload"]["r2a_autograd_diagnostic"] = failed_r2a
        child["report"]["comparison_payload_sha256"] = pair.canonical_object_sha256(
            child["report"]["comparison_payload"]
        )

        contract = pair.child_canonical_contract(
            reader_loss_mode="listwise-choice",
            steps=1,
            child_result=child,
        )

        self.assertTrue(contract["checks"]["r2a_semantics_valid"])
        self.assertFalse(contract["checks"]["r2a_passed_for_listwise"])
        self.assertFalse(contract["valid"])

    def test_listwise_one_step_zero_updater_gradient_is_a_technical_failure(self):
        child = self.complete_child_result(
            passed=True,
            reader_loss_mode="listwise-choice",
            steps=1,
        )
        failed_r2a = dict(child["report"]["r2a_autograd_diagnostic"])
        failed_r2a.update(
            {
                "passed": False,
                "positive_updater_gradient_steps": 0,
                "every_step_has_positive_updater_gradient": False,
            }
        )
        child["report"]["r2a_autograd_diagnostic"] = failed_r2a
        child["report"]["comparison_payload"]["r2a_autograd_diagnostic"] = failed_r2a
        child["report"]["comparison_payload_sha256"] = pair.canonical_object_sha256(
            child["report"]["comparison_payload"]
        )

        contract = pair.child_canonical_contract(
            reader_loss_mode="listwise-choice",
            steps=1,
            child_result=child,
        )

        self.assertTrue(contract["checks"]["r2a_semantics_valid"])
        self.assertFalse(contract["checks"]["r2a_passed_for_listwise"])
        self.assertFalse(contract["valid"])

    def test_listwise_one_step_failed_r2a_makes_wrapper_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "pair"
            arguments = argparse.Namespace(
                train=Path(temporary) / "train.jsonl",
                reader=Path(temporary) / "reader",
                output_dir=output_dir,
                steps=1,
                device="cuda:0",
                reader_loss_mode="listwise-choice",
            )
            environment = {
                **REQUIRED_DETERMINISM_ENV,
                "SLURM_JOB_ID": "12348",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            child_report = self.complete_child_result(
                passed=True,
                reader_loss_mode="listwise-choice",
                steps=1,
            )["report"]
            failed_r2a = dict(child_report["r2a_autograd_diagnostic"])
            failed_r2a.update(
                {
                    "passed": False,
                    "positive_image_gradient_query_count": 0,
                    "all_query_image_gradients_positive": False,
                    "steps_with_positive_image_gradient": 0,
                    "every_step_has_positive_image_gradient": False,
                }
            )
            child_report["r2a_autograd_diagnostic"] = failed_r2a
            child_report["comparison_payload"]["r2a_autograd_diagnostic"] = failed_r2a
            child_report["comparison_payload_sha256"] = pair.canonical_object_sha256(child_report["comparison_payload"])

            def complete_with_failed_r2a(command, **_kwargs):
                replica_dir = Path(command[command.index("--output-dir") + 1])
                replica_dir.mkdir(parents=True, exist_ok=True)
                (replica_dir / "report.json").write_text(json.dumps(child_report) + "\n", encoding="utf-8")
                return subprocess.CompletedProcess(args=command, returncode=0)

            with (
                mock.patch.object(pair, "parse_args", return_value=arguments),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(pair.subprocess, "run", side_effect=complete_with_failed_r2a),
                mock.patch("builtins.print"),
            ):
                returncode = pair.main()

            report = json.loads((output_dir / "pair_report.json").read_text(encoding="utf-8"))
            self.assertEqual(returncode, 1)
            self.assertFalse(report["reproducibility_valid"])
            self.assertIsNone(report["scientific_gate_name"])
            self.assertIsNone(report["r2_gate_passed"])
            self.assertFalse(report["overall_passed"])

    def test_listwise_main_passes_mode_and_requires_r2(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "pair"
            arguments = argparse.Namespace(
                train=Path(temporary) / "train.jsonl",
                reader=Path(temporary) / "reader",
                output_dir=output_dir,
                steps=2000,
                device="cuda:0",
                reader_loss_mode="listwise-choice",
            )
            environment = {
                **REQUIRED_DETERMINISM_ENV,
                "SLURM_JOB_ID": "12347",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            child_report = self.complete_child_result(
                passed=True,
                reader_loss_mode="listwise-choice",
            )["report"]

            def complete_listwise(command, **_kwargs):
                replica_dir = Path(command[command.index("--output-dir") + 1])
                replica_dir.mkdir(parents=True, exist_ok=True)
                (replica_dir / "report.json").write_text(
                    json.dumps(child_report) + "\n",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(args=command, returncode=0)

            with (
                mock.patch.object(pair, "parse_args", return_value=arguments),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(pair.subprocess, "run", side_effect=complete_listwise) as run,
                mock.patch("builtins.print"),
            ):
                returncode = pair.main()

            self.assertEqual(returncode, 0)
            self.assertEqual(run.call_count, 2)
            report = json.loads((output_dir / "pair_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], "vision_memory.lightweight_determinism_pair.v3")
            self.assertTrue(report["reproducibility_valid"])
            self.assertIsNone(report["reachability_gate_passed"])
            self.assertTrue(report["r2_gate_passed"])
            self.assertTrue(report["r2a_autograd_diagnostic"]["passed"])
            self.assertEqual(report["scientific_gate_name"], "R2/D2L-listwise-choice")
            self.assertTrue(report["overall_passed"])
            for call in run.call_args_list:
                command = call.args[0]
                self.assertEqual(command[command.index("--reader-loss-mode") + 1], "listwise-choice")


if __name__ == "__main__":
    unittest.main()
