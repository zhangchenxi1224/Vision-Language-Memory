from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


engine = load_script(
    "r8_common_descent_engine_under_test",
    ROOT / "scripts" / "train" / "dreamlite_r7_gradient_balance.py",
)
controller = load_script(
    "r8_common_descent_controller_under_test",
    ROOT / "scripts" / "inspire" / "run_r7_gradient_balance_arm.py",
)
comparison = load_script(
    "r8_common_descent_comparison_under_test",
    ROOT / "scripts" / "experiments" / "compare_r8_common_descent.py",
)


def summary(arm: str, *, overfit: bool, dev: bool, technical: bool = True):
    return {
        "schema": "vision_memory.r8-common-descent-summary.v1",
        "status": "completed",
        "arm": arm,
        "gradient_aggregation": comparison.EXPECTED_MODE[arm],
        "edit_start_sigma": 0.5,
        "full_success_claim_allowed": False,
        "selected_segments_sha256": "same",
        "implementation_revision": "implementation",
        "git_commit": "a" * 40,
        "gates": {
            "technical_gate": technical,
            "hard8_overfit_learnability_gate": overfit,
            "fixed_dev_generalization_gate": dev,
        },
        "comparisons": {
            "train_overfit_hard8_endpoint_vs_m0": {"m0_mean_ce": 10.0, "estimate": -1.0},
            "formal_select_32_endpoint_vs_m0": {"m0_mean_ce": 11.0, "estimate": -0.5},
            "mechanism_select_32_endpoint_vs_m0": {"m0_mean_ce": 12.0, "estimate": -0.5},
        },
    }


class R8CommonDescentTest(unittest.TestCase):
    def test_projection_removes_raw_micro_ascent_and_matches_norm(self):
        vectors = [torch.tensor([1.0, 0.0]) for _ in range(7)] + [torch.tensor([-2.0, 1.0])]
        raw, _raw_report = engine._aggregate_micro_gradients(
            vectors,
            mode="raw-mean",
            protocol_revision="r8",
        )
        projected, report = engine._aggregate_micro_gradients(
            vectors,
            mode="common-descent-projected-norm-matched",
            protocol_revision="r8",
        )
        projection = report["common_descent_projection"]
        self.assertEqual(projection["raw_violating_micro_count"], 1)
        self.assertEqual(projection["projected_violating_micro_count_at_tolerance"], 0)
        self.assertGreaterEqual(projection["minimum_projected_micro_cosine"], -1e-5)
        self.assertTrue(report["intervention_active"])
        self.assertLess(report["raw_vs_applied_cosine"], 0.9)
        self.assertLessEqual(report["norm_match_relative_error"], 1e-5)
        self.assertTrue(math.isclose(float(raw.norm()), float(projected.norm()), rel_tol=1e-6))

    def test_projection_is_exact_noop_when_raw_is_already_common_descent(self):
        vectors = [torch.tensor([1.0, float(index + 1) / 20.0]) for index in range(8)]
        raw, _ = engine._aggregate_micro_gradients(
            vectors,
            mode="raw-mean",
            protocol_revision="r8",
        )
        projected, report = engine._aggregate_micro_gradients(
            vectors,
            mode="common-descent-projected-norm-matched",
            protocol_revision="r8",
        )
        torch.testing.assert_close(projected, raw)
        self.assertFalse(report["intervention_active"])
        self.assertEqual(report["common_descent_projection"]["selected_active_set_mask"], 0)

    def test_projection_fails_closed_when_feasible_projection_collapses(self):
        vectors = [torch.tensor([1.0]) for _ in range(7)] + [torch.tensor([-1.0])]
        with self.assertRaisesRegex(RuntimeError, "collapsed to a zero direction"):
            engine._aggregate_micro_gradients(
                vectors,
                mode="common-descent-projected-norm-matched",
                protocol_revision="r8",
            )

    def test_projected_gate_requires_all_constraints_and_active_intervention(self):
        def metric(*, active: bool, minimum: float = 0.0):
            return {
                "kind": "optimizer_step",
                "gradient_aggregation": {
                    "mode": "common-descent-projected-norm-matched",
                    "norm_match_relative_error": 1e-7,
                    "raw_mean_norm": 4.0,
                    "unit_mean_norm_before_match": 0.5,
                    "applied_norm_before_clip": 4.0,
                    "raw_vs_unit_balanced_cosine": 0.8,
                    "raw_vs_applied_cosine": 0.8 if active else 1.0,
                    "intervention_active": active,
                    "common_descent_projection": {
                        "minimum_projected_micro_cosine": minimum,
                        "projected_violating_micro_count_at_tolerance": 0,
                        "raw_violating_micro_count": 1 if active else 0,
                    },
                },
            }

        metrics = [metric(active=index == 0) for index in range(128)]
        gate = engine._aggregation_technical_gate(
            metrics,
            expected_mode="common-descent-projected-norm-matched",
            protocol_revision="r8",
        )
        self.assertTrue(gate["passed"])
        metrics[-1]["gradient_aggregation"]["common_descent_projection"][
            "projected_violating_micro_count_at_tolerance"
        ] = 1
        self.assertFalse(
            engine._aggregation_technical_gate(
                metrics,
                expected_mode="common-descent-projected-norm-matched",
                protocol_revision="r8",
            )["passed"]
        )

    def test_r8_controller_command_has_frozen_hyperparameters(self):
        args = SimpleNamespace(
            protocol_revision="r8",
            arm="common-descent-projected-norm-matched",
            train=Path("train.jsonl"),
            dev=Path("dev.jsonl"),
            dreamlite=Path("dreamlite"),
            reader=Path("reader"),
            seed=0,
            dreamlite_device="cuda:0",
            reader_device="cuda:1",
        )
        command = controller._command(args, Path("run"))
        self.assertIn("dreamlite_r8_conflict_projection.py", command[1])
        self.assertEqual(command[command.index("--arm") + 1], args.arm)
        self.assertIn("--strict-determinism", command)
        for forbidden in ("--learning-rate", "--gradient-clip", "--lora-rank", "--edit-start-sigma"):
            self.assertNotIn(forbidden, command)

    def test_pair_decision_routes_joint_failure_to_per_segment_decomposition(self):
        raw = summary("raw-mean-control", overfit=False, dev=False)
        projected = summary("common-descent-projected-norm-matched", overfit=False, dev=False)
        comparison._validate(raw, projected)
        decision, _reason = comparison._decision(raw, projected)
        self.assertEqual(decision, "reject_batch_conflict_as_sufficient_test_per_segment_learnability")

    def test_pair_decision_does_not_interpret_projection_technical_failure(self):
        raw = summary("raw-mean-control", overfit=False, dev=False)
        projected = summary(
            "common-descent-projected-norm-matched",
            overfit=False,
            dev=False,
            technical=False,
        )
        decision, _reason = comparison._decision(raw, projected)
        self.assertEqual(decision, "common_descent_technical_failure_no_scientific_decision")


if __name__ == "__main__":
    unittest.main()
