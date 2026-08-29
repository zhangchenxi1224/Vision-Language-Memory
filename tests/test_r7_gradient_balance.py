from __future__ import annotations

import importlib.util
import inspect
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


r7 = load_script(
    "r7_gradient_balance_under_test",
    ROOT / "scripts" / "train" / "dreamlite_r7_gradient_balance.py",
)
controller = load_script(
    "r7_gradient_balance_controller_under_test",
    ROOT / "scripts" / "inspire" / "run_r7_gradient_balance_arm.py",
)
comparison = load_script(
    "r7_gradient_balance_comparison_under_test",
    ROOT / "scripts" / "experiments" / "compare_r7_gradient_balance.py",
)


def summary(arm: str, *, overfit: bool, dev: bool, formal_delta: float = -1.0):
    mode = comparison.EXPECTED_MODE[arm]
    return {
        "schema": "vision_memory.r7-gradient-balance-summary.v1",
        "status": "completed",
        "arm": arm,
        "gradient_aggregation": mode,
        "edit_start_sigma": 0.5,
        "full_success_claim_allowed": False,
        "selected_segments_sha256": "same",
        "implementation_revision": "implementation",
        "git_commit": "a" * 40,
        "gates": {
            "hard8_overfit_learnability_gate": overfit,
            "fixed_dev_generalization_gate": dev,
        },
        "comparisons": {
            "train_overfit_hard8_endpoint_vs_m0": {"m0_mean_ce": 10.0, "estimate": -1.0},
            "formal_select_32_endpoint_vs_m0": {"m0_mean_ce": 11.0, "estimate": formal_delta},
            "mechanism_select_32_endpoint_vs_m0": {"m0_mean_ce": 12.0, "estimate": -0.5},
        },
    }


class R7GradientBalanceTest(unittest.TestCase):
    def test_raw_and_unit_aggregation_change_only_direction_and_match_norm(self):
        vectors = [torch.tensor([100.0, 0.0]) for _ in range(4)] + [torch.tensor([0.0, 1.0]) for _ in range(4)]
        raw, raw_report = r7._aggregate_micro_gradients(vectors, mode="raw-mean")
        balanced, balanced_report = r7._aggregate_micro_gradients(
            vectors,
            mode="unit-balanced-norm-matched",
        )
        torch.testing.assert_close(raw, torch.tensor([50.0, 0.5]))
        self.assertLess(balanced_report["raw_vs_applied_cosine"], 0.8)
        self.assertTrue(balanced_report["intervention_active"])
        self.assertLessEqual(balanced_report["norm_match_relative_error"], 1e-5)
        self.assertTrue(math.isclose(float(raw.norm()), float(balanced.norm()), rel_tol=1e-6))
        self.assertTrue(math.isclose(raw_report["raw_vs_applied_cosine"], 1.0, abs_tol=1e-6))

    def test_aggregation_fails_closed_on_zero_micro_gradient(self):
        vectors = [torch.ones(2) for _ in range(7)] + [torch.zeros(2)]
        with self.assertRaisesRegex(RuntimeError, "invalid micro-gradient norms"):
            r7._aggregate_micro_gradients(vectors, mode="unit-balanced-norm-matched")

    def test_aggregation_gate_requires_all_steps_and_active_intervention(self):
        def metric(cosine: float, *, active: bool):
            return {
                "kind": "optimizer_step",
                "gradient_aggregation": {
                    "mode": "unit-balanced-norm-matched",
                    "norm_match_relative_error": 1e-7,
                    "raw_mean_norm": 4.0,
                    "unit_mean_norm_before_match": 0.5,
                    "applied_norm_before_clip": 4.0,
                    "raw_vs_unit_balanced_cosine": cosine,
                    "raw_vs_applied_cosine": cosine,
                    "intervention_active": active,
                },
            }

        metrics = [metric(0.9, active=True) for _ in range(128)]
        gate = r7._aggregation_technical_gate(
            metrics,
            expected_mode="unit-balanced-norm-matched",
        )
        self.assertTrue(gate["passed"])
        self.assertFalse(
            r7._aggregation_technical_gate(
                metrics[:-1],
                expected_mode="unit-balanced-norm-matched",
            )["passed"]
        )

    def test_r5_training_core_exposes_explicit_optimizer_step_hook(self):
        signature = inspect.signature(r7.r5.run_training_profile)
        self.assertIn("optimizer_step_fn", signature.parameters)
        self.assertIsNone(signature.parameters["optimizer_step_fn"].default)

    def test_controller_command_has_no_hyperparameter_sweep(self):
        args = SimpleNamespace(
            arm="unit-balanced-norm-matched",
            train=Path("train.jsonl"),
            dev=Path("dev.jsonl"),
            dreamlite=Path("dreamlite"),
            reader=Path("reader"),
            seed=0,
            dreamlite_device="cuda:0",
            reader_device="cuda:1",
        )
        command = controller._command(args, Path("run"))
        self.assertIn("--strict-determinism", command)
        self.assertEqual(command[command.index("--arm") + 1], "unit-balanced-norm-matched")
        for forbidden in ("--learning-rate", "--gradient-clip", "--lora-rank", "--edit-start-sigma"):
            self.assertNotIn(forbidden, command)

    def test_pair_decision_routes_failed_balancing_to_conflict_projection(self):
        raw = summary("raw-mean-control", overfit=False, dev=False)
        balanced = summary("unit-balanced-norm-matched", overfit=False, dev=False)
        comparison._validate(raw, balanced)
        decision, _reason = comparison._decision(raw, balanced)
        self.assertEqual(decision, "reject_unit_balance_as_sufficient_test_conflict_projection")

    def test_pair_validation_rejects_m0_drift(self):
        raw = summary("raw-mean-control", overfit=False, dev=False)
        balanced = summary("unit-balanced-norm-matched", overfit=False, dev=False)
        balanced["comparisons"]["formal_select_32_endpoint_vs_m0"]["m0_mean_ce"] = 11.1
        with self.assertRaisesRegex(ValueError, "M0 drift"):
            comparison._validate(raw, balanced)


if __name__ == "__main__":
    unittest.main()
