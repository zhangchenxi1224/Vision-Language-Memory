from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


controller = load_script(
    "r6_controller_under_test",
    ROOT / "scripts" / "inspire" / "run_r6_source_anchor_arm.py",
)
comparison = load_script(
    "r6_comparison_under_test",
    ROOT / "scripts" / "experiments" / "compare_r6_source_anchor.py",
)


def summary(arm: str, *, overfit: bool, dev: bool, formal_delta: float = -1.0):
    return {
        "schema": "vision_memory.r6-source-anchor-summary.v1",
        "status": "completed",
        "arm": arm,
        "edit_start_sigma": 1.0 if arm == "legacy-pure-noise" else 0.5,
        "full_success_claim_allowed": False,
        "selected_segments_sha256": "same",
        "gates": {
            "hard8_overfit_learnability_gate": overfit,
            "fixed_dev_generalization_gate": dev,
        },
        "comparisons": {
            "formal_select_32_endpoint_vs_m0": {"estimate": formal_delta},
        },
    }


class R6OrchestrationTest(unittest.TestCase):
    def test_controller_command_has_no_free_hyperparameter_sweep(self):
        args = SimpleNamespace(
            arm="source-anchored",
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
        self.assertEqual(command[command.index("--arm") + 1], "source-anchored")
        for forbidden in ("--learning-rate", "--gradient-clip", "--lora-rank"):
            self.assertNotIn(forbidden, command)

    def test_source_anchor_only_pass_advances_without_formal_claim(self):
        legacy = summary("legacy-pure-noise", overfit=False, dev=False)
        anchored = summary("source-anchored", overfit=True, dev=False)
        decision, _reason = comparison._decision(legacy, anchored)
        self.assertEqual(decision, "advance_source_anchor_full_data_pilot")

    def test_neither_local_gate_passes_routes_to_gradient_balancing(self):
        legacy = summary("legacy-pure-noise", overfit=False, dev=False)
        anchored = summary("source-anchored", overfit=False, dev=False)
        decision, _reason = comparison._decision(legacy, anchored)
        self.assertEqual(decision, "reject_sigma_as_sufficient_test_gradient_balancing")

    def test_both_fixed_dev_pass_use_preregistered_formal_delta_tie_break(self):
        legacy = summary("legacy-pure-noise", overfit=True, dev=True, formal_delta=-0.2)
        anchored = summary("source-anchored", overfit=True, dev=True, formal_delta=-0.5)
        decision, _reason = comparison._decision(legacy, anchored)
        self.assertEqual(decision, "advance_source_anchor_full_data_pilot")

    def test_pair_validation_rejects_lineage_drift_and_success_claim(self):
        legacy = summary("legacy-pure-noise", overfit=True, dev=True)
        anchored = summary("source-anchored", overfit=True, dev=True)
        anchored["selected_segments_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "drift"):
            comparison._validate(legacy, anchored)
        anchored["selected_segments_sha256"] = "same"
        anchored["full_success_claim_allowed"] = True
        with self.assertRaisesRegex(ValueError, "formal success"):
            comparison._validate(legacy, anchored)


if __name__ == "__main__":
    unittest.main()
