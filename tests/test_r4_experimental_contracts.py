from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train" / "dreamlite_r4_free_pixel.py"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

spec = importlib.util.spec_from_file_location("r4_experimental_under_test", SCRIPT)
assert spec and spec.loader
R4 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = R4
spec.loader.exec_module(R4)


class R4ExperimentalContractTest(unittest.TestCase):
    def args(self, **overrides):
        values = {
            "max_optimizer_steps": 32,
            "set_warmup_optimizer_steps": 8,
            "eval_limit": 16,
            "identity_calibration_count": 8,
            "checkpoint_every": 8,
            "gradient_accumulation": 4,
            "experimental": True,
            "smoke": False,
            "strict_determinism": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_experimental_budget_is_configurable(self):
        budget = R4.resolve_budget(self.args())
        self.assertEqual(budget.scope, "experimental")
        self.assertEqual(budget.gradient_accumulation, 4)
        self.assertEqual(budget.total_micro_transitions, 128)
        self.assertEqual(budget.balanced_micro_transitions, 96)

    def test_selected_step_count_two_covers_early_and_late_pairs(self):
        self.assertEqual(R4._selected_step_indices(0, 2), (0, 2))
        self.assertEqual(R4._selected_step_indices(1, 2), (1, 3))
        self.assertEqual(R4._selected_step_indices(2, 2), (0, 2))
        self.assertEqual(R4._selected_step_indices(3, 2), (1, 3))

    def test_formal_profile_still_rejects_custom_accumulation(self):
        with self.assertRaisesRegex(ValueError, "fixed"):
            R4.resolve_budget(self.args(experimental=False, gradient_accumulation=4, max_optimizer_steps=256, set_warmup_optimizer_steps=32))

if __name__ == "__main__":
    unittest.main()

