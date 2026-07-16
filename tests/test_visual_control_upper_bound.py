from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probes.qwen_visual_control_upper_bound import (  # noqa: E402
    DISCLAIMER,
    deterministic_query_order,
    grouped_accuracy,
)


class VisualControlUpperBoundTest(unittest.TestCase):
    def test_query_order_is_deterministic_complete_and_seeded(self):
        first = deterministic_query_order(8, steps=19, seed=7)
        second = deterministic_query_order(8, steps=19, seed=7)
        different = deterministic_query_order(8, steps=19, seed=8)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(set(first[:8]), set(range(8)))
        self.assertEqual(len(first), 19)

    def test_grouped_accuracy_and_disclaimer_make_supervision_explicit(self):
        records = [
            {"pattern": "p0", "correct": True},
            {"pattern": "p0", "correct": False},
            {"pattern": "p1", "correct": True},
        ]
        result = grouped_accuracy(records, "pattern")
        self.assertEqual(result["p0"], {"correct": 1, "count": 2, "accuracy": 0.5})
        self.assertEqual(result["p1"], {"correct": 1, "count": 1, "accuracy": 1.0})
        self.assertIn("TARGET-SUPERVISED DIAGNOSTIC ONLY", DISCLAIMER)
        self.assertIn("not a memory updater", DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
