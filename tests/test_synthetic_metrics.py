from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.eval import compute_synthetic_metrics  # noqa: E402


def row(episode, prediction, *, condition="standard", variant=None, **extra):
    return {
        "episode_id": episode,
        "query_id": "q",
        "method": "main",
        "seed": 0,
        "prediction_index": prediction,
        "target_index": 0,
        "subtype": "overwrite",
        "split": "test_id",
        "condition": condition,
        "counterfactual_pair_id": episode,
        "counterfactual_variant": variant,
        **extra,
    }


class SyntheticMetricTest(unittest.TestCase):
    def test_intervention_and_counterfactual_metrics(self):
        rows = [
            row("a", 0, variant="clean"),
            row("a", 1, variant="distractor"),
            row("b", 0),
            row("b", 1, condition="reset"),
            row("b", 1, condition="shuffle"),
            row("b", 1, condition="state_swap", donor_target_index=1),
        ]
        result = compute_synthetic_metrics(rows)
        self.assertEqual(result["matched_distractor_damage"]["accuracy_damage"], 1.0)
        self.assertEqual(result["reset"]["accuracy_drop"], 1.0)
        self.assertEqual(result["shuffle"]["accuracy_drop"], 1.0)
        self.assertEqual(result["state_swap_donor_answer"]["rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
