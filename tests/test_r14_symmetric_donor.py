from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training.r14_symmetric_donor import (  # noqa: E402
    R14_PAIR_SEED,
    R14_RANKING_MARGIN,
    choice_target_margin,
    symmetric_donor_mapping,
    symmetric_pairing_audit,
    symmetric_ranking_loss,
)


def _segment(target: str, member: int) -> SimpleNamespace:
    return SimpleNamespace(
        segment_id=f"{target}-{member}",
        family="F1",
        events=(object(),),
        query=SimpleNamespace(target_index=0, choices=(target, "alt-a", "alt-b", "alt-c")),
    )


class R14SymmetricDonorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = tuple(
            _segment(target, member) for target in ("alpha", "beta", "gamma", "delta") for member in range(2)
        )

    def test_pairing_is_order_invariant_fixed_point_free_and_cross_value(self) -> None:
        expected = symmetric_donor_mapping(self.segments)
        shuffled = list(self.segments)
        random.Random(17).shuffle(shuffled)
        self.assertEqual(expected, symmetric_donor_mapping(shuffled))
        by_id = {segment.segment_id: segment for segment in self.segments}
        for source, donor in expected.items():
            self.assertNotEqual(source, donor)
            self.assertEqual(expected[donor], source)
            self.assertNotEqual(by_id[source].query.choices[0], by_id[donor].query.choices[0])

    def test_pairing_audit_is_deterministic_and_complete(self) -> None:
        mapping = symmetric_donor_mapping(self.segments)
        first = symmetric_pairing_audit(self.segments, mapping)
        second = symmetric_pairing_audit(tuple(reversed(self.segments)), mapping)
        self.assertEqual(first, second)
        self.assertEqual(first["seed"], R14_PAIR_SEED)
        self.assertEqual(first["segment_count"], 8)
        self.assertEqual(first["pair_count"], 4)
        self.assertTrue(first["different_target_value"])
        self.assertTrue(first["involution"])

    def test_pairing_rejects_odd_target_group_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "even number"):
            symmetric_donor_mapping(tuple(_segment(target, 0) for target in ("a", "b", "c")))

    def test_active_ranking_hinge_pushes_own_down_and_donor_up(self) -> None:
        own = torch.tensor(2.0, requires_grad=True)
        donor = torch.tensor(0.5, requires_grad=True)
        loss = symmetric_ranking_loss(own, donor)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), math.log(4.0) + 1.5, places=6)
        self.assertEqual(float(own.grad), 1.0)
        self.assertEqual(float(donor.grad), -1.0)

    def test_satisfied_ranking_hinge_is_zero(self) -> None:
        own = torch.tensor(0.2)
        donor = torch.tensor(0.2 + R14_RANKING_MARGIN + 0.1)
        self.assertEqual(float(symmetric_ranking_loss(own, donor)), 0.0)

    def test_choice_target_margin_uses_strongest_alternative(self) -> None:
        logits = torch.tensor([1.0, 4.0, 2.0, 3.0])
        self.assertEqual(float(choice_target_margin(logits, 1)), 1.0)


if __name__ == "__main__":
    unittest.main()
