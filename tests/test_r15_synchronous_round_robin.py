from __future__ import annotations

import random
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training.r15_synchronous_round_robin import (  # noqa: E402
    R15_BACKWARD_LOSS_DIVISOR,
    R15_CHECKPOINT_STEPS,
    R15_DIRECTIONAL_EXAMPLES,
    R15_OPTIMIZER_STEPS,
    R15_PAIR_MICRO_STEPS,
    R15_READER_CALLS,
    build_synchronous_round_robin_schedule,
    round_robin_target_rounds,
    synchronous_pair_objective,
    synchronous_schedule_audit,
)


def _segment(target: str, member: int) -> SimpleNamespace:
    return SimpleNamespace(
        segment_id=f"{target}-{member}",
        family="F1",
        events=(object(),),
        query=SimpleNamespace(target_index=0, choices=(target, "alt-a", "alt-b", "alt-c")),
    )


class R15SynchronousRoundRobinTest(unittest.TestCase):
    def setUp(self) -> None:
        self.segments = tuple(
            _segment(target, member) for target in ("alpha", "beta", "gamma", "delta") for member in range(2)
        )

    def _schedule(self, segments=None):
        return build_synchronous_round_robin_schedule(
            self.segments if segments is None else segments,
            epochs=4,
            pair_gradient_accumulation=2,
            enforce_locked=False,
        )

    def test_locked_arithmetic_is_exact(self) -> None:
        self.assertEqual(R15_PAIR_MICRO_STEPS, 2592)
        self.assertEqual(R15_DIRECTIONAL_EXAMPLES, 5184)
        self.assertEqual(R15_READER_CALLS, 10368)
        self.assertEqual(R15_OPTIMIZER_STEPS, 1296)
        self.assertEqual(R15_CHECKPOINT_STEPS, (0, 324, 648, 972, 1296))
        self.assertEqual(R15_BACKWARD_LOSS_DIVISOR, 4.0)

    def test_one_factorization_covers_every_value_pair_once(self) -> None:
        rounds = round_robin_target_rounds(("alpha", "beta", "gamma", "delta"))
        self.assertEqual(len(rounds), 3)
        pairs = [frozenset(pair) for round_pairs in rounds for pair in round_pairs]
        self.assertEqual(len(pairs), 6)
        self.assertEqual(len(set(pairs)), 6)
        for round_pairs in rounds:
            self.assertEqual({value for pair in round_pairs for value in pair}, {"alpha", "beta", "gamma", "delta"})

    def test_schedule_is_input_order_invariant_and_hash_stable(self) -> None:
        expected = self._schedule()
        shuffled = list(self.segments)
        random.Random(17).shuffle(shuffled)
        observed = self._schedule(shuffled)
        self.assertEqual(
            [unit.receipt() for unit in expected],
            [unit.receipt() for unit in observed],
        )
        first = synchronous_schedule_audit(
            self.segments,
            expected,
            epochs=4,
            pair_gradient_accumulation=2,
        )
        second = synchronous_schedule_audit(
            tuple(reversed(self.segments)),
            observed,
            epochs=4,
            pair_gradient_accumulation=2,
        )
        self.assertTrue(first["passed"])
        self.assertEqual(first["schedule_sha256"], second["schedule_sha256"])

    def test_synthetic_schedule_has_complete_balanced_atomic_coverage(self) -> None:
        schedule = self._schedule()
        audit = synchronous_schedule_audit(
            self.segments,
            schedule,
            epochs=4,
            pair_gradient_accumulation=2,
        )
        self.assertTrue(audit["passed"], audit["checks"])
        self.assertEqual(audit["pair_micro_steps"], 16)
        self.assertEqual(audit["directional_examples"], 32)
        self.assertEqual(audit["reader_calls"], 64)
        self.assertEqual(audit["optimizer_steps"], 8)
        self.assertEqual(audit["member_shifts"], [0, 1, 0, 1])
        self.assertEqual(audit["value_pairs_in_complete_factorization"], 6)
        self.assertTrue(audit["checks"]["repeat_event_pairs_disjoint"])
        for step in range(8):
            batch = [unit for unit in schedule if unit.optimizer_step_zero == step]
            self.assertEqual(len(batch), 2)
            self.assertEqual(len({unit.epoch_zero for unit in batch}), 1)
            self.assertEqual(
                len(
                    {segment.query.choices[0] for unit in batch for segment in (unit.left_segment, unit.right_segment)}
                ),
                4,
            )

    def test_audit_fails_closed_on_optimizer_boundary_drift(self) -> None:
        schedule = list(self._schedule())
        schedule[1] = replace(schedule[1], optimizer_step_zero=1, pair_in_optimizer_step=0)
        audit = synchronous_schedule_audit(
            self.segments,
            schedule,
            epochs=4,
            pair_gradient_accumulation=2,
        )
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["checks"]["optimizer_atomic_distinct_pairs"])

    def test_pair_objective_is_scaled_exactly_once(self) -> None:
        left_own = torch.tensor(2.0, requires_grad=True)
        left_donor = torch.tensor(0.5, requires_grad=True)
        right_own = torch.tensor(1.5, requires_grad=True)
        right_donor = torch.tensor(0.25, requires_grad=True)
        left_regularizer = torch.tensor(0.4, requires_grad=True)
        right_regularizer = torch.tensor(0.8, requires_grad=True)
        result = synchronous_pair_objective(
            left_own,
            left_donor,
            right_own,
            right_donor,
            left_regularizer,
            right_regularizer,
        )
        self.assertAlmostEqual(float(result.backward_loss.detach()), float(result.pair_sum.detach()) / 4.0)
        result.backward_loss.backward()
        self.assertEqual(float(left_own.grad), 0.5)
        self.assertEqual(float(right_own.grad), 0.5)
        self.assertEqual(float(left_donor.grad), -0.25)
        self.assertEqual(float(right_donor.grad), -0.25)
        self.assertEqual(float(left_regularizer.grad), 0.25)
        self.assertEqual(float(right_regularizer.grad), 0.25)

    def test_invalid_group_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "even number"):
            build_synchronous_round_robin_schedule(
                tuple(_segment(target, member) for target in ("a", "b", "c") for member in range(2)),
                epochs=4,
                enforce_locked=False,
            )
        with self.assertRaisesRegex(ValueError, "equal members"):
            build_synchronous_round_robin_schedule(
                (*self.segments, _segment("alpha", 2)),
                epochs=4,
                enforce_locked=False,
            )


if __name__ == "__main__":
    unittest.main()
