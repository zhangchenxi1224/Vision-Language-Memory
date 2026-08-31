from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.schema import QuerySpec  # noqa: E402
from vision_memory.training.r5_compose import R5Event, R5Segment  # noqa: E402
from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_OPTIMIZER_STEPS,
    build_single_target_schedule,
    select_f1_targets,
    target_gate,
    target_statistics,
    target_training_view_counts,
)


def segment(index: int) -> R5Segment:
    event = R5Event(
        source_episode_id=f"episode-{index}",
        source_turn_index=0,
        entity_id=f"entity-{index}",
        event_kind="set",
        event_text=f"Remember value {index}.",
    )
    return R5Segment(
        segment_id=f"r5-f1-test-{index:02d}",
        family="F1",
        events=(event,),
        query=QuerySpec(
            text="What is the value?",
            choices=(f"value-{index}", "other-a", "other-b", "other-c"),
            target_index=0,
            comparison_id=f"comparison-{index}",
            target_token_count=1,
        ),
        query_source_episode_id=f"episode-{index}",
        query_turn_index=1,
        query_entity_id=f"entity-{index}",
        target_event_position=0,
        cross_slot_interference=False,
    )


class R10AlignmentTest(unittest.TestCase):
    def test_machine_readable_preregistration_is_locked(self):
        config = json.loads(
            (ROOT / "configs" / "experiments" / "r10_visual_alignment_lower_bound.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["protocol"], "R10-VisualAlignment-LowerBound")
        self.assertEqual(config["target_selection"]["target_count"], 8)
        self.assertEqual(config["arm_gate"]["required_target_passes"], 8)
        self.assertTrue(config["success_boundary"]["diagnostic_only"])

    def test_selection_is_hash_stable_without_outcome_fields(self):
        values = tuple(segment(index) for index in range(20))
        first = select_f1_targets({"F1": values}, count=8, seed=17, enforce_locked=False)
        second = select_f1_targets({"F1": tuple(reversed(values))}, count=8, seed=17, enforce_locked=False)
        self.assertEqual([value.segment_id for value in first], [value.segment_id for value in second])

    def test_schedule_has_one_target_and_exactly_32_training_views(self):
        values = tuple(segment(index) for index in range(8))
        schedule = build_single_target_schedule(values, target_index=3)
        self.assertEqual(len(schedule), R10_OPTIMIZER_STEPS * 8)
        target_id = values[3].segment_id
        self.assertEqual(sum(unit.segment.segment_id == target_id for unit in schedule), 128)
        self.assertEqual(
            target_training_view_counts(schedule, target_segment_id=target_id),
            {0: 32, 1: 32, 2: 32, 3: 32},
        )
        for step in range(R10_OPTIMIZER_STEPS):
            block = schedule[step * 8 : (step + 1) * 8]
            self.assertEqual(len({unit.segment.segment_id for unit in block}), 8)

    def test_target_gate_requires_all_preregistered_conditions(self):
        target = "target"
        rows = []
        for checkpoint in ("m0", "endpoint"):
            for condition in ("normal", "reset"):
                for view in range(4):
                    learned = checkpoint == "endpoint" and condition == "normal"
                    rows.append(
                        {
                            "suite": "r10",
                            "checkpoint": checkpoint,
                            "condition": condition,
                            "pair_unit": target,
                            "view_index": view,
                            "ce": 6.0 + view / 10 if learned else 10.0 + view / 10,
                            "correct": learned,
                        }
                    )
        statistics = target_statistics(
            rows,
            suite="r10",
            target_segment_id=target,
            endpoint="endpoint",
        )
        self.assertTrue(target_gate(statistics, technical_gate=True))
        self.assertFalse(target_gate(statistics, technical_gate=False))
        self.assertEqual(statistics["improved_choice_views"], 4)
        self.assertEqual(statistics["accuracy_delta"], 1.0)


if __name__ == "__main__":
    unittest.main()
