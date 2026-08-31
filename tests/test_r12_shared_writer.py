from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.schema import QuerySpec  # noqa: E402
from vision_memory.training.r5_compose import R5Event, R5Segment  # noqa: E402
from vision_memory.training.r12_shared_writer import (  # noqa: E402
    R12_CHECKPOINT_STEPS,
    R12_MICRO_STEPS,
    R12_OPTIMIZER_STEPS,
    build_training_schedule,
    conditioned_target_gate,
    conditioned_target_statistics,
    donor_derangement,
    select_balanced_train_f1,
    select_entity_disjoint_dev_f1,
    selection_audit,
)


def segment(*, value: str, position: int, serial: int, prefix: str) -> R5Segment:
    choices = [f"distractor-{serial}-{index}" for index in range(4)]
    choices[position] = value
    entity = f"{prefix}-entity-{serial}"
    episode = f"{prefix}-episode-{serial}"
    return R5Segment(
        segment_id=f"r5-f1-{prefix}-{serial:05d}",
        family="F1",
        events=(
            R5Event(
                source_episode_id=episode,
                source_turn_index=0,
                entity_id=entity,
                event_kind="set",
                event_text=f"Remember {value} for {entity}.",
            ),
        ),
        query=QuerySpec(
            text=f"What is remembered for {entity}?",
            choices=tuple(choices),
            target_index=position,
            comparison_id=f"comparison-{prefix}-{serial}",
            target_token_count=1,
        ),
        query_source_episode_id=episode,
        query_turn_index=1,
        query_entity_id=entity,
        target_event_position=0,
        cross_slot_interference=False,
    )


def balanced_pool(*, value_count: int, copies_per_cell: int, prefix: str) -> tuple[R5Segment, ...]:
    values = []
    serial = 0
    for value_index in range(value_count):
        for position in range(4):
            for _copy in range(copies_per_cell):
                values.append(
                    segment(
                        value=f"value-{value_index:02d}",
                        position=position,
                        serial=serial,
                        prefix=prefix,
                    )
                )
                serial += 1
    return tuple(values)


class R12SharedWriterContractTest(unittest.TestCase):
    def test_train_selection_is_stable_balanced_and_unique_entity(self):
        pool = balanced_pool(value_count=36, copies_per_cell=2, prefix="train")
        selected, audit = select_balanced_train_f1(pool, enforce_locked=False)
        reversed_selected, reversed_audit = select_balanced_train_f1(
            tuple(reversed(pool)), enforce_locked=False
        )
        self.assertEqual(
            [value.segment_id for value in selected],
            [value.segment_id for value in reversed_selected],
        )
        self.assertEqual(
            [value.segment_id for value in audit],
            [value.segment_id for value in reversed_audit],
        )
        report = selection_audit(selected)
        self.assertEqual(report["count"], 144)
        self.assertEqual(report["unique_entities"], 144)
        self.assertEqual(report["unique_target_values"], 36)
        self.assertEqual(report["target_index_counts"], {0: 36, 1: 36, 2: 36, 3: 36})
        self.assertEqual(selection_audit(audit)["target_index_counts"], {0: 9, 1: 9, 2: 9, 3: 9})

    def test_dev_splits_are_value_balanced_and_entity_disjoint(self):
        pool = balanced_pool(value_count=24, copies_per_cell=3, prefix="dev")
        select, final = select_entity_disjoint_dev_f1(pool, enforce_locked=False)
        self.assertEqual(len(select), 24)
        self.assertEqual(len(final), 24)
        self.assertEqual(selection_audit(select)["target_index_counts"], {0: 6, 1: 6, 2: 6, 3: 6})
        self.assertEqual(selection_audit(final)["target_index_counts"], {0: 6, 1: 6, 2: 6, 3: 6})
        self.assertFalse(
            {value.query_entity_id for value in select}
            & {value.query_entity_id for value in final}
        )
        self.assertEqual(set(donor_derangement(select)), {value.segment_id for value in select})

    def test_schedule_has_exact_views_and_fixed_endpoint(self):
        pool = balanced_pool(value_count=36, copies_per_cell=1, prefix="train")
        selected, _audit = select_balanced_train_f1(pool, enforce_locked=False)
        schedule = build_training_schedule(selected)
        self.assertEqual(len(schedule), R12_MICRO_STEPS)
        self.assertEqual(schedule[-1].optimizer_step_zero + 1, R12_OPTIMIZER_STEPS)
        self.assertEqual(R12_CHECKPOINT_STEPS, (0, 288, 576, 864, 1152))
        counts = Counter(
            (unit.segment.segment_id, unit.forward_cyclic_training_view)
            for unit in schedule
        )
        for value in selected:
            self.assertEqual([counts[(value.segment_id, view)] for view in range(4)], [8] * 4)

    def test_conditioned_gate_requires_reset_and_wrong_event_controls(self):
        target_id = "target"
        rows = []
        for checkpoint in ("m0", "shared_step1152"):
            for condition in ("normal", "reset", "donor"):
                for view in range(4):
                    if checkpoint == "m0":
                        ce, correct = 10.0, 0
                    elif condition == "normal":
                        ce, correct = 5.0, 1
                    elif condition == "reset":
                        ce, correct = 10.0, 0
                    else:
                        ce, correct = 8.0, 0
                    rows.append(
                        {
                            "suite": "suite",
                            "checkpoint": checkpoint,
                            "condition": condition,
                            "pair_unit": target_id,
                            "view_index": view,
                            "ce": ce,
                            "correct": correct,
                        }
                    )
        statistics = conditioned_target_statistics(
            rows,
            suite="suite",
            target_segment_id=target_id,
            endpoint="shared_step1152",
        )
        self.assertTrue(conditioned_target_gate(statistics, technical_gate=True))
        self.assertEqual(statistics["normal_better_than_donor_views"], 4)
        for row in rows:
            if row["checkpoint"] == "shared_step1152" and row["condition"] == "donor":
                row["ce"] = 4.0
                row["correct"] = 1
        shortcut = conditioned_target_statistics(
            rows,
            suite="suite",
            target_segment_id=target_id,
            endpoint="shared_step1152",
        )
        self.assertFalse(conditioned_target_gate(shortcut, technical_gate=True))


if __name__ == "__main__":
    unittest.main()
