from __future__ import annotations

import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import DatasetSizes, generate_dataset, read_jsonl  # noqa: E402
from vision_memory.training.r5_compose import (  # noqa: E402
    build_r5_family_pools,
    build_r5_schedule,
    family_composition_for_step,
    family_pool_audit,
    mechanism_subsets,
    schedule_audit,
    stable_record_split,
)


class R5ComposeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        generate_dataset(
            root,
            sizes=DatasetSizes(train=504, dev=504, test_id=504, test_ood=504),
            seed=2026,
            transition_profile="full",
        )
        cls.train = read_jsonl(root / "train.jsonl")
        cls.dev = read_jsonl(root / "dev.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_family_pools_encode_the_preregistered_state_algebra(self):
        pools = build_r5_family_pools(self.train, pairing_seed=0)
        audit = family_pool_audit(pools)
        self.assertEqual(set(audit["family_counts"]), {"F1", "F2", "F3", "F4", "F5", "F6"})
        self.assertGreater(min(audit["family_counts"].values()), 0)

        examples = {family: pools[family][0] for family in pools}
        self.assertEqual([event.event_kind for event in examples["F1"].events], ["set"])
        self.assertEqual([event.event_kind for event in examples["F3"].events], ["set", "overwrite"])
        self.assertEqual([event.event_kind for event in examples["F4"].events], ["set", "clear"])
        self.assertEqual([event.event_kind for event in examples["F5"].events], ["set", "set", "overwrite"])
        self.assertEqual([event.event_kind for event in examples["F6"].events], ["set", "set", "overwrite"])
        for family in ("F2", "F5", "F6"):
            self.assertGreater(len({event.entity_id for event in examples[family].events}), 1)
            self.assertTrue(examples[family].cross_slot_interference)
        self.assertEqual(examples["F5"].query_gap, 3)
        self.assertEqual(examples["F6"].query_gap, 3)

    def test_640_step_schedule_has_fixed_eight_micro_composition(self):
        pools = build_r5_family_pools(self.train, pairing_seed=0)
        schedule = build_r5_schedule(
            pools,
            optimizer_steps=640,
            schedule_seed=0,
            selected_step_count=2,
        )
        self.assertEqual(len(schedule), 5120)
        for step in range(640):
            units = schedule[step * 8 : (step + 1) * 8]
            self.assertEqual(Counter(unit.segment.family for unit in units), Counter(family_composition_for_step(step)))
        audit = schedule_audit(schedule)
        self.assertEqual(audit["optimizer_steps"], 640)
        self.assertEqual(audit["selected_step_set_counts"], {"(0, 2)": 2560, "(1, 3)": 2560})
        self.assertEqual(Counter(unit.segment.family for unit in schedule[256 * 8 : 257 * 8]), Counter({
            "F1": 2,
            "F2": 2,
            "F3": 2,
            "F4": 1,
            "F5": 1,
        }))
        self.assertEqual(Counter(unit.segment.family for unit in schedule[257 * 8 : 258 * 8]), Counter({
            "F1": 2,
            "F2": 2,
            "F3": 2,
            "F4": 1,
            "F6": 1,
        }))

    def test_dev_and_mechanism_selection_are_disjoint_and_deterministic(self):
        split_a = stable_record_split(self.dev, seed=17)
        split_b = stable_record_split(self.dev, seed=17)
        self.assertEqual(
            [[episode.episode_id for episode in split_a[key]] for key in ("select", "final", "reserve")],
            [[episode.episode_id for episode in split_b[key]] for key in ("select", "final", "reserve")],
        )
        ids = [{episode.episode_id for episode in split_a[key]} for key in ("select", "final", "reserve")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

        pools = build_r5_family_pools(self.dev, pairing_seed=17)
        mechanism = mechanism_subsets(pools, seed=17)
        self.assertEqual(len(mechanism["select"]), 32)
        self.assertEqual(len(mechanism["final"]), 128)
        self.assertFalse(
            {item.segment_id for item in mechanism["select"]}
            & {item.segment_id for item in mechanism["final"]}
        )
        self.assertGreaterEqual(sum(item.query_gap >= 3 for item in mechanism["final"]), 64)


if __name__ == "__main__":
    unittest.main()
