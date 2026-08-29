from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts" / "train" / "dreamlite_r6_source_anchor.py"
SPEC = importlib.util.spec_from_file_location("dreamlite_r6_source_anchor_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
r6 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r6
SPEC.loader.exec_module(r6)

from vision_memory.data.schema import QuerySpec  # noqa: E402
from vision_memory.training.r5_compose import R5Event, R5Segment  # noqa: E402


def make_segment(index: int, family: str) -> R5Segment:
    events = (
        R5Event(f"episode-{index}", 0, f"entity-{index}", "set", f"set value {index}"),
        R5Event(f"episode-{index}", 2, f"entity-{index}", "overwrite", f"overwrite value {index}"),
    )
    return R5Segment(
        segment_id=f"r6-segment-{index}",
        family=family,
        events=events,
        query=QuerySpec("Which value?", ("a", "b", "c", "d"), index % 4),
        query_source_episode_id=f"episode-{index}",
        query_turn_index=3,
        query_entity_id=f"entity-{index}",
        target_event_position=1,
        cross_slot_interference=family in {"F5", "F6"},
    )


class R6SourceAnchorContractTest(unittest.TestCase):
    def test_arm_changes_only_preregistered_start_sigma(self):
        self.assertEqual(r6.ARM_SIGMA, {"legacy-pure-noise": 1.0, "source-anchored": 0.5})

    def test_repeated_hard8_schedule_is_complete_and_deterministic(self):
        families = ("F2", "F2", "F3", "F3", "F5", "F5", "F6", "F6")
        segments = tuple(make_segment(index, family) for index, family in enumerate(families))
        first = r6._overfit_schedule(segments, optimizer_steps=3, schedule_seed=7)
        second = r6._overfit_schedule(segments, optimizer_steps=3, schedule_seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        self.assertEqual([unit.global_micro_index for unit in first], list(range(24)))
        for step in range(3):
            subset = first[step * 8 : (step + 1) * 8]
            self.assertEqual({unit.segment.segment_id for unit in subset}, {s.segment_id for s in segments})
            self.assertTrue(all(unit.selected_step_indices is None for unit in subset))

    def test_endpoint_bootstrap_and_training_induced_state_did_use_matched_units(self):
        rows = []
        for unit in ("a", "b", "c"):
            for checkpoint, normal, reset in (
                ("m0", 4.0, 5.0),
                ("ema_step128", 2.0, 5.0),
            ):
                rows.extend(
                    (
                        {
                            "checkpoint": checkpoint,
                            "suite": "suite",
                            "condition": "normal",
                            "pair_unit": unit,
                            "ce": normal,
                        },
                        {
                            "checkpoint": checkpoint,
                            "suite": "suite",
                            "condition": "reset",
                            "pair_unit": unit,
                            "ce": reset,
                        },
                    )
                )
        endpoint = r6._checkpoint_comparison(
            rows,
            suite="suite",
            endpoint="ema_step128",
            iterations=100,
            seed=1,
        )
        did = r6._difference_in_differences(
            rows,
            suite="suite",
            endpoint="ema_step128",
            iterations=100,
            seed=2,
        )
        self.assertEqual(endpoint["estimate"], -2.0)
        self.assertEqual(endpoint["ci95"], [-2.0, -2.0])
        self.assertEqual(endpoint["improved_pair_units"], 3)
        self.assertEqual(did["estimate"], -2.0)
        self.assertEqual(did["ci95"], [-2.0, -2.0])

    def test_hard_noop_is_same_tensor_without_updater(self):
        model = object.__new__(r6.R6SourceAnchorModel)
        nn.Module.__init__(model)
        state = torch.zeros(1, 4, 2, 2)
        event = R5Event("episode", 0, "entity", "noop", "unrelated")
        output = model.apply_event(
            state,
            event,
            gradient_mode="full",
            selected_step_indices=None,
        )
        self.assertIs(output, state)
        self.assertEqual(output.data_ptr(), state.data_ptr())


if __name__ == "__main__":
    unittest.main()
