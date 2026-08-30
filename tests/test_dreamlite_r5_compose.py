from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
SCRIPT = ROOT / "scripts" / "train" / "dreamlite_r5_compose.py"
SPEC = importlib.util.spec_from_file_location("dreamlite_r5_compose_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
r5 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r5
SPEC.loader.exec_module(r5)

from vision_memory.data.schema import QuerySpec  # noqa: E402
from vision_memory.training.r5_compose import R5Event, R5Segment  # noqa: E402


def segment_three_events() -> R5Segment:
    query = QuerySpec("Which value?", ("a", "b", "c", "d"), 0)
    events = (
        R5Event("episode-a", 0, "entity-a", "set", "set A"),
        R5Event("episode-b", 0, "entity-b", "set", "set B"),
        R5Event("episode-b", 2, "entity-b", "overwrite", "overwrite B"),
    )
    return R5Segment(
        segment_id="r5-test-segment",
        family="F5",
        events=events,
        query=query,
        query_source_episode_id="episode-a",
        query_turn_index=1,
        query_entity_id="entity-a",
        target_event_position=0,
        cross_slot_interference=True,
    )


class TinyStateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.1))
        self.initial_state = torch.ones(1, 3, 2, 2)
        self.calls = []

    def reset_state(self):
        return self.initial_state.clone()

    def apply_event(self, state, event, *, gradient_mode, selected_step_indices):
        self.calls.append((event.event_text, gradient_mode, torch.is_grad_enabled(), state.requires_grad))
        return state + self.gain * (1.0 if event.entity_id == "entity-a" else 2.0)

    def reader_image(self, state):
        return state


def tiny_reader(image, query, choices, target_index):
    del query, choices, target_index
    logits = torch.stack((image.mean(), image.mean() * 0, image.mean() * 0, image.mean() * 0))
    return SimpleNamespace(loss=image.square().mean(), choice_logits=logits)


class R5TrainerCoreTest(unittest.TestCase):
    def test_lr_schedule_matches_fixed_endpoints(self):
        self.assertEqual(r5.r5_learning_rate(0), r5.LR_START)
        self.assertAlmostEqual(r5.r5_learning_rate(15), r5.LR_PEAK)
        self.assertAlmostEqual(r5.r5_learning_rate(639), r5.LR_FINAL)
        self.assertGreater(r5.r5_learning_rate(16), r5.r5_learning_rate(638))
        self.assertEqual(r5.r5_learning_rate(127), r5.r5_learning_rate(127, endpoint_steps=128))

    def test_h2_replays_oldest_event_and_keeps_two_event_state_path(self):
        model = TinyStateModel()
        forward = r5._segment_forward(
            model=model,
            segment=segment_three_events(),
            reader_fn=tiny_reader,
            tbptt_horizon=2,
            gradient_mode="drtune_stateful",
            selected_step_indices=(0, 2),
            choice_permutation=(0, 1, 2, 3),
            audit_state_gradients=True,
        )
        forward.loss.backward()
        self.assertEqual(forward.prefix_event_count, 1)
        self.assertFalse(model.calls[0][2])
        self.assertEqual([call[1] for call in model.calls[1:]], ["drtune_stateful", "drtune_stateful"])
        self.assertGreater(forward.boundary_state.grad.norm().item(), 0.0)
        self.assertGreater(forward.earliest_output_state.grad.norm().item(), 0.0)
        self.assertGreater(model.gain.grad.abs().item(), 0.0)

    def test_h4_unrolls_all_three_events(self):
        model = TinyStateModel()
        forward = r5._segment_forward(
            model=model,
            segment=segment_three_events(),
            reader_fn=tiny_reader,
            tbptt_horizon=4,
            gradient_mode="drtune_stateful",
            selected_step_indices=(1, 3),
            choice_permutation=(0, 1, 2, 3),
            audit_state_gradients=True,
        )
        forward.loss.backward()
        self.assertEqual(forward.prefix_event_count, 0)
        self.assertTrue(all(call[2] for call in model.calls))
        self.assertGreater(forward.boundary_state.grad.norm().item(), 0.0)

    def test_hard_noop_returns_same_tensor_without_updater_call(self):
        model = object.__new__(r5.R5ComposeModel)
        nn.Module.__init__(model)
        model.persistent_state = "float_rgb"
        model.register_buffer("initial_state", torch.zeros(1, 3, 2, 2), persistent=False)
        model.updater = SimpleNamespace()
        state = model.reset_state()
        event = R5Event("episode", 0, "entity", "noop", "unrelated")
        output = model.apply_event(state, event, gradient_mode="full", selected_step_indices=None)
        self.assertIs(output, state)
        self.assertEqual(output.data_ptr(), state.data_ptr())

    def test_ema_apply_restores_raw_parameters(self):
        parameter = nn.Parameter(torch.tensor([1.0]))
        named = [("lora", parameter)]
        ema = r5.TrainableEMA(named, decay=0.5)
        with torch.no_grad():
            parameter.fill_(3.0)
        ema.update(named)
        self.assertEqual(ema.shadow["lora"].item(), 2.0)
        with ema.apply(named):
            self.assertEqual(parameter.item(), 2.0)
        self.assertEqual(parameter.item(), 3.0)

    def test_gradient_policy_selection_and_fallback_are_preregistered(self):
        records = []
        for kind in ("set", "overwrite", "clear"):
            for index in range(8):
                records.append({
                    "count_group": "K1",
                    "cosine_to_full": 0.2,
                    "norm_ratio_to_full": 1.0,
                    "target_event_kind": kind,
                })
                records.append({
                    "count_group": "K2",
                    "cosine_to_full": 0.6,
                    "norm_ratio_to_full": 1.0,
                    "target_event_kind": kind,
                })
        selected = r5._select_gradient_policy(records)
        self.assertEqual(selected["selected_step_count"], 2)
        for record in records:
            record["cosine_to_full"] = -0.1
        fallback = r5._select_gradient_policy(records)
        self.assertEqual(fallback["gradient_mode"], "full")
        self.assertEqual((fallback["persistent_state"], fallback["tbptt_horizon"]), ("latent", 2))

    def test_paired_bootstrap_uses_matched_units(self):
        rows = []
        for unit in ("a", "b", "c"):
            rows.extend((
                {"pair_unit": unit, "condition": "normal", "ce": 1.0},
                {"pair_unit": unit, "condition": "reset", "ce": 2.0},
            ))
        report = r5._paired_bootstrap(
            rows,
            condition_a="normal",
            condition_b="reset",
            iterations=100,
            seed=7,
        )
        self.assertEqual(report["estimate"], -1.0)
        self.assertEqual(report["ci95"], [-1.0, -1.0])


if __name__ == "__main__":
    unittest.main()
