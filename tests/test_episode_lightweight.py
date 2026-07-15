from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import EventKind, Episode, QuerySpec, Turn, TurnType, run_episode  # noqa: E402
from vision_memory.lightweight import HashChoiceReader, LightweightVisualUpdater  # noqa: E402


def make_episode() -> Episode:
    query = QuerySpec("What is preferred?", ("red", "blue", "green", "yellow"), 0)
    return Episode(
        episode_id="train-0",
        split="train",
        seed=0,
        entity_id="entity",
        template_id="template",
        pair_id="pair",
        counterfactual_episode_id="train-1",
        topic="color",
        turns=(
            Turn(TurnType.EVENT, EventKind.SET, "The preferred color is red."),
            Turn(TurnType.EVENT, EventKind.NOOP, "A clock ticked."),
            Turn(TurnType.MIXED, EventKind.NOOP, "It rained elsewhere.", query),
            Turn(TurnType.QUERY, query=query),
        ),
    )


class LightweightUpdaterTest(unittest.TestCase):
    def make_updater(self):
        return LightweightVisualUpdater(
            state_channels=8,
            state_size=8,
            output_size=16,
            vocabulary_size=64,
            embedding_dim=16,
            text_hidden_dim=8,
        )

    def test_shape_and_episode_gradient_contract(self):
        torch.manual_seed(0)
        updater = self.make_updater()
        reader = HashChoiceReader(feature_size=2)
        output = run_episode(make_episode(), updater=updater, reader=reader)
        self.assertEqual(tuple(output.final_state.shape), (1, 8, 8, 8))
        self.assertEqual(tuple(output.final_image.shape), (1, 3, 16, 16))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        gradients = [parameter.grad for parameter in updater.parameters() if parameter.requires_grad]
        self.assertTrue(any(gradient is not None and gradient.norm().item() > 0 for gradient in gradients))
        self.assertEqual(sum(parameter.numel() for parameter in reader.parameters()), 0)

    def test_hashed_encoder_is_deterministic(self):
        updater = self.make_updater()
        first, lengths_a = updater.event_encoder.tokenize(["Remember red."], device=torch.device("cpu"))
        second, lengths_b = updater.event_encoder.tokenize(["Remember red."], device=torch.device("cpu"))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(lengths_a, lengths_b)


if __name__ == "__main__":
    unittest.main()
