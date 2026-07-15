from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import EventKind, Episode, QuerySpec, Turn, TurnType, run_episode  # noqa: E402


def make_routing_episode() -> Episode:
    query_a = QuerySpec("first query", ("a", "b", "c", "d"), 0, target_token_count=1)
    query_b = QuerySpec("mixed query", ("a", "b", "c", "d"), 1, target_token_count=2)
    return Episode(
        episode_id="test-0",
        split="test_id",
        seed=0,
        entity_id="entity",
        template_id="template",
        pair_id="pair",
        counterfactual_episode_id="test-1",
        topic="topic",
        turns=(
            Turn(TurnType.EVENT, EventKind.SET, "set"),
            Turn(TurnType.QUERY, query=query_a),
            Turn(TurnType.MIXED, EventKind.OVERWRITE, "overwrite", query_b),
            Turn(TurnType.EVENT, EventKind.NOOP, "distractor"),
        ),
    )


class LoggingUpdater:
    def __init__(self, log: list[str]):
        self.log = log
        self.scale = torch.nn.Parameter(torch.tensor(0.1))
        self.decode_calls = 0

    def initial_state(self, *, batch_size, device, dtype):
        return torch.zeros(batch_size, 1, 1, 1, device=device, dtype=dtype)

    def update(self, state, event_text):
        self.log.append(f"update:{event_text}")
        return state + self.scale * len(event_text)

    def render(self, state):
        return state.expand(-1, 3, 2, 2)

    def decode_reencode(self, state):
        self.decode_calls += 1
        return state + 0.01


class LoggingReader:
    def __init__(self, log: list[str]):
        self.log = log

    def __call__(self, *, image, query, choices):
        del choices
        self.log.append(f"read:{query}:{image.mean().item():.2f}")
        value = image.mean()
        return torch.stack([value, -value, value * 0.5, -value * 0.5]).unsqueeze(0)


class EpisodeRoutingTest(unittest.TestCase):
    def test_pure_query_is_read_only_and_mixed_updates_before_read(self):
        log: list[str] = []
        updater = LoggingUpdater(log)
        output = run_episode(make_routing_episode(), updater=updater, reader=LoggingReader(log))

        self.assertEqual(output.update_count, 3)
        self.assertEqual(output.query_count, 2)
        self.assertEqual(
            [entry.split(":", 1)[0] for entry in log],
            ["update", "read", "update", "read", "update"],
        )
        self.assertIn("read:first query:0.30", log)
        self.assertIn("read:mixed query:1.20", log)
        output.loss.backward()
        self.assertIsNotNone(updater.scale.grad)
        self.assertGreater(abs(updater.scale.grad.item()), 0.0)

    def test_detach_between_events_cuts_previous_state_gradient(self):
        query = QuerySpec("final query", ("a", "b", "c", "d"), 0)
        episode = Episode(
            episode_id="test-detach",
            split="test_id",
            seed=0,
            entity_id="entity",
            template_id="template",
            pair_id="pair-detach",
            counterfactual_episode_id="test-detach-mate",
            topic="topic",
            turns=(
                Turn(TurnType.EVENT, EventKind.SET, "first"),
                Turn(TurnType.EVENT, EventKind.OVERWRITE, "second"),
                Turn(TurnType.EVENT, EventKind.NOOP, "third"),
                Turn(TurnType.QUERY, query=query),
            ),
        )
        initial = torch.zeros(1, 1, 1, 1, requires_grad=True)
        updater = LoggingUpdater([])
        output = run_episode(
            episode,
            updater=updater,
            reader=LoggingReader([]),
            initial_state=initial,
            detach_between_events=True,
        )
        output.loss.backward()
        self.assertIsNone(initial.grad)

    def test_decode_reencode_is_applied_only_between_updates(self):
        updater = LoggingUpdater([])
        run_episode(
            make_routing_episode(),
            updater=updater,
            reader=LoggingReader([]),
            recurrence_mode="decode_reencode",
        )
        self.assertEqual(updater.decode_calls, 2)


if __name__ == "__main__":
    unittest.main()
