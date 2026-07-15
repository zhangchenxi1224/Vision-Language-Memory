from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.schema import EventKind, Episode, QuerySpec, Turn, TurnType  # noqa: E402


def make_episode() -> Episode:
    query = QuerySpec("Which option?", ("a", "b", "c", "d"), 1)
    return Episode(
        episode_id="train-0",
        split="train",
        seed=0,
        entity_id="entity-0",
        template_id="template-0",
        pair_id="pair-0",
        counterfactual_episode_id="train-1",
        topic="color",
        turns=(
            Turn(TurnType.EVENT, EventKind.SET, "Remember b."),
            Turn(TurnType.QUERY, query=query),
            Turn(TurnType.MIXED, EventKind.NOOP, "Unrelated detail.", query),
            Turn(TurnType.EVENT, EventKind.CLEAR, "Clear it."),
        ),
    )


class EpisodeSchemaTest(unittest.TestCase):
    def test_round_trip_is_lossless(self):
        episode = make_episode()
        self.assertEqual(Episode.from_dict(episode.to_dict()), episode)

    def test_pure_query_forbids_event_fields(self):
        with self.assertRaisesRegex(ValueError, "query turn"):
            Turn(
                TurnType.QUERY,
                EventKind.NOOP,
                "This must not reach the updater.",
                QuerySpec("q", ("a", "b", "c", "d"), 0),
            )

    def test_hidden_ledger_is_rejected_recursively(self):
        value = make_episode().to_dict()
        value["turns"][0]["hidden_ledger"] = {"answer": "b"}
        with self.assertRaisesRegex(ValueError, "Hidden ledger"):
            Episode.from_dict(value)

    def test_unknown_fields_are_rejected(self):
        value = make_episode().to_dict()
        value["answer_hint"] = "b"
        with self.assertRaisesRegex(ValueError, "Unknown episode fields"):
            Episode.from_dict(value)


if __name__ == "__main__":
    unittest.main()
