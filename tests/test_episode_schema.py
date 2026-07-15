from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.schema import (  # noqa: E402
    DistractorVariant,
    EventKind,
    Episode,
    QuerySpec,
    Turn,
    TurnType,
)


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

    def test_pairing_metadata_is_structural_and_answer_agnostic(self):
        episode = Episode(
            **{
                **make_episode().__dict__,
                "entity_surface": "mug 4af2",
                "template_family": "memory memo",
                "distractor_variant": DistractorVariant.CLEAN,
                "distractor_pair_id": "stream-0",
                "distractor_episode_id": "train-2",
            }
        )
        value = episode.to_dict()
        self.assertEqual(value["distractor_variant"], "clean")
        self.assertEqual(value["distractor_pair_id"], "stream-0")
        self.assertNotIn("target", value)
        self.assertEqual(Episode.from_dict(value), episode)

    def test_paired_variant_requires_reciprocal_link_fields(self):
        with self.assertRaisesRegex(ValueError, "require both distractor link fields"):
            Episode(
                **{
                    **make_episode().__dict__,
                    "distractor_variant": DistractorVariant.DISTRACTOR,
                }
            )


if __name__ == "__main__":
    unittest.main()
