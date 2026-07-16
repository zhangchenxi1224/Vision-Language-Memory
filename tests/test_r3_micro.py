from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from vision_memory.data import (
    CYCLIC4,
    REVERSE_CYCLIC4,
    build_set8,
    build_transition16,
    episode_choice_view,
    presentation_permutation,
    read_jsonl,
    write_r3_micro_suite,
)


class R3MicroTest(unittest.TestCase):
    def test_permutation_families_are_valid_and_disjoint(self) -> None:
        self.assertEqual(len(set(CYCLIC4)), 4)
        self.assertEqual(len(set(REVERSE_CYCLIC4)), 4)
        self.assertFalse(set(CYCLIC4) & set(REVERSE_CYCLIC4))
        for family in (CYCLIC4, REVERSE_CYCLIC4):
            for permutation in family:
                self.assertEqual(set(permutation), {0, 1, 2, 3})

    def test_choice_views_preserve_target_and_model_state_inputs(self) -> None:
        episode = build_set8().train_episodes[0]
        original_target = episode.turns[-1].query.target
        for permutation in REVERSE_CYCLIC4:
            view = episode_choice_view(episode, permutation)
            self.assertEqual(view.turns[0], episode.turns[0])
            self.assertEqual(view.turns[-1].query.text, episode.turns[-1].query.text)
            self.assertEqual(view.turns[-1].query.target, original_target)

    def test_presentation_schedule_balances_positions(self) -> None:
        episode = build_set8().train_episodes[0]
        query = episode.turns[-1].query
        positions = Counter()
        for presentation in range(64):
            view = episode_choice_view(
                episode,
                presentation_permutation(presentation, query.comparison_id),
            )
            positions[view.turns[-1].query.target_index] += 1
        self.assertEqual(positions, Counter({0: 16, 1: 16, 2: 16, 3: 16}))

    def test_set8_contract(self) -> None:
        suite = build_set8()
        self.assertEqual(len(suite.train_episodes), 8)
        self.assertEqual(len(suite.gate_episodes), 8)
        self.assertEqual(len(suite.teacher_sidecar), 8)
        self.assertTrue(all(len(episode.turns) == 2 for episode in suite.train_episodes))
        self.assertEqual(suite.manifest["heldout_view_count"], 32)

    def test_transition16_contract_and_mixed_delayed_probe(self) -> None:
        suite = build_transition16()
        self.assertEqual(len(suite.train_episodes), 16)
        self.assertEqual(len(suite.gate_episodes), 16)
        mixed = [episode for episode in suite.train_episodes if "-mixed-" in episode.episode_id]
        self.assertEqual(len(mixed), 8)
        for episode in mixed:
            self.assertTrue(any(turn.type.value == "mixed" for turn in episode.turns))
            self.assertEqual(episode.turns[-1].type.value, "query")
        terminal_counts = Counter(episode.episode_id.split("-")[2] for episode in suite.train_episodes)
        self.assertEqual(terminal_counts, Counter({"set": 4, "overwrite": 4, "clear": 4, "noop": 4}))
        by_id = {episode.episode_id: episode for episode in suite.train_episodes}
        for episode in suite.train_episodes:
            if "-set-" not in episode.episode_id:
                continue
            donor = by_id[episode.distractor_episode_id]
            self.assertIn("-noop-", donor.episode_id)
            self.assertEqual(episode.distractor_pair_id, donor.distractor_pair_id)
            self.assertEqual(episode.turns[-1].query, donor.turns[-1].query)

    def test_sidecar_is_train_only_and_no_ledger_enters_episode_json(self) -> None:
        suite = build_transition16()
        for episode in suite.train_episodes:
            serialized = json.dumps(episode.to_dict(), sort_keys=True)
            self.assertNotIn("ledger", serialized.casefold())
        self.assertTrue(all(item["split"] == "train" for item in suite.teacher_sidecar))
        noop_records = [item for item in suite.teacher_sidecar if item["event_kind"] == "noop"]
        self.assertTrue(noop_records)
        self.assertTrue(all(item["before_state"] == item["after_state"] for item in noop_records))

    def test_written_artifacts_roundtrip_and_lock_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite = build_set8()
            manifest = write_r3_micro_suite(root, suite)
            self.assertEqual(read_jsonl(root / "set8_train.jsonl"), list(suite.train_episodes))
            self.assertEqual(manifest["artifacts"]["set8_train.jsonl"]["count"], 8)
            self.assertEqual(len(manifest["artifacts"]["set8_train.jsonl"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
