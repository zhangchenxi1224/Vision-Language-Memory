from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.r3_synthetic import NO_ACTIVE_PREFERENCE  # noqa: E402
from vision_memory.data.r4_baseline_lockbox import build_transition32 as build_r4_transition32  # noqa: E402
from vision_memory.data.r5_baseline_lockbox import (  # noqa: E402
    R5_ARTIFACT_NAMES,
    R5_BASELINE_SEED,
    R5_FORMAL_SIZES,
    R5_HISTORY_LENGTHS,
    R5_READ_FORMS,
    R5_TERMINAL_KINDS,
    _prepare_output_dir,
    build_smoke4,
    build_transition32,
    r5_target_for,
    validate_same_entity_pair_contract,
)
from vision_memory.data.schema import EventKind, TurnType  # noqa: E402


class R5BaselineLockboxTest(unittest.TestCase):
    def test_seed_and_inventory_are_fixed(self) -> None:
        self.assertEqual(R5_BASELINE_SEED, 20260723)
        self.assertEqual(
            R5_FORMAL_SIZES,
            {"train": 5000, "dev": 500, "test_id": 1000, "test_ood": 1000},
        )
        self.assertEqual(
            R5_ARTIFACT_NAMES,
            (
                "smoke4.jsonl",
                "transition32.jsonl",
                "formal_train.jsonl",
                "formal_dev.jsonl",
                "formal_test_id.jsonl",
                "formal_test_ood.jsonl",
            ),
        )
        with self.assertRaisesRegex(ValueError, "fixed to seed"):
            build_smoke4(seed=1)

    def test_targets_are_kind_specific_inside_each_entity_scope(self) -> None:
        for replica in range(2):
            self.assertEqual(
                r5_target_for(EventKind.SET, replica),
                r5_target_for(EventKind.NOOP, replica),
            )
            self.assertNotEqual(
                r5_target_for(EventKind.SET, replica),
                r5_target_for(EventKind.OVERWRITE, replica),
            )
            self.assertEqual(r5_target_for(EventKind.CLEAR, replica), NO_ACTIVE_PREFERENCE)

    def test_transition32_exact_factorial_and_delayed_probes(self) -> None:
        episodes = build_transition32()
        self.assertEqual(len(episodes), 32)
        cells = Counter()
        for episode in episodes:
            parts = episode.episode_id.split("-")
            read_form = next(value for value in R5_READ_FORMS if value in parts)
            history_length = next(value for value in R5_HISTORY_LENGTHS if value in parts)
            terminal = next(kind for kind in R5_TERMINAL_KINDS if episode.episode_id.endswith(kind.value))
            replica = int(episode.semantic_group_id.rsplit("r", 1)[1])
            cells[(terminal, read_form, history_length, replica)] += 1
            self.assertIs(episode.turns[-1].type, TurnType.QUERY)
            if read_form == "mixed":
                mixed = next(i for i, turn in enumerate(episode.turns) if turn.type is TurnType.MIXED)
                self.assertLess(mixed, len(episode.turns) - 1)
        self.assertEqual(len(cells), 32)
        self.assertTrue(all(count == 1 for count in cells.values()))

    def test_every_counterfactual_is_same_scope_different_target(self) -> None:
        episodes = {episode.episode_id: episode for episode in build_transition32()}
        report = validate_same_entity_pair_contract(tuple(episodes.values()), expected_delayed_states=32)
        self.assertEqual(report["episode_count"], 32)
        self.assertEqual(report["delayed_state_count"], 32)
        self.assertTrue(report["same_entity_query_scope_valid"])
        for episode in episodes.values():
            donor = episodes[episode.counterfactual_episode_id]
            self.assertEqual(donor.counterfactual_episode_id, episode.episode_id)
            self.assertEqual(donor.pair_id, episode.pair_id)
            self.assertEqual(donor.entity_id, episode.entity_id)
            self.assertEqual(donor.entity_surface, episode.entity_surface)
            self.assertEqual(donor.semantic_group_id, episode.semantic_group_id)
            self.assertEqual(donor.topic, episode.topic)
            self.assertEqual(donor.template_id, episode.template_id)
            own_queries = [turn.query for turn in episode.turns if turn.query is not None]
            donor_queries = [turn.query for turn in donor.turns if turn.query is not None]
            for own, other in zip(own_queries, donor_queries, strict=True):
                self.assertEqual(own.text, other.text)
                self.assertEqual(own.choices, other.choices)
                self.assertNotEqual(own.target, other.target)
                self.assertEqual(own.choices.count(other.target), 1)

    def test_set_noop_pairs_preserve_clean_noop_contract(self) -> None:
        episodes = {episode.episode_id: episode for episode in build_transition32()}
        for episode in episodes.values():
            if episode.distractor_variant is None or episode.distractor_variant.value != "clean":
                continue
            noop = episodes[episode.distractor_episode_id]
            clean_events = tuple((turn.event_kind, turn.event_text) for turn in episode.turns if turn.calls_updater)
            noop_events = tuple((turn.event_kind, turn.event_text) for turn in noop.turns if turn.calls_updater)
            self.assertEqual(clean_events, noop_events[:-1])
            self.assertIs(noop_events[-1][0], EventKind.NOOP)
            self.assertEqual(episode.turns[-1].query.target, noop.turns[-1].query.target)
            self.assertEqual(episode.turns[-1].query.text, noop.turns[-1].query.text)
            self.assertEqual(episode.turns[-1].query.choices, noop.turns[-1].query.choices)

    def test_r4_cross_entity_pairs_are_rejected_by_r5_validator(self) -> None:
        with self.assertRaisesRegex(ValueError, "crossed entity/query scope"):
            validate_same_entity_pair_contract(build_r4_transition32(), expected_delayed_states=32)

    def test_micro_namespace_and_surfaces_are_new(self) -> None:
        raw = json.dumps(
            [episode.to_dict() for episode in build_transition32()],
            ensure_ascii=False,
            sort_keys=True,
        ).casefold()
        self.assertIn("r5-", raw)
        self.assertNotIn("r4 lockbox", raw)
        self.assertNotIn('"r4-', raw)
        for value in ("cobalt", "saffron", "plum", "jade"):
            self.assertIn(value, raw)

    def test_build_is_deterministic(self) -> None:
        self.assertEqual(build_smoke4(), build_smoke4())
        self.assertEqual(build_transition32(), build_transition32())

    def test_output_directory_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "lockbox"
            _prepare_output_dir(root)
            (root / "existing.txt").write_text("do not replace", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refuses to overwrite"):
                _prepare_output_dir(root)
            self.assertEqual((root / "existing.txt").read_text(encoding="utf-8"), "do not replace")


if __name__ == "__main__":
    unittest.main()
