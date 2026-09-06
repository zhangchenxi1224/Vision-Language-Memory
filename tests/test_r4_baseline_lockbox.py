from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.r3_synthetic import R3SyntheticSizes  # noqa: E402
from vision_memory.data.r4_baseline_lockbox import (  # noqa: E402
    R4_ARTIFACT_NAMES,
    R4_BASELINE_SEED,
    R4_FORMAL_SIZES,
    R4_HISTORY_LENGTHS,
    R4_READ_FORMS,
    R4_TERMINAL_KINDS,
    _prepare_output_dir,
    build_r4_formal_episodes,
    build_smoke4,
    build_transition32,
)
from vision_memory.data.schema import EventKind, TurnType  # noqa: E402


SMALL_FORMAL_SIZES = R3SyntheticSizes(train=8, dev=8, test_id=8, test_ood=8)


class R4BaselineLockboxTest(unittest.TestCase):
    def test_seed_and_formal_inventory_are_fixed(self) -> None:
        self.assertEqual(R4_BASELINE_SEED, 20260722)
        self.assertEqual(
            R4_FORMAL_SIZES.as_dict(),
            {"train": 5000, "dev": 500, "test_id": 1000, "test_ood": 1000},
        )
        self.assertEqual(
            R4_ARTIFACT_NAMES,
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

    def test_smoke4_has_one_of_each_terminal_kind(self) -> None:
        episodes = build_smoke4()
        self.assertEqual(len(episodes), 4)
        terminal = Counter(
            next(turn.event_kind for turn in reversed(episode.turns) if turn.calls_updater)
            for episode in episodes
        )
        self.assertEqual(terminal, Counter({kind: 1 for kind in R4_TERMINAL_KINDS}))
        self.assertTrue(all(episode.turns[-1].type is TurnType.QUERY for episode in episodes))
        self.assertFalse(any("r3-" in json.dumps(episode.to_dict()).casefold() for episode in episodes))

    def test_transition32_exact_factorial_and_mixed_delayed_probe(self) -> None:
        episodes = build_transition32()
        self.assertEqual(len(episodes), 32)
        cells = Counter()
        for episode in episodes:
            parts = episode.episode_id.split("-")
            read_form = next(value for value in R4_READ_FORMS if value in parts)
            history_length = next(value for value in R4_HISTORY_LENGTHS if value in parts)
            terminal = next(kind for kind in R4_TERMINAL_KINDS if episode.episode_id.endswith(kind.value))
            replica = int(episode.semantic_group_id.rsplit("r", 1)[1])
            cells[(terminal, read_form, history_length, replica)] += 1
            self.assertIs(episode.turns[-1].type, TurnType.QUERY)
            if read_form == "mixed":
                self.assertTrue(any(turn.type is TurnType.MIXED for turn in episode.turns))
                mixed_index = next(i for i, turn in enumerate(episode.turns) if turn.type is TurnType.MIXED)
                self.assertLess(mixed_index, len(episode.turns) - 1)
            update_count = episode.update_count
            if history_length == "short":
                self.assertLessEqual(update_count, 2)
            else:
                self.assertGreaterEqual(update_count, 3)
        self.assertEqual(len(cells), 32)
        self.assertTrue(all(count == 1 for count in cells.values()))

    def test_set_noop_pairs_have_same_final_target_and_noop_only_appends(self) -> None:
        episodes = {episode.episode_id: episode for episode in build_transition32()}
        for episode in episodes.values():
            if episode.distractor_variant is None or episode.distractor_variant.value != "clean":
                continue
            noop = episodes[episode.distractor_episode_id]
            clean_events = tuple(
                (turn.event_kind, turn.event_text) for turn in episode.turns if turn.calls_updater
            )
            noop_events = tuple((turn.event_kind, turn.event_text) for turn in noop.turns if turn.calls_updater)
            self.assertEqual(clean_events, noop_events[:-1])
            self.assertEqual(noop_events[-1][0], EventKind.NOOP)
            self.assertEqual(episode.turns[-1].query.target, noop.turns[-1].query.target)
            self.assertEqual(episode.turns[-1].query.text, noop.turns[-1].query.text)

    def test_every_transition_state_has_a_different_target_counterfactual_donor(self) -> None:
        episodes = {episode.episode_id: episode for episode in build_transition32()}
        self.assertEqual(len(episodes), 32)
        for episode in episodes.values():
            donor = episodes[episode.counterfactual_episode_id]
            self.assertEqual(donor.counterfactual_episode_id, episode.episode_id)
            self.assertEqual(donor.pair_id, episode.pair_id)
            self.assertNotEqual(episode.turns[-1].query.target, donor.turns[-1].query.target)

    def test_formal_remap_is_deterministic_fresh_and_split_safe(self) -> None:
        first, source_sha_a = build_r4_formal_episodes(sizes=SMALL_FORMAL_SIZES)
        second, source_sha_b = build_r4_formal_episodes(sizes=SMALL_FORMAL_SIZES)
        self.assertEqual(source_sha_a, source_sha_b)
        self.assertEqual(first, second)
        groups: dict[str, set[str]] = {}
        for split, expected in SMALL_FORMAL_SIZES.as_dict().items():
            self.assertEqual(len(first[split]), expected)
            groups[split] = {episode.semantic_group_id for episode in first[split]}  # type: ignore[misc]
            for episode in first[split]:
                raw = json.dumps(episode.to_dict(), ensure_ascii=False, sort_keys=True).casefold()
                self.assertNotIn("r3-", raw)
                self.assertNotIn("ledger", raw)
                self.assertNotIn("teacher", raw)
                self.assertEqual(episode.seed, R4_BASELINE_SEED)
                self.assertTrue(episode.episode_id.startswith("r4-"))
                self.assertIn(" r4 ", episode.entity_surface)
        names = tuple(groups)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                self.assertTrue(groups[left].isdisjoint(groups[right]))

    def test_output_directory_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "lockbox"
            _prepare_output_dir(root)
            self.assertTrue(root.is_dir())
            (root / "existing.txt").write_text("do not replace", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "refuses to overwrite"):
                _prepare_output_dir(root)
            self.assertEqual((root / "existing.txt").read_text(encoding="utf-8"), "do not replace")


if __name__ == "__main__":
    unittest.main()
