from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import (  # noqa: E402
    DistractorVariant,
    R3SyntheticSizes,
    TurnType,
    generate_r3_synthetic,
    read_jsonl,
    validate_r3_synthetic,
)


SMALL_SIZES = R3SyntheticSizes(train=64, dev=64, test_id=64, test_ood=64)


class R3SyntheticTest(unittest.TestCase):
    def test_generation_is_deterministic_balanced_and_split_safe(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root, second_root = Path(first), Path(second)
            manifest_a = generate_r3_synthetic(
                first_root,
                sizes=SMALL_SIZES,
                seed=2026,
                profile="custom",
            )
            manifest_b = generate_r3_synthetic(
                second_root,
                sizes=SMALL_SIZES,
                seed=2026,
                profile="custom",
            )
            self.assertEqual(manifest_a["artifacts"], manifest_b["artifacts"])
            report = validate_r3_synthetic(first_root, expected_sizes=SMALL_SIZES.as_dict())
            self.assertTrue(report.valid)
            self.assertEqual(report.total_episodes, 256)

            groups: dict[str, set[str]] = {}
            entities: dict[str, set[str]] = {}
            templates: dict[str, set[str]] = {}
            for split in SMALL_SIZES.as_dict():
                episodes = read_jsonl(first_root / f"{split}.jsonl")
                groups[split] = {episode.semantic_group_id for episode in episodes}  # type: ignore[misc]
                entities[split] = {episode.entity_id for episode in episodes}
                templates[split] = {episode.template_family for episode in episodes}  # type: ignore[misc]
                stats = report.split_statistics[split]
                self.assertEqual(
                    stats["target_position_counts"],
                    {
                        "0": stats["query_count"] // 4,
                        "1": stats["query_count"] // 4,
                        "2": stats["query_count"] // 4,
                        "3": stats["query_count"] // 4,
                    },
                )
                self.assertEqual(set(stats["event_kind_counts"]), {"set", "overwrite", "clear", "noop"})
            split_names = tuple(SMALL_SIZES.as_dict())
            for left_index, left in enumerate(split_names):
                for right in split_names[left_index + 1 :]:
                    self.assertTrue(groups[left].isdisjoint(groups[right]))
                    self.assertTrue(entities[left].isdisjoint(entities[right]))
                    self.assertTrue(templates[left].isdisjoint(templates[right]))

    def test_related_members_stay_in_group_and_mixed_has_delayed_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_r3_synthetic(root, sizes=SMALL_SIZES, profile="custom")
            saw_mixed = False
            saw_noop_pair = False
            for split in SMALL_SIZES.as_dict():
                episodes = read_jsonl(root / f"{split}.jsonl")
                by_id = {episode.episode_id: episode for episode in episodes}
                for episode in episodes:
                    counterfactual = by_id[episode.counterfactual_episode_id]
                    self.assertEqual(episode.semantic_group_id, counterfactual.semantic_group_id)
                    self.assertNotEqual(
                        next(turn.query.target for turn in reversed(episode.turns) if turn.query),
                        next(turn.query.target for turn in reversed(counterfactual.turns) if turn.query),
                    )
                    if episode.distractor_variant in {DistractorVariant.CLEAN, DistractorVariant.DISTRACTOR}:
                        saw_noop_pair = True
                        distractor = by_id[episode.distractor_episode_id]
                        self.assertEqual(episode.semantic_group_id, distractor.semantic_group_id)
                    for index, turn in enumerate(episode.turns):
                        if turn.type is not TurnType.MIXED:
                            continue
                        saw_mixed = True
                        self.assertLess(index + 1, len(episode.turns))
                        delayed = episode.turns[index + 1]
                        self.assertIs(delayed.type, TurnType.QUERY)
                        self.assertEqual(turn.query.target, delayed.query.target)
            self.assertTrue(saw_mixed)
            self.assertTrue(saw_noop_pair)

    def test_analysis_ledger_is_sidecar_only_and_train_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = generate_r3_synthetic(root, sizes=SMALL_SIZES, profile="custom")
            for split in SMALL_SIZES.as_dict():
                raw = (root / f"{split}.jsonl").read_text(encoding="utf-8")
                self.assertNotIn("hidden_ledger", raw)
                self.assertNotIn("before_state", raw)
                self.assertNotIn("after_state", raw)
                self.assertNotIn("teacher", raw)
                for line in raw.splitlines():
                    value = json.loads(line)
                    self.assertIn("semantic_group_id", value)
                    self.assertNotIn("ledger", value)

            sidecar_path = root / "train_analysis_teacher_sidecar.jsonl"
            sidecar = [json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(sidecar)
            self.assertTrue(all(row["split"] == "train" for row in sidecar))
            self.assertTrue(all("before_state" in row and "after_state" in row for row in sidecar))
            self.assertFalse(any(name.startswith("dev_") and "sidecar" in name for name in manifest["artifacts"]))
            self.assertEqual(
                manifest["artifacts"]["train_analysis_teacher_sidecar.jsonl"]["model_visible"],
                False,
            )

    def test_ood_strata_and_default_profiles_are_locked(self) -> None:
        self.assertEqual(R3SyntheticSizes.pilot().train, 1_000)
        self.assertEqual(
            R3SyntheticSizes.formal().as_dict(),
            {
                "train": 5_000,
                "dev": 500,
                "test_id": 1_000,
                "test_ood": 1_000,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "pilot profile has fixed"):
                generate_r3_synthetic(Path(temporary), sizes=SMALL_SIZES, profile="pilot")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_r3_synthetic(root, sizes=SMALL_SIZES, profile="custom")
            counts = Counter(episode.ood_group for episode in read_jsonl(root / "test_ood.jsonl"))
            self.assertEqual(
                counts,
                Counter(
                    {
                        "heldout_entity": 16,
                        "heldout_topic": 16,
                        "heldout_paraphrase": 16,
                        "heldout_length": 16,
                    }
                ),
            )


if __name__ == "__main__":
    unittest.main()
