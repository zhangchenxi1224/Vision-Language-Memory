from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import (  # noqa: E402
    DatasetSizes,
    DatasetValidationError,
    DistractorVariant,
    EventKind,
    generate_dataset,
    read_jsonl,
    validate_dataset,
)
from vision_memory.training import select_curriculum_episodes  # noqa: E402


class EpisodeGeneratorTest(unittest.TestCase):
    def test_set_only_profile_is_separate_valid_and_curriculum_selectable(self):
        sizes = DatasetSizes(train=16, dev=16, test_id=16, test_ood=16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = generate_dataset(
                root,
                sizes=sizes,
                seed=41,
                transition_profile="set_only",
            )
            report = validate_dataset(root, expected_sizes=sizes.as_dict())

            self.assertEqual(manifest["transition_profile"], "set_only")
            self.assertTrue(report.valid)
            for split, split_report in report.splits.items():
                self.assertEqual(set(split_report.event_kind_counts), {"set", "noop"})
                self.assertGreater(split_report.mixed_queries, 0, split)
                self.assertEqual(split_report.canonical_target_share_reference, 0.25)
                self.assertFalse(split_report.canonical_balance_enforced)

            train = read_jsonl(root / "train.jsonl")
            self.assertTrue(train)
            for episode in train:
                self.assertGreaterEqual(episode.update_count, 2)
                self.assertTrue(
                    all(
                        turn.event_kind in {EventKind.SET, EventKind.NOOP}
                        for turn in episode.turns
                        if turn.event_kind is not None
                    )
                )
                targets = [turn.query.target for turn in episode.turns if turn.query]
                self.assertTrue(targets)
                self.assertEqual(len(set(targets)), 1)
                for turn in episode.turns:
                    if turn.event_kind is EventKind.SET:
                        self.assertIn(targets[0], turn.event_text)

            selected, audit = select_curriculum_episodes(train, curriculum="set-only")
            self.assertEqual(len(selected), len(train))
            self.assertGreater(audit.selected_count, 0)
            self.assertEqual(audit.excluded_count, 0)

    def test_generation_is_deterministic_balanced_and_valid(self):
        sizes = DatasetSizes(train=16, dev=16, test_id=16, test_ood=16)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            manifest_a = generate_dataset(Path(first), sizes=sizes, seed=17)
            manifest_b = generate_dataset(Path(second), sizes=sizes, seed=17)
            hashes_a = {key: value["sha256"] for key, value in manifest_a["splits"].items()}
            hashes_b = {key: value["sha256"] for key, value in manifest_b["splits"].items()}
            self.assertEqual(hashes_a, hashes_b)

            report = validate_dataset(Path(first), expected_sizes=sizes.as_dict())
            self.assertTrue(report.valid)
            self.assertEqual(report.total_episodes, 64)
            for split, split_report in report.splits.items():
                self.assertGreater(split_report.mixed_queries, 0, split)
                self.assertGreater(split_report.matched_distractor_pairs, 0, split)
                self.assertGreater(split_report.entity_surface_count, 0, split)
                self.assertGreater(split_report.template_family_count, 0, split)
                self.assertGreater(split_report.surface_template_signature_count, 0, split)
                self.assertLessEqual(split_report.max_target_position_deviation, 0.02)
                self.assertGreater(split_report.canonical_payloads, 0, split)
                self.assertEqual(split_report.canonical_target_share_reference, 0.25, split)
                self.assertTrue(split_report.canonical_balance_enforced, split)
                self.assertEqual(
                    split_report.max_canonical_target_share_deviation,
                    0.0,
                    split,
                )
                self.assertEqual(split_report.max_canonical_order_variants, 1, split)
                self.assertEqual(
                    set(split_report.event_kind_counts),
                    {"set", "overwrite", "clear", "noop"},
                )

            episodes_by_split = {split: read_jsonl(Path(first) / f"{split}.jsonl") for split in sizes.as_dict()}
            for left_index, left in enumerate(episodes_by_split):
                left_entities = {episode.entity_surface for episode in episodes_by_split[left]}
                left_families = {episode.template_family for episode in episodes_by_split[left]}
                for right in tuple(episodes_by_split)[left_index + 1 :]:
                    self.assertTrue(
                        left_entities.isdisjoint(episode.entity_surface for episode in episodes_by_split[right])
                    )
                    self.assertTrue(
                        left_families.isdisjoint(episode.template_family for episode in episodes_by_split[right])
                    )

            train = episodes_by_split["train"]
            by_id = {episode.episode_id: episode for episode in train}
            canonical_payloads = defaultdict(list)
            for episode in train:
                for turn in episode.turns:
                    if turn.query is None:
                        continue
                    key = (turn.query.text, tuple(sorted(turn.query.choices)))
                    canonical_payloads[key].append(turn.query)
            null_members = 0
            for (_, candidate_set), queries in canonical_payloads.items():
                ordered_choices = {query.choices for query in queries}
                target_counts = Counter(query.target for query in queries)
                self.assertEqual(len(ordered_choices), 1)
                self.assertEqual(set(target_counts), set(candidate_set))
                self.assertEqual(
                    {count * 4 for count in target_counts.values()},
                    {len(queries)},
                )
                if "no active preference" in candidate_set:
                    null_members += len(queries)
                    self.assertEqual(
                        target_counts["no active preference"] * 4,
                        len(queries),
                    )
            self.assertGreater(null_members, 0)

            clear_counterfactuals = 0
            for episode in train:
                has_clear = any(turn.event_kind is EventKind.CLEAR for turn in episode.turns)
                counterfactual = by_id[episode.counterfactual_episode_id]
                if has_clear:
                    clear_counterfactuals += 1
                    self.assertFalse(any(turn.event_kind is EventKind.CLEAR for turn in counterfactual.turns))
                    self.assertIs(
                        next(turn for turn in reversed(episode.turns) if turn.query is not None).event_kind,
                        EventKind.SET,
                    )
                    self.assertIs(
                        next(
                            turn
                            for turn in reversed(counterfactual.turns)
                            if turn.query is not None
                        ).event_kind,
                        EventKind.OVERWRITE,
                    )
                    pair_queries = [
                        turn.query
                        for member in (episode, counterfactual)
                        for turn in member.turns
                        if turn.query is not None
                    ]
                    self.assertEqual(
                        len({(query.text, query.choices) for query in pair_queries}),
                        1,
                    )
                    self.assertEqual(
                        {query.target for query in pair_queries},
                        set(pair_queries[0].choices),
                    )
                if episode.distractor_variant is DistractorVariant.UNPAIRED:
                    continue
                mate = by_id[episode.distractor_episode_id]
                self.assertEqual(episode.query_comparison_ids, mate.query_comparison_ids)
                self.assertEqual(
                    [turn.query.to_dict() for turn in episode.turns if turn.query],
                    [turn.query.to_dict() for turn in mate.turns if turn.query],
                )
                if episode.distractor_variant is DistractorVariant.CLEAN:
                    self.assertFalse(episode.distractor_turn_indices)
                else:
                    self.assertTrue(episode.distractor_turn_indices)
            self.assertGreater(clear_counterfactuals, 0)

            self.assertEqual(manifest_a["schema_version"], 2)
            self.assertEqual(set(manifest_a["surface_partitions"]), set(sizes.as_dict()))

    def test_full_profile_covariates_are_not_direct_pattern_aliases(self):
        sizes = DatasetSizes(train=64, dev=16, test_id=16, test_ood=16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_dataset(root, sizes=sizes, seed=53)
            train = read_jsonl(root / "train.jsonl")
            representatives = {}
            for episode in train:
                representatives.setdefault(episode.entity_id, episode)

            covariates = {
                pattern: {
                    "template": set(),
                    "entity": set(),
                    "topic": set(),
                }
                for pattern in range(4)
            }
            for episode in representatives.values():
                pattern = int(episode.template_id.split("-pattern-")[1].split("-")[0])
                covariates[pattern]["template"].add(episode.template_family)
                covariates[pattern]["entity"].add(episode.entity_surface.split()[0])
                covariates[pattern]["topic"].add(episode.topic)

            for pattern, values in covariates.items():
                self.assertGreater(len(values["template"]), 1, pattern)
                self.assertGreater(len(values["entity"]), 1, pattern)
                self.assertGreater(len(values["topic"]), 1, pattern)

    def test_validator_rejects_serialized_hidden_ledger(self):
        sizes = DatasetSizes(train=8, dev=8, test_id=8, test_ood=16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_dataset(root, sizes=sizes, seed=3)
            path = root / "train.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["hidden_ledger"] = {"current": "red"}
            lines[0] = json.dumps(first)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetValidationError, "Hidden ledger"):
                validate_dataset(root, verify_manifest_hashes=False)

    def test_validator_rejects_full_profile_choice_order_leakage(self):
        sizes = DatasetSizes(train=16, dev=16, test_id=16, test_ood=16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_dataset(root, sizes=sizes, seed=59)
            path = root / "train.jsonl"
            episodes = [json.loads(line) for line in path.read_text().splitlines()]
            comparison_id = episodes[0]["turns"][1]["query"]["comparison_id"]
            for episode in episodes:
                for turn in episode["turns"]:
                    query = turn.get("query")
                    if query is None or query["comparison_id"] != comparison_id:
                        continue
                    target = query["choices"][query["target_index"]]
                    query["choices"][0], query["choices"][1] = (
                        query["choices"][1],
                        query["choices"][0],
                    )
                    query["target_index"] = query["choices"].index(target)
            path.write_text(
                "".join(json.dumps(episode) + "\n" for episode in episodes),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetValidationError, "ordered choices tuple"):
                validate_dataset(root, verify_manifest_hashes=False)

    def test_validator_rejects_full_profile_conditional_target_imbalance(self):
        sizes = DatasetSizes(train=16, dev=16, test_id=16, test_ood=16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_dataset(root, sizes=sizes, seed=61)
            path = root / "train.jsonl"
            episodes = [json.loads(line) for line in path.read_text().splitlines()]
            comparison_id = episodes[0]["turns"][1]["query"]["comparison_id"]
            for episode in episodes:
                for turn in episode["turns"]:
                    query = turn.get("query")
                    if query is None or query["comparison_id"] != comparison_id:
                        continue
                    query["target_index"] = (query["target_index"] + 1) % 4
            path.write_text(
                "".join(json.dumps(episode) + "\n" for episode in episodes),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetValidationError, "target shares"):
                validate_dataset(root, verify_manifest_hashes=False)

    def test_validator_rejects_model_visible_entity_leakage_despite_disjoint_ids(self):
        sizes = DatasetSizes(train=16, dev=16, test_id=16, test_ood=16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_dataset(root, sizes=sizes, seed=29)
            train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
            dev_path = root / "dev.jsonl"
            dev = [json.loads(line) for line in dev_path.read_text().splitlines()]
            leaked_surface = train[0]["entity_surface"]
            affected_entity_id = dev[0]["entity_id"]
            for episode in dev:
                if episode["entity_id"] != affected_entity_id:
                    continue
                previous = episode["entity_surface"]
                episode["entity_surface"] = leaked_surface
                for turn in episode["turns"]:
                    if "event_text" in turn:
                        turn["event_text"] = turn["event_text"].replace(previous, leaked_surface)
                    if "query" in turn:
                        turn["query"]["text"] = turn["query"]["text"].replace(
                            previous,
                            leaked_surface,
                        )
            dev_path.write_text(
                "".join(json.dumps(episode) + "\n" for episode in dev),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetValidationError, "entity_surface leakage"):
                validate_dataset(root, verify_manifest_hashes=False)

    def test_validator_rejects_visible_template_family_leakage(self):
        sizes = DatasetSizes(train=16, dev=16, test_id=16, test_ood=16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_dataset(root, sizes=sizes, seed=31)
            train = [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()]
            dev_path = root / "dev.jsonl"
            dev = [json.loads(line) for line in dev_path.read_text().splitlines()]
            leaked_family = train[0]["template_family"]
            affected_template_id = dev[0]["template_id"]
            for episode in dev:
                if episode["template_id"] != affected_template_id:
                    continue
                previous = episode["template_family"]
                episode["template_family"] = leaked_family
                for turn in episode["turns"]:
                    if "event_text" in turn:
                        turn["event_text"] = turn["event_text"].replace(
                            previous.title(),
                            leaked_family.title(),
                        )
                    if "query" in turn:
                        turn["query"]["text"] = turn["query"]["text"].replace(
                            previous.title(),
                            leaked_family.title(),
                        )
            dev_path.write_text(
                "".join(json.dumps(episode) + "\n" for episode in dev),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DatasetValidationError, "template_family leakage"):
                validate_dataset(root, verify_manifest_hashes=False)


if __name__ == "__main__":
    unittest.main()
