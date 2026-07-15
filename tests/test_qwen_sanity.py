from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.data.qwen_sanity import (  # noqa: E402
    build_summary,
    collect_unique_queries,
    prediction_record,
    require_exact_episode_count,
    validate_model_input_inventory,
    validate_query_inventory,
)
from vision_memory.data import (  # noqa: E402
    DistractorVariant,
    Episode,
    EventKind,
    QuerySpec,
    Turn,
    TurnType,
)
from vision_memory.reader import ChoiceScoreOutput  # noqa: E402


class QwenSanityAuditTest(unittest.TestCase):
    def audit_episodes(self) -> list[Episode]:
        episodes = []
        choices = ("no active preference", "red", "blue", "green")
        for comparison_number in range(4):
            comparison_id = f"comparison-{comparison_number}"
            clean_id = f"dev-clean-{comparison_number}"
            distractor_id = f"dev-distractor-{comparison_number}"
            query = QuerySpec(
                text=f"What is the preference for entity {comparison_number}?",
                choices=choices,
                target_index=comparison_number,
                comparison_id=comparison_id,
            )
            for variant, episode_id, mate_id in (
                (DistractorVariant.CLEAN, clean_id, distractor_id),
                (DistractorVariant.DISTRACTOR, distractor_id, clean_id),
            ):
                episodes.append(
                    Episode(
                        episode_id=episode_id,
                        split="dev",
                        seed=comparison_number,
                        entity_id=f"entity-{comparison_number}",
                        entity_surface=f"entity surface {comparison_number}",
                        template_id=f"template-{comparison_number}",
                        template_family="user brief",
                        pair_id=f"counterfactual-{variant.value}-{comparison_number}",
                        counterfactual_episode_id=f"counterfactual-mate-{episode_id}",
                        distractor_variant=variant,
                        distractor_pair_id=comparison_id,
                        distractor_episode_id=mate_id,
                        topic="color",
                        turns=(
                            Turn(TurnType.EVENT, EventKind.SET, "Remember red."),
                            Turn(TurnType.EVENT, EventKind.NOOP, "Unrelated update."),
                            Turn(TurnType.EVENT, EventKind.OVERWRITE, "Remember blue."),
                            Turn(TurnType.QUERY, query=query),
                        ),
                    )
                )
        return episodes

    def test_clean_distractor_queries_are_deduplicated_and_balanced(self):
        episodes = self.audit_episodes()
        raw_count, unique = collect_unique_queries(episodes)

        self.assertEqual(raw_count, 8)
        self.assertEqual(len(unique), 4)
        self.assertTrue(all(len(item.members) == 2 for item in unique))
        self.assertTrue(
            all(
                {member.distractor_variant for member in item.members}
                == {"clean", "distractor"}
                for item in unique
            )
        )
        validate_query_inventory(
            raw_count,
            unique,
            expected_raw_queries=8,
            expected_comparison_queries=4,
            expected_target_position_count=1,
        )

    def test_formal_inventory_rejects_silent_episode_truncation(self):
        episodes = self.audit_episodes()
        self.assertEqual(require_exact_episode_count(episodes, expected=8), episodes)
        with self.assertRaisesRegex(ValueError, "exactly 7 episodes, found 8"):
            require_exact_episode_count(episodes, expected=7)

    def test_duplicate_comparison_id_with_changed_target_fails_closed(self):
        episodes = self.audit_episodes()
        first_query = next(turn.query for turn in episodes[0].turns if turn.query is not None)
        comparison_id = first_query.comparison_id
        changed = list(episodes)
        for episode_index, episode in enumerate(changed):
            updated_turns = list(episode.turns)
            for turn_index, turn in enumerate(updated_turns):
                if turn.query is None or turn.query.comparison_id != comparison_id:
                    continue
                if episode_index == 0:
                    break
                updated_query = replace(
                    turn.query,
                    target_index=(turn.query.target_index + 1) % len(turn.query.choices),
                )
                updated_turns[turn_index] = replace(turn, query=updated_query)
                changed[episode_index] = replace(episode, turns=tuple(updated_turns))
                with self.assertRaisesRegex(ValueError, "inconsistent payload/target"):
                    collect_unique_queries(changed)
                return
        self.fail("Generated fixture did not contain the expected duplicate comparison_id")

    def test_model_visible_payload_requires_four_balanced_targets(self):
        _, unique = collect_unique_queries(self.audit_episodes())
        for item in unique:
            item.query = replace(item.query, text="What is the shared preference?")
        validate_model_input_inventory(unique, expected_model_inputs=1)

        unbalanced = unique[:-1]
        with self.assertRaisesRegex(ValueError, "all four target positions equally"):
            validate_model_input_inventory(unbalanced, expected_model_inputs=1)

    def test_prediction_record_and_summary_retain_audit_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = self.audit_episodes()
            raw_count, unique = collect_unique_queries(episodes)
            records = []
            for index, item in enumerate(unique[:2]):
                target = item.query.target_index
                blank_index = target if index == 0 else (target + 1) % 4
                records.append(
                    prediction_record(
                        item,
                        blank_result=ChoiceScoreOutput((1.0, 2.0, 3.0, 4.0), blank_index),
                        oracle_result=ChoiceScoreOutput((1.0, 2.0, 3.0, 4.0), target),
                    )
                )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )
            report = build_summary(
                records,
                episodes=len(episodes),
                raw_query_count=sum(record["member_count"] for record in records),
                dataset_sha256="dataset-sha",
                reader_revision="reader-revision",
                predictions_path=predictions,
                oracle_threshold=0.95,
                query_only_ceiling=0.30,
                elapsed_seconds=1.5,
                peak_vram_gib=2.0,
                requested_limit=8,
                expected_raw_queries=4,
                expected_comparison_queries=2,
                expected_target_position_count=None,
                expected_model_inputs=None,
                device="cuda:0",
                dtype="bfloat16",
            )

        first = records[0]
        self.assertEqual(len(first["episode_ids"]), 2)
        self.assertEqual(len(first["blank_choice_mean_nll"]), 4)
        self.assertEqual(len(first["oracle_choice_mean_nll"]), 4)
        self.assertEqual(first["target"], first["target_text"])
        self.assertEqual(len(first["input_sha256"]), 64)
        self.assertNotEqual(first["input_sha256"], first["oracle_input_sha256"])
        self.assertIn("member_turn_types", first)
        self.assertIn("candidate_has_clear_sentinel", first)
        self.assertEqual(report["raw_query_count"], 4)
        self.assertEqual(report["comparison_query_count"], 2)
        self.assertEqual(report["clean_distractor_duplicates_removed"], 2)
        self.assertEqual(report["unique_model_input_count"], 2)
        self.assertEqual(report["requested_episode_limit"], 8)
        self.assertEqual(report["device"], "cuda:0")
        self.assertEqual(report["dtype"], "bfloat16")
        self.assertEqual(report["query_only_blank_accuracy"], 0.5)
        self.assertEqual(report["oracle_text_accuracy"], 1.0)
        self.assertIn("target_position_breakdown", report)
        self.assertIn("clear_breakdown", report)
        self.assertIn("blank_prediction_index_counts", report)


if __name__ == "__main__":
    unittest.main()
