from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scripts.eval.dreamlite_mcq import (  # noqa: E402
    QueryState,
    _target_index_in_choices,
    collect_synthetic,
    intervention_states,
    semantic_group_report,
)
from scripts.eval import dreamlite_mcq  # noqa: E402
from scripts.eval.qwen_text_baselines import synthetic_queries  # noqa: E402
from scripts.eval.score_synthetic import DEFAULT_MAIN_CONTRASTS  # noqa: E402
from vision_memory.data import DatasetSizes, generate_dataset, read_jsonl, write_jsonl  # noqa: E402
from vision_memory.eval import (  # noqa: E402
    compute_synthetic_metrics,
    filter_preregistered_records,
    seeded_stratified_accuracy,
)


def row(episode, prediction, *, condition="standard", distractor_variant=None, **extra):
    return {
        "episode_id": episode,
        "query_id": "q",
        "method": "main",
        "seed": 0,
        "prediction_index": prediction,
        "target_index": 0,
        "topic": "topic-a",
        "subtype": "overwrite",
        "split": "test_id",
        "condition": condition,
        "counterfactual_pair_id": f"semantic:{episode}",
        "distractor_variant": distractor_variant,
        "noop_policy": "keep",
        **extra,
    }


class SyntheticMetricTest(unittest.TestCase):
    def test_strict_evaluation_configures_runtime_before_first_cuda_probe(self):
        argv = [
            "dreamlite_mcq.py",
            "--episodes",
            "episodes.jsonl",
            "--format",
            "synthetic",
            "--dreamlite",
            "dreamlite",
            "--reader",
            "reader",
            "--output",
            "predictions.jsonl",
            "--method",
            "test",
            "--seed",
            "7",
            "--strict-determinism",
        ]
        sentinel = RuntimeError("strict-runtime-configured")
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(
                dreamlite_mcq,
                "configure_strict_cuda_determinism",
                side_effect=sentinel,
            ) as configure,
            mock.patch.object(dreamlite_mcq.torch.cuda, "is_available") as cuda_available,
            self.assertRaisesRegex(RuntimeError, "strict-runtime-configured"),
        ):
            dreamlite_mcq.main()
        configure.assert_called_once_with(seed=7)
        cuda_available.assert_not_called()

    def test_four_main_comparisons_are_fixed_and_distinct(self):
        self.assertEqual(len(DEFAULT_MAIN_CONTRASTS), 4)
        self.assertEqual(len(set(DEFAULT_MAIN_CONTRASTS)), 4)

    def test_intervention_and_counterfactual_metrics(self):
        rows = [
            row(
                "clean-a",
                0,
                distractor_variant="clean",
                distractor_pair_id="stream-a",
                query_comparison_id="stream-a:q0",
            ),
            row(
                "distractor-a",
                1,
                distractor_variant="distractor",
                distractor_pair_id="stream-a",
                query_comparison_id="stream-a:q0",
            ),
            row("b", 0),
            row("b", 1, condition="reset"),
            row("b", 1, condition="shuffle"),
            row("b", 1, condition="state_swap", donor_target_index=1),
        ]
        result = compute_synthetic_metrics(rows)
        self.assertAlmostEqual(result["macro_mcq_accuracy"], 2 / 6)
        self.assertEqual(result["macro_mcq_cells"], 1)
        self.assertEqual(result["matched_distractor_damage"]["accuracy_damage"], 1.0)
        self.assertEqual(result["reset"]["accuracy_drop"], 1.0)
        self.assertEqual(result["shuffle"]["accuracy_drop"], 1.0)
        self.assertEqual(result["state_swap_donor_answer"]["rate"], 1.0)

    def test_noop_filter_effect_is_not_matched_distractor_damage(self):
        keep = row(
            "distractor-a",
            1,
            distractor_variant="distractor",
            distractor_pair_id="stream-a",
            query_comparison_id="stream-a:q0",
            noop_policy="keep",
            noop_intervention_pair_id="distractor-a:q0",
        )
        skip = {**keep, "prediction_index": 0, "noop_policy": "skip"}
        result = compute_synthetic_metrics([keep, skip])
        self.assertIsNone(result["matched_distractor_damage"])
        self.assertEqual(result["noop_filter_effect"]["n_pairs"], 1)
        self.assertEqual(result["noop_filter_effect"]["skip_minus_keep_accuracy"], 1.0)
        unpaired = [
            {
                key: value
                for key, value in item.items()
                if key != "noop_intervention_pair_id"
            }
            for item in (keep, skip)
        ]
        self.assertIsNone(compute_synthetic_metrics(unpaired)["noop_filter_effect"])

        semantic_only = [
            row("semantic-a", 0, counterfactual_pair_id="semantic-pair"),
            row("semantic-b", 1, counterfactual_pair_id="semantic-pair"),
        ]
        self.assertIsNone(compute_synthetic_metrics(semantic_only)["matched_distractor_damage"])

    def test_prediction_collectors_preserve_dataset_pairs_and_separate_noop_policy(self):
        class Updater:
            def __call__(self, state, _event, _episode_id, _turn_id):
                return state + 1

        class Model:
            updater = Updater()

            @staticmethod
            def reset_state():
                return torch.zeros(1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_dataset(
                root,
                sizes=DatasetSizes(train=4, dev=4, test_id=4, test_ood=16),
                seed=19,
            )
            path = root / "test_id.jsonl"
            episode = read_jsonl(path)[0]
            text_item = next(synthetic_queries(path, limit=1))
            keep_item = collect_synthetic(
                Model(), path, limit=1, recurrence_mode="direct_latent", skip_noop=False
            )[0]
            skip_item = collect_synthetic(
                Model(), path, limit=1, recurrence_mode="direct_latent", skip_noop=True
            )[0]

        for metadata in (text_item["metadata"], keep_item.metadata, skip_item.metadata):
            self.assertEqual(metadata["counterfactual_pair_id"], episode.pair_id)
            self.assertEqual(metadata["distractor_pair_id"], episode.distractor_pair_id)
            self.assertEqual(metadata["distractor_variant"], episode.distractor_variant.value)
            self.assertEqual(
                metadata["query_comparison_id"], episode.query_comparison_ids[0]
            )
            self.assertNotIn("counterfactual_variant", metadata)
        self.assertEqual(keep_item.metadata["noop_policy"], "keep")
        self.assertEqual(skip_item.metadata["noop_policy"], "skip")
        self.assertEqual(
            keep_item.metadata["counterfactual_pair_id"],
            skip_item.metadata["counterfactual_pair_id"],
        )

    def test_dreamlite_predictions_and_report_preserve_semantic_group_provenance(self):
        class Updater:
            def __call__(self, state, _event, _episode_id, _turn_id):
                return state + 1

        class Model:
            updater = Updater()

            @staticmethod
            def reset_state():
                return torch.zeros(1)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_dataset(
                root,
                sizes=DatasetSizes(train=4, dev=4, test_id=4, test_ood=16),
                seed=23,
            )
            path = root / "test_id.jsonl"
            episode = replace(read_jsonl(path)[0], semantic_group_id="r3-semantic-000001")
            write_jsonl(path, [episode])
            item = collect_synthetic(
                Model(), path, limit=None, recurrence_mode="direct_latent", skip_noop=False
            )[0]

        self.assertEqual(item.metadata["semantic_group_id"], "r3-semantic-000001")
        report = semantic_group_report([item])
        expected_sha = hashlib.sha256(b'["r3-semantic-000001"]').hexdigest()
        self.assertEqual(report["semantic_group_count"], 1)
        self.assertEqual(report["semantic_group_ids_sha256"], expected_sha)
        self.assertTrue(report["semantic_group_metadata_complete"])
        self.assertEqual(report["query_states_missing_semantic_group_id"], 0)

        missing = QueryState(
            metadata={},
            query="q",
            choices=("a", "b", "c", "d"),
            target_index=0,
            state=torch.zeros(1),
        )
        incomplete = semantic_group_report([item, missing])
        self.assertFalse(incomplete["semantic_group_metadata_complete"])
        self.assertEqual(incomplete["query_states_missing_semantic_group_id"], 1)

    def test_state_swap_maps_donor_semantics_into_recipient_choice_order(self):
        first = QueryState(
            metadata={
                "episode_id": "a",
                "counterfactual_episode_id": "b",
                "query_ordinal": 0,
                "distractor_variant": "distractor",
                "noop_policy": "keep",
            },
            query="q",
            choices=("red", "green", "blue", "yellow"),
            target_index=0,
            state=torch.zeros(1),
        )
        donor = QueryState(
            metadata={
                "episode_id": "b",
                "counterfactual_episode_id": "a",
                "query_ordinal": 0,
                "distractor_variant": "distractor",
                "noop_policy": "keep",
            },
            query="q",
            choices=("blue", "yellow", "red", "green"),
            target_index=0,
            state=torch.ones(1),
        )
        swapped = intervention_states(
            [first, donor],
            condition="state_swap",
            initial_state=torch.full((1,), 2.0),
            seed=0,
        )
        self.assertEqual(swapped[0].donor_target_index, 2)
        self.assertEqual(swapped[0].donor_episode_id, "b")
        self.assertTrue(torch.equal(swapped[0].state, donor.state))

        clean_first = QueryState(
            metadata={**first.metadata, "episode_id": "c", "distractor_variant": "clean"},
            query="q",
            choices=first.choices,
            target_index=first.target_index,
            state=torch.full((1,), 3.0),
        )
        clean_second = QueryState(
            metadata={**donor.metadata, "episode_id": "d", "distractor_variant": "clean"},
            query="q",
            choices=donor.choices,
            target_index=donor.target_index,
            state=torch.full((1,), 4.0),
        )
        shuffled = intervention_states(
            [first, donor, clean_first, clean_second],
            condition="shuffle",
            initial_state=torch.full((1,), 2.0),
            seed=0,
        )
        for source, shuffled_state in zip(
            [first, donor, clean_first, clean_second],
            shuffled,
            strict=True,
        ):
            self.assertFalse(torch.equal(source.state, shuffled_state.state))
        self.assertIn(float(shuffled[0].state.item()), {1.0, 4.0})
        self.assertIn(float(shuffled[2].state.item()), {1.0, 4.0})

    def test_stale_target_uses_semantic_text_not_previous_position(self):
        choices = ("new", "other", "old", "none")
        self.assertEqual(_target_index_in_choices("old", choices), 2)
        self.assertIsNone(_target_index_in_choices("missing", choices))

    def test_headline_filter_and_strata_never_pool_interventions(self):
        records = [
            {
                **row("a", 0),
                "topic": "t",
                "form": "overwrite",
                "protocol": "synthetic",
            },
            {
                **row("a", 1, condition="reset"),
                "topic": "t",
                "form": "overwrite",
                "protocol": "synthetic",
            },
            {
                **row("b", 1),
                "seed": 1,
                "topic": "t",
                "form": "overwrite",
                "protocol": "synthetic",
            },
        ]
        headline = filter_preregistered_records(records, protocol="synthetic", split="test_id")
        self.assertEqual(len(headline), 2)
        summary = seeded_stratified_accuracy(records)
        cells = {cell["condition"]: cell for cell in summary["cells"]}
        self.assertEqual(cells["standard"]["n_seeds"], 2)
        self.assertEqual(cells["standard"]["seed_mean_accuracy"], 0.5)
        self.assertEqual(cells["reset"]["seed_mean_accuracy"], 0.0)

    def test_noise_robustness_requires_repeated_diffusion_seeds(self):
        single = compute_synthetic_metrics([row("a", 0, diffusion_seed=0)])
        self.assertIsNone(single["noise_robustness"])
        repeated = compute_synthetic_metrics(
            [
                row("a", 0, diffusion_seed=0),
                row("a", 1, diffusion_seed=1),
            ]
        )
        self.assertEqual(repeated["noise_robustness"]["n_episode_queries"], 1)
        self.assertEqual(repeated["noise_robustness"]["diffusion_seed_counts"], [2])


if __name__ == "__main__":
    unittest.main()
