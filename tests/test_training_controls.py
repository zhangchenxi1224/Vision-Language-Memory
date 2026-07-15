from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import (  # noqa: E402
    AdaptedPrefEvalRecord,
    StaticLearnedInitialImage,
    read_prefeval_adapted_jsonl,
    read_prefeval_supervised_jsonl,
    run_episode,
    select_curriculum_episodes,
)
from scripts.train.lightweight_episode import (  # noqa: E402
    evaluate_accuracy,
    training_budget_open,
    validate_overfit_gate_configuration,
    validate_overfit_gate_episodes,
)


def scalar_loss(image, query, target):
    del query, target
    return SimpleNamespace(loss=image.mean(), target_ids=torch.ones(1, 1, dtype=torch.long))


class RoutingControlTest(unittest.TestCase):
    def episode(self):
        return {
            "episode_id": "routing",
            "turns": [
                {"kind": "event", "event_kind": "set", "event_text": "remember red"},
                {"kind": "event", "event_kind": "noop", "event_text": "weather"},
                {"kind": "query", "query_text": "Which?", "choices": ["a", "b", "c", "d"], "target_index": 0},
            ],
        }

    def test_noop_policy_is_explicit_and_audited(self):
        for policy, expected_calls, expected_trace in (
            ("update", ["remember red", "weather"], ("0:set:update", "1:noop:update")),
            ("skip", ["remember red"], ("0:set:update", "1:noop:skip")),
        ):
            calls: list[str] = []

            def updater(state, text, episode_id, turn_id):
                del episode_id, turn_id
                calls.append(text)
                return state + 1

            result = run_episode(
                episode=self.episode(),
                initial_state=torch.zeros(1, 1, 1, 1),
                update_fn=updater,
                decode_fn=lambda state: state,
                reader_loss_fn=scalar_loss,
                noop_policy=policy,
            )
            self.assertEqual(calls, expected_calls)
            self.assertEqual(result.updater_trace, expected_trace)

    def test_skip_policy_fails_closed_without_event_label(self):
        episode = self.episode()
        del episode["turns"][0]["event_kind"]
        with self.assertRaisesRegex(ValueError, "fail closed"):
            run_episode(
                episode=episode,
                initial_state=torch.zeros(1, 1, 1, 1),
                update_fn=lambda state, *_: state,
                decode_fn=lambda state: state,
                reader_loss_fn=scalar_loss,
                noop_policy="skip",
            )

    def test_set_only_selects_whole_episodes_without_rewriting(self):
        set_episode = self.episode()
        other = {
            **self.episode(),
            "episode_id": "overwrite",
            "turns": [dict(turn) for turn in self.episode()["turns"]],
        }
        other["turns"][0]["event_kind"] = "overwrite"
        selected, audit = select_curriculum_episodes([set_episode, other], curriculum="set-only")
        self.assertEqual([episode["episode_id"] for episode in selected], ["routing"])
        self.assertEqual(audit.excluded_by_reason, {"contains_non_set_transition": 1})
        self.assertEqual(set_episode["turns"][1]["event_kind"], "noop")


class PrefEvalTrainingBoundaryTest(unittest.TestCase):
    def record(self):
        return {
            "schema_version": "vision_memory.prefeval.episode.v1",
            "model_input": {
                "schema_version": "vision_memory.prefeval.model-input.v1",
                "sample_id": "topic:0000:explicit:oracle-sparse:k0",
                "base_pair_id": "topic:0000",
                "topic": "education_resources",
                "form": "explicit",
                "split": "adapt_train",
                "protocol": "oracle-sparse",
                "forced_write_k": 0,
                "turns": [
                    {"type": "event", "text": "I prefer red.", "event_type": "set", "evidence_source": "raw"},
                    {"type": "query", "text": "Which?", "options": ["red", "blue", "green", "yellow"]},
                ],
            },
            "label": {"target_index": 0, "target_choice": "A"},
            "audit": {"row_index": 0},
        }

    def test_safe_conversion_places_target_only_at_loss_boundary(self):
        adapted = AdaptedPrefEvalRecord.from_record(self.record())
        episode = adapted.supervised_episode()
        self.assertNotIn("label", json.dumps(adapted.model_input))
        calls: list[tuple[str, str]] = []

        def updater(state, text, *_):
            calls.append(("updater", text))
            return state

        def reader(image, query, target):
            del image
            calls.append(("reader", query))
            calls.append(("loss_target", target))
            return torch.ones((), requires_grad=True)

        run_episode(
            episode=episode,
            initial_state=torch.zeros(1, 1, 1, 1),
            update_fn=updater,
            decode_fn=lambda state: state,
            reader_loss_fn=reader,
        )
        self.assertEqual(calls[0], ("updater", "I prefer red."))
        self.assertNotIn("target_index", calls[1][1])
        self.assertNotIn("Correct answer", calls[1][1])
        self.assertEqual(calls[2], ("loss_target", "red"))

    def test_model_input_label_key_is_rejected_and_reader_filters_split(self):
        bad = self.record()
        bad["model_input"]["turns"][0]["target_index"] = 0
        with self.assertRaisesRegex(ValueError, "Supervision key|Unknown event fields"):
            AdaptedPrefEvalRecord.from_record(bad)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefeval.jsonl"
            path.write_text(json.dumps(self.record()) + "\n", encoding="utf-8")
            episodes = read_prefeval_adapted_jsonl(path, allowed_splits={"adapt_train"})
            self.assertEqual(len(episodes), 1)
            converted = Path(directory) / "converted.jsonl"
            converted.write_text(json.dumps(episodes[0]) + "\n", encoding="utf-8")
            self.assertEqual(
                read_prefeval_supervised_jsonl(converted, allowed_splits={"adapt_train"}),
                episodes,
            )
            with self.assertRaisesRegex(ValueError, "No eligible"):
                read_prefeval_adapted_jsonl(path, allowed_splits={"adapt_dev"})


class StaticImageBaselineTest(unittest.TestCase):
    def test_static_image_ignores_events_but_receives_gradient(self):
        model = StaticLearnedInitialImage(output_size=8)
        state = model.initial_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        updated = model.update(state, "event text")
        self.assertIs(updated, state)
        image = model.render(updated)
        image.mean().backward()
        self.assertIsNotNone(model.image_logits.grad)
        self.assertGreater(float(model.image_logits.grad.norm()), 0.0)


class LightweightPredictionSchemaTest(unittest.TestCase):
    def test_predictions_keep_dataset_pairing_separate_from_noop_policy(self):
        class Model(torch.nn.Module):
            @staticmethod
            def initial_state(*, batch_size, device, dtype):
                return torch.zeros(batch_size, 1, 1, 1, device=device, dtype=dtype)

            @staticmethod
            def update(state, _text):
                return state + 1

            @staticmethod
            def render(state):
                return state.expand(1, 3, 2, 2)

        episode = {
            "episode_id": "distractor-a",
            "split": "test_id",
            "topic": "topic",
            "pair_id": "semantic-pair",
            "counterfactual_episode_id": "distractor-b",
            "distractor_pair_id": "stream-pair",
            "distractor_episode_id": "clean-a",
            "distractor_variant": "distractor",
            "turns": [
                {"kind": "event", "event_kind": "noop", "event_text": "irrelevant"},
                {
                    "kind": "query",
                    "query": {
                        "text": "Which?",
                        "choices": ["a", "b", "c", "d"],
                        "target_index": 0,
                        "comparison_id": "stream-pair:q0",
                    },
                },
            ],
        }
        score = SimpleNamespace(predicted_index=0, mean_nll=(0.0, 1.0, 2.0, 3.0))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            with patch("scripts.train.lightweight_episode.qwen3vl_choice_nll", return_value=score):
                accuracy = evaluate_accuracy(
                    episodes=[episode],
                    model=Model(),
                    reader=object(),
                    processor=object(),
                    device=torch.device("cpu"),
                    noop_policy="skip",
                    predictions_path=output,
                    method="lightweight_recurrent",
                    seed=0,
                )
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(accuracy, 1.0)
        self.assertEqual(record["counterfactual_pair_id"], "semantic-pair")
        self.assertEqual(record["distractor_pair_id"], "stream-pair")
        self.assertEqual(record["distractor_variant"], "distractor")
        self.assertEqual(record["query_comparison_id"], "stream-pair:q0")
        self.assertEqual(record["noop_policy"], "skip")
        self.assertEqual(record["noop_events_since_query"], 1)
        self.assertEqual(record["noop_events_applied_since_query"], 0)
        self.assertNotIn("counterfactual_variant", record)


class LightweightOverfitGateTest(unittest.TestCase):
    @staticmethod
    def arguments(**updates):
        values = {
            "overfit_gate": True,
            "method": "recurrent",
            "dataset_format": "synthetic",
            "curriculum": "full",
            "noop_policy": "update",
            "learn_initial_state": False,
            "overfit_episodes": 64,
            "max_optimizer_steps": 2_000,
            "overfit_threshold": 0.90,
            "epochs": 2,
        }
        values.update(updates)
        return SimpleNamespace(**values)

    @staticmethod
    def paired_episodes():
        episodes = []
        for pair_index in range(32):
            pair_id = f"stream-{pair_index}"
            comparison_id = f"{pair_id}:q0"
            for variant, episode_suffix, counterpart_suffix in (
                ("clean", "clean", "distractor"),
                ("distractor", "distractor", "clean"),
            ):
                episode_id = f"{pair_id}:{episode_suffix}"
                counterpart_id = f"{pair_id}:{counterpart_suffix}"
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "distractor_variant": variant,
                        "distractor_pair_id": pair_id,
                        "distractor_episode_id": counterpart_id,
                        "turns": [
                            {
                                "kind": "query",
                                "query_text": "Which?",
                                "choices": ["a", "b", "c", "d"],
                                "target_index": 0,
                                "comparison_id": comparison_id,
                            }
                        ],
                    }
                )
        return episodes

    def test_formal_gate_constants_cannot_be_weakened(self):
        validate_overfit_gate_configuration(self.arguments())
        for update in (
            {"overfit_episodes": 32},
            {"max_optimizer_steps": 1_000},
            {"overfit_threshold": 0.50},
            {"noop_policy": "skip"},
        ):
            with self.subTest(update=update), self.assertRaises(SystemExit):
                validate_overfit_gate_configuration(self.arguments(**update))

    def test_formal_gate_budget_is_optimizer_step_driven(self):
        args = self.arguments()
        self.assertTrue(
            training_budget_open(args, epoch=2, optimizer_step=16, gate_passed=False)
        )
        self.assertFalse(
            training_budget_open(args, epoch=250, optimizer_step=2_000, gate_passed=False)
        )
        self.assertFalse(
            training_budget_open(args, epoch=1, optimizer_step=250, gate_passed=True)
        )

    def test_gate_subset_requires_reciprocal_clean_distractor_pairs(self):
        episodes = self.paired_episodes()
        validate_overfit_gate_episodes(episodes)
        episodes[1]["distractor_episode_id"] = "wrong"
        with self.assertRaisesRegex(SystemExit, "incomplete|reciprocally"):
            validate_overfit_gate_episodes(episodes)


if __name__ == "__main__":
    unittest.main()
