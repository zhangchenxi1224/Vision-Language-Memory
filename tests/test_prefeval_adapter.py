from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from dreamlite_mcq import collect_prefeval  # noqa: E402
from vision_memory.event_noise import event_seed  # noqa: E402
from vision_memory.prefeval import (  # noqa: E402
    FORBIDDEN_MODEL_KEYS,
    TOPICS,
    PrefEvalAdapter,
    adaptation_topic_split,
    assign_base_pair_splits,
    prefeval_noise_episode_key,
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_fixture(root: Path) -> Path:
    data = root / "benchmark_dataset"
    for topic in TOPICS:
        question = f"Which venue should I choose for {topic}?"
        write_json(
            data / "mcq_options" / f"{topic}.json",
            [
                {
                    "preference": f"MCQ HIDDEN {topic}",
                    "question": question,
                    "explanation": f"MCQ EXPLANATION {topic}",
                    "classification_task_options": ["calm option", "loud option", "busy option", "crowded option"],
                }
            ],
        )
        write_json(
            data / "explicit_preference" / f"{topic}.json",
            [
                {
                    "preference": f"I explicitly prefer a quiet venue for {topic}.",
                    "question": question,
                    "explanation": f"EXPLICIT EXPLANATION {topic}",
                }
            ],
        )
        write_json(
            data / "implicit_preference" / "choice-based" / f"{topic}.json",
            [
                {
                    "preference": f"CHOICE HIDDEN SECRET {topic}",
                    "question": question,
                    "explanation": f"CHOICE EXPLANATION SECRET {topic}",
                    "aligned_op": f"ALIGNED OP SECRET {topic}",
                    "conversation": {
                        "query": "Where should we meet?",
                        "assistant_options": "1. Calm cafe. 2. Loud club.",
                        "user_selection": "The calm cafe sounds best.",
                        "assistant_acknowledgment": "I inferred your hidden preference.",
                    },
                }
            ],
        )
        write_json(
            data / "implicit_preference" / "persona-driven" / f"{topic}.json",
            [
                {
                    "preference": f"I strongly prefer a calm venue for {topic}",
                    "question": question,
                    "explanation": f"PERSONA EXPLANATION SECRET {topic}",
                    "persona": "A test persona",
                    "conversation": {
                        "0": {"user": "Tell me about the weather.", "assistant": "It is sunny."},
                        "1": {
                            "user": f"For {topic}, a calm venue works best for my meetings.",
                            "assistant": "I can help with that.",
                        },
                    },
                }
            ],
        )

    # This deliberately malformed near-duplicate must never be discovered by globbing.
    (data / "mcq_options" / "travel_hotel copy.json").write_text("not json", encoding="utf-8")
    messages = []
    for index in range(12):
        messages.extend(
            [
                {"role": "user", "content": f"Distractor question {index}"},
                {"role": "assistant", "content": f"Distractor answer {index}"},
            ]
        )
    write_json(data / "filtered_inter_turns.json", [{"conversation_id": "fixture", "conversation": messages}])
    return root


def recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


class PrefEvalAdapterTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = make_fixture(Path(self.temporary.name))
        self.adapter = PrefEvalAdapter(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_fixed_manifest_binds_three_forms_and_ignores_copy_file(self):
        manifest = self.adapter.manifest()
        self.assertEqual(manifest["topics"], list(TOPICS))
        self.assertEqual(manifest["base_pair_count"], 20)
        self.assertEqual(manifest["form_bound_sample_count"], 60)
        episodes = list(self.adapter.iter_episodes())
        self.assertEqual(len(episodes), 60)
        grouped = [episode for episode in episodes if episode.base_pair_id == f"{TOPICS[0]}:0000"]
        self.assertEqual({episode.form for episode in grouped}, {"explicit", "implicit_choice", "implicit_persona"})
        self.assertEqual(len({episode.turns[-1].options for episode in grouped}), 1)
        self.assertEqual(len({episode.target_index for episode in grouped}), 1)
        self.assertEqual(len({episode.split for episode in grouped}), 1)

    def test_implicit_model_inputs_do_not_expose_privileged_fields(self):
        episodes = list(self.adapter.iter_episodes(forms=("implicit_choice", "implicit_persona")))
        for episode in episodes:
            model_input = episode.model_input()
            keys = {str(key).lower() for key in recursive_keys(model_input)}
            self.assertTrue(FORBIDDEN_MODEL_KEYS.isdisjoint(keys))
            serialized = json.dumps(model_input)
            self.assertNotIn("EXPLANATION SECRET", serialized)
            self.assertNotIn("ALIGNED OP SECRET", serialized)
            self.assertNotIn("CHOICE HIDDEN SECRET", serialized)
            self.assertNotIn("I strongly prefer a calm venue", serialized)
            self.assertNotIn("assistant_acknowledgment", serialized)
        choice = next(episode for episode in episodes if episode.form == "implicit_choice")
        self.assertIn("The calm cafe sounds best", choice.turns[0].text)
        persona = next(episode for episode in episodes if episode.form == "implicit_persona")
        self.assertIn("a calm venue works best", persona.turns[0].text)

    def test_explicit_disclosure_is_legitimate_event_but_label_is_separate(self):
        episode = next(self.adapter.iter_episodes(forms=("explicit",)))
        self.assertIn("I explicitly prefer", episode.turns[0].text)
        model_input = episode.model_input()
        self.assertNotIn("target_index", json.dumps(model_input))
        record = episode.to_record()
        self.assertEqual(record["label"]["target_index"], episode.target_index)

    def test_forced_write_is_deterministic_and_has_requested_noops(self):
        first = next(
            self.adapter.iter_episodes(forms=("explicit",), protocol="forced-write", forced_write_k=10)
        )
        second_adapter = PrefEvalAdapter(self.root)
        second = next(
            second_adapter.iter_episodes(forms=("explicit",), protocol="forced-write", forced_write_k=10)
        )
        self.assertEqual(first.to_record(), second.to_record())
        self.assertEqual(len(first.turns), 12)
        self.assertEqual([turn.event_type for turn in first.turns[1:11]], ["noop"] * 10)
        self.assertEqual(first.turns[-1].type, "query")

    def test_forced_write_noise_and_distractors_are_nested_across_k(self):
        oracle = next(self.adapter.iter_episodes(forms=("explicit",), protocol="oracle-sparse"))
        forced = {
            count: next(
                self.adapter.iter_episodes(
                    forms=("explicit",),
                    protocol="forced-write",
                    forced_write_k=count,
                )
            )
            for count in (0, 2, 5, 10)
        }
        noise_key = prefeval_noise_episode_key(oracle.base_pair_id, oracle.form)
        self.assertEqual(
            event_seed(7, noise_key, 0),
            event_seed(
                7,
                prefeval_noise_episode_key(forced[0].base_pair_id, forced[0].form),
                0,
            ),
        )

        distractor_texts = {
            count: [turn.text for turn in episode.turns[1:-1]]
            for count, episode in forced.items()
        }
        self.assertEqual(distractor_texts[2], distractor_texts[5][:2])
        self.assertEqual(distractor_texts[5], distractor_texts[10][:5])

        event_seeds = {
            count: [event_seed(7, noise_key, turn_index) for turn_index in range(1, count + 1)]
            for count in (0, 2, 5, 10)
        }
        self.assertEqual(event_seeds[2], event_seeds[5][:2])
        self.assertEqual(event_seeds[5], event_seeds[10][:5])

    def test_collect_prefeval_uses_paired_noise_key_not_protocol_sample_id(self):
        episodes = [
            next(self.adapter.iter_episodes(forms=("explicit",), protocol="oracle-sparse")),
            next(
                self.adapter.iter_episodes(
                    forms=("explicit",), protocol="forced-write", forced_write_k=2
                )
            ),
        ]
        path = Path(self.temporary.name) / "paired.jsonl"
        path.write_text(
            "".join(json.dumps(episode.to_record()) + "\n" for episode in episodes),
            encoding="utf-8",
        )

        class RecordingUpdater:
            def __init__(self):
                self.calls = []

            def __call__(self, state, text, episode_id, turn_id):
                self.calls.append((text, episode_id, turn_id))
                return state

        class RecordingModel:
            def __init__(self):
                self.updater = RecordingUpdater()

            @staticmethod
            def reset_state():
                return torch.zeros(1)

        model = RecordingModel()
        queries = collect_prefeval(
            model,
            path,
            limit=None,
            recurrence_mode="direct_latent",
            skip_noop=False,
        )
        expected_key = prefeval_noise_episode_key(episodes[0].base_pair_id, episodes[0].form)
        set_calls = [call for call in model.updater.calls if call[2] == 0]
        self.assertEqual(len(set_calls), 2)
        self.assertEqual({call[1] for call in set_calls}, {expected_key})
        self.assertEqual({query.metadata["noise_episode_key"] for query in queries}, {expected_key})

    def test_seed_2026_topic_and_pair_splits_are_group_safe(self):
        topic_split = adaptation_topic_split(2026)
        self.assertEqual(
            topic_split.ood_topics,
            ("entertain_sports", "entertain_games", "education_learning_styles", "education_resources"),
        )
        topic = "travel_hotel"
        pair_ids = [f"{topic}:{index:04d}" for index in range(20)]
        split = assign_base_pair_splits({topic: pair_ids}, seed=2026)
        self.assertEqual(sum(value == "adapt_dev" for value in split.values()), 2)
        self.assertEqual(sum(value == "adapt_train" for value in split.values()), 18)

    def test_invalid_protocol_contracts_fail_closed(self):
        with self.assertRaises(ValueError):
            list(self.adapter.iter_episodes(protocol="oracle-sparse", forced_write_k=2))
        with self.assertRaises(ValueError):
            list(self.adapter.iter_episodes(protocol="forced-write", forced_write_k=3))


if __name__ == "__main__":
    unittest.main()
