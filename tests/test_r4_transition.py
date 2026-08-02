from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.schema import EventKind, Episode, QuerySpec, Turn, TurnType  # noqa: E402
from vision_memory.training.r4_transition import (  # noqa: E402
    R4_BALANCED_BLOCK_SIZE,
    R4_DIFFUSION_STEPS,
    R4_EVENT_KINDS,
    R4_SCHEDULE_SCHEMA,
    R4ScheduleCursor,
    TransitionObjective,
    build_balanced_schedule_block,
    build_transition_index,
    iter_balanced_schedule,
    make_r4_manifest_contract,
    target_update_kwargs,
    trainable_transition_examples,
    transition_index_audit,
    validate_r4_manifest_contract,
    validate_teacher_free_bindings,
)


def query(name: str, target_index: int = 0) -> QuerySpec:
    return QuerySpec(
        text=f"{name}?",
        choices=(f"{name}-a", f"{name}-b", f"{name}-c", f"{name}-d"),
        target_index=target_index,
    )


def routing_episode(*, episode_id: str = "r4-routing") -> Episode:
    return Episode(
        episode_id=episode_id,
        split="train",
        seed=7,
        entity_id=f"entity-{episode_id}",
        template_id="template-r4",
        pair_id=f"pair-{episode_id}",
        counterfactual_episode_id=f"{episode_id}-mate",
        topic="color",
        turns=(
            Turn(TurnType.EVENT, EventKind.SET, "set blue"),
            Turn(TurnType.QUERY, query=query("set", 1)),
            Turn(TurnType.MIXED, EventKind.OVERWRITE, "overwrite green", query("overwrite", 2)),
            Turn(TurnType.QUERY, query=query("overwrite-probe", 2)),
            Turn(TurnType.EVENT, EventKind.NOOP, "unrelated weather"),
            Turn(TurnType.QUERY, query=query("noop-incidental", 2)),
            Turn(TurnType.EVENT, EventKind.CLEAR, "clear color"),
            Turn(TurnType.QUERY, query=query("clear", 3)),
        ),
    )


class R4TransitionIndexTest(unittest.TestCase):
    def test_indexes_by_event_kind_and_attributes_only_local_queries(self):
        episode = routing_episode()
        examples = build_transition_index([episode])

        self.assertEqual([example.event_kind for example in examples], ["set", "overwrite", "noop", "clear"])
        self.assertEqual([example.target_turn_index for example in examples], [0, 2, 4, 6])
        self.assertEqual(examples[0].local_query_indices, (1,))
        self.assertEqual(examples[1].local_query_indices, (2, 3))
        self.assertEqual(examples[2].local_query_indices, (5,))
        self.assertEqual(examples[3].local_query_indices, (7,))

        # The mixed query at turn 2 belongs to the overwrite, never to the SET.
        self.assertNotIn(2, examples[0].local_query_indices)
        self.assertEqual(examples[0].next_updater_index, 2)
        self.assertEqual(examples[1].next_updater_index, 4)

    def test_preserves_original_episode_turns_and_exact_prefix(self):
        episode = routing_episode()
        examples = build_transition_index([episode])
        overwrite = examples[1]

        self.assertIs(overwrite.episode, episode)
        self.assertIs(overwrite.target_turn, episode.turns[2])
        self.assertIs(overwrite.prefix_turns[0], episode.turns[0])
        self.assertIs(overwrite.prefix_turns[1], episode.turns[1])
        self.assertEqual(overwrite.prefix_updater_indices, (0,))
        self.assertIs(overwrite.local_query_turns[0], episode.turns[2])
        self.assertIs(overwrite.local_query_turns[1], episode.turns[3])

    def test_intermediate_noop_is_identity_only_even_with_incidental_query(self):
        noop = build_transition_index([routing_episode()])[2]

        self.assertEqual(noop.objective, TransitionObjective.IDENTITY_ONLY)
        self.assertTrue(noop.uses_identity_loss)
        self.assertFalse(noop.uses_qa_loss)
        self.assertEqual(noop.local_query_indices, (5,))
        self.assertEqual(noop.qa_query_indices, ())

    def test_terminal_noop_can_combine_identity_and_local_qa(self):
        episode = Episode(
            episode_id="terminal-noop",
            split="train",
            seed=0,
            entity_id="entity-terminal-noop",
            template_id="template",
            pair_id="pair-terminal-noop",
            counterfactual_episode_id="terminal-noop-mate",
            topic="food",
            turns=(
                Turn(TurnType.EVENT, EventKind.SET, "set sushi"),
                Turn(TurnType.QUERY, query=query("food", 1)),
                Turn(TurnType.MIXED, EventKind.NOOP, "weather", query("food-again", 1)),
                Turn(TurnType.QUERY, query=query("food-probe", 1)),
            ),
        )
        terminal_noop = build_transition_index([episode])[1]

        self.assertTrue(terminal_noop.is_terminal_update)
        self.assertEqual(terminal_noop.objective, TransitionObjective.QA_AND_IDENTITY)
        self.assertEqual(terminal_noop.qa_query_indices, (2, 3))

    def test_mutation_without_local_query_is_prefix_only(self):
        episode = Episode(
            episode_id="prefix-only",
            split="train",
            seed=0,
            entity_id="entity-prefix-only",
            template_id="template",
            pair_id="pair-prefix-only",
            counterfactual_episode_id="prefix-only-mate",
            topic="hotel",
            turns=(
                Turn(TurnType.EVENT, EventKind.SET, "set first hotel"),
                Turn(TurnType.MIXED, EventKind.OVERWRITE, "overwrite hotel", query("hotel", 1)),
            ),
        )
        examples = build_transition_index([episode])

        self.assertEqual(examples[0].objective, TransitionObjective.PREFIX_ONLY)
        self.assertFalse(examples[0].is_trainable)
        self.assertEqual(trainable_transition_examples(examples), (examples[1],))

    def test_malformed_updater_missing_event_kind_fails_closed(self):
        malformed = {
            "episode_id": "malformed",
            "turns": [
                {"type": "event", "event_text": "missing kind"},
                {"type": "query", "query": {"text": "q"}},
            ],
        }
        with self.assertRaisesRegex(ValueError, "missing event_kind"):
            build_transition_index([malformed])

    def test_malformed_updater_missing_event_text_fails_closed(self):
        malformed = {
            "episode_id": "malformed-event-text",
            "turns": [
                {"type": "event", "event_kind": "set", "event_text": "   "},
                {"type": "query", "query": {"text": "q"}},
            ],
        }
        with self.assertRaisesRegex(ValueError, "non-empty event_text"):
            build_transition_index([malformed])

    def test_hidden_ledger_is_rejected_before_indexing(self):
        value = routing_episode().to_dict()
        value["hidden_ledger"] = {"color": "blue"}
        with self.assertRaisesRegex(ValueError, "Hidden ledger"):
            build_transition_index([value])

    def test_duplicate_episode_ids_fail_closed(self):
        episode = routing_episode()
        with self.assertRaisesRegex(ValueError, "Duplicate episode_id"):
            build_transition_index([episode, episode])

    def test_audit_makes_prefix_only_exclusions_explicit(self):
        examples = build_transition_index([routing_episode()])
        audit = transition_index_audit(examples)

        self.assertEqual(audit["total_transitions"], 4)
        self.assertEqual(audit["trainable_transitions"], 4)
        self.assertEqual(audit["local_query_count"], 5)
        self.assertEqual(audit["by_event_kind"], {"set": 1, "overwrite": 1, "clear": 1, "noop": 1})


class R4BalancedScheduleTest(unittest.TestCase):
    def setUp(self):
        self.examples = build_transition_index([routing_episode()])

    def test_each_block_contains_exact_event_kind_step_cross_product(self):
        block = build_balanced_schedule_block(self.examples, seed=19, block_index=0)

        self.assertEqual(len(block), R4_BALANCED_BLOCK_SIZE)
        self.assertEqual(
            {(item.event_kind, item.selected_step_index) for item in block},
            {(kind, step) for kind in R4_EVENT_KINDS for step in R4_DIFFUSION_STEPS},
        )
        self.assertEqual([item.unit_index for item in block], list(range(16)))
        self.assertEqual([item.block_offset for item in block], list(range(16)))

    def test_resume_suffix_matches_uninterrupted_schedule_exactly(self):
        uninterrupted = list(iter_balanced_schedule(self.examples, seed=23, num_units=49))
        resumed = list(
            iter_balanced_schedule(
                self.examples,
                seed=23,
                start_unit_index=17,
                num_units=32,
            )
        )

        self.assertEqual(resumed, uninterrupted[17:])
        self.assertEqual([item.receipt() for item in resumed], [item.receipt() for item in uninterrupted[17:]])

    def test_schedule_cursor_round_trip_is_strict(self):
        cursor = R4ScheduleCursor(seed=23, next_unit_index=17)
        self.assertEqual(R4ScheduleCursor.from_dict(cursor.to_dict()), cursor)
        self.assertEqual(cursor.schema, R4_SCHEDULE_SCHEMA)

        with self.assertRaisesRegex(ValueError, "unknown"):
            R4ScheduleCursor.from_dict({**cursor.to_dict(), "epoch": 1})
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            R4ScheduleCursor.from_dict({**cursor.to_dict(), "schema": "stale"})

    def test_schedule_requires_a_trainable_bucket_for_every_event_kind(self):
        without_clear = tuple(example for example in self.examples if example.event_kind != "clear")
        with self.assertRaisesRegex(ValueError, "every event kind"):
            build_balanced_schedule_block(without_clear, seed=0, block_index=0)

    def test_target_kwargs_lock_drtune_single_step_and_rgb_persistence(self):
        item = build_balanced_schedule_block(self.examples, seed=0, block_index=0)[0]
        self.assertEqual(
            target_update_kwargs(item),
            {
                "gradient_mode": "drtune",
                "selected_step_indices": (item.selected_step_index,),
                "persistent_state": "float_rgb",
            },
        )


class R4ManifestContractTest(unittest.TestCase):
    def test_contract_locks_free_pixel_teacher_free_boundary(self):
        contract = make_r4_manifest_contract(schedule_seed=41)
        self.assertEqual(validate_r4_manifest_contract(contract), contract)
        self.assertEqual(contract["persistent_state"], "float_rgb")
        self.assertEqual(contract["state_target_policy"], "none")
        self.assertFalse(contract["canonical_teacher_artifacts_loaded"])
        self.assertTrue(contract["pixel_or_latent_targets_forbidden"])
        self.assertTrue(contract["codebook_forbidden"])
        self.assertEqual(
            contract["updater_learned_conditioning"],
            "previous_rgb+visible_event_text",
        )
        self.assertTrue(contract["sampling_noise_keys_answer_agnostic"])
        self.assertTrue(contract["query_or_target_in_noise_seed_forbidden"])
        self.assertEqual(contract["balanced_block_size"], 16)

    def test_contract_tampering_and_unknown_fields_fail_closed(self):
        contract = make_r4_manifest_contract(schedule_seed=41)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            validate_r4_manifest_contract({**contract, "persistent_state": "latent"})
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_r4_manifest_contract({**contract, "teacher_manifest": "teacher.json"})
        with self.assertRaises(TypeError):
            validate_r4_manifest_contract({**contract, "balanced_schedule_seed": True})

    def test_any_non_none_teacher_or_target_binding_is_forbidden(self):
        validate_teacher_free_bindings(
            teacher_manifest=None,
            teacher_sidecar=None,
            canonical_target=None,
            codebook=None,
        )
        with self.assertRaisesRegex(ValueError, "teacher_manifest"):
            validate_teacher_free_bindings(teacher_manifest=Path("teacher.json"))
        with self.assertRaisesRegex(ValueError, "codebook"):
            validate_teacher_free_bindings(codebook=False)


if __name__ == "__main__":
    unittest.main()
