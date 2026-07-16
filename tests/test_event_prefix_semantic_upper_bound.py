from __future__ import annotations

import sys
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probes.qwen_event_prefix_semantic_upper_bound import (  # noqa: E402
    DISCLAIMER,
    build_permutation_views,
    collect_event_prefix_reads,
    event_prefix_key,
    permutation_parity,
    select_code_index,
    selector_leakage_audit,
    split_choice_permutations,
    target_position_counts,
)
from vision_memory.data import Episode, EventKind, QuerySpec, Turn, TurnType  # noqa: E402


class EventPrefixSemanticUpperBoundTest(unittest.TestCase):
    @staticmethod
    def episode(
        *,
        episode_id: str,
        query: QuerySpec,
        event_texts: tuple[str, str, str] = (
            "Remember red for the mug.",
            "The hallway clock was repaired.",
            "Remember blue for the mug.",
        ),
    ) -> Episode:
        return Episode(
            episode_id=episode_id,
            split="train",
            seed=0,
            entity_id="entity-0",
            entity_surface="mug fixture",
            template_id="template-0",
            template_family="memory memo",
            pair_id=f"pair-{episode_id}",
            counterfactual_episode_id=f"counterfactual-{episode_id}",
            topic="color",
            turns=(
                Turn(TurnType.EVENT, EventKind.SET, event_texts[0]),
                Turn(TurnType.EVENT, EventKind.NOOP, event_texts[1]),
                Turn(TurnType.EVENT, EventKind.OVERWRITE, event_texts[2]),
                Turn(TurnType.QUERY, query=query),
            ),
        )

    def test_prefix_key_is_ordered_visible_event_text_only(self):
        first = event_prefix_key(("visible event one", "visible event two"))
        repeated = event_prefix_key(("visible event one", "visible event two"))
        reversed_key = event_prefix_key(("visible event two", "visible event one"))
        changed = event_prefix_key(("visible event one", "different visible event"))

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, reversed_key)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)
        with self.assertRaisesRegex(ValueError, "at least one"):
            event_prefix_key(())
        with self.assertRaisesRegex(ValueError, "non-empty"):
            event_prefix_key(("",))

    def test_query_and_target_mutations_cannot_change_selection_key(self):
        choices = ("red", "blue", "green", "yellow")
        original_query = QuerySpec(
            text="Which color is current?",
            choices=choices,
            target_index=0,
            comparison_id="comparison-original",
        )
        mutated_query = QuerySpec(
            text="Completely different question wording?",
            choices=("yellow", "green", "blue", "red"),
            target_index=1,
            comparison_id="comparison-mutated",
        )
        original = collect_event_prefix_reads([self.episode(episode_id="episode-original", query=original_query)])[0]
        mutated = collect_event_prefix_reads([self.episode(episode_id="episode-mutated", query=mutated_query)])[0]

        self.assertNotEqual(original.query.text, mutated.query.text)
        self.assertNotEqual(original.query.target, mutated.query.target)
        self.assertEqual(original.event_texts, mutated.event_texts)
        self.assertEqual(original.state_key, mutated.state_key)
        mapping = {original.state_key: 7}
        self.assertEqual(select_code_index(original.state_key, mapping), 7)
        self.assertEqual(select_code_index(mutated.state_key, mapping), 7)
        audit = selector_leakage_audit((original, mutated))
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["forbidden_selector_fields_used"])
        self.assertEqual(audit["selector_signature"], ["state_key", "code_index_by_key"])

    def test_mixed_turn_event_enters_prefix_before_its_query(self):
        choices = ("red", "blue", "green", "yellow")
        first_query = QuerySpec("First read?", choices, 0, comparison_id="first")
        second_query = QuerySpec("Second read?", choices, 1, comparison_id="second")
        episode = Episode(
            episode_id="mixed-order",
            split="train",
            seed=0,
            entity_id="entity-0",
            template_id="template-0",
            pair_id="pair-0",
            counterfactual_episode_id="counterfactual-0",
            topic="color",
            turns=(
                Turn(TurnType.EVENT, EventKind.SET, "event one"),
                Turn(TurnType.QUERY, query=first_query),
                Turn(TurnType.EVENT, EventKind.OVERWRITE, "event two"),
                Turn(
                    TurnType.MIXED,
                    EventKind.NOOP,
                    "event three in mixed turn",
                    second_query,
                ),
            ),
        )

        reads = collect_event_prefix_reads((episode,))
        self.assertEqual(len(reads), 2)
        self.assertEqual(reads[0].event_texts, ("event one",))
        self.assertEqual(
            reads[1].event_texts,
            ("event one", "event two", "event three in mixed turn"),
        )
        self.assertEqual(reads[1].state_key, event_prefix_key(reads[1].event_texts))

    def test_even_odd_permutation_split_is_disjoint_exhaustive_and_balanced(self):
        choices = ("delta", "alpha", "charlie", "bravo")
        train, heldout = split_choice_permutations(choices)
        reversed_train, reversed_heldout = split_choice_permutations(tuple(reversed(choices)))

        self.assertEqual(train, reversed_train)
        self.assertEqual(heldout, reversed_heldout)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(heldout), 12)
        self.assertFalse(set(train) & set(heldout))
        self.assertEqual(len(set(train) | set(heldout)), 24)
        canonical = tuple(sorted(choices))
        for parity, permutations in ((0, train), (1, heldout)):
            for permutation in permutations:
                indices = tuple(canonical.index(choice) for choice in permutation)
                self.assertEqual(permutation_parity(indices), parity)
            target_positions = Counter(permutation.index("charlie") for permutation in permutations)
            self.assertEqual(target_positions, Counter({0: 3, 1: 3, 2: 3, 3: 3}))

    def test_views_use_same_state_key_across_unseen_target_positions(self):
        choices = ("red", "blue", "green", "yellow")
        query = QuerySpec(
            text="Which color is current?",
            choices=choices,
            target_index=1,
            comparison_id="comparison-0",
        )
        read = collect_event_prefix_reads([self.episode(episode_id="episode-0", query=query)])[0]
        train, heldout = build_permutation_views((read,))

        self.assertEqual(len(train), 12)
        self.assertEqual(len(heldout), 12)
        self.assertEqual(target_position_counts(train), {0: 3, 1: 3, 2: 3, 3: 3})
        self.assertEqual(target_position_counts(heldout), {0: 3, 1: 3, 2: 3, 3: 3})
        self.assertTrue(all(view.read_index == 0 for view in train + heldout))
        self.assertEqual(select_code_index(read.state_key, {read.state_key: 0}), 0)
        self.assertIn("EVENT-PREFIX CODEBOOK DIAGNOSTIC ONLY", DISCLAIMER)
        self.assertIn("not a learned updater", DISCLAIMER)

        changed_query = replace(query, target_index=2)
        changed_read = collect_event_prefix_reads([self.episode(episode_id="episode-1", query=changed_query)])[0]
        self.assertEqual(read.state_key, changed_read.state_key)


if __name__ == "__main__":
    unittest.main()
