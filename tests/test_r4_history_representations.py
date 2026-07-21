from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.schema import EventKind, QuerySpec, Turn, TurnType  # noqa: E402
from vision_memory.eval.r4_history_representations import (  # noqa: E402
    QWEN_R4_LAST_EFFECTIVE_EVENT,
    QWEN_R4_OPERATION_TAGGED_HISTORY,
    QWEN_R4_RAW_HISTORY,
    R4_EMPTY_MEMORY,
    R4_HISTORY_METHODS,
    R4_HISTORY_TASK_INSTRUCTION,
    VisibleEvent,
    render_history_representation,
    reset_event_stream,
    shuffle_event_stream,
    source_event_stream_sha256,
    state_swap_event_stream,
    visible_event_streams_at_queries,
)


def _query(text: str = "Which option applies?", target_index: int = 0) -> QuerySpec:
    return QuerySpec(
        text=text,
        choices=("teal", "burgundy", "ivory", "no active preference"),
        target_index=target_index,
    )


class R4HistoryRepresentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.events = (
            VisibleEvent(EventKind.SET, "Store teal for the brass carafe."),
            VisibleEvent(EventKind.OVERWRITE, "Replace teal with burgundy."),
            VisibleEvent(EventKind.NOOP, "An unrelated cart moved."),
        )

    def test_method_contract_and_exact_rendering_are_distinct(self) -> None:
        self.assertEqual(
            R4_HISTORY_METHODS,
            (
                "qwen_r4_raw_history",
                "qwen_r4_operation_tagged_history",
                "qwen_r4_last_effective_event",
            ),
        )
        raw = render_history_representation(QWEN_R4_RAW_HISTORY, self.events)
        tagged = render_history_representation(QWEN_R4_OPERATION_TAGGED_HISTORY, self.events)
        reduced = render_history_representation(QWEN_R4_LAST_EFFECTIVE_EVENT, self.events)
        self.assertTrue(raw.memory_text.startswith(R4_HISTORY_TASK_INSTRUCTION))
        self.assertIn("1. Store teal for the brass carafe.", raw.memory_text)
        self.assertNotIn("[SET]", raw.memory_text)
        self.assertIn("1. [SET] Store teal for the brass carafe.", tagged.memory_text)
        self.assertIn("2. [OVERWRITE] Replace teal with burgundy.", tagged.memory_text)
        self.assertNotIn("NOOP", reduced.memory_text)
        self.assertTrue(reduced.memory_text.endswith("[OVERWRITE] Replace teal with burgundy."))
        self.assertEqual((raw.retained_event_count, tagged.retained_event_count), (3, 3))
        self.assertEqual(reduced.retained_event_count, 1)
        for representation in (raw, tagged, reduced):
            self.assertEqual(len(representation.representation_contract_sha256), 64)
            self.assertEqual(len(representation.source_event_stream_sha256), 64)
            self.assertEqual(len(representation.memory_text_sha256), 64)

    def test_reducer_noop_is_a_memory_noop_and_empty_is_unset(self) -> None:
        before = render_history_representation(QWEN_R4_LAST_EFFECTIVE_EVENT, self.events[:2])
        after = render_history_representation(QWEN_R4_LAST_EFFECTIVE_EVENT, self.events)
        self.assertEqual(before.memory_text, after.memory_text)
        self.assertEqual(before.memory_text_sha256, after.memory_text_sha256)
        self.assertNotEqual(before.source_event_stream_sha256, after.source_event_stream_sha256)
        empty = render_history_representation(QWEN_R4_LAST_EFFECTIVE_EVENT, ())
        noop_only = render_history_representation(
            QWEN_R4_LAST_EFFECTIVE_EVENT,
            (VisibleEvent(EventKind.NOOP, "Unrelated fact."),),
        )
        self.assertTrue(empty.memory_text.endswith(R4_EMPTY_MEMORY))
        self.assertEqual(empty.memory_text, noop_only.memory_text)
        self.assertEqual(empty.retained_event_count, 0)

    def test_mixed_is_write_then_read_and_future_events_are_isolated(self) -> None:
        turns = (
            Turn(TurnType.EVENT, EventKind.SET, "Store teal."),
            Turn(TurnType.QUERY, query=_query()),
            Turn(
                TurnType.MIXED,
                EventKind.OVERWRITE,
                "Replace teal with burgundy.",
                _query(target_index=1),
            ),
            Turn(TurnType.EVENT, EventKind.CLEAR, "Clear the preference."),
            Turn(TurnType.QUERY, query=_query(target_index=3)),
        )
        prefixes = visible_event_streams_at_queries(turns)
        self.assertEqual(tuple(len(prefix) for prefix in prefixes), (1, 2, 3))
        self.assertEqual(prefixes[1][-1].kind, EventKind.OVERWRITE)
        self.assertNotIn(EventKind.CLEAR, tuple(event.kind for event in prefixes[1]))
        self.assertEqual(prefixes[2][-1].kind, EventKind.CLEAR)

    def test_query_label_and_episode_metadata_cannot_affect_representation(self) -> None:
        events = (VisibleEvent(EventKind.SET, "Store teal."),)
        left_turns = (
            Turn(TurnType.EVENT, EventKind.SET, "Store teal."),
            Turn(TurnType.QUERY, query=_query("Question A", 0)),
        )
        right_turns = (
            Turn(TurnType.EVENT, EventKind.SET, "Store teal."),
            Turn(TurnType.QUERY, query=_query("Question B", 2)),
        )
        self.assertEqual(visible_event_streams_at_queries(left_turns)[0], events)
        self.assertEqual(visible_event_streams_at_queries(right_turns)[0], events)
        left = render_history_representation(QWEN_R4_OPERATION_TAGGED_HISTORY, events)
        right = render_history_representation(
            QWEN_R4_OPERATION_TAGGED_HISTORY,
            visible_event_streams_at_queries(right_turns)[0],
        )
        self.assertEqual(left, right)
        parameters = tuple(inspect.signature(render_history_representation).parameters)
        self.assertEqual(parameters, ("method", "events"))
        with self.assertRaises(TypeError):
            render_history_representation(QWEN_R4_RAW_HISTORY, ({"kind": "set", "text": "x"},))  # type: ignore[arg-type]

    def test_reset_shuffle_and_swap_are_pure_event_stream_interventions(self) -> None:
        original = self.events
        self.assertEqual(reset_event_stream(original), ())
        shuffled = shuffle_event_stream(original, (2, 0, 1))
        self.assertEqual(shuffled, (original[2], original[0], original[1]))
        self.assertEqual(original, self.events)
        donor = (VisibleEvent(EventKind.CLEAR, "Clear it."),)
        self.assertEqual(state_swap_event_stream(original, donor), donor)
        with self.assertRaisesRegex(ValueError, "every event index"):
            shuffle_event_stream(original, (0, 0, 1))

    def test_stream_hash_is_path_invariant_and_content_sensitive(self) -> None:
        same = tuple(VisibleEvent(event.kind, event.text) for event in self.events)
        self.assertEqual(source_event_stream_sha256(self.events), source_event_stream_sha256(same))
        changed_kind = (VisibleEvent(EventKind.CLEAR, self.events[0].text), *self.events[1:])
        changed_text = (VisibleEvent(EventKind.SET, "Different text."), *self.events[1:])
        self.assertNotEqual(source_event_stream_sha256(self.events), source_event_stream_sha256(changed_kind))
        self.assertNotEqual(source_event_stream_sha256(self.events), source_event_stream_sha256(changed_text))

    def test_raw_model_visible_memory_is_invariant_to_router_kind_only(self) -> None:
        original = (VisibleEvent(EventKind.SET, "The same visible sentence."),)
        changed_kind = (VisibleEvent(EventKind.CLEAR, "The same visible sentence."),)
        left = render_history_representation(QWEN_R4_RAW_HISTORY, original)
        right = render_history_representation(QWEN_R4_RAW_HISTORY, changed_kind)
        self.assertEqual(left.memory_text, right.memory_text)
        self.assertEqual(left.memory_text_sha256, right.memory_text_sha256)
        self.assertNotEqual(left.source_event_stream_sha256, right.source_event_stream_sha256)


if __name__ == "__main__":
    unittest.main()
