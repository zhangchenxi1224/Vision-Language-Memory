from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vision_memory.prefeval.official_pipeline import (  # noqa: E402
    EVAL_TOPICS, TRAIN_TOPICS, context_messages, context_prefix,
    disclosure_messages, render_released_mistral, sft_record,
    tokenize_answer_only, writer_training_view,
    benchmark_generation_messages,
)


def example(n=0):
    row = {"preference": "Prefer quiet venues.", "question": "Where shall we meet?",
           "response_to_pref": "Prefer quiet venues.", "response_to_q": "Prefer quiet venues."}
    contexts = [{"role": role, "content": f"context {i} {role}"}
                for i in range(10) for role in ("user", "assistant")]
    return sft_record(row, topic="travel_restaurant", row_index=0, contexts=contexts, inter_turns=n)


class ToyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}|" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode("utf-8"))


class OfficialPipelineTest(unittest.TestCase):
    def test_topic_split_is_official_and_disjoint(self):
        self.assertEqual(EVAL_TOPICS, ("travel_transportation", "shop_technology", "education_resources", "shop_motors"))
        self.assertEqual(len(TRAIN_TOPICS), 16)
        self.assertFalse(set(TRAIN_TOPICS) & set(EVAL_TOPICS))

    def test_full_implicit_disclosure_ignores_hidden_ground_truth(self):
        row = {"preference": "HIDDEN", "explanation": "SECRET", "aligned_op": "SECRET",
               "conversation": {"query": "Lunch?", "assistant_options": "A or B", "user_selection": "A",
                                "assistant_acknowledgment": "OK"}}
        self.assertEqual([m["content"] for m in disclosure_messages(row, "implicit_choice")], ["Lunch?", "A or B", "A", "OK"])
        row["conversation"] = {"0": {"user": "Unrelated", "assistant": "Hello"},
                               "1": {"user": "Quiet please", "assistant": "Sure"}}
        self.assertEqual([m["content"] for m in disclosure_messages(row, "implicit_persona")],
                         ["Unrelated", "Hello", "Quiet please", "Sure"])
        row["preference"] = "CHANGED HIDDEN LABEL"
        self.assertEqual(len(disclosure_messages(row, "implicit_persona")), 4)

    def test_contexts_preserve_exchanges_and_upstream_selection(self):
        pool = [{"conversation": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "ok"}]}
                for i in range(6)]
        self.assertEqual(context_messages(pool, for_sft=True)[0]["content"], "2")
        self.assertEqual(context_messages(pool, for_sft=False)[0]["content"], "0")
        self.assertEqual(len(context_prefix(context_messages(pool, for_sft=False), 5)), 10)
        with self.assertRaises(ValueError):
            context_prefix(context_messages(pool, for_sft=True), 5)

    def test_context_conditions_share_target_and_do_not_leak_query_to_writer(self):
        records = [example(n) for n in (0, 5, 10)]
        self.assertEqual([len(r["history"]) // 2 + 1 for r in records], [2, 7, 12])
        self.assertEqual(len({r["target"]["content"] for r in records}), 1)
        view = writer_training_view(records[-1])
        self.assertEqual(len(view["updates"]), 11)
        self.assertNotIn("Where shall we meet?", str(view["updates"]))
        self.assertEqual(view["reader_query"]["content"], "Where shall we meet?")

    def test_loss_masks_history_even_when_answer_already_occurs_there(self):
        record = example()
        tokens = tokenize_answer_only(ToyTokenizer(), record, max_length=2000)
        supervised = [x for x in tokens["labels"] if x != -100]
        self.assertEqual(bytes(supervised).decode(), record["target"]["content"] + "|")
        self.assertEqual(len(tokens["input_ids"]), len(tokens["labels"]))
        with self.assertRaises(ValueError):
            tokenize_answer_only(ToyTokenizer(), record, max_length=5)

    def test_released_mistral_layout_retains_ack_and_plain_query(self):
        rendered = render_released_mistral(example())
        self.assertEqual(rendered, "<s>[INST]\nYou are an AI assistant.\nPrefer quiet venues.\n[/INST]\n"
                         "Prefer quiet venues.</s>\n\n[INST]\nWhere shall we meet?\n[/INST]\nPrefer quiet venues.\n</s>")

    def test_benchmark_requires_model_ack_and_never_passes_evaluator_fields(self):
        record = {"input": {"disclosure": [{"role": "user", "content": "Prefer quiet."}],
                            "needs_model_acknowledgment": True, "query": {"role": "user", "content": "Where?"}},
                  "evaluation_only": {"preference": "SECRET", "explanation": "SECRET"}}
        with self.assertRaises(ValueError):
            benchmark_generation_messages(record, [], inter_turns=0)
        messages = benchmark_generation_messages(record, [], inter_turns=0, explicit_acknowledgment="Noted.")
        self.assertEqual(messages[1]["content"], "Noted.")
        self.assertEqual(messages[-1]["content"], "Where? (Please respond within 300 words.)")
        self.assertNotIn("SECRET", str(messages))


if __name__ == "__main__":
    unittest.main()
