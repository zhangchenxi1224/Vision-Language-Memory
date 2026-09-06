from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from vision_memory.reader.open_answer import generate_short_answer, normalize_short_answer, score_short_answer


class FakeProcessor:
    def __init__(self, *, locked: bool = False, bad_grid: bool = False):
        self.locked = locked
        self.bad_grid = bad_grid
        self.tokenizer = SimpleNamespace(eos_token_id=99)
        self.messages = None
        self.inputs = None
        self.decoded_ids = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "USER IMAGE " + messages[0]["content"][1]["text"] + " ASSISTANT"

    def __call__(self, **kwargs):
        self.inputs = kwargs
        image = kwargs["images"][0]
        if self.locked:
            assert tuple(image.shape) == (3, 256, 256)
            pixels = image.reshape(-1).repeat(2).reshape(256, 1536).float()
        else:
            pixels = image
        self.batch = {
            # Visual expansion means prompt length is determined from batch input_ids,
            # never by separately tokenizing the unexpanded text template.
            "input_ids": torch.tensor([[31, 32, 33, 34, 35]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "pixel_values": pixels,
            "image_grid_thw": torch.tensor([[1, 8 if self.bad_grid else 16, 16]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0]]),
        }
        return self.batch

    def batch_decode(self, token_ids, **kwargs):
        assert set(kwargs) == {"skip_special_tokens", "clean_up_tokenization_spaces"}
        assert kwargs["clean_up_tokenization_spaces"] is False
        self.decoded_ids = token_ids.clone()
        vocabulary = {11: "Green", 12: ".", 13: " extra", 99: "", 100: ""}
        if not kwargs["skip_special_tokens"]:
            vocabulary.update({99: "<|endoftext|>", 100: "<|im_end|>"})
        return ["".join(vocabulary.get(int(item), "PROMPT_LEAK") for item in row) for row in token_ids]


class FakeModel(torch.nn.Module):
    def __init__(self, continuation=(11, 12, 100), *, bad_prefix=False, fail=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.generation_config = SimpleNamespace(eos_token_id=[99, 100])
        self.continuation = continuation
        self.bad_prefix = bad_prefix
        self.fail = fail

    def generate(self, **kwargs):
        self.observed = kwargs
        self.was_training_at_generate = self.training
        self.grad_enabled_at_generate = torch.is_grad_enabled()
        if self.fail:
            raise RuntimeError("simulated generation failure")
        prefix = kwargs["input_ids"].clone()
        if self.bad_prefix:
            prefix[0, 0] += 1
        suffix = torch.tensor([self.continuation], dtype=torch.long)
        return SimpleNamespace(sequences=torch.cat((prefix, suffix), dim=1))


class OpenAnswerTest(unittest.TestCase):
    def call(self, model=None, processor=None, **kwargs):
        return generate_short_answer(
            model=model if model is not None else FakeModel(),
            processor=processor if processor is not None else FakeProcessor(),
            image=torch.rand(3, 4, 4, requires_grad=True),
            query="What color is recorded? Answer only with the color.",
            device="cpu",
            reader_resize_contract=None,
            **kwargs,
        )

    def test_answer_blind_prompt_greedy_generation_and_exact_expanded_prefix_trim(self):
        model, processor = FakeModel(), FakeProcessor()
        result = self.call(model, processor, max_new_tokens=7)
        self.assertEqual(result["raw"], "Green.")
        self.assertEqual(result["raw_with_special_tokens"], "Green.<|im_end|>")
        self.assertEqual(result["input_token_ids"], [31, 32, 33, 34, 35])
        self.assertNotIn(11, result["input_token_ids"])  # The gold answer token never enters the prompt.
        self.assertEqual(
            result["chat_prompt"],
            "USER IMAGE What color is recorded? Answer only with the color. ASSISTANT",
        )
        self.assertEqual(result["chat_prompt"], processor.inputs["text"][0])
        self.assertEqual(result["prompt_token_count"], 5)
        self.assertEqual(processor.decoded_ids.tolist(), [[11, 12, 100]])
        self.assertEqual(processor.messages, [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What color is recorded? Answer only with the color."},
        ]}])
        self.assertNotIn("Green", processor.inputs["text"][0])
        self.assertEqual(model.observed["max_new_tokens"], 7)
        self.assertIs(model.observed["do_sample"], False)
        self.assertEqual(model.observed["num_beams"], 1)
        self.assertEqual(model.observed["eos_token_id"], [99, 100])
        self.assertTrue(torch.equal(model.observed["mm_token_type_ids"], processor.batch["mm_token_type_ids"]))
        self.assertIs(processor.inputs["do_rescale"], False)
        self.assertFalse(model.was_training_at_generate)
        self.assertFalse(model.grad_enabled_at_generate)
        self.assertTrue(model.training)
        self.assertTrue(model.weight.requires_grad)
        self.assertTrue(result["eos_reached"])
        self.assertFalse(result["truncated"])

    def test_expected_answer_and_choices_are_rejected_by_generation_api(self):
        for name in ("expected_answer", "target", "choices", "answer"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                self.call(**{name: "green"})

    def test_missing_eos_is_retained_and_flagged_at_token_budget(self):
        result = self.call(FakeModel(continuation=(11, 13)), max_new_tokens=2)
        self.assertEqual(result["raw"], "Green extra")
        self.assertEqual(result["generated_token_ids"], [11, 13])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["finish_reason"], "token_limit")

    def test_missing_eos_before_budget_is_conservatively_flagged(self):
        result = self.call(FakeModel(continuation=(11,)), max_new_tokens=2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["finish_reason"], "other_stop")

    def test_incorrect_generation_prefix_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "prefix"):
            self.call(FakeModel(bad_prefix=True))

    def test_model_state_restored_on_failure(self):
        model = FakeModel(fail=True)
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.call(model)
        self.assertTrue(model.training)
        self.assertTrue(torch.is_grad_enabled())

    def test_locked_tensor_preprocessing_and_contract_drift(self):
        processor = FakeProcessor(locked=True)
        image = torch.rand(1, 3, 1024, 1024, requires_grad=True)
        result = generate_short_answer(
            model=FakeModel(), processor=processor, image=image, query="What color?", device="cpu"
        )
        self.assertEqual(result["raw"], "Green.")
        self.assertIs(processor.inputs["do_resize"], False)
        self.assertFalse(processor.inputs["images"][0].requires_grad)
        self.assertEqual(processor.batch["pixel_values"].dtype, torch.float32)
        with self.assertRaisesRegex(RuntimeError, "grid drifted"):
            generate_short_answer(
                model=FakeModel(), processor=FakeProcessor(locked=True, bad_grid=True),
                image=image, query="What color?", device="cpu",
            )

    def test_score_normalizes_case_whitespace_and_trailing_punctuation_only(self):
        for raw in (" green ", "GREEN.", "Green!\n", "green 。  "):
            with self.subTest(raw=raw):
                self.assertTrue(score_short_answer(raw, "green")["strict_correct"])
        self.assertEqual(normalize_short_answer(" NO\n ACTIVE   preference. "), "no active preference")

    def test_unicode_trailing_quotes_and_brackets_removed_but_prefixes_and_internal_signs_preserved(self):
        for raw in ('green"', "green”）", "green ) 。 」\n", "green ’ ] !"):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_short_answer(raw), "green")
                self.assertTrue(score_short_answer(raw, "green")["strict_correct"])
        self.assertEqual(normalize_short_answer('"green"'), '"green')
        self.assertEqual(normalize_short_answer("（green）"), "（green")
        self.assertFalse(score_short_answer('"green"', "green")["strict_correct"])
        self.assertEqual(normalize_short_answer("minus -1.5。"), "minus -1.5")
        self.assertEqual(normalize_short_answer("−1.5。"), "−1.5")
        self.assertEqual(normalize_short_answer("red-green!"), "red-green")
        self.assertEqual(normalize_short_answer("green−"), "green−")  # Math symbol, not category P.

    def test_negations_explanations_substrings_and_synonyms_never_receive_credit(self):
        for raw in ("not green", "green is not correct", "The answer is green.", "green because of the image",
                    "green or blue", "greener", "lime", '"green"', ""):
            with self.subTest(raw=raw):
                self.assertFalse(score_short_answer(raw, "green")["strict_correct"])
        negated = score_short_answer("not green", "green")
        self.assertEqual(negated["normalized"], "not green")
        self.assertTrue(negated["has_extra_text"])
        self.assertEqual(negated["format_status"], "extra_text_or_explanation")
        self.assertFalse(score_short_answer("greener", "green")["expected_text_present"])
        self.assertEqual(score_short_answer("", "green")["format_status"], "empty")

    def test_empty_expected_answer_is_an_error(self):
        with self.assertRaises(ValueError):
            score_short_answer("", " . ")


if __name__ == "__main__":
    unittest.main()
