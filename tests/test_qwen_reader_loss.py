from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import (  # noqa: E402
    qwen3vl_choice_nll,
    qwen3vl_listwise_choice_ce,
    qwen3vl_target_only_ce,
)


class MockBatch(dict):
    def to(self, device):
        return MockBatch({key: value.to(device) for key, value in self.items()})


class MockTokenizer:
    def __init__(self):
        self.observed_texts: list[str] = []

    def __call__(self, text, add_special_tokens, return_tensors):
        del add_special_tokens, return_tensors
        self.observed_texts.append(text)
        mapping = {
            "prompt ": [0, 1, 1],
            "prompt answer": [0, 1, 1, 6, 7],
            "answer": [2, 3],
            "prompt a": [0, 1, 1, 2],
            "prompt b": [0, 1, 1, 3],
            "prompt c": [0, 1, 1, 4],
            "prompt d": [0, 1, 1, 5],
            "a": [2],
            "b": [3],
            "c": [4],
            "d": [5],
        }
        ids = mapping.get(text, [2, 3])
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class MockProcessor:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        self.observed_do_resize = "not-passed"
        self.observed_do_resize_calls: list[object] = []
        self.observed_processor_texts: list[str] = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        del messages, tokenize, add_generation_prompt
        return "prompt "

    def __call__(self, *, text, images, return_tensors, do_rescale, do_resize="not-passed"):
        del return_tensors, do_rescale
        self.observed_do_resize = do_resize
        self.observed_do_resize_calls.append(do_resize)
        rendered_text = text[0]
        self.observed_processor_texts.append(rendered_text)
        image = images[0]
        return MockBatch(
            {
                "input_ids": self.tokenizer(
                    rendered_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"],
                "attention_mask": torch.ones(
                    1,
                    len(self.tokenizer(rendered_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]),
                    dtype=torch.long,
                ),
                "pixel_values": image.unsqueeze(0),
                "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        )


class MockBaseModel(nn.Module):
    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        use_cache,
        return_dict,
    ):
        del attention_mask, image_grid_thw, use_cache, return_dict
        hidden = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        hidden = hidden + pixel_values.mean()
        return SimpleNamespace(last_hidden_state=hidden)


class MockQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = MockBaseModel()
        self.lm_head = nn.Linear(4, 8, bias=False)


class ReaderLossContractTest(unittest.TestCase):
    def test_target_only_ce_preserves_image_gradient_and_freezes_parameters(self):
        model = freeze_module(MockQwen())
        image = torch.rand(3, 8, 8, requires_grad=True)
        processor = MockProcessor()
        result = qwen3vl_target_only_ce(
            model=model,
            processor=processor,
            image=image,
            query="question",
            target="answer",
            device=torch.device("cpu"),
        )
        result.loss.backward()

        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(image.grad.norm().item(), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(tuple(result.target_logits.shape[:2]), (1, 2))
        self.assertEqual(result.target_ids.tolist(), [[6, 7]])
        self.assertEqual(processor.observed_processor_texts, ["prompt answer"])
        self.assertNotIn("answer", processor.tokenizer.observed_texts)
        self.assertEqual(processor.observed_do_resize, "not-passed")

    def test_joint_target_tokenization_fails_if_appending_target_retokenizes_prefix(self):
        class RetokenizingTokenizer(MockTokenizer):
            def __call__(self, text, add_special_tokens, return_tensors):
                if text == "prompt ":
                    return {"input_ids": torch.tensor([[0, 1]], dtype=torch.long)}
                if text == "prompt answer":
                    return {"input_ids": torch.tensor([[0, 9, 6]], dtype=torch.long)}
                return super().__call__(text, add_special_tokens, return_tensors)

        model = freeze_module(MockQwen())
        processor = MockProcessor()
        processor.tokenizer = RetokenizingTokenizer()
        image = torch.rand(3, 8, 8, requires_grad=True)

        with self.assertRaisesRegex(RuntimeError, "retokenized the chat-template prefix"):
            qwen3vl_target_only_ce(
                model=model,
                processor=processor,
                image=image,
                query="question",
                target="answer",
                device=torch.device("cpu"),
            )

    def test_target_only_ce_can_explicitly_disable_processor_resize(self):
        model = freeze_module(MockQwen())
        processor = MockProcessor()
        image = torch.rand(3, 8, 8, requires_grad=True)

        result = qwen3vl_target_only_ce(
            model=model,
            processor=processor,
            image=image,
            query="question",
            target="answer",
            device=torch.device("cpu"),
            do_resize=False,
        )

        self.assertEqual(processor.observed_do_resize, False)
        result.loss.backward()
        self.assertIsNotNone(image.grad)

    def test_choice_scorer_propagates_disabled_resize_to_every_choice(self):
        model = freeze_module(MockQwen())
        processor = MockProcessor()
        image = torch.rand(3, 8, 8)

        score = qwen3vl_choice_nll(
            model=model,
            processor=processor,
            image=image,
            query="question",
            choices=("a", "b", "c", "d"),
            device=torch.device("cpu"),
            do_resize=False,
        )

        self.assertEqual(processor.observed_do_resize, False)
        self.assertEqual(len(score.mean_nll), 4)

    def test_deterministic_ce_matches_default_and_preserves_gradient(self):
        torch.manual_seed(17)
        model = freeze_module(MockQwen())
        ordinary_image = torch.rand(3, 8, 8, requires_grad=True)
        deterministic_image = ordinary_image.detach().clone().requires_grad_(True)
        ordinary = qwen3vl_target_only_ce(
            model=model,
            processor=MockProcessor(),
            image=ordinary_image,
            query="question",
            target="answer",
            device=torch.device("cpu"),
        )
        deterministic = qwen3vl_target_only_ce(
            model=model,
            processor=MockProcessor(),
            image=deterministic_image,
            query="question",
            target="answer",
            device=torch.device("cpu"),
            deterministic_ce=True,
        )

        torch.testing.assert_close(deterministic.loss, ordinary.loss, rtol=1e-6, atol=1e-6)
        deterministic.loss.backward()
        self.assertIsNotNone(deterministic_image.grad)
        self.assertGreater(deterministic_image.grad.norm().item(), 0.0)

    def test_listwise_choice_ce_matches_explicit_formula_and_preserves_image_gradient(self):
        torch.manual_seed(19)
        model = freeze_module(MockQwen())
        image = torch.rand(3, 8, 8, requires_grad=True)
        result = qwen3vl_listwise_choice_ce(
            model=model,
            processor=MockProcessor(),
            image=image,
            query="question with A-D options",
            choices=("a", "b", "c", "d"),
            target_index=2,
            device=torch.device("cpu"),
        )

        expected = torch.logsumexp(-result.choice_mean_nll, dim=0) + result.choice_mean_nll[2]
        torch.testing.assert_close(result.loss, expected)
        self.assertEqual(tuple(result.choice_mean_nll.shape), (4,))
        self.assertEqual(result.choice_token_counts, (1, 1, 1, 1))
        self.assertEqual(result.target_ids.tolist(), [[4]])

        result.loss.backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(image.grad.norm().item(), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_deterministic_listwise_ce_matches_default_and_propagates_resize(self):
        torch.manual_seed(23)
        model = freeze_module(MockQwen())
        ordinary_image = torch.rand(3, 8, 8, requires_grad=True)
        deterministic_image = ordinary_image.detach().clone().requires_grad_(True)
        ordinary_processor = MockProcessor()
        deterministic_processor = MockProcessor()
        ordinary = qwen3vl_listwise_choice_ce(
            model=model,
            processor=ordinary_processor,
            image=ordinary_image,
            query="question",
            choices=("a", "b", "c", "d"),
            target_index=1,
            device=torch.device("cpu"),
            do_resize=False,
        )
        deterministic = qwen3vl_listwise_choice_ce(
            model=model,
            processor=deterministic_processor,
            image=deterministic_image,
            query="question",
            choices=("a", "b", "c", "d"),
            target_index=1,
            device=torch.device("cpu"),
            do_resize=False,
            deterministic_ce=True,
        )

        torch.testing.assert_close(deterministic.loss, ordinary.loss, rtol=1e-6, atol=1e-6)
        deterministic.loss.backward()
        self.assertIsNotNone(deterministic_image.grad)
        self.assertGreater(deterministic_image.grad.norm().item(), 0.0)
        self.assertEqual(ordinary_processor.observed_do_resize_calls, [False] * 4)
        self.assertEqual(deterministic_processor.observed_do_resize_calls, [False] * 4)

    def test_listwise_choice_ce_rejects_invalid_choices_and_target_index(self):
        model = freeze_module(MockQwen())
        common = {
            "model": model,
            "processor": MockProcessor(),
            "image": torch.rand(3, 8, 8, requires_grad=True),
            "query": "question",
            "device": torch.device("cpu"),
        }
        with self.assertRaisesRegex(ValueError, "exactly four"):
            qwen3vl_listwise_choice_ce(**common, choices=("a", "b", "c"), target_index=0)
        with self.assertRaisesRegex(ValueError, "distinct"):
            qwen3vl_listwise_choice_ce(**common, choices=("a", "b", "c", "c"), target_index=0)
        with self.assertRaisesRegex(ValueError, "target_index"):
            qwen3vl_listwise_choice_ce(**common, choices=("a", "b", "c", "d"), target_index=4)


if __name__ == "__main__":
    unittest.main()
