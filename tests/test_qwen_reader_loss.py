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
from vision_memory.reader import qwen3vl_choice_nll, qwen3vl_target_only_ce  # noqa: E402


class MockBatch(dict):
    def to(self, device):
        return MockBatch({key: value.to(device) for key, value in self.items()})


class MockTokenizer:
    def __call__(self, text, add_special_tokens, return_tensors):
        del text, add_special_tokens, return_tensors
        return {"input_ids": torch.tensor([[2, 3]], dtype=torch.long)}


class MockProcessor:
    tokenizer = MockTokenizer()

    def __init__(self):
        self.observed_do_resize = "not-passed"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        del messages, tokenize, add_generation_prompt
        return "prompt"

    def __call__(self, *, text, images, return_tensors, do_rescale, do_resize="not-passed"):
        del text, return_tensors, do_rescale
        self.observed_do_resize = do_resize
        image = images[0]
        return MockBatch(
            {
                "input_ids": torch.tensor([[0, 1, 1]], dtype=torch.long),
                "attention_mask": torch.ones(1, 3, dtype=torch.long),
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
        self.assertEqual(processor.observed_do_resize, "not-passed")

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


if __name__ == "__main__":
    unittest.main()
