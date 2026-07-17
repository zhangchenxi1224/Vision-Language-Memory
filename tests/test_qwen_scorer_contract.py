from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probes import qwen_scorer_contract  # noqa: E402
from scripts.probes.qwen_scorer_contract import (  # noqa: E402
    NLL_ATOL,
    NLL_RTOL,
    audit_joint_chat_tokenization,
    compare_nll_vectors,
    contract_exit_code,
    dihedral_choice_views,
    run_scorer_contract,
    validate_choice_view_mappings,
)
from vision_memory.reader import R3_QWEN_READER_RESIZE_CONTRACT  # noqa: E402


class CharacterTokenizer:
    def __call__(self, text, add_special_tokens, return_tensors):
        del add_special_tokens, return_tensors
        return {"input_ids": torch.tensor([[ord(character) for character in text]], dtype=torch.long)}


class FakeProcessor:
    tokenizer = CharacterTokenizer()

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        del messages, tokenize, add_generation_prompt
        return "<assistant>\n"


class FakeScorers:
    def __init__(self, *, repeat_drift: float = 0.0):
        self.eval_calls = 0
        self.repeat_drift = repeat_drift
        self.score_by_choice = {"alpha": 0.5, "beta": 1.0, "gamma": 1.5, "delta": 2.0}

    def listwise(self, **kwargs):
        scores = torch.tensor(
            [self.score_by_choice[choice] for choice in kwargs["choices"]],
            dtype=torch.float32,
            requires_grad=True,
        )
        target = kwargs["choices"][kwargs["target_index"]]
        return SimpleNamespace(
            loss=scores.sum(),
            choice_mean_nll=scores,
            target_ids=torch.tensor([[ord(character) for character in target]], dtype=torch.long),
            choice_token_counts=tuple(len(choice) for choice in kwargs["choices"]),
        )

    def evaluate(self, **kwargs):
        call_within_view = self.eval_calls % 2
        self.eval_calls += 1
        drift = self.repeat_drift if call_within_view == 1 else 0.0
        scores = tuple(self.score_by_choice[choice] + drift for choice in kwargs["choices"])
        return SimpleNamespace(mean_nll=scores, predicted_index=min(range(4), key=scores.__getitem__))


class QwenScorerContractProbeTest(unittest.TestCase):
    def test_dihedral_views_cover_four_cyclic_and_four_reverse_target_mappings(self):
        choices = ("alpha", "beta", "gamma", "delta")
        views = dihedral_choice_views(choices, 2)
        validation = validate_choice_view_mappings(views, canonical_target="gamma")

        self.assertTrue(validation["passed"])
        self.assertEqual([view.name for view in views[:4]], [f"cyclic-{index}" for index in range(4)])
        self.assertEqual(
            [view.name for view in views[4:]],
            [f"reverse-cyclic-{index}" for index in range(4)],
        )
        self.assertEqual([view.target_index for view in views[:4]], [2, 1, 0, 3])
        self.assertEqual([view.target_index for view in views[4:]], [2, 3, 0, 1])
        self.assertTrue(all(view.choices[view.target_index] == "gamma" for view in views))
        for canonical_target_index, canonical_target in enumerate(choices):
            target_views = dihedral_choice_views(choices, canonical_target_index)
            self.assertTrue(validate_choice_view_mappings(target_views, canonical_target=canonical_target)["passed"])

    def test_joint_chat_audit_uses_contextual_suffix_and_rejects_prefix_retokenization(self):
        audit = audit_joint_chat_tokenization(
            processor=FakeProcessor(),
            query="question",
            choices=("alpha", "beta", "gamma", "delta"),
        )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["choices"][1]["joint_suffix_token_ids"], [ord(value) for value in "beta"])

        class RetokenizingTokenizer(CharacterTokenizer):
            def __call__(self, text, add_special_tokens, return_tensors):
                encoded = super().__call__(text, add_special_tokens, return_tensors)
                if text.startswith("<assistant>\n") and text != "<assistant>\n":
                    encoded["input_ids"][0, 0] += 1
                return encoded

        processor = FakeProcessor()
        processor.tokenizer = RetokenizingTokenizer()
        failed = audit_joint_chat_tokenization(
            processor=processor,
            query="question",
            choices=("alpha", "beta", "gamma", "delta"),
        )
        self.assertFalse(failed["passed"])

    def test_nll_comparison_uses_locked_train_eval_tolerance(self):
        passed = compare_nll_vectors(
            (1.0, 2.0, 3.0, 4.0),
            (1.0 + 1e-7, 2.0, 3.0, 4.0),
            rtol=NLL_RTOL,
            atol=NLL_ATOL,
        )
        failed = compare_nll_vectors(
            (1.0, 2.0, 3.0, 4.0),
            (1.001, 2.0, 3.0, 4.0),
            rtol=NLL_RTOL,
            atol=NLL_ATOL,
        )

        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])

        nonfinite = compare_nll_vectors(
            (float("nan"), 2.0, 3.0, 4.0),
            (1.0, 2.0, 3.0, 4.0),
            rtol=NLL_RTOL,
            atol=NLL_ATOL,
        )
        self.assertFalse(nonfinite["passed"])
        self.assertIsNone(nonfinite["maximum_absolute_difference"])

    def test_full_contract_passes_all_eight_views_and_repeat_drift_fails_closed(self):
        common = {
            "model": object(),
            "processor": FakeProcessor(),
            "image": torch.ones(3, 2, 2, requires_grad=True),
            "query": "Which?",
            "choices": ("alpha", "beta", "gamma", "delta"),
            "target_index": 1,
            "device": torch.device("cpu"),
        }
        passing_scorers = FakeScorers()
        passed = run_scorer_contract(
            **common,
            listwise_scorer=passing_scorers.listwise,
            eval_scorer=passing_scorers.evaluate,
        )
        drifting_scorers = FakeScorers(repeat_drift=1e-5)
        failed = run_scorer_contract(
            **common,
            listwise_scorer=drifting_scorers.listwise,
            eval_scorer=drifting_scorers.evaluate,
        )

        self.assertTrue(passed["passed"])
        self.assertEqual(
            passed["contract"]["reader_resize_contract"],
            R3_QWEN_READER_RESIZE_CONTRACT,
        )
        self.assertEqual(passed["summary"]["views_passed"], 8)
        self.assertEqual(passed["summary"]["joint_tokenization_views_passed"], 8)
        self.assertEqual(passed["summary"]["train_eval_views_passed"], 8)
        self.assertEqual(passed["summary"]["repeat_eval_views_passed"], 8)
        self.assertEqual(contract_exit_code(passed), 0)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["summary"]["repeat_eval_views_passed"], 0)
        self.assertEqual(contract_exit_code(failed), 1)

    def test_main_emits_failure_json_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with mock.patch.object(qwen_scorer_contract.torch.cuda, "is_available", return_value=False):
                with redirect_stdout(io.StringIO()):
                    status = qwen_scorer_contract.main(["--output-json", str(output)])
            loaded = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 1)
        self.assertFalse(loaded["passed"])
        self.assertEqual(loaded["error"]["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
