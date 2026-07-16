from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probes.qwen_renderer_control_upper_bound import (  # noqa: E402
    CONTROL_CODE_COUNT,
    DIAGNOSTIC_SCOPE,
    DISCLAIMER,
    TARGET_SUPERVISED_LABEL_LEAK,
    TargetSelectedRendererControl,
    evaluate_renderer_control,
    optimize_target_selected_image,
)
from scripts.data.qwen_sanity import QueryMember, UniqueQuery  # noqa: E402
from vision_memory.data import QuerySpec  # noqa: E402
from vision_memory.lightweight import LightweightVisualUpdater  # noqa: E402
from vision_memory.reader import ChoiceScoreOutput  # noqa: E402


def make_diagnostic() -> TargetSelectedRendererControl:
    torch.manual_seed(0)
    renderer = LightweightVisualUpdater(
        state_channels=8,
        state_size=8,
        output_size=16,
        vocabulary_size=64,
        embedding_dim=16,
        text_hidden_dim=8,
    )
    return TargetSelectedRendererControl(renderer)


def make_query(target_index: int) -> UniqueQuery:
    choices = ("red", "blue", "green", "yellow")
    return UniqueQuery(
        comparison_id=f"comparison-{target_index}",
        query=QuerySpec(
            text="Which option is encoded?",
            choices=choices,
            target_index=target_index,
            comparison_id=f"comparison-{target_index}",
        ),
        query_ordinal=0,
        topic="color",
        entity_id="entity",
        template_id="template-pattern-0-a",
        template_family="test",
        members=[
            QueryMember(
                episode_id=f"episode-{target_index}",
                query_id=f"episode-{target_index}:q0",
                pair_id=f"pair-{target_index}",
                counterfactual_episode_id=f"counterfactual-{target_index}",
                distractor_pair_id=None,
                distractor_episode_id=None,
                distractor_variant=None,
                turn_index=1,
                turn_type="query",
            )
        ],
    )


class RendererControlUpperBoundTest(unittest.TestCase):
    def test_target_index_selects_exactly_one_of_four_hidden_codes(self):
        diagnostic = make_diagnostic()
        with torch.no_grad():
            for index in range(CONTROL_CODE_COUNT):
                diagnostic.state_codes[index].fill_(float(index))

        selected = diagnostic.select_codes(torch.tensor([3, 0, 2], dtype=torch.long))
        self.assertEqual(tuple(diagnostic.state_codes.shape), (4, 8, 8, 8))
        self.assertEqual(tuple(selected.shape), (3, 8, 8, 8))
        self.assertTrue(torch.equal(selected[0], diagnostic.state_codes[3]))
        self.assertTrue(torch.equal(selected[1], diagnostic.state_codes[0]))
        self.assertTrue(torch.equal(selected[2], diagnostic.state_codes[2]))
        with self.assertRaisesRegex(ValueError, "lie in"):
            diagnostic.select_codes(4)
        with self.assertRaisesRegex(ValueError, "torch.long"):
            diagnostic.select_codes(torch.tensor([0.0]))

    def test_only_state_codes_and_rgb_head_are_trainable_and_receive_gradients(self):
        diagnostic = make_diagnostic()
        expected = {"state_codes"} | {
            f"renderer.rgb_head.{name}" for name, _parameter in diagnostic.renderer.rgb_head.named_parameters()
        }
        self.assertEqual(diagnostic.trainable_parameter_names(), expected)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in diagnostic.renderer.event_encoder.parameters())
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in diagnostic.renderer.cell.parameters()))

        optimizer = torch.optim.SGD(
            [parameter for parameter in diagnostic.parameters() if parameter.requires_grad],
            lr=0.01,
        )
        result = optimize_target_selected_image(
            diagnostic,
            target_index=2,
            optimizer=optimizer,
            loss_from_image=lambda image: image.square().mean(),
        )
        self.assertGreater(result.state_code_gradient_norm, 0)
        self.assertGreater(result.rgb_head_gradient_norm, 0)
        self.assertTrue(torch.count_nonzero(diagnostic.state_codes.grad[2]).item() > 0)
        self.assertEqual(torch.count_nonzero(diagnostic.state_codes.grad[:2]).item(), 0)
        self.assertEqual(torch.count_nonzero(diagnostic.state_codes.grad[3:]).item(), 0)
        frozen_gradients = [
            parameter.grad
            for name, parameter in diagnostic.named_parameters()
            if name not in diagnostic.allowed_trainable_parameter_names()
        ]
        self.assertTrue(all(gradient is None for gradient in frozen_gradients))
        self.assertEqual(diagnostic.forbidden_call_counts(), {"event_encoder": 0, "convgru": 0})

    def test_render_path_does_not_call_event_encoder_or_convgru(self):
        diagnostic = make_diagnostic()
        image = diagnostic(1)
        self.assertEqual(tuple(image.shape), (1, 3, 16, 16))
        self.assertEqual(diagnostic.forbidden_call_counts(), {"event_encoder": 0, "convgru": 0})
        with self.assertRaisesRegex(RuntimeError, "event_encoder"):
            diagnostic.renderer.event_encoder(["must not be called"])
        self.assertEqual(diagnostic.forbidden_call_counts()["event_encoder"], 1)

    def test_mock_choice_reader_uses_four_choice_mean_nll(self):
        diagnostic = make_diagnostic()
        queries = [make_query(index) for index in range(CONTROL_CODE_COUNT)]
        seen_images: list[torch.Tensor] = []

        def mock_choice_reader(item: UniqueQuery, image: torch.Tensor) -> ChoiceScoreOutput:
            seen_images.append(image)
            target = item.query.target_index
            scores = tuple(0.0 if index == target else 1.0 for index in range(CONTROL_CODE_COUNT))
            return ChoiceScoreOutput(mean_nll=scores, predicted_index=target)

        accuracy, records = evaluate_renderer_control(
            diagnostic,
            queries,
            choice_scorer=mock_choice_reader,
        )
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(len(seen_images), CONTROL_CODE_COUNT)
        self.assertEqual(len(records), CONTROL_CODE_COUNT)
        self.assertTrue(all(len(record["choice_mean_nll"]) == 4 for record in records))
        self.assertTrue(all(record["target_supervised_label_leak"] for record in records))

    def test_disclaimer_and_scope_forbid_method_claims(self):
        self.assertTrue(TARGET_SUPERVISED_LABEL_LEAK)
        self.assertEqual(DIAGNOSTIC_SCOPE, "renderer_manifold_only")
        self.assertIn("TARGET-SUPERVISED LABEL-LEAK DIAGNOSTIC ONLY", DISCLAIMER)
        self.assertIn("only probes the renderer manifold", DISCLAIMER)
        self.assertIn("not a method, baseline, ablation", DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
