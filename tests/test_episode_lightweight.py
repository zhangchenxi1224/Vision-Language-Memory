from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import EventKind, Episode, QuerySpec, Turn, TurnType, run_episode  # noqa: E402
from vision_memory.lightweight import HashChoiceReader, LightweightVisualUpdater  # noqa: E402


def make_episode() -> Episode:
    query = QuerySpec("What is preferred?", ("red", "blue", "green", "yellow"), 0)
    return Episode(
        episode_id="train-0",
        split="train",
        seed=0,
        entity_id="entity",
        template_id="template",
        pair_id="pair",
        counterfactual_episode_id="train-1",
        topic="color",
        turns=(
            Turn(TurnType.EVENT, EventKind.SET, "The preferred color is red."),
            Turn(TurnType.EVENT, EventKind.NOOP, "A clock ticked."),
            Turn(TurnType.MIXED, EventKind.NOOP, "It rained elsewhere.", query),
            Turn(TurnType.QUERY, query=query),
        ),
    )


class LightweightUpdaterTest(unittest.TestCase):
    def make_updater(self):
        return LightweightVisualUpdater(
            state_channels=8,
            state_size=8,
            output_size=16,
            vocabulary_size=64,
            embedding_dim=16,
            text_hidden_dim=8,
        )

    def test_shape_and_episode_gradient_contract(self):
        torch.manual_seed(0)
        updater = self.make_updater()
        reader = HashChoiceReader(feature_size=2)
        output = run_episode(make_episode(), updater=updater, reader=reader)
        self.assertEqual(tuple(output.final_state.shape), (1, 8, 8, 8))
        self.assertEqual(tuple(output.final_image.shape), (1, 3, 16, 16))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        gradients = [parameter.grad for parameter in updater.parameters() if parameter.requires_grad]
        self.assertTrue(any(gradient is not None and gradient.norm().item() > 0 for gradient in gradients))
        self.assertEqual(sum(parameter.numel() for parameter in reader.parameters()), 0)

    def test_default_renderer_keeps_bilinear_production_behavior(self):
        torch.manual_seed(5)
        updater = self.make_updater()
        state = updater.initial_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        head_image = updater.rgb_head(state)
        expected = torch.nn.functional.interpolate(
            head_image,
            size=(16, 16),
            mode="bilinear",
            align_corners=False,
        )

        with mock.patch(
            "vision_memory.lightweight.model.F.interpolate", wraps=torch.nn.functional.interpolate
        ) as resize:
            actual = updater.render(state)

        resize.assert_called_once()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_deterministic_repro_renderer_repeats_and_center_crops_without_interpolate(self):
        torch.manual_seed(7)
        updater = self.make_updater()
        state = updater.initial_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        expected_expanded = updater.rgb_head(state).repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
        expected = expected_expanded[..., 1:15, 1:15]

        with mock.patch("vision_memory.lightweight.model.F.interpolate", side_effect=AssertionError("called")):
            actual = updater.render_deterministic_repro(state, target_size=14)

        self.assertEqual(tuple(actual.shape), (1, 3, 14, 14))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in updater.rgb_head.parameters()
            )
        )

    def test_deterministic_repro_renderer_rejects_non_integer_repeat_or_asymmetric_crop(self):
        updater = self.make_updater()
        state = updater.initial_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "even centered crop"):
            updater.render_deterministic_repro(state, target_size=15)

        updater.output_size = 15
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            updater.render_deterministic_repro(state, target_size=13)

    def test_formal_deterministic_renderer_contract_is_64_repeat_to_256_without_crop(self):
        torch.manual_seed(9)
        updater = LightweightVisualUpdater(
            state_channels=2,
            state_size=64,
            output_size=256,
            vocabulary_size=64,
            embedding_dim=8,
            text_hidden_dim=4,
        )
        state = updater.initial_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        head = updater.rgb_head(state)
        expanded = head.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)

        image = updater.render_deterministic_repro(state)

        self.assertEqual(tuple(head.shape), (1, 3, 64, 64))
        self.assertEqual(tuple(image.shape), (1, 3, 256, 256))
        torch.testing.assert_close(image, expanded, rtol=0, atol=0)

    def test_hashed_encoder_is_deterministic(self):
        updater = self.make_updater()
        first, lengths_a = updater.event_encoder.tokenize(["Remember red."], device=torch.device("cpu"))
        second, lengths_b = updater.event_encoder.tokenize(["Remember red."], device=torch.device("cpu"))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(lengths_a, lengths_b)

    def test_fixed_spatial_basis_is_shape_checked_deterministic_and_orthogonal(self):
        torch.manual_seed(11)
        first = self.make_updater()
        torch.manual_seed(97)
        second = self.make_updater()

        self.assertEqual(tuple(first.spatial_basis.shape), (16, 8, 8))
        torch.testing.assert_close(first.spatial_basis, second.spatial_basis, rtol=0, atol=0)
        flattened = first.spatial_basis.flatten(1)
        gram = flattened @ flattened.T / (8 * 8)
        torch.testing.assert_close(gram, torch.eye(16), rtol=1e-5, atol=1e-5)
        self.assertNotIsInstance(first.initial_hidden, torch.nn.Parameter)
        self.assertEqual(torch.count_nonzero(first.initial_hidden).item(), 0)

        first.spatial_basis = torch.zeros(15, 8, 8)
        state = first.initial_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "spatial_basis shape"):
            first.update(state, "Remember red.")

    def test_event_creates_central_spatial_variance_and_backpropagates(self):
        torch.manual_seed(23)
        updater = self.make_updater()
        state = updater.initial_state(batch_size=1, device=torch.device("cpu"), dtype=torch.float32)
        image = updater.render(updater.update(state, "The preferred color is red."))
        central_image = image[..., 4:12, 4:12]
        spatial_variance = central_image.var(dim=(-2, -1), unbiased=False).mean()

        self.assertTrue(torch.isfinite(spatial_variance))
        self.assertGreater(spatial_variance.item(), 0.0)
        spatial_variance.backward()

        encoder_gradients = [
            parameter.grad for parameter in updater.event_encoder.parameters() if parameter.requires_grad
        ]
        writer_gradient = updater.event_spatial_projection.weight.grad
        self.assertTrue(any(gradient is not None and gradient.norm().item() > 0 for gradient in encoder_gradients))
        self.assertIsNotNone(writer_gradient)
        self.assertGreater(writer_gradient.norm().item(), 0.0)

    def test_event_conditioned_spatial_maps_are_not_channelwise_rank_one(self):
        torch.manual_seed(29)
        updater = self.make_updater()
        captured_inputs = []

        def capture_cell_inputs(_module, inputs):
            captured_inputs.append(inputs[0].detach())

        handle = updater.cell.register_forward_pre_hook(capture_cell_inputs)
        try:
            state = updater.initial_state(batch_size=2, device=torch.device("cpu"), dtype=torch.float32)
            updater.update(
                state,
                ["The preferred color is red.", "The preferred destination is Kyoto."],
            )
        finally:
            handle.remove()

        self.assertEqual(tuple(updater.event_spatial_projection.weight.shape), (8 * 16, 16))
        self.assertEqual(len(captured_inputs), 1)
        event_maps = captured_inputs[0]
        centered = event_maps - event_maps.mean(dim=(-2, -1), keepdim=True)
        similarities = torch.nn.functional.cosine_similarity(
            centered[0].flatten(1),
            centered[1].flatten(1),
            dim=1,
        ).abs()
        self.assertGreater((similarities < 0.99).float().mean().item(), 0.5)

    def test_update_gate_bias_is_initialized_to_negative_one(self):
        torch.manual_seed(31)
        updater = self.make_updater()
        reset_bias, update_bias = updater.cell.gates.bias.chunk(2)

        torch.testing.assert_close(update_bias, torch.full_like(update_bias, -1.0), rtol=0, atol=0)
        self.assertFalse(torch.equal(reset_bias, torch.full_like(reset_bias, -1.0)))

    def test_learned_initial_state_is_tanh_parameterized_but_fixed_zero_path_is_unchanged(self):
        torch.manual_seed(33)
        learned = LightweightVisualUpdater(
            state_channels=8,
            state_size=8,
            output_size=16,
            vocabulary_size=64,
            embedding_dim=16,
            text_hidden_dim=8,
            learned_initial_state=True,
        )
        with torch.no_grad():
            learned.initial_hidden.fill_(2.0)
        learned_state = learned.initial_state(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertIsInstance(learned.initial_hidden, torch.nn.Parameter)
        torch.testing.assert_close(learned_state, torch.full_like(learned_state, torch.tanh(torch.tensor(2.0))))
        self.assertLess(learned_state.abs().max().item(), 1.0)
        learned_state.sum().backward()
        self.assertIsNotNone(learned.initial_hidden.grad)
        self.assertGreater(learned.initial_hidden.grad.norm().item(), 0.0)

        fixed = self.make_updater()
        fixed_state = fixed.initial_state(
            batch_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertNotIsInstance(fixed.initial_hidden, torch.nn.Parameter)
        self.assertTrue(torch.equal(fixed_state, torch.zeros_like(fixed_state)))

    def test_cell_event_input_is_bounded_and_initial_gates_are_not_saturated(self):
        torch.manual_seed(37)
        updater = self.make_updater()
        captured_inputs = []
        captured_gate_logits = []

        def capture_cell_inputs(_module, inputs):
            captured_inputs.append(inputs[0].detach())

        def capture_gate_logits(_module, _inputs, output):
            captured_gate_logits.append(output.detach())

        cell_handle = updater.cell.register_forward_pre_hook(capture_cell_inputs)
        gate_handle = updater.cell.gates.register_forward_hook(capture_gate_logits)
        try:
            state = updater.initial_state(batch_size=4, device=torch.device("cpu"), dtype=torch.float32)
            updater.update(
                state,
                [
                    "The preferred color is red.",
                    "The preferred destination is Kyoto.",
                    "Clear the saved restaurant preference.",
                    "A clock ticked in another room.",
                ],
            )
        finally:
            cell_handle.remove()
            gate_handle.remove()

        self.assertEqual(len(captured_inputs), 1)
        event_map = captured_inputs[0]
        self.assertTrue(torch.isfinite(event_map).all())
        self.assertLessEqual(event_map.abs().max().item(), 1.0)

        self.assertEqual(len(captured_gate_logits), 1)
        gate_values = torch.sigmoid(captured_gate_logits[0])
        saturation_ratio = ((gate_values < 0.01) | (gate_values > 0.99)).float().mean().item()
        self.assertLess(saturation_ratio, 0.01)

    def test_random_multi_turn_states_remain_finite_and_strictly_bounded(self):
        torch.manual_seed(41)
        updater = self.make_updater()
        state = updater.initial_state(batch_size=3, device=torch.device("cpu"), dtype=torch.float32)
        events = (
            ["Remember red.", "Remember Kyoto.", "Remember jazz."],
            ["Overwrite with blue.", "Overwrite with Oslo.", "Overwrite with folk."],
            ["A bird flew past.", "A train arrived.", "A clock ticked."],
            ["Clear the color.", "Clear the city.", "Clear the music."],
        )

        for _ in range(8):
            for batch_events in events:
                state = updater.update(state, batch_events)
                self.assertTrue(torch.isfinite(state).all())
                self.assertLess(state.abs().max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
