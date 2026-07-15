from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import (  # noqa: E402
    event_seed,
    load_training_checkpoint,
    run_episode,
    save_training_checkpoint,
)
from vision_memory.data import Episode, EventKind, QuerySpec, Turn, TurnType  # noqa: E402
from vision_memory.dreamlite.recurrent import _encoded_latent  # noqa: E402


class ScalarUpdater(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))
        self.calls: list[tuple[str, str, str | int]] = []

    def forward(self, state, event_text, episode_id, turn_id):
        self.calls.append((event_text, episode_id, turn_id))
        return state * self.gain + float(len(event_text))


def reader_loss(image, query, target):
    del query
    token_count = 2 if target == "red" else 1
    return SimpleNamespace(loss=image.mean(), target_ids=torch.ones(1, token_count, dtype=torch.long))


class EpisodeRunnerTest(unittest.TestCase):
    def episode(self):
        return {
            "episode_id": "ep-1",
            "hidden_ledger": {"must": "not leak"},
            "turns": [
                {"turn_id": 0, "kind": "event", "event_text": "set red", "transition": "set"},
                {
                    "turn_id": 1,
                    "kind": "query",
                    "query_text": "Which color?",
                    "choices": ["blue", "red", "green", "yellow"],
                    "target_index": 1,
                },
                {
                    "turn_id": 2,
                    "kind": "mixed",
                    "event_text": "room wood",
                    "query_text": "Which color?",
                    "choices": ["red", "blue", "green", "yellow"],
                    "target_index": 0,
                },
            ],
        }

    def test_oracle_routes_query_read_only_and_mixed_update_then_read(self):
        updater = ScalarUpdater()
        result = run_episode(
            episode=self.episode(),
            initial_state=torch.zeros(1, 1, 2, 2),
            update_fn=updater,
            decode_fn=lambda state: state,
            reader_loss_fn=reader_loss,
        )
        self.assertEqual([call[0] for call in updater.calls], ["set red", "room wood"])
        self.assertEqual(result.route_trace, ("0:update", "1:read", "2:update", "2:read"))
        self.assertEqual(result.query_count, 2)
        self.assertEqual(result.target_token_count, 4)

    def test_detach_control_breaks_first_state_gradient(self):
        episode = {
            "episode_id": "grad",
            "turns": [
                {"kind": "event", "event_text": "a"},
                {"kind": "event", "event_text": "b"},
                {"kind": "query", "query_text": "q", "target_text": "red"},
            ],
        }
        for detach, expect_gradient in ((False, True), (True, False)):
            updater = ScalarUpdater()
            result = run_episode(
                episode=episode,
                initial_state=torch.ones(1, 1, 2, 2, requires_grad=True),
                update_fn=updater,
                decode_fn=lambda state: state,
                reader_loss_fn=reader_loss,
                detach_between_events=detach,
            )
            result.states[0].retain_grad()
            result.loss.backward()
            self.assertEqual(result.states[0].grad is not None, expect_gradient)

    def test_decode_reencode_uses_separate_unclamped_decode(self):
        updater = ScalarUpdater()
        calls: list[str] = []
        episode = {
            "episode_id": "bottleneck",
            "turns": [
                {"kind": "event", "event_text": "a"},
                {"kind": "query", "query_text": "q", "target_text": "x"},
            ],
        }
        run_episode(
            episode=episode,
            initial_state=torch.zeros(1, 1, 2, 2),
            update_fn=updater,
            decode_fn=lambda state: calls.append("reader") or state,
            reencode_decode_fn=lambda state: calls.append("bottleneck") or state,
            reencode_fn=lambda image: calls.append("encode") or image,
            reader_loss_fn=reader_loss,
            recurrence_mode="decode_reencode",
        )
        self.assertEqual(calls, ["bottleneck", "encode", "reader"])

    def test_canonical_episode_schema_is_consumed_without_flattening_labels(self):
        episode = Episode(
            episode_id="canonical",
            split="train",
            seed=0,
            entity_id="entity",
            template_id="template",
            pair_id="pair",
            counterfactual_episode_id="other",
            topic="color",
            turns=(
                Turn(TurnType.EVENT, EventKind.SET, "remember red"),
                Turn(TurnType.EVENT, EventKind.NOOP, "unrelated weather"),
                Turn(TurnType.QUERY, query=QuerySpec("Which?", ("red", "blue", "green", "yellow"), 0)),
                Turn(TurnType.MIXED, EventKind.OVERWRITE, "now blue", QuerySpec("Which?", ("red", "blue", "green", "yellow"), 1)),
            ),
        )
        updater = ScalarUpdater()
        output = run_episode(
            episode=episode,
            initial_state=torch.zeros(1, 1, 2, 2),
            update_fn=updater,
            decode_fn=lambda state: state,
            reader_loss_fn=reader_loss,
        )
        self.assertEqual([call[0] for call in updater.calls], ["remember red", "unrelated weather", "now blue"])
        self.assertEqual(output.query_count, 2)


class ReproducibilityTest(unittest.TestCase):
    def test_autoencoder_tiny_output_latents_are_supported(self):
        expected = torch.randn(1, 4, 8, 8)
        self.assertIs(_encoded_latent(SimpleNamespace(latents=expected)), expected)

    def test_event_seed_is_stable_and_turn_specific(self):
        first = event_seed(7, "episode", 3)
        self.assertEqual(first, event_seed(7, "episode", 3))
        self.assertNotEqual(first, event_seed(7, "episode", 4))

    def test_checkpoint_contains_only_trainable_weights_and_restores(self):
        model = nn.Linear(2, 1)
        model.bias.requires_grad_(False)
        optimizer = torch.optim.AdamW([model.weight], lr=1e-3)
        original = model.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_training_checkpoint(
                path,
                trainable_module=model,
                optimizer=optimizer,
                epoch=1,
                episode_cursor=2,
                optimizer_step=3,
                manifest={"run": "test"},
                trainer_state={"best_dev": 1.5, "stale_evals": 2},
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(set(payload["trainable_state"]), {"weight"})
            with torch.no_grad():
                model.weight.add_(10)
            restored = load_training_checkpoint(
                path,
                trainable_module=model,
                optimizer=optimizer,
                expected_manifest={"run": "test"},
            )
            torch.testing.assert_close(model.weight, original)
            self.assertEqual(restored["episode_cursor"], 2)
            self.assertEqual(restored["trainer_state"], {"best_dev": 1.5, "stale_evals": 2})


if __name__ == "__main__":
    unittest.main()
