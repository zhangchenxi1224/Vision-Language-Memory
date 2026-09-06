from __future__ import annotations

import copy
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.experiments import run_r11_mcq_open_multistart as pilot
from scripts.experiments import run_r11_open_answer_replay as replay
from scripts.train.latent_r11_vae_oracle import _target_phase
from vision_memory.data import CYCLIC4, REVERSE_CYCLIC4
from vision_memory.reader.open_answer import score_short_answer
from vision_memory.repro import canonical_tensor_sha256


class MultistartTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        replay_config = replay.load_config(root / "configs/experiments/r11_open_answer_replay.json")
        cls.config = pilot.load_config(root / "configs/experiments/r11_mcq_open_multistart.json", replay_config)

    def test_paired_initials_are_exactly_shared_with_independent_rng(self):
        reference = torch.arange(1, 513, dtype=torch.float32).reshape(1, 2, 16, 16) / 100
        torch.manual_seed(100)
        before = torch.get_rng_state().clone()
        item = pilot.make_initial_latent(reference, seed=3, rho=0.1)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        left, right = item["latent_fp32"].clone(), item["latent_fp32"].clone()
        self.assertTrue(torch.equal(left, right))
        self.assertEqual(canonical_tensor_sha256(left), canonical_tensor_sha256(right))
        left.add_(1)
        self.assertTrue(torch.equal(right, item["latent_fp32"]))
        repeated = pilot.make_initial_latent(reference, seed=3, rho=0.1)
        self.assertTrue(torch.equal(right, repeated["latent_fp32"]))

    def test_noise_direction_rms_and_radius_follow_fp64_preregistration(self):
        reference = torch.linspace(-2, 3, 1024).reshape(1, 4, 16, 16)
        item = pilot.make_initial_latent(reference, seed=0, rho=0.1)
        self.assertEqual(item["raw_noise_fp32"].dtype, torch.float32)
        self.assertEqual(item["epsilon_fp64"].dtype, torch.float64)
        self.assertAlmostEqual(float(item["epsilon_fp64"].square().mean().sqrt()), 1.0, places=14)
        expected = (reference.double() + 0.1 * reference.double().square().mean().sqrt()
                    * item["epsilon_fp64"]).float()
        self.assertTrue(torch.equal(item["latent_fp32"], expected))
        measured = (item["latent_fp32"].double() - reference.double()).square().mean().sqrt()
        self.assertAlmostEqual(float(measured / reference.double().square().mean().sqrt()), 0.1, places=7)
        other = pilot.make_initial_latent(reference, seed=1, rho=0.1)
        self.assertNotEqual(item["latent_sha256"], other["latent_sha256"])
        blank = pilot.make_initial_latent(reference, seed=None, rho=0.0)
        self.assertTrue(torch.equal(blank["latent_fp32"], reference))

    def test_original_choice_phase_covers_all_views_and_open_prompt_has_no_options(self):
        target = self.config["target"]
        counts = [0, 0, 0, 0]
        for step in range(256):
            case = pilot.training_case(target, "mcq", step)
            view = (step + _target_phase(target["segment_id"])) % 4
            self.assertEqual(case["permutation"], list(CYCLIC4[view]))
            self.assertEqual(case["choices"][case["target_index"]], target["scorer_metadata"]["gold"])
            counts[view] += 1
        self.assertEqual(counts, [64] * 4)
        case = pilot.training_case(target, "open", 17)
        self.assertEqual(case["query"], target["inputs"]["original_open"])
        self.assertEqual(case["target"], "ambient")
        self.assertNotIn("choices", case)
        self.assertNotIn("ambient", case["query"].casefold())
        self.assertNotIn("Choose exactly one option", case["query"])
        self.assertNotIn("A.", case["query"])
        self.assertNotIn("<|im_end|>", case["target"])

    def test_rng_preservation_restores_python_numpy_torch_even_on_failure(self):
        random.seed(4)
        np.random.seed(5)
        torch.manual_seed(6)
        before = pilot.capture_rng()
        expected = (random.random(), float(np.random.random()), torch.rand(5))
        pilot.restore_rng(before)
        with self.assertRaisesRegex(RuntimeError, "probe"):
            with pilot.preserve_rng():
                random.random()
                np.random.random()
                torch.rand(19)
                raise RuntimeError("probe failed")
        actual = (random.random(), float(np.random.random()), torch.rand(5))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rng.pt"
            pilot.save_tensor_payload(path, {"rng": before})
            loaded = torch.load(path, weights_only=True)["rng"]
            pilot.restore_rng(loaded)
            self.assertEqual(random.random(), expected[0])

    def full_records(self):
        runs, generations, mcq = [], [], []
        gold = self.config["target"]["scorer_metadata"]["gold"]
        for spec in self.config["training"]["run_order"]:
            runs.append({**spec, "status": "completed", "optimizer_steps": 256, "latent_count": 257,
                         "probe_count": 25, "checkpoint_count": 5, "generation_count": 6, "mcq_count": 4,
                         "initial_latent_sha256": "same-init-" + spec["init_id"]})
            for condition in pilot.CONDITIONS:
                for prompt in pilot.PROMPTS:
                    # New open arm succeeds; original MCQ arm fails free generation.
                    raw = gold if spec["arm"] == "open" and condition == "matched" else "not " + gold
                    generations.append({**spec, "optimizer_step": 256, "condition": condition,
                                        "prompt_id": prompt, "query": self.config["target"]["inputs"][prompt],
                                        "raw": raw, "scorer": score_short_answer(raw, gold)})
            for view, permutation in enumerate(REVERSE_CYCLIC4):
                mcq.append({**spec, "optimizer_step": 256, "view_index": view, "permutation": list(permutation),
                            "correct": True, "ce": 0.1})
        return runs, generations, mcq

    def test_complete_aggregation_is_paired_and_blank_separate(self):
        runs, generations, mcq = self.full_records()
        summary = pilot.aggregate_results(self.config, runs, generations, mcq)
        self.assertEqual(summary["random_start_count"], 8)
        self.assertEqual(len(summary["paired_by_seed"]), 8)
        self.assertEqual(summary["blank_separate"]["init_id"], "blank")
        self.assertEqual(summary["by_arm"]["open"]["open_original"], 1.0)
        self.assertEqual(summary["by_arm"]["mcq"]["open_original"], 0.0)
        self.assertEqual(summary["paired_open_minus_mcq_accuracy"]["open_original"], 1.0)

    def test_incomplete_duplicate_or_mismatched_initialization_cannot_aggregate(self):
        runs, generations, mcq = self.full_records()
        cases = [(runs[:-1], generations, mcq), (runs, generations[:-1], mcq), (runs, generations, mcq[:-1]),
                 (runs, generations[:-1] + generations[:1], mcq)]
        for records in cases:
            with self.subTest(sizes=[len(items) for items in records]), self.assertRaises(ValueError):
                pilot.aggregate_results(self.config, *records)
        bad = copy.deepcopy(runs)
        bad[1]["initial_latent_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "identical"):
            pilot.aggregate_results(self.config, bad, generations, mcq)
        bad = copy.deepcopy(runs)
        bad[0]["optimizer_steps"] = 255
        with self.assertRaisesRegex(ValueError, "updates"):
            pilot.aggregate_results(self.config, bad, generations, mcq)
        bad_generations = copy.deepcopy(generations)
        bad_generations[0]["scorer"]["strict_correct"] = True
        with self.assertRaisesRegex(ValueError, "score drifted"):
            pilot.aggregate_results(self.config, runs, bad_generations, mcq)

    def test_geometry_retains_zero_delta_undefined_cosines_and_absolute_spread(self):
        vectors = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        result = pilot.pairwise_geometry(vectors)
        self.assertEqual(result["pair_count"], 3)
        self.assertEqual(result["independent_start_count"], 3)
        self.assertIsNone(result["cosine_matrix"][0][1])
        self.assertAlmostEqual(result["mean_pairwise_squared_rmse"], 8 / 3)
        self.assertAlmostEqual(result["rmse_matrix"][1][2], 2.0)
        zero = pilot.pairwise_geometry(torch.zeros_like(vectors))
        self.assertEqual(zero["mean_pairwise_squared_rmse"], 0.0)
        self.assertIsNone(zero["mean_pairwise_cosine"])
        distance = pilot.vector_geometry(torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 0.0]))
        self.assertEqual(distance["cosine"], -1.0)
        self.assertEqual(distance["l2"], 2.0)

    def test_zero_gradient_after_connected_preflight_still_executes_all_256_adam_steps(self):
        class FakeOracle(torch.nn.Module):
            def __init__(self, *, vae, initial_latent, compute_dtype):
                super().__init__()
                self.vae = vae
                self.latent_fp32 = torch.nn.Parameter(initial_latent.clone())

            def image(self):
                return self.latent_fp32[:, :3]

        calls = []

        def fake_loss(**kwargs):
            calls.append(kwargs["arm"])
            image = kwargs["image"]
            # First two calls verify the live gradient; all subsequent gradients
            # are finite zero, which must be logged without ending optimization.
            return SimpleNamespace(loss=image.square().mean() if len(calls) <= 2 else image.sum() * 0)

        def fake_probes(**kwargs):
            return [{"optimizer_step": kwargs["step"], "probe": str(index), "ce": 0.0} for index in range(5)]

        def fake_endpoints(**kwargs):
            generations = [{"row": index} for index in range(6)]
            mcq = [{"row": index} for index in range(4)]
            for row in generations:
                kwargs["on_generation"](row)
            for row in mcq:
                kwargs["on_mcq"](row)
            return generations, mcq

        reference = torch.ones(1, 4, 2, 2)
        initial = pilot.make_initial_latent(reference, seed=None, rho=0.0)
        model = torch.nn.Linear(1, 1).requires_grad_(False).eval()
        with TemporaryDirectory() as directory, patch.object(pilot, "VAELatentOracle", FakeOracle), \
                patch.object(pilot, "configure_strict_cuda_determinism", return_value={}), \
                patch.object(pilot, "loss_output", side_effect=fake_loss), \
                patch.object(pilot, "fixed_probes", side_effect=fake_probes), \
                patch.object(pilot, "endpoint_rows", side_effect=fake_endpoints):
            output = Path(directory)
            result = pilot.run_one(
                run_spec={"run_id": "blank-open", "init_id": "blank", "arm": "open"}, initial=initial,
                reference=reference, vae=model, reader=model, processor=None, reader_device=torch.device("cpu"),
                vae_device=torch.device("cpu"), config=self.config, controls={}, output_dir=output,
            )
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["optimizer_steps"], 256)
            self.assertEqual(len(calls), 258)
            run_dir = output / "runs/blank-open"
            metrics = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(metrics), 256)
            self.assertTrue(all(row["gradient_all_zero"] for row in metrics))
            self.assertTrue(all(row["actual_update_l2"] == 0 for row in metrics))
            self.assertEqual(len(list((run_dir / "latents").glob("*.pt"))), 257)
            checkpoint = torch.load(run_dir / "checkpoints/step-256.pt", weights_only=True)
            self.assertEqual(checkpoint["optimizer_step"], 256)
            adam_state = next(iter(checkpoint["optimizer"]["state"].values()))
            self.assertEqual(int(adam_state["step"]), 256)
            self.assertIn("rng", checkpoint)


if __name__ == "__main__":
    unittest.main()
