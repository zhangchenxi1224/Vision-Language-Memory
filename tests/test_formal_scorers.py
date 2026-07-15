from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def base_record(*, method: str, seed: int, episode: int) -> dict:
    return {
        "episode_id": f"episode-{episode}",
        "query_id": f"episode-{episode}:q0",
        "method": method,
        "seed": seed,
        "diffusion_seed": 0,
        "split": "test_id",
        "condition": "standard",
        "protocol": "synthetic",
        "form": "set",
        "subtype": "set",
        "topic": f"topic-{episode}",
        "distractor_variant": "clean",
        "noop_policy": "keep",
        "recurrence_mode": "direct_latent",
        "prediction_index": 0,
        "target_index": 0,
    }


class FormalScorerTest(unittest.TestCase):
    def test_ablation_scorer_requires_and_compares_matched_budgets(self):
        three_seed = {
            "dreamlite_latent",
            "ablation_detach",
            "ablation_decode_reencode",
            "ablation_noop_skip",
            "ablation_set_only",
        }
        one_seed = {
            "ablation_rank4_blank_1k",
            "ablation_rank8",
            "ablation_learned_initial",
        }
        records = []
        for method in sorted(three_seed | one_seed):
            seeds = range(3) if method in three_seed else (0,)
            for seed in seeds:
                for episode in range(2):
                    record = base_record(method=method, seed=seed, episode=episode)
                    if method == "ablation_noop_skip":
                        record["noop_policy"] = "skip"
                    if method == "ablation_decode_reencode":
                        record["recurrence_mode"] = "decode_reencode"
                    records.append(record)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            output = root / "scores.json"
            write_jsonl(predictions, records)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "eval" / "score_ablations.py"),
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(output),
                    "--bootstrap-iterations",
                    "100",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["contrasts"]), 6)
            self.assertEqual(report["methods"]["dreamlite_latent"]["n_seeds"], 3)
            self.assertEqual(report["methods"]["ablation_rank8"]["n_seeds"], 1)

    def test_noise_scorer_checks_complete_episode_seed_grid(self):
        records = []
        for training_seed in range(3):
            for diffusion_seed in (0, 1):
                for episode in range(2):
                    record = base_record(method="dreamlite_latent", seed=training_seed, episode=episode)
                    record["diffusion_seed"] = diffusion_seed
                    records.append(record)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            output = root / "scores.json"
            write_jsonl(predictions, records)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "eval" / "score_noise_robustness.py"),
                    "--predictions",
                    str(predictions),
                    "--output",
                    str(output),
                    "--expected-episodes-per-training-seed",
                    "2",
                    "--expected-diffusion-seeds",
                    "0",
                    "1",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["per_training_seed"]), 3)
            self.assertEqual(report["per_training_seed"][0]["diffusion_seed_counts"], [2])


if __name__ == "__main__":
    unittest.main()
