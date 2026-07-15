from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.eval import compute_synthetic_metrics, read_records  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Score five-seed diffusion-noise robustness")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes-per-training-seed", type=int, default=200)
    parser.add_argument("--expected-diffusion-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--expected-training-seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    expected_noise = set(args.expected_diffusion_seeds)
    if len(expected_noise) != len(args.expected_diffusion_seeds) or any(seed < 0 for seed in expected_noise):
        raise SystemExit("--expected-diffusion-seeds must be distinct non-negative integers.")
    expected_training = set(args.expected_training_seeds)
    if len(expected_training) != len(args.expected_training_seeds) or any(seed < 0 for seed in expected_training):
        raise SystemExit("--expected-training-seeds must be distinct non-negative integers.")

    rows = []
    for raw in read_records(args.predictions):
        row = dict(raw)
        row.setdefault("condition", "standard")
        row.setdefault("split", "test_id")
        row.setdefault("protocol", "synthetic")
        row.setdefault("noop_policy", "keep")
        row.setdefault("recurrence_mode", "direct_latent")
        if (
            row["condition"] == "standard"
            and row["split"] == "test_id"
            and row["protocol"] == "synthetic"
            and row["noop_policy"] == "keep"
            and row["recurrence_mode"] == "direct_latent"
        ):
            rows.append(row)
    if not rows:
        raise ValueError("No preregistered noise-robustness predictions found.")

    grouped: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("seed", 0)].append(row)
    observed_training = {int(seed) for seed in grouped}
    if observed_training != expected_training:
        raise ValueError(
            f"Observed training seeds {sorted(observed_training)} do not match {sorted(expected_training)}."
        )
    per_training_seed = []
    for training_seed, seed_rows in sorted(grouped.items(), key=lambda item: repr(item[0])):
        episodes = {str(row["episode_id"]) for row in seed_rows}
        observed_noise = {int(row["diffusion_seed"]) for row in seed_rows}
        if len(episodes) != args.expected_episodes_per_training_seed:
            raise ValueError(
                f"Training seed {training_seed!r} has {len(episodes)} episodes; "
                f"expected {args.expected_episodes_per_training_seed}."
            )
        if observed_noise != expected_noise:
            raise ValueError(
                f"Training seed {training_seed!r} diffusion seeds {sorted(observed_noise)} "
                f"do not match {sorted(expected_noise)}."
            )
        metrics = compute_synthetic_metrics(seed_rows)
        robustness = metrics["noise_robustness"]
        if robustness is None or robustness["diffusion_seed_counts"] != [len(expected_noise)]:
            raise ValueError(f"Training seed {training_seed!r} does not have a complete paired noise grid.")
        by_noise = {}
        for diffusion_seed in sorted(expected_noise):
            selected = [row for row in seed_rows if int(row["diffusion_seed"]) == diffusion_seed]
            by_noise[str(diffusion_seed)] = compute_synthetic_metrics(selected)["macro_mcq_accuracy"]
        per_training_seed.append(
            {
                "training_seed": training_seed,
                "episodes": len(episodes),
                "queries": robustness["n_episode_queries"],
                "by_diffusion_seed_macro_accuracy": by_noise,
                **robustness,
            }
        )

    mean_noise_accuracy = [float(item["mean_accuracy_across_noise"]) for item in per_training_seed]
    invariance = [float(item["prediction_correctness_invariance_rate"]) for item in per_training_seed]
    report = {
        "schema_version": "vision_memory.synthetic.noise_robustness.v1",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "expected_diffusion_seeds": sorted(expected_noise),
        "expected_episodes_per_training_seed": args.expected_episodes_per_training_seed,
        "per_training_seed": per_training_seed,
        "mean_accuracy_across_noise": statistics.fmean(mean_noise_accuracy),
        "sd_accuracy_across_training_seeds": (
            statistics.stdev(mean_noise_accuracy) if len(mean_noise_accuracy) > 1 else 0.0
        ),
        "mean_prediction_correctness_invariance_rate": statistics.fmean(invariance),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "training_seeds": len(per_training_seed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
