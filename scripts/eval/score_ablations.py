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

from vision_memory.eval import (  # noqa: E402
    compute_synthetic_metrics,
    holm_correction,
    paired_hierarchical_bootstrap,
    read_records,
)


CONTRASTS = (
    ("dreamlite_latent", "ablation_detach"),
    ("dreamlite_latent", "ablation_decode_reencode"),
    ("dreamlite_latent", "ablation_noop_skip"),
    ("dreamlite_latent", "ablation_set_only"),
    ("ablation_rank4_blank_1k", "ablation_rank8"),
    ("ablation_rank4_blank_1k", "ablation_learned_initial"),
)
EXPECTED_SEEDS = {
    "dreamlite_latent": {0, 1, 2},
    "ablation_detach": {0, 1, 2},
    "ablation_decode_reencode": {0, 1, 2},
    "ablation_noop_skip": {0, 1, 2},
    "ablation_set_only": {0, 1, 2},
    "ablation_rank4_blank_1k": {0},
    "ablation_rank8": {0},
    "ablation_learned_initial": {0},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(record: dict) -> dict:
    value = dict(record)
    value.setdefault("split", "test_id")
    value.setdefault("condition", "standard")
    value.setdefault("protocol", "synthetic")
    value.setdefault("noop_policy", "keep")
    value.setdefault("diffusion_seed", 0)
    value.setdefault("recurrence_mode", "direct_latent")
    value.setdefault("distractor_variant", "unpaired")
    value.setdefault("subtype", value.get("transition", "unknown"))
    return value


def is_preregistered_slice(record: dict) -> bool:
    method = str(record.get("method"))
    expected_noop = "skip" if method == "ablation_noop_skip" else "keep"
    expected_recurrence = "decode_reencode" if method == "ablation_decode_reencode" else "direct_latent"
    return (
        record["split"] == "test_id"
        and record["condition"] == "standard"
        and record["protocol"] == "synthetic"
        and record["diffusion_seed"] == 0
        and record["noop_policy"] == expected_noop
        and record["recurrence_mode"] == expected_recurrence
    )


def method_summary(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict[object, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[str(record["method"])][record.get("seed", 0)].append(record)
    output: dict[str, dict] = {}
    for method, by_seed in sorted(grouped.items()):
        per_seed = []
        for seed, rows in sorted(by_seed.items(), key=lambda item: repr(item[0])):
            metrics = compute_synthetic_metrics(rows)
            per_seed.append(
                {
                    "seed": seed,
                    "queries": len(rows),
                    "macro_mcq_accuracy": metrics["macro_mcq_accuracy"],
                    "micro_accuracy": metrics["micro_accuracy"],
                    "by_subtype": metrics["by_subtype"],
                }
            )
        values = [float(row["macro_mcq_accuracy"]) for row in per_seed]
        output[method] = {
            "per_seed": per_seed,
            "n_seeds": len(per_seed),
            "mean_macro_mcq_accuracy": statistics.fmean(values),
            "sd_macro_mcq_accuracy": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the preregistered DreamLite ablation family")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    records = [normalized(row) for row in read_records(args.predictions)]
    selected = [row for row in records if is_preregistered_slice(row)]
    methods = {str(row.get("method")) for row in selected}
    required = {method for contrast in CONTRASTS for method in contrast}
    missing = sorted(required - methods)
    if missing:
        raise ValueError(f"Ablation predictions are missing required methods: {missing}")
    for method, expected_seeds in EXPECTED_SEEDS.items():
        observed = {int(row.get("seed", 0)) for row in selected if row.get("method") == method}
        if observed != expected_seeds:
            raise ValueError(
                f"Ablation method {method!r} has seeds {sorted(observed)}; expected {sorted(expected_seeds)}."
            )

    contrasts: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for method_a, method_b in CONTRASTS:
        name = f"{method_a}_vs_{method_b}"
        result = paired_hierarchical_bootstrap(
            selected,
            method_a=method_a,
            method_b=method_b,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
            pair_fields=(
                "episode_id",
                "query_id",
                "seed",
                "split",
                "condition",
                "protocol",
                "distractor_variant",
            ),
        )
        contrasts[name] = result
        p_values[name] = float(result["two_sided_p_value"])

    report = {
        "schema_version": "vision_memory.synthetic.ablations.v1",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "selection": {
            "split": "test_id",
            "condition": "standard",
            "diffusion_seed": 0,
            "method_specific_noop_and_recurrence": True,
        },
        "records": len(selected),
        "methods": method_summary(selected),
        "contrasts": contrasts,
        "holm_ablation_family": holm_correction(p_values, alpha=args.alpha),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(selected), "methods": sorted(methods)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
