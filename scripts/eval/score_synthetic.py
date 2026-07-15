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
    filter_preregistered_records,
    holm_correction,
    paired_hierarchical_bootstrap,
    read_records,
    seeded_stratified_accuracy,
)


DEFAULT_MAIN_CONTRASTS = (
    ("dreamlite_latent", "query_only"),
    ("dreamlite_latent", "frozen_dreamlite"),
    ("dreamlite_latent", "lightweight_recurrent"),
    ("dreamlite_latent", "full_history"),
)


def parse_contrast(raw: str) -> tuple[str, str]:
    values = raw.split(":", 1)
    if len(values) != 2 or not all(values):
        raise argparse.ArgumentTypeError("contrast must be METHOD_A:METHOD_B")
    return values[0], values[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_record(record: dict) -> dict:
    normalized = dict(record)
    normalized.setdefault("condition", "standard")
    normalized.setdefault("protocol", "synthetic")
    normalized.setdefault("subtype", normalized.get("transition", "unknown"))
    normalized.setdefault("form", normalized["subtype"])
    normalized.setdefault("split", "unknown")
    normalized.setdefault("topic", normalized.get("domain", "synthetic"))
    normalized.setdefault("noop_policy", "keep")
    normalized.setdefault("diffusion_seed", 0)
    normalized.setdefault("recurrence_mode", "direct_latent")
    return normalized


def headline_by_seed(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, dict[object, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[str(record.get("method", "default"))][record["seed"]].append(record)
    result: dict[str, dict] = {}
    for method, by_seed in sorted(grouped.items()):
        per_seed = []
        for seed, rows in sorted(by_seed.items(), key=lambda item: repr(item[0])):
            metrics = compute_synthetic_metrics(rows)
            per_seed.append(
                {
                    "seed": seed,
                    "n": len(rows),
                    "macro_mcq_accuracy": metrics["macro_mcq_accuracy"],
                    "macro_mcq_cells": metrics["macro_mcq_cells"],
                    "subtype_macro_accuracy": metrics["subtype_macro_accuracy"],
                    "micro_accuracy": metrics["micro_accuracy"],
                }
            )
        values = [float(item["macro_mcq_accuracy"]) for item in per_seed]
        result[method] = {
            "per_seed": per_seed,
            "n_seeds": len(per_seed),
            "primary_metric": "macro_mcq_accuracy",
            "macro_mcq_definition": "equal-weight mean over observed topic x transition-subtype cells",
            "mean_macro_mcq_accuracy": statistics.fmean(values),
            "sd_macro_mcq_accuracy": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Score synthetic state-memory predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contrast",
        action="append",
        type=parse_contrast,
        help="Exactly four preregistered METHOD_A:METHOD_B comparisons; fixed defaults are used if omitted.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--headline-split", default="test_id")
    parser.add_argument("--headline-condition", default="standard")
    parser.add_argument("--headline-protocol", default="synthetic")
    parser.add_argument("--headline-form")
    parser.add_argument(
        "--headline-distractor-variant",
        choices=("clean", "distractor", "unpaired"),
        help="Optional dataset stream slice; omitted means the full synthetic test split.",
    )
    parser.add_argument("--headline-noop-policy", choices=("keep", "skip"), default="keep")
    parser.add_argument("--headline-diffusion-seed", type=int, default=0)
    parser.add_argument("--headline-recurrence-mode", default="direct_latent")
    args = parser.parse_args()

    records = [normalize_record(record) for record in read_records(args.predictions)]
    by_method: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_method[str(record.get("method", "default"))].append(record)

    headline_records = list(
        filter_preregistered_records(
            records,
            condition=args.headline_condition,
            protocol=args.headline_protocol,
            form=args.headline_form,
            split=args.headline_split,
            distractor_variant=args.headline_distractor_variant,
            noop_policy=args.headline_noop_policy,
            diffusion_seed=args.headline_diffusion_seed,
            recurrence_mode=args.headline_recurrence_mode,
        )
    )

    requested_contrasts = tuple(args.contrast or DEFAULT_MAIN_CONTRASTS)
    if len(requested_contrasts) != 4 or len(set(requested_contrasts)) != 4:
        raise ValueError("Synthetic evaluation requires exactly four distinct preregistered contrasts.")

    contrasts: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for method_a, method_b in requested_contrasts:
        name = f"{method_a}_vs_{method_b}"
        result = paired_hierarchical_bootstrap(
            headline_records,
            method_a=method_a,
            method_b=method_b,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
            pair_fields=(
                "episode_id",
                "query_id",
                "seed",
                "diffusion_seed",
                "split",
                "condition",
                "protocol",
                "form",
                "recurrence_mode",
                "distractor_variant",
                "noop_policy",
            ),
        )
        contrasts[name] = result
        p_values[name] = float(result["two_sided_p_value"])

    output = {
        "schema_version": "vision_memory.synthetic.evaluation.v2",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "headline": {
            "selection": {
                "split": args.headline_split,
                "condition": args.headline_condition,
                "protocol": args.headline_protocol,
                "form": args.headline_form,
                "distractor_variant": args.headline_distractor_variant,
                "noop_policy": args.headline_noop_policy,
                "diffusion_seed": args.headline_diffusion_seed,
                "recurrence_mode": args.headline_recurrence_mode,
            },
            "n_records": len(headline_records),
            "methods": headline_by_seed(headline_records),
        },
        "stratified_accuracy": seeded_stratified_accuracy(
            records,
            strata_fields=(
                "split",
                "condition",
                "protocol",
                "form",
                "distractor_variant",
                "noop_policy",
                "diffusion_seed",
                "recurrence_mode",
            ),
        ),
        "diagnostics": {
            method: compute_synthetic_metrics(rows) for method, rows in sorted(by_method.items())
        },
        "preregistered_comparisons": [list(contrast) for contrast in requested_contrasts],
        "contrasts": contrasts,
        "holm": holm_correction(p_values, alpha=args.alpha),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records), "methods": sorted(by_method)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
