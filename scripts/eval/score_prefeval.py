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
    diagnostic_metrics,
    filter_preregistered_records,
    holm_correction,
    paired_hierarchical_bootstrap,
    read_records,
    seeded_stratified_accuracy,
    topic_form_metrics,
)


DEFAULT_MAIN_CONTRASTS = (
    ("dreamlite_latent", "query_only"),
    ("dreamlite_latent", "frozen_dreamlite"),
    ("dreamlite_latent", "lightweight_recurrent"),
    ("dreamlite_latent", "full_history"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_contrast(raw: str) -> tuple[str, str]:
    parts = raw.split(":")
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("contrast must have the form METHOD_A:METHOD_B")
    return parts[0], parts[1]


def normalize_record(record: dict) -> dict:
    normalized = dict(record)
    normalized.setdefault("condition", "standard")
    normalized.setdefault("counterfactual_variant", "distractor")
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
            metrics = topic_form_metrics(rows)
            per_seed.append(
                {
                    "seed": seed,
                    "n": len(rows),
                    "topic_form_macro_accuracy": metrics["topic_form_macro_accuracy"],
                    "micro_accuracy": metrics["micro_accuracy"],
                    "topic_form_cells": metrics["cells"],
                    "by_topic_macro_accuracy": metrics["by_topic_macro_accuracy"],
                    "by_form_macro_accuracy": metrics["by_form_macro_accuracy"],
                }
            )
        values = [float(item["topic_form_macro_accuracy"]) for item in per_seed]
        result[method] = {
            "per_seed": per_seed,
            "n_seeds": len(per_seed),
            "mean_topic_form_macro_accuracy": statistics.fmean(values),
            "sd_topic_form_macro_accuracy": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Score PrefEval predictions and paired contrasts")
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
    parser.add_argument("--allow-unpaired", action="store_true")
    parser.add_argument("--extra-binary-field", action="append", default=[])
    parser.add_argument("--headline-condition", default="standard")
    parser.add_argument("--headline-protocol", default="oracle-sparse")
    parser.add_argument("--headline-forced-write-k", type=int)
    parser.add_argument("--headline-form", default="explicit")
    parser.add_argument("--headline-split")
    parser.add_argument("--headline-counterfactual-variant", default="distractor")
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
            forced_write_k=args.headline_forced_write_k,
            form=args.headline_form,
            split=args.headline_split,
            counterfactual_variant=args.headline_counterfactual_variant,
            noop_policy=args.headline_noop_policy,
            diffusion_seed=args.headline_diffusion_seed,
            recurrence_mode=args.headline_recurrence_mode,
        )
    )
    requested_contrasts = tuple(args.contrast or DEFAULT_MAIN_CONTRASTS)
    if len(requested_contrasts) != 4 or len(set(requested_contrasts)) != 4:
        raise ValueError("PrefEval evaluation requires exactly four distinct preregistered contrasts.")

    contrasts = {}
    p_values = {}
    for method_a, method_b in requested_contrasts:
        name = f"{method_a}_vs_{method_b}"
        if name in contrasts:
            raise ValueError(f"Duplicate contrast: {name}")
        result = paired_hierarchical_bootstrap(
            headline_records,
            method_a=method_a,
            method_b=method_b,
            subtype_field="form",
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
            strict_pairs=not args.allow_unpaired,
            pair_fields=(
                "base_pair_id",
                "form",
                "split",
                "condition",
                "protocol",
                "forced_write_k",
                "seed",
                "diffusion_seed",
                "recurrence_mode",
                "counterfactual_variant",
                "noop_policy",
            ),
        )
        contrasts[name] = result
        p_values[name] = result["two_sided_p_value"]

    output = {
        "schema_version": "vision_memory.prefeval.evaluation.v2",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "n_records": len(records),
        "headline": {
            "selection": {
                "split": args.headline_split,
                "condition": args.headline_condition,
                "protocol": args.headline_protocol,
                "forced_write_k": args.headline_forced_write_k,
                "form": args.headline_form,
                "counterfactual_variant": args.headline_counterfactual_variant,
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
                "forced_write_k",
                "topic",
                "form",
                "counterfactual_variant",
                "noop_policy",
                "diffusion_seed",
                "recurrence_mode",
            ),
        ),
        "diagnostics": {
            method: diagnostic_metrics(rows, extra_binary_fields=args.extra_binary_field)
            for method, rows in sorted(by_method.items())
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
