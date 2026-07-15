from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.eval import (  # noqa: E402
    compute_prefeval_metrics,
    holm_correction,
    paired_hierarchical_bootstrap,
    read_records,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Score PrefEval predictions and paired contrasts")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrast", action="append", type=parse_contrast, default=[])
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--allow-unpaired", action="store_true")
    parser.add_argument("--extra-binary-field", action="append", default=[])
    args = parser.parse_args()

    records = read_records(args.predictions)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_method[str(record.get("method", "default"))].append(record)
    method_metrics = {
        method: compute_prefeval_metrics(rows, extra_binary_fields=args.extra_binary_field)
        for method, rows in sorted(by_method.items())
    }

    contrasts = {}
    p_values = {}
    for method_a, method_b in args.contrast:
        name = f"{method_a}_vs_{method_b}"
        if name in contrasts:
            raise ValueError(f"Duplicate contrast: {name}")
        result = paired_hierarchical_bootstrap(
            records,
            method_a=method_a,
            method_b=method_b,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
            strict_pairs=not args.allow_unpaired,
        )
        contrasts[name] = result
        p_values[name] = result["two_sided_p_value"]

    output = {
        "schema_version": "vision_memory.prefeval.evaluation.v1",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "n_records": len(records),
        "methods": method_metrics,
        "contrasts": contrasts,
        "holm": holm_correction(p_values, alpha=args.alpha),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records), "methods": sorted(by_method)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
