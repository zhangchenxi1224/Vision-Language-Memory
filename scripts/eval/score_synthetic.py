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
    compute_synthetic_metrics,
    holm_correction,
    paired_hierarchical_bootstrap,
    read_records,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Score synthetic state-memory predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrast", action="append", type=parse_contrast, default=[])
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    records = read_records(args.predictions)
    by_method: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        record.setdefault("topic", record.get("domain", "synthetic"))
        record.setdefault("subtype", record.get("transition", "unknown"))
        record.setdefault("form", record["subtype"])
        by_method[str(record.get("method", "default"))].append(record)

    contrasts: dict[str, dict] = {}
    p_values: dict[str, float] = {}
    for method_a, method_b in args.contrast:
        name = f"{method_a}_vs_{method_b}"
        result = paired_hierarchical_bootstrap(
            records,
            method_a=method_a,
            method_b=method_b,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
            pair_fields=("episode_id", "query_id", "condition", "seed"),
        )
        contrasts[name] = result
        p_values[name] = float(result["two_sided_p_value"])

    output = {
        "schema_version": "vision_memory.synthetic.evaluation.v1",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "methods": {method: compute_synthetic_metrics(rows) for method, rows in sorted(by_method.items())},
        "contrasts": contrasts,
        "holm": holm_correction(p_values),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(records), "methods": sorted(by_method)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
