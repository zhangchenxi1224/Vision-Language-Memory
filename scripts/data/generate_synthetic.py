from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import DatasetSizes, generate_dataset, validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic stateful-memory episode JSONL")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "synthetic_v1")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train", type=int, default=5_000)
    parser.add_argument("--dev", type=int, default=500)
    parser.add_argument("--test-id", type=int, default=1_000)
    parser.add_argument("--test-ood", type=int, default=1_000)
    args = parser.parse_args()

    sizes = DatasetSizes(train=args.train, dev=args.dev, test_id=args.test_id, test_ood=args.test_ood)
    manifest = generate_dataset(args.output_dir, sizes=sizes, seed=args.seed)
    report = validate_dataset(args.output_dir, expected_sizes=sizes.as_dict())
    print(json.dumps({"manifest": manifest, "validation": report.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
