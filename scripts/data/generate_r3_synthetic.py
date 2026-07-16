from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import (  # noqa: E402
    R3SyntheticSizes,
    generate_r3_synthetic,
    validate_r3_synthetic,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate the split-before-expansion R3 synthetic pilot or formal corpus"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "r3_synthetic_pilot")
    parser.add_argument("--profile", choices=("pilot", "formal", "custom"), default="pilot")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train", type=int)
    parser.add_argument("--dev", type=int)
    parser.add_argument("--test-id", type=int)
    parser.add_argument("--test-ood", type=int)
    parser.add_argument("--balance-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    defaults = R3SyntheticSizes.formal() if args.profile == "formal" else R3SyntheticSizes.pilot()
    overrides = (args.train, args.dev, args.test_id, args.test_ood)
    if args.profile != "custom" and any(value is not None for value in overrides):
        raise SystemExit("Size overrides require --profile custom so pilot/formal preregistered counts stay fixed.")
    sizes = R3SyntheticSizes(
        train=args.train if args.train is not None else defaults.train,
        dev=args.dev if args.dev is not None else defaults.dev,
        test_id=args.test_id if args.test_id is not None else defaults.test_id,
        test_ood=args.test_ood if args.test_ood is not None else defaults.test_ood,
    )
    manifest = generate_r3_synthetic(
        args.output_dir,
        sizes=sizes,
        seed=args.seed,
        profile=args.profile,
    )
    report = validate_r3_synthetic(
        args.output_dir,
        expected_sizes=sizes.as_dict(),
        balance_tolerance=args.balance_tolerance,
    )
    print(json.dumps({"manifest": manifest, "validation": report.to_dict()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
