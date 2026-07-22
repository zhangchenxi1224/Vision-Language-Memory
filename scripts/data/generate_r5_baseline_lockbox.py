from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.r5_baseline_lockbox import (  # noqa: E402
    R5_BASELINE_SEED,
    generate_r5_baseline_lockbox,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the prospective R5 same-entity micro lockbox and inherit the SHA-verified sealed R4 formal files"
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal-source-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=R5_BASELINE_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_r5_baseline_lockbox(
        args.output_dir,
        formal_source_dir=args.formal_source_dir,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
