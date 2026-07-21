from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data.r4_baseline_lockbox import (  # noqa: E402
    R4_BASELINE_SEED,
    generate_r4_baseline_lockbox,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the prospective R4 Qwen baseline lockbox without privileged artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "r4_qwen_baseline_lockbox",
    )
    parser.add_argument("--seed", type=int, default=R4_BASELINE_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = generate_r4_baseline_lockbox(args.output_dir, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
