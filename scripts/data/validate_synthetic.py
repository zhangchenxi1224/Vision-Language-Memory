from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate stateful-memory episode JSONL")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--balance-tolerance", type=float, default=0.02)
    parser.add_argument("--skip-manifest-hashes", action="store_true")
    args = parser.parse_args()
    report = validate_dataset(
        args.dataset,
        balance_tolerance=args.balance_tolerance,
        verify_manifest_hashes=not args.skip_manifest_hashes,
    )
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
