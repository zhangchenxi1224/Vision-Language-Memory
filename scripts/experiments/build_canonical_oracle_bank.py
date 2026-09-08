"""Build Stage B only from preregistered independent tasks and audited Stage A evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_memory.training.frozen_oracle_writer import build_bank  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Preregistered schema-v1 task/run inventory")
    parser.add_argument("--geometry-decision", type=Path, required=True, help="Completed Stage A decision with canonical thresholds")
    parser.add_argument("--output", type=Path, required=True, help="New bank JSON including all excluded/missing tasks")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; preserve previous bank artifacts")
    result = build_bank(args.manifest, args.geometry_decision, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "blocking_reasons": result["blocking_reasons"]}, ensure_ascii=False))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
