from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.repro import emit_json_report, validate_e2e_pair_reports  # noqa: E402


def _read_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate paired two-event E2E BPTT and detach reports")
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--detached", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    report = validate_e2e_pair_reports(
        _read_report(args.positive),
        _read_report(args.detached),
        atol=args.atol,
        rtol=args.rtol,
    )
    report["positive_report"] = str(args.positive.resolve())
    report["detached_report"] = str(args.detached.resolve())
    emit_json_report(report, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
