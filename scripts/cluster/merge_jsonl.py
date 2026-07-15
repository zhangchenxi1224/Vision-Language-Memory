from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_method_map(raw: str) -> tuple[str, str]:
    values = raw.split("=", 1)
    if len(values) != 2 or not values[0] or not values[1]:
        raise argparse.ArgumentTypeError("method map must be OLD=NEW")
    return values[0], values[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Atomically merge prediction JSONL files")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-map", type=parse_method_map, action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite merged predictions: {args.output}")
    method_map = dict(args.method_map)
    if len(method_map) != len(args.method_map):
        raise SystemExit("Duplicate source method in --method-map.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for source in args.input:
                with source.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError(f"{source}:{line_number} is not a JSON object")
                        method = record.get("method")
                        if method in method_map:
                            record["method"] = method_map[method]
                        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                        count += 1
        if count == 0:
            raise ValueError("No prediction records were merged.")
        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps({"output": str(args.output.resolve()), "inputs": len(args.input), "records": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
