from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import read_prefeval_adapted_jsonl  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and convert separated PrefEval model_input/label exports for the episode trainer"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("adapt_train", "adapt_dev", "adapt_ood"), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")

    episodes = read_prefeval_adapted_jsonl(args.input, allowed_splits={args.split})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "vision_memory.prefeval.supervised-boundary.v1",
        "source": str(args.input.resolve()),
        "source_sha256": sha256_file(args.input),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "split": args.split,
        "episodes": len(episodes),
        "safety_contract": {
            "updater_fields": ["turns[].event_text"],
            "reader_fields": ["turns[].query_text", "turns[].choices"],
            "loss_only_fields": ["turns[].target_index"],
        },
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
