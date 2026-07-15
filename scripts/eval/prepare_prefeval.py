from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.eval import write_jsonl  # noqa: E402
from vision_memory.prefeval import FORCED_WRITE_COUNTS, FORMS, PrefEvalAdapter  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_snapshot(path: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()
    return commit, not bool(status)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export leakage-aware PrefEval visual-state episodes")
    parser.add_argument("--prefeval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--forms", nargs="+", choices=FORMS, default=list(FORMS))
    parser.add_argument(
        "--protocol",
        choices=("oracle-sparse", "forced-write", "all"),
        default="all",
    )
    parser.add_argument(
        "--forced-write-k",
        nargs="+",
        type=int,
        choices=FORCED_WRITE_COUNTS,
        default=list(FORCED_WRITE_COUNTS),
    )
    parser.add_argument(
        "--subset",
        choices=("all", "adapt_train", "adapt_dev", "adapt_ood"),
        default="all",
    )
    parser.add_argument("--adaptation-seed", type=int, default=2026)
    parser.add_argument("--option-seed", type=int, default=41)
    parser.add_argument("--distractor-seed", type=int, default=2026)
    parser.add_argument("--expected-base-pairs", type=int)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument(
        "--max-base-pairs-per-topic",
        type=int,
        help="Deterministic stratified subset; use 10 for the 200-pair forced-write study.",
    )
    args = parser.parse_args()
    if args.max_base_pairs_per_topic is not None and args.max_base_pairs_per_topic <= 0:
        raise SystemExit("--max-base-pairs-per-topic must be positive.")

    adapter = PrefEvalAdapter(
        args.prefeval_root,
        adaptation_seed=args.adaptation_seed,
        option_seed=args.option_seed,
        distractor_seed=args.distractor_seed,
    )
    splits = None if args.subset == "all" else (args.subset,)
    protocols: list[tuple[str, int]] = []
    if args.protocol in ("oracle-sparse", "all"):
        protocols.append(("oracle-sparse", 0))
    if args.protocol in ("forced-write", "all"):
        protocols.extend(("forced-write", count) for count in args.forced_write_k)

    def records():
        for protocol, count in protocols:
            for episode in adapter.iter_episodes(
                forms=args.forms,
                protocol=protocol,
                forced_write_k=count,
                splits=splits,
            ):
                if args.max_base_pairs_per_topic is not None and episode.row_index >= args.max_base_pairs_per_topic:
                    continue
                yield episode.to_record()

    count = write_jsonl(args.output, records())
    manifest_path = args.manifest_output or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    export_manifest = adapter.manifest()
    if args.expected_base_pairs is not None and export_manifest["base_pair_count"] != args.expected_base_pairs:
        raise RuntimeError(
            f"PrefEval base-pair count mismatch: {export_manifest['base_pair_count']} != {args.expected_base_pairs}"
        )
    if args.expected_records is not None and count != args.expected_records:
        raise RuntimeError(f"PrefEval export count mismatch: {count} != {args.expected_records}")
    git_revision, git_clean = git_snapshot(args.prefeval_root)
    if not git_clean:
        raise RuntimeError("PrefEval export refuses a dirty official snapshot.")
    locked_revision = json.loads((ROOT / "data.lock.json").read_text(encoding="utf-8"))["datasets"][
        "prefeval"
    ]["revision"]
    if git_revision != locked_revision:
        raise RuntimeError(f"PrefEval revision mismatch: {git_revision} != {locked_revision}")
    export_manifest["export"] = {
        "output": str(args.output.resolve()),
        "records": count,
        "output_sha256": sha256_file(args.output),
        "prefeval_git_revision": git_revision,
        "prefeval_git_clean": git_clean,
        "expected_base_pairs": args.expected_base_pairs,
        "expected_records": args.expected_records,
        "forms": list(args.forms),
        "protocols": [{"protocol": protocol, "forced_write_k": k} for protocol, k in protocols],
        "subset": args.subset,
        "max_base_pairs_per_topic": args.max_base_pairs_per_topic,
    }
    manifest_path.write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), "records": count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
