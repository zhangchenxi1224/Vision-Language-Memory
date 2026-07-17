from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "inspire"))

from model_snapshot_manifest import create_snapshot_manifest  # noqa: E402
METADATA_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.jinja",
    "*.py",
    "*.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct immutable HF snapshots from models.lock.json")
    parser.add_argument(
        "--only",
        choices=["all", "dreamlite_mobile", "qwen_reader"],
        default="all",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Fetch configs/tokenizers/processors but skip large weight files",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Override the model directory (or set VLM_MODEL_ROOT).",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install requirements/runtime-pinned.txt before fetching models.") from exc

    lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
    model_root = args.model_root or Path(os.environ.get("VLM_MODEL_ROOT", ROOT / "models"))
    model_root = model_root.expanduser().resolve()
    selected = lock["models"].items()
    if args.only != "all":
        selected = [(args.only, lock["models"][args.only])]

    for name, spec in selected:
        local_dir = model_root / Path(spec["local_dir"]).name
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {name}: {spec['repo_id']}@{spec['revision']} -> {local_dir}")
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_dir=local_dir,
            allow_patterns=METADATA_PATTERNS if args.metadata_only else None,
            max_workers=4,
        )
        (local_dir / ".locked_revision").write_text(spec["revision"] + "\n", encoding="utf-8")
        marker = ".metadata_complete" if args.metadata_only else ".snapshot_complete"
        if not args.metadata_only:
            weight_files = [
                *local_dir.rglob("*.safetensors"),
                *local_dir.rglob("*.bin"),
            ]
            weight_bytes = sum(path.stat().st_size for path in weight_files)
            expected = int(spec.get("weight_bytes", spec.get("approx_bytes", 0)))
            if not weight_files or (expected and weight_bytes < int(expected * 0.85)):
                raise RuntimeError(
                    f"Incomplete {name} snapshot: files={len(weight_files)}, "
                    f"weight_bytes={weight_bytes}, expected_at_least={int(expected * 0.85)}"
                )
            (local_dir / ".metadata_complete").write_text(spec["revision"] + "\n", encoding="utf-8")
        (local_dir / marker).write_text(spec["revision"] + "\n", encoding="utf-8")
        if not args.metadata_only:
            create_snapshot_manifest(
                model_dir=local_dir,
                repo_id=str(spec["repo_id"]),
                revision=str(spec["revision"]),
                overwrite=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
