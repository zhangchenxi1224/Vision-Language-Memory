from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def git(repo: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch immutable external dataset snapshots")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override VLM_DATA_ROOT (default: project data directory)",
    )
    args = parser.parse_args()
    lock = json.loads((ROOT / "data.lock.json").read_text(encoding="utf-8"))
    data_root = args.data_root or Path(os.environ.get("VLM_DATA_ROOT", ROOT / "data"))
    data_root = data_root.expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    for name, specification in lock["datasets"].items():
        destination = data_root / specification["local_dir"]
        revision = specification["revision"]
        created = False
        if destination.exists():
            inside = git(destination, "rev-parse", "--is-inside-work-tree", check=False)
            if inside.returncode != 0 or inside.stdout.strip() != "true":
                raise SystemExit(f"Refusing to replace non-Git dataset path: {destination}")
            status = git(destination, "status", "--short").stdout.strip()
            if status:
                raise SystemExit(f"Refusing to modify dirty {name} checkout:\n{status}")
        else:
            created = True
            subprocess.run(
                ["git", "clone", "--filter=blob:none", "--no-checkout", specification["url"], str(destination)],
                check=True,
            )

        current = git(destination, "rev-parse", "HEAD", check=False)
        if current.returncode != 0 or current.stdout.strip() != revision:
            git(destination, "fetch", "origin", revision)
        if created:
            git(destination, "checkout", "--force", "--detach", revision)
        elif current.returncode != 0 or current.stdout.strip() != revision:
            git(destination, "checkout", "--detach", revision)

        actual = git(destination, "rev-parse", "HEAD").stdout.strip()
        status = git(destination, "status", "--short").stdout.strip()
        if actual != revision or status:
            raise SystemExit(f"Dataset lock failed for {name}: revision={actual}, dirty={bool(status)}")
        print(f"{name} ready: {destination}@{actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
