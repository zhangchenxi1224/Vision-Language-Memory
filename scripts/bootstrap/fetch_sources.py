from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def main() -> int:
    lock = json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))
    spec = lock["sources"]["dreamlite_reference"]
    destination = ROOT / spec["local_dir"]
    revision = spec["revision"]

    created = False
    if destination.exists():
        inside = git(destination, "rev-parse", "--is-inside-work-tree", check=False)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            raise SystemExit(f"Refusing to replace non-Git path: {destination}")
        status = git(destination, "status", "--short").stdout.strip()
        if status:
            raise SystemExit(f"Refusing to change dirty DreamLite checkout:\n{status}")
    else:
        created = True
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", spec["url"], str(destination)],
            check=True,
        )

    current = git(destination, "rev-parse", "HEAD", check=False)
    if current.returncode != 0 or current.stdout.strip() != revision:
        git(destination, "fetch", "origin", revision)
    if created or current.returncode != 0 or current.stdout.strip() != revision:
        git(destination, "checkout", "--detach", revision)

    actual = git(destination, "rev-parse", "HEAD").stdout.strip()
    if actual != revision:
        raise SystemExit(f"DreamLite revision mismatch: expected {revision}, got {actual}")
    if git(destination, "status", "--short").stdout.strip():
        raise SystemExit("DreamLite checkout became dirty during bootstrap.")

    print(f"DreamLite source ready: {destination}@{actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
