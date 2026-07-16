from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stage_status(run_dir: Path) -> tuple[dict[str, Any], int]:
    run_dir = run_dir.resolve()
    terminal = load_json(run_dir / "terminal.json")
    if terminal is not None:
        summary = {
            "run_dir": str(run_dir),
            "status": terminal.get("status"),
            "passed": terminal.get("passed"),
            "exit_code": terminal.get("exit_code"),
            "finished_at": terminal.get("finished_at"),
            "stdout_sha256": terminal.get("stdout_sha256"),
            "stderr_sha256": terminal.get("stderr_sha256"),
        }
        return summary, 0 if terminal.get("passed") is True else 2

    running = load_json(run_dir / "running.json")
    if running is not None:
        alive = pid_alive(int(running["pid"]))
        summary = {
            "run_dir": str(run_dir),
            "status": "running" if alive else "orphaned_without_terminal_sentinel",
            "passed": None,
            "pid": running["pid"],
            "pid_alive": alive,
            "started_at": running.get("started_at"),
        }
        return summary, 3 if alive else 2

    launch = load_json(run_dir / "launch.json")
    if launch is not None:
        return {
            "run_dir": str(run_dir),
            "status": "launching_without_worker_sentinel",
            "passed": None,
            "created_at": launch.get("created_at"),
        }, 2
    return {"run_dir": str(run_dir), "status": "missing", "passed": None}, 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Read (but never modify) an Inspire background-stage sentinel")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report, exit_code = stage_status(args.run_dir)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
