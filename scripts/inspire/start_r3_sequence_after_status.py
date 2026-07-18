from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from r3_dag_contract import require_absolute_executable, verify_sha_sidecar  # noqa: E402


PLATFORM_STATUS_PROTOCOL = "vision-memory-inspire-platform-status.v1"
FORBIDDEN_FORWARDED_OPTIONS = {
    "--expected-node",
    "--platform-status",
    "--platform-status-sha256",
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def load_running_receipt(path: Path) -> tuple[dict[str, Any], str]:
    digest = verify_sha_sidecar(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Platform status receipt must be a JSON object")
    if value.get("schema_version") != 1 or value.get("protocol") != PLATFORM_STATUS_PROTOCOL:
        raise ValueError("Platform status receipt has the wrong protocol")
    if value.get("status") != "RUNNING" or value.get("node_status") != "READY":
        raise ValueError("Platform status receipt does not prove a running ready instance")
    node = value.get("node")
    if not isinstance(node, str) or not node:
        raise ValueError("Platform status receipt does not contain a scheduled node")
    return value, digest


def validate_forwarded(arguments: list[str]) -> None:
    if not arguments:
        raise ValueError("R3 sequence arguments are required after --")
    for argument in arguments:
        key = argument.split("=", 1)[0]
        if key in FORBIDDEN_FORWARDED_OPTIONS:
            raise ValueError(f"{key} is derived exclusively from the platform status receipt")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for a fresh RUNNING Inspire receipt, then exec the fail-stop R3 technical sequence"
    )
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--platform-status", type=Path, required=True)
    parser.add_argument("--launcher-status", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("sequence_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = list(args.sequence_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    try:
        validate_forwarded(forwarded)
        python = require_absolute_executable(args.python, "--python")
        receipt_path = Path(os.path.abspath(args.platform_status.expanduser()))
        launcher_status = Path(os.path.abspath(args.launcher_status.expanduser()))
        if args.wait_seconds <= 0 or args.poll_seconds <= 0:
            raise ValueError("wait and poll intervals must be positive")
        atomic_json(
            launcher_status,
            {
                "schema": "vision_memory.r3-poststart-receipt-waiter.v1",
                "state": "waiting_for_platform_status",
                "started_at": utc_now(),
                "platform_status": str(receipt_path),
                "wait_seconds": args.wait_seconds,
            },
        )
        deadline = time.monotonic() + args.wait_seconds
        last_receipt_error: str | None = None
        while time.monotonic() < deadline:
            if receipt_path.is_file() and receipt_path.with_suffix(receipt_path.suffix + ".sha256").is_file():
                try:
                    receipt, digest = load_running_receipt(receipt_path)
                except (OSError, ValueError) as exc:
                    last_receipt_error = f"{type(exc).__name__}: {exc}"
                    atomic_json(
                        launcher_status,
                        {
                            "schema": "vision_memory.r3-poststart-receipt-waiter.v1",
                            "state": "waiting_for_valid_platform_status",
                            "updated_at": utc_now(),
                            "platform_status": str(receipt_path),
                            "last_receipt_error": last_receipt_error,
                        },
                    )
                    time.sleep(args.poll_seconds)
                    continue
                command = [
                    str(python),
                    str(ROOT / "scripts" / "inspire" / "run_r3_technical_sequence.py"),
                    *forwarded,
                    "--expected-node",
                    str(receipt["node"]),
                    "--platform-status",
                    str(receipt_path),
                    "--platform-status-sha256",
                    digest,
                ]
                atomic_json(
                    launcher_status,
                    {
                        "schema": "vision_memory.r3-poststart-receipt-waiter.v1",
                        "state": "execing_technical_sequence",
                        "updated_at": utc_now(),
                        "platform_status": str(receipt_path),
                        "platform_status_sha256": digest,
                        "expected_node": receipt["node"],
                        "argv": command,
                    },
                )
                os.execv(str(python), command)
            time.sleep(args.poll_seconds)
        detail = f"; last receipt error: {last_receipt_error}" if last_receipt_error else ""
        raise TimeoutError(f"A RUNNING platform status receipt did not arrive before the waiter timeout{detail}")
    except (OSError, ValueError, TimeoutError) as exc:
        atomic_json(
            args.launcher_status,
            {
                "schema": "vision_memory.r3-poststart-receipt-waiter.v1",
                "state": "failed",
                "finished_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
