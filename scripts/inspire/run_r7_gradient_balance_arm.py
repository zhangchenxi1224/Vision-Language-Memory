"""Fail-closed Inspire controller for one R7 gradient-balance arm."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
R7_TRAINER = ROOT / "scripts" / "train" / "dreamlite_r7_gradient_balance.py"
R8_TRAINER = ROOT / "scripts" / "train" / "dreamlite_r8_conflict_projection.py"
TRAINER = R7_TRAINER
EXPECTED_DATA_SHA = {
    "train": "24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184",
    "dev": "8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303",
}
EXPECTED_SELECTED_SHA = "eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6"
R7_EXPECTED_MODE = {
    "raw-mean-control": "raw-mean",
    "unit-balanced-norm-matched": "unit-balanced-norm-matched",
}
R8_EXPECTED_MODE = {
    "raw-mean-control": "raw-mean",
    "common-descent-projected-norm-matched": "common-descent-projected-norm-matched",
}
EXPECTED_MODE = {**R7_EXPECTED_MODE, **R8_EXPECTED_MODE}
PROTOCOL = {
    "r7": {
        "trainer": R7_TRAINER,
        "expected_mode": R7_EXPECTED_MODE,
        "summary_filename": "r7_summary.json",
        "summary_schema": "vision_memory.r7-gradient-balance-summary.v1",
        "launch_schema": "vision_memory.r7-inspire-arm-launch.v1",
        "terminal_schema": "vision_memory.r7-inspire-arm-terminal.v1",
        "inventory_schema": "vision_memory.r7-artifact-inventory.v1",
    },
    "r8": {
        "trainer": R8_TRAINER,
        "expected_mode": R8_EXPECTED_MODE,
        "summary_filename": "r8_summary.json",
        "summary_schema": "vision_memory.r8-common-descent-summary.v1",
        "launch_schema": "vision_memory.r8-inspire-arm-launch.v1",
        "terminal_schema": "vision_memory.r8-inspire-arm-terminal.v1",
        "inventory_schema": "vision_memory.r8-artifact-inventory.v1",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "artifact_inventory.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-revision", choices=tuple(PROTOCOL), default="r7")
    parser.add_argument("--arm", choices=tuple(EXPECTED_MODE), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    return parser


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    protocol_revision = getattr(args, "protocol_revision", "r7")
    contract = PROTOCOL[protocol_revision]
    if args.arm not in contract["expected_mode"]:
        raise ValueError(f"{protocol_revision.upper()} controller does not define arm {args.arm}.")
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    if head != args.expected_commit:
        raise ValueError(
            f"{protocol_revision.upper()} controller commit mismatch: expected {args.expected_commit}, got {head}"
        )
    if dirty:
        raise ValueError(f"{protocol_revision.upper()} controller requires a clean detached experiment snapshot.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError(f"{protocol_revision.upper()} controller refuses a non-empty output root.")
    observed = {"train": _sha256(args.train), "dev": _sha256(args.dev)}
    if observed != EXPECTED_DATA_SHA:
        raise ValueError(f"{protocol_revision.upper()} fixed data SHA mismatch: {observed}")
    required_environment = (
        "PYTHONHASHSEED",
        "CUBLAS_WORKSPACE_CONFIG",
        "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256",
        "VLM_READER_SNAPSHOT_MANIFEST_SHA256",
    )
    missing = [name for name in required_environment if not os.environ.get(name)]
    if missing:
        raise ValueError(
            f"{protocol_revision.upper()} controller is missing strict environment variables: {missing}"
        )
    return {
        "git_commit": head,
        "git_dirty": False,
        "data_sha256": observed,
        "trainer": str(Path(contract["trainer"]).resolve()),
        "trainer_sha256": _sha256(Path(contract["trainer"])),
        "python": sys.executable,
        "host": platform.node(),
    }


def _command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    contract = PROTOCOL[getattr(args, "protocol_revision", "r7")]
    return [
        sys.executable,
        str(contract["trainer"]),
        "--arm",
        args.arm,
        "--train",
        str(args.train),
        "--dev",
        str(args.dev),
        "--dreamlite",
        str(args.dreamlite),
        "--reader",
        str(args.reader),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(args.seed),
        "--dreamlite-device",
        args.dreamlite_device,
        "--reader-device",
        args.reader_device,
        "--strict-determinism",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = PROTOCOL[args.protocol_revision]
    expected_mode = contract["expected_mode"]
    try:
        validated = _validate(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_root / "run"
    command = _command(args, run_dir)
    launch = {
        "schema": contract["launch_schema"],
        "status": "running",
        "started_at_utc": _utc_now(),
        "arm": args.arm,
        "gradient_aggregation": expected_mode[args.arm],
        "seed": args.seed,
        "command": command,
        **validated,
    }
    _write_json(args.output_root / "launch.json", launch)
    stdout_path = args.output_root / "stdout.log"
    stderr_path = args.output_root / "stderr.log"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr)
    elapsed = time.monotonic() - started
    summary_path = run_dir / str(contract["summary_filename"])
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    checks = {
        "child_exit_zero": result.returncode == 0,
        "summary_exists": summary_path.is_file(),
        "summary_schema": isinstance(summary, dict)
        and summary.get("schema") == contract["summary_schema"],
        "summary_completed": isinstance(summary, dict) and summary.get("status") == "completed",
        "arm_matches": isinstance(summary, dict) and summary.get("arm") == args.arm,
        "aggregation_matches": isinstance(summary, dict)
        and summary.get("gradient_aggregation") == expected_mode[args.arm],
        "summary_commit_matches": isinstance(summary, dict) and summary.get("git_commit") == args.expected_commit,
        "selected_segments_match": isinstance(summary, dict)
        and summary.get("selected_segments_sha256") == EXPECTED_SELECTED_SHA,
        "implementation_revision_recorded": isinstance(summary, dict) and bool(summary.get("implementation_revision")),
        "formal_success_not_claimed": isinstance(summary, dict) and summary.get("full_success_claim_allowed") is False,
    }
    terminal = {
        "schema": contract["terminal_schema"],
        "status": "completed_diagnostic" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scientific_success_claim": False,
        "arm": args.arm,
        "gradient_aggregation": expected_mode[args.arm],
        "child_exit_code": result.returncode,
        "checks": checks,
        "started_at_utc": launch["started_at_utc"],
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": elapsed,
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path) if summary_path.is_file() else None,
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
    }
    _write_json(args.output_root / "terminal.json", terminal)
    inventory = {
        "schema": contract["inventory_schema"],
        "root": str(args.output_root.resolve()),
        "artifacts": _inventory(args.output_root),
    }
    _write_json(args.output_root / "artifact_inventory.json", inventory)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    return 0 if terminal["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
