"""Fail-closed Inspire controller for one fixed R8 checkpoint trajectory."""

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
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = ROOT / "scripts" / "eval" / "dreamlite_r8_checkpoint_trajectory.py"
ARMS = ("raw-mean-control", "common-descent-projected-norm-matched")
EXPECTED_MODE = {
    "raw-mean-control": "raw-mean",
    "common-descent-projected-norm-matched": "common-descent-projected-norm-matched",
}
EXPECTED_DATA_SHA = {
    "train": "24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184",
    "dev": "8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303",
}
EXPECTED_SELECTED_SHA = "eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6"
EXPECTED_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256": (
        "1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159"
    ),
    "VLM_READER_SNAPSHOT_MANIFEST_SHA256": (
        "159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c"
    ),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
SOURCE_TERMINAL_SCHEMA = "vision_memory.r8-inspire-arm-terminal.v1"
SOURCE_MANIFEST_SCHEMA = "vision_memory.r8-common-descent-manifest.v1"
SOURCE_INVENTORY_SCHEMA = "vision_memory.r8-artifact-inventory.v1"
TRAJECTORY_SCHEMA = "vision_memory.r8-checkpoint-trajectory.v1"
TRAJECTORY_INVENTORY_SCHEMA = "vision_memory.r8-checkpoint-trajectory-inventory.v1"
LAUNCH_SCHEMA = "vision_memory.r8-checkpoint-trajectory-launch.v1"
TERMINAL_SCHEMA = "vision_memory.r8-checkpoint-trajectory-terminal.v1"
CONTROLLER_INVENTORY_SCHEMA = "vision_memory.r8-checkpoint-trajectory-controller-inventory.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


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


def _validate_inventory(root: Path, *, expected_schema: str) -> dict[str, Any]:
    inventory_path = root / "artifact_inventory.json"
    inventory = _load_json(inventory_path)
    if inventory.get("schema") != expected_schema:
        raise ValueError(f"Artifact inventory schema mismatch: {root}")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Artifact inventory is empty: {root}")
    for record in records:
        path = root / str(record["path"])
        if not path.is_file():
            raise ValueError(f"Inventory artifact is missing: {path}")
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise ValueError(f"Inventory artifact failed size/SHA validation: {path}")
    return inventory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-training-commit", required=True)
    parser.add_argument("--expected-analysis-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    return parser


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    if head != args.expected_analysis_commit:
        raise ValueError(
            f"R8 trajectory controller commit mismatch: expected {args.expected_analysis_commit}, got {head}"
        )
    if _git("status", "--porcelain"):
        raise ValueError("R8 trajectory controller requires a clean detached analysis snapshot.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("R8 trajectory controller refuses a non-empty output root.")
    observed_data = {"train": _sha256(args.train), "dev": _sha256(args.dev)}
    if observed_data != EXPECTED_DATA_SHA:
        raise ValueError(f"R8 trajectory fixed data SHA mismatch: {observed_data}")
    environment_drift = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if environment_drift:
        raise ValueError(f"R8 trajectory strict environment drift: {environment_drift}")

    terminal = _load_json(args.arm_root / "terminal.json")
    if (
        terminal.get("schema") != SOURCE_TERMINAL_SCHEMA
        or terminal.get("status") != "completed_diagnostic"
        or terminal.get("passed") is not True
        or terminal.get("arm") != args.arm
    ):
        raise ValueError("R8 trajectory source arm terminal is not valid.")
    manifest = _load_json(args.arm_root / "run" / "manifest.json")
    if (
        manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
        or manifest.get("arm") != args.arm
        or manifest.get("git_commit") != args.expected_training_commit
        or manifest.get("selected_segments_sha256") != EXPECTED_SELECTED_SHA
    ):
        raise ValueError("R8 trajectory source manifest lineage drifted.")
    _validate_inventory(args.arm_root, expected_schema=SOURCE_INVENTORY_SCHEMA)
    return {
        "analysis_commit": head,
        "analysis_dirty": False,
        "data_sha256": observed_data,
        "environment": dict(EXPECTED_ENVIRONMENT),
        "source_inventory_sha256": _sha256(args.arm_root / "artifact_inventory.json"),
        "source_terminal_sha256": _sha256(args.arm_root / "terminal.json"),
        "evaluator": str(EVALUATOR.resolve()),
        "evaluator_sha256": _sha256(EVALUATOR),
        "python": sys.executable,
        "host": platform.node(),
    }


def _command(args: argparse.Namespace, evaluation_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(EVALUATOR),
        "--arm",
        args.arm,
        "--arm-root",
        str(args.arm_root),
        "--train",
        str(args.train),
        "--dev",
        str(args.dev),
        "--dreamlite",
        str(args.dreamlite),
        "--reader",
        str(args.reader),
        "--output-dir",
        str(evaluation_dir),
        "--expected-training-commit",
        args.expected_training_commit,
        "--seed",
        str(args.seed),
        "--dreamlite-device",
        args.dreamlite_device,
        "--reader-device",
        args.reader_device,
        "--bootstrap-iterations",
        "10000",
    ]


def _evaluation_checks(
    evaluation_dir: Path,
    *,
    arm: str,
    expected_training_commit: str,
    expected_analysis_commit: str,
) -> dict[str, bool]:
    summary_path = evaluation_dir / "trajectory_summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else None
    inventory_valid = False
    if (evaluation_dir / "artifact_inventory.json").is_file():
        try:
            _validate_inventory(evaluation_dir, expected_schema=TRAJECTORY_INVENTORY_SCHEMA)
            inventory_valid = True
        except (KeyError, TypeError, ValueError):
            inventory_valid = False
    return {
        "summary_exists": summary_path.is_file(),
        "summary_schema": isinstance(summary, Mapping) and summary.get("schema") == TRAJECTORY_SCHEMA,
        "summary_completed": isinstance(summary, Mapping) and summary.get("status") == "completed",
        "formal_success_not_claimed": isinstance(summary, Mapping)
        and summary.get("formal_success_claim") is False,
        "arm_matches": isinstance(summary, Mapping) and summary.get("arm") == arm,
        "aggregation_matches": isinstance(summary, Mapping)
        and summary.get("gradient_aggregation") == EXPECTED_MODE[arm],
        "training_commit_matches": isinstance(summary, Mapping)
        and summary.get("training_commit") == expected_training_commit,
        "analysis_commit_matches": isinstance(summary, Mapping)
        and summary.get("analysis_commit") == expected_analysis_commit,
        "selected_segments_match": isinstance(summary, Mapping)
        and summary.get("selected_segments_sha256") == EXPECTED_SELECTED_SHA,
        "row_count_exact": isinstance(summary, Mapping) and summary.get("rows") == 640,
        "endpoint_binding_passed": isinstance(summary, Mapping)
        and isinstance(summary.get("endpoint_binding"), Mapping)
        and summary["endpoint_binding"].get("passed") is True,
        "artifact_inventory_valid": inventory_valid,
    }


def _controller_inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "controller_artifact_inventory.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validated = _validate(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output_root.mkdir(parents=True, exist_ok=True)
    evaluation_dir = args.output_root / "evaluation"
    command = _command(args, evaluation_dir)
    launch = {
        "schema": LAUNCH_SCHEMA,
        "status": "running",
        "started_at_utc": _utc_now(),
        "arm": args.arm,
        "gradient_aggregation": EXPECTED_MODE[args.arm],
        "seed": args.seed,
        "expected_training_commit": args.expected_training_commit,
        "command": command,
        **validated,
    }
    _write_json(args.output_root / "launch.json", launch)
    stdout_path = args.output_root / "stdout.log"
    stderr_path = args.output_root / "stderr.log"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr)
    checks = {
        "child_exit_zero": result.returncode == 0,
        **_evaluation_checks(
            evaluation_dir,
            arm=args.arm,
            expected_training_commit=args.expected_training_commit,
            expected_analysis_commit=args.expected_analysis_commit,
        ),
    }
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": "completed_descriptive_analysis" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scientific_success_claim": False,
        "checkpoint_selection_allowed": False,
        "primary_endpoint": "ema_step128",
        "arm": args.arm,
        "child_exit_code": result.returncode,
        "checks": checks,
        "started_at_utc": launch["started_at_utc"],
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
    }
    _write_json(args.output_root / "terminal.json", terminal)
    _write_json(
        args.output_root / "controller_artifact_inventory.json",
        {"schema": CONTROLLER_INVENTORY_SCHEMA, "artifacts": _controller_inventory(args.output_root)},
    )
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "passed": terminal["passed"],
                "arm": args.arm,
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0 if terminal["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
