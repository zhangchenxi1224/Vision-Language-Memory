"""Fail-closed Inspire controller for one conditionally activated R9 target."""

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
TRAINER = ROOT / "scripts" / "train" / "dreamlite_r9_individual_learnability.py"
EXPECTED_DATA_SHA = {
    "train": "24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184",
    "dev": "8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303",
}
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
EXPECTED_R8_TRAINING_COMMIT = "82e983743be73919f257d441f1cacb2b7f601288"
EXPECTED_SELECTED_SHA = "eeade3e006791aeea87aa12cf897956d34b4e2c3769c162db494e42fb7828ea6"
R8_ARMS = ("raw-mean-control", "common-descent-projected-norm-matched")
ACTIVATION_DECISION = "reject_batch_conflict_as_sufficient_test_per_segment_learnability"
TARGETS = (
    ("F2", "r5-f2-d1803aae745ed83f142b0bae"),
    ("F2", "r5-f2-6db739261bbb0fc74169dc1f"),
    ("F3", "r5-f3-dbc9791cf4ba3569afc7a3d0"),
    ("F3", "r5-f3-913c95f41d8e72f9b0aabac4"),
    ("F5", "r5-f5-9fec14ebb444f5f1fe201087"),
    ("F5", "r5-f5-56ef217cc7251269d003b5c8"),
    ("F6", "r5-f6-b6302052a821cfd9e9d9261e"),
    ("F6", "r5-f6-1ccc652bfea4b3e698683232"),
)
SUMMARY_SCHEMA = "vision_memory.r9-individual-learnability-summary.v1"
LAUNCH_SCHEMA = "vision_memory.r9-individual-target-launch.v1"
TERMINAL_SCHEMA = "vision_memory.r9-individual-target-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r9-individual-target-inventory.v1"


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


def _validate_parent(comparison_path: Path) -> dict[str, Any]:
    comparison = _load_json(comparison_path)
    if (
        comparison.get("schema") != "vision_memory.r8-common-descent-comparison.v1"
        or comparison.get("status") != "completed"
        or comparison.get("decision") != ACTIVATION_DECISION
        or comparison.get("formal_success_claim") is not False
    ):
        raise ValueError("R9 activation requires the exact valid R8 joint-failure decision.")
    arms = comparison.get("arms")
    paths = comparison.get("summary_paths")
    hashes = comparison.get("summary_sha256")
    if not all(isinstance(value, Mapping) for value in (arms, paths, hashes)):
        raise ValueError("R9 activation comparison lacks bound R8 arm summaries.")
    for arm in R8_ARMS:
        summary = arms.get(arm)
        if not isinstance(summary, Mapping):
            raise ValueError(f"R9 activation lacks embedded R8 summary: {arm}")
        if (
            summary.get("schema") != "vision_memory.r8-common-descent-summary.v1"
            or summary.get("status") != "completed"
            or summary.get("arm") != arm
            or summary.get("git_commit") != EXPECTED_R8_TRAINING_COMMIT
            or summary.get("selected_segments_sha256") != EXPECTED_SELECTED_SHA
            or summary.get("full_success_claim_allowed") is not False
            or summary.get("gates", {}).get("technical_gate") is not True
            or summary.get("gates", {}).get("hard8_overfit_learnability_gate") is not False
        ):
            raise ValueError(f"R9 activation R8 arm is invalid or did not jointly fail: {arm}")
        summary_path = Path(str(paths.get(arm)))
        if not summary_path.is_file() or _sha256(summary_path) != hashes.get(arm):
            raise ValueError(f"R9 activation R8 source summary binding failed: {arm}")
        if _load_json(summary_path) != dict(summary):
            raise ValueError(f"R9 activation embedded and source R8 summaries differ: {arm}")
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index", type=int, choices=range(8), required=True)
    parser.add_argument("--parent-comparison", type=Path, required=True)
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
    head = _git("rev-parse", "HEAD")
    if head != args.expected_commit:
        raise ValueError(f"R9 controller commit mismatch: expected {args.expected_commit}, got {head}")
    if _git("status", "--porcelain"):
        raise ValueError("R9 controller requires a clean detached experiment snapshot.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("R9 controller refuses a non-empty output root.")
    observed_data = {"train": _sha256(args.train), "dev": _sha256(args.dev)}
    if observed_data != EXPECTED_DATA_SHA:
        raise ValueError(f"R9 fixed data SHA mismatch: {observed_data}")
    environment_drift = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if environment_drift:
        raise ValueError(f"R9 strict environment drift: {environment_drift}")
    parent = _validate_parent(args.parent_comparison)
    family, segment_id = TARGETS[args.target_index]
    return {
        "git_commit": head,
        "git_dirty": False,
        "data_sha256": observed_data,
        "environment": dict(EXPECTED_ENVIRONMENT),
        "target_family": family,
        "target_segment_id": segment_id,
        "parent_comparison_sha256": _sha256(args.parent_comparison),
        "parent_decision": parent["decision"],
        "trainer": str(TRAINER.resolve()),
        "trainer_sha256": _sha256(TRAINER),
        "python": sys.executable,
        "host": platform.node(),
    }


def _command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(TRAINER),
        "--target-index",
        str(args.target_index),
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validated = _validate(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_root / "run"
    command = _command(args, run_dir)
    launch = {
        "schema": LAUNCH_SCHEMA,
        "status": "running",
        "started_at_utc": _utc_now(),
        "target_index": args.target_index,
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
    summary_path = run_dir / "r9_summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else None
    expected_family, expected_segment_id = TARGETS[args.target_index]
    checks = {
        "child_exit_zero": result.returncode == 0,
        "summary_exists": summary_path.is_file(),
        "summary_schema": isinstance(summary, Mapping) and summary.get("schema") == SUMMARY_SCHEMA,
        "summary_completed": isinstance(summary, Mapping) and summary.get("status") == "completed",
        "summary_commit_matches": isinstance(summary, Mapping)
        and summary.get("git_commit") == args.expected_commit,
        "target_index_matches": isinstance(summary, Mapping)
        and summary.get("target_index") == args.target_index,
        "target_segment_matches": isinstance(summary, Mapping)
        and summary.get("target_segment_id") == expected_segment_id
        and summary.get("target_family") == expected_family,
        "hard8_matches": isinstance(summary, Mapping)
        and summary.get("selected_segments_sha256") == EXPECTED_SELECTED_SHA,
        "gradient_contract_matches": isinstance(summary, Mapping)
        and summary.get("gradient_aggregation") == "single-target-one-eighth"
        and summary.get("gradient_coefficient") == 0.125,
        "technical_gate_passed": isinstance(summary, Mapping)
        and summary.get("gates", {}).get("technical_gate") is True,
        "formal_success_not_claimed": isinstance(summary, Mapping)
        and summary.get("full_success_claim_allowed") is False
        and summary.get("gates", {}).get("formal_success_gate") is False,
    }
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": "completed_diagnostic" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scientific_success_claim": False,
        "target_individual_learnability_gate": (
            summary.get("gates", {}).get("target_individual_learnability_gate")
            if isinstance(summary, Mapping)
            else None
        ),
        "target_index": args.target_index,
        "target_segment_id": expected_segment_id,
        "child_exit_code": result.returncode,
        "checks": checks,
        "started_at_utc": launch["started_at_utc"],
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "summary_sha256": _sha256(summary_path) if summary_path.is_file() else None,
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
    }
    _write_json(args.output_root / "terminal.json", terminal)
    _write_json(
        args.output_root / "artifact_inventory.json",
        {
            "schema": INVENTORY_SCHEMA,
            "root": str(args.output_root.resolve()),
            "artifacts": _inventory(args.output_root),
        },
    )
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    return 0 if terminal["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
