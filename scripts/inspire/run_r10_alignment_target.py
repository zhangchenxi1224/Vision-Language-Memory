"""Fail-closed Inspire controller for one conditionally activated R10 target arm."""

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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.experiments import compare_r9_individual_learnability as r9_compare  # noqa: E402
from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_SELECTED_SEGMENTS_SHA256,
    R10_TARGET_IDS,
)


TRAINERS = {
    "direct-pixel-oracle": ROOT / "scripts" / "train" / "pixel_r10_visual_oracle.py",
    "dreamlite-single-set": ROOT / "scripts" / "train" / "dreamlite_r10_single_set.py",
}
SUMMARY_SCHEMAS = {
    "direct-pixel-oracle": "vision_memory.r10-direct-pixel-oracle-summary.v1",
    "dreamlite-single-set": "vision_memory.r10-dreamlite-single-set-summary.v1",
}
SUMMARY_FILES = {
    "direct-pixel-oracle": "r10_pixel_summary.json",
    "dreamlite-single-set": "r10_dreamlite_summary.json",
}
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
EXPECTED_R9_TRAINING_COMMIT = "0eddfa273ae159deac8304db37f3c2a7baf04cee"
R9_COMPARISON_SCHEMA = "vision_memory.r9-individual-learnability-comparison.v1"
R9_DELIVERY_SCHEMA = "vision_memory.r9-individual-learnability-delivery.v1"
LAUNCH_SCHEMA = "vision_memory.r10-alignment-target-launch.v1"
TERMINAL_SCHEMA = "vision_memory.r10-alignment-target-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r10-alignment-target-inventory.v1"


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


def _validate_delivery(comparison_path: Path, comparison: Mapping[str, Any]) -> dict[str, Any]:
    delivery_path = comparison_path.parent / "DELIVERY_MANIFEST.json"
    delivery = _load_json(delivery_path)
    if delivery.get("schema") != R9_DELIVERY_SCHEMA:
        raise ValueError("R10 activation requires the exact R9 delivery manifest schema.")
    artifacts = delivery.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("R10 activation R9 delivery artifact list is missing.")
    bound = [value for value in artifacts if isinstance(value, Mapping) and value.get("path") == comparison_path.name]
    if len(bound) != 1:
        raise ValueError("R10 activation R9 delivery does not bind comparison.json exactly once.")
    record = bound[0]
    if record.get("sha256") != _sha256(comparison_path) or int(record.get("bytes", -1)) != comparison_path.stat().st_size:
        raise ValueError("R10 activation R9 comparison failed its delivery size/SHA binding.")
    expected_sources = {
        str(value["target_index"]): value["inventory_sha256"]
        for value in comparison["targets"]
    }
    if delivery.get("source_inventory_sha256") != expected_sources:
        raise ValueError("R10 activation R9 delivery/source inventory bindings drifted.")
    return delivery


def _validate_parent(comparison_path: Path) -> dict[str, Any]:
    comparison = _load_json(comparison_path)
    if (
        comparison.get("schema") != R9_COMPARISON_SCHEMA
        or comparison.get("status") != "completed"
        or comparison.get("formal_success_claim") is not False
        or comparison.get("selected_segments_sha256") != r9_compare.SELECTED_SEGMENTS_SHA256
        or comparison.get("git_commit") != EXPECTED_R9_TRAINING_COMMIT
    ):
        raise ValueError("R10 activation requires a complete, exact, non-formal R9 comparison.")
    targets = comparison.get("targets")
    if not isinstance(targets, list) or len(targets) != 8:
        raise ValueError("R10 activation requires all eight validated R9 target records.")
    ordered = sorted(targets, key=lambda value: int(value.get("target_index", -1)))
    if [value.get("target_index") for value in ordered] != list(range(8)):
        raise ValueError("R10 activation R9 target indices are incomplete or duplicated.")
    revalidated: list[dict[str, Any]] = []
    for index, embedded in enumerate(ordered):
        source_root = Path(str(embedded.get("source_root")))
        observed = r9_compare._validate_target(source_root, index)
        if observed != dict(embedded):
            raise ValueError(f"R10 activation embedded/source R9 target differs: {index}")
        revalidated.append(observed)
    pass_count = sum(bool(value["passed"]) for value in revalidated)
    if comparison.get("pass_count") != pass_count:
        raise ValueError("R10 activation R9 pass count does not match revalidated targets.")
    expected_decision, _reason = r9_compare._decision(pass_count)
    if comparison.get("decision") != expected_decision:
        raise ValueError("R10 activation R9 decision does not match the fixed pass boundary.")
    if pass_count == 8:
        raise ValueError("R10 is inactive because all eight valid R9 targets passed.")
    failed = [value["target_index"] for value in revalidated if not value["passed"]]
    passed = [value["target_index"] for value in revalidated if value["passed"]]
    if comparison.get("failed_target_indices") != failed or comparison.get("passed_target_indices") != passed:
        raise ValueError("R10 activation R9 pass/fail target lists drifted.")
    _validate_delivery(comparison_path, comparison)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(TRAINERS), required=True)
    parser.add_argument("--target-index", type=int, choices=range(8), required=True)
    parser.add_argument("--parent-comparison", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    return parser


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    if head != args.expected_commit:
        raise ValueError(f"R10 controller commit mismatch: expected {args.expected_commit}, got {head}")
    if _git("status", "--porcelain"):
        raise ValueError("R10 controller requires a clean detached experiment snapshot.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("R10 controller refuses a non-empty output root.")
    if args.arm == "dreamlite-single-set" and (args.dreamlite is None or not args.dreamlite.is_dir()):
        raise ValueError("R10 DreamLite arm requires a valid --dreamlite directory.")
    observed_data = {"train": _sha256(args.train), "dev": _sha256(args.dev)}
    if observed_data != EXPECTED_DATA_SHA:
        raise ValueError(f"R10 fixed data SHA mismatch: {observed_data}")
    environment_drift = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if environment_drift:
        raise ValueError(f"R10 strict environment drift: {environment_drift}")
    parent = _validate_parent(args.parent_comparison)
    trainer = TRAINERS[args.arm]
    return {
        "git_commit": head,
        "git_dirty": False,
        "data_sha256": observed_data,
        "environment": dict(EXPECTED_ENVIRONMENT),
        "target_segment_id": R10_TARGET_IDS[args.target_index],
        "parent_comparison_sha256": _sha256(args.parent_comparison),
        "parent_decision": parent["decision"],
        "parent_pass_count": parent["pass_count"],
        "trainer": str(trainer.resolve()),
        "trainer_sha256": _sha256(trainer),
        "python": sys.executable,
        "host": platform.node(),
    }


def _command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(TRAINERS[args.arm]),
        "--target-index",
        str(args.target_index),
        "--train",
        str(args.train),
        "--dev",
        str(args.dev),
        "--reader",
        str(args.reader),
        "--output-dir",
        str(run_dir),
        "--seed",
        str(args.seed),
        "--reader-device",
        args.reader_device,
        "--strict-determinism",
    ]
    if args.arm == "dreamlite-single-set":
        command.extend(
            (
                "--dreamlite",
                str(args.dreamlite),
                "--dreamlite-device",
                args.dreamlite_device,
            )
        )
    return command


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


def _summary_checks(
    *,
    args: argparse.Namespace,
    summary: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    child_exit_code: int,
) -> dict[str, bool]:
    gates = summary.get("gates", {}) if isinstance(summary, Mapping) else {}
    checks = {
        "child_exit_zero": child_exit_code == 0,
        "summary_schema": isinstance(summary, Mapping)
        and summary.get("schema") == SUMMARY_SCHEMAS[args.arm],
        "summary_completed": isinstance(summary, Mapping) and summary.get("status") == "completed",
        "summary_commit_matches": isinstance(summary, Mapping)
        and summary.get("git_commit") == args.expected_commit,
        "arm_matches": isinstance(summary, Mapping) and summary.get("arm") == args.arm,
        "target_matches": isinstance(summary, Mapping)
        and summary.get("target_index") == args.target_index
        and summary.get("target_segment_id") == R10_TARGET_IDS[args.target_index]
        and summary.get("target_family") == "F1",
        "selected_targets_match": isinstance(summary, Mapping)
        and summary.get("selected_segments") == list(R10_TARGET_IDS)
        and summary.get("selected_segments_sha256") == R10_SELECTED_SEGMENTS_SHA256,
        "fixed_execution_matches": isinstance(summary, Mapping)
        and summary.get("checkpoint_steps_observed") == [0, 32, 64, 96, 128],
        "technical_gate_passed": gates.get("technical_gate") is True,
        "formal_success_not_claimed": isinstance(summary, Mapping)
        and summary.get("full_success_claim_allowed") is False
        and summary.get("diagnostic_only_not_formal_success") is True
        and gates.get("formal_success_gate") is False,
        "manifest_exists": isinstance(manifest, Mapping),
        "manifest_commit_matches": isinstance(manifest, Mapping)
        and manifest.get("git_commit") == args.expected_commit,
        "manifest_arm_matches": isinstance(manifest, Mapping)
        and manifest.get("arm") == args.arm,
        "manifest_target_matches": isinstance(manifest, Mapping)
        and manifest.get("target_index") == args.target_index
        and manifest.get("target_segment_id") == R10_TARGET_IDS[args.target_index],
        "manifest_data_matches": isinstance(manifest, Mapping)
        and manifest.get("train_sha256") == EXPECTED_DATA_SHA["train"]
        and manifest.get("dev_sha256") == EXPECTED_DATA_SHA["dev"],
    }
    if args.arm == "direct-pixel-oracle":
        checks["pixel_optimizer_steps_match"] = (
            isinstance(summary, Mapping) and summary.get("optimizer_steps") == 128
        )
    else:
        checks["dreamlite_gradient_contract_matches"] = bool(
            isinstance(summary, Mapping)
            and summary.get("gradient_aggregation") == "single-target-full"
            and summary.get("gradient_coefficient") == 1.0
        )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validated = _validate(args)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_root / "run"
    command = _command(args, run_dir)
    launch = {
        "schema": LAUNCH_SCHEMA,
        "status": "running",
        "started_at_utc": _utc_now(),
        "arm": args.arm,
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
    summary_path = run_dir / SUMMARY_FILES[args.arm]
    manifest_path = run_dir / "manifest.json"
    summary = _load_json(summary_path) if summary_path.is_file() else None
    manifest = _load_json(manifest_path) if manifest_path.is_file() else None
    checks = _summary_checks(
        args=args,
        summary=summary,
        manifest=manifest,
        child_exit_code=result.returncode,
    )
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": "completed_diagnostic" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scientific_success_claim": False,
        "arm": args.arm,
        "target_lower_bound_gate": (
            summary.get("gates", {}).get("target_lower_bound_gate")
            if isinstance(summary, Mapping)
            else None
        ),
        "target_index": args.target_index,
        "target_segment_id": R10_TARGET_IDS[args.target_index],
        "child_exit_code": result.returncode,
        "checks": checks,
        "started_at_utc": launch["started_at_utc"],
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "summary_sha256": _sha256(summary_path) if summary_path.is_file() else None,
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
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
