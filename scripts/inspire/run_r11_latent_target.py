"""Fail-closed Inspire controller for one activated R11 VAE-latent target."""

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

from scripts.experiments import compare_r10_visual_alignment as r10  # noqa: E402
from scripts.reporting import render_r10_raw_endpoint_attribution as raw  # noqa: E402
from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_SELECTED_SEGMENTS_SHA256,
    R10_TARGET_IDS,
)
from vision_memory.training.r11_latent import (  # noqa: E402
    R11_CHECKPOINT_STEPS,
    R11_OPTIMIZER_STEPS,
)


TRAINER = ROOT / "scripts" / "train" / "latent_r11_vae_oracle.py"
SUMMARY_SCHEMA = "vision_memory.r11-vae-latent-oracle-summary.v1"
SUMMARY_FILE = "r11_latent_summary.json"
LAUNCH_SCHEMA = "vision_memory.r11-latent-target-launch.v1"
TERMINAL_SCHEMA = "vision_memory.r11-latent-target-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r11-latent-target-inventory.v1"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.strip()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _validate_delivery(path: Path, *, schema: str, required_name: str) -> dict[str, Any]:
    delivery = _load(path)
    if delivery.get("schema") != schema:
        raise ValueError(f"R11 parent delivery schema mismatch: {path}")
    records = delivery.get("artifacts")
    if not isinstance(records, list):
        raise ValueError(f"R11 parent delivery artifact list is missing: {path}")
    matches = [record for record in records if isinstance(record, Mapping) and record.get("path") == required_name]
    if len(matches) != 1:
        raise ValueError(f"R11 parent delivery does not bind {required_name} exactly once.")
    artifact = path.parent / required_name
    record = matches[0]
    if artifact.stat().st_size != int(record.get("bytes", -1)) or _sha256(artifact) != record.get("sha256"):
        raise ValueError(f"R11 parent delivery size/SHA binding failed: {artifact}")
    return delivery


def _validate_r10_parent(path: Path) -> dict[str, Any]:
    comparison = _load(path)
    if (
        comparison.get("schema") != r10.COMPARISON_SCHEMA
        or comparison.get("status") != "completed"
        or comparison.get("formal_success_claim") is not False
        or comparison.get("git_commit") != r10.EXPECTED_GIT_COMMIT
        or comparison.get("selected_segments_sha256") != R10_SELECTED_SEGMENTS_SHA256
        or comparison.get("arm_pass_counts")
        != {"direct-pixel-oracle": 8, "dreamlite-single-set": 0}
        or comparison.get("decision") != "redesign_dreamlite_updater_only"
    ):
        raise ValueError("R11 activation requires the exact completed R10 8/0 branch.")
    arms = comparison.get("arms")
    if not isinstance(arms, Mapping):
        raise ValueError("R11 R10 parent lacks arm records.")
    for arm in r10.ARMS:
        embedded = arms.get(arm, {}).get("targets") if isinstance(arms.get(arm), Mapping) else None
        if not isinstance(embedded, list) or len(embedded) != 8:
            raise ValueError(f"R11 R10 parent arm is incomplete: {arm}")
        for index, target in enumerate(embedded):
            if not isinstance(target, Mapping):
                raise ValueError(f"R11 R10 parent target is malformed: {arm}:{index}")
            observed = r10._validate_target(Path(str(target.get("source_root"))), arm, index)
            if observed != dict(target):
                raise ValueError(f"R11 R10 embedded/source target differs: {arm}:{index}")
    delivery = _validate_delivery(
        path.parent / "DELIVERY_MANIFEST.json",
        schema=r10.DELIVERY_SCHEMA,
        required_name="comparison.json",
    )
    expected_inventories = {
        arm: {
            str(target["target_index"]): target["inventory_sha256"]
            for target in comparison["arms"][arm]["targets"]
        }
        for arm in r10.ARMS
    }
    if delivery.get("source_inventory_sha256") != expected_inventories:
        raise ValueError("R11 R10 parent delivery/source inventory binding drifted.")
    return comparison


def _validate_raw_parent(path: Path) -> dict[str, Any]:
    analysis = _load(path)
    if (
        analysis.get("schema") != raw.ANALYSIS_SCHEMA
        or analysis.get("status") != "completed"
        or analysis.get("formal_success_claim") is not False
        or analysis.get("cannot_replace_preregistered_ema_endpoint") is not True
        or analysis.get("analysis_git_commit") != raw.ANALYSIS_COMMIT
        or analysis.get("source_training_git_commit") != r10.EXPECTED_GIT_COMMIT
        or analysis.get("selected_segments_sha256") != R10_SELECTED_SEGMENTS_SHA256
        or analysis.get("raw_pass_count") != 0
        or analysis.get("ema_pass_count") != 0
        or analysis.get("decision") != "ema_is_not_sufficient_explanation_run_vae_latent_oracle"
    ):
        raise ValueError("R11 activation requires the exact completed raw 0/8 branch.")
    targets = analysis.get("targets")
    if not isinstance(targets, list) or len(targets) != 8:
        raise ValueError("R11 raw parent target list is incomplete.")
    for index, embedded in enumerate(targets):
        if not isinstance(embedded, Mapping):
            raise ValueError(f"R11 raw parent target is malformed: {index}")
        observed = raw._validate_target(Path(str(embedded.get("attribution_root"))), index)
        if observed != dict(embedded):
            raise ValueError(f"R11 raw embedded/source target differs: {index}")
    delivery = _validate_delivery(
        path.parent / "DELIVERY_MANIFEST.json",
        schema=raw.DELIVERY_SCHEMA,
        required_name="ANALYSIS.json",
    )
    expected = {str(target["target_index"]): target["inventory_sha256"] for target in targets}
    if delivery.get("source_inventory_sha256") != expected:
        raise ValueError("R11 raw parent delivery/source inventory binding drifted.")
    return analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index", type=int, choices=range(8), required=True)
    parser.add_argument("--r10-comparison", type=Path, required=True)
    parser.add_argument("--raw-analysis", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    return parser


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    if head != args.expected_commit:
        raise ValueError(f"R11 controller commit mismatch: expected {args.expected_commit}, got {head}")
    if _git("status", "--porcelain"):
        raise ValueError("R11 controller requires a clean experiment snapshot.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("R11 controller refuses a non-empty output root.")
    if not args.dreamlite.is_dir() or not args.reader.is_dir():
        raise ValueError("R11 model snapshot path is missing.")
    observed_data = {"train": _sha256(args.train), "dev": _sha256(args.dev)}
    if observed_data != EXPECTED_DATA_SHA:
        raise ValueError(f"R11 fixed data SHA mismatch: {observed_data}")
    drift = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if drift:
        raise ValueError(f"R11 strict environment drift: {drift}")
    r10_parent = _validate_r10_parent(args.r10_comparison)
    raw_parent = _validate_raw_parent(args.raw_analysis)
    return {
        "git_commit": head,
        "git_dirty": False,
        "data_sha256": observed_data,
        "environment": dict(EXPECTED_ENVIRONMENT),
        "target_segment_id": R10_TARGET_IDS[args.target_index],
        "r10_comparison_sha256": _sha256(args.r10_comparison),
        "r10_decision": r10_parent["decision"],
        "raw_analysis_sha256": _sha256(args.raw_analysis),
        "raw_decision": raw_parent["decision"],
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
        "0",
        "--dreamlite-device",
        args.dreamlite_device,
        "--reader-device",
        args.reader_device,
        "--strict-determinism",
    ]


def _inventory(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
        if path.name != "artifact_inventory.json"
    ]


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
        "target_index": args.target_index,
        "command": command,
        **validated,
    }
    _write_json(args.output_root / "launch.json", launch)
    stdout_path = args.output_root / "stdout.log"
    stderr_path = args.output_root / "stderr.log"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr)
    summary_path = run_dir / SUMMARY_FILE
    manifest_path = run_dir / "manifest.json"
    summary = _load(summary_path) if summary_path.is_file() else None
    manifest = _load(manifest_path) if manifest_path.is_file() else None
    gates = summary.get("gates", {}) if isinstance(summary, Mapping) else {}
    checks = {
        "child_exit_zero": result.returncode == 0,
        "summary_schema": isinstance(summary, Mapping) and summary.get("schema") == SUMMARY_SCHEMA,
        "summary_completed": isinstance(summary, Mapping) and summary.get("status") == "completed",
        "summary_commit_matches": isinstance(summary, Mapping)
        and summary.get("git_commit") == args.expected_commit,
        "target_matches": isinstance(summary, Mapping)
        and summary.get("target_index") == args.target_index
        and summary.get("target_segment_id") == R10_TARGET_IDS[args.target_index]
        and summary.get("target_family") == "F1",
        "selection_matches": isinstance(summary, Mapping)
        and summary.get("selected_segments") == list(R10_TARGET_IDS)
        and summary.get("selected_segments_sha256") == R10_SELECTED_SEGMENTS_SHA256,
        "fixed_execution_matches": isinstance(summary, Mapping)
        and summary.get("optimizer_steps") == R11_OPTIMIZER_STEPS
        and summary.get("checkpoint_steps_observed") == list(R11_CHECKPOINT_STEPS),
        "technical_gate_passed": gates.get("technical_gate") is True,
        "formal_success_not_claimed": isinstance(summary, Mapping)
        and summary.get("full_success_claim_allowed") is False
        and summary.get("diagnostic_only_not_formal_success") is True
        and gates.get("formal_success_gate") is False,
        "manifest_commit_matches": isinstance(manifest, Mapping)
        and manifest.get("git_commit") == args.expected_commit,
        "manifest_target_matches": isinstance(manifest, Mapping)
        and manifest.get("target_index") == args.target_index
        and manifest.get("target_segment_id") == R10_TARGET_IDS[args.target_index],
        "manifest_data_matches": isinstance(manifest, Mapping)
        and manifest.get("train_sha256") == EXPECTED_DATA_SHA["train"]
        and manifest.get("dev_sha256") == EXPECTED_DATA_SHA["dev"],
    }
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": "completed_diagnostic" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "scientific_success_claim": False,
        "target_latent_reachability_gate": (
            gates.get("target_latent_reachability_gate") if isinstance(gates, Mapping) else None
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
