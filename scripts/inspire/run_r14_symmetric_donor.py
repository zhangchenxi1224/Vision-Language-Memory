"""Fail-closed Inspire controller for the preregistered R14 diagnostic."""

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

from scripts.train import r14_symmetric_donor_writer as trainer  # noqa: E402
from vision_memory.training.r12_shared_writer import (  # noqa: E402
    R12_CHECKPOINT_STEPS,
    R12_DEV_FINAL_SHA256,
    R12_DEV_SELECT_SHA256,
    R12_MICRO_STEPS,
    R12_OPTIMIZER_STEPS,
    R12_TRAIN_AUDIT_SHA256,
    R12_TRAIN_SELECTION_SHA256,
)
from vision_memory.training.r14_symmetric_donor import (  # noqa: E402
    R14_FRESH_DEV_FINAL_SHA256,
    R14_PAIR_SEED,
)


LAUNCH_SCHEMA = "vision_memory.r14-symmetric-donor-ranking-launch.v1"
TERMINAL_SCHEMA = "vision_memory.r14-symmetric-donor-ranking-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r14-symmetric-donor-ranking-inventory.v1"
SUMMARY_FILE = "r14_symmetric_donor_summary.json"
EXPECTED_DATA_SHA = {
    "train": "24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184",
    "dev": "8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303",
}
EXPECTED_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256": ("1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159"),
    "VLM_READER_SNAPSHOT_MANIFEST_SHA256": ("159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c"),
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


def _text_sha256_lf(path: Path) -> str:
    """Hash repository text identically on Windows and Linux checkouts."""

    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R14 controller expected a JSON object: {path}")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--r12-conditioned-root", type=Path, required=True)
    parser.add_argument("--r12-control-root", type=Path, required=True)
    parser.add_argument("--r12-comparison", type=Path, required=True)
    parser.add_argument("--r12-collapse-audit", type=Path, required=True)
    parser.add_argument("--r13-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    return parser


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    if head != args.expected_commit:
        raise ValueError(f"R14 controller commit mismatch: expected {args.expected_commit}, got {head}")
    if _git("status", "--porcelain"):
        raise ValueError("R14 controller requires a clean experiment snapshot.")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise ValueError("R14 controller refuses a non-empty output root.")
    for name in ("train", "dev", "r12_comparison", "r12_collapse_audit"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R14 controller file is missing: {name}")
    for name in ("dreamlite", "reader", "r12_conditioned_root", "r12_control_root", "r13_root"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R14 controller directory is missing: {name}")
    observed_data = {"train": _sha256(args.train), "dev": _sha256(args.dev)}
    if observed_data != EXPECTED_DATA_SHA:
        raise ValueError(f"R14 controller data hash drift: {observed_data}")
    drift = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if drift:
        raise ValueError(f"R14 strict environment drift: {drift}")
    config = _load(trainer.CONFIG_PATH)
    if (
        config.get("schema") != "vision_memory.r14-symmetric-donor-ranking-config.v1"
        or config.get("status") != "preregistered_before_any_r14_model_outcome"
    ):
        raise ValueError("R14 controller requires the exact preregistered config.")
    if not trainer.R14_PREREG_PATH.is_file() or _text_sha256_lf(trainer.R14_PREREG_PATH) != config.get(
        "preregistration", {}
    ).get("sha256"):
        raise ValueError("R14 controller detected preregistration artifact drift.")
    return {
        "git_commit": head,
        "git_dirty": False,
        "data_sha256": observed_data,
        "environment": dict(EXPECTED_ENVIRONMENT),
        "trainer": str(trainer.__file__),
        "trainer_sha256": _sha256(Path(trainer.__file__)),
        "config_sha256": _sha256(trainer.CONFIG_PATH),
        "preregistration_sha256": _text_sha256_lf(trainer.R14_PREREG_PATH),
        "pairing_seed": config["training_pairing"]["seed"],
        "pairing_sha256": config["training_pairing"]["pairs_sha256"],
        "python": sys.executable,
        "host": platform.node(),
    }


def _command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(trainer.__file__).resolve()),
        "--train",
        str(args.train),
        "--dev",
        str(args.dev),
        "--dreamlite",
        str(args.dreamlite),
        "--reader",
        str(args.reader),
        "--r12-conditioned-root",
        str(args.r12_conditioned_root),
        "--r12-control-root",
        str(args.r12_control_root),
        "--r12-comparison",
        str(args.r12_comparison),
        "--r12-collapse-audit",
        str(args.r12_collapse_audit),
        "--r13-root",
        str(args.r13_root),
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
        "command": command,
        **validated,
    }
    _write_json(args.output_root / "launch.json", launch)
    stdout_path = args.output_root / "stdout.log"
    stderr_path = args.output_root / "stderr.log"
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stdout,
            stderr=stderr,
        )
    summary_path = run_dir / SUMMARY_FILE
    manifest_path = run_dir / "manifest.json"
    technical_path = run_dir / "technical_gate.json"
    summary = _load(summary_path) if summary_path.is_file() else None
    manifest = _load(manifest_path) if manifest_path.is_file() else None
    technical = _load(technical_path) if technical_path.is_file() else None
    selections = summary.get("selection_audits", {}) if isinstance(summary, Mapping) else {}
    gates = summary.get("gates", {}) if isinstance(summary, Mapping) else {}
    pairing = summary.get("training_pairing", {}) if isinstance(summary, Mapping) else {}
    execution_checks = {
        "child_exit_zero": result.returncode == 0,
        "summary_schema": isinstance(summary, Mapping) and summary.get("schema") == trainer.SUMMARY_SCHEMA,
        "summary_completed": isinstance(summary, Mapping) and summary.get("status") == "completed",
        "summary_commit_matches": isinstance(summary, Mapping) and summary.get("git_commit") == args.expected_commit,
        "fixed_execution_matches": isinstance(summary, Mapping)
        and summary.get("micro_steps") == R12_MICRO_STEPS
        and summary.get("optimizer_steps") == R12_OPTIMIZER_STEPS
        and summary.get("checkpoint_steps_observed") == list(R12_CHECKPOINT_STEPS)
        and summary.get("endpoint") == trainer.PRIMARY_ENDPOINT,
        "selection_hashes_match": isinstance(selections, Mapping)
        and selections.get("train", {}).get("payload_sha256") == R12_TRAIN_SELECTION_SHA256
        and selections.get("train_audit", {}).get("payload_sha256") == R12_TRAIN_AUDIT_SHA256
        and selections.get("dev_select", {}).get("payload_sha256") == R12_DEV_SELECT_SHA256
        and selections.get("dev_replay", {}).get("payload_sha256") == R12_DEV_FINAL_SHA256
        and selections.get("dev_final", {}).get("payload_sha256") == R14_FRESH_DEV_FINAL_SHA256,
        "symmetric_pairing_matches": isinstance(pairing, Mapping)
        and pairing.get("schema") == "vision_memory.r14-symmetric-donor-pairing-audit.v1"
        and pairing.get("seed") == validated["pairing_seed"] == R14_PAIR_SEED
        and pairing.get("pairs_sha256") == validated["pairing_sha256"]
        and pairing.get("pair_count") == 72
        and pairing.get("different_target_value") is True
        and pairing.get("involution") is True,
        "technical_gate_passed": isinstance(technical, Mapping)
        and technical.get("schema") == trainer.TECHNICAL_GATE_SCHEMA
        and technical.get("passed") is True
        and gates.get("technical_gate") is True,
        "formal_success_not_claimed": isinstance(summary, Mapping)
        and summary.get("full_success_claim_allowed") is False
        and summary.get("diagnostic_only_not_formal_success") is True
        and gates.get("formal_success_gate") is False,
        "manifest_matches": isinstance(manifest, Mapping)
        and manifest.get("schema") == trainer.MANIFEST_SCHEMA
        and manifest.get("git_commit") == args.expected_commit,
    }
    execution_passed = all(execution_checks.values())
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": "completed_diagnostic" if execution_passed else "failed",
        "passed": execution_passed,
        "scientific_arm_gate": gates.get("arm_gate") if isinstance(gates, Mapping) else None,
        "formal_success_claim": False,
        "child_exit_code": result.returncode,
        "execution_checks": execution_checks,
        "started_at_utc": launch["started_at_utc"],
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "summary_sha256": _sha256(summary_path) if summary_path.is_file() else None,
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "technical_gate_sha256": _sha256(technical_path) if technical_path.is_file() else None,
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
    return 0 if execution_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
