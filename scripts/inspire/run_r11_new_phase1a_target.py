"""Fail-closed Inspire controller for one preregistered R11_new Phase 1A target.

This controller intentionally exposes no optimizer, scheduler, sigma, or
checkpoint knobs.  Those values live only in the immutable Phase 1A config.
It treats query-level reachability as a diagnostic result: either ``True`` or
``False`` is a technically valid formal run and neither is formal scientific
success.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from vision_memory.training import r11_new_oracle as core  # noqa: E402
from vision_memory.data import REVERSE_CYCLIC4  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402


TRAINER = ROOT / "scripts" / "train" / "r11_new_frozen_dreamlite_oracle.py"
CONFIG_PATH = (
    ROOT / "configs" / "experiments" / "r11_new_frozen_dreamlite_oracle_phase1a.json"
)
SUMMARY_FILE = "r11_new_phase1a_summary.json"
PREFLIGHT_FILE = "technical_preflight.json"
METRICS_FILE = "metrics.jsonl"
MANIFEST_FILE = "manifest.json"
TECHNICAL_GATE_FILE = "technical_gate.json"
EVALUATION_ROWS_FILE = "target_evaluation_rows.jsonl"

SUMMARY_SCHEMA = "vision_memory.r11-new-phase1a-summary.v1"
MANIFEST_SCHEMA = "vision_memory.r11-new-phase1a-manifest.v1"
METRICS_SCHEMA = "vision_memory.r11-new-phase1a-metrics.v1"
TECHNICAL_GATE_SCHEMA = "vision_memory.r11-new-phase1a-technical-gate.v1"
CHECKPOINT_HASH_SCHEMA = "vision_memory.r11-new-phase1a-checkpoint-hashes.v1"
EVALUATION_SCHEMA = "vision_memory.r5-compose-causal-evaluation.v1"
EVALUATION_SUITE = "r11_new_f1_frozen_dreamlite_oracle"
LAUNCH_SCHEMA = "vision_memory.r11-new-phase1a-target-launch.v1"
TERMINAL_SCHEMA = "vision_memory.r11-new-phase1a-target-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r11-new-phase1a-target-inventory.v1"
LOCK_SCHEMA = "vision_memory.r11-new-phase1a-suite-lock.v1"
TRAINER_TERMINAL_SCHEMA = "vision_memory.r11-new-phase1a-terminal.v1"
TRAINER_INVENTORY_SCHEMA = "vision_memory.r11-new-phase1a-artifact-inventory.v1"
TRAINER_CHECKPOINT_SCHEMA = "vision_memory.r11-new-phase1a-checkpoint.v1"
TRAINER_OUTPUT_OWNER_SCHEMA = "vision_memory.r11-new-phase1a-output-owner.v1"

EXPECTED_HOST_PREFIX = "vlm-r3-h200x2-live-20260717"
INSPIRE_SSD_ROOT = Path("/inspire/ssd")
MINIMUM_FREE_BYTES = 50 * 1024**3
LOCK_ROOT = Path("/tmp")

CANONICAL_R11_SCHEMA = "vision_memory.r11-vae-latent-reachability-comparison.v1"
CANONICAL_R11_STATUS = "completed"
CANONICAL_R11_PASS_COUNT = 8
CANONICAL_R11_DECISION = "replace_semantic_editor_with_shared_event_to_latent_writer"

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

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_COMMAND_OPTIONS = {
    "--learning-rate",
    "--lr",
    "--optimizer",
    "--optimizer-steps",
    "--steps",
    "--weight-decay",
    "--gradient-clipping",
    "--gradient-clip",
    "--sigma",
    "--effective-sigma",
    "--diffusion-steps",
    "--checkpoint-steps",
    "--seed",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R11_new controller expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"R11_new metrics contain a blank row at line {line_number}.")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"R11_new metrics row {line_number} is not an object.")
            rows.append(value)
    return rows


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


def _normalise_device(value: str) -> str:
    normalised = value.strip().lower()
    return "cuda:0" if normalised == "cuda" else normalised


def _require_fresh_root(path: Path) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError("R11_new controller output root exists and is not a directory.")
    if path.exists() and any(path.iterdir()):
        raise ValueError("R11_new controller refuses a non-empty output root.")


def _deployment_audit(
    output_root: Path,
    *,
    ssd_root: Path = INSPIRE_SSD_ROOT,
    hostname: str | None = None,
) -> dict[str, Any]:
    observed_host = platform.node() if hostname is None else hostname
    if not observed_host.startswith(EXPECTED_HOST_PREFIX):
        raise ValueError(
            "R11_new controller is pinned to the requested Inspire instance: "
            f"expected hostname prefix {EXPECTED_HOST_PREFIX!r}, observed {observed_host!r}."
        )
    resolved_root = ssd_root.resolve()
    resolved_output = output_root.resolve()
    try:
        resolved_output.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"R11_new output root must be under {resolved_root}, observed {resolved_output}."
        ) from exc
    existing_parent = resolved_output
    while not existing_parent.exists():
        parent = existing_parent.parent
        if parent == existing_parent:
            raise ValueError(f"R11_new cannot locate an existing storage parent for {resolved_output}.")
        existing_parent = parent
    if not existing_parent.is_dir():
        raise ValueError(f"R11_new nearest existing storage parent is not a directory: {existing_parent}")
    usage = shutil.disk_usage(existing_parent)
    if usage.free < MINIMUM_FREE_BYTES:
        raise ValueError(
            "R11_new Inspire SSD free-space gate failed: "
            f"required >= {MINIMUM_FREE_BYTES} bytes, observed {usage.free} bytes at {existing_parent}."
        )
    return {
        "hostname": observed_host,
        "required_hostname_prefix": EXPECTED_HOST_PREFIX,
        "ssd_root": str(resolved_root),
        "output_root": str(resolved_output),
        "disk_usage_path": str(existing_parent),
        "disk_total_bytes": usage.total,
        "disk_used_bytes": usage.used,
        "disk_free_bytes": usage.free,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "passed": True,
    }


def _suite_lock_path(*, mode: str, target_index: int, lock_root: Path | None = None) -> Path:
    root = LOCK_ROOT if lock_root is None else lock_root
    # One two-GPU instance can safely host only one Phase 1A process at a time.
    # ``mode`` and ``target_index`` remain explicit inputs for call-site audit,
    # but deliberately do not partition the suite-wide mutex.
    del mode, target_index
    return root / "vision-memory-r11-new-phase1a.lock"


def _acquire_suite_lock(
    args: argparse.Namespace,
    *,
    lock_root: Path | None = None,
) -> dict[str, Any]:
    """Atomically acquire the suite-wide Phase 1A lock without waiting.

    A directory lock is intentionally used instead of a process-local mutex so
    distinct modes, targets, controller processes, and fresh output roots all
    conflict on the single two-GPU instance.  A stale lock fails closed and
    must be investigated rather than silently stolen.
    """

    path = _suite_lock_path(mode=args.mode, target_index=args.target_index, lock_root=lock_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {
        "schema": LOCK_SCHEMA,
        "lock_key": "r11-new-phase1a-suite",
        "pid": os.getpid(),
        "hostname": platform.node(),
        "git_commit": args.expected_commit,
        "mode": args.mode,
        "target_index": args.target_index,
        "target_segment_id": core.R11_NEW_TARGET_IDS[args.target_index],
        "output_root": str(args.output_root.resolve()),
        "acquired_at_utc": _utc_now(),
        "owner_token": uuid.uuid4().hex,
    }
    try:
        path.mkdir()
    except FileExistsError as exc:
        metadata_path = path / "owner.json"
        try:
            existing: Any = _load(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError):
            existing = {"unreadable_or_missing_owner": str(metadata_path)}
        raise ValueError(
            "R11_new duplicate-process lock is already held; refusing a second fresh-root run: "
            f"lock={path}, owner={existing}"
        ) from exc
    metadata_path = path / "owner.json"
    try:
        _write_json(metadata_path, owner)
    except Exception:
        path.rmdir()
        raise
    return {
        "path": str(path.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "owner": owner,
    }


def _release_suite_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(lock["path"]))
    metadata_path = Path(str(lock["metadata_path"]))
    expected_owner = lock.get("owner")
    if not path.is_dir() or not metadata_path.is_file():
        raise RuntimeError("R11_new suite lock disappeared before controlled release.")
    observed_owner = _load(metadata_path)
    if observed_owner != expected_owner or _sha256(metadata_path) != lock.get("metadata_sha256"):
        raise RuntimeError("R11_new suite lock ownership changed; refusing to remove another owner.")
    metadata_path.unlink()
    path.rmdir()
    return {
        **dict(lock),
        "released": True,
        "released_at_utc": _utc_now(),
    }


def _validate_snapshot_manifest(model_dir: Path, *, env_name: str) -> dict[str, Any]:
    expected = EXPECTED_ENVIRONMENT[env_name]
    manifest = model_dir / ".snapshot_manifest.json"
    sidecar = model_dir / ".snapshot_manifest.json.sha256"
    if not manifest.is_file() or not sidecar.is_file():
        raise ValueError(f"R11_new model snapshot manifest is missing: {model_dir}")
    observed = _sha256(manifest)
    if observed != expected:
        raise ValueError(
            f"R11_new {env_name} mismatch: expected {expected}, observed {observed}."
        )
    expected_sidecar = f"{observed}  {manifest.name}"
    if sidecar.read_text(encoding="utf-8").strip() != expected_sidecar:
        raise ValueError(f"R11_new model snapshot manifest sidecar mismatch: {sidecar}")
    return {
        "model_dir": str(model_dir.resolve()),
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": observed,
        "sidecar_path": str(sidecar.resolve()),
    }


def _validate_r11_comparison(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"R11_new canonical R11 comparison is missing: {path}")
    observed_sha = _sha256(path)
    if observed_sha != core.R11_NEW_PARENT_R11_COMPARISON_SHA256:
        raise ValueError(
            "R11_new requires the exact canonical R11 comparison SHA: "
            f"expected {core.R11_NEW_PARENT_R11_COMPARISON_SHA256}, observed {observed_sha}."
        )
    comparison = _load(path)
    expected = {
        "schema": CANONICAL_R11_SCHEMA,
        "status": CANONICAL_R11_STATUS,
        "target_pass_count": CANONICAL_R11_PASS_COUNT,
        "decision": CANONICAL_R11_DECISION,
    }
    observed = {key: comparison.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            "R11_new canonical R11 comparison content drifted: "
            f"expected {expected}, observed {observed}."
        )
    return comparison


def _validate_prior_terminal(
    path: Path | None,
    *,
    expected_mode: str,
    expected_target_index: int,
    git_commit: str,
    config_sha256: str,
    trainer_sha256: str,
) -> dict[str, Any]:
    if path is None or not path.is_file() or path.name != "terminal.json":
        raise ValueError(
            f"R11_new requires a prior {expected_mode} target-{expected_target_index:02d} "
            "controller terminal.json."
        )
    terminal = _load(path)
    checks = terminal.get("execution_checks")
    expected_target_id = core.R11_NEW_TARGET_IDS[expected_target_index]
    valid = bool(
        terminal.get("schema") == TERMINAL_SCHEMA
        and terminal.get("status") == "technical_completed"
        and terminal.get("technical_completed") is True
        and terminal.get("mode") == expected_mode
        and terminal.get("target_index") == expected_target_index
        and terminal.get("target_segment_id") == expected_target_id
        and terminal.get("child_exit_code") == 0
        and terminal.get("git_commit") == git_commit
        and terminal.get("config_sha256") == config_sha256
        and terminal.get("trainer_sha256") == trainer_sha256
        and terminal.get("formal_success") is False
        and terminal.get("scientific_success_claim") is False
        and isinstance(checks, Mapping)
        and bool(checks)
        and all(value is True for value in checks.values())
    )
    if expected_mode == "technical-preflight":
        diagnostic = terminal.get("diagnostic_result")
        valid = bool(
            valid
            and isinstance(diagnostic, Mapping)
            and diagnostic.get("evaluated") is False
            and diagnostic.get("phase1a_query_level_reachability_gate") is None
            and diagnostic.get("result") == "not_evaluated_in_technical_preflight"
        )
    if not valid:
        raise ValueError(f"R11_new prerequisite terminal failed exact validation: {path}")

    inventory_path = path.parent / "artifact_inventory.json"
    if not inventory_path.is_file():
        raise ValueError(f"R11_new prerequisite inventory is missing: {inventory_path}")
    inventory = _load(inventory_path)
    artifacts = inventory.get("artifacts")
    terminal_rows = [
        row
        for row in artifacts
        if isinstance(row, Mapping) and row.get("path") == "terminal.json"
    ] if isinstance(artifacts, list) else []
    if not (
        inventory.get("schema") == INVENTORY_SCHEMA
        and len(terminal_rows) == 1
        and terminal_rows[0].get("bytes") == path.stat().st_size
        and terminal_rows[0].get("sha256") == _sha256(path)
    ):
        raise ValueError(f"R11_new prerequisite terminal is not bound by its inventory: {path}")
    return {
        "terminal_path": str(path.resolve()),
        "terminal_sha256": _sha256(path),
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": _sha256(inventory_path),
        "mode": expected_mode,
        "target_index": expected_target_index,
        "target_segment_id": expected_target_id,
        "passed": True,
    }


def _validate_prerequisites(
    args: argparse.Namespace,
    *,
    git_commit: str,
    config_sha256: str,
    trainer_sha256: str,
) -> dict[str, Any]:
    preflight_path = getattr(args, "preflight_terminal", None)
    target0_path = getattr(args, "target0_formal_terminal", None)
    if args.mode == "technical-preflight":
        if preflight_path is not None or target0_path is not None:
            raise ValueError("R11_new technical preflight forbids prerequisite terminal arguments.")
        if args.target_index != 0:
            raise ValueError("R11_new technical preflight is locked to target 0.")
        return {
            "preflight_required": False,
            "target0_formal_required": False,
            "passed": True,
        }
    preflight = _validate_prior_terminal(
        preflight_path,
        expected_mode="technical-preflight",
        expected_target_index=0,
        git_commit=git_commit,
        config_sha256=config_sha256,
        trainer_sha256=trainer_sha256,
    )
    target0: dict[str, Any] | None = None
    if args.target_index == 0:
        if target0_path is not None:
            raise ValueError("R11_new formal target 0 forbids a target0 formal prerequisite.")
    else:
        target0 = _validate_prior_terminal(
            target0_path,
            expected_mode="formal",
            expected_target_index=0,
            git_commit=git_commit,
            config_sha256=config_sha256,
            trainer_sha256=trainer_sha256,
        )
    return {
        "preflight_required": True,
        "preflight": preflight,
        "target0_formal_required": args.target_index > 0,
        "target0_formal": target0,
        "passed": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("technical-preflight", "formal"), required=True)
    parser.add_argument("--target-index", type=int, choices=range(8), required=True)
    parser.add_argument("--r11-comparison", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--preflight-terminal", type=Path)
    parser.add_argument("--target0-formal-terminal", type=Path)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    return parser


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    _require_fresh_root(args.output_root)
    storage_audit = _deployment_audit(args.output_root)
    if not isinstance(args.expected_commit, str) or not _COMMIT_RE.fullmatch(args.expected_commit):
        raise ValueError("R11_new expected commit must be an exact lowercase 40-character SHA-1.")
    head = _git("rev-parse", "HEAD")
    if not _COMMIT_RE.fullmatch(head):
        raise ValueError(f"R11_new observed Git HEAD is not a full lowercase commit: {head!r}")
    if head != args.expected_commit:
        raise ValueError(
            f"R11_new controller commit mismatch: expected {args.expected_commit}, got {head}"
        )
    dirty = _git("status", "--porcelain")
    if dirty:
        raise ValueError("R11_new controller requires a clean experiment snapshot.")

    for name in ("train", "dev", "r11_comparison"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R11_new controller path is missing or not a file: {name}")
    for name in ("dreamlite", "reader"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R11_new controller model directory is missing: {name}")
    if _normalise_device(args.dreamlite_device) == _normalise_device(args.reader_device):
        raise ValueError("R11_new DreamLite and Reader devices must be distinct.")

    observed_data = {"train": _sha256(args.train), "dev": _sha256(args.dev)}
    expected_data = {"train": core.R11_NEW_TRAIN_SHA256, "dev": core.R11_NEW_DEV_SHA256}
    if observed_data != expected_data:
        raise ValueError(
            f"R11_new fixed data SHA mismatch: expected {expected_data}, observed {observed_data}."
        )

    drift = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in EXPECTED_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if drift:
        raise ValueError(f"R11_new strict environment drift: {drift}")

    snapshots = {
        "dreamlite": _validate_snapshot_manifest(
            args.dreamlite, env_name="VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256"
        ),
        "reader": _validate_snapshot_manifest(
            args.reader, env_name="VLM_READER_SNAPSHOT_MANIFEST_SHA256"
        ),
    }
    comparison = _validate_r11_comparison(args.r11_comparison)
    if not CONFIG_PATH.is_file():
        raise ValueError(f"R11_new immutable config is missing: {CONFIG_PATH}")
    config = _load(CONFIG_PATH)
    config_validation = core.validate_phase1a_config(config)
    if config_validation.get("passed") is not True:
        raise ValueError("R11_new immutable config validation did not pass.")
    if not TRAINER.is_file():
        raise ValueError(f"R11_new trainer is missing: {TRAINER}")

    prerequisite_audit = _validate_prerequisites(
        args,
        git_commit=head,
        config_sha256=_sha256(CONFIG_PATH),
        trainer_sha256=_sha256(TRAINER),
    )

    return {
        "git_commit": head,
        "git_dirty": False,
        "mode": args.mode,
        "target_index": args.target_index,
        "target_segment_id": core.R11_NEW_TARGET_IDS[args.target_index],
        "data_sha256": observed_data,
        "environment": dict(EXPECTED_ENVIRONMENT),
        "snapshot_manifests": snapshots,
        "storage_audit": storage_audit,
        "config_path": str(CONFIG_PATH.resolve()),
        "config_sha256": _sha256(CONFIG_PATH),
        "config_validation": config_validation,
        "trainer": str(TRAINER.resolve()),
        "trainer_sha256": _sha256(TRAINER),
        "canonical_r11_comparison_sha256": _sha256(args.r11_comparison),
        "canonical_r11_comparison_schema": comparison["schema"],
        "canonical_r11_decision": comparison["decision"],
        "prerequisites": prerequisite_audit,
        "python": sys.executable,
        "host": platform.node(),
    }


def _command(args: argparse.Namespace, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(TRAINER.resolve()),
        "--mode",
        args.mode,
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
        "--dreamlite-device",
        args.dreamlite_device,
        "--reader-device",
        args.reader_device,
        "--strict-determinism",
    ]
    leaked = sorted(_FORBIDDEN_COMMAND_OPTIONS & set(command))
    if leaked:
        raise RuntimeError(f"R11_new controller command exposed locked hyperparameters: {leaked}")
    return command


def _inventory(root: Path) -> list[dict[str, Any]]:
    root_inventory = (root / "artifact_inventory.json").resolve()
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
        if path.resolve() != root_inventory
    ]


def _manifest_checks(
    manifest: Mapping[str, Any] | None,
    *,
    args: argparse.Namespace,
    validated: Mapping[str, Any],
) -> dict[str, bool]:
    snapshots = manifest.get("model_snapshot_payloads_start") if isinstance(manifest, Mapping) else None
    dreamlite_snapshot = snapshots.get("dreamlite_mobile") if isinstance(snapshots, Mapping) else None
    reader_snapshot = snapshots.get("qwen_reader") if isinstance(snapshots, Mapping) else None
    return {
        "manifest_schema": isinstance(manifest, Mapping)
        and manifest.get("schema") == MANIFEST_SCHEMA,
        "manifest_mode": isinstance(manifest, Mapping) and manifest.get("mode") == args.mode,
        "manifest_commit": isinstance(manifest, Mapping)
        and manifest.get("git_commit") == args.expected_commit
        and manifest.get("git_dirty") is False,
        "manifest_target": isinstance(manifest, Mapping)
        and manifest.get("target_index") == args.target_index
        and manifest.get("target_segment_id") == core.R11_NEW_TARGET_IDS[args.target_index]
        and manifest.get("selected_segment_ids") == list(core.R11_NEW_TARGET_IDS)
        and manifest.get("selected_segments_sha256") == core.R11_NEW_TARGETS_PAYLOAD_SHA256,
        "manifest_data": isinstance(manifest, Mapping)
        and manifest.get("train_sha256") == core.R11_NEW_TRAIN_SHA256
        and manifest.get("dev_sha256") == core.R11_NEW_DEV_SHA256,
        "manifest_config": isinstance(manifest, Mapping)
        and manifest.get("preregistered_config_sha256") == validated["config_sha256"],
        "manifest_information_boundary": isinstance(manifest, Mapping)
        and isinstance(manifest.get("information_boundary"), Mapping)
        and manifest["information_boundary"].get("passed") is True,
        "manifest_snapshot_hashes": isinstance(dreamlite_snapshot, Mapping)
        and isinstance(reader_snapshot, Mapping)
        and dreamlite_snapshot.get("manifest_sha256")
        == EXPECTED_ENVIRONMENT["VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256"]
        and reader_snapshot.get("manifest_sha256")
        == EXPECTED_ENVIRONMENT["VLM_READER_SNAPSHOT_MANIFEST_SHA256"],
    }


def _summary_base_checks(
    summary: Mapping[str, Any] | None,
    *,
    args: argparse.Namespace,
) -> dict[str, bool]:
    return {
        "summary_schema": isinstance(summary, Mapping) and summary.get("schema") == SUMMARY_SCHEMA,
        "summary_mode": isinstance(summary, Mapping) and summary.get("mode") == args.mode,
        "summary_commit": isinstance(summary, Mapping)
        and summary.get("git_commit") == args.expected_commit,
        "summary_target": isinstance(summary, Mapping)
        and summary.get("target_index") == args.target_index
        and summary.get("target_segment_id") == core.R11_NEW_TARGET_IDS[args.target_index],
        "summary_diagnostic_boundary": isinstance(summary, Mapping)
        and summary.get("query_level_diagnostic_only") is True
        and summary.get("formal_success_gate") is False
        and summary.get("full_success_claim_allowed") is False,
    }


def _checkpoint_artifacts_valid(run_dir: Path) -> bool:
    expected_steps = tuple(core.R11_NEW_CHECKPOINT_STEPS)
    expected_pt = {f"step-{step:03d}.pt" for step in expected_steps}
    expected_png = {f"step-{step:03d}.png" for step in expected_steps}
    expected_hashes = {f"step-{step:03d}.json" for step in expected_steps}
    checkpoint_dir = run_dir / "checkpoints"
    image_dir = run_dir / "images"
    hash_dir = run_dir / "checkpoint_hashes"
    if not checkpoint_dir.is_dir() or not image_dir.is_dir() or not hash_dir.is_dir():
        return False
    if {path.name for path in checkpoint_dir.glob("step-*.pt")} != expected_pt:
        return False
    if {path.name for path in image_dir.glob("step-*.png")} != expected_png:
        return False
    if {path.name for path in hash_dir.glob("step-*.json")} != expected_hashes:
        return False
    for step in expected_steps:
        checkpoint = checkpoint_dir / f"step-{step:03d}.pt"
        image = image_dir / f"step-{step:03d}.png"
        record = _load(hash_dir / f"step-{step:03d}.json")
        if (
            record.get("schema") != CHECKPOINT_HASH_SCHEMA
            or record.get("optimizer_step") != step
            or Path(str(record.get("checkpoint_path", ""))).resolve() != checkpoint.resolve()
            or Path(str(record.get("png_path", ""))).resolve() != image.resolve()
            or record.get("checkpoint_bytes") != checkpoint.stat().st_size
            or record.get("png_bytes") != image.stat().st_size
            or record.get("checkpoint_sha256") != _sha256(checkpoint)
            or record.get("png_sha256") != _sha256(image)
            or record.get("trajectory_points") != core.R11_NEW_DIFFUSION_STEPS + 1
            or list(record.get("effective_sigmas", ()))
            != list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE)
        ):
            return False
    return True


def _formal_receipts_valid(rows: Sequence[Mapping[str, Any]], *, target_id: str) -> bool:
    return bool(
        len(rows) == core.R11_NEW_OPTIMIZER_STEPS
        and all(
            row.get("schema") == METRICS_SCHEMA
            and row.get("kind") == "optimizer_step"
            and row.get("optimizer_step") == position
            and row.get("target_segment_id") == target_id
            for position, row in enumerate(rows, start=1)
        )
    )


def _evaluation_rows_valid(rows: Sequence[Mapping[str, Any]], *, target_id: str) -> bool:
    expected_cells = {
        (checkpoint, condition, view)
        for checkpoint in ("m0", core.R11_NEW_PRIMARY_ENDPOINT)
        for condition in ("normal", "reset")
        for view in range(4)
    }
    observed_cells: set[tuple[str, str, int]] = set()
    target_indices: set[int] = set()
    if len(rows) != len(expected_cells):
        return False
    for row in rows:
        try:
            view = int(row["view_index"])
            checkpoint = str(row["checkpoint"])
            condition = str(row["condition"])
            target_index = int(row["target_index"])
            permutation = [int(value) for value in row["permutation"]]
            logits = [float(value) for value in row["choice_logits_ordered"]]
            ordered_target = permutation.index(target_index)
            predicted_ordered = max(range(4), key=logits.__getitem__)
            predicted_original = permutation[predicted_ordered]
            alternatives = [value for index, value in enumerate(logits) if index != ordered_target]
            expected_margin = logits[ordered_target] - max(alternatives)
            maximum = max(logits)
            expected_ce = maximum + math.log(
                sum(math.exp(value - maximum) for value in logits)
            ) - logits[ordered_target]
            valid = bool(
                row.get("schema") == EVALUATION_SCHEMA
                and row.get("suite") == EVALUATION_SUITE
                and row.get("item_id") == target_id
                and row.get("pair_unit") == target_id
                and row.get("family") == "F1"
                and (checkpoint, condition, view) in expected_cells
                and permutation == list(REVERSE_CYCLIC4[view])
                and 0 <= target_index < 4
                and type(row.get("correct")) is bool
                and isinstance(row.get("predicted_index"), int)
                and 0 <= int(row["predicted_index"]) < 4
                and math.isfinite(float(row["ce"]))
                and math.isfinite(float(row["margin"]))
                and len(logits) == 4
                and all(math.isfinite(value) for value in logits)
                and int(row["predicted_index"]) == predicted_original
                and row["correct"] is (predicted_original == target_index)
                and math.isclose(float(row["margin"]), expected_margin, rel_tol=1e-6, abs_tol=1e-6)
                and math.isclose(float(row["ce"]), expected_ce, rel_tol=1e-5, abs_tol=1e-5)
            )
        except (IndexError, KeyError, TypeError, ValueError):
            return False
        if not valid or (checkpoint, condition, view) in observed_cells:
            return False
        observed_cells.add((checkpoint, condition, view))
        target_indices.add(target_index)
    return observed_cells == expected_cells and len(target_indices) == 1


def _json_equivalent(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _declared_artifacts_valid(summary: Mapping[str, Any] | None, *, run_dir: Path) -> bool:
    if not isinstance(summary, Mapping) or not isinstance(summary.get("artifacts"), Mapping):
        return False
    artifacts = summary["artifacts"]
    expected_paths = {
        "manifest_sha256": run_dir / MANIFEST_FILE,
        "metrics_sha256": run_dir / METRICS_FILE,
        "evaluation_rows_sha256": run_dir / EVALUATION_ROWS_FILE,
        "endpoint_raw_sha256": run_dir / "endpoint_raw.pt",
        "endpoint_png_sha256": run_dir / "endpoint_raw.png",
        "snapshot_end_sha256": run_dir / "model_snapshot_verification_end.json",
        "technical_gate_sha256": run_dir / TECHNICAL_GATE_FILE,
    }
    if set(artifacts) != set(expected_paths):
        return False
    for name, path in expected_paths.items():
        if not path.is_file() or artifacts.get(name) != _sha256(path):
            return False
    try:
        snapshot_end = _load(expected_paths["snapshot_end_sha256"])
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        snapshot_end.get("schema") == "vision_memory.r11-new-phase1a-model-snapshot-end.v1"
        and snapshot_end.get("passed") is True
    )


def _reachability_gate(summary: Mapping[str, Any] | None) -> Any:
    if not isinstance(summary, Mapping):
        return None
    gates = summary.get("gates")
    if isinstance(gates, Mapping):
        return gates.get("phase1a_query_level_reachability_gate")
    return None


def _finite_scalar(value: Any, *, positive: bool = False, at_most_one: bool = False) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(
        math.isfinite(number)
        and (not positive or number > 0.0)
        and (not at_most_one or number <= 1.0)
    )


def _preflight_fixed_contract_valid(manifest: Mapping[str, Any] | None) -> bool:
    if not isinstance(manifest, Mapping):
        return False
    expected = {
        "resolution": 1024,
        "only_trainable": "x_T_fp32",
        "dreamlite_pipeline_load": "DreamLiteMobilePipeline.from_pretrained directly; no PEFT",
        "unet_frozen": True,
        "vae_frozen": True,
        "text_encoder_frozen": True,
        "reader_frozen": True,
        "dreamlite_unet_executed": True,
        "official_full_conditioner_precomputed": True,
        "gradient_mode": "full",
        "checkpoint_unet": True,
        "num_denoising_steps": core.R11_NEW_DIFFUSION_STEPS,
        "edit_start_sigma": core.R11_NEW_EFFECTIVE_START_SIGMA,
        "effective_sigmas": list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE),
        "optimizer": "Adam",
        "learning_rate": core.R11_NEW_LEARNING_RATE,
        "weight_decay": core.R11_NEW_WEIGHT_DECAY,
        "optimizer_steps": 0,
        "preflight_backward_calls": 1,
        "gradient_clipping": None,
        "checkpoint_steps": [0],
        "training_views": "forward cyclic",
        "endpoint_views": "reverse cyclic",
        "primary_endpoint": core.R11_NEW_PRIMARY_ENDPOINT,
        "best_checkpoint_selection_forbidden": True,
        "reset": "decode(blank_source_latent)",
    }
    return _json_equivalent(manifest.get("fixed_contract"), expected)


def _preflight_information_boundary_valid(manifest: Mapping[str, Any] | None) -> bool:
    information = manifest.get("information_boundary") if isinstance(manifest, Mapping) else None
    if not isinstance(information, Mapping):
        return False
    try:
        recomputed = core.validate_information_boundary(
            dreamlite_inputs=information["dreamlite_inputs"],
            noise_key=information["noise_key"],
            reader_loss_inputs=information["reader_loss_inputs"],
        )
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        recomputed.get("passed") is True
        and information.get("schema")
        == "vision_memory.r11-new-phase1a-information-boundary.v1"
        and information.get("passed") is True
        and information.get("leaked_fields") == []
        and information.get("conditioner_input_names") == ["source_latent", "event_text"]
        and information.get("sampler_input_names") == list(core.R11_NEW_DREAMLITE_INPUTS)
        and information.get("oracle_initialization_key_names") == list(core.R11_NEW_NOISE_KEY)
        and information.get("query_used_only_by_frozen_reader_loss") is True
        and information.get("choices_used_only_by_frozen_reader_loss") is True
        and information.get("target_index_used_only_by_frozen_reader_loss") is True
        and information.get("forbidden_writer_key_violations") == []
    )


def _preflight_condition_artifact_valid(
    manifest: Mapping[str, Any] | None,
    *,
    run_dir: Path,
) -> bool:
    record = manifest.get("condition_artifact") if isinstance(manifest, Mapping) else None
    if not isinstance(record, Mapping):
        return False
    path = run_dir / "condition" / "official_full_condition.pt"
    tensor_hashes = record.get("tensor_sha256")
    try:
        resolved_record_path = Path(str(record.get("path", ""))).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        file_valid = bool(
            path.is_file()
            and resolved_record_path == path.resolve()
            and type(record.get("bytes")) is int
            and record.get("bytes") == path.stat().st_size
            and record.get("sha256") == _sha256(path)
        )
    except Exception:
        return False
    prompt_embeds = payload.get("prompt_embeds") if isinstance(payload, Mapping) else None
    attention_mask = payload.get("attention_mask") if isinstance(payload, Mapping) else None
    payload_hashes = payload.get("tensor_sha256") if isinstance(payload, Mapping) else None
    return bool(
        file_valid
        and set(record) == {
            "path",
            "sha256",
            "bytes",
            "tensor_sha256",
            "event_text_sha256",
            "recompute_matches",
        }
        and isinstance(record.get("sha256"), str)
        and _HEX64_RE.fullmatch(str(record["sha256"]))
        and isinstance(record.get("event_text_sha256"), str)
        and _HEX64_RE.fullmatch(str(record["event_text_sha256"]))
        and record.get("recompute_matches") is True
        and isinstance(tensor_hashes, Mapping)
        and set(tensor_hashes) == {"prompt_embeds", "attention_mask"}
        and all(
            isinstance(value, str) and _HEX64_RE.fullmatch(value)
            for value in tensor_hashes.values()
        )
        and isinstance(payload, Mapping)
        and set(payload)
        == {
            "schema",
            "prompt_embeds",
            "attention_mask",
            "tensor_sha256",
            "event_text_sha256",
            "recompute_matches",
        }
        and payload.get("schema") == "vision_memory.r11-new-phase1a-condition.v1"
        and isinstance(prompt_embeds, torch.Tensor)
        and prompt_embeds.is_floating_point()
        and isinstance(attention_mask, torch.Tensor)
        and attention_mask.shape[:2] == prompt_embeds.shape[:2]
        and isinstance(payload_hashes, Mapping)
        and dict(payload_hashes) == dict(tensor_hashes)
        and payload_hashes.get("prompt_embeds") == canonical_tensor_sha256(prompt_embeds)
        and payload_hashes.get("attention_mask") == canonical_tensor_sha256(attention_mask)
        and payload.get("event_text_sha256") == record.get("event_text_sha256")
        and payload.get("recompute_matches") is True
    )


def _preflight_checkpoint_artifact_valid(
    summary: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    *,
    run_dir: Path,
) -> bool:
    if not isinstance(summary, Mapping) or not isinstance(manifest, Mapping):
        return False
    checkpoint_dir = run_dir / "checkpoints"
    image_dir = run_dir / "images"
    hash_dir = run_dir / "checkpoint_hashes"
    if not checkpoint_dir.is_dir() or not image_dir.is_dir() or not hash_dir.is_dir():
        return False
    if {path.name for path in checkpoint_dir.glob("step-*.pt")} != {"step-000.pt"}:
        return False
    if {path.name for path in image_dir.glob("step-*.png")} != {"step-000.png"}:
        return False
    if {path.name for path in hash_dir.glob("step-*.json")} != {"step-000.json"}:
        return False
    checkpoint = checkpoint_dir / "step-000.pt"
    image = image_dir / "step-000.png"
    record_path = hash_dir / "step-000.json"
    try:
        record = _load(record_path)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception:
        return False
    tensor_hashes = record.get("tensor_sha256")
    trajectory_hashes = (
        tensor_hashes.get("trajectory_fp32") if isinstance(tensor_hashes, Mapping) else None
    )
    x_t = payload.get("x_T_fp32") if isinstance(payload, Mapping) else None
    z_t = payload.get("z_t_fp32") if isinstance(payload, Mapping) else None
    trajectory = payload.get("trajectory_fp32") if isinstance(payload, Mapping) else None
    payload_hashes = payload.get("tensor_sha256") if isinstance(payload, Mapping) else None
    manifest_sha256 = _sha256(run_dir / MANIFEST_FILE)
    condition = manifest.get("condition_artifact")
    condition_sha256 = condition.get("sha256") if isinstance(condition, Mapping) else None
    try:
        record_files_valid = bool(
            checkpoint.is_file()
            and image.is_file()
            and Path(str(record.get("checkpoint_path", ""))).resolve() == checkpoint.resolve()
            and Path(str(record.get("png_path", ""))).resolve() == image.resolve()
            and type(record.get("checkpoint_bytes")) is int
            and type(record.get("png_bytes")) is int
            and record.get("checkpoint_bytes") == checkpoint.stat().st_size
            and record.get("png_bytes") == image.stat().st_size
            and record.get("checkpoint_sha256") == _sha256(checkpoint)
            and record.get("png_sha256") == _sha256(image)
        )
    except OSError:
        return False
    return bool(
        record_files_valid
        and record.get("schema") == CHECKPOINT_HASH_SCHEMA
        and record.get("optimizer_step") == 0
        and record.get("trajectory_points") == core.R11_NEW_DIFFUSION_STEPS + 1
        and list(record.get("effective_sigmas", ()))
        == list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE)
        and isinstance(tensor_hashes, Mapping)
        and set(tensor_hashes) == {"x_T_fp32", "z_t_fp32", "trajectory_fp32"}
        and isinstance(trajectory_hashes, list)
        and len(trajectory_hashes) == core.R11_NEW_DIFFUSION_STEPS + 1
        and all(
            isinstance(value, str) and _HEX64_RE.fullmatch(value)
            for value in (
                tensor_hashes.get("x_T_fp32"),
                tensor_hashes.get("z_t_fp32"),
                *trajectory_hashes,
            )
        )
        and _json_equivalent(summary.get("checkpoint"), record)
        and isinstance(payload, Mapping)
        and payload.get("schema") == TRAINER_CHECKPOINT_SCHEMA
        and payload.get("optimizer_step") == 0
        and payload.get("manifest_sha256") == manifest_sha256
        and payload.get("condition_artifact_sha256") == condition_sha256
        and list(payload.get("effective_sigmas", ()))
        == list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE)
        and isinstance(payload.get("optimizer"), Mapping)
        and isinstance(x_t, torch.Tensor)
        and isinstance(z_t, torch.Tensor)
        and x_t.dtype == torch.float32
        and z_t.dtype == torch.float32
        and x_t.shape == z_t.shape
        and isinstance(trajectory, (tuple, list))
        and len(trajectory) == core.R11_NEW_DIFFUSION_STEPS + 1
        and all(
            isinstance(value, torch.Tensor)
            and value.dtype == torch.float32
            and value.shape == x_t.shape
            for value in trajectory
        )
        and isinstance(payload_hashes, Mapping)
        and dict(payload_hashes) == dict(tensor_hashes)
        and payload_hashes.get("x_T_fp32") == canonical_tensor_sha256(x_t)
        and payload_hashes.get("z_t_fp32") == canonical_tensor_sha256(z_t)
        and payload_hashes.get("trajectory_fp32")
        == [canonical_tensor_sha256(value) for value in trajectory]
    )


def _preflight_snapshot_end_valid(
    manifest: Mapping[str, Any] | None,
    *,
    run_dir: Path,
) -> bool:
    path = run_dir / "model_snapshot_verification_end.json"
    try:
        value = _load(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(manifest, Mapping)
        and value.get("schema") == "vision_memory.r11-new-phase1a-model-snapshot-end.v1"
        and value.get("passed") is True
        and isinstance(value.get("bindings"), Mapping)
        and _json_equivalent(value.get("bindings"), manifest.get("model_snapshot_payloads_start"))
    )


def _preflight_start_runtime_valid(
    manifest: Mapping[str, Any] | None,
    *,
    run_dir: Path,
) -> bool:
    try:
        start = _load(run_dir / "model_snapshot_verification_start.json")
        runtime = _load(run_dir / "runtime.json")
        owner = _load(run_dir / ".r11_new_output_owner.json")
        environment_text = (run_dir / "environment.txt").read_text(encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    determinism = manifest.get("strict_determinism") if isinstance(manifest, Mapping) else None
    required_determinism_environment = {
        key: EXPECTED_ENVIRONMENT[key]
        for key in (
            "PYTHONHASHSEED",
            "CUBLAS_WORKSPACE_CONFIG",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
        )
    }
    return bool(
        isinstance(manifest, Mapping)
        and start.get("schema") == "vision_memory.r11-new-phase1a-model-snapshot-start.v1"
        and _json_equivalent(start.get("bindings"), manifest.get("model_snapshot_payloads_start"))
        and set(runtime)
        == {"python", "platform", "packages", "cuda_available", "torch_cuda", "gpu_names"}
        and runtime.get("cuda_available") is True
        and isinstance(runtime.get("packages"), Mapping)
        and isinstance(runtime.get("gpu_names"), list)
        and len(runtime["gpu_names"]) >= 2
        and all(isinstance(value, str) and value for value in runtime["gpu_names"])
        and isinstance(runtime.get("python"), str)
        and bool(runtime["python"])
        and isinstance(runtime.get("platform"), str)
        and bool(runtime["platform"])
        and isinstance(runtime.get("torch_cuda"), str)
        and bool(runtime["torch_cuda"])
        and bool(environment_text.strip())
        and set(owner) == {"schema", "pid", "created_at_utc"}
        and owner.get("schema") == TRAINER_OUTPUT_OWNER_SCHEMA
        and type(owner.get("pid")) is int
        and owner["pid"] > 0
        and isinstance(owner.get("created_at_utc"), str)
        and bool(owner["created_at_utc"])
        and isinstance(determinism, Mapping)
        and determinism.get("seed") == core.R11_NEW_SEED
        and determinism.get("environment") == required_determinism_environment
        and determinism.get("deterministic_algorithms") is True
        and determinism.get("deterministic_warn_only") is False
        and determinism.get("cudnn_benchmark") is False
        and determinism.get("cudnn_deterministic") is True
        and determinism.get("cuda_matmul_allow_tf32") is False
        and determinism.get("cudnn_allow_tf32") is False
        and determinism.get("float32_matmul_precision") == "highest"
        and determinism.get("sdpa")
        == {"flash": False, "memory_efficient": False, "cudnn": False, "math": True}
    )


def _preflight_trainer_terminal_valid(*, run_dir: Path) -> bool:
    path = run_dir / "terminal.json"
    try:
        value = _load(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        set(value)
        == {
            "schema",
            "status",
            "mode",
            "technical_gate",
            "diagnostic_evaluated",
            "formal_success_gate",
        }
        and value.get("schema") == TRAINER_TERMINAL_SCHEMA
        and value.get("status") == "succeeded"
        and value.get("mode") == "technical-preflight"
        and value.get("technical_gate") is True
        and value.get("diagnostic_evaluated") is False
        and value.get("formal_success_gate") is False
    )


def _preflight_trainer_inventory_valid(*, run_dir: Path) -> bool:
    path = run_dir / "artifact_inventory.json"
    try:
        value = _load(path)
        observed = _inventory(run_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        set(value) == {"schema", "artifact_count", "artifacts"}
        and value.get("schema") == TRAINER_INVENTORY_SCHEMA
        and value.get("artifact_count") == len(observed)
        and _json_equivalent(value.get("artifacts"), observed)
    )


def _preflight_evidence_checks(
    summary: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    *,
    run_dir: Path,
) -> dict[str, bool]:
    gates = summary.get("gates") if isinstance(summary, Mapping) else None
    preflight_path = run_dir / PREFLIGHT_FILE
    try:
        preflight_summary = _load(preflight_path)
    except (OSError, ValueError, json.JSONDecodeError):
        preflight_summary = None
    forbidden_artifacts = (
        run_dir / METRICS_FILE,
        run_dir / EVALUATION_ROWS_FILE,
        run_dir / TECHNICAL_GATE_FILE,
        run_dir / "endpoint_raw.pt",
        run_dir / "endpoint_raw.png",
    )
    summary_contract = bool(
        isinstance(summary, Mapping)
        and summary.get("schema") == SUMMARY_SCHEMA
        and summary.get("status") == "completed_technical"
        and summary.get("mode") == "technical-preflight"
        and summary.get("passed") is True
        and type(summary.get("backward_calls")) is int
        and summary.get("backward_calls") == 1
        and type(summary.get("optimizer_steps")) is int
        and summary.get("optimizer_steps") == 0
        and _finite_scalar(summary.get("loss"))
        and _finite_scalar(summary.get("gradient_norm"), positive=True)
        and _finite_scalar(
            summary.get("gradient_nonzero_fraction"), positive=True, at_most_one=True
        )
        and summary.get("trajectory_points") == core.R11_NEW_DIFFUSION_STEPS + 1
        and summary.get("dreamlite_denoising_steps") == core.R11_NEW_DIFFUSION_STEPS
        and list(summary.get("effective_sigmas", ()))
        == list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE)
        and summary.get("trainable_parameter_names") == ["x_T_fp32"]
        and summary.get("all_models_frozen") is True
        and summary.get("snapshots_unchanged") is True
        and summary.get("condition_artifact_valid") is True
        and summary.get("scientific_gate_evaluated") is False
        and summary.get("phase1a_reachability_gate") is None
        and summary.get("formal_success_gate") is False
        and summary.get("query_level_diagnostic_only") is True
        and isinstance(gates, Mapping)
        and dict(gates)
        == {
            "technical_gate": True,
            "phase1a_query_level_reachability_gate": None,
            "formal_success_gate": False,
        }
    )
    manifest_contract = bool(
        isinstance(manifest, Mapping)
        and manifest.get("protocol") == core.R11_NEW_PROTOCOL
        and manifest.get("implementation_revision") == "full-frozen-dreamlite-x-t-fp32-v1"
        and manifest.get("source_contract_verified") is True
        and manifest.get("event_noise_contract_verified") is True
        and manifest.get("query_level_diagnostic_only") is True
        and manifest.get("formal_success_gate") is False
        and _preflight_fixed_contract_valid(manifest)
        and _preflight_information_boundary_valid(manifest)
    )
    return {
        "preflight_summary_technical_fields": summary_contract,
        "preflight_summary_twin_exact": isinstance(preflight_summary, Mapping)
        and isinstance(summary, Mapping)
        and _json_equivalent(preflight_summary, summary),
        "preflight_manifest_contract": manifest_contract,
        "preflight_condition_artifact": _preflight_condition_artifact_valid(
            manifest, run_dir=run_dir
        ),
        "preflight_checkpoint_artifact": _preflight_checkpoint_artifact_valid(
            summary, manifest, run_dir=run_dir
        ),
        "preflight_snapshot_end": _preflight_snapshot_end_valid(manifest, run_dir=run_dir),
        "preflight_start_runtime": _preflight_start_runtime_valid(manifest, run_dir=run_dir),
        "preflight_trainer_terminal": _preflight_trainer_terminal_valid(run_dir=run_dir),
        "preflight_forbidden_formal_artifacts_absent": not any(
            path.exists() for path in forbidden_artifacts
        ),
        "preflight_report_present": (run_dir / "REPORT.md").is_file()
        and (run_dir / "REPORT.md").stat().st_size > 0,
        "preflight_trainer_inventory": _preflight_trainer_inventory_valid(run_dir=run_dir),
    }


def _assess_run(
    *,
    args: argparse.Namespace,
    validated: Mapping[str, Any],
    run_dir: Path,
    child_exit_code: int,
) -> tuple[dict[str, bool], dict[str, Any]]:
    summary_path = run_dir / SUMMARY_FILE
    manifest_path = run_dir / MANIFEST_FILE
    summary = _load(summary_path) if summary_path.is_file() else None
    manifest = _load(manifest_path) if manifest_path.is_file() else None
    checks: dict[str, bool] = {"child_exit_zero": child_exit_code == 0}
    checks.update(_manifest_checks(manifest, args=args, validated=validated))

    if args.mode == "technical-preflight":
        gate = _reachability_gate(summary)
        checks.update(
            {
                "preflight_summary_schema": isinstance(summary, Mapping)
                and summary.get("schema") == SUMMARY_SCHEMA,
                "preflight_summary_mode": isinstance(summary, Mapping)
                and summary.get("mode") == "technical-preflight",
                "preflight_completed_technical": isinstance(summary, Mapping)
                and summary.get("status") == "completed_technical"
                and summary.get("passed") is True,
                "preflight_no_scientific_conclusion": gate is None
                and isinstance(summary, Mapping)
                and summary.get("scientific_gate_evaluated") is False
                and summary.get("query_level_diagnostic_only") is True
                and summary.get("formal_success_gate") is False,
            }
        )
        checks.update(
            _preflight_evidence_checks(
                summary,
                manifest,
                run_dir=run_dir,
            )
        )
        diagnostic = {
            "evaluated": False,
            "phase1a_query_level_reachability_gate": None,
            "result": "not_evaluated_in_technical_preflight",
        }
        return checks, diagnostic

    checks.update(_summary_base_checks(summary, args=args))
    metrics_path = run_dir / METRICS_FILE
    evaluation_path = run_dir / EVALUATION_ROWS_FILE
    technical_path = run_dir / TECHNICAL_GATE_FILE
    rows = _load_jsonl(metrics_path) if metrics_path.is_file() else []
    evaluation_rows = _load_jsonl(evaluation_path) if evaluation_path.is_file() else []
    technical = _load(technical_path) if technical_path.is_file() else None
    recomputed_technical: Mapping[str, Any] | None = None
    technical_core_matches = False
    if isinstance(technical, Mapping) and isinstance(technical.get("core_audit"), Mapping):
        try:
            recomputed_technical = core.phase1a_technical_gate(
                rows,
                target_segment_id=core.R11_NEW_TARGET_IDS[args.target_index],
                audit=technical["core_audit"],
            )
            technical_core_matches = all(
                _json_equivalent(technical.get(key), value)
                for key, value in recomputed_technical.items()
            )
        except (KeyError, TypeError, ValueError):
            recomputed_technical = None
            technical_core_matches = False
    gate = _reachability_gate(summary)
    checkpoint_steps = summary.get("checkpoint_steps_observed") if isinstance(summary, Mapping) else None
    summary_technical = summary.get("technical_gate") if isinstance(summary, Mapping) else None
    gates = summary.get("gates") if isinstance(summary, Mapping) else None
    raw_rows_valid = _evaluation_rows_valid(
        evaluation_rows,
        target_id=core.R11_NEW_TARGET_IDS[args.target_index],
    )
    recomputed_statistics: Mapping[str, Any] | None = None
    recomputed_gate: bool | None = None
    if raw_rows_valid:
        try:
            recomputed_statistics = core.phase1a_target_statistics(
                evaluation_rows,
                suite=EVALUATION_SUITE,
                target_segment_id=core.R11_NEW_TARGET_IDS[args.target_index],
                endpoint=core.R11_NEW_PRIMARY_ENDPOINT,
            )
            recomputed_gate = core.phase1a_target_gate(
                recomputed_statistics,
                technical_gate=True,
            )
        except (KeyError, TypeError, ValueError):
            recomputed_statistics = None
            recomputed_gate = None
    checks.update(
        {
            "formal_completed": isinstance(summary, Mapping) and summary.get("status") == "completed",
            "formal_receipts_exact": _formal_receipts_valid(
                rows, target_id=core.R11_NEW_TARGET_IDS[args.target_index]
            ),
            "formal_optimizer_steps_exact": isinstance(summary, Mapping)
            and summary.get("optimizer_steps") == core.R11_NEW_OPTIMIZER_STEPS,
            "formal_checkpoints_exact": checkpoint_steps == list(core.R11_NEW_CHECKPOINT_STEPS)
            and _checkpoint_artifacts_valid(run_dir),
            "formal_technical_gate_file": isinstance(technical, Mapping)
            and technical.get("schema") == TECHNICAL_GATE_SCHEMA
            and technical.get("passed") is True
            and technical.get("optimizer_step_records") == core.R11_NEW_OPTIMIZER_STEPS,
            "formal_raw_technical_gate_recomputed": recomputed_technical is not None
            and recomputed_technical.get("passed") is True
            and technical_core_matches,
            "formal_technical_gate_bound": isinstance(summary_technical, Mapping)
            and isinstance(technical, Mapping)
            and dict(summary_technical) == dict(technical)
            and isinstance(gates, Mapping)
            and gates.get("technical_gate") is True,
            "formal_evaluation_rows_exact": raw_rows_valid,
            "formal_statistics_recomputed": isinstance(summary, Mapping)
            and recomputed_statistics is not None
            and _json_equivalent(summary.get("target_statistics"), recomputed_statistics),
            "formal_gate_recomputed": type(gate) is bool
            and type(recomputed_gate) is bool
            and gate is recomputed_gate,
            "formal_artifact_hashes_bound": _declared_artifacts_valid(summary, run_dir=run_dir),
            "formal_diagnostic_boolean": type(gate) is bool,
            "formal_success_false": isinstance(summary, Mapping)
            and summary.get("formal_success_gate") is False
            and summary.get("full_success_claim_allowed") is False
            and isinstance(gates, Mapping)
            and gates.get("formal_success_gate") is False,
        }
    )
    if not all(checks.values()):
        diagnostic = {
            "evaluated": False,
            "phase1a_query_level_reachability_gate": None,
            "result": "not_evaluated_due_to_technical_failure",
        }
    else:
        diagnostic = {
            "evaluated": True,
            "phase1a_query_level_reachability_gate": gate,
            "result": (
                "query_level_reachability_passed"
                if gate is True
                else "query_level_reachability_not_found"
            ),
        }
    return checks, diagnostic


def _write_inventory(root: Path) -> None:
    _write_json(
        root / "artifact_inventory.json",
        {
            "schema": INVENTORY_SCHEMA,
            "root": str(root.resolve()),
            "artifacts": _inventory(root),
        },
    )


def _validation_failure_artifacts(
    args: argparse.Namespace,
    *,
    started_at: str,
    error: Exception,
) -> None:
    # Never overwrite a pre-existing run root merely to report that it was not fresh.
    if args.output_root.exists() and any(args.output_root.iterdir()):
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "stdout.log").touch()
    (args.output_root / "stderr.log").write_text(f"{error}\n", encoding="utf-8")
    _write_json(
        args.output_root / "launch.json",
        {
            "schema": LAUNCH_SCHEMA,
            "status": "failed_validation",
            "started_at_utc": started_at,
            "mode": args.mode,
            "target_index": args.target_index,
            "expected_commit": args.expected_commit,
            "command": None,
        },
    )
    _write_json(
        args.output_root / "terminal.json",
        {
            "schema": TERMINAL_SCHEMA,
            "status": "failed",
            "technical_completed": False,
            "diagnostic_result": {
                "evaluated": False,
                "phase1a_query_level_reachability_gate": None,
                "result": "not_evaluated_due_to_technical_failure",
            },
            "formal_success": False,
            "scientific_success_claim": False,
            "error": str(error),
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
        },
    )
    _write_inventory(args.output_root)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_at = _utc_now()
    suite_lock: dict[str, Any] | None = None
    try:
        validated = _validate(args)
        suite_lock = _acquire_suite_lock(args)
        args.output_root.mkdir(parents=True, exist_ok=False)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        if suite_lock is not None:
            try:
                _release_suite_lock(suite_lock)
            except (OSError, RuntimeError, ValueError):
                pass
        _validation_failure_artifacts(args, started_at=started_at, error=exc)
        print(str(exc), file=sys.stderr, flush=True)
        return 1

    stdout_path = args.output_root / "stdout.log"
    stderr_path = args.output_root / "stderr.log"
    run_dir = args.output_root / "run"
    try:
        command = _command(args, run_dir)
        launch = {
            "schema": LAUNCH_SCHEMA,
            "status": "running",
            "started_at_utc": started_at,
            "command": command,
            "suite_lock": suite_lock,
            **validated,
        }
        _write_json(args.output_root / "launch.json", launch)
    except (OSError, RuntimeError, ValueError) as exc:
        if suite_lock is not None:
            try:
                _release_suite_lock(suite_lock)
            except (OSError, RuntimeError, ValueError):
                pass
        _validation_failure_artifacts(args, started_at=started_at, error=exc)
        print(str(exc), file=sys.stderr, flush=True)
        return 1
    started = time.monotonic()
    child_exit_code = -1
    execution_error: str | None = None
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ.copy(),
                stdout=stdout,
                stderr=stderr,
            )
        child_exit_code = result.returncode
        checks, diagnostic = _assess_run(
            args=args,
            validated=validated,
            run_dir=run_dir,
            child_exit_code=child_exit_code,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        execution_error = str(exc)
        checks = {"controller_assessment_completed": False}
        diagnostic = {
            "evaluated": False,
            "phase1a_query_level_reachability_gate": None,
            "result": "not_evaluated_due_to_technical_failure",
        }
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write(f"\ncontroller assessment failure: {exc}\n")

    checks["suite_lock_acquired"] = suite_lock is not None
    released_lock: dict[str, Any] | None = None
    try:
        if suite_lock is None:
            raise RuntimeError("R11_new suite lock was not acquired.")
        released_lock = _release_suite_lock(suite_lock)
        checks["suite_lock_released"] = True
    except (OSError, RuntimeError, ValueError) as exc:
        checks["suite_lock_released"] = False
        release_error = f"suite lock release failure: {exc}"
        execution_error = (
            release_error if execution_error is None else f"{execution_error}; {release_error}"
        )

    technical_completed = all(checks.values())
    if not technical_completed:
        diagnostic = {
            "evaluated": False,
            "phase1a_query_level_reachability_gate": None,
            "result": "not_evaluated_due_to_technical_failure",
        }
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": "technical_completed" if technical_completed else "failed",
        "technical_completed": technical_completed,
        "diagnostic_result": diagnostic,
        "formal_success": False,
        "scientific_success_claim": False,
        "mode": args.mode,
        "target_index": args.target_index,
        "target_segment_id": core.R11_NEW_TARGET_IDS[args.target_index],
        "git_commit": validated["git_commit"],
        "child_exit_code": child_exit_code,
        "execution_checks": checks,
        "suite_lock": released_lock if released_lock is not None else suite_lock,
        "error": execution_error,
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "summary_sha256": _sha256(run_dir / SUMMARY_FILE)
        if (run_dir / SUMMARY_FILE).is_file()
        else None,
        "manifest_sha256": _sha256(run_dir / MANIFEST_FILE)
        if (run_dir / MANIFEST_FILE).is_file()
        else None,
        "technical_gate_sha256": _sha256(run_dir / TECHNICAL_GATE_FILE)
        if (run_dir / TECHNICAL_GATE_FILE).is_file()
        else None,
        "stdout_sha256": _sha256(stdout_path),
        "stderr_sha256": _sha256(stderr_path),
        "config_sha256": validated["config_sha256"],
        "trainer_sha256": validated["trainer_sha256"],
    }
    _write_json(args.output_root / "terminal.json", terminal)
    _write_inventory(args.output_root)
    print(json.dumps(terminal, indent=2, sort_keys=True), flush=True)
    return 0 if technical_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
