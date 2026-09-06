"""Fail-closed Inspire controller for the source-only initialization bridge diagnostic."""

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
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train import r11_new_source_init_bridge as trainer  # noqa: E402
from vision_memory.training import r11_new_source_init_bridge as core  # noqa: E402


TRAINER = ROOT / "scripts" / "train" / "r11_new_source_init_bridge.py"
CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "r11_new_canonical_latent_bridge_target01_source_only_init.json"
)
SUMMARY_FILE = "r11_new_bridge_summary.json"
PREFLIGHT_FILE = "technical_preflight.json"
METRICS_FILE = "metrics.jsonl"
ROWS_FILE = "evaluation_rows.jsonl"
PREFLIGHT_REQUIRED_AUDIT_TRUE = (
    "four_dreamlite_steps", "effective_sigmas_exact", "finite_nonzero_x_T_gradient",
    "only_x_T_fp32_trainable", "frozen_gradients_absent", "step0_parity_valid",
    "source_only_initialization_artifact_valid", "teacher_binding_valid",
    "teacher_replay_gate", "checkpoint_hash_valid", "snapshots_unchanged",
)

LAUNCH_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-controller-launch.v1"
TERMINAL_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-controller-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-controller-inventory.v1"
LOCK_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-suite-lock.v1"
EXPECTED_HOST_PREFIX = "vlm-r3-h200x2-live-20260717"
INSPIRE_SSD_ROOT = Path("/inspire/ssd")
MINIMUM_FREE_BYTES = 50 * 1024**3
LOCK_PATH = Path("/tmp/vision-memory-r11-new-canonical-latent-bridge-source-only-init.lock")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"R11_new bridge controller expected an object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"Blank JSONL row at {path}:{line_number}.")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"Non-object JSONL row at {path}:{line_number}.")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git(*args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return process.stdout.strip()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Malformed artifact inventory path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe artifact inventory path: {value!r}")
    return value


def _write_inventory(root: Path) -> dict[str, Any]:
    artifacts = []
    inventory_path = root / "artifact_inventory.json"
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path == inventory_path:
            continue
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    value = {
        "schema": INVENTORY_SCHEMA,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    _write_json(inventory_path, value)
    return value


def _validate_inventory(root: Path, *, schema: str) -> str:
    path = root / "artifact_inventory.json"
    value = _load(path)
    if value.get("schema") != schema or not isinstance(value.get("artifacts"), list):
        raise ValueError(f"Invalid artifact inventory schema: {path}")
    declared: set[str] = set()
    for row in value["artifacts"]:
        if not isinstance(row, Mapping):
            raise ValueError(f"Malformed artifact inventory row: {path}")
        relative = _safe_relative(row.get("path"))
        if relative in declared:
            raise ValueError(f"Duplicate artifact inventory path: {relative}")
        declared.add(relative)
        artifact = root.joinpath(*PurePosixPath(relative).parts)
        if (
            not artifact.is_file()
            or artifact.stat().st_size != row.get("bytes")
            or _sha256(artifact) != row.get("sha256")
        ):
            raise ValueError(f"Artifact inventory size/hash mismatch: {artifact}")
    actual = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != path
    }
    if declared != actual or value.get("artifact_count") != len(declared):
        raise ValueError(f"Artifact inventory file set mismatch: {root}")
    return _sha256(path)


def _deployment_audit(output_root: Path, *, hostname: str | None = None) -> dict[str, Any]:
    observed_host = platform.node() if hostname is None else hostname
    if not observed_host.startswith(EXPECTED_HOST_PREFIX):
        raise ValueError(f"Bridge is pinned to {EXPECTED_HOST_PREFIX!r}; observed host {observed_host!r}.")
    resolved_ssd = INSPIRE_SSD_ROOT.resolve()
    resolved_output = output_root.resolve()
    try:
        resolved_output.relative_to(resolved_ssd)
    except ValueError as exc:
        raise ValueError(f"Bridge output must be below {resolved_ssd}: {resolved_output}") from exc
    if output_root.exists():
        raise ValueError("Bridge controller requires a nonexistent fresh output root.")
    parent = resolved_output.parent
    while not parent.exists():
        if parent == parent.parent:
            raise ValueError("Bridge controller found no existing storage parent.")
        parent = parent.parent
    usage = shutil.disk_usage(parent)
    if usage.free < MINIMUM_FREE_BYTES:
        raise ValueError(f"Bridge free-space gate failed: {usage.free} < {MINIMUM_FREE_BYTES} bytes.")
    return {
        "passed": True,
        "hostname": observed_host,
        "required_hostname_prefix": EXPECTED_HOST_PREFIX,
        "ssd_root": str(resolved_ssd),
        "output_root": str(resolved_output),
        "disk_usage_path": str(parent),
        "disk_free_bytes": usage.free,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
    }


def _validate_snapshot(model_dir: Path, *, environment_key: str) -> dict[str, Any]:
    manifest = model_dir / ".snapshot_manifest.json"
    sidecar = model_dir / ".snapshot_manifest.json.sha256"
    expected = EXPECTED_ENVIRONMENT[environment_key]
    if not manifest.is_file() or not sidecar.is_file():
        raise ValueError(f"Model snapshot binding is missing: {model_dir}")
    observed = _sha256(manifest)
    if observed != expected:
        raise ValueError(f"{environment_key} snapshot hash drifted: {observed}")
    if sidecar.read_text(encoding="utf-8").strip() != f"{observed}  {manifest.name}":
        raise ValueError(f"Model snapshot sidecar drifted: {sidecar}")
    return {
        "model_dir": str(model_dir.resolve()),
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": observed,
        "sidecar_path": str(sidecar.resolve()),
    }


def _acquire_lock(*, mode: str, output_root: Path, expected_commit: str) -> dict[str, Any]:
    owner = {
        "schema": LOCK_SCHEMA,
        "owner_token": uuid.uuid4().hex,
        "pid": os.getpid(),
        "hostname": platform.node(),
        "mode": mode,
        "target_index": core.BRIDGE_TARGET_INDEX,
        "output_root": str(output_root.resolve()),
        "git_commit": expected_commit,
        "acquired_at_utc": _utc_now(),
    }
    try:
        LOCK_PATH.mkdir()
    except FileExistsError as exc:
        existing_path = LOCK_PATH / "owner.json"
        existing = _load(existing_path) if existing_path.is_file() else {"owner": "missing"}
        raise ValueError(f"Bridge suite lock is already held: {existing}") from exc
    owner_path = LOCK_PATH / "owner.json"
    try:
        _write_json(owner_path, owner)
    except Exception:
        LOCK_PATH.rmdir()
        raise
    return {
        "path": str(LOCK_PATH),
        "owner_path": str(owner_path),
        "owner_sha256": _sha256(owner_path),
        "owner": owner,
    }


def _release_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(lock["path"]))
    owner_path = Path(str(lock["owner_path"]))
    if (
        not path.is_dir()
        or not owner_path.is_file()
        or _load(owner_path) != lock.get("owner")
        or _sha256(owner_path) != lock.get("owner_sha256")
    ):
        raise RuntimeError("Bridge lock ownership changed before release.")
    owner_path.unlink()
    path.rmdir()
    return {**dict(lock), "released": True, "released_at_utc": _utc_now()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("technical-preflight", "formal"), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--phase1a-comparison", type=Path, required=True)
    parser.add_argument("--phase1a-raw-artifacts", type=Path, required=True)
    parser.add_argument("--phase1a-target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--preflight-terminal", type=Path)
    return parser


def _validate_preflight_terminal(
    path: Path | None,
    *,
    expected_commit: str,
    config_sha256: str,
    trainer_sha256: str,
    controller_sha256: str,
) -> dict[str, Any]:
    if path is None or not path.is_file() or path.name != "terminal.json":
        raise ValueError("Formal bridge requires the controller technical-preflight terminal.json.")
    value = _load(path)
    checks = value.get("execution_checks")
    valid = bool(
        value.get("schema") == TERMINAL_SCHEMA
        and value.get("status") == "technical_completed"
        and value.get("mode") == "technical-preflight"
        and value.get("target_index") == core.BRIDGE_TARGET_INDEX
        and value.get("target_segment_id") == core.BRIDGE_TARGET_SEGMENT_ID
        and value.get("technical_gate") is True
        and value.get("bridge_diagnostic_gate") is None
        and value.get("formal_success") is False
        and value.get("phase2_allowed") is False
        and value.get("child_exit_code") == 0
        and value.get("git_commit") == expected_commit
        and value.get("config_sha256") == config_sha256
        and value.get("trainer_sha256") == trainer_sha256
        and value.get("controller_sha256") == controller_sha256
        and isinstance(checks, Mapping)
        and bool(checks)
        and all(item is True for item in checks.values())
    )
    if not valid:
        raise ValueError("Bridge technical-preflight terminal failed exact validation.")
    inventory_sha = _validate_inventory(path.parent, schema=INVENTORY_SCHEMA)
    run = path.parent / "run"
    # Repeat the same strict raw child audit at formal admission; the previous
    # controller's top-level passed flag is not a substitute for current evidence.
    child = _validate_child(run, mode="technical-preflight", expected_commit=expected_commit)
    for field in ("summary_sha256", "manifest_sha256", "child_terminal_sha256"):
        if value.get(field) != child[field]:
            raise ValueError(f"Bridge technical-preflight terminal {field} binding drifted.")
    if value.get("child_inventory_sha256") != child["inventory_sha256"]:
        raise ValueError("Bridge technical-preflight child inventory binding drifted.")
    for field, filename in (("stdout_sha256", "stdout.log"), ("stderr_sha256", "stderr.log")):
        artifact = path.parent / filename
        if not artifact.is_file() or value.get(field) != _sha256(artifact):
            raise ValueError(f"Bridge technical-preflight terminal {field} binding drifted.")
    return {
        "terminal_path": str(path.resolve()),
        "terminal_sha256": _sha256(path),
        "inventory_path": str((path.parent / "artifact_inventory.json").resolve()),
        "inventory_sha256": inventory_sha,
        "summary_path": str((run / SUMMARY_FILE).resolve()),
        "summary_sha256": _sha256(run / SUMMARY_FILE),
        "passed": True,
    }


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    deployment = _deployment_audit(args.output_root)
    if not isinstance(args.expected_commit, str) or not _COMMIT_RE.fullmatch(args.expected_commit):
        raise ValueError("Bridge expected commit must be a full lowercase SHA-1.")
    head = _git("rev-parse", "HEAD")
    if head != args.expected_commit:
        raise ValueError(f"Bridge checkout mismatch: expected {args.expected_commit}, observed {head}.")
    if _git("status", "--porcelain"):
        raise ValueError("Bridge controller requires a clean source tree.")
    if _git("branch", "--show-current"):
        raise ValueError("Bridge controller requires a detached clean checkout.")
    config_sha256 = _sha256(CONFIG)
    if config_sha256 != core.BRIDGE_CONFIG_FILE_SHA256:
        raise ValueError("Bridge preregistered config file hash drifted.")
    config = core.validate_bridge_config(_load(CONFIG))
    for name in (
        "train",
        "dev",
        "teacher",
        "phase1a_comparison",
        "phase1a_raw_artifacts",
    ):
        if not getattr(args, name).is_file():
            raise ValueError(f"Bridge input file is missing: {name}.")
    for name in ("dreamlite", "reader", "phase1a_target_root"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"Bridge input directory is missing: {name}.")
    fixed = config["unchanged_contract"]
    if _sha256(args.train) != fixed["train_sha256"] or _sha256(args.dev) != fixed["dev_sha256"]:
        raise ValueError("Bridge train/dev binding drifted.")
    if _sha256(args.teacher) != core.BRIDGE_TEACHER_FILE_SHA256:
        raise ValueError("Bridge canonical teacher binding drifted.")
    parent = config["parent_phase1a"]
    if (
        _sha256(args.phase1a_comparison) != parent["comparison_sha256"]
        or _sha256(args.phase1a_raw_artifacts) != parent["raw_artifacts_sha256"]
    ):
        raise ValueError("Bridge Phase1A parent evidence binding drifted.")
    expected_target_root = Path(core.BRIDGE_PHASE1A_SOURCE_ROOT)
    if args.phase1a_target_root.resolve() != expected_target_root.resolve():
        raise ValueError("Bridge Phase1A target root differs from preregistration.")
    expected_teacher = Path(config["canonical_teacher"]["artifact_path"])
    if args.teacher.resolve() != expected_teacher.resolve():
        raise ValueError("Bridge teacher path differs from preregistration.")
    observed_environment = {name: os.environ.get(name) for name in EXPECTED_ENVIRONMENT}
    if observed_environment != EXPECTED_ENVIRONMENT:
        raise ValueError(f"Bridge deterministic/offline environment drifted: {observed_environment}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise ValueError("Bridge controller requires exactly two visible CUDA devices.")
    gpu_names = [torch.cuda.get_device_name(index) for index in range(2)]
    if any("H200" not in name.upper() for name in gpu_names):
        raise ValueError(f"Bridge controller requires two H200 GPUs: {gpu_names}")
    snapshots = {
        "dreamlite": _validate_snapshot(
            args.dreamlite,
            environment_key="VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256",
        ),
        "reader": _validate_snapshot(
            args.reader,
            environment_key="VLM_READER_SNAPSHOT_MANIFEST_SHA256",
        ),
    }
    trainer_sha256 = _sha256(TRAINER)
    controller_sha256 = _sha256(Path(__file__))
    core_sha256 = _sha256(Path(core.__file__))
    if args.mode == "technical-preflight":
        if args.preflight_terminal is not None:
            raise ValueError("Technical preflight forbids a prior preflight terminal.")
        prerequisite: dict[str, Any] | None = None
    else:
        prerequisite = _validate_preflight_terminal(
            args.preflight_terminal,
            expected_commit=args.expected_commit,
            config_sha256=config_sha256,
            trainer_sha256=trainer_sha256,
            controller_sha256=controller_sha256,
        )
        if (
            args.preflight_terminal is not None
            and args.output_root.resolve() == args.preflight_terminal.parent.resolve()
        ):
            raise ValueError("Formal bridge requires a fresh root distinct from preflight.")
    return {
        "deployment": deployment,
        "git_commit": head,
        "git_dirty": False,
        "git_detached": True,
        "config_sha256": config_sha256,
        "config_canonical_sha256": core.BRIDGE_CONFIG_CANONICAL_SHA256,
        "trainer_sha256": trainer_sha256,
        "controller_sha256": controller_sha256,
        "core_sha256": core_sha256,
        "environment": observed_environment,
        "gpu_names": gpu_names,
        "snapshots": snapshots,
        "preflight_prerequisite": prerequisite,
        "input_hashes": {
            "train": _sha256(args.train),
            "dev": _sha256(args.dev),
            "teacher": _sha256(args.teacher),
            "phase1a_comparison": _sha256(args.phase1a_comparison),
            "phase1a_raw_artifacts": _sha256(args.phase1a_raw_artifacts),
        },
    }


def _command(args: argparse.Namespace, run: Path) -> list[str]:
    return [
        sys.executable,
        str(TRAINER),
        "--mode",
        args.mode,
        "--train",
        str(args.train.resolve()),
        "--dev",
        str(args.dev.resolve()),
        "--dreamlite",
        str(args.dreamlite.resolve()),
        "--reader",
        str(args.reader.resolve()),
        "--teacher",
        str(args.teacher.resolve()),
        "--phase1a-comparison",
        str(args.phase1a_comparison.resolve()),
        "--phase1a-raw-artifacts",
        str(args.phase1a_raw_artifacts.resolve()),
        "--phase1a-target-root",
        str(args.phase1a_target_root.resolve()),
        "--output-dir",
        str(run.resolve()),
        "--dreamlite-device",
        "cuda:0",
        "--reader-device",
        "cuda:1",
        "--strict-determinism",
    ]


def _validate_hash_binding(summary: Mapping[str, Any], run: Path) -> dict[str, bool]:
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Bridge formal summary lacks artifact hash bindings.")
    mapping = {
        "manifest_sha256": "manifest.json",
        "metrics_sha256": METRICS_FILE,
        "evaluation_rows_sha256": ROWS_FILE,
        "endpoint_raw_sha256": "endpoint_raw.pt",
        "endpoint_png_sha256": "endpoint_raw.png",
        "technical_gate_sha256": "technical_gate.json",
    }
    checks = {
        key: (run / relative).is_file() and artifacts.get(key) == _sha256(run / relative)
        for key, relative in mapping.items()
    }
    if not all(checks.values()):
        raise ValueError(f"Bridge formal summary artifact hash drifted: {checks}")
    return checks


def _validate_initialization_binding(manifest: Mapping[str, Any], run: Path) -> bool:
    """Bind the pre-forward artifact; tensor arithmetic is audited by the worker and aggregator."""
    record = manifest.get("initialization_binding")
    if not isinstance(record, Mapping):
        return False
    path = run / "initialization" / "source_only_initialization.pt"
    return bool(
        record.get("schema") == trainer.INITIALIZATION_SCHEMA
        and record.get("passed") is True
        and Path(str(record.get("artifact_path", ""))).resolve() == path.resolve()
        and path.is_file()
        and record.get("artifact_bytes") == path.stat().st_size
        and record.get("artifact_sha256") == _sha256(path)
    )


def _validate_locked_preflight_target(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    """Anchor the complete answer-bearing target to immutable Phase1A bytes."""
    path = Path(core.BRIDGE_PHASE1A_SOURCE_ROOT) / "run" / "manifest.json"
    if not path.is_file():
        raise ValueError("Bridge locked Phase1A target manifest is missing.")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != core.BRIDGE_PARENT_TARGET_MANIFEST_SHA256:
        raise ValueError("Bridge locked Phase1A target manifest hash drifted.")
    parent = json.loads(payload)
    if not isinstance(parent, Mapping):
        raise ValueError("Bridge locked Phase1A target manifest is malformed.")
    target = parent.get("target_segment")
    if (
        parent.get("schema") != "vision_memory.r11-new-phase1a-manifest.v1"
        or parent.get("target_index") != core.BRIDGE_TARGET_INDEX
        or parent.get("target_segment_id") != core.BRIDGE_TARGET_SEGMENT_ID
        or not isinstance(target, Mapping)
        or target.get("schema") != "vision_memory.r5-compose-segments.v1"
        or target.get("segment_id") != core.BRIDGE_TARGET_SEGMENT_ID
    ):
        raise ValueError("Bridge locked Phase1A target manifest identity drifted.")
    candidate = manifest.get("target_segment")
    if (
        not isinstance(candidate, Mapping)
        or candidate != target
        or core.canonical_json_sha256(candidate) != core.canonical_json_sha256(target)
    ):
        raise ValueError("Bridge preflight target differs from the locked Phase1A parent manifest.")
    return target


def _validate_preflight_teacher_replay(
    rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the fixed teacher's four views without importing the aggregator."""
    target = _validate_locked_preflight_target(manifest)
    if not isinstance(target, Mapping) or target.get("segment_id") != core.BRIDGE_TARGET_SEGMENT_ID:
        raise ValueError("Bridge preflight teacher replay target manifest drifted.")
    query = target.get("query")
    events = target.get("events")
    if (
        not isinstance(query, Mapping) or not isinstance(query.get("choices"), list)
        or len(query["choices"]) != 4 or not all(isinstance(value, str) for value in query["choices"])
        or isinstance(query.get("target_index"), bool) or not isinstance(query.get("target_index"), int)
        or query["target_index"] not in range(4)
        or not isinstance(events, list) or not events or not isinstance(events[-1], Mapping)
        or not isinstance(target.get("query_source_episode_id"), str)
        or not isinstance(target.get("query_turn_index"), int)
        or len(rows) != 4
    ):
        raise ValueError("Bridge preflight teacher replay requires a valid target and exactly four rows.")
    choices, answer_index = query["choices"], query["target_index"]
    query_id = f"{target['query_source_episode_id']}:{target['query_turn_index']}"
    for row in rows:
        predicted = row.get("predicted_index")
        if (
            row.get("schema") != "vision_memory.r5-compose-causal-evaluation.v1"
            or row.get("suite") != trainer.SUITE
            or row.get("checkpoint") != "canonical_teacher" or row.get("condition") != "teacher"
            or row.get("item_id") != target["segment_id"] or row.get("pair_unit") != target["segment_id"]
            or row.get("donor_item_id") is not None
            or row.get("episode_id") != target["query_source_episode_id"] or row.get("query_id") != query_id
            or row.get("query_gap") != target.get("query_gap") or row.get("updater_count") != len(events)
            or row.get("target_event_kind") != events[-1].get("event_kind")
            or isinstance(row.get("target_index"), bool) or row.get("target_index") != answer_index
            or row.get("target_text") != choices[answer_index]
            or isinstance(predicted, bool) or not isinstance(predicted, int) or predicted not in range(4)
            or row.get("predicted_text") != choices[predicted]
        ):
            raise ValueError("Bridge preflight teacher replay rows are not bound to the manifest target.")
    # The shared pure helper checks exact reverse-cyclic views, finite ordered
    # logits, CE, and correctness. It invokes neither a model nor the aggregator.
    try:
        statistics = core.reader_checkpoint_statistics(rows, checkpoint="canonical_teacher", condition="teacher")
        for row in rows:
            permutation = tuple(row["permutation"])
            logits = [float(value) for value in row["choice_logits_ordered"]]
            predicted = permutation[max(range(4), key=logits.__getitem__)]
            ordered_target = permutation.index(answer_index)
            margin = logits[ordered_target] - max(value for i, value in enumerate(logits) if i != ordered_target)
            if row["predicted_index"] != predicted or not math.isclose(
                float(row.get("margin", math.nan)), margin, rel_tol=1e-5, abs_tol=1e-5,
            ):
                raise ValueError("prediction/margin drifted")
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"Bridge preflight teacher replay raw logits/permutations failed: {error}") from error
    if not core.teacher_replay_gate(statistics) or summary.get("teacher_replay_statistics") != statistics:
        raise ValueError("Bridge preflight teacher replay gate or summary statistics failed independently.")
    return statistics


def _validate_child(run: Path, *, mode: str, expected_commit: str) -> dict[str, Any]:
    inventory_sha256 = _validate_inventory(run, schema=trainer.INVENTORY_SCHEMA)
    summary = _load(run / SUMMARY_FILE)
    child_terminal = _load(run / "terminal.json")
    manifest = _load(run / "manifest.json")
    snapshot_end = _load(run / "model_snapshot_verification_end.json")
    expected_summary_schema = (
        trainer.SUMMARY_SCHEMA if mode == "formal" else "vision_memory.r11-new-canonical-latent-bridge-preflight.v1"
    )
    common = {
        "summary_schema": summary.get("schema") == expected_summary_schema,
        "summary_mode": summary.get("mode") == mode,
        "manifest_schema": manifest.get("schema") == trainer.MANIFEST_SCHEMA,
        "manifest_mode": manifest.get("mode") == mode,
        "protocol_exact": manifest.get("protocol") == core.BRIDGE_PROTOCOL,
        "source_only_information_boundary": (
            manifest.get("information_boundary", {}).get("initialization_teacher_assisted") is False
            and manifest.get("information_boundary", {}).get("optimization_teacher_supervised") is True
            and manifest.get("information_boundary", {}).get("answer_independent_writer_usable") is False
        ),
        "initialization_artifact_bound": _validate_initialization_binding(manifest, run),
        "commit_exact": manifest.get("git_commit") == expected_commit,
        "manifest_clean": manifest.get("git_dirty") is False,
        "target_exact": (
            summary.get("target_index") == core.BRIDGE_TARGET_INDEX
            and summary.get("target_segment_id") == core.BRIDGE_TARGET_SEGMENT_ID
            and manifest.get("target_index") == core.BRIDGE_TARGET_INDEX
            and manifest.get("target_segment_id") == core.BRIDGE_TARGET_SEGMENT_ID
        ),
        "formal_success_false": (
            summary.get("formal_success_gate") is False and child_terminal.get("formal_success") is False
        ),
        "phase2_blocked": summary.get("phase2_allowed") is False,
        "snapshot_end_passed": snapshot_end.get("passed") is True,
        "child_terminal_schema": child_terminal.get("schema") == trainer.TERMINAL_SCHEMA,
        "child_terminal_technical": (
            child_terminal.get("status") == "technical_completed" and child_terminal.get("technical_gate") is True
        ),
    }
    if not all(common.values()):
        raise ValueError(f"Bridge child common contract failed: {common}")
    rows = _load_jsonl(run / ROWS_FILE)
    if mode == "technical-preflight":
        preflight = _load(run / PREFLIGHT_FILE)
        audit = summary.get("audit")
        if not isinstance(audit, Mapping):
            raise ValueError("Bridge preflight audit must be a mapping.")
        teacher_statistics = _validate_preflight_teacher_replay(rows, summary, manifest)
        mode_checks = {
            "summary_matches_preflight": summary == preflight,
            "preflight_passed": summary.get("passed") is True,
            "optimizer_steps_zero": type(audit.get("optimizer_steps")) is int and audit["optimizer_steps"] == 0,
            "single_forward": type(audit.get("full_forward_calls")) is int and audit["full_forward_calls"] == 1,
            "single_backward": type(audit.get("backward_calls")) is int and audit["backward_calls"] == 1,
            **{f"audit_{name}": audit.get(name) is True for name in PREFLIGHT_REQUIRED_AUDIT_TRUE},
            "teacher_replay_recomputed": core.teacher_replay_gate(teacher_statistics),
            "child_bridge_not_evaluated": child_terminal.get("bridge_diagnostic_gate") is None,
            "bridge_not_evaluated": summary.get("bridge_result_evaluated") is False,
            "teacher_rows_exact": len(rows) == 4,
            "metrics_absent": not (run / METRICS_FILE).exists(),
            "checkpoint_zero_only": sorted(path.name for path in (run / "checkpoints").glob("*.pt")) == ["step-000.pt"],
        }
        artifact_checks: dict[str, bool] = {}
    else:
        metrics = _load_jsonl(run / METRICS_FILE)
        technical = _load(run / "technical_gate.json")
        observed_steps = [row.get("optimizer_step") for row in metrics]
        gates = summary.get("gates", {})
        initialization_audit = core.bridge_initialization_hypothesis_audit(
            endpoint_mse=summary.get("endpoint_distance_statistics", {}).get("mse"),
            endpoint_reader_mean_ce=summary.get("endpoint_reader_statistics", {}).get("mean_ce"),
            technical_gate=gates.get("technical_gate") is True,
            teacher_replay_gate=gates.get("teacher_replay_gate") is True,
            distance_pass=gates.get("bridge_distance_gate") is True,
            reader_transfer_pass=gates.get("endpoint_reader_transfer_gate") is True,
        )
        mode_checks = {
            "formal_completed": summary.get("status") == "completed",
            "technical_gate_passed": (
                technical.get("passed") is True
                and summary.get("technical_gate") == technical
                and summary.get("gates", {}).get("technical_gate") is True
            ),
            "receipts_exact": len(metrics) == core.BRIDGE_OPTIMIZER_STEPS,
            "steps_contiguous": observed_steps == list(range(1, core.BRIDGE_OPTIMIZER_STEPS + 1)),
            "rows_exact": len(rows) == 20,
            "checkpoints_exact": summary.get("checkpoint_steps_observed") == list(core.BRIDGE_CHECKPOINT_STEPS),
            "decision_exact": summary.get("decision") == core.bridge_decision(
                distance_pass=gates.get("bridge_distance_gate") is True,
                reader_transfer_pass=gates.get("endpoint_reader_transfer_gate") is True,
            ),
            "diagnostic_boolean": isinstance(summary.get("gates", {}).get("bridge_diagnostic_gate"), bool),
            "initialization_audit_exact": summary.get("secondary_solver_hypothesis_audit")
            == initialization_audit,
            "initialization_audit_decision_exact": summary.get("secondary_solver_hypothesis_decision")
            == core.bridge_initialization_hypothesis_decision(
                distance_pass=bool(
                    summary.get("gates", {}).get("bridge_distance_gate")
                ),
                reader_transfer_pass=bool(
                    summary.get("gates", {}).get("endpoint_reader_transfer_gate")
                ),
                audit=initialization_audit,
            ),
        }
        artifact_checks = _validate_hash_binding(summary, run)
    if not all(mode_checks.values()):
        raise ValueError(f"Bridge child mode contract failed: {mode_checks}")
    return {
        "passed": True,
        "common_checks": common,
        "mode_checks": mode_checks,
        "artifact_hash_checks": artifact_checks,
        "inventory_sha256": inventory_sha256,
        "summary": summary,
        "summary_sha256": _sha256(run / SUMMARY_FILE),
        "manifest_sha256": _sha256(run / "manifest.json"),
        "child_terminal_sha256": _sha256(run / "terminal.json"),
    }


def _run(args: argparse.Namespace, validation: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    if not args.output_root.is_dir() or any(args.output_root.iterdir()):
        raise RuntimeError("Bridge controller lost ownership of its fresh output root.")
    run = args.output_root / "run"
    command = _command(args, run)
    launch = {
        "schema": LAUNCH_SCHEMA,
        "status": "running",
        "created_at_utc": _utc_now(),
        "mode": args.mode,
        "target_index": core.BRIDGE_TARGET_INDEX,
        "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID,
        **dict(validation),
        "suite_lock": dict(lock),
        "command": command,
        "scientific_gate_evaluated_by_controller": False,
        "formal_success": False,
    }
    _write_json(args.output_root / "launch.json", launch)
    started = time.monotonic()
    with (
        (args.output_root / "stdout.log").open("x", encoding="utf-8") as stdout,
        (args.output_root / "stderr.log").open("x", encoding="utf-8") as stderr,
    ):
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=os.environ.copy(),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if process.returncode != 0:
        raise RuntimeError(f"Bridge trainer exited with code {process.returncode}.")
    child = _validate_child(run, mode=args.mode, expected_commit=args.expected_commit)
    summary = child.pop("summary")
    diagnostic = (
        {
            "evaluated": False,
            "bridge_diagnostic_gate": None,
            "distance_gate": None,
            "reader_transfer_gate": None,
            "decision": "not_evaluated_in_technical_preflight",
        }
        if args.mode == "technical-preflight"
        else {
            "evaluated": True,
            "bridge_diagnostic_gate": summary["gates"]["bridge_diagnostic_gate"],
            "distance_gate": summary["gates"]["bridge_distance_gate"],
            "reader_transfer_gate": summary["gates"]["endpoint_reader_transfer_gate"],
            "decision": summary["decision"],
        }
    )
    return {
        "schema": TERMINAL_SCHEMA,
        "status": "technical_completed",
        "completed_at_utc": _utc_now(),
        "mode": args.mode,
        "target_index": core.BRIDGE_TARGET_INDEX,
        "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID,
        "git_commit": args.expected_commit,
        "child_exit_code": process.returncode,
        "technical_gate": True,
        "bridge_diagnostic_gate": diagnostic["bridge_diagnostic_gate"],
        "diagnostic_result": diagnostic,
        "formal_success": False,
        "phase2_allowed": False,
        "execution_checks": {
            "child_contract_passed": child["passed"],
            "common_checks_passed": all(child["common_checks"].values()),
            "mode_checks_passed": all(child["mode_checks"].values()),
            "artifact_hash_checks_passed": all(child["artifact_hash_checks"].values()),
        },
        "summary_sha256": child["summary_sha256"],
        "manifest_sha256": child["manifest_sha256"],
        "child_terminal_sha256": child["child_terminal_sha256"],
        "child_inventory_sha256": child["inventory_sha256"],
        "stdout_sha256": _sha256(args.output_root / "stdout.log"),
        "stderr_sha256": _sha256(args.output_root / "stderr.log"),
        "config_sha256": validation["config_sha256"],
        "trainer_sha256": validation["trainer_sha256"],
        "controller_sha256": validation["controller_sha256"],
        "core_sha256": validation["core_sha256"],
        "elapsed_seconds": time.monotonic() - started,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    claimed = False
    lock: dict[str, Any] | None = None
    released: dict[str, Any] | None = None
    try:
        validation = _validate(args)
        lock = _acquire_lock(
            mode=args.mode,
            output_root=args.output_root,
            expected_commit=args.expected_commit,
        )
        args.output_root.mkdir(parents=True, exist_ok=False)
        claimed = True
        terminal = _run(args, validation, lock)
        released = _release_lock(lock)
        lock = None
        terminal["suite_lock_release"] = released
        _write_json(args.output_root / "terminal.json", terminal)
        _write_inventory(args.output_root)
    except Exception as exc:
        release_error: str | None = None
        if lock is not None:
            try:
                released = _release_lock(lock)
            except Exception as lock_exc:
                release_error = str(lock_exc)
        if claimed:
            try:
                _write_json(
                    args.output_root / "terminal.json",
                    {
                        "schema": TERMINAL_SCHEMA,
                        "status": "failed",
                        "completed_at_utc": _utc_now(),
                        "mode": args.mode,
                        "target_index": core.BRIDGE_TARGET_INDEX,
                        "technical_gate": False,
                        "bridge_diagnostic_gate": None,
                        "formal_success": False,
                        "phase2_allowed": False,
                        "error": str(exc),
                        "suite_lock_release": released,
                        "suite_lock_release_error": release_error,
                    },
                )
                _write_inventory(args.output_root)
            except Exception:
                pass
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "milestone": "r11_new_bridge_controller_completed",
                "mode": args.mode,
                "technical_gate": True,
                "bridge_diagnostic_gate": terminal["bridge_diagnostic_gate"],
                "formal_success": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
