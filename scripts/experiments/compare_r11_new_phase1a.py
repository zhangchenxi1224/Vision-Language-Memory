"""Independently aggregate the eight preregistered R11_new Phase 1A targets.

The controller and trainer summaries are treated as claims to verify, never as
the source of the result.  This script re-hashes the complete artifact trees,
validates the raw 256-step receipts and 16 fixed evaluation cells, verifies all
five checkpoint/image/hash triplets, and recomputes the core technical, target,
and eight-target arm gates.

Phase 1A is a query-level frozen-DreamLite reachability diagnostic.  Therefore
``formal_success`` is unconditionally false, including when all eight targets
pass.  Per the MVP route, an 8/8 diagnostic proceeds to the query-level Phase 2
oracle bank; Phase 1B remains a future state-level confirmation experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import REVERSE_CYCLIC4  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402
from vision_memory.training import r11_new_oracle as core  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "experiments" / "r11_new_frozen_dreamlite_oracle_phase1a.json"
TRAINER_PATH = ROOT / "scripts" / "train" / "r11_new_frozen_dreamlite_oracle.py"

CONTROLLER_LAUNCH_SCHEMA = "vision_memory.r11-new-phase1a-target-launch.v1"
CONTROLLER_TERMINAL_SCHEMA = "vision_memory.r11-new-phase1a-target-terminal.v1"
CONTROLLER_INVENTORY_SCHEMA = "vision_memory.r11-new-phase1a-target-inventory.v1"
TRAINER_MANIFEST_SCHEMA = "vision_memory.r11-new-phase1a-manifest.v1"
TRAINER_SUMMARY_SCHEMA = "vision_memory.r11-new-phase1a-summary.v1"
TRAINER_TERMINAL_SCHEMA = "vision_memory.r11-new-phase1a-terminal.v1"
TRAINER_INVENTORY_SCHEMA = "vision_memory.r11-new-phase1a-artifact-inventory.v1"
METRICS_SCHEMA = "vision_memory.r11-new-phase1a-metrics.v1"
TECHNICAL_GATE_SCHEMA = "vision_memory.r11-new-phase1a-technical-gate.v1"
CHECKPOINT_HASH_SCHEMA = "vision_memory.r11-new-phase1a-checkpoint-hashes.v1"
EVALUATION_SCHEMA = "vision_memory.r5-compose-causal-evaluation.v1"
EVALUATION_SUITE = "r11_new_f1_frozen_dreamlite_oracle"
IMPLEMENTATION_REVISION = "full-frozen-dreamlite-x-t-fp32-v1"

RAW_ARTIFACTS_SCHEMA = "vision_memory.r11-new-phase1a-raw-artifacts.v1"
COMPARISON_SCHEMA = "vision_memory.r11-new-phase1a-comparison.v1"

EXPECTED_MODEL_MANIFEST_SHA256 = {
    "dreamlite_mobile": "1bcf41b170c4b4a806bac6701cbdf4fabd5c3c53fa67415d065ab95ce2703159",
    "qwen_reader": "159a504daaae6dc412535978f087150a0eb8e50164afd70a8a17f83906f1127c",
}
EXPECTED_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256": EXPECTED_MODEL_MANIFEST_SHA256["dreamlite_mobile"],
    "VLM_READER_SNAPSHOT_MANIFEST_SHA256": EXPECTED_MODEL_MANIFEST_SHA256["qwen_reader"],
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
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
        raise ValueError(f"R11_new expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"R11_new JSONL contains a blank row: {path}:{line_number}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"R11_new JSONL row is not an object: {path}:{line_number}")
            values.append(value)
    return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _same_json(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _require_false(mapping: Mapping[str, Any], key: str, *, context: str) -> None:
    if mapping.get(key) is not False:
        raise ValueError(f"R11_new {context} must keep {key}=false.")


def _safe_relative(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"R11_new {context} has a malformed artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"R11_new {context} has an unsafe artifact path: {value!r}")
    return value


def _validate_inventory(
    root: Path,
    *,
    schema: str,
    required: set[str],
) -> tuple[str, list[dict[str, Any]]]:
    inventory_path = root / "artifact_inventory.json"
    inventory = _load(inventory_path)
    if inventory.get("schema") != schema:
        raise ValueError(f"R11_new artifact inventory schema mismatch: {root}")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"R11_new artifact inventory is empty: {root}")
    names: set[str] = set()
    verified: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"R11_new artifact inventory row is malformed: {root}")
        relative = _safe_relative(record.get("path"), context="artifact inventory")
        if relative in names:
            raise ValueError(f"R11_new artifact inventory contains duplicate path: {relative}")
        names.add(relative)
        artifact = root.joinpath(*PurePosixPath(relative).parts)
        byte_count = record.get("bytes")
        digest = record.get("sha256")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
            or not artifact.is_file()
            or artifact.stat().st_size != byte_count
            or _sha256(artifact) != digest
        ):
            raise ValueError(f"R11_new artifact failed size/SHA validation: {artifact}")
        verified.append({"path": relative, "bytes": byte_count, "sha256": digest})
    # Each writer omits only its own inventory.  In particular, the controller
    # inventory must bind the nested ``run/artifact_inventory.json`` rather than
    # excluding it by basename.
    inventory_resolved = inventory_path.resolve()
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != inventory_resolved
    }
    if names != actual:
        raise ValueError(
            f"R11_new inventory/file-set mismatch: missing={sorted(actual - names)}, stale={sorted(names - actual)}"
        )
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"R11_new inventory lacks required artifacts at {root}: {missing}")
    if schema == TRAINER_INVENTORY_SCHEMA and inventory.get("artifact_count") != len(records):
        raise ValueError(f"R11_new trainer inventory count mismatch: {root}")
    return _sha256(inventory_path), verified


def _run_required_artifacts() -> set[str]:
    checkpoints = {f"checkpoints/step-{step:03d}.pt" for step in core.R11_NEW_CHECKPOINT_STEPS}
    images = {f"images/step-{step:03d}.png" for step in core.R11_NEW_CHECKPOINT_STEPS}
    hashes = {f"checkpoint_hashes/step-{step:03d}.json" for step in core.R11_NEW_CHECKPOINT_STEPS}
    return (
        {
            ".r11_new_output_owner.json",
            "environment.txt",
            "runtime.json",
            "model_snapshot_verification_start.json",
            "model_snapshot_verification_end.json",
            "condition/official_full_condition.pt",
            "manifest.json",
            "metrics.jsonl",
            "target_evaluation_rows.jsonl",
            "endpoint_raw.pt",
            "endpoint_raw.png",
            "technical_gate.json",
            "r11_new_phase1a_summary.json",
            "terminal.json",
            "REPORT.md",
        }
        | checkpoints
        | images
        | hashes
    )


def _controller_required_artifacts() -> set[str]:
    return {
        "launch.json",
        "stdout.log",
        "stderr.log",
        "terminal.json",
        "run/artifact_inventory.json",
    } | {f"run/{name}" for name in _run_required_artifacts()}


def _validate_prerequisite_binding(
    value: Any,
    *,
    context: str,
    expected_mode: str,
    expected_target_index: int,
    expected_commit: str,
    config_sha256: str,
    trainer_sha256: str,
    expected_terminal_path: Path | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "terminal_path",
        "terminal_sha256",
        "inventory_path",
        "inventory_sha256",
        "mode",
        "target_index",
        "target_segment_id",
        "passed",
    }
    expected_target_id = core.R11_NEW_TARGET_IDS[expected_target_index]
    if (
        not isinstance(value, Mapping)
        or set(value) != expected_keys
        or value.get("mode") != expected_mode
        or type(value.get("target_index")) is not int
        or value.get("target_index") != expected_target_index
        or value.get("target_segment_id") != expected_target_id
        or value.get("passed") is not True
    ):
        raise ValueError(f"R11_new {context} prerequisite metadata drifted.")

    terminal_value = value.get("terminal_path")
    inventory_value = value.get("inventory_path")
    if not isinstance(terminal_value, str) or not isinstance(inventory_value, str):
        raise ValueError(f"R11_new {context} prerequisite paths are malformed.")
    terminal_path = Path(terminal_value)
    inventory_path = Path(inventory_value)
    if not terminal_path.is_absolute() or not inventory_path.is_absolute():
        raise ValueError(f"R11_new {context} prerequisite paths must be absolute.")
    terminal_path = terminal_path.resolve()
    inventory_path = inventory_path.resolve()
    if expected_terminal_path is not None and terminal_path != expected_terminal_path.resolve():
        raise ValueError(f"R11_new {context} must bind this aggregation's target-00 terminal.")
    expected_inventory_path = terminal_path.parent / "artifact_inventory.json"
    if (
        terminal_path.name != "terminal.json"
        or inventory_path != expected_inventory_path
        or not terminal_path.is_file()
        or not inventory_path.is_file()
    ):
        raise ValueError(f"R11_new {context} prerequisite artifacts are missing or misbound.")

    terminal_sha256 = value.get("terminal_sha256")
    inventory_sha256 = value.get("inventory_sha256")
    if (
        not isinstance(terminal_sha256, str)
        or _HEX64.fullmatch(terminal_sha256) is None
        or terminal_sha256 != _sha256(terminal_path)
    ):
        raise ValueError(f"R11_new {context} prerequisite terminal SHA256 mismatch.")
    if (
        not isinstance(inventory_sha256, str)
        or _HEX64.fullmatch(inventory_sha256) is None
        or inventory_sha256 != _sha256(inventory_path)
    ):
        raise ValueError(f"R11_new {context} prerequisite inventory SHA256 mismatch.")

    terminal = _load(terminal_path)
    execution_checks = terminal.get("execution_checks")
    valid_terminal = bool(
        terminal.get("schema") == CONTROLLER_TERMINAL_SCHEMA
        and terminal.get("status") == "technical_completed"
        and terminal.get("technical_completed") is True
        and terminal.get("mode") == expected_mode
        and type(terminal.get("target_index")) is int
        and terminal.get("target_index") == expected_target_index
        and terminal.get("target_segment_id") == expected_target_id
        and terminal.get("child_exit_code") == 0
        and terminal.get("git_commit") == expected_commit
        and terminal.get("config_sha256") == config_sha256
        and terminal.get("trainer_sha256") == trainer_sha256
        and terminal.get("formal_success") is False
        and terminal.get("scientific_success_claim") is False
        and isinstance(execution_checks, Mapping)
        and bool(execution_checks)
        and all(check is True for check in execution_checks.values())
    )
    if expected_mode == "technical-preflight":
        diagnostic = terminal.get("diagnostic_result")
        valid_terminal = bool(
            valid_terminal
            and isinstance(diagnostic, Mapping)
            and diagnostic.get("evaluated") is False
            and diagnostic.get("phase1a_query_level_reachability_gate") is None
            and diagnostic.get("result") == "not_evaluated_in_technical_preflight"
        )
    if not valid_terminal:
        raise ValueError(f"R11_new {context} prerequisite terminal failed exact validation.")

    inventory = _load(inventory_path)
    inventory_rows = inventory.get("artifacts")
    terminal_rows = (
        [row for row in inventory_rows if isinstance(row, Mapping) and row.get("path") == "terminal.json"]
        if isinstance(inventory_rows, list)
        else []
    )
    if (
        inventory.get("schema") != CONTROLLER_INVENTORY_SCHEMA
        or inventory.get("root") != str(terminal_path.parent.resolve())
        or len(terminal_rows) != 1
        or terminal_rows[0].get("bytes") != terminal_path.stat().st_size
        or terminal_rows[0].get("sha256") != terminal_sha256
    ):
        raise ValueError(f"R11_new {context} prerequisite terminal is not bound by its inventory.")
    return {
        "terminal_path": str(terminal_path),
        "terminal_sha256": terminal_sha256,
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_sha256,
        "mode": expected_mode,
        "target_index": expected_target_index,
        "target_segment_id": expected_target_id,
        "passed": True,
    }


def _validate_launch_prerequisites(
    launch: Mapping[str, Any],
    *,
    target_index: int,
    expected_commit: str,
    config_sha256: str,
    trainer_sha256: str,
    formal_target0_root: Path,
) -> dict[str, Any]:
    value = launch.get("prerequisites")
    if not isinstance(value, Mapping) or set(value) != {
        "preflight_required",
        "preflight",
        "target0_formal_required",
        "target0_formal",
        "passed",
    }:
        raise ValueError(f"R11_new launch prerequisites contract is missing or malformed: target {target_index}")
    if value.get("preflight_required") is not True or value.get("passed") is not True:
        raise ValueError(f"R11_new launch prerequisites failed closed: target {target_index}")
    preflight = _validate_prerequisite_binding(
        value.get("preflight"),
        context=f"target {target_index} preflight",
        expected_mode="technical-preflight",
        expected_target_index=0,
        expected_commit=expected_commit,
        config_sha256=config_sha256,
        trainer_sha256=trainer_sha256,
    )

    target0_formal: dict[str, Any] | None = None
    if target_index == 0:
        if value.get("target0_formal_required") is not False or value.get("target0_formal") is not None:
            raise ValueError("R11_new formal target 0 must not declare itself as a prerequisite.")
    else:
        if value.get("target0_formal_required") is not True:
            raise ValueError(f"R11_new formal target {target_index} must require target 0.")
        target0_formal = _validate_prerequisite_binding(
            value.get("target0_formal"),
            context=f"target {target_index} formal target 0",
            expected_mode="formal",
            expected_target_index=0,
            expected_commit=expected_commit,
            config_sha256=config_sha256,
            trainer_sha256=trainer_sha256,
            expected_terminal_path=formal_target0_root / "terminal.json",
        )
    return {"preflight": preflight, "target0_formal": target0_formal}


def _validate_checkpoint_artifacts(
    run_dir: Path,
    *,
    manifest_sha256: str,
    condition_sha256: str,
) -> dict[str, Any]:
    import torch

    expected_steps = tuple(core.R11_NEW_CHECKPOINT_STEPS)
    expected_pt = {f"step-{step:03d}.pt" for step in expected_steps}
    expected_png = {f"step-{step:03d}.png" for step in expected_steps}
    expected_records = {f"step-{step:03d}.json" for step in expected_steps}
    checkpoint_dir = run_dir / "checkpoints"
    image_dir = run_dir / "images"
    record_dir = run_dir / "checkpoint_hashes"
    if (
        not checkpoint_dir.is_dir()
        or not image_dir.is_dir()
        or not record_dir.is_dir()
        or {path.name for path in checkpoint_dir.glob("step-*.pt")} != expected_pt
        or {path.name for path in image_dir.glob("step-*.png")} != expected_png
        or {path.name for path in record_dir.glob("step-*.json")} != expected_records
    ):
        raise ValueError(f"R11_new fixed checkpoint file set drifted: {run_dir}")

    records: list[dict[str, Any]] = []
    for step in expected_steps:
        checkpoint = checkpoint_dir / f"step-{step:03d}.pt"
        image = image_dir / f"step-{step:03d}.png"
        record_path = record_dir / f"step-{step:03d}.json"
        record = _load(record_path)
        tensor_hashes = record.get("tensor_sha256")
        trajectory_hashes = tensor_hashes.get("trajectory_fp32") if isinstance(tensor_hashes, Mapping) else None
        checkpoint_suffix = f"/checkpoints/step-{step:03d}.pt"
        png_suffix = f"/images/step-{step:03d}.png"
        checkpoint_record_path = str(record.get("checkpoint_path", "")).replace("\\", "/")
        png_record_path = str(record.get("png_path", "")).replace("\\", "/")
        tensor_hash_values = []
        if isinstance(tensor_hashes, Mapping):
            tensor_hash_values = [
                tensor_hashes.get("x_T_fp32"),
                tensor_hashes.get("z_t_fp32"),
                *(trajectory_hashes if isinstance(trajectory_hashes, list) else []),
            ]
        valid = bool(
            record.get("schema") == CHECKPOINT_HASH_SCHEMA
            and record.get("optimizer_step") == step
            and checkpoint_record_path.endswith(checkpoint_suffix)
            and png_record_path.endswith(png_suffix)
            and record.get("checkpoint_bytes") == checkpoint.stat().st_size
            and record.get("png_bytes") == image.stat().st_size
            and record.get("checkpoint_sha256") == _sha256(checkpoint)
            and record.get("png_sha256") == _sha256(image)
            and record.get("trajectory_points") == core.R11_NEW_DIFFUSION_STEPS + 1
            and core.phase1a_effective_sigmas_match(record.get("effective_sigmas"))
            and isinstance(tensor_hashes, Mapping)
            and set(tensor_hashes) == {"x_T_fp32", "z_t_fp32", "trajectory_fp32"}
            and isinstance(trajectory_hashes, list)
            and len(trajectory_hashes) == core.R11_NEW_DIFFUSION_STEPS + 1
            and all(isinstance(value, str) and _HEX64.fullmatch(value) for value in tensor_hash_values)
        )
        if not valid:
            raise ValueError(f"R11_new checkpoint/hash record validation failed: {record_path}")
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"R11_new checkpoint payload cannot be read: {checkpoint}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"R11_new checkpoint payload is not a mapping: {checkpoint}")
        x_t = payload.get("x_T_fp32")
        z_t = payload.get("z_t_fp32")
        trajectory = payload.get("trajectory_fp32")
        payload_hashes = payload.get("tensor_sha256")
        tensors_valid = bool(
            isinstance(x_t, torch.Tensor)
            and isinstance(z_t, torch.Tensor)
            and isinstance(trajectory, (tuple, list))
            and len(trajectory) == core.R11_NEW_DIFFUSION_STEPS + 1
            and all(isinstance(value, torch.Tensor) for value in trajectory)
            and x_t.dtype == torch.float32
            and z_t.dtype == torch.float32
            and all(value.dtype == torch.float32 for value in trajectory)
            and x_t.shape == z_t.shape
            and all(value.shape == x_t.shape for value in trajectory)
            and payload.get("schema") == "vision_memory.r11-new-phase1a-checkpoint.v1"
            and payload.get("optimizer_step") == step
            and payload.get("manifest_sha256") == manifest_sha256
            and payload.get("condition_artifact_sha256") == condition_sha256
            and core.phase1a_effective_sigmas_match(payload.get("effective_sigmas"))
            # Keep raw observation binding exact; nominal-schedule tolerance
            # must not conceal a changed checkpoint record or tensor payload.
            and payload.get("effective_sigmas") == record.get("effective_sigmas")
            and isinstance(payload.get("optimizer"), Mapping)
            and isinstance(payload_hashes, Mapping)
            and payload_hashes == tensor_hashes
            and payload_hashes.get("x_T_fp32") == canonical_tensor_sha256(x_t)
            and payload_hashes.get("z_t_fp32") == canonical_tensor_sha256(z_t)
            and payload_hashes.get("trajectory_fp32") == [canonical_tensor_sha256(value) for value in trajectory]
        )
        if not tensors_valid:
            raise ValueError(f"R11_new checkpoint tensor hashes/payload drifted: {checkpoint}")
        records.append(
            {
                "optimizer_step": step,
                "checkpoint_sha256": record["checkpoint_sha256"],
                "png_sha256": record["png_sha256"],
                "record_sha256": _sha256(record_path),
                "tensor_sha256": dict(tensor_hashes),
            }
        )

    endpoint_pt = run_dir / "endpoint_raw.pt"
    endpoint_png = run_dir / "endpoint_raw.png"
    if _sha256(endpoint_pt) != _sha256(checkpoint_dir / "step-256.pt") or _sha256(endpoint_png) != _sha256(
        image_dir / "step-256.png"
    ):
        raise ValueError(f"R11_new raw endpoint is not the fixed step-256 checkpoint: {run_dir}")
    return {
        "steps": list(expected_steps),
        "records": records,
        "endpoint_raw_sha256": _sha256(endpoint_pt),
        "endpoint_png_sha256": _sha256(endpoint_png),
    }


def _finite_stat_block(value: Any, *, context: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"R11_new missing metric stat block: {context}")
    for key in ("minimum", "maximum", "rms", "norm"):
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"R11_new malformed metric stat block: {context}:{key}") from exc
        if not math.isfinite(number):
            raise ValueError(f"R11_new non-finite metric stat block: {context}:{key}")


def _validate_receipts(run_dir: Path, *, target_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_jsonl(run_dir / "metrics.jsonl")
    if len(rows) != core.R11_NEW_OPTIMIZER_STEPS:
        raise ValueError(f"R11_new target lacks exact 256 raw optimizer receipts: {run_dir} ({len(rows)})")
    for position, row in enumerate(rows, start=1):
        try:
            numeric = {
                key: float(row[key])
                for key in (
                    "loss_before_step",
                    "gradient_norm",
                    "gradient_nonzero_fraction",
                    "x_T_update_norm",
                    "learning_rate",
                    "weight_decay",
                    "elapsed_seconds",
                )
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"R11_new malformed optimizer receipt {position}: {run_dir}") from exc
        if (
            row.get("schema") != METRICS_SCHEMA
            or row.get("kind") != "optimizer_step"
            or row.get("optimizer_step") != position
            or row.get("target_segment_id") != target_id
            or not all(math.isfinite(value) for value in numeric.values())
            or numeric["gradient_norm"] <= 0.0
            or numeric["gradient_nonzero_fraction"] <= 0.0
            or numeric["x_T_update_norm"] <= 0.0
            or not math.isclose(numeric["learning_rate"], core.R11_NEW_LEARNING_RATE)
            or not math.isclose(numeric["weight_decay"], core.R11_NEW_WEIGHT_DECAY)
            or numeric["elapsed_seconds"] < 0.0
            or row.get("gradient_mode") != "full"
            or row.get("gradient_clipping_applied") is not False
            or row.get("trajectory_points") != core.R11_NEW_DIFFUSION_STEPS + 1
            or row.get("dreamlite_denoising_steps") != core.R11_NEW_DIFFUSION_STEPS
        ):
            raise ValueError(f"R11_new raw optimizer receipt contract drifted at step {position}: {run_dir}")
        for name in ("x_T_before_step", "x_T_after_step", "z_t_before_step", "image_before_step"):
            _finite_stat_block(row.get(name), context=f"step-{position}:{name}")
        trajectory = row.get("trajectory_before_step")
        if not isinstance(trajectory, list) or len(trajectory) != core.R11_NEW_DIFFUSION_STEPS + 1:
            raise ValueError(f"R11_new trajectory receipt drifted at step {position}: {run_dir}")
        for trajectory_index, state in enumerate(trajectory):
            _finite_stat_block(state, context=f"step-{position}:trajectory-{trajectory_index}")
            if state.get("trajectory_index") != trajectory_index:
                raise ValueError(f"R11_new trajectory index drifted at step {position}: {run_dir}")
            if trajectory_index:
                delta = float(state.get("delta_from_previous_norm", math.nan))
                if not math.isfinite(delta):
                    raise ValueError(f"R11_new trajectory delta is invalid at step {position}: {run_dir}")
    diagnostics = {
        "first_loss": float(rows[0]["loss_before_step"]),
        "final_loss": float(rows[-1]["loss_before_step"]),
        "minimum_loss": min(float(row["loss_before_step"]) for row in rows),
        "minimum_gradient_norm": min(float(row["gradient_norm"]) for row in rows),
        "minimum_gradient_nonzero_fraction": min(float(row["gradient_nonzero_fraction"]) for row in rows),
        "final_x_T_update_norm": float(rows[-1]["x_T_update_norm"]),
        "final_image_saturation_fraction": float(rows[-1]["image_before_step"]["saturation_fraction"]),
    }
    return rows, diagnostics


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _validate_evaluation_rows(
    run_dir: Path,
    *,
    target_id: str,
    manifest: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_jsonl(run_dir / "target_evaluation_rows.jsonl")
    expected_cells = {
        (checkpoint, condition, view)
        for checkpoint in ("m0", core.R11_NEW_PRIMARY_ENDPOINT)
        for condition in ("normal", "reset")
        for view in range(4)
    }
    if len(rows) != len(expected_cells):
        raise ValueError(f"R11_new target lacks exact 16 raw evaluation cells: {run_dir}")
    target_segment = manifest.get("target_segment")
    query = target_segment.get("query") if isinstance(target_segment, Mapping) else None
    if not isinstance(query, Mapping):
        raise ValueError(f"R11_new manifest lacks the target query contract: {run_dir}")
    choices = query.get("choices")
    original_target = query.get("target_index")
    if (
        not isinstance(choices, list)
        or len(choices) != 4
        or len(set(choices)) != 4
        or isinstance(original_target, bool)
        or not isinstance(original_target, int)
        or not 0 <= original_target < 4
    ):
        raise ValueError(f"R11_new manifest target query is malformed: {run_dir}")
    target_text = choices[original_target]

    observed: set[tuple[str, str, int]] = set()
    by_cell: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        try:
            checkpoint = str(row["checkpoint"])
            condition = str(row["condition"])
            view = int(row["view_index"])
            cell = (checkpoint, condition, view)
            permutation = list(row["permutation"])
            logits = [float(value) for value in row["choice_logits_ordered"]]
            target_index = int(row["target_index"])
            predicted_index = int(row["predicted_index"])
            ce = float(row["ce"])
            margin = float(row["margin"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"R11_new malformed raw evaluation row: {run_dir}") from exc
        if cell not in expected_cells or cell in observed:
            raise ValueError(f"R11_new evaluation cell is duplicate or unlocked: {run_dir}:{cell}")
        ordered_target = permutation.index(original_target) if original_target in permutation else -1
        predicted_ordered = max(range(4), key=logits.__getitem__) if len(logits) == 4 else -1
        predicted_from_logits = permutation[predicted_ordered] if predicted_ordered >= 0 else -1
        alternatives = [logits[index] for index in range(4) if index != ordered_target]
        expected_margin = logits[ordered_target] - max(alternatives) if ordered_target >= 0 else math.nan
        expected_ce = _logsumexp(logits) - logits[ordered_target] if ordered_target >= 0 else math.nan
        valid = bool(
            row.get("schema") == EVALUATION_SCHEMA
            and row.get("suite") == EVALUATION_SUITE
            and row.get("item_id") == target_id
            and row.get("pair_unit") == target_id
            and row.get("family") == "F1"
            and permutation == list(REVERSE_CYCLIC4[view])
            and target_index == original_target
            and 0 <= predicted_index < 4
            and predicted_index == predicted_from_logits
            and type(row.get("correct")) is bool
            and row["correct"] is (predicted_index == original_target)
            and row.get("target_text") == target_text
            and row.get("predicted_text") == choices[predicted_index]
            and len(logits) == 4
            and all(math.isfinite(value) for value in logits)
            and math.isfinite(ce)
            and math.isfinite(margin)
            and math.isclose(ce, expected_ce, rel_tol=2e-6, abs_tol=2e-6)
            and math.isclose(margin, expected_margin, rel_tol=2e-6, abs_tol=2e-6)
            and row.get("donor_item_id") is None
        )
        if not valid:
            raise ValueError(f"R11_new raw evaluation row failed recomputation: {run_dir}:{cell}")
        observed.add(cell)
        by_cell[cell] = row
    if observed != expected_cells:
        raise ValueError(f"R11_new raw evaluation coverage drifted: {run_dir}")

    # Reset is defined as decoding the same blank source state at both checkpoints.
    for view in range(4):
        m0 = by_cell[("m0", "reset", view)]
        endpoint = by_cell[(core.R11_NEW_PRIMARY_ENDPOINT, "reset", view)]
        for key in (
            "choice_logits_ordered",
            "ce",
            "margin",
            "predicted_index",
            "predicted_text",
            "correct",
        ):
            if not _same_json(m0.get(key), endpoint.get(key)):
                raise ValueError(f"R11_new reset changed across checkpoints: {run_dir}:view-{view}:{key}")

    statistics = core.phase1a_target_statistics(
        rows,
        suite=EVALUATION_SUITE,
        target_segment_id=target_id,
        endpoint=core.R11_NEW_PRIMARY_ENDPOINT,
    )
    return rows, json.loads(json.dumps(statistics, sort_keys=True))


def _expected_core_audit() -> dict[str, Any]:
    return {
        "checkpoint_steps_observed": list(core.R11_NEW_CHECKPOINT_STEPS),
        "latent_checkpoint_steps_observed": list(core.R11_NEW_CHECKPOINT_STEPS),
        "image_checkpoint_steps_observed": list(core.R11_NEW_CHECKPOINT_STEPS),
        "trainable_parameter_names": [core.R11_NEW_TRAINABLE_PARAMETER],
        "trainable_parameter_dtypes": {core.R11_NEW_TRAINABLE_PARAMETER: "torch.float32"},
        "frozen_components": {
            "dreamlite_unet": True,
            "condition_encoder": True,
            "vae": True,
            "reader": True,
        },
        "frozen_gradients_absent": True,
        "full_model_snapshots_unchanged": True,
        "information_boundary_passed": True,
        "source_contract_verified": True,
        "event_noise_contract_verified": True,
        "optimizer": "Adam",
        "learning_rate": core.R11_NEW_LEARNING_RATE,
        "weight_decay": core.R11_NEW_WEIGHT_DECAY,
        "gradient_clip": None,
        "primary_endpoint": core.R11_NEW_PRIMARY_ENDPOINT,
    }


def _validate_manifest(
    run_dir: Path,
    *,
    target_index: int,
    target_id: str,
    expected_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    manifest = _load(run_dir / "manifest.json")
    target_segment = manifest.get("target_segment")
    fixed = manifest.get("fixed_contract")
    information = manifest.get("information_boundary")
    source_rgb = manifest.get("source_rgb")
    source_latents = manifest.get("source_latents")
    initial_x_t = manifest.get("initial_x_T_fp32")
    if (
        manifest.get("schema") != TRAINER_MANIFEST_SCHEMA
        or manifest.get("protocol") != core.R11_NEW_PROTOCOL
        or manifest.get("implementation_revision") != IMPLEMENTATION_REVISION
        or manifest.get("mode") != "formal"
        or manifest.get("git_commit") != expected_commit
        or manifest.get("git_dirty") is not False
        or manifest.get("target_index") != target_index
        or manifest.get("target_segment_id") != target_id
        or manifest.get("selected_segment_ids") != list(core.R11_NEW_TARGET_IDS)
        or manifest.get("selected_segments_sha256") != core.R11_NEW_TARGETS_PAYLOAD_SHA256
        or manifest.get("train_sha256") != core.R11_NEW_TRAIN_SHA256
        or manifest.get("dev_sha256") != core.R11_NEW_DEV_SHA256
        or manifest.get("preregistered_config_sha256") != config_sha256
        or manifest.get("query_level_diagnostic_only") is not True
        or manifest.get("formal_success_gate") is not False
        or not isinstance(target_segment, Mapping)
        or target_segment.get("segment_id") != target_id
        or target_segment.get("family") != "F1"
        or not isinstance(target_segment.get("events"), list)
        or len(target_segment["events"]) != 1
        or not isinstance(source_rgb, Mapping)
        or source_rgb.get("value") != "127/255"
        or manifest.get("source_contract_verified") is not True
        or manifest.get("event_noise_contract_verified") is not True
        or not isinstance(source_latents, Mapping)
        or not isinstance(initial_x_t, Mapping)
        or source_latents.get("shape") != initial_x_t.get("shape")
        or initial_x_t.get("dtype") != "torch.float32"
        or not isinstance(fixed, Mapping)
    ):
        raise ValueError(f"R11_new manifest identity/fixed evidence drifted: target {target_index}")
    expected_fixed = {
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
        "optimizer_steps": core.R11_NEW_OPTIMIZER_STEPS,
        "gradient_clipping": None,
        "checkpoint_steps": list(core.R11_NEW_CHECKPOINT_STEPS),
        "training_views": "forward cyclic",
        "endpoint_views": "reverse cyclic",
        "primary_endpoint": core.R11_NEW_PRIMARY_ENDPOINT,
        "best_checkpoint_selection_forbidden": True,
        "reset": "decode(blank_source_latent)",
    }
    if any(fixed.get(key) != value for key, value in expected_fixed.items()):
        raise ValueError(f"R11_new manifest DreamLite/optimizer contract drifted: target {target_index}")
    if not isinstance(information, Mapping):
        raise ValueError(f"R11_new information-boundary artifact is missing: target {target_index}")
    try:
        boundary = core.validate_information_boundary(
            dreamlite_inputs=information["dreamlite_inputs"],
            noise_key=information["noise_key"],
            reader_loss_inputs=information["reader_loss_inputs"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"R11_new information boundary failed: target {target_index}") from exc
    if (
        boundary.get("passed") is not True
        or information.get("passed") is not True
        or information.get("query_used_only_by_frozen_reader_loss") is not True
        or information.get("choices_used_only_by_frozen_reader_loss") is not True
        or information.get("target_index_used_only_by_frozen_reader_loss") is not True
        or information.get("forbidden_writer_key_violations") != []
    ):
        raise ValueError(f"R11_new information boundary drifted: target {target_index}")
    condition = manifest.get("condition_artifact")
    condition_path = run_dir / "condition" / "official_full_condition.pt"
    if (
        not isinstance(condition, Mapping)
        or condition.get("recompute_matches") is not True
        or condition.get("bytes") != condition_path.stat().st_size
        or condition.get("sha256") != _sha256(condition_path)
    ):
        raise ValueError(f"R11_new condition artifact hash/recompute check failed: target {target_index}")
    return manifest


def _validate_snapshot_evidence(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    start = _load(run_dir / "model_snapshot_verification_start.json")
    end = _load(run_dir / "model_snapshot_verification_end.json")
    bindings = manifest.get("model_snapshot_payloads_start")
    if (
        start.get("schema") != "vision_memory.r11-new-phase1a-model-snapshot-start.v1"
        or start.get("bindings") != bindings
        or end.get("schema") != "vision_memory.r11-new-phase1a-model-snapshot-end.v1"
        or end.get("passed") is not True
        or end.get("bindings") != bindings
        or not isinstance(bindings, Mapping)
    ):
        raise ValueError(f"R11_new model snapshot start/end binding failed: {run_dir}")
    for name, expected_sha in EXPECTED_MODEL_MANIFEST_SHA256.items():
        binding = bindings.get(name)
        if not isinstance(binding, Mapping) or binding.get("passed") is not True:
            raise ValueError(f"R11_new model snapshot binding is missing: {run_dir}:{name}")
        if binding.get("manifest_sha256") != expected_sha:
            raise ValueError(f"R11_new model snapshot manifest SHA drifted: {run_dir}:{name}")


def _validate_summary_and_terminals(
    root: Path,
    *,
    target_index: int,
    target_id: str,
    expected_commit: str,
    config_sha256: str,
    trainer_sha256: str,
    recomputed_technical: Mapping[str, Any],
    recomputed_statistics: Mapping[str, Any],
    target_gate: bool,
    formal_target0_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_dir = root / "run"
    launch = _load(root / "launch.json")
    controller_terminal = _load(root / "terminal.json")
    trainer_terminal = _load(run_dir / "terminal.json")
    summary = _load(run_dir / "r11_new_phase1a_summary.json")
    technical = _load(run_dir / "technical_gate.json")
    command = launch.get("command")
    snapshot_manifests = launch.get("snapshot_manifests")
    storage = launch.get("storage_audit")
    if (
        launch.get("schema") != CONTROLLER_LAUNCH_SCHEMA
        or launch.get("status") != "running"
        or launch.get("mode") != "formal"
        or launch.get("git_commit") != expected_commit
        or launch.get("git_dirty") is not False
        or launch.get("target_index") != target_index
        or launch.get("target_segment_id") != target_id
        or launch.get("data_sha256") != {"train": core.R11_NEW_TRAIN_SHA256, "dev": core.R11_NEW_DEV_SHA256}
        or launch.get("environment") != EXPECTED_ENVIRONMENT
        or launch.get("config_sha256") != config_sha256
        or launch.get("trainer_sha256") != trainer_sha256
        or launch.get("canonical_r11_comparison_sha256") != core.R11_NEW_PARENT_R11_COMPARISON_SHA256
        or not isinstance(storage, Mapping)
        or storage.get("passed") is not True
        or storage.get("minimum_free_bytes") != core.R11_NEW_MINIMUM_FREE_BYTES
        or not isinstance(command, list)
        or any(not isinstance(item, str) for item in command)
        or "--mode" not in command
        or command[command.index("--mode") + 1] != "formal"
        or "--target-index" not in command
        or command[command.index("--target-index") + 1] != str(target_index)
        or bool(_FORBIDDEN_COMMAND_OPTIONS & set(command))
        or not isinstance(snapshot_manifests, Mapping)
        or snapshot_manifests.get("dreamlite", {}).get("manifest_sha256")
        != EXPECTED_MODEL_MANIFEST_SHA256["dreamlite_mobile"]
        or snapshot_manifests.get("reader", {}).get("manifest_sha256") != EXPECTED_MODEL_MANIFEST_SHA256["qwen_reader"]
    ):
        raise ValueError(f"R11_new controller launch contract drifted: target {target_index}")
    prerequisites = _validate_launch_prerequisites(
        launch,
        target_index=target_index,
        expected_commit=expected_commit,
        config_sha256=config_sha256,
        trainer_sha256=trainer_sha256,
        formal_target0_root=formal_target0_root,
    )

    execution_checks = controller_terminal.get("execution_checks")
    diagnostic = controller_terminal.get("diagnostic_result")
    if (
        controller_terminal.get("schema") != CONTROLLER_TERMINAL_SCHEMA
        or controller_terminal.get("status") != "technical_completed"
        or controller_terminal.get("technical_completed") is not True
        or controller_terminal.get("mode") != "formal"
        or controller_terminal.get("target_index") != target_index
        or controller_terminal.get("target_segment_id") != target_id
        or controller_terminal.get("child_exit_code") != 0
        or controller_terminal.get("error") is not None
        or not isinstance(execution_checks, Mapping)
        or not execution_checks
        or any(value is not True for value in execution_checks.values())
        or not isinstance(diagnostic, Mapping)
        or diagnostic.get("evaluated") is not True
        or diagnostic.get("phase1a_query_level_reachability_gate") is not target_gate
        or diagnostic.get("result")
        != ("query_level_reachability_passed" if target_gate else "query_level_reachability_not_found")
    ):
        raise ValueError(f"R11_new controller terminal failed closed: target {target_index}")
    _require_false(controller_terminal, "formal_success", context="controller terminal")
    _require_false(
        controller_terminal,
        "scientific_success_claim",
        context="controller terminal",
    )
    expected_terminal_hashes = {
        "summary_sha256": run_dir / "r11_new_phase1a_summary.json",
        "manifest_sha256": run_dir / "manifest.json",
        "technical_gate_sha256": run_dir / "technical_gate.json",
        "stdout_sha256": root / "stdout.log",
        "stderr_sha256": root / "stderr.log",
    }
    if any(controller_terminal.get(key) != _sha256(path) for key, path in expected_terminal_hashes.items()):
        raise ValueError(f"R11_new controller terminal hash binding failed: target {target_index}")
    if (
        controller_terminal.get("config_sha256") != config_sha256
        or controller_terminal.get("trainer_sha256") != trainer_sha256
    ):
        raise ValueError(f"R11_new controller executable/config hash drifted: target {target_index}")

    if (
        trainer_terminal.get("schema") != TRAINER_TERMINAL_SCHEMA
        or trainer_terminal.get("status") != "succeeded"
        or trainer_terminal.get("mode") != "formal"
        or trainer_terminal.get("technical_gate") is not True
        or trainer_terminal.get("diagnostic_evaluated") is not True
    ):
        raise ValueError(f"R11_new trainer terminal failed closed: target {target_index}")
    _require_false(trainer_terminal, "formal_success_gate", context="trainer terminal")

    gates = summary.get("gates")
    if (
        summary.get("schema") != TRAINER_SUMMARY_SCHEMA
        or summary.get("status") != "completed"
        or summary.get("mode") != "formal"
        or summary.get("protocol") != core.R11_NEW_PROTOCOL
        or summary.get("implementation_revision") != IMPLEMENTATION_REVISION
        or summary.get("git_commit") != expected_commit
        or summary.get("target_index") != target_index
        or summary.get("target_segment_id") != target_id
        or summary.get("optimizer_steps") != core.R11_NEW_OPTIMIZER_STEPS
        or summary.get("primary_endpoint") != core.R11_NEW_PRIMARY_ENDPOINT
        or summary.get("checkpoint_steps_observed") != list(core.R11_NEW_CHECKPOINT_STEPS)
        or not isinstance(gates, Mapping)
        or gates.get("technical_gate") is not True
        or gates.get("phase1a_query_level_reachability_gate") is not target_gate
        or gates.get("formal_success_gate") is not False
        or summary.get("query_level_diagnostic_only") is not True
        or summary.get("state_level_reachability_not_tested") is not True
        or summary.get("shared_writer_not_trained") is not True
        or summary.get("full_success_claim_allowed") is not False
        or not _same_json(summary.get("technical_gate"), technical)
        or not _same_json(summary.get("target_statistics"), recomputed_statistics)
    ):
        raise ValueError(f"R11_new trainer summary/recomputed result differs: target {target_index}")
    _require_false(summary, "formal_success_gate", context="trainer summary")

    technical_checks = {
        "schema": technical.get("schema") == TECHNICAL_GATE_SCHEMA,
        "stored_pass": technical.get("passed") is True,
        "core_audit": technical.get("core_audit") == _expected_core_audit(),
        "core_recompute": all(_same_json(technical.get(key), value) for key, value in recomputed_technical.items()),
        "optimizer_steps": technical.get("optimizer_steps_exact") is True,
        "four_step_schedule": technical.get("four_step_schedule_exact") is True,
        "checkpoint_hashes": technical.get("checkpoint_hashes_valid") is True,
        "condition": technical.get("condition_artifact_valid") is True,
        "only_x_T": technical.get("only_x_T_fp32_trainable") is True,
        "unet_frozen": technical.get("unet_frozen") is True,
        "vae_frozen": technical.get("vae_frozen") is True,
        "conditioner_frozen": technical.get("text_encoder_frozen") is True,
        "reader_frozen": technical.get("reader_frozen") is True,
        "optimizer": technical.get("optimizer_contract_valid") is True,
        "no_clip": technical.get("gradient_clipping_forbidden_and_absent") is True,
        "snapshots": technical.get("snapshots_unchanged") is True,
        "finite": technical.get("finite_metrics") is True,
    }
    if not all(technical_checks.values()):
        raise ValueError(
            f"R11_new stored technical audit differs from independent recomputation: "
            f"target {target_index}:{technical_checks}"
        )
    artifacts = summary.get("artifacts")
    expected_artifacts = {
        "manifest_sha256": run_dir / "manifest.json",
        "metrics_sha256": run_dir / "metrics.jsonl",
        "evaluation_rows_sha256": run_dir / "target_evaluation_rows.jsonl",
        "endpoint_raw_sha256": run_dir / "endpoint_raw.pt",
        "endpoint_png_sha256": run_dir / "endpoint_raw.png",
        "snapshot_end_sha256": run_dir / "model_snapshot_verification_end.json",
        "technical_gate_sha256": run_dir / "technical_gate.json",
    }
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != set(expected_artifacts)
        or any(artifacts.get(key) != _sha256(path) for key, path in expected_artifacts.items())
    ):
        raise ValueError(f"R11_new trainer summary artifact binding failed: target {target_index}")
    return launch, controller_terminal, summary, prerequisites


def _validate_target(
    root: Path,
    *,
    target_index: int,
    expected_commit: str,
    config_sha256: str,
    trainer_sha256: str,
    formal_target0_root: Path,
) -> dict[str, Any]:
    target_id = core.R11_NEW_TARGET_IDS[target_index]
    run_dir = root / "run"
    if not root.is_dir() or not run_dir.is_dir():
        raise ValueError(f"R11_new controller/run root is missing: {root}")
    controller_inventory_sha, controller_inventory = _validate_inventory(
        root,
        schema=CONTROLLER_INVENTORY_SCHEMA,
        required=_controller_required_artifacts(),
    )
    trainer_inventory_sha, _trainer_inventory = _validate_inventory(
        run_dir,
        schema=TRAINER_INVENTORY_SCHEMA,
        required=_run_required_artifacts(),
    )
    manifest = _validate_manifest(
        run_dir,
        target_index=target_index,
        target_id=target_id,
        expected_commit=expected_commit,
        config_sha256=config_sha256,
    )
    _validate_snapshot_evidence(run_dir, manifest)
    checkpoints = _validate_checkpoint_artifacts(
        run_dir,
        manifest_sha256=_sha256(run_dir / "manifest.json"),
        condition_sha256=str(manifest["condition_artifact"]["sha256"]),
    )
    receipts, training_diagnostics = _validate_receipts(run_dir, target_id=target_id)
    _evaluation_rows, statistics = _validate_evaluation_rows(
        run_dir,
        target_id=target_id,
        manifest=manifest,
    )
    recomputed_technical = core.phase1a_technical_gate(
        receipts,
        target_segment_id=target_id,
        audit=_expected_core_audit(),
    )
    if recomputed_technical.get("passed") is not True:
        raise ValueError(f"R11_new independent technical gate failed: target {target_index}")
    target_gate = core.phase1a_target_gate(statistics, technical_gate=True)
    launch, controller_terminal, summary, prerequisites = _validate_summary_and_terminals(
        root,
        target_index=target_index,
        target_id=target_id,
        expected_commit=expected_commit,
        config_sha256=config_sha256,
        trainer_sha256=trainer_sha256,
        recomputed_technical=recomputed_technical,
        recomputed_statistics=statistics,
        target_gate=target_gate,
        formal_target0_root=formal_target0_root,
    )
    return {
        "_validated_prerequisites": prerequisites,
        "target_index": target_index,
        "target_segment_id": target_id,
        "technical_gate": True,
        "target_reachability_gate": target_gate,
        "target_statistics": statistics,
        "training_diagnostics": training_diagnostics,
        "recomputed_technical_gate": recomputed_technical,
        "wall_clock_seconds": summary.get("wall_clock_seconds"),
        "source_root": str(root.resolve()),
        "source_hashes": {
            "launch_sha256": _sha256(root / "launch.json"),
            "controller_terminal_sha256": _sha256(root / "terminal.json"),
            "controller_inventory_sha256": controller_inventory_sha,
            "trainer_inventory_sha256": trainer_inventory_sha,
            "manifest_sha256": _sha256(run_dir / "manifest.json"),
            "metrics_sha256": _sha256(run_dir / "metrics.jsonl"),
            "evaluation_rows_sha256": _sha256(run_dir / "target_evaluation_rows.jsonl"),
            "technical_gate_sha256": _sha256(run_dir / "technical_gate.json"),
            "summary_sha256": _sha256(run_dir / "r11_new_phase1a_summary.json"),
        },
        "checkpoint_artifacts": checkpoints,
        "raw_artifact_count": len(controller_inventory),
        "controller_diagnostic_result": controller_terminal["diagnostic_result"]["result"],
        "source_training_git_commit": launch["git_commit"],
    }


def _precheck_unique_targets(roots: Sequence[Path]) -> list[tuple[int, Path]]:
    if len(roots) != len(core.R11_NEW_TARGET_IDS):
        raise ValueError("R11_new aggregation requires exactly eight controller roots.")
    declared: list[tuple[int, str, Path]] = []
    for root in roots:
        terminal_path = root / "terminal.json"
        if not terminal_path.is_file():
            raise ValueError(f"R11_new controller terminal is missing: {terminal_path}")
        terminal = _load(terminal_path)
        index = terminal.get("target_index")
        target_id = terminal.get("target_segment_id")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"R11_new controller terminal target index is malformed: {root}")
        declared.append((index, str(target_id), root))
    indices = [value[0] for value in declared]
    target_ids = [value[1] for value in declared]
    if len(set(indices)) != len(indices) or len(set(target_ids)) != len(target_ids):
        raise ValueError("R11_new aggregation rejects a duplicate target controller root.")
    if set(indices) != set(range(8)):
        raise ValueError(f"R11_new target-index coverage is not exact 0..7: {sorted(indices)}")
    expected_ids = set(core.R11_NEW_TARGET_IDS)
    if set(target_ids) != expected_ids:
        raise ValueError("R11_new target-segment coverage differs from the locked unique eight targets.")
    for index, target_id, _root in declared:
        if target_id != core.R11_NEW_TARGET_IDS[index]:
            raise ValueError(f"R11_new target index/segment mapping drifted at index {index}.")
    return sorted(((index, root) for index, _target_id, root in declared), key=lambda pair: pair[0])


def _decision(arm_passed: bool) -> tuple[str, str]:
    if arm_passed:
        return (
            "proceed_to_phase2_oracle_bank_mvp",
            "All eight independently optimized x_T targets passed the fixed query-level full-chain "
            "reachability gate. Build the train-split query-level Oracle Latent Bank next; Phase 1B "
            "remains deferred for future state-level confirmation.",
        )
    return (
        "run_canonical_r11_latent_bridge_distance_diagnostic",
        "The preregistered 8/8 Phase 1A arm gate was not met despite technically valid runs. "
        "Next run the minimal bridge-distance diagnostic against the known readable canonical-R11 "
        "latent; do not lower the gate or infer mathematical non-reachability.",
    )


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# R11_new Phase 1A 八目标独立复算报告",
        "",
        "> 本报告由 controller roots、原始 256-step receipts、16-cell endpoint rows 与固定 checkpoint hashes 独立复算；不以 trainer/controller summary 作为结果来源。",
        "",
        "## 三层结论",
        "",
        "| 层级 | 结果 | 含义 |",
        "| --- | --- | --- |",
        "| 工程通过 | 通过 | 8 个 target 均满足 exact receipts、四步 DreamLite、冻结/梯度、固定 checkpoint、snapshot 与 artifact hash 契约。 |",
        (
            "| 诊断通过 | "
            + ("通过" if result["phase1a_query_level_reachability_gate"] else "未通过")
            + f"（{result['target_pass_count']}/8） | 仅表示 query-level Frozen-DreamLite endpoint reachability。 |"
        ),
        "| 科学成功 | `false` | 未证明 state-level memory，未训练或验证共享 writer，也未做 held-out、多 seed、rollout 和完整因果确认。 |",
        "",
        f"- 决策：`{result['decision']}`",
        f"- 原因：{result['decision_reason']}",
        f"- 训练 commit：`{result['source_training_git_commit']}`",
        f"- 固定 endpoint：`{core.R11_NEW_PRIMARY_ENDPOINT}`",
        "- Phase 1B：本 MVP 路线中不作为 Phase 2 的前置门，仅保留为未来 state-level confirmation。",
        "- `formal_success = false`；即使 8/8，也不能表述为 R11_new 或 Picture Memory 训练成功。",
        "",
        "## 逐目标原始 endpoint 复算",
        "",
        "| Target | Gate | M0 normal CE | Endpoint normal CE | Relative change | Improved views | Accuracy delta | Normal/reset DiD |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in result["targets"]:
        statistics = target["target_statistics"]
        lines.append(
            f"| {target['target_index']} | "
            f"{'PASS' if target['target_reachability_gate'] else 'FAIL'} | "
            f"{float(statistics['m0_normal_mean_ce']):.6f} | "
            f"{float(statistics['endpoint_normal_mean_ce']):.6f} | "
            f"{float(statistics['relative_change']):+.2%} | "
            f"{statistics['improved_choice_views']}/4 | "
            f"{float(statistics['accuracy_delta']):+.3f} | "
            f"{float(statistics['normal_reset_difference_in_differences']):+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "Phase 1A 优化的是每个 query 各自的 initial latent `x_T`。它不证明一个共同 endpoint 支持同状态多 query，也不证明 query-free shared writer 已学会。若 8/8 通过，下一步只允许构建现有 train split 的 query-level Phase 2 MVP oracle bank；后续 Phase 3 仍只是共享 writer 可学习性诊断，必须继续保留固定门槛和因果评测。",
            "",
        ]
    )
    return "\n".join(lines)


def compare(run_root: Path, output_dir: Path, *, expected_commit: str) -> dict[str, Any]:
    if _HEX40.fullmatch(expected_commit) is None:
        raise ValueError("R11_new expected training commit must be an exact lowercase 40-character SHA-1.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R11_new aggregation refuses a non-empty output directory.")
    if not CONFIG_PATH.is_file() or not TRAINER_PATH.is_file():
        raise ValueError("R11_new aggregation cannot locate the locked config/trainer.")
    config = _load(CONFIG_PATH)
    validation = core.validate_phase1a_config(config)
    if validation.get("passed") is not True:
        raise ValueError("R11_new immutable config validation failed during aggregation.")
    config_sha256 = _sha256(CONFIG_PATH)
    trainer_sha256 = _sha256(TRAINER_PATH)

    roots = [run_root / f"target-{index:02d}" for index in range(8)]
    ordered = _precheck_unique_targets(roots)
    formal_target0_root = next(root for index, root in ordered if index == 0)
    targets = [
        _validate_target(
            root,
            target_index=index,
            expected_commit=expected_commit,
            config_sha256=config_sha256,
            trainer_sha256=trainer_sha256,
            formal_target0_root=formal_target0_root,
        )
        for index, root in ordered
    ]
    prerequisite_audits = [target.pop("_validated_prerequisites") for target in targets]
    shared_preflight = prerequisite_audits[0]["preflight"]
    if any(not _same_json(audit.get("preflight"), shared_preflight) for audit in prerequisite_audits[1:]):
        raise ValueError("R11_new all formal targets must bind one identical technical-preflight prerequisite.")
    if {target["source_training_git_commit"] for target in targets} != {expected_commit}:
        raise ValueError("R11_new immutable source commit drifted across targets.")
    arm_gate = core.phase1a_arm_gate(targets)
    if arm_gate.get("coverage_valid") is not True:
        raise ValueError("R11_new independently recomputed arm coverage is invalid.")
    decision, reason = _decision(bool(arm_gate["passed"]))
    pass_count = int(arm_gate["target_pass_count"])
    aggregation_commit = _git("rev-parse", "HEAD")

    raw_artifacts = {
        "schema": RAW_ARTIFACTS_SCHEMA,
        "status": "verified",
        "source_training_git_commit": expected_commit,
        "aggregation_git_commit": aggregation_commit,
        "immutable_config_sha256": config_sha256,
        "trainer_sha256": trainer_sha256,
        "selected_segments_sha256": core.R11_NEW_TARGETS_PAYLOAD_SHA256,
        "target_count": len(targets),
        "optimizer_receipts_per_target": core.R11_NEW_OPTIMIZER_STEPS,
        "evaluation_rows_per_target": 16,
        "checkpoint_steps": list(core.R11_NEW_CHECKPOINT_STEPS),
        "formal_success": False,
        "targets": [
            {
                "target_index": target["target_index"],
                "target_segment_id": target["target_segment_id"],
                "source_root": target["source_root"],
                "source_hashes": target["source_hashes"],
                "checkpoint_artifacts": target["checkpoint_artifacts"],
                "raw_artifact_count": target["raw_artifact_count"],
            }
            for target in targets
        ],
    }
    result = {
        "schema": COMPARISON_SCHEMA,
        "status": "completed",
        "phase": "phase1a_query_level_frozen_dreamlite_bridge_oracle",
        "engineering_gate": True,
        "phase1a_query_level_reachability_gate": bool(arm_gate["passed"]),
        "arm_gate": arm_gate,
        "target_pass_count": pass_count,
        "passed_target_indices": [target["target_index"] for target in targets if target["target_reachability_gate"]],
        "failed_target_indices": [
            target["target_index"] for target in targets if not target["target_reachability_gate"]
        ],
        "decision": decision,
        "decision_reason": reason,
        "next_stage": "phase2_oracle_bank_mvp" if arm_gate["passed"] else "bridge_distance_diagnostic",
        "phase1b_status": "deferred_future_state_level_confirmation",
        "query_level_diagnostic_only": True,
        "state_level_reachability_not_tested": True,
        "shared_writer_not_trained": True,
        "formal_success": False,
        "scientific_success_claim": False,
        "source_training_git_commit": expected_commit,
        "aggregation_git_commit": aggregation_commit,
        "selected_segments_sha256": core.R11_NEW_TARGETS_PAYLOAD_SHA256,
        "raw_artifacts_sha256": None,
        "targets": targets,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "RAW_ARTIFACTS.json", raw_artifacts)
    result["raw_artifacts_sha256"] = _sha256(output_dir / "RAW_ARTIFACTS.json")
    _write_json(output_dir / "comparison.json", result)
    (output_dir / "REPORT.md").write_text(_report(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args(argv)
    result = compare(args.run_root, args.output_dir, expected_commit=args.expected_commit)
    print(
        json.dumps(
            {
                "engineering_gate": result["engineering_gate"],
                "phase1a_query_level_reachability_gate": result["phase1a_query_level_reachability_gate"],
                "target_pass_count": result["target_pass_count"],
                "decision": result["decision"],
                "formal_success": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
