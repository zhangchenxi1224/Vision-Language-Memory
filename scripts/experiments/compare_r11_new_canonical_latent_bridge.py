"""Independent raw-artifact aggregation for the R11_new canonical-latent bridge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.inspire import run_r11_new_canonical_latent_bridge as controller  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402
from vision_memory.training import r11_new_bridge as core  # noqa: E402
from vision_memory.training.r11_new_oracle import (  # noqa: E402
    phase1a_effective_sigmas_match,
)


CONFIG = ROOT / "configs" / "experiments" / "r11_new_canonical_latent_bridge_target01.json"
COMPARISON_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-comparison.v1"
RAW_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-raw-artifacts.v1"
INVENTORY_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-aggregation-inventory.v1"
TRAINER_MANIFEST_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-manifest.v1"
TRAINER_METRICS_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-metrics.v1"
TRAINER_CHECKPOINT_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-checkpoint.v1"
TRAINER_CHECKPOINT_HASH_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-checkpoint-hashes.v1"
TRAINER_TECHNICAL_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-technical-gate.v1"
TRAINER_SUMMARY_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-summary.v1"
TRAINER_PREFLIGHT_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-preflight.v1"
TRAINER_TERMINAL_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-terminal.v1"
TRAINER_INVENTORY_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-artifact-inventory.v1"
EXPECTED_SIGMAS = (0.5, 0.375, 0.25, 0.125)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Bridge aggregation expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"Bridge aggregation found blank JSONL row {line_number}: {path}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"Bridge aggregation found non-object JSONL row {line_number}.")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Malformed inventory path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe inventory path: {value!r}")
    return value


def _validate_inventory(
    root: Path,
    *,
    schema: str,
    require_artifact_count: bool = True,
) -> dict[str, Any]:
    inventory_path = root / "artifact_inventory.json"
    inventory = _load(inventory_path)
    rows = inventory.get("artifacts")
    if inventory.get("schema") != schema or not isinstance(rows, list):
        raise ValueError(f"Bridge aggregation inventory schema drifted: {inventory_path}")
    declared: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"Malformed inventory row: {inventory_path}")
        relative = _safe_relative(row.get("path"))
        if relative in declared:
            raise ValueError(f"Duplicate inventory member: {relative}")
        declared.add(relative)
        artifact = root.joinpath(*PurePosixPath(relative).parts)
        if (
            not artifact.is_file()
            or artifact.stat().st_size != row.get("bytes")
            or _sha256(artifact) != row.get("sha256")
        ):
            raise ValueError(f"Inventory size/hash mismatch: {artifact}")
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path != inventory_path
    }
    count_valid = (
        inventory.get("artifact_count") == len(declared)
        if require_artifact_count
        else "artifact_count" not in inventory or inventory.get("artifact_count") == len(declared)
    )
    if declared != actual or not count_valid:
        raise ValueError(f"Inventory file set mismatch: {root}")
    return {
        "path": str(inventory_path.resolve()),
        "sha256": _sha256(inventory_path),
        "artifact_count": len(declared),
    }


def _write_inventory(root: Path) -> dict[str, Any]:
    inventory_path = root / "artifact_inventory.json"
    rows = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path == inventory_path:
            continue
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    value = {
        "schema": INVENTORY_SCHEMA,
        "artifact_count": len(rows),
        "artifacts": rows,
    }
    _write_json(inventory_path, value)
    return value


def _equal_float(left: Any, right: Any, *, rel_tol: float = 1e-6, abs_tol: float = 1e-8) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left_float)
        and math.isfinite(right_float)
        and math.isclose(left_float, right_float, rel_tol=rel_tol, abs_tol=abs_tol)
    )


def _sigmas_exact(values: Any) -> bool:
    return phase1a_effective_sigmas_match(values)


def _validate_config() -> dict[str, Any]:
    if _sha256(CONFIG) != core.BRIDGE_CONFIG_FILE_SHA256:
        raise ValueError("Bridge aggregation preregistered config file hash drifted.")
    return core.validate_bridge_config(_load(CONFIG))


def _validate_parent_target(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(config["exact_parity_bindings"]["phase1a_valid_source_root"])
    if not root.is_dir():
        raise ValueError(f"Bridge locked Phase1A target root is missing: {root}")
    inventory = _validate_inventory(
        root,
        schema="vision_memory.r11-new-phase1a-target-inventory.v1",
        require_artifact_count=False,
    )
    manifest_path = root / "run" / "manifest.json"
    if _sha256(manifest_path) != core.BRIDGE_PARENT_TARGET_MANIFEST_SHA256:
        raise ValueError("Bridge locked Phase1A target manifest hash drifted.")
    manifest = _load(manifest_path)
    segment = manifest.get("target_segment")
    query = segment.get("query") if isinstance(segment, Mapping) else None
    if (
        manifest.get("schema") != "vision_memory.r11-new-phase1a-manifest.v1"
        or manifest.get("target_index") != core.BRIDGE_TARGET_INDEX
        or manifest.get("target_segment_id") != core.BRIDGE_TARGET_SEGMENT_ID
        or not isinstance(segment, Mapping)
        or segment.get("schema") != "vision_memory.r5-compose-segments.v1"
        or segment.get("segment_id") != core.BRIDGE_TARGET_SEGMENT_ID
        or not isinstance(query, Mapping)
        or not isinstance(query.get("choices"), list)
        or len(query["choices"]) != 4
        or not isinstance(query.get("target_index"), int)
        or isinstance(query.get("target_index"), bool)
        or query["target_index"] not in range(4)
    ):
        raise ValueError("Bridge locked Phase1A target/query contract drifted.")
    return {
        "root": str(root.resolve()),
        "inventory": inventory,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "target_segment": dict(segment),
    }


def _validate_controller_root(
    root: Path,
    *,
    mode: str,
    expected_commit: str,
) -> dict[str, Any]:
    inventory = _validate_inventory(root, schema=controller.INVENTORY_SCHEMA)
    launch = _load(root / "launch.json")
    terminal = _load(root / "terminal.json")
    if (
        launch.get("schema") != controller.LAUNCH_SCHEMA
        or launch.get("mode") != mode
        or launch.get("git_commit") != expected_commit
        or launch.get("git_dirty") is not False
        or launch.get("git_detached") is not True
        or launch.get("target_index") != core.BRIDGE_TARGET_INDEX
        or launch.get("target_segment_id") != core.BRIDGE_TARGET_SEGMENT_ID
        or launch.get("config_sha256") != core.BRIDGE_CONFIG_FILE_SHA256
        or launch.get("config_canonical_sha256") != core.BRIDGE_CONFIG_CANONICAL_SHA256
        or launch.get("formal_success") is not False
    ):
        raise ValueError(f"Bridge {mode} controller launch contract drifted.")
    checks = terminal.get("execution_checks")
    if (
        terminal.get("schema") != controller.TERMINAL_SCHEMA
        or terminal.get("status") != "technical_completed"
        or terminal.get("mode") != mode
        or terminal.get("git_commit") != expected_commit
        or terminal.get("target_index") != core.BRIDGE_TARGET_INDEX
        or terminal.get("target_segment_id") != core.BRIDGE_TARGET_SEGMENT_ID
        or terminal.get("child_exit_code") != 0
        or terminal.get("technical_gate") is not True
        or terminal.get("formal_success") is not False
        or terminal.get("phase2_allowed") is not False
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError(f"Bridge {mode} controller terminal contract drifted.")
    for field in ("config_sha256", "trainer_sha256", "controller_sha256", "core_sha256"):
        if terminal.get(field) != launch.get(field):
            raise ValueError(f"Bridge {mode} controller {field} binding drifted.")
    for field, relative in (
        ("stdout_sha256", "stdout.log"),
        ("stderr_sha256", "stderr.log"),
        ("summary_sha256", f"run/{controller.SUMMARY_FILE}"),
        ("manifest_sha256", "run/manifest.json"),
        ("child_terminal_sha256", "run/terminal.json"),
        ("child_inventory_sha256", "run/artifact_inventory.json"),
    ):
        artifact = root / relative
        if not artifact.is_file() or terminal.get(field) != _sha256(artifact):
            raise ValueError(f"Bridge {mode} controller terminal {field} drifted.")
    command = launch.get("command")
    if not isinstance(command, list) or "--strict-determinism" not in command:
        raise ValueError(f"Bridge {mode} controller command contract drifted.")
    forbidden = {
        "--learning-rate",
        "--lr",
        "--optimizer",
        "--optimizer-steps",
        "--steps",
        "--gradient-clipping",
        "--gradient-clip",
        "--checkpoint-steps",
        "--target-index",
    }
    if any(value in forbidden for value in command):
        raise ValueError(f"Bridge {mode} controller exposed a forbidden knob.")
    trainer_inventory = _validate_inventory(root / "run", schema=TRAINER_INVENTORY_SCHEMA)
    return {
        "root": str(root.resolve()),
        "root_inventory": inventory,
        "trainer_inventory": trainer_inventory,
        "launch": launch,
        "terminal": terminal,
    }


def _load_teacher(formal_run: Path, preflight_run: Path) -> tuple[Tensor, dict[str, Any]]:
    formal_path = formal_run / "teacher" / "canonical_r11_target01.pt"
    preflight_path = preflight_run / "teacher" / "canonical_r11_target01.pt"
    for path in (formal_path, preflight_path):
        if not path.is_file() or _sha256(path) != core.BRIDGE_TEACHER_FILE_SHA256:
            raise ValueError(f"Bridge aggregation canonical teacher file drifted: {path}")
    payload = torch.load(formal_path, map_location="cpu", weights_only=False)
    teacher = payload.get("latent_fp32")
    if (
        payload.get("schema") != "vision_memory.r11-vae-latent-endpoint.v1"
        or not isinstance(teacher, Tensor)
        or teacher.dtype != torch.float32
        or tuple(teacher.shape) != (1, 4, 128, 128)
        or not torch.isfinite(teacher).all()
        or canonical_tensor_sha256(teacher) != core.BRIDGE_TEACHER_TENSOR_SHA256
        or not _equal_float(
            teacher.std(unbiased=False),
            core.BRIDGE_TEACHER_STD,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise ValueError("Bridge aggregation canonical teacher tensor contract drifted.")
    return teacher, {
        "formal_path": str(formal_path.resolve()),
        "preflight_path": str(preflight_path.resolve()),
        "file_sha256": core.BRIDGE_TEACHER_FILE_SHA256,
        "tensor_sha256": core.BRIDGE_TEACHER_TENSOR_SHA256,
        "shape": list(teacher.shape),
        "dtype": str(teacher.dtype),
        "finite": True,
        "population_std": float(teacher.std(unbiased=False)),
    }


def _validate_manifest(
    run: Path,
    *,
    mode: str,
    expected_commit: str,
    config: Mapping[str, Any],
    parent_target: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = run / "manifest.json"
    manifest = _load(manifest_path)
    parity = manifest.get("parity_checks")
    fixed = manifest.get("fixed_contract")
    information = manifest.get("information_boundary")
    if (
        manifest.get("schema") != TRAINER_MANIFEST_SCHEMA
        or manifest.get("protocol") != core.BRIDGE_PROTOCOL
        or manifest.get("mode") != mode
        or manifest.get("git_commit") != expected_commit
        or manifest.get("git_dirty") is not False
        or manifest.get("preregistered_config_file_sha256") != core.BRIDGE_CONFIG_FILE_SHA256
        or manifest.get("preregistered_config_canonical_sha256") != core.BRIDGE_CONFIG_CANONICAL_SHA256
        or manifest.get("target_index") != core.BRIDGE_TARGET_INDEX
        or manifest.get("target_segment_id") != core.BRIDGE_TARGET_SEGMENT_ID
        or manifest.get("target_segment") != parent_target.get("target_segment")
        or manifest.get("bridge_diagnostic_only") is not True
        or manifest.get("phase2_allowed") is not False
        or manifest.get("formal_success_gate") is not False
        or not isinstance(parity, Mapping)
        or parity.get("passed") is not True
        or parity.get("observed") != parity.get("expected")
        or not isinstance(fixed, Mapping)
        or not isinstance(information, Mapping)
        or information.get("passed") is not True
        or information.get("canonical_teacher_used_only_by_dense_loss") is not True
        or information.get("reader_used_only_for_fixed_audit") is not True
        or information.get("reader_gradient_calls_during_optimization") != 0
    ):
        raise ValueError(f"Bridge aggregation {mode} manifest contract drifted.")
    expected_fixed = config["unchanged_contract"]
    fixed_checks = {
        "only_trainable": fixed.get("only_trainable") == "x_T_fp32",
        "steps": fixed.get("optimizer_steps") == (core.BRIDGE_OPTIMIZER_STEPS if mode == "formal" else 0),
        "sigmas": _sigmas_exact(fixed.get("effective_sigmas")),
        "optimizer": fixed.get("optimizer") == "Adam",
        "learning_rate": _equal_float(
            fixed.get("learning_rate"),
            expected_fixed["learning_rate"],
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "weight_decay": _equal_float(
            fixed.get("weight_decay"),
            expected_fixed["weight_decay"],
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "gradient_clipping": fixed.get("gradient_clipping") is None,
        "checkpoints": fixed.get("checkpoint_steps")
        == (list(core.BRIDGE_CHECKPOINT_STEPS) if mode == "formal" else [0]),
        "primary_endpoint": fixed.get("primary_endpoint") == core.BRIDGE_PRIMARY_ENDPOINT,
        "no_best": fixed.get("best_checkpoint_selection_forbidden") is True,
        "reader_gradient_calls": fixed.get("reader_gradient_calls_during_optimization") == 0,
    }
    if not all(fixed_checks.values()):
        raise ValueError(f"Bridge aggregation {mode} fixed contract drifted: {fixed_checks}")
    parent = manifest.get("parent_phase1a")
    expected_parent = config["parent_phase1a"]
    if (
        not isinstance(parent, Mapping)
        or parent.get("comparison_sha256") != expected_parent["comparison_sha256"]
        or parent.get("raw_artifacts_sha256") != expected_parent["raw_artifacts_sha256"]
        or parent.get("target_inventory_sha256") != parent_target.get("inventory", {}).get("sha256")
        or Path(str(parent.get("target_root", ""))).resolve() != Path(str(parent_target.get("root", ""))).resolve()
    ):
        raise ValueError("Bridge aggregation parent Phase1A binding drifted.")
    teacher = manifest.get("teacher")
    if (
        not isinstance(teacher, Mapping)
        or teacher.get("file_sha256") != core.BRIDGE_TEACHER_FILE_SHA256
        or teacher.get("tensor_sha256") != core.BRIDGE_TEACHER_TENSOR_SHA256
        or teacher.get("copied_sha256") != core.BRIDGE_TEACHER_FILE_SHA256
    ):
        raise ValueError("Bridge aggregation manifest teacher binding drifted.")
    condition_path = run / "condition" / "official_full_condition.pt"
    condition = manifest.get("condition_artifact")
    if (
        not condition_path.is_file()
        or not isinstance(condition, Mapping)
        or condition.get("sha256") != _sha256(condition_path)
    ):
        raise ValueError("Bridge aggregation condition artifact binding drifted.")
    start = _load(run / "model_snapshot_verification_start.json")
    end = _load(run / "model_snapshot_verification_end.json")
    if (
        end.get("passed") is not True
        or start.get("bindings") != end.get("bindings")
        or start.get("bindings") != manifest.get("model_snapshot_payloads_start")
    ):
        raise ValueError("Bridge aggregation model snapshot start/end binding drifted.")
    return {
        "manifest": manifest,
        "manifest_sha256": _sha256(manifest_path),
        "condition_sha256": _sha256(condition_path),
        "fixed_checks": fixed_checks,
    }


def _validate_metrics(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_jsonl(path)
    expected_steps = list(range(1, core.BRIDGE_OPTIMIZER_STEPS + 1))
    finite_fields = (
        "loss_before_step",
        "mse",
        "rmse",
        "l2_distance",
        "m0_mse",
        "m0_rmse",
        "mse_ratio_to_m0",
        "l2_distance_ratio_to_m0",
        "relative_mse_change",
        "teacher_population_std",
        "teacher_normalized_rmse",
        "gradient_norm",
        "gradient_nonzero_fraction",
        "x_T_update_norm",
        "elapsed_seconds",
    )
    checks = {
        "count": len(rows) == core.BRIDGE_OPTIMIZER_STEPS,
        "steps": [row.get("optimizer_step") for row in rows] == expected_steps,
        "schemas": all(row.get("schema") == TRAINER_METRICS_SCHEMA for row in rows),
        "targets": all(
            row.get("target_index") == core.BRIDGE_TARGET_INDEX
            and row.get("target_segment_id") == core.BRIDGE_TARGET_SEGMENT_ID
            for row in rows
        ),
        "objectives": all(row.get("objective") == "canonical_r11_latent_fp32_mean_mse" for row in rows),
        "finite": all(
            field in row and _equal_float(row[field], row[field], rel_tol=0.0, abs_tol=0.0)
            for row in rows
            for field in finite_fields
        ),
        "gradient": all(
            float(row.get("gradient_norm", 0.0)) > 0.0 and float(row.get("gradient_nonzero_fraction", 0.0)) > 0.0
            for row in rows
        ),
        "updates": all(float(row.get("x_T_update_norm", -1.0)) >= 0.0 for row in rows),
        "path": all(
            row.get("full_dreamlite_forward_executed") is True
            and row.get("dreamlite_denoising_steps") == 4
            and row.get("trajectory_points") == 5
            and _sigmas_exact(row.get("effective_sigmas"))
            for row in rows
        ),
        "optimizer": all(
            _equal_float(row.get("learning_rate"), 0.05, rel_tol=0.0, abs_tol=0.0)
            and _equal_float(row.get("weight_decay"), 0.0, rel_tol=0.0, abs_tol=0.0)
            for row in rows
        ),
        "no_clip": all(row.get("gradient_clipping_applied") is False for row in rows),
        "no_reader_gradient": all(row.get("reader_gradient_calls") == 0 for row in rows),
        "teacher": all(row.get("teacher_tensor_sha256") == core.BRIDGE_TEACHER_TENSOR_SHA256 for row in rows),
    }
    if not all(checks.values()):
        raise ValueError(f"Bridge raw optimizer receipt contract failed: {checks}")
    m0_values = {float(row["m0_mse"]) for row in rows}
    if len(m0_values) != 1:
        raise ValueError("Bridge raw optimizer receipts disagree on M0 MSE.")
    for row in rows:
        recomputed = core.bridge_distance_statistics(
            mse=float(row["mse"]),
            m0_mse=float(row["m0_mse"]),
            tensor_numel=int(row["tensor_numel"]),
        )
        if not all(_equal_float(row.get(key), value) for key, value in recomputed.items()):
            raise ValueError(f"Bridge receipt distance arithmetic drifted at step {row['optimizer_step']}.")
        if not _equal_float(row["loss_before_step"], row["mse"], rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"Bridge receipt loss/MSE mismatch at step {row['optimizer_step']}.")
    return rows, {"checks": checks, "m0_mse": next(iter(m0_values)), "sha256": _sha256(path)}


def _validate_checkpoint(
    run: Path,
    *,
    step: int,
    teacher: Tensor,
    m0_mse: float,
    manifest_sha256: str,
    condition_sha256: str,
) -> dict[str, Any]:
    checkpoint_path = run / "checkpoints" / f"step-{step:03d}.pt"
    image_path = run / "images" / f"step-{step:03d}.png"
    record_path = run / "checkpoint_hashes" / f"step-{step:03d}.json"
    record = _load(record_path)
    if (
        record.get("schema") != TRAINER_CHECKPOINT_HASH_SCHEMA
        or record.get("optimizer_step") != step
        or not checkpoint_path.is_file()
        or not image_path.is_file()
        or record.get("checkpoint_bytes") != checkpoint_path.stat().st_size
        or record.get("checkpoint_sha256") != _sha256(checkpoint_path)
        or record.get("png_bytes") != image_path.stat().st_size
        or record.get("png_sha256") != _sha256(image_path)
        or record.get("trajectory_points") != 5
        or not _sigmas_exact(record.get("effective_sigmas"))
    ):
        raise ValueError(f"Bridge checkpoint/hash record contract drifted at step {step}.")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    x_t = payload.get("x_T_fp32")
    z_t = payload.get("z_t_fp32")
    trajectory = payload.get("trajectory_fp32")
    hashes = payload.get("tensor_sha256")
    if (
        payload.get("schema") != TRAINER_CHECKPOINT_SCHEMA
        or payload.get("optimizer_step") != step
        or payload.get("objective") != "canonical_r11_latent_fp32_mean_mse"
        or not isinstance(x_t, Tensor)
        or not isinstance(z_t, Tensor)
        or x_t.dtype != torch.float32
        or z_t.dtype != torch.float32
        or tuple(x_t.shape) != (1, 4, 128, 128)
        or tuple(z_t.shape) != tuple(teacher.shape)
        or not torch.isfinite(x_t).all()
        or not torch.isfinite(z_t).all()
        or not isinstance(trajectory, (list, tuple))
        or len(trajectory) != 5
        or not all(
            isinstance(value, Tensor)
            and value.dtype == torch.float32
            and tuple(value.shape) == tuple(teacher.shape)
            and torch.isfinite(value).all()
            for value in trajectory
        )
        or not _sigmas_exact(payload.get("effective_sigmas"))
        or payload.get("effective_sigmas") != record.get("effective_sigmas")
        or payload.get("teacher_tensor_sha256") != core.BRIDGE_TEACHER_TENSOR_SHA256
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("condition_artifact_sha256") != condition_sha256
        or not isinstance(payload.get("optimizer"), Mapping)
        or not isinstance(hashes, Mapping)
    ):
        raise ValueError(f"Bridge checkpoint tensor payload contract drifted at step {step}.")
    observed_hashes = {
        "x_T_fp32": canonical_tensor_sha256(x_t),
        "z_t_fp32": canonical_tensor_sha256(z_t),
        "trajectory_fp32": [canonical_tensor_sha256(value) for value in trajectory],
    }
    if hashes != observed_hashes or record.get("tensor_sha256") != observed_hashes:
        raise ValueError(f"Bridge checkpoint tensor hashes drifted at step {step}.")
    mse = float((z_t - teacher).square().mean())
    distance = core.bridge_distance_statistics(
        mse=mse,
        m0_mse=m0_mse,
        tensor_numel=teacher.numel(),
    )
    payload_distance = payload.get("distance_statistics")
    record_distance = record.get("distance_statistics")
    if (
        not isinstance(payload_distance, Mapping)
        or not isinstance(record_distance, Mapping)
        or set(payload_distance) != set(distance)
        or set(record_distance) != set(distance)
        or not all(_equal_float(payload_distance[key], value) for key, value in distance.items())
        or not all(_equal_float(record_distance[key], value) for key, value in distance.items())
    ):
        raise ValueError(f"Bridge checkpoint distance arithmetic drifted at step {step}.")
    if step == 0 and (
        observed_hashes["x_T_fp32"] != core.BRIDGE_INITIAL_X_T_SHA256
        or observed_hashes["z_t_fp32"] != core.BRIDGE_INITIAL_Z_T_SHA256
    ):
        raise ValueError("Bridge checkpoint step-0 parity hashes drifted.")
    return {
        "optimizer_step": step,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "image_path": str(image_path.resolve()),
        "image_sha256": _sha256(image_path),
        "record_path": str(record_path.resolve()),
        "record_sha256": _sha256(record_path),
        "tensor_sha256": observed_hashes,
        "distance_statistics": distance,
    }


def _reader_row_matches_target(row: Mapping[str, Any], target_segment: Mapping[str, Any]) -> bool:
    query = target_segment["query"]
    choices = query["choices"]
    answer_index = query["target_index"]
    predicted_index = row.get("predicted_index")
    expected_query_id = f"{target_segment['query_source_episode_id']}:{target_segment['query_turn_index']}"
    events = target_segment["events"]
    return bool(
        row.get("item_id") == target_segment["segment_id"]
        and row.get("pair_unit") == target_segment["segment_id"]
        and row.get("episode_id") == target_segment["query_source_episode_id"]
        and row.get("query_id") == expected_query_id
        and row.get("query_gap") == target_segment["query_gap"]
        and row.get("updater_count") == len(events)
        and row.get("target_event_kind") == events[-1]["event_kind"]
        and row.get("target_index") == answer_index
        and row.get("target_text") == choices[answer_index]
        and isinstance(predicted_index, int)
        and not isinstance(predicted_index, bool)
        and predicted_index in range(4)
        and row.get("predicted_text") == choices[predicted_index]
    )


def _validate_rows(path: Path, *, target_segment: Mapping[str, Any]) -> dict[str, Any]:
    rows = _load_jsonl(path)
    expected_groups = {
        ("canonical_teacher", "teacher"),
        ("m0", "normal"),
        ("m0", "reset"),
        (core.BRIDGE_PRIMARY_ENDPOINT, "normal"),
        (core.BRIDGE_PRIMARY_ENDPOINT, "reset"),
    }
    groups = {(row.get("checkpoint"), row.get("condition")) for row in rows}
    common = all(
        row.get("schema") == "vision_memory.r5-compose-causal-evaluation.v1"
        and row.get("suite") == "r11_new_target01_canonical_latent_bridge"
        and row.get("item_id") == core.BRIDGE_TARGET_SEGMENT_ID
        and row.get("pair_unit") == core.BRIDGE_TARGET_SEGMENT_ID
        and row.get("donor_item_id") is None
        and isinstance(row.get("target_index"), int)
        and not isinstance(row.get("target_index"), bool)
        and row.get("target_index") in range(4)
        and _reader_row_matches_target(row, target_segment)
        for row in rows
    )
    if len(rows) != 20 or groups != expected_groups or not common:
        raise ValueError("Bridge endpoint Reader raw-row group contract drifted.")
    statistics: dict[str, Any] = {}
    for checkpoint, condition in sorted(expected_groups):
        stats = core.reader_checkpoint_statistics(
            rows,
            checkpoint=checkpoint,
            condition=condition,
        )
        statistics[f"{checkpoint}/{condition}"] = stats
        selected = [row for row in rows if row.get("checkpoint") == checkpoint and row.get("condition") == condition]
        for row in selected:
            permutation = tuple(row["permutation"])
            logits = [float(value) for value in row["choice_logits_ordered"]]
            predicted_original = permutation[max(range(4), key=logits.__getitem__)]
            ordered_target = permutation.index(int(row["target_index"]))
            target_logit = logits[ordered_target]
            margin = target_logit - max(value for index, value in enumerate(logits) if index != ordered_target)
            if row.get("predicted_index") != predicted_original or not _equal_float(
                row.get("margin"),
                margin,
                rel_tol=1e-5,
                abs_tol=1e-5,
            ):
                raise ValueError("Bridge endpoint Reader raw-row prediction/margin drifted.")
    return {
        "rows": rows,
        "statistics": statistics,
        "sha256": _sha256(path),
    }


def _validate_preflight(
    run: Path,
    *,
    teacher: Tensor,
    manifest_sha256: str,
    condition_sha256: str,
    target_segment: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _load(run / controller.SUMMARY_FILE)
    report = _load(run / controller.PREFLIGHT_FILE)
    terminal = _load(run / "terminal.json")
    audit = summary.get("audit")
    required_true = (
        "four_dreamlite_steps",
        "effective_sigmas_exact",
        "finite_nonzero_x_T_gradient",
        "only_x_T_fp32_trainable",
        "frozen_gradients_absent",
        "step0_parity_valid",
        "teacher_binding_valid",
        "teacher_replay_gate",
        "checkpoint_hash_valid",
        "snapshots_unchanged",
    )
    if (
        summary != report
        or summary.get("schema") != TRAINER_PREFLIGHT_SCHEMA
        or summary.get("mode") != "technical-preflight"
        or summary.get("passed") is not True
        or summary.get("bridge_result_evaluated") is not False
        or summary.get("formal_success_gate") is not False
        or summary.get("phase2_allowed") is not False
        or not isinstance(audit, Mapping)
        or audit.get("optimizer_steps") != 0
        or audit.get("full_forward_calls") != 1
        or audit.get("backward_calls") != 1
        or not all(audit.get(name) is True for name in required_true)
        or terminal.get("schema") != TRAINER_TERMINAL_SCHEMA
        or terminal.get("technical_gate") is not True
        or terminal.get("bridge_diagnostic_gate") is not None
        or (run / controller.METRICS_FILE).exists()
    ):
        raise ValueError("Bridge technical-preflight raw contract drifted.")
    rows = _load_jsonl(run / controller.ROWS_FILE)
    if not all(_reader_row_matches_target(row, target_segment) for row in rows):
        raise ValueError("Bridge preflight Reader rows are not bound to the locked target label.")
    teacher_stats = core.reader_checkpoint_statistics(
        rows,
        checkpoint="canonical_teacher",
        condition="teacher",
    )
    if (
        len(rows) != 4
        or not core.teacher_replay_gate(teacher_stats)
        or summary.get("teacher_replay_statistics") != teacher_stats
    ):
        raise ValueError("Bridge technical-preflight teacher replay gate failed independently.")
    checkpoint = _validate_checkpoint(
        run,
        step=0,
        teacher=teacher,
        m0_mse=float(summary["initial_distance_statistics"]["m0_mse"]),
        manifest_sha256=manifest_sha256,
        condition_sha256=condition_sha256,
    )
    return {
        "summary_sha256": _sha256(run / controller.SUMMARY_FILE),
        "rows_sha256": _sha256(run / controller.ROWS_FILE),
        "teacher_statistics": teacher_stats,
        "checkpoint": checkpoint,
        "passed": True,
    }


def _write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "optimizer_step",
        "mse",
        "rmse",
        "l2_distance",
        "mse_ratio_to_m0",
        "l2_distance_ratio_to_m0",
        "teacher_normalized_rmse",
        "gradient_norm",
        "gradient_nonzero_fraction",
        "x_T_update_norm",
        "elapsed_seconds",
    )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def _write_report(path: Path, comparison: Mapping[str, Any]) -> None:
    distance = comparison["endpoint_distance_statistics"]
    reader = comparison["endpoint_reader_statistics"]
    lines = [
        "# R11_new canonical-latent bridge 独立复算报告",
        "",
        "## 结论",
        "",
        f"- 工程技术门：`{str(comparison['engineering_gate']).lower()}`",
        f"- Teacher replay：`{str(comparison['teacher_replay_gate']).lower()}`",
        f"- 距离门：`{str(comparison['bridge_distance_gate']).lower()}`",
        f"- Endpoint Reader transfer：`{str(comparison['endpoint_reader_transfer_gate']).lower()}`",
        f"- Bridge diagnostic：`{str(comparison['bridge_diagnostic_gate']).lower()}`",
        "- Picture Memory 正式科学成功：`false`",
        "- Phase 2：仍阻塞。",
        "",
        "## Raw step 256",
        "",
        f"- MSE：`{distance['mse']:.12g}`",
        f"- MSE/M0：`{distance['mse_ratio_to_m0']:.12g}`（门槛 <= 0.01）",
        f"- L2/M0：`{distance['l2_distance_ratio_to_m0']:.12g}`（门槛 <= 0.10）",
        f"- teacher-normalized RMSE：`{distance['teacher_normalized_rmse']:.12g}`（门槛 <= 0.10）",
        f"- Reader accuracy：`{reader['accuracy']:.6g}`（门槛 = 1.0）",
        f"- 决策：`{comparison['decision']}`",
        "",
        "## 证据边界",
        "",
        "本报告从原始 tensor、receipt、checkpoint 和 logits 独立复算，不采信 trainer 的 PASS 字段。",
        "该实验只区分已知可读 latent 在锁定 DreamLite solver/预算下的经验可达性；",
        "无论结果如何，都不证明 shared writer、ID/OOD、递归状态更新或 Picture Memory 正式成功。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _source_file(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"Bridge aggregation source artifact is missing: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def compare(
    *,
    preflight_root: Path,
    formal_root: Path,
    output_dir: Path,
    expected_commit: str,
) -> dict[str, Any]:
    if not _COMMIT_RE.fullmatch(expected_commit):
        raise ValueError("Bridge aggregation expected commit must be a full lowercase SHA-1.")
    if not preflight_root.is_dir() or not formal_root.is_dir():
        raise ValueError("Bridge aggregation requires existing preflight and formal controller roots.")
    if preflight_root.resolve() == formal_root.resolve():
        raise ValueError("Bridge preflight and formal roots must be distinct.")
    if output_dir.exists():
        raise ValueError("Bridge aggregation requires a nonexistent fresh output directory.")
    config = _validate_config()
    parent_target = _validate_parent_target(config)
    preflight_controller = _validate_controller_root(
        preflight_root,
        mode="technical-preflight",
        expected_commit=expected_commit,
    )
    formal_controller = _validate_controller_root(
        formal_root,
        mode="formal",
        expected_commit=expected_commit,
    )
    for field in ("trainer_sha256", "controller_sha256", "core_sha256", "config_sha256"):
        if preflight_controller["launch"].get(field) != formal_controller["launch"].get(field):
            raise ValueError(f"Bridge preflight/formal code binding differs: {field}.")
    prerequisite = formal_controller["launch"].get("preflight_prerequisite")
    if (
        not isinstance(prerequisite, Mapping)
        or prerequisite.get("passed") is not True
        or prerequisite.get("terminal_sha256") != _sha256(preflight_root / "terminal.json")
        or prerequisite.get("inventory_sha256") != _sha256(preflight_root / "artifact_inventory.json")
        or prerequisite.get("summary_sha256") != _sha256(preflight_root / "run" / controller.SUMMARY_FILE)
    ):
        raise ValueError("Bridge formal launch is not bound to this exact preflight.")
    preflight_run = preflight_root / "run"
    formal_run = formal_root / "run"
    teacher, teacher_record = _load_teacher(formal_run, preflight_run)
    preflight_manifest = _validate_manifest(
        preflight_run,
        mode="technical-preflight",
        expected_commit=expected_commit,
        config=config,
        parent_target=parent_target,
    )
    formal_manifest = _validate_manifest(
        formal_run,
        mode="formal",
        expected_commit=expected_commit,
        config=config,
        parent_target=parent_target,
    )
    preflight = _validate_preflight(
        preflight_run,
        teacher=teacher,
        manifest_sha256=preflight_manifest["manifest_sha256"],
        condition_sha256=preflight_manifest["condition_sha256"],
        target_segment=parent_target["target_segment"],
    )
    metrics, metrics_record = _validate_metrics(formal_run / controller.METRICS_FILE)
    checkpoints = [
        _validate_checkpoint(
            formal_run,
            step=step,
            teacher=teacher,
            m0_mse=float(metrics_record["m0_mse"]),
            manifest_sha256=formal_manifest["manifest_sha256"],
            condition_sha256=formal_manifest["condition_sha256"],
        )
        for step in core.BRIDGE_CHECKPOINT_STEPS
    ]
    checkpoint_by_step = {row["optimizer_step"]: row for row in checkpoints}
    step0_distance = checkpoint_by_step[0]["distance_statistics"]
    if not _equal_float(step0_distance["mse"], metrics_record["m0_mse"]) or not _equal_float(
        step0_distance["mse_ratio_to_m0"], 1.0
    ):
        raise ValueError("Bridge formal step-0 checkpoint disagrees with raw receipt M0.")
    endpoint_checkpoint = formal_run / "checkpoints" / "step-256.pt"
    endpoint_image = formal_run / "images" / "step-256.png"
    if _sha256(formal_run / "endpoint_raw.pt") != _sha256(endpoint_checkpoint) or _sha256(
        formal_run / "endpoint_raw.png"
    ) != _sha256(endpoint_image):
        raise ValueError("Bridge raw endpoint is not byte-identical to checkpoint step 256.")
    row_record = _validate_rows(
        formal_run / controller.ROWS_FILE,
        target_segment=parent_target["target_segment"],
    )
    teacher_stats = row_record["statistics"]["canonical_teacher/teacher"]
    endpoint_stats = row_record["statistics"][f"{core.BRIDGE_PRIMARY_ENDPOINT}/normal"]
    if teacher_stats != preflight["teacher_statistics"]:
        raise ValueError("Bridge formal and preflight teacher replay statistics differ.")
    engineering_gate = True
    teacher_gate = bool(
        core.teacher_replay_gate(preflight["teacher_statistics"]) and core.teacher_replay_gate(teacher_stats)
    )
    endpoint_distance = checkpoint_by_step[256]["distance_statistics"]
    distance_gate = core.bridge_distance_gate(
        endpoint_distance,
        technical_gate=engineering_gate and teacher_gate,
    )
    reader_gate = core.endpoint_reader_transfer_gate(endpoint_stats)
    bridge_gate = bool(engineering_gate and teacher_gate and distance_gate and reader_gate)
    decision = core.bridge_decision(
        distance_pass=distance_gate,
        reader_transfer_pass=reader_gate,
    )
    technical = _load(formal_run / "technical_gate.json")
    summary = _load(formal_run / controller.SUMMARY_FILE)
    expected_gates = {
        "technical_gate": True,
        "teacher_replay_gate": teacher_gate,
        "bridge_distance_gate": distance_gate,
        "endpoint_reader_transfer_gate": reader_gate,
        "bridge_diagnostic_gate": bridge_gate,
        "formal_success_gate": False,
    }
    if (
        technical.get("schema") != TRAINER_TECHNICAL_SCHEMA
        or technical.get("passed") is not True
        or core.bridge_technical_gate(technical) is not True
        or technical.get("optimizer_step_records") != core.BRIDGE_OPTIMIZER_STEPS
        or technical.get("checkpoint_steps_observed") != list(core.BRIDGE_CHECKPOINT_STEPS)
        or technical.get("trainable_parameter_names") != ["x_T_fp32"]
        or not _equal_float(technical.get("minimum_gradient_norm"), technical.get("minimum_gradient_norm"))
        or float(technical.get("minimum_gradient_norm", 0.0)) <= 0.0
        or not _equal_float(
            technical.get("minimum_gradient_nonzero_fraction"),
            technical.get("minimum_gradient_nonzero_fraction"),
        )
        or float(technical.get("minimum_gradient_nonzero_fraction", 0.0)) <= 0.0
        or summary.get("schema") != TRAINER_SUMMARY_SCHEMA
        or summary.get("status") != "completed"
        or summary.get("mode") != "formal"
        or summary.get("optimizer_steps") != core.BRIDGE_OPTIMIZER_STEPS
        or summary.get("primary_endpoint") != core.BRIDGE_PRIMARY_ENDPOINT
        or summary.get("technical_gate") != technical
        or summary.get("gates") != expected_gates
        or summary.get("decision") != decision
        or summary.get("formal_success_gate") is not False
        or summary.get("full_success_claim_allowed") is not False
        or summary.get("phase2_allowed") is not False
    ):
        raise ValueError("Bridge trainer summary disagrees with independent recomputation.")
    declared_distance = summary.get("endpoint_distance_statistics")
    declared_reader = summary.get("endpoint_reader_statistics")
    if (
        not isinstance(declared_distance, Mapping)
        or set(declared_distance) != set(endpoint_distance)
        or not all(_equal_float(declared_distance[key], value) for key, value in endpoint_distance.items())
        or declared_reader != endpoint_stats
    ):
        raise ValueError("Bridge trainer endpoint statistics disagree with raw artifacts.")
    controller_terminal = formal_controller["terminal"]
    diagnostic = controller_terminal.get("diagnostic_result")
    if (
        controller_terminal.get("bridge_diagnostic_gate") is not bridge_gate
        or not isinstance(diagnostic, Mapping)
        or diagnostic.get("evaluated") is not True
        or diagnostic.get("bridge_diagnostic_gate") is not bridge_gate
        or diagnostic.get("distance_gate") is not distance_gate
        or diagnostic.get("reader_transfer_gate") is not reader_gate
        or diagnostic.get("decision") != decision
    ):
        raise ValueError("Bridge controller diagnostic disagrees with independent recomputation.")
    output_dir.mkdir(parents=True)
    raw = {
        "schema": RAW_SCHEMA,
        "expected_commit": expected_commit,
        "preflight_controller": {
            "root": preflight_controller["root"],
            "root_inventory": preflight_controller["root_inventory"],
            "trainer_inventory": preflight_controller["trainer_inventory"],
        },
        "formal_controller": {
            "root": formal_controller["root"],
            "root_inventory": formal_controller["root_inventory"],
            "trainer_inventory": formal_controller["trainer_inventory"],
        },
        "teacher": teacher_record,
        "locked_parent_target": parent_target,
        "preflight": preflight,
        "metrics": metrics_record,
        "checkpoints": checkpoints,
        "evaluation_rows": {
            "path": str((formal_run / controller.ROWS_FILE).resolve()),
            "sha256": row_record["sha256"],
            "row_count": len(row_record["rows"]),
        },
        "key_source_files": {
            "formal_summary": _source_file(formal_run, controller.SUMMARY_FILE),
            "formal_technical_gate": _source_file(formal_run, "technical_gate.json"),
            "formal_endpoint_raw": _source_file(formal_run, "endpoint_raw.pt"),
            "formal_environment": _source_file(formal_run, "environment.txt"),
            "formal_runtime": _source_file(formal_run, "runtime.json"),
            "formal_stdout": _source_file(formal_root, "stdout.log"),
            "formal_stderr": _source_file(formal_root, "stderr.log"),
        },
    }
    _write_json(output_dir / "RAW_ARTIFACTS.json", raw)
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "status": "completed",
        "protocol": core.BRIDGE_PROTOCOL,
        "git_commit": expected_commit,
        "target_index": core.BRIDGE_TARGET_INDEX,
        "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID,
        "primary_endpoint": core.BRIDGE_PRIMARY_ENDPOINT,
        "optimizer_steps": core.BRIDGE_OPTIMIZER_STEPS,
        "engineering_gate": engineering_gate,
        "teacher_replay_gate": teacher_gate,
        "bridge_distance_gate": distance_gate,
        "endpoint_reader_transfer_gate": reader_gate,
        "bridge_diagnostic_gate": bridge_gate,
        "teacher_replay_statistics": teacher_stats,
        "m0_reader_statistics": row_record["statistics"]["m0/normal"],
        "m0_reset_statistics": row_record["statistics"]["m0/reset"],
        "endpoint_reader_statistics": endpoint_stats,
        "endpoint_reset_statistics": row_record["statistics"][f"{core.BRIDGE_PRIMARY_ENDPOINT}/reset"],
        "endpoint_distance_statistics": endpoint_distance,
        "checkpoint_distance_statistics": {
            str(row["optimizer_step"]): row["distance_statistics"] for row in checkpoints
        },
        "decision": decision,
        "phase2_allowed": False,
        "formal_success": False,
        "scientific_success_claim": False,
        "interpretation": "diagnostic_only",
        "raw_artifacts_sha256": _sha256(output_dir / "RAW_ARTIFACTS.json"),
    }
    _write_json(output_dir / "comparison.json", comparison)
    _write_metrics_csv(output_dir / "distance_trajectory.csv", metrics)
    _write_report(output_dir / "REPORT.md", comparison)
    _write_inventory(output_dir)
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare(
        preflight_root=args.preflight_root,
        formal_root=args.formal_root,
        output_dir=args.output_dir,
        expected_commit=args.expected_commit,
    )
    print(
        json.dumps(
            {
                "milestone": "r11_new_bridge_independent_aggregation_completed",
                "engineering_gate": result["engineering_gate"],
                "bridge_diagnostic_gate": result["bridge_diagnostic_gate"],
                "decision": result["decision"],
                "formal_success": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
