"""R11_new Target-01 source-only-initialization canonical-latent bridge diagnostic.

Only the FP32 initial latent x_T is trainable.  The complete DreamLite path is
frozen and the sole training objective is dense FP32 MSE to the immutable
canonical-R11 target-01 latent.  Qwen is used only for fixed teacher/endpoint
audits and never contributes a training gradient.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.inspire.model_snapshot_manifest import verify_snapshot_binding  # noqa: E402
from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r7_gradient_balance as r8  # noqa: E402
from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a  # noqa: E402
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval  # noqa: E402
from vision_memory.repro import canonical_object_sha256, canonical_tensor_sha256  # noqa: E402
from vision_memory.training import r11_new_source_init_bridge as core  # noqa: E402


CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "r11_new_canonical_latent_bridge_target01_source_only_init.json"
)
PROTOCOL = core.BRIDGE_PROTOCOL
IMPLEMENTATION_REVISION = "canonical-r11-latent-mse-source-only-init-target01-v1"
SUITE = "r11_new_target01_canonical_latent_bridge_source_only_init"
MANIFEST_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-manifest.v1"
METRICS_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-metrics.v1"
CHECKPOINT_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-checkpoint.v1"
CHECKPOINT_HASH_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-checkpoint-hashes.v1"
TECHNICAL_GATE_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-technical-gate.v1"
SUMMARY_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-summary.v1"
TERMINAL_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-artifact-inventory.v1"
OWNER_SCHEMA = "vision_memory.r11-new-canonical-latent-bridge-output-owner.v1"
INITIALIZATION_SCHEMA = "vision_memory.r11-new-source-only-initialization.v1"
INITIALIZATION_FORMULA = "x_T_init_fp32=source_latents_fp32.clone()"
INITIALIZATION_RECONSTRUCTION_ORDER = (
    "source_compute.mul(1-actual_sigma).add("
    "x_T_init_fp32.to(compute_dtype), alpha=actual_sigma)"
)
INITIALIZATION_TENSOR_KEYS = (
    "source_latents_fp32", "x_T_init_fp32", "reconstructed_start_state_compute",
)
INITIALIZATION_METADATA_KEYS = (
    "schema", "compute_device_type", "parameter_dtype", "compute_dtype", "shape",
    "formula", "reconstruction_operator_order", "nominal_effective_sigmas",
    "actual_effective_sigmas", "actual_effective_sigma0", "initial_x_t_equals_source",
    "initialization_teacher_assisted",
)


OPTIMIZER_STEPS = core.BRIDGE_OPTIMIZER_STEPS
CHECKPOINT_STEPS = core.BRIDGE_CHECKPOINT_STEPS
LEARNING_RATE = core.BRIDGE_BASE_LEARNING_RATE
WEIGHT_DECAY = phase1a.WEIGHT_DECAY


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.target_index = core.BRIDGE_TARGET_INDEX
    args.seed = phase1a.SEED
    args.adapter_seed = phase1a.SEED
    args.pairing_seed = 0
    args.split_seed = 20260730
    args.schedule_seed = phase1a.R10_SELECTION_SEED
    args.bootstrap_iterations = 10_000
    args.resolution = phase1a.RESOLUTION
    args.lora_rank = 0
    args.selected_step_count = 0
    args.gradient_mode = "full"
    return args


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _claim_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    owner = path / ".r11_new_bridge_output_owner.json"
    unexpected = [candidate for candidate in path.iterdir() if candidate != owner]
    if unexpected or owner.exists():
        raise ValueError("R11_new bridge requires a fresh output directory.")
    payload = {
        "schema": OWNER_SCHEMA,
        "created_at_utc": _utc_now(),
        "pid": os.getpid(),
    }
    with owner.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Malformed inventory path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe inventory path: {value!r}")
    return value


def _validate_inventory(root: Path, *, schema: str) -> str:
    path = root / "artifact_inventory.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != schema or not isinstance(value.get("artifacts"), list):
        raise ValueError(f"Invalid inventory schema at {root}.")
    names: set[str] = set()
    for row in value["artifacts"]:
        relative = _safe_relative(row.get("path"))
        if relative in names:
            raise ValueError(f"Duplicate inventory path: {relative}")
        names.add(relative)
        artifact = root.joinpath(*PurePosixPath(relative).parts)
        if (
            not artifact.is_file()
            or artifact.stat().st_size != row.get("bytes")
            or phase1a._sha256(artifact) != row.get("sha256")
        ):
            raise ValueError(f"Inventory hash/size mismatch: {artifact}")
    actual = {
        candidate.relative_to(root).as_posix()
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != path
    }
    if names != actual:
        raise ValueError("Inventory file set differs from the immutable tree.")
    return phase1a._sha256(path)


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file() or phase1a._sha256(CONFIG_PATH) != core.BRIDGE_CONFIG_FILE_SHA256:
        raise ValueError("R11_new bridge preregistered config file hash drifted.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return core.validate_bridge_config(value)


def _validate_parent_evidence(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    parent = config["parent_phase1a"]
    if phase1a._sha256(args.phase1a_comparison) != parent["comparison_sha256"]:
        raise ValueError("R11_new bridge parent comparison hash drifted.")
    if phase1a._sha256(args.phase1a_raw_artifacts) != parent["raw_artifacts_sha256"]:
        raise ValueError("R11_new bridge parent RAW_ARTIFACTS hash drifted.")
    comparison = json.loads(args.phase1a_comparison.read_text(encoding="utf-8"))
    expected = {
        "source_training_git_commit": parent["training_source_commit"],
        "engineering_gate": True,
        "target_pass_count": 6,
        "failed_target_indices": [1, 7],
        "decision": "run_canonical_r11_latent_bridge_distance_diagnostic",
        "formal_success": False,
    }
    if any(comparison.get(key) != value for key, value in expected.items()):
        raise ValueError("R11_new bridge parent aggregation semantics drifted.")
    targets = {int(row["target_index"]): row for row in comparison.get("targets", [])}
    target = targets.get(core.BRIDGE_TARGET_INDEX)
    if (
        not isinstance(target, Mapping)
        or target.get("target_segment_id") != core.BRIDGE_TARGET_SEGMENT_ID
        or target.get("technical_gate") is not True
        or target.get("target_reachability_gate") is not False
        or Path(str(target.get("source_root"))).resolve() != args.phase1a_target_root.resolve()
        or args.phase1a_target_root.resolve() != Path(core.BRIDGE_PHASE1A_SOURCE_ROOT).resolve()
    ):
        raise ValueError("R11_new bridge target-01 parent binding drifted.")
    inventory_sha = _validate_inventory(
        args.phase1a_target_root,
        schema="vision_memory.r11-new-phase1a-target-inventory.v1",
    )
    terminal = json.loads((args.phase1a_target_root / "terminal.json").read_text(encoding="utf-8"))
    if (
        terminal.get("schema") != "vision_memory.r11-new-phase1a-target-terminal.v1"
        or terminal.get("technical_completed") is not True
        or terminal.get("target_index") != core.BRIDGE_TARGET_INDEX
        or terminal.get("target_segment_id") != core.BRIDGE_TARGET_SEGMENT_ID
        or terminal.get("diagnostic_result", {}).get("phase1a_query_level_reachability_gate") is not False
    ):
        raise ValueError("R11_new bridge parent target terminal drifted.")
    parent_manifest_path = args.phase1a_target_root / "run" / "manifest.json"
    if phase1a._sha256(parent_manifest_path) != core.BRIDGE_PARENT_TARGET_MANIFEST_SHA256:
        raise ValueError("R11_new bridge locked parent target manifest hash drifted.")
    checkpoint = torch.load(
        args.phase1a_target_root / "run" / "checkpoints" / "step-000.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        # These are historical parent receipts, not the new initialization.
        canonical_tensor_sha256(checkpoint["x_T_fp32"].float())
        != "c970092e2afca24ededea1aec2892bd6bd54ba0dd2193522dab22af10ac1d991"
        or canonical_tensor_sha256(checkpoint["z_t_fp32"].float())
        != "11c7216fe2a70f0caa314d182b2c176b4f50c78f2d081f5aa0271184e5c8e659"
    ):
        raise ValueError("R11_new bridge parent step-0 checkpoint parity drifted.")
    return {
        "comparison_path": str(args.phase1a_comparison.resolve()),
        "comparison_sha256": parent["comparison_sha256"],
        "raw_artifacts_path": str(args.phase1a_raw_artifacts.resolve()),
        "raw_artifacts_sha256": parent["raw_artifacts_sha256"],
        "target_root": str(args.phase1a_target_root.resolve()),
        "target_inventory_sha256": inventory_sha,
        "target_terminal_sha256": phase1a._sha256(args.phase1a_target_root / "terminal.json"),
        "target_manifest_sha256": core.BRIDGE_PARENT_TARGET_MANIFEST_SHA256,
        "step0_checkpoint_sha256": phase1a._sha256(args.phase1a_target_root / "run" / "checkpoints" / "step-000.pt"),
    }


def _load_teacher(path: Path) -> tuple[Tensor, dict[str, Any]]:
    if phase1a._sha256(path) != core.BRIDGE_TEACHER_FILE_SHA256:
        raise ValueError("R11_new bridge teacher file hash drifted.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    teacher = payload.get("latent_fp32")
    if (
        payload.get("schema") != "vision_memory.r11-vae-latent-endpoint.v1"
        or not isinstance(teacher, Tensor)
        or teacher.dtype != torch.float32
        or tuple(teacher.shape) != (1, 4, 128, 128)
        or not torch.isfinite(teacher).all()
        or canonical_tensor_sha256(teacher) != core.BRIDGE_TEACHER_TENSOR_SHA256
    ):
        raise ValueError("R11_new bridge teacher tensor contract drifted.")
    population_std = float(teacher.std(unbiased=False))
    if not math.isclose(population_std, core.BRIDGE_TEACHER_STD, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("R11_new bridge teacher population std drifted.")
    return teacher, {
        "source_path": str(path.resolve()),
        "file_sha256": core.BRIDGE_TEACHER_FILE_SHA256,
        "schema": payload["schema"],
        "tensor_key": "latent_fp32",
        "tensor_sha256": core.BRIDGE_TEACHER_TENSOR_SHA256,
        "shape": list(teacher.shape),
        "dtype": str(teacher.dtype),
        "finite": True,
        "mean": float(teacher.mean()),
        "population_std": population_std,
        "rms": float(teacher.square().mean().sqrt()),
        "minimum": float(teacher.min()),
        "maximum": float(teacher.max()),
    }


def _validate_args(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _load_config()
    for name in ("train", "dev", "teacher", "phase1a_comparison", "phase1a_raw_artifacts"):
        if not getattr(args, name).is_file():
            raise ValueError(f"R11_new bridge {name} path is not a file.")
    for name in ("dreamlite", "reader", "phase1a_target_root"):
        if not getattr(args, name).is_dir():
            raise ValueError(f"R11_new bridge {name} path is not a directory.")
    if (
        torch.device(args.dreamlite_device) != torch.device("cuda:0")
        or torch.device(args.reader_device) != torch.device("cuda:1")
    ):
        raise ValueError("R11_new bridge requires the locked cuda:0 DreamLite / cuda:1 Reader devices.")
    if not args.strict_determinism:
        raise ValueError("R11_new bridge requires strict determinism.")
    if r8.git_value("status", "--porcelain") and not args.allow_dirty:
        raise ValueError("R11_new bridge refuses a dirty source tree.")
    fixed = config["unchanged_contract"]
    if r5.sha256_file(args.train) != fixed["train_sha256"] or r5.sha256_file(args.dev) != fixed["dev_sha256"]:
        raise ValueError("R11_new bridge data binding drifted.")
    output = args.output_dir.resolve().as_posix()
    if not output.startswith("/inspire/ssd/"):
        raise ValueError("R11_new bridge formal artifacts must live under /inspire/ssd/.")
    parent = _validate_parent_evidence(args, config)
    _teacher, teacher_record = _load_teacher(args.teacher)
    return parent, teacher_record


def _evaluation_rows(
    *,
    reader_fn: Any,
    segment: Any,
    checkpoint: str,
    images: Sequence[tuple[str, Tensor]],
) -> list[dict[str, Any]]:
    item = phase1a._item(segment, images[0][1])
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for condition, image in images:
            for view_index, permutation in enumerate(r5.REVERSE_CYCLIC4):
                ordered = tuple(segment.query.choices[index] for index in permutation)
                ordered_target = permutation.index(segment.query.target_index)
                output = reader_fn(
                    image,
                    r5.format_mcq_query(segment.query.text, ordered),
                    ordered,
                    ordered_target,
                )
                rows.append(
                    r5._choice_row(
                        reader_output=output,
                        item=item,
                        checkpoint_label=checkpoint,
                        suite=SUITE,
                        condition=condition,
                        permutation=permutation,
                        view_index=view_index,
                        donor_item_id=None,
                    )
                )
    return rows


def _distance_statistics(output: phase1a.OracleForward, teacher: Tensor, *, m0_mse: float) -> dict[str, Any]:
    mse = float((output.z_t.float() - teacher).square().mean().detach())
    return core.bridge_distance_statistics(
        mse=mse,
        m0_mse=m0_mse,
        tensor_numel=teacher.numel(),
        teacher_population_std=core.BRIDGE_TEACHER_STD,
    )


def _reconstruct_start_state(source_compute: Tensor, x_t_fp32: Tensor, *, actual_sigma: float) -> Tensor:
    """Use the sampler's two operations in its original compute dtype."""

    if (
        source_compute.shape != x_t_fp32.shape
        or x_t_fp32.dtype != torch.float32
        or source_compute.device != x_t_fp32.device
        or source_compute.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        or isinstance(actual_sigma, bool)
        or not isinstance(actual_sigma, (int, float))
        or not math.isfinite(actual_sigma)
        or not 0.0 < actual_sigma <= 1.0
    ):
        raise ValueError("Bridge start-state reconstruction tensor/sigma contract drifted.")
    return source_compute.mul(1.0 - actual_sigma).add(
        x_t_fp32.to(dtype=source_compute.dtype), alpha=actual_sigma
    )


def _source_only_initialization(
    *,
    source_latents: Tensor,
    scheduler: Any,
    compute_device: torch.device,
    output_dir: Path,
) -> tuple[Tensor, dict[str, Any]]:
    """Construct x_T from source only; scheduler/device/output are execution metadata.

    No oracle/model, teacher, query, choices, answer, target index, or sample ID
    enters this function. The caller installs the returned tensor only after the
    initialization artifact has been persisted and before its first forward.
    """
    if (
        not isinstance(source_latents, Tensor)
        or tuple(source_latents.shape) != (1, 4, 128, 128)
        or source_latents.dtype not in (torch.float32, torch.bfloat16)
        or source_latents.device != torch.device(compute_device)
        or not torch.isfinite(source_latents).all()
        or canonical_tensor_sha256(source_latents.detach().float().cpu()) != core.BRIDGE_SOURCE_LATENTS_SHA256
    ):
        raise ValueError("Source-only initialization source/device contract drifted.")
    artifact_path = output_dir / "initialization" / "source_only_initialization.pt"
    if artifact_path.exists() or artifact_path.is_symlink():
        raise ValueError("Source-only initialization requires a fresh artifact path.")
    nominal_sigmas = torch.linspace(
        phase1a.EDIT_START_SIGMA,
        phase1a.EDIT_START_SIGMA / phase1a.NUM_DENOISING_STEPS,
        phase1a.NUM_DENOISING_STEPS,
    ).tolist()
    _, effective_sigmas = scheduler._prepare_timesteps(
        source_latents, phase1a.NUM_DENOISING_STEPS, nominal_sigmas,
        sigmas_are_effective=True,
    )
    if not phase1a.phase1a_effective_sigmas_match(effective_sigmas):
        raise RuntimeError("Source-only initialization effective schedule drifted.")
    actual_sigmas = [float(value) for value in effective_sigmas]
    actual_sigma = actual_sigmas[0]
    source_fp32 = source_latents.detach().float()
    x_t_fp32 = source_fp32.clone()
    reconstructed_start = _reconstruct_start_state(
        source_latents.detach(), x_t_fp32, actual_sigma=actual_sigma,
    )
    if not torch.isfinite(reconstructed_start).all() or not torch.equal(x_t_fp32, source_fp32):
        raise RuntimeError("Source-only initialization source equality or finite reconstruction failed.")
    metadata = {
        "schema": INITIALIZATION_SCHEMA,
        "compute_device_type": source_latents.device.type,
        "parameter_dtype": "torch.float32",
        "compute_dtype": str(source_latents.dtype),
        "shape": [1, 4, 128, 128],
        "formula": INITIALIZATION_FORMULA,
        "reconstruction_operator_order": INITIALIZATION_RECONSTRUCTION_ORDER,
        "nominal_effective_sigmas": list(phase1a.EFFECTIVE_SIGMAS),
        "actual_effective_sigmas": actual_sigmas,
        "actual_effective_sigma0": actual_sigma,
        "initial_x_t_equals_source": True,
        "initialization_teacher_assisted": False,
    }
    tensors = {
        "source_latents_fp32": source_fp32.detach().cpu().clone(),
        "x_T_init_fp32": x_t_fp32.detach().cpu().clone(),
        "reconstructed_start_state_compute": reconstructed_start.detach().cpu().clone(),
    }
    hashes = {key: canonical_tensor_sha256(value) for key, value in tensors.items()}
    phase1a._atomic_torch_save(artifact_path, {**metadata, **tensors, "tensor_sha256": hashes})
    record = {
        **metadata, "artifact_path": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "artifact_sha256": phase1a._sha256(artifact_path),
        "tensor_sha256": hashes, "passed": True,
    }
    return x_t_fp32, record


def _verify_source_only_initialization(
    record: Mapping[str, Any],
    *,
    oracle: phase1a.FrozenDreamLiteOracle,
    initial_output: phase1a.OracleForward,
) -> bool:
    """Verify step-zero source equality and real-backend arithmetic, never updated xT."""
    path = Path(str(record.get("artifact_path", "")))
    if (
        record.get("schema") != INITIALIZATION_SCHEMA
        or record.get("passed") is not True
        or not path.is_file()
        or path.stat().st_size != record.get("artifact_bytes")
        or phase1a._sha256(path) != record.get("artifact_sha256")
        or set(record) != set(INITIALIZATION_METADATA_KEYS) | {
            "artifact_path", "artifact_bytes", "artifact_sha256", "tensor_sha256", "passed",
        }
    ):
        return False
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != (
        set(INITIALIZATION_METADATA_KEYS) | set(INITIALIZATION_TENSOR_KEYS) | {"tensor_sha256"}
    ):
        return False
    source, x_t, start = (payload[key] for key in INITIALIZATION_TENSOR_KEYS)
    trajectory, actual_sigmas = initial_output.trajectory, initial_output.effective_sigmas
    device = oracle.source_latents.device
    if (
        not isinstance(trajectory, (tuple, list)) or len(trajectory) != 5
        or not isinstance(trajectory[0], Tensor)
        or trajectory[0].device != device
        or oracle.x_T_fp32.device != device or oracle.initial_x_T_fp32.device != device
        or not phase1a.phase1a_effective_sigmas_match(actual_sigmas)
        or not all(isinstance(value, Tensor) and tuple(value.shape) == (1, 4, 128, 128)
                   and torch.isfinite(value).all() for value in (source, x_t, start))
        or source.dtype != torch.float32 or x_t.dtype != torch.float32
        or start.dtype != oracle.compute_dtype or start.dtype != oracle.source_latents.dtype
        or start.dtype != trajectory[0].dtype
    ):
        return False
    expected_metadata = {
        "schema": INITIALIZATION_SCHEMA, "compute_device_type": device.type,
        "parameter_dtype": "torch.float32", "compute_dtype": str(start.dtype), "shape": [1, 4, 128, 128],
        "formula": INITIALIZATION_FORMULA, "reconstruction_operator_order": INITIALIZATION_RECONSTRUCTION_ORDER,
        "nominal_effective_sigmas": list(phase1a.EFFECTIVE_SIGMAS),
        "actual_effective_sigmas": list(actual_sigmas), "actual_effective_sigma0": float(actual_sigmas[0]),
        "initial_x_t_equals_source": True, "initialization_teacher_assisted": False,
    }
    for item in (record, payload):
        if any(item.get(key) != value for key, value in expected_metadata.items()):
            return False
        if item.get("initial_x_t_equals_source") is not True or item.get("initialization_teacher_assisted") is not False:
            return False
        sigma = item.get("actual_effective_sigma0")
        if isinstance(sigma, bool) or not isinstance(sigma, (float, int)) or not math.isfinite(sigma):
            return False
    # CPU/CUDA BF16 kernels can differ. Preserve exact backend and operator order.
    source_on_device = source.to(device=device)
    recomputed_x_t = source_on_device.clone()
    recomputed_start = _reconstruct_start_state(
        source_on_device.to(dtype=start.dtype), recomputed_x_t,
        actual_sigma=float(actual_sigmas[0]),
    )
    hashes = {key: canonical_tensor_sha256(payload[key]) for key in INITIALIZATION_TENSOR_KEYS}
    return bool(
        torch.equal(x_t, source)
        and torch.equal(x_t, recomputed_x_t.cpu())
        and torch.equal(start, recomputed_start.cpu())
        and hashes == payload.get("tensor_sha256") == record.get("tensor_sha256")
        and hashes["source_latents_fp32"] == core.BRIDGE_SOURCE_LATENTS_SHA256
        and hashes["source_latents_fp32"] == canonical_tensor_sha256(oracle.source_latents.detach().float().cpu())
        and hashes["x_T_init_fp32"] == canonical_tensor_sha256(oracle.initial_x_T_fp32.detach().cpu())
        and hashes["x_T_init_fp32"] == canonical_tensor_sha256(oracle.x_T_fp32.detach().cpu())
        and hashes["reconstructed_start_state_compute"] == canonical_tensor_sha256(trajectory[0].detach().cpu())
    )


def _checkpoint_payload(
    *,
    step: int,
    oracle: phase1a.FrozenDreamLiteOracle,
    optimizer: torch.optim.Optimizer,
    output: phase1a.OracleForward,
    teacher: Tensor,
    m0_mse: float,
    manifest_sha256: str,
    condition_sha256: str,
) -> dict[str, Any]:
    trajectory = tuple(value.detach().float().cpu() for value in output.trajectory)
    x_t = oracle.x_T_fp32.detach().float().cpu()
    z_t = output.z_t.detach().float().cpu()
    distance = _distance_statistics(output, teacher, m0_mse=m0_mse)
    expected_start = _reconstruct_start_state(
        oracle.source_latents, oracle.x_T_fp32,
        actual_sigma=float(output.effective_sigmas[0]),
    )
    trajectory_point0_formula_valid = bool(
        output.trajectory[0].device == oracle.source_latents.device
        and output.trajectory[0].dtype == oracle.source_latents.dtype == oracle.compute_dtype
        and torch.equal(output.trajectory[0], expected_start)
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "optimizer_step": step,
        "objective": "canonical_r11_latent_fp32_mean_mse",
        "x_T_fp32": x_t,
        "z_t_fp32": z_t,
        "trajectory_fp32": trajectory,
        "effective_sigmas": list(output.effective_sigmas),
        "compute_dtype": str(oracle.source_latents.dtype),
        "compute_device_type": oracle.source_latents.device.type,
        "optimizer": optimizer.state_dict(),
        "optimizer_state_sha256": canonical_object_sha256(optimizer.state_dict()),
        "trajectory_point0_formula_valid": trajectory_point0_formula_valid,
        "distance_statistics": distance,
        "teacher_tensor_sha256": core.BRIDGE_TEACHER_TENSOR_SHA256,
        "manifest_sha256": manifest_sha256,
        "condition_artifact_sha256": condition_sha256,
        "tensor_sha256": {
            "x_T_fp32": canonical_tensor_sha256(x_t),
            "z_t_fp32": canonical_tensor_sha256(z_t),
            "trajectory_fp32": [canonical_tensor_sha256(value) for value in trajectory],
        },
    }


def _save_checkpoint(
    *,
    step: int,
    oracle: phase1a.FrozenDreamLiteOracle,
    optimizer: torch.optim.Optimizer,
    teacher: Tensor,
    m0_mse: float,
    manifest_sha256: str,
    condition_sha256: str,
    output_dir: Path,
    output: phase1a.OracleForward | None = None,
) -> tuple[dict[str, Any], phase1a.OracleForward]:
    if output is None:
        with torch.no_grad():
            output = oracle()
    pt_path = output_dir / "checkpoints" / f"step-{step:03d}.pt"
    png_path = output_dir / "images" / f"step-{step:03d}.png"
    payload = _checkpoint_payload(
        step=step,
        oracle=oracle,
        optimizer=optimizer,
        output=output,
        teacher=teacher,
        m0_mse=m0_mse,
        manifest_sha256=manifest_sha256,
        condition_sha256=condition_sha256,
    )
    phase1a._atomic_torch_save(pt_path, payload)
    phase1a._save_image(png_path, output.image)
    record = {
        "schema": CHECKPOINT_HASH_SCHEMA,
        "optimizer_step": step,
        "checkpoint_path": str(pt_path),
        "checkpoint_bytes": pt_path.stat().st_size,
        "checkpoint_sha256": phase1a._sha256(pt_path),
        "png_path": str(png_path),
        "png_bytes": png_path.stat().st_size,
        "png_sha256": phase1a._sha256(png_path),
        "trajectory_points": len(output.trajectory),
        "effective_sigmas": list(output.effective_sigmas),
        "distance_statistics": payload["distance_statistics"],
        "tensor_sha256": payload["tensor_sha256"],
        "optimizer_state_sha256": payload["optimizer_state_sha256"],
        "trajectory_point0_formula_valid": payload[
            "trajectory_point0_formula_valid"
        ],
        "compute_dtype": payload["compute_dtype"],
        "compute_device_type": payload["compute_device_type"],
    }
    phase1a._atomic_json(output_dir / "checkpoint_hashes" / f"step-{step:03d}.json", record)
    return record, output


def _optimizer_state_matches_step(value: Any, *, expected_step: int) -> bool:
    if not isinstance(value, Mapping):
        return False
    state = value.get("state")
    groups = value.get("param_groups")
    if not isinstance(state, Mapping) or not isinstance(groups, list) or len(groups) != 1:
        return False
    group = groups[0]
    expected_lr = (
        LEARNING_RATE
        if expected_step == 0
        else core.bridge_optimizer_learning_rate(expected_step)
    )
    if (
        not isinstance(group, Mapping)
        or group.get("params") != [0]
        or not math.isclose(
            float(group.get("lr", math.nan)),
            expected_lr,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        return False
    if expected_step == 0:
        return not state
    if set(state) != {0}:
        return False
    parameter_state = state[0]
    if not isinstance(parameter_state, Mapping) or set(parameter_state) != {
        "step",
        "exp_avg",
        "exp_avg_sq",
    }:
        return False
    step_value = parameter_state["step"]
    if not isinstance(step_value, Tensor) or step_value.numel() != 1:
        return False
    moments = (parameter_state["exp_avg"], parameter_state["exp_avg_sq"])
    return bool(
        float(step_value.item()) == float(expected_step)
        and all(
            isinstance(moment, Tensor)
            and moment.dtype == torch.float32
            and tuple(moment.shape) == (1, 4, 128, 128)
            and torch.isfinite(moment).all()
            for moment in moments
        )
    )


def _verify_checkpoint_record(
    record: Mapping[str, Any],
    *,
    expected_step: int,
    teacher_cpu: Tensor,
    m0_mse: float,
    source_latents_cpu: Tensor,
    compute_dtype: torch.dtype,
    compute_device: torch.device,
) -> bool:
    compute_device = torch.device(compute_device)
    pt_path = Path(str(record.get("checkpoint_path", "")))
    png_path = Path(str(record.get("png_path", "")))
    if not pt_path.is_file() or not png_path.is_file():
        return False
    if (
        pt_path.stat().st_size != record.get("checkpoint_bytes")
        or png_path.stat().st_size != record.get("png_bytes")
        or phase1a._sha256(pt_path) != record.get("checkpoint_sha256")
        or phase1a._sha256(png_path) != record.get("png_sha256")
    ):
        return False
    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    trajectory = payload.get("trajectory_fp32")
    hashes = payload.get("tensor_sha256", {})
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("optimizer_step") != expected_step
        or record.get("optimizer_step") != expected_step
        or not isinstance(trajectory, (tuple, list))
        or len(trajectory) != 5
        or payload.get("compute_dtype") != str(compute_dtype)
        or record.get("compute_dtype") != str(compute_dtype)
        or payload.get("compute_device_type") != compute_device.type
        or record.get("compute_device_type") != compute_device.type
        or payload.get("trajectory_point0_formula_valid") is not True
        or record.get("trajectory_point0_formula_valid") is not True
        or any(
            not isinstance(value, Tensor)
            or value.dtype != torch.float32
            or tuple(value.shape) != (1, 4, 128, 128)
            or not torch.isfinite(value).all()
            for value in (payload.get("x_T_fp32"), payload.get("z_t_fp32"), *trajectory)
        )
        or hashes != record.get("tensor_sha256")
        or hashes.get("x_T_fp32") != canonical_tensor_sha256(payload["x_T_fp32"].float())
        or hashes.get("z_t_fp32") != canonical_tensor_sha256(payload["z_t_fp32"].float())
        or hashes.get("trajectory_fp32") != [canonical_tensor_sha256(value.float()) for value in trajectory]
        or payload.get("optimizer_state_sha256")
        != canonical_object_sha256(payload.get("optimizer"))
        or record.get("optimizer_state_sha256") != payload.get("optimizer_state_sha256")
        or not _optimizer_state_matches_step(
            payload.get("optimizer"),
            expected_step=expected_step,
        )
        or not phase1a.phase1a_effective_sigmas_match(payload.get("effective_sigmas"))
        or payload.get("effective_sigmas") != record.get("effective_sigmas")
        or payload.get("teacher_tensor_sha256") != core.BRIDGE_TEACHER_TENSOR_SHA256
    ):
        return False
    if canonical_tensor_sha256(source_latents_cpu.float()) != core.BRIDGE_SOURCE_LATENTS_SHA256:
        return False
    expected_start = _reconstruct_start_state(
        source_latents_cpu.to(device=compute_device, dtype=compute_dtype),
        payload["x_T_fp32"].to(device=compute_device),
        actual_sigma=float(payload["effective_sigmas"][0]),
    )
    if not torch.equal(trajectory[0], expected_start.float().cpu()) or not torch.equal(trajectory[-1], payload["z_t_fp32"]):
        return False
    mse = float((payload["z_t_fp32"].to(device=compute_device) - teacher_cpu.to(device=compute_device)).square().mean())
    recomputed = core.bridge_distance_statistics(
        mse=mse,
        m0_mse=m0_mse,
        tensor_numel=teacher_cpu.numel(),
    )
    for key, value in recomputed.items():
        if isinstance(value, float) and not math.isclose(
            value, float(payload["distance_statistics"][key]), rel_tol=1e-6, abs_tol=1e-8
        ):
            return False
    return payload["distance_statistics"] == record.get("distance_statistics")


def _train_step(
    *,
    step_zero: int,
    oracle: phase1a.FrozenDreamLiteOracle,
    optimizer: torch.optim.Optimizer,
    teacher: Tensor,
    m0_mse: float,
    reader: nn.Module,
) -> dict[str, Any]:
    update_index = step_zero + 1
    learning_rate = core.bridge_optimizer_learning_rate(update_index)
    if len(optimizer.param_groups) != 1:
        raise RuntimeError("R11_new bridge optimizer param-group count drifted.")
    optimizer.param_groups[0]["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    output = oracle()
    loss = (output.z_t.float() - teacher).square().mean()
    if loss.numel() != 1 or not torch.isfinite(loss):
        raise RuntimeError("R11_new bridge produced an invalid scalar MSE.")
    distance = core.bridge_distance_statistics(
        mse=float(loss.detach()),
        m0_mse=m0_mse,
        tensor_numel=teacher.numel(),
    )
    loss.backward()
    gradient = oracle.x_T_fp32.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("R11_new bridge x_T received no finite gradient.")
    gradient_norm = float(gradient.double().norm())
    nonzero_fraction = float((gradient != 0).double().mean())
    if gradient_norm <= 0.0 or nonzero_fraction <= 0.0:
        raise RuntimeError("R11_new bridge x_T received a zero gradient.")
    frozen_modules = (oracle.unet, oracle.vae, oracle.text_encoder, reader)
    if any(parameter.grad is not None for module in frozen_modules for parameter in module.parameters()):
        raise RuntimeError("A frozen model parameter received a bridge gradient.")
    x_before = oracle.x_T_fp32.detach().clone()
    optimizer.step()
    x_after = oracle.x_T_fp32.detach()
    optimizer.zero_grad(set_to_none=True)
    return {
        "schema": METRICS_SCHEMA,
        "kind": "optimizer_step",
        "optimizer_step": update_index,
        "target_index": core.BRIDGE_TARGET_INDEX,
        "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID,
        "objective": "canonical_r11_latent_fp32_mean_mse",
        "loss_before_step": float(loss.detach()),
        **distance,
        "gradient_norm": gradient_norm,
        "gradient_nonzero_fraction": nonzero_fraction,
        "x_T_before_step": phase1a._tensor_stats(x_before),
        "x_T_after_step": phase1a._tensor_stats(x_after),
        "x_T_update_norm": float((x_after - x_before).double().norm()),
        "z_t_before_step": phase1a._tensor_stats(output.z_t),
        "image_before_step": phase1a._image_stats(output.image),
        "trajectory_before_step": phase1a._trajectory_stats(output.trajectory),
        "trajectory_points": len(output.trajectory),
        "dreamlite_denoising_steps": len(output.trajectory) - 1,
        "effective_sigmas": list(output.effective_sigmas),
        "full_dreamlite_forward_executed": True,
        "gradient_mode": "full",
        "gradient_clipping_applied": False,
        "learning_rate": learning_rate,
        "weight_decay": WEIGHT_DECAY,
        "reader_gradient_calls": 0,
        "teacher_tensor_sha256": core.BRIDGE_TEACHER_TENSOR_SHA256,
    }


def _all_frozen(module: nn.Module) -> bool:
    return all(not parameter.requires_grad and parameter.grad is None for parameter in module.parameters())


def _required_formal_artifacts(output_dir: Path) -> bool:
    required = {
        ".r11_new_bridge_output_owner.json",
        "environment.txt",
        "runtime.json",
        "model_snapshot_verification_start.json",
        "model_snapshot_verification_end.json",
        "condition/official_full_condition.pt",
        "initialization/source_only_initialization.pt",
        "teacher/canonical_r11_target01.pt",
        "teacher/canonical_r11_target01.png",
        "manifest.json",
        "metrics.jsonl",
        "evaluation_rows.jsonl",
        "endpoint_raw.pt",
        "endpoint_raw.png",
    }
    required |= {f"checkpoints/step-{step:03d}.pt" for step in CHECKPOINT_STEPS}
    required |= {f"images/step-{step:03d}.png" for step in CHECKPOINT_STEPS}
    required |= {f"checkpoint_hashes/step-{step:03d}.json" for step in CHECKPOINT_STEPS}
    return all((output_dir / relative).is_file() for relative in required)


def _technical_gate(
    metrics: Sequence[Mapping[str, Any]],
    *,
    oracle: phase1a.FrozenDreamLiteOracle,
    reader: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_records: Sequence[Mapping[str, Any]],
    teacher_cpu: Tensor,
    m0_mse: float,
    manifest: Mapping[str, Any],
    snapshots_unchanged: bool,
    output_dir: Path,
    initialization_verified: bool,
) -> dict[str, Any]:
    expected_steps = list(range(1, OPTIMIZER_STEPS + 1))
    trainable = [name for name, parameter in oracle.named_parameters() if parameter.requires_grad]
    finite_fields = (
        "mse",
        "rmse",
        "l2_distance",
        "mse_ratio_to_m0",
        "l2_distance_ratio_to_m0",
        "teacher_normalized_rmse",
        "gradient_norm",
        "gradient_nonzero_fraction",
        "x_T_update_norm",
    )
    finite_metrics = all(math.isfinite(float(row[field])) for row in metrics for field in finite_fields)
    four_step = all(
        int(row.get("dreamlite_denoising_steps", -1)) == 4
        and int(row.get("trajectory_points", -1)) == 5
        and bool(row.get("full_dreamlite_forward_executed"))
        and phase1a.phase1a_effective_sigmas_match(row.get("effective_sigmas"))
        for row in metrics
    )
    gradient_valid = all(
        float(row["gradient_norm"]) > 0.0 and float(row["gradient_nonzero_fraction"]) > 0.0 for row in metrics
    )
    checkpoint_map = {int(row.get("optimizer_step", -1)): row for row in checkpoint_records}
    checkpoint_hashes_valid = set(checkpoint_map) == set(CHECKPOINT_STEPS) and all(
        _verify_checkpoint_record(
            checkpoint_map[step],
            expected_step=step,
            teacher_cpu=teacher_cpu,
            m0_mse=m0_mse,
            source_latents_cpu=oracle.source_latents.detach().cpu(),
            compute_dtype=oracle.compute_dtype,
            compute_device=oracle.source_latents.device,
        )
        for step in CHECKPOINT_STEPS
    )
    optimizer_valid = bool(
        type(optimizer) is torch.optim.Adam
        and len(optimizer.param_groups) == 1
        and float(optimizer.param_groups[0]["lr"])
        == core.bridge_optimizer_learning_rate(OPTIMIZER_STEPS)
        and float(optimizer.param_groups[0]["weight_decay"]) == WEIGHT_DECAY
        and len(optimizer.param_groups[0]["params"]) == 1
        and optimizer.param_groups[0]["params"][0] is oracle.x_T_fp32
    )
    initialization = manifest.get("initialization_binding", {})
    initialization_path = Path(str(initialization.get("artifact_path", "")))
    initialization_unchanged = bool(
        initialization_verified
        and initialization.get("passed") is True
        and initialization_path.is_file()
        and initialization_path.stat().st_size == initialization.get("artifact_bytes")
        and phase1a._sha256(initialization_path) == initialization.get("artifact_sha256")
    )
    step0 = checkpoint_map.get(0, {}).get("tensor_sha256", {})
    step0_parity = bool(
        initialization_unchanged
        and step0.get("x_T_fp32") == initialization.get("tensor_sha256", {}).get("x_T_init_fp32")
        and step0.get("z_t_fp32") == manifest.get("parity_checks", {}).get("observed", {}).get("initial_z_t_fp32_sha256")
        and manifest.get("parity_checks", {}).get("passed") is True
    )
    lr_schedule = all(
        math.isclose(
            float(row.get("learning_rate", math.nan)),
            core.bridge_optimizer_learning_rate(int(row["optimizer_step"])),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        for row in metrics
    )
    teacher_binding = bool(
        manifest.get("teacher", {}).get("file_sha256") == core.BRIDGE_TEACHER_FILE_SHA256
        and manifest.get("teacher", {}).get("tensor_sha256") == core.BRIDGE_TEACHER_TENSOR_SHA256
        and phase1a._sha256(output_dir / "teacher" / "canonical_r11_target01.pt") == core.BRIDGE_TEACHER_FILE_SHA256
    )
    frozen = all(_all_frozen(module) for module in (oracle.unet, oracle.vae, oracle.text_encoder, reader))
    audit = {
        "receipts_exact": len(metrics) == OPTIMIZER_STEPS,
        "steps_contiguous": [int(row["optimizer_step"]) for row in metrics] == expected_steps,
        "four_step_schedule_exact": four_step,
        "finite_metrics": finite_metrics,
        "finite_nonzero_gradient_every_step": gradient_valid,
        "only_x_T_fp32_trainable": trainable == ["x_T_fp32"],
        "frozen_gradients_absent": frozen,
        "snapshots_unchanged": snapshots_unchanged,
        "optimizer_contract_valid": optimizer_valid,
        "optimizer_lr_schedule_exact": lr_schedule,
        "source_only_initialization_artifact_valid": initialization_unchanged,
        "trajectory_point0_binding_valid_every_checkpoint": checkpoint_hashes_valid,
        "gradient_clipping_absent": all(row.get("gradient_clipping_applied") is False for row in metrics),
        "checkpoint_hashes_valid": checkpoint_hashes_valid,
        "condition_artifact_valid": phase1a._verify_condition_record(manifest["condition_artifact"]),
        "step0_parity_valid": step0_parity,
        "teacher_binding_valid": teacher_binding,
        "artifact_contract_valid": _required_formal_artifacts(output_dir),
    }
    return {
        "schema": TECHNICAL_GATE_SCHEMA,
        "passed": core.bridge_technical_gate(audit),
        **audit,
        "optimizer_step_records": len(metrics),
        "checkpoint_steps_observed": sorted(checkpoint_map),
        "trainable_parameter_names": trainable,
        "minimum_gradient_norm": min(float(row["gradient_norm"]) for row in metrics),
        "minimum_gradient_nonzero_fraction": min(float(row["gradient_nonzero_fraction"]) for row in metrics),
        "formal_success_gate": False,
    }


def _parity_checks(
    *,
    context: Mapping[str, Any],
    oracle: phase1a.FrozenDreamLiteOracle,
    initial_output: phase1a.OracleForward,
    initialization_binding: Mapping[str, Any],
) -> dict[str, Any]:
    condition = context["condition_record"]
    observed = {
        "target_index": core.BRIDGE_TARGET_INDEX,
        "target_segment_id": context["target"].segment_id,
        "initial_x_T_fp32_sha256": canonical_tensor_sha256(oracle.initial_x_T_fp32.detach().float().cpu()),
        "initial_z_t_fp32_sha256": canonical_tensor_sha256(initial_output.z_t.detach().float().cpu()),
        "blank_source_rgb_sha256": canonical_tensor_sha256(context["source_rgb"].detach().cpu()),
        "source_latents_fp32_sha256": canonical_tensor_sha256(context["source_latents"].detach().float().cpu()),
        "event_text_sha256": condition["event_text_sha256"],
        "condition_prompt_embeds_sha256": condition["tensor_sha256"]["prompt_embeds"],
        "condition_attention_mask_sha256": condition["tensor_sha256"]["attention_mask"],
    }
    expected = dict(core.BRIDGE_UNCHANGED_PARITY_BINDINGS)
    initialization_valid = _verify_source_only_initialization(
        initialization_binding, oracle=oracle, initial_output=initial_output
    )
    return {
        "passed": all(observed[key] == value for key, value in expected.items()) and initialization_valid,
        "observed": observed,
        "expected": expected,
        "source_only_initialization_artifact_valid": initialization_valid,
    }


def _manifest(
    *,
    args: argparse.Namespace,
    context: Mapping[str, Any],
    oracle: phase1a.FrozenDreamLiteOracle,
    initial_output: phase1a.OracleForward,
    parent_binding: Mapping[str, Any],
    teacher_record: Mapping[str, Any],
    snapshot_bindings: Mapping[str, Any],
    determinism: Mapping[str, Any] | None,
    initialization_binding: Mapping[str, Any],
) -> dict[str, Any]:
    parity = _parity_checks(
        context=context, oracle=oracle, initial_output=initial_output,
        initialization_binding=initialization_binding,
    )
    if not parity["passed"]:
        raise RuntimeError(f"R11_new bridge step-0 parity failed: {parity}")
    information_boundary = phase1a._writer_information_boundary(context["target"])
    if not information_boundary["passed"]:
        raise RuntimeError("R11_new bridge information boundary failed.")
    information_boundary = {
        **information_boundary,
        "canonical_teacher_used_only_by_dense_loss": True,
        "canonical_teacher_used_by_initialization_and_dense_loss": False,
        "teacher_assisted_initialization": False,
        "initialization_teacher_assisted": False,
        "optimization_teacher_supervised": True,
        "answer_independent_writer_usable": False,
        "formal_success": False,
        "phase2_allowed": False,
        "reader_used_only_for_fixed_audit": True,
        "reader_gradient_calls_during_optimization": 0,
    }
    return {
        "schema": MANIFEST_SCHEMA,
        "protocol": PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "mode": args.mode,
        "git_commit": r8.git_value("rev-parse", "HEAD"),
        "git_dirty": bool(r8.git_value("status", "--porcelain")),
        "preregistered_config_path": str(CONFIG_PATH),
        "preregistered_config_file_sha256": core.BRIDGE_CONFIG_FILE_SHA256,
        "preregistered_config_canonical_sha256": core.BRIDGE_CONFIG_CANONICAL_SHA256,
        "target_index": core.BRIDGE_TARGET_INDEX,
        "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID,
        "target_segment": context["target"].to_dict(),
        "parent_phase1a": dict(parent_binding),
        "teacher": dict(teacher_record),
        "initialization_binding": dict(initialization_binding),
        "parity_checks": parity,
        "train_sha256": r5.sha256_file(args.train),
        "dev_sha256": r5.sha256_file(args.dev),
        "selected_segment_ids": [segment.segment_id for segment in context["selected"]],
        "selected_segments_sha256": context["selected_sha"],
        "model_snapshot_payloads_start": dict(snapshot_bindings),
        "strict_determinism": dict(determinism) if determinism is not None else None,
        "source_rgb": {
            "value": "127/255",
            "shape": list(context["source_rgb"].shape),
            "dtype": str(context["source_rgb"].dtype),
            "sha256": parity["observed"]["blank_source_rgb_sha256"],
        },
        "source_latents": {
            "shape": list(context["source_latents"].shape),
            "dtype": str(context["source_latents"].dtype),
            "device": str(context["source_latents"].device),
            "sha256": parity["observed"]["source_latents_fp32_sha256"],
        },
        "initial_x_T_fp32": {
            "distribution": "source-only deterministic initialization",
            "shape": list(oracle.initial_x_T_fp32.shape),
            "dtype": str(oracle.initial_x_T_fp32.dtype),
            "sha256": parity["observed"]["initial_x_T_fp32_sha256"],
        },
        "condition_artifact": dict(context["condition_record"]),
        "information_boundary": information_boundary,
        "single_changed_factor": {
            "factor": "x_T_initialization",
            "from": "answer-independent event-keyed standard Gaussian with global seed 0",
            "to": "source-only deterministic initialization",
            "initialization_teacher_assisted": False,
            "optimization_teacher_supervised": True,
            "answer_independent_writer_usable": False,
        },
        "fixed_contract": {
            "dreamlite_device": str(torch.device(args.dreamlite_device)),
            "reader_device": str(torch.device(args.reader_device)),
            "only_trainable": "x_T_fp32",
            "dreamlite_path": "complete frozen source-anchored four-step path",
            "effective_sigmas": list(phase1a.EFFECTIVE_SIGMAS),
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "learning_rate_schedule": {
                "name": "constant_then_post128_cosine_to_zero",
                "intervention_first_update": core.BRIDGE_LR_INTERVENTION_FIRST_UPDATE,
                "formula": "0.05 for u<=128; 0.025*(1+cos(pi*(u-128)/128)) otherwise",
            },
            "weight_decay": WEIGHT_DECAY,
            "optimizer_steps": OPTIMIZER_STEPS if args.mode == "formal" else 0,
            "gradient_clipping": None,
            "checkpoint_steps": list(CHECKPOINT_STEPS) if args.mode == "formal" else [0],
            "primary_endpoint": core.BRIDGE_PRIMARY_ENDPOINT,
            "m0_definition": "the unoptimized endpoint after the complete four-step DreamLite path from the new source-only x_T initialization",
            "best_checkpoint_selection_forbidden": True,
            "reader_gradient_calls_during_optimization": 0,
        },
        "bridge_diagnostic_only": True,
        "phase2_allowed": False,
        "formal_success_gate": False,
        "initialization_teacher_assisted": False,
        "optimization_teacher_supervised": True,
        "answer_independent_writer_usable": False,
    }


def _write_snapshot_end(output_dir: Path, snapshot_bindings: Mapping[str, Any]) -> bool:
    observed = {name: verify_snapshot_binding(value) for name, value in snapshot_bindings.items()}
    unchanged = observed == dict(snapshot_bindings)
    phase1a._atomic_json(
        output_dir / "model_snapshot_verification_end.json",
        {
            "schema": "vision_memory.r11-new-canonical-latent-bridge-model-snapshot-end.v1",
            "passed": unchanged,
            "bindings": observed,
        },
    )
    return unchanged


def _preflight(
    *,
    args: argparse.Namespace,
    oracle: phase1a.FrozenDreamLiteOracle,
    reader: nn.Module,
    eval_reader: Any,
    target: Any,
    teacher: Tensor,
    teacher_image: Tensor,
    initial_output: phase1a.OracleForward,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    snapshot_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    m0_mse = float((initial_output.z_t.float() - teacher).square().mean().detach())
    initialization_verified = _verify_source_only_initialization(
        manifest["initialization_binding"], oracle=oracle, initial_output=initial_output
    )
    teacher_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        checkpoint="canonical_teacher",
        images=(("teacher", teacher_image),),
    )
    phase1a._append_jsonl(args.output_dir / "evaluation_rows.jsonl", teacher_rows)
    teacher_stats = core.reader_checkpoint_statistics(teacher_rows, checkpoint="canonical_teacher", condition="teacher")
    optimizer.zero_grad(set_to_none=True)
    output = initial_output
    loss = (output.z_t.float() - teacher).square().mean()
    loss.backward()
    gradient = oracle.x_T_fp32.grad
    gradient_valid = bool(
        gradient is not None
        and torch.isfinite(gradient).all()
        and float(gradient.double().norm()) > 0.0
        and float((gradient != 0).double().mean()) > 0.0
    )
    frozen = all(_all_frozen(module) for module in (oracle.unet, oracle.vae, oracle.text_encoder, reader))
    optimizer.zero_grad(set_to_none=True)
    manifest_sha = phase1a._sha256(args.output_dir / "manifest.json")
    checkpoint, _ = _save_checkpoint(
        step=0,
        oracle=oracle,
        optimizer=optimizer,
        teacher=teacher,
        m0_mse=m0_mse,
        manifest_sha256=manifest_sha,
        condition_sha256=str(manifest["condition_artifact"]["sha256"]),
        output_dir=args.output_dir,
        output=initial_output,
    )
    snapshots_unchanged = _write_snapshot_end(args.output_dir, snapshot_bindings)
    audit = {
        "full_forward_calls": 1,
        "backward_calls": 1,
        "optimizer_steps": 0,
        "four_dreamlite_steps": len(output.trajectory) == 5,
        "effective_sigmas_exact": phase1a.phase1a_effective_sigmas_match(output.effective_sigmas),
        "finite_nonzero_x_T_gradient": gradient_valid,
        "only_x_T_fp32_trainable": [name for name, parameter in oracle.named_parameters() if parameter.requires_grad]
        == ["x_T_fp32"],
        "frozen_gradients_absent": frozen,
        "step0_parity_valid": manifest["parity_checks"]["passed"],
        "source_only_initialization_artifact_valid": initialization_verified,
        "teacher_binding_valid": manifest["teacher"]["tensor_sha256"] == core.BRIDGE_TEACHER_TENSOR_SHA256,
        "teacher_replay_gate": core.teacher_replay_gate(teacher_stats),
        "checkpoint_hash_valid": _verify_checkpoint_record(
            checkpoint,
            expected_step=0,
            teacher_cpu=teacher.detach().cpu(),
            m0_mse=m0_mse,
            source_latents_cpu=oracle.source_latents.detach().cpu(),
            compute_dtype=oracle.compute_dtype,
            compute_device=oracle.source_latents.device,
        ),
        "snapshots_unchanged": snapshots_unchanged,
    }
    passed = (
        all(bool(value) for key, value in audit.items() if key not in {"optimizer_steps"})
        and audit["optimizer_steps"] == 0
    )
    report = {
        "schema": "vision_memory.r11-new-canonical-latent-bridge-preflight.v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "mode": "technical-preflight",
        "target_index": core.BRIDGE_TARGET_INDEX,
        "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID,
        "audit": audit,
        "teacher_replay_statistics": teacher_stats,
        "initial_distance_statistics": core.bridge_distance_statistics(
            mse=m0_mse, m0_mse=m0_mse, tensor_numel=teacher.numel()
        ),
        "bridge_result_evaluated": False,
        "phase2_allowed": False,
        "formal_success_gate": False,
        "initialization_teacher_assisted": False,
        "optimization_teacher_supervised": True,
        "answer_independent_writer_usable": False,
    }
    phase1a._atomic_json(args.output_dir / "technical_preflight.json", report)
    return report


def _formal(
    *,
    args: argparse.Namespace,
    oracle: phase1a.FrozenDreamLiteOracle,
    reader: nn.Module,
    eval_reader: Any,
    target: Any,
    source_latents: Tensor,
    teacher: Tensor,
    teacher_image: Tensor,
    initial_output: phase1a.OracleForward,
    optimizer: torch.optim.Optimizer,
    manifest: Mapping[str, Any],
    snapshot_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_sha = phase1a._sha256(args.output_dir / "manifest.json")
    reset_image = decode_model_latents_unit_interval(oracle.vae, source_latents, clamp=True).detach()
    m0_mse = float((initial_output.z_t.float() - teacher).square().mean().detach())
    rows_path = args.output_dir / "evaluation_rows.jsonl"
    teacher_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        checkpoint="canonical_teacher",
        images=(("teacher", teacher_image),),
    )
    phase1a._append_jsonl(rows_path, teacher_rows)
    teacher_stats = core.reader_checkpoint_statistics(
        teacher_rows,
        checkpoint="canonical_teacher",
        condition="teacher",
    )
    if not core.teacher_replay_gate(teacher_stats):
        raise RuntimeError("R11_new bridge formal teacher replay gate failed before optimizer step 0.")
    initialization_verified = _verify_source_only_initialization(
        manifest["initialization_binding"], oracle=oracle, initial_output=initial_output
    )
    if not initialization_verified:
        raise RuntimeError("R11_new source-only initialization verification failed before optimizer step 0.")
    m0_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        checkpoint="m0",
        images=(("normal", initial_output.image.detach()), ("reset", reset_image)),
    )
    phase1a._append_jsonl(rows_path, m0_rows)
    checkpoint_records: list[dict[str, Any]] = []
    checkpoint, _ = _save_checkpoint(
        step=0,
        oracle=oracle,
        optimizer=optimizer,
        teacher=teacher,
        m0_mse=m0_mse,
        manifest_sha256=manifest_sha,
        condition_sha256=str(manifest["condition_artifact"]["sha256"]),
        output_dir=args.output_dir,
        output=initial_output,
    )
    checkpoint_records.append(checkpoint)
    metrics: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "metrics.jsonl"
    started = time.monotonic()
    endpoint_output: phase1a.OracleForward | None = None
    for step_zero in range(OPTIMIZER_STEPS):
        metric = _train_step(
            step_zero=step_zero,
            oracle=oracle,
            optimizer=optimizer,
            teacher=teacher,
            m0_mse=m0_mse,
            reader=reader,
        )
        metric["elapsed_seconds"] = time.monotonic() - started
        metrics.append(metric)
        phase1a._append_jsonl(metrics_path, (metric,))
        step = step_zero + 1
        if step in CHECKPOINT_STEPS:
            checkpoint, snapshot = _save_checkpoint(
                step=step,
                oracle=oracle,
                optimizer=optimizer,
                teacher=teacher,
                m0_mse=m0_mse,
                manifest_sha256=manifest_sha,
                condition_sha256=str(manifest["condition_artifact"]["sha256"]),
                output_dir=args.output_dir,
            )
            checkpoint_records.append(checkpoint)
            if step == OPTIMIZER_STEPS:
                endpoint_output = snapshot
        print(
            json.dumps(
                {
                    "milestone": "r11_new_bridge_optimizer_step",
                    "optimizer_step": step,
                    "mse": metric["mse"],
                    "mse_ratio_to_m0": metric["mse_ratio_to_m0"],
                    "gradient_norm": metric["gradient_norm"],
                    "elapsed_seconds": metric["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if endpoint_output is None:
        raise RuntimeError("R11_new bridge raw step-256 endpoint is missing.")
    endpoint_rows = _evaluation_rows(
        reader_fn=eval_reader,
        segment=target,
        checkpoint=core.BRIDGE_PRIMARY_ENDPOINT,
        images=(("normal", endpoint_output.image.detach()), ("reset", reset_image)),
    )
    phase1a._append_jsonl(rows_path, endpoint_rows)
    shutil.copyfile(
        args.output_dir / "checkpoints" / "step-256.pt",
        args.output_dir / "endpoint_raw.pt",
    )
    shutil.copyfile(
        args.output_dir / "images" / "step-256.png",
        args.output_dir / "endpoint_raw.png",
    )
    snapshots_unchanged = _write_snapshot_end(args.output_dir, snapshot_bindings)
    technical = _technical_gate(
        metrics,
        oracle=oracle,
        reader=reader,
        optimizer=optimizer,
        checkpoint_records=checkpoint_records,
        teacher_cpu=teacher.detach().cpu(),
        m0_mse=m0_mse,
        manifest=manifest,
        snapshots_unchanged=snapshots_unchanged,
        output_dir=args.output_dir,
        initialization_verified=initialization_verified,
    )
    phase1a._atomic_json(args.output_dir / "technical_gate.json", technical)
    if technical.get("passed") is not True:
        raise RuntimeError("R11_new bridge formal technical gate failed; scientific bridge result is unevaluated.")
    distance = _distance_statistics(endpoint_output, teacher, m0_mse=m0_mse)
    m0_reader = core.reader_checkpoint_statistics(m0_rows, checkpoint="m0", condition="normal")
    endpoint_reader = core.reader_checkpoint_statistics(
        endpoint_rows, checkpoint=core.BRIDGE_PRIMARY_ENDPOINT, condition="normal"
    )
    teacher_gate = core.teacher_replay_gate(teacher_stats)
    distance_gate = core.bridge_distance_gate(distance, technical_gate=bool(technical["passed"] and teacher_gate))
    reader_gate = core.endpoint_reader_transfer_gate(endpoint_reader)
    bridge_gate = bool(technical["passed"] and teacher_gate and distance_gate and reader_gate)
    decision = core.bridge_decision(distance_pass=distance_gate, reader_transfer_pass=reader_gate)
    initialization_audit = core.bridge_initialization_hypothesis_audit(
        endpoint_mse=float(distance["mse"]),
        endpoint_reader_mean_ce=float(endpoint_reader["mean_ce"]),
        technical_gate=bool(technical["passed"]),
        teacher_replay_gate=teacher_gate,
        distance_pass=distance_gate,
        reader_transfer_pass=reader_gate,
    )
    initialization_decision = core.bridge_initialization_hypothesis_decision(
        distance_pass=distance_gate,
        reader_transfer_pass=reader_gate,
        audit=initialization_audit,
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "mode": "formal",
        "protocol": PROTOCOL,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "git_commit": manifest["git_commit"],
        "target_index": core.BRIDGE_TARGET_INDEX,
        "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID,
        "optimizer_steps": OPTIMIZER_STEPS,
        "primary_endpoint": core.BRIDGE_PRIMARY_ENDPOINT,
        "technical_gate": technical,
        "teacher_replay_statistics": teacher_stats,
        "m0_reader_statistics": m0_reader,
        "endpoint_reader_statistics": endpoint_reader,
        "endpoint_distance_statistics": distance,
        "gates": {
            "technical_gate": bool(technical["passed"]),
            "teacher_replay_gate": teacher_gate,
            "bridge_distance_gate": distance_gate,
            "endpoint_reader_transfer_gate": reader_gate,
            "bridge_diagnostic_gate": bridge_gate,
            "formal_success_gate": False,
        },
        "decision": decision,
        "secondary_solver_hypothesis_audit": initialization_audit,
        "secondary_solver_hypothesis_decision": initialization_decision,
        "phase2_allowed": False,
        "formal_success_gate": False,
        "initialization_teacher_assisted": False,
        "optimization_teacher_supervised": True,
        "answer_independent_writer_usable": False,
        "full_success_claim_allowed": False,
        "checkpoint_steps_observed": [row["optimizer_step"] for row in checkpoint_records],
        "artifacts": {
            "manifest_sha256": manifest_sha,
            "metrics_sha256": phase1a._sha256(metrics_path),
            "evaluation_rows_sha256": phase1a._sha256(rows_path),
            "endpoint_raw_sha256": phase1a._sha256(args.output_dir / "endpoint_raw.pt"),
            "endpoint_png_sha256": phase1a._sha256(args.output_dir / "endpoint_raw.png"),
            "technical_gate_sha256": phase1a._sha256(args.output_dir / "technical_gate.json"),
        },
        "wall_clock_seconds": time.monotonic() - started,
    }


def _artifact_inventory(root: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.name == "artifact_inventory.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": phase1a._sha256(path),
            }
        )
    return {
        "schema": INVENTORY_SCHEMA,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def _write_report(*, output_dir: Path, summary: Mapping[str, Any], error: Exception | None = None) -> None:
    gates = summary.get("gates", {})
    lines = [
        "# R11_new Target-01 canonical-latent bridge 运行报告",
        "",
        "## 三层结论",
        "",
        f"- 工程技术门：`{gates.get('technical_gate', summary.get('passed', False))}`",
        f"- Bridge diagnostic gate：`{gates.get('bridge_diagnostic_gate', 'not_evaluated')}`",
        "- Picture Memory 科学成功：`false`",
        "- Phase 2：仍阻塞。",
        "",
        f"- 模式：`{summary.get('mode', 'unknown')}`",
        f"- Target：`{core.BRIDGE_TARGET_INDEX}` / `{core.BRIDGE_TARGET_SEGMENT_ID}`",
        f"- Git commit：`{summary.get('git_commit', 'unknown')}`",
    ]
    if error is not None:
        lines.extend(
            [
                f"- 技术错误：`{str(error).replace(chr(10), ' ')}`",
                "- 科学解释：无；必须在 fresh root 修复重跑。",
            ]
        )
    elif summary.get("mode") == "formal":
        distance = summary["endpoint_distance_statistics"]
        lines.extend(
            [
                f"- raw step-256 MSE ratio：`{distance['mse_ratio_to_m0']}`",
                f"- raw step-256 L2 ratio：`{distance['l2_distance_ratio_to_m0']}`",
                f"- teacher-normalized RMSE：`{distance['teacher_normalized_rmse']}`",
                f"- endpoint Reader accuracy：`{summary['endpoint_reader_statistics']['accuracy']}`",
                f"- 决策：`{summary['decision']}`",
            ]
        )
    else:
        lines.append("- Preflight 不计算 bridge 结果。")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "本实验初始化仅使用固定 source；canonical teacher 仅用于稠密监督及 replay。",
            "初始化不使用答案并不等于优化不依赖 teacher；完整 writer 成功仍为 false。",
            "任何结果都不证明 shared writer、state-level memory、ID/OOD、长期递归或正式 Picture Memory 成功。",
            "",
        ]
    )
    destination = output_dir / "REPORT.md"
    temporary = destination.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, destination)


def _run(
    args: argparse.Namespace,
    *,
    parent_binding: Mapping[str, Any],
    validated_teacher_record: Mapping[str, Any],
) -> dict[str, Any]:
    determinism = r8.configure_strict_cuda_determinism(phase1a.SEED)
    r8.set_all_seeds(phase1a.SEED)
    phase1a._write_environment(args.output_dir / "environment.txt")
    phase1a._atomic_json(args.output_dir / "runtime.json", phase1a._runtime_versions())
    snapshot_bindings = phase1a._snapshot_bindings(args)
    phase1a._atomic_json(
        args.output_dir / "model_snapshot_verification_start.json",
        {
            "schema": "vision_memory.r11-new-canonical-latent-bridge-model-snapshot-start.v1",
            "bindings": snapshot_bindings,
        },
    )
    processor, _pipe, reader, oracle, source_latents, context = phase1a._load_runtime(args)
    teacher_cpu, teacher_record = _load_teacher(args.teacher)
    if teacher_record != dict(validated_teacher_record):
        raise RuntimeError("R11_new bridge teacher changed between validation and load.")
    teacher_dir = args.output_dir / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_copy = teacher_dir / "canonical_r11_target01.pt"
    shutil.copyfile(args.teacher, teacher_copy)
    if phase1a._sha256(teacher_copy) != core.BRIDGE_TEACHER_FILE_SHA256:
        raise RuntimeError("R11_new bridge teacher copy hash drifted.")
    teacher = teacher_cpu.to(device=context["updater_device"], dtype=torch.float32, copy=True)
    initial_x_t, initialization_binding = _source_only_initialization(
        source_latents=source_latents, scheduler=oracle.sampler,
        compute_device=oracle.source_latents.device, output_dir=args.output_dir,
    )
    if (
        oracle.x_T_fp32.dtype != torch.float32
        or oracle.initial_x_T_fp32.dtype != torch.float32
        or oracle.compute_dtype != source_latents.dtype
        or oracle.x_T_fp32.device != initial_x_t.device
        or oracle.initial_x_T_fp32.device != initial_x_t.device
    ):
        raise RuntimeError("Source-only initialization installation contract drifted.")
    with torch.no_grad():
        oracle.x_T_fp32.copy_(initial_x_t)
        oracle.initial_x_T_fp32.copy_(initial_x_t)
    if args.mode == "technical-preflight":
        # The preregistration locks preflight to exactly one full DreamLite
        # forward plus one backward. Keep that graph for the audit rather
        # than executing a second hidden forward in the preflight helper.
        initial_output = oracle()
    else:
        with torch.no_grad():
            initial_output = oracle()
    with torch.no_grad():
        teacher_image = decode_model_latents_unit_interval(
            oracle.vae,
            teacher.to(dtype=source_latents.dtype),
            clamp=True,
        )
    phase1a._save_image(teacher_dir / "canonical_r11_target01.png", teacher_image)
    teacher_record = {
        **teacher_record,
        "copied_path": str(teacher_copy),
        "copied_sha256": phase1a._sha256(teacher_copy),
        "decoded_image_path": str(teacher_dir / "canonical_r11_target01.png"),
        "decoded_image_sha256": phase1a._sha256(teacher_dir / "canonical_r11_target01.png"),
    }
    manifest = _manifest(
        args=args,
        context=context,
        oracle=oracle,
        initial_output=initial_output,
        parent_binding=parent_binding,
        teacher_record=teacher_record,
        snapshot_bindings=snapshot_bindings,
        determinism=determinism,
        initialization_binding=initialization_binding,
    )
    phase1a._atomic_json(args.output_dir / "manifest.json", manifest)
    eval_reader = r8.choice_reader_callable(
        reader=reader,
        processor=processor,
        reader_device=context["reader_device"],
        require_grad=False,
        deterministic_ce=True,
    )
    optimizer = torch.optim.Adam((oracle.x_T_fp32,), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if args.mode == "technical-preflight":
        summary = _preflight(
            args=args,
            oracle=oracle,
            reader=reader,
            eval_reader=eval_reader,
            target=context["target"],
            teacher=teacher,
            teacher_image=teacher_image,
            initial_output=initial_output,
            optimizer=optimizer,
            manifest=manifest,
            snapshot_bindings=snapshot_bindings,
        )
    else:
        summary = _formal(
            args=args,
            oracle=oracle,
            reader=reader,
            eval_reader=eval_reader,
            target=context["target"],
            source_latents=source_latents,
            teacher=teacher,
            teacher_image=teacher_image,
            initial_output=initial_output,
            optimizer=optimizer,
            manifest=manifest,
            snapshot_bindings=snapshot_bindings,
        )
    phase1a._atomic_json(args.output_dir / "r11_new_bridge_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    owns_output = False
    try:
        _claim_output_dir(args.output_dir)
        owns_output = True
        parent, teacher_record = _validate_args(args)
        summary = _run(args, parent_binding=parent, validated_teacher_record=teacher_record)
        technical = bool(summary.get("passed", summary.get("gates", {}).get("technical_gate", False)))
        phase1a._atomic_json(
            args.output_dir / "terminal.json",
            {
                "schema": TERMINAL_SCHEMA,
                "status": "technical_completed" if technical else "failed_technical",
                "mode": args.mode,
                "target_index": core.BRIDGE_TARGET_INDEX,
                "technical_gate": technical,
                "bridge_diagnostic_gate": summary.get("gates", {}).get("bridge_diagnostic_gate"),
                "formal_success": False,
            },
        )
        _write_report(output_dir=args.output_dir, summary=summary)
        phase1a._atomic_json(args.output_dir / "artifact_inventory.json", _artifact_inventory(args.output_dir))
    except Exception as exc:
        if owns_output:
            failure = {
                "schema": SUMMARY_SCHEMA,
                "status": "failed",
                "mode": args.mode,
                "target_index": core.BRIDGE_TARGET_INDEX,
                "formal_success_gate": False,
            }
            try:
                phase1a._atomic_json(
                    args.output_dir / "terminal.json",
                    {
                        "schema": TERMINAL_SCHEMA,
                        "status": "failed",
                        "mode": args.mode,
                        "target_index": core.BRIDGE_TARGET_INDEX,
                        "error": str(exc),
                        "formal_success": False,
                    },
                )
                _write_report(output_dir=args.output_dir, summary=failure, error=exc)
                phase1a._atomic_json(
                    args.output_dir / "artifact_inventory.json",
                    _artifact_inventory(args.output_dir),
                )
            except Exception:
                pass
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "milestone": "r11_new_bridge_completed",
                "mode": args.mode,
                "target_index": core.BRIDGE_TARGET_INDEX,
                "technical_gate": technical,
                "bridge_diagnostic_gate": summary.get("gates", {}).get("bridge_diagnostic_gate"),
                "formal_success": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if technical else 1


if __name__ == "__main__":
    raise SystemExit(main())
