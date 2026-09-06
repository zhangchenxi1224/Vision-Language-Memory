"""Locked contracts and independent audit for the R11_new reachable-target control."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROTOCOL = "R11-New-Reachable-Endpoint-Control-Identity-Target01"
PREFIX = "vision_memory.r11-new-reachable-endpoint-control"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/experiments/r11_new_reachable_endpoint_control_target01.json"
CONFIG_BYTES_SHA256 = "08ea3d3a5ea41605dc245a891c5c38597148c73f3b4e36c09d2df0b4a0e26ce0"
CONFIG_CANONICAL_SHA256 = "6844f978e1032c096301e5918e5facbd4331262b2521f81170deb11886be15f3"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_config() -> dict[str, Any]:
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "Reachable-control config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(canonical_sha(value) == CONFIG_CANONICAL_SHA256, "Reachable-control config canonical hash drift.")
    require(value.get("protocol") == PROTOCOL and value.get("schema") == f"{PREFIX}-config.v1",
            "Reachable-control config identity drift.")
    require(value["fixed_inputs"]["target_index"] == 1, "Target selection drift.")
    require(value["interpretation_boundaries"]["formal_success_always_false"] is True
            and value["interpretation_boundaries"]["phase2_always_false"] is True,
            "Diagnostic boundary drift.")
    return value


def normalized_teacher_noise(shape: Sequence[int], config: Mapping[str, Any]) -> torch.Tensor:
    construction = config["reachable_teacher_construction"]
    require(tuple(shape) == (1, 4, 128, 128), "Teacher-noise shape drift.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(construction["seed"]))
    raw = torch.randn(tuple(shape), generator=generator, device="cpu", dtype=torch.float32)
    rms = raw.double().square().mean().sqrt()
    require(bool(torch.isfinite(rms)) and float(rms) > 0.0, "Invalid raw teacher noise.")
    normalized = (raw.double() / rms).float()
    require(bool(torch.isfinite(normalized).all()), "Invalid normalized teacher noise.")
    return normalized


def optimizer_learning_rate(update_index: int, config: Mapping[str, Any]) -> float:
    contract = config["unchanged_optimizer_contract"]
    steps = int(contract["optimizer_steps"])
    base = float(contract["base_learning_rate"])
    require(type(update_index) is int and 1 <= update_index <= steps == 256, "Invalid optimizer update index.")
    if update_index <= 128:
        return base
    return 0.5 * base * (1.0 + math.cos(math.pi * (update_index - 128) / 128.0))


def distance_statistics(*, mse: float, m0_mse: float, tensor_numel: int) -> dict[str, float]:
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (mse, m0_mse))
            and mse >= 0.0 and m0_mse > 0.0 and type(tensor_numel) is int and tensor_numel > 0,
            "Invalid distance inputs.")
    ratio = mse / m0_mse
    return {
        "mse": mse,
        "mse_ratio_to_m0": ratio,
        "l2_distance": math.sqrt(mse * tensor_numel),
        "l2_ratio_to_m0": math.sqrt(ratio),
    }


def reachable_control_gate(distance: Mapping[str, Any], *, technical_gate: bool,
                           config: Mapping[str, Any]) -> bool:
    gate = config["reachable_control_gate"]
    values = {name: distance.get(name) for name in ("mse", "mse_ratio_to_m0", "l2_ratio_to_m0")}
    require(type(technical_gate) is bool and all(type(value) in (int, float) and math.isfinite(value)
            and value >= 0.0 for value in values.values()), "Invalid control-gate inputs.")
    return bool(technical_gate
                and values["mse"] <= gate["endpoint_absolute_mse_lte"]
                and values["mse_ratio_to_m0"] <= gate["endpoint_mse_ratio_to_m0_lte"]
                and values["l2_ratio_to_m0"] <= gate["endpoint_l2_ratio_to_m0_lte"])


def validate_metrics(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["unchanged_optimizer_contract"]
    expected_steps = int(contract["optimizer_steps"])
    require(len(rows) == expected_steps, "Formal metrics row count drift.")
    gradient_norms = []
    nonzero_fractions = []
    for expected, row in enumerate(rows, start=1):
        require(row.get("schema") == f"{PREFIX}-metrics.v1" and row.get("optimizer_step") == expected,
                "Non-contiguous or wrong-schema optimizer receipt.")
        expected_lr = optimizer_learning_rate(expected, config)
        lr = row.get("learning_rate")
        require(type(lr) in (int, float) and math.isclose(float(lr), expected_lr, rel_tol=0.0, abs_tol=1e-15),
                "Learning-rate schedule drift.")
        require(row.get("gradient_clipping_applied") is False and row.get("optimizer_step_applied") is True,
                "Optimizer/clipping contract drift.")
        grad = row.get("gradient_norm")
        fraction = row.get("gradient_nonzero_fraction")
        loss = row.get("loss_before_step")
        require(all(type(value) in (int, float) and math.isfinite(value) for value in (grad, fraction, loss))
                and grad > 0.0 and 0.0 < fraction <= 1.0 and loss >= 0.0,
                "Invalid gradient or loss receipt.")
        gradient_norms.append(float(grad))
        nonzero_fractions.append(float(fraction))
    return {
        "optimizer_receipts": len(rows),
        "optimizer_steps_contiguous": True,
        "optimizer_lr_schedule_exact": True,
        "gradient_clipping_absent": True,
        "finite_nonzero_gradient_every_step": True,
        "minimum_gradient_norm": min(gradient_norms),
        "minimum_gradient_nonzero_fraction": min(nonzero_fractions),
    }


def validate_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory = json.loads((root / "artifact_inventory.json").read_text(encoding="utf-8"))
    require(inventory.get("schema") == f"{PREFIX}-inventory.v1", "Inventory schema drift.")
    listed: set[str] = set()
    for item in inventory.get("artifacts", []):
        relative = item.get("path")
        require(isinstance(relative, str) and relative not in listed and "\\" not in relative
                and not Path(relative).is_absolute()
                and all(part not in ("", ".", "..") for part in relative.split("/")),
                "Unsafe or duplicate inventory path.")
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(), "Escaping/missing artifact.")
        require(path.stat().st_size == item.get("bytes") and sha256_file(path) == item.get("sha256"),
                "Artifact bytes/hash mismatch.")
        listed.add(relative)
    observed = {path.relative_to(root).as_posix() for path in root.rglob("*")
                if path.is_file() and path.name != "artifact_inventory.json"}
    require(observed == listed and inventory.get("artifact_count") == len(listed), "Incomplete inventory.")
    return {"artifact_count": len(listed), "listed": listed}


def audit_delivery(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    inventory = validate_inventory(root)
    listed = inventory["listed"]
    required = {"config.json", "manifest.json", "result.json", "terminal.json", "target/reachable_target.pt",
                "target/reachable_target.png", "initialization/source_only_initialization.pt",
                "model_snapshot_verification_start.json", "model_snapshot_verification_end.json"}
    require(required.issubset(listed), "Required control artifacts missing.")
    saved_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    require(saved_config == config and manifest.get("config_sha256") == canonical_sha(config), "Config binding drift.")
    require(manifest.get("protocol") == result.get("protocol") == terminal.get("protocol") == PROTOCOL,
            "Protocol drift.")
    require(manifest.get("git_commit") == result.get("git_commit") == terminal.get("git_commit"),
            "Git binding drift.")
    require(terminal.get("status") == "technical_completed" and terminal.get("engineering_gate") is True
            and terminal.get("exit_code") == 0 and terminal.get("formal_success") is False
            and terminal.get("phase2_allowed") is False, "Invalid terminal status.")
    require(result.get("engineering_gate") is True and result.get("formal_success") is False
            and result.get("phase2_allowed") is False, "Invalid result boundaries.")
    snapshots_start = json.loads((root / "model_snapshot_verification_start.json").read_text(encoding="utf-8"))
    snapshots_end = json.loads((root / "model_snapshot_verification_end.json").read_text(encoding="utf-8"))
    require(snapshots_start == snapshots_end and result.get("snapshots_unchanged") is True,
            "Model snapshot drift.")
    require(result.get("models_frozen") is True and result.get("only_student_x_T_trainable") is True,
            "Frozen/trainable evidence missing.")
    target = torch.load(root / "target/reachable_target.pt", map_location="cpu", weights_only=True)
    from vision_memory.repro import canonical_tensor_sha256

    for name in ("teacher_x_T_fp32", "teacher_endpoint_fp32", "normalized_noise_fp32"):
        tensor = target.get(name)
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                and tuple(tensor.shape) == (1, 4, 128, 128) and bool(torch.isfinite(tensor).all())
                and target["tensor_sha256"][name] == canonical_tensor_sha256(tensor), "Target tensor binding drift.")
    expected_noise = normalized_teacher_noise((1, 4, 128, 128), config)
    source_x_t = target.get("source_only_x_T_init_fp32")
    amplitude = float(config["reachable_teacher_construction"]["perturbation_rms"])
    require(isinstance(source_x_t, torch.Tensor) and source_x_t.dtype == torch.float32
            and target["tensor_sha256"]["source_only_x_T_init_fp32"] == canonical_tensor_sha256(source_x_t)
            and torch.equal(target["normalized_noise_fp32"], expected_noise)
            and torch.equal(target["teacher_x_T_fp32"], source_x_t + expected_noise * amplitude),
            "Reachable-target construction drift.")
    target_record = result.get("target_record", {})
    require(target_record.get("sha256") == sha256_file(root / "target/reachable_target.pt")
            and target_record.get("tensor_sha256") == target["tensor_sha256"]
            and target_record.get("teacher_replay_bitwise_equal") is True,
            "Target record binding drift.")
    mode = result.get("mode")
    require(mode in ("technical-preflight", "formal") and terminal.get("mode") == mode, "Mode drift.")
    if mode == "technical-preflight":
        require(result.get("preflight_gate") is True and result.get("optimizer_steps") == 0
                and result.get("backward_calls") == 1 and result.get("full_chain_forward_calls") == 4,
                "Preflight execution contract failed.")
        require(result.get("counters") == {"full_chain_forward_calls": 4, "reader_forward_calls": 0,
                                          "backward_calls": 1, "optimizer_steps": 0},
                "Preflight counters drift.")
    else:
        require({"metrics.jsonl", "checkpoints/step-000.pt", "checkpoints/step-064.pt",
                 "checkpoints/step-128.pt", "checkpoints/step-192.pt", "checkpoints/step-256.pt"}.issubset(listed),
                "Formal training artifacts missing.")
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        metric_audit = validate_metrics(rows, config)
        require(result.get("technical_audit") == metric_audit and result.get("optimizer_steps") == 256,
                "Formal metrics audit drift.")
        require(result.get("counters") == {"full_chain_forward_calls": 263, "reader_forward_calls": 0,
                                          "backward_calls": 256, "optimizer_steps": 256},
                "Formal counters drift.")
        prerequisite = manifest.get("validation", {}).get("preflight_prerequisite", {})
        require(prerequisite.get("passed") is True and prerequisite.get("mode") == "technical-preflight"
                and prerequisite.get("git_commit") == result["git_commit"]
                and prerequisite.get("target_tensor_sha256") == target["tensor_sha256"]
                and isinstance(prerequisite.get("target_artifact_sha256"), str)
                and len(prerequisite["target_artifact_sha256"]) == 64
                and target_record.get("matches_preflight") is True,
                "Formal/preflight target chain drift.")
        checkpoints = {}
        for step in config["unchanged_optimizer_contract"]["checkpoint_steps"]:
            payload = torch.load(root / f"checkpoints/step-{step:03d}.pt", map_location="cpu", weights_only=True)
            require(payload.get("optimizer_step") == step, "Checkpoint step drift.")
            for name in ("student_x_T_fp32", "endpoint_fp32"):
                tensor = payload.get(name)
                require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                        and tuple(tensor.shape) == (1, 4, 128, 128) and bool(torch.isfinite(tensor).all())
                        and payload["tensor_sha256"][name] == canonical_tensor_sha256(tensor),
                        "Checkpoint tensor binding drift.")
            checkpoints[step] = payload
        teacher = target["teacher_endpoint_fp32"]
        m0 = float((checkpoints[0]["endpoint_fp32"] - teacher).square().mean())
        endpoint_mse = float((checkpoints[256]["endpoint_fp32"] - teacher).square().mean())
        distance = distance_statistics(mse=endpoint_mse, m0_mse=m0, tensor_numel=teacher.numel())
        require(math.isclose(m0, result.get("m0_mse"), rel_tol=1e-6, abs_tol=1e-8)
                and all(math.isclose(value, result["endpoint_distance"][name], rel_tol=1e-6, abs_tol=1e-8)
                        for name, value in distance.items()), "Formal endpoint recomputation drift.")
        gate = reachable_control_gate(distance, technical_gate=True, config=config)
        require(result.get("reachable_control_gate") is gate, "Control-gate claim drift.")
    return {"passed": True, "mode": mode, "artifact_count": inventory["artifact_count"],
            "git_commit": result["git_commit"], "reachable_control_gate": result.get("reachable_control_gate", False),
            "formal_success": False, "phase2_allowed": False}
