"""Locked contracts and independent audit for the R11_new reachable low-LR control."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROTOCOL = "R11-New-Reachable-LR-Control-Target01"
PREFIX = "vision_memory.r11-new-reachable-lr-control"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/experiments/r11_new_reachable_lr_control_target01.json"
CONFIG_BYTES_SHA256 = "2d87a5f3f669409d7e8e5b45029e5a5b75229f11353049a4a2a07c23b44c13cd"
CONFIG_CANONICAL_SHA256 = "5ca9dd8882ef397777f8fb8f4f5b8c070506a7452814c8a4f55ebbe891b7b339"
TARGET_TENSOR_NAMES = (
    "source_only_x_T_init_fp32",
    "teacher_x_T_fp32",
    "teacher_endpoint_fp32",
    "normalized_noise_fp32",
)


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
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "Reachable low-LR config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(canonical_sha(value) == CONFIG_CANONICAL_SHA256, "Reachable low-LR config canonical hash drift.")
    require(value.get("protocol") == PROTOCOL and value.get("schema") == f"{PREFIX}-config.v1",
            "Reachable low-LR config identity drift.")
    conditions = value.get("learning_rate_conditions")
    require(isinstance(conditions, list)
            and [item.get("name") for item in conditions] == ["lr-005", "lr-001"]
            and [item.get("base_learning_rate") for item in conditions] == [0.005, 0.001],
            "Learning-rate conditions drift.")
    require(value["fixed_target_and_initialization_contract"]["regenerate_noise_or_target_forbidden"] is True
            and value["fixed_target_and_initialization_contract"]["initialization_alpha"] == 0.99,
            "Fixed-target contract drift.")
    require(value["interpretation_boundaries"]["formal_success_always_false"] is True
            and value["interpretation_boundaries"]["phase2_always_false"] is True,
            "Diagnostic boundary drift.")
    return value


def condition_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    conditions = {item["name"]: item for item in config["learning_rate_conditions"]}
    require(tuple(conditions) == ("lr-005", "lr-001"), "Condition order drift.")
    return conditions


def warm_start(source: torch.Tensor, teacher: torch.Tensor, alpha: float) -> torch.Tensor:
    require(source.dtype == teacher.dtype == torch.float32 and source.shape == teacher.shape,
            "Invalid warm-start tensors.")
    require(type(alpha) is float and 0.0 < alpha < 1.0, "Invalid warm-start alpha.")
    return source + alpha * (teacher - source)


def optimizer_learning_rate(update_index: int, condition_name: str, config: Mapping[str, Any]) -> float:
    contract = config["optimizer_contract"]
    steps = int(contract["optimizer_steps_per_condition"])
    base = float(condition_map(config)[condition_name]["base_learning_rate"])
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


def per_condition_gate(distance: Mapping[str, Any], *, technical_gate: bool,
                       config: Mapping[str, Any]) -> bool:
    gate = config["per_condition_gate"]
    values = {name: distance.get(name) for name in ("mse", "mse_ratio_to_m0", "l2_ratio_to_m0")}
    require(type(technical_gate) is bool and all(type(value) in (int, float) and math.isfinite(value)
            and value >= 0.0 for value in values.values()), "Invalid low-LR gate inputs.")
    return bool(technical_gate
                and values["mse"] <= gate["endpoint_absolute_mse_lte"]
                and values["mse_ratio_to_m0"] <= gate["endpoint_mse_ratio_to_m0_lte"]
                and values["l2_ratio_to_m0"] <= gate["endpoint_l2_ratio_to_m0_lte"])


def classify_outcome(gates: Mapping[str, Any]) -> str:
    lr005 = gates.get("lr-005")
    lr001 = gates.get("lr-001")
    require(type(lr005) is bool and type(lr001) is bool, "Missing condition gates.")
    if lr005 and lr001:
        return "stable_low_lr_region_both_pass"
    if lr005:
        return "lr005_sufficient_lr001_underfits"
    if lr001:
        return "conservative_lr001_required"
    return "learning_rate_reduction_insufficient"


def validate_metrics(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
                     condition_name: str) -> dict[str, Any]:
    expected_steps = int(config["optimizer_contract"]["optimizer_steps_per_condition"])
    require(condition_name in condition_map(config) and len(rows) == expected_steps,
            "Formal condition or metrics row count drift.")
    gradient_norms = []
    nonzero_fractions = []
    for expected, row in enumerate(rows, start=1):
        require(row.get("schema") == f"{PREFIX}-metrics.v1"
                and row.get("condition") == condition_name
                and row.get("optimizer_step") == expected,
                "Non-contiguous or wrong-schema optimizer receipt.")
        expected_lr = optimizer_learning_rate(expected, condition_name, config)
        lr = row.get("learning_rate")
        require(type(lr) in (int, float) and math.isclose(float(lr), expected_lr, rel_tol=0.0, abs_tol=1e-15),
                "Learning-rate schedule drift.")
        require(row.get("base_learning_rate") == condition_map(config)[condition_name]["base_learning_rate"]
                and row.get("initialization_alpha") == 0.99
                and row.get("gradient_clipping_applied") is False and row.get("optimizer_step_applied") is True,
                "Optimizer/clipping contract drift.")
        values = (row.get("gradient_norm"), row.get("gradient_nonzero_fraction"), row.get("loss_before_step"))
        require(all(type(value) in (int, float) and math.isfinite(value) for value in values)
                and values[0] > 0.0 and 0.0 < values[1] <= 1.0 and values[2] >= 0.0,
                "Invalid gradient or loss receipt.")
        gradient_norms.append(float(values[0]))
        nonzero_fractions.append(float(values[1]))
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


def validate_fixed_target(root: Path, result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    from vision_memory.repro import canonical_tensor_sha256

    path = root / "target/fixed_parent_target.pt"
    parent = config["parent_basin_failure"]
    require(sha256_file(path) == parent["source_target_artifact_sha256"], "Parent target artifact drift.")
    target = torch.load(path, map_location="cpu", weights_only=True)
    require(target.get("tensor_sha256") == parent["target_tensor_sha256"], "Parent target hash map drift.")
    for name in TARGET_TENSOR_NAMES:
        tensor = target.get(name)
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                and tuple(tensor.shape) == (1, 4, 128, 128) and bool(torch.isfinite(tensor).all())
                and canonical_tensor_sha256(tensor) == parent["target_tensor_sha256"][name],
                f"Parent target tensor drift: {name}.")
    source = target["source_only_x_T_init_fp32"]
    teacher = target["teacher_x_T_fp32"]
    noise = target["normalized_noise_fp32"]
    require(torch.equal(teacher, source + noise), "Stored parent target formula drift.")
    record = result.get("target_record", {})
    require(record.get("artifact_sha256") == parent["source_target_artifact_sha256"]
            and record.get("tensor_sha256") == parent["target_tensor_sha256"]
            and record.get("teacher_endpoint_replay_bitwise_equal") is True,
            "Target result binding drift.")
    return target


def _validate_checkpoint(path: Path, *, condition_name: str, step: int,
                         target: Mapping[str, torch.Tensor], base_learning_rate: float,
                         config: Mapping[str, Any],
                         m0_mse: float | None = None) -> tuple[dict[str, Any], dict[str, float]]:
    from vision_memory.repro import canonical_tensor_sha256

    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("schema") == f"{PREFIX}-checkpoint.v1"
            and payload.get("protocol") == PROTOCOL
            and payload.get("condition") == condition_name
            and payload.get("base_learning_rate") == base_learning_rate
            and payload.get("initialization_alpha")
            == config["fixed_target_and_initialization_contract"]["initialization_alpha"]
            and payload.get("optimizer_step") == step, "Checkpoint identity drift.")
    for name in ("student_x_T_fp32", "endpoint_fp32"):
        tensor = payload.get(name)
        require(isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float32
                and tuple(tensor.shape) == (1, 4, 128, 128) and bool(torch.isfinite(tensor).all())
                and payload["tensor_sha256"][name] == canonical_tensor_sha256(tensor),
                "Checkpoint tensor drift.")
    optimizer = payload.get("optimizer")
    require(isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict)
            and isinstance(optimizer.get("param_groups"), list) and len(optimizer["param_groups"]) == 1,
            "Checkpoint optimizer receipt missing.")
    expected_lr = (base_learning_rate if step == 0
                   else optimizer_learning_rate(step, condition_name, config))
    require(math.isclose(float(optimizer["param_groups"][0].get("lr")), expected_lr,
                         rel_tol=0.0, abs_tol=1e-15)
            and (optimizer["state"] == {} if step == 0 else bool(optimizer["state"])),
            "Checkpoint optimizer state/LR drift.")
    if step == 0:
        alpha = float(config["fixed_target_and_initialization_contract"]["initialization_alpha"])
        expected = warm_start(target["source_only_x_T_init_fp32"], target["teacher_x_T_fp32"], alpha)
        require(torch.equal(payload["student_x_T_fp32"], expected), "Warm-start checkpoint drift.")
    endpoint_mse = float((payload["endpoint_fp32"] - target["teacher_endpoint_fp32"]).square().mean())
    baseline = endpoint_mse if m0_mse is None else m0_mse
    distance = distance_statistics(mse=endpoint_mse, m0_mse=baseline,
                                   tensor_numel=target["teacher_endpoint_fp32"].numel())
    require(all(math.isclose(distance[name], payload["distance"][name], rel_tol=1e-6, abs_tol=1e-8)
                for name in distance), "Checkpoint distance drift.")
    return payload, distance


def audit_delivery(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    inventory = validate_inventory(root)
    listed = inventory["listed"]
    required = {
        "config.json", "manifest.json", "result.json", "terminal.json", "runtime.json",
        "target/fixed_parent_target.pt", "model_snapshot_verification_start.json",
        "model_snapshot_verification_end.json",
    }
    require(required.issubset(listed), "Required low-LR-control artifacts missing.")
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
    target = validate_fixed_target(root, result, config)
    mode = result.get("mode")
    require(mode in ("technical-preflight", "formal") and terminal.get("mode") == mode, "Mode drift.")
    conditions = condition_map(config)
    result_conditions = result.get("conditions", {})
    require(set(result_conditions) == set(conditions)
            and manifest.get("condition_order") == config["optimizer_contract"]["condition_order"],
            "Result conditions or declared execution order drift.")
    initial_payloads = {}
    for name, condition in conditions.items():
        relative = f"conditions/{name}/checkpoints/step-000.pt"
        require(relative in listed, f"Missing initial checkpoint: {name}.")
        initial_payload, initial_distance = _validate_checkpoint(
            root / relative, condition_name=name, step=0, target=target,
            base_learning_rate=float(condition["base_learning_rate"]), config=config
        )
        initial_payloads[name] = initial_payload
        record = result_conditions[name]
        require(record.get("base_learning_rate") == condition["base_learning_rate"]
                and record.get("initialization_alpha") == 0.99
                and math.isclose(record.get("m0_mse"), initial_distance["mse"], rel_tol=1e-6, abs_tol=1e-8),
                "Initial condition result drift.")
    require(len({payload["tensor_sha256"]["student_x_T_fp32"] for payload in initial_payloads.values()}) == 1
            and len({payload["tensor_sha256"]["endpoint_fp32"] for payload in initial_payloads.values()}) == 1
            and len({result_conditions[name]["m0_mse"] for name in conditions}) == 1,
            "Condition initial states/endpoints are not bitwise identical.")
    if mode == "technical-preflight":
        require(result.get("preflight_gate") is True
                and result.get("counters") == {"full_chain_forward_calls": 5, "reader_forward_calls": 0,
                                                "backward_calls": 2, "optimizer_steps": 0},
                "Preflight execution contract failed.")
        probe_records = []
        for name, condition in conditions.items():
            probe_path = root / f"conditions/{name}/gradient_probe.json"
            require(probe_path.relative_to(root).as_posix() in listed, "Missing gradient probe.")
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            require(probe == result_conditions[name]["gradient_probe"]
                    and probe.get("base_learning_rate") == condition["base_learning_rate"]
                    and probe.get("initialization_alpha") == 0.99
                    and isinstance(probe.get("gradient_fp32_sha256"), str)
                    and len(probe["gradient_fp32_sha256"]) == 64
                    and probe.get("gradient_norm", 0.0) > 0.0
                    and probe.get("gradient_nonzero_fraction", 0.0)
                    >= config["technical_preflight_gate"]["gradient_nonzero_fraction_gte"],
                    "Gradient probe drift.")
            probe_records.append(probe)
        probe_keys = ("loss", "gradient_norm", "gradient_nonzero_fraction", "gradient_fp32_sha256")
        require(all(tuple(probe[key] for key in probe_keys)
                    == tuple(probe_records[0][key] for key in probe_keys)
                    for probe in probe_records[1:]), "Condition gradient probes are not bitwise identical.")
    else:
        require(result.get("counters") == {"full_chain_forward_calls": 523, "reader_forward_calls": 0,
                                           "backward_calls": 512, "optimizer_steps": 512},
                "Formal counters drift.")
        prerequisite = manifest.get("validation", {}).get("preflight_prerequisite", {})
        require(prerequisite.get("passed") is True and prerequisite.get("mode") == "technical-preflight"
                and prerequisite.get("git_commit") == result["git_commit"]
                and prerequisite.get("target_artifact_sha256")
                == config["parent_basin_failure"]["source_target_artifact_sha256"]
                and prerequisite.get("target_tensor_sha256")
                == config["parent_basin_failure"]["target_tensor_sha256"],
                "Formal/preflight target chain drift.")
        gates = {}
        for name, condition in conditions.items():
            record = result_conditions[name]
            metrics_path = root / f"conditions/{name}/metrics.jsonl"
            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            metric_audit = validate_metrics(rows, config, name)
            require(record.get("technical_audit") == metric_audit, "Condition metric audit drift.")
            m0_mse = float(record["m0_mse"])
            checkpoint_steps = []
            final_distance = None
            for step in config["optimizer_contract"]["checkpoint_steps"]:
                relative = f"conditions/{name}/checkpoints/step-{step:03d}.pt"
                require(relative in listed, "Formal checkpoint missing.")
                _, final_distance = _validate_checkpoint(
                    root / relative, condition_name=name, step=step, target=target,
                    base_learning_rate=float(condition["base_learning_rate"]), config=config, m0_mse=m0_mse
                )
                checkpoint_steps.append(step)
            require(checkpoint_steps == config["optimizer_contract"]["checkpoint_steps"]
                    and all(math.isclose(final_distance[key], record["endpoint_distance"][key],
                                         rel_tol=1e-6, abs_tol=1e-8) for key in final_distance),
                    "Condition endpoint distance drift.")
            gate = per_condition_gate(final_distance, technical_gate=True, config=config)
            require(record.get("condition_gate") is gate, "Condition gate claim drift.")
            gates[name] = gate
        classification = classify_outcome(gates)
        require(result.get("condition_gates") == gates and result.get("classification") == classification,
                "Low-LR classification drift.")
    return {
        "passed": True,
        "mode": mode,
        "artifact_count": inventory["artifact_count"],
        "git_commit": result["git_commit"],
        "condition_gates": result.get("condition_gates", {}),
        "classification": result.get("classification"),
        "formal_success": False,
        "phase2_allowed": False,
    }
