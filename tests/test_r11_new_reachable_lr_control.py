from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tarfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.repro import canonical_tensor_sha256  # noqa: E402
from vision_memory.training import r11_new_reachable_lr_control as core  # noqa: E402


RUNNER = ROOT / "scripts/experiments/run_r11_new_reachable_lr_control.py"


def _metric(config: dict, condition: str, step: int) -> dict:
    base_learning_rate = core.condition_map(config)[condition]["base_learning_rate"]
    return {
        "schema": f"{core.PREFIX}-metrics.v1",
        "protocol": core.PROTOCOL,
        "condition": condition,
        "base_learning_rate": base_learning_rate,
        "initialization_alpha": 0.99,
        "optimizer_step": step,
        "learning_rate": core.optimizer_learning_rate(step, condition, config),
        "loss_before_step": 0.1 / step,
        "gradient_norm": 0.01,
        "gradient_nonzero_fraction": 1.0,
        "x_T_update_norm": 0.1,
        "gradient_clipping_applied": False,
        "optimizer_step_applied": True,
        "teacher_endpoint_sha256": "a" * 64,
    }


def test_config_is_immutable_fixed_target_and_diagnostic_only() -> None:
    config = core.load_config()
    assert core.sha256_file(core.CONFIG_PATH) == core.CONFIG_BYTES_SHA256
    assert core.canonical_sha(config) == core.CONFIG_CANONICAL_SHA256
    assert config["status"] == "preregistered_after_alpha099_lr0p05_overshoot_before_any_low_lr_forward"
    assert config["fixed_target_and_initialization_contract"]["regenerate_noise_or_target_forbidden"] is True
    assert config["fixed_target_and_initialization_contract"]["initialization_alpha"] == 0.99
    assert config["optimizer_contract"]["condition_order"] == ["lr-005", "lr-001"]
    assert config["deployment"]["python_executable"].endswith("/envs/vlm-r3-ngc2502/bin/python")
    assert config["technical_preflight_gate"]["total_full_chain_forward_calls"] == 5
    assert config["formal_technical_gate"]["total_full_chain_forward_calls"] == 523
    assert config["formal_technical_gate"]["backward_calls"] == 512
    assert config["formal_technical_gate"]["optimizer_steps"] == 512
    assert config["interpretation_boundaries"]["formal_success_always_false"] is True
    assert config["interpretation_boundaries"]["phase2_always_false"] is True


def test_parent_hash_bindings_match_the_delivered_run() -> None:
    config = core.load_config()
    parent = config["parent_basin_failure"]
    delivery = ROOT / "reports/r11-new-reachable-basin-control-results-20260907"
    assert core.sha256_file(delivery / "formal/result.json") == parent["source_result_sha256"]
    assert core.sha256_file(delivery / "formal/manifest.json") == parent["source_manifest_sha256"]
    assert core.sha256_file(delivery / "formal/artifact_inventory.json") == parent["source_inventory_sha256"]
    archive = delivery / "raw/r11-new-basin-7d4a845-20260907-round01.tar.gz"
    assert core.sha256_file(archive) == parent["source_archive_sha256"]
    member = (
        "r11-new-basin-7d4a845-20260907-round01-formal/target/fixed_parent_target.pt"
    )
    with tarfile.open(archive, "r:gz") as bundle:
        handle = bundle.extractfile(member)
        assert handle is not None
        assert hashlib.sha256(handle.read()).hexdigest() == parent["source_target_artifact_sha256"]


def test_both_lr_conditions_share_the_exact_alpha099_initialization() -> None:
    config = core.load_config()
    source = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
    teacher = torch.ones_like(source)
    alpha = float(config["fixed_target_and_initialization_contract"]["initialization_alpha"])
    starts = {name: core.warm_start(source, teacher, alpha) for name in core.condition_map(config)}
    assert torch.equal(starts["lr-005"], torch.full_like(source, 0.99))
    assert torch.equal(starts["lr-005"], starts["lr-001"])
    with pytest.raises(ValueError, match="alpha"):
        core.warm_start(source, teacher, 0.0)


def test_learning_rate_conditions_share_schedule_shape_but_not_scale() -> None:
    config = core.load_config()
    assert core.optimizer_learning_rate(1, "lr-005", config) == 0.005
    assert core.optimizer_learning_rate(128, "lr-001", config) == 0.001
    assert core.optimizer_learning_rate(192, "lr-005", config) == pytest.approx(0.0025)
    assert core.optimizer_learning_rate(192, "lr-001", config) == pytest.approx(0.0005)
    assert core.optimizer_learning_rate(256, "lr-005", config) == 0.0
    assert core.optimizer_learning_rate(256, "lr-001", config) == 0.0


def test_condition_gate_and_classification_are_fail_closed() -> None:
    config = core.load_config()
    passing = core.distance_statistics(mse=0.0005, m0_mse=0.1, tensor_numel=100)
    failing = core.distance_statistics(mse=0.01, m0_mse=0.1, tensor_numel=100)
    assert core.per_condition_gate(passing, technical_gate=True, config=config) is True
    assert core.per_condition_gate(passing, technical_gate=False, config=config) is False
    assert core.per_condition_gate(failing, technical_gate=True, config=config) is False
    assert core.classify_outcome({"lr-005": False, "lr-001": False}) == "learning_rate_reduction_insufficient"
    assert core.classify_outcome({"lr-005": True, "lr-001": False}) == "lr005_sufficient_lr001_underfits"
    assert core.classify_outcome({"lr-005": False, "lr-001": True}) == "conservative_lr001_required"
    assert core.classify_outcome({"lr-005": True, "lr-001": True}) == "stable_low_lr_region_both_pass"


def test_formal_metrics_recompute_per_condition() -> None:
    config = core.load_config()
    for condition in ("lr-005", "lr-001"):
        rows = [_metric(config, condition, step) for step in range(1, 257)]
        audit = core.validate_metrics(rows, config, condition)
        assert audit["optimizer_receipts"] == 256
        assert audit["optimizer_lr_schedule_exact"] is True
        assert audit["gradient_clipping_absent"] is True


@pytest.mark.parametrize("mutation", ("missing", "condition", "lr", "clip", "nan", "step"))
def test_formal_metrics_fail_closed(mutation: str) -> None:
    config = core.load_config()
    rows = [_metric(config, "lr-005", step) for step in range(1, 257)]
    if mutation == "missing":
        rows.pop()
    elif mutation == "condition":
        rows[0]["condition"] = "lr-001"
    elif mutation == "lr":
        rows[128]["learning_rate"] = 0.05
    elif mutation == "clip":
        rows[0]["gradient_clipping_applied"] = True
    elif mutation == "nan":
        rows[0]["gradient_norm"] = math.nan
    else:
        rows[1]["optimizer_step"] = 1
    with pytest.raises(ValueError):
        core.validate_metrics(rows, config, "lr-005")


def test_runner_reuses_target_resets_optimizer_and_forbids_reader() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "shutil.copyfile(parent_path, copied_path)" in source
    assert "normalized_teacher_noise" not in source and "torch.randn" not in source
    assert "reader.model.register_forward_pre_hook(forbidden_reader)" in source
    assert "Reader is forbidden" in source
    assert "gradient_clipping_applied\": False" in source
    assert "torch.optim.Adam(" in source
    assert "for condition in config[\"learning_rate_conditions\"]" in source
    assert "lr=base_learning_rate" in source
    assert "optimizer_learning_rate(update, name, config)" in source
    assert "Pinned Python environment drift." in source
    assert '"teacher_x_T_exposed_to_oracle_warm_start": True' in source
    assert "teacher_x_t.to(output.z_t.device)" not in source
    assert '"formal_success": False' in source and '"phase2_allowed": False' in source


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_delivery(root: Path, *, mode: str) -> dict:
    config = copy.deepcopy(core.load_config())
    commit = "a" * 40
    source = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
    noise = torch.ones_like(source)
    teacher_x_t = source + noise
    teacher_endpoint = torch.ones_like(source)
    tensors = {
        "source_only_x_T_init_fp32": source,
        "teacher_x_T_fp32": teacher_x_t,
        "teacher_endpoint_fp32": teacher_endpoint,
        "normalized_noise_fp32": noise,
    }
    target = {
        "schema": "legacy-fixed-parent-target",
        **tensors,
        "tensor_sha256": {name: canonical_tensor_sha256(value) for name, value in tensors.items()},
    }
    target_path = root / "target/fixed_parent_target.pt"
    target_path.parent.mkdir(parents=True)
    torch.save(target, target_path)
    config["parent_basin_failure"]["source_target_artifact_sha256"] = core.sha256_file(target_path)
    config["parent_basin_failure"]["target_tensor_sha256"] = target["tensor_sha256"]
    target_record = {
        "path": str(target_path),
        "artifact_sha256": core.sha256_file(target_path),
        "bytes": target_path.stat().st_size,
        "tensor_sha256": target["tensor_sha256"],
        "teacher_endpoint_replay_bitwise_equal": True,
        "source_parent_root": "synthetic",
    }
    result_conditions = {}
    for condition in config["learning_rate_conditions"]:
        name = condition["name"]
        base_learning_rate = float(condition["base_learning_rate"])
        alpha = float(config["fixed_target_and_initialization_contract"]["initialization_alpha"])
        condition_root = root / "conditions" / name
        checkpoints = condition_root / "checkpoints"
        checkpoints.mkdir(parents=True)
        start = core.warm_start(source, teacher_x_t, alpha)
        steps = (0,) if mode == "technical-preflight" else (0, 64, 128, 192, 256)
        checkpoint_records = []
        m0_mse = 1.0
        for step in steps:
            endpoint = torch.zeros_like(source) if step == 0 else (
                torch.full_like(source, 0.99) if step == 256 else torch.full_like(source, step / 256)
            )
            distance = core.distance_statistics(
                mse=float((endpoint - teacher_endpoint).square().mean()),
                m0_mse=m0_mse,
                tensor_numel=teacher_endpoint.numel(),
            )
            payload = {
                "schema": f"{core.PREFIX}-checkpoint.v1",
                "protocol": core.PROTOCOL,
                "condition": name,
                "base_learning_rate": base_learning_rate,
                "initialization_alpha": alpha,
                "optimizer_step": step,
                "student_x_T_fp32": start,
                "endpoint_fp32": endpoint,
                "distance": distance,
                "optimizer": {
                    "state": {} if step == 0 else {0: {"step": step}},
                    "param_groups": [{
                        "lr": (base_learning_rate if step == 0
                               else core.optimizer_learning_rate(step, name, config)),
                        "params": [0],
                    }],
                },
                "tensor_sha256": {
                    "student_x_T_fp32": canonical_tensor_sha256(start),
                    "endpoint_fp32": canonical_tensor_sha256(endpoint),
                },
            }
            checkpoint_path = checkpoints / f"step-{step:03d}.pt"
            torch.save(payload, checkpoint_path)
            checkpoint_records.append({"optimizer_step": step, "distance": distance})
        if mode == "technical-preflight":
            probe = {
                "schema": f"{core.PREFIX}-gradient-probe.v1",
                "condition": name,
                "base_learning_rate": base_learning_rate,
                "initialization_alpha": alpha,
                "loss": 1.0,
                "gradient_norm": 0.1,
                "gradient_nonzero_fraction": 1.0,
                "gradient_fp32_sha256": "b" * 64,
            }
            _write_json(condition_root / "gradient_probe.json", probe)
            result_conditions[name] = {
                "base_learning_rate": base_learning_rate,
                "initialization_alpha": alpha,
                "m0_mse": m0_mse,
                "gradient_probe": probe,
                "checkpoint_records": checkpoint_records,
            }
        else:
            rows = [_metric(config, name, step) for step in range(1, 257)]
            (condition_root / "metrics.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
            )
            final_distance = checkpoint_records[-1]["distance"]
            result_conditions[name] = {
                "base_learning_rate": base_learning_rate,
                "initialization_alpha": alpha,
                "m0_mse": m0_mse,
                "technical_audit": core.validate_metrics(rows, config, name),
                "checkpoint_records": checkpoint_records,
                "endpoint_distance": final_distance,
                "condition_gate": core.per_condition_gate(final_distance, technical_gate=True, config=config),
            }
    gates = ({name: record["condition_gate"] for name, record in result_conditions.items()}
             if mode == "formal" else {})
    result = {
        "schema": f"{core.PREFIX}-result.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": commit,
        "engineering_gate": True,
        "preflight_gate": True if mode == "technical-preflight" else None,
        "conditions": result_conditions,
        "counters": ({"full_chain_forward_calls": 5, "reader_forward_calls": 0,
                      "backward_calls": 2, "optimizer_steps": 0} if mode == "technical-preflight" else
                     {"full_chain_forward_calls": 523, "reader_forward_calls": 0,
                      "backward_calls": 512, "optimizer_steps": 512}),
        "target_record": target_record,
        "models_frozen": True,
        "only_student_x_T_trainable": True,
        "snapshots_unchanged": True,
        "condition_gates": gates,
        "classification": core.classify_outcome(gates) if mode == "formal" else None,
        "formal_success": False,
        "phase2_allowed": False,
    }
    validation = {"preflight_prerequisite": None}
    if mode == "formal":
        validation["preflight_prerequisite"] = {
            "passed": True,
            "mode": "technical-preflight",
            "git_commit": commit,
            "target_artifact_sha256": target_record["artifact_sha256"],
            "target_tensor_sha256": target["tensor_sha256"],
        }
    manifest = {
        "schema": f"{core.PREFIX}-manifest.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": commit,
        "config_sha256": core.canonical_sha(config),
        "condition_order": config["optimizer_contract"]["condition_order"],
        "validation": validation,
        "formal_success": False,
        "phase2_allowed": False,
    }
    terminal = {
        "schema": f"{core.PREFIX}-terminal.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": commit,
        "status": "technical_completed",
        "engineering_gate": True,
        "exit_code": 0,
        "formal_success": False,
        "phase2_allowed": False,
    }
    _write_json(root / "config.json", config)
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "result.json", result)
    _write_json(root / "terminal.json", terminal)
    _write_json(root / "runtime.json", {"platform": "synthetic"})
    _write_json(root / "model_snapshot_verification_start.json", {"same": True})
    _write_json(root / "model_snapshot_verification_end.json", {"same": True})
    artifacts = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
         "sha256": core.sha256_file(path)}
        for path in sorted(root.rglob("*")) if path.is_file()
    ]
    _write_json(root / "artifact_inventory.json", {
        "schema": f"{core.PREFIX}-inventory.v1", "artifact_count": len(artifacts), "artifacts": artifacts
    })
    return config


@pytest.mark.parametrize("mode", ("technical-preflight", "formal"))
def test_independent_delivery_audit_recomputes_synthetic_run(tmp_path: Path, mode: str) -> None:
    config = _synthetic_delivery(tmp_path, mode=mode)
    result = core.audit_delivery(tmp_path, config)
    assert result["passed"] is True
    assert result["mode"] == mode
    assert result["formal_success"] is False


def test_delivery_audit_rejects_target_byte_corruption(tmp_path: Path) -> None:
    config = _synthetic_delivery(tmp_path, mode="technical-preflight")
    target = tmp_path / "target/fixed_parent_target.pt"
    target.write_bytes(target.read_bytes() + b"corruption")
    with pytest.raises(ValueError):
        core.audit_delivery(tmp_path, config)
