from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import r11_new_reachable_control as core  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402


RUNNER = ROOT / "scripts/experiments/run_r11_new_reachable_control.py"


def _metric(config: dict, step: int) -> dict:
    return {
        "schema": f"{core.PREFIX}-metrics.v1",
        "protocol": core.PROTOCOL,
        "optimizer_step": step,
        "learning_rate": core.optimizer_learning_rate(step, config),
        "loss_before_step": 0.1 / step,
        "gradient_norm": 0.01,
        "gradient_nonzero_fraction": 1.0,
        "x_T_update_norm": 0.1,
        "gradient_clipping_applied": False,
        "optimizer_step_applied": True,
        "teacher_endpoint_sha256": "a" * 64,
    }


def test_config_is_immutable_and_diagnostic_only() -> None:
    config = core.load_config()
    assert core.sha256_file(core.CONFIG_PATH) == core.CONFIG_BYTES_SHA256
    assert core.canonical_sha(config) == core.CONFIG_CANONICAL_SHA256
    assert config["status"] == "preregistered_after_teacher47_d0_before_any_reachable_target_forward"
    assert config["reachable_teacher_construction"]["no_candidate_search_or_amplitude_retry"] is True
    assert config["interpretation_boundaries"]["formal_success_always_false"] is True
    assert config["interpretation_boundaries"]["phase2_always_false"] is True


def test_teacher_noise_is_local_deterministic_and_unit_rms() -> None:
    config = core.load_config()
    torch.manual_seed(1)
    first = core.normalized_teacher_noise((1, 4, 128, 128), config)
    torch.manual_seed(999)
    second = core.normalized_teacher_noise((1, 4, 128, 128), config)
    assert torch.equal(first, second)
    assert first.dtype == torch.float32 and first.device.type == "cpu"
    assert math.isclose(float(first.double().square().mean().sqrt()), 1.0, rel_tol=0.0, abs_tol=1e-8)


def test_teacher_noise_rejects_shape_drift() -> None:
    with pytest.raises(ValueError, match="shape drift"):
        core.normalized_teacher_noise((1, 4, 64, 64), core.load_config())


def test_learning_rate_exact_parent_schedule() -> None:
    config = core.load_config()
    assert core.optimizer_learning_rate(1, config) == 0.05
    assert core.optimizer_learning_rate(128, config) == 0.05
    assert 0.0 < core.optimizer_learning_rate(129, config) < 0.05
    assert core.optimizer_learning_rate(192, config) == pytest.approx(0.025)
    assert core.optimizer_learning_rate(256, config) == 0.0
    with pytest.raises(ValueError, match="update index"):
        core.optimizer_learning_rate(0, config)


def test_distance_and_control_gate_require_all_three_thresholds() -> None:
    config = core.load_config()
    passing = core.distance_statistics(mse=0.0005, m0_mse=0.1, tensor_numel=100)
    assert core.reachable_control_gate(passing, technical_gate=True, config=config) is True
    assert core.reachable_control_gate(passing, technical_gate=False, config=config) is False
    absolute_fail = core.distance_statistics(mse=0.0011, m0_mse=1.0, tensor_numel=100)
    assert core.reachable_control_gate(absolute_fail, technical_gate=True, config=config) is False
    ratio_fail = core.distance_statistics(mse=0.0005, m0_mse=0.01, tensor_numel=100)
    assert core.reachable_control_gate(ratio_fail, technical_gate=True, config=config) is False


def test_formal_metrics_recompute_exactly() -> None:
    config = core.load_config()
    rows = [_metric(config, step) for step in range(1, 257)]
    audit = core.validate_metrics(rows, config)
    assert audit["optimizer_receipts"] == 256
    assert audit["optimizer_lr_schedule_exact"] is True
    assert audit["gradient_clipping_absent"] is True


@pytest.mark.parametrize("mutation", ("missing", "lr", "clip", "nan", "step"))
def test_formal_metrics_fail_closed(mutation: str) -> None:
    config = core.load_config()
    rows = [_metric(config, step) for step in range(1, 257)]
    if mutation == "missing":
        rows.pop()
    elif mutation == "lr":
        rows[128]["learning_rate"] = 0.05
    elif mutation == "clip":
        rows[0]["gradient_clipping_applied"] = True
    elif mutation == "nan":
        rows[0]["gradient_norm"] = math.nan
    else:
        rows[1]["optimizer_step"] = 1
    with pytest.raises(ValueError):
        core.validate_metrics(rows, config)


def test_inventory_rejects_escape(tmp_path: Path) -> None:
    inventory = {
        "schema": f"{core.PREFIX}-inventory.v1",
        "artifact_count": 1,
        "artifacts": [{"path": "../escape", "bytes": 0, "sha256": "0" * 64}],
    }
    (tmp_path / "artifact_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe or duplicate"):
        core.validate_inventory(tmp_path)


def test_runner_keeps_teacher_parameter_out_of_loss_and_reader_out_of_run() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "torch.optim.Adam((oracle.x_T_fp32,)" in source
    assert "reader.model.register_forward_pre_hook(forbidden_reader)" in source
    assert "Reader is forbidden" in source
    assert "gradient_clipping_applied\": False" in source
    assert ".backward()" in source
    assert "teacher_endpoint.to(output.z_t.device)" in source
    assert "teacher_x_t.to(output.z_t.device)" not in source
    assert '"formal_success": False' in source and '"phase2_allowed": False' in source


def test_runner_restores_student_before_any_optimization() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    last_teacher_install = source.rindex("oracle.x_T_fp32.copy_(teacher_x_t.to(oracle.x_T_fp32.device))")
    student_restore = source.index("oracle.x_T_fp32.copy_(initial_x_t)", last_teacher_install)
    optimizer = source.index("torch.optim.Adam((oracle.x_T_fp32,)", student_restore)
    training_loop = source.index("for update in range", optimizer)
    assert last_teacher_install < student_restore < optimizer < training_loop
    assert source.count("oracle.x_T_fp32.copy_(teacher_x_t.to(oracle.x_T_fp32.device))") == 2


def test_config_mutation_is_detected_even_when_re_signed_in_memory() -> None:
    config = copy.deepcopy(core.load_config())
    config["reachable_control_gate"]["endpoint_mse_ratio_to_m0_lte"] = 1.0
    assert core.canonical_sha(config) != core.CONFIG_CANONICAL_SHA256


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_delivery(root: Path, *, mode: str) -> None:
    config = core.load_config()
    commit = "a" * 40
    source = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
    noise = core.normalized_teacher_noise(source.shape, config)
    teacher_x_t = source + noise
    teacher_endpoint = torch.ones_like(source)
    tensors = {
        "source_only_x_T_init_fp32": source,
        "teacher_x_T_fp32": teacher_x_t,
        "teacher_endpoint_fp32": teacher_endpoint,
        "normalized_noise_fp32": noise,
    }
    target = {
        "schema": f"{core.PREFIX}-target.v1",
        "protocol": core.PROTOCOL,
        "construction": config["reachable_teacher_construction"],
        **tensors,
        "tensor_sha256": {name: canonical_tensor_sha256(value) for name, value in tensors.items()},
    }
    target_path = root / "target/reachable_target.pt"
    target_path.parent.mkdir(parents=True)
    torch.save(target, target_path)
    (root / "target/reachable_target.png").write_bytes(b"synthetic-png")
    initialization = root / "initialization/source_only_initialization.pt"
    initialization.parent.mkdir(parents=True)
    torch.save({"source": source}, initialization)
    target_record = {
        "path": str(target_path),
        "sha256": core.sha256_file(target_path),
        "bytes": target_path.stat().st_size,
        "tensor_sha256": target["tensor_sha256"],
        "teacher_replay_bitwise_equal": True,
        "teacher_perturbation_rms": 1.0,
        "m0_mse": 1.0,
        "matches_preflight": None if mode == "technical-preflight" else True,
    }
    validation: dict = {"preflight_prerequisite": None}
    result = {
        "schema": f"{core.PREFIX}-result.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": commit,
        "engineering_gate": True,
        "m0_mse": 1.0,
        "construction_gate": True,
        "target_record": target_record,
        "models_frozen": True,
        "only_student_x_T_trainable": True,
        "snapshots_unchanged": True,
        "formal_success": False,
        "phase2_allowed": False,
    }
    if mode == "technical-preflight":
        result.update(
            preflight_gate=True,
            full_chain_forward_calls=4,
            backward_calls=1,
            optimizer_steps=0,
            counters={"full_chain_forward_calls": 4, "reader_forward_calls": 0,
                      "backward_calls": 1, "optimizer_steps": 0},
            reachable_control_gate=False,
        )
    else:
        validation["preflight_prerequisite"] = {
            "passed": True,
            "mode": "technical-preflight",
            "git_commit": commit,
            "target_tensor_sha256": target["tensor_sha256"],
            "target_artifact_sha256": "b" * 64,
        }
        rows = [_metric(config, step) for step in range(1, 257)]
        (root / "metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        checkpoints = root / "checkpoints"
        checkpoints.mkdir()
        for step in (0, 64, 128, 192, 256):
            endpoint = torch.zeros_like(source) if step == 0 else (
                torch.full_like(source, 0.99) if step == 256 else torch.full_like(source, step / 256)
            )
            checkpoint_tensors = {"student_x_T_fp32": source.clone(), "endpoint_fp32": endpoint}
            torch.save(
                {
                    "schema": f"{core.PREFIX}-checkpoint.v1",
                    "optimizer_step": step,
                    **checkpoint_tensors,
                    "tensor_sha256": {
                        name: canonical_tensor_sha256(value) for name, value in checkpoint_tensors.items()
                    },
                },
                checkpoints / f"step-{step:03d}.pt",
            )
        distance = core.distance_statistics(mse=float((teacher_endpoint - 0.99).square().mean()),
                                            m0_mse=1.0, tensor_numel=teacher_endpoint.numel())
        result.update(
            optimizer_steps=256,
            counters={"full_chain_forward_calls": 263, "reader_forward_calls": 0,
                      "backward_calls": 256, "optimizer_steps": 256},
            technical_audit=core.validate_metrics(rows, config),
            endpoint_distance=distance,
            reachable_control_gate=core.reachable_control_gate(distance, technical_gate=True, config=config),
        )
    manifest = {
        "schema": f"{core.PREFIX}-manifest.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": commit,
        "config_sha256": core.canonical_sha(config),
        "validation": validation,
        "target": target_record,
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


@pytest.mark.parametrize("mode", ("technical-preflight", "formal"))
def test_independent_delivery_audit_recomputes_synthetic_run(tmp_path: Path, mode: str) -> None:
    _synthetic_delivery(tmp_path, mode=mode)
    result = core.audit_delivery(tmp_path, core.load_config())
    assert result["passed"] is True
    assert result["mode"] == mode
    assert result["formal_success"] is False
