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
from vision_memory.training import r11_new_direction_fidelity as core  # noqa: E402


RUNNER = ROOT / "scripts/experiments/run_r11_new_direction_fidelity.py"


def _optimizer(gradient: torch.Tensor, *, populated: bool) -> dict:
    state = {}
    if populated:
        state = {
            0: {
                "step": torch.tensor(256.0),
                "exp_avg": gradient.mul(0.01),
                "exp_avg_sq": gradient.square().mul(0.001).add(1e-6),
            }
        }
    return {
        "state": state,
        "param_groups": [{
            "amsgrad": False,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "lr": 0.001 if not populated else 0.0,
            "maximize": False,
            "params": [0],
        }],
    }


def test_config_is_locked_and_diagnostic_only() -> None:
    config = core.load_config()
    assert core.sha256_file(core.CONFIG_PATH) == core.CONFIG_BYTES_SHA256
    assert core.canonical_sha(config) == core.CONFIG_CANONICAL_SHA256
    assert config["status"] == "preregistered_after_low_lr_failure_before_any_direction_scan_forward"
    assert tuple(config["fixed_target_and_anchor_contract"]["anchor_order"]) == core.ANCHORS
    assert tuple(config["direction_scan_contract"]["directions"]) == core.DIRECTIONS
    assert config["formal_technical_gate"]["scan_rows"] == 112
    assert config["formal_technical_gate"]["total_full_chain_forward_calls"] == 115
    assert config["interpretation_boundaries"]["formal_success_always_false"] is True
    assert config["interpretation_boundaries"]["phase2_always_false"] is True


def test_parent_hashes_bind_to_delivered_low_lr_run() -> None:
    config = core.load_config()
    parent = config["parent_low_lr_failure"]
    delivery = ROOT / "reports/r11-new-reachable-lr-control-results-20260907"
    formal = delivery / "formal"
    for name in ("result", "manifest", "terminal", "inventory"):
        filename = "artifact_inventory.json" if name == "inventory" else f"{name}.json"
        assert core.sha256_file(formal / filename) == parent[f"source_{name}_sha256"]
    archive = delivery / "raw/r11-new-lr-af00d04-20260907-round02.tar.gz"
    assert core.sha256_file(archive) == parent["source_archive_sha256"]
    prefix = "r11-new-lr-af00d04-20260907-round02-formal"
    members = {
        f"{prefix}/target/fixed_parent_target.pt": parent["source_target_artifact_sha256"],
        f"{prefix}/conditions/lr-001/checkpoints/step-000.pt": parent["source_alpha099_checkpoint_sha256"],
        f"{prefix}/conditions/lr-001/checkpoints/step-256.pt": parent["source_plateau_checkpoint_sha256"],
    }
    with tarfile.open(archive, "r:gz") as bundle:
        for member, expected in members.items():
            handle = bundle.extractfile(member)
            assert handle is not None
            assert hashlib.sha256(handle.read()).hexdigest() == expected


def test_unit_directions_and_orthogonal_control_are_deterministic() -> None:
    gradient = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(1, 1, 8, 8)
    residual = torch.ones_like(gradient)
    unit = core.unit_vector(gradient)
    assert float(unit.double().norm()) == pytest.approx(1.0, abs=1e-7)
    first = core.orthogonal_control(gradient, residual, seed=20260907)
    second = core.orthogonal_control(gradient, residual, seed=20260907)
    assert torch.equal(first, second)
    assert abs(core.cosine(first, gradient)) <= 1e-6
    assert abs(core.cosine(first, residual)) <= 1e-6


def test_adam_direction_reconstructs_empty_and_step256_states() -> None:
    gradient = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(1, 1, 8, 8)
    first = core.adam_preconditioned_direction(gradient, _optimizer(gradient, populated=False))
    expected_sign = core.unit_vector(-gradient.sign())
    assert core.cosine(first, expected_sign) > 0.999
    plateau = core.adam_preconditioned_direction(gradient, _optimizer(gradient, populated=True))
    assert float(plateau.double().norm()) == pytest.approx(1.0, abs=1e-7)
    assert bool(torch.isfinite(plateau).all())


def _minimal_rows(config: dict) -> list[dict]:
    rows = []
    for anchor in core.ANCHORS:
        for direction in core.DIRECTIONS:
            analytic = -0.1 if direction != "deterministic-orthogonal-control" else 0.0
            for index, radius in enumerate(config["direction_scan_contract"]["signed_radii_l2"]):
                for sign in ("plus", "minus"):
                    rows.append({
                        "schema": f"{core.PREFIX}-scan-row.v1",
                        "protocol": core.PROTOCOL,
                        "row_id": core.scan_row_id(anchor, direction, index, sign),
                        "anchor": anchor,
                        "direction": direction,
                        "radius_index": index,
                        "radius_l2": radius,
                        "sign": sign,
                        "loss": 1.0,
                        "loss_ratio_to_anchor": 1.0,
                        "analytic_directional_derivative": analytic,
                        "endpoint_fp32_sha256": "a" * 64,
                        "bf16_equal_fraction_to_teacher": 0.5,
                        "bf16_l2_to_teacher": 1.0,
                        "fp32_l2_to_teacher": 1.0,
                    })
    return rows


def test_scan_grid_and_central_differences_are_recomputed() -> None:
    config = core.load_config()
    summary = core.summarize_scan(_minimal_rows(config), config)
    assert set(summary) == set(core.ANCHORS)
    assert set(summary["lr001-raw256"]) == set(core.DIRECTIONS)
    assert summary["lr001-raw256"]["negative-autograd"]["best_positive_loss_ratio"] == 1.0
    assert summary["lr001-raw256"]["negative-autograd"]["central_finite_difference"] == [0.0] * 7


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "radius", "nan"))
def test_scan_grid_fails_closed(mutation: str) -> None:
    config = core.load_config()
    rows = _minimal_rows(config)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "radius":
        rows[0]["radius_l2"] = 9.0
    else:
        rows[0]["loss"] = math.nan
    with pytest.raises(ValueError):
        core.summarize_scan(rows, config)


def _classification_summary(*, negative_gradient: bool, adam: bool, oracle_ratio: float) -> dict:
    direction = lambda descent, ratio: {  # noqa: E731
        "meaningful_descent": descent,
        "best_positive_loss_ratio": ratio,
    }
    anchors = {}
    for anchor in core.ANCHORS:
        anchors[anchor] = {
            "negative-autograd": direction(negative_gradient, 0.5 if negative_gradient else 1.0),
            "negative-adam-preconditioned": direction(adam, 0.5 if adam else 1.0),
            "teacher-residual": direction(oracle_ratio <= 0.9, oracle_ratio),
            "deterministic-orthogonal-control": direction(False, 1.0),
        }
    return anchors


@pytest.mark.parametrize(
    ("negative_gradient", "adam", "oracle_ratio", "expected"),
    (
        (True, True, 0.01, "plateau_negative_gradient_and_adam_descent"),
        (True, False, 0.01, "plateau_negative_gradient_only"),
        (False, False, 0.01, "plateau_teacher_residual_only"),
        (False, True, 0.01, "adam_only_or_random_control_anomaly"),
        (True, True, 0.2, "plateau_oracle_not_strongly_descending"),
    ),
)
def test_classification_is_preregistered(negative_gradient: bool, adam: bool,
                                         oracle_ratio: float, expected: str) -> None:
    config = core.load_config()
    summary = _classification_summary(
        negative_gradient=negative_gradient, adam=adam, oracle_ratio=oracle_ratio
    )
    assert core.classify_outcome(summary, config) == expected


def test_runner_has_no_optimizer_and_scans_both_signs() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "optimizer.step(" not in source and "torch.optim." not in source
    assert "shutil.copyfile(source, destination)" in source
    assert "for sign_name, sign in ((\"plus\", 1.0), (\"minus\", -1.0))" in source
    assert "reader.model.register_forward_pre_hook(forbidden_reader)" in source
    assert "Reader is forbidden" in source
    assert '"teacher_x_T_exposed_to_oracle_direction": True' in source
    assert '"formal_success": False' in source and '"phase2_allowed": False' in source


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_delivery(root: Path, *, mode: str) -> dict:
    config = copy.deepcopy(core.load_config())
    parent = config["parent_low_lr_failure"]
    shape = (1, 4, 128, 128)
    source = torch.zeros(shape, dtype=torch.float32)
    noise = torch.ones(shape, dtype=torch.float32)
    teacher_x_t = source + noise
    teacher_endpoint = torch.ones(shape, dtype=torch.float32)
    target_tensors = {
        "source_only_x_T_init_fp32": source,
        "teacher_x_T_fp32": teacher_x_t,
        "teacher_endpoint_fp32": teacher_endpoint,
        "normalized_noise_fp32": noise,
    }
    target = {
        **target_tensors,
        "tensor_sha256": {name: canonical_tensor_sha256(value) for name, value in target_tensors.items()},
    }
    target_path = root / "target/fixed_parent_target.pt"
    target_path.parent.mkdir(parents=True)
    torch.save(target, target_path)
    alpha_x_t = torch.full(shape, 0.99, dtype=torch.float32)
    plateau_x_t = torch.full(shape, 0.80, dtype=torch.float32)
    alpha_endpoint = torch.full(shape, 0.90, dtype=torch.float32)
    plateau_endpoint = torch.full(shape, 0.80, dtype=torch.float32)
    gradient = torch.linspace(-1.0, 1.0, math.prod(shape), dtype=torch.float32).reshape(shape)
    checkpoints = {
        "alpha099-start": {
            "student_x_T_fp32": alpha_x_t,
            "endpoint_fp32": alpha_endpoint,
            "optimizer": _optimizer(gradient, populated=False),
        },
        "lr001-raw256": {
            "student_x_T_fp32": plateau_x_t,
            "endpoint_fp32": plateau_endpoint,
            "optimizer": _optimizer(gradient, populated=True),
        },
    }
    checkpoint_paths = {
        "alpha099-start": root / "parent/alpha099-step000.pt",
        "lr001-raw256": root / "parent/lr001-step256.pt",
    }
    for name, checkpoint in checkpoints.items():
        checkpoint["tensor_sha256"] = {
            key: canonical_tensor_sha256(checkpoint[key])
            for key in ("student_x_T_fp32", "endpoint_fp32")
        }
        checkpoint_paths[name].parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_paths[name])
    parent["source_target_artifact_sha256"] = core.sha256_file(target_path)
    parent["target_tensor_sha256"] = target["tensor_sha256"]
    parent["source_alpha099_checkpoint_sha256"] = core.sha256_file(checkpoint_paths["alpha099-start"])
    parent["source_plateau_checkpoint_sha256"] = core.sha256_file(checkpoint_paths["lr001-raw256"])
    parent["alpha099_tensor_sha256"] = checkpoints["alpha099-start"]["tensor_sha256"]
    parent["plateau_tensor_sha256"] = checkpoints["lr001-raw256"]["tensor_sha256"]
    anchors = {}
    gradients = {}
    baselines = {}
    directions = {}
    anchor_records = {}
    for name in core.ANCHORS:
        checkpoint = checkpoints[name]
        anchor = checkpoint["student_x_T_fp32"]
        residual = teacher_x_t - anchor
        anchor_gradient = gradient if name == "alpha099-start" else gradient.mul(0.5)
        anchor_directions = {
            "negative-autograd": core.unit_vector(-anchor_gradient),
            "negative-adam-preconditioned": core.adam_preconditioned_direction(
                anchor_gradient, checkpoint["optimizer"]
            ),
            "teacher-residual": core.unit_vector(residual),
            "deterministic-orthogonal-control": core.orthogonal_control(
                anchor_gradient, residual, seed=config["direction_scan_contract"]["orthogonal_control_seed"]
            ),
        }
        anchors[name] = anchor
        gradients[name] = anchor_gradient
        baselines[name] = checkpoint["endpoint_fp32"]
        directions[name] = anchor_directions
        anchor_records[name] = {
            "baseline_mse": float((baselines[name] - teacher_endpoint).square().mean()),
            "gradient_fp32_sha256": canonical_tensor_sha256(anchor_gradient),
        }
    rows = []
    endpoints = {}
    if mode == "formal":
        for anchor_name in core.ANCHORS:
            baseline_mse = anchor_records[anchor_name]["baseline_mse"]
            for direction_name in core.DIRECTIONS:
                direction = directions[anchor_name][direction_name]
                analytic = float((gradients[anchor_name].double() * direction.double()).sum())
                for index, radius in enumerate(config["direction_scan_contract"]["signed_radii_l2"]):
                    for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                        row_id = core.scan_row_id(anchor_name, direction_name, index, sign_name)
                        scanned_x_t = anchors[anchor_name] + sign * radius * direction
                        endpoint = baselines[anchor_name]
                        loss = float((endpoint - teacher_endpoint).square().mean())
                        row = {
                            "schema": f"{core.PREFIX}-scan-row.v1",
                            "protocol": core.PROTOCOL,
                            "row_id": row_id,
                            "anchor": anchor_name,
                            "direction": direction_name,
                            "radius_index": index,
                            "radius_l2": radius,
                            "sign": sign_name,
                            "loss": loss,
                            "loss_ratio_to_anchor": loss / baseline_mse,
                            "analytic_directional_derivative": analytic,
                            "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
                            **core.quantization_statistics(scanned_x_t, teacher_x_t),
                        }
                        rows.append(row)
                        endpoints[row_id] = endpoint
        (root / "scan_metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
    bundle = {
        "schema": f"{core.PREFIX}-tensor-bundle.v1",
        "protocol": core.PROTOCOL,
        "anchors": anchors,
        "gradients": gradients,
        "baseline_endpoints": baselines,
        "directions": directions,
        "scan_endpoints": endpoints,
    }
    bundle_path = root / "scan_tensors.pt"
    torch.save(bundle, bundle_path)
    scan_summary = core.summarize_scan(rows, config) if mode == "formal" else None
    classification = core.classify_outcome(scan_summary, config) if mode == "formal" else None
    counters = ({"full_chain_forward_calls": 3, "reader_forward_calls": 0,
                 "backward_calls": 2, "optimizer_steps": 0} if mode == "technical-preflight" else
                {"full_chain_forward_calls": 115, "reader_forward_calls": 0,
                 "backward_calls": 2, "optimizer_steps": 0})
    result = {
        "schema": f"{core.PREFIX}-result.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": "a" * 40,
        "engineering_gate": True,
        "preflight_gate": True if mode == "technical-preflight" else None,
        "anchors": anchor_records,
        "scan_summary": scan_summary,
        "classification": classification,
        "counters": counters,
        "tensor_bundle_sha256": core.sha256_file(bundle_path),
        "models_frozen": True,
        "only_student_x_T_trainable": True,
        "snapshots_unchanged": True,
        "formal_success": False,
        "phase2_allowed": False,
    }
    validation = {"preflight_prerequisite": None}
    if mode == "formal":
        validation["preflight_prerequisite"] = {
            "passed": True, "mode": "technical-preflight", "git_commit": "a" * 40
        }
    manifest = {
        "schema": f"{core.PREFIX}-manifest.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": "a" * 40,
        "config_sha256": core.canonical_sha(config),
        "validation": validation,
        "formal_success": False,
        "phase2_allowed": False,
    }
    terminal = {
        "schema": f"{core.PREFIX}-terminal.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": "a" * 40,
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
    _write_json(root / "runtime.json", {"synthetic": True})
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
def test_independent_audit_recomputes_synthetic_delivery(tmp_path: Path, mode: str) -> None:
    config = _synthetic_delivery(tmp_path, mode=mode)
    audit = core.audit_delivery(tmp_path, config)
    assert audit["passed"] is True
    assert audit["mode"] == mode
    assert audit["formal_success"] is False


def test_audit_rejects_corrupted_parent_checkpoint(tmp_path: Path) -> None:
    config = _synthetic_delivery(tmp_path, mode="technical-preflight")
    path = tmp_path / "parent/lr001-step256.pt"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError):
        core.audit_delivery(tmp_path, config)
