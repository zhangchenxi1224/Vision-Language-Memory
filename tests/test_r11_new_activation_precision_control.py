from __future__ import annotations

import io
import json
import math
import sys
import tarfile
from functools import lru_cache
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.repro import canonical_tensor_sha256  # noqa: E402
from vision_memory.training import r11_new_activation_precision_control as core  # noqa: E402
from vision_memory.training import r11_new_oracle_terminal_capture as terminal_core  # noqa: E402


RUNNER = ROOT / "scripts/experiments/run_r11_new_activation_precision_control.py"
PARENT_RESULTS = ROOT / "reports/r11-new-oracle-terminal-capture-results-20260907"
PARENT_ARCHIVE = PARENT_RESULTS / "raw/r11-new-terminal-0262545-20260907-round01.tar.gz"
PARENT_PREFIX = "r11-new-terminal-0262545-20260907-round01-formal"


def test_locked_config_and_scientific_boundaries() -> None:
    config = core.load_config()
    assert core.sha256_file(core.CONFIG_PATH) == core.CONFIG_BYTES_SHA256
    assert core.canonical_sha(config) == core.CONFIG_CANONICAL_SHA256
    assert config["status"].endswith("before_any_precision_control_dreamlite_forward")
    assert tuple(config["precision_conditions"]["order"]) == core.CONDITIONS
    assert tuple(config["path_control"]["source_point_indices_from_terminal_contract"]) == core.PATH_POINT_INDICES
    assert tuple(config["gradient_control"]["signed_radii_l2"]) == core.GRADIENT_RADII
    assert config["technical_preflight_gate"]["total_full_chain_forward_calls"] == 4
    assert config["formal_technical_gate"]["total_full_chain_forward_calls"] == 54
    assert config["formal_technical_gate"]["path_rows"] == 28
    assert config["formal_technical_gate"]["gradient_rows"] == 24
    assert config["interpretation_boundaries"]["formal_success_always_false"] is True
    assert config["interpretation_boundaries"]["phase2_always_false"] is True


def test_parent_terminal_failure_is_delivered_and_hash_bound() -> None:
    config = core.load_config()
    parent = config["parent_terminal_failure"]
    assert core.sha256_file(PARENT_ARCHIVE) == parent["source_archive_sha256"]
    for name in ("result", "manifest", "terminal", "inventory"):
        filename = "artifact_inventory.json" if name == "inventory" else f"{name}.json"
        assert core.sha256_file(PARENT_RESULTS / "formal" / filename) == parent[f"source_{name}_sha256"]
    result = json.loads((PARENT_RESULTS / "formal/result.json").read_text(encoding="utf-8"))
    assert result["classification"] == "microscopic_partial_code_capture"
    assert result["scan_summary"]["strong_capture_prefix_max_requested_remaining_l2"] == 0.00003
    assert result["formal_success"] is False


@lru_cache(maxsize=1)
def _parent_bytes() -> tuple[bytes, bytes, bytes]:
    with tarfile.open(PARENT_ARCHIVE, "r:gz") as archive:
        values = []
        for relative in (
            "target/fixed_parent_target.pt",
            "parent/lr001-step256.pt",
            "terminal_path_tensors.pt",
        ):
            handle = archive.extractfile(f"{PARENT_PREFIX}/{relative}")
            assert handle is not None
            values.append(handle.read())
    return values[0], values[1], values[2]


def _parent_tensors() -> tuple[dict, dict, dict]:
    target_bytes, plateau_bytes, bundle_bytes = _parent_bytes()
    return (
        torch.load(io.BytesIO(target_bytes), map_location="cpu", weights_only=True),
        torch.load(io.BytesIO(plateau_bytes), map_location="cpu", weights_only=True),
        torch.load(io.BytesIO(bundle_bytes), map_location="cpu", weights_only=True),
    )


def test_selected_path_points_are_exact_parent_candidates() -> None:
    target, plateau, parent_bundle = _parent_tensors()
    points = terminal_core.construct_points(
        target["teacher_x_T_fp32"], plateau["student_x_T_fp32"], terminal_core.load_config()
    )
    assert set(core.PATH_POINT_INDICES) == {0, 8, 9, 11, 13, 15, 16}
    assert all(torch.equal(points[index], parent_bundle["candidates"][index]) for index in core.PATH_POINT_INDICES)


@pytest.mark.parametrize("condition", core.CONDITIONS)
@pytest.mark.parametrize("pass_name", core.PASSES)
@pytest.mark.parametrize("point_index", (0, 9, 16))
def test_path_row_identity(condition: str, pass_name: str, point_index: int) -> None:
    assert core.path_row_id(condition, pass_name, point_index) == (
        f"{condition}__path__{pass_name}__p{point_index:02d}"
    )


@pytest.mark.parametrize("condition", core.CONDITIONS)
@pytest.mark.parametrize("sign", ("plus", "minus"))
def test_gradient_row_identity(condition: str, sign: str) -> None:
    assert core.gradient_row_id(condition, 3, sign) == (f"{condition}__negative-gradient__r03__{sign}")


def test_unit_vector_and_cosine_are_fp32_and_deterministic() -> None:
    value = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(1, 1, 8, 8)
    first = core.unit_vector(value)
    second = core.unit_vector(value)
    assert torch.equal(first, second)
    assert first.dtype == torch.float32
    assert float(first.double().norm()) == pytest.approx(1.0, abs=1e-7)
    assert core.cosine(first, value) == pytest.approx(1.0, abs=1e-7)


def _path_rows(*, fp32_ratios: list[float], corrupt_bf16: bool = False) -> list[dict]:
    config = core.load_config()
    terminal_config = terminal_core.load_config()
    expected = {row["source_point_index"]: row for row in config["path_control"]["bf16_parent_expected"]}
    rows = []
    for condition in core.CONDITIONS:
        for pass_name in core.PASSES:
            order = (
                core.PATH_POINT_INDICES if pass_name == "teacher-outward" else tuple(reversed(core.PATH_POINT_INDICES))
            )
            for local_index, point_index in enumerate(order):
                binding = terminal_config["terminal_path_contract"]["point_bindings"][point_index]
                if condition == "bf16-baseline":
                    ratio = expected[point_index]["loss_ratio_to_plateau"]
                    endpoint_hash = expected[point_index]["endpoint_fp32_sha256"]
                    if corrupt_bf16 and point_index == 9:
                        endpoint_hash = "f" * 64
                else:
                    ratio = fp32_ratios[core.PATH_POINT_INDICES.index(point_index)]
                    endpoint_hash = f"{1000 + point_index:064x}"
                rows.append(
                    {
                        "schema": f"{core.PREFIX}-path-row.v1",
                        "protocol": core.PROTOCOL,
                        "row_id": core.path_row_id(condition, pass_name, point_index),
                        "condition": condition,
                        "pass": pass_name,
                        "source_point_index": point_index,
                        "requested_remaining_l2": binding["requested_remaining_l2"],
                        "x_T_fp32_sha256": binding["x_T_fp32_sha256"],
                        "endpoint_fp32_sha256": endpoint_hash,
                        "endpoint_bitwise_equal_to_condition_teacher": ratio == 0.0,
                        "loss": ratio,
                        "loss_ratio_to_plateau": ratio,
                    }
                )
    return rows


def _gradient_rows(*, fp32_best: float) -> list[dict]:
    rows = []
    for condition in core.CONDITIONS:
        analytic = -0.25 if condition == "bf16-baseline" else -0.5
        for radius_index, radius in enumerate(core.GRADIENT_RADII):
            for sign_name in ("plus", "minus"):
                if condition == "fp32-lifted" and sign_name == "plus" and radius_index == 0:
                    ratio = fp32_best
                else:
                    ratio = 1.0 + (0.1 if sign_name == "minus" else 0.05)
                rows.append(
                    {
                        "schema": f"{core.PREFIX}-gradient-row.v1",
                        "protocol": core.PROTOCOL,
                        "row_id": core.gradient_row_id(condition, radius_index, sign_name),
                        "condition": condition,
                        "radius_index": radius_index,
                        "radius_l2": radius,
                        "sign": sign_name,
                        "loss": ratio,
                        "loss_ratio_to_plateau": ratio,
                        "analytic_directional_derivative": analytic,
                        "scanned_x_T_fp32_sha256": f"{2000 + radius_index:064x}",
                        "endpoint_fp32_sha256": f"{3000 + radius_index:064x}",
                    }
                )
    return rows


@pytest.mark.parametrize(
    ("fp32_ratios", "fp32_best", "expected"),
    (
        ([0.0, 0.01, 0.02, 0.05, 0.5, 0.8, 1.0], 0.5, "fp32_restores_capture_and_gradient"),
        ([0.0, 0.01, 0.02, 0.05, 0.5, 0.8, 1.0], 0.95, "fp32_widens_capture_without_actionable_gradient"),
        ([0.0, 0.01, 0.5, 0.5, 0.5, 0.8, 1.0], 0.5, "fp32_gradient_only"),
        ([0.0, 0.01, 0.5, 0.5, 0.5, 0.8, 1.0], 0.95, "precision_lift_insufficient"),
    ),
)
def test_classification_branches(fp32_ratios: list[float], fp32_best: float, expected: str) -> None:
    config = core.load_config()
    summary = core.summarize_experiment(
        _path_rows(fp32_ratios=fp32_ratios),
        _gradient_rows(fp32_best=fp32_best),
        config,
    )
    assert core.classify_outcome(summary) == expected


def test_bf16_control_corruption_fails_closed() -> None:
    summary = core.summarize_experiment(
        _path_rows(
            fp32_ratios=[0.0, 0.01, 0.02, 0.05, 0.5, 0.8, 1.0],
            corrupt_bf16=True,
        ),
        _gradient_rows(fp32_best=0.5),
        core.load_config(),
    )
    assert core.classify_outcome(summary) == "bf16_reproduction_failure"


def test_runner_has_no_optimizer_or_reader_and_lifts_in_memory() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert ".step(" not in source and "torch.optim" not in source
    assert "clip_grad" not in source
    assert source.count("loss.backward()") == 1
    assert "oracle.unet.to(dtype=torch.float32)" in source
    assert "oracle.vae.to(dtype=torch.float32)" in source
    assert "oracle.source_latents = oracle.source_latents.float()" in source
    assert "oracle.prompt_embeds = oracle.prompt_embeds.float()" in source
    assert 'model_reloaded": False' in source
    assert "Reader is forbidden" in source
    assert '"formal_success": False' in source and '"phase2_allowed": False' in source


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _endpoint_at_ratio(teacher: torch.Tensor, plateau: torch.Tensor, ratio: float) -> torch.Tensor:
    if ratio == 0.0:
        return teacher.clone()
    if ratio == 1.0:
        return plateau.clone()
    return teacher + math.sqrt(ratio) * (plateau - teacher)


def _synthetic_delivery(root: Path, *, mode: str) -> None:
    config = core.load_config()
    terminal_config = terminal_core.load_config()
    target_bytes, plateau_bytes, _bundle_bytes = _parent_bytes()
    target_path = root / "target/fixed_parent_target.pt"
    plateau_path = root / "parent/lr001-step256.pt"
    target_path.parent.mkdir(parents=True)
    plateau_path.parent.mkdir(parents=True)
    target_path.write_bytes(target_bytes)
    plateau_path.write_bytes(plateau_bytes)
    target, plateau, parent_bundle = _parent_tensors()
    all_points = terminal_core.construct_points(
        target["teacher_x_T_fp32"], plateau["student_x_T_fp32"], terminal_config
    )
    points = {index: all_points[index] for index in core.PATH_POINT_INDICES}
    teacher = target["teacher_endpoint_fp32"]
    plateau_endpoint = plateau["endpoint_fp32"]
    baseline = float((plateau_endpoint - teacher).square().mean())
    dense_gradient = torch.linspace(-1.0, 1.0, teacher.numel(), dtype=torch.float32).reshape_as(teacher)
    dense_gradient[dense_gradient == 0] = 1e-6
    direction = core.unit_vector(-dense_gradient)
    conditions = {}
    condition_results = {}
    path_rows = []
    gradient_rows = []
    expected = {row["source_point_index"]: row for row in config["path_control"]["bf16_parent_expected"]}
    for condition in core.CONDITIONS:
        condition_teacher = teacher.clone()
        condition_plateau = plateau_endpoint.clone()
        path_endpoints = {}
        gradient_endpoints = {}
        if mode == "formal":
            fp32_ratios = dict(zip(core.PATH_POINT_INDICES, (0.0, 0.01, 0.02, 0.05, 0.5, 0.8, 1.0), strict=True))
            for pass_name in core.PASSES:
                order = (
                    core.PATH_POINT_INDICES
                    if pass_name == "teacher-outward"
                    else tuple(reversed(core.PATH_POINT_INDICES))
                )
                for point_index in order:
                    if condition == "bf16-baseline":
                        endpoint = parent_bundle["scan_endpoints"][terminal_core.scan_row_id(pass_name, point_index)]
                    else:
                        endpoint = _endpoint_at_ratio(condition_teacher, condition_plateau, fp32_ratios[point_index])
                    row_id = core.path_row_id(condition, pass_name, point_index)
                    loss = float((endpoint - condition_teacher).square().mean())
                    binding = terminal_config["terminal_path_contract"]["point_bindings"][point_index]
                    path_rows.append(
                        {
                            "schema": f"{core.PREFIX}-path-row.v1",
                            "protocol": core.PROTOCOL,
                            "row_id": row_id,
                            "condition": condition,
                            "pass": pass_name,
                            "source_point_index": point_index,
                            "requested_remaining_l2": binding["requested_remaining_l2"],
                            "x_T_fp32_sha256": binding["x_T_fp32_sha256"],
                            "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
                            "endpoint_bitwise_equal_to_condition_teacher": torch.equal(endpoint, condition_teacher),
                            "loss": loss,
                            "loss_ratio_to_plateau": loss / baseline,
                        }
                    )
                    path_endpoints[row_id] = endpoint
            analytic = float((dense_gradient.double() * direction.double()).sum())
            for radius_index, radius in enumerate(core.GRADIENT_RADII):
                for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                    ratio = 0.5 if condition == "fp32-lifted" and radius_index == 0 and sign_name == "plus" else 1.1
                    endpoint = _endpoint_at_ratio(condition_teacher, condition_plateau, ratio)
                    scanned = plateau["student_x_T_fp32"] + sign * radius * direction
                    loss = float((endpoint - condition_teacher).square().mean())
                    row_id = core.gradient_row_id(condition, radius_index, sign_name)
                    gradient_rows.append(
                        {
                            "schema": f"{core.PREFIX}-gradient-row.v1",
                            "protocol": core.PROTOCOL,
                            "row_id": row_id,
                            "condition": condition,
                            "radius_index": radius_index,
                            "radius_l2": radius,
                            "sign": sign_name,
                            "loss": loss,
                            "loss_ratio_to_plateau": loss / baseline,
                            "analytic_directional_derivative": analytic,
                            "scanned_x_T_fp32_sha256": canonical_tensor_sha256(scanned),
                            "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
                        }
                    )
                    gradient_endpoints[row_id] = endpoint
        conditions[condition] = {
            "teacher_endpoint": condition_teacher,
            "plateau_endpoint": condition_plateau,
            "gradient": dense_gradient,
            "negative_gradient_direction": direction,
            "path_endpoints": path_endpoints,
            "gradient_scan_endpoints": gradient_endpoints,
        }
        condition_results[condition] = {
            "plateau_mse": baseline,
            "teacher_endpoint_fp32_sha256": canonical_tensor_sha256(condition_teacher),
            "plateau_endpoint_fp32_sha256": canonical_tensor_sha256(condition_plateau),
            "gradient_fp32_sha256": canonical_tensor_sha256(dense_gradient),
            "direction_fp32_sha256": canonical_tensor_sha256(direction),
            "gradient_norm": float(dense_gradient.double().norm()),
            "gradient_nonzero_fraction": float((dense_gradient != 0).double().mean()),
        }
    if mode == "formal":
        for row in path_rows:
            if row["condition"] == "bf16-baseline":
                binding = expected[row["source_point_index"]]
                assert row["endpoint_fp32_sha256"] == binding["endpoint_fp32_sha256"]
                assert row["loss_ratio_to_plateau"] == pytest.approx(
                    binding["loss_ratio_to_plateau"], rel=1e-6, abs=1e-12
                )
        (root / "path_metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in path_rows),
            encoding="utf-8",
        )
        (root / "gradient_metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in gradient_rows),
            encoding="utf-8",
        )
        summary = core.summarize_experiment(path_rows, gradient_rows, config)
        classification = core.classify_outcome(summary)
        counters = {
            "full_chain_forward_calls": 54,
            "reader_forward_calls": 0,
            "backward_calls": 2,
            "optimizer_steps": 0,
        }
    else:
        summary = None
        classification = None
        counters = {
            "full_chain_forward_calls": 4,
            "reader_forward_calls": 0,
            "backward_calls": 2,
            "optimizer_steps": 0,
        }
    bundle = {
        "schema": f"{core.PREFIX}-tensor-bundle.v1",
        "protocol": core.PROTOCOL,
        "candidates": points,
        "conditions": conditions,
    }
    bundle_path = root / "precision_tensors.pt"
    torch.save(bundle, bundle_path)
    git_commit = "a" * 40
    validation = {"preflight_prerequisite": None}
    if mode == "formal":
        validation["preflight_prerequisite"] = {
            "passed": True,
            "mode": "technical-preflight",
            "git_commit": git_commit,
        }
    manifest = {
        "schema": f"{core.PREFIX}-manifest.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": git_commit,
        "config_sha256": core.canonical_sha(config),
        "validation": validation,
        "formal_success": False,
        "phase2_allowed": False,
    }
    result = {
        "schema": f"{core.PREFIX}-result.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": git_commit,
        "engineering_gate": True,
        "preflight_gate": True if mode == "technical-preflight" else None,
        "conditions": condition_results,
        "scan_summary": summary,
        "classification": classification,
        "counters": counters,
        "tensor_bundle_sha256": core.sha256_file(bundle_path),
        "models_frozen": True,
        "only_student_x_T_trainable": True,
        "snapshots_unchanged": True,
        "formal_success": False,
        "phase2_allowed": False,
    }
    terminal = {
        "schema": f"{core.PREFIX}-terminal.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": git_commit,
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
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": core.sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    ]
    _write_json(
        root / "artifact_inventory.json",
        {
            "schema": f"{core.PREFIX}-inventory.v1",
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )


@pytest.mark.parametrize("mode", ("technical-preflight", "formal"))
def test_independent_audit_recomputes_delivery(tmp_path: Path, mode: str) -> None:
    _synthetic_delivery(tmp_path, mode=mode)
    audit = core.audit_delivery(tmp_path, core.load_config())
    assert audit["passed"] is True
    assert audit["mode"] == mode
    assert audit["formal_success"] is False
    assert audit["phase2_allowed"] is False


def test_inventory_corruption_fails_closed(tmp_path: Path) -> None:
    _synthetic_delivery(tmp_path, mode="formal")
    path = tmp_path / "gradient_metrics.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes/hash mismatch"):
        core.audit_delivery(tmp_path, core.load_config())
