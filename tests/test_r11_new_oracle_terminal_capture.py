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
from vision_memory.training import r11_new_oracle_terminal_capture as core  # noqa: E402


RUNNER = ROOT / "scripts/experiments/run_r11_new_oracle_terminal_capture.py"
PARENT_RESULTS = ROOT / "reports/r11-new-direction-fidelity-results-20260907"
PARENT_ARCHIVE = (
    PARENT_RESULTS / "raw/r11-new-direction-8f696fa-20260907-round01.tar.gz"
)
PARENT_FORMAL = "r11-new-direction-8f696fa-20260907-round01-formal"


def test_locked_config_and_execution_boundaries() -> None:
    config = core.load_config()
    assert config["protocol"] == core.PROTOCOL
    assert tuple(config["terminal_path_contract"]["requested_remaining_l2_grid"]) == core.GRID
    assert tuple(config["terminal_path_contract"]["formal_passes"]) == core.PASSES
    assert config["technical_preflight_gate"]["total_full_chain_forward_calls"] == 2
    assert config["formal_technical_gate"]["total_full_chain_forward_calls"] == 35
    assert config["formal_technical_gate"]["scan_rows"] == 34
    assert config["interpretation_boundaries"]["formal_success_always_false"] is True
    assert config["interpretation_boundaries"]["phase2_always_false"] is True


def test_parent_direction_failure_is_delivered_and_hash_bound() -> None:
    config = core.load_config()
    parent = config["parent_direction_failure"]
    assert core.sha256_file(PARENT_ARCHIVE) == parent["source_archive_sha256"]
    for name in ("result", "manifest", "terminal", "inventory"):
        local = PARENT_RESULTS / "formal" / (
            "artifact_inventory.json" if name == "inventory" else f"{name}.json"
        )
        assert core.sha256_file(local) == parent[f"source_{name}_sha256"]
    result = json.loads((PARENT_RESULTS / "formal/result.json").read_text(encoding="utf-8"))
    assert result["classification"] == "plateau_oracle_not_strongly_descending"
    assert result["formal_success"] is False


@lru_cache(maxsize=1)
def _parent_bytes() -> tuple[bytes, bytes]:
    with tarfile.open(PARENT_ARCHIVE, "r:gz") as bundle:
        target = bundle.extractfile(f"{PARENT_FORMAL}/target/fixed_parent_target.pt")
        plateau = bundle.extractfile(f"{PARENT_FORMAL}/parent/lr001-step256.pt")
        assert target is not None and plateau is not None
        return target.read(), plateau.read()


def _parent_tensors() -> tuple[dict, dict]:
    target_bytes, plateau_bytes = _parent_bytes()
    target = torch.load(io.BytesIO(target_bytes), map_location="cpu", weights_only=True)
    plateau = torch.load(io.BytesIO(plateau_bytes), map_location="cpu", weights_only=True)
    return target, plateau


def test_all_preregistered_candidate_tensors_reconstruct_exactly() -> None:
    config = core.load_config()
    target, plateau = _parent_tensors()
    points = core.construct_points(
        target["teacher_x_T_fp32"], plateau["student_x_T_fp32"], config
    )
    assert len(points) == 17
    assert torch.equal(points[0], target["teacher_x_T_fp32"])
    assert torch.equal(points[-1], plateau["student_x_T_fp32"])
    assert [canonical_tensor_sha256(point) for point in points] == [
        binding["x_T_fp32_sha256"]
        for binding in config["terminal_path_contract"]["point_bindings"]
    ]


@pytest.mark.parametrize("pass_name", core.PASSES)
@pytest.mark.parametrize("point_index", (0, 7, 16))
def test_scan_row_id_is_locked(pass_name: str, point_index: int) -> None:
    assert core.scan_row_id(pass_name, point_index) == f"{pass_name}__p{point_index:02d}"


def _endpoint_hash(index: int, *, target: str, plateau: str) -> str:
    if index == 0:
        return target
    if index == 16:
        return plateau
    return f"{index:064x}"


def _rows_for_ratios(ratios: list[float]) -> list[dict]:
    config = core.load_config()
    fixed = config["fixed_parent_artifacts"]
    rows = []
    for pass_name in core.PASSES:
        order = range(17) if pass_name == "teacher-outward" else range(16, -1, -1)
        for index in order:
            binding = config["terminal_path_contract"]["point_bindings"][index]
            ratio = ratios[index]
            rows.append(
                {
                    "schema": f"{core.PREFIX}-scan-row.v1",
                    "protocol": core.PROTOCOL,
                    "row_id": core.scan_row_id(pass_name, index),
                    "pass": pass_name,
                    "point_index": index,
                    "requested_remaining_l2": binding["requested_remaining_l2"],
                    "actual_fp32_l2_to_teacher": binding["actual_fp32_l2"],
                    "bf16_l2_to_teacher": binding["bf16_l2_to_teacher"],
                    "bf16_equal_fraction_to_teacher": binding[
                        "bf16_equal_fraction_to_teacher"
                    ],
                    "x_T_fp32_sha256": binding["x_T_fp32_sha256"],
                    "endpoint_fp32_sha256": _endpoint_hash(
                        index,
                        target=fixed["teacher_endpoint_fp32_sha256"],
                        plateau=fixed["plateau_endpoint_fp32_sha256"],
                    ),
                    "endpoint_bitwise_equal_to_teacher": index == 0,
                    "loss": ratio * fixed["plateau_endpoint_mse"],
                    "loss_ratio_to_plateau": ratio,
                }
            )
    return rows


@pytest.mark.parametrize(
    ("prefix_end", "expected"),
    (
        (13, "macroscopic_terminal_capture"),
        (9, "narrow_terminal_capture"),
        (2, "microscopic_partial_code_capture"),
        (1, "bf16_exact_cell_only"),
        (0, "exact_teacher_only"),
    ),
)
def test_classification_branches(prefix_end: int, expected: str) -> None:
    ratios = [0.01 if index <= prefix_end else 0.5 for index in range(17)]
    ratios[0] = 0.0
    ratios[-1] = 1.0
    summary = core.summarize_scan(_rows_for_ratios(ratios), core.load_config())
    assert core.classify_outcome(summary, core.load_config()) == expected


def test_disconnected_strong_capture_is_anomaly() -> None:
    ratios = [0.0, 0.5, 0.01, *([0.5] * 13), 1.0]
    summary = core.summarize_scan(_rows_for_ratios(ratios), core.load_config())
    assert summary["strong_capture_disconnected"] is True
    assert core.classify_outcome(summary, core.load_config()) == "nonmonotone_capture_anomaly"


def test_duplicate_endpoint_drift_is_rejected() -> None:
    ratios = [0.01] * 16 + [1.0]
    ratios[0] = 0.0
    rows = _rows_for_ratios(ratios)
    rows[-1]["endpoint_fp32_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="Duplicate terminal endpoint mismatch"):
        core.summarize_scan(rows, core.load_config())


def test_runner_has_no_training_or_reader_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert ".backward(" not in source
    assert ".step(" not in source
    assert "optimizer_steps\": 0" in source
    assert "Reader is forbidden" in source
    assert "teacher-outward" in source and "plateau-inward" in source


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _make_delivery(root: Path, *, mode: str) -> None:
    config = core.load_config()
    target_bytes, plateau_bytes = _parent_bytes()
    target_path = root / "target/fixed_parent_target.pt"
    plateau_path = root / "parent/lr001-step256.pt"
    target_path.parent.mkdir(parents=True)
    plateau_path.parent.mkdir(parents=True)
    target_path.write_bytes(target_bytes)
    plateau_path.write_bytes(plateau_bytes)
    target, plateau = _parent_tensors()
    teacher_x_t = target["teacher_x_T_fp32"]
    teacher_endpoint = target["teacher_endpoint_fp32"]
    plateau_x_t = plateau["student_x_T_fp32"]
    plateau_endpoint = plateau["endpoint_fp32"]
    plateau_mse = float((plateau_endpoint - teacher_endpoint).square().mean())
    points = core.construct_points(teacher_x_t, plateau_x_t, config)
    rows = []
    endpoints = {}
    if mode == "technical-preflight":
        endpoints = {
            "teacher-replay": teacher_endpoint,
            "plateau-replay": plateau_endpoint,
        }
        summary = None
        classification = None
        counters = {
            "full_chain_forward_calls": 2,
            "reader_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }
    else:
        intended = [0.0, 0.0, *([0.01] * 12), 0.25, 0.5, 1.0]
        generated = []
        for index, ratio in enumerate(intended):
            if index == 0 or index == 1:
                endpoint = teacher_endpoint.clone()
            elif index == 16:
                endpoint = plateau_endpoint.clone()
            else:
                endpoint = teacher_endpoint + math.sqrt(ratio) * (
                    plateau_endpoint - teacher_endpoint
                )
            generated.append(endpoint)
        for pass_name in core.PASSES:
            order = range(17) if pass_name == "teacher-outward" else range(16, -1, -1)
            for index in order:
                endpoint = generated[index]
                loss = float((endpoint - teacher_endpoint).square().mean())
                binding = config["terminal_path_contract"]["point_bindings"][index]
                row_id = core.scan_row_id(pass_name, index)
                rows.append(
                    {
                        "schema": f"{core.PREFIX}-scan-row.v1",
                        "protocol": core.PROTOCOL,
                        "row_id": row_id,
                        "pass": pass_name,
                        "point_index": index,
                        "requested_remaining_l2": binding["requested_remaining_l2"],
                        "actual_fp32_l2_to_teacher": binding["actual_fp32_l2"],
                        "bf16_l2_to_teacher": binding["bf16_l2_to_teacher"],
                        "bf16_equal_fraction_to_teacher": binding[
                            "bf16_equal_fraction_to_teacher"
                        ],
                        "x_T_fp32_sha256": binding["x_T_fp32_sha256"],
                        "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
                        "endpoint_bitwise_equal_to_teacher": torch.equal(
                            endpoint, teacher_endpoint
                        ),
                        "loss": loss,
                        "loss_ratio_to_plateau": loss / plateau_mse,
                    }
                )
                endpoints[row_id] = endpoint
        (root / "scan_metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary = core.summarize_scan(rows, config)
        classification = core.classify_outcome(summary, config)
        counters = {
            "full_chain_forward_calls": 35,
            "reader_forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        }
    bundle = {
        "schema": f"{core.PREFIX}-tensor-bundle.v1",
        "protocol": core.PROTOCOL,
        "candidates": points,
        "scan_endpoints": endpoints,
    }
    bundle_path = root / "terminal_path_tensors.pt"
    torch.save(bundle, bundle_path)
    git_commit = "a" * 40
    manifest = {
        "schema": f"{core.PREFIX}-manifest.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": git_commit,
        "config_sha256": core.canonical_sha(config),
        "validation": {
            "preflight_prerequisite": (
                None
                if mode == "technical-preflight"
                else {"passed": True, "mode": "technical-preflight", "git_commit": git_commit}
            )
        },
    }
    result = {
        "schema": f"{core.PREFIX}-result.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": git_commit,
        "engineering_gate": True,
        "preflight_gate": True if mode == "technical-preflight" else None,
        "plateau_endpoint_mse": plateau_mse,
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
    _write_json(root / "runtime.json", {"python": sys.version})
    _write_json(root / "model_snapshot_verification_start.json", {"stable": True})
    _write_json(root / "model_snapshot_verification_end.json", {"stable": True})
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
    _make_delivery(tmp_path, mode=mode)
    audit = core.audit_delivery(tmp_path, core.load_config())
    assert audit["passed"] is True
    assert audit["mode"] == mode
    assert audit["formal_success"] is False
    assert audit["phase2_allowed"] is False


def test_audit_rejects_metric_corruption(tmp_path: Path) -> None:
    _make_delivery(tmp_path, mode="formal")
    path = tmp_path / "scan_metrics.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes/hash mismatch"):
        core.audit_delivery(tmp_path, core.load_config())
