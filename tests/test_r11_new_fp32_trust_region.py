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
from vision_memory.training import r11_new_activation_precision_control as precision_core  # noqa: E402
from vision_memory.training import r11_new_fp32_trust_region as core  # noqa: E402


RUNNER = ROOT / "scripts/experiments/run_r11_new_fp32_trust_region.py"
PARENT_RESULTS = ROOT / "reports/r11-new-activation-precision-control-results-20260907"
PARENT_ARCHIVE = PARENT_RESULTS / "raw/r11-new-precision-2753094-20260907-round01.tar.gz"
PARENT_PREFIX = "r11-new-precision-2753094-20260907-round01-formal"


def test_config_is_locked_and_scientifically_bounded() -> None:
    config = core.load_config()
    assert core.sha256_file(core.CONFIG_PATH) == core.CONFIG_BYTES_SHA256
    assert core.canonical_sha(config) == core.CONFIG_CANONICAL_SHA256
    assert config["status"].endswith("before_any_trust_region_dreamlite_forward")
    assert tuple(config["algorithm"]["candidate_radii_l2_in_fixed_order"]) == core.RADII
    assert config["algorithm"]["maximum_formal_iterations"] == 32
    assert config["technical_preflight_gate"]["full_chain_forward_calls"] == 9
    assert config["formal_gate"]["maximum_full_chain_forward_calls"] == 195
    assert config["interpretation_boundaries"]["teacher_x_T_not_used_by_optimizer"] is True
    assert config["interpretation_boundaries"]["formal_picture_memory_success_always_false"] is True
    assert config["interpretation_boundaries"]["phase2_always_false"] is True


def test_parent_precision_result_is_delivered_and_hash_bound() -> None:
    config = core.load_config()
    parent = config["parent_precision_result"]
    assert core.sha256_file(PARENT_ARCHIVE) == parent["source_archive_sha256"]
    for name in ("result", "manifest", "terminal", "inventory"):
        filename = "artifact_inventory.json" if name == "inventory" else f"{name}.json"
        assert core.sha256_file(PARENT_RESULTS / "formal" / filename) == parent[f"source_{name}_sha256"]
    assert (
        core.sha256_file(PARENT_RESULTS / "formal/gradient_metrics.jsonl") == parent["source_gradient_metrics_sha256"]
    )


@pytest.mark.parametrize("iteration", (0, 1, 31))
def test_iteration_and_candidate_ids(iteration: int) -> None:
    assert core.iteration_id(iteration) == f"iteration-{iteration:02d}"
    assert core.candidate_id(iteration, 2) == f"iteration-{iteration:02d}__radius-02"


def _trace(final_ratio: float, *, accepted: bool = True) -> tuple[list[dict], list[dict]]:
    current_loss = 1.0
    losses = [1.5, 1.4, 1.6, 1.7, 1.8]
    selected_index = 1
    losses[selected_index] = final_ratio
    candidates = []
    for radius_index, (radius, loss) in enumerate(zip(core.RADII, losses, strict=True)):
        candidates.append(
            {
                "schema": f"{core.PREFIX}-candidate-row.v1",
                "protocol": core.PROTOCOL,
                "candidate_id": core.candidate_id(0, radius_index),
                "iteration": 0,
                "radius_index": radius_index,
                "radius_l2": radius,
                "candidate_x_T_fp32_sha256": f"{100 + radius_index:064x}",
                "endpoint_fp32_sha256": f"{200 + radius_index:064x}",
                "loss": loss,
                "loss_ratio_to_plateau": loss,
                "loss_ratio_to_current": loss,
            }
        )
    relative = (current_loss - final_ratio) / current_loss
    row = {
        "schema": f"{core.PREFIX}-iteration-row.v1",
        "protocol": core.PROTOCOL,
        "iteration_id": core.iteration_id(0),
        "iteration": 0,
        "current_loss": current_loss,
        "current_loss_ratio_to_plateau": 1.0,
        "current_x_T_fp32_sha256": "1" * 64,
        "current_endpoint_fp32_sha256": "2" * 64,
        "gradient_fp32_sha256": "3" * 64,
        "direction_fp32_sha256": "4" * 64,
        "gradient_norm": 1.0,
        "gradient_nonzero_fraction": 1.0,
        "analytic_directional_derivative": -1.0,
        "selected_candidate_id": core.candidate_id(0, selected_index),
        "selected_radius_index": selected_index,
        "selected_radius_l2": core.RADII[selected_index],
        "selected_loss": final_ratio,
        "selected_loss_ratio_to_plateau": final_ratio,
        "selected_x_T_fp32_sha256": f"{100 + selected_index:064x}",
        "selected_endpoint_fp32_sha256": f"{200 + selected_index:064x}",
        "selected_candidate_residual_l2": core.RADII[selected_index],
        "relative_improvement": relative,
        "accepted": accepted,
    }
    return [row], candidates


@pytest.mark.parametrize(
    ("ratio", "expected"),
    (
        (0.005, "fixed_target_trust_region_success"),
        (0.05, "strong_capture_reached_only"),
        (0.5, "monotone_progress_above_capture"),
    ),
)
def test_preregistered_classification_branches(ratio: float, expected: str) -> None:
    iterations, candidates = _trace(ratio)
    summary = core.summarize_trace(iterations, candidates, core.load_config())
    assert core.classify_outcome(summary, core.load_config()) == expected


def test_stalled_classification() -> None:
    iterations, candidates = _trace(1.01, accepted=False)
    summary = core.summarize_trace(iterations, candidates, core.load_config())
    assert summary["accepted_update_count"] == 0
    assert summary["stop_reason"] == "no_acceptable_candidate"
    assert core.classify_outcome(summary, core.load_config()) == "trust_region_stalled"


def test_trace_rejects_missing_or_wrong_selection() -> None:
    iterations, candidates = _trace(0.5)
    with pytest.raises(ValueError, match="candidate count"):
        core.summarize_trace(iterations, candidates[:-1], core.load_config())
    iterations, candidates = _trace(0.5)
    iterations[0]["selected_radius_index"] = 0
    with pytest.raises(ValueError):
        core.summarize_trace(iterations, candidates, core.load_config())


def test_runner_uses_fixed_normalized_update_without_optimizer_or_reader() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert ".step(" not in source and "torch.optim" not in source
    assert "clip_grad" not in source
    assert "direction = core.unit_vector(-gradient_cpu)" in source
    assert "candidate_x = current_x + radius * direction" in source
    assert 'min(local_records, key=lambda row: (row["loss"], row["radius_l2"]))' in source
    assert "Reader is forbidden" in source
    search_source = source[source.index("def run_search(") : source.index("def write_report(")]
    assert "teacher_x_T" not in search_source
    assert '"formal_success": False' in source and '"phase2_allowed": False' in source


@lru_cache(maxsize=1)
def _parent_bytes() -> tuple[bytes, bytes, bytes]:
    with tarfile.open(PARENT_ARCHIVE, "r:gz") as archive:
        payloads = []
        for relative in (
            "target/fixed_parent_target.pt",
            "parent/lr001-step256.pt",
            "precision_tensors.pt",
        ):
            handle = archive.extractfile(f"{PARENT_PREFIX}/{relative}")
            assert handle is not None
            payloads.append(handle.read())
    return payloads[0], payloads[1], payloads[2]


def _parent_tensors() -> tuple[dict, dict, dict]:
    target_bytes, plateau_bytes, precision_bytes = _parent_bytes()
    return (
        torch.load(io.BytesIO(target_bytes), map_location="cpu", weights_only=True),
        torch.load(io.BytesIO(plateau_bytes), map_location="cpu", weights_only=True),
        torch.load(io.BytesIO(precision_bytes), map_location="cpu", weights_only=True),
    )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _endpoint_at_ratio(teacher: torch.Tensor, plateau: torch.Tensor, ratio: float) -> torch.Tensor:
    return teacher + math.sqrt(ratio) * (plateau - teacher)


def _synthetic_delivery(root: Path, *, mode: str) -> None:
    config = core.load_config()
    target_bytes, plateau_bytes, _precision_bytes = _parent_bytes()
    target_path = root / "target/fixed_parent_target.pt"
    plateau_path = root / "parent/lr001-step256.pt"
    target_path.parent.mkdir(parents=True)
    plateau_path.parent.mkdir(parents=True)
    target_path.write_bytes(target_bytes)
    plateau_path.write_bytes(plateau_bytes)
    target, plateau, precision = _parent_tensors()
    fp32 = precision["conditions"]["fp32-lifted"]
    teacher = fp32["teacher_endpoint"]
    plateau_endpoint = fp32["plateau_endpoint"]
    gradient = fp32["gradient"]
    direction = fp32["negative_gradient_direction"]
    plateau_x = plateau["student_x_T_fp32"]
    baseline = float((plateau_endpoint - teacher).square().mean())
    parent_endpoint_by_radius = {}
    for parent_index, radius in enumerate(precision_core.GRADIENT_RADII):
        if radius in core.RADII:
            parent_endpoint_by_radius[radius] = fp32["gradient_scan_endpoints"][
                precision_core.gradient_row_id("fp32-lifted", parent_index, "plus")
            ]
    synthetic_ratios = {0.03: 0.7, 0.003: 0.95}
    candidate_rows = []
    candidate_tensors = {}
    for radius_index, radius in enumerate(core.RADII):
        candidate_x = plateau_x + radius * direction
        endpoint = parent_endpoint_by_radius.get(radius)
        if endpoint is None:
            endpoint = _endpoint_at_ratio(teacher, plateau_endpoint, synthetic_ratios[radius])
        loss = float((endpoint - teacher).square().mean())
        row = {
            "schema": f"{core.PREFIX}-candidate-row.v1",
            "protocol": core.PROTOCOL,
            "candidate_id": core.candidate_id(0, radius_index),
            "iteration": 0,
            "radius_index": radius_index,
            "radius_l2": radius,
            "candidate_x_T_fp32_sha256": canonical_tensor_sha256(candidate_x),
            "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
            "loss": loss,
            "loss_ratio_to_plateau": loss / baseline,
            "loss_ratio_to_current": loss / baseline,
        }
        candidate_rows.append(row)
        candidate_tensors[radius_index] = {
            "x_T_fp32": candidate_x,
            "endpoint_fp32": endpoint,
        }
    best = min(candidate_rows, key=lambda row: (row["loss"], row["radius_l2"]))
    selected = candidate_tensors[best["radius_index"]]
    relative_improvement = (baseline - best["loss"]) / baseline
    iteration_row = {
        "schema": f"{core.PREFIX}-iteration-row.v1",
        "protocol": core.PROTOCOL,
        "iteration_id": core.iteration_id(0),
        "iteration": 0,
        "current_loss": baseline,
        "current_loss_ratio_to_plateau": 1.0,
        "current_x_T_fp32_sha256": canonical_tensor_sha256(plateau_x),
        "current_endpoint_fp32_sha256": canonical_tensor_sha256(plateau_endpoint),
        "gradient_fp32_sha256": canonical_tensor_sha256(gradient),
        "direction_fp32_sha256": canonical_tensor_sha256(direction),
        "gradient_norm": float(gradient.double().norm()),
        "gradient_nonzero_fraction": float((gradient != 0).double().mean()),
        "analytic_directional_derivative": float((gradient.double() * direction.double()).sum()),
        "selected_candidate_id": best["candidate_id"],
        "selected_radius_index": best["radius_index"],
        "selected_radius_l2": best["radius_l2"],
        "selected_loss": best["loss"],
        "selected_loss_ratio_to_plateau": best["loss_ratio_to_plateau"],
        "selected_x_T_fp32_sha256": best["candidate_x_T_fp32_sha256"],
        "selected_endpoint_fp32_sha256": best["endpoint_fp32_sha256"],
        "selected_candidate_residual_l2": float((selected["x_T_fp32"] - plateau_x).double().norm()),
        "relative_improvement": relative_improvement,
        "accepted": True,
    }
    iteration_rows = [iteration_row]
    (root / "iteration_metrics.jsonl").write_text(json.dumps(iteration_row, sort_keys=True) + "\n", encoding="utf-8")
    (root / "candidate_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate_rows),
        encoding="utf-8",
    )
    checkpoint = {
        "schema": f"{core.PREFIX}-checkpoint.v1",
        "protocol": core.PROTOCOL,
        "iteration": 0,
        "student_x_T_fp32": selected["x_T_fp32"],
        "endpoint_fp32": selected["endpoint_fp32"],
    }
    checkpoint_path = root / "checkpoints/iteration-00.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(checkpoint, checkpoint_path)
    bundle = {
        "schema": f"{core.PREFIX}-tensor-bundle.v1",
        "protocol": core.PROTOCOL,
        "bf16_teacher_replay_endpoint_fp32": target["teacher_endpoint_fp32"],
        "plateau_x_T_fp32": plateau_x,
        "teacher_endpoint_fp32": teacher,
        "plateau_endpoint_fp32": plateau_endpoint,
        "iterations": {
            0: {
                "current_x_T_fp32": plateau_x,
                "current_endpoint_fp32": plateau_endpoint,
                "gradient_fp32": gradient,
                "negative_gradient_direction_fp32": direction,
                "candidates": candidate_tensors,
            }
        },
        "final_x_T_fp32": selected["x_T_fp32"],
        "final_endpoint_fp32": selected["endpoint_fp32"],
        "final_replay_endpoint_fp32": selected["endpoint_fp32"].clone(),
    }
    bundle_path = root / "trust_region_tensors.pt"
    torch.save(bundle, bundle_path)
    summary = core.summarize_trace(iteration_rows, candidate_rows, config)
    classification = core.classify_outcome(summary, config) if mode == "formal" else None
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
    counters = {
        "full_chain_forward_calls": 9,
        "reader_forward_calls": 0,
        "backward_calls": 1,
        "optimizer_steps": 0,
        "parameter_updates": 1,
    }
    result = {
        "schema": f"{core.PREFIX}-result.v1",
        "protocol": core.PROTOCOL,
        "mode": mode,
        "git_commit": git_commit,
        "engineering_gate": True,
        "preflight_gate": True if mode == "technical-preflight" else None,
        "fp32_lift_audit": {"passed": True},
        "trace_summary": summary,
        "classification": classification,
        "fixed_target_optimization_success": classification == "fixed_target_trust_region_success",
        "final_replay_bitwise_equal": True,
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
    path = tmp_path / "candidate_metrics.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes/hash mismatch"):
        core.audit_delivery(tmp_path, core.load_config())
