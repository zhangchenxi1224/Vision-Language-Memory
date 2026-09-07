from __future__ import annotations

import inspect
import json
import math
import runpy
from pathlib import Path

import pytest
import torch

from scripts.experiments import audit_r11_new_fp32_lbfgs_trust_region as external_audit
from scripts.experiments import run_r11_new_fp32_lbfgs_trust_region as runner
from scripts.experiments import run_r11_new_fp32_trust_region as parent_runner
from vision_memory.repro import canonical_tensor_sha256
from vision_memory.training import r11_new_fp32_lbfgs_trust_region as core


ROOT = Path(__file__).resolve().parents[1]


def _accepted_pair(
    *,
    iteration: int = 1,
    previous_x: torch.Tensor | None = None,
    current_x: torch.Tensor | None = None,
    previous_gradient: torch.Tensor | None = None,
    gradient: torch.Tensor | None = None,
) -> dict[str, object]:
    previous_x = torch.tensor([0.0, 0.0], dtype=torch.float32) if previous_x is None else previous_x
    current_x = torch.tensor([1.0, 0.0], dtype=torch.float32) if current_x is None else current_x
    previous_gradient = (
        torch.tensor([0.0, 0.0], dtype=torch.float32) if previous_gradient is None else previous_gradient
    )
    gradient = torch.tensor([2.0, 0.0], dtype=torch.float32) if gradient is None else gradient
    state = core.form_curvature_pair(
        iteration=iteration,
        current_x=current_x,
        previous_x=previous_x,
        gradient=gradient,
        previous_gradient=previous_gradient,
    )
    assert state["accepted"] is True
    assert state["pair"] is not None
    return state["pair"]


def test_config_is_hash_locked_and_preserves_non_direction_contract() -> None:
    config = core.load_config()
    algorithm = config["algorithm"]
    assert config["protocol"] == core.PROTOCOL
    assert algorithm["direction_method"] == "explicit limited-memory BFGS inverse-curvature two-loop recursion"
    assert algorithm["history_size"] == core.HISTORY_SIZE == 10
    assert algorithm["curvature_relative_threshold"] == core.CURVATURE_RELATIVE_THRESHOLD == 1e-8
    assert tuple(algorithm["candidate_radii_l2_in_fixed_order"]) == core.RADII
    assert algorithm["maximum_formal_iterations"] == 128
    assert algorithm["stop_after_loss_ratio_lte"] == 0.01
    assert algorithm["no_torch_optimizer"] is True
    assert algorithm["no_gradient_clipping"] is True


def test_prpplus_parent_delivery_is_hash_bound() -> None:
    contract = core.load_config()["parent_prpplus_result"]
    assert contract["training_commit"] == "de26e3a918861ab3d3214e229469ee246b395876"
    assert contract["delivery_commit"] == "efc2a9c626cf5846d9d7970cc15d45ea85c8f9ae"
    assert contract["source_archive_sha256"] == "e6abd466e26302a052ce43df1f7a3c326c8620aa76151df4c38540bc1eb68015"
    assert contract["source_tensor_bundle_sha256"] == "80d58818e04c1acaa2968666e9fab62c48a23cd0fe5c0fb90c7d346412641f07"
    assert contract["classification"] == "prpplus_material_improvement_only"
    assert contract["iteration_count"] == 65
    assert contract["accepted_update_count"] == 64
    assert contract["final_loss_ratio_to_plateau"] == 0.02859118701878591


def test_initial_direction_exactly_reproduces_normalized_negative_gradient() -> None:
    gradient = torch.tensor([3.0, 4.0], dtype=torch.float32)
    state = core.lbfgs_direction(gradient, [], iteration=0)
    assert torch.equal(state["raw_direction"], -gradient)
    assert torch.equal(state["direction"], core.unit_vector(-gradient))
    assert state["restart_reason"] == "initial"
    assert state["history_pair_iterations_used"] == []
    assert state["initial_inverse_hessian_scale"] == 1.0
    assert state["two_loop_coefficients"] == []
    assert state["history_cleared"] is False


def test_curvature_gate_accepts_positive_and_rejects_nonpositive_pairs() -> None:
    positive = core.form_curvature_pair(
        iteration=1,
        current_x=torch.tensor([1.0, 0.0], dtype=torch.float32),
        previous_x=torch.zeros(2, dtype=torch.float32),
        gradient=torch.tensor([2.0, 1.0], dtype=torch.float32),
        previous_gradient=torch.zeros(2, dtype=torch.float32),
    )
    assert positive["accepted"] is True
    assert positive["s_dot_y"] == 2.0
    assert positive["pair"]["rho"] == 0.5

    nonpositive = core.form_curvature_pair(
        iteration=1,
        current_x=torch.tensor([1.0, 0.0], dtype=torch.float32),
        previous_x=torch.zeros(2, dtype=torch.float32),
        gradient=torch.tensor([-1.0, 0.0], dtype=torch.float32),
        previous_gradient=torch.zeros(2, dtype=torch.float32),
    )
    assert nonpositive["accepted"] is False
    assert nonpositive["reason"] == "curvature_gate_rejected"
    assert nonpositive["pair"] is None


def test_one_pair_two_loop_recursion_is_exact() -> None:
    pair = _accepted_pair(gradient=torch.tensor([2.0, 0.0], dtype=torch.float32))
    gradient = torch.tensor([2.0, 1.0], dtype=torch.float32)
    state = core.lbfgs_direction(gradient, [pair], iteration=1)
    assert torch.equal(state["raw_direction"], torch.tensor([-1.0, -0.5], dtype=torch.float32))
    assert state["history_pair_iterations_before_direction"] == [1]
    assert state["history_pair_iterations_after_direction"] == [1]
    assert state["initial_inverse_hessian_scale"] == 0.5
    assert state["restart_reason"] == "none"
    coefficient = state["two_loop_coefficients"][0]
    assert coefficient == {"pair_iteration": 1, "rho": 0.5, "alpha": 1.0, "beta": 0.0}
    assert state["directional_derivative"] < 0.0


def test_two_loop_is_deterministic_and_history_order_is_enforced() -> None:
    pair1 = _accepted_pair(iteration=1)
    pair2 = _accepted_pair(
        iteration=2,
        previous_x=torch.tensor([1.0, 0.0], dtype=torch.float32),
        current_x=torch.tensor([1.0, 1.0], dtype=torch.float32),
        previous_gradient=torch.tensor([2.0, 0.0], dtype=torch.float32),
        gradient=torch.tensor([2.0, 3.0], dtype=torch.float32),
    )
    gradient = torch.tensor([1.5, 2.5], dtype=torch.float32)
    first = core.lbfgs_direction(gradient, [pair1, pair2], iteration=2)
    second = core.lbfgs_direction(gradient, [pair1, pair2], iteration=2)
    assert torch.equal(first["raw_direction"], second["raw_direction"])
    assert torch.equal(first["direction"], second["direction"])
    assert first["two_loop_coefficients"] == second["two_loop_coefficients"]
    with pytest.raises(ValueError, match="history order"):
        core.lbfgs_direction(gradient, [pair2, pair1], iteration=2)


def test_empty_post_initial_history_uses_negative_gradient_without_false_curvature_claim() -> None:
    gradient = torch.tensor([1.0, -2.0], dtype=torch.float32)
    state = core.lbfgs_direction(gradient, [], iteration=3)
    assert torch.equal(state["raw_direction"], -gradient)
    assert state["restart_reason"] == "no_usable_history"
    assert state["history_cleared"] is False


@pytest.mark.parametrize(
    ("ratio", "updates", "expected"),
    [
        (0.009, 1, "fixed_target_trust_region_success"),
        (0.02, 128, "lbfgs_material_improvement_only"),
        (0.028, 128, "lbfgs_no_matched_budget_advantage"),
        (1.0, 0, "lbfgs_stalled"),
    ],
)
def test_classification_is_preregistered(ratio: float, updates: int, expected: str) -> None:
    summary = {
        "final_loss_ratio_to_plateau": ratio,
        "accepted_update_count": updates,
        "all_accepted_losses_strictly_monotone": True,
    }
    assert core.classify_outcome(summary, core.load_config()) == expected


def test_runner_uses_only_accepted_state_history_and_keeps_exact_budget() -> None:
    source = inspect.getsource(runner.run_search)
    assert "teacher_x_t" not in source.lower()
    assert "torch.optim" not in source
    assert ".clip_grad" not in source
    assert "history = history[-core.HISTORY_SIZE :]" in source
    assert "form_curvature_pair" in source
    assert "lbfgs_direction" in source
    config = core.load_config()
    assert config["formal_gate"]["maximum_full_chain_forward_calls"] == 3 + 6 * 128


def test_external_audit_is_bound_to_delivered_prpplus_parent() -> None:
    source = inspect.getsource(external_audit._verify_parent_bindings)
    assert "iteration-63.pt" in source
    assert "source_external_parent_audit_sha256" in source
    assert "r11_new_fp32_prpplus_trust_region" in (
        ROOT / "scripts/experiments/audit_r11_new_fp32_lbfgs_trust_region.py"
    ).read_text(encoding="utf-8")


def test_curvature_metadata_hashes_actual_tensors() -> None:
    pair_state = core.form_curvature_pair(
        iteration=1,
        current_x=torch.tensor([1.0, 0.0], dtype=torch.float32),
        previous_x=torch.zeros(2, dtype=torch.float32),
        gradient=torch.tensor([2.0, 1.0], dtype=torch.float32),
        previous_gradient=torch.zeros(2, dtype=torch.float32),
    )
    metadata = core.curvature_row_metadata(pair_state)
    assert metadata["curvature_s_fp32_sha256"] == canonical_tensor_sha256(pair_state["s"])
    assert metadata["curvature_y_fp32_sha256"] == canonical_tensor_sha256(pair_state["y"])
    assert math.isclose(metadata["curvature_cosine"], 2.0 / math.sqrt(5.0), rel_tol=1e-15)


def test_invalid_history_pair_is_rejected() -> None:
    bad_pair = dict(_accepted_pair())
    bad_pair["rho"] = 7.0
    with pytest.raises(ValueError, match="positive-curvature history"):
        core.lbfgs_direction(torch.tensor([1.0, 1.0], dtype=torch.float32), [bad_pair], iteration=1)


def test_runner_temporarily_swaps_parent_strategy(monkeypatch: pytest.MonkeyPatch) -> None:
    originals = {
        "core": parent_runner.core,
        "validate_environment": parent_runner.validate_environment,
        "run_search": parent_runner.run_search,
        "run_experiment": parent_runner.run_experiment,
    }
    observed = {}

    def fake_main(argv: list[str] | None = None) -> int:
        observed.update(
            core=parent_runner.core,
            validate_environment=parent_runner.validate_environment,
            run_search=parent_runner.run_search,
            run_experiment=parent_runner.run_experiment,
            argv=argv,
        )
        return 23

    monkeypatch.setattr(parent_runner, "main", fake_main)
    assert runner.main(["formal"]) == 23
    assert observed == {
        "core": core,
        "validate_environment": runner.validate_environment,
        "run_search": runner.run_search,
        "run_experiment": runner.run_experiment,
        "argv": ["formal"],
    }
    assert parent_runner.core is originals["core"]
    assert parent_runner.validate_environment is originals["validate_environment"]
    assert parent_runner.run_search is originals["run_search"]
    assert parent_runner.run_experiment is originals["run_experiment"]


def test_external_audit_compares_prpplus_first_step_semantically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_root = ROOT / "reports/r11-new-fp32-prpplus-trust-region-results-20260907/formal"
    output_root = tmp_path / "output"
    output_root.mkdir()
    parent_iteration = json.loads((parent_root / "iteration_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    run_iteration = {
        **parent_iteration,
        "schema": f"{core.PREFIX}-iteration-row.v1",
        "protocol": core.PROTOCOL,
        "curvature_pair_iteration": None,
        "curvature_pair_accepted": None,
        "curvature_pair_reason": "initial",
        "history_pair_iterations_before_direction": [],
        "history_pair_iterations_used": [],
        "history_pair_iterations_after_direction": [],
        "history_cleared": False,
        "initial_inverse_hessian_scale": 1.0,
        "two_loop_coefficients": [],
    }
    parent_candidates = [
        json.loads(line)
        for line in (parent_root / "candidate_metrics.jsonl").read_text(encoding="utf-8").splitlines()[:5]
    ]
    run_candidates = [
        {**row, "schema": f"{core.PREFIX}-candidate-row.v1", "protocol": core.PROTOCOL} for row in parent_candidates
    ]
    (output_root / "iteration_metrics.jsonl").write_text(
        json.dumps(run_iteration, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "candidate_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in run_candidates), encoding="utf-8"
    )
    monkeypatch.setattr(
        core,
        "load_config",
        lambda: {"parent_prpplus_result": {"source_root": str(parent_root)}},
    )
    monkeypatch.setattr(core, "audit_delivery", lambda *_args, **_kwargs: {"mode": "formal", "passed": True})
    monkeypatch.setattr(external_audit, "_verify_parent_bindings", lambda *_args, **_kwargs: {"passed": True})
    assert external_audit.audit_parent_first_step(output_root, parent_root)["passed"] is True
    run_candidates[0]["loss"] += 1e-12
    (output_root / "candidate_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in run_candidates), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="exactly reproduce"):
        external_audit.audit_parent_first_step(output_root, parent_root)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_independent_audit_reconstructs_lbfgs_curvature_and_two_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_helpers = runpy.run_path(str(ROOT / "tests/test_r11_new_fp32_trust_region.py"))
    base_helpers["_synthetic_delivery"](tmp_path, mode="formal")
    config = core.load_config()
    iteration_path = tmp_path / "iteration_metrics.jsonl"
    candidate_path = tmp_path / "candidate_metrics.jsonl"
    row = json.loads(iteration_path.read_text(encoding="utf-8").strip())
    candidates = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    for candidate in candidates:
        candidate.update(schema=f"{core.PREFIX}-candidate-row.v1", protocol=core.PROTOCOL)
    bundle_path = tmp_path / "trust_region_tensors.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    record = bundle["iterations"][0]
    pair_state = core.form_curvature_pair(
        iteration=0,
        current_x=record["current_x_T_fp32"],
        previous_x=None,
        gradient=record["gradient_fp32"],
        previous_gradient=None,
    )
    direction_state = core.lbfgs_direction(record["gradient_fp32"], [], iteration=0)
    raw_direction = direction_state["raw_direction"]
    row.update(
        schema=f"{core.PREFIX}-iteration-row.v1",
        protocol=core.PROTOCOL,
        raw_direction_fp32_sha256=canonical_tensor_sha256(raw_direction),
        **core.curvature_row_metadata(pair_state),
        **core.direction_row_metadata(direction_state),
    )
    bundle.update(schema=f"{core.PREFIX}-tensor-bundle.v1", protocol=core.PROTOCOL)
    record.update(
        curvature_s_fp32=None,
        curvature_y_fp32=None,
        inverse_hessian_raw_direction_fp32=raw_direction,
        history_pair_iterations_after_direction=[],
    )
    checkpoint_path = tmp_path / "checkpoints/iteration-00.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint.update(
        schema=f"{core.PREFIX}-checkpoint.v1",
        protocol=core.PROTOCOL,
        gradient_fp32=record["gradient_fp32"],
        curvature_s_fp32=None,
        curvature_y_fp32=None,
        inverse_hessian_raw_direction_fp32=raw_direction,
        negative_gradient_direction_fp32=record["negative_gradient_direction_fp32"],
        **core.curvature_row_metadata(pair_state),
        **core.direction_row_metadata(direction_state),
    )
    torch.save(checkpoint, checkpoint_path)

    plateau_x = bundle["plateau_x_T_fp32"]
    teacher_endpoint = bundle["teacher_endpoint_fp32"]
    plateau_endpoint = bundle["plateau_endpoint_fp32"]
    baseline = float((plateau_endpoint - teacher_endpoint).square().mean())
    current_x = checkpoint["student_x_T_fp32"]
    current_endpoint = checkpoint["endpoint_fp32"]
    gradient = record["gradient_fp32"] + (current_x - record["current_x_T_fp32"])
    pair_state_1 = core.form_curvature_pair(
        iteration=1,
        current_x=current_x,
        previous_x=record["current_x_T_fp32"],
        gradient=gradient,
        previous_gradient=record["gradient_fp32"],
    )
    assert pair_state_1["accepted"] is True
    direction_state_1 = core.lbfgs_direction(gradient, [pair_state_1["pair"]], iteration=1)
    direction_1 = direction_state_1["direction"]
    current_loss = float((current_endpoint - teacher_endpoint).square().mean())
    current_ratio = current_loss / baseline
    candidate_ratios = (0.9, 0.5, 0.7, 0.8, 0.95)
    candidates_1 = []
    candidate_tensors_1 = {}
    for radius_index, (radius, ratio_factor) in enumerate(zip(core.RADII, candidate_ratios, strict=True)):
        candidate_x = current_x + radius * direction_1
        endpoint = base_helpers["_endpoint_at_ratio"](
            teacher_endpoint,
            plateau_endpoint,
            current_ratio * ratio_factor,
        )
        candidate_loss = float((endpoint - teacher_endpoint).square().mean())
        candidate = {
            "schema": f"{core.PREFIX}-candidate-row.v1",
            "protocol": core.PROTOCOL,
            "candidate_id": core.candidate_id(1, radius_index),
            "iteration": 1,
            "radius_index": radius_index,
            "radius_l2": radius,
            "candidate_x_T_fp32_sha256": canonical_tensor_sha256(candidate_x),
            "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
            "loss": candidate_loss,
            "loss_ratio_to_plateau": candidate_loss / baseline,
            "loss_ratio_to_current": candidate_loss / current_loss,
        }
        candidates_1.append(candidate)
        candidate_tensors_1[radius_index] = {"x_T_fp32": candidate_x, "endpoint_fp32": endpoint}
    best_1 = min(candidates_1, key=lambda item: (item["loss"], item["radius_l2"]))
    selected_1 = candidate_tensors_1[best_1["radius_index"]]
    row_1 = {
        "schema": f"{core.PREFIX}-iteration-row.v1",
        "protocol": core.PROTOCOL,
        "iteration_id": core.iteration_id(1),
        "iteration": 1,
        "current_loss": current_loss,
        "current_loss_ratio_to_plateau": current_ratio,
        "current_x_T_fp32_sha256": canonical_tensor_sha256(current_x),
        "current_endpoint_fp32_sha256": canonical_tensor_sha256(current_endpoint),
        "gradient_fp32_sha256": canonical_tensor_sha256(gradient),
        "raw_direction_fp32_sha256": canonical_tensor_sha256(direction_state_1["raw_direction"]),
        "direction_fp32_sha256": canonical_tensor_sha256(direction_1),
        "gradient_norm": float(gradient.double().norm()),
        "gradient_nonzero_fraction": float((gradient != 0).double().mean()),
        **core.curvature_row_metadata(pair_state_1),
        **core.direction_row_metadata(direction_state_1),
        "selected_candidate_id": best_1["candidate_id"],
        "selected_radius_index": best_1["radius_index"],
        "selected_radius_l2": best_1["radius_l2"],
        "selected_loss": best_1["loss"],
        "selected_loss_ratio_to_plateau": best_1["loss_ratio_to_plateau"],
        "selected_x_T_fp32_sha256": best_1["candidate_x_T_fp32_sha256"],
        "selected_endpoint_fp32_sha256": best_1["endpoint_fp32_sha256"],
        "selected_candidate_residual_l2": float((selected_1["x_T_fp32"] - plateau_x).double().norm()),
        "relative_improvement": (current_loss - best_1["loss"]) / current_loss,
        "accepted": True,
    }
    bundle["iterations"][1] = {
        "current_x_T_fp32": current_x,
        "current_endpoint_fp32": current_endpoint,
        "gradient_fp32": gradient,
        "curvature_s_fp32": pair_state_1["s"],
        "curvature_y_fp32": pair_state_1["y"],
        "inverse_hessian_raw_direction_fp32": direction_state_1["raw_direction"],
        "negative_gradient_direction_fp32": direction_1,
        "history_pair_iterations_after_direction": [1],
        "candidates": candidate_tensors_1,
    }
    bundle["final_x_T_fp32"] = selected_1["x_T_fp32"]
    bundle["final_endpoint_fp32"] = selected_1["endpoint_fp32"]
    bundle["final_replay_endpoint_fp32"] = selected_1["endpoint_fp32"].clone()
    checkpoint_1 = {
        "schema": f"{core.PREFIX}-checkpoint.v1",
        "protocol": core.PROTOCOL,
        "iteration": 1,
        "student_x_T_fp32": selected_1["x_T_fp32"],
        "endpoint_fp32": selected_1["endpoint_fp32"],
        "gradient_fp32": gradient,
        "curvature_s_fp32": pair_state_1["s"],
        "curvature_y_fp32": pair_state_1["y"],
        "inverse_hessian_raw_direction_fp32": direction_state_1["raw_direction"],
        "negative_gradient_direction_fp32": direction_1,
        **core.curvature_row_metadata(pair_state_1),
        **core.direction_row_metadata(direction_state_1),
    }
    torch.save(checkpoint_1, tmp_path / "checkpoints/iteration-01.pt")
    torch.save(bundle, bundle_path)
    iteration_rows = [row, row_1]
    candidates.extend(candidates_1)
    iteration_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in iteration_rows), encoding="utf-8"
    )
    candidate_path.write_text(
        "".join(json.dumps(candidate, sort_keys=True) + "\n" for candidate in candidates),
        encoding="utf-8",
    )
    summary = core.summarize_trace(iteration_rows, candidates, config)
    classification = core.classify_outcome(summary, config)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        schema=f"{core.PREFIX}-manifest.v1",
        protocol=core.PROTOCOL,
        config_sha256=core.canonical_sha(config),
    )
    manifest["optimization_information_boundary"] = {
        "direction_method": config["algorithm"]["direction_method"],
        "direction_uses_only_current_gradient_and_accepted_history": True,
        "curvature_pairs_use_only_consecutive_accepted_x_and_gradients": True,
        "history_size": core.HISTORY_SIZE,
        "curvature_relative_threshold": core.CURVATURE_RELATIVE_THRESHOLD,
        "two_loop_cpu_float64": True,
        "matched_parent_prpplus_updates": config["algorithm"]["maximum_formal_iterations"],
    }
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    result.update(
        schema=f"{core.PREFIX}-result.v1",
        protocol=core.PROTOCOL,
        trace_summary=summary,
        classification=classification,
        fixed_target_optimization_success=classification == "fixed_target_trust_region_success",
        tensor_bundle_sha256=core.sha256_file(bundle_path),
        counters={
            "full_chain_forward_calls": 15,
            "reader_forward_calls": 0,
            "backward_calls": 2,
            "optimizer_steps": 0,
            "parameter_updates": 2,
        },
    )
    terminal = json.loads((tmp_path / "terminal.json").read_text(encoding="utf-8"))
    terminal.update(schema=f"{core.PREFIX}-terminal.v1", protocol=core.PROTOCOL, classification=classification)
    _write_json(tmp_path / "config.json", config)
    _write_json(tmp_path / "manifest.json", manifest)
    _write_json(tmp_path / "result.json", result)
    _write_json(tmp_path / "terminal.json", terminal)
    artifacts = [
        {
            "path": path.relative_to(tmp_path).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": core.sha256_file(path),
        }
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file() and path.name != "artifact_inventory.json"
    ]
    _write_json(
        tmp_path / "artifact_inventory.json",
        {"schema": f"{core.PREFIX}-inventory.v1", "artifact_count": len(artifacts), "artifacts": artifacts},
    )
    monkeypatch.setattr(core, "_audit_parent_first_step", lambda *_args, **_kwargs: None)
    audit = core.audit_delivery(tmp_path, config)
    assert audit["passed"] is True
    assert audit["classification"] == classification
    assert audit["formal_success"] is False
    assert audit["phase2_allowed"] is False
