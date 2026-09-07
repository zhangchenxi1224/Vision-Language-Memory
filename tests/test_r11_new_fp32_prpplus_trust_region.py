from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest
import torch

from scripts.experiments import audit_r11_new_fp32_prpplus_trust_region as external_audit
from scripts.experiments import run_r11_new_fp32_prpplus_trust_region as runner
from scripts.experiments import run_r11_new_fp32_trust_region as parent_runner
from vision_memory.repro import canonical_tensor_sha256
from vision_memory.training import r11_new_fp32_prpplus_trust_region as core
from vision_memory.training import r11_new_fp32_trust_region_horizon128 as parent_core


ROOT = Path(__file__).resolve().parents[1]


def test_config_is_hash_locked_and_matched_budget() -> None:
    config = core.load_config()
    parent = parent_core.load_config()
    assert core.sha256_file(core.CONFIG_PATH) == core.CONFIG_BYTES_SHA256
    assert core.canonical_sha(config) == core.CONFIG_CANONICAL_SHA256
    assert config["algorithm"]["maximum_formal_iterations"] == parent["algorithm"]["maximum_formal_iterations"]
    assert (
        config["algorithm"]["candidate_radii_l2_in_fixed_order"]
        == parent["algorithm"]["candidate_radii_l2_in_fixed_order"]
    )
    assert config["algorithm"]["loss"] == parent["algorithm"]["loss"]
    assert config["algorithm"]["direction_method"] == "Polak-Ribiere-Polyak plus nonlinear conjugate gradient"
    assert config["interpretation_boundaries"]["formal_picture_memory_success_always_false"] is True


def test_parent_horizon_delivery_is_hash_bound() -> None:
    contract = core.load_config()["parent_horizon_result"]
    delivery = ROOT / contract["source_delivery_root"]
    bindings = {
        "result.json": "source_result_sha256",
        "manifest.json": "source_manifest_sha256",
        "terminal.json": "source_terminal_sha256",
        "artifact_inventory.json": "source_inventory_sha256",
        "iteration_metrics.jsonl": "source_iteration_metrics_sha256",
        "candidate_metrics.jsonl": "source_candidate_metrics_sha256",
    }
    for name, key in bindings.items():
        assert core.sha256_file(delivery / name) == contract[key]
    inventory = json.loads((delivery / "artifact_inventory.json").read_text(encoding="utf-8"))
    artifacts = {item["path"]: item["sha256"] for item in inventory["artifacts"]}
    assert artifacts["trust_region_tensors.pt"] == contract["source_tensor_bundle_sha256"]
    assert artifacts["checkpoints/iteration-127.pt"] == contract["source_final_checkpoint_sha256"]


def test_prpplus_direction_initial_and_positive_recurrence() -> None:
    gradient = torch.tensor([3.0, 4.0], dtype=torch.float32)
    initial = core.prp_plus_direction(gradient, None, None)
    assert initial["restart_reason"] == "initial"
    assert initial["beta_raw"] == initial["beta"] == 0.0
    assert torch.equal(initial["raw_direction"], -gradient)
    assert torch.allclose(initial["direction"], torch.tensor([-0.6, -0.8]))

    previous_gradient = torch.tensor([1.0, 0.0], dtype=torch.float32)
    previous_raw = torch.tensor([-1.0, 0.0], dtype=torch.float32)
    current = torch.tensor([0.5, 1.0], dtype=torch.float32)
    state = core.prp_plus_direction(current, previous_gradient, previous_raw)
    assert state["restart_reason"] == "none"
    assert state["beta_raw"] == state["beta"] == 0.75
    assert torch.equal(state["raw_direction"], torch.tensor([-1.25, -1.0]))
    assert state["directional_derivative"] < 0.0


def test_prpplus_clamps_negative_beta_and_restarts_non_descent() -> None:
    previous_gradient = torch.tensor([1.0, 0.0], dtype=torch.float32)
    negative = core.prp_plus_direction(
        torch.tensor([0.5, 0.0], dtype=torch.float32),
        previous_gradient,
        torch.tensor([-1.0, 0.0], dtype=torch.float32),
    )
    assert negative["beta_raw"] == -0.25
    assert negative["beta"] == 0.0
    assert negative["restart_reason"] == "negative_beta_clamped"
    assert torch.equal(negative["raw_direction"], torch.tensor([-0.5, -0.0]))

    restarted = core.prp_plus_direction(
        torch.tensor([2.0, 1.0], dtype=torch.float32),
        previous_gradient,
        torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    assert restarted["beta_raw"] == 3.0
    assert restarted["beta"] == 0.0
    assert restarted["restart_reason"] == "non_descent"
    assert torch.equal(restarted["raw_direction"], torch.tensor([-2.0, -1.0]))


@pytest.mark.parametrize(
    ("ratio", "accepted", "expected"),
    (
        (0.009, 10, "fixed_target_trust_region_success"),
        (0.04, 128, "prpplus_material_improvement_only"),
        (0.052, 128, "prpplus_no_matched_budget_advantage"),
        (1.0, 0, "prpplus_stalled"),
    ),
)
def test_preregistered_classification(ratio: float, accepted: int, expected: str) -> None:
    summary = {
        "final_loss_ratio_to_plateau": ratio,
        "accepted_update_count": accepted,
        "all_accepted_losses_strictly_monotone": True,
    }
    assert core.classify_outcome(summary, core.load_config()) == expected


def test_runner_is_optimizer_reader_and_teacher_x_free() -> None:
    source = (ROOT / "scripts/experiments/run_r11_new_fp32_prpplus_trust_region.py").read_text(encoding="utf-8")
    assert "torch.optim" not in source and ".step(" not in source and "clip_grad" not in source
    search = source[source.index("def run_search(") : source.index("def write_report(")]
    assert "teacher_x_T" not in search
    assert "core.prp_plus_direction" in search
    assert "candidate_x = current_x + radius * direction" in search


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
        return 19

    monkeypatch.setattr(parent_runner, "main", fake_main)
    assert runner.main(["formal"]) == 19
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


def test_external_audit_compares_parent_first_step_semantically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_root = ROOT / "reports/r11-new-fp32-trust-region-horizon128-results-20260907/formal"
    output_root = tmp_path / "output"
    output_root.mkdir()
    parent_iteration = json.loads((parent_root / "iteration_metrics.jsonl").read_text(encoding="utf-8").splitlines()[0])
    run_iteration = {
        **parent_iteration,
        "schema": f"{core.PREFIX}-iteration-row.v1",
        "protocol": core.PROTOCOL,
        "restart_reason": "initial",
        "beta_raw": 0.0,
        "beta": 0.0,
        "previous_gradient_fp32_sha256": None,
        "previous_raw_direction_fp32_sha256": None,
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
        lambda: {"parent_horizon_result": {"source_root": str(parent_root)}},
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


def test_independent_audit_reconstructs_prpplus_recurrence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_helpers = runpy.run_path(str(ROOT / "tests/test_r11_new_fp32_trust_region.py"))
    base_helpers["_synthetic_delivery"](tmp_path, mode="formal")
    config = core.load_config()
    iteration_path = tmp_path / "iteration_metrics.jsonl"
    candidate_path = tmp_path / "candidate_metrics.jsonl"
    row = json.loads(iteration_path.read_text(encoding="utf-8").strip())
    row.update(
        schema=f"{core.PREFIX}-iteration-row.v1",
        protocol=core.PROTOCOL,
        previous_gradient_fp32_sha256=None,
        previous_raw_direction_fp32_sha256=None,
        beta_raw=0.0,
        beta=0.0,
        restart_reason="initial",
        negative_gradient_cosine=1.0,
    )
    candidates = [json.loads(line) for line in candidate_path.read_text(encoding="utf-8").splitlines()]
    for candidate in candidates:
        candidate.update(schema=f"{core.PREFIX}-candidate-row.v1", protocol=core.PROTOCOL)
    bundle_path = tmp_path / "trust_region_tensors.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    record = bundle["iterations"][0]
    gradient = record["gradient_fp32"]
    raw_direction = -gradient
    row["raw_direction_fp32_sha256"] = canonical_tensor_sha256(raw_direction)
    bundle.update(schema=f"{core.PREFIX}-tensor-bundle.v1", protocol=core.PROTOCOL)
    record.update(
        previous_gradient_fp32=None,
        previous_raw_direction_fp32=None,
        conjugate_raw_direction_fp32=raw_direction,
    )
    torch.save(bundle, bundle_path)
    checkpoint_path = tmp_path / "checkpoints/iteration-00.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint.update(
        schema=f"{core.PREFIX}-checkpoint.v1",
        protocol=core.PROTOCOL,
        gradient_fp32=gradient,
        conjugate_raw_direction_fp32=raw_direction,
        negative_gradient_direction_fp32=record["negative_gradient_direction_fp32"],
        beta_raw=0.0,
        beta=0.0,
        restart_reason="initial",
    )
    torch.save(checkpoint, checkpoint_path)
    iteration_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    candidate_path.write_text(
        "".join(json.dumps(candidate, sort_keys=True) + "\n" for candidate in candidates),
        encoding="utf-8",
    )
    summary = core.summarize_trace([row], candidates, config)
    classification = core.classify_outcome(summary, config)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(
        schema=f"{core.PREFIX}-manifest.v1",
        protocol=core.PROTOCOL,
        config_sha256=core.canonical_sha(config),
    )
    manifest["optimization_information_boundary"] = {
        "direction_method": config["algorithm"]["direction_method"],
        "direction_uses_only_current_gradient_previous_gradient_and_previous_raw_direction": True,
        "matched_parent_horizon_updates": config["algorithm"]["maximum_formal_iterations"],
    }
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    result.update(
        schema=f"{core.PREFIX}-result.v1",
        protocol=core.PROTOCOL,
        trace_summary=summary,
        classification=classification,
        fixed_target_optimization_success=classification == "fixed_target_trust_region_success",
        tensor_bundle_sha256=core.sha256_file(bundle_path),
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
