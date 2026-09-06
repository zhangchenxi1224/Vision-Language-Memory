from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import r11_new_bridge as old  # noqa: E402
from vision_memory.training import r11_new_source_init_bridge as bridge  # noqa: E402


CONFIG = ROOT / "configs/experiments/r11_new_canonical_latent_bridge_target01_source_only_init.json"


def _config() -> dict:
    return json.loads(CONFIG.read_bytes())


def _audit(*, mse: float = 0.08, ce: float = 20.0, **gates: bool) -> dict:
    kwargs = {
        "technical_gate": True, "teacher_replay_gate": True,
        "distance_pass": False, "reader_transfer_pass": False,
    }
    kwargs.update(gates)
    return bridge.bridge_initialization_hypothesis_audit(
        endpoint_mse=mse, endpoint_reader_mean_ce=ce, **kwargs,
    )


def test_new_config_is_exact_and_old_config_is_unchanged() -> None:
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == bridge.BRIDGE_CONFIG_FILE_SHA256
    assert bridge.canonical_json_sha256(_config()) == bridge.BRIDGE_CONFIG_CANONICAL_SHA256
    assert bridge.validate_bridge_config(_config()) == _config()
    old_path = CONFIG.with_name("r11_new_canonical_latent_bridge_target01_teacher_matched_init.json")
    old_config = json.loads(old_path.read_bytes())
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == old.BRIDGE_CONFIG_FILE_SHA256
    assert old.validate_bridge_config(old_config) == old_config
    with pytest.raises(ValueError, match="schema"):
        bridge.validate_bridge_config(old_config)
    with pytest.raises(ValueError, match="schema"):
        old.validate_bridge_config(_config())


def test_unchanged_math_is_explicitly_reused_not_monkeypatched() -> None:
    for name in (
        "canonical_json_sha256", "bridge_distance_statistics", "bridge_distance_gate",
        "bridge_optimizer_learning_rate", "bridge_initialization_hypothesis_audit",
        "reader_checkpoint_statistics", "teacher_replay_gate", "endpoint_reader_transfer_gate",
    ):
        assert getattr(bridge, name) is getattr(old, name)
        assert name in bridge.__all__
    assert old.BRIDGE_PROTOCOL == "R11-New-Canonical-Latent-Bridge-Teacher-Matched-Init-Target01"
    assert bridge.BRIDGE_PROTOCOL != old.BRIDGE_PROTOCOL
    assert not hasattr(bridge, "BRIDGE_START_TEACHER_NRMSE_MAX")
    assert not hasattr(bridge, "BRIDGE_TEACHER_MATCHED_SIGMA")


def test_single_factor_and_information_boundary_are_explicit() -> None:
    config = _config()
    changed = config["single_changed_solver_factor"]
    assert changed["factor"] == "x_T_initialization"
    assert changed["formula"] == "x_T_init_fp32=source_latents_fp32.clone()"
    assert changed["initializer_allowed_inputs"] == ["source_latents"]
    assert set(changed["initializer_forbidden_inputs"]) == {
        "teacher", "query", "answer", "choices", "target_index", "sample_id",
    }
    init = config["initialization_binding"]
    assert init["artifact_must_store"] == [
        "source_latents_fp32", "x_T_init_fp32", "reconstructed_start_state_compute",
    ]
    assert init["no_teacher_neighborhood_threshold"] is True
    assert init["no_cpu_fallback_for_cuda_recomputation"] is True
    assert "trajectory_point0_teacher_normalized_rmse_lte" not in init
    assert "require_trajectory_point0_teacher_normalized_rmse_lte" not in config["preflight_gate"]
    assert "teacher_fp32" not in init["artifact_must_store"]
    assert "teacher_matched_initialization_artifact_valid" not in config["formal_technical_gate"]
    assert config["formal_technical_gate"]["source_only_initialization_artifact_valid"] is True
    boundary = bridge.bridge_information_boundary()
    assert boundary == {
        "initialization_teacher_assisted": False, "optimization_teacher_supervised": True,
        "answer_independent_writer_usable": False, "formal_success": False, "phase2_allowed": False,
    }
    assert bridge.validate_bridge_information_boundary(boundary)


@pytest.mark.parametrize("name", tuple(bridge.bridge_information_boundary()))
def test_missing_or_inverted_boundary_is_rejected(name: str) -> None:
    correct = bridge.bridge_information_boundary()
    for replacement in (not correct[name], None, int(correct[name])):
        changed = dict(correct, **{name: replacement})
        assert not bridge.validate_bridge_information_boundary(changed)
    del correct[name]
    assert not bridge.validate_bridge_information_boundary(correct)


@pytest.mark.parametrize(
    ("section", "key", "replacement"),
    [
        ("target_selection", "target_index", 7),
        ("single_changed_solver_factor", "formula", "x_T_init_fp32=teacher_fp32.clone()"),
        ("single_changed_solver_factor", "initializer_allowed_inputs", ["source_latents", "teacher"]),
        ("initialization_binding", "compute_device_type", "cpu"),
        ("initialization_binding", "compute_dtype", "torch.float32"),
        ("initialization_binding", "initial_x_t_equals_source", False),
        ("initialization_binding", "artifact_must_store", ["source_latents_fp32", "teacher_fp32"]),
        ("unchanged_contract", "optimizer_steps", 128),
        ("unchanged_contract", "gradient_clipping", 1.0),
        ("unchanged_contract", "diffusion_steps", 1),
        ("primary_bridge_gate", "mse_ratio_to_m0_lte", 0.5),
        ("primary_bridge_gate", "endpoint_reverse_cyclic_accuracy_eq", 0.25),
        ("parent_bridge", "endpoint_mse", bridge.BRIDGE_TEACHER_REFERENCE_ENDPOINT_MSE),
        ("teacher_matched_reference", "role", "secondary_comparator"),
        ("interpretation_boundaries", "formal_success_always_false", False),
        ("interpretation_boundaries", "phase2_remains_blocked", False),
    ],
)
def test_any_config_mutation_fails_closed(section: str, key: str, replacement: object) -> None:
    changed = copy.deepcopy(_config())
    changed[section][key] = replacement
    with pytest.raises(ValueError, match="canonical"):
        bridge.validate_bridge_config(changed)


@pytest.mark.parametrize(
    ("constant", "replacement"),
    [
        ("BRIDGE_TARGET_INDEX", 7), ("BRIDGE_INITIALIZATION_FORMULA", "wrong"),
        ("BRIDGE_OPTIMIZER_STEPS", 128), ("BRIDGE_MSE_RATIO_MAX", 0.5),
        ("BRIDGE_TEACHER_NRMSE_MAX", 0.5), ("BRIDGE_PARENT_ENDPOINT_MSE", 0.0657),
        ("BRIDGE_TEACHER_REFERENCE_ENDPOINT_MSE", 0.5),
    ],
)
def test_implementation_constant_drift_also_fails(monkeypatch: pytest.MonkeyPatch, constant: str, replacement: object) -> None:
    monkeypatch.setattr(bridge, constant, replacement)
    with pytest.raises(ValueError, match="constant mismatch"):
        bridge.validate_bridge_config(_config())


@pytest.mark.parametrize("name", bridge.BRIDGE_REQUIRED_TECHNICAL_FIELDS)
def test_each_technical_gate_field_is_required_and_strict_bool(name: str) -> None:
    audit = dict.fromkeys(bridge.BRIDGE_REQUIRED_TECHNICAL_FIELDS, True)
    assert bridge.bridge_technical_gate(audit)
    for replacement in (False, None, 1, "true"):
        assert not bridge.bridge_technical_gate(dict(audit, **{name: replacement}))
    del audit[name]
    assert not bridge.bridge_technical_gate(audit)


def test_old_initializer_gate_does_not_satisfy_new_gate() -> None:
    audit = dict.fromkeys(bridge.BRIDGE_REQUIRED_TECHNICAL_FIELDS, True)
    del audit["source_only_initialization_artifact_valid"]
    audit["teacher_matched_initialization_artifact_valid"] = True
    assert not bridge.bridge_technical_gate(audit)


def test_secondary_comparator_remains_gaussian_not_teacher_reference() -> None:
    # Worse than teacher-matched but better than Gaussian: qualifies only for
    # the non-rescuing Gaussian comparison, exactly as preregistered.
    audit = _audit(mse=0.08, ce=20.0)
    assert audit["passed"] is True
    assert _audit(mse=bridge.BRIDGE_PARENT_ENDPOINT_MSE)["passed"] is False
    assert _audit(ce=bridge.BRIDGE_PARENT_ENDPOINT_READER_CE)["passed"] is False
    assert _audit(mse=0.10, ce=20.0)["passed"] is False
    assert _audit(mse=0.08, ce=26.0)["passed"] is False


@pytest.mark.parametrize(
    "gates", [
        {"technical_gate": False}, {"teacher_replay_gate": False},
        {"distance_pass": True}, {"reader_transfer_pass": True},
    ],
)
def test_secondary_is_ineligible_when_technical_invalid_or_primary_passes(gates: dict) -> None:
    audit = _audit(**gates)
    assert audit["eligible"] is False
    assert audit["passed"] is False


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), -1.0, True, "0.1", None])
@pytest.mark.parametrize("field", ["mse", "ce"])
def test_secondary_rejects_invalid_metrics(metric: object, field: str) -> None:
    with pytest.raises(ValueError, match="invalid metric"):
        _audit(**{field: metric})


@pytest.mark.parametrize(
    ("distance", "reader", "expected"),
    [
        (True, True, "distance_pass_reader_pass_revalidate_original_qa_solver"),
        (True, False, "distance_pass_reader_fail_test_teacher_neighborhood"),
        (False, True, "distance_fail_reader_pass_prioritize_qa_objective"),
        (False, False, "distance_fail_reader_fail_change_one_solver_factor"),
    ],
)
def test_primary_decision_and_boundary_do_not_promote_to_phase2(distance: bool, reader: bool, expected: str) -> None:
    assert bridge.bridge_decision(distance_pass=distance, reader_transfer_pass=reader) == expected
    audit = _audit(distance_pass=distance, reader_transfer_pass=reader)
    secondary = bridge.bridge_initialization_hypothesis_decision(
        distance_pass=distance, reader_transfer_pass=reader, audit=audit,
    )
    assert secondary == (f"primary_branch_{expected}" if distance or reader else "distance_fail_reader_fail_secondary_init_improves")
    assert bridge.bridge_information_boundary()["phase2_allowed"] is False
    assert bridge.bridge_information_boundary()["formal_success"] is False


def test_secondary_claim_without_eligibility_cannot_pass() -> None:
    assert bridge.bridge_initialization_hypothesis_decision(
        distance_pass=False, reader_transfer_pass=False, audit={"eligible": False, "passed": True},
    ) == "distance_fail_reader_fail_secondary_init_not_improve"


def test_optimizer_and_primary_thresholds_are_unchanged() -> None:
    for update in range(1, 257):
        assert bridge.bridge_optimizer_learning_rate(update) == old.bridge_optimizer_learning_rate(update)
    assert bridge.bridge_optimizer_learning_rate(256) == 0.0
    assert bridge.BRIDGE_CHECKPOINT_STEPS == (0, 64, 128, 192, 256)
    stats = bridge.bridge_distance_statistics(mse=0.0001, m0_mse=0.1, tensor_numel=65536)
    assert bridge.bridge_distance_gate(stats, technical_gate=True)
    assert stats["l2_distance_ratio_to_m0"] == math.sqrt(0.0001) / math.sqrt(0.1)
    assert not bridge.bridge_distance_gate(stats, technical_gate=False)
    assert bridge.BRIDGE_UNCHANGED_PARITY_BINDINGS == old.BRIDGE_UNCHANGED_PARITY_BINDINGS
    assert bridge.BRIDGE_MODEL_SNAPSHOT_BINDINGS == old.BRIDGE_MODEL_SNAPSHOT_BINDINGS


def test_teacher_reference_hashes_and_values_match_closed_local_delivery() -> None:
    folder = ROOT / "reports/r11-new-teacher-matched-init-results-20260906/aggregation-v1"
    comparison_bytes = (folder / "comparison.json").read_bytes()
    assert hashlib.sha256(comparison_bytes).hexdigest() == bridge.BRIDGE_TEACHER_REFERENCE_COMPARISON_SHA256
    assert hashlib.sha256((folder / "RAW_ARTIFACTS.json").read_bytes()).hexdigest() == bridge.BRIDGE_TEACHER_REFERENCE_RAW_SHA256
    comparison = json.loads(comparison_bytes)
    assert comparison["endpoint_distance_statistics"]["mse"] == bridge.BRIDGE_TEACHER_REFERENCE_ENDPOINT_MSE
    assert comparison["endpoint_reader_statistics"]["mean_ce"] == bridge.BRIDGE_TEACHER_REFERENCE_ENDPOINT_READER_CE
    assert comparison["engineering_gate"] is True
    assert comparison["formal_success"] is False
