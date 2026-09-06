from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import r11_new_identity_condition_bridge as core  # noqa: E402
from vision_memory.training import r11_new_source_init_bridge as source  # noqa: E402


CONFIG = ROOT / "configs/experiments/r11_new_identity_conditioning_bridge_target01.json"


def test_locked_config_and_unchanged_contracts() -> None:
    config = core.load_bridge_config(CONFIG)
    parent = json.loads(
        (
            ROOT
            / "configs/experiments/r11_new_canonical_latent_bridge_target01_source_only_init.json"
        ).read_text(encoding="utf-8")
    )
    for section in ("parent_phase1a", "canonical_teacher", "initialization_binding", "primary_bridge_gate"):
        assert config[section] == parent[section]
    new_fixed, old_fixed = dict(config["unchanged_contract"]), dict(parent["unchanged_contract"])
    assert new_fixed.pop("m0_definition") != old_fixed.pop("m0_definition")
    assert new_fixed == old_fixed
    assert config["parent_bridge"]["protocol"] == source.BRIDGE_PROTOCOL
    assert config["single_changed_solver_factor"]["factor"] == "actual_conditioning_text"
    assert config["single_changed_solver_factor"]["new_value"] == "no changes"
    assert config["condition_probe_binding"]["optimizer_steps"] == 0
    assert config["deployment"]["physical_resource"].startswith("4x NVIDIA H200")
    assert config["deployment"]["runtime_cuda_visible_devices"] == "0,1"
    assert config["deployment"]["runtime_visible_gpu_count"] == 2
    assert config["deployment"]["resource_migration_is_operational_not_scientific_factor"] is True


@pytest.mark.parametrize("section,key,value", [
    ("single_changed_solver_factor", "new_value", "no change"),
    ("single_changed_solver_factor", "full_prompt_sha256", "0" * 64),
    ("parent_bridge", "endpoint_mse", 0.09553645551204681),
    ("parent_bridge", "endpoint_reader_mean_ce", 25.536474171257463),
    ("primary_bridge_gate", "mse_ratio_to_m0_lte", 0.1),
    ("unchanged_contract", "optimizer_steps", 128),
    ("unchanged_contract", "gradient_clipping", 1.0),
    ("condition_probe_binding", "manifest_sha256", "0" * 64),
    ("condition_probe_binding", "probe_tensors_exact_match_required", False),
    ("deployment", "runtime_cuda_visible_devices", "0,1,2,3"),
    ("deployment", "resource_migration_is_operational_not_scientific_factor", False),
])
def test_config_mutation_fails_closed(section: str, key: str, value) -> None:
    config = copy.deepcopy(core.load_bridge_config(CONFIG))
    config[section][key] = value
    with pytest.raises(ValueError, match="canonical config"):
        core.validate_bridge_config(config)


def test_config_byte_drift_fails_even_if_json_equal(tmp_path: Path) -> None:
    path = tmp_path / "changed.json"
    path.write_bytes(CONFIG.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file bytes"):
        core.load_bridge_config(path)


def test_only_condition_parity_changes_and_old_module_is_untouched() -> None:
    changed = {k for k in source.BRIDGE_UNCHANGED_PARITY_BINDINGS
               if source.BRIDGE_UNCHANGED_PARITY_BINDINGS[k] != core.BRIDGE_UNCHANGED_PARITY_BINDINGS[k]}
    assert changed == {"condition_prompt_embeds_sha256", "condition_attention_mask_sha256"}
    assert source.BRIDGE_PARENT_ENDPOINT_MSE == 0.09553645551204681
    assert core.BRIDGE_PARENT_ENDPOINT_MSE == 0.11177215725183487
    assert core.BRIDGE_CONDITION_EMBEDS_SHA256 == core.BRIDGE_UNCHANGED_PARITY_BINDINGS["condition_prompt_embeds_sha256"]


def audit(mse=0.10, ce=25.6, **flags):
    gates = dict(technical_gate=True, teacher_replay_gate=True, distance_pass=False, reader_transfer_pass=False)
    gates.update(flags)
    return core.bridge_conditioning_hypothesis_audit(endpoint_mse=mse, endpoint_reader_mean_ce=ce, **gates)


def test_secondary_uses_source_only_comparator_not_gaussian() -> None:
    assert audit()["passed"] is True
    old = source.bridge_initialization_hypothesis_audit(endpoint_mse=0.10, endpoint_reader_mean_ce=25.6,
        technical_gate=True, teacher_replay_gate=True, distance_pass=False, reader_transfer_pass=False)
    assert old["passed"] is False
    assert core.bridge_conditioning_hypothesis_decision(distance_pass=False, reader_transfer_pass=False,
        audit=audit()) == "distance_fail_reader_fail_secondary_condition_improves"


@pytest.mark.parametrize("mse,ce", [
    (core.BRIDGE_PARENT_ENDPOINT_MSE, 0.0), (0.0, core.BRIDGE_PARENT_ENDPOINT_READER_CE),
    (math.nextafter(core.BRIDGE_PARENT_ENDPOINT_MSE, math.inf), 0.0),
    (0.0, math.nextafter(core.BRIDGE_PARENT_ENDPOINT_READER_CE, math.inf)),
])
def test_two_strict_improvements_required(mse, ce) -> None:
    assert audit(mse, ce)["passed"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True, "0.0", None])
@pytest.mark.parametrize("key", ["mse", "ce"])
def test_invalid_metrics_rejected(bad, key: str) -> None:
    with pytest.raises(ValueError, match="invalid metric"):
        audit(**{key: bad})


@pytest.mark.parametrize("flags", [dict(technical_gate=False), dict(teacher_replay_gate=False),
                                    dict(distance_pass=True), dict(reader_transfer_pass=True)])
def test_secondary_cannot_rescue_or_override_primary(flags) -> None:
    result = audit(0.0, 0.0, **flags)
    assert result["eligible"] is False and result["passed"] is False


@pytest.mark.parametrize("key", ["technical_gate", "teacher_replay_gate", "distance_pass", "reader_transfer_pass"])
def test_non_boolean_gates_rejected(key) -> None:
    with pytest.raises(ValueError, match="exact boolean"):
        audit(**{key: 1})


def test_complete_technical_and_information_boundaries() -> None:
    fields = {name: True for name in core.BRIDGE_REQUIRED_TECHNICAL_FIELDS}
    assert core.bridge_technical_gate(fields)
    for name in fields:
        changed = dict(fields)
        changed[name] = False
        assert not core.bridge_technical_gate(changed)
    boundary = core.bridge_information_boundary()
    assert boundary["formal_success"] is False and boundary["phase2_allowed"] is False
    assert boundary["original_event_used_as_condition"] is False
    assert boundary["event_to_state_learning_evaluated"] is False
    assert core.validate_bridge_information_boundary(boundary)
    for name in boundary:
        changed = dict(boundary)
        changed[name] = not changed[name]
        assert not core.validate_bridge_information_boundary(changed)
