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

from vision_memory.data import REVERSE_CYCLIC4  # noqa: E402
from vision_memory.training import r11_new_bridge as bridge  # noqa: E402


CONFIG = (
    ROOT
    / "configs"
    / "experiments"
    / "r11_new_canonical_latent_bridge_target01_teacher_matched_init.json"
)


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _rows(*, checkpoint: str = "canonical_teacher", condition: str = "teacher") -> list[dict]:
    rows = []
    original_target = 2
    for view_index, permutation in enumerate(REVERSE_CYCLIC4):
        ordered_target = permutation.index(original_target)
        logits = [-2.0, -2.0, -2.0, -2.0]
        logits[ordered_target] = 8.0
        maximum = max(logits)
        ce = maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[ordered_target]
        rows.append(
            {
                "checkpoint": checkpoint,
                "condition": condition,
                "view_index": view_index,
                "permutation": list(permutation),
                "target_index": original_target,
                "choice_logits_ordered": logits,
                "ce": ce,
                "correct": True,
            }
        )
    return rows


def test_preregistered_config_is_exact() -> None:
    config = _config()
    assert bridge.validate_bridge_config(config) == config
    assert bridge.canonical_json_sha256(config) == bridge.BRIDGE_CONFIG_CANONICAL_SHA256
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == bridge.BRIDGE_CONFIG_FILE_SHA256


def test_post128_optimizer_lr_schedule_is_exact() -> None:
    assert all(
        bridge.bridge_optimizer_learning_rate(update) == bridge.BRIDGE_BASE_LEARNING_RATE
        for update in range(1, 129)
    )
    assert bridge.bridge_optimizer_learning_rate(129) == pytest.approx(
        0.025 * (1.0 + math.cos(math.pi / 128.0)),
        rel=0.0,
        abs=0.0,
    )
    assert bridge.bridge_optimizer_learning_rate(192) == pytest.approx(0.025, rel=0.0, abs=1e-16)
    assert bridge.bridge_optimizer_learning_rate(256) == pytest.approx(0.0, rel=0.0, abs=1e-16)


@pytest.mark.parametrize("update", [True, 1.0, "1"])
def test_optimizer_lr_schedule_rejects_non_integer_indices(update: object) -> None:
    with pytest.raises(TypeError):
        bridge.bridge_optimizer_learning_rate(update)  # type: ignore[arg-type]


@pytest.mark.parametrize("update", [0, 257])
def test_optimizer_lr_schedule_rejects_out_of_range_indices(update: int) -> None:
    with pytest.raises(ValueError):
        bridge.bridge_optimizer_learning_rate(update)


def test_initialization_intervention_preserves_other_parent_bindings() -> None:
    parent_path = CONFIG.with_name("r11_new_canonical_latent_bridge_target01_post128_cosine.json")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    bindings = parent["exact_parity_bindings"]
    assert bridge.BRIDGE_PHASE1A_SOURCE_ROOT == bindings["phase1a_valid_source_root"]
    assert bridge.BRIDGE_UNCHANGED_PARITY_BINDINGS == {
        "target_index": parent["target_selection"]["target_index"],
        "target_segment_id": parent["target_selection"]["target_segment_id"],
        **{
            key: bindings[key]
            for key in (
                "blank_source_rgb_sha256",
                "source_latents_fp32_sha256",
                "event_text_sha256",
                "condition_prompt_embeds_sha256",
                "condition_attention_mask_sha256",
            )
        },
    }
    assert bridge.BRIDGE_MODEL_SNAPSHOT_BINDINGS == {
        key: bindings[key]
        for key in ("dreamlite_snapshot_manifest_sha256", "reader_snapshot_manifest_sha256")
    }
    assert "initial_x_T_fp32_sha256" not in bridge.BRIDGE_UNCHANGED_PARITY_BINDINGS
    assert "initial_z_t_fp32_sha256" not in bridge.BRIDGE_UNCHANGED_PARITY_BINDINGS


@pytest.mark.parametrize(
    ("distance", "reader", "endpoint", "expected"),
    [
        (True, True, 0.001, "primary_branch_distance_pass_reader_pass_design_answer_independent_initializer"),
        (True, False, 0.001, "primary_branch_distance_pass_reader_fail_test_teacher_neighborhood"),
        (False, True, 0.07, "primary_branch_distance_fail_reader_pass_prioritize_qa_objective"),
        (False, False, 0.07, "distance_fail_reader_fail_secondary_init_improves"),
        (False, False, 0.10, "distance_fail_reader_fail_secondary_init_not_improve"),
    ],
)
def test_initialization_hypothesis_is_secondary_and_non_rescuing(
    distance: bool,
    reader: bool,
    endpoint: float,
    expected: str,
) -> None:
    audit = bridge.bridge_initialization_hypothesis_audit(
        endpoint_mse=endpoint,
        endpoint_reader_mean_ce=20.0,
        technical_gate=True,
        teacher_replay_gate=True,
        distance_pass=distance,
        reader_transfer_pass=reader,
    )
    assert audit["eligible"] is (not distance and not reader)
    assert bridge.bridge_initialization_hypothesis_decision(
        distance_pass=distance,
        reader_transfer_pass=reader,
        audit=audit,
    ) == expected


@pytest.mark.parametrize(
    ("mse", "ce", "mse_improves", "ce_improves"),
    [
        (bridge.BRIDGE_PARENT_ENDPOINT_MSE, 20.0, False, True),
        (0.07, bridge.BRIDGE_PARENT_ENDPOINT_READER_CE, True, False),
        (0.10, 26.0, False, False),
        (0.07, 20.0, True, True),
    ],
)
def test_initialization_secondary_requires_both_strict_absolute_improvements(
    mse: float, ce: float, mse_improves: bool, ce_improves: bool,
) -> None:
    audit = bridge.bridge_initialization_hypothesis_audit(
        endpoint_mse=mse,
        endpoint_reader_mean_ce=ce,
        technical_gate=True,
        teacher_replay_gate=True,
        distance_pass=False,
        reader_transfer_pass=False,
    )
    assert audit == {
        "eligible": True,
        "absolute_endpoint_mse_improves_parent": mse_improves,
        "endpoint_reader_ce_improves_parent": ce_improves,
        "passed": mse_improves and ce_improves,
    }


@pytest.mark.parametrize("field", ["technical_gate", "teacher_replay_gate"])
@pytest.mark.parametrize("invalid", [False, None, 1, "true"])
def test_initialization_secondary_requires_valid_technical_evidence(field: str, invalid: object) -> None:
    kwargs = {
        "endpoint_mse": 0.07,
        "endpoint_reader_mean_ce": 20.0,
        "technical_gate": True,
        "teacher_replay_gate": True,
        "distance_pass": False,
        "reader_transfer_pass": False,
    }
    kwargs[field] = invalid
    audit = bridge.bridge_initialization_hypothesis_audit(**kwargs)
    assert audit["eligible"] is False
    assert audit["passed"] is False


@pytest.mark.parametrize("field", ["endpoint_mse", "endpoint_reader_mean_ce"])
@pytest.mark.parametrize("invalid", [-1.0, float("nan"), float("inf"), True, "0.01", None])
def test_initialization_secondary_rejects_invalid_metrics(field: str, invalid: object) -> None:
    kwargs = {
        "endpoint_mse": 0.07,
        "endpoint_reader_mean_ce": 20.0,
        "technical_gate": True,
        "teacher_replay_gate": True,
        "distance_pass": False,
        "reader_transfer_pass": False,
    }
    kwargs[field] = invalid
    with pytest.raises(ValueError, match="invalid metric"):
        bridge.bridge_initialization_hypothesis_audit(**kwargs)


@pytest.mark.parametrize("distance,reader", [(True, True), (True, False), (False, True)])
def test_primary_branch_precedes_even_a_claimed_secondary_pass(distance: bool, reader: bool) -> None:
    decision = bridge.bridge_initialization_hypothesis_decision(
        distance_pass=distance, reader_transfer_pass=reader, audit={"passed": True},
    )
    assert decision == "primary_branch_" + bridge.bridge_decision(
        distance_pass=distance, reader_transfer_pass=reader,
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unchanged_contract", "base_learning_rate"), 0.01),
        (("unchanged_contract", "gradient_clipping"), 1.0),
        (("unchanged_contract", "optimizer_steps"), 512),
        (("unchanged_contract", "m0_definition"), "the initial trajectory point zero"),
        (("unchanged_contract", "train_sha256"), "0" * 64),
        (("primary_bridge_gate", "mse_ratio_to_m0_lte"), 0.02),
        (("target_selection", "target_index"), 7),
        (("canonical_teacher", "tensor_sha256"), "0" * 64),
        (("single_changed_solver_factor", "teacher_assisted"), False),
        (("single_changed_solver_factor", "parameterization_dtype"), "torch.bfloat16"),
        (("initialization_binding", "actual_sigma_must_come_from_scheduler_setup"), False),
        (("initialization_binding", "reconstruction_operator_order"), "source + sigma * (x_T - source)"),
        (("initialization_binding", "trajectory_point0_teacher_normalized_rmse_lte"), 0.02),
        (("formal_technical_gate", "trajectory_point0_binding_valid_every_checkpoint"), False),
        (("interpretation_boundaries", "formal_success_always_false"), False),
        (("interpretation_boundaries", "phase2_remains_blocked"), False),
    ],
)
def test_config_mutations_fail_closed(path: tuple[str, str], value: object) -> None:
    config = copy.deepcopy(_config())
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError, match="config"):
        bridge.validate_bridge_config(config)


def test_config_rejects_implementation_constant_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge, "BRIDGE_START_TEACHER_NRMSE_MAX", 0.02)
    with pytest.raises(ValueError, match="constant mismatch"):
        bridge.validate_bridge_config(_config())


def test_distance_statistics_and_strict_gate() -> None:
    statistics = bridge.bridge_distance_statistics(
        mse=0.0025,
        m0_mse=1.0,
        tensor_numel=100,
        teacher_population_std=1.0,
    )
    assert statistics["rmse"] == pytest.approx(0.05)
    assert statistics["l2_distance"] == pytest.approx(0.5)
    assert statistics["mse_ratio_to_m0"] == pytest.approx(0.0025)
    assert statistics["l2_distance_ratio_to_m0"] == pytest.approx(0.05)
    assert statistics["teacher_normalized_rmse"] == pytest.approx(0.05)
    assert bridge.bridge_distance_gate(statistics, technical_gate=True)
    assert not bridge.bridge_distance_gate(statistics, technical_gate=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mse_ratio_to_m0", 0.0100001),
        ("l2_distance_ratio_to_m0", 0.100001),
        ("teacher_normalized_rmse", 0.100001),
    ],
)
def test_distance_gate_rejects_each_threshold_failure(field: str, value: float) -> None:
    statistics = {
        "mse_ratio_to_m0": 0.01,
        "l2_distance_ratio_to_m0": 0.1,
        "teacher_normalized_rmse": 0.1,
    }
    statistics[field] = value
    assert not bridge.bridge_distance_gate(statistics, technical_gate=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mse": -1.0, "m0_mse": 1.0, "tensor_numel": 4},
        {"mse": 1.0, "m0_mse": 0.0, "tensor_numel": 4},
        {"mse": 1.0, "m0_mse": 1.0, "tensor_numel": 0},
        {"mse": float("nan"), "m0_mse": 1.0, "tensor_numel": 4},
    ],
)
def test_distance_statistics_reject_invalid_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        bridge.bridge_distance_statistics(**kwargs)


def test_reader_rows_are_recomputed_from_logits() -> None:
    statistics = bridge.reader_checkpoint_statistics(_rows(), checkpoint="canonical_teacher", condition="teacher")
    assert statistics["row_count"] == 4
    assert statistics["accuracy"] == 1.0
    assert statistics["all_four_correct"]
    assert statistics["permutations_are_independent_statistical_units"] is False
    assert bridge.teacher_replay_gate(statistics)
    assert bridge.endpoint_reader_transfer_gate(statistics)


def test_reader_rows_reject_declared_ce_or_correctness_drift() -> None:
    rows = _rows()
    rows[0]["ce"] += 1.0
    with pytest.raises(ValueError, match="CE"):
        bridge.reader_checkpoint_statistics(rows, checkpoint="canonical_teacher", condition="teacher")
    rows = _rows()
    rows[0]["correct"] = False
    with pytest.raises(ValueError, match="correctness"):
        bridge.reader_checkpoint_statistics(rows, checkpoint="canonical_teacher", condition="teacher")


def test_reader_rows_reject_duplicate_or_unlocked_views() -> None:
    rows = _rows()
    rows[3]["view_index"] = 2
    with pytest.raises(ValueError, match="view indices"):
        bridge.reader_checkpoint_statistics(rows, checkpoint="canonical_teacher", condition="teacher")
    rows = _rows()
    rows[0]["permutation"] = [0, 2, 1, 3]
    with pytest.raises(ValueError, match="reverse-cyclic"):
        bridge.reader_checkpoint_statistics(rows, checkpoint="canonical_teacher", condition="teacher")


def test_teacher_replay_mean_ce_is_strict() -> None:
    statistics = {"row_count": 4, "all_four_correct": True, "mean_ce": 0.001}
    assert bridge.teacher_replay_gate(statistics)
    statistics["mean_ce"] = 0.00100001
    assert not bridge.teacher_replay_gate(statistics)


@pytest.mark.parametrize(
    ("distance", "reader", "expected"),
    [
        (True, True, "distance_pass_reader_pass_design_answer_independent_initializer"),
        (True, False, "distance_pass_reader_fail_test_teacher_neighborhood"),
        (False, True, "distance_fail_reader_pass_prioritize_qa_objective"),
        (False, False, "distance_fail_reader_fail_change_one_solver_factor"),
    ],
)
def test_bridge_decision_table(distance: bool, reader: bool, expected: str) -> None:
    assert bridge.bridge_decision(distance_pass=distance, reader_transfer_pass=reader) == expected


def test_technical_gate_requires_every_contract() -> None:
    keys = (
        "receipts_exact",
        "steps_contiguous",
        "four_step_schedule_exact",
        "finite_metrics",
        "finite_nonzero_gradient_every_step",
        "only_x_T_fp32_trainable",
        "frozen_gradients_absent",
        "snapshots_unchanged",
        "optimizer_contract_valid",
        "optimizer_lr_schedule_exact",
        "teacher_matched_initialization_artifact_valid",
        "trajectory_point0_binding_valid_every_checkpoint",
        "gradient_clipping_absent",
        "checkpoint_hashes_valid",
        "condition_artifact_valid",
        "step0_parity_valid",
        "teacher_binding_valid",
        "artifact_contract_valid",
    )
    audit = {key: True for key in keys}
    assert bridge.bridge_technical_gate(audit)
    for key in keys:
        mutated = dict(audit)
        mutated[key] = False
        assert not bridge.bridge_technical_gate(mutated), key
