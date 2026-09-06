from __future__ import annotations

import copy
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
    / "r11_new_canonical_latent_bridge_target01_post128_cosine.json"
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


def test_pre_intervention_parity_requires_every_parent_anchor() -> None:
    record = {
        "optimizer_step": 128,
        "tensor_sha256": copy.deepcopy(bridge.BRIDGE_PARENT_STEP128_TENSOR_SHA256),
        "optimizer_state_sha256": bridge.BRIDGE_PARENT_STEP128_OPTIMIZER_SHA256,
        "png_sha256": bridge.BRIDGE_PARENT_STEP128_PNG_SHA256,
    }
    assert bridge.bridge_pre_intervention_parity(record)
    for key in tuple(record):
        mutated = copy.deepcopy(record)
        mutated[key] = 127 if key == "optimizer_step" else "0" * 64
        assert not bridge.bridge_pre_intervention_parity(mutated), key


@pytest.mark.parametrize(
    ("distance", "reader", "endpoint", "expected"),
    [
        (True, True, 0.001, "primary_branch_distance_pass_reader_pass_prioritize_qa_objective"),
        (True, False, 0.001, "primary_branch_distance_pass_reader_fail_test_teacher_neighborhood"),
        (False, True, 0.70, "primary_branch_distance_fail_reader_pass_prioritize_qa_objective"),
        (False, False, 0.70, "distance_fail_reader_fail_secondary_schedule_pass"),
        (False, False, 0.80, "distance_fail_reader_fail_secondary_schedule_fail"),
    ],
)
def test_schedule_hypothesis_is_secondary_and_non_rescuing(
    distance: bool,
    reader: bool,
    endpoint: float,
    expected: str,
) -> None:
    audit = bridge.bridge_schedule_hypothesis_audit(
        step128_mse_ratio=0.7737632777116902,
        endpoint_mse_ratio=endpoint,
        technical_gate=True,
        teacher_replay_gate=True,
        pre_intervention_parity=True,
        distance_pass=distance,
        reader_transfer_pass=reader,
    )
    assert bridge.bridge_schedule_hypothesis_decision(
        distance_pass=distance,
        reader_transfer_pass=reader,
        audit=audit,
    ) == expected


def test_schedule_hypothesis_fails_on_rebound_even_if_it_beats_parent() -> None:
    audit = bridge.bridge_schedule_hypothesis_audit(
        step128_mse_ratio=0.7737632777116902,
        endpoint_mse_ratio=0.80,
        technical_gate=True,
        teacher_replay_gate=True,
        pre_intervention_parity=True,
        distance_pass=False,
        reader_transfer_pass=False,
    )
    assert audit == {
        "eligible": True,
        "post128_non_rebound": False,
        "beats_parent_endpoint": True,
        "passed": False,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("unchanged_contract", "learning_rate"), 0.01),
        (("unchanged_contract", "gradient_clipping"), 1.0),
        (("primary_bridge_gate", "mse_ratio_to_m0_lte"), 0.02),
        (("target_selection", "target_index"), 7),
        (("interpretation_boundaries", "phase2_remains_blocked"), False),
    ],
)
def test_config_mutations_fail_closed(path: tuple[str, str], value: object) -> None:
    config = copy.deepcopy(_config())
    config[path[0]][path[1]] = value
    with pytest.raises(ValueError, match="config"):
        bridge.validate_bridge_config(config)


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
        (True, True, "distance_pass_reader_pass_prioritize_qa_objective"),
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
        "pre_intervention_step128_parity_valid",
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
