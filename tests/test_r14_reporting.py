from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reporting" / "render_r14_symmetric_donor_report.py"
SPEC = importlib.util.spec_from_file_location("r14_reporter_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def _passing_statistic() -> dict[str, Any]:
    return {
        "relative_change": -0.5,
        "improved_choice_views": 4,
        "accuracy_delta": 1.0,
        "normal_reset_difference_in_differences": -1.0,
        "normal_vs_donor_relative_change": -0.5,
        "normal_better_than_donor_views": 4,
        "normal_vs_donor_accuracy_delta": 0.25,
        "normal_donor_difference_in_differences": -1.0,
        "relative_normal_ce_vs_base": -0.5,
        "normal_better_than_base_views": 4,
        "normal_accuracy_delta_vs_base": 0.25,
        "normal_base_difference_in_differences": -1.0,
    }


def _micro(step: int, *, clipped: bool = False) -> dict[str, Any]:
    del clipped
    return {
        "optimizer_step_zero": step,
        "own_ce": 1.0,
        "donor_ce": 2.0,
        "ranking_loss": 0.5,
        "ranking_satisfied": True,
        "objective": 1.5,
        "residual_coefficient_norm": 0.1,
        "residual_rms": 0.2,
        "image_saturation_fraction": 0.05,
        "mean_residual_coefficient_max_abs": 1e-7,
        "mean_residual_delta_max_abs": 2e-7,
    }


def test_r14_report_criteria_cover_all_locked_gates() -> None:
    row = _passing_statistic()
    assert len(report.CRITERIA) == 12
    assert all(predicate(row) for _key, _label, predicate in report.CRITERIA)
    by_key = {key: predicate for key, _label, predicate in report.CRITERIA}
    row["normal_vs_donor_accuracy_delta"] = 0.0
    assert not by_key["donor_acc"](row)
    row["normal_vs_donor_accuracy_delta"] = 0.25
    row["normal_accuracy_delta_vs_base"] = 0.0
    assert not by_key["base_acc"](row)


def test_r14_training_rows_reconstruct_gradient_accumulation() -> None:
    micro = [_micro(step) for step in range(report.OPTIMIZER_STEPS) for _ in range(report.MICRO_PER_OPTIMIZER)]
    optimizer = [
        {
            "optimizer_step": step + 1,
            "epoch_zero": step // 36,
            "learning_rate": 1e-3,
            "gradient_norm": 1.0,
            "gradient_nonzero_fraction": 1.0,
            "basis_parameter_norm": 2.0,
            "gradient_clipped": False,
        }
        for step in range(report.OPTIMIZER_STEPS)
    ]
    rows = report._training_rows(micro, optimizer)
    assert len(rows) == report.OPTIMIZER_STEPS
    assert rows[0]["own_ce_mean"] == pytest.approx(1.0)
    assert rows[-1]["donor_ce_mean"] == pytest.approx(2.0)
    assert rows[15]["ranking_satisfied_fraction_moving_mean_16"] == pytest.approx(1.0)


def test_r14_training_rows_reject_clipping() -> None:
    micro = [_micro(step) for step in range(report.OPTIMIZER_STEPS) for _ in range(report.MICRO_PER_OPTIMIZER)]
    optimizer = [
        {
            "optimizer_step": step + 1,
            "epoch_zero": 0,
            "learning_rate": 1e-3,
            "gradient_norm": 1.0,
            "gradient_nonzero_fraction": 1.0,
            "basis_parameter_norm": 2.0,
            "gradient_clipped": step == 0,
        }
        for step in range(report.OPTIMIZER_STEPS)
    ]
    with pytest.raises(ValueError, match="invalid optimizer diagnostic"):
        report._training_rows(micro, optimizer)


def test_r14_pair_update_lag_audit_counts_same_update_pairs() -> None:
    rows = []
    for epoch in range(32):
        for pair in range(72):
            left = f"left-{pair:02d}"
            right = f"right-{pair:02d}"
            first = epoch * 100 + pair
            lag = 0 if pair == 0 else 3
            rows.extend(
                (
                    {
                        "epoch_zero": epoch,
                        "segment_id": left,
                        "donor_segment_id": right,
                        "optimizer_step_zero": first,
                    },
                    {
                        "epoch_zero": epoch,
                        "segment_id": right,
                        "donor_segment_id": left,
                        "optimizer_step_zero": first + lag,
                    },
                )
            )
    detail, audit = report._pair_update_lag_rows(rows)
    assert len(detail) == 2304
    assert audit["same_optimizer_update_count"] == 32
    assert audit["same_optimizer_update_fraction"] == pytest.approx(32 / 2304)
    assert audit["median_optimizer_step_lag"] == 3


def test_r14_negative_coverage_distinguishes_identity_and_value_overlap() -> None:
    data = {
        "micro": [
            {
                "segment_id": "a",
                "target_value": "red",
                "donor_segment_id": "b",
                "donor_target_value": "blue",
            },
            {
                "segment_id": "b",
                "target_value": "blue",
                "donor_segment_id": "a",
                "donor_target_value": "red",
            },
            {
                "segment_id": "c",
                "target_value": "blue",
                "donor_segment_id": "a",
                "donor_target_value": "red",
            },
        ],
        "split_rows": {
            "train_audit": [
                {
                    "checkpoint": report.ENDPOINT,
                    "condition": "donor",
                    "item_id": "a",
                    "donor_item_id": "c",
                }
            ]
        },
    }
    detail, audit = report._negative_coverage_rows(data)
    assert detail[0]["exact_donor_seen_in_training"] is False
    assert detail[0]["donor_target_value_seen_in_training"] is True
    assert audit["exact_evaluation_donor_overlap_count"] == 0
    assert audit["evaluation_donor_target_value_overlap_count"] == 1
