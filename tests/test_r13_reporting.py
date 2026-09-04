from __future__ import annotations

import importlib.util
from typing import Any
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reporting" / "render_r13_centered_residual_report.py"
SPEC = importlib.util.spec_from_file_location("r13_reporter_under_test", SCRIPT)
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


def test_r13_report_criteria_cover_all_locked_gates() -> None:
    row = _passing_statistic()
    assert len(report.CRITERIA) == 12
    assert all(predicate(row) for _key, _label, predicate in report.CRITERIA)

    by_key = {key: predicate for key, _label, predicate in report.CRITERIA}
    row["normal_vs_donor_accuracy_delta"] = 0.0
    assert not by_key["donor_acc"](row)
    row["normal_vs_donor_accuracy_delta"] = 0.25
    row["normal_accuracy_delta_vs_base"] = 0.0
    assert not by_key["base_acc"](row)


def test_r13_training_rows_reconstruct_gradient_accumulation() -> None:
    micro = []
    optimizer = []
    for step in range(1, report.OPTIMIZER_STEPS + 1):
        for offset in range(report.MICRO_PER_OPTIMIZER):
            micro.append(
                {
                    "optimizer_step_zero": step - 1,
                    "ce": 1.0 / step,
                    "objective": 1.0 / step,
                    "residual_penalty": 0.0,
                    "coefficient_penalty": 0.0,
                    "residual_coefficient_norm": 0.1,
                    "residual_rms": 0.2,
                    "image_saturation_fraction": 0.1,
                    "mean_residual_coefficient_max_abs": 1e-7 + offset * 1e-9,
                    "mean_residual_delta_max_abs": 2e-7 + offset * 1e-9,
                }
            )
        optimizer.append(
            {
                "optimizer_step": step,
                "epoch_zero": (step - 1) // 36,
                "learning_rate": 1e-3,
                "gradient_norm": 1.0,
                "gradient_nonzero_fraction": 1.0,
                "basis_parameter_norm": 2.0,
                "gradient_clipped": False,
            }
        )

    rows = report._training_rows(micro, optimizer)
    assert len(rows) == report.OPTIMIZER_STEPS
    assert rows[0]["ce_mean"] == pytest.approx(1.0)
    assert rows[-1]["ce_mean"] == pytest.approx(1.0 / report.OPTIMIZER_STEPS)
    assert rows[15]["ce_moving_mean_16_optimizer_steps"] == pytest.approx(
        sum(1.0 / step for step in range(1, 17)) / 16
    )


def test_r13_training_rows_fail_on_clipped_gradient() -> None:
    micro = [
        {
            "optimizer_step_zero": step,
            "ce": 1.0,
            "objective": 1.0,
            "residual_penalty": 0.0,
            "coefficient_penalty": 0.0,
            "residual_coefficient_norm": 0.1,
            "residual_rms": 0.2,
            "image_saturation_fraction": 0.1,
            "mean_residual_coefficient_max_abs": 1e-7,
            "mean_residual_delta_max_abs": 2e-7,
        }
        for step in range(report.OPTIMIZER_STEPS)
        for _ in range(report.MICRO_PER_OPTIMIZER)
    ]
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
