from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "reporting"))

from render_qwen_history_r4_report import render_stage_reports, sha256_file  # noqa: E402
from test_qwen_history_r4_reporting import _fixture, _write_json  # noqa: E402


def _failed_bh1_fixture(
    tmp_path: Path,
) -> tuple[dict[str, dict[str, Path]], Path, Path]:
    arm_inputs, comparison = _fixture(tmp_path)
    last_score_path = arm_inputs["last_effective"]["score"]
    last_score = json.loads(last_score_path.read_text(encoding="utf-8"))
    last_score.update(
        {
            "suite": "transition32",
            "passed": False,
            "execution_passed": False,
            "blocking_policy": {"scientific_gate_blocks_execution": True},
            "scientific_gate": {
                "passed": False,
                "performance_only": False,
                "data_readability_required": True,
            },
        }
    )
    _write_json(last_score_path, last_score)

    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    stdout = stage_dir / "stdout.log"
    stderr = stage_dir / "stderr.log"
    stdout.write_text("scientific gate failed\n", encoding="utf-8")
    stderr.write_text("exit 3\n", encoding="utf-8")
    terminal = stage_dir / "terminal.json"
    _write_json(
        terminal,
        {
            "stage": "qwen-history-r4-bh1",
            "status": "failed",
            "passed": False,
            "exit_code": 3,
            "stdout_sha256": sha256_file(stdout),
            "stderr_sha256": sha256_file(stderr),
        },
    )
    return arm_inputs, comparison, terminal


def test_failed_bh1_report_is_audited_without_becoming_a_pass(tmp_path: Path) -> None:
    arm_inputs, _, terminal = _failed_bh1_fixture(tmp_path)
    output = tmp_path / "failed-report"
    result = render_stage_reports(
        stage="BH1",
        dataset="transition32",
        arm_inputs=arm_inputs,
        comparison=None,
        output_dir=output,
        terminal=terminal,
        evidence=None,
        strict_execution=True,
        report_scientific_failure=True,
    )

    assert result["passed"] is False
    assert result["report_generation_passed"] is True
    assert result["scientific_stage_passed"] is False
    assert result["strict_execution"]["audit_validation_passed"] is True
    assert result["strict_execution"]["report_mode"] == "failed-scientific-bh1"
    assert result["strict_execution"]["evidence"] is None

    on_disk = json.loads((output / "report_set.json").read_text(encoding="utf-8"))
    assert on_disk["passed"] is False
    assert on_disk["report_generation_passed"] is True
    assert on_disk["scientific_stage_passed"] is False
    combined = json.loads((output / "combined" / "report.json").read_text(encoding="utf-8"))
    assert combined["passed"] is False
    assert combined["report_generation_passed"] is True
    assert combined["scientific_stage_passed"] is False

    for arm in ("raw", "tagged", "last_effective"):
        report = json.loads((output / arm / "report.json").read_text(encoding="utf-8"))
        assert report["report_generation_passed"] is True
        assert report["scientific_stage_passed"] is False
        assert "SCIENTIFIC STAGE STATUS: FAILED" in (
            output / arm / "report.md"
        ).read_text(encoding="utf-8")
    assert json.loads(
        (output / "last_effective" / "report.json").read_text(encoding="utf-8")
    )["passed"] is False


def test_default_strict_mode_does_not_accept_failed_stage(tmp_path: Path) -> None:
    arm_inputs, _, terminal = _failed_bh1_fixture(tmp_path)
    with pytest.raises(ValueError, match="terminal and evidence"):
        render_stage_reports(
            stage="BH1",
            dataset="transition32",
            arm_inputs=arm_inputs,
            comparison=None,
            output_dir=tmp_path / "default-report",
            terminal=terminal,
            evidence=None,
            strict_execution=True,
        )


@pytest.mark.parametrize(
    ("stage", "strict_execution", "with_evidence", "with_comparison", "message"),
    (
        ("BH2", True, False, False, "restricted to BH1"),
        ("BH1", False, False, False, "requires strict execution"),
        ("BH1", True, True, False, "must not claim successful stage evidence"),
        ("BH1", True, False, True, "cannot have a successful comparison"),
    ),
)
def test_failed_stage_mode_rejects_scope_drift(
    tmp_path: Path,
    stage: str,
    strict_execution: bool,
    with_evidence: bool,
    with_comparison: bool,
    message: str,
) -> None:
    arm_inputs, comparison, terminal = _failed_bh1_fixture(tmp_path)
    evidence = tmp_path / "evidence.json"
    if with_evidence:
        _write_json(evidence, {"passed": True})
    with pytest.raises(ValueError, match=message):
        render_stage_reports(
            stage=stage,
            dataset="transition32",
            arm_inputs=arm_inputs,
            comparison=comparison if with_comparison else None,
            output_dir=tmp_path / "drift-report",
            terminal=terminal,
            evidence=evidence if with_evidence else None,
            strict_execution=strict_execution,
            report_scientific_failure=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"stage": "qwen-history-r4-bh0"}, "failed nonzero terminal"),
        ({"status": "succeeded", "passed": True, "exit_code": 0}, "failed nonzero terminal"),
    ),
)
def test_failed_stage_mode_rejects_wrong_terminal(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    arm_inputs, _, terminal = _failed_bh1_fixture(tmp_path)
    value = json.loads(terminal.read_text(encoding="utf-8"))
    value.update(mutation)
    _write_json(terminal, value)
    with pytest.raises(ValueError, match=message):
        render_stage_reports(
            stage="BH1",
            dataset="transition32",
            arm_inputs=arm_inputs,
            comparison=None,
            output_dir=tmp_path / "bad-terminal-report",
            terminal=terminal,
            evidence=None,
            strict_execution=True,
            report_scientific_failure=True,
        )


def test_failed_stage_mode_requires_downloaded_bound_logs(tmp_path: Path) -> None:
    arm_inputs, _, terminal = _failed_bh1_fixture(tmp_path)
    (terminal.parent / "stdout.log").unlink()
    with pytest.raises(ValueError, match="requires downloaded stdout.log"):
        render_stage_reports(
            stage="BH1",
            dataset="transition32",
            arm_inputs=arm_inputs,
            comparison=None,
            output_dir=tmp_path / "missing-log-report",
            terminal=terminal,
            evidence=None,
            strict_execution=True,
            report_scientific_failure=True,
        )
