from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "reporting"))
sys.path.insert(0, str(ROOT / "scripts" / "inspire"))

from qwen_history_r4_contract import (  # noqa: E402
    ARM_METHODS,
    ARM_ORDER,
    COMPARISON_SCHEMA,
    SCORE_SCHEMA,
    STAGE_EVIDENCE_PROTOCOL,
)
from render_qwen_history_r4_report import (  # noqa: E402
    PLOT_NAMES,
    render_stage_reports,
    sha256_file,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _arm_fixture(root: Path, arm: str) -> dict[str, Path]:
    method = ARM_METHODS[arm]
    values: dict[str, Path] = {}
    rows_by_replica: dict[str, list[dict]] = {}
    for replica in ("A", "B"):
        rows = [
            {
                "method": method,
                "condition": "standard",
                "prediction_index": 0,
                "target_index": 0,
                "nll_margin": 0.4,
                "memory_utf8_bytes": 96 if arm != "last_effective" else 40,
                "latency_seconds": 0.012,
            },
            {
                "method": method,
                "condition": "standard",
                "prediction_index": 1,
                "target_index": 0,
                "nll_margin": -0.1,
                "memory_utf8_bytes": 128 if arm != "last_effective" else 44,
                "latency_seconds": 0.014,
            },
        ]
        rows_by_replica[replica] = rows
        prediction = root / arm / f"predictions-{replica.lower()}.jsonl"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        report = root / arm / f"report-{replica.lower()}.json"
        _write_json(
            report,
            {
                "status": "complete",
                "method": method,
                "replica_id": replica,
                "input_mode": "blank_image",
                "choice_view_family": "reverse-cyclic4",
                "output_sha256": sha256_file(prediction),
            },
        )
        values[f"predictions_{replica.lower()}"] = prediction
        values[f"report_{replica.lower()}"] = report
    score = root / arm / "score.json"
    _write_json(
        score,
        {
            "schema": SCORE_SCHEMA,
            "method": method,
            "suite": "formal",
            "passed": True,
            "execution_passed": True,
            "integrity": {
                "passed": True,
                "prediction_sha256": sha256_file(values["predictions_a"]),
                "prediction_report_sha256": sha256_file(values["report_a"]),
                "replica_b_prediction_sha256": sha256_file(values["predictions_b"]),
                "replica_b_report_sha256": sha256_file(values["report_b"]),
            },
            "replication": {
                "passed": True,
                "bitwise_scientific_payload_match": True,
            },
            "scientific_gate": {
                "passed": None,
                "performance_only": True,
                "data_readability_required": False,
            },
            "descriptive_metrics": {
                "standard": {"correct": 1, "count": 2, "accuracy": 0.5},
                "conditions": {
                    "standard": {"correct": 1, "count": 2, "accuracy": 0.5},
                    "reset": {"correct": 0, "count": 0, "accuracy": None},
                    "shuffle": {"correct": 0, "count": 0, "accuracy": None},
                    "state_swap": {"correct": 0, "count": 0, "accuracy": None},
                },
                "by_target_position": {"0": {"correct": 1, "count": 2, "accuracy": 0.5}},
                "by_event_kind": {"set": {"correct": 1, "count": 2, "accuracy": 0.5}},
                "by_form": {"separate": {"correct": 1, "count": 2, "accuracy": 0.5}},
                "by_ood_group": {},
                "rotation": {"consistent": 1, "count": 2, "agreement_rate": 0.5},
                "state_swap_donor_answer": {"correct": 0, "count": 0, "rate": None},
            },
        },
    )
    values["score"] = score
    return values


def _fixture(tmp_path: Path) -> tuple[dict[str, dict[str, Path]], Path]:
    arm_inputs = {arm: _arm_fixture(tmp_path / "inputs", arm) for arm in ARM_ORDER}
    comparison = tmp_path / "inputs" / "comparison.json"
    _write_json(
        comparison,
        {
            "schema": COMPARISON_SCHEMA,
            "suite": "formal",
            "passed": True,
            "identity_pairing": {"passed": True},
            "comparisons": {
                "tagged_minus_raw": {"b_minus_a": {"difference": 0.0}},
                "last_effective_minus_tagged": {"b_minus_a": {"difference": 0.0}},
            },
        },
    )
    return arm_inputs, comparison


def test_report_set_writes_each_arm_all_formats_and_eight_plots(tmp_path: Path) -> None:
    arm_inputs, comparison = _fixture(tmp_path)
    output = tmp_path / "report"
    result = render_stage_reports(
        stage="BH2",
        dataset="formal_dev",
        arm_inputs=arm_inputs,
        comparison=comparison,
        output_dir=output,
        terminal=None,
        evidence=None,
        strict_execution=False,
    )
    assert result["passed"] is True
    for arm in ARM_ORDER:
        arm_dir = output / arm
        for filename in ("report.md", "report.html", "report.json", "metrics.csv", "sha256_manifest.json"):
            assert (arm_dir / filename).is_file()
        assert len(list((arm_dir / "plots").glob("*.png"))) == len(PLOT_NAMES) == 8
        assert all(path.stat().st_size > 0 for path in (arm_dir / "plots").glob("*.png"))
        report = json.loads((arm_dir / "report.json").read_text(encoding="utf-8"))
        assert report["training_performed"] is False
        assert report["loss_curve_available"] is False
        assert "no loss curve" in (arm_dir / "report.html").read_text(encoding="utf-8").lower()
    for filename in ("report.md", "report.html", "report.json", "comparison.csv"):
        assert (output / "combined" / filename).is_file()
    assert len(list((output / "combined" / "plots").glob("*.png"))) == 3


def test_report_refuses_overwrite(tmp_path: Path) -> None:
    arm_inputs, comparison = _fixture(tmp_path)
    output = tmp_path / "already-there"
    output.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        render_stage_reports(
            stage="BH1",
            dataset="transition32",
            arm_inputs=arm_inputs,
            comparison=comparison,
            output_dir=output,
            terminal=None,
            evidence=None,
            strict_execution=False,
        )


def test_strict_report_requires_terminal_evidence_and_binds_every_input(tmp_path: Path) -> None:
    arm_inputs, comparison = _fixture(tmp_path)
    scientific_paths = [path for values in arm_inputs.values() for path in values.values()] + [comparison]
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    terminal = stage_dir / "terminal.json"
    evidence = stage_dir / "evidence.json"
    _write_json(
        terminal,
        {"status": "succeeded", "passed": True, "exit_code": 0},
    )
    _write_json(
        evidence,
        {
            "protocol": STAGE_EVIDENCE_PROTOCOL,
            "stage": "BH2",
            "passed": True,
            "execution_mode": "sequential_within_arm",
            "outputs": [{"sha256": sha256_file(path)} for path in scientific_paths],
        },
    )
    result = render_stage_reports(
        stage="BH2",
        dataset="formal_dev",
        arm_inputs=arm_inputs,
        comparison=comparison,
        output_dir=tmp_path / "strict-report",
        terminal=terminal,
        evidence=evidence,
        strict_execution=True,
    )
    assert result["strict_execution"]["passed"] is True


def test_strict_report_rejects_unbound_scientific_artifact(tmp_path: Path) -> None:
    arm_inputs, comparison = _fixture(tmp_path)
    terminal = tmp_path / "terminal.json"
    evidence = tmp_path / "evidence.json"
    _write_json(terminal, {"status": "succeeded", "passed": True, "exit_code": 0})
    _write_json(
        evidence,
        {
            "protocol": STAGE_EVIDENCE_PROTOCOL,
            "stage": "BH2",
            "passed": True,
            "execution_mode": "sequential_within_arm",
            "outputs": [],
        },
    )
    with pytest.raises(ValueError, match="does not bind"):
        render_stage_reports(
            stage="BH2",
            dataset="formal_dev",
            arm_inputs=arm_inputs,
            comparison=comparison,
            output_dir=tmp_path / "bad-report",
            terminal=terminal,
            evidence=evidence,
            strict_execution=True,
        )
