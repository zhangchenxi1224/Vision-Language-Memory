from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "reporting"))

from render_training_report import FIGURE_NAMES, build_training_report  # noqa: E402


METRICS_SCHEMA = "vision_memory.dreamlite-training-metrics.v1"
SUMMARY_SCHEMA = "vision_memory.dreamlite-training-summary.v1"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_fixture(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    training = tmp_path / "training"
    training.mkdir()
    write_json(
        training / "manifest.json",
        {
            "schema_version": 2,
            "metrics_schema": METRICS_SCHEMA,
            "summary_schema": SUMMARY_SCHEMA,
            "git_commit": "a" * 40,
            "git_dirty": False,
            "reader_resize_contract": "resize.v1",
            "model_snapshot_manifests": {"dreamlite_mobile": "b" * 64, "qwen_reader": "c" * 64},
            "arguments": {
                "seed": 0,
                "training_regime": "qa_only",
                "objective_stage": "qa",
                "learning_rate": 0.0001,
                "gradient_clip": 1.0,
            },
            "training_lineage": {"training_regime": "qa_only"},
        },
    )
    rows = [
        {
            "schema": METRICS_SCHEMA,
            "kind": "train",
            "optimizer_step": 1,
            "epoch": 0,
            "loss": 2.0,
            "qa_loss": 2.0,
            "training_regime": "qa_only",
            "objective_stage": "qa",
            "gradient_norm": 2.5,
            "group_episode_count": 8,
            "elapsed_seconds": 10.0,
        },
        {"schema": METRICS_SCHEMA, "kind": "dev", "optimizer_step": 1, "loss": 1.8},
        {
            "schema": METRICS_SCHEMA,
            "kind": "train",
            "optimizer_step": 2,
            "epoch": 0,
            "loss": 1.5,
            "qa_loss": 1.5,
            "training_regime": "qa_only",
            "objective_stage": "qa",
            "gradient_norm": 0.5,
            "group_episode_count": 8,
            "elapsed_seconds": 19.0,
        },
    ]
    (training / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    write_json(
        training / "summary.json",
        {
            "schema": SUMMARY_SCHEMA,
            "optimizer_steps": 2,
            "training_regime": "qa_only",
            "objective_stage": "qa",
            "best_dev_loss": 1.8,
            "elapsed_seconds": 19.0,
            "peak_vram_gib": {"cuda:0": 12.5, "cuda:1": 10.7},
        },
    )
    write_json(training / "state_gradient_audit.json", {"passed": True})
    terminal = tmp_path / "terminal.json"
    evidence = tmp_path / "evidence.json"
    evaluation = tmp_path / "gate_report.json"
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("training completed\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    write_json(
        terminal,
        {
            "status": "succeeded",
            "passed": True,
            "exit_code": 0,
            "finished_at": "2026-07-18T00:00:00+00:00",
            "stage": "qa8-a",
            "expected_commit": "a" * 40,
            "stdout_sha256": sha256_file(stdout),
            "stderr_sha256": sha256_file(stderr),
        },
    )
    write_json(evidence, {"passed": True, "stage": "QA8-A"})
    write_json(evaluation, {"passed": True, "accuracy": 0.9375})
    return training, {
        "terminal": terminal,
        "evidence": evidence,
        "evaluation": evaluation,
        "stdout": stdout,
        "stderr": stderr,
    }


def test_report_contains_plots_metrics_lineage_and_sha_manifest(tmp_path: Path) -> None:
    training, attachments = training_fixture(tmp_path)
    output = tmp_path / "report"
    result = build_training_report(
        training_dir=training,
        output_dir=output,
        stage="QA8-A",
        run_id="unit-run",
        terminal_path=attachments["terminal"],
        stage_evidence_path=attachments["evidence"],
        evaluation_paths=[attachments["evaluation"]],
        stdout_path=attachments["stdout"],
        stderr_path=attachments["stderr"],
        ema_span=2,
        strict_complete=True,
    )

    assert result["passed"] is True
    assert (output / "report.html").is_file()
    assert (output / "report.md").is_file()
    html_text = (output / "report.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html_text
    assert "qa_only" in html_text
    assert "unit-run" in html_text
    for name in FIGURE_NAMES:
        payload = (output / "figures" / name).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 1000

    csv_payload = (output / "metrics" / "training_curve.csv").read_bytes()
    assert csv_payload.startswith(b"\xef\xbb\xbf")
    summary = json.loads((output / "metrics" / "report_summary.json").read_text(encoding="utf-8"))
    assert summary["optimizer_steps"] == 2
    assert summary["complete"] is True
    assert summary["gradient_clip_count"] == 1
    assert summary["final_train_loss"] == 1.5

    lines = (output / "artifacts.sha256").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        digest, relative = line.split("  ", 1)
        assert sha256_file(output / relative) == digest


def test_failed_early_training_still_gets_a_report(tmp_path: Path) -> None:
    training = tmp_path / "failed-training"
    training.mkdir()
    write_json(training / "manifest.json", {"git_commit": "d" * 40, "arguments": {"seed": 0}})
    terminal = tmp_path / "terminal.json"
    stderr = tmp_path / "stderr.log"
    write_json(terminal, {"status": "failed", "passed": False, "exit_code": 1})
    stderr.write_text("RuntimeError: synthetic failure\n", encoding="utf-8")

    output = tmp_path / "failed-report"
    result = build_training_report(
        training_dir=training,
        output_dir=output,
        stage="DL-S",
        run_id="failed-unit",
        terminal_path=terminal,
        stderr_path=stderr,
    )
    assert result["passed"] is False
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "No completed optimizer-step metrics" in report
    assert "synthetic failure" in report
    assert all((output / "figures" / name).is_file() for name in FIGURE_NAMES)


def test_report_refuses_nonempty_output_directory(tmp_path: Path) -> None:
    training, _attachments = training_fixture(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ValueError, match="absent or empty"):
        build_training_report(training_dir=training, output_dir=output, stage="QA8-A", run_id="unit-run")


def test_strict_report_rejects_terminal_log_sha_mismatch(tmp_path: Path) -> None:
    training, attachments = training_fixture(tmp_path)
    attachments["stdout"].write_text("mutated after terminal\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stdout log does not match"):
        build_training_report(
            training_dir=training,
            output_dir=tmp_path / "report",
            stage="QA8-A",
            run_id="unit-run",
            terminal_path=attachments["terminal"],
            stdout_path=attachments["stdout"],
            stderr_path=attachments["stderr"],
            strict_complete=True,
        )


def test_strict_report_rejects_missing_objective_metric(tmp_path: Path) -> None:
    training, attachments = training_fixture(tmp_path)
    rows = [json.loads(line) for line in (training / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    del rows[0]["qa_loss"]
    (training / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="missing finite qa_loss"):
        build_training_report(
            training_dir=training,
            output_dir=tmp_path / "report",
            stage="QA8-A",
            run_id="unit-run",
            terminal_path=attachments["terminal"],
            stdout_path=attachments["stdout"],
            stderr_path=attachments["stderr"],
            strict_complete=True,
        )
