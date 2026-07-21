from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "reporting"))

from render_baseline_report import FIGURE_NAMES, build_baseline_report  # noqa: E402
from score_qwen_history_baseline import (  # noqa: E402
    descriptive_metrics,
    replication_report,
    score_baseline,
    scientific_prediction_payload,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set8_rows() -> list[dict]:
    rows: list[dict] = []
    for state in range(8):
        target_text = f"value-{state}"
        for view in range(4):
            wrong = [f"wrong-{state}-{index}" for index in range(3)]
            choices = wrong[:]
            choices.insert(view, target_text)
            for condition in ("standard", "reset", "shuffle"):
                prediction = view if condition == "standard" else (view + 1) % 4
                scores = [2.0, 2.0, 2.0, 2.0]
                scores[prediction] = 0.1
                history = f"state={state}" if condition == "standard" else condition
                rows.append(
                    {
                        "schema_version": "vision_memory.qwen_full_event_history_predictions.v1",
                        "method": "qwen_full_event_history",
                        "input_mode": "blank_image",
                        "micro_sensitivity": False,
                        "episode_id": f"r3-set8-{state:02d}",
                        "query_id": f"r3-set8-{state:02d}:q0",
                        "query_ordinal": 0,
                        "probe_role": "delayed",
                        "query_turn_type": "query",
                        "choice_view_family": "reverse-cyclic4",
                        "choice_view_index": view,
                        "condition": condition,
                        "choices": choices,
                        "target_index": view,
                        "target_text": target_text,
                        "prediction_index": prediction,
                        "prediction_text": choices[prediction],
                        "choice_mean_nll": scores,
                        "history_sha256": hashlib.sha256(history.encode()).hexdigest(),
                        "prompt_sha256": hashlib.sha256(f"{history}:{view}".encode()).hexdigest(),
                        "history_token_count": 3 + state,
                        "history_utf8_bytes": len(history.encode()),
                        "constant_visual_input_bytes": 3 * 1024 * 1024 * 4,
                        "semantic_group_id": f"group-{state}",
                        "split": "dev",
                        "topic": f"topic-{state % 2}",
                        "subtype": "set",
                        "form": "separate",
                        "noop_policy": "keep",
                        "recurrence_mode": "text_history",
                        "seed": 0,
                        "diffusion_seed": 0,
                        "deterministic_ce": True,
                        "latency_seconds": 0.1 + state * 0.01,
                        "query_latency_seconds": 0.1 + state * 0.01,
                        "peak_reader_vram_gib": 7.0,
                        "peak_vram_gib": 7.0,
                    }
                )
    return rows


def write_prediction_fixture(
    root: Path,
    rows: list[dict],
    *,
    method: str = "qwen_full_event_history",
    input_mode: str = "blank_image",
    micro_sensitivity: bool = False,
) -> tuple[Path, Path]:
    predictions = root / "predictions.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    companion = root / "predictions.jsonl.report.json"
    write_json(
        companion,
        {
            "schema_version": "vision_memory.qwen_full_event_history_report.v1",
            "output": str(predictions),
            "output_sha256": sha256_file(predictions),
            "episodes_sha256": "a" * 64,
            "prediction_records": len(rows),
            "method": method,
            "input_mode": input_mode,
            "micro_sensitivity": micro_sensitivity,
            "reader_revision": "b" * 40,
            "scientific_payload_sha256": scientific_prediction_payload(rows)["sha256"],
        },
    )
    return predictions, companion


def text_only_rows() -> list[dict]:
    rows = json.loads(json.dumps(set8_rows()))
    for row in rows:
        row["method"] = "qwen_full_event_history_text_only"
        row["input_mode"] = "text_only"
        row["micro_sensitivity"] = True
        row["constant_visual_input_bytes"] = 0
    return rows


def test_set8_baseline_reuses_gate_without_checkpoint_lineage(tmp_path: Path) -> None:
    predictions, companion = write_prediction_fixture(tmp_path, set8_rows())
    report = score_baseline(
        predictions=predictions,
        prediction_report=companion,
        suite="set8",
        bootstrap_iterations=200,
    )

    assert report["passed"] is True
    assert report["micro_gate"]["correct"] == 32
    assert report["micro_gate"]["interventions"]["reset"]["drop"] == 32
    assert "artifact_provenance" not in report
    assert report["descriptive_metrics"]["paired_conditions"]["shuffle"]["accuracy_drop"] == 1.0
    assert report["descriptive_metrics"]["rotation"]["agreement_rate"] == 1.0


def test_scientific_replication_excludes_latency_but_includes_nll_and_history(tmp_path: Path) -> None:
    rows_a = set8_rows()
    rows_b = json.loads(json.dumps(rows_a))
    for row in rows_b:
        row["latency_seconds"] += 9.0
        row["peak_vram_gib"] = 99.0
    assert scientific_prediction_payload(rows_a) == scientific_prediction_payload(rows_b)
    assert replication_report(rows_a, rows_b)["passed"] is True

    rows_b[0]["choice_mean_nll"][0] += 1e-6
    mismatch = replication_report(rows_a, rows_b)
    assert mismatch["passed"] is False
    assert mismatch["identity_sets_match"] is True


def test_text_only_is_micro_sensitivity_and_formal_rejects_it(tmp_path: Path) -> None:
    blank, blank_report = write_prediction_fixture(tmp_path / "blank", set8_rows())
    text, text_report = write_prediction_fixture(
        tmp_path / "text",
        text_only_rows(),
        method="qwen_full_event_history_text_only",
        input_mode="text_only",
        micro_sensitivity=True,
    )
    micro = score_baseline(
        predictions=blank,
        prediction_report=blank_report,
        text_only_predictions=text,
        text_only_report=text_report,
        suite="set8",
        bootstrap_iterations=20,
    )
    assert micro["text_only_sensitivity"]["role"] == "micro_sensitivity_not_formal_baseline"
    assert micro["text_only_sensitivity"]["text_only_minus_blank_accuracy"] == 0.0

    with pytest.raises(ValueError, match="prohibited in formal"):
        score_baseline(
            predictions=blank,
            prediction_report=blank_report,
            text_only_predictions=text,
            text_only_report=text_report,
            suite="formal",
            bootstrap_iterations=20,
        )


def test_formal_score_reports_paired_metrics_stale_noop_and_efficiency(tmp_path: Path) -> None:
    rows = set8_rows()
    for row in rows:
        row["split"] = "test_ood"
        row["ood_group"] = "new_entity"
        if row["episode_id"] == "r3-set8-00":
            row["stale_target_index"] = (row["target_index"] + 1) % 4
            row["stale_target_text"] = row["choices"][row["stale_target_index"]]
    predictions, companion = write_prediction_fixture(tmp_path, rows)
    report = score_baseline(
        predictions=predictions,
        prediction_report=companion,
        suite="formal",
        bootstrap_iterations=200,
    )
    metrics = report["descriptive_metrics"]

    assert report["passed"] is True
    assert metrics["by_ood_group"]["new_entity"]["count"] == 32
    assert metrics["stale_answer_error"]["n"] == 4
    assert metrics["nll_margin"]["positive_target_margin_rate"] == 1.0
    assert metrics["efficiency"]["condition_scope"] == "standard"
    assert metrics["efficiency"]["history_token_count"]["n"] == 32


def test_state_swap_donor_rate_excludes_equal_target_rows() -> None:
    rows = set8_rows()
    standard_rows = [row for row in rows if row["condition"] == "standard"]
    swapped = []
    for index, standard in enumerate(standard_rows):
        donor_index = (
            standard["target_index"]
            if index % 2 == 0
            else (standard["target_index"] + 1) % 4
        )
        swapped.append(
            {
                **standard,
                "condition": "state_swap",
                "donor_target_index": donor_index,
                "prediction_index": donor_index,
            }
        )
    metrics = descriptive_metrics(
        [*rows, *swapped],
        bootstrap_iterations=20,
        bootstrap_seed=2026,
    )

    donor = metrics["state_swap_donor_answer"]
    assert donor["n"] == 16
    assert donor["count"] == 16
    assert donor["excluded_equal_target"] == 16
    assert donor["all_mapped_rows"]["n"] == 32


def test_report_renders_markdown_html_csv_json_png_and_sha_manifest(tmp_path: Path) -> None:
    predictions, companion = write_prediction_fixture(tmp_path / "source", set8_rows())
    replica_b_predictions, replica_b_companion = write_prediction_fixture(
        tmp_path / "replica-b",
        set8_rows(),
    )
    score = score_baseline(
        predictions=predictions,
        prediction_report=companion,
        replica_b_predictions=replica_b_predictions,
        replica_b_report=replica_b_companion,
        suite="set8",
        bootstrap_iterations=200,
    )
    score_path = tmp_path / "source" / "score.json"
    write_json(score_path, score)
    output = tmp_path / "report"

    result = build_baseline_report(
        predictions=predictions,
        prediction_report=companion,
        replica_b_predictions=replica_b_predictions,
        replica_b_prediction_report=replica_b_companion,
        score_report=score_path,
        output_dir=output,
        stage="BH1-Set8",
        run_id="baseline-unit",
    )

    assert result["passed"] is True
    assert (output / "report.md").is_file()
    assert (output / "report.html").is_file()
    assert "no training loss curve" in (output / "report.md").read_text(encoding="utf-8")
    assert "data:image/png;base64," in (output / "report.html").read_text(encoding="utf-8")
    for name in FIGURE_NAMES:
        payload = (output / "figures" / name).read_bytes()
        assert payload.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(payload) > 1000
    assert (output / "metrics" / "predictions.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert (output / "sources" / "replica_b_predictions.jsonl").is_file()
    assert (output / "sources" / "replica_b_prediction_report.json").is_file()
    for line in (output / "artifacts.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert sha256_file(output / relative) == digest


def test_report_refuses_overwrite_and_unbound_predictions(tmp_path: Path) -> None:
    predictions, companion = write_prediction_fixture(tmp_path / "source", set8_rows())
    score = score_baseline(
        predictions=predictions,
        prediction_report=companion,
        suite="set8",
        bootstrap_iterations=20,
    )
    score_path = tmp_path / "source" / "score.json"
    write_json(score_path, score)
    output = tmp_path / "report"
    output.mkdir()
    (output / "keep.txt").write_text("owner data", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        build_baseline_report(
            predictions=predictions,
            score_report=score_path,
            output_dir=output,
            stage="BH1",
            run_id="x",
        )

    rows = set8_rows()
    rows[0]["latency_seconds"] += 1
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        build_baseline_report(
            predictions=predictions,
            score_report=score_path,
            output_dir=tmp_path / "fresh-report",
            stage="BH1",
            run_id="x",
        )


def test_strict_report_binds_terminal_evidence_and_scientific_artifacts(tmp_path: Path) -> None:
    predictions, companion = write_prediction_fixture(tmp_path / "source", set8_rows())
    score = score_baseline(
        predictions=predictions,
        prediction_report=companion,
        suite="set8",
        bootstrap_iterations=20,
    )
    score_path = tmp_path / "source" / "score.json"
    write_json(score_path, score)
    evidence = tmp_path / "stage_evidence.json"
    write_json(
        evidence,
        {
            "passed": True,
            "outputs": [
                {"label": path.name, "sha256": sha256_file(path)}
                for path in (predictions, companion, score_path)
            ],
        },
    )
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text(json.dumps({"evidence_sha256": sha256_file(evidence)}) + "\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    terminal = tmp_path / "terminal.json"
    write_json(
        terminal,
        {
            "passed": True,
            "exit_code": 0,
            "stdout_sha256": sha256_file(stdout),
            "stderr_sha256": sha256_file(stderr),
        },
    )
    result = build_baseline_report(
        predictions=predictions,
        prediction_report=companion,
        score_report=score_path,
        output_dir=tmp_path / "strict-report",
        stage="BH1",
        run_id="strict",
        terminal_path=terminal,
        stage_evidence_path=evidence,
        stdout_path=stdout,
        stderr_path=stderr,
        strict_complete=True,
    )
    assert result["passed"] is True
