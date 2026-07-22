from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "reporting"))

from render_qwen_history_r4_state_swap_diagnosis import (  # noqa: E402
    MANIFEST_SCHEMA,
    PLOT_NAMES,
    SCORE_SCHEMA,
    SCHEMA,
    QWEN_R4_LAST_EFFECTIVE_EVENT,
    _state_swap_rows,
    canonical_object_sha256,
    render_diagnosis,
    scientific_prediction_payload,
    sha256_file,
)


DATASET_SHA = "2" * 64
CHOICES = ["no active preference", "ivory", "burgundy", "teal"]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prediction_rows(replica_id: str) -> list[dict]:
    rows: list[dict] = []
    kinds = ("set", "overwrite", "clear", "noop")
    conditions = ("standard", "reset", "shuffle", "state_swap")
    for kind in kinds:
        for recipient in (0, 1):
            for donor in (0, 1):
                for replicate in (0, 1):
                    episode_id = (
                        f"r4-test-{kind}-r{recipient}-d{donor}-s{replicate}"
                    )
                    donor_episode_id = f"r4-test-donor-r{donor}-{kind}-s{replicate}"
                    for condition in conditions:
                        for view in range(4):
                            view_choices = CHOICES[view:] + CHOICES[:view]
                            target = (-view) % 4
                            donor_target = (1 - view) % 4
                            if condition == "state_swap":
                                prediction = (
                                    donor_target
                                    if recipient == donor
                                    else (donor_target + 1) % 4
                                )
                            else:
                                prediction = target
                            row = {
                                "schema_version": "vision_memory.qwen_r4_history_predictions.v1",
                                "method": QWEN_R4_LAST_EFFECTIVE_EVENT,
                                "replica_id": replica_id,
                                "input_mode": "blank_image",
                                "condition": condition,
                                "episode_id": episode_id,
                                "query_id": f"{episode_id}:q0:reverse{view}",
                                "query_ordinal": 0,
                                "probe_role": "delayed",
                                "choice_view_family": "reverse-cyclic4",
                                "choice_view_index": view,
                                "dataset_sha256": DATASET_SHA,
                                "episodes_sha256": DATASET_SHA,
                                "choices": view_choices,
                                "target_index": target,
                                "target_text": CHOICES[0],
                                "prediction_index": prediction,
                                "prediction_text": view_choices[prediction],
                                "subtype": kind,
                                "form": "separate" if replicate == 0 else "mixed",
                                "topic": "accent",
                                "event_latency_seconds": 0.01,
                                "reader_latency_seconds": 0.02,
                            }
                            if condition == "state_swap":
                                row.update(
                                    {
                                        "donor_episode_id": donor_episode_id,
                                        "donor_target_index": donor_target,
                                    }
                                )
                            rows.append(row)
    assert len(rows) == 512
    return rows


def _score_fixture(predictions_a: Path, predictions_b: Path, rows_a: list[dict]) -> dict:
    scientific_sha = scientific_prediction_payload(rows_a)["sha256"]
    checks = {
        "overall": True,
        "positions": True,
        "event_kinds": True,
        "mixed": True,
        "cells": True,
        "rotation": True,
        "state_swap_donor": False,
        "reset_drop": True,
        "shuffle_drop": True,
        "clean_noop": True,
    }
    gate = {
        "suite": "transition32",
        "method": QWEN_R4_LAST_EFFECTIVE_EVENT,
        "passed": False,
        "performance_only": False,
        "data_readability_required": True,
        "state_swap": {"donor_answers": 16, "count": 32},
        "thresholds": {"donor": 30},
        "checks": checks,
    }
    gate["scientific_payload_sha256"] = canonical_object_sha256(gate)
    payload = {
        "schema": SCORE_SCHEMA,
        "suite": "transition32",
        "method": QWEN_R4_LAST_EFFECTIVE_EVENT,
        "passed": False,
        "execution_passed": False,
        "integrity": {
            "passed": True,
            "prediction_sha256": sha256_file(predictions_a),
            "prediction_report_sha256": "a" * 64,
            "replica_b_prediction_sha256": sha256_file(predictions_b),
            "replica_b_report_sha256": "b" * 64,
            "replica_a": {
                "replica_id": "A",
                "method": QWEN_R4_LAST_EFFECTIVE_EVENT,
                "prediction_records": 512,
                "unique_identities": 512,
                "dataset_sha256": DATASET_SHA,
            },
            "replica_b": {
                "replica_id": "B",
                "method": QWEN_R4_LAST_EFFECTIVE_EVENT,
                "prediction_records": 512,
                "unique_identities": 512,
                "dataset_sha256": DATASET_SHA,
            },
        },
        "replication": {
            "passed": True,
            "identity_sets_match": True,
            "bitwise_scientific_payload_match": True,
            "replica_a_scientific_payload_sha256": scientific_sha,
            "replica_b_scientific_payload_sha256": scientific_sha,
            "records_a": 512,
            "records_b": 512,
        },
        "scientific_gate": gate,
        "descriptive_metrics": {
            "state_swap_donor_answer": {"correct": 64, "count": 128, "rate": 0.5}
        },
    }
    return {**payload, "report_sha256": canonical_object_sha256(payload)}


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, list[dict], list[dict]]:
    rows_a = _prediction_rows("A")
    rows_b = _prediction_rows("B")
    predictions_a = tmp_path / "replica-a.jsonl"
    predictions_b = tmp_path / "replica-b.jsonl"
    score_path = tmp_path / "score.json"
    _write_jsonl(predictions_a, rows_a)
    _write_jsonl(predictions_b, rows_b)
    _write_json(score_path, _score_fixture(predictions_a, predictions_b, rows_a))
    return predictions_a, predictions_b, score_path, rows_a, rows_b


def test_render_state_swap_diagnosis_preserves_failure_and_derives_breakdowns(
    tmp_path: Path,
) -> None:
    predictions_a, predictions_b, score_path, _, _ = _fixture(tmp_path)
    output = tmp_path / "diagnosis"

    result = render_diagnosis(
        predictions_a=predictions_a,
        predictions_b=predictions_b,
        score_path=score_path,
        output_dir=output,
    )

    assert result["schema"] == SCHEMA
    assert result["scientific_stage_passed"] is False
    assert result["r4_failure_preserved"] is True
    assert result["locked_gate"] == {
        "donor_answers": 16,
        "count": 32,
        "rate": 0.5,
        "threshold": 30,
        "passed": False,
    }
    # The synthetic fixture balances same/cross directions within every kind;
    # the reporter must derive the split instead of hard-coding the real run.
    for kind in ("set", "overwrite", "clear", "noop"):
        assert result["by_event_kind"][kind] == {
            "donor_answers": 4, "count": 8, "rate": 0.5
        }
    assert result["by_lexical_replica_direction"]["r0->r0"]["donor_answers"] == 8
    assert result["by_lexical_replica_direction"]["r1->r1"]["donor_answers"] == 8
    assert result["by_lexical_replica_direction"]["r0->r1"]["donor_answers"] == 0
    assert result["by_lexical_replica_direction"]["r1->r0"]["donor_answers"] == 0
    assert result["diagnosis"]["classification"] == "protocol_semantic_confound_detected"
    assert result["loss_curve_available"] is False

    for name in ("diagnosis.md", "diagnosis.html", "diagnosis.csv", "diagnosis.json"):
        assert (output / name).is_file()
    for name in PLOT_NAMES:
        assert (output / "plots" / name).is_file()
    markdown = (output / "diagnosis.md").read_text(encoding="utf-8")
    assert "R4 SCIENTIFIC STATUS: FAILED" in markdown
    assert "loss curve: unavailable" in markdown.lower()
    manifest = json.loads((output / "sha256_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == MANIFEST_SCHEMA
    for item in manifest["files"]:
        path = output / item["path"]
        assert path.stat().st_size == item["size"]
        assert sha256_file(path) == item["sha256"]


def test_state_swap_diagnosis_rejects_ab_scientific_payload_drift(tmp_path: Path) -> None:
    predictions_a, predictions_b, score_path, _, rows_b = _fixture(tmp_path)
    rows_b[0]["topic"] = "tampered"
    _write_jsonl(predictions_b, rows_b)

    with pytest.raises(ValueError, match="scientific payloads are not bitwise identical"):
        render_diagnosis(
            predictions_a=predictions_a,
            predictions_b=predictions_b,
            score_path=score_path,
            output_dir=tmp_path / "diagnosis",
        )


def test_state_swap_diagnosis_rejects_incomplete_four_view_state(tmp_path: Path) -> None:
    rows = _prediction_rows("A")
    state_swap_rows = [row for row in rows if row["condition"] == "state_swap"]
    state_swap_rows.pop()

    with pytest.raises(ValueError, match="128 state-swap views"):
        _state_swap_rows(state_swap_rows)


def test_state_swap_diagnosis_rejects_score_prediction_sha_drift(tmp_path: Path) -> None:
    predictions_a, predictions_b, score_path, _, _ = _fixture(tmp_path)
    score = json.loads(score_path.read_text(encoding="utf-8"))
    score["integrity"]["prediction_sha256"] = "f" * 64
    payload = {key: value for key, value in score.items() if key != "report_sha256"}
    score["report_sha256"] = canonical_object_sha256(payload)
    _write_json(score_path, score)

    with pytest.raises(ValueError, match="Score does not bind prediction_sha256"):
        render_diagnosis(
            predictions_a=predictions_a,
            predictions_b=predictions_b,
            score_path=score_path,
            output_dir=tmp_path / "diagnosis",
        )
