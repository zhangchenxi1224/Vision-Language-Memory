from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.eval.compare_qwen_history_r4 import _assert_semantic_match  # noqa: E402
from scripts.eval.score_qwen_history_r4 import (  # noqa: E402
    EXPECTED_BLANK_IMAGE_CONTRACT,
    EXPECTED_READER_RESIZE_CONTRACT,
    EXPECTED_READER_REVISION,
    PREDICTION_REPORT_SCHEMA,
    PREDICTION_SCHEMA,
    replication_report,
    scientific_prediction_payload,
    sha256_file,
    validate_prediction_report,
    validate_rows,
)
from vision_memory.eval.r4_history_representations import (  # noqa: E402
    QWEN_R4_RAW_HISTORY,
    representation_contract_sha256,
)


def _sha(character: str) -> str:
    return character * 64


def _valid_row(*, replica_id: str = "A") -> dict[str, Any]:
    return {
        "schema_version": PREDICTION_SCHEMA,
        "episode_id": "r4-audit-episode",
        "query_id": "r4-audit-episode:q0:reverse0",
        "query_ordinal": 0,
        "probe_role": "delayed",
        "choice_view_family": "reverse-cyclic4",
        "choice_view_index": 0,
        "condition": "standard",
        "method": QWEN_R4_RAW_HISTORY,
        "input_mode": "blank_image",
        "deterministic_ce": True,
        "context_truncated": False,
        "query_text_sha256": _sha("1"),
        "dataset_sha256": _sha("2"),
        "episodes_sha256": _sha("2"),
        "representation_contract_sha256": representation_contract_sha256(
            QWEN_R4_RAW_HISTORY
        ),
        "source_event_stream_sha256": _sha("3"),
        "memory_text_sha256": _sha("4"),
        "prompt_sha256": _sha("5"),
        "chat_prompt_sha256": _sha("6"),
        "choices": ["teal", "burgundy", "ivory", "no active preference"],
        "choice_mean_nll": [0.1, 1.0, 2.0, 3.0],
        "target_index": 0,
        "prediction_index": 0,
        "target_text": "teal",
        "prediction_text": "teal",
        "source_event_count": 1,
        "retained_event_count": 1,
        "memory_token_count": 8,
        "memory_utf8_bytes": 32,
        "prompt_token_count": 24,
        "prompt_utf8_bytes": 96,
        "state_bytes": 32,
        "constant_visual_input_bytes": 12_582_912,
        "blank_image": copy.deepcopy(EXPECTED_BLANK_IMAGE_CONTRACT),
        "reader_resize_contract": EXPECTED_READER_RESIZE_CONTRACT,
        "replica_id": replica_id,
        "semantic_group_id": "r4-audit-group",
        "subtype": "set",
        "form": "separate",
        "split": "dev",
        "ood_group": None,
    }


def _write_prediction_bundle(
    root: Path, rows: list[dict[str, Any]]
) -> tuple[Path, Path, dict[str, Any]]:
    predictions = root / "predictions.jsonl"
    predictions.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": PREDICTION_REPORT_SCHEMA,
        "status": "complete",
        "method": QWEN_R4_RAW_HISTORY,
        "input_mode": "blank_image",
        "reader_revision": EXPECTED_READER_REVISION,
        "reader_resize_contract": EXPECTED_READER_RESIZE_CONTRACT,
        "blank_image": copy.deepcopy(EXPECTED_BLANK_IMAGE_CONTRACT),
        "representation_contract_sha256": representation_contract_sha256(
            QWEN_R4_RAW_HISTORY
        ),
        "output_sha256": sha256_file(predictions),
        "prediction_records": len(rows),
        "dataset_sha256": _sha("2"),
        "episodes_sha256": _sha("2"),
        "scientific_payload_sha256": scientific_prediction_payload(rows)["sha256"],
        "deterministic_ce": True,
        "context_truncation_policy": "fail_closed",
    }
    report_path = root / "predictions.jsonl.report.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return predictions, report_path, report


def test_exact_row_and_report_audit_contract_passes(tmp_path: Path) -> None:
    rows = [_valid_row()]
    integrity = validate_rows(rows, method=QWEN_R4_RAW_HISTORY)
    assert integrity["dataset_sha256"] == _sha("2")
    predictions, report_path, _ = _write_prediction_bundle(tmp_path, rows)
    report = validate_prediction_report(
        report_path,
        predictions,
        rows,
        method=QWEN_R4_RAW_HISTORY,
    )
    assert report["blank_image"] == {
        "shape": [3, 1024, 1024],
        "dtype": "float32",
        "value": 0.5,
        "bytes": 12_582_912,
        "reader_resize_contract": EXPECTED_READER_RESIZE_CONTRACT,
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("query_text_sha256", None),
        ("dataset_sha256", None),
        ("episodes_sha256", None),
        ("source_event_stream_sha256", None),
    ],
)
def test_row_rejects_missing_or_invalid_scientific_binding(
    field: str, replacement: object
) -> None:
    row = _valid_row()
    row[field] = replacement
    with pytest.raises(ValueError):
        validate_rows([row], method=QWEN_R4_RAW_HISTORY)


def test_row_rejects_dataset_episodes_sha_divergence() -> None:
    row = _valid_row()
    row["episodes_sha256"] = _sha("7")
    with pytest.raises(ValueError, match="dataset/episodes SHA binding drifted"):
        validate_rows([row], method=QWEN_R4_RAW_HISTORY)


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("shape", [1, 3, 1024, 1024]),
        ("dtype", "bfloat16"),
        ("value", 0.0),
        ("bytes", 12_582_911),
        ("reader_resize_contract", "drifted-resize-contract"),
    ],
)
def test_row_rejects_each_blank_image_contract_drift(key: str, replacement: object) -> None:
    row = _valid_row()
    row["blank_image"][key] = replacement
    with pytest.raises(ValueError, match="blank-image contract drifted"):
        validate_rows([row], method=QWEN_R4_RAW_HISTORY)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("constant_visual_input_bytes", 0, "byte accounting drifted"),
        ("reader_resize_contract", "drifted", "Reader resize contract drifted"),
    ],
)
def test_row_rejects_flat_blank_contract_drift(
    field: str, replacement: object, message: str
) -> None:
    row = _valid_row()
    row[field] = replacement
    with pytest.raises(ValueError, match=message):
        validate_rows([row], method=QWEN_R4_RAW_HISTORY)


@pytest.mark.parametrize(
    "field",
    [
        "query_text_sha256",
        "source_event_stream_sha256",
        "dataset_sha256",
        "episodes_sha256",
    ],
)
def test_cross_arm_pairing_rejects_each_scientific_binding_drift(field: str) -> None:
    row_a = _valid_row()
    row_b = _valid_row()
    row_b[field] = _sha("7")
    with pytest.raises(ValueError, match=field):
        _assert_semantic_match(("audit",), (row_a, row_b))


@pytest.mark.parametrize(
    "field",
    [
        "query_text_sha256",
        "source_event_stream_sha256",
        "dataset_sha256",
        "episodes_sha256",
        "blank_image",
        "reader_resize_contract",
        "constant_visual_input_bytes",
    ],
)
def test_ab_scientific_payload_includes_every_new_binding(field: str) -> None:
    row_a = _valid_row(replica_id="A")
    row_b = _valid_row(replica_id="B")
    assert replication_report([row_a], [row_b])["passed"] is True
    if field == "blank_image":
        row_b[field] = {**row_b[field], "value": 0.0}
    elif field in {"reader_resize_contract"}:
        row_b[field] = "drifted"
    elif field == "constant_visual_input_bytes":
        row_b[field] = 0
    else:
        row_b[field] = _sha("7")
    assert replication_report([row_a], [row_b])["passed"] is False


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("blank_image", "shape", [1, 3, 1024, 1024]),
        ("blank_image", "dtype", "bfloat16"),
        ("blank_image", "value", 0.0),
        ("blank_image", "bytes", 0),
        ("blank_image", "reader_resize_contract", "drifted"),
        ("report", "reader_resize_contract", "drifted"),
        ("report", "dataset_sha256", "7" * 64),
        ("report", "episodes_sha256", "7" * 64),
    ],
)
def test_prediction_report_rejects_each_contract_drift(
    tmp_path: Path, section: str, field: str, replacement: object
) -> None:
    rows = [_valid_row()]
    predictions, report_path, report = _write_prediction_bundle(tmp_path, rows)
    if section == "blank_image":
        report["blank_image"][field] = replacement
    else:
        report[field] = replacement
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        validate_prediction_report(
            report_path,
            predictions,
            rows,
            method=QWEN_R4_RAW_HISTORY,
        )
