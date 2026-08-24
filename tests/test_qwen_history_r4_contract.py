from __future__ import annotations

import json
from copy import deepcopy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSPIRE = ROOT / "scripts" / "inspire"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(INSPIRE))

from qwen_history_r4_contract import (  # noqa: E402
    ARM_METHODS,
    load_amendment,
    validate_scientific_command,
    verify_bh2_last_effective_dev_gate,
)
from vision_memory.data import REVERSE_CYCLIC4, permutation_family_sha256  # noqa: E402
from vision_memory.eval.r4_history_representations import (  # noqa: E402
    QWEN_R4_LAST_EFFECTIVE_EVENT,
    QWEN_R4_OPERATION_TAGGED_HISTORY,
    QWEN_R4_RAW_HISTORY,
    R4_HISTORY_TASK_INSTRUCTION,
    representation_contract_sha256,
)


AMENDMENT = ROOT / "configs" / "experiments" / "r4_qwen_history_comparison_20260722.json"


def test_prospective_amendment_locks_data_inventory_and_router_privilege() -> None:
    amendment, digest = load_amendment(AMENDMENT)
    assert len(digest) == 64
    assert amendment["lockbox"]["independent_generations_bitwise_identical"] is True
    assert amendment["lockbox"]["pre_bh2_test_policy"] == {
        "scope": ["formal_test_id", "formal_test_ood"],
        "allowed_before_bh2_unlock": ["sha256_byte_binding"],
        "json_semantic_parse_before_bh2_unlock": False,
        "scoring_before_bh2_unlock": False,
        "metric_access_before_bh2_unlock": False,
        "evaluation_before_bh2_unlock": False,
        "unlock_condition": "BH2_last_effective_dev_gate_passed",
    }
    assert amendment["expected_inventory"]["smoke4"]["prediction_records_per_arm"] == 16
    assert amendment["expected_inventory"]["transition32"]["prediction_records_per_arm"] == 512
    assert amendment["expected_inventory"]["formal_dev"]["prediction_records_per_arm"] == 5008
    assert amendment["expected_inventory"]["formal_test_id"]["prediction_records_per_arm"] == 9952
    assert amendment["arms"]["raw"]["oracle_router_metadata"] is False
    assert amendment["arms"]["tagged"]["oracle_router_metadata"] is True
    assert amendment["arms"]["last_effective"]["oracle_router_metadata"] is True
    assert amendment["research_role"]["teacher_or_oracle_state_used"] is False


def test_amendment_representation_sha_matches_implementation() -> None:
    amendment, _ = load_amendment(AMENDMENT)
    expected = {
        "raw": representation_contract_sha256(QWEN_R4_RAW_HISTORY),
        "tagged": representation_contract_sha256(QWEN_R4_OPERATION_TAGGED_HISTORY),
        "last_effective": representation_contract_sha256(QWEN_R4_LAST_EFFECTIVE_EVENT),
    }
    assert {name: arm["representation_contract_sha256"] for name, arm in amendment["arms"].items()} == expected
    assert amendment["reader"]["choice_family_sha256"] == permutation_family_sha256(REVERSE_CYCLIC4)
    assert amendment["prompt_contract"]["instruction"] == R4_HISTORY_TASK_INSTRUCTION


def _write_mutated_amendment(tmp_path: Path, mutate) -> Path:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    mutated = deepcopy(payload)
    mutate(mutated)
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["prompt_contract"].__setitem__("instruction", "Use the memory."),
            "prompt contract",
        ),
        (
            lambda value: value["prompt_contract"].__setitem__("instruction_sha256", "0" * 64),
            "prompt contract",
        ),
        (
            lambda value: value["reader"].__setitem__("choice_family_sha256", "0" * 64),
            "Reader contract",
        ),
        (
            lambda value: value["arms"]["tagged"].__setitem__(
                "representation_contract_sha256", "0" * 64
            ),
            "representation_contract_sha256",
        ),
        (
            lambda value: value["arms"]["raw"].__setitem__(
                "representation_contract_sha256", "0" * 64
            ),
            "representation_contract_sha256",
        ),
        (
            lambda value: value["arms"]["last_effective"].__setitem__(
                "representation_contract_sha256", "0" * 64
            ),
            "representation_contract_sha256",
        ),
        (
            lambda value: value["blank_image"].__setitem__("rgb_value", 0.0),
            "visual input contract drifted",
        ),
        (
            lambda value: value["thresholds"]["bh1_last_effective_blocking"].__setitem__(
                "overall", "121/128"
            ),
            "threshold contract drifted",
        ),
        (
            lambda value: value["stages"]["BH2"].__setitem__(
                "last_effective_gate_required", False
            ),
            "gate policy drifted",
        ),
        (
            lambda value: value["lockbox"]["pre_bh2_test_policy"].__setitem__(
                "json_semantic_parse_before_bh2_unlock", True
            ),
            "pre-BH2 test policy",
        ),
    ],
)
def test_amendment_contract_mutations_fail_closed(tmp_path: Path, mutation, message: str) -> None:
    path = _write_mutated_amendment(tmp_path, mutation)
    with pytest.raises(ValueError, match=message):
        load_amendment(path)


def test_obsolete_test_unread_claim_is_rejected(tmp_path: Path) -> None:
    def mutate(value: dict[str, object]) -> None:
        lockbox = value["lockbox"]
        assert isinstance(lockbox, dict)
        lockbox["test_contents_unread_before_bh2_unlock"] = True

    path = _write_mutated_amendment(tmp_path, mutate)
    with pytest.raises(ValueError, match="obsolete misleading test-unread"):
        load_amendment(path)


def test_scientific_command_allowlist_rejects_privileged_fragments() -> None:
    valid = [
        "/abs/python",
        "/abs/repo/scripts/eval/qwen_history_r4.py",
        "--method",
        ARM_METHODS["raw"],
        "--choice-view-family",
        "reverse-cyclic4",
        "--strict-determinism",
    ]
    validate_scientific_command(valid)
    with pytest.raises(ValueError, match="forbidden"):
        validate_scientific_command([*valid, "--teacher-sidecar", "/secret.json"])


def _formal_score(accuracy: float, count: int = 5008) -> dict[str, object]:
    return {
        "schema": "vlm.qwen-history-r4-score.v1",
        "method": ARM_METHODS["last_effective"],
        "suite": "formal",
        "passed": True,
        "integrity": {"passed": True},
        "replication": {"passed": True, "bitwise_scientific_payload_match": True},
        "scientific_gate": {"data_readability_required": False, "passed": None},
        "descriptive_metrics": {"standard": {"accuracy": accuracy, "count": count}},
    }


def test_bh2_last_effective_dev_gate_is_independent_and_fail_closed(tmp_path: Path) -> None:
    passed = tmp_path / "passed.json"
    passed.write_text(json.dumps(_formal_score(0.95)), encoding="utf-8")
    report = verify_bh2_last_effective_dev_gate(passed)
    assert report["bh2_last_effective_dev_gate"]["passed"] is True

    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps(_formal_score(0.949)), encoding="utf-8")
    with pytest.raises(ValueError, match="data-readability gate failed"):
        verify_bh2_last_effective_dev_gate(failed)

    wrong_count = tmp_path / "wrong-count.json"
    wrong_count.write_text(json.dumps(_formal_score(1.0, 5007)), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 5008"):
        verify_bh2_last_effective_dev_gate(wrong_count)
