from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSPIRE = ROOT / "scripts" / "inspire"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(INSPIRE))

from qwen_history_r4_contract import (  # noqa: E402
    AMENDMENT_SCHEMA,
    R5_AMENDMENT_SCHEMA,
    load_amendment,
)


AMENDMENT = ROOT / "configs" / "experiments" / "r5_qwen_history_same_entity_20260722.json"


def test_r5_amendment_is_lf_locked_in_git_attributes() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert (
        "configs/experiments/r5_qwen_history_same_entity_20260722.json text eol=lf"
        in attributes
    )
    assert b"\r" not in AMENDMENT.read_bytes()


def test_r5_amendment_is_prospective_and_changes_only_pair_semantics() -> None:
    amendment, digest = load_amendment(AMENDMENT)
    assert len(digest) == 64
    assert amendment["schema"] == R5_AMENDMENT_SCHEMA
    assert amendment["status"] == "prospective_before_any_r5_gpu_prediction"
    assert amendment["parent_r4_outcome"] == {
        "scientific_stage_passed": False,
        "state_swap_donor": "16/32",
        "locked_threshold": "30/32",
        "result_reinterpreted": False,
    }
    assert amendment["protocol_delta"] == {
        "parent_protocol": AMENDMENT_SCHEMA,
        "only_scientific_change": "same_entity_counterfactual_micro_lockbox",
        "r4_result_reinterpreted": False,
        "reader_prompt_arms_thresholds_unchanged": True,
        "formal_artifacts_byte_inherited": True,
        "r4_model_inference_and_scorer_unchanged": True,
    }
    assert amendment["lockbox"]["same_entity_pair_contract_validated"] is True
    assert amendment["lockbox"]["independent_generations_bitwise_identical"] is True
    assert amendment["thresholds"]["bh1_last_effective_blocking"]["state_swap_donor"] == "30/32"


def _write_mutation(tmp_path: Path, mutate) -> Path:
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    value = deepcopy(payload)
    mutate(value)
    path = tmp_path / "r5-amendment.json"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["protocol_delta"].__setitem__("only_scientific_change", "prompt_and_data"),
            "protocol delta",
        ),
        (
            lambda value: value["protocol_delta"].__setitem__("r4_result_reinterpreted", True),
            "protocol delta",
        ),
        (
            lambda value: value["lockbox"].__setitem__("seed", 20260722),
            "lockbox contract",
        ),
        (
            lambda value: value["lockbox"].__setitem__("same_entity_pair_contract_validated", False),
            "lockbox contract",
        ),
        (
            lambda value: value["lockbox"].__setitem__("transition32_pair_map_sha256", "0" * 64),
            "lockbox contract",
        ),
        (
            lambda value: value["thresholds"]["bh1_last_effective_blocking"].__setitem__("state_swap_donor", "28/32"),
            "threshold contract drifted",
        ),
        (
            lambda value: value["prompt_contract"].__setitem__("instruction", "Use a changed prompt."),
            "prompt contract",
        ),
    ],
)
def test_r5_contract_mutations_fail_closed(tmp_path: Path, mutation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_amendment(_write_mutation(tmp_path, mutation))
