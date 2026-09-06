from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import r11_teacher_swap as core  # noqa: E402


RUNNER = ROOT / "scripts" / "experiments" / "audit_r11_teacher47.py"


def _row(
    config: dict,
    *,
    target_index: int,
    condition: str,
    view_index: int,
    image_hashes: dict[str, str],
    own_like: bool,
) -> dict:
    target = next(item for item in config["targets"] if item["target_index"] == target_index)
    segment = target["target_segment"]
    query = segment["query"]
    permutation = config["permutations"][view_index]
    ordered_target = permutation.index(query["target_index"])
    logits = [0.0, 0.0, 0.0, 0.0]
    logits[ordered_target if own_like else (ordered_target + 1) % 4] = 12.0
    derived = core.recompute(logits, ordered_target, permutation)
    image_key = core.expected_image(target_index, condition)
    return {
        "schema": f"{core.PREFIX}-row.v1",
        "target_index": target_index,
        "segment_id": segment["segment_id"],
        "query_sha256": core.canonical_sha(query),
        "condition": condition,
        "view_index": view_index,
        "permutation": permutation,
        "original_answer_index": query["target_index"],
        "image_key": image_key,
        "image_sha256": image_hashes[image_key],
        "choice_logits_ordered": logits,
        **derived,
    }


def _rows(config: dict, *, donor_acts_like_own: bool = False) -> tuple[list[dict], dict[str, str]]:
    image_hashes = {"4": "4" * 64, "7": "7" * 64, "reset": "a" * 64}
    rows = []
    for target_index in (4, 7):
        for condition in config["conditions"]:
            for view_index in range(4):
                rows.append(
                    _row(
                        config,
                        target_index=target_index,
                        condition=condition,
                        view_index=view_index,
                        image_hashes=image_hashes,
                        own_like=condition == "own" or (donor_acts_like_own and condition == "donor"),
                    )
                )
    return rows, image_hashes


def test_config_is_immutable_and_preregistered_before_donor_outcome() -> None:
    config = core.load_config()
    assert core.sha256_file(core.CONFIG_PATH) == core.CONFIG_BYTES_SHA256
    assert core.canonical_sha(config) == core.CONFIG_CANONICAL_SHA256
    assert config["status"] == "preregistered_before_any_d0_donor_or_reset_reader_outcome"
    assert [item["target_index"] for item in config["targets"]] == [4, 7]
    assert config["expected_rows"] == 24
    assert config["expected_reader_forward_calls"] == 96


def test_deployment_and_zero_training_contract_are_explicit() -> None:
    config = core.load_config()
    assert config["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert config["deployment"]["instance"] == "vlm-r11-identity-h200x4-20260907"
    assert config["deployment"]["physical_gpu_count"] == 4
    assert config["deployment"]["runtime_visible_gpu_count"] == 2
    assert config["guardrails"] == {
        "optimizer_steps": 0,
        "unet_forward_calls": 0,
        "full_dreamlite_forward_calls": 0,
        "phase2_allowed": False,
        "formal_success": False,
        "shared_training_allowed": False,
    }


def test_expected_image_is_symmetric_and_reset_is_fixed() -> None:
    assert core.expected_image(4, "own") == "4"
    assert core.expected_image(4, "donor") == "7"
    assert core.expected_image(7, "donor") == "4"
    assert core.expected_image(7, "reset") == "reset"
    with pytest.raises(ValueError, match="Invalid audit cell"):
        core.expected_image(1, "own")


def test_recompute_maps_ordered_prediction_back_to_original_choice() -> None:
    permutation = [2, 1, 0, 3]
    output = core.recompute([9.0, 0.0, 0.0, 0.0], 0, permutation)
    assert output["predicted_index"] == 2
    assert output["correct"] is True
    assert 0.0 < output["ce"] < 0.001


def test_strong_sample_specific_rows_pass_d0_gate() -> None:
    config = core.load_config()
    rows, image_hashes = _rows(config)
    result = core.aggregate(rows, config, image_hashes)
    assert result["raw_row_recomputation_passed"] is True
    assert result["teacher_replay_gate"] is True
    assert result["distinguishability_gate"] is True
    assert result["d0_diagnostic_gate"] is True
    assert result["decision"] == "eligible_for_d1_implementation_and_preflight"
    assert result["formal_success"] is False
    assert result["phase2_allowed"] is False


def test_universal_trigger_behavior_fails_specificity_gate() -> None:
    config = core.load_config()
    rows, image_hashes = _rows(config, donor_acts_like_own=True)
    result = core.aggregate(rows, config, image_hashes)
    assert result["teacher_replay_gate"] is True
    assert result["distinguishability_gate"] is False
    assert result["d0_diagnostic_gate"] is False
    assert result["decision"] == "hold_d1_teacher_specificity_failed"
    assert all(target["contrasts"]["donor"]["accuracy_gap"] == 0 for target in result["targets"])


def test_row_image_binding_drift_is_rejected() -> None:
    config = core.load_config()
    rows, image_hashes = _rows(config)
    rows[0] = {**rows[0], "image_sha256": "f" * 64}
    with pytest.raises(ValueError, match="Actual image/donor binding drift"):
        core.aggregate(rows, config, image_hashes)


def test_duplicate_grid_cell_is_rejected() -> None:
    config = core.load_config()
    rows, image_hashes = _rows(config)
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match="Duplicate row"):
        core.aggregate(rows, config, image_hashes)


def test_delivery_audit_rejects_escaping_inventory_path(tmp_path: Path) -> None:
    inventory = {
        "schema": f"{core.PREFIX}-inventory.v1",
        "artifact_count": 1,
        "artifacts": [{"path": "../escape", "bytes": 0, "sha256": "0" * 64}],
    }
    (tmp_path / "artifact_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe or duplicate artifact path"):
        core.audit_delivery(tmp_path, core.load_config())


def test_runner_uses_current_host_contract_and_contains_no_training_step() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert source.count("run_r11_new_identity_condition_bridge as safety") == 2
    assert "run_r11_new_source_init_bridge as safety" not in source
    assert ".backward(" not in source
    assert "torch.optim" not in source
    assert '"unet_forward_calls": 0' in source
    assert '"optimizer_steps": 0' in source
    assert "forbidden_unet" in source
    assert '"reader_forward_calls": 96' in source
