"""Synthetic unit fixtures test gates; they are not scientific experiment results."""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vision_memory.training import frozen_oracle_writer as core


def decision() -> dict:
    return {"status": "complete", "bank_action": "go", "canonicalization": "medoid",
            "controller_supervision": "single_mse", "controller_target_space": "xT", "canonical_policy": {
                "min_margin": 0.1, "max_prior_penalty": 4.0, "min_perturbations": 4,
                "min_perturbation_success": 1.0, "prior_weight": 0.01, "margin_weight": 0.01,
                "min_passing_endpoints": 2}}


@pytest.fixture
def inventory(tmp_path: Path) -> tuple[Path, Path, dict]:
    tasks, runs = [], []
    for index in range(33):
        folder = tmp_path / str(index)
        folder.mkdir()
        source = np.full((1, 2, 4, 4), index / 33, dtype=np.float32)
        np.save(folder / "source.npy", source)
        np.save(folder / "event.npy", np.array([index / 33, 1, 0, -1], dtype=np.float32))
        tasks.append({"task_id": f"task-{index}", "source_episode_id": f"episode-{index}",
                      "source_event_id": f"event-{index}", "split": "heldout" if index == 32 else "train",
                      "source_latent_path": f"{index}/source.npy", "event_embedding_path": f"{index}/event.npy",
                      "conditioning_provenance": {"input_fields": ["source_latent", "event"],
                                                  "query_answer_excluded": True}})
        for seed, offset in enumerate((0, 0.1, 0.4)):
            np.save(folder / f"x-{seed}.npy", source + 0.25 + offset)
            np.save(folder / f"z-{seed}.npy", source + offset)
            runs.append({"run_id": f"run-{index}-{seed}", "task_id": f"task-{index}", "status": "complete",
                         "optimized_xT_path": f"{index}/x-{seed}.npy", "endpoint_z_path": f"{index}/z-{seed}.npy",
                         "qa_pass": True, "functional_chain": core.CHAIN, "reader_margin": 1.0,
                         "prior_penalty": 1.0, "perturbation": {"n": 4, "passed": 4}})
    manifest = {"schema_version": 1, "tasks": tasks, "runs": runs,
                "checkpoint_hashes": {key: "a" * 64 for key in ("dreamlite", "vae", "reader")}}
    manifest_path, decision_path = tmp_path / "manifest.json", tmp_path / "decision.json"
    core.write_json(manifest_path, manifest)
    core.write_json(decision_path, decision())
    return manifest_path, decision_path, manifest


def test_bank_selects_an_observed_medoid_and_hash_binds_inventory(inventory, tmp_path):
    manifest_path, decision_path, _ = inventory
    bank = core.build_bank(manifest_path, decision_path, tmp_path / "bank.json")
    assert bank["status"] == "ready"
    assert bank["independent_task_count"] == 33
    assert all(row["targets"][0]["run_id"].endswith("-1") for row in bank["records"])
    assert all(len(row["runs"]) == 3 for row in bank["ledger"])
    assert bank["formal_success"] is False
    inputs = core.load_bank_tensors(bank)
    assert inputs[0].shape == (33, 2, 4, 4)
    assert inputs[2].shape == (33, 1, 2, 4, 4)


def test_failed_task_is_not_replaced_by_more_seeds(inventory, tmp_path):
    manifest_path, decision_path, manifest = inventory
    for row in manifest["runs"]:
        if row["task_id"] == "task-0":
            row["perturbation"] = None
    core.write_json(manifest_path, manifest)
    bank = core.build_bank(manifest_path, decision_path, tmp_path / "bank.json")
    assert bank["status"] == "blocked" and bank["selected_task_count"] == 32
    assert bank["failed_tasks_replaced"] is False
    assert bank["ledger"][0]["status"] == "blocked"
    assert all(row["status"] == "excluded" for row in bank["ledger"][0]["runs"])
    with pytest.raises(ValueError, match="partial/failed"):
        core.load_bank_tensors(bank)


def test_eight_tasks_times_eight_seeds_is_not_bank64(inventory, tmp_path):
    manifest_path, decision_path, manifest = inventory
    manifest["tasks"] = manifest["tasks"][:8]
    core.write_json(manifest_path, manifest)
    bank = core.build_bank(manifest_path, decision_path, tmp_path / "bank.json")
    assert bank["status"] == "blocked"
    assert "32-64 independent tasks" in bank["blocking_reasons"][0]


def test_source_group_leakage_is_rejected(inventory):
    tasks = copy.deepcopy(inventory[2]["tasks"])
    tasks[-1]["source_episode_id"] = tasks[0]["source_episode_id"]
    with pytest.raises(ValueError, match="Source leakage"):
        core.validate_task_inventory(tasks)


def test_same_turn_number_in_unrelated_episodes_is_allowed(inventory):
    tasks = copy.deepcopy(inventory[2]["tasks"])
    for row in tasks:
        row["source_event_id"] = "1"
    core.validate_task_inventory(tasks)


@pytest.mark.parametrize("changes", [
    {"status": "incomplete"}, {"bank_action": "modify"},
    {"controller_supervision": "behavior"}, {"canonicalization": "cluster_medoids"},
])
def test_unknown_or_incompatible_geometry_fails_closed(changes):
    with pytest.raises(ValueError):
        core.validate_geometry_decision({**decision(), **changes})


def test_artifact_mutation_after_selection_is_rejected(inventory, tmp_path):
    bank = core.build_bank(inventory[0], inventory[1], tmp_path / "bank.json")
    np.save(bank["records"][0]["source_latent_path"], np.ones((1, 2, 4, 4), np.float32))
    with pytest.raises(ValueError, match="Bank input changed"):
        core.load_bank_tensors(bank)


def test_event_mask_pooling_excludes_padding():
    result = core.event_pool(np.array([[1, 3], [3, 1], [100, 100]], dtype=np.float32), np.array([1, 1, 0]))
    np.testing.assert_array_equal(result, [2, 2])
    np.testing.assert_array_equal(core.event_pool(np.array([[2, 2]])), [2, 2])
    with pytest.raises(ValueError, match="attention mask"):
        core.event_pool(np.ones((3, 4)))


def test_set_loss_uses_nearest_valid_target_without_averaging_basins():
    prediction = torch.tensor([[[[3.0]]]], requires_grad=True)
    targets = torch.tensor([[[[[0.0]]], [[[4.0]]], [[[3.0]]]]])
    mask = torch.tensor([[True, True, False]])
    loss = core.latent_supervision(prediction, targets, mask, "set_mse")
    assert float(loss.detach()) == 1
    loss.backward()
    assert float(prediction.grad) == -2
    with pytest.raises(ValueError, match="cannot discard"):
        core.latent_supervision(prediction, targets, mask, "single_mse")


def test_controller_has_no_query_answer_or_identity_input():
    assert list(inspect.signature(core.FrozenOracleController.forward).parameters) == [
        "self", "source_latent", "event_embedding"]
    model = core.FrozenOracleController((2, 4, 4), 4, basis_rank=2, hidden=8)
    source, event = torch.randn(3, 2, 4, 4), torch.randn(3, 4)
    predicted = model(source, event)
    assert predicted.shape == source.shape
    torch.testing.assert_close(model(source[[2, 0]], event[[2, 0]]), predicted[[2, 0]])


def test_cpu_training_emits_predictions_but_cannot_claim_qa(inventory, tmp_path):
    bank_path = tmp_path / "bank.json"
    core.build_bank(inventory[0], inventory[1], bank_path)
    result = core.train_controller(bank_path=bank_path, decision_path=inventory[1], output=tmp_path / "train",
                                   steps=2, basis_rank=2, hidden=8, overfit_tasks=2, device="cpu")
    assert result["status"] == "awaiting_functional_evaluation" and result["formal_success"] is False
    assert result["prediction_count"] == 2
    spec = json.loads(Path(result["evaluation_spec"]).read_text())
    assert all(case["space"] == "xT" and case["split"] == "train" for case in spec["cases"])
    assert len(list((tmp_path / "train" / "predictions").glob("*.npy"))) == 2
    with pytest.raises(ValueError, match="functionally validate"):
        core.train_controller(bank_path=bank_path, decision_path=inventory[1], output=tmp_path / "heldout",
                               phase="heldout", steps=1, device="cpu")


def test_overfit_partial_or_unbound_qa_cannot_unlock_heldout(inventory, tmp_path):
    bank = core.build_bank(inventory[0], inventory[1], tmp_path / "bank.json")
    evaluation = {"bank_sha256": bank["bank_sha256"], "phase": "overfit", "status": "complete",
                  "functional_chain": core.CHAIN, "checkpoint_hashes": bank["checkpoint_hashes"],
                  "rows": [{"split": "train", "qa_pass": True}]}
    with pytest.raises(ValueError, match="original prediction specification"):
        core.validate_overfit_evaluation(evaluation, bank)


def test_capacity_ladder_requires_functional_failure_in_order():
    bank = {"status": "ready", "bank_sha256": "abc"}
    assert core.next_capacity_stage(bank, []) == "controller"
    base = {"stage": "controller", "bank_sha256": "abc", "functional_chain": core.CHAIN,
            "functional_evaluation_complete": True, "optimization_audit_pass": True,
            "data_audit_pass": True, "outcome": "fail"}
    assert core.next_capacity_stage(bank, [base]) == "lora_r4"
    assert core.next_capacity_stage(bank, [{**base, "outcome": "pass"}]) is None
    with pytest.raises(ValueError, match="MSE alone"):
        core.next_capacity_stage(bank, [{**base, "functional_evaluation_complete": False}])
    with pytest.raises(ValueError, match="cannot be skipped"):
        core.next_capacity_stage(bank, [{**base, "stage": "full_finetune_upper_bound"}])
