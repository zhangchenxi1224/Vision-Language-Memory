from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import (  # noqa: E402
    compare_r11_new_source_init_bridge as comparison,
)
from vision_memory.data import REVERSE_CYCLIC4  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _metric(step: int, *, mse: float = 1.0) -> dict:
    distance = comparison.core.bridge_distance_statistics(
        mse=mse,
        m0_mse=4.0,
        tensor_numel=65536,
    )
    return {
        "schema": comparison.TRAINER_METRICS_SCHEMA,
        "kind": "optimizer_step",
        "optimizer_step": step,
        "target_index": comparison.core.BRIDGE_TARGET_INDEX,
        "target_segment_id": comparison.core.BRIDGE_TARGET_SEGMENT_ID,
        "objective": "canonical_r11_latent_fp32_mean_mse",
        "loss_before_step": mse,
        **distance,
        "gradient_norm": 1.0,
        "gradient_nonzero_fraction": 1.0,
        "x_T_update_norm": 0.1,
        "elapsed_seconds": float(step),
        "trajectory_points": 5,
        "dreamlite_denoising_steps": 4,
        "effective_sigmas": list(comparison.EXPECTED_SIGMAS),
        "full_dreamlite_forward_executed": True,
        "gradient_clipping_applied": False,
        "learning_rate": comparison.core.bridge_optimizer_learning_rate(step),
        "weight_decay": 0.0,
        "reader_gradient_calls": 0,
        "teacher_tensor_sha256": comparison.core.BRIDGE_TEACHER_TENSOR_SHA256,
    }


def _target_segment() -> dict:
    return {
        "schema": "vision_memory.r5-compose-segments.v1",
        "segment_id": comparison.core.BRIDGE_TARGET_SEGMENT_ID,
        "events": [{"event_kind": "set"}],
        "query_source_episode_id": "episode",
        "query_turn_index": 3,
        "query_gap": 1,
        "query": {
            "choices": ["none", "ambient", "blues", "jazz"],
            "target_index": 1,
        },
    }


def _reader_rows(path: Path) -> list[dict]:
    rows = []
    groups = (
        ("canonical_teacher", "teacher"),
        ("m0", "normal"),
        ("m0", "reset"),
        (comparison.core.BRIDGE_PRIMARY_ENDPOINT, "normal"),
        (comparison.core.BRIDGE_PRIMARY_ENDPOINT, "reset"),
    )
    target = 1
    for checkpoint, condition in groups:
        for view_index, permutation in enumerate(REVERSE_CYCLIC4):
            ordered_target = permutation.index(target)
            logits = [-2.0] * 4
            logits[ordered_target] = 8.0
            maximum = max(logits)
            ce = maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[ordered_target]
            rows.append(
                {
                    "schema": "vision_memory.r5-compose-causal-evaluation.v1",
                    "suite": comparison.controller.trainer.SUITE,
                    "item_id": comparison.core.BRIDGE_TARGET_SEGMENT_ID,
                    "pair_unit": comparison.core.BRIDGE_TARGET_SEGMENT_ID,
                    "donor_item_id": None,
                    "episode_id": "episode",
                    "query_id": "episode:3",
                    "query_gap": 1,
                    "updater_count": 1,
                    "target_event_kind": "set",
                    "checkpoint": checkpoint,
                    "condition": condition,
                    "view_index": view_index,
                    "permutation": list(permutation),
                    "target_index": target,
                    "choice_logits_ordered": logits,
                    "predicted_index": target,
                    "predicted_text": "ambient",
                    "target_text": "ambient",
                    "ce": ce,
                    "correct": True,
                    "margin": 10.0,
                }
            )
    _write_jsonl(path, rows)
    return rows


def test_raw_metrics_are_recomputed_not_trusted(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    rows = [_metric(step, mse=4.0 / (step + 1)) for step in range(1, 257)]
    _write_jsonl(path, rows)
    observed, record = comparison._validate_metrics(path)
    assert len(observed) == 256
    assert record["m0_mse"] == pytest.approx(4.0)
    rows[100]["mse_ratio_to_m0"] = 0.0
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="distance arithmetic"):
        comparison._validate_metrics(path)


def test_checkpoint_tensor_distance_and_hashes_are_recomputed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "images").mkdir()
    (run / "checkpoint_hashes").mkdir()
    teacher = torch.zeros((1, 4, 128, 128), dtype=torch.float32)
    x_t = torch.ones_like(teacher)
    z_t = torch.full_like(teacher, 0.5)
    source = torch.zeros_like(teacher)
    trajectory = (z_t.clone(), *[torch.full_like(teacher, float(index)) for index in range(1, 4)], z_t.clone())
    initialization = {
        "source_latents_fp32": source,
        "reconstructed_start_state_compute": z_t.clone(),
        "actual_effective_sigmas": list(comparison.EXPECTED_SIGMAS),
        "compute_dtype": "torch.float32",
        "compute_device_type": "cpu",
    }
    distance = comparison.core.bridge_distance_statistics(
        mse=0.25,
        m0_mse=1.0,
        tensor_numel=teacher.numel(),
    )
    manifest_sha = "1" * 64
    condition_sha = "2" * 64
    checkpoint_step = 256
    optimizer_state = {
        "state": {
            0: {
                "step": torch.tensor(float(checkpoint_step)),
                "exp_avg": torch.zeros_like(teacher),
                "exp_avg_sq": torch.zeros_like(teacher),
            }
        },
        "param_groups": [
            {
                "lr": comparison.core.bridge_optimizer_learning_rate(checkpoint_step),
                "params": [0],
            }
        ],
    }
    optimizer_state_sha256 = comparison.canonical_object_sha256(optimizer_state)
    payload = {
        "schema": comparison.TRAINER_CHECKPOINT_SCHEMA,
        "optimizer_step": checkpoint_step,
        "objective": "canonical_r11_latent_fp32_mean_mse",
        "x_T_fp32": x_t,
        "z_t_fp32": z_t,
        "trajectory_fp32": trajectory,
        "effective_sigmas": list(comparison.EXPECTED_SIGMAS),
        "compute_dtype": "torch.float32",
        "compute_device_type": "cpu",
        "trajectory_point0_formula_valid": True,
        "optimizer": optimizer_state,
        "optimizer_state_sha256": optimizer_state_sha256,
        "distance_statistics": dict(distance),
        "teacher_tensor_sha256": comparison.core.BRIDGE_TEACHER_TENSOR_SHA256,
        "manifest_sha256": manifest_sha,
        "condition_artifact_sha256": condition_sha,
        "tensor_sha256": {
            "x_T_fp32": canonical_tensor_sha256(x_t),
            "z_t_fp32": canonical_tensor_sha256(z_t),
            "trajectory_fp32": [canonical_tensor_sha256(value) for value in trajectory],
        },
    }
    checkpoint = run / "checkpoints" / "step-256.pt"
    torch.save(payload, checkpoint)
    image = run / "images" / "step-256.png"
    image.write_bytes(b"png")
    record = {
        "schema": comparison.TRAINER_CHECKPOINT_HASH_SCHEMA,
        "optimizer_step": checkpoint_step,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": comparison._sha256(checkpoint),
        "png_bytes": image.stat().st_size,
        "png_sha256": comparison._sha256(image),
        "trajectory_points": 5,
        "effective_sigmas": list(comparison.EXPECTED_SIGMAS),
        "distance_statistics": dict(distance),
        "tensor_sha256": payload["tensor_sha256"],
        "optimizer_state_sha256": optimizer_state_sha256,
    }
    record_path = run / "checkpoint_hashes" / "step-256.json"
    record.update({"compute_dtype": "torch.float32", "compute_device_type": "cpu",
                   "trajectory_point0_formula_valid": True})
    _write_json(record_path, record)
    validated = comparison._validate_checkpoint(
        run,
        step=checkpoint_step,
        teacher=teacher,
        m0_mse=1.0,
        manifest_sha256=manifest_sha,
        condition_sha256=condition_sha,
        initialization=initialization,
    )
    assert validated["distance_statistics"]["mse"] == pytest.approx(0.25)
    original_payload = copy.deepcopy(payload)
    original_record = copy.deepcopy(record)
    for mutation in ("current_x_t", "start", "final", "actual_sigma", "compute_dtype"):
        changed = copy.deepcopy(original_payload)
        if mutation == "current_x_t":
            changed["x_T_fp32"] += 1.0
        elif mutation == "start":
            changed["trajectory_fp32"][0].add_(1.0)
        elif mutation == "final":
            changed["trajectory_fp32"][-1].add_(1.0)
        elif mutation == "actual_sigma":
            changed["effective_sigmas"][0] = 0.5000001
        else:
            changed["compute_dtype"] = "torch.bfloat16"
        changed["tensor_sha256"] = {
            "x_T_fp32": canonical_tensor_sha256(changed["x_T_fp32"]),
            "z_t_fp32": canonical_tensor_sha256(changed["z_t_fp32"]),
            "trajectory_fp32": [canonical_tensor_sha256(value) for value in changed["trajectory_fp32"]],
        }
        torch.save(changed, checkpoint)
        changed_record = {
            **original_record,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": comparison._sha256(checkpoint),
            "tensor_sha256": changed["tensor_sha256"],
            "effective_sigmas": changed["effective_sigmas"],
        }
        _write_json(record_path, changed_record)
        with pytest.raises(ValueError, match="tensor payload contract|current x_T/trajectory binding"):
            comparison._validate_checkpoint(
                run, step=checkpoint_step, teacher=teacher, m0_mse=1.0,
                manifest_sha256=manifest_sha, condition_sha256=condition_sha,
                initialization=initialization,
            )
    torch.save(payload, checkpoint)
    record["checkpoint_bytes"] = checkpoint.stat().st_size
    record["checkpoint_sha256"] = comparison._sha256(checkpoint)
    record["distance_statistics"]["mse"] = 0.0
    _write_json(record_path, record)
    with pytest.raises(ValueError, match="distance arithmetic"):
        comparison._validate_checkpoint(
            run,
            step=checkpoint_step,
            teacher=teacher,
            m0_mse=1.0,
            manifest_sha256=manifest_sha,
            condition_sha256=condition_sha,
            initialization=initialization,
        )
    record["distance_statistics"] = distance
    payload["optimizer"]["state"][0]["step"] = torch.tensor(255.0)
    payload["optimizer_state_sha256"] = comparison.canonical_object_sha256(payload["optimizer"])
    torch.save(payload, checkpoint)
    record["checkpoint_bytes"] = checkpoint.stat().st_size
    record["checkpoint_sha256"] = comparison._sha256(checkpoint)
    record["optimizer_state_sha256"] = payload["optimizer_state_sha256"]
    _write_json(record_path, record)
    with pytest.raises(ValueError, match="tensor payload contract"):
        comparison._validate_checkpoint(
            run,
            step=checkpoint_step,
            teacher=teacher,
            m0_mse=1.0,
            manifest_sha256=manifest_sha,
            condition_sha256=condition_sha,
            initialization=initialization,
        )


def test_reader_ce_argmax_and_margin_are_recomputed(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_rows.jsonl"
    rows = _reader_rows(path)
    result = comparison._validate_rows(path, target_segment=_target_segment())
    assert len(result["rows"]) == 20
    assert result["statistics"][f"{comparison.core.BRIDGE_PRIMARY_ENDPOINT}/normal"]["accuracy"] == 1.0
    rows[0]["correct"] = False
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="correctness"):
        comparison._validate_rows(path, target_segment=_target_segment())


def test_self_consistent_wrong_answer_label_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "evaluation_rows.jsonl"
    rows = _reader_rows(path)
    for row in rows:
        permutation = tuple(row["permutation"])
        ordered_target = permutation.index(0)
        logits = [-2.0] * 4
        logits[ordered_target] = 8.0
        maximum = max(logits)
        row["target_index"] = 0
        row["target_text"] = "none"
        row["predicted_index"] = 0
        row["predicted_text"] = "none"
        row["choice_logits_ordered"] = logits
        row["ce"] = maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[ordered_target]
        row["correct"] = True
        row["margin"] = 10.0
    _write_jsonl(path, rows)
    with pytest.raises(ValueError, match="group contract"):
        comparison._validate_rows(path, target_segment=_target_segment())


def test_inventory_includes_nested_inventory_and_rejects_extra_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    nested = root / "run"
    nested.mkdir(parents=True)
    (nested / "artifact_inventory.json").write_text("{}\n", encoding="utf-8")
    (root / "terminal.json").write_text("{}\n", encoding="utf-8")
    comparison._write_inventory(root)
    validated = comparison._validate_inventory(
        root,
        schema=comparison.INVENTORY_SCHEMA,
    )
    assert validated["artifact_count"] == 2
    (root / "unlisted.txt").write_text("new", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        comparison._validate_inventory(root, schema=comparison.INVENTORY_SCHEMA)


def test_sigma_validation_uses_the_locked_phase1a_tolerance() -> None:
    assert comparison._sigmas_exact([0.5000015, 0.375, 0.25, 0.125])
    assert not comparison._sigmas_exact([0.500003, 0.375, 0.25, 0.125])
    assert not comparison._equal_float(True, 1.0)


def test_preflight_rejects_top_level_pass_with_false_freeze_subgate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    rows_path = run / comparison.controller.ROWS_FILE
    rows = _reader_rows(rows_path)[:4]
    _write_jsonl(rows_path, rows)
    teacher_stats = comparison.core.reader_checkpoint_statistics(
        rows,
        checkpoint="canonical_teacher",
        condition="teacher",
    )
    required = {
        "four_dreamlite_steps": True,
        "effective_sigmas_exact": True,
        "finite_nonzero_x_T_gradient": True,
        "only_x_T_fp32_trainable": True,
        "frozen_gradients_absent": True,
        "step0_parity_valid": True,
        "source_only_initialization_artifact_valid": True,
        "teacher_binding_valid": True,
        "teacher_replay_gate": True,
        "checkpoint_hash_valid": True,
        "snapshots_unchanged": True,
    }
    summary = {
        "schema": comparison.TRAINER_PREFLIGHT_SCHEMA,
        "mode": "technical-preflight",
        "passed": True,
        "bridge_result_evaluated": False,
        "formal_success_gate": False,
        "phase2_allowed": False,
        "audit": {
            "optimizer_steps": 0,
            "full_forward_calls": 1,
            "backward_calls": 1,
            **required,
        },
        "teacher_replay_statistics": teacher_stats,
        "initial_distance_statistics": {"m0_mse": 1.0},
    }
    _write_json(run / comparison.controller.SUMMARY_FILE, summary)
    _write_json(run / comparison.controller.PREFLIGHT_FILE, summary)
    _write_json(
        run / "terminal.json",
        {
            "schema": comparison.TRAINER_TERMINAL_SCHEMA,
            "technical_gate": True,
            "bridge_diagnostic_gate": None,
        },
    )
    monkeypatch.setattr(
        comparison,
        "_validate_checkpoint",
        lambda *_args, **_kwargs: {"optimizer_step": 0},
    )
    result = comparison._validate_preflight(
        run,
        teacher=torch.zeros((1, 4, 128, 128)),
        manifest_sha256="1" * 64,
        condition_sha256="2" * 64,
        target_segment=_target_segment(),
        initialization={},
    )
    assert result["passed"]
    summary["audit"]["frozen_gradients_absent"] = False
    _write_json(run / comparison.controller.SUMMARY_FILE, summary)
    _write_json(run / comparison.controller.PREFLIGHT_FILE, summary)
    with pytest.raises(ValueError, match="raw contract"):
        comparison._validate_preflight(
            run,
            teacher=torch.zeros((1, 4, 128, 128)),
            manifest_sha256="1" * 64,
            condition_sha256="2" * 64,
            target_segment=_target_segment(),
            initialization={},
        )


def test_compare_integration_never_promotes_bridge_pass_to_scientific_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_root = tmp_path / "preflight"
    formal_root = tmp_path / "formal"
    preflight_run = preflight_root / "run"
    formal_run = formal_root / "run"
    preflight_run.mkdir(parents=True)
    (formal_run / "checkpoints").mkdir(parents=True)
    (formal_run / "images").mkdir()
    for path, payload in (
        (preflight_root / "terminal.json", b"preflight-terminal"),
        (preflight_root / "artifact_inventory.json", b"preflight-inventory"),
        (preflight_run / comparison.controller.SUMMARY_FILE, b"preflight-summary"),
        (formal_run / "checkpoints" / "step-256.pt", b"endpoint"),
        (formal_run / "endpoint_raw.pt", b"endpoint"),
        (formal_run / "images" / "step-256.png", b"image"),
        (formal_run / "endpoint_raw.png", b"image"),
        (formal_run / "environment.txt", b"environment"),
        (formal_run / "runtime.json", b"{}\n"),
        (formal_root / "stdout.log", b"stdout"),
        (formal_root / "stderr.log", b""),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    code_hashes = {
        "trainer_sha256": "1" * 64,
        "controller_sha256": "2" * 64,
        "core_sha256": "3" * 64,
        "config_sha256": comparison.core.BRIDGE_CONFIG_FILE_SHA256,
    }
    prerequisite = {
        "passed": True,
        "terminal_sha256": comparison._sha256(preflight_root / "terminal.json"),
        "inventory_sha256": comparison._sha256(preflight_root / "artifact_inventory.json"),
        "summary_sha256": comparison._sha256(preflight_run / comparison.controller.SUMMARY_FILE),
    }
    teacher_stats = {
        "row_count": 4,
        "all_four_correct": True,
        "mean_ce": 0.0,
        "accuracy": 1.0,
    }
    m0_stats = {
        "row_count": 4,
        "all_four_correct": False,
        "mean_ce": 1.0,
        "accuracy": 0.0,
    }
    distance = comparison.core.bridge_distance_statistics(
        mse=0.0001,
        m0_mse=1.0,
        tensor_numel=65536,
    )
    step0_distance = comparison.core.bridge_distance_statistics(
        mse=1.0,
        m0_mse=1.0,
        tensor_numel=65536,
    )
    technical = {
        "schema": comparison.TRAINER_TECHNICAL_SCHEMA,
        "passed": True,
        "optimizer_step_records": 256,
        "checkpoint_steps_observed": list(comparison.core.BRIDGE_CHECKPOINT_STEPS),
        "trainable_parameter_names": ["x_T_fp32"],
        "minimum_gradient_norm": 1.0,
        "minimum_gradient_nonzero_fraction": 1.0,
        **{
            key: True
            for key in (
                "receipts_exact",
                "steps_contiguous",
                "four_step_schedule_exact",
                "finite_metrics",
                "finite_nonzero_gradient_every_step",
                "only_x_T_fp32_trainable",
                "frozen_gradients_absent",
                "snapshots_unchanged",
                "optimizer_contract_valid",
                "optimizer_lr_schedule_exact",
                "source_only_initialization_artifact_valid",
                "trajectory_point0_binding_valid_every_checkpoint",
                "gradient_clipping_absent",
                "checkpoint_hashes_valid",
                "condition_artifact_valid",
                "step0_parity_valid",
                "teacher_binding_valid",
                "artifact_contract_valid",
            )
        },
    }
    gates = {
        "technical_gate": True,
        "teacher_replay_gate": True,
        "bridge_distance_gate": True,
        "endpoint_reader_transfer_gate": True,
        "bridge_diagnostic_gate": True,
        "formal_success_gate": False,
    }
    schedule_audit = comparison.core.bridge_initialization_hypothesis_audit(
        endpoint_mse=float(distance["mse"]),
        endpoint_reader_mean_ce=float(teacher_stats["mean_ce"]),
        technical_gate=True,
        teacher_replay_gate=True,
        distance_pass=True,
        reader_transfer_pass=True,
    )
    summary = {
        "schema": comparison.TRAINER_SUMMARY_SCHEMA,
        "status": "completed",
        "mode": "formal",
        "optimizer_steps": 256,
        "primary_endpoint": comparison.core.BRIDGE_PRIMARY_ENDPOINT,
        "technical_gate": technical,
        "gates": gates,
        "decision": comparison.core.bridge_decision(
            distance_pass=True,
            reader_transfer_pass=True,
        ),
        "secondary_solver_hypothesis_audit": schedule_audit,
        "secondary_solver_hypothesis_decision": comparison.core.bridge_initialization_hypothesis_decision(
            distance_pass=True,
            reader_transfer_pass=True,
            audit=schedule_audit,
        ),
        "endpoint_distance_statistics": distance,
        "endpoint_reader_statistics": teacher_stats,
        "formal_success_gate": False,
        "full_success_claim_allowed": False,
        "phase2_allowed": False,
        "initialization_teacher_assisted": False,
        "optimization_teacher_supervised": True,
        "answer_independent_writer_usable": False,
    }
    _write_json(formal_run / "technical_gate.json", technical)
    _write_json(formal_run / comparison.controller.SUMMARY_FILE, summary)

    monkeypatch.setattr(comparison, "_validate_config", lambda: {})
    monkeypatch.setattr(comparison, "_validate_parent_bridge", lambda _config: {})
    monkeypatch.setattr(comparison, "_validate_teacher_matched_reference", lambda _config: {})
    monkeypatch.setattr(
        comparison,
        "_validate_parent_target",
        lambda _config: {
            "root": "parent",
            "inventory": {"sha256": "4" * 64},
            "target_segment": _target_segment(),
        },
    )

    def controller_record(root: Path, *, mode: str, expected_commit: str) -> dict:
        launch = {
            **code_hashes,
            "preflight_prerequisite": prerequisite if mode == "formal" else None,
        }
        terminal = {
            "bridge_diagnostic_gate": True if mode == "formal" else None,
            "diagnostic_result": {
                "evaluated": True,
                "bridge_diagnostic_gate": True,
                "distance_gate": True,
                "reader_transfer_gate": True,
                "decision": summary["decision"],
            }
            if mode == "formal"
            else {"evaluated": False},
        }
        return {
            "root": str(root),
            "root_inventory": {"sha256": "9" * 64},
            "trainer_inventory": {"sha256": "0" * 64},
            "launch": launch,
            "terminal": terminal,
        }

    monkeypatch.setattr(comparison, "_validate_controller_root", controller_record)
    monkeypatch.setattr(
        comparison,
        "_load_teacher",
        lambda *_args: (
            torch.zeros((1, 4, 128, 128), dtype=torch.float32),
            {"tensor_sha256": comparison.core.BRIDGE_TEACHER_TENSOR_SHA256},
        ),
    )
    monkeypatch.setattr(
        comparison,
        "_validate_manifest",
        lambda run, **_kwargs: {
            "manifest": {
                "initialization_binding": {"compute_device_type": "cuda"}, "source_latents": {"dtype": "torch.bfloat16"},
                "parity_checks": {"observed": {"initial_x_T_fp32_sha256": "1" * 64,
                                               "initial_z_t_fp32_sha256": "2" * 64}},
            },
            "manifest_sha256": "5" * 64 if run == preflight_run else "6" * 64,
            "condition_sha256": "7" * 64,
        },
    )
    monkeypatch.setattr(
        comparison,
        "_validate_initialization",
        lambda *_args, **_kwargs: {
            "record": {},
            "actual_effective_sigmas": list(comparison.EXPECTED_SIGMAS),
        },
    )
    monkeypatch.setattr(
        comparison,
        "_validate_preflight",
        lambda *_args, **_kwargs: {
            "teacher_statistics": teacher_stats,
            "passed": True,
            "checkpoint": {"tensor_sha256": {"x_T_fp32": "1" * 64, "z_t_fp32": "2" * 64},
                           "image_sha256": "0" * 64},
        },
    )
    metric_rows = [
        {
            "effective_sigmas": list(comparison.EXPECTED_SIGMAS),
            **{
            field: value
            for field, value in zip(
                (
                    "optimizer_step",
                    "mse",
                    "rmse",
                    "l2_distance",
                    "mse_ratio_to_m0",
                    "l2_distance_ratio_to_m0",
                    "teacher_normalized_rmse",
                    "gradient_norm",
                    "gradient_nonzero_fraction",
                    "x_T_update_norm",
                    "elapsed_seconds",
                ),
                (step, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1, step),
            )
            },
        }
        for step in range(1, 257)
    ]
    monkeypatch.setattr(
        comparison,
        "_validate_metrics",
        lambda _path: (metric_rows, {"m0_mse": 1.0}),
    )
    def checkpoint_record(_run: Path, *, step: int, **_kwargs: object) -> dict:
        record = {
            "optimizer_step": step,
            "distance_statistics": distance if step == 256 else step0_distance,
            "tensor_sha256": {"x_T_fp32": "1" * 64, "z_t_fp32": "2" * 64},
            "image_sha256": "0" * 64,
        }
        return record

    monkeypatch.setattr(comparison, "_validate_checkpoint", checkpoint_record)
    monkeypatch.setattr(
        comparison,
        "_validate_rows",
        lambda *_args, **_kwargs: {
            "rows": [{}] * 20,
            "sha256": "8" * 64,
            "statistics": {
                "canonical_teacher/teacher": teacher_stats,
                "m0/normal": m0_stats,
                "m0/reset": m0_stats,
                f"{comparison.core.BRIDGE_PRIMARY_ENDPOINT}/normal": teacher_stats,
                f"{comparison.core.BRIDGE_PRIMARY_ENDPOINT}/reset": m0_stats,
            },
        },
    )
    result = comparison.compare(
        preflight_root=preflight_root,
        formal_root=formal_root,
        output_dir=tmp_path / "aggregation",
        expected_commit="a" * 40,
    )
    assert result["bridge_diagnostic_gate"] is True
    assert result["formal_success"] is False
    assert result["phase2_allowed"] is False
    assert result["initialization_teacher_assisted"] is False
    assert result["optimization_teacher_supervised"] is True
    assert result["answer_independent_writer_usable"] is False
    assert (tmp_path / "aggregation" / "artifact_inventory.json").is_file()


def test_parent_query_and_inventory_cannot_be_resigned_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "parent"
    manifest_path = root / "run" / "manifest.json"
    manifest = {
        "schema": "vision_memory.r11-new-phase1a-manifest.v1",
        "target_index": comparison.core.BRIDGE_TARGET_INDEX,
        "target_segment_id": comparison.core.BRIDGE_TARGET_SEGMENT_ID,
        "target_segment": _target_segment(),
    }
    _write_json(manifest_path, manifest)

    def resign() -> None:
        _write_json(
            root / "artifact_inventory.json",
            {
                "schema": "vision_memory.r11-new-phase1a-target-inventory.v1",
                "artifacts": [
                    {
                        "path": "run/manifest.json",
                        "bytes": manifest_path.stat().st_size,
                        "sha256": comparison._sha256(manifest_path),
                    }
                ],
            },
        )

    resign()
    immutable_sha = comparison._sha256(manifest_path)
    monkeypatch.setattr(
        comparison.core,
        "BRIDGE_PARENT_TARGET_MANIFEST_SHA256",
        immutable_sha,
    )
    monkeypatch.setattr(comparison.core, "BRIDGE_PHASE1A_SOURCE_ROOT", str(root))
    config = {}
    assert comparison._validate_parent_target(config)["target_segment"] == _target_segment()
    manifest["target_segment"]["query"]["choices"][1] = "tampered-answer"
    _write_json(manifest_path, manifest)
    resign()
    with pytest.raises(ValueError, match="manifest hash drifted"):
        comparison._validate_parent_target(config)


def _initialization_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype):
    run = tmp_path / "run"
    path = run / "initialization" / "source_only_initialization.pt"
    path.parent.mkdir(parents=True)
    source = torch.linspace(-2.0, 2.0, 65536).reshape(1, 4, 128, 128).to(dtype).float()
    actual = [0.4999999701976776, 0.375, 0.25, 0.1249999925494194]
    sigma = actual[0]
    x_t = source.clone()
    start = source.to(dtype).mul(1.0 - sigma).add(x_t.to(dtype), alpha=sigma)
    tensors = {
        "source_latents_fp32": source,
        "x_T_init_fp32": x_t, "reconstructed_start_state_compute": start,
    }
    metadata = {
        "schema": comparison.INITIALIZATION_SCHEMA, "compute_device_type": "cpu",
        "formula": comparison.INITIALIZATION_FORMULA,
        "reconstruction_operator_order": comparison.INITIALIZATION_OPERATOR,
        "nominal_effective_sigmas": list(comparison.EXPECTED_SIGMAS),
        "actual_effective_sigmas": actual, "actual_effective_sigma0": sigma,
        "parameter_dtype": "torch.float32", "compute_dtype": str(dtype),
        "shape": [1, 4, 128, 128], "initial_x_t_equals_source": True,
        "initialization_teacher_assisted": False,
        "tensor_sha256": {key: canonical_tensor_sha256(value) for key, value in tensors.items()},
    }
    payload = {**copy.deepcopy(metadata), **tensors}
    record = {**copy.deepcopy(metadata), "artifact_path": str(path), "passed": True}
    monkeypatch.setattr(comparison.core, "BRIDGE_SOURCE_LATENTS_SHA256", canonical_tensor_sha256(source))

    def resign():
        torch.save(payload, path)
        record["artifact_bytes"] = path.stat().st_size
        record["artifact_sha256"] = comparison._sha256(path)

    resign()
    return run, record, payload, resign


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mutation", [
    None, "record_sigma", "payload_sigma", "nominal_schedule", "actual_schedule",
    "formula", "operator", "x_t", "start", "dtype",
    "record_backend", "payload_backend", "both_backend", "parameter_dtype", "shape",
    "source_equality_claim", "teacher_assist_claim", "forbidden_teacher", "forbidden_query", "forbidden_std",
])
def test_source_only_initialization_is_recomputed_after_resigning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, mutation: str | None,
) -> None:
    run, record, payload, resign = _initialization_fixture(tmp_path, monkeypatch, dtype)
    if mutation == "record_sigma":
        record["actual_effective_sigma0"] = 0.5
    elif mutation == "payload_sigma":
        payload["actual_effective_sigma0"] = 0.5
    elif mutation in {"nominal_schedule", "actual_schedule"}:
        key = "nominal_effective_sigmas" if mutation == "nominal_schedule" else "actual_effective_sigmas"
        for item in (record, payload):
            item[key][0] = 0.499999
    elif mutation in {"formula", "operator"}:
        key = "formula" if mutation == "formula" else "reconstruction_operator_order"
        record[key] = payload[key] = "self-consistent but wrong"
    elif mutation in {"x_t", "start"}:
        key = "x_T_init_fp32" if mutation == "x_t" else "reconstructed_start_state_compute"
        payload[key] = payload[key] + 0.125
        record["tensor_sha256"][key] = payload["tensor_sha256"][key] = canonical_tensor_sha256(payload[key])
    elif mutation == "dtype":
        key = "reconstructed_start_state_compute"
        payload[key] = payload[key].to(torch.bfloat16 if dtype == torch.float32 else torch.float32)
        record["tensor_sha256"][key] = payload["tensor_sha256"][key] = canonical_tensor_sha256(payload[key])
    elif mutation in {"record_backend", "payload_backend", "both_backend"}:
        if mutation != "payload_backend":
            record["compute_device_type"] = "cuda"
        if mutation != "record_backend":
            payload["compute_device_type"] = "cuda"
    elif mutation in {"parameter_dtype", "shape", "source_equality_claim", "teacher_assist_claim"}:
        key, value = {
            "parameter_dtype": ("parameter_dtype", "torch.bfloat16"),
            "shape": ("shape", [1, 4, 64, 256]),
            "source_equality_claim": ("initial_x_t_equals_source", False),
            "teacher_assist_claim": ("initialization_teacher_assisted", True),
        }[mutation]
        record[key] = payload[key] = value
    elif mutation in {"forbidden_teacher", "forbidden_query", "forbidden_std"}:
        key, value = {
            "forbidden_teacher": ("teacher_fp32", torch.ones_like(payload["source_latents_fp32"])),
            "forbidden_query": ("query", "answer-bearing question"),
            "forbidden_std": ("teacher_population_std", 0.6546660661697388),
        }[mutation]
        payload[key] = value
    resign()
    kwargs = dict(record=record, compute_dtype=str(dtype), compute_device_type="cpu")
    if mutation is None:
        verified = comparison._validate_initialization(run, **kwargs)
        assert verified["record"]["passed"] is True
        assert verified["record"]["initial_x_t_equals_source"] is True
        assert verified["record"]["initialization_teacher_assisted"] is False
        assert torch.equal(verified["x_T_init_fp32"], payload["source_latents_fp32"])
    else:
        with pytest.raises(ValueError, match="initialization"):
            comparison._validate_initialization(run, **kwargs)


def test_cuda_initialization_refuses_cpu_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, record, payload, resign = _initialization_fixture(tmp_path, monkeypatch, torch.bfloat16)
    record["compute_device_type"] = payload["compute_device_type"] = "cuda"
    resign()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        comparison._validate_initialization(run, record=record,
                                              compute_dtype="torch.bfloat16", compute_device_type="cuda")


@pytest.mark.parametrize("mutation", [None, "parity", "nominal_sigma", "m0", "information",
                                      "source_dtype", "backend", "models", "condition"])
def test_manifest_keeps_fixed_source_condition_models_and_new_initializer_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str | None,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    config = comparison._load(comparison.CONFIG)
    parity = dict(comparison.core.BRIDGE_UNCHANGED_PARITY_BINDINGS)
    condition_path = run / "condition" / "official_full_condition.pt"
    condition_path.parent.mkdir()
    prompt = torch.ones((1, 4))
    mask = torch.ones((1, 4), dtype=torch.bool)
    condition_hashes = {"prompt_embeds": canonical_tensor_sha256(prompt),
                        "attention_mask": canonical_tensor_sha256(mask)}
    parity["condition_prompt_embeds_sha256"] = condition_hashes["prompt_embeds"]
    parity["condition_attention_mask_sha256"] = condition_hashes["attention_mask"]
    monkeypatch.setattr(comparison.core, "BRIDGE_UNCHANGED_PARITY_BINDINGS", parity)
    condition_payload = {
        "schema": "vision_memory.r11-new-phase1a-condition.v1",
        "prompt_embeds": prompt, "attention_mask": mask,
        "tensor_sha256": condition_hashes,
        "event_text_sha256": parity["event_text_sha256"], "recompute_matches": True,
    }
    torch.save(condition_payload, condition_path)
    condition_record = {
        "sha256": comparison._sha256(condition_path), "bytes": condition_path.stat().st_size,
        "tensor_sha256": copy.deepcopy(condition_hashes), "event_text_sha256": parity["event_text_sha256"],
        "recompute_matches": True,
    }
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_binding = {"target_root": str(parent), "target_inventory_sha256": "9" * 64}
    for name in ("comparison", "raw_artifacts"):
        path = parent / f"{name}.json"
        _write_json(path, {"kind": name})
        parent_binding[f"{name}_path"] = str(path)
        parent_binding[f"{name}_sha256"] = comparison._sha256(path)
        config["parent_phase1a"][f"{name}_sha256"] = comparison._sha256(path)
    snapshots = {
        model: {"manifest_sha256": comparison.core.BRIDGE_MODEL_SNAPSHOT_BINDINGS[key]}
        for model, key in (("dreamlite_mobile", "dreamlite_snapshot_manifest_sha256"),
                           ("qwen_reader", "reader_snapshot_manifest_sha256"))
    }
    manifest = {
        "schema": comparison.TRAINER_MANIFEST_SCHEMA,
        "protocol": comparison.core.BRIDGE_PROTOCOL, "mode": "formal",
        "git_commit": "a" * 40, "git_dirty": False,
        "preregistered_config_file_sha256": comparison.core.BRIDGE_CONFIG_FILE_SHA256,
        "preregistered_config_canonical_sha256": comparison.core.BRIDGE_CONFIG_CANONICAL_SHA256,
        "target_index": comparison.core.BRIDGE_TARGET_INDEX,
        "target_segment_id": comparison.core.BRIDGE_TARGET_SEGMENT_ID,
        "target_segment": _target_segment(), "bridge_diagnostic_only": True,
        "phase2_allowed": False, "formal_success_gate": False,
        "parity_checks": {"passed": True, "expected": copy.deepcopy(parity),
                          "observed": {**parity, "initial_x_T_fp32_sha256": "1" * 64,
                                       "initial_z_t_fp32_sha256": "2" * 64},
                          "source_only_initialization_artifact_valid": True},
        "initialization_binding": {"compute_device_type": "cuda", "tensor_sha256": {"x_T_init_fp32": "1" * 64}},
        "source_latents": {"sha256": comparison.core.BRIDGE_SOURCE_LATENTS_SHA256,
                           "shape": [1, 4, 128, 128], "dtype": "torch.bfloat16", "device": "cuda:0"},
        "initial_x_T_fp32": {"sha256": "1" * 64, "shape": [1, 4, 128, 128], "dtype": "torch.float32"},
        "single_changed_factor": {
            "factor": "x_T_initialization",
            "from": "answer-independent event-keyed standard Gaussian with global seed 0",
            "to": "source-only deterministic initialization",
            "initialization_teacher_assisted": False, "optimization_teacher_supervised": True,
            "answer_independent_writer_usable": False,
        },
        "information_boundary": {
            "passed": True, "canonical_teacher_used_only_by_dense_loss": True,
            "canonical_teacher_used_by_initialization_and_dense_loss": False,
            "teacher_assisted_initialization": False, "answer_independent_writer_usable": False,
            "initialization_teacher_assisted": False, "optimization_teacher_supervised": True,
            "reader_used_only_for_fixed_audit": True, "reader_gradient_calls_during_optimization": 0,
        },
        "fixed_contract": {
            "only_trainable": "x_T_fp32", "optimizer_steps": 256,
            "effective_sigmas": list(comparison.EXPECTED_SIGMAS), "optimizer": "Adam",
            "learning_rate": 0.05, "weight_decay": 0.0, "gradient_clipping": None,
            "learning_rate_schedule": {"name": "constant_then_post128_cosine_to_zero",
                                       "intervention_first_update": 129,
                                       "formula": "0.05 for u<=128; 0.025*(1+cos(pi*(u-128)/128)) otherwise"},
            "checkpoint_steps": list(comparison.core.BRIDGE_CHECKPOINT_STEPS),
            "primary_endpoint": comparison.core.BRIDGE_PRIMARY_ENDPOINT,
            "best_checkpoint_selection_forbidden": True, "reader_gradient_calls_during_optimization": 0,
            "m0_definition": config["unchanged_contract"]["m0_definition"],
            "dreamlite_device": "cuda:0", "reader_device": "cuda:1",
        },
        "train_sha256": config["unchanged_contract"]["train_sha256"],
        "dev_sha256": config["unchanged_contract"]["dev_sha256"],
        "selected_segments_sha256": config["parent_phase1a"]["selected_segments_sha256"],
        "parent_phase1a": parent_binding,
        "teacher": {"file_sha256": comparison.core.BRIDGE_TEACHER_FILE_SHA256,
                    "tensor_sha256": comparison.core.BRIDGE_TEACHER_TENSOR_SHA256,
                    "copied_sha256": comparison.core.BRIDGE_TEACHER_FILE_SHA256},
        "condition_artifact": condition_record, "model_snapshot_payloads_start": snapshots,
    }
    if mutation == "parity":
        manifest["parity_checks"]["observed"]["event_text_sha256"] = "3" * 64
        manifest["parity_checks"]["expected"]["event_text_sha256"] = "3" * 64
    elif mutation == "nominal_sigma":
        manifest["fixed_contract"]["effective_sigmas"][0] = 0.499999
    elif mutation == "m0":
        manifest["fixed_contract"]["m0_definition"] = "initial teacher state"
    elif mutation == "information":
        manifest["information_boundary"]["answer_independent_writer_usable"] = True
    elif mutation == "source_dtype":
        manifest["source_latents"]["dtype"] = "torch.float32"
    elif mutation == "backend":
        manifest["initialization_binding"]["compute_device_type"] = "cpu"
    elif mutation == "models":
        snapshots["qwen_reader"]["manifest_sha256"] = "8" * 64
    elif mutation == "condition":
        condition_payload["prompt_embeds"].add_(1.0)
        condition_payload["tensor_sha256"]["prompt_embeds"] = canonical_tensor_sha256(condition_payload["prompt_embeds"])
        torch.save(condition_payload, condition_path)
        condition_record.update({"sha256": comparison._sha256(condition_path),
                                 "bytes": condition_path.stat().st_size,
                                 "tensor_sha256": condition_payload["tensor_sha256"]})
    _write_json(run / "manifest.json", manifest)
    _write_json(run / "model_snapshot_verification_start.json", {"bindings": snapshots})
    _write_json(run / "model_snapshot_verification_end.json", {"bindings": snapshots, "passed": True})
    kwargs = dict(mode="formal", expected_commit="a" * 40, config=config,
                  parent_target={"target_segment": _target_segment(), "root": str(parent),
                                 "inventory": {"sha256": "9" * 64}})
    if mutation is None:
        assert all(comparison._validate_manifest(run, **kwargs)["fixed_checks"].values())
    else:
        with pytest.raises(ValueError, match="Bridge"):
            comparison._validate_manifest(run, **kwargs)


@pytest.mark.parametrize("mutation", [None, "mse", "ce", "formal_success"])
def test_parent_bridge_keeps_absolute_endpoint_reference_after_resigning(
    tmp_path: Path, mutation: str | None,
) -> None:
    config = comparison._load(comparison.CONFIG)
    report = ROOT / "reports" / "r11-new-canonical-latent-bridge-post128-cosine-results-20260906" / "aggregation-v1"
    destination = tmp_path / "parent" / "aggregation-v1"
    destination.mkdir(parents=True)
    parent = comparison._load(report / "comparison.json")
    if mutation == "mse":
        parent["endpoint_distance_statistics"]["mse"] = 0.0
    elif mutation == "ce":
        parent["endpoint_reader_statistics"]["mean_ce"] = 0.0
    elif mutation == "formal_success":
        parent["formal_success"] = True
    _write_json(destination / "comparison.json", parent)
    (destination / "RAW_ARTIFACTS.json").write_bytes((report / "RAW_ARTIFACTS.json").read_bytes())
    config["parent_bridge"]["source_root"] = str(destination.parent)
    config["parent_bridge"]["comparison_sha256"] = comparison._sha256(destination / "comparison.json")
    comparison._write_inventory(destination)
    if mutation is None:
        assert comparison._validate_parent_bridge(config)["training_git_commit"] == "16318e005b496a16b7712ad4ff3cea50e2be34fa"
    else:
        with pytest.raises(ValueError, match="endpoint values drifted"):
            comparison._validate_parent_bridge(config)


@pytest.mark.parametrize("mutation", [None, "role", "mse", "teacher_flag"])
def test_teacher_matched_result_is_a_bound_descriptive_reference_only(tmp_path: Path, mutation: str | None) -> None:
    config = comparison._load(comparison.CONFIG)
    source = ROOT / "reports" / "r11-new-teacher-matched-init-results-20260906" / "aggregation-v1"
    destination = tmp_path / "teacher-reference" / "aggregation-v1"
    destination.mkdir(parents=True)
    reference = comparison._load(source / "comparison.json")
    if mutation == "role":
        config["teacher_matched_reference"]["role"] = "secondary_comparator"
    elif mutation == "mse":
        reference["endpoint_distance_statistics"]["mse"] = 0.0
    elif mutation == "teacher_flag":
        reference["teacher_assisted_initialization"] = False
    _write_json(destination / "comparison.json", reference)
    (destination / "RAW_ARTIFACTS.json").write_bytes((source / "RAW_ARTIFACTS.json").read_bytes())
    config["teacher_matched_reference"]["source_root"] = str(destination.parent)
    config["teacher_matched_reference"]["comparison_sha256"] = comparison._sha256(destination / "comparison.json")
    comparison._write_inventory(destination)
    if mutation is None:
        verified = comparison._validate_teacher_matched_reference(config)
        assert verified["role"] == "descriptive_only_not_the_secondary_comparator"
        audit = comparison.core.bridge_initialization_hypothesis_audit(
            endpoint_mse=0.08, endpoint_reader_mean_ce=15.0, technical_gate=True,
            teacher_replay_gate=True, distance_pass=False, reader_transfer_pass=False,
        )
        # Worse than the descriptive teacher reference, but better than both fixed Gaussian metrics.
        assert audit["passed"] is True
    else:
        with pytest.raises(ValueError, match="teacher-matched reference"):
            comparison._validate_teacher_matched_reference(config)
