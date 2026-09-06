from __future__ import annotations

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
    compare_r11_new_canonical_latent_bridge as comparison,
)
from vision_memory.data import REVERSE_CYCLIC4  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256  # noqa: E402


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
        tensor_numel=16,
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
    trajectory = tuple(torch.full_like(teacher, float(index)) for index in range(5))
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
    _write_json(record_path, record)
    validated = comparison._validate_checkpoint(
        run,
        step=checkpoint_step,
        teacher=teacher,
        m0_mse=1.0,
        manifest_sha256=manifest_sha,
        condition_sha256=condition_sha,
    )
    assert validated["distance_statistics"]["mse"] == pytest.approx(0.25)
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
                "pre_intervention_step128_parity_valid",
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
    schedule_audit = comparison.core.bridge_schedule_hypothesis_audit(
        step128_mse_ratio=1.0,
        endpoint_mse_ratio=float(distance["mse_ratio_to_m0"]),
        technical_gate=True,
        teacher_replay_gate=True,
        pre_intervention_parity=True,
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
        "secondary_solver_hypothesis_decision": comparison.core.bridge_schedule_hypothesis_decision(
            distance_pass=True,
            reader_transfer_pass=True,
            audit=schedule_audit,
        ),
        "endpoint_distance_statistics": distance,
        "endpoint_reader_statistics": teacher_stats,
        "formal_success_gate": False,
        "full_success_claim_allowed": False,
        "phase2_allowed": False,
    }
    _write_json(formal_run / "technical_gate.json", technical)
    _write_json(formal_run / comparison.controller.SUMMARY_FILE, summary)

    monkeypatch.setattr(comparison, "_validate_config", lambda: {})
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
            "manifest_sha256": "5" * 64 if run == preflight_run else "6" * 64,
            "condition_sha256": "7" * 64,
        },
    )
    monkeypatch.setattr(
        comparison,
        "_validate_preflight",
        lambda *_args, **_kwargs: {
            "teacher_statistics": teacher_stats,
            "passed": True,
        },
    )
    metric_rows = [
        {
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
        }
        if step == 128:
            record.update(
                {
                    "tensor_sha256": comparison.core.BRIDGE_PARENT_STEP128_TENSOR_SHA256,
                    "optimizer_state_sha256": comparison.core.BRIDGE_PARENT_STEP128_OPTIMIZER_SHA256,
                    "image_sha256": comparison.core.BRIDGE_PARENT_STEP128_PNG_SHA256,
                }
            )
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
    config = {
        "exact_parity_bindings": {
            "phase1a_valid_source_root": str(root),
        }
    }
    assert comparison._validate_parent_target(config)["target_segment"] == _target_segment()
    manifest["target_segment"]["query"]["choices"][1] = "tampered-answer"
    _write_json(manifest_path, manifest)
    resign()
    with pytest.raises(ValueError, match="manifest hash drifted"):
        comparison._validate_parent_target(config)
