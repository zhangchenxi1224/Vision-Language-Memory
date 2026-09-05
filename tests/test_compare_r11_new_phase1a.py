from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "experiments" / "compare_r11_new_phase1a.py"
    specification = importlib.util.spec_from_file_location("compare_r11_new_phase1a_test", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


comparison = _load_script()
core = comparison.core
EXPECTED_COMMIT = "a" * 40
FLOAT32_EFFECTIVE_SIGMAS = [0.4999999701976776, 0.375, 0.25, 0.1249999925494194]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _inventory(root: Path, schema: str, *, count: bool) -> None:
    own_inventory = (root / "artifact_inventory.json").resolve()
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in sorted(value for value in root.rglob("*") if value.is_file())
        if path.resolve() != own_inventory
    ]
    value = {"schema": schema, "artifacts": rows}
    if count:
        value["artifact_count"] = len(rows)
    else:
        value["root"] = str(root.resolve())
    _write_json(root / "artifact_inventory.json", value)


def _stat_block(*, saturation: bool = False) -> dict[str, object]:
    result: dict[str, object] = {
        "shape": [1, 4, 2, 2],
        "dtype": "torch.float32",
        "minimum": -1.0,
        "maximum": 1.0,
        "rms": 0.5,
        "norm": 2.0,
    }
    if saturation:
        result["saturation_fraction"] = 0.0
    return result


def _metrics(index: int, *, effective_sigmas: list[float] | None = None) -> list[dict[str, object]]:
    if effective_sigmas is None:
        effective_sigmas = list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE)
    target_id = core.R11_NEW_TARGET_IDS[index]
    rows = []
    for scheduled in core.build_phase1a_schedule(index):
        trajectory = []
        for trajectory_index in range(core.R11_NEW_DIFFUSION_STEPS + 1):
            state = {"trajectory_index": trajectory_index, **_stat_block()}
            if trajectory_index:
                state["delta_from_previous_norm"] = 0.5
            trajectory.append(state)
        rows.append(
            {
                "schema": comparison.METRICS_SCHEMA,
                "kind": "optimizer_step",
                "optimizer_step": scheduled.optimizer_step,
                "target_segment_id": target_id,
                "forward_cyclic_training_view": scheduled.forward_cyclic_training_view,
                "permutation": list(scheduled.permutation),
                "loss_before_step": 2.0 - scheduled.optimizer_step / 512.0,
                "gradient_norm": 1.0,
                "gradient_nonzero_fraction": 1.0,
                "x_T_before_step": _stat_block(),
                "x_T_after_step": _stat_block(),
                "x_T_update_norm": 0.1,
                "z_t_before_step": _stat_block(),
                "image_before_step": _stat_block(saturation=True),
                "trajectory_before_step": trajectory,
                "trajectory_points": core.R11_NEW_DIFFUSION_STEPS + 1,
                "dreamlite_denoising_steps": core.R11_NEW_DIFFUSION_STEPS,
                "effective_sigmas": effective_sigmas,
                "full_dreamlite_forward_executed": True,
                "denoiser_steps_executed": core.R11_NEW_DIFFUSION_STEPS,
                "effective_sigma_schedule": effective_sigmas,
                "gradient_mode": "full",
                "gradient_clipping_applied": False,
                "learning_rate": core.R11_NEW_LEARNING_RATE,
                "weight_decay": core.R11_NEW_WEIGHT_DECAY,
                "elapsed_seconds": float(scheduled.optimizer_step),
            }
        )
    return rows


def _choice_row(
    *,
    target_id: str,
    checkpoint: str,
    condition: str,
    view: int,
    passing_endpoint: bool,
) -> dict[str, object]:
    permutation = list(comparison.REVERSE_CYCLIC4[view])
    original_target = 0
    ordered_target = permutation.index(original_target)
    logits = [0.0, 0.0, 0.0, 0.0]
    if checkpoint == core.R11_NEW_PRIMARY_ENDPOINT and condition == "normal" and passing_endpoint:
        logits[ordered_target] = 3.0
    else:
        alternative = next(index for index in range(4) if index != ordered_target)
        logits[alternative] = 1.0
    predicted_ordered = max(range(4), key=logits.__getitem__)
    predicted_original = permutation[predicted_ordered]
    maximum = max(logits)
    ce = maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[ordered_target]
    margin = logits[ordered_target] - max(
        value for index, value in enumerate(logits) if index != ordered_target
    )
    choices = ["alpha", "beta", "gamma", "delta"]
    return {
        "schema": comparison.EVALUATION_SCHEMA,
        "suite": comparison.EVALUATION_SUITE,
        "checkpoint": checkpoint,
        "condition": condition,
        "item_id": target_id,
        "pair_unit": target_id,
        "family": "F1",
        "view_index": view,
        "permutation": permutation,
        "target_index": original_target,
        "predicted_index": predicted_original,
        "correct": predicted_original == original_target,
        "ce": ce,
        "margin": margin,
        "choice_logits_ordered": logits,
        "target_text": choices[original_target],
        "predicted_text": choices[predicted_original],
        "donor_item_id": None,
    }


def _evaluation(index: int, *, passing: bool) -> list[dict[str, object]]:
    target_id = core.R11_NEW_TARGET_IDS[index]
    return [
        _choice_row(
            target_id=target_id,
            checkpoint=checkpoint,
            condition=condition,
            view=view,
            passing_endpoint=passing,
        )
        for checkpoint in ("m0", core.R11_NEW_PRIMARY_ENDPOINT)
        for condition in ("normal", "reset")
        for view in range(4)
    ]


def _core_audit() -> dict[str, object]:
    return comparison._expected_core_audit()


def _make_prerequisite_terminal(root: Path, *, mode: str) -> Path:
    root.mkdir()
    terminal_path = root / "terminal.json"
    diagnostic = (
        {
            "evaluated": False,
            "phase1a_query_level_reachability_gate": None,
            "result": "not_evaluated_in_technical_preflight",
        }
        if mode == "technical-preflight"
        else {
            "evaluated": True,
            "phase1a_query_level_reachability_gate": True,
            "result": "query_level_reachability_passed",
        }
    )
    _write_json(
        terminal_path,
        {
            "schema": comparison.CONTROLLER_TERMINAL_SCHEMA,
            "status": "technical_completed",
            "technical_completed": True,
            "diagnostic_result": diagnostic,
            "formal_success": False,
            "scientific_success_claim": False,
            "mode": mode,
            "target_index": 0,
            "target_segment_id": core.R11_NEW_TARGET_IDS[0],
            "git_commit": EXPECTED_COMMIT,
            "child_exit_code": 0,
            "execution_checks": {"synthetic_prerequisite_complete": True},
            "error": None,
            "config_sha256": comparison._sha256(comparison.CONFIG_PATH),
            "trainer_sha256": comparison._sha256(comparison.TRAINER_PATH),
        },
    )
    _inventory(root, comparison.CONTROLLER_INVENTORY_SCHEMA, count=False)
    return terminal_path


def _prerequisite_binding(terminal_path: Path, *, mode: str) -> dict[str, object]:
    inventory_path = terminal_path.parent / "artifact_inventory.json"
    return {
        "terminal_path": str(terminal_path.resolve()),
        "terminal_sha256": _sha(terminal_path),
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": _sha(inventory_path),
        "mode": mode,
        "target_index": 0,
        "target_segment_id": core.R11_NEW_TARGET_IDS[0],
        "passed": True,
    }


def _make_target(
    root: Path,
    index: int,
    *,
    preflight_terminal: Path,
    target0_formal_terminal: Path | None,
    passing: bool = True,
    effective_sigmas: list[float] | None = None,
) -> None:
    if effective_sigmas is None:
        effective_sigmas = list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE)
    target_id = core.R11_NEW_TARGET_IDS[index]
    run = root / "run"
    for directory in (run / "checkpoints", run / "images", run / "checkpoint_hashes", run / "condition"):
        directory.mkdir(parents=True, exist_ok=True)
    (root / "stdout.log").write_text("synthetic stdout\n", encoding="utf-8")
    (root / "stderr.log").write_text("", encoding="utf-8")
    (run / ".r11_new_output_owner.json").write_text("{}\n", encoding="utf-8")
    (run / "environment.txt").write_text("synthetic==1\n", encoding="utf-8")
    _write_json(run / "runtime.json", {"python": "test"})
    (run / "REPORT.md").write_text("# synthetic per-target report\n", encoding="utf-8")
    condition_path = run / "condition" / "official_full_condition.pt"
    condition_path.write_bytes(b"condition")

    tensor_hash = "b" * 64
    checkpoint_records = []
    for step in core.R11_NEW_CHECKPOINT_STEPS:
        checkpoint = run / "checkpoints" / f"step-{step:03d}.pt"
        image = run / "images" / f"step-{step:03d}.png"
        checkpoint.write_bytes(f"checkpoint-{step}".encode())
        image.write_bytes(f"image-{step}".encode())
        record = {
            "schema": comparison.CHECKPOINT_HASH_SCHEMA,
            "optimizer_step": step,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _sha(checkpoint),
            "png_path": str(image.resolve()),
            "png_bytes": image.stat().st_size,
            "png_sha256": _sha(image),
            "trajectory_points": core.R11_NEW_DIFFUSION_STEPS + 1,
            "effective_sigmas": effective_sigmas,
            "tensor_sha256": {
                "x_T_fp32": tensor_hash,
                "z_t_fp32": tensor_hash,
                "trajectory_fp32": [tensor_hash] * (core.R11_NEW_DIFFUSION_STEPS + 1),
            },
        }
        record_path = run / "checkpoint_hashes" / f"step-{step:03d}.json"
        _write_json(record_path, record)
        checkpoint_records.append(record)
    (run / "endpoint_raw.pt").write_bytes((run / "checkpoints" / "step-256.pt").read_bytes())
    (run / "endpoint_raw.png").write_bytes((run / "images" / "step-256.png").read_bytes())

    bindings = {
        name: {"passed": True, "manifest_sha256": digest}
        for name, digest in comparison.EXPECTED_MODEL_MANIFEST_SHA256.items()
    }
    _write_json(
        run / "model_snapshot_verification_start.json",
        {
            "schema": "vision_memory.r11-new-phase1a-model-snapshot-start.v1",
            "bindings": bindings,
        },
    )
    _write_json(
        run / "model_snapshot_verification_end.json",
        {
            "schema": "vision_memory.r11-new-phase1a-model-snapshot-end.v1",
            "passed": True,
            "bindings": bindings,
        },
    )
    config_sha = comparison._sha256(comparison.CONFIG_PATH)
    query = {
        "text": "Which value is current?",
        "choices": ["alpha", "beta", "gamma", "delta"],
        "target_index": 0,
        "target_token_count": 1,
    }
    fixed_contract = {
        "resolution": 1024,
        "only_trainable": "x_T_fp32",
        "dreamlite_pipeline_load": "DreamLiteMobilePipeline.from_pretrained directly; no PEFT",
        "unet_frozen": True,
        "vae_frozen": True,
        "text_encoder_frozen": True,
        "reader_frozen": True,
        "dreamlite_unet_executed": True,
        "official_full_conditioner_precomputed": True,
        "gradient_mode": "full",
        "checkpoint_unet": True,
        "num_denoising_steps": core.R11_NEW_DIFFUSION_STEPS,
        "edit_start_sigma": core.R11_NEW_EFFECTIVE_START_SIGMA,
        "effective_sigmas": list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE),
        "optimizer": "Adam",
        "learning_rate": core.R11_NEW_LEARNING_RATE,
        "weight_decay": core.R11_NEW_WEIGHT_DECAY,
        "optimizer_steps": core.R11_NEW_OPTIMIZER_STEPS,
        "preflight_backward_calls": 0,
        "gradient_clipping": None,
        "checkpoint_steps": list(core.R11_NEW_CHECKPOINT_STEPS),
        "training_views": "forward cyclic",
        "endpoint_views": "reverse cyclic",
        "primary_endpoint": core.R11_NEW_PRIMARY_ENDPOINT,
        "best_checkpoint_selection_forbidden": True,
        "reset": "decode(blank_source_latent)",
    }
    information = {
        "schema": "vision_memory.r11-new-phase1a-information-boundary.v1",
        "passed": True,
        "dreamlite_inputs": list(core.R11_NEW_DREAMLITE_INPUTS),
        "noise_key": list(core.R11_NEW_NOISE_KEY),
        "reader_loss_inputs": list(core.R11_NEW_READER_LOSS_INPUTS),
        "leaked_fields": [],
        "query_used_only_by_frozen_reader_loss": True,
        "choices_used_only_by_frozen_reader_loss": True,
        "target_index_used_only_by_frozen_reader_loss": True,
        "forbidden_writer_key_violations": [],
    }
    manifest = {
        "schema": comparison.TRAINER_MANIFEST_SCHEMA,
        "protocol": core.R11_NEW_PROTOCOL,
        "implementation_revision": comparison.IMPLEMENTATION_REVISION,
        "mode": "formal",
        "git_commit": EXPECTED_COMMIT,
        "git_dirty": False,
        "preregistered_config_sha256": config_sha,
        "target_index": index,
        "target_segment_id": target_id,
        "target_segment": {
            "segment_id": target_id,
            "family": "F1",
            "events": [{"event_text": "set alpha"}],
            "query": query,
        },
        "selected_segment_ids": list(core.R11_NEW_TARGET_IDS),
        "selected_segments_sha256": core.R11_NEW_TARGETS_PAYLOAD_SHA256,
        "train_sha256": core.R11_NEW_TRAIN_SHA256,
        "dev_sha256": core.R11_NEW_DEV_SHA256,
        "model_snapshot_payloads_start": bindings,
        "source_rgb": {"value": "127/255"},
        "source_latents": {"shape": [1, 4, 2, 2]},
        "initial_x_T_fp32": {"shape": [1, 4, 2, 2], "dtype": "torch.float32"},
        "condition_artifact": {
            "path": str(condition_path.resolve()),
            "sha256": _sha(condition_path),
            "bytes": condition_path.stat().st_size,
            "recompute_matches": True,
        },
        "information_boundary": information,
        "source_contract_verified": True,
        "event_noise_contract_verified": True,
        "fixed_contract": fixed_contract,
        "query_level_diagnostic_only": True,
        "formal_success_gate": False,
    }
    _write_json(run / "manifest.json", manifest)

    manifest_sha = _sha(run / "manifest.json")
    condition_sha = _sha(condition_path)
    for step in core.R11_NEW_CHECKPOINT_STEPS:
        checkpoint = run / "checkpoints" / f"step-{step:03d}.pt"
        image = run / "images" / f"step-{step:03d}.png"
        x_t = torch.full((1, 4, 2, 2), float(step) / 256.0, dtype=torch.float32)
        z_t = x_t + 0.25
        trajectory = tuple(
            x_t + float(index) / 10.0
            for index in range(core.R11_NEW_DIFFUSION_STEPS + 1)
        )
        tensor_hashes = {
            "x_T_fp32": comparison.canonical_tensor_sha256(x_t),
            "z_t_fp32": comparison.canonical_tensor_sha256(z_t),
            "trajectory_fp32": [
                comparison.canonical_tensor_sha256(value) for value in trajectory
            ],
        }
        torch.save(
            {
                "schema": "vision_memory.r11-new-phase1a-checkpoint.v1",
                "optimizer_step": step,
                "x_T_fp32": x_t,
                "z_t_fp32": z_t,
                "trajectory_fp32": trajectory,
                "effective_sigmas": effective_sigmas,
                "optimizer": {},
                "manifest_sha256": manifest_sha,
                "condition_artifact_sha256": condition_sha,
                "tensor_sha256": tensor_hashes,
            },
            checkpoint,
        )
        record = {
            "schema": comparison.CHECKPOINT_HASH_SCHEMA,
            "optimizer_step": step,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _sha(checkpoint),
            "png_path": str(image.resolve()),
            "png_bytes": image.stat().st_size,
            "png_sha256": _sha(image),
            "trajectory_points": core.R11_NEW_DIFFUSION_STEPS + 1,
            "effective_sigmas": effective_sigmas,
            "tensor_sha256": tensor_hashes,
        }
        _write_json(run / "checkpoint_hashes" / f"step-{step:03d}.json", record)
    (run / "endpoint_raw.pt").write_bytes((run / "checkpoints" / "step-256.pt").read_bytes())
    (run / "endpoint_raw.png").write_bytes((run / "images" / "step-256.png").read_bytes())

    metrics = _metrics(index, effective_sigmas=effective_sigmas)
    _write_jsonl(run / "metrics.jsonl", metrics)
    evaluation = _evaluation(index, passing=passing)
    _write_jsonl(run / "target_evaluation_rows.jsonl", evaluation)
    statistics = core.phase1a_target_statistics(
        evaluation,
        suite=comparison.EVALUATION_SUITE,
        target_segment_id=target_id,
        endpoint=core.R11_NEW_PRIMARY_ENDPOINT,
    )
    target_gate = core.phase1a_target_gate(statistics, technical_gate=True)
    technical = {
        **core.phase1a_technical_gate(metrics, target_segment_id=target_id, audit=_core_audit()),
        "schema": comparison.TECHNICAL_GATE_SCHEMA,
        "passed": True,
        "optimizer_step_records": core.R11_NEW_OPTIMIZER_STEPS,
        "optimizer_steps_exact": True,
        "training_view_counts": {0: 64, 1: 64, 2: 64, 3: 64},
        "expected_training_view_counts": {0: 64, 1: 64, 2: 64, 3: 64},
        "four_step_schedule_exact": True,
        "effective_sigmas_expected": list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE),
        "checkpoint_steps_observed": list(core.R11_NEW_CHECKPOINT_STEPS),
        "checkpoint_hashes_valid": True,
        "condition_artifact_valid": True,
        "core_audit": _core_audit(),
        "trainable_parameter_names": [core.R11_NEW_TRAINABLE_PARAMETER],
        "only_x_T_fp32_trainable": True,
        "unet_frozen": True,
        "vae_frozen": True,
        "text_encoder_frozen": True,
        "reader_frozen": True,
        "optimizer_contract_valid": True,
        "gradient_clipping_forbidden_and_absent": True,
        "snapshots_unchanged": True,
        "finite_metrics": True,
        "minimum_gradient_norm": 1.0,
        "minimum_gradient_nonzero_fraction": 1.0,
    }
    _write_json(run / "technical_gate.json", technical)
    artifacts = {
        "manifest_sha256": _sha(run / "manifest.json"),
        "metrics_sha256": _sha(run / "metrics.jsonl"),
        "evaluation_rows_sha256": _sha(run / "target_evaluation_rows.jsonl"),
        "endpoint_raw_sha256": _sha(run / "endpoint_raw.pt"),
        "endpoint_png_sha256": _sha(run / "endpoint_raw.png"),
        "snapshot_end_sha256": _sha(run / "model_snapshot_verification_end.json"),
        "technical_gate_sha256": _sha(run / "technical_gate.json"),
    }
    summary = {
        "schema": comparison.TRAINER_SUMMARY_SCHEMA,
        "status": "completed",
        "mode": "formal",
        "protocol": core.R11_NEW_PROTOCOL,
        "implementation_revision": comparison.IMPLEMENTATION_REVISION,
        "git_commit": EXPECTED_COMMIT,
        "target_index": index,
        "target_segment_id": target_id,
        "optimizer_steps": core.R11_NEW_OPTIMIZER_STEPS,
        "primary_endpoint": core.R11_NEW_PRIMARY_ENDPOINT,
        "technical_gate": technical,
        "target_statistics": statistics,
        "gates": {
            "technical_gate": True,
            "phase1a_query_level_reachability_gate": target_gate,
            "formal_success_gate": False,
        },
        "query_level_diagnostic_only": True,
        "state_level_reachability_not_tested": True,
        "shared_writer_not_trained": True,
        "formal_success_gate": False,
        "full_success_claim_allowed": False,
        "checkpoint_steps_observed": list(core.R11_NEW_CHECKPOINT_STEPS),
        "artifacts": artifacts,
        "wall_clock_seconds": 256.0,
    }
    _write_json(run / "r11_new_phase1a_summary.json", summary)
    _write_json(
        run / "terminal.json",
        {
            "schema": comparison.TRAINER_TERMINAL_SCHEMA,
            "status": "succeeded",
            "mode": "formal",
            "technical_gate": True,
            "diagnostic_evaluated": True,
            "formal_success_gate": False,
        },
    )
    _inventory(run, comparison.TRAINER_INVENTORY_SCHEMA, count=True)

    launch = {
        "schema": comparison.CONTROLLER_LAUNCH_SCHEMA,
        "status": "running",
        "mode": "formal",
        "git_commit": EXPECTED_COMMIT,
        "git_dirty": False,
        "target_index": index,
        "target_segment_id": target_id,
        "data_sha256": {"train": core.R11_NEW_TRAIN_SHA256, "dev": core.R11_NEW_DEV_SHA256},
        "environment": comparison.EXPECTED_ENVIRONMENT,
        "config_sha256": config_sha,
        "trainer_sha256": comparison._sha256(comparison.TRAINER_PATH),
        "canonical_r11_comparison_sha256": core.R11_NEW_PARENT_R11_COMPARISON_SHA256,
        "storage_audit": {
            "passed": True,
            "minimum_free_bytes": core.R11_NEW_MINIMUM_FREE_BYTES,
        },
        "command": [
            "python",
            str(comparison.TRAINER_PATH),
            "--mode",
            "formal",
            "--target-index",
            str(index),
        ],
        "snapshot_manifests": {
            "dreamlite": {
                "manifest_sha256": comparison.EXPECTED_MODEL_MANIFEST_SHA256["dreamlite_mobile"]
            },
            "reader": {
                "manifest_sha256": comparison.EXPECTED_MODEL_MANIFEST_SHA256["qwen_reader"]
            },
        },
        "prerequisites": {
            "preflight_required": True,
            "preflight": _prerequisite_binding(
                preflight_terminal,
                mode="technical-preflight",
            ),
            "target0_formal_required": index > 0,
            "target0_formal": (
                _prerequisite_binding(target0_formal_terminal, mode="formal")
                if target0_formal_terminal is not None
                else None
            ),
            "passed": True,
        },
    }
    _write_json(root / "launch.json", launch)
    diagnostic_result = {
        "evaluated": True,
        "phase1a_query_level_reachability_gate": target_gate,
        "result": "query_level_reachability_passed" if target_gate else "query_level_reachability_not_found",
    }
    controller_terminal = {
        "schema": comparison.CONTROLLER_TERMINAL_SCHEMA,
        "status": "technical_completed",
        "technical_completed": True,
        "diagnostic_result": diagnostic_result,
        "formal_success": False,
        "scientific_success_claim": False,
        "mode": "formal",
        "target_index": index,
        "target_segment_id": target_id,
        "git_commit": EXPECTED_COMMIT,
        "child_exit_code": 0,
        "execution_checks": {"synthetic_all_independent_checks": True},
        "error": None,
        "summary_sha256": _sha(run / "r11_new_phase1a_summary.json"),
        "manifest_sha256": _sha(run / "manifest.json"),
        "technical_gate_sha256": _sha(run / "technical_gate.json"),
        "stdout_sha256": _sha(root / "stdout.log"),
        "stderr_sha256": _sha(root / "stderr.log"),
        "config_sha256": config_sha,
        "trainer_sha256": comparison._sha256(comparison.TRAINER_PATH),
    }
    _write_json(root / "terminal.json", controller_terminal)
    _inventory(root, comparison.CONTROLLER_INVENTORY_SCHEMA, count=False)


def _make_suite(
    root: Path, *, failing_index: int | None = None, effective_sigmas: list[float] | None = None
) -> None:
    root.mkdir(parents=True)
    preflight_terminal = _make_prerequisite_terminal(
        root.parent / f"{root.name}-technical-preflight",
        mode="technical-preflight",
    )
    for index in range(8):
        target0_terminal = root / "target-00" / "terminal.json" if index > 0 else None
        _make_target(
            root / f"target-{index:02d}",
            index,
            preflight_terminal=preflight_terminal,
            target0_formal_terminal=target0_terminal,
            passing=index != failing_index,
            effective_sigmas=effective_sigmas,
        )


def _resign(root: Path) -> None:
    run = root / "run"
    summary_path = run / "r11_new_phase1a_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key, relative in {
        "manifest_sha256": "manifest.json",
        "metrics_sha256": "metrics.jsonl",
        "evaluation_rows_sha256": "target_evaluation_rows.jsonl",
        "endpoint_raw_sha256": "endpoint_raw.pt",
        "endpoint_png_sha256": "endpoint_raw.png",
        "snapshot_end_sha256": "model_snapshot_verification_end.json",
        "technical_gate_sha256": "technical_gate.json",
    }.items():
        summary["artifacts"][key] = _sha(run / relative)
    _write_json(summary_path, summary)
    terminal_path = root / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal["summary_sha256"] = _sha(summary_path)
    terminal["manifest_sha256"] = _sha(run / "manifest.json")
    terminal["technical_gate_sha256"] = _sha(run / "technical_gate.json")
    terminal["stdout_sha256"] = _sha(root / "stdout.log")
    terminal["stderr_sha256"] = _sha(root / "stderr.log")
    _write_json(terminal_path, terminal)
    _inventory(run, comparison.TRAINER_INVENTORY_SCHEMA, count=True)
    _inventory(root, comparison.CONTROLLER_INVENTORY_SCHEMA, count=False)


class R11NewPhase1AComparisonTest(unittest.TestCase):
    def test_raw_float32_scheduler_observations_survive_full_recomputation(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            _make_suite(run_root, effective_sigmas=FLOAT32_EFFECTIVE_SIGMAS)
            result = comparison.compare(run_root, workspace / "report", expected_commit=EXPECTED_COMMIT)
            self.assertTrue(result["engineering_gate"])
            self.assertTrue(result["phase1a_query_level_reachability_gate"])
            self.assertFalse(result["formal_success"])
            self.assertFalse(result["scientific_success_claim"])
            run = run_root / "target-00" / "run"
            payload = torch.load(run / "checkpoints" / "step-000.pt", weights_only=False)
            self.assertEqual(payload["effective_sigmas"], FLOAT32_EFFECTIVE_SIGMAS)
            manifest = comparison._load(run / "manifest.json")
            self.assertEqual(
                manifest["fixed_contract"]["effective_sigmas"], list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE)
            )

    def test_checkpoint_rejects_invalid_runtime_sigmas(self):
        cases = {
            "beyond-tolerance": [0.49999, 0.375, 0.25, 0.125],
            "wrong-length": [0.5, 0.375, 0.25],
            "nan": [float("nan"), 0.375, 0.25, 0.125],
        }
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prerequisite = _make_prerequisite_terminal(
                workspace / "preflight", mode="technical-preflight"
            )
            for name, sigmas in cases.items():
                with self.subTest(name=name):
                    target = workspace / name
                    _make_target(
                        target, 0, preflight_terminal=prerequisite,
                        target0_formal_terminal=None, effective_sigmas=sigmas,
                    )
                    run = target / "run"
                    with self.assertRaisesRegex(ValueError, "checkpoint/hash record validation failed"):
                        comparison._validate_checkpoint_artifacts(
                            run,
                            manifest_sha256=_sha(run / "manifest.json"),
                            condition_sha256=_sha(run / "condition" / "official_full_condition.pt"),
                        )

    def test_checkpoint_rejects_different_in_tolerance_record_and_payload_sigmas(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            prerequisite = _make_prerequisite_terminal(
                workspace / "preflight", mode="technical-preflight"
            )
            target = workspace / "target-00"
            _make_target(
                target, 0, preflight_terminal=prerequisite,
                target0_formal_terminal=None, effective_sigmas=FLOAT32_EFFECTIVE_SIGMAS,
            )
            run = target / "run"
            checkpoint = run / "checkpoints" / "step-000.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            payload["effective_sigmas"] = list(core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE)
            torch.save(payload, checkpoint)
            record_path = run / "checkpoint_hashes" / "step-000.json"
            record = comparison._load(record_path)
            self.assertTrue(core.phase1a_effective_sigmas_match(payload["effective_sigmas"]))
            self.assertTrue(core.phase1a_effective_sigmas_match(record["effective_sigmas"]))
            self.assertNotEqual(payload["effective_sigmas"], record["effective_sigmas"])
            # Valid file hashes cannot make two different raw observations agree.
            record["checkpoint_bytes"] = checkpoint.stat().st_size
            record["checkpoint_sha256"] = _sha(checkpoint)
            _write_json(record_path, record)
            with self.assertRaisesRegex(ValueError, "checkpoint tensor hashes/payload drifted"):
                comparison._validate_checkpoint_artifacts(
                    run,
                    manifest_sha256=_sha(run / "manifest.json"),
                    condition_sha256=_sha(run / "condition" / "official_full_condition.pt"),
                )

    def test_valid_eight_of_eight_proceeds_to_phase2_but_never_claims_success(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            output = workspace / "report"
            _make_suite(run_root)
            result = comparison.compare(run_root, output, expected_commit=EXPECTED_COMMIT)
            self.assertTrue(result["engineering_gate"])
            self.assertTrue(result["phase1a_query_level_reachability_gate"])
            self.assertEqual(result["target_pass_count"], 8)
            self.assertEqual(result["decision"], "proceed_to_phase2_oracle_bank_mvp")
            self.assertFalse(result["formal_success"])
            self.assertFalse(result["scientific_success_claim"])
            self.assertEqual(result["phase1b_status"], "deferred_future_state_level_confirmation")
            self.assertTrue((output / "RAW_ARTIFACTS.json").is_file())
            self.assertTrue((output / "comparison.json").is_file())
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("科学成功 | `false`", report)
            self.assertIn("Phase 1B", report)

    def test_missing_launch_prerequisites_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            _make_suite(run_root)
            target = run_root / "target-03"
            launch_path = target / "launch.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            del launch["prerequisites"]
            _write_json(launch_path, launch)
            _resign(target)
            with self.assertRaisesRegex(ValueError, "launch prerequisites contract"):
                comparison.compare(run_root, workspace / "report", expected_commit=EXPECTED_COMMIT)

    def test_wrong_preflight_prerequisite_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            _make_suite(run_root)
            target = run_root / "target-04"
            launch_path = target / "launch.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["prerequisites"]["preflight"]["terminal_sha256"] = "0" * 64
            _write_json(launch_path, launch)
            _resign(target)
            with self.assertRaisesRegex(ValueError, "preflight prerequisite terminal SHA256"):
                comparison.compare(run_root, workspace / "report", expected_commit=EXPECTED_COMMIT)

    def test_all_targets_must_bind_the_same_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            _make_suite(run_root)
            other_preflight = _make_prerequisite_terminal(
                workspace / "other-technical-preflight",
                mode="technical-preflight",
            )
            target = run_root / "target-07"
            launch_path = target / "launch.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["prerequisites"]["preflight"] = _prerequisite_binding(
                other_preflight,
                mode="technical-preflight",
            )
            _write_json(launch_path, launch)
            _resign(target)
            with self.assertRaisesRegex(ValueError, "one identical technical-preflight"):
                comparison.compare(run_root, workspace / "report", expected_commit=EXPECTED_COMMIT)

    def test_later_target_must_bind_this_suite_target0(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            _make_suite(run_root)
            other_target0 = _make_prerequisite_terminal(
                workspace / "other-formal-target0",
                mode="formal",
            )
            target = run_root / "target-02"
            launch_path = target / "launch.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["prerequisites"]["target0_formal"] = _prerequisite_binding(
                other_target0,
                mode="formal",
            )
            _write_json(launch_path, launch)
            _resign(target)
            with self.assertRaisesRegex(ValueError, "this aggregation's target-00 terminal"):
                comparison.compare(run_root, workspace / "report", expected_commit=EXPECTED_COMMIT)

    def test_reachability_false_is_a_valid_diagnostic_not_a_technical_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            output = workspace / "report"
            _make_suite(run_root, failing_index=3)
            result = comparison.compare(run_root, output, expected_commit=EXPECTED_COMMIT)
            self.assertTrue(result["engineering_gate"])
            self.assertFalse(result["phase1a_query_level_reachability_gate"])
            self.assertEqual(result["target_pass_count"], 7)
            self.assertEqual(
                result["decision"], "run_canonical_r11_latent_bridge_distance_diagnostic"
            )
            self.assertFalse(result["formal_success"])

    def test_tampered_checkpoint_is_rejected_by_raw_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            _make_suite(run_root)
            (run_root / "target-04" / "run" / "checkpoints" / "step-128.pt").write_bytes(
                b"tampered"
            )
            with self.assertRaisesRegex(ValueError, "size/SHA validation"):
                comparison.compare(run_root, workspace / "report", expected_commit=EXPECTED_COMMIT)

    def test_truncated_receipts_are_rejected_even_after_hashes_are_resigned(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            _make_suite(run_root)
            target = run_root / "target-02"
            metrics = target / "run" / "metrics.jsonl"
            metrics.write_text("\n".join(metrics.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
            _resign(target)
            with self.assertRaisesRegex(ValueError, "exact 256 raw optimizer receipts"):
                comparison.compare(run_root, workspace / "report", expected_commit=EXPECTED_COMMIT)

    def test_duplicate_target_root_is_rejected_before_summary_use(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            _make_suite(run_root)
            target = run_root / "target-01"
            terminal_path = target / "terminal.json"
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            terminal["target_index"] = 0
            terminal["target_segment_id"] = core.R11_NEW_TARGET_IDS[0]
            _write_json(terminal_path, terminal)
            _inventory(target, comparison.CONTROLLER_INVENTORY_SCHEMA, count=False)
            with self.assertRaisesRegex(ValueError, "duplicate target"):
                comparison.compare(run_root, workspace / "report", expected_commit=EXPECTED_COMMIT)

    def test_any_formal_success_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_root = workspace / "runs"
            _make_suite(run_root)
            target = run_root / "target-05"
            terminal_path = target / "terminal.json"
            terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
            terminal["formal_success"] = True
            _write_json(terminal_path, terminal)
            _inventory(target, comparison.CONTROLLER_INVENTORY_SCHEMA, count=False)
            with self.assertRaisesRegex(ValueError, "formal_success=false"):
                comparison.compare(run_root, workspace / "report", expected_commit=EXPECTED_COMMIT)


if __name__ == "__main__":
    unittest.main()
