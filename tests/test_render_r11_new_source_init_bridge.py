from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import render_r11_new_source_init_bridge_delivery as delivery  # noqa: E402
from vision_memory.data import REVERSE_CYCLIC4  # noqa: E402


COMMIT = "c" * 40


@pytest.fixture(scope="module", autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def _inventory(root: Path, schema: str) -> dict:
    inventory = root / "artifact_inventory.json"
    rows = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
             "sha256": delivery.sha256(path)}
            for path in sorted(root.rglob("*")) if path.is_file() and path != inventory]
    _write(inventory, {"schema": schema, "artifact_count": len(rows), "artifacts": rows})
    return {"sha256": delivery.sha256(inventory), "artifact_count": len(rows)}


def _resign(paths: dict) -> None:
    raw = paths["raw"]
    for label in ("preflight", "formal"):
        root = paths[f"{label}_root"]
        binding = raw[f"{label}_controller"]
        binding["trainer_inventory"] = _inventory(root / "run", delivery.aggregate.TRAINER_INVENTORY_SCHEMA)
        binding["root_inventory"] = _inventory(root, delivery.aggregate.controller.INVENTORY_SCHEMA)
    agg = paths["aggregation_root"]
    _write(agg / "RAW_ARTIFACTS.json", raw)
    paths["comparison"]["raw_artifacts_sha256"] = delivery.sha256(agg / "RAW_ARTIFACTS.json")
    _write(agg / "comparison.json", paths["comparison"])
    _inventory(agg, delivery.aggregate.INVENTORY_SCHEMA)


def _reader_rows(target: dict, *, endpoint_correct: bool) -> list[dict]:
    rows = []
    for checkpoint, condition in (
        ("canonical_teacher", "teacher"), ("m0", "normal"), ("m0", "reset"),
        (delivery.core.BRIDGE_PRIMARY_ENDPOINT, "normal"), (delivery.core.BRIDGE_PRIMARY_ENDPOINT, "reset"),
    ):
        correct = checkpoint == "canonical_teacher" or (
            checkpoint == delivery.core.BRIDGE_PRIMARY_ENDPOINT and condition == "normal" and endpoint_correct
        )
        prediction = 1 if correct else 0
        for view, permutation in enumerate(REVERSE_CYCLIC4):
            logits = [-2.0] * 4
            logits[permutation.index(prediction)] = 8.0
            ordered_target = permutation.index(1)
            ce = 8.0 + math.log(sum(math.exp(value - 8.0) for value in logits)) - logits[ordered_target]
            rows.append({
                "schema": "vision_memory.r5-compose-causal-evaluation.v1",
                "suite": delivery.aggregate.controller.trainer.SUITE,
                "checkpoint": checkpoint, "condition": condition, "view_index": view,
                "permutation": list(permutation), "choice_logits_ordered": logits,
                "target_index": 1, "target_text": "ambient", "predicted_index": prediction,
                "predicted_text": target["query"]["choices"][prediction], "ce": ce, "correct": correct,
                "margin": logits[ordered_target] - max(value for i, value in enumerate(logits) if i != ordered_target),
                "item_id": target["segment_id"], "pair_unit": target["segment_id"], "donor_item_id": None,
                "episode_id": "episode", "query_id": "episode:3", "query_gap": 1,
                "updater_count": 1, "target_event_kind": "set",
            })
    return rows


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, endpoint_correct: bool = False,
             endpoint_offset: float = 0.2) -> dict:
    core, aggregate = delivery.core, delivery.aggregate
    preflight, formal, agg = [tmp_path / name for name in ("preflight", "formal", "aggregation")]
    for root in (preflight, formal):
        (root / "run" / "teacher").mkdir(parents=True)
        (root / "run" / "initialization").mkdir()
        _write(root / "terminal.json", {"git_commit": COMMIT, "technical_gate": True,
                                       "formal_success": False, "phase2_allowed": False})
    agg.mkdir()
    teacher = torch.linspace(-1, 1, 65536, dtype=torch.float32).reshape(1, 4, 128, 128)
    teacher_path = formal / "run" / "teacher" / "canonical_r11_target01.pt"
    torch.save({"schema": "vision_memory.r11-vae-latent-endpoint.v1", "latent_fp32": teacher}, teacher_path)
    shutil.copyfile(teacher_path, preflight / "run" / "teacher" / teacher_path.name)
    monkeypatch.setattr(core, "BRIDGE_TEACHER_FILE_SHA256", delivery.sha256(teacher_path))
    monkeypatch.setattr(core, "BRIDGE_TEACHER_TENSOR_SHA256", delivery.canonical_tensor_sha256(teacher))
    monkeypatch.setattr(core, "BRIDGE_TEACHER_STD", float(teacher.std(unbiased=False)))
    Image.new("RGB", (12, 12), "white").save(formal / "run" / "teacher" / "canonical_r11_target01.png")
    initialization = {}
    for label, root in (("preflight", preflight), ("formal", formal)):
        artifact = root / "run" / "initialization" / "source_only_initialization.pt"
        # This fixture covers hash binding; only the CUDA aggregator tests arithmetic.
        artifact.write_bytes(b"fixture CUDA-aggregator-bound initialization")
        initialization[label] = {"passed": True, "compute_device_type": "cuda",
                                 "compute_dtype": "torch.bfloat16", "artifact_sha256": delivery.sha256(artifact),
                                 "initial_x_t_equals_source": True, "initialization_teacher_assisted": False}
    preflight_report = {
        "passed": True, "bridge_result_evaluated": False, "formal_success_gate": False, "phase2_allowed": False,
        "audit": {"full_forward_calls": 1, "backward_calls": 1, "optimizer_steps": 0,
                  **{key: True for key in ("four_dreamlite_steps", "finite_nonzero_x_T_gradient", "only_x_T_fp32_trainable",
                                          "frozen_gradients_absent", "source_only_initialization_artifact_valid", "snapshots_unchanged")}},
    }
    _write(preflight / "run" / "technical_preflight.json", preflight_report)
    _write(preflight / "run" / "r11_new_bridge_summary.json", preflight_report)
    run = formal / "run"
    for folder in ("checkpoints", "checkpoint_hashes", "images"):
        (run / folder).mkdir()
    m0_mse = float((teacher.add(0.5) - teacher).square().mean())
    checkpoint_records = []
    for step in delivery.STEPS:
        z_t = teacher.add(0.5 + (endpoint_offset - 0.5) * step / 256)
        x_t = teacher.mul(2).add(step / 256)
        trajectory = [x_t.clone()] * 4 + [z_t]
        distance = core.bridge_distance_statistics(mse=float((z_t - teacher).square().mean()),
                                                   m0_mse=m0_mse, tensor_numel=65536)
        optimizer = {"state": {} if step == 0 else {0: {"step": torch.tensor(float(step)),
                     "exp_avg": torch.zeros_like(teacher), "exp_avg_sq": torch.zeros_like(teacher)}},
                     "param_groups": [{"params": [0], "lr": 0.05 if step == 0 else core.bridge_optimizer_learning_rate(step)}]}
        tensor_hashes = {"x_T_fp32": delivery.canonical_tensor_sha256(x_t),
                         "z_t_fp32": delivery.canonical_tensor_sha256(z_t),
                         "trajectory_fp32": [delivery.canonical_tensor_sha256(t) for t in trajectory]}
        payload = {"optimizer_step": step, "x_T_fp32": x_t, "z_t_fp32": z_t, "trajectory_fp32": trajectory,
                   "tensor_sha256": tensor_hashes, "optimizer": optimizer,
                   "optimizer_state_sha256": delivery.canonical_object_sha256(optimizer), "distance_statistics": distance}
        path = run / "checkpoints" / f"step-{step:03d}.pt"
        torch.save(payload, path)
        image = run / "images" / f"step-{step:03d}.png"
        Image.new("RGB", (12, 12), (step % 256, 100, 200)).save(image)
        hash_record = run / "checkpoint_hashes" / f"step-{step:03d}.json"
        _write(hash_record, {"checkpoint_sha256": delivery.sha256(path)})
        checkpoint_records.append({"optimizer_step": step, "checkpoint_sha256": delivery.sha256(path),
                                   "image_sha256": delivery.sha256(image), "record_sha256": delivery.sha256(hash_record),
                                   "tensor_sha256": tensor_hashes, "optimizer_state_sha256": payload["optimizer_state_sha256"],
                                   "distance_statistics": distance})
    shutil.copyfile(run / "checkpoints" / "step-256.pt", run / "endpoint_raw.pt")
    shutil.copyfile(run / "images" / "step-256.png", run / "endpoint_raw.png")
    metrics = []
    for step in range(1, 257):
        mse = m0_mse * (1 - (step - 1) / 512)
        metrics.append({"schema": aggregate.TRAINER_METRICS_SCHEMA, "optimizer_step": step,
                        "target_index": 1, "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID,
                        "objective": "canonical_r11_latent_fp32_mean_mse", "loss_before_step": mse,
                        **core.bridge_distance_statistics(mse=mse, m0_mse=m0_mse, tensor_numel=65536),
                        "gradient_norm": 1.0, "gradient_nonzero_fraction": 1.0, "x_T_update_norm": 0.0 if step == 256 else 0.1,
                        "elapsed_seconds": float(step), "full_dreamlite_forward_executed": True,
                        "dreamlite_denoising_steps": 4, "trajectory_points": 5, "effective_sigmas": list(aggregate.EXPECTED_SIGMAS),
                        "learning_rate": core.bridge_optimizer_learning_rate(step), "weight_decay": 0.0,
                        "gradient_clipping_applied": False, "reader_gradient_calls": 0,
                        "teacher_tensor_sha256": core.BRIDGE_TEACHER_TENSOR_SHA256})
    _write_rows(run / "metrics.jsonl", metrics)
    target = {"segment_id": core.BRIDGE_TARGET_SEGMENT_ID, "events": [{"event_kind": "set"}],
              "query_source_episode_id": "episode", "query_turn_index": 3, "query_gap": 1,
              "query": {"choices": ["none", "ambient", "blues", "jazz"], "target_index": 1}}
    rows = _reader_rows(target, endpoint_correct=endpoint_correct)
    _write_rows(run / "evaluation_rows.jsonl", rows)
    def stat(checkpoint, condition):
        return core.reader_checkpoint_statistics(rows, checkpoint=checkpoint, condition=condition)

    endpoint = checkpoint_records[-1]["distance_statistics"]
    endpoint_reader = stat(core.BRIDGE_PRIMARY_ENDPOINT, "normal")
    distance_gate = core.bridge_distance_gate(endpoint, technical_gate=True)
    reader_gate = core.endpoint_reader_transfer_gate(endpoint_reader)
    secondary = core.bridge_initialization_hypothesis_audit(
        endpoint_mse=endpoint["mse"], endpoint_reader_mean_ce=endpoint_reader["mean_ce"], technical_gate=True,
        teacher_replay_gate=True, distance_pass=distance_gate, reader_transfer_pass=reader_gate)
    parent = {"training_git_commit": "1" * 40, "comparison_sha256": "2" * 64,
              "raw_artifacts_sha256": "3" * 64, "config_sha256": "4" * 64}
    monkeypatch.setattr(aggregate, "_validate_config", lambda: {"parent_bridge": parent})
    comparison = {
        "schema": aggregate.COMPARISON_SCHEMA, "status": "completed", "protocol": core.BRIDGE_PROTOCOL,
        "git_commit": COMMIT, "engineering_gate": True, "teacher_replay_gate": True, "optimizer_steps": 256,
        "target_index": 1, "target_segment_id": core.BRIDGE_TARGET_SEGMENT_ID, "primary_endpoint": core.BRIDGE_PRIMARY_ENDPOINT,
        "initialization_teacher_assisted": False, "optimization_teacher_supervised": True, "answer_independent_writer_usable": False,
        "formal_success": False, "phase2_allowed": False, "scientific_success_claim": False, "interpretation": "diagnostic_only",
        "bridge_distance_gate": distance_gate, "endpoint_reader_transfer_gate": reader_gate,
        "bridge_diagnostic_gate": distance_gate and reader_gate,
        "endpoint_distance_statistics": endpoint, "endpoint_reader_statistics": endpoint_reader,
        "m0_reader_statistics": stat("m0", "normal"), "m0_reset_statistics": stat("m0", "reset"),
        "endpoint_reset_statistics": stat(core.BRIDGE_PRIMARY_ENDPOINT, "reset"),
        "teacher_replay_statistics": stat("canonical_teacher", "teacher"),
        "checkpoint_distance_statistics": {str(row["optimizer_step"]): row["distance_statistics"] for row in checkpoint_records},
        "decision": core.bridge_decision(distance_pass=distance_gate, reader_transfer_pass=reader_gate),
        "secondary_solver_hypothesis_audit": secondary,
        "secondary_solver_hypothesis_decision": core.bridge_initialization_hypothesis_decision(
            distance_pass=distance_gate, reader_transfer_pass=reader_gate, audit=secondary),
    }
    raw = {"schema": aggregate.RAW_SCHEMA, "expected_commit": COMMIT, "preflight_controller": {}, "formal_controller": {},
           "preflight": {"summary_sha256": delivery.sha256(preflight / "run" / "r11_new_bridge_summary.json")},
           "initialization": initialization, "metrics": {"sha256": delivery.sha256(run / "metrics.jsonl"), "checks": {"optimizer": True}},
           "checkpoints": checkpoint_records, "evaluation_rows": {"sha256": delivery.sha256(run / "evaluation_rows.jsonl")},
           "locked_parent_target": {"target_segment": target}, "locked_parent_bridge": parent}
    paths = {"preflight_root": preflight, "formal_root": formal, "aggregation_root": agg,
             "comparison": comparison, "raw": raw}
    _resign(paths)
    return paths


def _inputs(paths: dict) -> dict:
    return {key: paths[key] for key in ("preflight_root", "formal_root", "aggregation_root")}


@pytest.mark.parametrize(("distance_pass", "reader_pass"), [(False, False), (True, False), (False, True), (True, True)])
def test_all_primary_outcomes_remain_diagnostic_and_have_true_coordinates(tmp_path, monkeypatch, distance_pass, reader_pass):
    paths = _fixture(tmp_path, monkeypatch, endpoint_correct=reader_pass, endpoint_offset=0.001 if distance_pass else 0.2)
    evidence = delivery.collect_evidence(**_inputs(paths), expected_commit=COMMIT)
    assert evidence["comparison"]["bridge_distance_gate"] is distance_pass
    assert evidence["comparison"]["endpoint_reader_transfer_gate"] is reader_pass
    assert evidence["formal_success"] is False and evidence["phase2_allowed"] is False
    assert [row["completed_updates"] for row in evidence["receipts"]] == list(range(256))
    assert [row["completed_updates"] for row in evidence["checkpoints"]] == list(delivery.STEPS)
    assert evidence["phase1a_passed"] == 6
    text = delivery._readme(evidence)
    assert "phase2_allowed=false" in text and "不同 M0" in text
    assert "prefix_parity" not in evidence


def test_render_has_real_figures_relative_links_and_complete_manifest(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "delivery"
    manifest = delivery.render(**_inputs(paths), output_dir=output, expected_commit=COMMIT)
    assert len(manifest["artifacts"]) == 7
    for name, record in manifest["artifacts"].items():
        assert delivery.sha256(output / name) == record["sha256"]
    for name in ("distance_trajectory.png", "optimizer_diagnostics.png", "reader_transfer.png", "checkpoint_images.png"):
        with Image.open(output / name) as image:
            assert image.width > 1000 and image.height > 300
        assert f"]({name})" in (output / "README.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="fresh"):
        delivery.render(**_inputs(paths), output_dir=output)


@pytest.mark.parametrize("field", ["formal_success", "phase2_allowed", "answer_independent_writer_usable"])
def test_resigned_promotion_is_rejected_before_output_creation(tmp_path, monkeypatch, field):
    paths = _fixture(tmp_path, monkeypatch)
    paths["comparison"][field] = True
    _resign(paths)
    with pytest.raises(ValueError, match="boundary"):
        delivery.render(**_inputs(paths), output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_resigned_checkpoint_distance_is_recomputed(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    paths["comparison"]["checkpoint_distance_statistics"]["64"]["mse"] = 0.0
    # The fixture's RAW and comparison share this dict; both claims are forged.
    _resign(paths)
    with pytest.raises(ValueError, match="arithmetic"):
        delivery.collect_evidence(**_inputs(paths))


def test_resigned_reader_ce_is_recomputed(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    row_path = paths["formal_root"] / "run" / "evaluation_rows.jsonl"
    rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["ce"] += 1
    _write_rows(row_path, rows)
    paths["raw"]["evaluation_rows"]["sha256"] = delivery.sha256(row_path)
    _resign(paths)
    with pytest.raises(ValueError, match="CE"):
        delivery.collect_evidence(**_inputs(paths))


def test_preflight_updates_rejected_even_if_resigned(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    run = paths["preflight_root"] / "run"
    summary = json.loads((run / "technical_preflight.json").read_text(encoding="utf-8"))
    summary["audit"]["optimizer_steps"] = 1
    _write(run / "technical_preflight.json", summary)
    _write(run / "r11_new_bridge_summary.json", summary)
    paths["raw"]["preflight"]["summary_sha256"] = delivery.sha256(run / "r11_new_bridge_summary.json")
    _resign(paths)
    with pytest.raises(ValueError, match="zero updates"):
        delivery.collect_evidence(**_inputs(paths))


def test_no_cpu_initializer_fallback_and_no_output_in_inputs(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="immutable input"):
        delivery.render(**_inputs(paths), output_dir=paths["formal_root"] / "new-report")
    paths["raw"]["initialization"]["formal"]["compute_device_type"] = "cpu"
    _resign(paths)
    with pytest.raises(ValueError, match="CUDA-verified"):
        delivery.collect_evidence(**_inputs(paths))


def test_wrong_commit_and_unbound_source_bytes_are_rejected(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="training commit"):
        delivery.collect_evidence(**_inputs(paths), expected_commit="f" * 40)
    (paths["formal_root"] / "run" / "metrics.jsonl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Inventory size/hash"):
        delivery.collect_evidence(**_inputs(paths))


def test_cross_platform_cosine_is_not_substituted_for_byte_bound_runtime_audit(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    metric_path = paths["formal_root"] / "run" / "metrics.jsonl"
    rows = [json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines()]
    rows[162]["learning_rate"] = math.nextafter(rows[162]["learning_rate"], 0)
    _write_rows(metric_path, rows)
    paths["raw"]["metrics"]["sha256"] = delivery.sha256(metric_path)
    _resign(paths)
    evidence = delivery.collect_evidence(**_inputs(paths))
    assert evidence["receipts"][162]["learning_rate"] == rows[162]["learning_rate"]
    paths["raw"]["metrics"]["checks"]["optimizer"] = False
    _resign(paths)
    with pytest.raises(ValueError, match="Source-runtime exact optimizer"):
        delivery.collect_evidence(**_inputs(paths))


def test_local_cpu_reduction_never_replaces_original_display_statistics(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    original = dict(paths["raw"]["checkpoints"][-1]["distance_statistics"])
    remote_statistics = delivery.core.bridge_distance_statistics(
        mse=original["mse"] + 1e-9, m0_mse=original["m0_mse"], tensor_numel=65536,
    )
    paths["raw"]["checkpoints"][-1]["distance_statistics"] = remote_statistics
    paths["comparison"]["endpoint_distance_statistics"] = remote_statistics
    paths["comparison"]["checkpoint_distance_statistics"]["256"] = remote_statistics
    _resign(paths)
    evidence = delivery.collect_evidence(**_inputs(paths))
    checkpoint = evidence["checkpoints"][-1]
    assert checkpoint["mse"] == remote_statistics["mse"]
    assert checkpoint["local_cpu_recomputed_distance_statistics"]["mse"] == original["mse"]
    assert checkpoint["mse"] != checkpoint["local_cpu_recomputed_distance_statistics"]["mse"]
    assert evidence["comparison"]["endpoint_distance_statistics"]["mse"] == checkpoint["mse"]
    report = delivery._readme(evidence)
    assert f"| 256 | {remote_statistics['mse']:.10g} |" in report
    assert "local_cpu_recomputed_distance_statistics" in report
