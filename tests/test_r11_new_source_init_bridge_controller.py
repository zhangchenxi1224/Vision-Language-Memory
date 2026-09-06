from __future__ import annotations

import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.inspire import run_r11_new_source_init_bridge as controller  # noqa: E402
from vision_memory.data import REVERSE_CYCLIC4  # noqa: E402


COMMIT = "a" * 40


def _target() -> dict:
    return {
        "schema": "vision_memory.r5-compose-segments.v1",
        "segment_id": controller.core.BRIDGE_TARGET_SEGMENT_ID,
        "query_source_episode_id": "episode", "query_turn_index": 3, "query_gap": 1,
        "events": [{"event_kind": "set"}],
        "query": {"choices": ["none", "ambient", "blues", "jazz"], "target_index": 1},
    }


@pytest.fixture(autouse=True)
def _locked_phase1a_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "locked-phase1a-parent"
    path = root / "run" / "manifest.json"
    _write_json(path, {
        "schema": "vision_memory.r11-new-phase1a-manifest.v1",
        "target_index": controller.core.BRIDGE_TARGET_INDEX,
        "target_segment_id": controller.core.BRIDGE_TARGET_SEGMENT_ID,
        "target_segment": _target(),
    })
    monkeypatch.setattr(controller.core, "BRIDGE_PHASE1A_SOURCE_ROOT", str(root))
    monkeypatch.setattr(controller.core, "BRIDGE_PARENT_TARGET_MANIFEST_SHA256", controller._sha256(path))
    return path


def _teacher_rows() -> list[dict]:
    target = _target()
    rows = []
    for view, permutation in enumerate(REVERSE_CYCLIC4):
        logits = [-2.0] * 4
        logits[permutation.index(1)] = 8.0
        ce = 8.0 + math.log(sum(math.exp(value - 8.0) for value in logits)) - 8.0
        rows.append({
            "schema": "vision_memory.r5-compose-causal-evaluation.v1", "suite": controller.trainer.SUITE,
            "checkpoint": "canonical_teacher", "condition": "teacher", "view_index": view,
            "permutation": list(permutation), "choice_logits_ordered": logits, "ce": ce,
            "target_index": 1, "target_text": "ambient", "predicted_index": 1,
            "predicted_text": "ambient", "correct": True, "margin": 10.0,
            "item_id": target["segment_id"], "pair_unit": target["segment_id"], "donor_item_id": None,
            "episode_id": "episode", "query_id": "episode:3", "query_gap": 1,
            "updater_count": 1, "target_event_kind": "set",
        })
    return rows


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _inventory(root: Path, schema: str) -> None:
    artifacts = []
    inventory_path = root / "artifact_inventory.json"
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path == inventory_path:
            continue
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": controller._sha256(path),
            }
        )
    _write_json(
        inventory_path,
        {
            "schema": schema,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
        },
    )


def _common_child(run: Path, *, mode: str) -> tuple[dict, dict]:
    run.mkdir(parents=True)
    initialization_path = run / "initialization" / "source_only_initialization.pt"
    initialization_path.parent.mkdir()
    initialization_path.write_bytes(b"controller-test-artifact-not-a-tensor-arithmetic-fixture")
    manifest = {
        "schema": controller.trainer.MANIFEST_SCHEMA,
        "protocol": controller.core.BRIDGE_PROTOCOL,
        "mode": mode,
        "git_commit": COMMIT,
        "git_dirty": False,
        "target_index": controller.core.BRIDGE_TARGET_INDEX,
        "target_segment_id": controller.core.BRIDGE_TARGET_SEGMENT_ID,
        "target_segment": _target(),
        "information_boundary": {
            "initialization_teacher_assisted": False,
            "optimization_teacher_supervised": True,
            "answer_independent_writer_usable": False,
        },
        "initialization_binding": {
            "schema": controller.trainer.INITIALIZATION_SCHEMA,
            "passed": True,
            "artifact_path": str(initialization_path.resolve()),
            "artifact_bytes": initialization_path.stat().st_size,
            "artifact_sha256": controller._sha256(initialization_path),
        },
    }
    terminal = {
        "schema": controller.trainer.TERMINAL_SCHEMA,
        "status": "technical_completed",
        "mode": mode,
        "target_index": controller.core.BRIDGE_TARGET_INDEX,
        "technical_gate": True,
        "formal_success": False,
        "bridge_diagnostic_gate": None if mode == "technical-preflight" else False,
    }
    _write_json(run / "manifest.json", manifest)
    _write_json(
        run / "model_snapshot_verification_end.json",
        {"passed": True},
    )
    _write_json(run / "terminal.json", terminal)
    return manifest, terminal


def _make_preflight(run: Path) -> dict:
    _common_child(run, mode="technical-preflight")
    summary = {
        "schema": "vision_memory.r11-new-canonical-latent-bridge-preflight.v1",
        "mode": "technical-preflight",
        "target_index": controller.core.BRIDGE_TARGET_INDEX,
        "target_segment_id": controller.core.BRIDGE_TARGET_SEGMENT_ID,
        "passed": True,
        "audit": {
            "optimizer_steps": 0,
            "full_forward_calls": 1,
            "backward_calls": 1,
            **{name: True for name in controller.PREFLIGHT_REQUIRED_AUDIT_TRUE},
        },
        "bridge_result_evaluated": False,
        "formal_success_gate": False,
        "phase2_allowed": False,
        "teacher_replay_statistics": controller.core.reader_checkpoint_statistics(
            _teacher_rows(), checkpoint="canonical_teacher", condition="teacher",
        ),
    }
    _write_json(run / controller.SUMMARY_FILE, summary)
    _write_json(run / controller.PREFLIGHT_FILE, summary)
    _write_jsonl(run / controller.ROWS_FILE, _teacher_rows())
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "step-000.pt").write_bytes(b"step0")
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    return summary


def _sign_preflight_controller(root: Path) -> dict:
    """Re-sign all bytes without validating them, to test semantic fail-closed gates."""
    run = root / "run"
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    for name in ("stdout.log", "stderr.log"):
        (root / name).write_text("fixture log\n", encoding="utf-8")
    hashes = {"config_sha256": "1" * 64, "trainer_sha256": "2" * 64, "controller_sha256": "3" * 64}
    terminal = {
        "schema": controller.TERMINAL_SCHEMA, "status": "technical_completed", "mode": "technical-preflight",
        "target_index": controller.core.BRIDGE_TARGET_INDEX, "target_segment_id": controller.core.BRIDGE_TARGET_SEGMENT_ID,
        "technical_gate": True, "bridge_diagnostic_gate": None, "formal_success": False, "phase2_allowed": False,
        "child_exit_code": 0, "git_commit": COMMIT,
        "execution_checks": {key: True for key in (
            "child_contract_passed", "common_checks_passed", "mode_checks_passed", "artifact_hash_checks_passed",
        )},
        "summary_sha256": controller._sha256(run / controller.SUMMARY_FILE),
        "manifest_sha256": controller._sha256(run / "manifest.json"),
        "child_terminal_sha256": controller._sha256(run / "terminal.json"),
        "child_inventory_sha256": controller._sha256(run / "artifact_inventory.json"),
        "stdout_sha256": controller._sha256(root / "stdout.log"),
        "stderr_sha256": controller._sha256(root / "stderr.log"), **hashes,
    }
    _write_json(root / "terminal.json", terminal)
    _inventory(root, controller.INVENTORY_SCHEMA)
    return hashes


def _make_formal(run: Path, *, bridge_gate: bool = False) -> dict:
    _common_child(run, mode="formal")
    technical = {"passed": True}
    _write_json(run / "technical_gate.json", technical)
    _write_jsonl(
        run / controller.METRICS_FILE,
        [{"optimizer_step": step} for step in range(1, 257)],
    )
    _write_jsonl(run / controller.ROWS_FILE, [{"row": index} for index in range(20)])
    (run / "endpoint_raw.pt").write_bytes(b"endpoint")
    (run / "endpoint_raw.png").write_bytes(b"png")
    initialization_audit = controller.core.bridge_initialization_hypothesis_audit(
        endpoint_mse=0.09,
        endpoint_reader_mean_ce=25.0,
        technical_gate=True,
        teacher_replay_gate=True,
        distance_pass=bridge_gate,
        reader_transfer_pass=bridge_gate,
    )
    summary = {
        "schema": controller.trainer.SUMMARY_SCHEMA,
        "status": "completed",
        "mode": "formal",
        "target_index": controller.core.BRIDGE_TARGET_INDEX,
        "target_segment_id": controller.core.BRIDGE_TARGET_SEGMENT_ID,
        "technical_gate": technical,
        "gates": {
            "technical_gate": True,
            "teacher_replay_gate": True,
            "bridge_distance_gate": bridge_gate,
            "endpoint_reader_transfer_gate": bridge_gate,
            "bridge_diagnostic_gate": bridge_gate,
        },
        "decision": controller.core.bridge_decision(
            distance_pass=bridge_gate,
            reader_transfer_pass=bridge_gate,
        ),
        "secondary_solver_hypothesis_audit": initialization_audit,
        "endpoint_distance_statistics": {"mse": 0.09},
        "endpoint_reader_statistics": {"mean_ce": 25.0},
        "secondary_solver_hypothesis_decision": controller.core.bridge_initialization_hypothesis_decision(
            distance_pass=bridge_gate,
            reader_transfer_pass=bridge_gate,
            audit=initialization_audit,
        ),
        "checkpoint_steps_observed": list(controller.core.BRIDGE_CHECKPOINT_STEPS),
        "formal_success_gate": False,
        "phase2_allowed": False,
        "artifacts": {},
    }
    bindings = {
        "manifest_sha256": "manifest.json",
        "metrics_sha256": controller.METRICS_FILE,
        "evaluation_rows_sha256": controller.ROWS_FILE,
        "endpoint_raw_sha256": "endpoint_raw.pt",
        "endpoint_png_sha256": "endpoint_raw.png",
        "technical_gate_sha256": "technical_gate.json",
    }
    summary["artifacts"] = {key: controller._sha256(run / relative) for key, relative in bindings.items()}
    _write_json(run / controller.SUMMARY_FILE, summary)
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    return summary


def test_deployment_audit_requires_pinned_host_ssd_and_fresh_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(controller, "INSPIRE_SSD_ROOT", tmp_path)
    monkeypatch.setattr(controller, "MINIMUM_FREE_BYTES", 1)
    output = tmp_path / "fresh"
    audit = controller._deployment_audit(
        output,
        hostname=controller.EXPECTED_HOST_PREFIX + "-pod",
    )
    assert audit["passed"]
    with pytest.raises(ValueError, match="pinned"):
        controller._deployment_audit(output, hostname="wrong-host")
    output.mkdir()
    with pytest.raises(ValueError, match="nonexistent fresh"):
        controller._deployment_audit(
            output,
            hostname=controller.EXPECTED_HOST_PREFIX + "-pod",
        )


def test_suite_lock_is_exclusive_and_owner_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(controller, "LOCK_PATH", tmp_path / "bridge.lock")
    lock = controller._acquire_lock(
        mode="technical-preflight",
        output_root=tmp_path / "output",
        expected_commit=COMMIT,
    )
    with pytest.raises(ValueError, match="already held"):
        controller._acquire_lock(
            mode="formal",
            output_root=tmp_path / "other",
            expected_commit=COMMIT,
        )
    released = controller._release_lock(lock)
    assert released["released"] is True
    assert not (tmp_path / "bridge.lock").exists()


def test_controller_command_has_no_scientific_or_optimizer_knobs(tmp_path: Path) -> None:
    args = Namespace(
        mode="formal",
        train=tmp_path / "train",
        dev=tmp_path / "dev",
        dreamlite=tmp_path / "dreamlite",
        reader=tmp_path / "reader",
        teacher=tmp_path / "teacher",
        phase1a_comparison=tmp_path / "comparison",
        phase1a_raw_artifacts=tmp_path / "raw",
        phase1a_target_root=tmp_path / "target",
    )
    command = controller._command(args, tmp_path / "run")
    assert "--strict-determinism" in command
    assert command[command.index("--dreamlite-device") + 1] == "cuda:0"
    assert command[command.index("--reader-device") + 1] == "cuda:1"
    forbidden = {
        "--learning-rate",
        "--lr",
        "--optimizer",
        "--optimizer-steps",
        "--steps",
        "--gradient-clip",
        "--checkpoint-steps",
        "--target-index",
    }
    assert forbidden.isdisjoint(command)


def test_valid_preflight_child_passes_and_has_no_bridge_result(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _make_preflight(run)
    result = controller._validate_child(
        run,
        mode="technical-preflight",
        expected_commit=COMMIT,
    )
    assert result["passed"]
    assert result["mode_checks"]["bridge_not_evaluated"]


def test_valid_formal_child_can_be_diagnostic_failure(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _make_formal(run, bridge_gate=False)
    result = controller._validate_child(run, mode="formal", expected_commit=COMMIT)
    assert result["passed"]
    assert result["mode_checks"]["diagnostic_boolean"]


@pytest.mark.parametrize("field", [
    "initialization_teacher_assisted", "optimization_teacher_supervised", "answer_independent_writer_usable",
])
def test_source_only_information_boundary_rejects_resigned_claim_drift(tmp_path: Path, field: str) -> None:
    run = tmp_path / "run"
    _make_preflight(run)
    manifest = controller._load(run / "manifest.json")
    manifest["information_boundary"][field] = not manifest["information_boundary"][field]
    _write_json(run / "manifest.json", manifest)
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    with pytest.raises(ValueError, match="common contract"):
        controller._validate_child(run, mode="technical-preflight", expected_commit=COMMIT)


@pytest.mark.parametrize("mutation", ["missing", "wrong_protocol", "external_path", "wrong_hash"])
def test_child_rejects_resigned_initialization_binding_drift(tmp_path: Path, mutation: str) -> None:
    run = tmp_path / "run"
    _make_preflight(run)
    manifest_path = run / "manifest.json"
    manifest = controller._load(manifest_path)
    if mutation == "missing":
        manifest.pop("initialization_binding")
    elif mutation == "wrong_protocol":
        manifest["protocol"] = "R11-New-Canonical-Latent-Bridge-Post128-Cosine-Target01"
    elif mutation == "external_path":
        original = run / "initialization" / "source_only_initialization.pt"
        external = tmp_path / "external.pt"
        external.write_bytes(original.read_bytes())
        manifest["initialization_binding"]["artifact_path"] = str(external.resolve())
    else:
        manifest["initialization_binding"]["artifact_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    with pytest.raises(ValueError, match="common contract"):
        controller._validate_child(run, mode="technical-preflight", expected_commit=COMMIT)


@pytest.mark.parametrize("mutation", ["audit", "decision", "endpoint_ce"])
def test_formal_child_rejects_resigned_secondary_audit_drift(tmp_path: Path, mutation: str) -> None:
    run = tmp_path / "run"
    summary = _make_formal(run)
    if mutation == "audit":
        summary["secondary_solver_hypothesis_audit"]["passed"] = False
    elif mutation == "decision":
        summary["decision"] = controller.core.bridge_decision(distance_pass=True, reader_transfer_pass=True)
    else:
        summary["endpoint_reader_statistics"]["mean_ce"] = 26.0
    _write_json(run / controller.SUMMARY_FILE, summary)
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    with pytest.raises(ValueError, match="mode contract"):
        controller._validate_child(run, mode="formal", expected_commit=COMMIT)


def test_formal_child_rejects_missing_or_reordered_receipt(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    _make_formal(run)
    metrics = run / controller.METRICS_FILE
    rows = metrics.read_text(encoding="utf-8").splitlines()
    metrics.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    summary_path = run / controller.SUMMARY_FILE
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["artifacts"]["metrics_sha256"] = controller._sha256(metrics)
    _write_json(summary_path, summary)
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    with pytest.raises(ValueError, match="mode contract"):
        controller._validate_child(run, mode="formal", expected_commit=COMMIT)


def test_preflight_terminal_is_bound_to_same_commit_and_code(
    tmp_path: Path,
) -> None:
    root = tmp_path / "preflight"
    run = root / "run"
    _make_preflight(run)
    hashes = _sign_preflight_controller(root)
    binding = controller._validate_preflight_terminal(
        root / "terminal.json",
        expected_commit=COMMIT,
        **hashes,
    )
    assert binding["passed"]
    with pytest.raises(ValueError, match="exact validation"):
        controller._validate_preflight_terminal(
            root / "terminal.json",
            expected_commit="b" * 40,
            **hashes,
        )


@pytest.mark.parametrize("field", controller.PREFLIGHT_REQUIRED_AUDIT_TRUE)
@pytest.mark.parametrize("mutation", ["false", "missing"])
def test_preflight_rejects_each_resigned_hard_subgate_despite_top_level_pass(tmp_path, field, mutation):
    run = tmp_path / "run"
    summary = _make_preflight(run)
    if mutation == "false":
        summary["audit"][field] = False
    else:
        summary["audit"].pop(field)
    _write_json(run / controller.SUMMARY_FILE, summary)
    _write_json(run / controller.PREFLIGHT_FILE, summary)
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    assert summary["passed"] is True
    with pytest.raises(ValueError, match="mode contract"):
        controller._validate_child(run, mode="technical-preflight", expected_commit=COMMIT)


@pytest.mark.parametrize("mutation", [
    "bad_logits", "wrong_permutation", "duplicate_view", "too_few_rows", "target_label", "query_id",
    "summary_mean_ce", "summary_per_view_ce", "correct_flag", "wrong_prediction", "margin",
    "consistent_wrong_answer", "replay_ce_too_high",
])
def test_preflight_recomputes_teacher_logits_and_target_binding_after_resigning(tmp_path, mutation):
    run = tmp_path / "run"
    summary = _make_preflight(run)
    rows = _teacher_rows()
    if mutation == "bad_logits":
        rows[0]["choice_logits_ordered"] = [0.0] * 4
    elif mutation == "wrong_permutation":
        rows[0]["permutation"] = rows[1]["permutation"]
    elif mutation == "duplicate_view":
        rows[1]["view_index"] = 0
    elif mutation == "too_few_rows":
        rows.pop()
    elif mutation == "target_label":
        rows[0]["target_index"] = 0
        rows[0]["target_text"] = "none"
    elif mutation == "query_id":
        rows[0]["query_id"] = "another:1"
    elif mutation == "summary_mean_ce":
        summary["teacher_replay_statistics"]["mean_ce"] = 0.0
    elif mutation == "summary_per_view_ce":
        summary["teacher_replay_statistics"]["per_view_ce"][0] = 0.0
    elif mutation == "correct_flag":
        rows[0]["correct"] = False
    elif mutation == "wrong_prediction":
        rows[0]["predicted_index"], rows[0]["predicted_text"] = 0, "none"
    elif mutation == "margin":
        rows[0]["margin"] = 99.0
    elif mutation in {"consistent_wrong_answer", "replay_ce_too_high"}:
        for row in rows:
            permutation = row["permutation"]
            logits = [-2.0] * 4
            prediction = 0 if mutation == "consistent_wrong_answer" else 1
            logits[permutation.index(prediction)] = 8.0 if prediction == 0 else 0.0
            maximum = max(logits)
            row["choice_logits_ordered"] = logits
            row["predicted_index"], row["predicted_text"] = prediction, _target()["query"]["choices"][prediction]
            row["correct"] = prediction == 1
            ordered_target = permutation.index(1)
            row["ce"] = maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[ordered_target]
            row["margin"] = logits[ordered_target] - max(value for i, value in enumerate(logits) if i != ordered_target)
        summary["teacher_replay_statistics"] = controller.core.reader_checkpoint_statistics(
            rows, checkpoint="canonical_teacher", condition="teacher",
        )
    _write_jsonl(run / controller.ROWS_FILE, rows)
    _write_json(run / controller.SUMMARY_FILE, summary)
    _write_json(run / controller.PREFLIGHT_FILE, summary)
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    with pytest.raises(ValueError, match="teacher replay"):
        controller._validate_child(run, mode="technical-preflight", expected_commit=COMMIT)


@pytest.mark.parametrize("field", controller.PREFLIGHT_REQUIRED_AUDIT_TRUE)
def test_formal_prerequisite_reaudits_each_false_subgate_even_with_resigned_terminal(tmp_path, field):
    root = tmp_path / "preflight"
    run = root / "run"
    summary = _make_preflight(run)
    summary["audit"][field] = False
    _write_json(run / controller.SUMMARY_FILE, summary)
    _write_json(run / controller.PREFLIGHT_FILE, summary)
    hashes = _sign_preflight_controller(root)
    with pytest.raises(ValueError, match="mode contract"):
        controller._validate_preflight_terminal(root / "terminal.json", expected_commit=COMMIT, **hashes)


def test_formal_prerequisite_rechecks_raw_teacher_rows_even_with_resigned_terminal(tmp_path):
    root = tmp_path / "preflight"
    run = root / "run"
    _make_preflight(run)
    rows = _teacher_rows()
    rows[0]["choice_logits_ordered"] = [0.0] * 4
    _write_jsonl(run / controller.ROWS_FILE, rows)
    hashes = _sign_preflight_controller(root)
    with pytest.raises(ValueError, match="teacher replay"):
        controller._validate_preflight_terminal(root / "terminal.json", expected_commit=COMMIT, **hashes)


@pytest.mark.parametrize("admission", ["child", "formal"])
def test_self_consistent_relabelled_manifest_logits_and_summary_cannot_replace_locked_parent(tmp_path, admission):
    root = tmp_path / "preflight"
    run = root / "run"
    summary = _make_preflight(run)
    manifest = controller._load(run / "manifest.json")
    manifest["target_segment"]["query"]["target_index"] = 0
    rows = _teacher_rows()
    for row in rows:
        logits = [-2.0] * 4
        logits[row["permutation"].index(0)] = 8.0
        row["choice_logits_ordered"] = logits
        row["target_index"] = row["predicted_index"] = 0
        row["target_text"] = row["predicted_text"] = "none"
        row["ce"] = 8.0 + math.log(sum(math.exp(value - 8.0) for value in logits)) - 8.0
        row["correct"], row["margin"] = True, 10.0
    summary["teacher_replay_statistics"] = controller.core.reader_checkpoint_statistics(
        rows, checkpoint="canonical_teacher", condition="teacher",
    )
    assert controller.core.teacher_replay_gate(summary["teacher_replay_statistics"])
    _write_json(run / "manifest.json", manifest)
    _write_jsonl(run / controller.ROWS_FILE, rows)
    _write_json(run / controller.SUMMARY_FILE, summary)
    _write_json(run / controller.PREFLIGHT_FILE, summary)
    hashes = _sign_preflight_controller(root)
    with pytest.raises(ValueError, match="differs from the locked Phase1A parent"):
        if admission == "child":
            controller._validate_child(run, mode="technical-preflight", expected_commit=COMMIT)
        else:
            controller._validate_preflight_terminal(root / "terminal.json", expected_commit=COMMIT, **hashes)


@pytest.mark.parametrize("admission", ["child", "formal"])
def test_locked_parent_hash_drift_blocks_even_resigned_child_admission(tmp_path, _locked_phase1a_parent, admission):
    root = tmp_path / "preflight"
    run = root / "run"
    _make_preflight(run)
    parent = controller._load(_locked_phase1a_parent)
    parent["target_segment"]["query"]["target_index"] = 0
    _write_json(_locked_phase1a_parent, parent)
    hashes = _sign_preflight_controller(root)
    with pytest.raises(ValueError, match="locked Phase1A target manifest hash drifted"):
        if admission == "child":
            controller._validate_child(run, mode="technical-preflight", expected_commit=COMMIT)
        else:
            controller._validate_preflight_terminal(root / "terminal.json", expected_commit=COMMIT, **hashes)


@pytest.mark.parametrize("field", [
    "summary_sha256", "manifest_sha256", "child_terminal_sha256", "child_inventory_sha256", "stdout_sha256", "stderr_sha256",
])
def test_formal_prerequisite_binds_original_terminal_artifact_hashes(tmp_path, field):
    root = tmp_path / "preflight"
    _make_preflight(root / "run")
    hashes = _sign_preflight_controller(root)
    terminal = controller._load(root / "terminal.json")
    terminal[field] = "f" * 64
    _write_json(root / "terminal.json", terminal)
    _inventory(root, controller.INVENTORY_SCHEMA)
    with pytest.raises(ValueError, match="binding drifted"):
        controller._validate_preflight_terminal(root / "terminal.json", expected_commit=COMMIT, **hashes)


def test_required_preflight_audit_list_matches_preregistered_full_contract():
    assert set(controller.PREFLIGHT_REQUIRED_AUDIT_TRUE) == {
        "four_dreamlite_steps", "effective_sigmas_exact", "finite_nonzero_x_T_gradient",
        "only_x_T_fp32_trainable", "frozen_gradients_absent", "step0_parity_valid",
        "source_only_initialization_artifact_valid", "teacher_binding_valid", "teacher_replay_gate",
        "checkpoint_hash_valid", "snapshots_unchanged",
    }


def test_inventory_rejects_resigned_file_set_omission(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("a", encoding="utf-8")
    _inventory(root, controller.INVENTORY_SCHEMA)
    (root / "unlisted.txt").write_text("new", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        controller._validate_inventory(root, schema=controller.INVENTORY_SCHEMA)


def test_main_never_writes_an_existing_unowned_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "terminal.json"
    marker.write_bytes(b"immutable-existing-result\n")
    before = {
        path.relative_to(root).as_posix(): (path.stat().st_size, controller._sha256(path))
        for path in root.rglob("*")
        if path.is_file()
    }

    def reject(_args: Namespace) -> dict:
        raise ValueError("existing root rejected before ownership")

    monkeypatch.setattr(controller, "_validate", reject)
    argv = [
        "--mode",
        "technical-preflight",
        "--train",
        "train",
        "--dev",
        "dev",
        "--dreamlite",
        "dreamlite",
        "--reader",
        "reader",
        "--teacher",
        "teacher",
        "--phase1a-comparison",
        "comparison",
        "--phase1a-raw-artifacts",
        "raw",
        "--phase1a-target-root",
        "target",
        "--output-root",
        str(root),
        "--expected-commit",
        COMMIT,
    ]
    with pytest.raises(SystemExit, match="existing root rejected"):
        controller.main(argv)
    after = {
        path.relative_to(root).as_posix(): (path.stat().st_size, controller._sha256(path))
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
