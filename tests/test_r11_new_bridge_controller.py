from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.inspire import run_r11_new_canonical_latent_bridge as controller  # noqa: E402


COMMIT = "a" * 40


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
    manifest = {
        "schema": controller.trainer.MANIFEST_SCHEMA,
        "mode": mode,
        "git_commit": COMMIT,
        "git_dirty": False,
        "target_index": controller.core.BRIDGE_TARGET_INDEX,
        "target_segment_id": controller.core.BRIDGE_TARGET_SEGMENT_ID,
    }
    terminal = {
        "schema": controller.trainer.TERMINAL_SCHEMA,
        "status": "technical_completed",
        "mode": mode,
        "target_index": controller.core.BRIDGE_TARGET_INDEX,
        "technical_gate": True,
        "formal_success": False,
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
        },
        "bridge_result_evaluated": False,
        "formal_success_gate": False,
        "phase2_allowed": False,
    }
    _write_json(run / controller.SUMMARY_FILE, summary)
    _write_json(run / controller.PREFLIGHT_FILE, summary)
    _write_jsonl(run / controller.ROWS_FILE, [{"row": index} for index in range(4)])
    (run / "checkpoints").mkdir()
    (run / "checkpoints" / "step-000.pt").write_bytes(b"step0")
    _inventory(run, controller.trainer.INVENTORY_SCHEMA)
    return summary


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
    summary = {
        "schema": controller.trainer.SUMMARY_SCHEMA,
        "status": "completed",
        "mode": "formal",
        "target_index": controller.core.BRIDGE_TARGET_INDEX,
        "target_segment_id": controller.core.BRIDGE_TARGET_SEGMENT_ID,
        "technical_gate": technical,
        "gates": {
            "technical_gate": True,
            "bridge_distance_gate": bridge_gate,
            "endpoint_reader_transfer_gate": bridge_gate,
            "bridge_diagnostic_gate": bridge_gate,
        },
        "decision": controller.core.bridge_decision(
            distance_pass=bridge_gate,
            reader_transfer_pass=bridge_gate,
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
    hashes = {
        "config_sha256": "1" * 64,
        "trainer_sha256": "2" * 64,
        "controller_sha256": "3" * 64,
    }
    terminal = {
        "schema": controller.TERMINAL_SCHEMA,
        "status": "technical_completed",
        "mode": "technical-preflight",
        "target_index": controller.core.BRIDGE_TARGET_INDEX,
        "technical_gate": True,
        "bridge_diagnostic_gate": None,
        "formal_success": False,
        "child_exit_code": 0,
        "git_commit": COMMIT,
        "execution_checks": {"all": True},
        **hashes,
    }
    _write_json(root / "terminal.json", terminal)
    _inventory(root, controller.INVENTORY_SCHEMA)
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
