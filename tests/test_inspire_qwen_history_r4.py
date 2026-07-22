from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "inspire"))

import materialize_qwen_history_r4 as materialize  # noqa: E402
import run_qwen_history_r4_stage as worker  # noqa: E402
from qwen_history_r4_contract import ARM_METHODS, ARM_ORDER  # noqa: E402


def _inventory(*, query_states: int, conditions: list[str], probe_role: str) -> dict:
    return {
        "data_key": "fixture",
        "query_states": query_states,
        "conditions": conditions,
        "probe_role": probe_role,
        "limit": None,
        "records_per_condition": query_states * 4,
        "prediction_records_per_arm": query_states * 4 * len(conditions),
    }


def _fragment(tmp_path: Path, *, stage: str = "BH0"):
    if stage == "BH0":
        inventory = _inventory(query_states=4, conditions=["standard"], probe_role="delayed")
        suite = "smoke"
        comparison = None
    elif stage == "BH1":
        inventory = _inventory(
            query_states=32,
            conditions=["standard", "reset", "shuffle", "state_swap"],
            probe_role="delayed",
        )
        suite = "transition32"
        comparison = "bh1"
    else:
        inventory = _inventory(query_states=5, conditions=["standard"], probe_role="all")
        suite = "formal"
        comparison = "formal"
    return materialize._dataset_fragment(
        python=Path(sys.executable),
        repo=ROOT,
        reader=tmp_path / "reader",
        episodes=tmp_path / "episodes.jsonl",
        results=tmp_path / "results",
        dataset="fixture",
        inventory=inventory,
        score_suite=suite,
        comparison_suite=comparison,
        seed=20260722,
        bootstrap_iterations=10_000,
        bootstrap_seed=2026,
        stage=stage,
    )


def test_fragment_locks_three_arms_and_sequential_ab(tmp_path: Path) -> None:
    pairs, serial, outputs = _fragment(tmp_path, stage="BH1")
    assert [pair["arm"] for pair in pairs] == list(ARM_ORDER)
    assert [pair["method"] for pair in pairs] == [ARM_METHODS[arm] for arm in ARM_ORDER]
    for pair in pairs:
        assert pair["execution_mode"] == "sequential_a_then_b"
        assert [entry["replica"] for entry in pair["replicas"]] == ["A", "B"]
        assert [entry["device"] for entry in pair["replicas"]] == ["cuda:0", "cuda:1"]
        for entry in pair["replicas"]:
            argv = entry["argv"]
            assert Path(argv[1]).name == "qwen_history_r4.py"
            assert argv[argv.index("--method") + 1] == pair["method"]
            assert argv[argv.index("--replica-id") + 1] == entry["replica"]
            assert argv[argv.index("--device") + 1] == entry["device"]
            assert "--strict-determinism" in argv
    assert len(serial) == 4  # three scores plus one strictly paired comparison
    assert "--no-fail-on-gate" in serial[0]
    assert "--no-fail-on-gate" in serial[1]
    assert "--fail-on-gate" in serial[2]
    assert Path(serial[3][1]).name == "compare_qwen_history_r4.py"
    score_outputs = [output for output in outputs if output.get("validator") == "r4_score"]
    assert [output["require_data_readability"] for output in score_outputs] == [False, False, True]


def test_bh2_has_only_last_effective_accuracy_gate_and_bh3_has_none(tmp_path: Path) -> None:
    _, bh2_serial, bh2_outputs = _fragment(tmp_path / "bh2", stage="BH2")
    bh2_scores = [output for output in bh2_outputs if ":score" in output["label"]]
    assert [output["validator"] for output in bh2_scores] == [
        "r4_score",
        "r4_score",
        "r4_bh2_last_effective_dev",
    ]
    assert all("--no-fail-on-gate" in command for command in bh2_serial[:3])

    _, bh3_serial, bh3_outputs = _fragment(tmp_path / "bh3", stage="BH3")
    bh3_scores = [output for output in bh3_outputs if ":score" in output["label"]]
    assert all(output["validator"] == "r4_score" for output in bh3_scores)
    assert all(output["require_data_readability"] is False for output in bh3_scores)
    assert all("--no-fail-on-gate" in command for command in bh3_serial[:3])


def test_worker_executes_a_then_b_not_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, _, _ = _fragment(tmp_path, stage="BH0")
    calls: list[str] = []

    def fake_run(argv, **_kwargs):
        calls.append(argv[argv.index("--replica-id") + 1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker, "validate_command", lambda _argv: None)
    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    result = worker._run_sequential_pair(pairs[0], cwd=ROOT, environment={})
    assert calls == ["A", "B"]
    assert result["execution_mode"] == "sequential_a_then_b"
    assert result["replicas"] == [
        {"replica": "A", "device": "cuda:0", "exit_code": 0},
        {"replica": "B", "device": "cuda:1", "exit_code": 0},
    ]


def test_worker_stops_before_replica_b_when_a_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, _, _ = _fragment(tmp_path, stage="BH0")
    calls: list[str] = []

    def fake_run(argv, **_kwargs):
        replica = argv[argv.index("--replica-id") + 1]
        calls.append(replica)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(worker, "validate_command", lambda _argv: None)
    monkeypatch.setattr(worker.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="replica A failed"):
        worker._run_sequential_pair(pairs[0], cwd=ROOT, environment={})
    assert calls == ["A"]


def test_commands_never_include_training_teacher_or_ledger(tmp_path: Path) -> None:
    pairs, serial, _ = _fragment(tmp_path, stage="BH1")
    commands = [entry["argv"] for pair in pairs for entry in pair["replicas"]] + serial
    forbidden = ("dreamlite", "scripts/train", "teacher", "ledger", "optimizer")
    for command in commands:
        flattened = " ".join(command).lower()
        assert not any(value in flattened for value in forbidden)


def test_materialization_access_audit_allows_only_test_sha_binding() -> None:
    assert materialize.TEST_DATA_ACCESS_AUDIT == {
        "test_file_sha256_bytes_read_during_materialization": True,
        "test_json_semantics_parsed_during_materialization": False,
        "test_predictions_or_metrics_accessed_during_materialization": False,
        "test_evaluation_executed_during_materialization": False,
    }
    source = Path(materialize.__file__).read_text(encoding="utf-8")
    assert "test_contents_read_during_materialization" not in source
