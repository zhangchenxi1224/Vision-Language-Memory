from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
INSPIRE = ROOT / "scripts" / "inspire"
sys.path.insert(0, str(INSPIRE))

from launch_background import STRICT_ENVIRONMENT  # noqa: E402
from materialize_qwen_history_baseline import (  # noqa: E402
    authorize_stage,
    initialize,
)
from model_snapshot_manifest import create_snapshot_manifest  # noqa: E402
from qwen_history_baseline_contract import (  # noqa: E402
    STAGE_EVIDENCE_PROTOCOL,
    load_amendment,
    validate_replica_pair,
    validate_scientific_command,
    verify_prediction_jsonl,
)
from r3_dag_contract import atomic_json, load_json_object, sha256_file  # noqa: E402
from run_qwen_history_baseline_stage import run_bound_stage  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_fake_evaluator(repo: Path) -> None:
    path = repo / "scripts" / "eval" / "qwen_text_baselines.py"
    path.parent.mkdir(parents=True)
    path.write_text(
        """from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('episodes','reader','output','method','probe_role','choice_view_family','input_mode','replica_id','device'):
    p.add_argument('--'+name.replace('_','-'), required=True)
p.add_argument('--format', required=True); p.add_argument('--conditions', nargs='+', required=True)
p.add_argument('--strict-determinism', action='store_true'); p.add_argument('--limit')
a=p.parse_args(); out=Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
rows=[]
for condition in a.conditions:
    for view in range(4):
        rows.append({'replica':a.replica_id,'method':a.method,'input_mode':a.input_mode,'micro_sensitivity':False,'base_query_id':'fixture:q0','choice_view_family':'reverse-cyclic4','choice_view_index':view,'context_truncated':False,'condition':condition,'probe_role':'delayed'})
out.write_text(''.join(json.dumps(row)+'\\n' for row in rows), encoding='utf-8')
sha=hashlib.sha256(out.read_bytes()).hexdigest()
report={'status':'complete','method':a.method,'input_mode':a.input_mode,'micro_sensitivity':False,'output_sha256':sha,'episodes_sha256':hashlib.sha256(Path(a.episodes).read_bytes()).hexdigest(),'reader_revision':'ebb281ec70b05090aa6165b016eac8ec08e71b17','query_states':1,'prediction_records':len(rows),'conditions':a.conditions,'probe_role':a.probe_role,'choice_view_family':a.choice_view_family}
out.with_suffix(out.suffix+'.report.json').write_text(json.dumps(report)+'\\n', encoding='utf-8')
""",
        encoding="utf-8",
    )
    scorer = repo / "scripts" / "eval" / "score_qwen_history_baseline.py"
    scorer.write_text(
        """from __future__ import annotations
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('predictions','prediction_report','replica_b_predictions','replica_b_report','suite','output'):
    p.add_argument('--'+name.replace('_','-'), required=True)
p.add_argument('--bootstrap-iterations'); p.add_argument('--bootstrap-seed')
a=p.parse_args(); out=Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
payload={'schema':'vlm.qwen-history-baseline-score.v1','suite':a.suite,'method':'qwen_full_event_history','passed':True,'replication':{'passed':True,'bitwise_scientific_payload_match':True}}
out.write_text(json.dumps(payload)+'\\n', encoding='utf-8')
""",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "unit@example.com")
    _git(repo, "config", "user.name", "Unit Test")
    (repo / "scripts" / "inspire").mkdir(parents=True)
    for name in (
        "launch_background.py",
        "run_qwen_history_baseline_stage.py",
    ):
        (repo / "scripts" / "inspire" / name).write_text("# fixture path binding\n", encoding="utf-8")
    _write_fake_evaluator(repo)

    data_root = tmp_path / "data"
    data_root.mkdir()
    data_paths = {
        "set8_gate_sha256": data_root / "set8_gate.jsonl",
        "transition16_gate_sha256": data_root / "transition16_gate.jsonl",
        "formal_manifest_sha256": data_root / "formal_manifest.json",
        "formal_dev_sha256": data_root / "dev.jsonl",
        "formal_test_id_sha256": data_root / "test_id.jsonl",
        "formal_test_ood_sha256": data_root / "test_ood.jsonl",
    }
    query_counts = {
        "set8_gate_sha256": 8,
        "transition16_gate_sha256": 16,
        "formal_dev_sha256": 1252,
        "formal_test_id_sha256": 2488,
        "formal_test_ood_sha256": 2488,
    }
    for name, path in data_paths.items():
        if name == "formal_manifest_sha256":
            path.write_text('{"schema":"fixture"}\n', encoding="utf-8")
            continue
        count = query_counts[name]
        if name in {"set8_gate_sha256", "transition16_gate_sha256"}:
            episodes = [
                {"episode_id": f"{name}-{index}", "turns": [{"type": "query"}]}
                for index in range(count)
            ]
        else:
            episodes = [
                {
                    "episode_id": name,
                    "turns": [{"type": "query"} for _ in range(count)],
                }
            ]
        path.write_text(
            "".join(json.dumps(episode) + "\n" for episode in episodes),
            encoding="utf-8",
        )

    amendment = json.loads(
        (ROOT / "configs" / "experiments" / "r3_qwen_full_history_baseline_amendment.json").read_text(
            encoding="utf-8"
        )
    )
    amendment["data"] = {name: sha256_file(path) for name, path in data_paths.items()}
    amendment_path = repo / "configs" / "experiments" / "r3_qwen_full_history_baseline_amendment.json"
    amendment_path.parent.mkdir(parents=True)
    amendment_path.write_text(json.dumps(amendment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (repo / "models.lock.json").write_text(
        json.dumps(
            {
                "models": {
                    "qwen_reader": {
                        "repo_id": "Qwen/Qwen3-VL-4B-Instruct",
                        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
                        "local_dir": "models/Qwen3-VL-4B-Instruct",
                        "snapshot_manifest": ".snapshot_manifest.json",
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    model_root = tmp_path / "models"
    reader = model_root / "Qwen3-VL-4B-Instruct"
    reader.mkdir(parents=True)
    (reader / "config.json").write_text('{"model":"fixture"}\n', encoding="utf-8")
    (reader / ".locked_revision").write_text(
        "ebb281ec70b05090aa6165b016eac8ec08e71b17\n",
        encoding="utf-8",
    )
    (reader / ".snapshot_complete").write_text(
        "ebb281ec70b05090aa6165b016eac8ec08e71b17\n",
        encoding="utf-8",
    )
    snapshot = create_snapshot_manifest(
        model_dir=reader,
        repo_id="Qwen/Qwen3-VL-4B-Instruct",
        revision="ebb281ec70b05090aa6165b016eac8ec08e71b17",
    )
    snapshot.pop("created")

    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    preflight = tmp_path / "formal_preflight.json"
    atomic_json(
        preflight,
        {
            "passed": True,
            "formal_ready": True,
            "git": {"commit": commit},
            "models": {"qwen_reader": {"snapshot_manifest": snapshot}},
        },
    )
    return {
        "repo": repo,
        "commit": commit,
        "preflight": preflight,
        "model_root": model_root,
        "amendment": amendment_path,
        "data_paths": data_paths,
        "runs_root": tmp_path / "runs",
    }


def _initialize_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    fixture = _fixture(tmp_path)
    result = initialize(
        repo=fixture["repo"],
        python=Path(sys.executable),
        model_root=fixture["model_root"],
        runs_root=fixture["runs_root"],
        run_name="qwen-history-unit",
        preflight=fixture["preflight"],
        expected_commit=fixture["commit"],
        amendment=fixture["amendment"],
        data_paths=fixture["data_paths"],
    )
    return fixture, result


def test_prospective_amendment_locks_blank_image_and_ab_replication() -> None:
    amendment, digest = load_amendment(
        ROOT / "configs" / "experiments" / "r3_qwen_full_history_baseline_amendment.json"
    )
    assert len(digest) == 64
    assert amendment["blank_image"]["shape_chw"] == [3, 1024, 1024]
    assert amendment["research_role"]["dreamlite_loaded"] is False
    assert amendment["execution"]["strict_order"] == ["BH0", "BH1", "BH2", "BH3"]
    assert amendment["expected_inventory"]["formal_dev"]["prediction_records"] == 20_032
    assert amendment["expected_inventory"]["formal_test_id"]["prediction_records"] == 39_808


def test_command_contract_rejects_training_privilege_and_wrong_device() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_scientific_command([sys.executable, "scripts/train/dreamlite_episode.py"])
    with pytest.raises(ValueError, match="cuda:0"):
        validate_replica_pair(
            [
                {
                    "replica": "A",
                    "device": "cuda:1",
                    "argv": [sys.executable, "eval.py", "--device", "cuda:1"],
                },
                {
                    "replica": "B",
                    "device": "cuda:1",
                    "argv": [sys.executable, "eval.py", "--device", "cuda:1"],
                },
            ]
        )


def test_prediction_inventory_rejects_incomplete_reverse_views(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = [
        {
            "method": "qwen_full_event_history",
            "input_mode": "blank_image",
            "micro_sensitivity": False,
            "base_query_id": "episode:q0",
            "choice_view_family": "reverse-cyclic4",
            "choice_view_index": view,
            "context_truncated": False,
            "condition": "standard",
            "probe_role": "delayed",
        }
        for view in (0, 1, 2, 2)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = {
        "path": str(path),
        "expected_records": 4,
        "expected_method": "qwen_full_event_history",
        "expected_input_mode": "blank_image",
        "expected_micro_sensitivity": False,
        "expected_conditions": ["standard"],
        "expected_probe_role": "delayed",
    }

    with pytest.raises(ValueError, match="duplicates view"):
        verify_prediction_jsonl(output)


def test_materializer_binds_qwen_only_and_fail_stops_before_bh1(tmp_path: Path) -> None:
    fixture, result = _initialize_fixture(tmp_path)
    plan = load_json_object(Path(result["plan"]))
    assert plan["strict_order"] == ["BH0", "BH1", "BH2", "BH3"]
    assert plan["dreamlite_snapshot_bound"] is False
    assert plan["dreamlite_loaded"] is False
    assert plan["training_performed"] is False
    assert set(plan["qwen_reader_snapshot"]) >= {"revision", "manifest_sha256"}
    assert len(plan["stages"]["BH1"]["parallel_groups"]) == 2
    bh1 = json.dumps(plan["stages"]["BH1"])
    sensitivity_commands = [
        command
        for command in plan["stages"]["BH1"]["serial_commands"]
        if "--method" in command
        and command[command.index("--method") + 1] == "qwen_full_event_history_text_only"
    ]
    assert len(sensitivity_commands) == 2
    assert "micro_sensitivity_not_formal_baseline" in bh1
    assert "qwen_full_event_history_text_only" not in json.dumps(plan["stages"]["BH2"])
    assert "qwen_full_event_history_text_only" not in json.dumps(plan["stages"]["BH3"])
    for stage in plan["strict_order"]:
        flattened = json.dumps(plan["stages"][stage]).lower()
        assert "scripts/train" not in flattened
        assert "dreamlite" not in flattened
    launch = result["first_stage"]["launch"]
    assert launch["executed"] is False
    assert any(value.endswith("run_qwen_history_baseline_stage.py") for value in launch["argv"])
    with pytest.raises(ValueError, match="no terminal.json"):
        authorize_stage(fixture["runs_root"] / "qwen-history-unit", stage="BH1")


def test_worker_executes_ab_concurrently_and_writes_qwen_only_evidence(tmp_path: Path) -> None:
    fixture, result = _initialize_fixture(tmp_path)
    run_root = fixture["runs_root"] / "qwen-history-unit"
    plan = load_json_object(Path(result["plan"]))
    spec = result["first_stage"]["stage_spec"]
    spec_path = Path(result["first_stage"]["launch"]["stage_spec"])
    spec_sha256 = result["first_stage"]["stage_spec_sha256"]
    run_dir = Path(spec["run_dir"])
    run_dir.mkdir(parents=True)
    worker_input = run_dir / "worker_input.json"
    configuration_sha256 = atomic_json(
        worker_input,
        {
            "stage": spec["launcher_stage"],
            "repo": spec["repo"],
            "run_root": spec["run_root"],
            "run_dir": spec["run_dir"],
            "expected_commit": spec["expected_commit"],
            "preflight": spec["formal_preflight"]["path"],
            "preflight_sha256": spec["formal_preflight"]["sha256"],
            "infrastructure_stage": False,
            "strict_environment": STRICT_ENVIRONMENT,
            "command": [
                plan["python"],
                str(Path(spec["repo"]) / "scripts" / "inspire" / "run_qwen_history_baseline_stage.py"),
                "--spec",
                str(spec_path),
                "--spec-sha256",
                spec_sha256,
            ],
        },
    )
    environment = {
        "VLM_STAGE_WORKER_INPUT": str(worker_input),
        "VLM_STAGE_CONFIGURATION_SHA256": configuration_sha256,
        "VLM_STAGE_PREFLIGHT": spec["formal_preflight"]["path"],
        "VLM_STAGE_PREFLIGHT_SHA256": spec["formal_preflight"]["sha256"],
    }
    with patch.dict(os.environ, environment, clear=False):
        evidence = run_bound_stage(spec_path, spec_sha256)
    assert evidence["passed"] is True
    assert evidence["protocol"] == STAGE_EVIDENCE_PROTOCOL
    assert evidence["dreamlite_loaded"] is False
    assert evidence["training_performed"] is False
    assert {entry["replica"] for entry in evidence["parallel_groups"][0]["replicas"]} == {"A", "B"}
    assert all(entry["exit_code"] == 0 for entry in evidence["parallel_groups"][0]["replicas"])
    assert (run_root / "results" / "BH0" / "set8-smoke" / "score.json").is_file()
