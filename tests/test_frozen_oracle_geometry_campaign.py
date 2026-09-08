"""Controller integration tests using explicit fake CPU subprocesses, never GPU evidence."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.experiments import analyze_frozen_oracle_geometry as analysis  # noqa: E402
from scripts.experiments import run_frozen_oracle_geometry_campaign as campaign  # noqa: E402
from vision_memory.training.frozen_oracle_geometry import build_manifest, canonical_hash  # noqa: E402


FAKE_WORKER = r'''
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--run-spec');p.add_argument('--mode');p.add_argument('--output-dir')
a,_=p.parse_known_args(); spec=json.loads(a.run_spec); out=Path(a.output_dir);out.mkdir(parents=True)
with open(os.environ['GEOMETRY_TEST_EVENTS'],'a') as f:
 f.write(json.dumps({**spec,'mode':a.mode})+'\n')
fault=os.environ.get('GEOMETRY_TEST_FAULT','')
if fault=='probe_fail' and a.mode=='probe': raise SystemExit(7)
if fault=='malformed' and spec['stage']=='A1' and spec['target_index']==0:
 (out/'summary.json').write_text('{broken');raise SystemExit(0)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps({'run_spec':spec,'mode':a.mode,'fake_test_only':True}))
array=out/'fake.npy';np.save(array,np.zeros((1,4,2,2),dtype=np.float32))
record={'path':str(array),'file_sha256':sha(array),'sha256':'a'*64}
index=out/'trajectory_index.json'
index.write_text(json.dumps([{'step':i,'xT':record,'z':record} for i in range(257)]))
checkpoints=out/'checkpoint_index.json'
checkpoints.write_text(json.dumps([{'step':i,'path':str(array),'file_sha256':sha(array),
 'png_path':str(array),'png_sha256':sha(array)} for i in (0,1,2,4,8,16,32,64,128,192,256)]))
summary=dict(spec,status='completed',mode=a.mode,technical_pass=True,passed=True,bitwise_repeatability=True,
 model_snapshot_end_verified=True,manifest_sha256=sha(out/'manifest.json'),qa_pass=False,
 optimizer_steps=0 if a.mode=='probe' else 256,trajectory_index_path=str(index),checkpoint_index_path=str(checkpoints),
 initial_xT_sha256='a'*64,optimized_xT_sha256='b'*64,endpoint_z_sha256='c'*64,
 loss_trajectory_sha256='d'*64,gradient_trajectory_sha256='e'*64)
if fault=='a1_drift' and spec.get('target_index')==0 and spec.get('repeat')==2:
 summary['gradient_trajectory_sha256']='f'*64
if fault=='fresh_hash_mismatch' and spec['stage']=='A2': summary['manifest_sha256']='0'*64
(out/'summary.json').write_text(json.dumps(summary))
'''


@pytest.fixture
def harness(tmp_path, monkeypatch):
    full = build_manifest()
    manifest = {**full, "runs": [row for row in full["runs"] if row["stage"] == "A1"] +
                                [next(row for row in full["runs"] if row["stage"] == "A2")]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"manifest_sha256": canonical_hash(manifest)}))
    args = SimpleNamespace(config=config_path, manifest=manifest_path, output_root=tmp_path / "outputs",
                           expected_commit="test-commit", max_hours=0.1, aux_command_json=None,
                           train=tmp_path / "train", dev=tmp_path / "dev",
                           dreamlite=tmp_path / "dreamlite", reader=tmp_path / "reader")
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("GEOMETRY_TEST_EVENTS", str(events))
    fake_path = tmp_path / "fake_worker.py"
    fake_path.write_text(FAKE_WORKER)
    original_popen = subprocess.Popen

    def fake_popen(command, *positional, **kwargs):
        assert Path(command[1]).name == "run_frozen_oracle_geometry_worker.py"
        return original_popen([command[0], str(fake_path), *command[2:]], *positional, **kwargs)

    def fake_git(command, **kwargs):
        del kwargs
        if command == ["git", "rev-parse", "HEAD"]:
            return "test-commit\n"
        assert command == ["git", "status", "--porcelain"]
        return ""

    def fake_analysis(command, **kwargs):
        del kwargs
        assert Path(command[1]).name == "analyze_frozen_oracle_geometry.py"
        directory = args.output_root / "evaluation_inputs"
        directory.mkdir(exist_ok=True)
        (directory / "plan.json").write_text(json.dumps({"status": "not_applicable"}))
        reports = args.output_root / "analysis"
        reports.mkdir(exist_ok=True)
        (reports / "geometry_decision.json").write_text(json.dumps({"status": getattr(args, "analysis_status", "complete")}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(campaign.subprocess, "check_output", fake_git)
    monkeypatch.setattr(campaign.subprocess, "run", fake_analysis)
    # The real controller is Linux-only. A no-op lock suffices for this one-process
    # orchestration test on Windows; production fcntl behavior is not claimed tested.
    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=lambda *args: None))
    return args, events


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_probe_failure_stops_all_optimization(harness, monkeypatch):
    args, events = harness
    monkeypatch.setenv("GEOMETRY_TEST_FAULT", "probe_fail")
    with pytest.raises(RuntimeError, match="worker failed"):
        campaign.run_campaign(args)
    rows = _events(events)
    assert len(rows) == 1 and rows[0]["mode"] == "probe"
    assert json.loads((args.output_root / "status.json").read_text())["state"] == "failed"


def test_a1_hash_drift_stops_later_studies(harness, monkeypatch):
    args, events = harness
    monkeypatch.setenv("GEOMETRY_TEST_FAULT", "a1_drift")
    with pytest.raises(RuntimeError, match="A1 determinism gate failed"):
        campaign.run_campaign(args)
    assert not any(row["stage"] == "A2" for row in _events(events))
    assert json.loads((args.output_root / "A1_determinism_gate.json").read_text())["passed"] is False


def test_matching_a1_allows_later_studies(harness):
    args, events = harness
    campaign.run_campaign(args)
    rows = _events(events)
    assert rows[0]["mode"] == "probe"
    assert sum(row["mode"] == "optimize" and row["stage"] == "A1" for row in rows) == 6
    assert rows[-1]["stage"] == "A2"
    assert json.loads((args.output_root / "A1_determinism_gate.json").read_text())["passed"] is True
    assert json.loads((args.output_root / "status.json").read_text())["state"] == "completed"


def test_malformed_summary_stops_later_studies(harness, monkeypatch):
    args, events = harness
    monkeypatch.setenv("GEOMETRY_TEST_FAULT", "malformed")
    with pytest.raises((ValueError, RuntimeError)):
        campaign.run_campaign(args)
    assert not any(row["stage"] == "A2" for row in _events(events))


def test_reuse_rejects_changed_completed_summary(harness):
    args, _ = harness
    campaign.run_campaign(args)
    path = args.output_root / "probes/probe-frozen-full-path/summary.json"
    value = json.loads(path.read_text())
    value["tampered_after_completion"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="completed summary changed"):
        campaign.run_campaign(args)


def test_empty_analysis_stays_incomplete_and_preserves_denominators(tmp_path):
    manifest = build_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    analysis.analyze(SimpleNamespace(root=tmp_path, manifest=path, prepare_evaluations=True))
    decision = json.loads((tmp_path / "analysis/geometry_decision.json").read_text())
    assert decision["status"] == "incomplete" and decision["bank_action"] == "blocked"
    assert decision["completed_optimization_runs"] == 0 and decision["planned_optimization_runs"] == 158
    rates = json.loads((tmp_path / "analysis/success_rates.json").read_text())
    assert rates and all(row["completed"] == 0 and row["success_rate_completed"] is None for row in rates)
    assert all(row["unresolved"] == row["planned"] for row in rates)


def test_newly_completed_summary_requires_matching_manifest_hash(harness, monkeypatch):
    args, _ = harness
    monkeypatch.setenv("GEOMETRY_TEST_FAULT", "fresh_hash_mismatch")
    with pytest.raises((RuntimeError, ValueError), match="manifest|hash|binding"):
        campaign.run_campaign(args)


def test_incomplete_analysis_must_not_be_reported_completed(harness):
    args, _ = harness
    args.analysis_status = "incomplete"
    try:
        campaign.run_campaign(args)
    except RuntimeError:
        pass
    assert json.loads((args.output_root / "status.json").read_text())["state"] != "completed"
