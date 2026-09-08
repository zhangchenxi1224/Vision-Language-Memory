"""CPU contract tests; these are not GPU experiment outcomes."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import run_frozen_oracle_geometry_worker as worker  # noqa: E402
from vision_memory.training.frozen_oracle_geometry import build_manifest  # noqa: E402


def _spec(**changes):
    return {"stage": "A1", "run_id": "A1-t01-gaussian-s00-a1-r0", "target_index": 1,
            "seed": 0, "distribution": "gaussian", "scale": 1.0, "repeat": 0, **changes}


def _rows():
    return [{"condition": condition, "view_index": view, "correct": True, "margin": 0.2}
            for condition in ("normal", "reset") for view in range(4)]


def test_every_preregistered_manifest_run_is_accepted():
    manifest = build_manifest()
    assert len(manifest["runs"]) == 158
    for spec in manifest["runs"]:
        assert worker.select_run_spec(manifest, spec["run_id"]) == spec


def test_repeat_and_target_never_change_initialization():
    a = worker.initial_tensor((1, 4, 16, 16), _spec())
    b = worker.initial_tensor((1, 4, 16, 16), _spec(repeat=2, target_index=0))
    c = worker.initial_tensor((1, 4, 16, 16), _spec(seed=1))
    assert a.dtype == torch.float32 and torch.equal(a, b)
    assert not torch.equal(a, c)


def test_duplicate_run_and_bad_spec_fail_closed():
    with pytest.raises(ValueError, match="exactly one"):
        worker.select_run_spec({"runs": [_spec(), _spec()]}, _spec()["run_id"])
    for changes in ({"seed": True}, {"target_index": 8}, {"repeat": -1}, {"scale": float("nan")},
                    {"run_id": "../escape"}):
        with pytest.raises(ValueError):
            worker.validate_run_spec(_spec(**changes))


def test_strict_gate_rejects_ties_missing_duplicate_and_legacy_partial_success():
    rows = _rows()
    assert worker.strict_endpoint_gate(rows)["qa_pass"] is True
    rows[0]["margin"] = 0.0
    assert worker.strict_endpoint_gate(rows)["qa_pass"] is False
    rows[0]["margin"] = 0.2
    rows[2]["correct"] = False
    assert worker.strict_endpoint_gate(rows)["normal_correct"] == 3
    with pytest.raises(ValueError):
        worker.strict_endpoint_gate(rows[:-1])
    rows[-1]["view_index"] = 0
    with pytest.raises(ValueError):
        worker.strict_endpoint_gate(rows)


def test_path_length_accumulates_all_iterates_and_detects_curved_path():
    path = worker.PathAccumulator()
    assert path.add(torch.tensor([0.0, 0.0]))["tortuosity"] is None
    path.add(torch.tensor([3.0, 0.0]))
    final = path.add(torch.tensor([3.0, 4.0]))
    assert final == {"step_distance_l2": 4.0, "path_length_l2": 7.0,
                     "displacement_l2": 5.0, "tortuosity": 1.4}


def test_numpy_artifact_roundtrip_and_canonical_tensor_hash(tmp_path):
    tensor = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    path = tmp_path / "latent.npy"
    row = worker.save_array(path, tensor)
    loaded = torch.from_numpy(np.load(path, allow_pickle=False))
    assert torch.equal(tensor, loaded)
    assert row["file_sha256"] == worker.file_sha256(path)
    assert row["sha256"] == worker.canonical_tensor_sha256(loaded)
    assert Path(row["path"]).is_absolute()


class TinyOracle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x_T_fp32 = torch.nn.Parameter(torch.ones((1, 3, 2, 2)))
        self.unet = torch.nn.Linear(1, 1)
        self.vae = torch.nn.Linear(1, 1)
        self.text_encoder = torch.nn.Linear(1, 1)
        for module in (self.unet, self.vae, self.text_encoder):
            module.requires_grad_(False)
        self.eval()


def test_gradient_contract_detects_frozen_leak_and_zero_gradient():
    oracle = TinyOracle()
    reader = torch.nn.Linear(1, 1).requires_grad_(False).eval()
    oracle.x_T_fp32.sum().backward()
    assert worker.assert_frozen_gradient_contract(oracle, reader)["all_frozen_without_parameter_gradients"]
    reader.weight.grad = torch.ones_like(reader.weight)
    with pytest.raises(RuntimeError, match="Frozen reader"):
        worker.assert_frozen_gradient_contract(oracle, reader)
    reader.weight.grad = None
    oracle.x_T_fp32.grad.zero_()
    with pytest.raises(RuntimeError, match="nonzero"):
        worker.assert_frozen_gradient_contract(oracle, reader)


def test_probe_uses_real_frozen_oracle_full_four_step_graph(tmp_path):
    # Reuse the established mocked model components, while executing the real production
    # FrozenDreamLiteOracle and differentiable sampler implementation.
    import importlib.util
    path = ROOT / "tests/test_r11_new_phase1a_trainer.py"
    spec = importlib.util.spec_from_file_location("worker_oracle_test_fixtures", path)
    fixtures = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixtures
    spec.loader.exec_module(fixtures)
    oracle, unet, _, _ = fixtures.make_oracle()
    oracle.eval()
    reader = torch.nn.Linear(1, 1).requires_grad_(False).eval()
    target = SimpleNamespace(segment_id=fixtures.trainer.R11_NEW_TARGET_IDS[0],
                             query=SimpleNamespace(choices=("a", "b", "c", "d"), target_index=0, text="test"))

    def reader_fn(image, query, choices, target_index):
        del query, choices
        mean = image.mean()
        logits = torch.stack((mean, -mean, mean * 0, mean * 0.25))
        return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits[None], torch.tensor([target_index])),
                               choice_logits=logits)

    runtime = {"oracle": oracle, "reader": reader, "legacy": fixtures.trainer,
               "target": target, "train_reader": reader_fn}
    result = worker._probe(SimpleNamespace(output_dir=tmp_path, spec=_spec(target_index=0, mode="optimize")), runtime)
    assert result["passed"] is True
    assert result["mode"] == "probe"
    assert result["backward_calls"] == 2 and result["optimizer_steps"] == 0
    assert [row["denoiser_steps"] for row in result["rows"]] == [4, 4]
    assert unet.calls >= 8  # Checkpointed backward recomputes the executed denoiser calls.
    assert result["rows"][0]["gradient_audit"]["gradient"]["sha256"] == result["rows"][1]["gradient_audit"]["gradient"]["sha256"]


def test_evaluation_rejects_unbound_latents_before_model_execution(tmp_path):
    array_path = tmp_path / "latent.npy"
    np.save(array_path, np.zeros((1, 3, 2, 2), dtype=np.float32))
    args = SimpleNamespace(target_index=0, output_dir=tmp_path, spec=_spec(target_index=0), evaluation_spec=json.dumps({
        "cases": [{"case_id": "perturb-0", "space": "xT", "latent_path": str(array_path), "latent_sha256": "0" * 64}]
    }))
    with pytest.raises(ValueError, match="SHA binding"):
        worker._evaluate(args, {"oracle": TinyOracle(), "target": SimpleNamespace(segment_id="target-0")})


def test_a5_external_initialization_requires_hash_and_stage(tmp_path):
    path = tmp_path / "perturbed.npy"
    initial = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    np.save(path, initial)
    spec = _spec(stage="A5_reoptimization", initial_xT_path=str(path), initial_xT_file_sha256=worker.file_sha256(path))
    assert torch.equal(worker.initial_tensor(initial.shape, spec), torch.from_numpy(initial))
    with pytest.raises(ValueError, match="Stage A5"):
        worker.initial_tensor(initial.shape, {**spec, "stage": "A1"})
    with pytest.raises(ValueError, match="SHA"):
        worker.initial_tensor(initial.shape, {**spec, "initial_xT_file_sha256": "0" * 64})
    with pytest.raises(ValueError, match="wrong-shaped"):
        worker.initial_tensor((1, 4, 2, 2), spec)


def test_optimizer_saves_matching_states_and_repeat_hashes(tmp_path, monkeypatch):
    # Two CPU Adam updates are a bounded artifact/iteration contract test, not a shortened GPU run.
    import importlib.util
    fixture_path = ROOT / "tests/test_r11_new_phase1a_trainer.py"
    module_spec = importlib.util.spec_from_file_location("worker_optimizer_test_fixtures", fixture_path)
    fixtures = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = fixtures
    module_spec.loader.exec_module(fixtures)
    monkeypatch.setattr(worker, "STEPS", 2)
    monkeypatch.setattr(worker, "CHECKPOINT_STEPS", (0, 1, 2))

    def reader_fn(image, query, choices, target_index):
        del query, choices
        mean = image.mean()
        logits = torch.stack((mean, -mean, mean * 0, mean * 0.25))
        return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits[None], torch.tensor([target_index])),
                               choice_logits=logits)

    results = []
    for repeat in range(2):
        output_dir = tmp_path / f"repeat-{repeat}"
        output_dir.mkdir()
        oracle, _, _, _ = fixtures.make_oracle()
        oracle.eval()
        reader = torch.nn.Linear(1, 1).requires_grad_(False).eval()
        target = SimpleNamespace(segment_id=fixtures.trainer.R11_NEW_TARGET_IDS[0],
                                 events=[SimpleNamespace(source_episode_id="episode-a", noise_turn_id="turn-1")],
                                 query=SimpleNamespace(choices=("a", "b", "c", "d"), target_index=0, text="test"))
        runtime = {"oracle": oracle, "reader": reader, "legacy": fixtures.trainer, "target": target,
                   "train_reader": reader_fn, "eval_reader": reader_fn, "reset_image": torch.zeros((1, 3, 4, 4))}
        result = worker._optimize(SimpleNamespace(output_dir=output_dir, spec=_spec(target_index=0, repeat=repeat)), runtime)
        results.append(result)
        index = json.loads((output_dir / "trajectory_index.json").read_text())
        assert [record["step"] for record in index] == [0, 1, 2]
        assert len(list((output_dir / "xT").glob("*.npy"))) == 3
        assert len(list((output_dir / "z").glob("*.npy"))) == 3
        assert len(list((output_dir / "checkpoints").glob("*.pt"))) == 3
        checkpoint = torch.load(output_dir / "checkpoints/step-002.pt", weights_only=True)
        assert worker.canonical_tensor_sha256(checkpoint["x_T_fp32"]) == result["optimized_xT_sha256"]
        assert worker.canonical_tensor_sha256(checkpoint["z"]) == result["endpoint_z_sha256"]
        assert len(checkpoint["optimizer"]["state"]) == 1
        assert result["initial_xT_sha256"] != result["optimized_xT_sha256"]
        assert len(result["initial_evaluation_rows"]) == 8 and len(result["evaluation_rows"]) == 8
        assert result["legacy_reachability_gate"] in (True, False)
        assert result["perturbation"] is None
        raw = [np.load(record["z"]["path"]).astype(np.float64) for record in index]
        path = np.linalg.norm(raw[1] - raw[0]) + np.linalg.norm(raw[2] - raw[1])
        assert result["geometry"]["z"]["path_length_l2"] == pytest.approx(path)
    for field in ("initial_xT_sha256", "optimized_xT_sha256", "endpoint_z_sha256", "loss_trajectory_sha256",
                  "gradient_trajectory_sha256"):
        assert results[0][field] == results[1][field]
