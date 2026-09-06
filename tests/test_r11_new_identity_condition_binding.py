from __future__ import annotations

import ast
import copy
import inspect
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train import r11_new_identity_condition_binding as binding  # noqa: E402
from scripts.train import r11_new_identity_condition_bridge as trainer  # noqa: E402
from scripts.train import r11_new_source_init_bridge as source_trainer  # noqa: E402
from scripts.inspire import run_r11_new_identity_condition_bridge as controller  # noqa: E402
from scripts.experiments import compare_r11_new_identity_condition_bridge as comparison  # noqa: E402


@pytest.fixture(scope="module")
def raw_probe(tmp_path_factory):
    root = tmp_path_factory.mktemp("identity-probe-fixed-raw")
    archive = ROOT / "reports/r11-new-identity-condition-probe-results-20260907/probe-raw.tar.gz"
    with tarfile.open(archive) as handle:
        handle.extractall(root, filter="data")
    return root / "probe"


@pytest.fixture
def config():
    return trainer._load_config()


@pytest.fixture
def tensors(raw_probe):
    first = torch.load(raw_probe / "condition.pt", map_location="cpu", weights_only=True)
    second = torch.load(raw_probe / "condition-repeat.pt", map_location="cpu", weights_only=True)
    return (SimpleNamespace(prompt_embeds=first["prompt_embeds"].clone(), attention_mask=first["attention_mask"].clone()),
            SimpleNamespace(prompt_embeds=second["prompt_embeds"].clone(), attention_mask=second["attention_mask"].clone()))


def test_all_entries_share_new_locked_config_and_no_old_runtime(config) -> None:
    assert trainer.PROTOCOL == controller.core.BRIDGE_PROTOCOL == comparison.core.BRIDGE_PROTOCOL
    assert trainer.CONFIG_PATH == controller.CONFIG == comparison.CONFIG
    assert "identity_condition_bridge.py" in str(controller.TRAINER)
    assert "identity-conditioning-bridge.lock" in str(controller.LOCK_PATH)
    assert controller.LOCK_PATH != binding.probe.LOCK_PATH
    assert controller.EXPECTED_HOST_PREFIX == config["deployment"]["instance"]
    assert controller.EXPECTED_ENVIRONMENT["CUDA_VISIBLE_DEVICES"] == "0,1"
    text = inspect.getsource(trainer._run)
    assert "condition_binding.load_runtime(args, _load_config())" in text
    assert "phase1a._load_runtime" not in text
    assert "condition_binding.validate_manifest_identity" in inspect.getsource(controller._validate_child)
    assert "condition_binding.validate_manifest_identity" in inspect.getsource(comparison._validate_manifest)
    for field in ("identity_condition_probe_binding_valid", "actual_conditioning_text_binding_valid"):
        assert field in controller.PREFLIGHT_REQUIRED_AUDIT_TRUE
        assert field in inspect.getsource(comparison._validate_preflight)
        assert field in inspect.getsource(trainer._preflight)
        assert field in inspect.getsource(trainer._technical_gate)


@pytest.mark.parametrize("function", [
    "_train_step", "_source_only_initialization", "_verify_source_only_initialization",
    "_reconstruct_start_state", "_checkpoint_payload", "_save_checkpoint", "_verify_checkpoint_record",
    "_optimizer_state_matches_step", "_distance_statistics", "_evaluation_rows",
])
def test_unchanged_numerical_implementations_match_parent_ast(function):
    new = ast.dump(ast.parse(inspect.getsource(getattr(trainer, function))), include_attributes=False)
    old = ast.dump(ast.parse(inspect.getsource(getattr(source_trainer, function))), include_attributes=False)
    assert new == old


def test_actual_tensor_pair_roundtrip_keeps_original_and_actual_text_separate(tmp_path, tensors, config):
    record = binding.save_condition_pair(tmp_path, *tensors, config)
    assert binding.verify_condition_record(record, config=config)
    metadata = record["metadata"]
    assert metadata["actual_conditioning_text"] == "no changes"
    assert "ambient" in metadata["original_event_text"]
    assert metadata["original_event_is_provenance_only"] is True
    assert record["event_text_sha256"] == metadata["original_event_text_sha256"]
    assert record["repeat"]["path"] != record["path"]
    assert record["tensor_sha256"] == record["repeat"]["tensor_sha256"]


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("mutation", ["nan", "zero", "different", "mask", "shape"])
def test_bad_or_changed_raw_encodings_fail_before_training(tmp_path, tensors, config, which, mutation):
    value = tensors[which]
    if mutation == "nan":
        value.prompt_embeds[0, 0, 0] = float("nan")
    elif mutation == "zero":
        value.prompt_embeds.zero_()
    elif mutation == "different":
        value.prompt_embeds[0, 0, 0] += 1
    elif mutation == "mask":
        value.attention_mask[0, 0] = 2
    else:
        value.prompt_embeds = value.prompt_embeds[:, :292]
    with pytest.raises(ValueError):
        binding.save_condition_pair(tmp_path, *tensors, config)
    assert (tmp_path / "condition/official_full_condition.pt").is_file()
    assert (tmp_path / "condition/identity_condition_repeat.pt").is_file()


@pytest.mark.parametrize("key,value", [("actual_conditioning_text", "ambient"), ("original_event_text", "no changes"),
    ("full_prompt", "changed"), ("original_event_is_provenance_only", False), ("probe_manifest_sha256", "0" * 64)])
def test_self_resigned_metadata_cannot_change_fixed_text(tmp_path, tensors, config, key, value):
    record = binding.save_condition_pair(tmp_path, *tensors, config)
    for item in (record, record["repeat"]):
        path = Path(item["path"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["metadata"][key] = value
        item["metadata"][key] = value
        binding.phase1a._atomic_torch_save(path, payload)
        item.update(sha256=binding.phase1a._sha256(path), bytes=path.stat().st_size)
    assert not binding.verify_condition_record(record, config=config)


def test_same_file_cannot_impersonate_second_encoding(tmp_path, tensors, config):
    record = binding.save_condition_pair(tmp_path, *tensors, config)
    record["repeat"] = {k: v for k, v in record.items() if k not in {"repeat", "recompute_matches"}}
    assert not binding.verify_condition_record(record, config=config)


def test_probe_config_mutation_rejected_before_any_file_read(config):
    config["condition_probe_binding"]["source_root"] = "nonexistent"
    with pytest.raises(ValueError, match="canonical config"):
        binding.validate_probe_binding(config)


def test_full_probe_raw_audit_with_only_location_substituted(raw_probe, config, monkeypatch):
    # This unit fixture relocates already SHA-bound raw artifacts. Production
    # has no bypass flag and uses the unchanged canonical JSON and remote roots.
    config["condition_probe_binding"]["source_root"] = str(raw_probe)
    monkeypatch.setattr(binding.core, "validate_bridge_config", lambda value: value)
    manifest = json.loads((raw_probe / "manifest.json").read_text())
    monkeypatch.setattr(binding.probe, "_validate_parents", lambda _config: manifest["parent_binding"])
    audit = binding.validate_probe_binding(config)
    assert audit["passed"] is True
    assert audit["tensor_sha256"]["prompt_embeds"] == binding.core.BRIDGE_CONDITION_EMBEDS_SHA256


@pytest.mark.parametrize("field", ["identity_condition_probe_binding", "single_changed_factor", "information_boundary", "condition_artifact"])
def test_manifest_gate_rejects_each_binding_failure(tmp_path, tensors, config, monkeypatch, field):
    record = binding.save_condition_pair(tmp_path, *tensors, config)
    probe_binding = {"passed": True, "manifest_sha256": binding.core.BRIDGE_PROBE_MANIFEST_SHA256}
    monkeypatch.setattr(binding, "validate_probe_binding", lambda _config: probe_binding)
    manifest = {"identity_condition_probe_binding": probe_binding,
        "single_changed_factor": binding.single_changed_factor(config),
        "information_boundary": binding.core.bridge_information_boundary(), "condition_artifact": record}
    assert binding.validate_manifest_identity(manifest, config=config)
    changed = copy.deepcopy(manifest)
    changed[field] = {}
    assert not binding.validate_manifest_identity(changed, config=config)


class ToyOracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_T_fp32 = nn.Parameter(torch.ones(1))
        self.unet = nn.Linear(1, 1).requires_grad_(False)
        self.vae = nn.Linear(1, 1).requires_grad_(False)
        self.text_encoder = nn.Linear(1, 1).requires_grad_(False)
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1
        value = self.x_T_fp32 * 2
        return SimpleNamespace(z_t=value, image=value.sigmoid().reshape(1, 1, 1, 1),
            trajectory=tuple(value + i for i in range(5)), effective_sigmas=trainer.phase1a.EFFECTIVE_SIGMAS)


def test_new_trainer_executes_dense_mse_and_exact_256_adam_calls_without_reader_gradient():
    oracle, reader = ToyOracle(), nn.Linear(1, 1).requires_grad_(False)
    optimizer = torch.optim.Adam([oracle.x_T_fp32], lr=trainer.LEARNING_RATE)
    rows = [trainer._train_step(step_zero=i, oracle=oracle, optimizer=optimizer,
            teacher=torch.zeros(1), m0_mse=4.0, reader=reader) for i in range(256)]
    assert oracle.forward_calls == 256
    assert rows[0]["mse"] == 4.0 and rows[0]["reader_gradient_calls"] == 0
    assert all(r["gradient_clipping_applied"] is False and r["dreamlite_denoising_steps"] == 4 for r in rows)
    assert rows[-1]["learning_rate"] == 0 and rows[-1]["gradient_norm"] > 0 and rows[-1]["x_T_update_norm"] == 0
    assert int(optimizer.state[oracle.x_T_fp32]["step"].item()) == 256
    assert all(p.grad is None for p in reader.parameters())


def test_native_condition_runtime_does_not_pass_original_event_or_query_to_encoder():
    tree = ast.parse(inspect.getsource(binding.load_runtime))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "encode_latent_path_condition"]
    assert len(calls) == 2
    assert all(ast.unparse(call.args[-1]) == "core.BRIDGE_CONDITION_TEXT" for call in calls)
    forbidden = {"_initial_x_t", "_load_runtime"}
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in forbidden for n in ast.walk(tree))
