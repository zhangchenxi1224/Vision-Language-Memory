from __future__ import annotations

import ast
import copy
import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import probe_r11_new_identity_condition as probe  # noqa: E402


def _write(path: Path, value: dict) -> str:
    probe.phase1a._atomic_json(path, value)
    return probe.safety._sha256(path)


@pytest.fixture
def config() -> dict:
    return copy.deepcopy(probe._load_config())


@pytest.fixture
def parents(tmp_path: Path, config: dict) -> tuple[dict, dict]:
    parent = config["parent_source_init"]
    parent["source_root"] = str(tmp_path / "source-parent")
    config["parent_phase1a"]["source_root"] = str(tmp_path / "phase1a-parent")
    target = {"schema": "vision_memory.r5-compose-segments.v1", "family": "F1",
              "segment_id": config["parent_phase1a"]["target_segment_id"],
              "events": [{"event_text": config["condition"]["original_event_text"]}],
              "query": {"choices": ["none", "ambient", "blues", "jazz"], "target_index": 1}}
    identity = {"target_index": 1, "target_segment_id": target["segment_id"], "target_segment": target}
    original = {"schema": probe.phase1a.MANIFEST_SCHEMA, **identity}
    original_path = Path(config["parent_phase1a"]["source_root"]) / "run/manifest.json"
    config["parent_phase1a"]["manifest_sha256"] = _write(original_path, original)
    formal = {"protocol": probe.safety.core.BRIDGE_PROTOCOL, **identity,
              "preregistered_config_file_sha256": parent["config_sha256"],
              "source_rgb": {"sha256": config["source"]["rgb_sha256"]},
              "source_latents": {"sha256": config["source"]["latents_fp32_sha256"]},
              "condition_artifact": {"event_text_sha256": config["condition"]["original_event_text_sha256"],
                  "tensor_sha256": {"prompt_embeds": config["condition"]["old_prompt_embeds_sha256"],
                                    "attention_mask": config["condition"]["old_attention_mask_sha256"]}}}
    root = Path(parent["source_root"])
    parent["formal_manifest_sha256"] = _write(root / "formal-target01/run/manifest.json", formal)
    parent["raw_artifacts_sha256"] = _write(root / "aggregation-v1/RAW_ARTIFACTS.json", {"test_raw": True})
    comparison = {"protocol": probe.safety.core.BRIDGE_PROTOCOL, **identity,
                  "git_commit": parent["training_git_commit"], "raw_artifacts_sha256": parent["raw_artifacts_sha256"]}
    for name in ("engineering_gate", "teacher_replay_gate", "bridge_distance_gate", "endpoint_reader_transfer_gate",
                 "formal_success", "phase2_allowed"):
        comparison[name] = parent[name]
    parent["comparison_sha256"] = _write(root / "aggregation-v1/comparison.json", comparison)
    return config, probe._validate_parents(config)


class FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def decode(self, latent: torch.Tensor, return_dict: bool = False):
        image = torch.zeros((1, 3, 8, 8), dtype=latent.dtype)
        return SimpleNamespace(sample=image) if return_dict else (image,)


class FakePipeline:
    def __init__(self, *, bad: str | None = None) -> None:
        self.vae = FakeVAE()
        self.unet = nn.Linear(1, 1).requires_grad_(False)
        self.text_encoder = nn.Linear(1, 1).requires_grad_(False)
        self.components = {"vae": self.vae, "unet": self.unet, "text_encoder": self.text_encoder}
        self.image_processor = SimpleNamespace(postprocess=lambda _decoded, output_type: ["fake-source-PIL"])
        self.calls: list[dict] = []
        self.bad = bad

    def encode_prompt(self, **kwargs):
        self.calls.append(kwargs)
        if self.bad == "unet":
            self.unet(torch.ones(1, 1))
        embeds = torch.ones((1, 4, 8), dtype=torch.bfloat16)
        mask = torch.ones((1, 4), dtype=torch.int64)
        if self.bad == "repeat" and len(self.calls) == 2:
            embeds[0, 0, 0] = 2
        if self.bad == "nan":
            embeds[0, 0, 0] = float("nan")
        if self.bad == "zero":
            embeds.zero_()
        if self.bad == "mask":
            mask[0, 0] = 2
        return embeds, mask


@pytest.fixture
def sample(parents: tuple[dict, dict]) -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    config, binding = parents
    rgb = probe.phase1a.blank_source_rgb(device=torch.device("cpu"), dtype=torch.bfloat16)
    latent = torch.ones((1, 4, 128, 128), dtype=torch.bfloat16)
    config["source"]["rgb_sha256"] = probe.canonical_tensor_sha256(rgb)
    config["source"]["latents_fp32_sha256"] = probe.canonical_tensor_sha256(latent.float())
    return config, binding, rgb, latent


def _collect(tmp_path: Path, sample: tuple, *, bad: str | None = None):
    config, binding, rgb, latent = sample
    root = tmp_path / "probe-output"
    root.mkdir()
    pipe = FakePipeline(bad=bad)
    counters = {**probe.ZERO_COUNTERS, "condition_encoder_calls": 0}
    records, audit = probe._collect_condition_artifacts(root, pipe, rgb, latent, config, binding, counters)
    return root, pipe, counters, records, audit


def test_real_config_hash_and_interface() -> None:
    config = probe._load_config()
    assert config["information_boundary"]["phase2_allowed"] is False
    assert list(inspect.signature(probe._encode_condition).parameters) == ["pipe", "source_latents", "fixed_text"]
    args = probe.build_parser().parse_args(["--output-root", "fresh", "--expected-commit", "a" * 40])
    assert vars(args) == {"output_root": Path("fresh"), "expected_commit": "a" * 40}
    calls = [node.func for node in ast.walk(ast.parse(inspect.getsource(probe))) if isinstance(node, ast.Call)]
    attributes = {node.attr for node in calls if isinstance(node, ast.Attribute)}
    assert "_load_pipeline" in attributes
    assert not {"_load_runtime", "_load_reader", "FrozenDreamLiteOracle", "backward", "step", "Adam"} & attributes


def test_locked_config_drift_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    path = tmp_path / "config.json"
    config["condition"]["actual_conditioning_text"] = "different"
    _write(path, config)
    monkeypatch.setattr(probe, "CONFIG", path)
    with pytest.raises(ValueError, match="config file hash"):
        probe._load_config()


def test_exact_two_native_condition_calls_and_no_training(tmp_path: Path, sample: tuple) -> None:
    config, binding, _, _ = sample
    root, pipe, counters, records, audit = _collect(tmp_path, sample)
    assert counters == {**probe.ZERO_COUNTERS, "condition_encoder_calls": 2}
    assert len(pipe.calls) == 2 and audit["passed"] is True
    assert pipe.calls[0] == pipe.calls[1]
    assert pipe.calls[0]["mode"] == "edit"
    assert pipe.calls[0]["prompts"] == [probe.official_mobile_edit_prompt("no changes")]
    assert "ambient" not in pipe.calls[0]["prompts"][0]
    assert probe._frozen_audit(probe._modules(pipe))["passed"] is True
    assert not pipe.unet._forward_pre_hooks
    payload = torch.load(root / "condition.pt", weights_only=True)
    assert payload["metadata"]["original_event_text"] == config["condition"]["original_event_text"]
    assert payload["metadata"]["actual_conditioning_text"] == "no changes"
    assert probe._validate_artifacts(root, config, binding, records) == audit


@pytest.mark.parametrize("text", ["", "no change", "No changes", "no changes ", "ambient"])
def test_other_text_rejected(text: str) -> None:
    pipe = FakePipeline()
    with pytest.raises(ValueError, match="exactly 'no changes'"):
        probe._encode_condition(pipe, torch.ones(1, 4, 128, 128), text)
    assert not pipe.calls


@pytest.mark.parametrize("bad,match", [("repeat", "Repeated condition"), ("nan", "Nonfinite"),
                                      ("zero", "zero embeddings"), ("mask", "binary attention")])
def test_bad_tensor_or_repeat_fail_closed(tmp_path: Path, sample: tuple, bad: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _collect(tmp_path, sample, bad=bad)
    # Preserve the actual bad tensor, not just its rejected summary.
    assert (tmp_path / "probe-output/condition.pt").is_file()


def test_forbidden_unet_counts_and_raises(tmp_path: Path, sample: tuple) -> None:
    config, binding, rgb, latent = sample
    counters = {**probe.ZERO_COUNTERS, "condition_encoder_calls": 0}
    pipe = FakePipeline(bad="unet")
    with pytest.raises(RuntimeError, match="U-Net forward is forbidden"):
        probe._collect_condition_artifacts(tmp_path, pipe, rgb, latent, config, binding, counters)
    assert counters["unet_forward_calls"] == 1
    assert counters["condition_encoder_calls"] == 1
    assert not pipe.unet._forward_pre_hooks


def test_mask_may_equal_old(tmp_path: Path, sample: tuple) -> None:
    sample[0]["condition"]["old_attention_mask_sha256"] = probe.canonical_tensor_sha256(torch.ones(1, 4, dtype=torch.int64))
    assert _collect(tmp_path, sample)[4]["attention_mask_equals_old"] is True


def test_embedding_must_differ_from_old(tmp_path: Path, sample: tuple) -> None:
    sample[0]["condition"]["old_prompt_embeds_sha256"] = probe.canonical_tensor_sha256(torch.ones(1, 4, 8, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="did not change embedding"):
        _collect(tmp_path, sample)


@pytest.mark.parametrize("field,value", [("actual_conditioning_text", "ambient"), ("original_event_text", "rewritten"),
                                        ("full_prompt", "rewritten"), ("compute_device", "cpu"),
                                        ("source_latents_fp32_sha256", "0" * 64)])
def test_resigned_condition_metadata_cannot_override_preregistration(
    tmp_path: Path, sample: tuple, field: str, value: str,
) -> None:
    config, binding, _, _ = sample
    root, _, _, records, _ = _collect(tmp_path, sample)
    for name, filename in (("condition", "condition.pt"), ("repeat", "condition-repeat.pt")):
        path = root / filename
        payload = torch.load(path, weights_only=True)
        payload["metadata"][field] = value
        if f"{field}_sha256" in payload["metadata"]:
            payload["metadata"][f"{field}_sha256"] = probe._text_sha256(value)
        probe.phase1a._atomic_torch_save(path, payload)
        records[name].update(probe._file_record(path))
    with pytest.raises(ValueError, match="metadata disagrees"):
        probe._validate_artifacts(root, config, binding, records)


@pytest.mark.parametrize("name", ["comparison", "raw_artifacts", "formal_manifest", "phase1a_manifest"])
def test_parent_file_hash_drift_rejected(parents: tuple[dict, dict], name: str) -> None:
    config, binding = parents
    path = Path(binding["artifacts"][name]["path"])
    payload = probe.safety._load(path)
    payload["tampered"] = True
    _write(path, payload)
    with pytest.raises(ValueError, match="Parent hash drifted"):
        probe._validate_parents(config)


def test_parent_original_target_still_required_after_source_manifest_resign(parents: tuple[dict, dict]) -> None:
    config, binding = parents
    path = Path(binding["artifacts"]["formal_manifest"]["path"])
    payload = probe.safety._load(path)
    payload["target_segment"]["query"]["target_index"] = 0
    config["parent_source_init"]["formal_manifest_sha256"] = _write(path, payload)
    with pytest.raises(ValueError, match="Original target/event/query"):
        probe._validate_parents(config)


@pytest.mark.parametrize("which", ["rgb", "latent"])
def test_source_drift_rejected(sample: tuple, which: str) -> None:
    config, _, rgb, latent = sample
    if which == "rgb":
        rgb[0, 0, 0, 0] = 0
    else:
        latent[0, 0, 0, 0] = 0
    with pytest.raises(ValueError, match="hash drifted"):
        probe._source_checks(rgb, latent.float(), config)


def test_fresh_root_rejects_even_empty_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fresh output root"):
        probe._claim_root(tmp_path)
    fresh = tmp_path / "fresh"
    probe._claim_root(fresh)
    assert fresh.is_dir()


def test_independent_lock_collision_and_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_path = tmp_path / "probe.lock"
    monkeypatch.setattr(probe, "LOCK_PATH", lock_path)
    lock = probe._acquire_lock(tmp_path / "fresh", "a" * 40)
    with pytest.raises(FileExistsError):
        probe._acquire_lock(tmp_path / "other", "b" * 40)
    assert probe.safety._release_lock(lock)["released"] is True
    assert not lock_path.exists()


@pytest.mark.parametrize("bad", ["trainable", "gradient"])
def test_frozen_parameter_audit_rejects_bad_state(bad: str) -> None:
    pipe = FakePipeline()
    if bad == "trainable":
        pipe.text_encoder.weight.requires_grad_(True)
    else:
        pipe.vae.weight.grad = torch.zeros_like(pipe.vae.weight)
    with pytest.raises(ValueError, match="frozen without gradients"):
        probe._frozen_audit(probe._modules(pipe))


@pytest.mark.parametrize("bad", ["rank", "dtype", "mask_shape", "mask_dtype", "mask_zero", "grad"])
def test_condition_shape_dtype_and_gradient_checks(bad: str) -> None:
    embeds = torch.ones(1, 4, 8, dtype=torch.bfloat16)
    mask = torch.ones(1, 4, dtype=torch.int64)
    if bad == "rank":
        embeds = embeds[0]
    elif bad == "dtype":
        embeds = embeds.float()
    elif bad == "mask_shape":
        mask = mask[:, :2]
    elif bad == "mask_dtype":
        mask = mask.float()
    elif bad == "mask_zero":
        mask.zero_()
    else:
        embeds.requires_grad_(True)
    with pytest.raises(ValueError):
        probe._condition_tensor_checks(embeds, mask)


def test_failure_terminal_inventory_and_lock_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    root = tmp_path / "run"
    monkeypatch.setattr(probe, "LOCK_PATH", tmp_path / "probe.lock")
    monkeypatch.setattr(probe, "_validate_environment", lambda args, cfg: {"git_commit": args.expected_commit})
    monkeypatch.setattr(probe, "_validate_parents", lambda cfg: {"test": True})

    def fail(_root, _config, _validation, _parents, counters):
        counters["unet_forward_calls"] = 1
        raise RuntimeError("forbidden test U-Net")

    monkeypatch.setattr(probe, "_run_probe", fail)
    assert probe.main(["--output-root", str(root), "--expected-commit", "a" * 40]) == 2
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert terminal["engineering_gate"] is False and terminal["phase2_allowed"] is False
    assert terminal["formal_success"] is False and terminal["counters"]["unet_forward_calls"] == 1
    assert "forbidden test U-Net" in (root / "stderr.log").read_text(encoding="utf-8")
    assert terminal["lock_release"]["released"] is True
    probe.safety._validate_inventory(root, schema=probe.INVENTORY_SCHEMA)


def test_existing_root_is_never_overwritten_by_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "keep.json"
    _write(marker, {"keep": True})
    before = marker.read_bytes()
    monkeypatch.setattr(probe, "LOCK_PATH", tmp_path / "probe.lock")
    monkeypatch.setattr(probe, "_validate_environment", lambda args, cfg: {})
    monkeypatch.setattr(probe, "_validate_parents", lambda cfg: {})
    assert probe.main(["--output-root", str(root), "--expected-commit", "a" * 40]) == 2
    assert marker.read_bytes() == before
    assert list(root.iterdir()) == [marker]
    assert not probe.LOCK_PATH.exists()


@pytest.mark.parametrize("failure_count", [1, 2])
def test_inventory_failure_cannot_leave_pass_terminal_or_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_count: int,
) -> None:
    root = tmp_path / "run"
    monkeypatch.setattr(probe, "LOCK_PATH", tmp_path / "probe.lock")
    monkeypatch.setattr(probe, "_validate_environment", lambda args, cfg: {"git_commit": args.expected_commit})
    monkeypatch.setattr(probe, "_validate_parents", lambda cfg: {})

    def succeed(root, _config, _validation, _parents, counters):
        counters["condition_encoder_calls"] = 2
        _write(root / "manifest.json", {"test": True})
        _write(root / "result.json", {"engineering_gate": True, "formal_success": False, "phase2_allowed": False})

    original_inventory = probe._write_inventory
    calls = []

    def fail_inventory(path):
        calls.append(path)
        if len(calls) <= failure_count:
            raise OSError("injected inventory write or validation failure")
        original_inventory(path)

    monkeypatch.setattr(probe, "_run_probe", succeed)
    monkeypatch.setattr(probe, "_write_inventory", fail_inventory)
    assert probe.main(["--output-root", str(root), "--expected-commit", "a" * 40]) == 2
    terminal = probe.safety._load(root / "terminal.json")
    result = probe.safety._load(root / "result.json")
    for value in (terminal, result):
        assert value["engineering_gate"] is False
        assert value["formal_success"] is False and value["phase2_allowed"] is False
        assert value["status"] == "technical_failed"
        assert "injected inventory" in value["inventory_error"]
    assert terminal["result_sha256"] == probe.safety._sha256(root / "result.json")
    if failure_count == 1:
        probe.safety._validate_inventory(root, schema=probe.INVENTORY_SCHEMA)
    else:
        assert "inventory_recovery_error" in terminal


def test_runtime_requires_cuda_before_model_load(config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Test substitutes this module's facade, not a production module's globals.
    safety_stub = SimpleNamespace(_deployment_audit=lambda path: {}, _COMMIT_RE=probe.safety._COMMIT_RE,
        _git=lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(probe, "safety", safety_stub)
    monkeypatch.setattr(probe, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    for key, value in config["execution"]["environment"].items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match="two visible H200"):
        probe._validate_environment(Namespace(output_root=tmp_path, expected_commit="a" * 40), config)
