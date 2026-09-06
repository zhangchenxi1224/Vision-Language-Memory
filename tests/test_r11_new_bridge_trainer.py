from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train import r11_new_canonical_latent_bridge as trainer  # noqa: E402


class _FrozenScalar(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()), requires_grad=False)


class _ToyOracle(nn.Module):
    def __init__(self, *, forbid_forward: bool = False) -> None:
        super().__init__()
        self.x_T_fp32 = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("initial_x_T_fp32", self.x_T_fp32.detach().clone())
        self.register_buffer("source_latents", torch.zeros_like(self.x_T_fp32))
        self.compute_dtype = torch.float32
        self.unet = _FrozenScalar()
        self.vae = _FrozenScalar()
        self.text_encoder = _FrozenScalar()
        self.forward_calls = 0
        self.forbid_forward = forbid_forward

    def forward(self) -> SimpleNamespace:
        self.forward_calls += 1
        if self.forbid_forward:
            raise AssertionError("preflight executed an unregistered second forward")
        z_t = self.x_T_fp32 * 2.0
        return SimpleNamespace(
            z_t=z_t,
            image=z_t.sigmoid().reshape(1, 1, 1, 1),
            trajectory=tuple(z_t + float(index) for index in range(5)),
            effective_sigmas=tuple(trainer.phase1a.EFFECTIVE_SIGMAS),
        )


def _reader() -> nn.Module:
    module = _FrozenScalar()
    module.eval()
    return module


def test_parse_args_uses_locked_phase1a_schedule_seed() -> None:
    args = trainer.parse_args(
        [
            "--mode",
            "technical-preflight",
            "--train",
            "train.jsonl",
            "--dev",
            "dev.jsonl",
            "--dreamlite",
            "dreamlite",
            "--reader",
            "reader",
            "--teacher",
            "teacher.pt",
            "--phase1a-comparison",
            "comparison.json",
            "--phase1a-raw-artifacts",
            "RAW_ARTIFACTS.json",
            "--phase1a-target-root",
            "target-01-retry01",
            "--output-dir",
            "output",
        ]
    )
    assert args.schedule_seed == trainer.phase1a.R10_SELECTION_SEED
    assert args.target_index == 1
    assert args.seed == 0


def test_train_step_is_dense_mse_only_and_updates_x_t() -> None:
    oracle = _ToyOracle()
    reader = _reader()
    optimizer = torch.optim.Adam((oracle.x_T_fp32,), lr=trainer.LEARNING_RATE)
    before = oracle.x_T_fp32.detach().clone()
    metric = trainer._train_step(
        step_zero=0,
        oracle=oracle,
        optimizer=optimizer,
        teacher=torch.zeros(1),
        m0_mse=4.0,
        reader=reader,
    )
    assert oracle.forward_calls == 1
    assert metric["objective"] == "canonical_r11_latent_fp32_mean_mse"
    assert metric["optimizer_step"] == 1
    assert metric["reader_gradient_calls"] == 0
    assert metric["gradient_clipping_applied"] is False
    assert metric["dreamlite_denoising_steps"] == 4
    assert metric["mse"] == pytest.approx(4.0)
    assert metric["gradient_norm"] > 0.0
    assert metric["x_T_update_norm"] > 0.0
    assert not torch.equal(before, oracle.x_T_fp32.detach())
    assert oracle.x_T_fp32.grad is None
    assert all(parameter.grad is None for parameter in reader.parameters())


def test_train_step_executes_exact_schedule_through_zero_lr_update_256() -> None:
    oracle = _ToyOracle()
    reader = _reader()
    optimizer = torch.optim.Adam((oracle.x_T_fp32,), lr=trainer.LEARNING_RATE)
    metrics = [
        trainer._train_step(
            step_zero=step_zero,
            oracle=oracle,
            optimizer=optimizer,
            teacher=torch.zeros(1),
            m0_mse=4.0,
            reader=reader,
        )
        for step_zero in range(256)
    ]
    assert metrics[127]["learning_rate"] == trainer.LEARNING_RATE
    assert metrics[128]["learning_rate"] == trainer.core.bridge_optimizer_learning_rate(129)
    assert metrics[191]["learning_rate"] == pytest.approx(0.025, rel=0.0, abs=1e-16)
    assert metrics[255]["learning_rate"] == pytest.approx(0.0, rel=0.0, abs=1e-16)
    assert metrics[255]["gradient_norm"] > 0.0
    assert metrics[255]["x_T_update_norm"] == 0.0
    assert float(optimizer.state[oracle.x_T_fp32]["step"].item()) == 256.0


def test_optimizer_state_contract_binds_state_key_counter_and_group_param() -> None:
    parameter_state = {
        "step": torch.tensor(256.0),
        "exp_avg": torch.zeros((1, 4, 128, 128), dtype=torch.float32),
        "exp_avg_sq": torch.zeros((1, 4, 128, 128), dtype=torch.float32),
    }
    valid = {
        "state": {0: parameter_state},
        "param_groups": [{"lr": 0.0, "params": [0]}],
    }
    assert trainer._optimizer_state_matches_step(valid, expected_step=256)
    wrong_counter = copy.deepcopy(valid)
    wrong_counter["state"][0]["step"] = torch.tensor(255.0)
    assert not trainer._optimizer_state_matches_step(wrong_counter, expected_step=256)
    wrong_key = copy.deepcopy(valid)
    wrong_key["state"] = {1: wrong_key["state"].pop(0)}
    assert not trainer._optimizer_state_matches_step(wrong_key, expected_step=256)
    wrong_group_param = copy.deepcopy(valid)
    wrong_group_param["param_groups"][0]["params"] = [1]
    assert not trainer._optimizer_state_matches_step(wrong_group_param, expected_step=256)


def test_preflight_reuses_the_single_registered_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oracle = _ToyOracle(forbid_forward=True)
    reader = _reader()
    optimizer = torch.optim.Adam((oracle.x_T_fp32,), lr=trainer.LEARNING_RATE)
    z_t = oracle.x_T_fp32 * 2.0
    initial_output = SimpleNamespace(
        z_t=z_t,
        image=z_t.sigmoid().reshape(1, 1, 1, 1),
        trajectory=tuple(z_t + float(index) for index in range(5)),
        effective_sigmas=tuple(trainer.phase1a.EFFECTIVE_SIGMAS),
    )
    monkeypatch.setattr(trainer, "_evaluation_rows", lambda **_kwargs: [{"row": 1}])
    monkeypatch.setattr(
        trainer.core,
        "reader_checkpoint_statistics",
        lambda *_args, **_kwargs: {
            "row_count": 4,
            "all_four_correct": True,
            "mean_ce": 0.0,
        },
    )
    monkeypatch.setattr(trainer, "_save_checkpoint", lambda **_kwargs: ({"optimizer_step": 0}, initial_output))
    monkeypatch.setattr(trainer, "_verify_checkpoint_record", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(trainer, "_verify_teacher_matched_initialization", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(trainer, "_write_snapshot_end", lambda *_args, **_kwargs: True)
    output_dir = tmp_path / "preflight"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    summary = trainer._preflight(
        args=SimpleNamespace(output_dir=output_dir),
        oracle=oracle,
        reader=reader,
        eval_reader=object(),
        target=object(),
        teacher=torch.zeros(1),
        teacher_image=torch.zeros(1),
        initial_output=initial_output,
        optimizer=optimizer,
        manifest={
            "initialization_binding": {},
            "parity_checks": {"passed": True},
            "teacher": {"tensor_sha256": trainer.core.BRIDGE_TEACHER_TENSOR_SHA256},
            "condition_artifact": {"sha256": "condition"},
        },
        snapshot_bindings={},
    )
    assert oracle.forward_calls == 0
    assert summary["passed"] is True
    assert summary["bridge_result_evaluated"] is False
    assert summary["audit"]["full_forward_calls"] == 1
    assert summary["audit"]["optimizer_steps"] == 0
    assert oracle.x_T_fp32.grad is None


def test_distance_statistics_match_tensor_mse() -> None:
    output = SimpleNamespace(z_t=torch.tensor([1.0, 3.0]))
    statistics = trainer._distance_statistics(output, torch.tensor([0.0, 1.0]), m0_mse=2.5)
    assert statistics["mse"] == pytest.approx(2.5)
    assert statistics["mse_ratio_to_m0"] == pytest.approx(1.0)
    assert math.isfinite(statistics["teacher_normalized_rmse"])


def test_fresh_output_claim_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "run"
    trainer._claim_output_dir(output)
    assert (output / ".r11_new_bridge_output_owner.json").is_file()
    with pytest.raises(ValueError, match="fresh output"):
        trainer._claim_output_dir(output)


def test_formal_saves_teacher_rows_and_stops_before_step_zero_on_replay_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _ToyOracle()
    reader = _reader()
    output_dir = tmp_path / "formal"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    initial_output = oracle()
    teacher_rows = [{"evidence": index} for index in range(4)]
    monkeypatch.setattr(
        trainer,
        "decode_model_latents_unit_interval",
        lambda *_args, **_kwargs: torch.zeros(1, 1, 1, 1),
    )
    monkeypatch.setattr(trainer, "_evaluation_rows", lambda **_kwargs: teacher_rows)
    monkeypatch.setattr(
        trainer.core,
        "reader_checkpoint_statistics",
        lambda *_args, **_kwargs: {
            "row_count": 4,
            "all_four_correct": False,
            "mean_ce": 1.0,
        },
    )
    monkeypatch.setattr(
        trainer,
        "_train_step",
        lambda **_kwargs: pytest.fail("optimizer step ran after teacher replay failure"),
    )
    with pytest.raises(RuntimeError, match="before optimizer step 0"):
        trainer._formal(
            args=SimpleNamespace(output_dir=output_dir),
            oracle=oracle,
            reader=reader,
            eval_reader=object(),
            target=object(),
            source_latents=torch.zeros(1),
            teacher=torch.zeros(1),
            teacher_image=torch.zeros(1),
            initial_output=initial_output,
            optimizer=torch.optim.Adam((oracle.x_T_fp32,), lr=trainer.LEARNING_RATE),
            manifest={"condition_artifact": {"sha256": "condition"}},
            snapshot_bindings={},
        )
    saved = (output_dir / "evaluation_rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(saved) == 4
    assert not (output_dir / "metrics.jsonl").exists()


def test_formal_blocks_before_first_update_when_initialization_is_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _ToyOracle()
    reader = _reader()
    output_dir = tmp_path / "formal"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    initial_output = oracle()
    monkeypatch.setattr(trainer, "_verify_teacher_matched_initialization", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        trainer,
        "decode_model_latents_unit_interval",
        lambda *_args, **_kwargs: torch.zeros(1, 1, 1, 1),
    )
    monkeypatch.setattr(
        trainer,
        "_evaluation_rows",
        lambda **_kwargs: [{"evidence": index} for index in range(4)],
    )
    monkeypatch.setattr(
        trainer.core,
        "reader_checkpoint_statistics",
        lambda *_args, **_kwargs: {
            "row_count": 4,
            "all_four_correct": True,
            "mean_ce": 0.0,
        },
    )
    calls: list[int] = []

    def fake_train_step(*, step_zero: int, **_kwargs: object) -> dict:
        calls.append(step_zero + 1)
        pytest.fail("an optimizer step ran before initialization verification")

    def fake_save_checkpoint(*, step: int, **_kwargs: object) -> tuple[dict, SimpleNamespace]:
        return (
            {
                "optimizer_step": step,
                "tensor_sha256": {},
                "optimizer_state_sha256": "0" * 64,
                "png_sha256": "0" * 64,
            },
            initial_output,
        )

    monkeypatch.setattr(trainer, "_train_step", fake_train_step)
    monkeypatch.setattr(trainer, "_save_checkpoint", fake_save_checkpoint)
    with pytest.raises(RuntimeError, match="initialization verification failed before optimizer step 0"):
        trainer._formal(
            args=SimpleNamespace(output_dir=output_dir),
            oracle=oracle,
            reader=reader,
            eval_reader=object(),
            target=object(),
            source_latents=torch.zeros(1),
            teacher=torch.zeros(1),
            teacher_image=torch.zeros(1),
            initial_output=initial_output,
            optimizer=torch.optim.Adam((oracle.x_T_fp32,), lr=trainer.LEARNING_RATE),
            manifest={"condition_artifact": {"sha256": "condition"}, "initialization_binding": {}},
            snapshot_bindings={},
        )
    assert calls == []


def test_formal_technical_failure_never_emits_scientific_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _ToyOracle()
    reader = _reader()
    output_dir = tmp_path / "formal"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "checkpoints").mkdir()
    (output_dir / "images").mkdir()
    (output_dir / "checkpoints" / "step-256.pt").write_bytes(b"checkpoint")
    (output_dir / "images" / "step-256.png").write_bytes(b"image")
    initial_output = oracle()
    monkeypatch.setattr(trainer, "OPTIMIZER_STEPS", 1)
    monkeypatch.setattr(trainer, "CHECKPOINT_STEPS", (0, 1))
    monkeypatch.setattr(
        trainer,
        "decode_model_latents_unit_interval",
        lambda *_args, **_kwargs: torch.zeros(1, 1, 1, 1),
    )
    monkeypatch.setattr(
        trainer,
        "_evaluation_rows",
        lambda **kwargs: [
            {
                "checkpoint": kwargs["checkpoint"],
                "condition": "teacher",
                "row": index,
            }
            for index in range(4)
        ],
    )
    monkeypatch.setattr(
        trainer.core,
        "reader_checkpoint_statistics",
        lambda *_args, **_kwargs: {
            "row_count": 4,
            "all_four_correct": True,
            "mean_ce": 0.0,
        },
    )
    monkeypatch.setattr(
        trainer,
        "_save_checkpoint",
        lambda **kwargs: ({"optimizer_step": kwargs["step"]}, initial_output),
    )
    monkeypatch.setattr(
        trainer,
        "_train_step",
        lambda **_kwargs: {
            "optimizer_step": 1,
            "mse": 1.0,
            "mse_ratio_to_m0": 1.0,
            "gradient_norm": 1.0,
        },
    )
    monkeypatch.setattr(trainer, "_write_snapshot_end", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(trainer, "_verify_teacher_matched_initialization", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(trainer, "_technical_gate", lambda *_args, **_kwargs: {"passed": False})
    monkeypatch.setattr(
        trainer.core,
        "bridge_decision",
        lambda **_kwargs: pytest.fail("scientific decision emitted after technical failure"),
    )
    with pytest.raises(RuntimeError, match="scientific bridge result is unevaluated"):
        trainer._formal(
            args=SimpleNamespace(output_dir=output_dir),
            oracle=oracle,
            reader=reader,
            eval_reader=object(),
            target=object(),
            source_latents=torch.zeros(1),
            teacher=torch.zeros(1),
            teacher_image=torch.zeros(1),
            initial_output=initial_output,
            optimizer=torch.optim.Adam((oracle.x_T_fp32,), lr=trainer.LEARNING_RATE),
            manifest={"condition_artifact": {"sha256": "condition"}, "initialization_binding": {}},
            snapshot_bindings={},
        )
    technical = json.loads((output_dir / "technical_gate.json").read_text(encoding="utf-8"))
    assert technical["passed"] is False


def _initialization_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype):
    source = torch.linspace(-0.27, 0.31, 65536).reshape(1, 4, 128, 128).to(dtype)
    teacher = torch.linspace(-0.53, 0.69, 65536).reshape(1, 4, 128, 128)
    monkeypatch.setattr(trainer.core, "BRIDGE_SOURCE_LATENTS_SHA256", trainer.canonical_tensor_sha256(source.float()))
    monkeypatch.setattr(trainer.core, "BRIDGE_TEACHER_TENSOR_SHA256", trainer.canonical_tensor_sha256(teacher))
    sigmas = (0.4999999701976776, 0.375, 0.25, 0.1249999925494194)
    prepare_calls = []

    def prepare(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return torch.arange(4), sigmas

    oracle = SimpleNamespace(
        source_latents=source, compute_dtype=dtype,
        x_T_fp32=nn.Parameter(torch.zeros_like(source, dtype=torch.float32)),
        initial_x_T_fp32=torch.zeros_like(source, dtype=torch.float32),
        sampler=SimpleNamespace(_prepare_timesteps=prepare),
    )
    record = trainer._teacher_matched_initialization(
        oracle=oracle, source_latents=source, teacher=teacher, output_dir=tmp_path,
    )
    start = source.mul(1.0 - sigmas[0]).add(oracle.x_T_fp32.to(dtype), alpha=sigmas[0])
    trajectory = tuple(start + index * 0.02 for index in range(5))
    output = SimpleNamespace(trajectory=trajectory, effective_sigmas=sigmas, z_t=trajectory[-1])
    return oracle, teacher, record, output, prepare_calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_teacher_matched_initializer_uses_actual_sigma_fp32_inverse_and_compute_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype,
) -> None:
    oracle, teacher, record, output, prepare_calls = _initialization_case(tmp_path, monkeypatch, dtype)
    sigma = output.effective_sigmas[0]
    expected_x_t = teacher.sub(oracle.source_latents.float().mul(1.0 - sigma)).div(sigma)
    assert len(prepare_calls) == 1
    assert prepare_calls[0][1] == {"sigmas_are_effective": True}
    assert record["actual_effective_sigma0"] == sigma != 0.5
    assert torch.equal(oracle.x_T_fp32, expected_x_t)
    assert oracle.x_T_fp32.dtype == torch.float32
    assert not torch.equal(expected_x_t, teacher.sub(oracle.source_latents.float().mul(0.5)).div(0.5))
    payload = torch.load(record["artifact_path"], weights_only=True)
    assert record["compute_device_type"] == payload["compute_device_type"] == "cpu"
    assert payload["reconstructed_start_state_compute"].dtype == dtype
    assert torch.equal(payload["reconstructed_start_state_compute"], output.trajectory[0])
    mse = float((output.trajectory[0].detach().float() - teacher).square().mean())
    assert record["trajectory_point0_teacher_normalized_rmse"] == math.sqrt(mse) / trainer.core.BRIDGE_TEACHER_STD
    assert trainer._verify_teacher_matched_initialization(record, oracle=oracle, initial_output=output)
    with torch.no_grad():
        oracle.x_T_fp32.add_(0.125)
    assert not trainer._verify_teacher_matched_initialization(record, oracle=oracle, initial_output=output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mutation", [
    "record_sigma", "output_schedule", "both_schedules", "both_nominal", "both_std",
    "payload_mse", "payload_nrmse", "record_mse", "record_nrmse", "both_formula",
    "both_operator", "wrong_x_t", "wrong_start", "wrong_output_h0",
    "payload_device", "record_device", "both_device",
])
def test_teacher_matched_initialization_rejects_resigned_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, mutation: str,
) -> None:
    oracle, _teacher, record, output, _calls = _initialization_case(tmp_path, monkeypatch, dtype)
    payload = torch.load(record["artifact_path"], weights_only=True)
    if mutation == "record_sigma":
        record["actual_effective_sigma0"] = 0.25
    elif mutation == "output_schedule":
        output.effective_sigmas = (0.5, 0.375, 0.25, 0.125)
    elif mutation in {"both_schedules", "both_nominal"}:
        key = "actual_effective_sigmas" if mutation == "both_schedules" else "nominal_effective_sigmas"
        payload[key] = record[key] = [0.5, 0.4, 0.3, 0.2]
    elif mutation == "both_std":
        payload["teacher_population_std"] = record["teacher_population_std"] = 999.0
    elif mutation in {"payload_mse", "record_mse", "payload_nrmse", "record_nrmse"}:
        declared = payload if mutation.startswith("payload") else record
        key = "trajectory_point0_teacher_mse" if mutation.endswith("_mse") else "trajectory_point0_teacher_normalized_rmse"
        declared[key] = 999.0
    elif mutation in {"both_formula", "both_operator"}:
        key = "formula" if mutation == "both_formula" else "reconstruction_operator_order"
        payload[key] = record[key] = "forged but mutually consistent"
    elif mutation == "wrong_x_t":
        payload["x_T_init_fp32"] += 0.125
    elif mutation == "wrong_start":
        payload["reconstructed_start_state_compute"] += 0.125
    elif mutation == "wrong_output_h0":
        output.trajectory = (output.trajectory[0] + 0.125, *output.trajectory[1:])
    elif mutation == "payload_device":
        payload["compute_device_type"] = "cuda"
    elif mutation == "record_device":
        record["compute_device_type"] = "cuda"
    elif mutation == "both_device":
        payload["compute_device_type"] = record["compute_device_type"] = "cuda"
    payload["tensor_sha256"] = {
        key: trainer.canonical_tensor_sha256(payload[key]) for key in payload["tensor_sha256"]
    }
    record["tensor_sha256"] = payload["tensor_sha256"]
    # Re-sign the changed artifact, so the semantic verifier (not stale hashes) rejects it.
    torch.save(payload, record["artifact_path"])
    record["artifact_bytes"] = Path(record["artifact_path"]).stat().st_size
    record["artifact_sha256"] = trainer.phase1a._sha256(Path(record["artifact_path"]))
    assert not trainer._verify_teacher_matched_initialization(record, oracle=oracle, initial_output=output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("step", [0, 64])
def test_checkpoint_independently_reconstructs_h0_from_current_x_t(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype, step: int,
) -> None:
    oracle, teacher, _initialization, output, _calls = _initialization_case(tmp_path, monkeypatch, dtype)
    with torch.no_grad():
        oracle.x_T_fp32.add_(0.125)
    start = trainer._reconstruct_start_state(
        oracle.source_latents, oracle.x_T_fp32, actual_sigma=output.effective_sigmas[0]
    )
    output.trajectory = tuple(start + index * 0.02 for index in range(5))
    output.z_t = output.trajectory[-1]
    optimizer = SimpleNamespace(state_dict=lambda: {
        "state": {} if step == 0 else {0: {
            "step": torch.tensor(float(step)), "exp_avg": torch.zeros_like(teacher),
            "exp_avg_sq": torch.zeros_like(teacher),
        }},
        "param_groups": [{"params": [0], "lr": 0.05}],
    })
    payload = trainer._checkpoint_payload(
        step=step, oracle=oracle, optimizer=optimizer, output=output, teacher=teacher,
        m0_mse=1.0, manifest_sha256="manifest", condition_sha256="condition",
    )
    path, png = tmp_path / "checkpoint.pt", tmp_path / "checkpoint.png"
    png.write_bytes(b"test png bytes")

    def signed_record():
        torch.save(payload, path)
        return {
            "checkpoint_path": str(path), "checkpoint_bytes": path.stat().st_size,
            "checkpoint_sha256": trainer.phase1a._sha256(path), "png_path": str(png),
            "png_bytes": png.stat().st_size, "png_sha256": trainer.phase1a._sha256(png),
            **{key: payload[key] for key in (
                "optimizer_step", "tensor_sha256", "optimizer_state_sha256", "effective_sigmas",
                "compute_dtype", "compute_device_type", "trajectory_point0_formula_valid", "distance_statistics",
            )},
        }

    def verify(record):
        return trainer._verify_checkpoint_record(
            record, expected_step=step, teacher_cpu=teacher, m0_mse=1.0,
            source_latents_cpu=oracle.source_latents, compute_dtype=dtype,
            compute_device=torch.device("cpu"),
        )

    valid_record = signed_record()
    assert valid_record["compute_dtype"] == payload["compute_dtype"] == str(dtype)
    assert valid_record["compute_device_type"] == payload["compute_device_type"] == "cpu"
    assert all(value.dtype == torch.float32 for value in payload["trajectory_fp32"])
    assert verify(valid_record)
    actual_dtype = payload["compute_dtype"]
    payload["compute_dtype"] = "torch.float32" if dtype == torch.bfloat16 else "torch.bfloat16"
    assert not verify(signed_record())
    payload["compute_dtype"] = actual_dtype
    for key, wrong in (("compute_device_type", "cuda"), ("compute_dtype", "forged")):
        record = signed_record()
        record[key] = wrong
        assert not verify(record)
    payload["compute_device_type"] = "cuda"
    assert not verify(signed_record())
    payload["compute_device_type"] = "cpu"
    wrong_start = payload["trajectory_fp32"][0] + 0.125
    payload["trajectory_fp32"] = (wrong_start, *payload["trajectory_fp32"][1:])
    payload["tensor_sha256"]["trajectory_fp32"][0] = trainer.canonical_tensor_sha256(wrong_start)
    payload["trajectory_point0_formula_valid"] = True  # A forged boolean cannot rescue wrong actual h0.
    assert not verify(signed_record())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_run_installs_artifact_before_one_preflight_forward_and_uses_four_step_m0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype,
) -> None:
    source = torch.linspace(-0.27, 0.31, 65536).reshape(1, 4, 128, 128).to(dtype)
    teacher = torch.linspace(-0.53, 0.69, 65536).reshape(1, 4, 128, 128)
    events = []
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    artifact_path = output_dir / "initialization" / "teacher_matched_initialization.pt"

    class Oracle(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_T_fp32 = nn.Parameter(torch.ones_like(source, dtype=torch.float32))
            self.register_buffer("initial_x_T_fp32", self.x_T_fp32.detach().clone())
            self.register_buffer("source_latents", source)
            self.compute_dtype = dtype
            self.unet, self.vae, self.text_encoder = _FrozenScalar(), _FrozenScalar(), _FrozenScalar()
            self.sampler = SimpleNamespace(_prepare_timesteps=lambda *_args, **_kwargs: (
                torch.arange(4), tuple(trainer.phase1a.EFFECTIVE_SIGMAS)
            ))
            self.x_T_fp32.register_hook(lambda grad: events.append("backward"))

        def forward(self):
            assert artifact_path.is_file(), "initialization artifact must precede the first complete forward"
            events.append("forward")
            start = trainer._reconstruct_start_state(self.source_latents, self.x_T_fp32, actual_sigma=0.5)
            trajectory = [start]
            for _step in range(4):
                trajectory.append(trajectory[-1] + 0.02)
            return SimpleNamespace(
                z_t=trajectory[-1], trajectory=tuple(trajectory),
                image=torch.zeros(1, 3, 4, 4), effective_sigmas=tuple(trainer.phase1a.EFFECTIVE_SIGMAS),
            )

    oracle, reader = Oracle(), _reader()
    monkeypatch.setattr(trainer.core, "BRIDGE_SOURCE_LATENTS_SHA256", trainer.canonical_tensor_sha256(source.float()))
    monkeypatch.setattr(trainer.core, "BRIDGE_TEACHER_TENSOR_SHA256", trainer.canonical_tensor_sha256(teacher))
    teacher_file = tmp_path / "teacher.pt"
    torch.save({"latent_fp32": teacher}, teacher_file)
    monkeypatch.setattr(trainer.core, "BRIDGE_TEACHER_FILE_SHA256", trainer.phase1a._sha256(teacher_file))
    teacher_record = {
        "file_sha256": trainer.core.BRIDGE_TEACHER_FILE_SHA256,
        "tensor_sha256": trainer.core.BRIDGE_TEACHER_TENSOR_SHA256,
    }
    monkeypatch.setattr(trainer, "_load_teacher", lambda _path: (teacher, teacher_record))
    monkeypatch.setattr(trainer.r8, "configure_strict_cuda_determinism", lambda _seed: {})
    monkeypatch.setattr(trainer.r8, "set_all_seeds", lambda _seed: None)
    monkeypatch.setattr(trainer.phase1a, "_write_environment", lambda _path: None)
    monkeypatch.setattr(trainer.phase1a, "_runtime_versions", lambda: {})
    monkeypatch.setattr(trainer.phase1a, "_snapshot_bindings", lambda _args: {})
    monkeypatch.setattr(trainer.phase1a, "_load_runtime", lambda _args: (
        None, None, reader, oracle, source,
        {"updater_device": torch.device("cpu"), "reader_device": torch.device("cpu"), "target": object()},
    ))
    monkeypatch.setattr(trainer, "decode_model_latents_unit_interval", lambda *_args, **_kwargs: torch.zeros(1, 3, 4, 4))
    monkeypatch.setattr(trainer.r8, "choice_reader_callable", lambda **_kwargs: object())
    monkeypatch.setattr(trainer, "_evaluation_rows", lambda **_kwargs: [{"row": i} for i in range(4)])
    monkeypatch.setattr(trainer.core, "reader_checkpoint_statistics", lambda *_args, **_kwargs: {
        "row_count": 4, "all_four_correct": True, "mean_ce": 0.0,
    })
    monkeypatch.setattr(trainer, "_write_snapshot_end", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda *_args, **_kwargs: pytest.fail("preflight called Adam.step"))

    def manifest(**kwargs):
        assert trainer._verify_teacher_matched_initialization(
            kwargs["initialization_binding"], oracle=oracle, initial_output=kwargs["initial_output"]
        )
        return {
            "initialization_binding": kwargs["initialization_binding"],
            "parity_checks": {"passed": True}, "teacher": teacher_record,
            "condition_artifact": {"sha256": "condition"},
        }

    monkeypatch.setattr(trainer, "_manifest", manifest)
    result = trainer._run(
        SimpleNamespace(mode="technical-preflight", output_dir=output_dir, teacher=teacher_file),
        parent_binding={}, validated_teacher_record=teacher_record,
    )
    saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert events == ["forward", "backward"]
    assert result["passed"] is True
    assert result["audit"]["optimizer_steps"] == 0
    assert result["bridge_result_evaluated"] is False
    assert result["initial_distance_statistics"]["mse"] > saved_manifest["initialization_binding"]["trajectory_point0_teacher_mse"]
    checkpoint = torch.load(output_dir / "checkpoints" / "step-000.pt", weights_only=True)
    assert result["initial_distance_statistics"]["mse"] == float((checkpoint["z_t_fp32"] - teacher).square().mean())
    record = json.loads((output_dir / "checkpoint_hashes" / "step-000.json").read_text(encoding="utf-8"))
    assert checkpoint["compute_dtype"] == record["compute_dtype"] == str(dtype)
    assert checkpoint["compute_device_type"] == record["compute_device_type"] == "cpu"
    assert all(value.dtype == torch.float32 for value in checkpoint["trajectory_fp32"])
    assert checkpoint["optimizer"]["state"] == {}
