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


def test_formal_blocks_before_update_129_when_step128_prefix_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _ToyOracle()
    reader = _reader()
    output_dir = tmp_path / "formal"
    output_dir.mkdir()
    (output_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    initial_output = oracle()
    monkeypatch.setattr(trainer, "OPTIMIZER_STEPS", 129)
    monkeypatch.setattr(trainer, "CHECKPOINT_STEPS", (0, 128, 129))
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
        if step_zero + 1 >= 129:
            pytest.fail("the changed LR update ran before prefix parity was accepted")
        return {
            "optimizer_step": step_zero + 1,
            "mse": 1.0,
            "mse_ratio_to_m0": 1.0,
            "gradient_norm": 1.0,
        }

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
    with pytest.raises(RuntimeError, match="blocking before the first changed LR update 129"):
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
    assert calls == list(range(1, 129))


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
            manifest={"condition_artifact": {"sha256": "condition"}},
            snapshot_bindings={},
        )
    technical = json.loads((output_dir / "technical_gate.json").read_text(encoding="utf-8"))
    assert technical["passed"] is False
