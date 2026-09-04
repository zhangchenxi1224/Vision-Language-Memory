from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    path = ROOT / relative
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


trainer = load_script(
    "r11_new_phase1a_trainer_under_test",
    "scripts/train/r11_new_frozen_dreamlite_oracle.py",
)


class FakeScheduler:
    config = {
        "base_image_seq_len": 256,
        "max_image_seq_len": 4096,
        "base_shift": 0.5,
        "max_shift": 1.16,
        "shift": 1.0,
    }

    def set_timesteps(self, *, sigmas, device, mu):
        del mu
        values = torch.tensor(list(sigmas) + [0.0], device=device, dtype=torch.float32)
        self.sigmas = values
        self.timesteps = values[:-1] * 1000.0
        self._step_index = 0

    def step(self, model_output, timestep, sample, return_dict=False):
        del timestep, return_dict
        delta = self.sigmas[self._step_index + 1] - self.sigmas[self._step_index]
        self._step_index += 1
        return (sample + delta.to(sample.dtype) * model_output,)


class FakeUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.5))
        self.calls = 0

    def forward(
        self,
        sample,
        *,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        added_cond_kwargs,
        return_dict,
    ):
        del timestep, encoder_attention_mask, added_cond_kwargs, return_dict
        self.calls += 1
        target, source = sample.chunk(2, dim=-1)
        prompt = encoder_hidden_states.mean().to(sample.dtype)
        velocity = self.weight * target + 0.1 * source + 0.01 * prompt
        return (torch.cat((velocity, torch.zeros_like(source)), dim=-1),)


class FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def encode(self, image, return_dict=True):
        del return_dict
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: image))

    def decode(self, latent, return_dict=False):
        del return_dict
        return (latent[:, :3] * self.weight,)


class FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))


def make_oracle(*, checkpoint_unet: bool = True):
    unet = FakeUNet()
    vae = FakeVAE()
    text_encoder = FakeTextEncoder()
    source = torch.full((1, 3, 4, 4), 0.25)
    initial_x_t = torch.randn((1, 3, 4, 4), generator=torch.Generator().manual_seed(7))
    oracle = trainer.FrozenDreamLiteOracle(
        unet=unet,
        scheduler=FakeScheduler(),
        vae=vae,
        text_encoder=text_encoder,
        source_latents=source,
        prompt_embeds=torch.ones((1, 2, 3)),
        prompt_attention_mask=torch.ones((1, 2), dtype=torch.long),
        initial_x_t=initial_x_t,
        compute_dtype=torch.float32,
        checkpoint_unet=checkpoint_unet,
        vae_scale_factor=1,
    )
    return oracle, unet, vae, text_encoder


class R11NewPhase1ATrainerTest(unittest.TestCase):
    def test_parser_locks_all_formal_hyperparameters_and_primary_endpoint(self):
        args = trainer.parse_args(
            (
                "--mode",
                "formal",
                "--target-index",
                "3",
                "--train",
                "train.jsonl",
                "--dev",
                "dev.jsonl",
                "--dreamlite",
                "dreamlite",
                "--reader",
                "reader",
                "--output-dir",
                "output",
                "--strict-determinism",
            )
        )
        self.assertEqual(args.mode, "formal")
        self.assertEqual(args.seed, 0)
        self.assertEqual(trainer.OPTIMIZER_STEPS, 256)
        self.assertEqual(trainer.LEARNING_RATE, 0.05)
        self.assertEqual(trainer.WEIGHT_DECAY, 0.0)
        self.assertEqual(trainer.CHECKPOINT_STEPS, (0, 64, 128, 192, 256))
        self.assertEqual(trainer.R11_NEW_PRIMARY_ENDPOINT, "raw_x_T_step256")
        help_text = trainer.build_parser().format_help()
        for forbidden in (
            "--seed",
            "--learning-rate",
            "--optimizer-steps",
            "--weight-decay",
            "--gradient-clip",
            "--checkpoint-steps",
            "--edit-start-sigma",
            "--num-steps",
        ):
            self.assertNotIn(forbidden, help_text)

    def test_oracle_backpropagates_only_to_x_T_through_exact_four_steps(self):
        oracle, unet, vae, text_encoder = make_oracle()
        trainable = [(name, parameter) for name, parameter in oracle.named_parameters() if parameter.requires_grad]
        self.assertEqual([name for name, _parameter in trainable], ["x_T_fp32"])
        self.assertEqual(trainable[0][1].dtype, torch.float32)
        self.assertTrue(all(not parameter.requires_grad for parameter in unet.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in vae.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in text_encoder.parameters()))

        output = oracle()
        self.assertEqual(unet.calls, 4)
        self.assertEqual(len(output.trajectory), 5)
        self.assertEqual(output.effective_sigmas, (0.5, 0.375, 0.25, 0.125))
        output.image.square().mean().backward()
        self.assertIsNotNone(oracle.x_T_fp32.grad)
        self.assertGreater(float(oracle.x_T_fp32.grad.norm()), 0.0)
        self.assertIsNone(unet.weight.grad)
        self.assertIsNone(vae.weight.grad)
        self.assertIsNone(text_encoder.weight.grad)

    def test_checkpoint_contains_x_T_endpoint_five_point_trajectory_png_and_hashes(self):
        oracle, _unet, _vae, _text_encoder = make_oracle(checkpoint_unet=False)
        optimizer = torch.optim.Adam((oracle.x_T_fp32,), lr=0.05, weight_decay=0.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition = trainer._condition_record(
                root / "condition.pt",
                prompt_embeds=oracle.prompt_embeds,
                attention_mask=oracle.prompt_attention_mask,
                event_text="set alpha",
                recompute_matches=True,
            )
            record, _output = trainer._save_checkpoint(
                step=0,
                oracle=oracle,
                optimizer=optimizer,
                manifest_sha256="1" * 64,
                condition_sha256=condition["sha256"],
                output_dir=root,
            )
            self.assertTrue(trainer._verify_checkpoint_record(record, expected_step=0))
            payload = torch.load(root / "checkpoints" / "step-000.pt", map_location="cpu", weights_only=False)
            self.assertEqual(payload["schema"], trainer.CHECKPOINT_SCHEMA)
            self.assertEqual(payload["x_T_fp32"].dtype, torch.float32)
            self.assertEqual(payload["z_t_fp32"].dtype, torch.float32)
            self.assertEqual(len(payload["trajectory_fp32"]), 5)
            self.assertTrue((root / "images" / "step-000.png").is_file())
            self.assertEqual(len(record["checkpoint_sha256"]), 64)
            self.assertEqual(len(record["png_sha256"]), 64)

    def test_formal_technical_gate_delegates_to_locked_core_and_checks_artifacts(self):
        oracle, _unet, _vae, _text_encoder = make_oracle(checkpoint_unet=False)
        reader = nn.Linear(2, 2).requires_grad_(False)
        optimizer = torch.optim.Adam((oracle.x_T_fp32,), lr=0.05, weight_decay=0.0)
        target_id = trainer.R11_NEW_TARGET_IDS[0]
        schedule = trainer.build_phase1a_schedule(0)
        receipts = [
            {
                "schema": trainer.METRICS_SCHEMA,
                "kind": "optimizer_step",
                "optimizer_step": row.optimizer_step,
                "target_segment_id": target_id,
                "forward_cyclic_training_view": row.forward_cyclic_training_view,
                "permutation": list(row.permutation),
                "loss_before_step": 1.0,
                "gradient_norm": 0.5,
                "gradient_nonzero_fraction": 1.0,
                "x_T_update_norm": 0.1,
                "trajectory_points": 5,
                "full_dreamlite_forward_executed": True,
                "denoiser_steps_executed": 4,
                "effective_sigma_schedule": [0.5, 0.375, 0.25, 0.125],
                "gradient_clipping_applied": False,
            }
            for row in schedule
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            condition = trainer._condition_record(
                root / "condition.pt",
                prompt_embeds=oracle.prompt_embeds,
                attention_mask=oracle.prompt_attention_mask,
                event_text="set alpha",
                recompute_matches=True,
            )
            checkpoints = [
                trainer._save_checkpoint(
                    step=step,
                    oracle=oracle,
                    optimizer=optimizer,
                    manifest_sha256="2" * 64,
                    condition_sha256=condition["sha256"],
                    output_dir=root,
                )[0]
                for step in trainer.CHECKPOINT_STEPS
            ]
            gate = trainer._technical_gate(
                receipts,
                target_segment_id=target_id,
                oracle=oracle,
                reader=reader,
                optimizer=optimizer,
                checkpoint_records=checkpoints,
                condition_record=condition,
                snapshots_unchanged=True,
            )
            self.assertTrue(gate["passed"])
            self.assertTrue(gate["audit_valid"])
            self.assertTrue(gate["checkpoint_hashes_valid"])
            self.assertEqual(gate["training_view_counts"], {0: 64, 1: 64, 2: 64, 3: 64})

            invalid = [dict(row) for row in receipts]
            invalid[7]["denoiser_steps_executed"] = 3
            self.assertFalse(
                trainer._technical_gate(
                    invalid,
                    target_segment_id=target_id,
                    oracle=oracle,
                    reader=reader,
                    optimizer=optimizer,
                    checkpoint_records=checkpoints,
                    condition_record=condition,
                    snapshots_unchanged=True,
                )["passed"]
            )

    def test_information_boundary_keeps_query_and_answer_out_of_dreamlite(self):
        event = SimpleNamespace(
            event_text="set the color to blue",
            source_episode_id="episode-1",
            noise_turn_id=2,
        )
        audit = trainer._writer_information_boundary(SimpleNamespace(events=(event,)))
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["dreamlite_inputs"], ["source_latent", "event_text", "x_T"])
        self.assertEqual(audit["leaked_fields"], [])
        self.assertTrue(audit["query_used_only_by_frozen_reader_loss"])
        with self.assertRaisesRegex(ValueError, "drifted or contain supervision"):
            trainer.validate_information_boundary(
                dreamlite_inputs=("source_latent", "event_text", "x_T", "query_text"),
                noise_key=trainer.R11_NEW_NOISE_KEY,
                reader_loss_inputs=trainer.R11_NEW_READER_LOSS_INPUTS,
            )

    def test_pipeline_loader_is_direct_and_contains_no_peft_path(self):
        source = (ROOT / "scripts" / "train" / "r11_new_frozen_dreamlite_oracle.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("DreamLiteMobilePipeline.from_pretrained", source)
        self.assertNotIn("r5._load_pipeline", source)
        self.assertNotIn("from peft", source)

    def test_formal_report_is_derived_from_gates_and_indexes_raw_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "metrics.jsonl"
            raw.write_text('{"optimizer_step":1}\n', encoding="utf-8")
            manifest = {
                "git_commit": "a" * 40,
                "target_index": 2,
                "target_segment_id": trainer.R11_NEW_TARGET_IDS[2],
                "fixed_contract": {"primary_endpoint": trainer.R11_NEW_PRIMARY_ENDPOINT},
            }
            summary = {
                "mode": "formal",
                "target_index": 2,
                "target_segment_id": trainer.R11_NEW_TARGET_IDS[2],
                "optimizer_steps": 256,
                "primary_endpoint": trainer.R11_NEW_PRIMARY_ENDPOINT,
                "gates": {
                    "technical_gate": True,
                    "phase1a_query_level_reachability_gate": True,
                    "formal_success_gate": False,
                },
                "formal_success_gate": False,
                "target_statistics": {
                    "relative_change": -0.25,
                    "improved_choice_views": 4,
                    "accuracy_delta": 0.25,
                },
            }
            trainer._write_report(output_dir=root, summary=summary, manifest=manifest)
            report = (root / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("| 工程通过 | 通过 |", report)
            self.assertIn("query-level full-chain reachability", report)
            self.assertIn("| 科学成功 | `false` |", report)
            self.assertIn("`raw_x_T_step256`", report)
            self.assertIn("`metrics.jsonl`", report)
            self.assertIn(trainer._sha256(raw), report)

    def test_preflight_report_has_no_diagnostic_or_scientific_judgment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = {
                "mode": "technical-preflight",
                "passed": True,
                "backward_calls": 1,
                "optimizer_steps": 0,
                "dreamlite_denoising_steps": 4,
                "effective_sigmas": [0.5, 0.375, 0.25, 0.125],
                "scientific_gate_evaluated": False,
                "gates": {
                    "technical_gate": True,
                    "phase1a_query_level_reachability_gate": None,
                    "formal_success_gate": False,
                },
                "formal_success_gate": False,
            }
            trainer._write_report(
                output_dir=root,
                summary=summary,
                manifest={
                    "git_commit": "b" * 40,
                    "target_index": 0,
                    "target_segment_id": trainer.R11_NEW_TARGET_IDS[0],
                    "fixed_contract": {"primary_endpoint": trainer.R11_NEW_PRIMARY_ENDPOINT},
                },
            )
            report = (root / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("preflight 只执行一次 backward", report)
            self.assertIn("Query-level reachability：未评估", report)
            self.assertIn("| 科学成功 | `false` |", report)

    def test_failure_report_marks_technical_failure_and_no_scientific_conclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "terminal.json").write_text('{"status":"failed"}\n', encoding="utf-8")
            args = SimpleNamespace(mode="formal", target_index=5, output_dir=root)
            trainer._write_failure_report(args=args, error=RuntimeError("synthetic failure"))
            report = (root / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("| 工程通过 | 失败 |", report)
            self.assertIn("技术失败不产生 reachability 结论", report)
            self.assertIn("| 科学成功 | `false` |", report)
            self.assertIn("synthetic failure", report)
            self.assertIn("`terminal.json`", report)

    def test_trainer_refuses_to_overwrite_an_existing_run_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "existing"
            root.mkdir()
            terminal = root / "terminal.json"
            terminal.write_text('{"status":"user-owned"}\n', encoding="utf-8")
            args = SimpleNamespace(mode="formal", target_index=0, output_dir=root)
            with mock.patch.object(trainer, "parse_args", return_value=args):
                with self.assertRaises(SystemExit):
                    trainer.main(())
            self.assertEqual(
                terminal.read_text(encoding="utf-8"),
                '{"status":"user-owned"}\n',
            )
            self.assertFalse((root / "REPORT.md").exists())
            self.assertFalse((root / "artifact_inventory.json").exists())

    def test_technical_failure_has_no_reachability_judgment_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "fresh"
            args = SimpleNamespace(mode="formal", target_index=0, output_dir=root)

            def fake_run(_args):
                trainer._atomic_json(
                    root / "manifest.json",
                    {
                        "git_commit": "c" * 40,
                        "target_index": 0,
                        "target_segment_id": trainer.R11_NEW_TARGET_IDS[0],
                        "fixed_contract": {
                            "primary_endpoint": trainer.R11_NEW_PRIMARY_ENDPOINT
                        },
                    },
                )
                return {
                    "mode": "formal",
                    "target_index": 0,
                    "target_segment_id": trainer.R11_NEW_TARGET_IDS[0],
                    "optimizer_steps": 256,
                    "primary_endpoint": trainer.R11_NEW_PRIMARY_ENDPOINT,
                    "gates": {
                        "technical_gate": False,
                        "phase1a_query_level_reachability_gate": None,
                        "formal_success_gate": False,
                    },
                    "formal_success_gate": False,
                    "target_statistics": {
                        "relative_change": -0.9,
                        "improved_choice_views": 4,
                        "accuracy_delta": 1.0,
                    },
                }

            with (
                mock.patch.object(trainer, "parse_args", return_value=args),
                mock.patch.object(trainer, "_validate_args"),
                mock.patch.object(trainer, "_run", side_effect=fake_run),
            ):
                self.assertEqual(trainer.main(()), 1)
            terminal = trainer.json.loads((root / "terminal.json").read_text(encoding="utf-8"))
            self.assertEqual(terminal["status"], "failed_technical")
            self.assertFalse(terminal["technical_gate"])
            self.assertFalse(terminal["diagnostic_evaluated"])
            report = (root / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("未评估（技术门失败）", report)
            self.assertIn("| 科学成功 | `false` |", report)


if __name__ == "__main__":
    unittest.main()
