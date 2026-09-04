from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    path = ROOT / relative
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


controller = load_script(
    "r11_new_phase1a_controller_under_test",
    "scripts/inspire/run_r11_new_phase1a_target.py",
)


class R11NewPhase1AControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.train = self.root / "train.jsonl"
        self.dev = self.root / "dev.jsonl"
        self.comparison = self.root / "comparison.json"
        self.dreamlite = self.root / "dreamlite"
        self.reader = self.root / "reader"
        self.output = self.root / "output"
        self.train.write_text("train\n", encoding="utf-8")
        self.dev.write_text("dev\n", encoding="utf-8")
        self.comparison.write_text(
            json.dumps(
                {
                    "schema": controller.CANONICAL_R11_SCHEMA,
                    "status": controller.CANONICAL_R11_STATUS,
                    "target_pass_count": 8,
                    "decision": controller.CANONICAL_R11_DECISION,
                }
            ),
            encoding="utf-8",
        )
        for model_dir, env_name in (
            (self.dreamlite, "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256"),
            (self.reader, "VLM_READER_SNAPSHOT_MANIFEST_SHA256"),
        ):
            model_dir.mkdir()
            manifest = model_dir / ".snapshot_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            expected = controller.EXPECTED_ENVIRONMENT[env_name]
            (model_dir / ".snapshot_manifest.json.sha256").write_text(
                f"{expected}  .snapshot_manifest.json\n",
                encoding="utf-8",
            )
        self.commit = "a" * 40
        self.args = SimpleNamespace(
            mode="technical-preflight",
            target_index=0,
            r11_comparison=self.comparison,
            train=self.train,
            dev=self.dev,
            dreamlite=self.dreamlite,
            reader=self.reader,
            output_root=self.output,
            expected_commit=self.commit,
            preflight_terminal=None,
            target0_formal_terminal=None,
            dreamlite_device="cuda:0",
            reader_device="cuda:1",
        )
        self.real_sha256 = controller._sha256

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _git(self, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return self.commit
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    def _locked_sha(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved == self.train.resolve():
            return controller.core.R11_NEW_TRAIN_SHA256
        if resolved == self.dev.resolve():
            return controller.core.R11_NEW_DEV_SHA256
        if resolved == self.comparison.resolve():
            return controller.core.R11_NEW_PARENT_R11_COMPARISON_SHA256
        if resolved == (self.dreamlite / ".snapshot_manifest.json").resolve():
            return controller.EXPECTED_ENVIRONMENT[
                "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256"
            ]
        if resolved == (self.reader / ".snapshot_manifest.json").resolve():
            return controller.EXPECTED_ENVIRONMENT["VLM_READER_SNAPSHOT_MANIFEST_SHA256"]
        return self.real_sha256(path)

    def _validation_context(self):
        return (
            mock.patch.object(controller, "_git", side_effect=self._git),
            mock.patch.object(controller, "_sha256", side_effect=self._locked_sha),
            mock.patch.object(
                controller,
                "_deployment_audit",
                return_value={"passed": True, "disk_free_bytes": 100 * 1024**3},
            ),
            mock.patch.dict(os.environ, controller.EXPECTED_ENVIRONMENT, clear=False),
        )

    def _validate(self):
        contexts = self._validation_context()
        with contexts[0], contexts[1], contexts[2], contexts[3]:
            return controller._validate(self.args)

    def _prior_terminal(self, *, mode: str, target_index: int) -> Path:
        root = self.root / f"prior-{mode}-{target_index}"
        root.mkdir()
        terminal_path = root / "terminal.json"
        diagnostic = (
            {
                "evaluated": False,
                "phase1a_query_level_reachability_gate": None,
                "result": "not_evaluated_in_technical_preflight",
            }
            if mode == "technical-preflight"
            else {
                "evaluated": True,
                "phase1a_query_level_reachability_gate": False,
                "result": "query_level_reachability_not_found",
            }
        )
        terminal = {
            "schema": controller.TERMINAL_SCHEMA,
            "status": "technical_completed",
            "technical_completed": True,
            "diagnostic_result": diagnostic,
            "formal_success": False,
            "scientific_success_claim": False,
            "mode": mode,
            "target_index": target_index,
            "target_segment_id": controller.core.R11_NEW_TARGET_IDS[target_index],
            "git_commit": self.commit,
            "child_exit_code": 0,
            "execution_checks": {"synthetic_complete": True},
            "config_sha256": self.real_sha256(controller.CONFIG_PATH),
            "trainer_sha256": self.real_sha256(controller.TRAINER),
        }
        terminal_path.write_text(json.dumps(terminal), encoding="utf-8")
        inventory = {
            "schema": controller.INVENTORY_SCHEMA,
            "root": str(root.resolve()),
            "artifacts": [
                {
                    "path": "terminal.json",
                    "bytes": terminal_path.stat().st_size,
                    "sha256": self.real_sha256(terminal_path),
                }
            ],
        }
        (root / "artifact_inventory.json").write_text(
            json.dumps(inventory), encoding="utf-8"
        )
        return terminal_path

    def test_parser_and_command_do_not_expose_locked_hyperparameters(self):
        parser = controller.build_parser()
        help_text = parser.format_help()
        for forbidden in controller._FORBIDDEN_COMMAND_OPTIONS:
            self.assertNotIn(forbidden, help_text)
        parsed = parser.parse_args(
            (
                "--mode",
                "formal",
                "--target-index",
                "0",
                "--r11-comparison",
                str(self.comparison),
                "--train",
                str(self.train),
                "--dev",
                str(self.dev),
                "--dreamlite",
                str(self.dreamlite),
                "--reader",
                str(self.reader),
                "--output-root",
                str(self.output),
                "--expected-commit",
                self.commit,
            )
        )
        command = controller._command(parsed, self.output / "run")
        self.assertTrue(controller._FORBIDDEN_COMMAND_OPTIONS.isdisjoint(command))
        self.assertIn("--strict-determinism", command)

    def test_validation_records_locked_config_trainer_parent_and_snapshots(self):
        result = self._validate()
        self.assertEqual(result["git_commit"], self.commit)
        self.assertTrue(result["config_validation"]["passed"])
        self.assertEqual(
            result["canonical_r11_comparison_sha256"],
            controller.core.R11_NEW_PARENT_R11_COMPARISON_SHA256,
        )
        self.assertEqual(result["target_segment_id"], controller.core.R11_NEW_TARGET_IDS[0])
        self.assertIn("config_sha256", result)
        self.assertIn("trainer_sha256", result)
        self.assertTrue(result["storage_audit"]["passed"])
        self.assertTrue(result["prerequisites"]["passed"])
        self.assertFalse(result["prerequisites"]["preflight_required"])

    def test_formal_prerequisites_enforce_preflight_then_target0_order(self):
        formal = SimpleNamespace(**{**vars(self.args), "mode": "formal"})
        contexts = self._validation_context()
        with contexts[0], contexts[1], contexts[2], contexts[3]:
            with self.assertRaisesRegex(ValueError, "prior technical-preflight"):
                controller._validate(formal)

        preflight = self._prior_terminal(mode="technical-preflight", target_index=0)
        formal.preflight_terminal = preflight
        contexts = self._validation_context()
        with contexts[0], contexts[1], contexts[2], contexts[3]:
            result = controller._validate(formal)
        self.assertTrue(result["prerequisites"]["preflight"]["passed"])
        self.assertFalse(result["prerequisites"]["target0_formal_required"])

        target_one = SimpleNamespace(
            **{
                **vars(formal),
                "target_index": 1,
                "target0_formal_terminal": None,
            }
        )
        contexts = self._validation_context()
        with contexts[0], contexts[1], contexts[2], contexts[3]:
            with self.assertRaisesRegex(ValueError, "prior formal target-00"):
                controller._validate(target_one)
        target_one.target0_formal_terminal = self._prior_terminal(mode="formal", target_index=0)
        contexts = self._validation_context()
        with contexts[0], contexts[1], contexts[2], contexts[3]:
            result = controller._validate(target_one)
        self.assertTrue(result["prerequisites"]["target0_formal"]["passed"])

    def test_parent_content_tamper_fails_even_if_hash_check_is_mocked(self):
        value = json.loads(self.comparison.read_text(encoding="utf-8"))
        value["target_pass_count"] = 7
        self.comparison.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content drifted"):
            self._validate()

    def test_commit_dirty_fresh_root_and_environment_fail_closed(self):
        self.args.expected_commit = "short"
        with self.assertRaisesRegex(ValueError, "40-character"):
            self._validate()
        self.args.expected_commit = self.commit

        def mismatched_git(*args: str) -> str:
            return "b" * 40 if args == ("rev-parse", "HEAD") else ""

        contexts = self._validation_context()
        with contexts[1], contexts[2], contexts[3], mock.patch.object(
            controller, "_git", side_effect=mismatched_git
        ):
            with self.assertRaisesRegex(ValueError, "commit mismatch"):
                controller._validate(self.args)

        def dirty_git(*args: str) -> str:
            return self.commit if args == ("rev-parse", "HEAD") else " M tracked.py"

        contexts = self._validation_context()
        with contexts[1], contexts[2], contexts[3], mock.patch.object(
            controller, "_git", side_effect=dirty_git
        ):
            with self.assertRaisesRegex(ValueError, "clean experiment snapshot"):
                controller._validate(self.args)

        self.output.mkdir()
        (self.output / "old.txt").write_text("occupied", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-empty output root"):
            self._validate()
        (self.output / "old.txt").unlink()
        self.output.rmdir()

        contexts = self._validation_context()
        bad_environment = dict(controller.EXPECTED_ENVIRONMENT)
        bad_environment["HF_HUB_OFFLINE"] = "0"
        with contexts[0], contexts[1], contexts[2], mock.patch.dict(
            os.environ, bad_environment, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "environment drift"):
                controller._validate(self.args)

    def test_same_device_and_data_drift_fail_closed(self):
        self.args.reader_device = "cuda"
        with self.assertRaisesRegex(ValueError, "devices must be distinct"):
            self._validate()
        self.args.reader_device = "cuda:1"

        def bad_data_sha(path: Path) -> str:
            if path.resolve() == self.train.resolve():
                return "0" * 64
            return self._locked_sha(path)

        contexts = self._validation_context()
        with contexts[0], contexts[2], contexts[3], mock.patch.object(
            controller, "_sha256", side_effect=bad_data_sha
        ):
            with self.assertRaisesRegex(ValueError, "data SHA mismatch"):
                controller._validate(self.args)

    def test_deployment_audit_requires_instance_ssd_and_free_space(self):
        ssd = self.root / "inspire" / "ssd"
        ssd.mkdir(parents=True)
        output = ssd / "runs" / "new-run"
        usage = SimpleNamespace(total=200 * 1024**3, used=100 * 1024**3, free=100 * 1024**3)
        with mock.patch.object(controller.shutil, "disk_usage", return_value=usage):
            audit = controller._deployment_audit(
                output,
                ssd_root=ssd,
                hostname=controller.EXPECTED_HOST_PREFIX + "-0",
            )
        self.assertTrue(audit["passed"])
        with self.assertRaisesRegex(ValueError, "pinned"):
            controller._deployment_audit(output, ssd_root=ssd, hostname="wrong-instance")
        low = SimpleNamespace(total=200 * 1024**3, used=190 * 1024**3, free=10 * 1024**3)
        with mock.patch.object(controller.shutil, "disk_usage", return_value=low):
            with self.assertRaisesRegex(ValueError, "free-space"):
                controller._deployment_audit(
                    output,
                    ssd_root=ssd,
                    hostname=controller.EXPECTED_HOST_PREFIX,
                )
        outside = self.root / "hdd" / "run"
        with self.assertRaisesRegex(ValueError, "must be under"):
            controller._deployment_audit(
                outside,
                ssd_root=ssd,
                hostname=controller.EXPECTED_HOST_PREFIX,
            )

    def test_validation_failure_still_writes_fail_closed_terminal_and_inventory(self):
        argv = (
            "--mode",
            "formal",
            "--target-index",
            "0",
            "--r11-comparison",
            str(self.comparison),
            "--train",
            str(self.train),
            "--dev",
            str(self.dev),
            "--dreamlite",
            str(self.dreamlite),
            "--reader",
            str(self.reader),
            "--output-root",
            str(self.output),
            "--expected-commit",
            self.commit,
        )
        with mock.patch.object(controller, "_validate", side_effect=ValueError("locked failure")):
            self.assertEqual(controller.main(argv), 1)
        for name in (
            "launch.json",
            "stdout.log",
            "stderr.log",
            "terminal.json",
            "artifact_inventory.json",
        ):
            self.assertTrue((self.output / name).is_file(), name)
        terminal = json.loads((self.output / "terminal.json").read_text(encoding="utf-8"))
        self.assertEqual(terminal["status"], "failed")
        self.assertFalse(terminal["technical_completed"])
        self.assertFalse(terminal["formal_success"])
        self.assertFalse(terminal["diagnostic_result"]["evaluated"])

    def test_second_process_same_mode_and_target_fails_closed_across_fresh_roots(self):
        lock_root = self.root / "tmp-locks"
        first_output = self.root / "first-run"
        first_output.mkdir()
        sentinel = first_output / "preserve.txt"
        sentinel.write_text("first run remains untouched", encoding="utf-8")
        first_args = SimpleNamespace(**{**vars(self.args), "output_root": first_output})
        held = controller._acquire_suite_lock(first_args, lock_root=lock_root)
        owner_path = Path(held["metadata_path"])
        original_owner = owner_path.read_bytes()
        second_output = self.root / "second-fresh-run"
        argv = (
            "--mode",
            "formal",
            "--target-index",
            "0",
            "--r11-comparison",
            str(self.comparison),
            "--train",
            str(self.train),
            "--dev",
            str(self.dev),
            "--dreamlite",
            str(self.dreamlite),
            "--reader",
            str(self.reader),
            "--output-root",
            str(second_output),
            "--expected-commit",
            self.commit,
        )
        validated = {
            "config_sha256": "config-sha",
            "trainer_sha256": "trainer-sha",
        }
        try:
            with (
                mock.patch.object(controller, "LOCK_ROOT", lock_root),
                mock.patch.object(controller, "_validate", return_value=validated),
                mock.patch.object(controller.subprocess, "run") as run,
            ):
                self.assertEqual(controller.main(argv), 1)
                run.assert_not_called()
            self.assertEqual(owner_path.read_bytes(), original_owner)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "first run remains untouched")
            terminal = json.loads(
                (second_output / "terminal.json").read_text(encoding="utf-8")
            )
            self.assertIn("duplicate-process lock", terminal["error"])
            self.assertFalse(terminal["technical_completed"])
            self.assertTrue((second_output / "artifact_inventory.json").is_file())
        finally:
            controller._release_suite_lock(held)

    def test_suite_lock_serializes_different_modes_and_targets(self):
        lock_root = self.root / "mode-locks"
        preflight = SimpleNamespace(**{**vars(self.args), "mode": "technical-preflight"})
        formal = SimpleNamespace(**{**vars(self.args), "mode": "formal"})
        first = controller._acquire_suite_lock(preflight, lock_root=lock_root)
        try:
            with self.assertRaisesRegex(ValueError, "duplicate-process lock"):
                controller._acquire_suite_lock(formal, lock_root=lock_root)
            other_target = SimpleNamespace(**{**vars(formal), "target_index": 1})
            with self.assertRaisesRegex(ValueError, "duplicate-process lock"):
                controller._acquire_suite_lock(other_target, lock_root=lock_root)
        finally:
            controller._release_suite_lock(first)
        second = controller._acquire_suite_lock(formal, lock_root=lock_root)
        self.assertEqual(second["owner"]["lock_key"], "r11-new-phase1a-suite")
        controller._release_suite_lock(second)

    def test_outer_inventory_binds_nested_trainer_inventory(self):
        root = self.root / "inventory-root"
        nested = root / "run"
        nested.mkdir(parents=True)
        (nested / "artifact_inventory.json").write_text(
            '{"schema":"trainer-inventory"}\n', encoding="utf-8"
        )
        (root / "terminal.json").write_text('{"status":"done"}\n', encoding="utf-8")
        controller._write_inventory(root)
        inventory = json.loads((root / "artifact_inventory.json").read_text(encoding="utf-8"))
        paths = {row["path"] for row in inventory["artifacts"]}
        self.assertIn("run/artifact_inventory.json", paths)
        self.assertIn("terminal.json", paths)
        self.assertNotIn("artifact_inventory.json", paths)

    def _manifest(self, mode: str) -> dict:
        return {
            "schema": controller.MANIFEST_SCHEMA,
            "mode": mode,
            "git_commit": self.commit,
            "git_dirty": False,
            "target_index": 0,
            "target_segment_id": controller.core.R11_NEW_TARGET_IDS[0],
            "selected_segment_ids": list(controller.core.R11_NEW_TARGET_IDS),
            "selected_segments_sha256": controller.core.R11_NEW_TARGETS_PAYLOAD_SHA256,
            "train_sha256": controller.core.R11_NEW_TRAIN_SHA256,
            "dev_sha256": controller.core.R11_NEW_DEV_SHA256,
            "preregistered_config_sha256": "config-sha",
            "information_boundary": {"passed": True},
            "model_snapshot_payloads_start": {
                "dreamlite_mobile": {
                    "manifest_sha256": controller.EXPECTED_ENVIRONMENT[
                        "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256"
                    ]
                },
                "qwen_reader": {
                    "manifest_sha256": controller.EXPECTED_ENVIRONMENT[
                        "VLM_READER_SNAPSHOT_MANIFEST_SHA256"
                    ]
                },
            },
        }

    def _summary_base(self, mode: str) -> dict:
        return {
            "schema": controller.SUMMARY_SCHEMA,
            "mode": mode,
            "git_commit": self.commit,
            "target_index": 0,
            "target_segment_id": controller.core.R11_NEW_TARGET_IDS[0],
            "query_level_diagnostic_only": True,
            "formal_success_gate": False,
            "full_success_claim_allowed": False,
        }

    def _write_trainer_inventory(self, run: Path) -> None:
        inventory_path = run / "artifact_inventory.json"
        inventory_path.unlink(missing_ok=True)
        artifacts = controller._inventory(run)
        inventory_path.write_text(
            json.dumps(
                {
                    "schema": controller.TRAINER_INVENTORY_SCHEMA,
                    "artifact_count": len(artifacts),
                    "artifacts": artifacts,
                }
            ),
            encoding="utf-8",
        )

    def _materialize_preflight_run(self, run: Path) -> tuple[dict, dict]:
        run.mkdir()
        (run / ".r11_new_output_owner.json").write_text(
            json.dumps(
                {
                    "schema": controller.TRAINER_OUTPUT_OWNER_SCHEMA,
                    "pid": 123,
                    "created_at_utc": "2026-09-05T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )
        condition_path = run / "condition" / "official_full_condition.pt"
        condition_path.parent.mkdir()
        prompt_embeds = torch.ones((1, 2, 3))
        attention_mask = torch.ones((1, 2), dtype=torch.long)
        condition_tensor_hashes = {
            "prompt_embeds": controller.canonical_tensor_sha256(prompt_embeds),
            "attention_mask": controller.canonical_tensor_sha256(attention_mask),
        }
        event_text_sha256 = "3" * 64
        torch.save(
            {
                "schema": "vision_memory.r11-new-phase1a-condition.v1",
                "prompt_embeds": prompt_embeds,
                "attention_mask": attention_mask,
                "tensor_sha256": condition_tensor_hashes,
                "event_text_sha256": event_text_sha256,
                "recompute_matches": True,
            },
            condition_path,
        )
        condition_record = {
            "path": str(condition_path),
            "sha256": controller._sha256(condition_path),
            "bytes": condition_path.stat().st_size,
            "tensor_sha256": condition_tensor_hashes,
            "event_text_sha256": event_text_sha256,
            "recompute_matches": True,
        }
        manifest = self._manifest("technical-preflight")
        manifest.update(
            {
                "protocol": controller.core.R11_NEW_PROTOCOL,
                "implementation_revision": "full-frozen-dreamlite-x-t-fp32-v1",
                "source_contract_verified": True,
                "event_noise_contract_verified": True,
                "query_level_diagnostic_only": True,
                "formal_success_gate": False,
                "strict_determinism": {
                    "seed": controller.core.R11_NEW_SEED,
                    "environment": {
                        key: controller.EXPECTED_ENVIRONMENT[key]
                        for key in (
                            "PYTHONHASHSEED",
                            "CUBLAS_WORKSPACE_CONFIG",
                            "OMP_NUM_THREADS",
                            "MKL_NUM_THREADS",
                            "TOKENIZERS_PARALLELISM",
                        )
                    },
                    "deterministic_algorithms": True,
                    "deterministic_warn_only": False,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                    "float32_matmul_precision": "highest",
                    "sdpa": {
                        "flash": False,
                        "memory_efficient": False,
                        "cudnn": False,
                        "math": True,
                    },
                },
                "condition_artifact": condition_record,
                "information_boundary": {
                    "schema": "vision_memory.r11-new-phase1a-information-boundary.v1",
                    "passed": True,
                    "dreamlite_inputs": list(controller.core.R11_NEW_DREAMLITE_INPUTS),
                    "noise_key": list(controller.core.R11_NEW_NOISE_KEY),
                    "reader_loss_inputs": list(controller.core.R11_NEW_READER_LOSS_INPUTS),
                    "leaked_fields": [],
                    "conditioner_input_names": ["source_latent", "event_text"],
                    "sampler_input_names": list(controller.core.R11_NEW_DREAMLITE_INPUTS),
                    "oracle_initialization_key_names": list(controller.core.R11_NEW_NOISE_KEY),
                    "event_source_episode_id_sha256": "4" * 64,
                    "event_source_turn_id": 0,
                    "query_used_only_by_frozen_reader_loss": True,
                    "choices_used_only_by_frozen_reader_loss": True,
                    "target_index_used_only_by_frozen_reader_loss": True,
                    "forbidden_writer_key_violations": [],
                },
                "fixed_contract": {
                    "resolution": 1024,
                    "only_trainable": "x_T_fp32",
                    "dreamlite_pipeline_load": (
                        "DreamLiteMobilePipeline.from_pretrained directly; no PEFT"
                    ),
                    "unet_frozen": True,
                    "vae_frozen": True,
                    "text_encoder_frozen": True,
                    "reader_frozen": True,
                    "dreamlite_unet_executed": True,
                    "official_full_conditioner_precomputed": True,
                    "gradient_mode": "full",
                    "checkpoint_unet": True,
                    "num_denoising_steps": controller.core.R11_NEW_DIFFUSION_STEPS,
                    "edit_start_sigma": controller.core.R11_NEW_EFFECTIVE_START_SIGMA,
                    "effective_sigmas": list(
                        controller.core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE
                    ),
                    "optimizer": "Adam",
                    "learning_rate": controller.core.R11_NEW_LEARNING_RATE,
                    "weight_decay": controller.core.R11_NEW_WEIGHT_DECAY,
                    "optimizer_steps": 0,
                    "preflight_backward_calls": 1,
                    "gradient_clipping": None,
                    "checkpoint_steps": [0],
                    "training_views": "forward cyclic",
                    "endpoint_views": "reverse cyclic",
                    "primary_endpoint": controller.core.R11_NEW_PRIMARY_ENDPOINT,
                    "best_checkpoint_selection_forbidden": True,
                    "reset": "decode(blank_source_latent)",
                },
            }
        )
        (run / controller.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")

        x_t = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
        z_t = x_t.add(1.0)
        trajectory = tuple(x_t.add(float(index)) for index in range(5))
        tensor_hashes = {
            "x_T_fp32": controller.canonical_tensor_sha256(x_t),
            "z_t_fp32": controller.canonical_tensor_sha256(z_t),
            "trajectory_fp32": [
                controller.canonical_tensor_sha256(value) for value in trajectory
            ],
        }
        checkpoint_path = run / "checkpoints" / "step-000.pt"
        checkpoint_path.parent.mkdir()
        torch.save(
            {
                "schema": controller.TRAINER_CHECKPOINT_SCHEMA,
                "optimizer_step": 0,
                "x_T_fp32": x_t,
                "z_t_fp32": z_t,
                "trajectory_fp32": trajectory,
                "effective_sigmas": list(
                    controller.core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE
                ),
                "optimizer": {"state": {}, "param_groups": []},
                "manifest_sha256": controller._sha256(run / controller.MANIFEST_FILE),
                "condition_artifact_sha256": condition_record["sha256"],
                "tensor_sha256": tensor_hashes,
            },
            checkpoint_path,
        )
        png_path = run / "images" / "step-000.png"
        png_path.parent.mkdir()
        png_path.write_bytes(b"synthetic-png")
        checkpoint_record = {
            "schema": controller.CHECKPOINT_HASH_SCHEMA,
            "optimizer_step": 0,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": controller._sha256(checkpoint_path),
            "png_path": str(png_path),
            "png_bytes": png_path.stat().st_size,
            "png_sha256": controller._sha256(png_path),
            "trajectory_points": controller.core.R11_NEW_DIFFUSION_STEPS + 1,
            "effective_sigmas": list(controller.core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE),
            "tensor_sha256": tensor_hashes,
        }
        checkpoint_record_path = run / "checkpoint_hashes" / "step-000.json"
        checkpoint_record_path.parent.mkdir()
        checkpoint_record_path.write_text(json.dumps(checkpoint_record), encoding="utf-8")

        summary = {
            "schema": controller.SUMMARY_SCHEMA,
            "status": "completed_technical",
            "mode": "technical-preflight",
            "passed": True,
            "backward_calls": 1,
            "optimizer_steps": 0,
            "loss": 1.25,
            "gradient_norm": 0.5,
            "gradient_nonzero_fraction": 0.75,
            "trajectory_points": controller.core.R11_NEW_DIFFUSION_STEPS + 1,
            "dreamlite_denoising_steps": controller.core.R11_NEW_DIFFUSION_STEPS,
            "effective_sigmas": list(controller.core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE),
            "trainable_parameter_names": ["x_T_fp32"],
            "all_models_frozen": True,
            "snapshots_unchanged": True,
            "condition_artifact_valid": True,
            "checkpoint": checkpoint_record,
            "scientific_gate_evaluated": False,
            "phase1a_reachability_gate": None,
            "gates": {
                "technical_gate": True,
                "phase1a_query_level_reachability_gate": None,
                "formal_success_gate": False,
            },
            "formal_success_gate": False,
            "query_level_diagnostic_only": True,
        }
        encoded_summary = json.dumps(summary)
        (run / controller.SUMMARY_FILE).write_text(encoded_summary, encoding="utf-8")
        (run / controller.PREFLIGHT_FILE).write_text(encoded_summary, encoding="utf-8")
        (run / "model_snapshot_verification_end.json").write_text(
            json.dumps(
                {
                    "schema": "vision_memory.r11-new-phase1a-model-snapshot-end.v1",
                    "passed": True,
                    "bindings": manifest["model_snapshot_payloads_start"],
                }
            ),
            encoding="utf-8",
        )
        (run / "terminal.json").write_text(
            json.dumps(
                {
                    "schema": controller.TRAINER_TERMINAL_SCHEMA,
                    "status": "succeeded",
                    "mode": "technical-preflight",
                    "technical_gate": True,
                    "diagnostic_evaluated": False,
                    "formal_success_gate": False,
                }
            ),
            encoding="utf-8",
        )
        (run / "environment.txt").write_text("torch==fixture\n", encoding="utf-8")
        (run / "runtime.json").write_text(
            json.dumps(
                {
                    "python": "fixture-python",
                    "platform": "fixture-platform",
                    "packages": {"torch": "fixture"},
                    "cuda_available": True,
                    "torch_cuda": "fixture-cuda",
                    "gpu_names": ["fixture-gpu-0", "fixture-gpu-1"],
                }
            ),
            encoding="utf-8",
        )
        (run / "model_snapshot_verification_start.json").write_text(
            json.dumps(
                {
                    "schema": "vision_memory.r11-new-phase1a-model-snapshot-start.v1",
                    "bindings": manifest["model_snapshot_payloads_start"],
                }
            ),
            encoding="utf-8",
        )
        (run / "REPORT.md").write_text("# valid preflight\n", encoding="utf-8")
        self._write_trainer_inventory(run)
        return summary, manifest

    def test_preflight_completion_has_no_scientific_result(self):
        run = self.root / "preflight"
        self._materialize_preflight_run(run)
        checks, diagnostic = controller._assess_run(
            args=SimpleNamespace(**{**vars(self.args), "mode": "technical-preflight"}),
            validated={"config_sha256": "config-sha"},
            run_dir=run,
            child_exit_code=0,
        )
        self.assertTrue(all(checks.values()), checks)
        self.assertFalse(diagnostic["evaluated"])

    def test_preflight_summary_and_manifest_tampering_fail_independent_checks(self):
        summary_mutations = {
            "zero-gradient": ("gradient_norm", 0.0),
            "nan-gradient": ("gradient_norm", float("nan")),
            "optimizer-step": ("optimizer_steps", 1),
            "second-backward": ("backward_calls", 2),
            "wrong-trajectory": ("trajectory_points", 4),
            "wrong-sigmas": ("effective_sigmas", [0.5, 0.3, 0.2, 0.1]),
            "wrong-trainable": ("trainable_parameter_names", ["x_T_fp32", "unet"]),
            "reader-not-frozen": ("all_models_frozen", False),
        }
        for name, (field, value) in summary_mutations.items():
            with self.subTest(name=name):
                run = self.root / f"preflight-summary-{name}"
                summary, _manifest = self._materialize_preflight_run(run)
                summary[field] = value
                encoded = json.dumps(summary)
                (run / controller.SUMMARY_FILE).write_text(encoded, encoding="utf-8")
                (run / controller.PREFLIGHT_FILE).write_text(encoded, encoding="utf-8")
                self._write_trainer_inventory(run)
                checks, diagnostic = controller._assess_run(
                    args=SimpleNamespace(
                        **{**vars(self.args), "mode": "technical-preflight"}
                    ),
                    validated={"config_sha256": "config-sha"},
                    run_dir=run,
                    child_exit_code=0,
                )
                self.assertFalse(checks["preflight_summary_technical_fields"], checks)
                self.assertFalse(all(checks.values()))
                self.assertFalse(diagnostic["evaluated"])
                self.assertIsNone(diagnostic["phase1a_query_level_reachability_gate"])

        manifest_mutations = {
            "source-contract": lambda value: value.update(
                {"source_contract_verified": False}
            ),
            "event-noise-contract": lambda value: value.update(
                {"event_noise_contract_verified": False}
            ),
            "optimizer-count": lambda value: value["fixed_contract"].update(
                {"optimizer_steps": 1}
            ),
            "information-leak": lambda value: value["information_boundary"].update(
                {"query_used_only_by_frozen_reader_loss": False}
            ),
        }
        for name, mutate in manifest_mutations.items():
            with self.subTest(name=name):
                run = self.root / f"preflight-manifest-{name}"
                _summary, manifest = self._materialize_preflight_run(run)
                mutate(manifest)
                (run / controller.MANIFEST_FILE).write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                self._write_trainer_inventory(run)
                checks, diagnostic = controller._assess_run(
                    args=SimpleNamespace(
                        **{**vars(self.args), "mode": "technical-preflight"}
                    ),
                    validated={"config_sha256": "config-sha"},
                    run_dir=run,
                    child_exit_code=0,
                )
                self.assertFalse(checks["preflight_manifest_contract"], checks)
                self.assertFalse(all(checks.values()))
                self.assertFalse(diagnostic["evaluated"])
                self.assertIsNone(diagnostic["phase1a_query_level_reachability_gate"])

    def test_preflight_missing_tampered_or_formal_artifacts_fail_closed(self):
        cases = {
            "missing-checkpoint": lambda run: (run / "checkpoints" / "step-000.pt").unlink(),
            "tampered-condition": lambda run: (run / "condition" / "official_full_condition.pt").write_bytes(
                b"tampered"
            ),
            "snapshot-failed": lambda run: (run / "model_snapshot_verification_end.json").write_text(
                json.dumps({"schema": "vision_memory.r11-new-phase1a-model-snapshot-end.v1", "passed": False, "bindings": {}}),
                encoding="utf-8",
            ),
            "trainer-terminal-failed": lambda run: (run / "terminal.json").write_text(
                json.dumps(
                    {
                        "schema": controller.TRAINER_TERMINAL_SCHEMA,
                        "status": "failed",
                        "mode": "technical-preflight",
                        "technical_gate": False,
                        "diagnostic_evaluated": False,
                        "formal_success_gate": False,
                    }
                ),
                encoding="utf-8",
            ),
            "summary-twin-mismatch": lambda run: (run / controller.PREFLIGHT_FILE).write_text(
                json.dumps({"schema": controller.SUMMARY_SCHEMA, "passed": False}),
                encoding="utf-8",
            ),
            "unexpected-metrics": lambda run: (run / controller.METRICS_FILE).write_text(
                "{}\n", encoding="utf-8"
            ),
            "runtime-without-cuda": lambda run: (run / "runtime.json").write_text(
                json.dumps(
                    {
                        "python": "fixture-python",
                        "platform": "fixture-platform",
                        "packages": {"torch": "fixture"},
                        "cuda_available": False,
                        "torch_cuda": "fixture-cuda",
                        "gpu_names": [],
                    }
                ),
                encoding="utf-8",
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                run = self.root / f"preflight-artifact-{name}"
                self._materialize_preflight_run(run)
                mutate(run)
                self._write_trainer_inventory(run)
                checks, diagnostic = controller._assess_run(
                    args=SimpleNamespace(
                        **{**vars(self.args), "mode": "technical-preflight"}
                    ),
                    validated={"config_sha256": "config-sha"},
                    run_dir=run,
                    child_exit_code=0,
                )
                self.assertFalse(all(checks.values()), checks)
                self.assertFalse(diagnostic["evaluated"])
                self.assertIsNone(diagnostic["phase1a_query_level_reachability_gate"])

    def test_preflight_evidence_failure_returns_nonzero_and_no_diagnostic(self):
        output = self.root / "controller-preflight-failure"
        argv = (
            "--mode",
            "technical-preflight",
            "--target-index",
            "0",
            "--r11-comparison",
            str(self.comparison),
            "--train",
            str(self.train),
            "--dev",
            str(self.dev),
            "--dreamlite",
            str(self.dreamlite),
            "--reader",
            str(self.reader),
            "--output-root",
            str(output),
            "--expected-commit",
            self.commit,
        )
        validated = {
            "git_commit": self.commit,
            "config_sha256": "config-sha",
            "trainer_sha256": "trainer-sha",
        }

        def child_run(*_args, **_kwargs):
            run = output / "run"
            summary, _manifest = self._materialize_preflight_run(run)
            summary["gradient_norm"] = 0.0
            encoded = json.dumps(summary)
            (run / controller.SUMMARY_FILE).write_text(encoded, encoding="utf-8")
            (run / controller.PREFLIGHT_FILE).write_text(encoded, encoding="utf-8")
            self._write_trainer_inventory(run)
            return SimpleNamespace(returncode=0)

        held_lock = {"path": "synthetic", "metadata_path": "synthetic", "owner": {}}
        released_lock = {**held_lock, "released": True}
        with (
            mock.patch.object(controller, "_validate", return_value=validated),
            mock.patch.object(controller, "_acquire_suite_lock", return_value=held_lock),
            mock.patch.object(controller, "_release_suite_lock", return_value=released_lock),
            mock.patch.object(controller.subprocess, "run", side_effect=child_run),
        ):
            self.assertEqual(controller.main(argv), 1)
        terminal = json.loads((output / "terminal.json").read_text(encoding="utf-8"))
        self.assertEqual(terminal["status"], "failed")
        self.assertFalse(terminal["technical_completed"])
        self.assertFalse(terminal["diagnostic_result"]["evaluated"])
        self.assertIsNone(
            terminal["diagnostic_result"]["phase1a_query_level_reachability_gate"]
        )

    def test_formal_scientific_failure_is_controller_technical_success(self):
        run = self.root / "formal"
        run.mkdir()
        (run / controller.MANIFEST_FILE).write_text(
            json.dumps(self._manifest("formal")), encoding="utf-8"
        )
        target_id = controller.core.R11_NEW_TARGET_IDS[0]
        schedule = controller.core.build_phase1a_schedule(0)
        receipts = [
            {
                "schema": controller.METRICS_SCHEMA,
                "kind": "optimizer_step",
                "optimizer_step": unit.optimizer_step,
                "target_segment_id": target_id,
                "forward_cyclic_training_view": unit.forward_cyclic_training_view,
                "permutation": list(unit.permutation),
                "loss_before_step": 1.0,
                "gradient_norm": 0.5,
                "gradient_nonzero_fraction": 1.0,
                "full_dreamlite_forward_executed": True,
                "denoiser_steps_executed": 4,
                "effective_sigma_schedule": list(
                    controller.core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE
                ),
            }
            for unit in schedule
        ]
        with (run / controller.METRICS_FILE).open("w", encoding="utf-8") as handle:
            for receipt in receipts:
                handle.write(json.dumps(receipt) + "\n")
        for name in ("checkpoints", "images", "checkpoint_hashes"):
            (run / name).mkdir()
        for step in controller.core.R11_NEW_CHECKPOINT_STEPS:
            checkpoint = run / "checkpoints" / f"step-{step:03d}.pt"
            image = run / "images" / f"step-{step:03d}.png"
            checkpoint.write_bytes(f"checkpoint-{step}".encode())
            image.write_bytes(f"image-{step}".encode())
            record = {
                "schema": controller.CHECKPOINT_HASH_SCHEMA,
                "optimizer_step": step,
                "checkpoint_path": str(checkpoint),
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": controller._sha256(checkpoint),
                "png_path": str(image),
                "png_bytes": image.stat().st_size,
                "png_sha256": controller._sha256(image),
                "trajectory_points": controller.core.R11_NEW_DIFFUSION_STEPS + 1,
                "effective_sigmas": list(controller.core.R11_NEW_EFFECTIVE_SIGMA_SCHEDULE),
            }
            (run / "checkpoint_hashes" / f"step-{step:03d}.json").write_text(
                json.dumps(record), encoding="utf-8"
            )
        core_audit = {
            "checkpoint_steps_observed": list(controller.core.R11_NEW_CHECKPOINT_STEPS),
            "latent_checkpoint_steps_observed": list(controller.core.R11_NEW_CHECKPOINT_STEPS),
            "image_checkpoint_steps_observed": list(controller.core.R11_NEW_CHECKPOINT_STEPS),
            "trainable_parameter_names": ["x_T_fp32"],
            "trainable_parameter_dtypes": {"x_T_fp32": "torch.float32"},
            "frozen_components": {
                "dreamlite_unet": True,
                "condition_encoder": True,
                "vae": True,
                "reader": True,
            },
            "frozen_gradients_absent": True,
            "full_model_snapshots_unchanged": True,
            "information_boundary_passed": True,
            "source_contract_verified": True,
            "event_noise_contract_verified": True,
            "optimizer": "Adam",
            "learning_rate": 0.05,
            "weight_decay": 0.0,
            "gradient_clip": None,
            "primary_endpoint": controller.core.R11_NEW_PRIMARY_ENDPOINT,
        }
        technical = {
            **controller.core.phase1a_technical_gate(
                receipts,
                target_segment_id=target_id,
                audit=core_audit,
            ),
            "schema": controller.TECHNICAL_GATE_SCHEMA,
            "core_audit": core_audit,
        }
        (run / controller.TECHNICAL_GATE_FILE).write_text(
            json.dumps(technical), encoding="utf-8"
        )
        evaluation_rows = []
        for checkpoint in ("m0", controller.core.R11_NEW_PRIMARY_ENDPOINT):
            for condition in ("normal", "reset"):
                for view, permutation in enumerate(controller.REVERSE_CYCLIC4):
                    ordered_target = permutation.index(0)
                    ordered_incorrect = permutation.index(1)
                    logits = [-2.0, -2.0, -2.0, -2.0]
                    logits[ordered_target] = 0.0
                    logits[ordered_incorrect] = 1.0
                    expected_ce = math.log(sum(math.exp(value) for value in logits))
                    evaluation_rows.append(
                        {
                            "schema": controller.EVALUATION_SCHEMA,
                            "suite": controller.EVALUATION_SUITE,
                            "checkpoint": checkpoint,
                            "condition": condition,
                            "item_id": target_id,
                            "pair_unit": target_id,
                            "family": "F1",
                            "view_index": view,
                            "permutation": list(permutation),
                            "target_index": 0,
                            "predicted_index": 1,
                            "correct": False,
                            "ce": expected_ce,
                            "margin": -1.0,
                            "choice_logits_ordered": logits,
                        }
                    )
        evaluation_path = run / controller.EVALUATION_ROWS_FILE
        evaluation_path.write_text(
            "".join(json.dumps(row) + "\n" for row in evaluation_rows),
            encoding="utf-8",
        )
        (run / "endpoint_raw.pt").write_bytes(b"endpoint")
        (run / "endpoint_raw.png").write_bytes(b"png")
        (run / "model_snapshot_verification_end.json").write_text(
            json.dumps(
                {
                    "schema": "vision_memory.r11-new-phase1a-model-snapshot-end.v1",
                    "passed": True,
                }
            ),
            encoding="utf-8",
        )
        statistics = controller.core.phase1a_target_statistics(
            evaluation_rows,
            suite=controller.EVALUATION_SUITE,
            target_segment_id=target_id,
            endpoint=controller.core.R11_NEW_PRIMARY_ENDPOINT,
        )
        summary = self._summary_base("formal")
        summary.update(
            {
                "status": "completed",
                "optimizer_steps": controller.core.R11_NEW_OPTIMIZER_STEPS,
                "checkpoint_steps_observed": list(controller.core.R11_NEW_CHECKPOINT_STEPS),
                "technical_gate": technical,
                "gates": {
                    "technical_gate": True,
                    "phase1a_query_level_reachability_gate": False,
                    "formal_success_gate": False,
                },
                "target_statistics": statistics,
            }
        )
        summary["artifacts"] = {
            "manifest_sha256": controller._sha256(run / controller.MANIFEST_FILE),
            "metrics_sha256": controller._sha256(run / controller.METRICS_FILE),
            "evaluation_rows_sha256": controller._sha256(evaluation_path),
            "endpoint_raw_sha256": controller._sha256(run / "endpoint_raw.pt"),
            "endpoint_png_sha256": controller._sha256(run / "endpoint_raw.png"),
            "snapshot_end_sha256": controller._sha256(
                run / "model_snapshot_verification_end.json"
            ),
            "technical_gate_sha256": controller._sha256(
                run / controller.TECHNICAL_GATE_FILE
            ),
        }
        (run / controller.SUMMARY_FILE).write_text(json.dumps(summary), encoding="utf-8")
        checks, diagnostic = controller._assess_run(
            args=SimpleNamespace(**{**vars(self.args), "mode": "formal"}),
            validated={"config_sha256": "config-sha"},
            run_dir=run,
            child_exit_code=0,
        )
        self.assertTrue(all(checks.values()), checks)
        self.assertFalse(diagnostic["phase1a_query_level_reachability_gate"])
        self.assertEqual(diagnostic["result"], "query_level_reachability_not_found")

        metrics_path = run / controller.METRICS_FILE
        original_metrics = metrics_path.read_text(encoding="utf-8")
        invalid_receipts = [dict(row) for row in receipts]
        invalid_receipts[0]["gradient_norm"] = 0.0
        metrics_path.write_text(
            "".join(json.dumps(row) + "\n" for row in invalid_receipts),
            encoding="utf-8",
        )
        invalid_receipt_checks, invalid_receipt_diagnostic = controller._assess_run(
            args=SimpleNamespace(**{**vars(self.args), "mode": "formal"}),
            validated={"config_sha256": "config-sha"},
            run_dir=run,
            child_exit_code=0,
        )
        self.assertFalse(invalid_receipt_checks["formal_raw_technical_gate_recomputed"])
        self.assertFalse(invalid_receipt_diagnostic["evaluated"])
        metrics_path.write_text(original_metrics, encoding="utf-8")

        checkpoint = run / "checkpoints" / "step-256.pt"
        checkpoint_bytes = checkpoint.read_bytes()
        checkpoint.unlink()
        missing_checkpoint_checks, invalid_diagnostic = controller._assess_run(
            args=SimpleNamespace(**{**vars(self.args), "mode": "formal"}),
            validated={"config_sha256": "config-sha"},
            run_dir=run,
            child_exit_code=0,
        )
        self.assertFalse(missing_checkpoint_checks["formal_checkpoints_exact"])
        self.assertFalse(invalid_diagnostic["evaluated"])
        self.assertIsNone(invalid_diagnostic["phase1a_query_level_reachability_gate"])
        self.assertEqual(
            invalid_diagnostic["result"], "not_evaluated_due_to_technical_failure"
        )
        checkpoint.write_bytes(checkpoint_bytes)

        original_rows = evaluation_path.read_text(encoding="utf-8")
        evaluation_path.write_text(
            "\n".join(original_rows.splitlines()[:-1]) + "\n",
            encoding="utf-8",
        )
        truncated_checks, truncated_diagnostic = controller._assess_run(
            args=SimpleNamespace(**{**vars(self.args), "mode": "formal"}),
            validated={"config_sha256": "config-sha"},
            run_dir=run,
            child_exit_code=0,
        )
        self.assertFalse(truncated_checks["formal_evaluation_rows_exact"])
        self.assertFalse(all(truncated_checks.values()))
        self.assertFalse(truncated_diagnostic["evaluated"])

        tampered_rows = [dict(row) for row in evaluation_rows]
        tampered_rows[0]["ce"] = 2.0
        evaluation_path.write_text(
            "".join(json.dumps(row) + "\n" for row in tampered_rows),
            encoding="utf-8",
        )
        tampered_checks, _ = controller._assess_run(
            args=SimpleNamespace(**{**vars(self.args), "mode": "formal"}),
            validated={"config_sha256": "config-sha"},
            run_dir=run,
            child_exit_code=0,
        )
        self.assertFalse(tampered_checks["formal_statistics_recomputed"])
        self.assertFalse(tampered_checks["formal_artifact_hashes_bound"])
        self.assertFalse(all(tampered_checks.values()))


if __name__ == "__main__":
    unittest.main()
