from __future__ import annotations

import importlib.util
import json
import subprocess
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


pixel = load_script(
    "r10_direct_pixel_under_test",
    "scripts/train/pixel_r10_visual_oracle.py",
)
dreamlite = load_script(
    "r10_dreamlite_under_test",
    "scripts/train/dreamlite_r10_single_set.py",
)
controller = load_script(
    "r10_alignment_controller_under_test",
    "scripts/inspire/run_r10_alignment_target.py",
)


def make_r9_parent(root: Path, *, pass_count: int) -> tuple[Path, list[dict[str, object]]]:
    targets: list[dict[str, object]] = []
    for index in range(8):
        targets.append(
            {
                "target_index": index,
                "target_family": controller.r9_compare.TARGETS[index][0],
                "target_segment_id": controller.r9_compare.TARGETS[index][1],
                "passed": index < pass_count,
                "source_root": str((root / f"target-{index:02d}").resolve()),
                "inventory_sha256": f"{index + 1:064x}",
            }
        )
    decision, reason = controller.r9_compare._decision(pass_count)
    comparison = {
        "schema": controller.R9_COMPARISON_SCHEMA,
        "status": "completed",
        "formal_success_claim": False,
        "decision": decision,
        "reason": reason,
        "pass_count": pass_count,
        "failed_target_indices": list(range(pass_count, 8)),
        "passed_target_indices": list(range(pass_count)),
        "git_commit": controller.EXPECTED_R9_TRAINING_COMMIT,
        "selected_segments_sha256": controller.r9_compare.SELECTED_SEGMENTS_SHA256,
        "targets": targets,
    }
    comparison_path = root / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    delivery = {
        "schema": controller.R9_DELIVERY_SCHEMA,
        "artifacts": [
            {
                "path": "comparison.json",
                "bytes": comparison_path.stat().st_size,
                "sha256": controller._sha256(comparison_path),
            }
        ],
        "source_inventory_sha256": {
            str(value["target_index"]): value["inventory_sha256"] for value in targets
        },
    }
    (root / "DELIVERY_MANIFEST.json").write_text(json.dumps(delivery), encoding="utf-8")
    return comparison_path, targets


class R10TrainingEntrypointsTest(unittest.TestCase):
    def test_direct_pixel_parser_freezes_preregistered_hyperparameters(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train/pixel_r10_visual_oracle.py"), "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--target-index {0,1,2,3,4,5,6,7}", result.stdout)
        for forbidden in ("--learning-rate", "--optimizer-steps", "--weight-decay", "--resolution"):
            self.assertNotIn(forbidden, result.stdout)
        args = pixel.parse_args(
            [
                "--target-index",
                "6",
                "--train",
                "train.jsonl",
                "--dev",
                "dev.jsonl",
                "--reader",
                "reader",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(args.target_index, 6)
        self.assertEqual(args.resolution, 1024)
        self.assertEqual(pixel.LEARNING_RATE, 0.05)
        self.assertEqual(pixel.CHECKPOINT_STEPS, (0, 32, 64, 96, 128))

    def test_pixel_oracle_has_exactly_one_trainable_logit_tensor(self):
        initial = torch.full((1, 3, 1024, 1024), 127 / 255, dtype=torch.float32)
        oracle = pixel.PixelOracle(initial)
        self.assertEqual([name for name, _ in oracle.named_parameters()], ["image_logits"])
        self.assertLessEqual(float((oracle.image().detach() - initial).abs().max()), 1e-7)
        self.assertTrue(bool(((oracle.image() > 0.0) & (oracle.image() < 1.0)).all()))

    def test_direct_pixel_technical_gate_requires_all_receipts(self):
        target = "target"
        metrics = []
        for step in range(1, 129):
            metrics.append(
                {
                    "kind": "optimizer_step",
                    "optimizer_step": step,
                    "target_segment_id": target,
                    "forward_cyclic_training_view": (step - 1) % 4,
                    "loss_before_step": 1.0,
                    "gradient_norm": 0.5,
                    "gradient_nonzero_fraction": 0.25,
                    "image_min_after_step": 0.1,
                    "image_max_after_step": 0.9,
                }
            )
        gate = pixel._technical_gate(
            metrics,
            target_segment_id=target,
            checkpoint_steps={0, 32, 64, 96, 128},
            png_steps={0, 32, 64, 96, 128},
            reader_frozen=True,
            snapshot_end_passed=True,
        )
        self.assertTrue(gate["passed"])
        metrics[-1]["gradient_norm"] = 0.0
        self.assertFalse(
            pixel._technical_gate(
                metrics,
                target_segment_id=target,
                checkpoint_steps={0, 32, 64, 96, 128},
                png_steps={0, 32, 64, 96, 128},
                reader_frozen=True,
                snapshot_end_passed=True,
            )["passed"]
        )

    def test_dreamlite_parser_freezes_full_target_gradient_contract(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/train/dreamlite_r10_single_set.py"), "--help"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for forbidden in ("--learning-rate", "--gradient-clip", "--lora-rank", "--optimizer-steps"):
            self.assertNotIn(forbidden, result.stdout)
        args = dreamlite.parse_args(
            [
                "--target-index",
                "2",
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
            ]
        )
        self.assertEqual(args.gradient_aggregation, "single-target-full")
        self.assertEqual(args.gradient_clip, 10.0)
        self.assertEqual(args.lora_rank, 4)
        self.assertEqual(args.max_optimizer_steps, 128)
        self.assertEqual(dreamlite.GRADIENT_COEFFICIENT, 1.0)

    def test_dreamlite_technical_gate_requires_exact_scaling_views_and_checkpoints(self):
        target = "target"
        metrics = []
        for step in range(128):
            metrics.append(
                {
                    "kind": "optimizer_step",
                    "gradient_aggregation": {
                        "mode": "single-target-full",
                        "target_segment_id": target,
                        "gradient_coefficient": 1.0,
                        "scale_relative_error": 0.0,
                        "raw_vs_applied_cosine": 1.0,
                        "forward_cyclic_training_view": step % 4,
                    },
                }
            )
        micro = [{"segment_id": target} for _ in range(128)]
        gate = dreamlite._technical_gate(
            metrics,
            micro,
            target_segment_id=target,
            training_gate={"passed": True},
            checkpoint_steps={0, 32, 64, 96, 128},
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["training_view_counts"], {0: 32, 1: 32, 2: 32, 3: 32})
        self.assertFalse(
            dreamlite._technical_gate(
                metrics,
                micro,
                target_segment_id=target,
                training_gate={"passed": True},
                checkpoint_steps={0, 32, 64, 96},
            )["passed"]
        )

    def test_controller_commands_do_not_expose_preregistered_hyperparameters(self):
        base = SimpleNamespace(
            target_index=4,
            train=Path("train.jsonl"),
            dev=Path("dev.jsonl"),
            reader=Path("reader"),
            dreamlite=Path("dreamlite"),
            seed=0,
            reader_device="cuda:1",
            dreamlite_device="cuda:0",
        )
        pixel_args = SimpleNamespace(**vars(base), arm="direct-pixel-oracle")
        pixel_command = controller._command(pixel_args, Path("run"))
        self.assertNotIn("--dreamlite", pixel_command)
        dreamlite_args = SimpleNamespace(**vars(base), arm="dreamlite-single-set")
        dreamlite_command = controller._command(dreamlite_args, Path("run"))
        self.assertIn("--dreamlite", dreamlite_command)
        for command in (pixel_command, dreamlite_command):
            self.assertIn("--strict-determinism", command)
            for forbidden in ("--learning-rate", "--optimizer-steps", "--gradient-clip", "--lora-rank"):
                self.assertNotIn(forbidden, command)

    def test_controller_revalidates_complete_r9_and_rejects_inactive_eight_of_eight(self):
        with tempfile.TemporaryDirectory() as directory:
            comparison_path, targets = make_r9_parent(Path(directory), pass_count=0)
            with mock.patch.object(
                controller.r9_compare,
                "_validate_target",
                side_effect=lambda _root, index: dict(targets[index]),
            ):
                observed = controller._validate_parent(comparison_path)
            self.assertEqual(observed["pass_count"], 0)
        with tempfile.TemporaryDirectory() as directory:
            comparison_path, targets = make_r9_parent(Path(directory), pass_count=8)
            with mock.patch.object(
                controller.r9_compare,
                "_validate_target",
                side_effect=lambda _root, index: dict(targets[index]),
            ), self.assertRaisesRegex(ValueError, "inactive"):
                controller._validate_parent(comparison_path)

    def test_controller_rejects_tampered_r9_delivery_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comparison_path, targets = make_r9_parent(root, pass_count=0)
            delivery_path = root / "DELIVERY_MANIFEST.json"
            delivery = json.loads(delivery_path.read_text(encoding="utf-8"))
            delivery["artifacts"][0]["sha256"] = "0" * 64
            delivery_path.write_text(json.dumps(delivery), encoding="utf-8")
            with mock.patch.object(
                controller.r9_compare,
                "_validate_target",
                side_effect=lambda _root, index: dict(targets[index]),
            ), self.assertRaisesRegex(ValueError, "size/SHA"):
                controller._validate_parent(comparison_path)


if __name__ == "__main__":
    unittest.main()
