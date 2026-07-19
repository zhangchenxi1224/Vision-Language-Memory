from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.train import dreamlite_episode  # noqa: E402


class DreamLiteTrainingContractTest(unittest.TestCase):
    def test_formal_cli_defaults_to_listwise_cyclic_qa_only(self):
        argv = [
            "dreamlite_episode.py",
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
        with mock.patch.object(sys, "argv", argv):
            args = dreamlite_episode.parse_args()

        self.assertEqual(args.reader_loss_mode, "listwise-choice")
        self.assertEqual(args.choice_view_schedule, "cyclic4")
        self.assertEqual(args.training_regime, "qa_only")
        self.assertIsNone(args.teacher_manifest)
        self.assertFalse(args.audit_optimizer_diagnostics)
        self.assertNotIn(
            "audit_optimizer_diagnostics",
            dreamlite_episode.manifest_compatibility_args(args),
        )
        args.audit_optimizer_diagnostics = True
        self.assertTrue(
            dreamlite_episode.manifest_compatibility_args(args)["audit_optimizer_diagnostics"]
        )

    def test_optimizer_diagnostic_groups_are_exhaustive_and_norms_partition_each_axis(self):
        named_tensors = [
            (
                "updater.pipeline.unet.base_model.model.down_blocks.0.attentions.0.to_q."
                "lora_A.default.weight",
                torch.tensor([3.0, 4.0]),
            ),
            (
                "updater.pipeline.unet.base_model.model.down_blocks.0.attentions.0.to_k."
                "lora_B.default.weight",
                torch.tensor([12.0]),
            ),
            (
                "updater.pipeline.unet.base_model.model.mid_block.attentions.0.to_v."
                "lora_A.default.weight",
                torch.tensor([5.0]),
            ),
            (
                "updater.pipeline.unet.base_model.model.up_blocks.0.attentions.0.to_out.0."
                "lora_B.default.weight",
                torch.tensor([8.0, 15.0]),
            ),
            ("initial_state", torch.tensor([7.0, 24.0])),
        ]

        summary = dreamlite_episode.grouped_tensor_norms(named_tensors)
        global_squared = summary["global"]["norm"] ** 2

        self.assertEqual(summary["global"]["tensor_count"], len(named_tensors))
        self.assertEqual(summary["global"]["missing_tensor_count"], 0)
        self.assertEqual(
            set(summary["by_stage"]),
            {"down_blocks", "mid_block", "up_blocks", "other"},
        )
        self.assertEqual(set(summary["by_projection"]), {"to_q", "to_k", "to_v", "to_out", "other"})
        self.assertEqual(set(summary["by_factor"]), {"lora_A", "lora_B", "initial_state"})
        self.assertEqual(len(summary["by_cross"]), len(named_tensors))
        for axis in dreamlite_episode._OPTIMIZER_DIAGNOSTIC_AXES:
            groups = summary[f"by_{axis}"]
            self.assertEqual(sum(item["tensor_count"] for item in groups.values()), len(named_tensors))
            self.assertTrue(
                math.isclose(
                    sum(item["norm"] ** 2 for item in groups.values()),
                    global_squared,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            )
            self.assertTrue(
                math.isclose(
                    sum(item["squared_norm_share"] for item in groups.values()),
                    1.0,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            )
        self.assertIsInstance(json.dumps(summary, allow_nan=False, sort_keys=True), str)

    def test_optimizer_diagnostics_record_actual_clip_and_update_ratios(self):
        down = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
        up = torch.nn.Parameter(torch.tensor([0.0, 12.0]))
        down.grad = torch.tensor([3.0, 4.0])
        up.grad = torch.tensor([0.0, 12.0])
        named_parameters = [
            (
                "updater.pipeline.unet.base_model.model.down_blocks.0.attentions.0.to_q."
                "lora_A.default.weight",
                down,
            ),
            (
                "updater.pipeline.unet.base_model.model.up_blocks.0.attentions.0.to_v."
                "lora_B.default.weight",
                up,
            ),
        ]
        optimizer = torch.optim.SGD([down, up], lr=0.1)

        snapshot, report = dreamlite_episode.begin_optimizer_diagnostics(named_parameters)
        gradient_norm = torch.nn.utils.clip_grad_norm_([down, up], 6.5)
        dreamlite_episode.record_optimizer_diagnostics_after_clip(
            report,
            named_parameters,
            gradient_norm=gradient_norm,
            max_norm=6.5,
        )
        optimizer.step()
        report = dreamlite_episode.finalize_optimizer_diagnostics_after_step(
            report,
            named_parameters,
            snapshot,
        )

        self.assertAlmostEqual(report["parameter_norms_before_step"]["global"]["norm"], 13.0, places=6)
        self.assertAlmostEqual(report["gradient_norms_before_clip"]["global"]["norm"], 13.0, places=6)
        self.assertAlmostEqual(report["gradient_clipping"]["pre_clip_norm_returned"], 13.0, places=6)
        self.assertTrue(report["gradient_clipping"]["pre_clip_norm_diagnostic_matches"])
        self.assertAlmostEqual(
            report["gradient_clipping"]["clipping_factor"],
            6.5 / (13.0 + 1e-6),
            places=7,
        )
        self.assertTrue(report["gradient_clipping"]["clipped"])
        self.assertAlmostEqual(report["gradient_norms_after_clip"]["global"]["norm"], 6.5, places=5)
        self.assertAlmostEqual(report["updates_after_step"]["global"]["norm"], 0.65, places=5)
        self.assertAlmostEqual(report["updates_after_step"]["global"]["update_weight_ratio"], 0.05, places=5)
        self.assertEqual(
            set(report["updates_after_step"]["by_stage"]),
            set(report["parameter_norms_before_step"]["by_stage"]),
        )
        self.assertIsInstance(json.dumps(report, allow_nan=False, sort_keys=True), str)

        with self.assertRaisesRegex(ValueError, "finite and positive"):
            dreamlite_episode.record_optimizer_diagnostics_after_clip(
                report,
                named_parameters,
                gradient_norm=gradient_norm,
                max_norm=float("inf"),
            )

    def test_optimizer_diagnostics_use_low_precision_norm_tolerance(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16))
        parameter.grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        named_parameters = [
            (
                "updater.pipeline.unet.base_model.model.mid_block.attentions.0.to_q."
                "lora_A.default.weight",
                parameter,
            )
        ]

        _snapshot, report = dreamlite_episode.begin_optimizer_diagnostics(named_parameters)
        gradient_norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0)
        dreamlite_episode.record_optimizer_diagnostics_after_clip(
            report,
            named_parameters,
            gradient_norm=gradient_norm,
            max_norm=1.0,
        )

        self.assertEqual(report["gradient_clipping"]["pre_clip_norm_comparison_rel_tol"], 5e-3)
        self.assertTrue(report["gradient_clipping"]["pre_clip_norm_diagnostic_matches"])

    def test_cyclic_rotation_synchronizes_target_index(self):
        choices = ("a", "b", "c", "d")
        rotated, target_index = dreamlite_episode.rotate_choice_view(choices, 1, rotation=3)

        self.assertEqual(rotated, ("d", "a", "b", "c"))
        self.assertEqual(target_index, 2)
        self.assertEqual(rotated[target_index], "b")

    def test_reverse_cyclic_eval_permutation_preserves_semantic_target(self):
        choices = ("a", "b", "c", "d")
        view = dreamlite_episode.choice_view_for_permutation("cyclic4", (3, 2, 1, 0))
        self.assertIsNotNone(view)
        permuted, target_index = view("episode", 1, choices, 1)
        self.assertEqual(permuted, ("d", "c", "b", "a"))
        self.assertEqual(target_index, 2)
        self.assertEqual(permuted[target_index], "b")

    def test_cyclic_training_schedule_advances_per_episode_exposure(self):
        observed = [
            dreamlite_episode.choice_rotation_for_training(
                "cyclic4",
                epoch=epoch,
                position=position,
                episodes_per_epoch=3,
            )
            for epoch in range(2)
            for position in range(3)
        ]

        self.assertEqual(observed, [0, 1, 2, 3, 0, 1])
        self.assertEqual(
            dreamlite_episode.choice_rotation_for_training(
                "canonical",
                epoch=9,
                position=2,
                episodes_per_epoch=3,
            ),
            0,
        )

        keyed = [
            dreamlite_episode.choice_rotation_for_training(
                "cyclic4",
                epoch=epoch,
                position=(epoch * 3) % 7,
                episodes_per_epoch=8,
                schedule_key="comparison:entity-slot",
            )
            for epoch in range(8)
        ]
        self.assertEqual(keyed[:4], keyed[4:])
        self.assertEqual(set(keyed[:4]), {0, 1, 2, 3})

    def test_training_lineage_is_fail_closed_and_hashes_teacher_manifest(self):
        qa_args = argparse.Namespace(
            training_regime="qa_only",
            objective_stage="qa",
            reader_loss_mode="listwise-choice",
            choice_view_schedule="cyclic4",
            teacher_manifest=None,
            teacher_sidecar=None,
            teacher_calibration=None,
            initialize_from=None,
            epochs=2,
            presentations_per_state=512,
            distill_presentations=0,
            qa_presentations=512,
        )
        qa_lineage = dreamlite_episode.training_lineage(qa_args)
        self.assertIsNone(qa_lineage["teacher_manifest_sha256"])
        self.assertTrue(qa_lineage["teacher_checkpoint_is_qa_only_eligible"])

        with tempfile.TemporaryDirectory() as directory:
            teacher_manifest = Path(directory) / "teacher.json"
            teacher_manifest.write_text('{"teacher":"fixed"}\n', encoding="utf-8")
            teacher_sidecar = Path(directory) / "sidecar.jsonl"
            teacher_sidecar.write_text('{"transition":"fixed"}\n', encoding="utf-8")
            teacher_calibration = Path(directory) / "calibration.json"
            teacher_calibration.write_text('{"scales":"fixed"}\n', encoding="utf-8")
            teacher_args = argparse.Namespace(
                training_regime="teacher_assisted",
                objective_stage="distill",
                reader_loss_mode="listwise-choice",
                choice_view_schedule="cyclic4",
                teacher_manifest=teacher_manifest,
                teacher_sidecar=teacher_sidecar,
                teacher_calibration=teacher_calibration,
                initialize_from=None,
                epochs=256,
                presentations_per_state=256,
                distill_presentations=256,
                qa_presentations=0,
            )
            with mock.patch.object(
                dreamlite_episode,
                "teacher_control_contract",
                return_value=("c" * 64, {"state": "state"}),
            ):
                teacher_lineage = dreamlite_episode.training_lineage(teacher_args)

            self.assertEqual(
                teacher_lineage["teacher_manifest_sha256"],
                dreamlite_episode.sha256_file(teacher_manifest),
            )
            self.assertFalse(teacher_lineage["teacher_checkpoint_is_qa_only_eligible"])
            self.assertEqual(teacher_lineage["objective_stage"], "distill")
            self.assertEqual(teacher_lineage["distill_presentations"], 256)

            qa_args.teacher_manifest = teacher_manifest
            with self.assertRaisesRegex(ValueError, "qa_only training forbids"):
                dreamlite_episode.training_lineage(qa_args)

        teacher_args.teacher_manifest = None
        with self.assertRaisesRegex(ValueError, "--teacher-manifest is required"):
            dreamlite_episode.training_lineage(teacher_args)

    def test_real_manifest_places_report_schemas_at_top_level(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            dev = root / "dev.jsonl"
            train.write_text('{"episode_id":"train"}\n', encoding="utf-8")
            dev.write_text('{"episode_id":"dev"}\n', encoding="utf-8")
            dreamlite = root / "dreamlite"
            reader = root / "reader"
            dreamlite.mkdir()
            reader.mkdir()
            argv = [
                "dreamlite_episode.py",
                "--train",
                str(train),
                "--dev",
                str(dev),
                "--dreamlite",
                str(dreamlite),
                "--reader",
                str(reader),
                "--output-dir",
                str(root / "output"),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(dreamlite_episode, "git_value", side_effect=["a" * 40, ""]),
                mock.patch.object(dreamlite_episode, "locked_revision", return_value="b" * 40),
                mock.patch.object(
                    dreamlite_episode,
                    "load_initial_image",
                    return_value=(torch.zeros(1, 3, 1, 1), {"initial_state_mode": "blank"}),
                ),
            ):
                manifest = dreamlite_episode.make_manifest(dreamlite_episode.parse_args())

        self.assertEqual(manifest["metrics_schema"], dreamlite_episode.DREAMLITE_METRICS_SCHEMA)
        self.assertEqual(manifest["summary_schema"], dreamlite_episode.DREAMLITE_SUMMARY_SCHEMA)
        self.assertNotIn("metrics_schema", manifest["training_lineage"])
        self.assertNotIn("summary_schema", manifest["training_lineage"])

    def test_parent_checkpoint_lineage_requires_locked_reader_resize_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "parent.pt"
            payload = {
                "schema_version": 1,
                "manifest": {
                    "training_lineage": {"training_regime": "teacher_assisted"},
                },
            }
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "Reader resize contract"):
                dreamlite_episode._checkpoint_lineage(checkpoint)

            payload["manifest"]["reader_resize_contract"] = "incompatible"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "Reader resize contract"):
                dreamlite_episode._checkpoint_lineage(checkpoint)

            payload["manifest"]["reader_resize_contract"] = dreamlite_episode.R3_QWEN_READER_RESIZE_CONTRACT
            torch.save(payload, checkpoint)
            lineage, digest = dreamlite_episode._checkpoint_lineage(checkpoint)
            self.assertEqual(lineage, payload["manifest"]["training_lineage"])
            self.assertEqual(digest, dreamlite_episode.sha256_file(checkpoint))

    def test_random_teacher_control_is_deterministic_and_preserves_channel_moments(self):
        image = torch.rand(1, 3, 8, 8)
        latent = torch.randn(1, 4, 4, 4)
        feature = torch.randn(1, 7, 6)
        teacher = dreamlite_episode.TeacherState(
            state_id="1" * 64,
            teacher_key="2" * 64,
            semantic_state_sha256="3" * 64,
            teacher_contract_sha256="4" * 64,
            renderer_contract_sha256="5" * 64,
            image=torch.nn.functional.interpolate(image, size=(1024, 1024), mode="nearest"),
            latent=latent,
            feature=feature,
        )
        first = dreamlite_episode.random_moment_matched_teacher(teacher)
        second = dreamlite_episode.random_moment_matched_teacher(teacher)
        torch.testing.assert_close(first.image, second.image, rtol=0, atol=0)
        torch.testing.assert_close(first.latent, second.latent, rtol=0, atol=0)
        torch.testing.assert_close(first.feature, second.feature, rtol=0, atol=0)
        for source, controlled in ((teacher.image, first.image), (teacher.latent, first.latent)):
            torch.testing.assert_close(
                source.mean(dim=tuple(range(2, source.ndim))), controlled.mean(dim=tuple(range(2, source.ndim)))
            )
            torch.testing.assert_close(
                source.var(dim=tuple(range(2, source.ndim)), unbiased=False),
                controlled.var(dim=tuple(range(2, source.ndim)), unbiased=False),
            )
        torch.testing.assert_close(teacher.feature.mean(dim=1), first.feature.mean(dim=1))
        torch.testing.assert_close(teacher.feature.var(dim=1, unbiased=False), first.feature.var(dim=1, unbiased=False))

    def test_listwise_callable_forwards_only_choice_boundary_supervision(self):
        expected = object()
        reader = object()
        processor = object()
        image = torch.rand(1, 3, 8, 8, requires_grad=True)
        audit_tensors: list[tuple[str, torch.Tensor]] = []
        with mock.patch.object(
            dreamlite_episode,
            "qwen3vl_listwise_choice_ce",
            return_value=expected,
        ) as scorer:
            callable_reader = dreamlite_episode.choice_reader_callable(
                reader=reader,
                processor=processor,
                reader_device=torch.device("cpu"),
                require_grad=True,
                gradient_audit_tensors=audit_tensors,
            )
            result = callable_reader(image, "formatted query", ("a", "b", "c", "d"), 2)

        self.assertIs(result, expected)
        kwargs = scorer.call_args.kwargs
        self.assertIs(kwargs["model"], reader)
        self.assertIs(kwargs["processor"], processor)
        self.assertEqual(tuple(kwargs["image"].shape), (3, 8, 8))
        self.assertEqual(kwargs["choices"], ("a", "b", "c", "d"))
        self.assertEqual(kwargs["target_index"], 2)
        self.assertTrue(kwargs["require_image_grad"])
        self.assertEqual(
            kwargs["reader_resize_contract"],
            dreamlite_episode.R3_QWEN_READER_RESIZE_CONTRACT,
        )
        self.assertEqual([category for category, _tensor in audit_tensors], ["query_image"])
        self.assertIs(audit_tensors[0][1], kwargs["image"])

    def test_target_callable_propagates_locked_reader_resize_contract(self):
        expected = object()
        reader = object()
        processor = object()
        image = torch.rand(1, 3, 8, 8, requires_grad=True)
        audit_tensors: list[tuple[str, torch.Tensor]] = []
        with mock.patch.object(
            dreamlite_episode,
            "qwen3vl_target_only_ce",
            return_value=expected,
        ) as scorer:
            callable_reader = dreamlite_episode.target_reader_callable(
                reader=reader,
                processor=processor,
                reader_device=torch.device("cpu"),
                require_grad=True,
                gradient_audit_tensors=audit_tensors,
            )
            result = callable_reader(image, "formatted query", "answer")

        self.assertIs(result, expected)
        kwargs = scorer.call_args.kwargs
        self.assertIs(kwargs["model"], reader)
        self.assertIs(kwargs["processor"], processor)
        self.assertEqual(tuple(kwargs["image"].shape), (3, 8, 8))
        self.assertEqual(kwargs["target"], "answer")
        self.assertTrue(kwargs["require_image_grad"])
        self.assertEqual(
            kwargs["reader_resize_contract"],
            dreamlite_episode.R3_QWEN_READER_RESIZE_CONTRACT,
        )
        self.assertEqual([category for category, _tensor in audit_tensors], ["query_image"])
        self.assertIs(audit_tensors[0][1], kwargs["image"])

    def test_state_gradient_audit_passes_connected_qa_graph_and_records_exact_counts(self):
        source = torch.tensor([1.0], requires_grad=True)
        first_state = source * 2.0
        final_state = first_state * 3.0
        query_image = final_state * 4.0
        tensors: list[tuple[str, torch.Tensor]] = []
        dreamlite_episode.retain_gradient_audit_tensor(
            tensors,
            category="query_image",
            tensor=query_image,
        )
        dreamlite_episode.retain_gradient_audit_tensor(
            tensors,
            category="final_state",
            tensor=final_state,
        )
        dreamlite_episode.retain_gradient_audit_tensor(
            tensors,
            category="first_intermediate_state",
            tensor=first_state,
        )
        query_image.square().sum().backward()

        accumulator: dict[str, list[float]] = {}
        dreamlite_episode.audit_episode_gradients(tensors, accumulator)
        expected = Counter(category for category, _tensor in tensors)
        contract = dreamlite_episode.state_gradient_audit_contract(
            argparse.Namespace(audit_state_gradients=True, objective_stage="qa")
        )
        summary = dreamlite_episode.gradient_audit_summary(
            accumulator,
            expected,
            contract=contract,
            multi_update_episode_count=1,
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(
            set(summary["required_categories"]),
            {"query_image", "final_state", "first_intermediate_state"},
        )
        self.assertTrue(all(item["expected"] == item["observed"] == 1 for item in summary["categories"].values()))

    def test_state_gradient_audit_fails_missing_zero_and_distill_feature_evidence(self):
        disconnected_source = torch.tensor([1.0], requires_grad=True)
        disconnected = disconnected_source * 2.0
        disconnected.retain_grad()
        (disconnected_source * 3.0).sum().backward()
        with self.assertRaisesRegex(RuntimeError, "no gradient"):
            dreamlite_episode.audit_episode_gradients([("query_image", disconnected)], {})

        zero_source = torch.tensor([1.0], requires_grad=True)
        zero = zero_source * 2.0
        zero.retain_grad()
        (zero * 0.0).sum().backward()
        with self.assertRaisesRegex(RuntimeError, "non-positive"):
            dreamlite_episode.audit_episode_gradients([("query_image", zero)], {})

        contract = dreamlite_episode.state_gradient_audit_contract(
            argparse.Namespace(audit_state_gradients=True, objective_stage="distill")
        )
        missing_feature = dreamlite_episode.gradient_audit_summary(
            {"final_state": [1.0], "state_image": [1.0]},
            Counter({"final_state": 1, "state_image": 1}),
            contract=contract,
            multi_update_episode_count=0,
        )
        self.assertFalse(missing_feature["passed"])
        self.assertIn("student_visual_feature", missing_feature["required_categories"])

        with self.assertRaisesRegex(RuntimeError, "does not require gradients"):
            dreamlite_episode.retain_gradient_audit_tensor(
                [],
                category="final_state",
                tensor=torch.tensor([1.0]),
            )

    def test_teacher_distill_audit_requires_and_accepts_student_visual_feature_gradient(self):
        source = torch.tensor([1.0], requires_grad=True)
        final_state = source * 2.0
        state_image = final_state * 3.0
        student_feature = state_image * 4.0
        tensors: list[tuple[str, torch.Tensor]] = []
        for category, tensor in (
            ("final_state", final_state),
            ("state_image", state_image),
            ("student_visual_feature", student_feature),
        ):
            dreamlite_episode.retain_gradient_audit_tensor(
                tensors,
                category=category,
                tensor=tensor,
            )
        (final_state.square() + state_image.square() + student_feature.square()).sum().backward()
        accumulator: dict[str, list[float]] = {}
        dreamlite_episode.audit_episode_gradients(tensors, accumulator)
        contract = dreamlite_episode.state_gradient_audit_contract(
            argparse.Namespace(audit_state_gradients=True, objective_stage="distill")
        )
        summary = dreamlite_episode.gradient_audit_summary(
            accumulator,
            Counter(category for category, _tensor in tensors),
            contract=contract,
            multi_update_episode_count=0,
        )

        self.assertTrue(summary["passed"])
        self.assertEqual(
            set(summary["required_categories"]),
            {"final_state", "state_image", "student_visual_feature"},
        )


if __name__ == "__main__":
    unittest.main()
