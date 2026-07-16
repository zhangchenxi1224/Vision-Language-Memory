from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.repro import (  # noqa: E402
    DETERMINISTIC_FIXTURE_RGB_SHA256_1024,
    assert_no_frozen_parameter_grads,
    assert_determinism_environment,
    canonical_json_sha256,
    canonical_object_sha256,
    canonical_tensor_sha256,
    compare_bitwise_repro_reports,
    load_initial_image,
    load_source_image,
    lora_trainable_parameters,
    model_optimizer_rng_manifest,
    named_tensors_manifest,
    seed_adapter_initialization,
    validate_e2e_pair_reports,
)
from scripts.probes.lightweight_determinism import (  # noqa: E402
    ALLOWED_STEP_COUNTS,
    episode_schedule,
    evaluation_views_for_reader_loss_mode,
    grouped_prediction_summary,
    listwise_gradient_trace_summary,
    listwise_query_trace_records,
    normalized_categorical_label,
    r2a_autograd_diagnostic,
    r2_gate_summary,
    reader_objective_contract,
    reachability_gate_summary,
    rotate_choices_left_one,
    validate_qwen_image_grid_contract,
)


def make_pair_report(*, detached: bool) -> dict:
    metadata = {
        "events": ["remember red", "the room has a table"],
        "adapter_seed": 11,
        "event_noise_seeds": [19, 20],
        "source_image": {"rgb_sha256": "abc"},
    }
    return {
        "events": 2,
        "detach_between_events": detached,
        "pair_id": canonical_json_sha256(metadata),
        "pair_metadata": metadata,
        "loss": 1.25,
        "intermediate_gradients": [{"norm": None if detached else 0.75, "nonfinite_elements": None if detached else 0}],
        "lora_grad_norm": 2.0,
        "unclamped_image_grad_norm": 3.0,
    }


class ReproProbeContractTest(unittest.TestCase):
    @staticmethod
    def r2_predictions(*, view: str = "canonical") -> list[dict]:
        predictions: list[dict] = []
        for pair_index in range(64):
            original_target = pair_index % 4
            target_index = original_target if view == "canonical" else (original_target - 1) % 4
            correct = pair_index >= 6
            predicted_index = target_index if correct else (target_index + 1) % 4
            canonical_choices = [f"pair-{pair_index:02d}-choice-{index}" for index in range(4)]
            choices = canonical_choices if view == "canonical" else canonical_choices[1:] + canonical_choices[:1]
            target_text = choices[target_index]
            predicted_text = choices[predicted_index]
            for variant in ("clean", "distractor"):
                predictions.append(
                    {
                        "episode_id": f"episode-{pair_index:02d}-{variant}",
                        "turn_id": 3,
                        "query_ordinal": 0,
                        "view": view,
                        "correct": correct,
                        "target_index": target_index,
                        "predicted_index": predicted_index,
                        "choices": list(choices),
                        "target_text": target_text,
                        "predicted_text": predicted_text,
                        "comparison_id": f"comparison-{pair_index:02d}",
                        "event_kind": "noop" if variant == "distractor" else "set",
                        "state_event_kind": "overwrite" if pair_index % 2 else "set",
                        "distractor_variant": variant,
                        "turn_type": "mixed" if pair_index >= 52 else "query",
                        "topic": "style",
                    }
                )
        return predictions

    @staticmethod
    def r2_trace(steps: int = 2000) -> list[dict]:
        return [
            {
                "optimizer_step": step,
                "gradient_norm_before_clip_float_hex": float(1.0).hex(),
                "listwise_queries": [
                    {
                        "all_values_finite": True,
                        "image_gradient_norm_float_hex": float(1.0).hex(),
                    }
                ],
            }
            for step in range(1, steps + 1)
        ]

    def test_categorical_metadata_uses_enum_value(self):
        self.assertEqual(normalized_categorical_label(SimpleNamespace(value="clean")), "clean")
        self.assertEqual(normalized_categorical_label("distractor"), "distractor")
        self.assertIsNone(normalized_categorical_label(None))

    def test_reachability_step_budget_and_schedule_are_locked(self):
        schedule = episode_schedule(64, 2000)

        self.assertEqual(ALLOWED_STEP_COUNTS, (1, 100, 2000))
        self.assertEqual(len(schedule), 2000)
        self.assertEqual(schedule[0], (0, 29))
        self.assertEqual(schedule[-1][0], 31)
        self.assertEqual(len(schedule[1984:]), 16)
        schedule_records = [
            {
                "optimizer_step": optimizer_step,
                "epoch": epoch,
                "episode_index": episode_index,
                "episode_id": f"train-{episode_index:07d}",
            }
            for optimizer_step, (epoch, episode_index) in enumerate(schedule, start=1)
        ]
        self.assertEqual(
            canonical_object_sha256(schedule_records),
            "a4c9bfb6e108ac6de97fc04c18a530245ae339a1250f54312205369d0be49dcc",
        )

    def test_reachability_gate_requires_116_of_128_at_exact_budget(self):
        base = [
            {
                "correct": index < 116,
                "target_index": index % 4,
                "event_kind": "set" if index % 2 == 0 else "overwrite",
                "distractor_variant": "clean" if index % 2 == 0 else "distractor",
                "turn_type": "query" if index % 2 == 0 else "mixed",
                "topic": "style",
            }
            for index in range(128)
        ]

        passed = reachability_gate_summary(
            steps=2000,
            optimizer_steps_completed=2000,
            predictions=base,
            positive_gradient_steps=2000,
            clipped_steps=17,
        )
        base[115]["correct"] = False
        failed = reachability_gate_summary(
            steps=2000,
            optimizer_steps_completed=2000,
            predictions=base,
            positive_gradient_steps=2000,
            clipped_steps=17,
        )
        audit_only = reachability_gate_summary(
            steps=100,
            optimizer_steps_completed=100,
            predictions=base,
            positive_gradient_steps=100,
            clipped_steps=4,
        )

        self.assertTrue(passed["passed"])
        self.assertEqual(passed["final_correct"], 116)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["final_correct"], 115)
        self.assertFalse(audit_only["applicable"])
        self.assertIsNone(audit_only["passed"])

        listwise_is_not_r1 = reachability_gate_summary(
            steps=2000,
            optimizer_steps_completed=2000,
            predictions=base,
            positive_gradient_steps=2000,
            clipped_steps=17,
            reader_loss_mode="listwise-choice",
        )
        self.assertFalse(listwise_is_not_r1["applicable"])
        self.assertIsNone(listwise_is_not_r1["passed"])

    def test_reader_objective_contract_and_rotation_are_locked(self):
        target_only = reader_objective_contract("target-only")
        listwise = reader_objective_contract("listwise-choice")
        rotated, target_index = rotate_choices_left_one(("a", "b", "c", "d"), 2)

        self.assertEqual(target_only["historical_scope"], "R1/D2R-only")
        self.assertEqual(target_only["token_ce"], "fp32-logsumexp-minus-target-token-score")
        self.assertIsNone(target_only["choice_ce"])
        self.assertEqual(evaluation_views_for_reader_loss_mode("target-only"), ("canonical",))
        self.assertEqual(
            evaluation_views_for_reader_loss_mode("listwise-choice"),
            ("canonical", "left-rotate-one"),
        )
        self.assertEqual(listwise["choice_training_scores"], "negative-mean-token-nll")
        self.assertEqual(listwise["token_ce"], "fp32-logsumexp-minus-target-token-score")
        self.assertEqual(listwise["choice_ce"], "fp32-logsumexp-minus-target-choice-logit")
        self.assertEqual(listwise["choice_logit_temperature_float_hex"], float(1.0).hex())
        self.assertEqual(rotated, ("b", "c", "d", "a"))
        self.assertEqual(target_index, 1)

    def test_r2_gate_requires_both_views_subgroups_pairs_and_image_gradients(self):
        canonical = self.r2_predictions(view="canonical")
        rotated = self.r2_predictions(view="left-rotate-one")
        trace = self.r2_trace()

        passed = r2_gate_summary(
            reader_loss_mode="listwise-choice",
            steps=2000,
            optimizer_steps_completed=2000,
            canonical_predictions=canonical,
            rotated_predictions=rotated,
            trace=trace,
            positive_gradient_steps=2000,
            clipped_steps=37,
        )
        failed_canonical = deepcopy(canonical)
        failed_canonical[12]["predicted_index"] = (failed_canonical[12]["target_index"] + 1) % 4
        failed_canonical[12]["predicted_text"] = failed_canonical[12]["choices"][
            failed_canonical[12]["predicted_index"]
        ]
        failed_canonical[12]["correct"] = False
        failed = r2_gate_summary(
            reader_loss_mode="listwise-choice",
            steps=2000,
            optimizer_steps_completed=2000,
            canonical_predictions=failed_canonical,
            rotated_predictions=rotated,
            trace=trace,
            positive_gradient_steps=2000,
            clipped_steps=37,
        )
        audit_only = r2_gate_summary(
            reader_loss_mode="listwise-choice",
            steps=100,
            optimizer_steps_completed=100,
            canonical_predictions=canonical,
            rotated_predictions=rotated,
            trace=self.r2_trace(100),
            positive_gradient_steps=100,
            clipped_steps=3,
        )

        self.assertTrue(passed["passed"])
        self.assertEqual(passed["canonical"]["correct"], 116)
        self.assertEqual(passed["left_rotate_one"]["correct"], 116)
        self.assertTrue(passed["view_alignment"]["passed"])
        self.assertNotIn("noop", passed["canonical"]["grouped_predictions"]["state_event_kind"])
        self.assertEqual(
            passed["canonical"]["distractor_prediction_agreement"]["predicted_text_agreements"],
            64,
        )
        self.assertTrue(passed["listwise_gradient_evidence_valid"])
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["canonical"]["correct"], 115)
        self.assertFalse(audit_only["applicable"])
        self.assertIsNone(audit_only["passed"])

        unrotated_control = r2_gate_summary(
            reader_loss_mode="listwise-choice",
            steps=2000,
            optimizer_steps_completed=2000,
            canonical_predictions=canonical,
            rotated_predictions=canonical,
            trace=trace,
            positive_gradient_steps=2000,
            clipped_steps=37,
        )
        self.assertFalse(unrotated_control["passed"])
        self.assertFalse(unrotated_control["view_alignment"]["passed"])

    def test_listwise_trace_requires_positive_image_gradient_in_every_step(self):
        trace = self.r2_trace(2)
        trace[1]["listwise_queries"][0]["image_gradient_norm_float_hex"] = float(0.0).hex()

        summary = listwise_gradient_trace_summary(trace)

        self.assertTrue(summary["all_records_finite"])
        self.assertEqual(summary["steps_with_positive_image_gradient"], 1)
        self.assertFalse(summary["every_step_has_positive_image_gradient"])

    def test_listwise_trace_requires_positive_updater_gradient_in_every_step(self):
        trace = self.r2_trace(2)
        trace[1]["gradient_norm_before_clip_float_hex"] = float(0.0).hex()

        summary = listwise_gradient_trace_summary(trace)
        diagnostic = r2a_autograd_diagnostic("listwise-choice", trace)

        self.assertEqual(summary["positive_updater_gradient_steps"], 1)
        self.assertFalse(summary["every_step_has_positive_updater_gradient"])
        self.assertFalse(diagnostic["passed"])

    def test_distractor_pair_requires_identical_ordered_choices(self):
        canonical = self.r2_predictions(view="canonical")
        canonical[1]["choices"] = canonical[1]["choices"][1:] + canonical[1]["choices"][:1]

        gate = r2_gate_summary(
            reader_loss_mode="listwise-choice",
            steps=2000,
            optimizer_steps_completed=2000,
            canonical_predictions=canonical,
            rotated_predictions=self.r2_predictions(view="left-rotate-one"),
            trace=self.r2_trace(),
            positive_gradient_steps=2000,
            clipped_steps=0,
        )

        self.assertFalse(gate["canonical"]["distractor_prediction_agreement"]["passed"])
        self.assertFalse(gate["passed"])

    def test_listwise_query_trace_records_scores_entropy_margin_and_retained_image_grad(self):
        image = torch.ones(1, 3, 2, 2, requires_grad=True) * 2.0
        image.retain_grad()
        image.square().sum().backward()
        output = SimpleNamespace(
            loss=torch.tensor(0.5),
            choice_mean_nll=torch.tensor([0.5, 1.0, 1.5, 2.0]),
            choice_logits=torch.tensor([-0.5, -1.0, -1.5, -2.0]),
            choice_token_counts=(1, 1, 2, 1),
        )

        records = listwise_query_trace_records(
            [
                {
                    "output": output,
                    "image": image,
                    "choices": ("a", "b", "two tokens", "d"),
                    "target_index": 0,
                }
            ]
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["target_rank"], 1)
        self.assertEqual(float.fromhex(records[0]["margin_float_hex"]), 0.5)
        self.assertGreater(float.fromhex(records[0]["choice_entropy_float_hex"]), 0.0)
        self.assertGreater(float.fromhex(records[0]["image_gradient_norm_float_hex"]), 0.0)
        self.assertTrue(records[0]["all_values_finite"])

    def test_grouped_prediction_summary_records_labels_and_subtypes(self):
        predictions = [
            {
                "correct": True,
                "target_index": 0,
                "event_kind": "set",
                "distractor_variant": "clean",
                "turn_type": "query",
                "topic": "style",
            },
            {
                "correct": False,
                "target_index": 0,
                "event_kind": "overwrite",
                "distractor_variant": "distractor",
                "turn_type": "mixed",
                "topic": "style",
            },
        ]

        summary = grouped_prediction_summary(predictions)
        r2_summary = grouped_prediction_summary(
            [
                {**predictions[0], "state_event_kind": "set"},
                {**predictions[1], "state_event_kind": "overwrite"},
            ],
            event_kind_field="state_event_kind",
        )

        self.assertEqual(summary["target_index"]["0"]["count"], 2)
        self.assertEqual(summary["target_index"]["0"]["correct"], 1)
        self.assertEqual(summary["event_kind"]["set"]["correct"], 1)
        self.assertEqual(summary["distractor_variant"]["clean"]["correct"], 1)
        self.assertEqual(summary["turn_type"]["mixed"]["correct"], 0)
        self.assertEqual(r2_summary["state_event_kind"]["set"]["correct"], 1)

    def test_qwen3_image_grid_contract_accepts_256_and_rejects_252(self):
        processor = SimpleNamespace(patch_size=16, temporal_patch_size=2, merge_size=2)

        contract = validate_qwen_image_grid_contract(processor, image_size=256)

        self.assertEqual(contract["spatial_factor"], 32)
        with self.assertRaisesRegex(RuntimeError, "divisible"):
            validate_qwen_image_grid_contract(processor, image_size=252)

    def test_qwen3_image_grid_contract_rejects_configuration_drift(self):
        processor = SimpleNamespace(patch_size=14, temporal_patch_size=2, merge_size=2)
        with self.assertRaisesRegex(RuntimeError, "patch_size drifted"):
            validate_qwen_image_grid_contract(processor, image_size=252)

    def test_canonical_tensor_and_object_hashes_are_bitwise_and_order_stable(self):
        tensor = torch.tensor([[1.0, -0.0], [2.0, 3.0]], dtype=torch.float32)
        clone = tensor.clone()
        changed = tensor.clone()
        changed[1, 1] = torch.nextafter(changed[1, 1], torch.tensor(float("inf")))

        self.assertEqual(canonical_tensor_sha256(tensor), canonical_tensor_sha256(clone))
        self.assertNotEqual(canonical_tensor_sha256(tensor), canonical_tensor_sha256(changed))
        self.assertNotEqual(canonical_tensor_sha256(tensor), canonical_tensor_sha256(tensor.double()))
        self.assertEqual(
            canonical_object_sha256({"b": [tensor], "a": 1}),
            canonical_object_sha256({"a": 1, "b": [clone]}),
        )
        self.assertNotEqual(canonical_object_sha256(0.0), canonical_object_sha256(-0.0))

        manifest = named_tensors_manifest({"second": changed, "first": tensor})
        self.assertEqual(list(manifest["tensors"]), ["first", "second"])
        self.assertEqual(manifest["tensors"]["first"]["sha256"], canonical_tensor_sha256(tensor))

    def test_determinism_environment_fails_closed(self):
        expected = {
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        self.assertEqual(assert_determinism_environment(expected), dict(sorted(expected.items())))
        with self.assertRaisesRegex(RuntimeError, "environment mismatch"):
            assert_determinism_environment({**expected, "OMP_NUM_THREADS": "2"})

    def test_model_manifest_includes_nonpersistent_buffers(self):
        module = nn.Linear(2, 2)
        module.register_buffer("ephemeral", torch.tensor([3.0]), persistent=False)
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)

        manifest = model_optimizer_rng_manifest(module, optimizer)

        self.assertIn("parameter:weight", manifest["model"]["tensors"])
        self.assertIn("parameter:bias", manifest["model"]["tensors"])
        self.assertIn("buffer:ephemeral", manifest["model"]["tensors"])

    def test_bitwise_report_comparator_ignores_runtime_but_rejects_payload_drift(self):
        first = {
            "status": "complete",
            "runtime": {"pid": 1},
            "comparison_payload": {"trace": [{"loss": "0x1.0p+0"}], "model": "abc"},
        }
        second = {
            "status": "complete",
            "runtime": {"pid": 2},
            "comparison_payload": {"model": "abc", "trace": [{"loss": "0x1.0p+0"}]},
        }
        self.assertTrue(compare_bitwise_repro_reports(first, second)["valid"])

        second["comparison_payload"]["trace"][0]["loss"] = "0x1.0000000000001p+0"
        comparison = compare_bitwise_repro_reports(first, second)
        self.assertFalse(comparison["valid"])
        self.assertTrue(any("trace[0].loss" in mismatch for mismatch in comparison["mismatches"]))

        second["status"] = "failed"
        self.assertFalse(compare_bitwise_repro_reports(first, second)["valid"])

    def test_lora_whitelist_rejects_accidentally_trainable_base_weights(self):
        module = nn.Module()
        module.lora_A = nn.ModuleDict({"default": nn.Linear(2, 2, bias=False)})
        module.base_weight = nn.Parameter(torch.ones(2, 2))
        with self.assertRaisesRegex(RuntimeError, "non-LoRA"):
            lora_trainable_parameters(module)
        module.base_weight.requires_grad_(False)
        self.assertEqual(len(lora_trainable_parameters(module)), 1)

    def test_deterministic_fixture_has_locked_rgb_sha(self):
        first, first_metadata = load_source_image(None, resolution=1024)
        second, second_metadata = load_source_image(None, resolution=1024)

        self.assertEqual(first.size, (1024, 1024))
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["rgb_sha256"], DETERMINISTIC_FIXTURE_RGB_SHA256_1024)

    def test_formal_blank_initial_image_is_uniform_and_fail_closed(self):
        first, first_metadata = load_initial_image("blank", resolution=64)
        second, second_metadata = load_initial_image("blank", resolution=64)

        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(first_metadata["initial_state_mode"], "blank")
        self.assertEqual(first.getextrema(), ((127, 127), (127, 127), (127, 127)))
        with self.assertRaisesRegex(ValueError, "does not accept"):
            load_initial_image("blank", Path("unexpected.png"), resolution=64)
        with self.assertRaisesRegex(ValueError, "requires"):
            load_initial_image("file", resolution=64)

    def test_adapter_seed_is_repeatable_and_explicit(self):
        seed_adapter_initialization(37)
        first = torch.randn(4)
        seed_adapter_initialization(37)
        second = torch.randn(4)
        torch.testing.assert_close(first, second)

    def test_frozen_gradient_contract_reports_and_rejects_gradients(self):
        module = nn.Linear(2, 2).requires_grad_(False)
        report = assert_no_frozen_parameter_grads({"reader": module}, fully_frozen={"reader"})
        self.assertEqual(report["reader"]["frozen_tensors_with_grad"], 0)

        module.weight.grad = torch.ones_like(module.weight)
        with self.assertRaisesRegex(RuntimeError, "reader accumulated gradients"):
            assert_no_frozen_parameter_grads({"reader": module}, fully_frozen={"reader"})

    def test_valid_e2e_positive_detach_pair(self):
        report = validate_e2e_pair_reports(
            make_pair_report(detached=False),
            make_pair_report(detached=True),
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["positive_intermediate_grad_norm"], 0.75)
        self.assertIsNone(report["detached_intermediate_grad_norm"])

    def test_pair_validator_rejects_metadata_drift(self):
        positive = make_pair_report(detached=False)
        detached = make_pair_report(detached=True)
        detached["pair_metadata"] = {**detached["pair_metadata"], "adapter_seed": 12}
        with self.assertRaisesRegex(ValueError, "metadata"):
            validate_e2e_pair_reports(positive, detached)

    def test_pair_validator_rejects_forward_drift(self):
        positive = make_pair_report(detached=False)
        detached = make_pair_report(detached=True)
        detached["loss"] = 1.5
        with self.assertRaisesRegex(ValueError, "Forward losses differ"):
            validate_e2e_pair_reports(positive, detached)


if __name__ == "__main__":
    unittest.main()
