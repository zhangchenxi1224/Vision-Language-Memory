from __future__ import annotations

import copy
import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.event_noise import event_seed  # noqa: E402
from vision_memory.training.r11_new_oracle import (  # noqa: E402
    R11_NEW_CHECKPOINT_STEPS,
    R11_NEW_EFFECTIVE_SIGMA_SCHEDULE,
    R11_NEW_EFFECTIVE_START_SIGMA,
    R11_NEW_NOISE_KEY,
    R11_NEW_OPTIMIZER_STEPS,
    R11_NEW_PARENT_R11_COMPARISON_SHA256,
    R11_NEW_PRIMARY_ENDPOINT,
    R11_NEW_READER_LOSS_INPUTS,
    R11_NEW_TARGETS_PAYLOAD_SHA256,
    R11_NEW_TARGET_IDS,
    build_phase1a_schedule,
    locked_event_noise_seed,
    phase1a_arm_gate,
    phase1a_effective_sigmas_match,
    phase1a_target_gate,
    phase1a_target_statistics,
    phase1a_technical_gate,
    validate_information_boundary,
    validate_phase1a_config,
)


def valid_audit() -> dict[str, object]:
    return {
        "checkpoint_steps_observed": list(R11_NEW_CHECKPOINT_STEPS),
        "latent_checkpoint_steps_observed": list(R11_NEW_CHECKPOINT_STEPS),
        "image_checkpoint_steps_observed": list(R11_NEW_CHECKPOINT_STEPS),
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
        "primary_endpoint": R11_NEW_PRIMARY_ENDPOINT,
    }


def valid_receipts(target_index: int = 0) -> list[dict[str, object]]:
    return [
        {
            "kind": "optimizer_step",
            "optimizer_step": unit.optimizer_step,
            "target_segment_id": unit.target_segment_id,
            "forward_cyclic_training_view": unit.forward_cyclic_training_view,
            "permutation": list(unit.permutation),
            "loss_before_step": 1.0,
            "gradient_norm": 0.5,
            "gradient_nonzero_fraction": 0.25,
            "full_dreamlite_forward_executed": True,
            "denoiser_steps_executed": 4,
            "effective_sigma_schedule": list(R11_NEW_EFFECTIVE_SIGMA_SCHEDULE),
        }
        for unit in build_phase1a_schedule(target_index)
    ]


class R11NewOracleContractTest(unittest.TestCase):
    def test_runtime_sigma_tolerance_preserves_raw_values_and_rejects_drift(self) -> None:
        observed = [0.4999999701976776, 0.375, 0.25, 0.1249999925494194]
        original = list(observed)
        self.assertTrue(phase1a_effective_sigmas_match(observed))
        self.assertTrue(phase1a_effective_sigmas_match(tuple(observed)))
        self.assertEqual(observed, original)
        receipts = valid_receipts()
        for row in receipts:
            row["effective_sigma_schedule"] = list(observed)
        self.assertTrue(
            phase1a_technical_gate(receipts, target_segment_id=R11_NEW_TARGET_IDS[0], audit=valid_audit())["passed"]
        )
        invalid = (
            [0.50001, 0.375, 0.25, 0.125],
            [float("nan"), 0.375, 0.25, 0.125],
            [float("inf"), 0.375, 0.25, 0.125],
            [True, 0.375, 0.25, 0.125],
            ["0.5", 0.375, 0.25, 0.125],
            [0.5, 0.375, 0.25],
            [0.5, 0.375, 0.25, 0.125, 0.0],
            None,
        )
        for sigmas in invalid:
            with self.subTest(sigmas=sigmas):
                self.assertFalse(phase1a_effective_sigmas_match(sigmas))
                receipts[-1]["effective_sigma_schedule"] = sigmas
                self.assertFalse(
                    phase1a_technical_gate(receipts, target_segment_id=R11_NEW_TARGET_IDS[0], audit=valid_audit())[
                        "passed"
                    ]
                )

    def test_machine_config_is_exactly_the_immutable_contract(self) -> None:
        path = ROOT / "configs" / "experiments" / "r11_new_frozen_dreamlite_oracle_phase1a.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        report = validate_phase1a_config(config)
        self.assertTrue(report["passed"])
        self.assertEqual(
            config["parent_evidence"]["canonical_r11_comparison_sha256"], R11_NEW_PARENT_R11_COMPARISON_SHA256
        )
        self.assertTrue(config["parent_evidence"]["canonical_r11_is_not_phase1a"])
        self.assertEqual(config["target_selection"]["target_ids"], list(R11_NEW_TARGET_IDS))
        self.assertEqual(config["target_selection"]["selected_segment_payload_sha256"], R11_NEW_TARGETS_PAYLOAD_SHA256)
        self.assertEqual(config["dreamlite_path"]["effective_start_sigma"], 0.5)
        self.assertEqual(R11_NEW_EFFECTIVE_START_SIGMA, 0.5)
        self.assertEqual(config["dreamlite_path"]["effective_sigma_schedule"], [0.5, 0.375, 0.25, 0.125])
        self.assertIsNone(config["optimization"]["gradient_clipping"])
        self.assertEqual(
            config["mvp_route"]["sequence"],
            ["phase1a", "phase2", "phase3a", "phase3b"],
        )
        self.assertFalse(config["mvp_route"]["phase1b_required_before_phase2"])
        self.assertTrue(config["mvp_route"]["phase2_uses_existing_train_split_only"])
        self.assertFalse(config["mvp_route"]["phase2_full_train_coverage_required"])
        self.assertEqual(config["mvp_route"]["phase2_supervision_scope"], "query_level")
        self.assertEqual(
            config["mvp_route"]["phase2_candidate_filter"],
            "locked_train_sha_build_r5_family_pools_pairing_seed0_f1_single_event_unique_segment_id",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_selection_protocol"],
            "reuse_r10_f1_hash_order_v1",
        )
        self.assertEqual(config["mvp_route"]["phase2_selection_seed"], 20260831)
        self.assertEqual(
            config["mvp_route"]["phase2_selection_key"],
            "sha256('R10-VisualAlignment-LowerBound' + unit-separator + '20260831' + unit-separator + 'F1' + unit-separator + segment_id), then segment_id",
        )
        self.assertTrue(config["mvp_route"]["phase2_phase1a_first8_reused"])
        self.assertTrue(config["mvp_route"]["phase2_membership_locked_before_any_phase1a_outcome"])
        self.assertEqual(config["mvp_route"]["phase2_candidate_count"], 7504)
        self.assertEqual(
            config["mvp_route"]["phase2_bank8_ids_sha256"],
            "08c25bbb753e7ffb3a0fd760d0bbf079b113f1db12be9eba4af1505ad57e86ff",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_bank8_payload_sha256"],
            R11_NEW_TARGETS_PAYLOAD_SHA256,
        )
        self.assertEqual(
            config["mvp_route"]["phase2_bank64_ids_sha256"],
            "5cbd99fdc537f67cba311ed39144516735d0da149e87095565118b162a872fcc",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_bank64_payload_sha256"],
            "d7d3a3d12182fd3169c5b9b5127617f9c1c5b81462a94c2d8afccb256973d98a",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_bank128_ids_sha256"],
            "c762b52c2b71ba8b977b6bec339a9586ef000440021c8bdf38bef28006d99f37",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_bank128_payload_sha256"],
            "6b817ffdc488df1925294aa6169e8d33cb738877432fb9da46b2844aec6a3665",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_required_technical_passes"],
            "all_selected_items",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_required_oracle_passes"],
            "all_selected_items",
        )
        self.assertTrue(config["mvp_route"]["phase2_failed_item_replacement_forbidden"])
        self.assertTrue(config["mvp_route"]["phase2_technical_retry_same_id_only"])
        self.assertTrue(config["mvp_route"]["phase3_requires_complete_selected_bank"])
        self.assertEqual(config["mvp_route"]["phase2_minimum_bank_size"], 64)
        self.assertEqual(config["mvp_route"]["phase2_optional_bank_size"], 128)
        self.assertEqual(config["mvp_route"]["phase2_calibration_items"], 8)
        self.assertTrue(config["mvp_route"]["phase2_bank64_is_bank128_prefix"])
        self.assertEqual(config["mvp_route"]["phase2_default_size_if_clock_unreliable"], 64)
        self.assertEqual(config["mvp_route"]["program_planning_window_hours"], 30.0)
        self.assertEqual(
            config["mvp_route"]["program_clock_start_event"],
            "phase1a_technical_preflight_controller_launch",
        )
        self.assertEqual(config["mvp_route"]["phase3_and_reporting_reserve_hours"], 9.0)
        self.assertEqual(config["mvp_route"]["phase2_projection_safety_factor"], 1.15)
        self.assertEqual(
            config["mvp_route"]["phase2_projection_statistic"],
            "p90_wall_clock_seconds_first_8",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_projection_p90_method"],
            "nearest_rank_ceil_0p90_n8",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_projection_remaining_items_formula"],
            "N-8",
        )
        self.assertEqual(
            config["mvp_route"]["phase2_projected_remaining_items_for_128"],
            120,
        )
        self.assertEqual(
            config["mvp_route"]["phase2_expand_to_128_if_projected_total_hours_lte"],
            30.0,
        )
        self.assertTrue(config["mvp_route"]["phase2_minimum64_even_if_window_infeasible"])
        self.assertTrue(config["mvp_route"]["phase2_size_decision_before_item_9"])
        self.assertEqual(
            config["mvp_route"]["phase3_initial_claim"],
            "locked_training_subset_learnability_diagnostic",
        )
        self.assertTrue(config["success_boundary"]["query_level_diagnostic_only"])
        self.assertFalse(config["success_boundary"]["scientific_success_gate"])

    def test_config_validator_rejects_any_scientific_parameter_drift(self) -> None:
        path = ROOT / "configs" / "experiments" / "r11_new_frozen_dreamlite_oracle_phase1a.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        mutations = (
            (("optimization", "learning_rate"), 0.1),
            (("optimization", "optimizer_steps"), 255),
            (("dreamlite_path", "effective_start_sigma"), 1.0),
            (("oracle_variable", "noise_key"), ["global_seed", "target_index"]),
            (("target_selection", "target_ids"), list(reversed(R11_NEW_TARGET_IDS))),
            (("mvp_route", "phase1b_required_before_phase2"), True),
            (("mvp_route", "phase2_minimum_bank_size"), 32),
            (("mvp_route", "phase2_optional_bank_size"), 96),
            (("mvp_route", "phase2_required_oracle_passes"), "successful_items_only"),
            (("mvp_route", "phase2_projection_safety_factor"), 1.0),
            (("mvp_route", "program_planning_window_hours"), 48.0),
        )
        for path_keys, replacement in mutations:
            with self.subTest(path=path_keys):
                config = copy.deepcopy(original)
                cursor = config
                for key in path_keys[:-1]:
                    cursor = cursor[key]
                cursor[path_keys[-1]] = replacement
                with self.assertRaisesRegex(ValueError, "config drifted"):
                    validate_phase1a_config(config)

    def test_schedule_is_per_target_and_exactly_balanced(self) -> None:
        for target_index, target_id in enumerate(R11_NEW_TARGET_IDS):
            schedule = build_phase1a_schedule(target_index)
            self.assertEqual(len(schedule), R11_NEW_OPTIMIZER_STEPS)
            self.assertEqual({row.target_segment_id for row in schedule}, {target_id})
            self.assertEqual(
                Counter(row.forward_cyclic_training_view for row in schedule),
                Counter({0: 64, 1: 64, 2: 64, 3: 64}),
            )
            for row in schedule:
                self.assertEqual(
                    row.permutation,
                    ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2))[row.forward_cyclic_training_view],
                )
        with self.assertRaises((TypeError, ValueError)):
            build_phase1a_schedule(True)
        with self.assertRaises(ValueError):
            build_phase1a_schedule(8)

    def test_noise_seed_uses_only_locked_source_episode_and_turn(self) -> None:
        self.assertEqual(R11_NEW_NOISE_KEY, ("global_seed", "source_episode_id", "source_turn_id"))
        self.assertEqual(
            locked_event_noise_seed("episode-a", 3),
            event_seed(0, "episode-a", 3),
        )

    def test_information_boundary_keeps_query_supervision_after_dreamlite(self) -> None:
        report = validate_information_boundary(
            dreamlite_inputs=("source_latent", "event_text", "x_T"),
            noise_key=R11_NEW_NOISE_KEY,
            reader_loss_inputs=R11_NEW_READER_LOSS_INPUTS,
        )
        self.assertTrue(report["passed"])
        with self.assertRaisesRegex(ValueError, "supervision"):
            validate_information_boundary(
                dreamlite_inputs=("source_latent", "event_text", "x_T", "query_text"),
                noise_key=R11_NEW_NOISE_KEY,
                reader_loss_inputs=R11_NEW_READER_LOSS_INPUTS,
            )
        with self.assertRaisesRegex(ValueError, "noise key drifted"):
            validate_information_boundary(
                dreamlite_inputs=("source_latent", "event_text", "x_T"),
                noise_key=("global_seed", "source_episode_id", "source_turn_id", "target_index"),
                reader_loss_inputs=R11_NEW_READER_LOSS_INPUTS,
            )

    def test_technical_gate_requires_full_frozen_path_and_exact_receipts(self) -> None:
        receipts = valid_receipts()
        gate = phase1a_technical_gate(
            receipts,
            target_segment_id=R11_NEW_TARGET_IDS[0],
            audit=valid_audit(),
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["training_view_counts"], {0: 64, 1: 64, 2: 64, 3: 64})
        self.assertFalse(gate["scientific_success_gate"])

        bad_sigma = copy.deepcopy(receipts)
        bad_sigma[-1]["effective_sigma_schedule"] = [1.0, 0.75, 0.5, 0.25]
        self.assertFalse(
            phase1a_technical_gate(
                bad_sigma,
                target_segment_id=R11_NEW_TARGET_IDS[0],
                audit=valid_audit(),
            )["passed"]
        )
        bad_audit = valid_audit()
        bad_audit["trainable_parameter_names"] = ["x_T_fp32", "unet.weight"]
        self.assertFalse(
            phase1a_technical_gate(
                receipts,
                target_segment_id=R11_NEW_TARGET_IDS[0],
                audit=bad_audit,
            )["passed"]
        )

    def test_target_and_arm_gates_remain_diagnostic_only(self) -> None:
        target_id = R11_NEW_TARGET_IDS[0]
        rows = []
        for checkpoint in ("m0", R11_NEW_PRIMARY_ENDPOINT):
            for condition in ("normal", "reset"):
                for view in range(4):
                    endpoint_normal = checkpoint == R11_NEW_PRIMARY_ENDPOINT and condition == "normal"
                    rows.append(
                        {
                            "suite": "r11_new_phase1a",
                            "checkpoint": checkpoint,
                            "condition": condition,
                            "pair_unit": target_id,
                            "view_index": view,
                            "ce": 5.0 if endpoint_normal else 10.0,
                            "correct": 1 if endpoint_normal else 0,
                        }
                    )
        statistics = phase1a_target_statistics(
            rows,
            suite="r11_new_phase1a",
            target_segment_id=target_id,
        )
        self.assertTrue(phase1a_target_gate(statistics, technical_gate=True))
        results = [
            {
                "target_segment_id": value,
                "technical_gate": True,
                "target_reachability_gate": True,
            }
            for value in R11_NEW_TARGET_IDS
        ]
        arm = phase1a_arm_gate(results)
        self.assertTrue(arm["passed"])
        self.assertTrue(arm["query_level_diagnostic_only"])
        self.assertFalse(arm["scientific_success_gate"])


if __name__ == "__main__":
    unittest.main()
