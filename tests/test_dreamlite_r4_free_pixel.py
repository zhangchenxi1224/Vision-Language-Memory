from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train" / "dreamlite_r4_free_pixel.py"
CONFIG = ROOT / "configs" / "experiments" / "r4_freepixel_mechanism_rescue.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _load_script():
    name = "dreamlite_r4_free_pixel_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R4 = _load_script()


def _budget_args(**overrides):
    values = {
        "max_optimizer_steps": 256,
        "set_warmup_optimizer_steps": 32,
        "eval_limit": 8,
        "identity_calibration_count": 8,
        "checkpoint_every": 16,
        "smoke": False,
        "strict_determinism": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _query(label: str) -> dict[str, object]:
    return {
        "text": f"What is {label}?",
        "choices": [f"{label}-{index}" for index in range(4)],
        "target_index": 1,
    }


def _trainable_four_kind_transitions():
    episode = {
        "episode_id": "r4-entrypoint-contract",
        "turns": [
            {
                "type": "mixed",
                "event_kind": "set",
                "event_text": "remember blue",
                "query": _query("set"),
            },
            {
                "type": "mixed",
                "event_kind": "overwrite",
                "event_text": "replace blue with green",
                "query": _query("overwrite"),
            },
            {
                "type": "event",
                "event_kind": "noop",
                "event_text": "the weather is mild",
            },
            {
                "type": "mixed",
                "event_kind": "clear",
                "event_text": "forget the color",
                "query": _query("clear"),
            },
        ],
    }
    return R4.build_transition_index([episode])


class R4BudgetContractTest(unittest.TestCase):
    def test_formal_budget_is_exactly_256_steps_32_warmup_and_2048_micros(self):
        budget = R4.resolve_budget(_budget_args())

        self.assertEqual(budget.scope, "mechanism_rescue")
        self.assertEqual(budget.optimizer_steps, 256)
        self.assertEqual(budget.warmup_optimizer_steps, 32)
        self.assertEqual(budget.balanced_optimizer_steps, 224)
        self.assertEqual(budget.gradient_accumulation, 8)
        self.assertEqual(budget.total_micro_transitions, 2048)
        self.assertEqual(budget.warmup_micro_transitions, 256)
        self.assertEqual(budget.balanced_micro_transitions, 1792)
        self.assertEqual(budget.to_dict()["endpoint_selection"], "fixed_final_only")
        self.assertFalse(budget.to_dict()["intermediate_dev_selection"])

    def test_smoke_is_exactly_one_complete_16_unit_balanced_block(self):
        budget = R4.resolve_budget(_budget_args(max_optimizer_steps=2, set_warmup_optimizer_steps=0, smoke=True))
        counts = R4.expected_schedule_counts(budget)

        self.assertEqual(budget.scope, "technical_smoke")
        self.assertEqual(budget.total_micro_transitions, 16)
        self.assertEqual(budget.warmup_micro_transitions, 0)
        self.assertEqual(budget.balanced_micro_transitions, 16)
        self.assertEqual(counts["by_event_kind"], {kind: 4 for kind in R4.R4_EVENT_KINDS})
        self.assertEqual(counts["by_diffusion_step"], {str(step): 4 for step in range(4)})
        self.assertEqual(set(counts["balanced_by_event_kind_step"].values()), {1})
        self.assertEqual(len(counts["balanced_by_event_kind_step"]), 16)

    def test_invalid_or_mutated_budgets_fail_closed(self):
        invalid = (
            (_budget_args(max_optimizer_steps=True), "integers"),
            (_budget_args(max_optimizer_steps=255), "fixed"),
            (_budget_args(set_warmup_optimizer_steps=33), "fixed"),
            (
                _budget_args(max_optimizer_steps=2, set_warmup_optimizer_steps=1, smoke=True),
                "smoke requires",
            ),
            (_budget_args(max_optimizer_steps=1, set_warmup_optimizer_steps=2), "cannot exceed"),
        )
        for args, message in invalid:
            with self.subTest(args=args, message=message):
                with self.assertRaisesRegex(ValueError, message):
                    R4.resolve_budget(args)

        incomplete = R4.RunBudget(
            scope="invalid",
            optimizer_steps=2,
            warmup_optimizer_steps=0,
            balanced_optimizer_steps=2,
            gradient_accumulation=7,
            eval_limit=8,
            identity_calibration_count=8,
            checkpoint_every=16,
        )
        with self.assertRaisesRegex(ValueError, "complete balanced blocks"):
            R4.expected_schedule_counts(incomplete)

    def test_runtime_contract_locks_resolution_and_distinct_devices(self):
        args = _budget_args()
        args.resolution = 1024
        args.dreamlite_device = "cuda:0"
        args.reader_device = "cuda:1"
        self.assertEqual(R4.validate_runtime_args(args).optimizer_steps, 256)

        args.resolution = 512
        with self.assertRaisesRegex(ValueError, "1024x1024"):
            R4.validate_runtime_args(args)
        args.resolution = 1024
        args.reader_device = "cuda:0"
        with self.assertRaisesRegex(ValueError, "distinct devices"):
            R4.validate_runtime_args(args)
        args.reader_device = "cuda:1"
        args.strict_determinism = False
        with self.assertRaisesRegex(ValueError, "strict-determinism"):
            R4.validate_runtime_args(args)

    def test_formal_data_gate_accepts_only_exact_hashes_and_transition_audit(self):
        args = SimpleNamespace(train=Path("formal-train.jsonl"), dev=Path("formal-dev.jsonl"))
        budget = R4.resolve_budget(_budget_args())
        audit = dict(json.loads(CONFIG.read_text(encoding="utf-8"))["data"]["transition_audit"])

        with mock.patch.object(
            R4,
            "sha256_file",
            side_effect=[R4.FORMAL_TRAIN_SHA256, R4.FORMAL_DEV_SHA256],
        ) as sha:
            R4.validate_formal_data_contract(
                args=args,
                budget=budget,
                audit=audit,
                transition_digest=R4.FORMAL_TRANSITION_INDEX_SHA256,
            )
        self.assertEqual(sha.call_args_list, [mock.call(args.train), mock.call(args.dev)])

        with mock.patch.object(
            R4,
            "sha256_file",
            side_effect=["0" * 64, R4.FORMAL_DEV_SHA256],
        ):
            with self.assertRaisesRegex(ValueError, "dataset SHA256 mismatch"):
                R4.validate_formal_data_contract(
                    args=args,
                    budget=budget,
                    audit=audit,
                    transition_digest=R4.FORMAL_TRANSITION_INDEX_SHA256,
                )

        count_drift = {
            **audit,
            "by_event_kind": {**audit["by_event_kind"], "set": audit["by_event_kind"]["set"] - 1},
        }
        with mock.patch.object(
            R4,
            "sha256_file",
            side_effect=[R4.FORMAL_TRAIN_SHA256, R4.FORMAL_DEV_SHA256],
        ):
            with self.assertRaisesRegex(ValueError, "transition-index contract mismatch"):
                R4.validate_formal_data_contract(
                    args=args,
                    budget=budget,
                    audit=count_drift,
                    transition_digest=R4.FORMAL_TRANSITION_INDEX_SHA256,
                )

        with mock.patch.object(
            R4,
            "sha256_file",
            side_effect=[R4.FORMAL_TRAIN_SHA256, R4.FORMAL_DEV_SHA256],
        ):
            with self.assertRaisesRegex(ValueError, "canonical transition digest mismatch"):
                R4.validate_formal_data_contract(
                    args=args,
                    budget=budget,
                    audit=audit,
                    transition_digest="0" * 64,
                )

    def test_smoke_data_gate_skips_formal_hash_and_count_bindings(self):
        args = SimpleNamespace(train=Path("tiny-train.jsonl"), dev=Path("tiny-dev.jsonl"))
        budget = R4.resolve_budget(_budget_args(max_optimizer_steps=2, set_warmup_optimizer_steps=0, smoke=True))
        with mock.patch.object(
            R4,
            "sha256_file",
            side_effect=AssertionError("smoke must not hash against the formal dataset contract"),
        ) as sha:
            R4.validate_formal_data_contract(
                args=args,
                budget=budget,
                audit={"total_transitions": 4},
                transition_digest="smoke-not-formal",
            )
        sha.assert_not_called()


class R4TrainingUnitContractTest(unittest.TestCase):
    def setUp(self):
        self.examples = _trainable_four_kind_transitions()

    def test_warmup_draws_only_trainable_set_from_blank_and_cycles_all_steps(self):
        units = [
            R4.training_unit_for_micro(
                self.examples,
                schedule_seed=13,
                global_micro_index=index,
                warmup_micro_transitions=8,
            )
            for index in range(8)
        ]

        self.assertTrue(all(unit.phase == "set_from_blank_warmup" for unit in units))
        self.assertTrue(all(unit.transition.event_kind == "set" for unit in units))
        self.assertTrue(all(unit.transition.is_trainable for unit in units))
        self.assertTrue(all(unit.transition.uses_qa_loss for unit in units))
        self.assertTrue(all(unit.transition.prefix_updater_indices == () for unit in units))
        self.assertEqual([unit.selected_step_index for unit in units], [0, 1, 2, 3, 0, 1, 2, 3])

        boundary = R4.training_unit_for_micro(
            self.examples,
            schedule_seed=13,
            global_micro_index=8,
            warmup_micro_transitions=8,
        )
        self.assertEqual(boundary.phase, "balanced_transition")
        self.assertEqual(boundary.phase_unit_index, 0)
        self.assertIsNotNone(boundary.balanced_schedule)

    def test_formal_expected_counts_are_exact_and_sum_to_2048(self):
        budget = R4.resolve_budget(_budget_args())
        counts = R4.expected_schedule_counts(budget)
        expected_pairs = {f"{kind}:{step}": 112 for kind in R4.R4_EVENT_KINDS for step in R4.R4_DIFFUSION_STEPS}

        self.assertEqual(
            counts["by_event_kind"],
            {"set": 704, "overwrite": 448, "clear": 448, "noop": 448},
        )
        self.assertEqual(counts["by_diffusion_step"], {str(step): 512 for step in range(4)})
        self.assertEqual(counts["balanced_by_event_kind_step"], expected_pairs)
        self.assertEqual(sum(counts["by_event_kind"].values()), 2048)
        self.assertEqual(sum(counts["by_diffusion_step"].values()), 2048)
        self.assertEqual(sum(counts["balanced_by_event_kind_step"].values()), 1792)

    def test_updater_kwargs_lock_free_rgb_drtune_one_step_and_presentation(self):
        unit = R4.training_unit_for_micro(
            self.examples,
            schedule_seed=29,
            global_micro_index=37,
            warmup_micro_transitions=0,
        )

        self.assertEqual(
            unit.updater_kwargs(),
            {
                "gradient_mode": "drtune",
                "selected_step_indices": (unit.selected_step_index,),
                "persistent_state": "float_rgb",
                "presentation_index": 37,
            },
        )
        self.assertEqual(len(unit.updater_kwargs()["selected_step_indices"]), 1)

    def test_normalized_identity_loss_uses_frozen_positive_scale(self):
        raw = torch.tensor(0.75, requires_grad=True)
        normalized = R4.normalized_identity_loss(raw, 0.25)
        self.assertEqual(float(normalized.detach()), 3.0)
        normalized.backward()
        self.assertEqual(float(raw.grad), 4.0)

        for invalid in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(scale=invalid):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    R4.normalized_identity_loss(raw.detach(), invalid)

    def test_objective_combination_has_exact_none_gates_sum_and_gradients(self):
        qa = torch.tensor(2.0, requires_grad=True)
        identity = torch.tensor(3.0, requires_grad=True)

        self.assertIs(R4.combine_objective_losses(qa, None), qa)
        self.assertIs(R4.combine_objective_losses(None, identity), identity)
        combined = R4.combine_objective_losses(qa, identity)
        self.assertEqual(float(combined.detach()), 5.0)
        combined.backward()
        self.assertEqual(float(qa.grad), 1.0)
        self.assertEqual(float(identity.grad), 1.0)

        with self.assertRaisesRegex(ValueError, "no admissible objective"):
            R4.combine_objective_losses(None, None)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_terminal_noop_identity_moves_differentiably_to_reader_device(self):
        qa = torch.tensor(2.0, device="cuda:1", requires_grad=True)
        identity = torch.tensor(3.0, device="cuda:0", requires_grad=True)

        combined = R4.combine_objective_losses(qa, identity)
        self.assertEqual(combined.device, qa.device)
        self.assertEqual(float(combined.detach().cpu()), 5.0)
        combined.backward()
        self.assertEqual(float(qa.grad.cpu()), 1.0)
        self.assertEqual(float(identity.grad.cpu()), 1.0)

    def test_checkpoint_state_resumes_at_the_exact_phase_and_schedule_cursor(self):
        budget = R4.resolve_budget(_budget_args())
        m0 = {"mean_episode_listwise_choice_ce": 1.25, "episode_count": 8}
        at_boundary = R4.checkpoint_trainer_state(
            optimizer_step=32,
            budget=budget,
            schedule_seed=17,
            identity_scale=0.125,
            m0_evaluation=m0,
        )

        self.assertEqual(at_boundary["next_global_micro_index"], 256)
        self.assertEqual(at_boundary["warmup_next_unit_index"], 256)
        self.assertEqual(
            at_boundary["balanced_schedule_cursor"],
            R4.R4ScheduleCursor(seed=17, next_unit_index=0).to_dict(),
        )
        self.assertEqual(at_boundary["identity_scale"], 0.125)
        self.assertEqual(at_boundary["m0_evaluation"], m0)
        R4.validate_checkpoint_trainer_state(
            at_boundary,
            optimizer_step=32,
            budget=budget,
            schedule_seed=17,
            identity_scale=0.125,
        )

        final_state = R4.checkpoint_trainer_state(
            optimizer_step=256,
            budget=budget,
            schedule_seed=17,
            identity_scale=0.125,
            m0_evaluation=m0,
        )
        self.assertEqual(final_state["next_global_micro_index"], 2048)
        self.assertEqual(final_state["balanced_schedule_cursor"]["next_unit_index"], 1792)

        tampered = {**at_boundary, "next_global_micro_index": 255}
        with self.assertRaisesRegex(ValueError, "cursor/phase mismatch"):
            R4.validate_checkpoint_trainer_state(
                tampered,
                optimizer_step=32,
                budget=budget,
                schedule_seed=17,
                identity_scale=0.125,
            )


class R4FixedConfigContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_config_embeds_the_canonical_teacher_free_protocol_contract(self):
        protocol = R4.make_r4_manifest_contract(schedule_seed=0)
        self.assertEqual(self.config["state_contract"], protocol)
        self.assertEqual(R4.validate_r4_manifest_contract(self.config["state_contract"]), protocol)
        self.assertEqual(protocol["persistent_state"], "float_rgb")
        self.assertEqual(protocol["state_target_policy"], "none")
        self.assertTrue(protocol["pixel_or_latent_targets_forbidden"])
        self.assertTrue(protocol["codebook_forbidden"])
        self.assertFalse(protocol["canonical_teacher_artifacts_loaded"])
        self.assertEqual(protocol["updater_learned_conditioning"], "previous_rgb+visible_event_text")
        self.assertTrue(protocol["sampling_noise_keys_answer_agnostic"])
        self.assertTrue(protocol["query_or_target_in_noise_seed_forbidden"])

        forbidden = set(self.config["forbidden_bindings"])
        self.assertTrue(
            {
                "teacher_image",
                "canonical_canvas",
                "pixel_target",
                "latent_target",
                "feature_target",
                "codebook",
            }
            <= forbidden
        )

    def test_config_locks_compute_optimizer_budget_and_evaluation(self):
        runtime = self.config["runtime"]
        models = self.config["models"]
        training = self.config["training"]
        evaluation = self.config["evaluation"]

        self.assertEqual(runtime["accelerator"], "NVIDIA H200 141GB")
        self.assertEqual(runtime["accelerator_count"], 2)
        self.assertEqual((runtime["dreamlite_device"], runtime["reader_device"]), ("cuda:0", "cuda:1"))
        self.assertEqual(models["lora"]["rank"], 4)
        self.assertEqual(training["resolution"], 1024)
        self.assertEqual(training["learning_rate"], 3e-5)
        self.assertEqual(training["weight_decay"], 1e-4)
        self.assertEqual(training["gradient_accumulation"], 8)
        self.assertEqual(training["gradient_clip_global_norm"], 1.0)
        self.assertEqual(training["checkpoint_every_optimizer_steps"], 16)
        self.assertEqual(training["identity_calibration_count"], 8)
        self.assertTrue(training["strict_determinism"])
        self.assertEqual(evaluation["eval_limit"], 8)
        self.assertEqual(evaluation["checkpoints"], ["M0", "final_step_256"])
        self.assertFalse(evaluation["intermediate_dev_evaluation"])
        self.assertFalse(evaluation["best_dev_checkpoint_selection"])

    def test_config_counts_match_code_and_authoritative_formal_audit(self):
        budget = R4.resolve_budget(_budget_args())
        training = self.config["training"]
        self.assertEqual(training["optimizer_steps"], budget.optimizer_steps)
        self.assertEqual(training["warmup_optimizer_steps"], budget.warmup_optimizer_steps)
        self.assertEqual(training["balanced_optimizer_steps"], budget.balanced_optimizer_steps)
        self.assertEqual(training["total_micro_transitions"], budget.total_micro_transitions)
        self.assertEqual(training["warmup_micro_transitions"], budget.warmup_micro_transitions)
        self.assertEqual(training["balanced_micro_transitions"], budget.balanced_micro_transitions)
        self.assertEqual(self.config["schedule"]["expected_counts"], R4.expected_schedule_counts(budget))

        audit = self.config["data"]["transition_audit"]
        self.assertEqual(audit["total_transitions"], 12500)
        self.assertEqual(audit["trainable_transitions"], 11876)
        self.assertEqual(audit["prefix_only_transitions"], 624)
        self.assertEqual(audit["local_query_count"], 12496)
        self.assertEqual(
            audit["by_event_kind"],
            {"set": 7504, "overwrite": 1872, "clear": 624, "noop": 2500},
        )
        self.assertEqual(
            audit["trainable_by_event_kind"],
            {"set": 6880, "overwrite": 1872, "clear": 624, "noop": 2500},
        )
        self.assertEqual(
            audit["by_objective"],
            {"qa_only": 9376, "identity_only": 1876, "qa_and_identity": 624, "prefix_only": 624},
        )
        self.assertEqual(
            self.config["data"]["canonical_transition_index_sha256"],
            "2b96ae090955c25092b0d415b5b5a7e36a3e9f9fc91bf693725c197ae94f4964",
        )

    def test_historical_comparator_is_metadata_only_and_never_rerun(self):
        configured = self.config["historical_comparator"]
        helper = R4._historical_comparator()

        self.assertEqual(configured["commit"], "10bde565d30d119a68e8460757d979b1c35e1b8f")
        self.assertEqual(configured["best_dev_loss"], 10.0844607353)
        self.assertEqual(configured["elapsed_seconds"], 11233.8007)
        self.assertFalse(configured["rerun"])
        self.assertEqual(helper["git_commit"], configured["commit"])
        self.assertEqual(helper["scheduled_dev_loss"], configured["best_dev_loss"])
        self.assertEqual(helper["elapsed_seconds"], configured["elapsed_seconds"])
        self.assertEqual(helper["role"], "metadata_only_not_loaded_or_rerun")


class R4EntrypointExecutionTest(unittest.TestCase):
    def test_help_subprocess_succeeds_from_repository_root(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("Teacher-free R4-FreePixel DreamLite transition training", result.stdout)
        self.assertIn("--smoke", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_main_execution_order_reaches_completion_without_touching_cuda(self):
        events: list[str] = []

        def marked(name, value=None):
            def call(*_args, **_kwargs):
                events.append(name)
                return value

            return call

        args = SimpleNamespace(strict_determinism=False, seed=0)
        budget = R4.resolve_budget(_budget_args(max_optimizer_steps=2, set_warmup_optimizer_steps=0, smoke=True))
        data = SimpleNamespace(all_transitions=(), audit={}, warmup_pool=(), trainable=())
        runtime = SimpleNamespace(
            updater_device="mock-cuda:0",
            reader_device="mock-cuda:1",
            model=object(),
            pipe=object(),
            reader=object(),
            optimizer=object(),
            named_trainable=(),
            trainable=(),
        )
        state = SimpleNamespace(
            start_step=0,
            prior_elapsed=0.0,
            identity_scale=0.25,
            m0={"mean_episode_listwise_choice_ce": 1.5},
            manifest={"bound": True},
        )
        static_manifest = {"transition_index_sha256": "a" * 64}
        final = {"mean_episode_listwise_choice_ce": 1.25}

        def reader_fn(_runtime, _args, *, require_grad):
            events.append("reader_train" if require_grad else "reader_eval")
            return "train-reader" if require_grad else "eval-reader"

        def reset_peak(device):
            events.append(f"reset_peak:{device}")

        with (
            mock.patch.object(R4, "parse_args", side_effect=marked("parse", args)),
            mock.patch.object(R4, "_startup", side_effect=marked("startup", budget)),
            mock.patch.object(R4, "set_all_seeds", side_effect=marked("seed")),
            mock.patch.object(R4, "_load_data", side_effect=marked("data", data)),
            mock.patch.object(R4, "make_static_manifest", side_effect=marked("manifest", static_manifest)),
            mock.patch.object(R4, "_load_runtime", side_effect=marked("runtime", runtime)),
            mock.patch.object(R4, "_reader_fn", side_effect=reader_fn),
            mock.patch.object(R4, "_prepare_run_state", side_effect=marked("prepare", state)),
            mock.patch.object(R4.torch.cuda, "reset_peak_memory_stats", side_effect=reset_peak),
            mock.patch.object(R4, "_run_training", side_effect=marked("train", 12.0)),
            mock.patch.object(
                R4,
                "_finish_evaluation",
                side_effect=marked("finish_eval", (Path("endpoint.pt"), final)),
            ),
            mock.patch.object(R4, "_save_last", side_effect=marked("save_last", Path("last.pt"))),
            mock.patch.object(R4, "_write_summary", side_effect=marked("summary")),
            mock.patch("builtins.print"),
        ):
            result = R4.main(["mocked-smoke"])

        self.assertEqual(result, 0)
        self.assertEqual(
            events,
            [
                "parse",
                "startup",
                "seed",
                "data",
                "manifest",
                "runtime",
                "reader_train",
                "reader_eval",
                "prepare",
                "reset_peak:mock-cuda:0",
                "reset_peak:mock-cuda:1",
                "train",
                "finish_eval",
                "save_last",
                "summary",
            ],
        )

    def test_main_guard_is_the_absolute_source_tail(self):
        source = SCRIPT.read_text(encoding="utf-8").rstrip()
        tail = 'if __name__ == "__main__":\n    raise SystemExit(main())'
        self.assertEqual(source.count('if __name__ == "__main__":'), 1)
        self.assertTrue(source.endswith(tail))


if __name__ == "__main__":
    unittest.main()
