from __future__ import annotations

import json
import hashlib
import sys
import unittest
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train.r13_centered_residual_writer import (  # noqa: E402
    MeanCenteredResidualWriter,
    _source_coefficients,
)
from vision_memory.data.schema import QuerySpec  # noqa: E402
from vision_memory.training.r5_compose import R5Event, R5Segment  # noqa: E402
from vision_memory.training.r13_centered_residual import (  # noqa: E402
    R13_FRESH_DEV_FINAL_SHA256,
    centered_target_gate,
    centered_target_statistics,
    select_fresh_dev_final_f1,
)


def segment(*, value: str, position: int, serial: int, prefix: str) -> R5Segment:
    choices = [f"distractor-{prefix}-{serial}-{index}" for index in range(4)]
    choices[position] = value
    entity = f"entity-{prefix}-{serial}"
    episode = f"episode-{prefix}-{serial}"
    return R5Segment(
        segment_id=f"{prefix}-{serial:04d}",
        family="F1",
        events=(
            R5Event(
                source_episode_id=episode,
                source_turn_index=0,
                entity_id=entity,
                event_kind="set",
                event_text=f"Entity {serial} now prefers {value}.",
            ),
        ),
        query=QuerySpec(
            text=f"What does entity {serial} prefer?",
            choices=tuple(choices),
            target_index=position,
            comparison_id=f"comparison-{prefix}-{serial}",
            target_token_count=1,
        ),
        query_source_episode_id=episode,
        query_turn_index=1,
        query_entity_id=entity,
        target_event_position=0,
        cross_slot_interference=False,
    )


class R13CenteredResidualContractTest(unittest.TestCase):
    def test_config_locks_parent_sources_fresh_final_and_unchanged_schedule(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "experiments" / "r13_centered_residual_writer.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["status"], "preregistered_before_any_r13_model_outcome")
        self.assertEqual(config["fixed_data"]["dev_final"]["payload_sha256"], R13_FRESH_DEV_FINAL_SHA256)
        self.assertEqual(config["optimization"]["optimizer_steps"], 1152)
        self.assertIsNone(config["optimization"]["gradient_clipping"])
        self.assertEqual(config["optimization"]["backward_loss_divisor"], 1.0)
        self.assertIn("donor", config["evaluation"]["conditions"])
        self.assertIn("base", config["evaluation"]["conditions"])
        self.assertTrue(config["success_boundary"]["diagnostic_only"])
        self.assertEqual(
            config["writer"]["fixed_base_latent_sha256"],
            "11281993533d2db0fcab6b890908bdddc986996552034fe57c8c4f5a432825e8",
        )
        source_prereg = ROOT / "reports" / "r13-source-decomposition-preregistration-20260904.json"
        self.assertEqual(
            hashlib.sha256(source_prereg.read_bytes()).hexdigest(),
            config["activation"]["source_decomposition_preregistration_sha256"],
        )

    def test_fresh_final_selection_is_stable_balanced_and_excludes_entities(self) -> None:
        pool = []
        serial = 0
        for value_index in range(24):
            for position in range(4):
                for _copy in range(2):
                    pool.append(
                        segment(
                            value=f"value-{value_index:02d}",
                            position=position,
                            serial=serial,
                            prefix="fresh",
                        )
                    )
                    serial += 1
        excluded = {pool[0].query_entity_id, pool[3].query_entity_id}
        selected = select_fresh_dev_final_f1(
            pool,
            excluded_entities=excluded,
            enforce_locked=False,
        )
        reversed_selected = select_fresh_dev_final_f1(
            tuple(reversed(pool)),
            excluded_entities=excluded,
            enforce_locked=False,
        )
        self.assertEqual(
            [value.segment_id for value in selected],
            [value.segment_id for value in reversed_selected],
        )
        self.assertEqual(len(selected), 24)
        self.assertEqual(Counter(value.query.target_index for value in selected), Counter({0: 6, 1: 6, 2: 6, 3: 6}))
        self.assertFalse({value.query_entity_id for value in selected} & excluded)

    def test_centered_writer_reconstructs_source_and_forbids_common_residual(self) -> None:
        generator = torch.Generator().manual_seed(13)
        initial = torch.randn((1, 4, 128, 128), generator=generator) * 0.01
        train_features = torch.randn((144, 2048), generator=generator) * 0.02
        source_state = {
            "basis_raw": torch.randn((48, 4, 128, 128), generator=generator),
            "coefficient_mlp.0.weight": torch.randn((512, 2048), generator=generator) * 0.001,
            "coefficient_mlp.0.bias": torch.randn((512,), generator=generator) * 0.001,
            "coefficient_mlp.2.weight": torch.randn((48, 512), generator=generator) * 0.001,
            "coefficient_mlp.2.bias": torch.randn((48,), generator=generator) * 0.001,
        }
        writer = MeanCenteredResidualWriter(
            initial_latent=initial,
            train_features=train_features,
            source_state=source_state,
        )
        feature = train_features[7:8]
        output, _ = writer(feature)
        source_coefficients = _source_coefficients(feature, source_state)
        flat_basis = source_state["basis_raw"].flatten(1)
        unit_basis = flat_basis / flat_basis.double().norm(dim=1).float().unsqueeze(1)
        expected = initial + (80.0 * source_coefficients @ unit_basis).reshape(1, 4, 128, 128)
        self.assertLess(float((output - expected).detach().abs().max()), 1e-5)
        invariants = writer.center_invariants()
        self.assertLess(float(invariants["mean_residual_coefficients"].detach().abs().max()), 1e-6)
        self.assertLess(float(invariants["mean_residual_delta_flat"].detach().abs().max()), 1e-5)
        with torch.no_grad():
            writer.coefficient_mlp[0].bias.add_(0.5)
            writer.basis_raw.mul_(1.3)
        invariants = writer.center_invariants()
        self.assertLess(float(invariants["mean_residual_coefficients"].detach().abs().max()), 1e-6)
        self.assertLess(float(invariants["mean_residual_delta_flat"].detach().abs().max()), 1e-5)
        base_output, diagnostics = writer(feature, conditioned=False)
        self.assertTrue(torch.equal(base_output, writer.fixed_base_latent_fp32))
        self.assertEqual(float(diagnostics["residual_coefficients"].abs().max()), 0.0)

    def test_gate_requires_reset_donor_and_frozen_base_attribution(self) -> None:
        rows = []
        for checkpoint in ("m0", "centered_step1152"):
            for condition in ("normal", "reset", "donor", "base"):
                for view in range(4):
                    if checkpoint == "m0":
                        ce, correct = 10.0, 0
                    elif condition == "normal":
                        ce, correct = 4.0, 1
                    elif condition == "reset":
                        ce, correct = 10.0, 0
                    else:
                        ce, correct = 6.0, 0
                    rows.append(
                        {
                            "suite": "suite",
                            "checkpoint": checkpoint,
                            "condition": condition,
                            "pair_unit": "target",
                            "view_index": view,
                            "ce": ce,
                            "correct": correct,
                        }
                    )
        statistics = centered_target_statistics(
            rows,
            suite="suite",
            target_segment_id="target",
            endpoint="centered_step1152",
        )
        self.assertTrue(centered_target_gate(statistics, technical_gate=True))
        for row in rows:
            if row["checkpoint"] == "centered_step1152" and row["condition"] == "base":
                row["ce"] = 3.0
                row["correct"] = 1
        shortcut = centered_target_statistics(
            rows,
            suite="suite",
            target_segment_id="target",
            endpoint="centered_step1152",
        )
        self.assertFalse(centered_target_gate(shortcut, technical_gate=True))


if __name__ == "__main__":
    unittest.main()
