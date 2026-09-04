from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiments" / "compare_r12_shared_writer.py"
SPEC = importlib.util.spec_from_file_location("compare_r12_shared_writer_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)

COLLAPSE_SCRIPT = ROOT / "scripts" / "analysis" / "analyze_r12_shared_writer_collapse.py"
COLLAPSE_SPEC = importlib.util.spec_from_file_location(
    "analyze_r12_shared_writer_collapse_under_test", COLLAPSE_SCRIPT
)
assert COLLAPSE_SPEC is not None and COLLAPSE_SPEC.loader is not None
collapse = importlib.util.module_from_spec(COLLAPSE_SPEC)
sys.modules[COLLAPSE_SPEC.name] = collapse
COLLAPSE_SPEC.loader.exec_module(collapse)

RENDERER_SCRIPT = ROOT / "scripts" / "reporting" / "render_r12_shared_writer_report.py"
RENDERER_SPEC = importlib.util.spec_from_file_location(
    "render_r12_shared_writer_report_under_test", RENDERER_SCRIPT
)
assert RENDERER_SPEC is not None and RENDERER_SPEC.loader is not None
renderer = importlib.util.module_from_spec(RENDERER_SPEC)
sys.modules[RENDERER_SPEC.name] = renderer
RENDERER_SPEC.loader.exec_module(renderer)


def arm(*, gate: bool, train: int, dev_select: int = 0, dev_final: int = 0) -> dict:
    return {
        "summary": {
            "gates": {
                "arm_gate": gate,
                "split_target_pass_counts": {
                    "train_audit": train,
                    "dev_select": dev_select,
                    "dev_final": dev_final,
                },
            }
        }
    }


class R12ReportingTest(unittest.TestCase):
    def test_decision_requires_conditioned_all_pass_and_control_fail(self):
        decision, _reason, passed = comparison._decision(
            arm(gate=True, train=36, dev_select=24, dev_final=24),
            arm(gate=False, train=0),
        )
        self.assertEqual(decision, "advance_to_recurrent_state_algebra")
        self.assertTrue(passed)

        decision, _reason, passed = comparison._decision(
            arm(gate=True, train=36, dev_select=24, dev_final=24),
            arm(gate=True, train=36, dev_select=24, dev_final=24),
        )
        self.assertEqual(decision, "reject_universal_image_false_positive")
        self.assertFalse(passed)

    def test_failure_localization_preserves_locked_interpretation(self):
        decision, _reason, passed = comparison._decision(
            arm(gate=False, train=36, dev_select=23, dev_final=20),
            arm(gate=False, train=0),
        )
        self.assertEqual(decision, "diagnose_shared_writer_generalization")
        self.assertFalse(passed)

        decision, _reason, passed = comparison._decision(
            arm(gate=False, train=35),
            arm(gate=False, train=0),
        )
        self.assertEqual(decision, "diagnose_shared_writer_fit_boundary")
        self.assertFalse(passed)

    def test_required_artifacts_bind_every_checkpoint_and_evaluation_split(self):
        required = comparison._required_artifacts()
        for step in comparison.EXPECTED_CHECKPOINT_STEPS:
            self.assertIn(f"run/checkpoints/step-{step:04d}.pt", required)
            self.assertIn(f"run/checkpoint_diagnostics/step-{step:04d}.json", required)
        for split in comparison.SPLITS:
            self.assertIn(f"run/{split}_evaluation_rows.jsonl", required)
            self.assertIn(f"run/{split}_statistics.json", required)

    def test_inventory_rejects_unlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = root / "bound.txt"
            bound.write_text("bound", encoding="utf-8")
            extra = root / "extra.txt"
            extra.write_text("extra", encoding="utf-8")
            (root / "artifact_inventory.json").write_text(
                json.dumps(
                    {
                        "schema": comparison.INVENTORY_SCHEMA,
                        "artifacts": [
                            {
                                "path": "bound.txt",
                                "bytes": bound.stat().st_size,
                                "sha256": hashlib.sha256(bound.read_bytes()).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inventory/file-set mismatch"):
                comparison._validate_inventory(root)

    def test_success_boundary_remains_diagnostic(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"formal_success_claim": False', source)
        self.assertIn("recurrent composition", source)
        self.assertEqual(
            comparison.EXPECTED_COMMIT,
            "c401954e5624e99347306a60c3f86202c941ab34",
        )

    def test_moving_mean_uses_only_past_and_current_steps(self):
        self.assertEqual(renderer._moving_mean([1.0, 3.0, 5.0], window=2), [1.0, 2.0, 4.0])

    def test_collapse_probe_recovers_linearly_separable_target_values(self):
        segment_ids = ("a1", "a2", "b1", "b2", "a3", "b3", "a4", "b4", "a5", "b5")
        values = torch.tensor(
            [
                [2.0, 0.0],
                [1.8, 0.1],
                [0.0, 2.0],
                [0.1, 1.8],
                [1.9, 0.0],
                [0.0, 1.9],
                [2.1, 0.1],
                [0.1, 2.1],
                [1.7, 0.0],
                [0.0, 1.7],
            ]
        )
        labels = {segment_id: segment_id[0] for segment_id in segment_ids}
        split_ids = {
            "train": list(segment_ids[:4]),
            "train_audit": list(segment_ids[4:6]),
            "dev_select": list(segment_ids[6:8]),
            "dev_final": list(segment_ids[8:10]),
        }
        result = collapse._ridge_probe(values, segment_ids, labels, split_ids)
        self.assertEqual(result["accuracy"], {split: 1.0 for split in collapse.SPLITS})

    def test_pairwise_audit_separates_same_and_different_targets(self):
        result = collapse._pairwise_summary(
            torch.tensor([[0.0, 0.0], [0.1, 0.0], [3.0, 0.0], [3.1, 0.0]]),
            ["a", "a", "b", "b"],
        )
        self.assertLess(result["same_target_mean_l2"], result["different_target_mean_l2"])


if __name__ == "__main__":
    unittest.main()
