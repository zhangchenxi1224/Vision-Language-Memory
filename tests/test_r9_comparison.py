from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiments" / "compare_r9_individual_learnability.py"
SPEC = importlib.util.spec_from_file_location("r9_comparison_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
r9 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r9
SPEC.loader.exec_module(r9)

RENDER_SCRIPT = ROOT / "scripts" / "reporting" / "render_r9_individual_learnability_report.py"
RENDER_SPEC = importlib.util.spec_from_file_location("r9_renderer_under_test", RENDER_SCRIPT)
assert RENDER_SPEC is not None and RENDER_SPEC.loader is not None
renderer = importlib.util.module_from_spec(RENDER_SPEC)
sys.modules[RENDER_SPEC.name] = renderer
RENDER_SPEC.loader.exec_module(renderer)


def sha256(path: Path) -> str:
    return r9._sha256(path)


def make_target(root: Path, index: int, *, passed: bool) -> None:
    target = root / f"target-{index:02d}"
    run = target / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    relative_change = -0.25 if passed else -0.05
    statistics = {
        "target_segment_id": r9.TARGETS[index][1],
        "m0_normal_mean_ce": 10.0,
        "endpoint_normal_mean_ce": 10.0 * (1.0 + relative_change),
        "delta_ce": 10.0 * relative_change,
        "relative_change": relative_change,
        "per_view_delta_ce": {str(view): -1.0 if passed else (1.0 if view == 3 else -0.1) for view in range(4)},
        "improved_choice_views": 4 if passed else 3,
        "m0_normal_accuracy": 0.0,
        "endpoint_normal_accuracy": 0.25 if passed else 0.0,
        "accuracy_delta": 0.25 if passed else 0.0,
        "normal_reset_difference_in_differences": -0.5 if passed else 0.1,
        "condition_mean_ce": {},
        "bootstrap_ci_used": False,
    }
    summary = {
        "schema": r9.SUMMARY_SCHEMA,
        "status": "completed",
        "implementation_revision": "fixed",
        "git_commit": "a" * 40,
        "target_index": index,
        "target_segment_id": r9.TARGETS[index][1],
        "target_family": r9.TARGETS[index][0],
        "selected_segments": [segment for _family, segment in r9.TARGETS],
        "selected_segments_sha256": r9.SELECTED_SEGMENTS_SHA256,
        "gradient_aggregation": "single-target-one-eighth",
        "gradient_coefficient": 0.125,
        "executed_micro_segments": 128,
        "schedule_cursor_segments": 1024,
        "full_success_claim_allowed": False,
        "checkpoint_steps_observed": [0, 32, 64, 96, 128],
        "training_summary": {"clip_rate": 0.0},
        "aggregation_technical_gate": {
            "passed": True,
            "minimum_raw_vs_applied_cosine": 1.0,
            "maximum_scale_relative_error": 0.0,
        },
        "target_statistics": statistics,
        "gates": {
            "technical_gate": True,
            "target_individual_learnability_gate": passed,
            "formal_success_gate": False,
        },
        "wall_clock_seconds": 100.0,
    }
    summary_path = run / "r9_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    terminal = {
        "schema": r9.TERMINAL_SCHEMA,
        "status": "completed_diagnostic",
        "passed": True,
        "scientific_success_claim": False,
        "target_individual_learnability_gate": passed,
        "target_index": index,
        "target_segment_id": r9.TARGETS[index][1],
        "child_exit_code": 0,
        "summary_sha256": sha256(summary_path),
    }
    (target / "terminal.json").write_text(json.dumps(terminal), encoding="utf-8")
    (target / "launch.json").write_text("{}", encoding="utf-8")
    metrics = []
    for step in range(1, 129):
        metrics.append(
            json.dumps(
                {
                    "kind": "optimizer_step",
                    "optimizer_step": step,
                    "learning_rate": 3e-5,
                    "loss_mean": 10.0 - step / 256.0,
                    "gradient_norm_before_clip": 0.5 + step / 512.0,
                    "gradient_clipped": False,
                    "state_gradient_nonzero_fraction": 1.0,
                    "optimizer_diagnostics": {
                        "updates_after_step": {"global": {"update_weight_ratio": 1e-3}}
                    },
                }
            )
        )
    (run / "metrics.jsonl").write_text("\n".join(metrics) + "\n", encoding="utf-8")
    (run / "micro_metrics.jsonl").write_text(
        "\n".join(json.dumps({"micro": step}) for step in range(128)) + "\n",
        encoding="utf-8",
    )
    for relative in (
        "manifest.json",
        "endpoint_ema.pt",
        "endpoint_raw.pt",
        "overfit_evaluation_rows.jsonl",
    ):
        (run / relative).write_text("placeholder", encoding="utf-8")
    artifacts = []
    for path in sorted(value for value in target.rglob("*") if value.is_file()):
        artifacts.append(
            {
                "path": path.relative_to(target).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (target / "artifact_inventory.json").write_text(
        json.dumps({"schema": r9.INVENTORY_SCHEMA, "artifacts": artifacts}),
        encoding="utf-8",
    )


class R9ComparisonTest(unittest.TestCase):
    def _root(self, directory: str, passing: set[int]) -> Path:
        root = Path(directory) / "run"
        for index in range(8):
            make_target(root, index, passed=index in passing)
        return root

    def test_locked_decision_boundaries(self):
        self.assertEqual(r9._decision(0)[0], "reject_batch_aggregation_reopen_recurrent_alignment_and_temporal_credit")
        self.assertEqual(r9._decision(3)[0], "transition_heterogeneity_repair_failing_structural_property")
        self.assertEqual(
            r9._decision(8)[0],
            "all_targets_individually_learnable_diagnose_realized_shared_update_interference",
        )

    def test_aggregate_zero_passes_and_write_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = self._root(directory, set())
            output = Path(directory) / "output"
            result = r9.compare(run_root, output)
            self.assertEqual(result["pass_count"], 0)
            self.assertFalse(result["formal_success_claim"])
            self.assertTrue((output / "comparison.json").is_file())
            self.assertTrue((output / "target_summary.csv").is_file())
            self.assertTrue((output / "REPORT.md").is_file())
            self.assertTrue((output / "DELIVERY_MANIFEST.json").is_file())

    def test_aggregate_partial_and_all_passes(self):
        for passing in ({0, 2, 4}, set(range(8))):
            with self.subTest(passing=passing), tempfile.TemporaryDirectory() as directory:
                run_root = self._root(directory, passing)
                result = r9.compare(run_root, Path(directory) / "output")
                self.assertEqual(result["pass_count"], len(passing))
                self.assertEqual(result["passed_target_indices"], sorted(passing))

    def test_tampered_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = self._root(directory, set())
            (run_root / "target-03" / "run" / "metrics.jsonl").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "size/SHA"):
                r9.compare(run_root, Path(directory) / "output")

    def test_renderer_writes_loss_and_endpoint_figures_with_refreshed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = self._root(directory, {0, 4})
            output = Path(directory) / "output"
            analysis = renderer.render(run_root, output)
            self.assertEqual(analysis["pass_count"], 2)
            for name in (
                "training_metrics.csv",
                "training_diagnostics.png",
                "endpoint_metrics.png",
                "RAW_ARTIFACTS.json",
                "ANALYSIS.json",
                "DELIVERY_MANIFEST.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            delivery = json.loads((output / "DELIVERY_MANIFEST.json").read_text(encoding="utf-8"))
            names = {record["path"] for record in delivery["artifacts"]}
            self.assertIn("training_diagnostics.png", names)
            self.assertIn("endpoint_metrics.png", names)


if __name__ == "__main__":
    unittest.main()
