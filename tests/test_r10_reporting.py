from __future__ import annotations

import base64
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiments" / "compare_r10_visual_alignment.py"
SPEC = importlib.util.spec_from_file_location("r10_comparison_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
r10 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = r10
SPEC.loader.exec_module(r10)

RENDER_SCRIPT = ROOT / "scripts" / "reporting" / "render_r10_visual_alignment_report.py"
RENDER_SPEC = importlib.util.spec_from_file_location("r10_renderer_under_test", RENDER_SCRIPT)
assert RENDER_SPEC is not None and RENDER_SPEC.loader is not None
renderer = importlib.util.module_from_spec(RENDER_SPEC)
sys.modules[RENDER_SPEC.name] = renderer
RENDER_SPEC.loader.exec_module(renderer)


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _write(path: Path, value: str = "placeholder") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def make_target(root: Path, arm: str, index: int, *, passed: bool) -> None:
    target = root / arm / f"target-{index:02d}"
    run = target / "run"
    run.mkdir(parents=True)
    _write(target / "launch.json", "{}")
    _write(run / "manifest.json", "{}")
    _write(run / "target_evaluation_rows.jsonl", "{}\n")
    _write(run / "endpoint_raw.pt")
    statistics = {
        "target_segment_id": r10.R10_TARGET_IDS[index],
        "m0_normal_mean_ce": 10.0,
        "endpoint_normal_mean_ce": 7.5 if passed else 9.5,
        "delta_ce": -2.5 if passed else -0.5,
        "relative_change": -0.25 if passed else -0.05,
        "per_view_delta_ce": {str(view): -0.5 if passed else (-0.1 if view < 3 else 0.1) for view in range(4)},
        "improved_choice_views": 4 if passed else 3,
        "m0_normal_accuracy": 0.0,
        "endpoint_normal_accuracy": 0.25 if passed else 0.0,
        "accuracy_delta": 0.25 if passed else 0.0,
        "normal_reset_difference_in_differences": -0.5 if passed else 0.1,
        "condition_mean_ce": {},
        "bootstrap_ci_used": False,
    }
    metrics = []
    for step in range(1, 129):
        if arm == "direct-pixel-oracle":
            metrics.append(
                {
                    "kind": "optimizer_step",
                    "optimizer_step": step,
                    "learning_rate": 0.05,
                    "loss_before_step": max(0.001, 10.0 - step / 16),
                    "gradient_norm": 0.5,
                    "gradient_nonzero_fraction": 1.0,
                    "image_min_after_step": 0.25,
                    "image_max_after_step": 0.75,
                    "image_saturation_fraction_after_step": 0.0,
                }
            )
        else:
            metrics.append(
                {
                    "kind": "optimizer_step",
                    "optimizer_step": step,
                    "learning_rate": 3e-5,
                    "loss_mean": 10.0 - step / 256,
                    "gradient_norm_before_clip": 0.5,
                    "gradient_clipped": False,
                    "state_gradient_nonzero_fraction": 1.0,
                    "image_min": 0.1,
                    "image_max": 0.9,
                    "image_saturation_fraction_mean": 0.0,
                    "gradient_aggregation": {
                        "mode": "single-target-full",
                        "gradient_coefficient": 1.0,
                    },
                    "optimizer_diagnostics": {
                        "updates_after_step": {"global": {"update_weight_ratio": 1e-3}}
                    },
                }
            )
    _write(run / "metrics.jsonl", "\n".join(json.dumps(row) for row in metrics) + "\n")
    artifacts: dict[str, str]
    if arm == "direct-pixel-oracle":
        (run / "endpoint_raw.png").write_bytes(PNG_1X1)
        _write(run / "snapshot_end_verification.json", "{}")
        artifacts = {
            "manifest_sha256": r10._sha256(run / "manifest.json"),
            "metrics_sha256": r10._sha256(run / "metrics.jsonl"),
            "evaluation_rows_sha256": r10._sha256(run / "target_evaluation_rows.jsonl"),
            "endpoint_raw_sha256": r10._sha256(run / "endpoint_raw.pt"),
            "endpoint_png_sha256": r10._sha256(run / "endpoint_raw.png"),
            "snapshot_end_verification_sha256": r10._sha256(run / "snapshot_end_verification.json"),
        }
        summary_file = "r10_pixel_summary.json"
        summary_schema = r10.SUMMARY_SCHEMAS[arm]
        arm_fields = {"optimizer_steps": 128}
    else:
        _write(run / "micro_metrics.jsonl", "\n".join("{}" for _ in range(128)) + "\n")
        _write(run / "endpoint_ema.pt")
        artifacts = {
            "manifest_sha256": r10._sha256(run / "manifest.json"),
            "metrics_sha256": r10._sha256(run / "metrics.jsonl"),
            "micro_metrics_sha256": r10._sha256(run / "micro_metrics.jsonl"),
            "evaluation_rows_sha256": r10._sha256(run / "target_evaluation_rows.jsonl"),
            "endpoint_ema_sha256": r10._sha256(run / "endpoint_ema.pt"),
            "endpoint_raw_sha256": r10._sha256(run / "endpoint_raw.pt"),
        }
        summary_file = "r10_dreamlite_summary.json"
        summary_schema = r10.SUMMARY_SCHEMAS[arm]
        arm_fields = {
            "gradient_aggregation": "single-target-full",
            "gradient_coefficient": 1.0,
            "training_summary": {"clip_rate": 0.0},
        }
    summary = {
        "schema": summary_schema,
        "status": "completed",
        "protocol": r10.R10_PROTOCOL,
        "implementation_revision": r10.IMPLEMENTATION_REVISIONS[arm],
        "git_commit": r10.EXPECTED_GIT_COMMIT,
        "arm": arm,
        "target_index": index,
        "target_family": "F1",
        "target_segment_id": r10.R10_TARGET_IDS[index],
        "selected_segments": list(r10.R10_TARGET_IDS),
        "selected_segments_sha256": r10.R10_SELECTED_SEGMENTS_SHA256,
        "checkpoint_steps_observed": [0, 32, 64, 96, 128],
        "technical_gate": {"passed": True},
        "target_statistics": statistics,
        "gates": {
            "technical_gate": True,
            "target_lower_bound_gate": passed,
            "formal_success_gate": False,
        },
        "diagnostic_only_not_formal_success": True,
        "full_success_claim_allowed": False,
        "artifacts": artifacts,
        "wall_clock_seconds": 100.0,
        **arm_fields,
    }
    summary_path = run / summary_file
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    terminal = {
        "schema": r10.TERMINAL_SCHEMA,
        "status": "completed_diagnostic",
        "passed": True,
        "scientific_success_claim": False,
        "arm": arm,
        "target_index": index,
        "target_segment_id": r10.R10_TARGET_IDS[index],
        "target_lower_bound_gate": passed,
        "child_exit_code": 0,
        "checks": {"all": True},
        "summary_sha256": r10._sha256(summary_path),
        "manifest_sha256": r10._sha256(run / "manifest.json"),
    }
    (target / "terminal.json").write_text(json.dumps(terminal), encoding="utf-8")
    inventory = []
    for path in sorted(value for value in target.rglob("*") if value.is_file()):
        inventory.append(
            {
                "path": path.relative_to(target).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": r10._sha256(path),
            }
        )
    (target / "artifact_inventory.json").write_text(
        json.dumps({"schema": r10.INVENTORY_SCHEMA, "artifacts": inventory}),
        encoding="utf-8",
    )


def make_root(directory: str, *, pixel: set[int], dreamlite: set[int]) -> Path:
    root = Path(directory) / "run"
    for arm, passing in (
        ("direct-pixel-oracle", pixel),
        ("dreamlite-single-set", dreamlite),
    ):
        for index in range(8):
            make_target(root, arm, index, passed=index in passing)
    return root


class R10ReportingTest(unittest.TestCase):
    def test_locked_decision_boundaries(self):
        self.assertEqual(r10._decision(8, 8)[0], "advance_shared_f1_train_heldout_alignment")
        self.assertEqual(r10._decision(8, 3)[0], "redesign_dreamlite_updater_only")
        self.assertEqual(r10._decision(4, 8)[0], "diagnose_target_dependent_visual_channel")
        self.assertEqual(r10._decision(0, 0)[0], "test_post_resize_pixel_token_oracle")

    def test_aggregate_two_arms_and_write_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = make_root(directory, pixel=set(range(8)), dreamlite={0, 2})
            output = Path(directory) / "output"
            result = r10.compare(run_root, output)
            self.assertEqual(result["arm_pass_counts"], {"direct-pixel-oracle": 8, "dreamlite-single-set": 2})
            self.assertEqual(result["decision"], "redesign_dreamlite_updater_only")
            self.assertFalse(result["formal_success_claim"])
            for name in ("comparison.json", "target_summary.csv", "REPORT.md", "DELIVERY_MANIFEST.json"):
                self.assertTrue((output / name).is_file(), name)

    def test_tampered_artifact_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = make_root(directory, pixel=set(), dreamlite=set())
            (run_root / "direct-pixel-oracle" / "target-03" / "run" / "metrics.jsonl").write_text(
                "tampered", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "hash binding|size/SHA"):
                r10.compare(run_root, Path(directory) / "output")

    def test_renderer_writes_figures_and_refreshes_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = make_root(directory, pixel=set(range(8)), dreamlite={0, 4})
            output = Path(directory) / "output"
            analysis = renderer.render(run_root, output)
            self.assertEqual(analysis["arm_pass_counts"]["direct-pixel-oracle"], 8)
            for name in (
                "training_metrics.csv",
                "training_diagnostics.png",
                "endpoint_metrics.png",
                "pixel_endpoint_contact_sheet.png",
                "RAW_ARTIFACTS.json",
                "ANALYSIS.json",
                "DELIVERY_MANIFEST.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            delivery = json.loads((output / "DELIVERY_MANIFEST.json").read_text(encoding="utf-8"))
            names = {record["path"] for record in delivery["artifacts"]}
            self.assertIn("pixel_endpoint_contact_sheet.png", names)


if __name__ == "__main__":
    unittest.main()
