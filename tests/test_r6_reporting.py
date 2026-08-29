from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reporting" / "render_r6_source_anchor_report.py"
SPEC = importlib.util.spec_from_file_location("render_r6_source_anchor_report_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reporting = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reporting
SPEC.loader.exec_module(reporting)


def comparison(m0: float, endpoint: float) -> dict:
    delta = endpoint - m0
    return {
        "estimate": delta,
        "ci95": [delta - 0.1, delta + 0.1],
        "m0_mean_ce": m0,
        "endpoint_mean_ce": endpoint,
        "relative_change": endpoint / m0 - 1.0,
        "improved_pair_units": 8 if delta < 0 else 0,
    }


def make_arm(root: Path, arm: str) -> None:
    run = root / "run"
    run.mkdir(parents=True)
    records = []
    for step in range(1, 129):
        records.append(
            {
                "optimizer_step": step,
                "learning_rate": 1e-4,
                "loss_mean": 10.0 - step / 64,
                "loss_by_family": {"F2": 8.0, "F3": 9.0, "F5": 10.0, "F6": 11.0},
                "gradient_norm_before_clip": 20.0 if arm == "legacy-pure-noise" else 5.0,
                "gradient_clipped": arm == "legacy-pure-noise",
                "state_gradient_nonzero_fraction": 1.0,
                "image_saturation_fraction_mean": 0.01,
                "optimizer_diagnostics": {
                    "updates_after_step": {"global": {"update_weight_ratio": 1e-3}}
                },
            }
        )
    (run / "metrics.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    gradient_records = [
        {
            "index": index,
            "segment_id": f"segment-{index}",
            "family": ("F2", "F2", "F3", "F3", "F5", "F5", "F6", "F6")[index],
            "loss": float(index + 1),
            "gradient_norm": float(index + 1),
            "norm_share_of_sum_of_norms": (index + 1) / 36,
            "cosine_to_raw_batch_gradient": 0.2,
            "cosine_to_unit_balanced_gradient": 0.3,
        }
        for index in range(8)
    ]
    hard8 = comparison(10.0, 7.0)
    summary = {
        "status": "completed",
        "arm": arm,
        "full_success_claim_allowed": False,
        "git_commit": "a" * 40,
        "implementation_revision": "scheduler-effective-sigma-v2",
        "selected_segments_sha256": "b" * 64,
        "edit_start_sigma": 1.0 if arm == "legacy-pure-noise" else 0.5,
        "gradient_conflict_audit": {
            "records": gradient_records,
            "pairwise_cosine": [
                [1.0 if first == second else 0.1 for second in range(8)]
                for first in range(8)
            ],
            "gradient_norm": {"median": 4.5, "max_to_min_ratio": 8.0},
            "off_diagonal": {"negative_fraction": 0.25, "median": 0.1},
            "raw_vs_unit_balanced_cosine": 0.9,
        },
        "comparisons": {
            "train_overfit_hard8_endpoint_vs_m0": hard8,
            "train_overfit_hard8_state_did": {"estimate": -0.5},
            "formal_select_32_endpoint_vs_m0": comparison(12.0, 11.0),
            "mechanism_select_32_endpoint_vs_m0": comparison(11.0, 10.0),
        },
        "overfit_accuracy": {"m0": 0.0, "endpoint": 0.5, "delta": 0.5},
        "gates": {
            "technical_gate": True,
            "hard8_overfit_learnability_gate": True,
            "fixed_dev_generalization_gate": True,
            "formal_success_gate": False,
        },
        "training_summary": {"clip_rate": 1.0 if arm == "legacy-pure-noise" else 0.0},
    }
    (run / "r6_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (root / "terminal.json").write_text(
        json.dumps({"status": "completed_diagnostic", "passed": True}),
        encoding="utf-8",
    )
    paths = (root / "terminal.json", run / "metrics.jsonl", run / "r6_summary.json")
    inventory = {
        "artifacts": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": reporting._sha256(path),
            }
            for path in paths
        ]
    }
    (root / "artifact_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")


class R6ReportingTest(unittest.TestCase):
    def test_render_validates_raw_artifacts_and_emits_complete_portable_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "legacy"
            anchored = root / "anchored"
            make_arm(legacy, "legacy-pure-noise")
            make_arm(anchored, "source-anchored")
            output = root / "delivery"
            analysis = reporting.render(legacy, anchored, output)

            self.assertEqual(analysis["status"], "completed")
            self.assertFalse(analysis["formal_success_claim"])
            expected = {
                "ANALYSIS.json",
                "DELIVERY_MANIFEST.json",
                "REPORT.md",
                "endpoint_metrics.png",
                "endpoint_summary.csv",
                "gradient_conflict.csv",
                "gradient_conflict.png",
                "training_diagnostics.png",
                "training_metrics.csv",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            manifest = json.loads((output / "DELIVERY_MANIFEST.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["artifacts"]), len(expected) - 1)
            for record in manifest["artifacts"]:
                path = output / record["path"]
                self.assertEqual(path.stat().st_size, record["bytes"])
                self.assertEqual(reporting._sha256(path), record["sha256"])


if __name__ == "__main__":
    unittest.main()
