from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cluster"))

from collect_slurm_results import build_report, parse_sacct  # noqa: E402


class SlurmLedgerTest(unittest.TestCase):
    def test_parse_sacct_selects_parent_row(self):
        text = "123|COMPLETED|3600|0:0|cpu=8,gres/gpu=2\n123.batch|COMPLETED|3590|0:0|cpu=8\n"
        parsed = parse_sacct(text, "123")
        self.assertEqual(parsed["state"], "COMPLETED")
        self.assertEqual(parsed["elapsed_seconds"], 3600)

    def test_report_accounts_gpu_hours_and_failed_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            (run / "logs").mkdir(parents=True)
            (run / "results").mkdir()
            (run / "logs" / "train_10.out").write_text("ok\n", encoding="utf-8")
            (run / "logs" / "train_10.err").write_text("", encoding="utf-8")
            (run / "results" / "result.json").write_text("{}\n", encoding="utf-8")
            manifest = root / "submission.json"
            manifest.write_text(
                json.dumps(
                    {
                        "run_root": str(run),
                        "commit": "abc",
                        "jobs": {
                            "train": {
                                "job_id": "10",
                                "stage": "full",
                                "resources": {"gpus": 2},
                            },
                            "score": {
                                "job_id": "11",
                                "stage": "eval",
                                "resources": {"gpus": 0},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            rows = {
                "10": "10|COMPLETED|3600|0:0|cpu=24,gres/gpu=2\n",
                "11": "11|FAILED|60|1:0|cpu=12\n",
            }
            report = build_report(manifest, sacct_query=rows.__getitem__)
            self.assertEqual(report["summary"]["gpu_hours"], 2.0)
            self.assertEqual(report["summary"]["failures"], 1)
            self.assertFalse(report["summary"]["all_completed"])
            self.assertEqual(len(report["result_artifacts"]), 1)


if __name__ == "__main__":
    unittest.main()
