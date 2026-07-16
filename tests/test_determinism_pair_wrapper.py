from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probes import run_lightweight_determinism_pair as pair  # noqa: E402
from vision_memory.repro import REQUIRED_DETERMINISM_ENV  # noqa: E402


class DeterminismPairWrapperTest(unittest.TestCase):
    def test_second_replica_runs_after_first_replica_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "pair"
            arguments = argparse.Namespace(
                train=Path(temporary) / "train.jsonl",
                reader=Path(temporary) / "reader",
                output_dir=output_dir,
                steps=1,
                device="cuda:0",
            )
            environment = {
                **REQUIRED_DETERMINISM_ENV,
                "SLURM_JOB_ID": "12345",
                "CUDA_VISIBLE_DEVICES": "0",
            }
            failures = [
                subprocess.CompletedProcess(args=["replica-a"], returncode=17),
                subprocess.CompletedProcess(args=["replica-b"], returncode=19),
            ]
            with (
                mock.patch.object(pair, "parse_args", return_value=arguments),
                mock.patch.dict(os.environ, environment, clear=False),
                mock.patch.object(pair.subprocess, "run", side_effect=failures) as run,
                mock.patch("builtins.print"),
            ):
                returncode = pair.main()

            self.assertEqual(returncode, 1)
            self.assertEqual(run.call_count, 2)
            report = json.loads((output_dir / "pair_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["valid"])
            self.assertEqual(report["children"]["a"]["returncode"], 17)
            self.assertEqual(report["children"]["b"]["returncode"], 19)


if __name__ == "__main__":
    unittest.main()
