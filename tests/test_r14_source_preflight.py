from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.probes import r14_source_preflight as preflight  # noqa: E402


class R14SourcePreflightTest(unittest.TestCase):
    def test_trainer_args_lock_execution_and_do_not_expose_scientific_tunables(self) -> None:
        cli = argparse.Namespace(
            train=Path("train.jsonl"),
            dev=Path("dev.jsonl"),
            dreamlite=Path("dreamlite"),
            reader=Path("reader"),
            r12_conditioned_root=Path("conditioned"),
            r12_control_root=Path("control"),
            r12_comparison=Path("comparison.json"),
            r12_collapse_audit=Path("collapse.json"),
            r13_root=Path("r13"),
            report=Path("preflight/report.json"),
            expected_commit="a" * 40,
        )
        args = preflight._trainer_args(cli)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.pairing_seed, 20260904)
        self.assertTrue(args.strict_determinism)
        self.assertFalse(args.allow_dirty)
        self.assertEqual(args.dreamlite_device, "cuda:0")
        self.assertEqual(args.reader_device, "cuda:1")
        self.assertEqual(args.r13_root, Path("r13"))
        self.assertEqual(
            args.output_dir,
            Path("preflight/.r14-source-preflight-output-must-not-exist"),
        )


if __name__ == "__main__":
    unittest.main()
