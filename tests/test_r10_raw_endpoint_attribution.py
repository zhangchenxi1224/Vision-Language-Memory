from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiments" / "evaluate_r10_raw_endpoint.py"
SPEC = importlib.util.spec_from_file_location("r10_raw_attribution_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
raw = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = raw
SPEC.loader.exec_module(raw)

RENDER_SCRIPT = ROOT / "scripts" / "reporting" / "render_r10_raw_endpoint_attribution.py"
RENDER_SPEC = importlib.util.spec_from_file_location("r10_raw_renderer_under_test", RENDER_SCRIPT)
assert RENDER_SPEC is not None and RENDER_SPEC.loader is not None
renderer = importlib.util.module_from_spec(RENDER_SPEC)
sys.modules[RENDER_SPEC.name] = renderer
RENDER_SPEC.loader.exec_module(renderer)


class R10RawEndpointAttributionTest(unittest.TestCase):
    def _args(self, directory: str):
        root = Path(directory)
        return raw.build_parser().parse_args(
            (
                "--target-index",
                "3",
                "--r10-run-root",
                str(root / "r10"),
                "--train",
                str(root / "train.jsonl"),
                "--dev",
                str(root / "dev.jsonl"),
                "--dreamlite",
                str(root / "dreamlite"),
                "--reader",
                str(root / "reader"),
                "--output-dir",
                str(root / "output"),
                "--expected-analysis-commit",
                "a" * 40,
            )
        )

    def test_runtime_contract_reuses_frozen_r10_single_set_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(directory)
            runtime = raw._runtime_args(args)
            self.assertEqual(runtime.target_index, 3)
            self.assertEqual(runtime.gradient_aggregation, "single-target-full")
            self.assertEqual(runtime.gradient_mode, "full")
            self.assertEqual(runtime.gradient_clip, 10.0)
            self.assertEqual(runtime.lora_rank, 4)
            self.assertEqual(runtime.max_optimizer_steps, 128)
            self.assertTrue(runtime.strict_determinism)

    def test_jsonl_writer_is_atomic_and_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = [{"checkpoint": "m0", "ce": 1.0}, {"checkpoint": raw.RAW_LABEL, "ce": 0.5}]
            raw._write_jsonl(path, rows)
            observed = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(observed, rows)
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())

    def test_plan_forbids_raw_endpoint_from_rescuing_ema_gate(self):
        plan = (ROOT / "reports" / "r10-dreamlite-raw-endpoint-attribution-plan-20260831.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("not allowed to replace that endpoint", plan)
        self.assertIn("all eight immutable R10 F1 targets", plan)
        self.assertIn("before any raw endpoint was evaluated", plan)

    def test_locked_attribution_decisions(self):
        self.assertEqual(renderer._decision(8)[0], "ema_lag_is_sufficient_endpoint_bottleneck")
        self.assertEqual(
            renderer._decision(3)[0],
            "ema_contributes_but_updater_remains_insufficient_run_vae_latent_oracle",
        )
        self.assertEqual(
            renderer._decision(0)[0],
            "ema_is_not_sufficient_explanation_run_vae_latent_oracle",
        )


if __name__ == "__main__":
    unittest.main()
