from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative: str):
    path = ROOT / relative
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


comparison = load_script(
    "r11_comparison_under_test", "scripts/experiments/compare_r11_vae_latent.py"
)
renderer = load_script(
    "r11_renderer_under_test", "scripts/reporting/render_r11_vae_latent_report.py"
)


class R11ReportingTest(unittest.TestCase):
    def test_decision_is_locked_to_eight_target_oracle_gate(self):
        self.assertEqual(
            comparison._decision(8)[0],
            "replace_semantic_editor_with_shared_event_to_latent_writer",
        )
        for count in range(1, 8):
            self.assertEqual(
                comparison._decision(count)[0],
                "attribute_target_dependent_vae_readability",
            )
        self.assertEqual(
            comparison._decision(0)[0],
            "augment_or_bypass_vae_with_image_residual_codec",
        )

    def test_required_artifacts_bind_every_fixed_checkpoint_and_image(self):
        required = comparison._required_artifacts()
        for step in (0, 64, 128, 192, 256):
            self.assertIn(f"run/checkpoints/step-{step:03d}.pt", required)
            self.assertIn(f"run/images/step-{step:03d}.png", required)
        self.assertIn("run/endpoint_raw.pt", required)
        self.assertIn("run/target_evaluation_rows.jsonl", required)

    def test_inventory_rejects_unlisted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = root / "bound.txt"
            bound.write_text("bound", encoding="utf-8")
            extra = root / "extra.txt"
            extra.write_text("unlisted", encoding="utf-8")
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

    def test_moving_mean_uses_only_past_and_current_values(self):
        self.assertEqual(renderer._moving_mean([1.0, 3.0, 5.0], window=2), [1.0, 2.0, 4.0])

    def test_json_domain_normalizes_integer_view_keys_losslessly(self):
        in_memory = {"per_view_delta_ce": {0: -1.0, 1: -2.0}}
        serialized = json.loads(json.dumps(in_memory, sort_keys=True))
        self.assertEqual(serialized, {"per_view_delta_ce": {"0": -1.0, "1": -2.0}})

    def test_success_boundary_remains_diagnostic(self):
        source = (ROOT / "scripts" / "experiments" / "compare_r11_vae_latent.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"formal_success_claim": False', source)
        self.assertIn("shared event-to-memory writer", source)
        self.assertEqual(
            comparison.EXPECTED_GIT_COMMIT,
            "f4a018f3c4eef453fff3367b049ea732332e8c37",
        )


if __name__ == "__main__":
    unittest.main()
