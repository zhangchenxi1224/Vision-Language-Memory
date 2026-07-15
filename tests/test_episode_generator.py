from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import (  # noqa: E402
    DatasetSizes,
    DatasetValidationError,
    generate_dataset,
    validate_dataset,
)


class EpisodeGeneratorTest(unittest.TestCase):
    def test_generation_is_deterministic_balanced_and_valid(self):
        sizes = DatasetSizes(train=16, dev=16, test_id=16, test_ood=16)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            manifest_a = generate_dataset(Path(first), sizes=sizes, seed=17)
            manifest_b = generate_dataset(Path(second), sizes=sizes, seed=17)
            hashes_a = {key: value["sha256"] for key, value in manifest_a["splits"].items()}
            hashes_b = {key: value["sha256"] for key, value in manifest_b["splits"].items()}
            self.assertEqual(hashes_a, hashes_b)

            report = validate_dataset(Path(first), expected_sizes=sizes.as_dict())
            self.assertTrue(report.valid)
            self.assertEqual(report.total_episodes, 64)
            for split in report.splits.values():
                self.assertLessEqual(split.max_target_position_deviation, 0.02)
                self.assertEqual(set(split.event_kind_counts), {"set", "overwrite", "clear", "noop"})

    def test_validator_rejects_serialized_hidden_ledger(self):
        sizes = DatasetSizes(train=8, dev=8, test_id=8, test_ood=16)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generate_dataset(root, sizes=sizes, seed=3)
            path = root / "train.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(lines[0])
            first["hidden_ledger"] = {"current": "red"}
            lines[0] = json.dumps(first)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DatasetValidationError, "Hidden ledger"):
                validate_dataset(root, verify_manifest_hashes=False)


if __name__ == "__main__":
    unittest.main()
