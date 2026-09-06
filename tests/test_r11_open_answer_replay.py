"""CPU safeguards for the finite R11 replay and fail-closed artifact boundary."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.experiments import run_r11_open_answer_replay as replay


CONFIG_PATH = replay.ROOT / "configs/experiments/r11_open_answer_replay.json"


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.config = replay.load_config(CONFIG_PATH)

    def records(self):
        rows = []
        for case in replay.open_cases(self.config):
            gold = self.config["targets"][case["target_index"]]["scorer_metadata"]["gold"]
            raw = gold if case["condition"] == "matched" else "unknown"
            rows.append({**case, "raw": raw, "scorer": replay.score_short_answer(raw, gold), "truncated": False})
        anchors = [
            {"target_index": target, "view_index": view, "correct": True, "ce": 0.0001,
             "permutation": list(replay.REVERSE_CYCLIC4[view])}
            for target in range(8) for view in range(4)
        ]
        return rows, anchors

    def test_plan_contains_all_48_unique_answer_blind_calls(self):
        cases = replay.open_cases(self.config)
        self.assertEqual(len(cases), 48)
        self.assertEqual(len({(c["target_index"], c["condition"], c["prompt_id"]) for c in cases}), 48)
        for case in cases:
            self.assertFalse({"gold", "scorer_metadata", "choices", "answer_index", "target"} & set(case))
            target = self.config["targets"][case["target_index"]]
            self.assertNotIn(target["scorer_metadata"]["gold"], case["query"].casefold().split())
            self.assertNotIn("Choose exactly one option", case["query"])

    def test_gold_leak_and_truncated_config_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = copy.deepcopy(self.config)
            config["targets"][0]["inputs"]["original_open"] += " The answer is linen."
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Gold answer leaked"):
                replay.load_config(path)
            config = copy.deepcopy(self.config)
            config["targets"].pop()
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly the eight"):
                replay.load_config(path)

    def test_file_hash_is_checked_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / self.config["targets"][0]["artifact"]["endpoint_member"]
            path.parent.mkdir(parents=True)
            path.write_bytes(b"untrusted changed payload")
            with patch.object(replay.torch, "load") as loader:
                with self.assertRaisesRegex(ValueError, "file SHA256 mismatch"):
                    replay.load_endpoint(root, self.config["targets"][0])
                loader.assert_not_called()

    def test_tensor_hash_is_checked_after_weights_only_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = copy.deepcopy(self.config["targets"][0])
            path = root / target["artifact"]["endpoint_member"]
            path.parent.mkdir(parents=True)
            path.write_bytes(b"mock serialized tensor")
            target["artifact"]["endpoint_file_sha256"] = replay.sha256_file(path)
            payload = {
                "schema": "vision_memory.r11-vae-latent-endpoint.v1",
                "latent_fp32": torch.zeros(1, 4, 128, 128),
                "image": torch.zeros(1, 3, 1024, 1024, dtype=torch.bfloat16),
            }
            with patch.object(replay.torch, "load", return_value=payload) as loader:
                with self.assertRaisesRegex(ValueError, "tensor SHA256 mismatch"):
                    replay.load_endpoint(root, target)
                loader.assert_called_once_with(path.resolve(), map_location="cpu", weights_only=True)

    def test_incomplete_or_duplicate_generations_never_aggregate(self):
        rows, anchors = self.records()
        for invalid in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.subTest(count=len(invalid)):
                with self.assertRaisesRegex(ValueError, "exactly 48"):
                    replay.aggregate_results(self.config, invalid, anchors)

    def test_missing_or_wrong_mcq_views_never_aggregate(self):
        rows, anchors = self.records()
        with self.assertRaisesRegex(ValueError, "exactly 32"):
            replay.aggregate_results(self.config, rows, anchors[:-1])
        anchors[0]["permutation"] = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "reverse-cyclic"):
            replay.aggregate_results(self.config, rows, anchors)

    def test_full_aggregation_has_paired_controls_without_semantic_credit(self):
        rows, anchors = self.records()
        summary = replay.aggregate_results(self.config, rows, anchors)
        self.assertEqual(summary["open_answer"]["count"], 48)
        self.assertEqual(summary["open_answer"]["strict_correct"], 16)
        self.assertEqual(summary["by_condition"]["matched"]["strict_accuracy"], 1.0)
        self.assertEqual(summary["mcq_anchor"]["correct"], 32)
        self.assertFalse(summary["formal_success"])
        self.assertFalse(summary["semantic_review"]["automatic_semantic_credit"])
        for prompt in replay.PROMPTS:
            for control in ("blank", "donor"):
                paired = summary["paired_matched_vs_controls"][prompt][control]
                self.assertEqual(paired["matched_only_correct"]["target_indices"], list(range(8)))

    def test_negated_extra_text_cannot_be_promoted_by_altering_score(self):
        rows, anchors = self.records()
        rows[0]["raw"] = "not linen"
        # The stale positive scorer is rejected, rather than silently counted.
        with self.assertRaisesRegex(ValueError, "scorer or question"):
            replay.aggregate_results(self.config, rows, anchors)
        rows[0]["scorer"] = replay.score_short_answer(rows[0]["raw"], "linen")
        summary = replay.aggregate_results(self.config, rows, anchors)
        self.assertEqual(summary["open_answer"]["strict_correct"], 15)
        self.assertEqual(summary["open_answer"]["extra_text_count"], 1)

    def test_existing_output_directory_is_never_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "existing"
            destination.mkdir()
            sentinel = destination / "terminal.json"
            sentinel.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError), patch.object(replay, "run") as run:
                replay.main(["--config", "unused", "--old-run-root", "unused", "--dreamlite", "unused",
                             "--reader", "unused", "--output-dir", str(destination)])
            run.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")

    def test_technical_failure_writes_failed_terminal_without_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "new"
            with patch.object(replay, "run", side_effect=RuntimeError("image parity failed")):
                status = replay.main(["--config", "unused", "--old-run-root", "unused", "--dreamlite", "unused",
                                      "--reader", "unused", "--output-dir", str(destination)])
            self.assertEqual(status, 1)
            terminal = json.loads((destination / "terminal.json").read_text(encoding="utf-8"))
            self.assertEqual(terminal["status"], "failed")
            self.assertFalse(terminal["formal_success"])
            self.assertFalse((destination / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
