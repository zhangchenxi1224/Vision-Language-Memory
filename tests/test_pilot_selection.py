from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cluster"))

from select_pilot import evaluate_selection  # noqa: E402


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def records(method: str, correct_counts: dict[str, int]) -> list[dict]:
    result = []
    for condition, count in correct_counts.items():
        for index in range(10):
            result.append(
                {
                    "episode_id": f"episode-{index}",
                    "query_id": f"episode-{index}:q0",
                    "method": method,
                    "condition": condition,
                    "noop_policy": "keep",
                    "prediction_index": 0 if index < count else 1,
                    "target_index": 0,
                }
            )
    return result


class PilotSelectionTest(unittest.TestCase):
    def fixture(self, root: Path, *, selected_is_good: bool = True) -> dict:
        blank = root / "blank.jsonl"
        frozen = root / "frozen.jsonl"
        write_jsonl(blank, records("query_only", {"standard": 2}))
        write_jsonl(frozen, records("frozen_dreamlite", {"standard": 3}))
        candidates = []
        for index, (learning_rate, dev_loss) in enumerate(((3e-5, 0.8), (1e-4, 0.4), (3e-4, 0.6))):
            directory = root / f"candidate-{index}"
            directory.mkdir()
            summary = directory / "summary.json"
            prediction = directory / "predictions.jsonl"
            checkpoint = directory / "best.pt"
            resume_checkpoint = directory / "checkpoint-000100.pt"
            write_json(summary, {"best_dev_loss": dev_loss, "optimizer_steps": 250})
            standard_count = 6 if selected_is_good or learning_rate != 1e-4 else 3
            write_jsonl(
                prediction,
                records(
                    f"pilot-{index}",
                    {"standard": standard_count, "reset": 3, "shuffle": 4, "state_swap": 2},
                ),
            )
            checkpoint.write_bytes(b"checkpoint")
            resume_checkpoint.write_bytes(b"resume")
            candidates.append(
                {
                    "learning_rate": learning_rate,
                    "candidate_dir": str(directory),
                    "summary": str(summary),
                    "predictions": str(prediction),
                    "checkpoint": str(checkpoint),
                    "resume_checkpoint": str(resume_checkpoint),
                }
            )
        return {
            "selection_split": "dev",
            "blank_predictions": str(blank),
            "frozen_predictions": str(frozen),
            "candidates": candidates,
        }

    def test_selects_by_dev_loss_then_applies_scientific_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_selection(
                self.fixture(Path(directory)),
                minimum_gain=0.10,
                minimum_intervention_drop=0.10,
            )
            self.assertEqual(report["selected"]["learning_rate"], 1e-4)
            self.assertAlmostEqual(report["selected"]["gain_over_blank"], 0.4)
            self.assertAlmostEqual(report["selected"]["gain_over_frozen"], 0.3)
            self.assertAlmostEqual(report["selected"]["reset_drop"], 0.3)
            self.assertTrue(report["passed"])

    def test_does_not_cherry_pick_a_worse_dev_candidate_that_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_selection(
                self.fixture(Path(directory), selected_is_good=False),
                minimum_gain=0.10,
                minimum_intervention_drop=0.10,
            )
            self.assertEqual(report["selected"]["learning_rate"], 1e-4)
            self.assertFalse(report["checks"]["gain_over_frozen"])
            self.assertFalse(report["passed"])

    def test_rejects_a_non_dev_selection_split(self):
        with tempfile.TemporaryDirectory() as directory:
            specification = self.fixture(Path(directory))
            specification["selection_split"] = "test_id"
            with self.assertRaisesRegex(ValueError, "locked to dev"):
                evaluate_selection(
                    specification,
                    minimum_gain=0.10,
                    minimum_intervention_drop=0.10,
                )


if __name__ == "__main__":
    unittest.main()
