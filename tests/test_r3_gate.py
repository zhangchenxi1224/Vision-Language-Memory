from __future__ import annotations

import unittest

from vision_memory.eval import score_r3_micro


def _row(
    episode_id: str,
    view: int,
    target: int,
    *,
    condition: str,
    correct: bool,
    variant: str | None = None,
    pair_id: str | None = None,
) -> dict:
    choices = ["red", "blue", "green", "no active preference"]
    prediction = target if correct else (target + 1) % 4
    return {
        "episode_id": episode_id,
        "query_ordinal": 1 if "transition" in episode_id else 0,
        "probe_role": "delayed",
        "choice_view_family": "reverse-cyclic4",
        "choice_view_index": view,
        "condition": condition,
        "target_index": target,
        "prediction_index": prediction,
        "choices": choices,
        "target_text": choices[target],
        "prediction_text": choices[prediction],
        "distractor_variant": variant,
        "distractor_pair_id": pair_id,
        "donor_target_index": target,
    }


class R3GateTest(unittest.TestCase):
    def test_set8_passes_exact_contract(self) -> None:
        rows = []
        for state in range(8):
            semantic_target = ("red", "blue", "green", "yellow")[state % 4]
            for view in range(4):
                target = view
                standard = _row(
                    f"r3-set8-r{state // 4}-v{state % 4}",
                    view,
                    target,
                    condition="standard",
                    correct=True,
                )
                standard["target_text"] = semantic_target
                standard["prediction_text"] = semantic_target
                rows.append(standard)
                rows.append(_row(f"r3-set8-r{state // 4}-v{state % 4}", view, target, condition="reset", correct=False))
                rows.append(_row(f"r3-set8-r{state // 4}-v{state % 4}", view, target, condition="shuffle", correct=False))
        report = score_r3_micro(rows, "set8")
        self.assertTrue(report["passed"])
        self.assertEqual(report["correct"], 32)

    def test_set8_fails_without_causal_drop(self) -> None:
        rows = []
        for state in range(8):
            for view in range(4):
                for condition in ("standard", "reset", "shuffle"):
                    rows.append(_row(f"r3-set8-r{state // 4}-v{state % 4}", view, view, condition=condition, correct=True))
        report = score_r3_micro(rows, "set8")
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["reset_drop"])

    def test_transition16_passes_exact_contract(self) -> None:
        rows = []
        kinds = ("set", "overwrite", "clear", "noop")
        for kind_index, kind in enumerate(kinds):
            for read_form in ("separate", "mixed"):
                for replica in range(2):
                    episode_id = f"r3-transition-{kind}-{read_form}-r{replica}"
                    variant = "clean" if kind == "set" else "distractor" if kind == "noop" else None
                    pair_id = f"r3-transition-noop-pair-{read_form}-r{replica}" if variant else None
                    for view in range(4):
                        target = view
                        for condition, correct in (("standard", True), ("reset", False), ("shuffle", False)):
                            rows.append(
                                _row(
                                    episode_id,
                                    view,
                                    target,
                                    condition=condition,
                                    correct=correct,
                                    variant=variant,
                                    pair_id=pair_id,
                                )
                            )
                        swap = _row(episode_id, view, target, condition="state_swap", correct=True)
                        swap["donor_target_index"] = swap["prediction_index"]
                        rows.append(swap)
        # Make clean/noop semantic targets and predictions identical as required.
        for row in rows:
            if row.get("distractor_variant") in {"clean", "distractor"}:
                row["target_text"] = "red"
                if row["condition"] == "standard":
                    row["prediction_text"] = "red"
        report = score_r3_micro(rows, "transition16")
        self.assertTrue(report["passed"])
        self.assertEqual(report["interventions"]["state_swap"]["count"], 16)


if __name__ == "__main__":
    unittest.main()
