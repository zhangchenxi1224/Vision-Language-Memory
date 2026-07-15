from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.eval import (  # noqa: E402
    compute_prefeval_metrics,
    filter_preregistered_records,
    holm_correction,
    paired_hierarchical_bootstrap,
    topic_form_metrics,
)


def row(
    pair: str,
    topic: str,
    form: str,
    prediction: int,
    *,
    target: int = 0,
    method: str = "main",
    condition: str = "standard",
    protocol: str = "oracle-sparse",
    forced_write_k: int = 0,
    **extra,
):
    return {
        "base_pair_id": pair,
        "topic": topic,
        "form": form,
        "prediction_index": prediction,
        "target_index": target,
        "method": method,
        "condition": condition,
        "protocol": protocol,
        "forced_write_k": forced_write_k,
        **extra,
    }


class PrefEvalMetricsTest(unittest.TestCase):
    def test_forced_write_headline_selects_one_k(self):
        records = [
            row("p", "t", "explicit", 0, protocol="forced-write", forced_write_k=0),
            row("p", "t", "explicit", 1, protocol="forced-write", forced_write_k=2),
        ]
        selected = filter_preregistered_records(
            records,
            protocol="forced-write",
            form="explicit",
            forced_write_k=2,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["forced_write_k"], 2)

    def test_topic_form_macro_equal_weights_cells(self):
        records = [
            row("a0", "a", "explicit", 0),
            row("a1", "a", "explicit", 0),
            row("a2", "a", "explicit", 0),
            row("b0", "b", "implicit_choice", 1),
        ]
        result = topic_form_metrics(records)
        self.assertEqual(result["micro_accuracy"], 0.75)
        self.assertEqual(result["topic_form_macro_accuracy"], 0.5)

    def test_diagnostics_cover_stale_distractor_reset_shuffle_and_swap(self):
        records = [
            row("p", "t", "explicit", 0, stale_target_index=1, query_id="p:oracle-sparse:k0:q0"),
            row("p", "t", "explicit", 1, condition="reset", query_id="p:oracle-sparse:k0:q0"),
            row("p", "t", "explicit", 1, condition="shuffle", query_id="p:oracle-sparse:k0:q0"),
            row(
                "p",
                "t",
                "explicit",
                1,
                condition="state_swap",
                donor_target_index=1,
                query_id="p:oracle-sparse:k0:q0",
            ),
            row(
                "p",
                "t",
                "explicit",
                1,
                protocol="forced-write",
                forced_write_k=5,
                query_id="p:forced-write:k5:q0",
            ),
            row(
                "noop",
                "t",
                "explicit",
                1,
                noop_policy="keep",
                noop_intervention_pair_id="noop:q0",
            ),
            row(
                "noop",
                "t",
                "explicit",
                0,
                noop_policy="skip",
                noop_intervention_pair_id="noop:q0",
            ),
        ]
        result = compute_prefeval_metrics(records)["diagnostics"]
        self.assertEqual(result["reset"]["accuracy_drop"], 1.0)
        self.assertEqual(result["shuffle"]["accuracy_drop"], 1.0)
        self.assertEqual(result["state_swap"]["accuracy_drop"], 1.0)
        self.assertEqual(result["state_swap_donor_answer"]["rate"], 1.0)
        self.assertEqual(result["distractor_damage_by_k"]["5"]["accuracy_damage"], 1.0)
        self.assertEqual(result["noop_filter_effect"]["skip_minus_keep_accuracy"], 1.0)
        self.assertEqual(result["stale_answer_error"]["rate"], 0.0)

    def test_paired_hierarchical_bootstrap_is_deterministic(self):
        records = []
        for topic in ("t1", "t2"):
            for form in ("explicit", "implicit_choice"):
                for index in range(4):
                    pair = f"{topic}:{index}"
                    records.append(row(pair, topic, form, 0, method="learned"))
                    records.append(row(pair, topic, form, 1 if index < 2 else 0, method="blank"))
        first = paired_hierarchical_bootstrap(
            records,
            method_a="learned",
            method_b="blank",
            iterations=500,
            seed=2026,
        )
        second = paired_hierarchical_bootstrap(
            records,
            method_a="learned",
            method_b="blank",
            iterations=500,
            seed=2026,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["observed_delta"], 0.5)
        self.assertEqual(first["n_pairs"], 16)
        self.assertEqual(first["n_topic_subtype_cells"], 4)

    def test_bootstrap_rejects_unpaired_inputs(self):
        records = [
            row("p1", "t", "explicit", 0, method="a"),
            row("p2", "t", "explicit", 0, method="b"),
        ]
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            paired_hierarchical_bootstrap(records, method_a="a", method_b="b", iterations=10)

    def test_holm_correction_is_monotone_and_step_down(self):
        result = holm_correction({"a": 0.01, "b": 0.03, "c": 0.04})
        self.assertAlmostEqual(result["a"]["adjusted_p_value"], 0.03)
        self.assertAlmostEqual(result["b"]["adjusted_p_value"], 0.06)
        self.assertAlmostEqual(result["c"]["adjusted_p_value"], 0.06)
        self.assertTrue(result["a"]["rejected"])
        self.assertFalse(result["b"]["rejected"])
        self.assertFalse(result["c"]["rejected"])


if __name__ == "__main__":
    unittest.main()
