from pathlib import Path
import json
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vision_memory.prefeval.rgb_protocol import (
    balanced_state_loss, transition, writer_input, official_mcq_valid, state_id)


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "reports/prefeval-rgb-20260917/registered/manifest.json").read_text(encoding="utf-8"))

    def test_components_and_nested_subsets(self):
        m = self.manifest
        self.assertLessEqual(set(m["pilot_train"]), set(m["small"]))
        self.assertLess(set(m["small"]), set(m["full"]))
        for group in m["groups"].values():
            self.assertEqual({m["records"][key]["split"] for key in group["members"]}, {group["split"]})
        for target in m["targets"].values():
            self.assertEqual(target["split"], "train")
        for ep in m["episodes"]:
            expected = ep["split"]
            self.assertEqual({m["groups"][gid]["split"] for gid in ep["semantic_groups"]}, {expected})

    def test_predecessor_graph_is_acyclic(self):
        targets = self.manifest["targets"]
        for sid in targets:
            seen = set()
            while sid is not None:
                self.assertNotIn(sid, seen)
                seen.add(sid)
                sid = targets[sid]["predecessor"]

    def test_teacher_training_excludes_eval_combinations(self):
        m = self.manifest
        planned = {t["target_state_id"] for ep in m["episodes"]
                   if ep["split"] == "train" and not ep.get("evaluation_only") for t in ep["transitions"]}
        self.assertEqual(set(m["targets"]), planned)

    def test_clear_is_selective_and_invalidates_old_mcq(self):
        before = {"hotel": "quiet rooms", "food": "no peanuts"}
        after = transition(before, "hotel", None, "clear")
        self.assertIsNone(after["hotel"])
        self.assertEqual(after["food"], "no peanuts")
        self.assertEqual(before["hotel"], "quiet rooms")
        self.assertFalse(official_mcq_valid("quiet rooms", after["hotel"]))
        self.assertTrue(official_mcq_valid("no peanuts", after["food"]))

    def test_separate_normalization_preserves_untouched_gradient(self):
        changed = torch.tensor(4., requires_grad=True)
        kept = [torch.tensor(2., requires_grad=True) for _ in range(3)]
        loss = balanced_state_loss({"x":changed, **{str(i):v for i,v in enumerate(kept)}}, "x")
        self.assertEqual(float(loss.detach()), 3.)
        loss.backward()
        self.assertAlmostEqual(float(changed.grad), .5)
        for v in kept:
            self.assertAlmostEqual(float(v.grad), 1/6)

    def test_writer_boundary_has_no_scoring_metadata(self):
        image = object()
        self.assertEqual(set(writer_input(image, "Keep all my preferences.")), {"image", "event"})
        self.assertIs(writer_input(image, "Current event")["image"], image)

    def test_transitions_reconstruct_registered_states(self):
        for ep in self.manifest["episodes"]:
            previous = {}
            for t in ep["transitions"]:
                self.assertEqual(t["before"], previous)
                value = t["state"].get(t["scope"])
                actual = transition(previous, t["scope"], value, t["operation"])
                self.assertEqual(actual, t["state"])
                self.assertEqual(state_id(actual), t["target_state_id"])
                previous = actual


if __name__ == "__main__":
    unittest.main()
