from __future__ import annotations

import io
import hashlib
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probes import teacher_t0_upper_bound  # noqa: E402
from scripts.probes.teacher_t0_upper_bound import (  # noqa: E402
    MODALITIES,
    align_transition16_delayed_queries,
    audit_cache_identity_and_paths,
    audit_cross_split_fail_closed,
    audit_identity_mutations,
    contract_exit_code,
    manifest_transition_records,
    parse_raw_sidecar_records,
    reverse_cyclic_query_views,
    score_upper_bound_predictions,
    semantic_state_registry,
)
from vision_memory.data import build_transition16  # noqa: E402


def perfect_predictions() -> list[dict]:
    rows: list[dict] = []
    for modality in MODALITIES:
        for episode_index in range(16):
            target_text = f"target-{episode_index}"
            template_id = f"r3-transition-{'mixed' if episode_index % 2 else 'separate'}-gate-b"
            canonical = [target_text, "choice-a", "choice-b", "choice-c"]
            for view_index in range(4):
                choices = canonical[view_index:] + canonical[:view_index]
                target_index = choices.index(target_text)
                rows.append(
                    {
                        "episode_id": f"episode-{episode_index:02d}",
                        "template_id": template_id,
                        "modality": modality,
                        "choice_view_family": "reverse-cyclic4",
                        "choice_view_index": view_index,
                        "choices": choices,
                        "target_index": target_index,
                        "target_text": target_text,
                        "prediction_index": target_index,
                        "prediction_text": target_text,
                    }
                )
    return rows


class TeacherT0UpperBoundTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = build_transition16()
        cls.transitions = parse_raw_sidecar_records(cls.suite.teacher_sidecar)

    def test_transition16_raw_sidecar_aligns_exactly_to_final_delayed_queries(self):
        cases = align_transition16_delayed_queries(self.suite.gate_episodes, self.transitions)
        registry = semantic_state_registry(self.transitions)

        self.assertEqual(len(cases), 16)
        self.assertEqual(len(self.transitions), 28)
        self.assertTrue(registry)
        self.assertTrue(all(case.query.comparison_id.endswith(":delayed") for case in cases))
        self.assertTrue(all("gate-b" in case.template_id for case in cases))

    def test_reverse_cyclic_views_preserve_target_and_cover_every_position(self):
        case = align_transition16_delayed_queries(self.suite.gate_episodes, self.transitions)[0]
        views = reverse_cyclic_query_views(case.query)

        self.assertEqual(len(views), 4)
        self.assertEqual(sorted(view.target_index for view in views), [0, 1, 2, 3])
        self.assertTrue(all(view.target == case.query.target for view in views))

    def test_query_choices_episode_future_and_target_deletion_do_not_change_teacher_identity(self):
        state = self.transitions[0].after_state
        audit = audit_identity_mutations(state)

        self.assertTrue(audit["passed"])
        self.assertEqual(len(set(audit["mutation_state_ids"].values())), 1)
        self.assertIn("target_deleted", audit["mutations_checked"])
        self.assertTrue(audit["state_boundary_rejects_supervision"])

    def test_raw_sidecar_rejects_cross_split_and_noop_state_change(self):
        raw = dict(self.suite.teacher_sidecar[-1])
        raw["split"] = "dev"
        with self.assertRaisesRegex(ValueError, "train-only"):
            parse_raw_sidecar_records([raw])

        noop = next(dict(record) for record in self.suite.teacher_sidecar if record["event_kind"] == "noop")
        changed_after = json.loads(json.dumps(noop["after_state"]))
        changed_after["entries"][0]["value_id"] = "magenta"
        changed_after["entries"][0]["value_text"] = "magenta"
        noop["after_state"] = changed_after
        with self.assertRaisesRegex(ValueError, "no-op"):
            parse_raw_sidecar_records([noop])

    def test_provider_cross_split_audit_fails_closed(self):
        class RefusingProvider:
            def get(self, state_id, *, split):
                del state_id, split
                raise ValueError("train only")

        class AcceptingProvider:
            def get(self, state_id, *, split):
                return state_id, split

        self.assertTrue(audit_cross_split_fail_closed(RefusingProvider(), "0" * 64)["passed"])
        self.assertFalse(audit_cross_split_fail_closed(AcceptingProvider(), "0" * 64)["passed"])

    def test_cache_audit_covers_path_invariance_noop_and_collisions(self):
        registry = semantic_state_registry(self.transitions)
        records = []
        for state_id, state in registry.items():
            prefix = f"artifacts/{state_id}"
            records.append(
                SimpleNamespace(
                    state_id=state_id,
                    teacher_key=hashlib.sha256(f"teacher:{state_id}".encode()).hexdigest(),
                    semantic_state_sha256=state.canonical_sha256,
                    image=SimpleNamespace(relative_path=f"{prefix}/image.pt"),
                    latent=SimpleNamespace(relative_path=f"{prefix}/latent.pt"),
                    feature=SimpleNamespace(relative_path=f"{prefix}/feature.pt"),
                )
            )
        manifest = SimpleNamespace(
            records=tuple(records),
            by_state_id={record.state_id: record for record in records},
        )

        converted = manifest_transition_records(self.transitions, manifest)
        audit = audit_cache_identity_and_paths(
            registry=registry,
            transitions=self.transitions,
            manifest=manifest,
        )

        self.assertEqual(len(converted), 28)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["noop_transition_count"], 4)
        self.assertGreater(audit["multi_path_state_count"], 0)

        records[1].image.relative_path = records[0].image.relative_path
        collision = audit_cache_identity_and_paths(
            registry=registry,
            transitions=self.transitions,
            manifest=manifest,
        )
        self.assertFalse(collision["passed"])
        self.assertFalse(collision["checks"]["unique_artifact_paths"])

    def test_upper_bound_metrics_enforce_macro_positions_templates_and_rotation(self):
        episode_ids = [f"episode-{index:02d}" for index in range(16)]
        passed = score_upper_bound_predictions(perfect_predictions(), expected_episode_ids=episode_ids)
        self.assertTrue(passed["passed"])
        for modality in MODALITIES:
            self.assertEqual(passed["modalities"][modality]["macro_accuracy"], 1.0)
            self.assertEqual(
                passed["modalities"][modality]["predicted_text_rotation_agreement"],
                1.0,
            )
            self.assertTrue(
                all(value["correct"] == 16 for value in passed["modalities"][modality]["positions"].values())
            )

        inconsistent = perfect_predictions()
        first = inconsistent[0]
        first["prediction_index"] = (first["target_index"] + 1) % 4
        first["prediction_text"] = first["choices"][first["prediction_index"]]
        rotation_failed = score_upper_bound_predictions(inconsistent, expected_episode_ids=episode_ids)
        self.assertFalse(rotation_failed["passed"])
        self.assertFalse(rotation_failed["modalities"]["raw_state_card"]["checks"]["predicted_text_rotation_agreement"])

        macro_failed_rows = perfect_predictions()
        for row in macro_failed_rows[:4]:
            row["prediction_index"] = (row["target_index"] + 1) % 4
            row["prediction_text"] = row["choices"][row["prediction_index"]]
        macro_failed = score_upper_bound_predictions(macro_failed_rows, expected_episode_ids=episode_ids)
        self.assertFalse(macro_failed["modalities"]["raw_state_card"]["checks"]["macro_accuracy"])

    def test_main_emits_json_and_returns_nonzero_without_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "teacher-t0.json"
            arguments = [
                "--gate-jsonl",
                "gate.jsonl",
                "--raw-sidecar",
                "sidecar.jsonl",
                "--teacher-manifest",
                "manifest.json",
                "--reader",
                "reader",
                "--output-json",
                str(output),
            ]
            with mock.patch.object(teacher_t0_upper_bound.torch.cuda, "is_available", return_value=False):
                with redirect_stdout(io.StringIO()):
                    status = teacher_t0_upper_bound.main(arguments)
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["error"]["type"], "RuntimeError")
        self.assertEqual(contract_exit_code(report), 1)


if __name__ == "__main__":
    unittest.main()
