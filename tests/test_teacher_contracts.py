from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import sys
import unittest
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import __version__ as PILLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.teacher import (  # noqa: E402
    CALIBRATION_DENOMINATOR_EPSILON,
    SEMANTIC_STATE_SCHEMA,
    FixedFontContract,
    FrozenTeacherLossCalibration,
    FullStateCardRenderer,
    SemanticState,
    SemanticStateEntry,
    TeacherArtifactRecord,
    TeacherBuildContract,
    TeacherCacheManifest,
    TeacherProvider,
    TeacherState,
    TeacherTransitionRecord,
    build_teacher_state,
    composite_teacher_distillation_loss,
    file_sha256,
    manifest_json,
    normalize_latent_per_channel,
    rgb_sha256,
    validate_teacher_sidecar,
)


def active_entry(
    *,
    entity_id: str = "entity-1",
    entity_text: str = "Cafe",
    slot_id: str = "drink",
    slot_text: str = "drink",
    value_id: str = "tea",
    value_text: str = "tea",
) -> SemanticStateEntry:
    return SemanticStateEntry(
        entity_id=entity_id,
        entity_text=entity_text,
        slot_id=slot_id,
        slot_text=slot_text,
        status="active",
        value_id=value_id,
        value_text=value_text,
    )


def cleared_entry() -> SemanticStateEntry:
    return SemanticStateEntry(
        entity_id="entity-2",
        entity_text="Studio",
        slot_id="music",
        slot_text="music",
        status="cleared",
    )


def available_font() -> Path | None:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    return next((path for path in candidates if path.is_file()), None)


def renderer_or_skip(test: unittest.TestCase) -> FullStateCardRenderer:
    path = available_font()
    if path is None:
        test.skipTest("No explicit TrueType font is available for the fixed-font renderer test.")
    assert path is not None
    return FullStateCardRenderer(
        FixedFontContract(
            font_id="test-fixed-font",
            path=path,
            sha256=file_sha256(path),
            pillow_version=PILLOW_VERSION,
        )
    )


def dummy_teacher(
    *,
    state_id: str = "1" * 64,
    teacher_key: str = "2" * 64,
    image_value: float = 0.0,
) -> TeacherState:
    latent = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4) / 31.0 + image_value
    return TeacherState(
        state_id=state_id,
        teacher_key=teacher_key,
        semantic_state_sha256="3" * 64,
        teacher_contract_sha256="4" * 64,
        renderer_contract_sha256="5" * 64,
        image=torch.full((1, 3, 1024, 1024), image_value),
        latent=latent,
        feature=torch.full((1, 8), image_value),
    )


class SemanticStateContractTest(unittest.TestCase):
    def test_nfc_and_entry_order_produce_one_compact_state_identity(self):
        decomposed = unicodedata.normalize("NFD", "Café")
        first = SemanticState(
            entries=(
                active_entry(entity_text=decomposed),
                cleared_entry(),
            )
        )
        second = SemanticState(
            entries=(
                cleared_entry(),
                active_entry(entity_text="Café"),
            )
        )

        self.assertEqual(first.canonical_bytes, second.canonical_bytes)
        self.assertEqual(first.state_id, second.state_id)
        self.assertEqual(first.canonical_sha256, second.canonical_sha256)
        payload = first.canonical_bytes.decode("utf-8")
        self.assertNotIn("\n", payload)
        self.assertNotIn(": ", payload)
        self.assertEqual(json.loads(payload)["schema"], SEMANTIC_STATE_SCHEMA)
        self.assertEqual(SEMANTIC_STATE_SCHEMA, "vlm.semantic_state.v1")
        self.assertEqual(
            first.state_id,
            hashlib.sha256(b"vlm.semantic_state.v1\0" + first.canonical_bytes).hexdigest(),
        )

    def test_state_rejects_supervision_and_ambiguous_rendering(self):
        value = SemanticState(entries=(active_entry(),)).to_dict()
        value["target_index"] = 0
        with self.assertRaisesRegex(ValueError, "Supervision key"):
            SemanticState.from_dict(value)

        with self.assertRaisesRegex(ValueError, "duplicate entity_text"):
            SemanticState(
                entries=(
                    active_entry(entity_id="one", slot_id="one"),
                    active_entry(entity_id="two", slot_id="two"),
                )
            )
        with self.assertRaisesRegex(ValueError, "single ASCII spaces"):
            active_entry(entity_text="two  spaces")

    def test_status_value_contract_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "requires value_id"):
            SemanticStateEntry("e", "entity", "s", "slot", "active")
        with self.assertRaisesRegex(ValueError, "cannot contain a value"):
            SemanticStateEntry("e", "entity", "s", "slot", "cleared", "v", "value")


class FullStateRendererTest(unittest.TestCase):
    def test_render_is_path_independent_deterministic_1024_rgb(self):
        renderer = renderer_or_skip(self)
        first = SemanticState(entries=(active_entry(entity_text="Café"), cleared_entry()))
        second = SemanticState(entries=(cleared_entry(), active_entry(entity_text=unicodedata.normalize("NFD", "Café"))))

        image_a = renderer.render(first)
        image_b = renderer.render(second)
        self.assertEqual(image_a.mode, "RGB")
        self.assertEqual(image_a.size, (1024, 1024))
        self.assertEqual(image_a.tobytes(), image_b.tobytes())
        self.assertEqual(rgb_sha256(image_a), rgb_sha256(image_b))
        tensor = renderer.render_tensor(first)
        self.assertEqual(tuple(tensor.shape), (1, 3, 1024, 1024))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_font_hash_and_pillow_version_fail_closed(self):
        path = available_font()
        if path is None:
            self.skipTest("No explicit TrueType font is available.")
        assert path is not None
        with self.assertRaisesRegex(RuntimeError, "SHA256 drifted"):
            FullStateCardRenderer(
                FixedFontContract(
                    font_id="wrong-hash",
                    path=path,
                    sha256="0" * 64,
                    pillow_version=PILLOW_VERSION,
                )
            )
        with self.assertRaisesRegex(RuntimeError, "Pillow version drifted"):
            FullStateCardRenderer(
                FixedFontContract(
                    font_id="wrong-pillow",
                    path=path,
                    sha256=file_sha256(path),
                    pillow_version="0.0.0",
                )
            )

    def test_renderer_refuses_capacity_and_cell_overflow(self):
        renderer = renderer_or_skip(self)
        entries = tuple(
            active_entry(
                entity_id=f"entity-{index}",
                entity_text=f"Entity {index}",
                slot_id="slot",
                slot_text="slot",
                value_id="value",
                value_text="value",
            )
            for index in range(17)
        )
        with self.assertRaisesRegex(ValueError, "16-entry"):
            renderer.render(SemanticState(entries=entries))

        long_value = "x" * 500
        with self.assertRaisesRegex(ValueError, "token too wide"):
            renderer.render(SemanticState(entries=(active_entry(value_id="long", value_text=long_value),)))


class TeacherArtifactBuildTest(unittest.TestCase):
    def test_callbacks_receive_image_only_and_build_path_invariant_artifacts(self):
        renderer = renderer_or_skip(self)
        calls: list[tuple[str, tuple[int, ...]]] = []

        def latent_callback(image: torch.Tensor) -> torch.Tensor:
            calls.append(("latent", tuple(image.shape)))
            return F.avg_pool2d(image, kernel_size=8)

        def decode_callback(latent: torch.Tensor) -> torch.Tensor:
            calls.append(("decode", tuple(latent.shape)))
            return F.interpolate(latent, size=(1024, 1024), mode="nearest").clamp(0.0, 1.0)

        def feature_callback(image: torch.Tensor) -> torch.Tensor:
            calls.append(("feature", tuple(image.shape)))
            return image.mean(dim=(-2, -1))

        contract = TeacherBuildContract(
            latent_callback_id="mock-vae-posterior-mean-v1",
            decode_callback_id="mock-vae-decode-v1",
            feature_callback_id="mock-qwen-query-free-vision-v1",
            vae_revision="vae-revision",
            reader_revision="reader-revision",
        )
        first_state = SemanticState(entries=(active_entry(), cleared_entry()))
        second_state = SemanticState(entries=(cleared_entry(), active_entry()))
        self.assertTrue(
            {"query", "query_text", "choices", "options", "target", "target_index"}.isdisjoint(
                inspect.signature(build_teacher_state).parameters
            )
        )
        first = build_teacher_state(
            first_state,
            renderer=renderer,
            contract=contract,
            encode_image=latent_callback,
            decode_latent=decode_callback,
            encode_visual_feature=feature_callback,
        )
        second = build_teacher_state(
            second_state,
            renderer=renderer,
            contract=contract,
            encode_image=latent_callback,
            decode_latent=decode_callback,
            encode_visual_feature=feature_callback,
        )

        self.assertEqual(
            calls,
            [
                ("latent", (1, 3, 1024, 1024)),
                ("decode", (1, 3, 128, 128)),
                ("feature", (1, 3, 1024, 1024)),
            ]
            * 2,
        )
        self.assertEqual(first.state_id, second.state_id)
        self.assertEqual(first.teacher_key, second.teacher_key)
        self.assertEqual(first.artifact_sha256, second.artifact_sha256)
        self.assertTrue(
            torch.equal(
                first.image,
                F.interpolate(first.latent, size=(1024, 1024), mode="nearest").clamp(0.0, 1.0),
            )
        )
        self.assertTrue(torch.equal(first.feature, first.image.mean(dim=(-2, -1))))
        self.assertFalse(first.image.requires_grad)
        self.assertFalse(first.latent.requires_grad)
        self.assertFalse(first.feature.requires_grad)

    def test_callback_output_must_be_finite_batch_one_tensor(self):
        renderer = renderer_or_skip(self)
        contract = TeacherBuildContract("latent", "decode", "feature", "vae", "reader")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_teacher_state(
                SemanticState(entries=(active_entry(),)),
                renderer=renderer,
                contract=contract,
                encode_image=lambda _image: torch.full((1, 1), float("nan")),
                decode_latent=lambda _latent: torch.zeros(1, 3, 1024, 1024),
                encode_visual_feature=lambda image: image.mean(dim=(-2, -1)),
            )


class TeacherCacheAndSidecarTest(unittest.TestCase):
    def manifest(self) -> tuple[TeacherCacheManifest, TeacherState, TeacherState]:
        first = dummy_teacher()
        second = TeacherState(
            state_id="6" * 64,
            teacher_key="7" * 64,
            semantic_state_sha256="8" * 64,
            teacher_contract_sha256=first.teacher_contract_sha256,
            renderer_contract_sha256=first.renderer_contract_sha256,
            image=first.image,
            latent=first.latent,
            feature=first.feature,
        )
        manifest = TeacherCacheManifest(
            teacher_contract_sha256=first.teacher_contract_sha256,
            renderer_contract_sha256=first.renderer_contract_sha256,
            records=(TeacherArtifactRecord.from_teacher_state(second), TeacherArtifactRecord.from_teacher_state(first)),
        )
        return manifest, first, second

    def test_manifest_round_trip_provider_and_train_only_boundary(self):
        manifest, first, second = self.manifest()
        round_trip = TeacherCacheManifest.from_dict(json.loads(manifest_json(manifest)))
        self.assertEqual(round_trip.to_dict(), manifest.to_dict())
        states = {first.state_id: first, second.state_id: second}
        provider = TeacherProvider(round_trip, lambda record: states[record.state_id])
        self.assertIs(provider.get(first.state_id, split="train"), first)
        with self.assertRaisesRegex(ValueError, "non-train"):
            provider.get(first.state_id, split="dev")
        self.assertNotIn("query", manifest_json(manifest).casefold())
        self.assertNotIn("target_index", manifest_json(manifest))

    def test_provider_rehashes_loaded_artifacts(self):
        manifest, first, _second = self.manifest()
        corrupted = TeacherState(
            state_id=first.state_id,
            teacher_key=first.teacher_key,
            semantic_state_sha256=first.semantic_state_sha256,
            teacher_contract_sha256=first.teacher_contract_sha256,
            renderer_contract_sha256=first.renderer_contract_sha256,
            image=torch.ones_like(first.image),
            latent=first.latent,
            feature=first.feature,
        )
        provider = TeacherProvider(manifest, lambda _record: corrupted)
        with self.assertRaisesRegex(ValueError, "image SHA256"):
            provider.get(first.state_id, split="train")

    def test_sidecar_enforces_continuity_noop_identity_and_no_leakage(self):
        manifest, first, second = self.manifest()
        records = (
            TeacherTransitionRecord(
                episode_id="episode",
                turn_id=0,
                before_state_id=first.state_id,
                after_state_id=second.state_id,
                event_kind="set",
                teacher_key=second.teacher_key,
            ),
            TeacherTransitionRecord(
                episode_id="episode",
                turn_id=1,
                before_state_id=second.state_id,
                after_state_id=second.state_id,
                event_kind="noop",
                teacher_key=second.teacher_key,
            ),
        )
        self.assertEqual(validate_teacher_sidecar(records, manifest=manifest), records)

        leaked = records[0].to_dict()
        leaked["target_index"] = 2
        with self.assertRaisesRegex(ValueError, "Supervision key"):
            validate_teacher_sidecar((leaked,), manifest=manifest)
        with self.assertRaisesRegex(ValueError, "preserve state_id"):
            TeacherTransitionRecord(
                episode_id="episode",
                turn_id=2,
                before_state_id=first.state_id,
                after_state_id=second.state_id,
                event_kind="noop",
                teacher_key=second.teacher_key,
            )

        discontinuous = (
            records[0],
            TeacherTransitionRecord(
                episode_id="episode",
                turn_id=1,
                before_state_id=first.state_id,
                after_state_id=first.state_id,
                event_kind="noop",
                teacher_key=first.teacher_key,
            ),
        )
        with self.assertRaisesRegex(ValueError, "continuity"):
            validate_teacher_sidecar(discontinuous, manifest=manifest)


class TeacherDistillationLossTest(unittest.TestCase):
    def test_frozen_scales_normalize_components_and_gradients_reach_students_only(self):
        teacher = dummy_teacher(image_value=1.0)
        calibration = FrozenTeacherLossCalibration(
            latent_scale=2.0,
            image_scale=4.0,
            feature_scale=8.0,
        )
        student_latent = torch.linspace(-0.25, 0.75, teacher.latent.numel()).reshape_as(teacher.latent)
        student_latent.requires_grad_(True)
        student_image = torch.zeros_like(teacher.image, requires_grad=True)
        student_feature = torch.zeros_like(teacher.feature, requires_grad=True)
        output = composite_teacher_distillation_loss(
            student_latent=student_latent,
            student_image=student_image,
            student_feature=student_feature,
            teacher=teacher,
            calibration=calibration,
        )

        expected_latent_raw = F.smooth_l1_loss(
            normalize_latent_per_channel(student_latent),
            normalize_latent_per_channel(teacher.latent),
        )
        expected_image_raw = F.smooth_l1_loss(student_image, teacher.image)
        expected_feature_raw = (
            1.0 - F.cosine_similarity(student_feature, teacher.feature, dim=-1)
        ).mean()
        torch.testing.assert_close(output.latent_raw, expected_latent_raw)
        torch.testing.assert_close(output.image_raw, expected_image_raw)
        torch.testing.assert_close(output.feature_raw, expected_feature_raw)
        expected_total = (
            expected_latent_raw / (2.0 + CALIBRATION_DENOMINATOR_EPSILON)
            + expected_image_raw / (4.0 + CALIBRATION_DENOMINATOR_EPSILON)
            + expected_feature_raw / (8.0 + CALIBRATION_DENOMINATOR_EPSILON)
        ) / 3.0
        torch.testing.assert_close(output.loss, expected_total)
        output.loss.backward()
        self.assertGreater(float(student_latent.grad.norm()), 0.0)
        self.assertGreater(float(student_image.grad.norm()), 0.0)
        self.assertGreater(float(student_feature.grad.norm()), 0.0)
        self.assertIsNone(teacher.image.grad)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            calibration.image_scale = 1.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            FrozenTeacherLossCalibration(1.0, 1.0, 1.0, latent_weight=2.0)  # type: ignore[call-arg]

    def test_shape_and_calibration_contracts_fail_closed(self):
        teacher = dummy_teacher()
        with self.assertRaisesRegex(ValueError, "positive finite"):
            FrozenTeacherLossCalibration(0.0, 1.0, 1.0)
        calibration = FrozenTeacherLossCalibration(1.0, 1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            composite_teacher_distillation_loss(
                student_latent=torch.zeros(1, 1),
                student_image=teacher.image,
                student_feature=teacher.feature,
                teacher=teacher,
                calibration=calibration,
            )


if __name__ == "__main__":
    unittest.main()
