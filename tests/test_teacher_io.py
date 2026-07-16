from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.teacher import (  # noqa: E402
    CALIBRATION_FILENAME,
    MANIFEST_FILENAME,
    SIDECAR_FILENAME,
    FrozenTeacherLossCalibration,
    TeacherArtifactRecord,
    TeacherCacheManifest,
    TeacherState,
    TeacherTransitionRecord,
    load_teacher_cache,
    load_teacher_cache_manifest,
    load_teacher_calibration,
    load_teacher_tensor,
    load_teacher_transition_sidecar,
    make_disk_teacher_provider,
    save_teacher_cache,
    save_teacher_calibration,
    save_teacher_tensor,
)


def teacher_fixture() -> TeacherState:
    return TeacherState(
        state_id="1" * 64,
        teacher_key="2" * 64,
        semantic_state_sha256="3" * 64,
        teacher_contract_sha256="4" * 64,
        renderer_contract_sha256="5" * 64,
        image=torch.zeros(1, 3, 1024, 1024, dtype=torch.float16),
        latent=torch.arange(256, dtype=torch.float32).reshape(1, 4, 8, 8) / 255.0,
        feature=torch.arange(18, dtype=torch.float32).reshape(1, 3, 6) / 17.0,
    )


def cache_fixture() -> tuple[
    TeacherState,
    TeacherCacheManifest,
    tuple[TeacherTransitionRecord, ...],
    FrozenTeacherLossCalibration,
]:
    teacher = teacher_fixture()
    manifest = TeacherCacheManifest(
        teacher_contract_sha256=teacher.teacher_contract_sha256,
        renderer_contract_sha256=teacher.renderer_contract_sha256,
        records=(TeacherArtifactRecord.from_teacher_state(teacher),),
    )
    sidecar = (
        TeacherTransitionRecord(
            episode_id="episode-0",
            turn_id=0,
            before_state_id=teacher.state_id,
            after_state_id=teacher.state_id,
            event_kind="noop",
            teacher_key=teacher.teacher_key,
        ),
    )
    calibration = FrozenTeacherLossCalibration(
        latent_scale=0.5,
        image_scale=0.25,
        feature_scale=0.125,
    )
    return teacher, manifest, sidecar, calibration


class TeacherCacheRoundTripTest(unittest.TestCase):
    def test_atomic_cache_round_trip_and_trainer_facing_apis(self):
        teacher, manifest, sidecar, calibration = cache_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = save_teacher_cache(
                root,
                manifest=manifest,
                teacher_states=(teacher,),
                sidecar=sidecar,
                calibration=calibration,
            )

            self.assertTrue((root / MANIFEST_FILENAME).is_file())
            self.assertTrue((root / SIDECAR_FILENAME).is_file())
            self.assertTrue((root / CALIBRATION_FILENAME).is_file())
            self.assertEqual(len(hashes.artifacts), 3)
            self.assertFalse(any(path.name.endswith(".tmp") for path in root.rglob("*")))

            cache = load_teacher_cache(
                root,
                expected_manifest_file_sha256=hashes.manifest,
                expected_sidecar_file_sha256=hashes.sidecar,
                expected_calibration_file_sha256=hashes.calibration,
                expected_calibration_contract_sha256=calibration.contract_sha256,
            )
            loaded = cache.get(teacher.state_id, split="train")
            self.assertTrue(torch.equal(loaded.image, teacher.image))
            self.assertTrue(torch.equal(loaded.latent, teacher.latent))
            self.assertTrue(torch.equal(loaded.feature, teacher.feature))
            self.assertEqual(cache.sidecar, sidecar)
            self.assertEqual(cache.calibration, calibration)
            with self.assertRaisesRegex(ValueError, "non-train"):
                cache.get(teacher.state_id, split="dev")

            loaded_manifest = load_teacher_cache_manifest(root / MANIFEST_FILENAME)
            self.assertEqual(loaded_manifest.to_dict(), manifest.to_dict())
            self.assertEqual(
                load_teacher_transition_sidecar(
                    root / SIDECAR_FILENAME,
                    manifest=loaded_manifest,
                ),
                sidecar,
            )
            self.assertEqual(load_teacher_calibration(root / CALIBRATION_FILENAME), calibration)
            provider = make_disk_teacher_provider(root)
            self.assertTrue(torch.equal(provider.get(teacher.state_id, split="train").feature, teacher.feature))

    def test_each_tensor_is_checked_for_sha_shape_and_dtype(self):
        teacher, manifest, sidecar, calibration = cache_fixture()
        record = manifest.records[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_teacher_cache(
                root,
                manifest=manifest,
                teacher_states=(teacher,),
                sidecar=sidecar,
                calibration=calibration,
            )
            feature_path = root / record.feature.relative_path

            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_teacher_tensor(
                    feature_path,
                    specification=dataclasses.replace(record.feature, shape=(1, 18)),
                )
            with self.assertRaisesRegex(ValueError, "dtype mismatch"):
                load_teacher_tensor(
                    feature_path,
                    specification=dataclasses.replace(record.feature, dtype="float64"),
                )

            payload = torch.load(feature_path, map_location="cpu", weights_only=True)
            payload["split"] = "dev"
            torch.save(payload, feature_path)
            with self.assertRaisesRegex(ValueError, "train-only"):
                load_teacher_tensor(feature_path, specification=record.feature)
            payload["split"] = "train"
            payload["tensor"] = payload["tensor"] + 1.0
            torch.save(payload, feature_path)
            with self.assertRaisesRegex(ValueError, "tensor SHA256 mismatch"):
                load_teacher_cache(root)


class TeacherAtomicAndFailClosedTest(unittest.TestCase):
    def test_failed_tensor_write_preserves_destination_and_cleans_temporary_file(self):
        teacher = teacher_fixture()
        specification = TeacherArtifactRecord.from_teacher_state(teacher).feature
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "feature.pt"
            destination.write_bytes(b"existing-cache-object")
            with patch("vision_memory.teacher.io.torch.save", side_effect=RuntimeError("injected write failure")):
                with self.assertRaisesRegex(RuntimeError, "injected write failure"):
                    save_teacher_tensor(destination, teacher.feature, specification=specification)
            self.assertEqual(destination.read_bytes(), b"existing-cache-object")
            self.assertFalse(any(path.name.endswith(".tmp") for path in root.iterdir()))

    def test_metadata_hash_and_train_split_drift_fail_closed(self):
        teacher, manifest, sidecar, calibration = cache_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = save_teacher_cache(
                root,
                manifest=manifest,
                teacher_states=(teacher,),
                sidecar=sidecar,
                calibration=calibration,
            )
            with self.assertRaisesRegex(ValueError, "File SHA256 mismatch"):
                load_teacher_cache_manifest(
                    root / MANIFEST_FILENAME,
                    expected_file_sha256="0" * 64,
                )
            manifest_path = root / MANIFEST_FILENAME
            manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_value["split"] = "dev"
            manifest_path.write_text(json.dumps(manifest_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train-only"):
                load_teacher_cache_manifest(manifest_path)

            calibration_path = root / CALIBRATION_FILENAME
            calibration_value = json.loads(calibration_path.read_text(encoding="utf-8"))
            calibration_value["split"] = "dev"
            calibration_path.write_text(json.dumps(calibration_value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train-only"):
                load_teacher_calibration(calibration_path)

            sidecar_path = root / SIDECAR_FILENAME
            sidecar_value = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar_value["split"] = "test"
            sidecar_path.write_text(json.dumps(sidecar_value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train-only"):
                load_teacher_transition_sidecar(sidecar_path, manifest=manifest)

            # A fresh atomic calibration save recovers the locked train-only file.
            recovered_sha = save_teacher_calibration(calibration_path, calibration)
            self.assertEqual(load_teacher_calibration(calibration_path), calibration)
            self.assertEqual(recovered_sha, hashes.calibration)


if __name__ == "__main__":
    unittest.main()
