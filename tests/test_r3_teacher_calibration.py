from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.probes.calibrate_r3_teacher_loss import parse_args  # noqa: E402
from vision_memory.teacher import (  # noqa: E402
    CALIBRATION_SAMPLE_SELECTION,
    TeacherCalibrationInputLock,
    file_sha256,
    load_teacher_calibration_input_lock,
    verify_teacher_calibration_input_files,
)


def test_calibration_defaults_match_locked_micro_student() -> None:
    argv = [
        "calibrate_r3_teacher_loss.py",
        "--train",
        "train.jsonl",
        "--cache-dir",
        "cache",
        "--suite",
        "set8",
        "--dreamlite",
        "DreamLite-mobile",
        "--reader",
        "Qwen3-VL-4B-Instruct",
        "--output",
        "calibration.json",
        "--report",
        "calibration_report.json",
    ]
    with mock.patch.object(sys, "argv", argv):
        args = parse_args()

    assert args.seed == 0
    assert args.adapter_seed == 0
    assert args.lora_rank == 4
    assert args.resolution == 1024
    assert args.dreamlite_device == "cuda:0"
    assert args.reader_device == "cuda:1"
    assert args.suite == "set8"
    assert args.preregistration == ROOT / "configs" / "experiments" / "r3_preregistration.json"


def test_preregistered_calibration_locks_bind_each_suite_train_and_cache() -> None:
    preregistration = ROOT / "configs" / "experiments" / "r3_preregistration.json"
    set8 = load_teacher_calibration_input_lock(preregistration, suite="set8")
    transition16 = load_teacher_calibration_input_lock(preregistration, suite="transition16")

    assert set8.preregistration_sha256 == transition16.preregistration_sha256 == file_sha256(
        preregistration
    )
    assert set8.train_sha256 == "b0b8896c14ea597379271bc485f57011a0028b3a50eef08185817bdf3442deb6"
    assert set8.manifest_sha256 == "1e611a8155ebc2b3055fd899271fdcc949bf12798ce49f7c64877fe71488ef02"
    assert set8.sidecar_sha256 == "1f76546dbf6f9e72ce764cf3c4fe519204667ef6609cfd9f293da0154ef11ad1"
    assert set8.transition_count == 8
    assert transition16.train_sha256 == (
        "879ff26cf638e87cea404ce5135e546b165f6a2a6da2a9415990b3e855651ae0"
    )
    assert transition16.manifest_sha256 == (
        "4d793bf76f44ddca573698832b19dfe242615e359bcc4c273bf180cb3bcd873e"
    )
    assert transition16.sidecar_sha256 == (
        "e80e6456db5f8fb513c10e02a42362b0478d2efb9863411492c309599e7527f0"
    )
    assert transition16.transition_count == 28
    assert CALIBRATION_SAMPLE_SELECTION == {
        "split": "train",
        "unit": "one-unweighted-sample-per-updater-transition",
        "query_turns_excluded": True,
        "duplicate_semantic_after_states_retained": True,
    }


def test_calibration_input_file_binding_round_trips_and_fails_closed(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    manifest = tmp_path / "manifest.json"
    sidecar = tmp_path / "transitions.jsonl"
    preregistration = tmp_path / "preregistration.json"
    train.write_bytes(b"train\n")
    manifest.write_bytes(b"manifest\n")
    sidecar.write_bytes(b"sidecar\n")
    preregistration.write_bytes(b"preregistration\n")
    lock = TeacherCalibrationInputLock(
        suite="set8",
        preregistration_sha256=file_sha256(preregistration),
        train_sha256=file_sha256(train),
        manifest_sha256=file_sha256(manifest),
        sidecar_sha256=file_sha256(sidecar),
        transition_count=8,
    )

    assert verify_teacher_calibration_input_files(
        lock,
        train=train,
        manifest=manifest,
        sidecar=sidecar,
    ) == lock.to_dict()

    sidecar.write_bytes(b"substituted\n")
    with pytest.raises(ValueError, match="inputs differ from preregistration"):
        verify_teacher_calibration_input_files(
            lock,
            train=train,
            manifest=manifest,
            sidecar=sidecar,
        )
