from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probes.calibrate_r3_teacher_loss import parse_args  # noqa: E402


def test_calibration_defaults_match_locked_micro_student() -> None:
    argv = [
        "calibrate_r3_teacher_loss.py",
        "--train",
        "train.jsonl",
        "--cache-dir",
        "cache",
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
