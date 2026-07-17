from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
INSPIRE = ROOT / "scripts" / "inspire"
sys.path.insert(0, str(INSPIRE))

import run_r3_teacher_prep_step as step  # noqa: E402
from r3_dag_contract import sha256_file  # noqa: E402


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_tc0_wrapper_injects_exact_runtime_hashes(tmp_path: Path) -> None:
    raw = _write(tmp_path / "raw.json", {"passed": True})
    preregistration = _write(
        tmp_path / "preregistration.json",
        {"schema": "vision_memory.r3-preregistration.v1"},
    )
    output = tmp_path / "validation.json"
    with patch.object(step.validate_tc0, "main", return_value=0) as validator:
        assert (
            step.main(
                [
                    "validate-tc0",
                    "--report",
                    str(raw),
                    "--preregistration",
                    str(preregistration),
                    "--expected-commit",
                    "a" * 40,
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
    argv = validator.call_args.args[0]
    assert argv[argv.index("--report-sha256") + 1] == sha256_file(raw)
    assert argv[argv.index("--preregistration-sha256") + 1] == sha256_file(preregistration)


def test_finalizer_binds_every_teacher_artifact_and_calibration_file(tmp_path: Path) -> None:
    values = {
        "tc0_raw": {"passed": True},
        "tc0_validation": {
            "passed": True,
            "protocol": "R3-TC0-cache-forward-compatibility-validation.v1",
        },
        "tf0_raw": {"passed": True},
        "tf0_validation": {
            "passed": True,
            "protocol": "R3-TF0-feature-backend-compatibility-validation.v1",
        },
        "t0": {"passed": True, "probe": "teacher_t0_real_qwen_integrity_upper_bound"},
        "set8_calibration": {"schema": "vision_memory.teacher-calibration-file.v1", "split": "train"},
        "transition16_calibration": {
            "schema": "vision_memory.teacher-calibration-file.v1",
            "split": "train",
        },
    }
    paths = {name: _write(tmp_path / f"{name}.json", value) for name, value in values.items()}
    for suite in ("set8", "transition16"):
        paths[f"{suite}_calibration_report"] = _write(
            tmp_path / f"{suite}_calibration_report.json",
            {
                "schema": "vision_memory.r3-teacher-calibration-report.v1",
                "suite": suite,
                "calibration_file_sha256": sha256_file(paths[f"{suite}_calibration"]),
            },
        )
    output = tmp_path / "final.json"
    argv = ["finalize", "--expected-commit", "a" * 40]
    for name, path in paths.items():
        argv.extend((f"--{name.replace('_', '-')}", str(path)))
    argv.extend(("--output", str(output)))

    assert step.main(argv) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["strict_order"] == [
        "R3-TC0",
        "R3-TF0",
        "T0",
        "CAL-Set8",
        "CAL-Transition16",
    ]
    assert set(report["artifacts"]) == set(paths)
    assert report["artifacts"]["set8_calibration"]["sha256"] == sha256_file(
        paths["set8_calibration"]
    )
