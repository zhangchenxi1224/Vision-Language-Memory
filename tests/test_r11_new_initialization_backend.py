from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "r11_new_initialization_backend_probe",
    ROOT / "scripts" / "experiments" / "probe_r11_new_initialization_backend.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_cpu_cli_produces_parseable_identical_stdout_and_json_artifact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "cpu_probe.json"
    assert probe.main(["--device", "cpu", "--output", str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert report["schema"] == probe.SCHEMA
    assert report["environment"]["actual_compute_device_type"] == "cpu"
    assert report["contract"]["seed"] == 0
    assert report["contract"]["shape"] == [1, 4, 128, 128]
    assert report["contract"]["sigma"] == 0.4999999701976776
    assert report["contract"]["model_forward_calls"] == 0
    assert report["contract"]["training_steps"] == 0
    assert report["contract"]["h0_comparison_uses_same_device_produced_x_T"] is True
    assert report["interpretation"]["model_or_scientific_result"] is False
    assert report["interpretation"]["cpu_mode_is_cpu_only_smoke_test"] is True
    assert report["interpretation"]["cross_backend_difference_required_for_success"] is False
    assert len(report["probe_script_sha256"]) == 64
    assert set(report["comparisons"]) == {
        "cpu_vs_device_x_T", "cpu_vs_device_h0_with_shared_device_x_T",
        "device_repeat_x_T", "device_repeat_h0",
    }
    for comparison in report["comparisons"].values():
        assert comparison["element_count"] == 65536
        assert comparison["different_element_count"] == 0
        assert comparison["max_abs_difference"] == 0.0
        assert comparison["torch_equal"] is True
        assert comparison["raw_bytes_equal"] is True


def test_cpu_stdout_only_and_repeated_seed_are_reproducible(capsys: pytest.CaptureFixture[str]) -> None:
    assert probe.main(["--device", "cpu"]) == 0
    first = json.loads(capsys.readouterr().out)
    second = probe.run_probe("cpu")
    assert first["inputs"] == second["inputs"]
    assert first["comparisons"] == second["comparisons"]


def test_existing_output_is_rejected_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "existing.json"
    output.write_text('{"preserve": true}\n', encoding="utf-8")

    def unexpected_probe(_: str) -> None:
        pytest.fail("An existing output must be rejected before running the probe.")

    monkeypatch.setattr(probe, "run_probe", unexpected_probe)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        probe.main(["--device", "cpu", "--output", str(output)])
    assert output.read_text(encoding="utf-8") == '{"preserve": true}\n'


def test_cuda_request_never_silently_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is forbidden"):
        probe.run_probe("cuda:0")


def test_unregistered_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="Only explicit CPU or CUDA"):
        probe.run_probe("meta")
