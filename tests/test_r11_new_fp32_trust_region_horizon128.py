from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.experiments import audit_r11_new_fp32_trust_region_horizon128 as prefix_audit
from scripts.experiments import run_r11_new_fp32_trust_region as parent_runner
from scripts.experiments import run_r11_new_fp32_trust_region_horizon128 as runner
from vision_memory.training import r11_new_fp32_trust_region as parent
from vision_memory.training import r11_new_fp32_trust_region_horizon128 as core


ROOT = Path(__file__).resolve().parents[1]


def test_config_is_hash_locked_and_changes_only_iteration_budget() -> None:
    config = core.load_config()
    base = parent.load_config()
    assert core.sha256_file(core.CONFIG_PATH) == core.CONFIG_BYTES_SHA256
    assert core.canonical_sha(config) == core.CONFIG_CANONICAL_SHA256
    assert config["experiment_variant"] == "horizon128"
    assert config["algorithm"] == {**base["algorithm"], "maximum_formal_iterations": 128}
    assert config["base_precision_contract"] == base["base_precision_contract"]
    assert config["fixed_fp32_anchor"] == base["fixed_fp32_anchor"]
    assert config["parent_precision_result"] == base["parent_precision_result"]
    assert config["technical_preflight_gate"] == base["technical_preflight_gate"]
    assert config["formal_gate"] == {
        **base["formal_gate"],
        "maximum_full_chain_forward_calls": 771,
        "maximum_backward_calls": 128,
        "maximum_parameter_updates": 128,
    }


def test_parent_delivery_bindings_match_committed_results() -> None:
    contract = core.load_config()["parent_trust_result"]
    delivery = ROOT / contract["source_delivery_root"]
    bindings = {
        "result.json": "source_result_sha256",
        "manifest.json": "source_manifest_sha256",
        "terminal.json": "source_terminal_sha256",
        "artifact_inventory.json": "source_inventory_sha256",
        "iteration_metrics.jsonl": "source_iteration_metrics_sha256",
        "candidate_metrics.jsonl": "source_candidate_metrics_sha256",
    }
    for name, key in bindings.items():
        assert core.sha256_file(delivery / name) == contract[key]
    summary = json.loads(
        (ROOT / "reports/r11-new-fp32-trust-region-results-20260907/derived/summary.json").read_text(encoding="utf-8")
    )
    assert summary["archive"]["sha256"] == contract["source_archive_sha256"]
    assert summary["formal_result"]["trace_summary"]["final_x_T_fp32_sha256"] == contract["final_x_T_fp32_sha256"]


def test_facade_reuses_exact_parent_numerical_and_audit_functions() -> None:
    assert core.PROTOCOL == parent.PROTOCOL
    assert core.PREFIX == parent.PREFIX
    assert core.RADII == parent.RADII
    assert core.unit_vector is parent.unit_vector
    assert core.summarize_trace is parent.summarize_trace
    assert core.classify_outcome is parent.classify_outcome
    assert core.audit_delivery is parent.audit_delivery


def test_runner_swaps_only_locked_config_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    original = parent_runner.core
    observed = {}

    def fake_main(argv: list[str] | None = None) -> int:
        observed["core"] = parent_runner.core
        observed["argv"] = argv
        return 17

    monkeypatch.setattr(parent_runner, "main", fake_main)
    assert runner.main(["formal"]) == 17
    assert observed == {"core": core, "argv": ["formal"]}
    assert parent_runner.core is original


def _write(path: Path, lines: list[str]) -> str:
    payload = "".join(f"{line}\n" for line in lines)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()


def test_external_prefix_audit_requires_byte_exact_parent_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_root = tmp_path / "parent"
    output_root = tmp_path / "output"
    parent_root.mkdir()
    output_root.mkdir()
    parent_iteration_sha = _write(parent_root / "iteration_metrics.jsonl", ['{"i":0}', '{"i":1}'])
    parent_candidate_sha = _write(parent_root / "candidate_metrics.jsonl", ['{"c":0}', '{"c":1}', '{"c":2}'])
    _write(output_root / "iteration_metrics.jsonl", ['{"i":0}', '{"i":1}', '{"i":2}'])
    _write(output_root / "candidate_metrics.jsonl", ['{"c":0}', '{"c":1}', '{"c":2}', '{"c":3}'])
    config = {
        "parent_trust_result": {
            "source_root": str(parent_root),
            "source_iteration_metrics_sha256": parent_iteration_sha,
            "source_candidate_metrics_sha256": parent_candidate_sha,
        },
        "parent_prefix_reproduction_gate": {
            "iteration_rows_exact_prefix_count": 2,
            "candidate_rows_exact_prefix_count": 3,
        },
    }
    monkeypatch.setattr(core, "load_config", lambda: config)
    monkeypatch.setattr(core, "audit_delivery", lambda *_args, **_kwargs: {"mode": "formal", "passed": True})
    monkeypatch.setattr(prefix_audit, "_verify_parent_bindings", lambda *_args, **_kwargs: {"passed": True})
    result = prefix_audit.audit_parent_prefix(output_root, parent_root)
    assert result["passed"] is True
    assert result["prefix"]["iteration_rows_byte_exact"] is True
    _write(output_root / "iteration_metrics.jsonl", ['{"i":9}', '{"i":1}', '{"i":2}'])
    with pytest.raises(ValueError, match="exactly reproduce"):
        prefix_audit.audit_parent_prefix(output_root, parent_root)
