from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probes" / "r15_schedule_preflight.py"
SPEC = importlib.util.spec_from_file_location("r15_schedule_preflight_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def test_r15_preflight_hashes_files_without_model_dependencies(tmp_path: Path) -> None:
    value = tmp_path / "value.bin"
    value.write_bytes(b"abc")
    assert preflight._sha256_file(value) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_r15_preflight_source_contains_no_model_loading_or_inference() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = (
        "from_pretrained",
        "_load_pipeline",
        "_load_reader",
        "choice_reader_callable",
        "cuda:",
        "reader_fn",
    )
    assert all(token not in source for token in forbidden)
    assert '"model_calls": 0' in source
    assert '"reader_calls_executed": 0' in source


def test_r15_preflight_parser_requires_train_and_output() -> None:
    parser = preflight.build_parser()
    args = parser.parse_args(["--train", "train.jsonl", "--output", "receipt.json"])
    assert args.train == Path("train.jsonl")
    assert args.output == Path("receipt.json")
    assert args.expected_schedule_sha256 is None


def test_r15_config_binds_preregistration_schedule_and_fixed_gate() -> None:
    config_path = ROOT / "configs" / "experiments" / "r15_synchronous_round_robin.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preregistration = ROOT / config["preregistration"]["path"]
    preregistration_sha256 = hashlib.sha256(preregistration.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert config["status"] == "preregistered_before_any_r15_model_outcome"
    assert preregistration_sha256 == config["preregistration"]["sha256"]
    assert (
        config["training_schedule"]["schedule_sha256"]
        == config["model_free_preflight"]["schedule_sha256"]
        == "2495ce15ed5242b88f1d88b6caed29694a6340a9982c3870b1720004cf75ffb8"
    )
    assert config["optimization"]["pair_gradient_accumulation"] == 2
    assert config["optimization"]["backward_loss_divisor"] == 4.0
    assert config["optimization"]["additional_accumulation_divisor"] is None
    assert config["optimization"]["checkpoints"] == [0, 324, 648, 972, 1296]
    assert config["arm_gate"] == {
        "required_train_audit_passes": 36,
        "required_dev_select_passes": 24,
        "required_dev_replay_passes": 24,
        "required_dev_final_passes": 24,
        "partial_counts_are_diagnostic_only": True,
        "formal_success_claim": False,
    }
    assert config["success_boundary"]["diagnostic_only"] is True
    assert config["success_boundary"]["cannot_establish_full_picture_memory_success"] is True
