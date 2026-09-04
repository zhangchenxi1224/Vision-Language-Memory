from __future__ import annotations

import importlib.util
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
