"""Independently audit horizon-128 output and its exact 32-step parent prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import r11_new_fp32_trust_region as parent_core  # noqa: E402
from vision_memory.training import r11_new_fp32_trust_region_horizon128 as core  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    core.require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    core.require(text.endswith("\n"), f"JSONL lacks terminal newline: {path}")
    return text.splitlines(keepends=True)


def _prefix_sha256(lines: list[str]) -> str:
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def _verify_parent_bindings(parent_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    bindings = {
        "result": (parent_root / "result.json", contract["source_result_sha256"]),
        "manifest": (parent_root / "manifest.json", contract["source_manifest_sha256"]),
        "terminal": (parent_root / "terminal.json", contract["source_terminal_sha256"]),
        "inventory": (parent_root / "artifact_inventory.json", contract["source_inventory_sha256"]),
        "iterations": (parent_root / "iteration_metrics.jsonl", contract["source_iteration_metrics_sha256"]),
        "candidates": (parent_root / "candidate_metrics.jsonl", contract["source_candidate_metrics_sha256"]),
        "tensor_bundle": (parent_root / "trust_region_tensors.pt", contract["source_tensor_bundle_sha256"]),
        "final_checkpoint": (
            parent_root / "checkpoints/iteration-31.pt",
            contract["source_final_checkpoint_sha256"],
        ),
    }
    for name, (path, expected) in bindings.items():
        core.require(path.is_file() and core.sha256_file(path) == expected, f"Horizon128 parent {name} drift.")
    parent_config = _read_json(parent_root / "config.json")
    parent_audit = parent_core.audit_delivery(parent_root, parent_config)
    core.require(
        parent_audit["mode"] == "formal"
        and parent_audit["classification"] == contract["classification"]
        and parent_audit["trace_summary"]["iteration_count"] == contract["iteration_count"]
        and parent_audit["trace_summary"]["final_x_T_fp32_sha256"] == contract["final_x_T_fp32_sha256"]
        and parent_audit["formal_success"] is False
        and parent_audit["phase2_allowed"] is False,
        "Horizon128 parent scientific identity drift.",
    )
    return {
        "passed": True,
        "audit": parent_audit,
        "bindings": {name: {"path": str(path), "sha256": expected} for name, (path, expected) in bindings.items()},
    }


def audit_parent_prefix(output_root: Path, parent_root: Path | None = None) -> dict[str, Any]:
    config = core.load_config()
    contract = config["parent_trust_result"]
    output_root = output_root.resolve()
    parent_root = Path(contract["source_root"]).resolve() if parent_root is None else parent_root.resolve()
    run_audit = core.audit_delivery(output_root, config)
    core.require(run_audit["mode"] == "formal", "Parent-prefix audit requires formal horizon128 output.")
    parent = _verify_parent_bindings(parent_root, contract)
    gate = config["parent_prefix_reproduction_gate"]
    iteration_count = gate["iteration_rows_exact_prefix_count"]
    candidate_count = gate["candidate_rows_exact_prefix_count"]
    parent_iterations = _lines(parent_root / "iteration_metrics.jsonl")
    parent_candidates = _lines(parent_root / "candidate_metrics.jsonl")
    run_iterations = _lines(output_root / "iteration_metrics.jsonl")
    run_candidates = _lines(output_root / "candidate_metrics.jsonl")
    core.require(
        len(parent_iterations) == iteration_count
        and len(parent_candidates) == candidate_count
        and len(run_iterations) >= iteration_count
        and len(run_candidates) >= candidate_count,
        "Horizon128 parent-prefix coverage drift.",
    )
    iteration_exact = run_iterations[:iteration_count] == parent_iterations
    candidate_exact = run_candidates[:candidate_count] == parent_candidates
    core.require(iteration_exact and candidate_exact, "Horizon128 does not exactly reproduce the parent metric prefix.")
    core.require(
        _prefix_sha256(run_iterations[:iteration_count]) == contract["source_iteration_metrics_sha256"]
        and _prefix_sha256(run_candidates[:candidate_count]) == contract["source_candidate_metrics_sha256"],
        "Horizon128 reproduced-prefix hash drift.",
    )
    return {
        "passed": True,
        "output_root": str(output_root),
        "parent_root": str(parent_root),
        "run_audit": run_audit,
        "parent": parent,
        "prefix": {
            "iteration_rows": iteration_count,
            "candidate_rows": candidate_count,
            "iteration_rows_byte_exact": iteration_exact,
            "candidate_rows_byte_exact": candidate_exact,
            "iteration_prefix_sha256": contract["source_iteration_metrics_sha256"],
            "candidate_prefix_sha256": contract["source_candidate_metrics_sha256"],
        },
        "classification_accepted": True,
        "formal_picture_memory_success": False,
        "phase2_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            audit_parent_prefix(args.output_root, args.parent_root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
