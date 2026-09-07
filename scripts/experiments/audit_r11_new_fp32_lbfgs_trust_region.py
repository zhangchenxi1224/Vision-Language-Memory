"""Independently audit L-BFGS output and its exact first-step PRP+ reproduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training import r11_new_fp32_lbfgs_trust_region as core  # noqa: E402
from vision_memory.training import r11_new_fp32_prpplus_trust_region as parent_core  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path)
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    core.require(text.endswith("\n"), f"JSONL lacks terminal newline: {path}")
    return [json.loads(line) for line in text.splitlines()]


def _verify_parent_bindings(parent_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    external_audit = parent_root.parent / "r11-new-prpplus-de26e3a-20260907-round01-parent-audit.json"
    bindings = {
        "result": (parent_root / "result.json", contract["source_result_sha256"]),
        "manifest": (parent_root / "manifest.json", contract["source_manifest_sha256"]),
        "terminal": (parent_root / "terminal.json", contract["source_terminal_sha256"]),
        "inventory": (parent_root / "artifact_inventory.json", contract["source_inventory_sha256"]),
        "iterations": (parent_root / "iteration_metrics.jsonl", contract["source_iteration_metrics_sha256"]),
        "candidates": (parent_root / "candidate_metrics.jsonl", contract["source_candidate_metrics_sha256"]),
        "tensor_bundle": (parent_root / "trust_region_tensors.pt", contract["source_tensor_bundle_sha256"]),
        "final_checkpoint": (
            parent_root / "checkpoints/iteration-63.pt",
            contract["source_final_checkpoint_sha256"],
        ),
        "archive": (Path(contract["source_archive_path"]), contract["source_archive_sha256"]),
        "external_parent_audit": (external_audit, contract["source_external_parent_audit_sha256"]),
    }
    for name, (path, expected) in bindings.items():
        core.require(path.is_file() and core.sha256_file(path) == expected, f"L-BFGS parent {name} drift.")
    parent_config = parent_core.load_config()
    parent_audit = parent_core.audit_delivery(parent_root, parent_config)
    core.require(
        parent_audit["mode"] == "formal"
        and parent_audit["classification"] == contract["classification"]
        and parent_audit["trace_summary"]["iteration_count"] == contract["iteration_count"]
        and parent_audit["trace_summary"]["accepted_update_count"] == contract["accepted_update_count"]
        and parent_audit["trace_summary"]["final_x_T_fp32_sha256"] == contract["final_x_T_fp32_sha256"]
        and parent_audit["trace_summary"]["final_endpoint_fp32_sha256"] == contract["final_endpoint_fp32_sha256"]
        and parent_audit["trace_summary"]["stop_reason"] == contract["stop_reason"]
        and parent_audit["formal_success"] is False
        and parent_audit["phase2_allowed"] is False,
        "L-BFGS parent scientific identity drift.",
    )
    return {
        "passed": True,
        "audit": parent_audit,
        "bindings": {name: {"path": str(path), "sha256": expected} for name, (path, expected) in bindings.items()},
    }


def _common(mapping: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    return {name: mapping[name] for name in names}


def audit_parent_first_step(output_root: Path, parent_root: Path | None = None) -> dict[str, Any]:
    config = core.load_config()
    contract = config["parent_prpplus_result"]
    output_root = output_root.resolve()
    parent_root = Path(contract["source_root"]).resolve() if parent_root is None else parent_root.resolve()
    run_audit = core.audit_delivery(output_root, config)
    core.require(run_audit["mode"] == "formal", "Parent audit requires formal L-BFGS output.")
    parent = _verify_parent_bindings(parent_root, contract)
    parent_iteration = _read_jsonl(parent_root / "iteration_metrics.jsonl")[0]
    run_iteration = _read_jsonl(output_root / "iteration_metrics.jsonl")[0]
    parent_candidates = _read_jsonl(parent_root / "candidate_metrics.jsonl")[: len(core.RADII)]
    run_candidates = _read_jsonl(output_root / "candidate_metrics.jsonl")[: len(core.RADII)]
    iteration_fields = (
        "iteration_id",
        "iteration",
        "current_loss",
        "current_loss_ratio_to_plateau",
        "current_x_T_fp32_sha256",
        "current_endpoint_fp32_sha256",
        "gradient_fp32_sha256",
        "direction_fp32_sha256",
        "gradient_norm",
        "gradient_nonzero_fraction",
        "analytic_directional_derivative",
        "selected_candidate_id",
        "selected_radius_index",
        "selected_radius_l2",
        "selected_loss",
        "selected_loss_ratio_to_plateau",
        "selected_x_T_fp32_sha256",
        "selected_endpoint_fp32_sha256",
        "selected_candidate_residual_l2",
        "relative_improvement",
        "accepted",
    )
    candidate_fields = (
        "candidate_id",
        "iteration",
        "radius_index",
        "radius_l2",
        "candidate_x_T_fp32_sha256",
        "endpoint_fp32_sha256",
        "loss",
        "loss_ratio_to_plateau",
        "loss_ratio_to_current",
    )
    iteration_exact = _common(run_iteration, iteration_fields) == _common(parent_iteration, iteration_fields)
    candidates_exact = all(
        _common(run, candidate_fields) == _common(parent_row, candidate_fields)
        for run, parent_row in zip(run_candidates, parent_candidates, strict=True)
    )
    core.require(iteration_exact and candidates_exact, "L-BFGS first step does not exactly reproduce PRP+.")
    core.require(
        run_iteration["restart_reason"] == "initial"
        and run_iteration["curvature_pair_iteration"] is None
        and run_iteration["curvature_pair_accepted"] is None
        and run_iteration["curvature_pair_reason"] == "initial"
        and run_iteration["history_pair_iterations_before_direction"] == []
        and run_iteration["history_pair_iterations_used"] == []
        and run_iteration["history_pair_iterations_after_direction"] == []
        and run_iteration["history_cleared"] is False
        and run_iteration["initial_inverse_hessian_scale"] == 1.0
        and run_iteration["two_loop_coefficients"] == [],
        "L-BFGS initial recurrence metadata drift.",
    )
    return {
        "passed": True,
        "output_root": str(output_root),
        "parent_root": str(parent_root),
        "run_audit": run_audit,
        "parent": parent,
        "first_step": {
            "iteration_common_fields_exact": iteration_exact,
            "all_five_candidate_common_fields_exact": candidates_exact,
            "initial_history_empty": True,
            "initial_direction_is_negative_gradient": True,
        },
        "classification_accepted": True,
        "formal_picture_memory_success": False,
        "phase2_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        json.dumps(
            audit_parent_first_step(args.output_root, args.parent_root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
