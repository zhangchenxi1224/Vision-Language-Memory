"""Model-free formal-input preflight for the preregistered R14 experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train import r14_symmetric_donor_writer as trainer  # noqa: E402
from vision_memory.training.r12_shared_writer import (  # noqa: E402
    R12_DEV_FINAL_SHA256,
    R12_DEV_SELECT_SHA256,
    R12_MICRO_STEPS,
    R12_OPTIMIZER_STEPS,
    R12_TRAIN_AUDIT_SHA256,
    R12_TRAIN_SELECTION_SHA256,
)
from vision_memory.training.r14_symmetric_donor import (  # noqa: E402
    R14_FRESH_DEV_FINAL_SHA256,
    R14_PAIR_SEED,
)


SCHEMA = "vision_memory.r14-symmetric-donor-source-preflight.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--r12-conditioned-root", type=Path, required=True)
    parser.add_argument("--r12-control-root", type=Path, required=True)
    parser.add_argument("--r12-comparison", type=Path, required=True)
    parser.add_argument("--r12-collapse-audit", type=Path, required=True)
    parser.add_argument("--r13-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser


def _trainer_args(cli: argparse.Namespace) -> argparse.Namespace:
    return trainer.parse_args(
        [
            "--train",
            str(cli.train),
            "--dev",
            str(cli.dev),
            "--dreamlite",
            str(cli.dreamlite),
            "--reader",
            str(cli.reader),
            "--r12-conditioned-root",
            str(cli.r12_conditioned_root),
            "--r12-control-root",
            str(cli.r12_control_root),
            "--r12-comparison",
            str(cli.r12_comparison),
            "--r12-collapse-audit",
            str(cli.r12_collapse_audit),
            "--r13-root",
            str(cli.r13_root),
            "--output-dir",
            str(cli.report.parent / ".r14-source-preflight-output-must-not-exist"),
            "--seed",
            "0",
            "--dreamlite-device",
            "cuda:0",
            "--reader-device",
            "cuda:1",
            "--strict-determinism",
        ]
    )


def run(cli: argparse.Namespace) -> dict[str, object]:
    if cli.report.exists():
        raise ValueError(f"R14 source preflight refuses an existing report: {cli.report}")
    head = trainer.r5.git_value("rev-parse", "HEAD")
    if head != cli.expected_commit:
        raise ValueError(f"R14 source preflight commit mismatch: expected {cli.expected_commit}, got {head}")

    args = _trainer_args(cli)
    validation = trainer._validate_args(args)
    data = trainer.r5._load_data(args, optimizer_steps=0)
    train_selected, train_audit = trainer.select_balanced_train_f1(data.train_pools["F1"])
    dev_records = tuple(trainer.r5.read_episode_jsonl(args.dev))
    dev_pool = trainer.r5.build_r5_family_pools(dev_records, pairing_seed=args.split_seed)["F1"]
    dev_select, dev_replay = trainer.select_entity_disjoint_dev_f1(dev_pool)
    excluded_entities = {segment.query_entity_id for segment in (*train_selected, *dev_select, *dev_replay)}
    dev_final = trainer.select_fresh_dev_final_f1(dev_pool, excluded_entities=excluded_entities)
    selections = {
        "train": train_selected,
        "train_audit": train_audit,
        "dev_select": dev_select,
        "dev_replay": dev_replay,
        "dev_final": dev_final,
    }
    selection_audits = {name: trainer.selection_audit(values) for name, values in selections.items()}
    expected_selection_hashes = {
        "train": R12_TRAIN_SELECTION_SHA256,
        "train_audit": R12_TRAIN_AUDIT_SHA256,
        "dev_select": R12_DEV_SELECT_SHA256,
        "dev_replay": R12_DEV_FINAL_SHA256,
        "dev_final": R14_FRESH_DEV_FINAL_SHA256,
    }
    observed_selection_hashes = {name: audit["payload_sha256"] for name, audit in selection_audits.items()}
    relationship_audit = trainer._selection_relationship_audit(selections)

    pairing_map = trainer.symmetric_donor_mapping(train_selected, seed=args.pairing_seed)
    pairing_audit = trainer.symmetric_pairing_audit(train_selected, pairing_map, seed=args.pairing_seed)
    schedule = trainer.build_training_schedule(train_selected)
    schedule_sha256 = trainer.r5.canonical_sha256([unit.receipt() for unit in schedule])
    config = validation["config"]
    checks = {
        "exact_clean_commit": validation["git_commit"] == cli.expected_commit and validation["git_dirty"] is False,
        "formal_data_hashes": validation["data_sha256"]
        == {
            "train": config["fixed_data"]["train_sha256"],
            "dev": config["fixed_data"]["dev_sha256"],
        },
        "r12_and_r13_parent_hashes_validated": bool(
            validation["conditioned_paths"] and validation["control_paths"] and validation["r13_parent_paths"]
        ),
        "selection_hashes": observed_selection_hashes == expected_selection_hashes,
        "selection_relationships": relationship_audit["passed"] is True,
        "fixed_schedule": len(schedule) == R12_MICRO_STEPS
        and config["optimization"]["micro_steps"] == R12_MICRO_STEPS
        and config["optimization"]["optimizer_steps"] == R12_OPTIMIZER_STEPS,
        "symmetric_pairing": pairing_audit["seed"] == R14_PAIR_SEED
        and pairing_audit["segment_count"] == 144
        and pairing_audit["pair_count"] == 72
        and pairing_audit["different_target_value"] is True
        and pairing_audit["involution"] is True
        and pairing_audit["pairs_sha256"] == config["training_pairing"]["pairs_sha256"],
        "no_model_loading_or_reader_call": True,
    }
    report: dict[str, object] = {
        "schema": SCHEMA,
        "status": "completed_source_only" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "git_commit": head,
        "config_sha256": validation["config_sha256"],
        "preregistration_sha256": validation["r14_preregistration_sha256"],
        "data_sha256": validation["data_sha256"],
        "parent_source_sha256": {
            "r12_conditioned": config["activation"]["conditioned_source_sha256"],
            "r12_control": config["activation"]["control_source_sha256"],
            "r13": validation["r13_source_sha256"],
            "r13_reports": validation["r13_report_sha256"],
        },
        "selection_audits": selection_audits,
        "selection_relationship_audit": relationship_audit,
        "training_pairing": pairing_audit,
        "schedule": {
            "micro_steps": len(schedule),
            "optimizer_steps": R12_OPTIMIZER_STEPS,
            "receipts_sha256": schedule_sha256,
        },
        "checks": checks,
        "model_outcomes_observed": False,
    }
    if not report["passed"]:
        raise RuntimeError(f"R14 source preflight failed: {checks}")
    trainer._write_json(cli.report, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run(build_parser().parse_args(argv))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
