"""Generate the model-free R15 schedule receipt from the locked formal train file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import read_jsonl  # noqa: E402
from vision_memory.training.r5_compose import build_r5_family_pools  # noqa: E402
from vision_memory.training.r12_shared_writer import (  # noqa: E402
    R12_TRAIN_AUDIT_SHA256,
    R12_TRAIN_SELECTION_SHA256,
    select_balanced_train_f1,
    selection_audit,
)
from vision_memory.training.r15_synchronous_round_robin import (  # noqa: E402
    R15_BACKWARD_LOSS_DIVISOR,
    R15_CHECKPOINT_STEPS,
    R15_DIRECTIONAL_EXAMPLES,
    R15_EPOCHS,
    R15_OPTIMIZER_STEPS,
    R15_PAIR_GRADIENT_ACCUMULATION,
    R15_PAIR_MICRO_STEPS,
    R15_PAIR_SEED,
    R15_POOL_PAIRING_SEED,
    R15_READER_CALLS,
    build_synchronous_round_robin_schedule,
    synchronous_schedule_audit,
)


SCHEMA = "vision_memory.r15-synchronous-round-robin-source-preflight.v1"
EXPECTED_TRAIN_FILE_SHA256 = "24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-schedule-sha256")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(
    *,
    train: Path,
    output: Path,
    expected_schedule_sha256: str | None = None,
) -> dict[str, Any]:
    if not train.is_file():
        raise ValueError(f"R15 preflight is missing the formal train file: {train}")
    train_sha256 = _sha256_file(train)
    if train_sha256 != EXPECTED_TRAIN_FILE_SHA256:
        raise ValueError(
            f"R15 formal train hash drift: expected {EXPECTED_TRAIN_FILE_SHA256}, observed {train_sha256}."
        )
    records = tuple(read_jsonl(train))
    pools = build_r5_family_pools(records, pairing_seed=R15_POOL_PAIRING_SEED)
    train_selected, train_audit = select_balanced_train_f1(pools["F1"])
    selected_receipt = selection_audit(train_selected)
    audit_receipt = selection_audit(train_audit)
    if (
        selected_receipt["payload_sha256"] != R12_TRAIN_SELECTION_SHA256
        or audit_receipt["payload_sha256"] != R12_TRAIN_AUDIT_SHA256
    ):
        raise RuntimeError("R15 fixed R12/R14 training selections drifted.")
    schedule = build_synchronous_round_robin_schedule(train_selected)
    schedule_receipt = synchronous_schedule_audit(train_selected, schedule)
    if not schedule_receipt["passed"]:
        raise RuntimeError(f"R15 model-free schedule audit failed: {schedule_receipt['checks']}")
    observed_schedule_sha256 = schedule_receipt["schedule_sha256"]
    if expected_schedule_sha256 is not None and observed_schedule_sha256 != expected_schedule_sha256:
        raise RuntimeError(
            f"R15 schedule hash drift: expected {expected_schedule_sha256}, observed {observed_schedule_sha256}."
        )
    checks = {
        "source_only_no_model_outcome": True,
        "formal_train_file_hash": train_sha256 == EXPECTED_TRAIN_FILE_SHA256,
        "fixed_train_selection": selected_receipt["payload_sha256"] == R12_TRAIN_SELECTION_SHA256,
        "fixed_train_audit_selection": audit_receipt["payload_sha256"] == R12_TRAIN_AUDIT_SHA256,
        "schedule_audit": schedule_receipt["passed"] is True,
        "pair_micro_steps": schedule_receipt["pair_micro_steps"] == R15_PAIR_MICRO_STEPS,
        "directional_examples": schedule_receipt["directional_examples"] == R15_DIRECTIONAL_EXAMPLES,
        "reader_calls": schedule_receipt["reader_calls"] == R15_READER_CALLS,
        "optimizer_steps": schedule_receipt["optimizer_steps"] == R15_OPTIMIZER_STEPS,
        "checkpoints": R15_CHECKPOINT_STEPS == (0, 324, 648, 972, 1296),
        "single_loss_scaling": schedule_receipt["backward_loss_divisor"] == R15_BACKWARD_LOSS_DIVISOR,
        "two_atomic_pairs_per_update": (
            schedule_receipt["pair_gradient_accumulation"] == R15_PAIR_GRADIENT_ACCUMULATION
            and schedule_receipt["checks"]["optimizer_atomic_distinct_pairs"] is True
            and schedule_receipt["checks"]["no_cross_round_accumulation"] is True
        ),
        "full_wrong_value_coverage": schedule_receipt["checks"]["every_wrong_target_seen"] is True,
        "balanced_member_rotation": schedule_receipt["checks"]["balanced_donor_member_ranks"] is True,
        "balanced_choice_views": schedule_receipt["checks"]["balanced_choice_views"] is True,
    }
    payload = {
        "schema": SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "model_calls": 0,
        "reader_calls_executed": 0,
        "checks": checks,
        "train_file": str(train.resolve()),
        "train_file_sha256": train_sha256,
        "pool_pairing_seed": R15_POOL_PAIRING_SEED,
        "schedule_seed": R15_PAIR_SEED,
        "epochs": R15_EPOCHS,
        "train_selection": selected_receipt,
        "train_audit_selection": audit_receipt,
        "schedule": schedule_receipt,
    }
    if not payload["passed"]:
        raise RuntimeError(f"R15 source preflight failed closed: {checks}")
    _write_json(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(
        train=args.train,
        output=args.output,
        expected_schedule_sha256=args.expected_schedule_sha256,
    )
    print(
        json.dumps(
            {
                "passed": payload["passed"],
                "train_selection_sha256": payload["train_selection"]["payload_sha256"],
                "schedule_sha256": payload["schedule"]["schedule_sha256"],
                "pair_micro_steps": payload["schedule"]["pair_micro_steps"],
                "optimizer_steps": payload["schedule"]["optimizer_steps"],
                "model_calls": payload["model_calls"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
