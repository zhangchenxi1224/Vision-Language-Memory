"""Fit event-only Controller; emit xT predictions for actual frozen DreamLite/Reader QA."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_memory.training.frozen_oracle_writer import train_controller  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--geometry-decision", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("overfit", "heldout"), default="overfit")
    parser.add_argument("--overfit-evaluation", type=Path)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--basis-rank", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--overfit-tasks", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        result = train_controller(bank_path=args.bank, decision_path=args.geometry_decision, output=args.output,
                                  phase=args.phase, overfit_evaluation=args.overfit_evaluation, steps=args.steps,
                                  learning_rate=args.learning_rate, seed=args.seed, basis_rank=args.basis_rank,
                                  hidden=args.hidden, overfit_tasks=args.overfit_tasks, device=args.device)
    except (ValueError, KeyError, OSError) as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({key: result[key] for key in ("status", "formal_success", "prediction_count", "evaluation_spec")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
