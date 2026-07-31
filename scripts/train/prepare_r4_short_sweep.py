"""Expand the R4 short-sweep matrix into auditable run manifests and commands."""

from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any


def _resolve(value: str, root: Path) -> str:
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], value)
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def expand(config: dict[str, Any], *, root: Path, output_dir: Path) -> list[dict[str, Any]]:
    budget = config["budget"]
    runs: list[dict[str, Any]] = []
    for dataset in config["datasets"]:
        for arm in config["arms"]:
            name = f"{dataset['name']}__{arm['name']}"
            run_dir = output_dir / name
            runs.append({
                "schema": "vision_memory.r4-short-sweep-run.v1",
                "name": name,
                "dataset": dataset["name"],
                "train": _resolve(dataset["train"], root),
                "dev": _resolve(dataset["dev"], root),
                "output_dir": str(run_dir),
                "budget": budget,
                "arm": arm,
            })
    return runs


def command(run: dict[str, Any], *, dreamlite: str, reader: str, resolution: int = 1024) -> str:
    budget = run["budget"]
    arm = run["arm"]
    args = [
        "python", "scripts/train/dreamlite_r4_free_pixel.py",
        "--experimental",
        "--strict-determinism",
        "--train", run["train"],
        "--dev", run["dev"],
        "--dreamlite", dreamlite,
        "--reader", reader,
        "--output-dir", run["output_dir"],
        "--resolution", str(resolution),
        "--max-optimizer-steps", str(budget["optimizer_steps"]),
        "--set-warmup-optimizer-steps", str(budget["set_warmup_optimizer_steps"]),
        "--gradient-accumulation", str(budget["gradient_accumulation"]),
        "--eval-limit", str(budget["eval_limit"]),
        "--identity-calibration-count", str(budget["identity_calibration_count"]),
        "--checkpoint-every", str(budget["checkpoint_every"]),
        "--learning-rate", str(arm["learning_rate"]),
        "--weight-decay", str(arm["weight_decay"]),
        "--gradient-clip", str(arm["gradient_clip"]),
        "--lora-rank", str(arm["lora_rank"]),
        "--selected-step-count", str(arm["selected_step_count"]),
        "--optimizer-diagnostics",
        "--record-micro-metrics",
        "--record-validation-metrics",
        "--validation-every", str(budget.get("validation_every", budget["checkpoint_every"])),
        "--save-step-zero",
    ]
    return shlex.join(args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dreamlite", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    runs = expand(config, root=args.root, output_dir=args.output_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        "\n".join(json.dumps(run, ensure_ascii=False, sort_keys=True) for run in runs) + "\n",
        encoding="utf-8",
    )
    commands = args.manifest.with_suffix(".sh")
    commands.write_text(
        "\n".join(command(run, dreamlite=args.dreamlite, reader=args.reader) for run in runs) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"run_count": len(runs), "manifest": str(args.manifest), "commands": str(commands)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

