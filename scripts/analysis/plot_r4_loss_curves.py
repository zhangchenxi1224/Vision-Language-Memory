"""Build train/validation loss curves from an R4 metrics.jsonl file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _loss(row: dict[str, Any]) -> float | None:
    for key in ("loss_mean", "validation_loss", "loss", "mean_episode_listwise_choice_ce"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def load_curve_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        kind = str(value.get("kind", ""))
        step = value.get("optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int):
            continue
        loss = _loss(value)
        if loss is None:
            continue
        if kind in {"optimizer_step", "train"}:
            series = "train"
        elif kind in {"validation", "dev", "fixed_evaluation_m0", "fixed_evaluation_final"}:
            series = "validation"
        else:
            continue
        rows.append({"kind": kind, "series": series, "optimizer_step": step, "loss": loss})
    return rows


def write_curve_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("kind", "series", "optimizer_step", "loss"))
        writer.writeheader()
        writer.writerows(rows)


def _rolling(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        output.append(sum(values[start : index + 1]) / (index - start + 1))
    return output


def plot(rows: list[dict[str, Any]], output: Path, *, rolling_window: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [row for row in rows if row["series"] == "train"]
    validation = [row for row in rows if row["series"] == "validation"]
    figure, axis = plt.subplots(figsize=(9, 5.5), dpi=160)
    if train:
        train_steps = [row["optimizer_step"] for row in train]
        train_values = [row["loss"] for row in train]
        axis.plot(train_steps, train_values, color="#4c78a8", alpha=0.28, linewidth=0.8, label="train loss")
        axis.plot(
            train_steps,
            _rolling(train_values, rolling_window),
            color="#1f4e79",
            linewidth=1.8,
            label=f"train loss ({rolling_window}-step mean)",
        )
    if validation:
        axis.plot(
            [row["optimizer_step"] for row in validation],
            [row["loss"] for row in validation],
            color="#d62728",
            marker="o",
            linewidth=1.8,
            label="validation loss",
        )
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("loss / fixed dev CE")
    axis.set_title("DreamLite R4 train and validation loss")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=8)
    args = parser.parse_args()
    if args.rolling_window <= 0:
        raise SystemExit("--rolling-window must be positive")
    rows = load_curve_rows(args.metrics)
    if not rows:
        raise SystemExit("metrics file contains no train or validation loss rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_curve_csv(rows, args.output_dir / "loss_curve.csv")
    plot(rows, args.output_dir / "train_validation_loss.png", rolling_window=args.rolling_window)
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
