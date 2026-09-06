from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.eval.score_qwen_history_r4 import (  # noqa: E402
    load_jsonl,
    prediction_identity,
    sha256_file,
    validate_rows,
)
from vision_memory.eval.r4_history_representations import (  # noqa: E402
    QWEN_R4_LAST_EFFECTIVE_EVENT,
    QWEN_R4_OPERATION_TAGGED_HISTORY,
    QWEN_R4_RAW_HISTORY,
)
from vision_memory.repro import canonical_object_sha256  # noqa: E402


SCHEMA = "vlm.qwen-history-r4-comparison.v1"


def _correct(row: Mapping[str, Any]) -> int:
    return int(row["prediction_index"] == row["target_index"])


def _paired_identity_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    selected = [row for row in rows if row.get("condition") == "standard"]
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in selected:
        identity = prediction_identity(row)
        if identity in result:
            raise ValueError(f"Duplicate standard comparison identity: {identity!r}")
        result[identity] = row
    if not result:
        raise ValueError("R4 comparison requires at least one standard prediction")
    return result


def _assert_semantic_match(
    identity: tuple[Any, ...], rows: Sequence[Mapping[str, Any]]
) -> None:
    reference = rows[0]
    fields = (
        "episode_id",
        "query_id",
        "query_ordinal",
        "probe_role",
        "choice_view_family",
        "choice_view_index",
        "condition",
        "query_text_sha256",
        "source_event_stream_sha256",
        "dataset_sha256",
        "episodes_sha256",
        "choices",
        "target_index",
        "target_text",
        "semantic_group_id",
        "subtype",
        "form",
        "split",
        "ood_group",
    )
    for field in fields:
        if any(row.get(field) != reference.get(field) for row in rows[1:]):
            raise ValueError(f"Strict R4 arm pairing changed {field!r} at identity {identity!r}")
    group = reference.get("semantic_group_id")
    if not isinstance(group, str) or not group:
        raise ValueError(f"Comparison identity {identity!r} lacks semantic_group_id")


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _semantic_group_bootstrap(
    row_differences: Sequence[tuple[str, int]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("bootstrap_iterations must be positive")
    grouped: dict[str, list[int]] = defaultdict(list)
    for group, difference in row_differences:
        grouped[group].append(difference)
    groups = sorted(grouped)
    if not groups:
        raise ValueError("Semantic-group bootstrap received no paired rows")
    observed = sum(value for _, value in row_differences) / len(row_differences)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_groups = [groups[rng.randrange(len(groups))] for _ in groups]
        values = [value for group in sampled_groups for value in grouped[group]]
        samples.append(sum(values) / len(values))
    return {
        "difference": observed,
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "iterations": iterations,
        "seed": seed,
        "resampling_unit": "semantic_group_id",
        "semantic_groups": len(groups),
        "paired_rows": len(row_differences),
    }


def _arm_comparison(
    rows_a: Mapping[tuple[Any, ...], Mapping[str, Any]],
    rows_b: Mapping[tuple[Any, ...], Mapping[str, Any]],
    *,
    name: str,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if set(rows_a) != set(rows_b):
        missing_a = sorted(set(rows_b) - set(rows_a), key=repr)[:5]
        missing_b = sorted(set(rows_a) - set(rows_b), key=repr)[:5]
        raise ValueError(
            f"{name} identity sets differ; missing_from_a={missing_a!r}, missing_from_b={missing_b!r}"
        )
    differences: list[tuple[str, int]] = []
    correct_a = 0
    correct_b = 0
    for identity in sorted(rows_a, key=repr):
        row_a = rows_a[identity]
        row_b = rows_b[identity]
        _assert_semantic_match(identity, (row_a, row_b))
        result_a = _correct(row_a)
        result_b = _correct(row_b)
        correct_a += result_a
        correct_b += result_b
        differences.append((str(row_a["semantic_group_id"]), result_b - result_a))
    bootstrap = _semantic_group_bootstrap(differences, iterations=iterations, seed=seed)
    return {
        "name": name,
        "arm_a_correct": correct_a,
        "arm_b_correct": correct_b,
        "count": len(differences),
        "arm_a_accuracy": correct_a / len(differences),
        "arm_b_accuracy": correct_b / len(differences),
        "b_minus_a": bootstrap,
        "gate_role": "descriptive_not_blocking",
    }


def compare_r4_history(
    *,
    raw_predictions: Path,
    tagged_predictions: Path,
    last_effective_predictions: Path,
    suite: str,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 2026,
) -> dict[str, Any]:
    if suite not in {"bh1", "formal"}:
        raise ValueError("suite must be bh1 or formal")
    raw_rows = load_jsonl(raw_predictions)
    tagged_rows = load_jsonl(tagged_predictions)
    reduced_rows = load_jsonl(last_effective_predictions)
    validate_rows(raw_rows, method=QWEN_R4_RAW_HISTORY)
    validate_rows(tagged_rows, method=QWEN_R4_OPERATION_TAGGED_HISTORY)
    validate_rows(reduced_rows, method=QWEN_R4_LAST_EFFECTIVE_EVENT)
    raw_map = _paired_identity_map(raw_rows)
    tagged_map = _paired_identity_map(tagged_rows)
    reduced_map = _paired_identity_map(reduced_rows)
    all_match = set(raw_map) == set(tagged_map) == set(reduced_map)
    if not all_match:
        raise ValueError("The three R4 arms do not have identical standard prediction identities")
    for identity in sorted(raw_map, key=repr):
        _assert_semantic_match(identity, (raw_map[identity], tagged_map[identity], reduced_map[identity]))
    comparisons = {
        "tagged_minus_raw": _arm_comparison(
            raw_map,
            tagged_map,
            name="B-A: operation-tagged minus raw chronological",
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "last_effective_minus_tagged": _arm_comparison(
            tagged_map,
            reduced_map,
            name="C-B: last-effective minus operation-tagged",
            iterations=bootstrap_iterations,
            seed=bootstrap_seed + 1,
        ),
    }
    payload = {
        "schema": SCHEMA,
        "suite": suite,
        "passed": True,
        "identity_pairing": {
            "passed": True,
            "standard_records_per_arm": len(raw_map),
            "semantic_fields_exact": True,
        },
        "comparisons": comparisons,
        "gate_policy": {
            "bh1_differences_are_not_a_gate": suite == "bh1",
            "formal_outputs_comparison_only": suite == "formal",
        },
        "inputs": {
            "raw_predictions_sha256": sha256_file(raw_predictions),
            "tagged_predictions_sha256": sha256_file(tagged_predictions),
            "last_effective_predictions_sha256": sha256_file(last_effective_predictions),
        },
    }
    return {**payload, "report_sha256": canonical_object_sha256(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly paired R4 history-arm comparison")
    parser.add_argument("--raw-predictions", type=Path, required=True)
    parser.add_argument("--tagged-predictions", type=Path, required=True)
    parser.add_argument("--last-effective-predictions", type=Path, required=True)
    parser.add_argument("--suite", choices=("bh1", "formal"), required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing comparison report: {args.output}")
    report = compare_r4_history(
        raw_predictions=args.raw_predictions,
        tagged_predictions=args.tagged_predictions,
        last_effective_predictions=args.last_effective_predictions,
        suite=args.suite,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
