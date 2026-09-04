"""Evaluate one completed R10 DreamLite raw endpoint without changing its EMA decision."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.experiments import compare_r10_visual_alignment as comparison  # noqa: E402
from scripts.inspire import run_r10_alignment_target as controller  # noqa: E402
from scripts.train import dreamlite_r10_single_set as trainer  # noqa: E402
from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r7_gradient_balance as r8  # noqa: E402
from vision_memory.training.r10_alignment import target_gate, target_statistics  # noqa: E402


RAW_LABEL = "raw_step128_attribution"
SUMMARY_SCHEMA = "vision_memory.r10-raw-endpoint-attribution-summary.v1"
MANIFEST_SCHEMA = "vision_memory.r10-raw-endpoint-attribution-manifest.v1"
TERMINAL_SCHEMA = "vision_memory.r10-raw-endpoint-attribution-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r10-raw-endpoint-attribution-inventory.v1"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            values.append(value)
    return values


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    os.replace(temporary, path)


def _inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        if path.name == "artifact_inventory.json":
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": comparison._sha256(path),
            }
        )
    return records


def _validate_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("R10 raw attribution refuses a non-empty output directory.")
    head = _git("rev-parse", "HEAD")
    if head != args.expected_analysis_commit:
        raise ValueError(f"R10 raw attribution commit mismatch: expected {args.expected_analysis_commit}, got {head}")
    if _git("status", "--porcelain"):
        raise ValueError("R10 raw attribution requires a clean analysis worktree.")
    expected_root = args.r10_run_root / "dreamlite-single-set" / f"target-{args.target_index:02d}"
    source = comparison._validate_target(expected_root, "dreamlite-single-set", args.target_index)
    observed_data = {
        "train": comparison._sha256(args.train),
        "dev": comparison._sha256(args.dev),
    }
    if observed_data != controller.EXPECTED_DATA_SHA:
        raise ValueError(f"R10 raw attribution fixed data SHA mismatch: {observed_data}")
    environment_drift = {
        name: {"expected": expected, "observed": os.environ.get(name)}
        for name, expected in controller.EXPECTED_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    }
    if environment_drift:
        raise ValueError(f"R10 raw attribution strict environment drift: {environment_drift}")
    if not args.dreamlite.is_dir() or not args.reader.is_dir():
        raise ValueError("R10 raw attribution model snapshot path is missing.")
    return {
        "analysis_git_commit": head,
        "source": source,
        "source_root": expected_root,
        "data_sha256": observed_data,
    }


def _runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    values = trainer.parse_args(
        (
            "--target-index",
            str(args.target_index),
            "--train",
            str(args.train),
            "--dev",
            str(args.dev),
            "--dreamlite",
            str(args.dreamlite),
            "--reader",
            str(args.reader),
            "--output-dir",
            str(args.output_dir),
            "--seed",
            "0",
            "--dreamlite-device",
            args.dreamlite_device,
            "--reader-device",
            args.reader_device,
            "--strict-determinism",
        )
    )
    return values


def _save_state_image(*, runtime: r5.RuntimeBundle, target: r5.R5Segment, path: Path) -> None:
    from torchvision.transforms.functional import to_pil_image

    item = r5._segment_eval_items(runtime.model, (target,))[0]
    state = item.normal_state.to(runtime.updater_device)
    with torch.no_grad():
        image = runtime.model.reader_image(state).detach().float().clamp(0.0, 1.0)[0].cpu()
    to_pil_image(image).save(path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    validated = _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    r5._write_environment(args.output_dir / "environment.txt")
    comparison._write_json(args.output_dir / "runtime.json", r5._runtime_versions())
    runtime_args = _runtime_args(args)
    determinism = r8.configure_strict_cuda_determinism(runtime_args.seed)
    r8.set_all_seeds(runtime_args.seed)
    data, selected = trainer._load_data(runtime_args)
    target = selected[args.target_index]
    runtime = trainer._load_runtime(runtime_args)
    eval_reader = r8.choice_reader_callable(
        reader=runtime.reader,
        processor=runtime.processor,
        reader_device=runtime.reader_device,
        require_grad=False,
        deterministic_ce=True,
    )
    source_root = Path(validated["source_root"])
    raw_path = source_root / "run" / "endpoint_raw.pt"
    r8.load_trainable_weights(raw_path, trainable_module=runtime.model)
    raw_rows = trainer._evaluation_rows(
        model=runtime.model,
        reader_fn=eval_reader,
        target=target,
        checkpoint=RAW_LABEL,
    )
    source_rows_path = source_root / "run" / "target_evaluation_rows.jsonl"
    source_rows = _load_jsonl(source_rows_path)
    m0_rows = [row for row in source_rows if row.get("checkpoint") == "m0"]
    if len(m0_rows) != 8 or len(raw_rows) != 8:
        raise ValueError("R10 raw attribution requires exact eight-row M0 and raw endpoint cells.")
    combined = m0_rows + raw_rows
    statistics = target_statistics(
        combined,
        suite=trainer.SUITE,
        target_segment_id=target.segment_id,
        endpoint=RAW_LABEL,
    )
    descriptive_gate = target_gate(statistics, technical_gate=True)
    r8.assert_frozen_contract(runtime.pipe, runtime.reader)
    rows_path = args.output_dir / "raw_endpoint_evaluation_rows.jsonl"
    _write_jsonl(rows_path, combined)
    image_path = args.output_dir / "raw_endpoint_state.png"
    _save_state_image(runtime=runtime, target=target, path=image_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "completed",
        "analysis_git_commit": validated["analysis_git_commit"],
        "source_training_git_commit": validated["source"]["git_commit"],
        "source_root": str(source_root.resolve()),
        "source_inventory_sha256": validated["source"]["inventory_sha256"],
        "source_summary_sha256": validated["source"]["summary_sha256"],
        "source_terminal_sha256": validated["source"]["terminal_sha256"],
        "source_raw_endpoint_sha256": comparison._sha256(raw_path),
        "source_evaluation_rows_sha256": comparison._sha256(source_rows_path),
        "target_index": args.target_index,
        "target_segment_id": target.segment_id,
        "selected_segments_sha256": comparison.R10_SELECTED_SEGMENTS_SHA256,
        "data_sha256": validated["data_sha256"],
        "environment": dict(controller.EXPECTED_ENVIRONMENT),
        "determinism": determinism,
        "host": platform.node(),
        "python": sys.executable,
        "formal_success_claim": False,
        "cannot_replace_preregistered_ema_endpoint": True,
    }
    comparison._write_json(args.output_dir / "manifest.json", manifest)
    source_ema = validated["source"]["target_statistics"]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_attribution",
        "formal_success_claim": False,
        "cannot_replace_preregistered_ema_endpoint": True,
        "target_index": args.target_index,
        "target_segment_id": target.segment_id,
        "technical_gate": True,
        "raw_descriptive_gate": descriptive_gate,
        "raw_statistics": statistics,
        "existing_ema_gate": validated["source"]["passed"],
        "existing_ema_statistics": source_ema,
        "raw_minus_ema_normal_ce": (
            float(statistics["endpoint_normal_mean_ce"])
            - float(source_ema["endpoint_normal_mean_ce"])
        ),
        "artifacts": {
            "manifest_sha256": comparison._sha256(args.output_dir / "manifest.json"),
            "rows_sha256": comparison._sha256(rows_path),
            "state_image_sha256": comparison._sha256(image_path),
        },
    }
    summary_path = args.output_dir / "summary.json"
    comparison._write_json(summary_path, summary)
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": "completed_attribution",
        "passed": True,
        "formal_success_claim": False,
        "target_index": args.target_index,
        "target_segment_id": target.segment_id,
        "summary_sha256": comparison._sha256(summary_path),
        "manifest_sha256": comparison._sha256(args.output_dir / "manifest.json"),
    }
    comparison._write_json(args.output_dir / "terminal.json", terminal)
    comparison._write_json(
        args.output_dir / "artifact_inventory.json",
        {
            "schema": INVENTORY_SCHEMA,
            "root": str(args.output_dir.resolve()),
            "artifacts": _inventory(args.output_dir),
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-index", type=int, choices=range(8), required=True)
    parser.add_argument("--r10-run-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-analysis-commit", required=True)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = evaluate(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "milestone": "r10_raw_endpoint_attribution_completed",
                "target_index": summary["target_index"],
                "raw_descriptive_gate": summary["raw_descriptive_gate"],
                "existing_ema_gate": summary["existing_ema_gate"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
