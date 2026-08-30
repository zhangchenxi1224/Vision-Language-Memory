"""Evaluate the preregistered R8 intermediate EMA checkpoints without selecting among them."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.train import dreamlite_r5_compose as r5  # noqa: E402
from scripts.train import dreamlite_r6_source_anchor as r6  # noqa: E402
from scripts.train import dreamlite_r7_gradient_balance as r8  # noqa: E402
from scripts.train.dreamlite_episode import (  # noqa: E402
    assert_frozen_contract,
    choice_reader_callable,
    set_all_seeds,
)
from vision_memory.repro import configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training.r5_compose import canonical_sha256  # noqa: E402


SCHEMA = "vision_memory.r8-checkpoint-trajectory.v1"
INVENTORY_SCHEMA = "vision_memory.r8-checkpoint-trajectory-inventory.v1"
ARMS = tuple(r8.R8_AGGREGATION_MODE)
STEPS = (0, 32, 64, 96, 128)
INTERMEDIATE_STEPS = (32, 64, 96)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"Expected JSON objects: {path}")
    return values


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_source_inventory(arm_root: Path) -> dict[str, Any]:
    inventory_path = arm_root / "artifact_inventory.json"
    inventory = _load_json(inventory_path)
    records = inventory.get("artifacts")
    if inventory.get("schema") != "vision_memory.r8-artifact-inventory.v1":
        raise ValueError("R8 source inventory schema drifted.")
    if not isinstance(records, list) or not records:
        raise ValueError("R8 source inventory is empty.")
    checked = 0
    total_bytes = 0
    for record in records:
        path = arm_root / str(record["path"])
        if not path.is_file():
            raise ValueError(f"R8 source artifact is missing: {path}")
        size = path.stat().st_size
        if size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise ValueError(f"R8 source artifact failed size/SHA validation: {path}")
        checked += 1
        total_bytes += size
    return {
        "inventory_path": str(inventory_path.resolve()),
        "inventory_sha256": _sha256(inventory_path),
        "records_checked": checked,
        "bytes_checked": total_bytes,
    }


def _checkpoint_path(arm_root: Path, step: int) -> Path:
    return arm_root / "run" / "checkpoints" / f"step-{step:06d}.pt"


def _checkpoint_payload(path: Path, *, expected_manifest: Mapping[str, Any], expected_step: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != 1 or int(payload.get("optimizer_step", -1)) != expected_step:
        raise ValueError(f"R8 checkpoint schema/step mismatch: {path}")
    if payload.get("manifest") != dict(expected_manifest):
        raise ValueError(f"R8 checkpoint manifest drift: {path}")
    trainer_state = payload.get("trainer_state")
    if not isinstance(trainer_state, Mapping) or trainer_state.get("schema") != r5.R5_TRAINER_STATE_SCHEMA:
        raise ValueError(f"R8 checkpoint trainer state is invalid: {path}")
    if int(trainer_state.get("next_optimizer_step", -1)) != expected_step:
        raise ValueError(f"R8 checkpoint trainer cursor drift: {path}")
    ema_state = trainer_state.get("ema_state")
    if not isinstance(ema_state, Mapping) or not ema_state:
        raise ValueError(f"R8 checkpoint EMA state is missing: {path}")
    return payload


@torch.no_grad()
def _install_ema(runtime: r5.RuntimeBundle, payload: Mapping[str, Any]) -> None:
    state = payload["trainer_state"]["ema_state"]
    parameters = dict(runtime.named_trainable)
    if set(state) != set(parameters):
        raise ValueError("R8 EMA parameter names differ from the runtime trainable topology.")
    for name, parameter in parameters.items():
        source = state[name]
        if not isinstance(source, torch.Tensor) or not bool(torch.isfinite(source).all()):
            raise ValueError(f"R8 EMA tensor is invalid: {name}")
        parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))


def _endpoint_binding(
    *,
    endpoint_path: Path,
    step128_payload: Mapping[str, Any],
) -> dict[str, Any]:
    endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
    endpoint_state = endpoint.get("trainable_state")
    ema_state = step128_payload["trainer_state"]["ema_state"]
    if not isinstance(endpoint_state, Mapping) or set(endpoint_state) != set(ema_state):
        raise ValueError("R8 endpoint EMA topology differs from step128 trainer EMA state.")
    mismatched = [name for name in endpoint_state if not torch.equal(endpoint_state[name], ema_state[name])]
    if mismatched:
        raise ValueError(f"R8 endpoint EMA is not the exact step128 EMA state: {mismatched[:5]}")
    return {
        "passed": True,
        "endpoint_path": str(endpoint_path.resolve()),
        "endpoint_sha256": _sha256(endpoint_path),
        "matched_tensors": len(endpoint_state),
    }


def _accuracy(rows: Sequence[Mapping[str, Any]], *, checkpoint: str) -> float:
    values = [
        float(bool(row["correct"]))
        for row in rows
        if row.get("checkpoint") == checkpoint
        and row.get("suite") == "train_overfit_hard8"
        and row.get("condition") == "normal"
    ]
    if not values:
        raise ValueError(f"R8 trajectory has no normal accuracy rows for {checkpoint}.")
    return float(np.mean(values))


def _trajectory_statistics(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    labels = ["m0", "ema_step32", "ema_step64", "ema_step96", "ema_step128"]
    comparisons: dict[str, Any] = {}
    state_did: dict[str, Any] = {}
    accuracies: dict[str, float] = {}
    for index, label in enumerate(labels):
        accuracies[label] = _accuracy(rows, checkpoint=label)
        if label == "m0":
            continue
        comparisons[label] = r6._checkpoint_comparison(
            rows,
            suite="train_overfit_hard8",
            endpoint=label,
            iterations=bootstrap_iterations,
            seed=20260860 + index,
        )
        state_did[label] = r6._difference_in_differences(
            rows,
            suite="train_overfit_hard8",
            endpoint=label,
            iterations=bootstrap_iterations,
            seed=20260870 + index,
        )
    return {
        "checkpoint_order": labels,
        "normal_accuracy": accuracies,
        "normal_ce_vs_m0": comparisons,
        "normal_reset_difference_in_differences_vs_m0": state_did,
        "descriptive_only_not_checkpoint_selection": True,
        "primary_endpoint_remains": "ema_step128",
    }


def _existing_endpoint_rows(arm_root: Path) -> list[dict[str, Any]]:
    rows = _load_jsonl(arm_root / "run" / "overfit_evaluation_rows.jsonl")
    labels = {str(row.get("checkpoint")) for row in rows}
    if labels != {"m0", "ema_step128"}:
        raise ValueError(f"R8 source overfit labels drifted: {labels}")
    expected = 2 * 8 * 4 * 4
    if len(rows) != expected:
        raise ValueError(f"R8 source overfit row count is {len(rows)}, expected {expected}.")
    return rows


def _runtime_args(cli: argparse.Namespace) -> argparse.Namespace:
    return r8.parse_args(
        [
            "--protocol-revision",
            "r8",
            "--arm",
            cli.arm,
            "--train",
            str(cli.train),
            "--dev",
            str(cli.dev),
            "--dreamlite",
            str(cli.dreamlite),
            "--reader",
            str(cli.reader),
            "--output-dir",
            str(cli.output_dir),
            "--seed",
            str(cli.seed),
            "--dreamlite-device",
            cli.dreamlite_device,
            "--reader-device",
            cli.reader_device,
            "--strict-determinism",
        ]
    )


def run(cli: argparse.Namespace) -> dict[str, Any]:
    if cli.output_dir.exists() and any(cli.output_dir.iterdir()):
        raise ValueError("R8 checkpoint trajectory refuses a non-empty output directory.")
    terminal = _load_json(cli.arm_root / "terminal.json")
    if terminal.get("status") != "completed_diagnostic" or terminal.get("passed") is not True:
        raise ValueError("R8 source arm does not have a passed terminal record.")
    manifest = _load_json(cli.arm_root / "run" / "manifest.json")
    if manifest.get("schema") != r8.R8_MANIFEST_SCHEMA or manifest.get("arm") != cli.arm:
        raise ValueError("R8 source manifest schema/arm drifted.")
    if manifest.get("git_commit") != cli.expected_training_commit:
        raise ValueError("R8 source manifest training commit drifted.")
    if manifest.get("selected_segments_sha256") != r8.R8_SELECTED_SEGMENTS_SHA256:
        raise ValueError("R8 source selected hard8 drifted.")
    source_validation = _validate_source_inventory(cli.arm_root)

    checkpoint_paths = {step: _checkpoint_path(cli.arm_root, step) for step in STEPS}
    if any(not path.is_file() for path in checkpoint_paths.values()):
        missing = [str(path) for path in checkpoint_paths.values() if not path.is_file()]
        raise ValueError(f"R8 trajectory checkpoints are missing: {missing}")
    payloads = {
        step: _checkpoint_payload(path, expected_manifest=manifest, expected_step=step)
        for step, path in checkpoint_paths.items()
    }
    endpoint_binding = _endpoint_binding(
        endpoint_path=cli.arm_root / "run" / "endpoint_ema.pt",
        step128_payload=payloads[128],
    )

    configure_strict_cuda_determinism(cli.seed)
    set_all_seeds(cli.seed)
    args = _runtime_args(cli)
    data, selected = r6._load_data(args)
    selected_sha = canonical_sha256([segment.to_dict() for segment in selected])
    if selected_sha != r8.R8_SELECTED_SEGMENTS_SHA256:
        raise RuntimeError("R8 trajectory runtime selected a different hard8.")
    runtime = r8._load_runtime(args)
    eval_reader = choice_reader_callable(
        reader=runtime.reader,
        processor=runtime.processor,
        reader_device=runtime.reader_device,
        require_grad=False,
        deterministic_ce=True,
    )
    assert_frozen_contract(runtime.pipe, runtime.reader)

    rows = _existing_endpoint_rows(cli.arm_root)
    started = time.monotonic()
    for step in INTERMEDIATE_STEPS:
        _install_ema(runtime, payloads[step])
        label = f"ema_step{step}"
        current = r6._evaluation_rows(
            model=runtime.model,
            reader_fn=eval_reader,
            segments=selected,
            checkpoint=label,
        )
        if len(current) != 8 * 4 * 4:
            raise RuntimeError(f"R8 trajectory row count drifted at {label}: {len(current)}")
        rows.extend(current)
        print(
            json.dumps(
                {
                    "milestone": "r8_checkpoint_trajectory",
                    "arm": cli.arm,
                    "checkpoint": label,
                    "rows": len(current),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    assert_frozen_contract(runtime.pipe, runtime.reader)
    expected_rows = len(STEPS) * 8 * 4 * 4
    if len(rows) != expected_rows:
        raise RuntimeError(f"R8 trajectory total row count is {len(rows)}, expected {expected_rows}.")
    rows.sort(key=lambda row: (
        STEPS.index(0 if row["checkpoint"] == "m0" else int(str(row["checkpoint"]).removeprefix("ema_step"))),
        str(row["pair_unit"]),
        str(row["condition"]),
        int(row["view_index"]),
    ))
    cli.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(cli.output_dir / "hard8_checkpoint_rows.jsonl", rows)
    evaluation_summary = r5.summarize_evaluation_rows(
        rows,
        bootstrap_iterations=cli.bootstrap_iterations,
        bootstrap_seed=20260859,
    )
    _write_json(cli.output_dir / "hard8_checkpoint_evaluation.json", evaluation_summary)
    trajectory = _trajectory_statistics(rows, bootstrap_iterations=cli.bootstrap_iterations)
    result = {
        "schema": SCHEMA,
        "status": "completed",
        "formal_success_claim": False,
        "arm": cli.arm,
        "gradient_aggregation": r8.R8_AGGREGATION_MODE[cli.arm],
        "training_commit": cli.expected_training_commit,
        "analysis_commit": r5.git_value("rev-parse", "HEAD"),
        "selected_segments_sha256": selected_sha,
        "source_arm_root": str(cli.arm_root.resolve()),
        "source_validation": source_validation,
        "checkpoint_sha256": {f"step{step}": _sha256(path) for step, path in checkpoint_paths.items()},
        "endpoint_binding": endpoint_binding,
        "trajectory": trajectory,
        "rows": len(rows),
        "elapsed_seconds": time.monotonic() - started,
        "updater_peak_memory_gib": r5._peak_gib(runtime.updater_device),
        "reader_peak_memory_gib": r5._peak_gib(runtime.reader_device),
        "interpretation_scope": (
            "Intermediate EMA checkpoints are descriptive trajectory points only. "
            "They cannot replace or select against the preregistered EMA step128 endpoint."
        ),
    }
    _write_json(cli.output_dir / "trajectory_summary.json", result)
    artifacts = []
    for path in sorted(value for value in cli.output_dir.rglob("*") if value.is_file()):
        if path.name == "artifact_inventory.json":
            continue
        artifacts.append(
            {
                "path": path.relative_to(cli.output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        cli.output_dir / "artifact_inventory.json",
        {"schema": INVENTORY_SCHEMA, "artifacts": artifacts},
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--arm-root", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-training-commit", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dreamlite-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(build_parser().parse_args(argv))
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": result["arm"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
