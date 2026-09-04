"""Fail-closed aggregation for the eight preregistered R11 VAE-latent targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vision_memory.training.r10_alignment import (  # noqa: E402
    R10_SELECTED_SEGMENTS_SHA256,
    R10_TARGET_IDS,
    target_gate,
    target_statistics,
)
from vision_memory.training.r11_latent import (  # noqa: E402
    R11_CHECKPOINT_STEPS,
    R11_LEARNING_RATE,
    R11_OPTIMIZER_STEPS,
    R11_PROTOCOL,
    training_view_counts,
)


SUMMARY_SCHEMA = "vision_memory.r11-vae-latent-oracle-summary.v1"
TERMINAL_SCHEMA = "vision_memory.r11-latent-target-terminal.v1"
INVENTORY_SCHEMA = "vision_memory.r11-latent-target-inventory.v1"
LAUNCH_SCHEMA = "vision_memory.r11-latent-target-launch.v1"
COMPARISON_SCHEMA = "vision_memory.r11-vae-latent-reachability-comparison.v1"
DELIVERY_SCHEMA = "vision_memory.r11-vae-latent-reachability-delivery.v1"
EXPECTED_GIT_COMMIT = "f4a018f3c4eef453fff3367b049ea732332e8c37"
EXPECTED_IMPLEMENTATION_REVISION = "direct-vae-latent-fp32-v1"
EXPECTED_DATA_SHA = {
    "train": "24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184",
    "dev": "8b167df38022a631d4e631d3c0d66e9fca74171f4224fec436030d6650047303",
}
SUITE = "r11_f1_vae_latent"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"R11 JSONL row is not an object: {path}:{line_number}")
            values.append(value)
    return values


def _sha256(path: Path) -> str:
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


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _required_artifacts() -> set[str]:
    checkpoint_files = {
        f"run/checkpoints/step-{step:03d}.pt" for step in R11_CHECKPOINT_STEPS
    }
    image_files = {f"run/images/step-{step:03d}.png" for step in R11_CHECKPOINT_STEPS}
    return {
        "launch.json",
        "stdout.log",
        "stderr.log",
        "terminal.json",
        "run/manifest.json",
        "run/environment.txt",
        "run/runtime.json",
        "run/family_pool_audit.json",
        "run/dev_split_audit.json",
        "run/metrics.jsonl",
        "run/target_evaluation_rows.jsonl",
        "run/endpoint_raw.pt",
        "run/endpoint_raw.png",
        "run/model_snapshot_verification_end.json",
        "run/r11_latent_summary.json",
    } | checkpoint_files | image_files


def _validate_inventory(root: Path) -> str:
    inventory_path = root / "artifact_inventory.json"
    inventory = _load(inventory_path)
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise ValueError(f"R11 inventory schema mismatch: {root}")
    records = inventory.get("artifacts")
    if not isinstance(records, list) or not records:
        raise ValueError(f"R11 artifact inventory is empty: {root}")
    names: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"R11 inventory record is malformed: {root}")
        relative = str(record.get("path"))
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"R11 inventory path escapes its root: {relative}")
        if relative in names:
            raise ValueError(f"R11 inventory contains a duplicate path: {relative}")
        names.add(relative)
        artifact = root / relative_path
        if not artifact.is_file():
            raise ValueError(f"R11 inventory artifact is missing: {artifact}")
        if (
            artifact.stat().st_size != int(record.get("bytes", -1))
            or _sha256(artifact) != record.get("sha256")
        ):
            raise ValueError(f"R11 inventory artifact failed size/SHA validation: {artifact}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    if names != actual:
        raise ValueError(
            f"R11 inventory/file-set mismatch: missing={sorted(actual - names)}, "
            f"stale={sorted(names - actual)}"
        )
    missing = sorted(_required_artifacts() - names)
    if missing:
        raise ValueError(f"R11 inventory lacks required artifacts for {root}: {missing}")
    return _sha256(inventory_path)


def _validate_summary_artifacts(root: Path, summary: Mapping[str, Any]) -> None:
    bindings = summary.get("artifacts")
    if not isinstance(bindings, Mapping):
        raise ValueError(f"R11 summary artifact bindings are missing: {root}")
    expected = {
        "manifest_sha256": "manifest.json",
        "metrics_sha256": "metrics.jsonl",
        "evaluation_rows_sha256": "target_evaluation_rows.jsonl",
        "endpoint_raw_sha256": "endpoint_raw.pt",
        "endpoint_png_sha256": "endpoint_raw.png",
        "snapshot_end_sha256": "model_snapshot_verification_end.json",
    }
    for key, relative in expected.items():
        if bindings.get(key) != _sha256(root / "run" / relative):
            raise ValueError(f"R11 summary artifact hash binding failed: {root}:{key}")


def _validate_metrics(root: Path, segment_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_jsonl(root / "run" / "metrics.jsonl")
    if (
        len(rows) != R11_OPTIMIZER_STEPS
        or [row.get("optimizer_step") for row in rows]
        != list(range(1, R11_OPTIMIZER_STEPS + 1))
    ):
        raise ValueError(f"R11 target lacks the exact 256-step metric sequence: {root}")
    numeric = (
        "loss_before_step",
        "gradient_norm",
        "gradient_nonzero_fraction",
        "latent_min_after_step",
        "latent_max_after_step",
        "latent_rms_after_step",
        "latent_delta_norm_after_step",
        "image_min_after_step",
        "image_max_after_step",
        "image_rms_after_step",
        "image_saturation_fraction_after_step",
        "learning_rate",
    )
    for row in rows:
        if row.get("kind") != "optimizer_step" or row.get("target_segment_id") != segment_id:
            raise ValueError(f"R11 optimizer metric identity drift: {root}")
        if any(not math.isfinite(float(row.get(key, math.nan))) for key in numeric):
            raise ValueError(f"R11 optimizer metric is non-finite: {root}")
        if not math.isclose(float(row["learning_rate"]), R11_LEARNING_RATE):
            raise ValueError(f"R11 optimizer learning-rate drift: {root}")
        if float(row["gradient_norm"]) <= 0.0 or float(row["gradient_nonzero_fraction"]) <= 0.0:
            raise ValueError(f"R11 optimizer gradient receipt failed: {root}")
    views = training_view_counts(rows)
    if views != {0: 64, 1: 64, 2: 64, 3: 64}:
        raise ValueError(f"R11 fixed training-view counts drifted: {root}:{views}")
    diagnostic = {
        "first_loss": float(rows[0]["loss_before_step"]),
        "final_loss": float(rows[-1]["loss_before_step"]),
        "minimum_loss": min(float(row["loss_before_step"]) for row in rows),
        "minimum_gradient_norm": min(float(row["gradient_norm"]) for row in rows),
        "final_latent_delta_norm": float(rows[-1]["latent_delta_norm_after_step"]),
        "final_image_saturation_fraction": float(
            rows[-1]["image_saturation_fraction_after_step"]
        ),
        "training_view_counts": views,
    }
    return rows, diagnostic


def _validate_evaluation(root: Path, segment_id: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = _load_jsonl(root / "run" / "target_evaluation_rows.jsonl")
    expected_counts = {
        (checkpoint, condition): 4
        for checkpoint in ("m0", "raw_latent_step256")
        for condition in ("normal", "reset")
    }
    observed = {
        key: sum(
            row.get("checkpoint") == key[0] and row.get("condition") == key[1]
            for row in rows
        )
        for key in expected_counts
    }
    if len(rows) != 16 or observed != expected_counts:
        raise ValueError(f"R11 fixed evaluation receipt drifted: {root}:{observed}")
    recomputed = target_statistics(
        rows,
        suite=SUITE,
        target_segment_id=segment_id,
        endpoint="raw_latent_step256",
    )
    # The in-memory statistic uses integer view keys; JSON necessarily stores object
    # keys as strings. Compare the canonical JSON-domain value so this check remains
    # strict without treating that lossless serialization boundary as scientific drift.
    serialized = json.loads(json.dumps(recomputed, sort_keys=True))
    if serialized != summary.get("target_statistics"):
        raise ValueError(f"R11 stored/recomputed target statistics differ: {root}")
    return serialized


def _validate_target(root: Path, index: int) -> dict[str, Any]:
    segment_id = R10_TARGET_IDS[index]
    inventory_sha = _validate_inventory(root)
    launch_path = root / "launch.json"
    terminal_path = root / "terminal.json"
    summary_path = root / "run" / "r11_latent_summary.json"
    manifest_path = root / "run" / "manifest.json"
    launch = _load(launch_path)
    terminal = _load(terminal_path)
    summary = _load(summary_path)
    manifest = _load(manifest_path)
    if (
        launch.get("schema") != LAUNCH_SCHEMA
        or launch.get("git_commit") != EXPECTED_GIT_COMMIT
        or launch.get("git_dirty") is not False
        or launch.get("data_sha256") != EXPECTED_DATA_SHA
        or launch.get("target_index") != index
        or launch.get("target_segment_id") != segment_id
    ):
        raise ValueError(f"R11 launch identity/immutability drift: target {index}")
    checks = terminal.get("checks")
    if (
        terminal.get("schema") != TERMINAL_SCHEMA
        or terminal.get("status") != "completed_diagnostic"
        or terminal.get("passed") is not True
        or terminal.get("child_exit_code") != 0
        or terminal.get("scientific_success_claim") is not False
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError(f"R11 target did not complete its technical contract: target {index}")
    if terminal.get("summary_sha256") != _sha256(summary_path):
        raise ValueError(f"R11 terminal/summary hash binding failed: target {index}")
    if terminal.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError(f"R11 terminal/manifest hash binding failed: target {index}")
    if (
        summary.get("schema") != SUMMARY_SCHEMA
        or summary.get("status") != "completed"
        or summary.get("protocol") != R11_PROTOCOL
        or summary.get("implementation_revision") != EXPECTED_IMPLEMENTATION_REVISION
        or summary.get("git_commit") != EXPECTED_GIT_COMMIT
        or summary.get("target_index") != index
        or summary.get("target_segment_id") != segment_id
        or summary.get("target_family") != "F1"
        or summary.get("selected_segments") != list(R10_TARGET_IDS)
        or summary.get("selected_segments_sha256") != R10_SELECTED_SEGMENTS_SHA256
        or summary.get("optimizer_steps") != R11_OPTIMIZER_STEPS
        or summary.get("checkpoint_steps_observed") != list(R11_CHECKPOINT_STEPS)
    ):
        raise ValueError(f"R11 summary fixed contract drift: target {index}")
    if (
        manifest.get("git_commit") != EXPECTED_GIT_COMMIT
        or manifest.get("git_dirty") is not False
        or manifest.get("target_index") != index
        or manifest.get("target_segment_id") != segment_id
        or manifest.get("selected_segment_ids") != list(R10_TARGET_IDS)
        or manifest.get("selected_segments_sha256") != R10_SELECTED_SEGMENTS_SHA256
        or manifest.get("train_sha256") != EXPECTED_DATA_SHA["train"]
        or manifest.get("dev_sha256") != EXPECTED_DATA_SHA["dev"]
    ):
        raise ValueError(f"R11 manifest fixed contract drift: target {index}")
    fixed = manifest.get("fixed_contract")
    if (
        not isinstance(fixed, Mapping)
        or fixed.get("vae_frozen") is not True
        or fixed.get("dreamlite_unet_executed") is not False
        or fixed.get("semantic_prompt_used") is not False
        or fixed.get("optimizer") != "Adam"
        or not math.isclose(float(fixed.get("learning_rate", math.nan)), R11_LEARNING_RATE)
        or not math.isclose(float(fixed.get("weight_decay", math.nan)), 0.0)
        or fixed.get("optimizer_steps") != R11_OPTIMIZER_STEPS
        or fixed.get("checkpoint_steps") != list(R11_CHECKPOINT_STEPS)
        or fixed.get("primary_endpoint") != "raw_latent_step256"
        or fixed.get("best_checkpoint_selection_forbidden") is not True
    ):
        raise ValueError(f"R11 manifest oracle contract drift: target {index}")
    gates = summary.get("gates")
    technical = summary.get("technical_gate")
    if (
        not isinstance(gates, Mapping)
        or not isinstance(technical, Mapping)
        or technical.get("passed") is not True
        or technical.get("optimizer_step_records") != R11_OPTIMIZER_STEPS
        or technical.get("training_view_counts") != {"0": 64, "1": 64, "2": 64, "3": 64}
        or technical.get("checkpoint_steps_observed") != list(R11_CHECKPOINT_STEPS)
        or technical.get("png_steps_observed") != list(R11_CHECKPOINT_STEPS)
        or technical.get("trainable_parameter_names") != ["latent_fp32"]
        or technical.get("vae_frozen") is not True
        or technical.get("reader_frozen") is not True
        or technical.get("snapshots_unchanged") is not True
        or gates.get("technical_gate") is not True
        or gates.get("formal_success_gate") is not False
        or summary.get("diagnostic_only_not_formal_success") is not True
        or summary.get("full_success_claim_allowed") is not False
    ):
        raise ValueError(f"R11 technical/fail-closed gate drift: target {index}")
    _validate_summary_artifacts(root, summary)
    _rows, training = _validate_metrics(root, segment_id)
    statistics = _validate_evaluation(root, segment_id, summary)
    passed = target_gate(statistics, technical_gate=True)
    if (
        gates.get("target_latent_reachability_gate") is not passed
        or terminal.get("target_latent_reachability_gate") is not passed
    ):
        raise ValueError(f"R11 stored/recomputed target gate differs: target {index}")
    return {
        "target_index": index,
        "target_family": "F1",
        "target_segment_id": segment_id,
        "passed": passed,
        "target_statistics": statistics,
        "training_diagnostics": training,
        "wall_clock_seconds": summary.get("wall_clock_seconds"),
        "source_root": str(root.resolve()),
        "launch_sha256": _sha256(launch_path),
        "summary_sha256": _sha256(summary_path),
        "terminal_sha256": _sha256(terminal_path),
        "inventory_sha256": inventory_sha,
        "git_commit": summary.get("git_commit"),
        "implementation_revision": summary.get("implementation_revision"),
    }


def _decision(pass_count: int) -> tuple[str, str]:
    if pass_count == 8:
        return (
            "replace_semantic_editor_with_shared_event_to_latent_writer",
            "All eight independently optimized VAE latents pass the fixed causal Reader gate. "
            "The frozen VAE representation can carry the code; the next bottleneck test is a shared "
            "event-to-latent writer with held-out F1 targets, not further semantic-prompt or EMA tuning.",
        )
    if pass_count > 0:
        return (
            "attribute_target_dependent_vae_readability",
            "VAE-space readability is target-dependent. Attribute fixed target/query/token properties "
            "before designing or claiming a shared writer.",
        )
    return (
        "augment_or_bypass_vae_with_image_residual_codec",
        "No target demonstrated a readable code through the frozen VAE. Test a direct image-space "
        "residual codec instead of another DreamLite U-Net tuning round.",
    )


def _write_csv(path: Path, targets: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "target_index",
                "family",
                "segment_id",
                "gate",
                "m0_normal_ce",
                "endpoint_normal_ce",
                "relative_change",
                "improved_choice_views",
                "accuracy_delta",
                "normal_reset_did",
                "first_train_loss",
                "final_train_loss",
                "minimum_train_loss",
                "final_latent_delta_norm",
                "final_image_saturation_fraction",
                "wall_clock_seconds",
            )
        )
        for target in targets:
            stat = target["target_statistics"]
            train = target["training_diagnostics"]
            writer.writerow(
                (
                    target["target_index"],
                    target["target_family"],
                    target["target_segment_id"],
                    target["passed"],
                    stat["m0_normal_mean_ce"],
                    stat["endpoint_normal_mean_ce"],
                    stat["relative_change"],
                    stat["improved_choice_views"],
                    stat["accuracy_delta"],
                    stat["normal_reset_difference_in_differences"],
                    train["first_loss"],
                    train["final_loss"],
                    train["minimum_loss"],
                    train["final_latent_delta_norm"],
                    train["final_image_saturation_fraction"],
                    target["wall_clock_seconds"],
                )
            )


def _report(result: Mapping[str, Any]) -> str:
    lines = [
        "# R11 VAE-latent reachability result",
        "",
        "R11 is a per-target representation oracle. It is diagnostic and is not a shared memory-writer success claim.",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Target passes: {result['target_pass_count']}/8.",
        f"- Reason: {result['reason']}",
        f"- Source training commit: `{result['source_training_git_commit']}`",
        f"- Aggregation commit: `{result['aggregation_git_commit']}`",
        f"- Fixed F1 payload SHA-256: `{R10_SELECTED_SEGMENTS_SHA256}`",
        "",
        "| Target | Gate | M0 CE | Endpoint CE | Relative change | Improved views | Accuracy delta | Normal/reset DiD |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target in result["targets"]:
        stat = target["target_statistics"]
        lines.append(
            f"| {target['target_index']} | {'PASS' if target['passed'] else 'FAIL'} | "
            f"{float(stat['m0_normal_mean_ce']):.6f} | "
            f"{float(stat['endpoint_normal_mean_ce']):.6f} | "
            f"{float(stat['relative_change']):+.4%} | {stat['improved_choice_views']}/4 | "
            f"{float(stat['accuracy_delta']):+.3f} | "
            f"{float(stat['normal_reset_difference_in_differences']):+.6f} |"
        )
    lines.extend(
        (
            "",
            "Even 8/8 establishes only that independently optimized VAE latents can carry target-specific codes. "
            "Formal Picture Memory success still requires one shared event-conditioned writer, held-out ID/OOD "
            "targets, recurrence, multiple seeds, and causal state controls.",
            "",
        )
    )
    return "\n".join(lines)


def _refresh_delivery(output_dir: Path, result: Mapping[str, Any]) -> None:
    artifacts = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(value for value in output_dir.iterdir() if value.is_file())
        if path.name != "DELIVERY_MANIFEST.json"
    ]
    _write_json(
        output_dir / "DELIVERY_MANIFEST.json",
        {
            "schema": DELIVERY_SCHEMA,
            "artifacts": artifacts,
            "source_inventory_sha256": {
                str(target["target_index"]): target["inventory_sha256"]
                for target in result["targets"]
            },
        },
    )


def compare(run_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("R11 comparison refuses a non-empty output directory.")
    dirty = _git("status", "--porcelain")
    if dirty:
        raise ValueError("R11 comparison requires a clean aggregation worktree.")
    targets = [_validate_target(run_root / f"target-{index:02d}", index) for index in range(8)]
    if {target["git_commit"] for target in targets} != {EXPECTED_GIT_COMMIT}:
        raise ValueError("R11 immutable source commit drifted across targets.")
    pass_count = sum(bool(target["passed"]) for target in targets)
    decision, reason = _decision(pass_count)
    result = {
        "schema": COMPARISON_SCHEMA,
        "status": "completed",
        "formal_success_claim": False,
        "formal_success_reason": (
            "R11 independently optimizes one answer-supervised latent per fixed target. It does not train or "
            "evaluate a shared event-to-memory writer on held-out ID/OOD data."
        ),
        "decision": decision,
        "reason": reason,
        "target_pass_count": pass_count,
        "passed_target_indices": [target["target_index"] for target in targets if target["passed"]],
        "failed_target_indices": [target["target_index"] for target in targets if not target["passed"]],
        "source_training_git_commit": EXPECTED_GIT_COMMIT,
        "aggregation_git_commit": _git("rev-parse", "HEAD"),
        "selected_segments_sha256": R10_SELECTED_SEGMENTS_SHA256,
        "targets": targets,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "comparison.json", result)
    _write_csv(output_dir / "target_summary.csv", targets)
    (output_dir / "REPORT.md").write_text(_report(result), encoding="utf-8")
    _refresh_delivery(output_dir, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.run_root, args.output_dir)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "target_pass_count": result["target_pass_count"],
                "formal_success_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
