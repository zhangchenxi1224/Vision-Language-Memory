from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.eval import read_prediction_jsonl, score_r3_micro  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _load_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object.")
        values.append(value)
    return values


def scientific_prediction_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Hash only deterministic scientific predictions, excluding paths, latency, and VRAM."""

    fields = (
        "episode_id",
        "query_ordinal",
        "probe_role",
        "choice_view_family",
        "choice_view_index",
        "condition",
        "choices",
        "target_index",
        "target_text",
        "prediction_index",
        "prediction_text",
        "donor_target_index",
        "donor_episode_id",
        "distractor_pair_id",
        "distractor_variant",
        "recurrence_mode",
        "initial_state_mode",
        "seed",
        "diffusion_seed",
        "deterministic_ce",
    )
    normalized: list[dict[str, Any]] = []
    for row in rows:
        value = {field: row.get(field) for field in fields}
        scores = row.get("choice_mean_nll")
        if not isinstance(scores, list) or len(scores) != 4:
            raise ValueError("R3 scientific prediction payload requires four choice NLL values per row.")
        value["choice_mean_nll_hex"] = [float(score).hex() for score in scores]
        normalized.append(value)
    normalized.sort(
        key=lambda row: (
            str(row["episode_id"]),
            int(row["query_ordinal"] or 0),
            str(row["probe_role"]),
            int(row["choice_view_index"] or 0),
            str(row["condition"]),
        )
    )
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema": "vlm.r3.micro_scientific_predictions.v1",
        "row_count": len(normalized),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_training_trace(
    *,
    checkpoint: Path,
    summary: Mapping[str, Any],
    arguments: Mapping[str, Any],
    lineage: Mapping[str, Any],
    suite: str,
) -> dict[str, Any]:
    episodes = 8 if suite == "set8" else 16 if suite == "transition16" else None
    if episodes is None:
        raise ValueError("Formal R3 trace validation requires set8 or transition16.")
    regime = lineage.get("training_regime")
    epochs = 512 if regime == "qa_only" else 256 if regime == "teacher_assisted" else None
    if epochs is None or lineage.get("objective_stage") != "qa":
        raise ValueError("Formal R3 micro score must bind a QA-stage checkpoint.")
    expected_lineage_presentations = {
        "presentations_per_state": 512,
        "distill_presentations": 0 if regime == "qa_only" else 256,
        "qa_presentations": 512 if regime == "qa_only" else 256,
    }
    lineage_drift = {
        field: {"expected": expected, "observed": lineage.get(field)}
        for field, expected in expected_lineage_presentations.items()
        if lineage.get(field) != expected
    }
    if lineage_drift:
        raise ValueError(f"R3 micro checkpoint presentation lineage drifted: {lineage_drift}.")
    step_per_presentation = episodes // 8
    expected_optimizer_steps = epochs * step_per_presentation
    eval_start = 64 * step_per_presentation
    eval_every = 32 * step_per_presentation
    expected_arguments = {
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "gradient_accumulation": 8,
        "gradient_clip": 1.0,
        "resolution": 1024,
        "checkpoint_unet": True,
        "curriculum": "full",
        "max_optimizer_steps": None,
        "max_train_episodes": episodes,
        "epochs": epochs,
        "presentations_per_state": epochs,
        "distill_presentations": 0 if regime == "qa_only" else 256,
        "qa_presentations": epochs,
        "checkpoint_every": eval_every,
        "eval_start_step": eval_start,
        "eval_every": eval_every,
        "eval_limit": episodes,
        "disable_early_stopping": True,
        "require_mixed_delayed_probe": True,
    }
    drift = {
        field: {"expected": expected, "observed": arguments.get(field)}
        for field, expected in expected_arguments.items()
        if arguments.get(field) != expected
    }
    if drift:
        raise ValueError(f"R3 micro optimizer/presentation protocol drifted: {drift}.")
    if int(summary.get("optimizer_steps", -1)) != expected_optimizer_steps:
        raise ValueError("R3 micro run did not complete its exact locked presentation budget.")

    metrics_path = checkpoint.parent / "metrics.jsonl"
    metrics = _load_jsonl_objects(metrics_path)
    train_rows = [row for row in metrics if row.get("kind") == "train"]
    dev_rows = [row for row in metrics if row.get("kind") == "dev"]
    if len(train_rows) != expected_optimizer_steps:
        raise ValueError("R3 micro metrics do not contain one train row per optimizer step.")
    if any(row.get("kind") == "resume" for row in metrics):
        raise ValueError("R3 micro A/B replicas must be fresh and cannot contain resume records.")
    if [int(row.get("optimizer_step", -1)) for row in train_rows] != list(range(1, expected_optimizer_steps + 1)):
        raise ValueError("R3 micro train optimizer-step sequence is incomplete or duplicated.")
    for row in train_rows:
        gradient_norm = float(row.get("gradient_norm", float("nan")))
        if not math.isfinite(gradient_norm) or gradient_norm <= 0:
            raise ValueError("R3 micro train metrics contain an invalid LoRA gradient norm.")
        state_audit = row.get("state_gradient_audit")
        if not isinstance(state_audit, Mapping) or state_audit.get("passed") is not True:
            raise ValueError("R3 micro train metrics contain a failed state/image gradient audit.")
    rotation_counts = [0, 0, 0, 0]
    for row in train_rows:
        counts = row.get("choice_rotation_counts")
        if not isinstance(counts, list) or len(counts) != 4:
            raise ValueError("R3 micro train metrics lack cyclic4 choice counts.")
        rotation_counts = [left + int(right) for left, right in zip(rotation_counts, counts, strict=True)]
    expected_rotation_count = epochs * episodes // 4
    if rotation_counts != [expected_rotation_count] * 4:
        raise ValueError("R3 micro cyclic4 presentations are not exactly balanced.")
    expected_dev_steps = list(range(eval_start, expected_optimizer_steps + 1, eval_every))
    if [int(row.get("optimizer_step", -1)) for row in dev_rows] != expected_dev_steps:
        raise ValueError("R3 micro dev evaluations do not occur from 64 then every 32 presentations/state.")
    expected_checkpoints = [
        checkpoint.parent / f"checkpoint-{step:06d}.pt"
        for step in range(eval_every, expected_optimizer_steps + 1, eval_every)
    ]
    missing = [str(path) for path in expected_checkpoints if not path.is_file()]
    if missing:
        raise ValueError(f"R3 micro checkpoint cadence is incomplete: {missing[:3]}.")
    return {
        "schema": "vlm.r3.micro_training_trace.v1",
        "suite": suite,
        "episodes": episodes,
        "epochs": epochs,
        "presentations_per_state": epochs,
        "optimizer_steps": expected_optimizer_steps,
        "choice_rotation_counts": rotation_counts,
        "dev_optimizer_steps": expected_dev_steps,
        "checkpoint_count": len(expected_checkpoints),
        "metrics_sha256": sha256_file(metrics_path),
        "passed": True,
    }


def build_artifact_provenance(
    *,
    predictions: Path,
    rows: Sequence[Mapping[str, Any]],
    prediction_report: Path,
    suite: str,
) -> dict[str, Any]:
    """Bind a scientific micro-gate payload to the evaluated checkpoint lineage."""

    report = _load_object(prediction_report)
    if report.get("output_sha256") != sha256_file(predictions):
        raise ValueError("Prediction report SHA does not match the scored JSONL.")
    manifest = report.get("checkpoint_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Formal R3 micro scoring requires a checkpoint manifest.")
    lineage = manifest.get("training_lineage")
    if not isinstance(lineage, Mapping) or lineage.get("schema_version") != 2:
        raise ValueError("Formal R3 micro scoring requires schema-v2 training lineage.")
    arguments = manifest.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError("Formal R3 micro scoring requires checkpoint arguments.")
    locked_arguments = {
        "reader_loss_mode": "listwise-choice",
        "choice_view_schedule": "cyclic4",
        "recurrence_mode": "direct_latent",
        "detach_between_events": False,
        "noop_policy": "update",
        "initial_state_mode": "blank",
        "learn_initial_state": False,
        "lora_rank": 4,
        "seed": 0,
        "adapter_seed": 0,
        "strict_determinism": True,
        "audit_state_gradients": True,
        "disable_early_stopping": True,
    }
    drift = {
        field: {"expected": expected, "observed": arguments.get(field)}
        for field, expected in locked_arguments.items()
        if arguments.get(field) != expected
    }
    if drift:
        raise ValueError(f"Checkpoint violates the locked R3 micro protocol: {drift}.")
    if not isinstance(manifest.get("strict_determinism"), Mapping):
        raise ValueError("Formal R3 micro checkpoint lacks strict-determinism provenance.")
    if lineage.get("reader_loss_mode") != "listwise-choice" or lineage.get("choice_view_schedule") != "cyclic4":
        raise ValueError("Checkpoint lineage does not use listwise-choice/cyclic4 training.")
    expected_conditions = ["standard", "reset", "shuffle"]
    if suite == "transition16":
        expected_conditions.append("state_swap")
    evaluation_drift = {
        "choice_view_family": ("reverse-cyclic4", report.get("choice_view_family")),
        "conditions": (expected_conditions, report.get("conditions")),
        "noop_policy": ("keep", report.get("noop_policy")),
        "episodes_sha256": (manifest.get("dev_sha256"), report.get("episodes_sha256")),
        "deterministic_ce": (True, report.get("deterministic_ce")),
    }
    bad_evaluation = {
        field: {"expected": expected, "observed": observed}
        for field, (expected, observed) in evaluation_drift.items()
        if expected != observed
    }
    if bad_evaluation:
        raise ValueError(f"R3 micro evaluation protocol drifted: {bad_evaluation}.")
    if lineage.get("training_regime") == "qa_only":
        if lineage.get("parent_checkpoint_regime") is not None or lineage.get("parent_checkpoint_sha256") is not None:
            raise ValueError("QA-only R3 micro replicas must use fresh LoRA initialization.")
    elif lineage.get("training_regime") == "teacher_assisted":
        if lineage.get("parent_checkpoint_regime") != "teacher_assisted":
            raise ValueError("Teacher-assisted QA must retain its distill-parent lineage.")
    else:
        raise ValueError("R3 micro checkpoint has an unsupported training regime.")

    checkpoint_values = {str(row.get("checkpoint", "")) for row in rows}
    if len(checkpoint_values) != 1 or not next(iter(checkpoint_values)):
        raise ValueError("Every prediction row must name the same non-empty checkpoint path.")
    checkpoint = Path(next(iter(checkpoint_values))).expanduser().resolve(strict=True)
    training_summary_path = checkpoint.parent / "summary.json"
    training_summary = _load_object(training_summary_path)
    state_gradient_audit = training_summary.get("state_gradient_audit")
    if not isinstance(state_gradient_audit, Mapping) or state_gradient_audit.get("passed") is not True:
        raise ValueError("Formal R3 micro checkpoint lacks a passing state/image gradient audit.")
    training_trace = validate_training_trace(
        checkpoint=checkpoint,
        summary=training_summary,
        arguments=arguments,
        lineage=lineage,
        suite=suite,
    )
    expected_row_values = {
        "training_regime": lineage.get("training_regime"),
        "parent_checkpoint_regime": lineage.get("parent_checkpoint_regime"),
        "teacher_control": lineage.get("teacher_control"),
    }
    for field, expected in expected_row_values.items():
        if any(row.get(field) != expected for row in rows):
            raise ValueError(f"Prediction rows disagree with checkpoint lineage field {field!r}.")
    row_protocol = {
        "recurrence_mode": "direct_latent",
        "initial_state_mode": "blank",
        "seed": 0,
        "diffusion_seed": 0,
        "deterministic_ce": True,
    }
    for field, expected in row_protocol.items():
        if any(row.get(field) != expected for row in rows):
            raise ValueError(f"Prediction rows violate the locked R3 field {field!r}.")

    return {
        "schema": "vlm.r3.micro_artifact_provenance.v1",
        "predictions_sha256": sha256_file(predictions),
        "prediction_report_sha256": sha256_file(prediction_report),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "training_summary_sha256": sha256_file(training_summary_path),
        "training_regime": lineage.get("training_regime"),
        "parent_checkpoint_regime": lineage.get("parent_checkpoint_regime"),
        "objective_stage": lineage.get("objective_stage"),
        "reader_loss_mode": lineage.get("reader_loss_mode"),
        "choice_permutation_family_sha256": lineage.get("choice_permutation_family_sha256"),
        "eval_choice_permutation_family_sha256": lineage.get("eval_choice_permutation_family_sha256"),
        "teacher_control": lineage.get("teacher_control"),
        "teacher_control_sha256": lineage.get("teacher_control_sha256"),
        "teacher_manifest_sha256": lineage.get("teacher_manifest_sha256"),
        "teacher_sidecar_sha256": lineage.get("teacher_sidecar_sha256"),
        "teacher_calibration_sha256": lineage.get("teacher_calibration_sha256"),
        "presentations_per_state": lineage.get("presentations_per_state"),
        "distill_presentations": lineage.get("distill_presentations"),
        "qa_presentations": lineage.get("qa_presentations"),
        "recurrence_mode": arguments.get("recurrence_mode"),
        "detach_between_events": arguments.get("detach_between_events"),
        "noop_policy": arguments.get("noop_policy"),
        "initial_state_mode": arguments.get("initial_state_mode"),
        "learn_initial_state": arguments.get("learn_initial_state"),
        "lora_rank": arguments.get("lora_rank"),
        "seed": arguments.get("seed"),
        "adapter_seed": arguments.get("adapter_seed"),
        "strict_determinism": dict(manifest["strict_determinism"]),
        "state_gradient_audit": dict(state_gradient_audit),
        "training_trace": training_trace,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Score preregistered R3 Set8/Transition16 gates")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--prediction-report",
        type=Path,
        help="dreamlite_mcq companion report; required by formal R3 orchestration for checkpoint binding.",
    )
    parser.add_argument("--suite", choices=("set8", "transition16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fail-on-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Controls may record a failed gate without aborting the attribution job.",
    )
    args = parser.parse_args()
    rows = read_prediction_jsonl(args.predictions)
    report = score_r3_micro(rows, args.suite)
    report["scientific_prediction_payload"] = scientific_prediction_payload(rows)
    if args.prediction_report is not None:
        report["artifact_provenance"] = build_artifact_provenance(
            predictions=args.predictions,
            rows=rows,
            prediction_report=args.prediction_report,
            suite=args.suite,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] or not args.fail_on_gate else 3


if __name__ == "__main__":
    raise SystemExit(main())
