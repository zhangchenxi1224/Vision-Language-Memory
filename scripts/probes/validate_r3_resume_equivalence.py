from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "cluster"))
sys.path.insert(0, str(ROOT / "src"))

from compare_checkpoints import compare_values  # noqa: E402
from vision_memory.reader import R3_QWEN_READER_RESIZE_CONTRACT  # noqa: E402


REQUIRED_CHECKPOINT_FIELDS = {
    "schema_version",
    "trainable_state",
    "optimizer",
    "epoch",
    "episode_cursor",
    "optimizer_step",
    "rng_state",
    "manifest",
    "trainer_state",
}

EXPECTED_ARGUMENTS: dict[str, Any] = {
    "reader_loss_mode": "listwise-choice",
    "choice_view_schedule": "cyclic4",
    "training_regime": "qa_only",
    "objective_stage": "qa",
    "initial_state_mode": "blank",
    "epochs": 2,
    "max_train_episodes": 16,
    "max_optimizer_steps": 17,
    "audit_gradient_sha": True,
    "strict_determinism": True,
    "require_mixed_delayed_probe": True,
    "gradient_accumulation": 1,
    "checkpoint_every": 8,
    "eval_every": 100000,
    "lora_rank": 4,
    "recurrence_mode": "direct_latent",
    "detach_between_events": False,
    "noop_policy": "update",
    "checkpoint_unet": True,
    "dreamlite_device": "cuda:0",
    "reader_device": "cuda:1",
}
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_DETERMINISM_ENV = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_state(payload: Mapping[str, Any]) -> dict[str, int]:
    return {
        "epoch": int(payload["epoch"]),
        "episode_cursor": int(payload["episode_cursor"]),
        "optimizer_step": int(payload["optimizer_step"]),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_checkpoint_paths(
    prefix: Path,
    reference: Path,
    resumed: Path,
    reference_next: Path,
    resumed_next: Path,
) -> list[str]:
    """Lock the common-prefix fork topology represented by the three checkpoint paths."""

    errors: list[str] = []
    resolved_prefix = prefix.resolve()
    resolved_reference = reference.resolve()
    resolved_resumed = resumed.resolve()
    if resolved_prefix.name != "checkpoint-000008.pt":
        errors.append("DL-S prefix must be named checkpoint-000008.pt.")
    if resolved_reference.name != "checkpoint-000016.pt" or resolved_resumed.name != "checkpoint-000016.pt":
        errors.append("DL-S step-16 reference and resumed endpoints must be checkpoint-000016.pt.")
    if resolved_prefix.parent != resolved_reference.parent:
        errors.append("DL-S prefix and uninterrupted endpoint must come from the same run directory.")
    if resolved_resumed.parent == resolved_reference.parent:
        errors.append("DL-S resumed endpoint must use a separate output directory from the reference run.")
    if reference_next.resolve().name != "last.pt" or resumed_next.resolve().name != "last.pt":
        errors.append("DL-S next-step endpoints must both be named last.pt.")
    if reference_next.resolve().parent != resolved_reference.parent:
        errors.append("DL-S uninterrupted step-16 and next-step checkpoints must share a run directory.")
    if resumed_next.resolve().parent != resolved_resumed.parent:
        errors.append("DL-S resumed step-16 and next-step checkpoints must share a run directory.")
    return errors


def read_step_metric(path: Path, *, optimizer_step: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"Metrics file contains a blank line at {path}:{line_number}.")
        value = json.loads(line)
        if isinstance(value, dict) and value.get("kind") == "train" and value.get("optimizer_step") == optimizer_step:
            records.append(value)
    if len(records) != 1:
        raise ValueError(f"Expected exactly one train metric for optimizer step {optimizer_step} in {path}.")
    required = {
        "loss_hex",
        "gradient_norm_hex",
        "raw_gradient_sha256",
        "clipped_gradient_sha256",
    }
    if not required.issubset(records[0]):
        raise ValueError(f"Step metric is missing bitwise loss/gradient fields: {sorted(required - set(records[0]))}.")
    for field in ("raw_gradient_sha256", "clipped_gradient_sha256"):
        if not isinstance(records[0][field], str) or _HEX_64.fullmatch(records[0][field]) is None:
            raise ValueError(f"Step metric {field} must be a SHA256 digest.")
    return {field: records[0][field] for field in sorted(required)}


def _manifest(payload: Mapping[str, Any], *, label: str, errors: list[str]) -> Mapping[str, Any] | None:
    manifest = payload.get("manifest")
    if not isinstance(manifest, Mapping):
        errors.append(f"{label}.manifest is missing or is not an object.")
        return None
    return manifest


def validate_resume_checkpoints(
    prefix: Mapping[str, Any],
    reference: Mapping[str, Any],
    resumed: Mapping[str, Any],
    reference_next: Mapping[str, Any] | None = None,
    resumed_next: Mapping[str, Any] | None = None,
    reference_next_metric: Mapping[str, Any] | None = None,
    resumed_next_metric: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact 16 versus shared-8-prefix + resume-8 continuation contract."""

    errors: list[str] = []
    payloads = {"prefix": prefix, "reference": reference, "resumed": resumed}
    missing: dict[str, list[str]] = {}
    for label, payload in payloads.items():
        missing[label] = sorted(REQUIRED_CHECKPOINT_FIELDS - set(payload))
        if missing[label]:
            errors.append(f"{label} checkpoint is missing required fields: {missing[label]}.")

    checkpoint_state: dict[str, dict[str, int] | None] = {name: None for name in payloads}
    if not any(missing.values()):
        for label, payload in payloads.items():
            try:
                checkpoint_state[label] = _checkpoint_state(payload)
            except (TypeError, ValueError) as exc:
                errors.append(f"{label} checkpoint state is invalid: {exc}.")

        expected_states = {
            "prefix": {"epoch": 0, "episode_cursor": 8, "optimizer_step": 8},
            "reference": {"epoch": 0, "episode_cursor": 16, "optimizer_step": 16},
            "resumed": {"epoch": 0, "episode_cursor": 16, "optimizer_step": 16},
        }
        for label, expected in expected_states.items():
            if checkpoint_state[label] != expected:
                errors.append(f"{label} checkpoint state must be {expected}, found {checkpoint_state[label]}.")

    reference_manifest = _manifest(reference, label="reference", errors=errors)
    prefix_manifest = _manifest(prefix, label="prefix", errors=errors)
    resumed_manifest = _manifest(resumed, label="resumed", errors=errors)
    if reference_manifest is not None and prefix_manifest is not None:
        errors.extend(
            compare_values(
                reference_manifest,
                prefix_manifest,
                atol=0.0,
                rtol=0.0,
                path="prefix.manifest",
            )
        )
    if reference_manifest is not None and resumed_manifest is not None:
        errors.extend(
            compare_values(
                reference_manifest,
                resumed_manifest,
                atol=0.0,
                rtol=0.0,
                path="resumed.manifest",
            )
        )

    lineage: dict[str, Any] = {}
    if reference_manifest is not None:
        raw_lineage = reference_manifest.get("training_lineage")
        if not isinstance(raw_lineage, Mapping):
            errors.append("reference.manifest.training_lineage is missing or invalid.")
        else:
            lineage = {
                "training_regime": raw_lineage.get("training_regime"),
                "reader_loss_mode": raw_lineage.get("reader_loss_mode"),
                "qa_supervision": raw_lineage.get("qa_supervision"),
                "choice_view_schedule": raw_lineage.get("choice_view_schedule"),
            }
            expected_lineage = {
                "training_regime": "qa_only",
                "reader_loss_mode": "listwise-choice",
                "qa_supervision": "listwise-choice",
                "choice_view_schedule": "cyclic4",
            }
            if lineage != expected_lineage:
                errors.append(f"R3 DL-S lineage must be {expected_lineage}, found {lineage}.")

        arguments = reference_manifest.get("arguments")
        if not isinstance(arguments, Mapping):
            errors.append("reference.manifest.arguments is missing or invalid.")
        else:
            for name, expected in EXPECTED_ARGUMENTS.items():
                actual = arguments.get(name)
                if actual != expected:
                    errors.append(f"reference.manifest.arguments.{name} must be {expected!r}, found {actual!r}.")
        if reference_manifest.get("git_dirty") is not False:
            errors.append("DL-S requires manifest.git_dirty=false.")
        if reference_manifest.get("reader_resize_contract") != R3_QWEN_READER_RESIZE_CONTRACT:
            errors.append("DL-S manifest has the wrong Reader resize contract.")
        for field in ("git_commit", "dreamlite_revision", "reader_revision"):
            value = reference_manifest.get(field)
            if not isinstance(value, str) or _HEX_40.fullmatch(value) is None:
                errors.append(f"reference.manifest.{field} must be a full 40-character revision.")
        for field in ("train_sha256", "dev_sha256"):
            value = reference_manifest.get(field)
            if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
                errors.append(f"reference.manifest.{field} must be a SHA256 digest.")
        expected_runtime = {
            "python": "3.12.3",
            "torch": "2.7.0a0+ecf3bae40a.nv25.02",
            "torchvision": "0.22.0a0",
            "cuda_runtime": "12.8",
            "diffusers": "0.39.0",
            "transformers": "4.57.3",
            "peft": "0.18.1",
        }
        if reference_manifest.get("environment") != expected_runtime:
            errors.append(
                "reference.manifest.environment must equal the locked Inspire H200 software runtime."
            )
        determinism = reference_manifest.get("strict_determinism")
        if not isinstance(determinism, Mapping):
            errors.append("reference.manifest.strict_determinism is missing or invalid.")
        else:
            expected_determinism = {
                "seed": 0,
                "environment": EXPECTED_DETERMINISM_ENV,
                "deterministic_algorithms": True,
                "deterministic_warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cuda_matmul_allow_tf32": False,
                "cudnn_allow_tf32": False,
                "float32_matmul_precision": "highest",
                "sdpa": {"flash": False, "memory_efficient": False, "cudnn": False, "math": True},
            }
            for field, expected in expected_determinism.items():
                actual = determinism.get(field)
                if actual != expected:
                    errors.append(
                        f"reference.manifest.strict_determinism.{field} must be {expected!r}, "
                        f"found {actual!r}."
                    )
        initial_image = reference_manifest.get("initial_image")
        if not isinstance(initial_image, Mapping):
            errors.append("reference.manifest.initial_image is missing or invalid.")
        else:
            expected_initial = {
                "initial_state_mode": "blank",
                "origin": "blank_fixture",
                "mode": "RGB",
                "size": [1024, 1024],
            }
            for field, expected in expected_initial.items():
                actual = initial_image.get(field)
                if actual != expected:
                    errors.append(f"reference.manifest.initial_image.{field} must be {expected!r}, found {actual!r}.")

    comparison_errors: list[str] = []
    if not missing["reference"] and not missing["resumed"]:
        for field in sorted(REQUIRED_CHECKPOINT_FIELDS):
            comparison_errors.extend(
                compare_values(
                    reference[field],
                    resumed[field],
                    atol=0.0,
                    rtol=0.0,
                    path=field,
                )
            )
    errors.extend(comparison_errors)

    next_comparison_errors: list[str] = []
    next_checkpoint_state: dict[str, dict[str, int] | None] = {"reference": None, "resumed": None}
    if reference_next is None or resumed_next is None:
        errors.append("DL-S requires both uninterrupted and resumed next-step checkpoints.")
    else:
        for label, payload in (("reference", reference_next), ("resumed", resumed_next)):
            missing_next = sorted(REQUIRED_CHECKPOINT_FIELDS - set(payload))
            if missing_next:
                errors.append(f"next_{label} checkpoint is missing required fields: {missing_next}.")
            else:
                next_checkpoint_state[label] = _checkpoint_state(payload)
        expected_next = {"epoch": 1, "episode_cursor": 1, "optimizer_step": 17}
        for label, state in next_checkpoint_state.items():
            if state != expected_next:
                errors.append(f"next_{label} checkpoint state must be {expected_next}, found {state}.")
        if not any(REQUIRED_CHECKPOINT_FIELDS - set(payload) for payload in (reference_next, resumed_next)):
            for field in sorted(REQUIRED_CHECKPOINT_FIELDS):
                next_comparison_errors.extend(
                    compare_values(
                        reference_next[field],
                        resumed_next[field],
                        atol=0.0,
                        rtol=0.0,
                        path=f"next_step.{field}",
                    )
                )
    errors.extend(next_comparison_errors)
    next_metric_equal = reference_next_metric is not None and dict(reference_next_metric) == dict(resumed_next_metric or {})
    if not next_metric_equal:
        errors.append("DL-S next-step loss/raw-gradient/clipped-gradient metrics are not bitwise identical.")

    return {
        "schema_version": 2,
        "protocol": "DL-S-common-prefix-16-vs-8-resume-8-next-step-v2",
        "git_commit": None if reference_manifest is None else reference_manifest.get("git_commit"),
        "dreamlite_revision": (
            None if reference_manifest is None else reference_manifest.get("dreamlite_revision")
        ),
        "reader_revision": None if reference_manifest is None else reference_manifest.get("reader_revision"),
        "reader_resize_contract": (
            None if reference_manifest is None else reference_manifest.get("reader_resize_contract")
        ),
        "runtime_environment": (
            None if reference_manifest is None else reference_manifest.get("environment")
        ),
        "presentations": {"uninterrupted": 16, "shared_prefix": 8, "resumed_suffix": 8, "next_step": 17},
        "atol": 0.0,
        "rtol": 0.0,
        "exact": not comparison_errors and not next_comparison_errors and next_metric_equal,
        "missing": missing,
        "checkpoint_state": checkpoint_state,
        "lineage": lineage,
        "next_checkpoint_state": next_checkpoint_state,
        "next_step_metric": None if reference_next_metric is None else dict(reference_next_metric),
        "mismatch_count": len(comparison_errors) + len(next_comparison_errors) + int(not next_metric_equal),
        "mismatches": (comparison_errors + next_comparison_errors)[:100],
        "errors": errors[:200],
        "passed": not errors,
    }


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint {path} is not a mapping.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate exact R3 16 vs 8+resume+8 continuation")
    parser.add_argument("--prefix", type=Path, required=True, help="Shared checkpoint after 8 episode presentations")
    parser.add_argument("--reference", type=Path, required=True, help="Uninterrupted checkpoint after 16 presentations")
    parser.add_argument(
        "--resumed", type=Path, required=True, help="Resumed checkpoint after the remaining 8 presentations"
    )
    parser.add_argument("--reference-next", type=Path, required=True)
    parser.add_argument("--resumed-next", type=Path, required=True)
    parser.add_argument("--reference-metrics", type=Path, required=True)
    parser.add_argument("--resumed-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        report = validate_resume_checkpoints(
            _load_checkpoint(args.prefix),
            _load_checkpoint(args.reference),
            _load_checkpoint(args.resumed),
            _load_checkpoint(args.reference_next),
            _load_checkpoint(args.resumed_next),
            read_step_metric(args.reference_metrics, optimizer_step=17),
            read_step_metric(args.resumed_metrics, optimizer_step=17),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        report = {
            "schema_version": 2,
            "protocol": "DL-S-common-prefix-16-vs-8-resume-8-next-step-v2",
            "presentations": {"uninterrupted": 16, "shared_prefix": 8, "resumed_suffix": 8, "next_step": 17},
            "atol": 0.0,
            "rtol": 0.0,
            "exact": False,
            "mismatch_count": 0,
            "mismatches": [],
            "errors": [str(exc)],
            "passed": False,
        }
    path_errors = validate_checkpoint_paths(
        args.prefix,
        args.reference,
        args.resumed,
        args.reference_next,
        args.resumed_next,
    )
    if path_errors:
        report["errors"] = path_errors + list(report.get("errors", []))
        report["passed"] = False
    report["checkpoint_paths"] = {
        "prefix": str(args.prefix.resolve()),
        "reference": str(args.reference.resolve()),
        "resumed": str(args.resumed.resolve()),
        "reference_next": str(args.reference_next.resolve()),
        "resumed_next": str(args.resumed_next.resolve()),
        "reference_metrics": str(args.reference_metrics.resolve()),
        "resumed_metrics": str(args.resumed_metrics.resolve()),
    }
    try:
        report["checkpoint_sha256"] = {
            "prefix": _sha256_file(args.prefix),
            "reference": _sha256_file(args.reference),
            "resumed": _sha256_file(args.resumed),
            "reference_next": _sha256_file(args.reference_next),
            "resumed_next": _sha256_file(args.resumed_next),
            "reference_metrics": _sha256_file(args.reference_metrics),
            "resumed_metrics": _sha256_file(args.resumed_metrics),
        }
    except OSError as exc:
        report["errors"] = [f"Could not hash checkpoint inputs: {exc}"] + list(report.get("errors", []))
        report["passed"] = False
    report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
