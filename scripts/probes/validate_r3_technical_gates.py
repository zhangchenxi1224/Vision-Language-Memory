from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FIXTURE_RGB_SHA256 = "c44093f3ad73d6a3d62b5bf9b8ad226f65e65afd7841d5ef3ed80bc7d14a841a"
SET_EVENT = "The user prefers red mugs."
OVERWRITE_EVENT = "The user now prefers blue mugs instead of red mugs."
QUERY = "Which mug color does the user prefer?"
CHOICES = ("red", "blue", "green", "yellow")

GATE_PROTOCOLS: dict[str, dict[str, Any]] = {
    "G4-L": {
        "semantic_operations": ("set",),
        "events": (SET_EVENT,),
        "target_index": 0,
        "detach_between_events": False,
    },
    "G5-L": {
        "semantic_operations": ("set", "overwrite"),
        "events": (SET_EVENT, OVERWRITE_EVENT),
        "target_index": 1,
        "detach_between_events": False,
    },
    "G6-L": {
        "semantic_operations": ("set", "overwrite"),
        "events": (SET_EVENT, OVERWRITE_EVENT),
        "target_index": 1,
        "detach_between_events": True,
    },
}

GATE_ORDER = ("G4-L", "G5-L", "G6-L", "DL-S")
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _canonical_sha256(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _finite(value: Any, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, found {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, found {result!r}.")
    return result


def _positive(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0:
        raise ValueError(f"{label} must be greater than zero, found {result!r}.")
    return result


def _require_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} must be {expected!r}, found {actual!r}.")


def _validate_provenance(report: Mapping[str, Any]) -> None:
    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Probe report is missing provenance.")
    git = provenance.get("git")
    if not isinstance(git, Mapping):
        raise ValueError("Probe provenance is missing git metadata.")
    commit = git.get("commit")
    if not isinstance(commit, str) or _HEX_40.fullmatch(commit) is None:
        raise ValueError("Probe provenance must contain a full 40-character git commit.")
    _require_equal(git.get("clean"), True, label="provenance.git.clean")

    models = provenance.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("Probe provenance is missing model metadata.")
    for model_name in ("dreamlite", "reader"):
        model = models.get(model_name)
        if not isinstance(model, Mapping):
            raise ValueError(f"Probe provenance is missing {model_name} metadata.")
        _require_equal(
            model.get("revision_matches_lock"),
            True,
            label=f"provenance.models.{model_name}.revision_matches_lock",
        )


def _validate_frozen_gradients(report: Mapping[str, Any]) -> None:
    frozen = report.get("frozen_gradients")
    if not isinstance(frozen, Mapping):
        raise ValueError("Probe report is missing frozen_gradients.")
    for module_name in ("base_unet", "vae", "internal_qwen", "reader"):
        module = frozen.get(module_name)
        if not isinstance(module, Mapping):
            raise ValueError(f"frozen_gradients is missing {module_name}.")
        _require_equal(
            module.get("frozen_tensors_with_grad"),
            0,
            label=f"frozen_gradients.{module_name}.frozen_tensors_with_grad",
        )
        _require_equal(
            module.get("frozen_nonfinite_grad_elements"),
            0,
            label=f"frozen_gradients.{module_name}.frozen_nonfinite_grad_elements",
        )
        if module_name in {"vae", "internal_qwen", "reader"}:
            _require_equal(
                module.get("trainable_parameter_tensors"),
                0,
                label=f"frozen_gradients.{module_name}.trainable_parameter_tensors",
            )


def _validate_memory_report(report: Mapping[str, Any]) -> None:
    memory = report.get("cuda_peak_memory")
    if not isinstance(memory, Mapping):
        raise ValueError("Probe report is missing cuda_peak_memory.")
    for device in ("cuda:0", "cuda:1"):
        record = memory.get(device)
        if not isinstance(record, Mapping):
            raise ValueError(f"cuda_peak_memory is missing {device}; both allocated A800s must be recorded.")
        for field in ("peak_allocated_gib", "peak_reserved_gib"):
            value = _finite(record.get(field), label=f"cuda_peak_memory.{device}.{field}")
            if value < 0:
                raise ValueError(f"cuda_peak_memory.{device}.{field} cannot be negative.")


def validate_probe_report(report: Mapping[str, Any], gate: str) -> dict[str, Any]:
    """Validate one R3 listwise technical gate against its locked semantic fixture."""

    if gate not in GATE_PROTOCOLS:
        raise ValueError(f"Unsupported probe gate: {gate}.")
    protocol = GATE_PROTOCOLS[gate]
    events = list(protocol["events"])
    expected_intermediate_count = len(events) - 1

    _require_equal(report.get("probe"), "e2e_episode_grad", label=f"{gate}.probe")
    _require_equal(report.get("events"), len(events), label=f"{gate}.events")
    _require_equal(
        report.get("detach_between_events"),
        protocol["detach_between_events"],
        label=f"{gate}.detach_between_events",
    )
    _require_equal(report.get("reader_loss_mode"), "listwise-choice", label=f"{gate}.reader_loss_mode")
    _require_equal(report.get("updater_device"), "cuda:0", label=f"{gate}.updater_device")
    _require_equal(report.get("reader_device"), "cuda:1", label=f"{gate}.reader_device")

    metadata = report.get("pair_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{gate} is missing pair_metadata.")
    _require_equal(metadata.get("event"), events, label=f"{gate}.pair_metadata.event")
    _require_equal(metadata.get("query"), QUERY, label=f"{gate}.pair_metadata.query")
    _require_equal(metadata.get("reader_loss_mode"), "listwise-choice", label=f"{gate}.pair_metadata.reader_loss_mode")
    _require_equal(metadata.get("target"), None, label=f"{gate}.pair_metadata.target")
    _require_equal(metadata.get("choices"), list(CHOICES), label=f"{gate}.pair_metadata.choices")
    _require_equal(
        metadata.get("target_index"),
        protocol["target_index"],
        label=f"{gate}.pair_metadata.target_index",
    )
    _require_equal(metadata.get("resolution"), 1024, label=f"{gate}.pair_metadata.resolution")
    _require_equal(metadata.get("adapter_seed"), 0, label=f"{gate}.pair_metadata.adapter_seed")
    _require_equal(
        metadata.get("event_noise_seeds"),
        list(range(len(events))),
        label=f"{gate}.pair_metadata.event_noise_seeds",
    )
    _require_equal(metadata.get("lora_rank"), 4, label=f"{gate}.pair_metadata.lora_rank")
    _require_equal(metadata.get("checkpoint_unet"), True, label=f"{gate}.pair_metadata.checkpoint_unet")
    _require_equal(metadata.get("dreamlite_device"), "cuda:0", label=f"{gate}.pair_metadata.dreamlite_device")
    _require_equal(metadata.get("reader_device"), "cuda:1", label=f"{gate}.pair_metadata.reader_device")

    source = metadata.get("source_image")
    if not isinstance(source, Mapping):
        raise ValueError(f"{gate}.pair_metadata is missing source_image.")
    _require_equal(source.get("origin"), "deterministic_fixture", label=f"{gate}.source_image.origin")
    _require_equal(source.get("mode"), "RGB", label=f"{gate}.source_image.mode")
    _require_equal(source.get("size"), [1024, 1024], label=f"{gate}.source_image.size")
    _require_equal(source.get("rgb_sha256"), FIXTURE_RGB_SHA256, label=f"{gate}.source_image.rgb_sha256")

    expected_pair_id = _canonical_sha256(dict(metadata))
    _require_equal(report.get("pair_id"), expected_pair_id, label=f"{gate}.pair_id")
    if _HEX_64.fullmatch(expected_pair_id) is None:  # pragma: no cover - hashlib contract
        raise ValueError(f"{gate}.pair_id is not a SHA256 digest.")

    _positive(report.get("loss"), label=f"{gate}.loss")
    choice_nll = report.get("choice_mean_nll")
    if not isinstance(choice_nll, Sequence) or isinstance(choice_nll, (str, bytes)) or len(choice_nll) != 4:
        raise ValueError(f"{gate}.choice_mean_nll must contain exactly four values.")
    for index, value in enumerate(choice_nll):
        _finite(value, label=f"{gate}.choice_mean_nll[{index}]")

    _positive(report.get("lora_grad_norm"), label=f"{gate}.lora_grad_norm")
    tensors_with_grad = report.get("lora_tensors_with_grad")
    if not isinstance(tensors_with_grad, int) or tensors_with_grad <= 0:
        raise ValueError(f"{gate}.lora_tensors_with_grad must be a positive integer.")
    _require_equal(report.get("lora_nonfinite_elements"), 0, label=f"{gate}.lora_nonfinite_elements")
    _positive(report.get("unclamped_image_grad_norm"), label=f"{gate}.unclamped_image_grad_norm")
    final_state_sha256 = report.get("final_state_sha256")
    if not isinstance(final_state_sha256, str) or _HEX_64.fullmatch(final_state_sha256) is None:
        raise ValueError(f"{gate}.final_state_sha256 must be a SHA256 digest.")
    final_state_gradient = report.get("final_state_gradient")
    if not isinstance(final_state_gradient, Mapping):
        raise ValueError(f"{gate}.final_state_gradient is missing.")
    _positive(final_state_gradient.get("norm"), label=f"{gate}.final_state_gradient.norm")
    _require_equal(
        final_state_gradient.get("nonfinite_elements"),
        0,
        label=f"{gate}.final_state_gradient.nonfinite_elements",
    )

    intermediate = report.get("intermediate_gradients")
    if not isinstance(intermediate, list) or len(intermediate) != expected_intermediate_count:
        raise ValueError(f"{gate}.intermediate_gradients must contain {expected_intermediate_count} record(s).")
    for index, record in enumerate(intermediate):
        if not isinstance(record, Mapping):
            raise ValueError(f"{gate}.intermediate_gradients[{index}] must be an object.")
        if protocol["detach_between_events"]:
            _require_equal(record.get("norm"), None, label=f"{gate}.intermediate_gradients[{index}].norm")
            _require_equal(
                record.get("nonfinite_elements"),
                None,
                label=f"{gate}.intermediate_gradients[{index}].nonfinite_elements",
            )
        else:
            _positive(record.get("norm"), label=f"{gate}.intermediate_gradients[{index}].norm")
            _require_equal(
                record.get("nonfinite_elements"),
                0,
                label=f"{gate}.intermediate_gradients[{index}].nonfinite_elements",
            )

    _validate_frozen_gradients(report)
    _validate_memory_report(report)
    _validate_provenance(report)
    return {
        "valid": True,
        "semantic_operations": list(protocol["semantic_operations"]),
        "event_count": len(events),
        "target_index": protocol["target_index"],
        "loss": float(report["loss"]),
        "pair_id": report["pair_id"],
    }


def validate_pair(
    positive: Mapping[str, Any],
    detached: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Validate the G5-L/G6-L intervention as a forward-identical detach pair."""

    if atol < 0 or rtol < 0:
        raise ValueError("Pair tolerances must be non-negative.")
    _require_equal(positive.get("detach_between_events"), False, label="G5-L.detach_between_events")
    _require_equal(detached.get("detach_between_events"), True, label="G6-L.detach_between_events")
    _require_equal(detached.get("pair_id"), positive.get("pair_id"), label="G5-L/G6-L.pair_id")
    _require_equal(
        detached.get("pair_metadata"),
        positive.get("pair_metadata"),
        label="G5-L/G6-L.pair_metadata",
    )
    _require_equal(
        detached.get("final_state_sha256"),
        positive.get("final_state_sha256"),
        label="G5-L/G6-L.final_state_sha256",
    )
    positive_loss = _finite(positive.get("loss"), label="G5-L.loss")
    detached_loss = _finite(detached.get("loss"), label="G6-L.loss")
    if not math.isclose(positive_loss, detached_loss, abs_tol=atol, rel_tol=rtol):
        raise ValueError(
            "G5-L/G6-L forward losses differ outside the locked tolerances: "
            f"positive={positive_loss}, detached={detached_loss}, atol={atol}, rtol={rtol}."
        )
    return {
        "valid": True,
        "pair_id": positive.get("pair_id"),
        "positive_loss": positive_loss,
        "detached_loss": detached_loss,
        "absolute_loss_difference": abs(positive_loss - detached_loss),
        "atol": atol,
        "rtol": rtol,
    }


def validate_resume_report(report: Mapping[str, Any]) -> dict[str, Any]:
    _require_equal(report.get("schema_version"), 2, label="DL-S.schema_version")
    _require_equal(
        report.get("protocol"),
        "DL-S-common-prefix-16-vs-8-resume-8-next-step-v2",
        label="DL-S.protocol",
    )
    _require_equal(report.get("passed"), True, label="DL-S.passed")
    _require_equal(report.get("exact"), True, label="DL-S.exact")
    _require_equal(report.get("atol"), 0.0, label="DL-S.atol")
    _require_equal(report.get("rtol"), 0.0, label="DL-S.rtol")
    _require_equal(report.get("mismatch_count"), 0, label="DL-S.mismatch_count")
    _require_equal(
        report.get("presentations"),
        {"uninterrupted": 16, "shared_prefix": 8, "resumed_suffix": 8, "next_step": 17},
        label="DL-S.presentations",
    )
    checkpoints = report.get("checkpoint_state")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("DL-S is missing checkpoint_state.")
    _require_equal(
        checkpoints.get("prefix"),
        {"epoch": 0, "episode_cursor": 8, "optimizer_step": 8},
        label="DL-S.checkpoint_state.prefix",
    )
    expected_final = {"epoch": 0, "episode_cursor": 16, "optimizer_step": 16}
    _require_equal(checkpoints.get("reference"), expected_final, label="DL-S.checkpoint_state.reference")
    _require_equal(checkpoints.get("resumed"), expected_final, label="DL-S.checkpoint_state.resumed")
    next_checkpoints = report.get("next_checkpoint_state")
    if not isinstance(next_checkpoints, Mapping):
        raise ValueError("DL-S is missing next_checkpoint_state.")
    expected_next = {"epoch": 1, "episode_cursor": 1, "optimizer_step": 17}
    _require_equal(next_checkpoints.get("reference"), expected_next, label="DL-S.next.reference")
    _require_equal(next_checkpoints.get("resumed"), expected_next, label="DL-S.next.resumed")
    next_metric = report.get("next_step_metric")
    if not isinstance(next_metric, Mapping):
        raise ValueError("DL-S is missing exact next_step_metric evidence.")
    for field in ("raw_gradient_sha256", "clipped_gradient_sha256"):
        digest = next_metric.get(field)
        if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
            raise ValueError(f"DL-S.next_step_metric.{field} must be a SHA256 digest.")
    for field in ("loss_hex", "gradient_norm_hex"):
        if not isinstance(next_metric.get(field), str):
            raise ValueError(f"DL-S.next_step_metric.{field} must be an exact hexadecimal float string.")
    lineage = report.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("DL-S is missing lineage.")
    _require_equal(lineage.get("training_regime"), "qa_only", label="DL-S.lineage.training_regime")
    _require_equal(lineage.get("reader_loss_mode"), "listwise-choice", label="DL-S.lineage.reader_loss_mode")
    _require_equal(lineage.get("qa_supervision"), "listwise-choice", label="DL-S.lineage.qa_supervision")
    _require_equal(lineage.get("choice_view_schedule"), "cyclic4", label="DL-S.lineage.choice_view_schedule")

    paths = report.get("checkpoint_paths")
    if not isinstance(paths, Mapping) or not all(
        isinstance(paths.get(name), str)
        for name in ("prefix", "reference", "resumed", "reference_next", "resumed_next")
    ):
        raise ValueError("DL-S must record prefix/reference/resumed checkpoint paths.")
    prefix_path = Path(str(paths["prefix"]))
    reference_path = Path(str(paths["reference"]))
    resumed_path = Path(str(paths["resumed"]))
    _require_equal(prefix_path.name, "checkpoint-000008.pt", label="DL-S.checkpoint_paths.prefix.name")
    _require_equal(reference_path.name, "checkpoint-000016.pt", label="DL-S.checkpoint_paths.reference.name")
    _require_equal(resumed_path.name, "checkpoint-000016.pt", label="DL-S.checkpoint_paths.resumed.name")
    _require_equal(prefix_path.parent, reference_path.parent, label="DL-S.common_prefix_parent")
    if resumed_path.parent == reference_path.parent:
        raise ValueError("DL-S resumed checkpoint must come from a separate output directory.")
    _require_equal(Path(str(paths["reference_next"])).parent, reference_path.parent, label="DL-S.reference_next.parent")
    _require_equal(Path(str(paths["resumed_next"])).parent, resumed_path.parent, label="DL-S.resumed_next.parent")

    checkpoint_sha256 = report.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, Mapping):
        raise ValueError("DL-S must record checkpoint SHA256 values.")
    for name in ("prefix", "reference", "resumed", "reference_next", "resumed_next"):
        digest = checkpoint_sha256.get(name)
        if not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None:
            raise ValueError(f"DL-S.checkpoint_sha256.{name} must be a SHA256 digest.")
    return {
        "valid": True,
        "exact": True,
        "presentations": dict(report["presentations"]),
    }


def validate_reports(
    *,
    through: str,
    g4: Mapping[str, Any] | None = None,
    g5: Mapping[str, Any] | None = None,
    g6: Mapping[str, Any] | None = None,
    resume: Mapping[str, Any] | None = None,
    pair_atol: float = 1e-5,
    pair_rtol: float = 1e-4,
) -> dict[str, Any]:
    if through not in GATE_ORDER:
        raise ValueError(f"Unknown through gate: {through}.")
    required = GATE_ORDER[: GATE_ORDER.index(through) + 1]
    inputs = {"G4-L": g4, "G5-L": g5, "G6-L": g6, "DL-S": resume}
    checks: dict[str, Any] = {}
    errors: list[str] = []
    probe_commits: set[str] = set()

    for gate in required:
        report = inputs[gate]
        if report is None:
            errors.append(f"{gate}: required report was not supplied.")
            continue
        try:
            checks[gate] = validate_resume_report(report) if gate == "DL-S" else validate_probe_report(report, gate)
            if gate != "DL-S":
                provenance = report.get("provenance")
                git = provenance.get("git") if isinstance(provenance, Mapping) else None
                commit = git.get("commit") if isinstance(git, Mapping) else None
                if isinstance(commit, str):
                    probe_commits.add(commit)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{gate}: {exc}")

    if "G6-L" in required and g5 is not None and g6 is not None:
        try:
            checks["G5-L/G6-L-pair"] = validate_pair(g5, g6, atol=pair_atol, rtol=pair_rtol)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"G5-L/G6-L-pair: {exc}")

    if len(probe_commits) != 1:
        errors.append("technical probe reports do not share one clean Git commit")
    git_commit = next(iter(probe_commits)) if len(probe_commits) == 1 else None
    return {
        "schema_version": 1,
        "protocol": "R3-technical-listwise-v1",
        "through": through,
        "required_gates": list(required),
        "failure_policy": "fail-closed; gates are serial and downstream gates require prior success",
        "pair_atol": pair_atol,
        "pair_rtol": pair_rtol,
        "checks": checks,
        "git_commit": git_commit,
        "errors": errors,
        "passed": not errors and all(gate in checks for gate in required),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed validation for R3 listwise technical gates")
    parser.add_argument("--through", choices=GATE_ORDER, required=True)
    parser.add_argument("--g4", type=Path)
    parser.add_argument("--g5", type=Path)
    parser.add_argument("--g6", type=Path)
    parser.add_argument("--resume-report", type=Path)
    parser.add_argument("--pair-atol", type=float, default=1e-5)
    parser.add_argument("--pair-rtol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "g4": args.g4,
        "g5": args.g5,
        "g6": args.g6,
        "resume": args.resume_report,
    }
    loaded: dict[str, dict[str, Any] | None] = {}
    load_errors: list[str] = []
    for name, path in paths.items():
        if path is None:
            loaded[name] = None
            continue
        try:
            loaded[name] = _read_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            loaded[name] = None
            load_errors.append(f"{name}: could not load {path}: {exc}")

    try:
        report = validate_reports(
            through=args.through,
            g4=loaded["g4"],
            g5=loaded["g5"],
            g6=loaded["g6"],
            resume=loaded["resume"],
            pair_atol=args.pair_atol,
            pair_rtol=args.pair_rtol,
        )
    except (TypeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "protocol": "R3-technical-listwise-v1",
            "through": args.through,
            "checks": {},
            "errors": [str(exc)],
            "passed": False,
        }
    if load_errors:
        report["errors"] = load_errors + list(report.get("errors", []))
        report["passed"] = False
    report["input_paths"] = {name: None if path is None else str(path.resolve()) for name, path in paths.items()}
    report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
