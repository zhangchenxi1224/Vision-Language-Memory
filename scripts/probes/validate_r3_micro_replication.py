from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SCHEMAS = {
    "set8": "vlm.r3.set8_gate.v1",
    "transition16": "vlm.r3.transition16_gate.v1",
}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def validate_replication(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    suite: str,
    training_regime: str,
    teacher_control: str,
) -> dict[str, Any]:
    if suite not in _SCHEMAS:
        raise ValueError("suite must be set8 or transition16.")
    if training_regime not in {"qa_only", "teacher_assisted"}:
        raise ValueError("training_regime must be qa_only or teacher_assisted.")
    expected_control = "none" if training_regime == "qa_only" else "correct"
    if teacher_control != expected_control:
        raise ValueError(
            f"teacher_control must be {expected_control!r} for training_regime={training_regime!r}."
        )
    errors: list[str] = []
    digests: dict[str, str | None] = {}
    checkpoints: dict[str, str | None] = {}
    checkpoint_paths: dict[str, str | None] = {}
    prediction_payloads: dict[str, str | None] = {}
    for replica, report in (("A", a), ("B", b)):
        if report.get("schema_version") != _SCHEMAS[suite] or report.get("suite") != suite:
            errors.append(f"replica {replica} has the wrong suite/schema")
        if report.get("passed") is not True:
            errors.append(f"replica {replica} did not pass the preregistered scientific gate")
        digest = report.get("scientific_payload_sha256")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            errors.append(f"replica {replica} lacks a scientific payload SHA256")
            digest = None
        digests[replica] = digest
        prediction_payload = report.get("scientific_prediction_payload")
        prediction_digest = prediction_payload.get("sha256") if isinstance(prediction_payload, Mapping) else None
        if not isinstance(prediction_digest, str) or _SHA256.fullmatch(prediction_digest) is None:
            errors.append(f"replica {replica} lacks a scientific prediction payload SHA256")
            prediction_digest = None
        prediction_payloads[replica] = prediction_digest
        provenance = report.get("artifact_provenance")
        if not isinstance(provenance, Mapping):
            errors.append(f"replica {replica} lacks artifact provenance")
            continue
        if provenance.get("training_regime") != training_regime:
            errors.append(f"replica {replica} training regime mismatch")
        if provenance.get("objective_stage") != "qa":
            errors.append(f"replica {replica} is not a final QA-stage checkpoint")
        if provenance.get("teacher_control") != expected_control:
            errors.append(f"replica {replica} teacher control mismatch")
        checkpoint_sha = provenance.get("checkpoint_sha256")
        if not isinstance(checkpoint_sha, str) or _SHA256.fullmatch(checkpoint_sha) is None:
            errors.append(f"replica {replica} lacks checkpoint SHA256")
            checkpoint_sha = None
        checkpoints[replica] = checkpoint_sha
        checkpoint_path = provenance.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            errors.append(f"replica {replica} lacks checkpoint path")
            checkpoint_path = None
        checkpoint_paths[replica] = checkpoint_path
    if digests.get("A") != digests.get("B"):
        errors.append("A/B scientific payloads are not bitwise identical")
    if prediction_payloads.get("A") != prediction_payloads.get("B"):
        errors.append("A/B per-view scientific predictions are not bitwise identical")
    if checkpoint_paths.get("A") is not None and checkpoint_paths.get("A") == checkpoint_paths.get("B"):
        errors.append("A/B must be fresh independent output paths")
    return {
        "schema_version": 1,
        "protocol": "R3-micro-A-B-replication-v1",
        "suite": suite,
        "training_regime": training_regime,
        "teacher_control": expected_control,
        "scientific_payload_sha256": digests,
        "scientific_prediction_payload_sha256": prediction_payloads,
        "checkpoint_sha256": checkpoints,
        "checkpoint_paths": checkpoint_paths,
        "artifact_provenance_validated": not errors,
        "bitwise_scientific_payload_match": (
            digests.get("A") == digests.get("B")
            and digests.get("A") is not None
            and prediction_payloads.get("A") == prediction_payloads.get("B")
            and prediction_payloads.get("A") is not None
        ),
        "errors": errors,
        "passed": not errors,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate bitwise-identical R3 micro A/B scientific payloads")
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--suite", choices=tuple(_SCHEMAS), required=True)
    parser.add_argument("--training-regime", choices=("qa_only", "teacher_assisted"), required=True)
    parser.add_argument("--teacher-control", choices=("none", "correct"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_replication(
            _load_object(args.a),
            _load_object(args.b),
            suite=args.suite,
            training_regime=args.training_regime,
            teacher_control=args.teacher_control,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "protocol": "R3-micro-A-B-replication-v1",
            "suite": args.suite,
            "errors": [str(exc)],
            "passed": False,
        }
    report["inputs"] = {"A": str(args.a), "B": str(args.b)}
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
