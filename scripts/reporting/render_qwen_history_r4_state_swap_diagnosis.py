from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "src"))

from score_qwen_history_r4 import (  # noqa: E402
    PREDICTION_SCHEMA,
    RUNTIME_FIELDS,
    SCHEMA as SCORE_SCHEMA,
    prediction_identity,
    scientific_prediction_payload,
)
from vision_memory.eval.r4_history_representations import (  # noqa: E402
    QWEN_R4_LAST_EFFECTIVE_EVENT,
)
from vision_memory.repro import canonical_object_sha256  # noqa: E402


SCHEMA = "vlm.qwen-history-r4-state-swap-diagnosis.v1"
MANIFEST_SCHEMA = "vlm.qwen-history-r4-state-swap-diagnosis-manifest.v1"
EXPECTED_CONDITIONS = {"standard", "reset", "shuffle", "state_swap"}
EXPECTED_KINDS = {"set", "overwrite", "clear", "noop"}
EXPECTED_DIRECTIONS = {"r0->r0", "r0->r1", "r1->r0", "r1->r1"}
LEXICAL_REPLICA_RE = re.compile(r"(?:^|-)r([01])(?:-|:|$)")
PLOT_NAMES = (
    "state_swap_by_event_kind.png",
    "state_swap_by_lexical_direction.png",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"Prediction file is empty: {path}")
    return rows


def _atomic_write(path: Path, payload: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if isinstance(payload, bytes):
        temporary.write_bytes(payload)
    else:
        temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _lexical_replica(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string containing an r0/r1 identifier")
    match = LEXICAL_REPLICA_RE.search(value)
    if match is None:
        raise ValueError(f"{field} does not encode the locked r0/r1 lexical replica")
    return f"r{match.group(1)}"


def _validate_replica_rows(
    rows: Sequence[Mapping[str, Any]], *, replica_id: str
) -> dict[str, Any]:
    if len(rows) != 512:
        raise ValueError(f"Transition32 replica {replica_id} requires 512 records, got {len(rows)}")
    identities = [prediction_identity(row) for row in rows]
    if len(set(identities)) != len(rows):
        raise ValueError(f"Replica {replica_id} contains duplicate prediction identities")
    conditions = Counter(str(row.get("condition")) for row in rows)
    if set(conditions) != EXPECTED_CONDITIONS or any(
        conditions[name] != 128 for name in EXPECTED_CONDITIONS
    ):
        raise ValueError(f"Replica {replica_id} does not contain 128 rows per intervention")
    datasets: set[str] = set()
    for index, row in enumerate(rows):
        prefix = f"replica {replica_id} row {index}"
        if row.get("schema_version") != PREDICTION_SCHEMA:
            raise ValueError(f"{prefix} prediction schema drifted")
        if row.get("method") != QWEN_R4_LAST_EFFECTIVE_EVENT:
            raise ValueError(f"{prefix} is not the locked last-effective arm")
        if row.get("replica_id") != replica_id:
            raise ValueError(f"{prefix} replica label drifted")
        if row.get("input_mode") != "blank_image":
            raise ValueError(f"{prefix} did not use the fixed blank image")
        if row.get("probe_role") != "delayed":
            raise ValueError(f"{prefix} is not a delayed probe")
        if row.get("choice_view_family") != "reverse-cyclic4":
            raise ValueError(f"{prefix} choice-view family drifted")
        view = row.get("choice_view_index")
        if isinstance(view, bool) or not isinstance(view, int) or view not in range(4):
            raise ValueError(f"{prefix} has an invalid reverse-cyclic view index")
        dataset_sha = row.get("dataset_sha256")
        if not _valid_sha256(dataset_sha) or row.get("episodes_sha256") != dataset_sha:
            raise ValueError(f"{prefix} dataset SHA binding drifted")
        datasets.add(str(dataset_sha))
        choices = row.get("choices")
        prediction = row.get("prediction_index")
        target = row.get("target_index")
        if (
            not isinstance(choices, list)
            or len(choices) != 4
            or len(set(choices)) != 4
            or isinstance(prediction, bool)
            or not isinstance(prediction, int)
            or prediction not in range(4)
            or isinstance(target, bool)
            or not isinstance(target, int)
            or target not in range(4)
        ):
            raise ValueError(f"{prefix} choice/target/prediction contract drifted")
        if row.get("prediction_text") != choices[prediction]:
            raise ValueError(f"{prefix} prediction text/index binding drifted")
        if row.get("target_text") != choices[target]:
            raise ValueError(f"{prefix} target text/index binding drifted")
    if len(datasets) != 1:
        raise ValueError(f"Replica {replica_id} must bind exactly one dataset SHA")
    return {
        "record_count": len(rows),
        "identity_count": len(identities),
        "dataset_sha256": next(iter(datasets)),
        "condition_counts": dict(sorted(conditions.items())),
    }


def _validate_ab_and_score(
    *,
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    predictions_a: Path,
    predictions_b: Path,
    score: Mapping[str, Any],
) -> dict[str, Any]:
    validation_a = _validate_replica_rows(rows_a, replica_id="A")
    validation_b = _validate_replica_rows(rows_b, replica_id="B")
    if validation_a["dataset_sha256"] != validation_b["dataset_sha256"]:
        raise ValueError("Replica A/B dataset SHA values differ")

    identities_a = [prediction_identity(row) for row in rows_a]
    identities_b = [prediction_identity(row) for row in rows_b]
    if identities_a != identities_b:
        raise ValueError("Replica A/B prediction identities or traversal order differ")
    payload_a = scientific_prediction_payload(rows_a)
    payload_b = scientific_prediction_payload(rows_b)
    if payload_a["sha256"] != payload_b["sha256"]:
        raise ValueError("Replica A/B scientific payloads are not bitwise identical")

    if (
        score.get("schema") != SCORE_SCHEMA
        or score.get("suite") != "transition32"
        or score.get("method") != QWEN_R4_LAST_EFFECTIVE_EVENT
    ):
        raise ValueError("Score schema/suite/method is not the locked Transition32 arm")
    integrity = score.get("integrity")
    replication = score.get("replication")
    if not isinstance(integrity, Mapping) or integrity.get("passed") is not True:
        raise ValueError("Score integrity did not pass")
    if (
        not isinstance(replication, Mapping)
        or replication.get("passed") is not True
        or replication.get("identity_sets_match") is not True
        or replication.get("bitwise_scientific_payload_match") is not True
    ):
        raise ValueError("Score does not attest exact A/B replication")
    expected_bindings = {
        "prediction_sha256": sha256_file(predictions_a),
        "replica_b_prediction_sha256": sha256_file(predictions_b),
    }
    for key, expected in expected_bindings.items():
        if integrity.get(key) != expected:
            raise ValueError(f"Score does not bind {key}")
    for key in ("prediction_report_sha256", "replica_b_report_sha256"):
        if not _valid_sha256(integrity.get(key)):
            raise ValueError(f"Score lacks a valid authenticated {key}")
    for key in (
        "replica_a_scientific_payload_sha256",
        "replica_b_scientific_payload_sha256",
    ):
        if replication.get(key) != payload_a["sha256"]:
            raise ValueError(f"Score does not bind {key}")
    if replication.get("records_a") != len(rows_a) or replication.get("records_b") != len(rows_b):
        raise ValueError("Score A/B record counts drifted")

    for key, expected in (("replica_a", validation_a), ("replica_b", validation_b)):
        score_replica = integrity.get(key)
        if not isinstance(score_replica, Mapping):
            raise ValueError(f"Score lacks {key} integrity metadata")
        expected_id = "A" if key == "replica_a" else "B"
        if (
            score_replica.get("replica_id") != expected_id
            or score_replica.get("method") != QWEN_R4_LAST_EFFECTIVE_EVENT
            or score_replica.get("prediction_records") != expected["record_count"]
            or score_replica.get("unique_identities") != expected["identity_count"]
            or score_replica.get("dataset_sha256") != expected["dataset_sha256"]
        ):
            raise ValueError(f"Score {key} integrity metadata drifted")

    score_without_hash = {key: value for key, value in score.items() if key != "report_sha256"}
    if score.get("report_sha256") != canonical_object_sha256(score_without_hash):
        raise ValueError("Score report_sha256 does not bind its canonical payload")
    gate = score.get("scientific_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("Score lacks a scientific gate")
    gate_without_hash = {
        key: value for key, value in gate.items() if key != "scientific_payload_sha256"
    }
    if gate.get("scientific_payload_sha256") != canonical_object_sha256(gate_without_hash):
        raise ValueError("Scientific-gate SHA does not bind its canonical payload")
    return {
        "replica_a": validation_a,
        "replica_b": validation_b,
        "prediction_identity_sequence_match": True,
        "bitwise_scientific_payload_match": True,
        "scientific_payload_sha256": payload_a["sha256"],
        "score_report_sha256_valid": True,
        "score_prediction_sha_bindings_valid": True,
        "score_prediction_report_sha_fields_well_formed": True,
    }


def _state_swap_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    all_views = [row for row in rows if row.get("condition") == "state_swap"]
    if len(all_views) != 128:
        raise ValueError(f"Transition32 requires 128 state-swap views, got {len(all_views)}")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(all_views):
        prefix = f"state-swap row {index}"
        donor = row.get("donor_target_index")
        choices = row.get("choices")
        if isinstance(donor, bool) or not isinstance(donor, int) or donor not in range(4):
            raise ValueError(f"{prefix} has no valid donor target index")
        if not isinstance(row.get("donor_episode_id"), str):
            raise ValueError(f"{prefix} has no donor episode ID")
        if row.get("subtype") not in EXPECTED_KINDS:
            raise ValueError(f"{prefix} has an invalid event kind")
        if not isinstance(choices, list) or row.get("donor_target_index") >= len(choices):
            raise ValueError(f"{prefix} donor target cannot be mapped into choices")
        grouped[
            (
                row.get("episode_id"),
                row.get("query_ordinal"),
                row.get("probe_role"),
                row.get("condition"),
            )
        ].append(row)
    if len(grouped) != 32:
        raise ValueError(f"Transition32 requires 32 state-swap states, got {len(grouped)}")
    for identity, values in grouped.items():
        views = {row.get("choice_view_index") for row in values}
        if len(values) != 4 or views != {0, 1, 2, 3}:
            raise ValueError(f"State-swap state {identity!r} lacks four reverse-cyclic views")
        invariant_fields = ("episode_id", "donor_episode_id", "subtype", "form")
        for field in invariant_fields:
            if len({row.get(field) for row in values}) != 1:
                raise ValueError(f"State-swap state {identity!r} changes {field} across views")
        donor_texts = {
            row["choices"][int(row["donor_target_index"])] for row in values
        }
        if len(donor_texts) != 1:
            raise ValueError(f"State-swap state {identity!r} changes donor answer text across views")
    locked = [row for row in all_views if row.get("choice_view_index") == 0]
    if len(locked) != 32:
        raise ValueError(f"Locked state-swap gate requires 32 view0 rows, got {len(locked)}")
    return all_views, locked


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(row["prediction_index"] == row["donor_target_index"] for row in rows)
    return {
        "donor_answers": correct,
        "count": len(rows),
        "rate": correct / len(rows) if rows else None,
    }


def _group_locked_rows(
    locked: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_kind: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_direction: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_relation: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in locked:
        kind = str(row["subtype"])
        recipient = _lexical_replica(row.get("episode_id"), field="episode_id")
        donor = _lexical_replica(row.get("donor_episode_id"), field="donor_episode_id")
        direction = f"{recipient}->{donor}"
        relation = "same_entity" if recipient == donor else "cross_entity"
        by_kind[kind].append(row)
        by_direction[direction].append(row)
        by_relation[relation].append(row)
    if set(by_kind) != EXPECTED_KINDS or any(len(values) != 8 for values in by_kind.values()):
        raise ValueError("Locked view0 probe must contain 8 states per event kind")
    if set(by_direction) != EXPECTED_DIRECTIONS or any(
        len(values) != 8 for values in by_direction.values()
    ):
        raise ValueError("Locked view0 probe must contain 8 states per r0/r1 donor direction")
    if set(by_relation) != {"same_entity", "cross_entity"} or any(
        len(values) != 16 for values in by_relation.values()
    ):
        raise ValueError("Locked view0 probe must balance same-entity and cross-entity states")
    return (
        {name: _summary(values) for name, values in sorted(by_kind.items())},
        {name: _summary(values) for name, values in sorted(by_direction.items())},
        {name: _summary(values) for name, values in sorted(by_relation.items())},
    )


def diagnose_state_swap(
    *,
    predictions_a: Path,
    predictions_b: Path,
    score_path: Path,
) -> dict[str, Any]:
    rows_a = _load_jsonl(predictions_a)
    rows_b = _load_jsonl(predictions_b)
    score = _load_json(score_path)
    validation = _validate_ab_and_score(
        rows_a=rows_a,
        rows_b=rows_b,
        predictions_a=predictions_a,
        predictions_b=predictions_b,
        score=score,
    )
    all_views_a, locked_a = _state_swap_rows(rows_a)
    all_views_b, locked_b = _state_swap_rows(rows_b)
    locked_payload_a = [
        {key: value for key, value in row.items() if key not in RUNTIME_FIELDS}
        for row in locked_a
    ]
    locked_payload_b = [
        {key: value for key, value in row.items() if key not in RUNTIME_FIELDS}
        for row in locked_b
    ]
    if locked_payload_a != locked_payload_b:
        raise ValueError("Replica A/B locked state-swap rows differ scientifically")

    by_kind, by_direction, by_relation = _group_locked_rows(locked_a)
    locked_summary = _summary(locked_a)
    all_views_summary = _summary(all_views_a)
    score_gate = score["scientific_gate"]
    score_swap = score_gate.get("state_swap")
    thresholds = score_gate.get("thresholds")
    checks = score_gate.get("checks")
    descriptive = score.get("descriptive_metrics")
    descriptive_swap = (
        descriptive.get("state_swap_donor_answer")
        if isinstance(descriptive, Mapping)
        else None
    )
    if not isinstance(score_swap, Mapping) or dict(score_swap) != {
        "donor_answers": locked_summary["donor_answers"],
        "count": locked_summary["count"],
    }:
        raise ValueError("Score state-swap gate does not match the locked view0 rows")
    if not isinstance(thresholds, Mapping) or not isinstance(thresholds.get("donor"), int):
        raise ValueError("Score lacks the locked donor threshold")
    donor_passed = locked_summary["donor_answers"] >= int(thresholds["donor"])
    if (
        not isinstance(checks, Mapping)
        or checks.get("state_swap_donor") is not donor_passed
        or score_gate.get("passed") is not all(value is True for value in checks.values())
    ):
        raise ValueError("Score state-swap check/pass value is inconsistent with its threshold")
    expected_descriptive_swap = {
        "correct": all_views_summary["donor_answers"],
        "count": all_views_summary["count"],
        "rate": all_views_summary["rate"],
    }
    if not isinstance(descriptive_swap, Mapping) or any(
        descriptive_swap.get(key) != value
        for key, value in expected_descriptive_swap.items()
    ):
        raise ValueError("Score all-view state-swap metric does not match predictions")
    if (
        score.get("passed") is not False
        or score.get("execution_passed") is not False
        or score_gate.get("passed") is not False
        or checks.get("state_swap_donor") is not False
        or any(value is not True for key, value in checks.items() if key != "state_swap_donor")
    ):
        raise ValueError("This reporter requires the isolated R4 BH1 state-swap scientific failure")

    same = by_relation["same_entity"]
    cross = by_relation["cross_entity"]
    exact_confound_pattern = (
        same["donor_answers"] == same["count"]
        and cross["donor_answers"] == 0
    )
    return {
        "schema": SCHEMA,
        "scientific_stage": "R4-BH1",
        "suite": "transition32",
        "method": QWEN_R4_LAST_EFFECTIVE_EVENT,
        "report_generation_passed": True,
        "scientific_stage_passed": False,
        "r4_failure_preserved": True,
        "locked_gate_scope": {
            "choice_view_index": 0,
            "state_count": len(locked_a),
            "all_state_swap_view_count": len(all_views_a),
            "choice_view_family": "reverse-cyclic4",
        },
        "locked_gate": {
            **locked_summary,
            "threshold": int(thresholds["donor"]),
            "passed": donor_passed,
        },
        "all_views_descriptive": all_views_summary,
        "by_event_kind": by_kind,
        "by_lexical_replica_direction": by_direction,
        "by_entity_relation": by_relation,
        "diagnosis": {
            "exact_same_vs_cross_pattern": exact_confound_pattern,
            "classification": (
                "protocol_semantic_confound_detected"
                if exact_confound_pattern
                else "state_swap_failure_without_exact_same_vs_cross_pattern"
            ),
            "interpretation": (
                "Under the locked Transition32 ID contract, r0/r1 are lexical/entity replicates. "
                "The model follows the donor for same-entity directions but not cross-entity "
                "directions. This supports a semantic mismatch in the state-swap diagnostic: "
                "cross-entity donor memory is paired with the recipient query."
                if exact_confound_pattern
                else "The input does not exhibit the exact same-entity-pass/cross-entity-fail pattern."
            ),
            "scientific_status": (
                "R4 remains a preregistered scientific failure. This post-hoc diagnosis does "
                "not rescore the gate or authorize BH2/BH3."
            ),
            "prospective_only_recommendation": (
                "If continued, define an R5 lockbox with same entity/query/slot and a different "
                "donor terminal value before observing new results."
            ),
        },
        "validation": validation,
        "training_performed": False,
        "loss_curve_available": False,
        "loss_curve_reason": (
            "This is frozen Qwen inference. No optimizer step or training loss exists, so no "
            "loss curve is generated."
        ),
        "inputs": {
            "predictions_a": {
                "path": str(predictions_a.resolve()),
                "sha256": sha256_file(predictions_a),
            },
            "predictions_b": {
                "path": str(predictions_b.resolve()),
                "sha256": sha256_file(predictions_b),
            },
            "score": {"path": str(score_path.resolve()), "sha256": sha256_file(score_path)},
        },
    }


def _plot_breakdown(
    groups: Mapping[str, Mapping[str, Any]], *, title: str, output: Path
) -> None:
    labels = list(groups)
    correct = [int(groups[label]["donor_answers"]) for label in labels]
    counts = [int(groups[label]["count"]) for label in labels]
    colors = ["#4C78A8" if value else "#E45756" for value in correct]
    fig, axis = plt.subplots(figsize=(7.4, 4.4))
    bars = axis.bar(labels, correct, color=colors)
    axis.set_ylim(0, max(counts) + 1.2)
    axis.set_ylabel("Donor answers (locked view0 states)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.grid(axis="x", visible=False)
    axis.tick_params(axis="x", rotation=20)
    for bar, value, count in zip(bars, correct, counts, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.16,
            f"{value}/{count}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _csv_rows(diagnosis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, field in (
        ("event_kind", "by_event_kind"),
        ("lexical_replica_direction", "by_lexical_replica_direction"),
        ("entity_relation", "by_entity_relation"),
    ):
        groups = diagnosis[field]
        for name, summary in groups.items():
            rows.append(
                {
                    "gate_scope": "locked_reverse_cyclic_view0",
                    "breakdown": section,
                    "group": name,
                    "donor_answers": summary["donor_answers"],
                    "count": summary["count"],
                    "rate": summary["rate"],
                }
            )
    gate = diagnosis["locked_gate"]
    rows.append(
        {
            "gate_scope": "locked_reverse_cyclic_view0",
            "breakdown": "overall_gate",
            "group": "all",
            "donor_answers": gate["donor_answers"],
            "count": gate["count"],
            "rate": gate["rate"],
            "threshold": gate["threshold"],
            "passed": gate["passed"],
        }
    )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "gate_scope",
        "breakdown",
        "group",
        "donor_answers",
        "count",
        "rate",
        "threshold",
        "passed",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _image_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _markdown_table(groups: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [
        "| group | donor answers | states | rate |",
        "|---|---:|---:|---:|",
        *[
            f"| {name} | {value['donor_answers']} | {value['count']} | {value['rate']:.4f} |"
            for name, value in groups.items()
        ],
    ]


def render_diagnosis(
    *,
    predictions_a: Path,
    predictions_b: Path,
    score_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite diagnosis directory: {output_dir}")
    diagnosis = diagnose_state_swap(
        predictions_a=predictions_a,
        predictions_b=predictions_b,
        score_path=score_path,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    plots = output_dir / "plots"
    sources = output_dir / "sources"
    plots.mkdir()
    sources.mkdir()
    for source, destination in (
        (predictions_a, sources / "replica-a-predictions.jsonl"),
        (predictions_b, sources / "replica-b-predictions.jsonl"),
        (score_path, sources / "score.json"),
    ):
        shutil.copyfile(source, destination)

    plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "savefig.dpi": 180})
    _plot_breakdown(
        diagnosis["by_event_kind"],
        title="Locked state-swap donor answers by event kind",
        output=plots / PLOT_NAMES[0],
    )
    _plot_breakdown(
        diagnosis["by_lexical_replica_direction"],
        title="Locked state-swap donor answers by lexical/entity direction",
        output=plots / PLOT_NAMES[1],
    )
    _write_csv(output_dir / "diagnosis.csv", _csv_rows(diagnosis))

    diagnosis["artifacts"] = {
        "plots": {
            name: {"sha256": sha256_file(plots / name), "size": (plots / name).stat().st_size}
            for name in PLOT_NAMES
        },
        "diagnosis_csv_sha256": sha256_file(output_dir / "diagnosis.csv"),
        "source_copies": {
            path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted(sources.iterdir())
        },
    }
    _atomic_write(
        output_dir / "diagnosis.json",
        json.dumps(diagnosis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    gate = diagnosis["locked_gate"]
    same = diagnosis["by_entity_relation"]["same_entity"]
    cross = diagnosis["by_entity_relation"]["cross_entity"]
    markdown = [
        "# R4 BH1 state-swap root-cause diagnosis",
        "",
        "> **R4 SCIENTIFIC STATUS: FAILED.** This diagnostic report preserves the preregistered failure and does not authorize BH2/BH3.",
        "",
        f"The locked gate uses reverse-cyclic `view0` only: **{gate['donor_answers']}/{gate['count']}** donor answers, below the fixed **{gate['threshold']}/{gate['count']}** threshold.",
        "",
        "## Locked gate by event kind",
        "",
        *_markdown_table(diagnosis["by_event_kind"]),
        "",
        "![State-swap by event kind](plots/state_swap_by_event_kind.png)",
        "",
        "## Locked gate by lexical/entity replica direction",
        "",
        *_markdown_table(diagnosis["by_lexical_replica_direction"]),
        "",
        "![State-swap by lexical/entity direction](plots/state_swap_by_lexical_direction.png)",
        "",
        "## Interpretation",
        "",
        f"Same-entity directions produce **{same['donor_answers']}/{same['count']}** donor answers; cross-entity directions produce **{cross['donor_answers']}/{cross['count']}**. Under the locked Transition32 ID contract, `r0` and `r1` identify different lexical/entity replicas. The exact split supports a protocol semantic confound: cross-entity donor memory is paired with the recipient query.",
        "",
        "This is a post-hoc root-cause diagnosis, not a rescore. R4 remains failed. Any corrected state-swap protocol must be prospective R5 with a new untouched lockbox and same entity/query/slot but a different donor terminal value.",
        "",
        "## Reproducibility and loss-curve note",
        "",
        f"- A/B exact scientific payload: `{diagnosis['validation']['bitwise_scientific_payload_match']}`",
        f"- Scientific payload SHA256: `{diagnosis['validation']['scientific_payload_sha256']}`",
        "- Score prediction SHA bindings and canonical score SHA: validated.",
        "- Prediction-report SHA fields are checked for well-formed authenticated values; the report files are outside this three-input diagnostic contract.",
        "- Training performed: `false`.",
        "- Loss curve: unavailable by design because this is frozen inference with no optimizer or training loss.",
    ]
    _atomic_write(output_dir / "diagnosis.md", "\n".join(markdown) + "\n")

    table_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value['donor_answers']}</td><td>{value['count']}</td><td>{value['rate']:.4f}</td></tr>"
        for name, value in diagnosis["by_lexical_replica_direction"].items()
    )
    html_payload = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>R4 BH1 state-swap diagnosis</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1100px}}img{{max-width:100%;display:block;margin:1rem 0}}table{{border-collapse:collapse}}th,td{{padding:.35rem .7rem;border-bottom:1px solid #ccc;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{background:#eee;padding:.1rem .25rem}}</style></head>
<body><h1>R4 BH1 state-swap root-cause diagnosis</h1>
<p><strong>R4 SCIENTIFIC STATUS: FAILED.</strong> The preregistered result is unchanged and BH2/BH3 remain blocked.</p>
<p>The locked gate uses view0 only: <strong>{gate['donor_answers']}/{gate['count']}</strong>, below <strong>{gate['threshold']}/{gate['count']}</strong>.</p>
<h2>Event-kind breakdown</h2><img alt="Donor answers by event kind" src="{_image_uri(plots / PLOT_NAMES[0])}">
<h2>Lexical/entity direction breakdown</h2><table><thead><tr><th>direction</th><th>donor answers</th><th>states</th><th>rate</th></tr></thead><tbody>{table_rows}</tbody></table>
<img alt="Donor answers by lexical and entity direction" src="{_image_uri(plots / PLOT_NAMES[1])}">
<h2>Interpretation</h2><p>Same-entity directions: <strong>{same['donor_answers']}/{same['count']}</strong>; cross-entity directions: <strong>{cross['donor_answers']}/{cross['count']}</strong>. This exact split supports a semantic confound in the diagnostic, but it does not rescore R4.</p>
<p>A corrected protocol must be a prospective R5 lockbox. This frozen-inference experiment performed no training, so no loss curve exists.</p>
</body></html>"""
    _atomic_write(output_dir / "diagnosis.html", html_payload)

    manifest_paths = [
        output_dir / "diagnosis.json",
        output_dir / "diagnosis.md",
        output_dir / "diagnosis.html",
        output_dir / "diagnosis.csv",
        *(plots / name for name in PLOT_NAMES),
        *sorted(sources.iterdir()),
    ]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "files": [
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_paths
        ],
    }
    _atomic_write(
        output_dir / "sha256_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {**diagnosis, "output_dir": str(output_dir.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an audited R4 BH1 last-effective state-swap failure diagnosis"
    )
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--score", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = render_diagnosis(
            predictions_a=args.predictions_a,
            predictions_b=args.predictions_b,
            score_path=args.score,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
