from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import __version__ as PILLOW_VERSION
from torch import Tensor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.data import REVERSE_CYCLIC4, Episode, QuerySpec, permute_query, read_jsonl  # noqa: E402
from vision_memory.dreamlite import freeze_module  # noqa: E402
from vision_memory.reader import R3_QWEN_READER_RESIZE_CONTRACT, qwen3vl_choice_nll  # noqa: E402
from vision_memory.repro import (  # noqa: E402
    assert_no_frozen_parameter_grads,
    configure_strict_cuda_determinism,
    cuda_peak_memory_report,
    emit_json_report,
    probe_provenance,
    reset_cuda_peak_memory,
)
from vision_memory.teacher import (  # noqa: E402
    TRAIN_SPLIT,
    FixedFontContract,
    FullStateCardRenderer,
    SemanticState,
    TeacherCacheManifest,
    TeacherTransitionRecord,
    file_sha256,
    load_teacher_manifest,
    make_disk_teacher_provider,
    validate_teacher_sidecar,
)
from vision_memory.training import format_mcq_query  # noqa: E402


RAW_SIDECAR_SCHEMA = "vlm.r3.teacher_transition.v1"
LOCKED_FONT_SHA256 = "3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22"
LOCKED_FONT_ID = "DejaVuSans-2.37-embedded"
PREREGISTRATION_PATH = ROOT / "configs" / "experiments" / "r3_preregistration.json"
MODALITIES = ("raw_state_card", "vae_decoded_teacher_card")
MACRO_ACCURACY_THRESHOLD = 0.95
HELDOUT_TEMPLATE_THRESHOLD = 0.90
POSITION_CORRECT_THRESHOLD = 15
ROTATION_AGREEMENT_THRESHOLD = 0.99


@dataclass(frozen=True)
class RawTransition:
    episode_id: str
    turn_id: int
    event_kind: str
    before_state: SemanticState
    after_state: SemanticState


@dataclass(frozen=True)
class DelayedQueryCase:
    episode_id: str
    template_id: str
    query: QuerySpec
    final_state: SemanticState


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_mapping(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(f"R3 preregistration is missing {'.'.join(keys)}.")
        current = current[key]
    if not isinstance(current, Mapping):
        raise ValueError(f"R3 preregistration field {'.'.join(keys)} must be an object.")
    return current


def audit_preregistered_inputs(
    *,
    preregistration_path: Path,
    gate_jsonl: Path,
    raw_sidecar: Path,
    teacher_manifest_path: Path,
    manifest: TeacherCacheManifest,
    font_path: Path,
    reader_path: Path,
) -> dict[str, Any]:
    """Bind T0 to the prospective R3 data/cache/model contract in Git."""

    value = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema") != "vision_memory.r3-preregistration.v1":
        raise ValueError("Teacher T0 requires the locked R3 preregistration schema.")
    transition16 = _nested_mapping(value, "micro_data", "transition16")
    teacher_contract = _nested_mapping(value, "teacher_contract")
    cache_locks = _nested_mapping(value, "teacher_contract", "cache_manifest_sha256")
    reader_lock = _nested_mapping(value, "models", "reader")
    reader_marker = reader_path / ".locked_revision"
    observed = {
        "gate_jsonl_sha256": sha256_file(gate_jsonl),
        "raw_sidecar_sha256": sha256_file(raw_sidecar),
        "teacher_manifest_sha256": sha256_file(teacher_manifest_path),
        "font_sha256": sha256_file(font_path),
        "renderer_contract_sha256": manifest.renderer_contract_sha256,
        "teacher_contract_sha256": manifest.teacher_contract_sha256,
        "reader_revision": reader_marker.read_text(encoding="utf-8").strip() if reader_marker.is_file() else None,
        "pillow_version": PILLOW_VERSION,
    }
    expected = {
        "gate_jsonl_sha256": transition16.get("gate_sha256"),
        "raw_sidecar_sha256": transition16.get("raw_teacher_sidecar_sha256"),
        "teacher_manifest_sha256": cache_locks.get("transition16"),
        "font_sha256": teacher_contract.get("font_sha256"),
        "renderer_contract_sha256": teacher_contract.get("renderer_contract_sha256"),
        "teacher_contract_sha256": teacher_contract.get("build_contract_sha256"),
        "reader_revision": reader_lock.get("revision"),
        "pillow_version": teacher_contract.get("pillow_version"),
    }
    checks = {
        name: isinstance(expected_value, str) and bool(expected_value) and observed[name] == expected_value
        for name, expected_value in expected.items()
    }
    return {
        "passed": all(checks.values()),
        "preregistration": {
            "path": str(preregistration_path),
            "sha256": sha256_file(preregistration_path),
        },
        "checks": checks,
        "expected": expected,
        "observed": observed,
    }


def parse_raw_sidecar_records(values: Iterable[Mapping[str, Any]]) -> tuple[RawTransition, ...]:
    expected = {
        "schema_version",
        "split",
        "episode_id",
        "turn_id",
        "event_kind",
        "before_state",
        "after_state",
    }
    records: list[RawTransition] = []
    seen: set[tuple[str, int]] = set()
    for index, value in enumerate(values, 1):
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError(f"Raw teacher sidecar record {index} differs from its locked schema.")
        if value["schema_version"] != RAW_SIDECAR_SCHEMA or value["split"] != TRAIN_SPLIT:
            raise ValueError("Raw R3 teacher transitions are schema-locked and train-only.")
        episode_id = value["episode_id"]
        turn_id = value["turn_id"]
        event_kind = value["event_kind"]
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("Raw teacher transition episode_id must be non-empty.")
        if isinstance(turn_id, bool) or not isinstance(turn_id, int) or turn_id < 0:
            raise ValueError("Raw teacher transition turn_id must be a non-negative integer.")
        if event_kind not in {"set", "overwrite", "clear", "noop"}:
            raise ValueError("Raw teacher transition has an unsupported event_kind.")
        identity = (episode_id, turn_id)
        if identity in seen:
            raise ValueError(f"Duplicate raw teacher transition identity: {identity!r}.")
        seen.add(identity)
        before = SemanticState.from_dict(value["before_state"])
        after = SemanticState.from_dict(value["after_state"])
        if event_kind == "noop" and before.state_id != after.state_id:
            raise ValueError("A raw no-op transition changed semantic-state identity.")
        records.append(
            RawTransition(
                episode_id=episode_id,
                turn_id=turn_id,
                event_kind=event_kind,
                before_state=before,
                after_state=after,
            )
        )
    if not records:
        raise ValueError("Raw teacher sidecar is empty.")
    return tuple(sorted(records, key=lambda record: (record.episode_id, record.turn_id)))


def read_raw_sidecar(path: Path) -> tuple[RawTransition, ...]:
    values: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Raw teacher sidecar contains a blank line at {line_number}.")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"Raw teacher sidecar line {line_number} is not an object.")
            values.append(value)
    return parse_raw_sidecar_records(values)


def semantic_state_registry(records: Sequence[RawTransition]) -> dict[str, SemanticState]:
    registry: dict[str, SemanticState] = {}
    for record in records:
        for state in (record.before_state, record.after_state):
            existing = registry.get(state.state_id)
            if existing is not None and existing.canonical_bytes != state.canonical_bytes:
                raise RuntimeError("Semantic state_id collision detected in the raw sidecar.")
            registry[state.state_id] = state
    return dict(sorted(registry.items()))


def align_transition16_delayed_queries(
    episodes: Sequence[Episode],
    transitions: Sequence[RawTransition],
) -> tuple[DelayedQueryCase, ...]:
    if len(episodes) != 16:
        raise ValueError(f"Teacher T0 requires exactly 16 Transition16 gate episodes, got {len(episodes)}.")
    if len(transitions) != 28:
        raise ValueError(f"Teacher T0 requires exactly 28 Transition16 raw transitions, got {len(transitions)}.")
    by_episode: dict[str, list[RawTransition]] = defaultdict(list)
    for transition in transitions:
        by_episode[transition.episode_id].append(transition)
    episode_ids = [episode.episode_id for episode in episodes]
    expected_episode_ids = {
        f"r3-transition-{kind}-{read_form}-r{replica}"
        for kind in ("set", "overwrite", "clear", "noop")
        for read_form in ("separate", "mixed")
        for replica in range(2)
    }
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Transition16 gate JSONL contains duplicate episode IDs.")
    if set(episode_ids) != expected_episode_ids:
        raise ValueError("Transition16 gate JSONL does not contain the locked 4x2x2 episode grid.")
    if set(by_episode) != set(episode_ids):
        raise ValueError("Gate episode IDs and raw teacher sidecar episode IDs differ.")
    event_kind_counts = {
        kind: sum(transition.event_kind == kind for transition in transitions)
        for kind in ("set", "overwrite", "clear", "noop")
    }
    if event_kind_counts != {"set": 16, "overwrite": 4, "clear": 4, "noop": 4}:
        raise ValueError(f"Transition16 raw event-kind counts drifted: {event_kind_counts}.")

    aligned: list[DelayedQueryCase] = []
    for episode in episodes:
        if episode.split != TRAIN_SPLIT:
            raise ValueError("Teacher T0 only accepts the train-scoped Transition16 micro gate.")
        if not episode.episode_id.startswith("r3-transition-"):
            raise ValueError(f"Unexpected non-Transition16 episode: {episode.episode_id!r}.")
        read_form = "mixed" if "-mixed-" in episode.episode_id else "separate"
        if episode.template_id != f"r3-transition-{read_form}-gate-b":
            raise ValueError("Teacher T0 requires the held-out gate-b query templates.")
        updater_turns = [turn for turn in episode.turns if turn.calls_updater]
        episode_transitions = sorted(by_episode[episode.episode_id], key=lambda record: record.turn_id)
        if len(updater_turns) != len(episode_transitions):
            raise ValueError(f"Updater/sidecar transition count mismatch for {episode.episode_id!r}.")
        for ordinal, (turn, transition) in enumerate(zip(updater_turns, episode_transitions, strict=True)):
            if transition.turn_id != ordinal:
                raise ValueError(f"Non-contiguous sidecar turn IDs for {episode.episode_id!r}.")
            if turn.event_kind is None or transition.event_kind != turn.event_kind.value:
                raise ValueError(f"Event-kind mismatch for {episode.episode_id!r} turn {ordinal}.")
        for previous, current in zip(episode_transitions, episode_transitions[1:]):
            if previous.after_state.state_id != current.before_state.state_id:
                raise ValueError(f"Semantic-state continuity failed for {episode.episode_id!r}.")

        reader_turns = [turn for turn in episode.turns if turn.calls_reader]
        if not reader_turns or reader_turns[-1].query is None:
            raise ValueError(f"Episode {episode.episode_id!r} lacks a final delayed query.")
        delayed_query = reader_turns[-1].query
        if delayed_query.comparison_id is None or not delayed_query.comparison_id.endswith(":delayed"):
            raise ValueError(f"Episode {episode.episode_id!r} final query is not the locked delayed query.")
        if episode.turns[-1].query != delayed_query:
            raise ValueError(f"Episode {episode.episode_id!r} delayed query is not the final turn.")
        aligned.append(
            DelayedQueryCase(
                episode_id=episode.episode_id,
                template_id=episode.template_id,
                query=delayed_query,
                final_state=episode_transitions[-1].after_state,
            )
        )
    return tuple(sorted(aligned, key=lambda item: item.episode_id))


def manifest_transition_records(
    transitions: Sequence[RawTransition],
    manifest: TeacherCacheManifest,
) -> tuple[TeacherTransitionRecord, ...]:
    by_state = manifest.by_state_id
    records: list[TeacherTransitionRecord] = []
    for transition in transitions:
        before = by_state.get(transition.before_state.state_id)
        after = by_state.get(transition.after_state.state_id)
        if before is None or after is None:
            raise ValueError("Raw transition references a semantic state absent from the teacher cache.")
        if before.semantic_state_sha256 != transition.before_state.canonical_sha256:
            raise ValueError("Cached before-state semantic SHA does not match the raw sidecar.")
        if after.semantic_state_sha256 != transition.after_state.canonical_sha256:
            raise ValueError("Cached after-state semantic SHA does not match the raw sidecar.")
        records.append(
            TeacherTransitionRecord(
                episode_id=transition.episode_id,
                turn_id=transition.turn_id,
                before_state_id=transition.before_state.state_id,
                after_state_id=transition.after_state.state_id,
                event_kind=transition.event_kind,
                teacher_key=after.teacher_key,
            )
        )
    return validate_teacher_sidecar(records, manifest=manifest)


def audit_cache_identity_and_paths(
    *,
    registry: Mapping[str, SemanticState],
    transitions: Sequence[RawTransition],
    manifest: TeacherCacheManifest,
) -> dict[str, Any]:
    records = tuple(manifest.records)
    manifest_by_state = manifest.by_state_id
    state_ids = [record.state_id for record in records]
    teacher_keys = [record.teacher_key for record in records]
    semantic_hashes = [record.semantic_state_sha256 for record in records]
    artifact_paths = [
        specification.relative_path
        for record in records
        for specification in (record.image, record.latent, record.feature)
    ]
    raw_coverage = set(registry).issubset(manifest_by_state)
    semantic_hashes_match = raw_coverage and all(
        manifest_by_state[state_id].semantic_state_sha256 == state.canonical_sha256
        for state_id, state in registry.items()
    )

    reached_by: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    for transition in transitions:
        reached_by[transition.after_state.state_id].add(
            (transition.episode_id, transition.turn_id, transition.event_kind)
        )
    multi_path_states = {
        state_id: sorted(paths)
        for state_id, paths in reached_by.items()
        if len({episode_id for episode_id, _turn_id, _kind in paths}) > 1
    }
    noop_records = [transition for transition in transitions if transition.event_kind == "noop"]
    noop_identity_preserved = bool(noop_records) and all(
        transition.before_state.state_id == transition.after_state.state_id for transition in noop_records
    )
    path_invariance = bool(multi_path_states) and all(
        state_id in manifest_by_state and bool(manifest_by_state[state_id].teacher_key)
        for state_id in multi_path_states
    )
    checks = {
        "raw_state_cache_coverage": raw_coverage,
        "semantic_hashes_match": semantic_hashes_match,
        "unique_state_ids": len(state_ids) == len(set(state_ids)),
        "unique_teacher_keys": len(teacher_keys) == len(set(teacher_keys)),
        "unique_semantic_hashes": len(semantic_hashes) == len(set(semantic_hashes)),
        "unique_artifact_paths": len(artifact_paths) == len(set(artifact_paths)),
        "path_invariance": path_invariance,
        "noop_identity": noop_identity_preserved,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "manifest_state_count": len(records),
        "raw_state_count": len(registry),
        "artifact_path_count": len(artifact_paths),
        "multi_path_state_count": len(multi_path_states),
        "multi_path_states": multi_path_states,
        "noop_transition_count": len(noop_records),
    }


def teacher_identity_from_context(context: Mapping[str, Any]) -> str:
    """Select only the semantic after-state at the privileged teacher boundary."""

    if "after_state" not in context:
        raise ValueError("Teacher identity context requires after_state.")
    return SemanticState.from_dict(context["after_state"]).state_id


def audit_identity_mutations(state: SemanticState) -> dict[str, Any]:
    base = {
        "after_state": state.to_dict(),
        "episode_id": "original-episode",
        "query": "Which option is correct?",
        "choices": ["a", "b", "c", "d"],
        "future": ["future event one"],
        "target": "b",
    }
    mutations: dict[str, dict[str, Any]] = {}
    query_mutated = copy.deepcopy(base)
    query_mutated["query"] = "A completely different query"
    mutations["query_mutated"] = query_mutated
    choices_mutated = copy.deepcopy(base)
    choices_mutated["choices"] = list(reversed(choices_mutated["choices"]))
    mutations["choices_reversed"] = choices_mutated
    episode_mutated = copy.deepcopy(base)
    episode_mutated["episode_id"] = "different-episode-and-path"
    mutations["episode_mutated"] = episode_mutated
    future_mutated = copy.deepcopy(base)
    future_mutated["future"] = ["unobserved future event", "another future event"]
    mutations["future_mutated"] = future_mutated
    target_deleted = copy.deepcopy(base)
    del target_deleted["target"]
    mutations["target_deleted"] = target_deleted
    supervision_deleted = {"after_state": copy.deepcopy(base["after_state"])}
    mutations["all_query_supervision_deleted"] = supervision_deleted

    base_identity = teacher_identity_from_context(base)
    identities = {name: teacher_identity_from_context(value) for name, value in mutations.items()}
    state_boundary_rejects_supervision = False
    contaminated_state = copy.deepcopy(state.to_dict())
    contaminated_state["query_text"] = "forbidden"
    try:
        SemanticState.from_dict(contaminated_state)
    except ValueError:
        state_boundary_rejects_supervision = True
    passed = all(identity == base_identity for identity in identities.values()) and state_boundary_rejects_supervision
    return {
        "passed": passed,
        "base_state_id": base_identity,
        "mutation_state_ids": identities,
        "mutations_checked": tuple(mutations),
        "state_boundary_rejects_supervision": state_boundary_rejects_supervision,
    }


def audit_cross_split_fail_closed(provider: Any, state_id: str) -> dict[str, Any]:
    refused: dict[str, bool] = {}
    for split in ("dev", "test", "test_ood"):
        try:
            provider.get(state_id, split=split)
        except ValueError:
            refused[split] = True
        else:
            refused[split] = False
    return {"passed": all(refused.values()), "refused_splits": refused}


def reverse_cyclic_query_views(query: QuerySpec) -> tuple[QuerySpec, ...]:
    views = tuple(permute_query(query, permutation) for permutation in REVERSE_CYCLIC4)
    if len(views) != 4 or sorted(view.target_index for view in views) != [0, 1, 2, 3]:
        raise RuntimeError("Reverse-cyclic query views failed to cover all four target positions.")
    if any(view.target != query.target for view in views):
        raise RuntimeError("Reverse-cyclic query views changed the semantic target text.")
    return views


def score_upper_bound_predictions(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_episode_ids: Sequence[str],
) -> dict[str, Any]:
    expected_episodes = set(expected_episode_ids)
    if len(expected_episodes) != 16:
        raise ValueError("Teacher T0 metrics require exactly 16 expected episode IDs.")
    by_modality: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str, int]] = set()
    for row in rows:
        modality = str(row.get("modality"))
        if modality not in MODALITIES:
            raise ValueError(f"Unexpected teacher-card modality: {modality!r}.")
        episode_id = str(row.get("episode_id"))
        view_index = int(row.get("choice_view_index", -1))
        identity = (modality, episode_id, view_index)
        if identity in identities:
            raise ValueError(f"Duplicate teacher T0 prediction identity: {identity!r}.")
        identities.add(identity)
        if episode_id not in expected_episodes or not 0 <= view_index < 4:
            raise ValueError("Teacher T0 prediction references an unexpected episode or view.")
        if row.get("choice_view_family") != "reverse-cyclic4":
            raise ValueError("Teacher T0 predictions must use reverse-cyclic4 views.")
        choices = tuple(row.get("choices", ()))
        target_index = int(row.get("target_index", -1))
        prediction_index = int(row.get("prediction_index", -1))
        if len(choices) != 4 or len(set(choices)) != 4:
            raise ValueError("Teacher T0 prediction choices must contain four distinct values.")
        if not 0 <= target_index < 4 or choices[target_index] != row.get("target_text"):
            raise ValueError("Teacher T0 target index/text mapping is inconsistent.")
        if not 0 <= prediction_index < 4 or choices[prediction_index] != row.get("prediction_text"):
            raise ValueError("Teacher T0 prediction index/text mapping is inconsistent.")
        by_modality[modality].append(row)

    modality_reports: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        values = by_modality.get(modality, [])
        if len(values) != 64:
            raise ValueError(f"Teacher T0 requires 64 {modality} predictions, got {len(values)}.")
        by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        by_position: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        by_template: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in values:
            by_episode[str(row["episode_id"])].append(row)
            by_position[int(row["target_index"])].append(row)
            by_template[str(row["template_id"])].append(row)
        if set(by_episode) != expected_episodes or any(len(group) != 4 for group in by_episode.values()):
            raise ValueError(f"Teacher T0 {modality} rows do not contain four views per episode.")
        if set(by_position) != {0, 1, 2, 3} or any(len(group) != 16 for group in by_position.values()):
            raise ValueError(f"Teacher T0 {modality} target positions are not balanced 16/16/16/16.")
        if any("gate-b" not in template_id for template_id in by_template):
            raise ValueError("Teacher T0 received a non-held-out query template.")

        def correct(row: Mapping[str, Any]) -> bool:
            return int(row["prediction_index"]) == int(row["target_index"])

        episode_accuracies = {
            episode_id: sum(correct(row) for row in group) / len(group)
            for episode_id, group in sorted(by_episode.items())
        }
        macro_accuracy = sum(episode_accuracies.values()) / len(episode_accuracies)
        template_accuracy = {
            template_id: sum(correct(row) for row in group) / len(group)
            for template_id, group in sorted(by_template.items())
        }
        positions = {
            str(position): {
                "correct": sum(correct(row) for row in group),
                "count": len(group),
            }
            for position, group in sorted(by_position.items())
        }
        rotation_consistent = {
            episode_id: len({str(row["prediction_text"]) for row in group}) == 1
            for episode_id, group in sorted(by_episode.items())
        }
        rotation_agreement = sum(rotation_consistent.values()) / len(rotation_consistent)
        checks = {
            "macro_accuracy": macro_accuracy >= MACRO_ACCURACY_THRESHOLD,
            "heldout_templates": all(value >= HELDOUT_TEMPLATE_THRESHOLD for value in template_accuracy.values()),
            "positions": all(summary["correct"] >= POSITION_CORRECT_THRESHOLD for summary in positions.values()),
            "predicted_text_rotation_agreement": rotation_agreement >= ROTATION_AGREEMENT_THRESHOLD,
        }
        modality_reports[modality] = {
            "passed": all(checks.values()),
            "checks": checks,
            "macro_accuracy": macro_accuracy,
            "correct": sum(correct(row) for row in values),
            "count": len(values),
            "episode_accuracy": episode_accuracies,
            "heldout_template_accuracy": template_accuracy,
            "positions": positions,
            "predicted_text_rotation_agreement": rotation_agreement,
            "rotation_consistent_by_episode": rotation_consistent,
        }

    paired: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        paired[(str(row["episode_id"]), int(row["choice_view_index"]))][str(row["modality"])] = row
    if len(paired) != 64 or any(set(pair) != set(MODALITIES) for pair in paired.values()):
        raise ValueError("Raw/decoded teacher predictions do not form 64 complete modality pairs.")
    paired_fields = ("template_id", "choices", "target_index", "target_text")
    if any(
        any(pair[MODALITIES[0]].get(field) != pair[MODALITIES[1]].get(field) for field in paired_fields)
        for pair in paired.values()
    ):
        raise ValueError("Raw/decoded teacher prediction pairs changed the query, choices, or semantic target.")
    cross_modality_agreements = sum(
        pair[MODALITIES[0]]["prediction_text"] == pair[MODALITIES[1]]["prediction_text"] for pair in paired.values()
    )
    return {
        "passed": all(report["passed"] for report in modality_reports.values()),
        "thresholds": {
            "macro_accuracy": MACRO_ACCURACY_THRESHOLD,
            "heldout_template_accuracy": HELDOUT_TEMPLATE_THRESHOLD,
            "position_correct": f">={POSITION_CORRECT_THRESHOLD}/16",
            "predicted_text_rotation_agreement": ROTATION_AGREEMENT_THRESHOLD,
        },
        "modalities": modality_reports,
        "cross_modality_predicted_text_agreement": {
            "agreements": cross_modality_agreements,
            "count": len(paired),
            "rate": cross_modality_agreements / len(paired),
            "thresholded": False,
        },
    }


def run_real_qwen_upper_bound(
    *,
    cases: Sequence[DelayedQueryCase],
    raw_cards: Mapping[str, Tensor],
    decoded_cards: Mapping[str, Tensor],
    model: Any,
    processor: Any,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for case in cases:
        state_id = case.final_state.state_id
        images = {
            "raw_state_card": raw_cards[state_id],
            "vae_decoded_teacher_card": decoded_cards[state_id],
        }
        for modality, batched_image in images.items():
            if tuple(batched_image.shape) != (1, 3, 1024, 1024):
                raise ValueError(f"{modality} image violated the locked [1,3,1024,1024] shape.")
            for view_index, view in enumerate(reverse_cyclic_query_views(case.query)):
                formatted_query = format_mcq_query(view.text, view.choices)
                result = qwen3vl_choice_nll(
                    model=model,
                    processor=processor,
                    image=batched_image[0].to(device=device),
                    query=formatted_query,
                    choices=view.choices,
                    device=device,
                    reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT,
                    deterministic_ce=True,
                )
                if len(result.mean_nll) != 4 or not all(math.isfinite(float(value)) for value in result.mean_nll):
                    raise RuntimeError(
                        f"{modality} scorer produced a malformed or non-finite NLL vector for {case.episode_id}."
                    )
                predicted_index = int(result.predicted_index)
                predictions.append(
                    {
                        "episode_id": case.episode_id,
                        "template_id": case.template_id,
                        "state_id": state_id,
                        "modality": modality,
                        "choice_view_family": "reverse-cyclic4",
                        "choice_view_index": view_index,
                        "permutation": list(REVERSE_CYCLIC4[view_index]),
                        "choices": list(view.choices),
                        "target_index": view.target_index,
                        "target_text": view.target,
                        "prediction_index": predicted_index,
                        "prediction_text": view.choices[predicted_index],
                        "choice_mean_nll": list(result.mean_nll),
                    }
                )
    metrics = score_upper_bound_predictions(
        predictions,
        expected_episode_ids=[case.episode_id for case in cases],
    )
    return predictions, metrics


def contract_exit_code(report: Mapping[str, Any]) -> int:
    return 0 if report.get("passed") is True else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher T0 real-Qwen integrity and upper-bound probe")
    parser.add_argument("--gate-jsonl", type=Path, required=True)
    parser.add_argument("--raw-sidecar", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=ROOT / "assets" / "fonts" / "DejaVuSans.ttf")
    parser.add_argument("--reader-device", default="cuda:0")
    parser.add_argument("--allow-small-gpu", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_report: dict[str, Any] = {
        "schema_version": 1,
        "probe": "teacher_t0_real_qwen_integrity_upper_bound",
        "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
        "passed": False,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the real Qwen teacher T0 probe.")
        device = torch.device(args.reader_device)
        if device.type != "cuda":
            raise ValueError("Teacher T0 requires a CUDA Reader device.")
        memory_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
        if memory_gib < 16 and not args.allow_small_gpu:
            raise RuntimeError(f"Only {memory_gib:.1f} GiB VRAM detected; run teacher T0 on the cluster.")

        strict_determinism = configure_strict_cuda_determinism(seed=0)

        if file_sha256(args.font) != LOCKED_FONT_SHA256:
            raise RuntimeError("Embedded DejaVuSans.ttf differs from the locked R3 font SHA256.")
        manifest = load_teacher_manifest(args.teacher_manifest)
        preregistered_inputs = audit_preregistered_inputs(
            preregistration_path=PREREGISTRATION_PATH,
            gate_jsonl=args.gate_jsonl,
            raw_sidecar=args.raw_sidecar,
            teacher_manifest_path=args.teacher_manifest,
            manifest=manifest,
            font_path=args.font,
            reader_path=args.reader,
        )
        if not preregistered_inputs["passed"]:
            raise RuntimeError(
                "Teacher T0 inputs differ from the prospective R3 locks: "
                f"{preregistered_inputs['checks']}"
            )
        font = FixedFontContract(
            font_id=LOCKED_FONT_ID,
            path=args.font,
            sha256=LOCKED_FONT_SHA256,
            pillow_version=PILLOW_VERSION,
        )
        renderer = FullStateCardRenderer(font)
        if renderer.contract_sha256 != manifest.renderer_contract_sha256:
            raise RuntimeError("Teacher cache renderer contract does not match the fixed embedded font/runtime.")

        episodes = read_jsonl(args.gate_jsonl)
        transitions = read_raw_sidecar(args.raw_sidecar)
        registry = semantic_state_registry(transitions)
        cases = align_transition16_delayed_queries(episodes, transitions)
        validated_transitions = manifest_transition_records(transitions, manifest)
        cache_integrity = audit_cache_identity_and_paths(
            registry=registry,
            transitions=transitions,
            manifest=manifest,
        )
        if not cache_integrity["passed"]:
            raise RuntimeError(f"Teacher cache identity/path audit failed: {cache_integrity['checks']}")

        provider = make_disk_teacher_provider(
            args.teacher_manifest,
            cache_root=args.teacher_cache_root,
        )
        final_state_ids = {case.final_state.state_id for case in cases}
        decoded_cards: dict[str, Tensor] = {}
        for state_id in registry:
            teacher = provider.get(state_id, split=TRAIN_SPLIT)
            if state_id in final_state_ids:
                decoded_cards[state_id] = teacher.image
        cross_split = audit_cross_split_fail_closed(provider, next(iter(sorted(final_state_ids))))
        if not cross_split["passed"]:
            raise RuntimeError("Teacher provider accepted a non-train split.")
        raw_cards = {
            state_id: renderer.render_tensor(registry[state_id]).detach().cpu().contiguous()
            for state_id in sorted(final_state_ids)
        }
        mutation_reports = {state_id: audit_identity_mutations(registry[state_id]) for state_id in sorted(registry)}
        if not all(report["passed"] for report in mutation_reports.values()):
            raise RuntimeError("Query/future/target mutation changed teacher identity.")

        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        reset_cuda_peak_memory([device])
        processor = AutoProcessor.from_pretrained(
            args.reader,
            local_files_only=True,
            use_fast=True,
            min_pixels=256 * 256,
            max_pixels=256 * 256,
        )
        processor_name = type(processor.image_processor).__name__
        if "Fast" not in processor_name:
            raise RuntimeError(f"Expected a fast tensor image processor, got {processor_name}")
        reader = Qwen3VLForConditionalGeneration.from_pretrained(
            args.reader,
            local_files_only=True,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        freeze_module(reader)
        reader.eval()
        reader.config.use_cache = False

        predictions, upper_bound = run_real_qwen_upper_bound(
            cases=cases,
            raw_cards=raw_cards,
            decoded_cards=decoded_cards,
            model=reader,
            processor=processor,
            device=device,
        )
        report = {
            **base_report,
            "passed": bool(upper_bound["passed"]),
            "inputs": {
                "gate_jsonl": {"path": str(args.gate_jsonl), "sha256": sha256_file(args.gate_jsonl)},
                "raw_sidecar": {"path": str(args.raw_sidecar), "sha256": sha256_file(args.raw_sidecar)},
                "teacher_manifest": {
                    "path": str(args.teacher_manifest),
                    "sha256": sha256_file(args.teacher_manifest),
                    "canonical_sha256": manifest.canonical_sha256,
                },
                "font": {"path": str(args.font), "sha256": LOCKED_FONT_SHA256},
            },
            "counts": {
                "episodes": len(episodes),
                "raw_transitions": len(transitions),
                "validated_transitions": len(validated_transitions),
                "semantic_states": len(registry),
                "final_states": len(final_state_ids),
                "predictions": len(predictions),
            },
            "renderer_contract_sha256": renderer.contract_sha256,
            "teacher_contract_sha256": manifest.teacher_contract_sha256,
            "preregistered_inputs": preregistered_inputs,
            "cache_integrity": cache_integrity,
            "cross_split_fail_closed": cross_split,
            "identity_mutations": mutation_reports,
            "upper_bound": upper_bound,
            "predictions": predictions,
            "reader_processor": processor_name,
            "reader_resize_contract": R3_QWEN_READER_RESIZE_CONTRACT,
            "reader_dtype": str(dtype),
            "reader_device": str(device),
            "strict_determinism": strict_determinism,
            "frozen_gradients": assert_no_frozen_parameter_grads(
                {"reader": reader},
                fully_frozen={"reader"},
            ),
            "cuda_peak_memory": cuda_peak_memory_report([device]),
            "provenance": probe_provenance(
                root=ROOT,
                arguments=args,
                models={"reader": args.reader},
            ),
        }
    except Exception as error:  # noqa: BLE001 - all probe failures must still produce JSON
        report = {
            **base_report,
            "error": {"type": type(error).__name__, "message": str(error)},
            "provenance": probe_provenance(
                root=ROOT,
                arguments=args,
                models={"reader": args.reader},
            ),
        }
    emit_json_report(report, args.output_json)
    return contract_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
