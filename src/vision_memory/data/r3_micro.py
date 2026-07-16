"""Prospectively locked R3 Set8 and Transition16 micro-overfit suites.

The model-visible episode JSONL intentionally contains no semantic ledger.
Oracle state snapshots live in a separate train-only sidecar used exclusively
by the privileged teacher-assisted lineage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schema import DistractorVariant, Episode, EventKind, QuerySpec, Turn, TurnType, write_jsonl


R3_MICRO_SEED = 2026
R3_STATE_SCHEMA = "vlm.semantic_state.v1"
SET8_VALUES = ("red", "blue", "green", "yellow")
TRANSITION_VALUES = ("red", "blue", "green", "no active preference")

CYCLIC4: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
)
REVERSE_CYCLIC4: tuple[tuple[int, int, int, int], ...] = (
    (3, 2, 1, 0),
    (2, 1, 0, 3),
    (1, 0, 3, 2),
    (0, 3, 2, 1),
)


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def permutation_family_sha256(family: Sequence[Sequence[int]]) -> str:
    return hashlib.sha256(_stable_json_bytes([list(item) for item in family])).hexdigest()


def permute_query(query: QuerySpec, permutation: Sequence[int]) -> QuerySpec:
    """Return one choice view while preserving the semantic target text."""

    order = tuple(int(index) for index in permutation)
    if len(order) != 4 or set(order) != {0, 1, 2, 3}:
        raise ValueError("A choice permutation must contain each index in [0, 3] exactly once.")
    target = query.target
    choices = tuple(query.choices[index] for index in order)
    return replace(query, choices=choices, target_index=choices.index(target))


def episode_choice_view(episode: Episode, permutation: Sequence[int]) -> Episode:
    """Apply a Reader-only choice view; events and query text remain unchanged."""

    turns = tuple(
        replace(turn, query=permute_query(turn.query, permutation)) if turn.query is not None else turn
        for turn in episode.turns
    )
    return replace(episode, turns=turns)


def presentation_permutation(presentation_index: int, comparison_id: str) -> tuple[int, int, int, int]:
    """Balanced cyclic training schedule with an answer-agnostic phase offset."""

    if presentation_index < 0:
        raise ValueError("presentation_index must be non-negative")
    if not comparison_id:
        raise ValueError("comparison_id must be non-empty")
    phase = int.from_bytes(hashlib.sha256(comparison_id.encode("utf-8")).digest()[:2], "big") % 4
    return CYCLIC4[(presentation_index + phase) % 4]


def _query(
    entity: str,
    target: str,
    *,
    choices: tuple[str, str, str, str],
    template: str,
    comparison_id: str,
) -> QuerySpec:
    if target not in choices:
        raise ValueError(f"Unsupported R3 target: {target!r}")
    if template == "train-a":
        text = f"Memory memo: Which color is currently preferred for {entity}? Choose exactly one option."
    elif template == "gate-b":
        text = f"Preference check: What color should be selected now for {entity}? Choose exactly one option."
    else:
        raise ValueError(f"Unknown query template: {template}")
    return QuerySpec(
        text=text,
        choices=choices,
        target_index=choices.index(target),
        comparison_id=comparison_id,
    )


def _event(entity: str, kind: EventKind, value: str | None, lexical_replica: int) -> str:
    if kind is EventKind.SET:
        forms = (
            f"Memory memo: For {entity}, remember that the preferred color is {value}.",
            f"Memory memo: Save {value} as the current color preference for {entity}.",
        )
    elif kind is EventKind.OVERWRITE:
        forms = (
            f"Memory memo: Replace the earlier color preference for {entity} with {value}.",
            f"Memory memo: The color preference for {entity} is now {value}, not the previous value.",
        )
    elif kind is EventKind.CLEAR:
        forms = (
            f"Memory memo: Clear the saved color preference for {entity}; none is active now.",
            f"Memory memo: {entity} no longer has an active color preference.",
        )
    elif kind is EventKind.NOOP:
        forms = (
            "Memory memo: An unrelated hallway clock was repaired yesterday.",
            "Memory memo: An unrelated delivery arrived at noon.",
        )
    else:  # pragma: no cover - exhaustive enum
        raise ValueError(f"Unsupported event kind: {kind}")
    return forms[lexical_replica % 2]


def _state(entity_id: str, entity_text: str, *, status: str, value: str | None) -> dict[str, Any]:
    if status not in {"active", "cleared", "unset"}:
        raise ValueError(f"Unsupported state status: {status}")
    if status == "active" and value is None:
        raise ValueError("An active state requires a value.")
    if status != "active" and value is not None:
        raise ValueError("Cleared/unset states cannot contain a value.")
    return {
        "schema": R3_STATE_SCHEMA,
        "entries": [
            {
                "entity_id": entity_id,
                "entity_text": entity_text,
                "slot_id": "color",
                "slot_text": "color",
                "status": status,
                "value_id": value,
                "value_text": value,
            }
        ],
    }


def _sidecar_record(
    *,
    episode_id: str,
    turn_index: int,
    event_kind: EventKind,
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "vlm.r3.teacher_transition.v1",
        "split": "train",
        "episode_id": episode_id,
        "turn_id": turn_index,
        "event_kind": event_kind.value,
        "before_state": before,
        "after_state": after,
    }


@dataclass(frozen=True)
class R3MicroSuite:
    name: str
    train_episodes: tuple[Episode, ...]
    gate_episodes: tuple[Episode, ...]
    teacher_sidecar: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def _episode(
    *,
    episode_id: str,
    entity_id: str,
    entity_text: str,
    turns: Iterable[Turn],
    counterfactual_episode_id: str,
    pair_id: str,
    template_id: str,
    distractor_variant: DistractorVariant | None = None,
    distractor_pair_id: str | None = None,
    distractor_episode_id: str | None = None,
) -> Episode:
    return Episode(
        episode_id=episode_id,
        split="train",
        seed=R3_MICRO_SEED,
        entity_id=entity_id,
        entity_surface=entity_text,
        template_id=template_id,
        template_family="r3-micro",
        turns=tuple(turns),
        pair_id=pair_id,
        counterfactual_episode_id=counterfactual_episode_id,
        topic="color",
        distractor_variant=distractor_variant,
        distractor_pair_id=distractor_pair_id,
        distractor_episode_id=distractor_episode_id,
    )


def build_set8(*, seed: int = R3_MICRO_SEED) -> R3MicroSuite:
    if seed != R3_MICRO_SEED:
        raise ValueError(f"R3 Set8 is preregistered with seed {R3_MICRO_SEED}.")
    train: list[Episode] = []
    gate: list[Episode] = []
    sidecar: list[dict[str, Any]] = []
    active_values = SET8_VALUES
    for replica in range(2):
        entity_id = f"r3-set8-entity-{replica}"
        entity_text = ("the copper mug", "the linen notebook")[replica]
        for value_index, value in enumerate(active_values):
            episode_id = f"r3-set8-r{replica}-v{value_index}"
            mate_index = value_index ^ 1
            counterfactual_id = f"r3-set8-r{replica}-v{mate_index}"
            comparison_id = f"r3-set8-r{replica}:state-query"
            event_turn = Turn(
                TurnType.EVENT,
                EventKind.SET,
                _event(entity_text, EventKind.SET, value, replica),
            )
            train_query = Turn(
                TurnType.QUERY,
                query=_query(
                    entity_text,
                    value,
                    choices=SET8_VALUES,
                    template="train-a",
                    comparison_id=comparison_id,
                ),
            )
            gate_query = Turn(
                TurnType.QUERY,
                query=_query(
                    entity_text,
                    value,
                    choices=SET8_VALUES,
                    template="gate-b",
                    comparison_id=comparison_id,
                ),
            )
            common = {
                "episode_id": episode_id,
                "entity_id": entity_id,
                "entity_text": entity_text,
                "counterfactual_episode_id": counterfactual_id,
                "pair_id": f"r3-set8-r{replica}-pair-{value_index // 2}",
            }
            train.append(_episode(turns=(event_turn, train_query), template_id="r3-set8-train-a", **common))
            gate.append(_episode(turns=(event_turn, gate_query), template_id="r3-set8-gate-b", **common))
            before = _state(entity_id, entity_text, status="unset", value=None)
            after = _state(entity_id, entity_text, status="active", value=value)
            sidecar.append(
                _sidecar_record(
                    episode_id=episode_id,
                    turn_index=0,
                    event_kind=EventKind.SET,
                    before=before,
                    after=after,
                )
            )
    manifest = {
        "schema_version": "vlm.r3.micro.v1",
        "suite": "set8",
        "seed": seed,
        "semantic_history_count": len(train),
        "heldout_view_count": len(gate) * len(REVERSE_CYCLIC4),
        "train_query_template": "train-a",
        "gate_query_template": "gate-b",
        "train_permutation_family": [list(item) for item in CYCLIC4],
        "gate_permutation_family": [list(item) for item in REVERSE_CYCLIC4],
        "train_permutation_family_sha256": permutation_family_sha256(CYCLIC4),
        "gate_permutation_family_sha256": permutation_family_sha256(REVERSE_CYCLIC4),
        "max_presentations_per_state": 512,
        "evaluation_start": 64,
        "evaluation_interval": 32,
    }
    return R3MicroSuite("set8", tuple(train), tuple(gate), tuple(sidecar), manifest)


def _transition_target(kind: EventKind, replica: int) -> str:
    if kind in {EventKind.SET, EventKind.NOOP}:
        return ("red", "blue")[replica]
    if kind is EventKind.OVERWRITE:
        return ("blue", "green")[replica]
    if kind is EventKind.CLEAR:
        return "no active preference"
    raise ValueError(f"Unsupported terminal kind: {kind}")


def build_transition16(*, seed: int = R3_MICRO_SEED) -> R3MicroSuite:
    if seed != R3_MICRO_SEED:
        raise ValueError(f"R3 Transition16 is preregistered with seed {R3_MICRO_SEED}.")
    train: list[Episode] = []
    gate: list[Episode] = []
    sidecar: list[dict[str, Any]] = []
    kinds = (EventKind.SET, EventKind.OVERWRITE, EventKind.CLEAR, EventKind.NOOP)
    read_forms = ("separate", "mixed")
    for kind_index, terminal_kind in enumerate(kinds):
        for read_form in read_forms:
            for replica in range(2):
                entity_id = f"r3-transition-{read_form}-entity-{replica}"
                entity_text = (
                    f"the {read_form} amber lamp",
                    f"the {read_form} canvas backpack",
                )[replica]
                episode_id = f"r3-transition-{terminal_kind.value}-{read_form}-r{replica}"
                target = _transition_target(terminal_kind, replica)
                stale = ("green", "red")[replica]
                comparison_id = f"r3-transition-{read_form}-r{replica}:delayed"
                immediate_id = f"r3-transition-{read_form}-r{replica}:immediate"

                current_status = "unset"
                current_value: str | None = None
                event_specs: list[tuple[EventKind, str | None, bool]] = []
                if terminal_kind in {EventKind.OVERWRITE, EventKind.CLEAR, EventKind.NOOP}:
                    initial_value = target if terminal_kind is EventKind.NOOP else stale
                    event_specs.append((EventKind.SET, initial_value, False))
                final_value = None if terminal_kind in {EventKind.CLEAR, EventKind.NOOP} else target
                event_specs.append((terminal_kind, final_value, read_form == "mixed"))

                train_turns: list[Turn] = []
                gate_turns: list[Turn] = []
                for event_index, (event_kind, event_value, is_mixed) in enumerate(event_specs):
                    event_text = _event(entity_text, event_kind, event_value, replica)
                    before = _state(
                        entity_id,
                        entity_text,
                        status=current_status,
                        value=current_value,
                    )
                    if event_kind in {EventKind.SET, EventKind.OVERWRITE}:
                        current_status, current_value = "active", event_value
                    elif event_kind is EventKind.CLEAR:
                        current_status, current_value = "cleared", None
                    elif event_kind is EventKind.NOOP:
                        pass
                    after = _state(
                        entity_id,
                        entity_text,
                        status=current_status,
                        value=current_value,
                    )
                    sidecar.append(
                        _sidecar_record(
                            episode_id=episode_id,
                            turn_index=event_index,
                            event_kind=event_kind,
                            before=before,
                            after=after,
                        )
                    )
                    if is_mixed:
                        train_turns.append(
                            Turn(
                                TurnType.MIXED,
                                event_kind,
                                event_text,
                                _query(
                                    entity_text,
                                    target,
                                    choices=TRANSITION_VALUES,
                                    template="train-a",
                                    comparison_id=immediate_id,
                                ),
                            )
                        )
                        gate_turns.append(
                            Turn(
                                TurnType.MIXED,
                                event_kind,
                                event_text,
                                _query(
                                    entity_text,
                                    target,
                                    choices=TRANSITION_VALUES,
                                    template="gate-b",
                                    comparison_id=immediate_id,
                                ),
                            )
                        )
                    else:
                        turn = Turn(TurnType.EVENT, event_kind, event_text)
                        train_turns.append(turn)
                        gate_turns.append(turn)

                train_turns.append(
                    Turn(
                        TurnType.QUERY,
                        query=_query(
                            entity_text,
                            target,
                            choices=TRANSITION_VALUES,
                            template="train-a",
                            comparison_id=comparison_id,
                        ),
                    )
                )
                gate_turns.append(
                    Turn(
                        TurnType.QUERY,
                        query=_query(
                            entity_text,
                            target,
                            choices=TRANSITION_VALUES,
                            template="gate-b",
                            comparison_id=comparison_id,
                        ),
                    )
                )

                mate_kind = kinds[kind_index ^ 1]
                counterfactual_id = f"r3-transition-{mate_kind.value}-{read_form}-r{replica}"
                pair_id = f"r3-transition-{read_form}-r{replica}-pair-{kind_index // 2}"
                common = {
                    "episode_id": episode_id,
                    "entity_id": entity_id,
                    "entity_text": entity_text,
                    "counterfactual_episode_id": counterfactual_id,
                    "pair_id": pair_id,
                }
                if terminal_kind is EventKind.SET:
                    common.update(
                        {
                            "distractor_variant": DistractorVariant.CLEAN,
                            "distractor_pair_id": f"r3-transition-noop-pair-{read_form}-r{replica}",
                            "distractor_episode_id": f"r3-transition-noop-{read_form}-r{replica}",
                        }
                    )
                elif terminal_kind is EventKind.NOOP:
                    common.update(
                        {
                            "distractor_variant": DistractorVariant.DISTRACTOR,
                            "distractor_pair_id": f"r3-transition-noop-pair-{read_form}-r{replica}",
                            "distractor_episode_id": f"r3-transition-set-{read_form}-r{replica}",
                        }
                    )
                train.append(
                    _episode(turns=train_turns, template_id=f"r3-transition-{read_form}-train-a", **common)
                )
                gate.append(
                    _episode(turns=gate_turns, template_id=f"r3-transition-{read_form}-gate-b", **common)
                )

    manifest = {
        "schema_version": "vlm.r3.micro.v1",
        "suite": "transition16",
        "seed": seed,
        "semantic_history_count": len(train),
        "heldout_delayed_view_count": len(gate) * len(REVERSE_CYCLIC4),
        "terminal_kinds": [kind.value for kind in kinds],
        "read_forms": list(read_forms),
        "lexical_replicas": 2,
        "train_query_template": "train-a",
        "gate_query_template": "gate-b",
        "train_permutation_family": [list(item) for item in CYCLIC4],
        "gate_permutation_family": [list(item) for item in REVERSE_CYCLIC4],
        "train_permutation_family_sha256": permutation_family_sha256(CYCLIC4),
        "gate_permutation_family_sha256": permutation_family_sha256(REVERSE_CYCLIC4),
        "max_presentations_per_history": 512,
        "evaluation_start": 64,
        "evaluation_interval": 32,
    }
    return R3MicroSuite("transition16", tuple(train), tuple(gate), tuple(sidecar), manifest)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_r3_micro_suite(output_dir: Path, suite: R3MicroSuite) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{suite.name}_train.jsonl"
    gate_path = output_dir / f"{suite.name}_gate.jsonl"
    sidecar_path = output_dir / f"{suite.name}_teacher_sidecar.jsonl"
    manifest_path = output_dir / f"{suite.name}_manifest.json"
    write_jsonl(train_path, suite.train_episodes)
    write_jsonl(gate_path, suite.gate_episodes)
    with sidecar_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in suite.teacher_sidecar:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        **suite.manifest,
        "artifacts": {
            train_path.name: {"sha256": _sha256_file(train_path), "count": len(suite.train_episodes)},
            gate_path.name: {"sha256": _sha256_file(gate_path), "count": len(suite.gate_episodes)},
            sidecar_path.name: {"sha256": _sha256_file(sidecar_path), "count": len(suite.teacher_sidecar)},
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
