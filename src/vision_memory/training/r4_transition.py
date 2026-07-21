"""Teacher-free transition indexing and scheduling for R4-FreePixel training.

This module deliberately contains no model imports.  It defines the data boundary
between the synthetic episodes and an R4 trainer while keeping the visual state
representation unconstrained by teacher images, latents, or codebooks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, TypeVar

from vision_memory.data.schema import EventKind, TurnType, reject_hidden_ledger


R4_PROTOCOL = "R4-FreePixel"
R4_MANIFEST_SCHEMA = "vision_memory.r4-free-pixel-contract.v1"
R4_SCHEDULE_SCHEMA = "vision_memory.r4-balanced-transition-schedule.v1"
R4_EVENT_KINDS = tuple(kind.value for kind in EventKind)
R4_DIFFUSION_STEPS = (0, 1, 2, 3)
R4_BALANCED_BLOCK_SIZE = len(R4_EVENT_KINDS) * len(R4_DIFFUSION_STEPS)


class TransitionObjective(str, Enum):
    """Losses an R4 trainer is allowed to apply to one transition."""

    QA_ONLY = "qa_only"
    IDENTITY_ONLY = "identity_only"
    QA_AND_IDENTITY = "qa_and_identity"
    PREFIX_ONLY = "prefix_only"


@dataclass(frozen=True)
class TransitionExample:
    """One updater call plus only the queries causally local to that update.

    ``episode``, ``target_turn``, ``prefix_turns`` and ``local_query_turns`` retain
    the original objects.  A trainer can therefore replay the exact visible prefix
    without reconstructing or enriching turns with oracle state.
    """

    transition_id: str
    episode_id: str
    episode: Any
    target_turn_index: int
    target_event_ordinal: int
    event_kind: str
    target_turn: Any
    prefix_turns: tuple[Any, ...]
    prefix_updater_indices: tuple[int, ...]
    local_query_turns: tuple[Any, ...]
    local_query_indices: tuple[int, ...]
    next_updater_index: int | None
    objective: TransitionObjective

    @property
    def is_terminal_update(self) -> bool:
        return self.next_updater_index is None

    @property
    def is_trainable(self) -> bool:
        return self.objective is not TransitionObjective.PREFIX_ONLY

    @property
    def uses_qa_loss(self) -> bool:
        return self.objective in {
            TransitionObjective.QA_ONLY,
            TransitionObjective.QA_AND_IDENTITY,
        }

    @property
    def uses_identity_loss(self) -> bool:
        return self.objective in {
            TransitionObjective.IDENTITY_ONLY,
            TransitionObjective.QA_AND_IDENTITY,
        }

    @property
    def qa_query_turns(self) -> tuple[Any, ...]:
        """Queries actually admitted to the loss (not merely locally attributable)."""

        return self.local_query_turns if self.uses_qa_loss else ()

    @property
    def qa_query_indices(self) -> tuple[int, ...]:
        return self.local_query_indices if self.uses_qa_loss else ()


@dataclass(frozen=True)
class ScheduledTransition:
    """A deterministic unit in the event-kind x diffusion-step schedule."""

    unit_index: int
    block_index: int
    block_offset: int
    selected_step_index: int
    transition: TransitionExample

    @property
    def selected_step_indices(self) -> tuple[int, ...]:
        return (self.selected_step_index,)

    @property
    def event_kind(self) -> str:
        return self.transition.event_kind

    def receipt(self) -> dict[str, Any]:
        return {
            "schedule_schema": R4_SCHEDULE_SCHEMA,
            "unit_index": self.unit_index,
            "block_index": self.block_index,
            "block_offset": self.block_offset,
            "transition_id": self.transition.transition_id,
            "event_kind": self.event_kind,
            "selected_step_index": self.selected_step_index,
        }


@dataclass(frozen=True)
class R4ScheduleCursor:
    """Minimal checkpoint state needed to resume the stateless schedule exactly."""

    seed: int
    next_unit_index: int
    schema: str = R4_SCHEDULE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != R4_SCHEDULE_SCHEMA:
            raise ValueError(f"Unsupported R4 schedule schema: {self.schema!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("R4 schedule seed must be an integer, not bool.")
        if (
            isinstance(self.next_unit_index, bool)
            or not isinstance(self.next_unit_index, int)
            or self.next_unit_index < 0
        ):
            raise ValueError("R4 next_unit_index must be a non-negative integer.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "next_unit_index": self.next_unit_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "R4ScheduleCursor":
        required = {"schema", "seed", "next_unit_index"}
        unknown = set(value) - required
        missing = required - set(value)
        if unknown or missing:
            raise ValueError(f"Invalid R4 schedule cursor fields; missing={sorted(missing)}, unknown={sorted(unknown)}")
        return cls(
            schema=str(value["schema"]),
            seed=value["seed"],
            next_unit_index=value["next_unit_index"],
        )


StateT = TypeVar("StateT")


class PrefixReplayFn(Protocol[StateT]):
    """Protocol for no-grad replay of the exact visible prefix."""

    def __call__(
        self,
        *,
        prefix_turns: Sequence[Any],
        prefix_updater_indices: Sequence[int],
    ) -> StateT: ...


class TargetUpdateFn(Protocol[StateT]):
    """Protocol implemented by the DreamLite target-event updater."""

    def __call__(
        self,
        state: StateT,
        event_text: str,
        episode_id: str,
        turn_id: str | int,
        *,
        gradient_mode: str,
        selected_step_indices: tuple[int, ...],
        persistent_state: str,
        presentation_index: int,
    ) -> StateT: ...


def _field(value: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _normalized_enum(value: Any) -> str | None:
    value = getattr(value, "value", value)
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return normalized or None


def _episode_id(episode: Mapping[str, Any] | Any) -> str:
    value = _field(episode, "episode_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("R4 transition indexing requires a non-empty episode_id.")
    return value.strip()


def _turns(episode: Mapping[str, Any] | Any) -> Sequence[Any]:
    turns = _field(episode, "turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)) or not turns:
        raise ValueError("R4 transition indexing requires a non-empty turns sequence.")
    return turns


def _turn_type(turn: Mapping[str, Any] | Any) -> str | None:
    return _normalized_enum(_field(turn, "type", _field(turn, "kind")))


def _event_kind(turn: Mapping[str, Any] | Any) -> str | None:
    # R4 intentionally indexes the explicit event_kind field.  Do not infer updater
    # calls merely from type==event; doing so would miss mixed turns and permit
    # malformed event records to cross the model boundary.
    return _normalized_enum(_field(turn, "event_kind"))


def _query(turn: Mapping[str, Any] | Any) -> Any:
    return _field(turn, "query")


def _validate_turn(turn: Mapping[str, Any] | Any, *, episode_id: str, index: int) -> None:
    turn_type = _turn_type(turn)
    event_kind = _event_kind(turn)
    event_text = _field(turn, "event_text")
    query = _query(turn)
    location = f"episode {episode_id!r} turn {index}"
    if turn_type not in {kind.value for kind in TurnType}:
        raise ValueError(f"{location} has unsupported type {turn_type!r}.")
    if event_kind is not None and event_kind not in R4_EVENT_KINDS:
        raise ValueError(f"{location} has unsupported event_kind {event_kind!r}.")
    if turn_type in {TurnType.EVENT.value, TurnType.MIXED.value} and event_kind is None:
        raise ValueError(f"{location} calls the updater but is missing event_kind; R4 fails closed.")
    if event_kind is not None and turn_type not in {TurnType.EVENT.value, TurnType.MIXED.value}:
        raise ValueError(f"{location} has event_kind on a non-updater turn.")
    if event_kind is not None and (not isinstance(event_text, str) or not event_text.strip()):
        raise ValueError(f"{location} calls the updater but has no non-empty event_text.")
    if event_kind is None and event_text is not None:
        raise ValueError(f"{location} has event_text on a non-updater turn.")
    if turn_type == TurnType.MIXED.value and query is None:
        raise ValueError(f"{location} is mixed but has no query.")
    if turn_type == TurnType.QUERY.value and query is None:
        raise ValueError(f"{location} is a query turn but has no query.")
    if turn_type == TurnType.EVENT.value and query is not None:
        raise ValueError(f"{location} is a pure event but contains a query.")


def _objective(
    event_kind: str,
    *,
    has_local_query: bool,
    next_updater_index: int | None,
) -> TransitionObjective:
    if event_kind == EventKind.NOOP.value:
        # Intermediate distractors are identity-only even if an incidental read is
        # present.  A terminal NOOP may use its causally local read in addition to
        # the appearance-free identity loss.
        if next_updater_index is None and has_local_query:
            return TransitionObjective.QA_AND_IDENTITY
        return TransitionObjective.IDENTITY_ONLY
    if has_local_query:
        return TransitionObjective.QA_ONLY
    # Such a state-changing update is still needed during prefix replay, but it
    # cannot become a teacher-free target without a local semantic query.
    return TransitionObjective.PREFIX_ONLY


def build_transition_index(
    episodes: Iterable[Mapping[str, Any] | Any],
) -> tuple[TransitionExample, ...]:
    """Index updater transitions without consulting any hidden/teacher state.

    A query belongs to a target only when it is embedded in that target's mixed
    turn or occurs after it and strictly before the next explicit event_kind.
    Queries on the next mixed updater therefore belong to that next transition.
    """

    examples: list[TransitionExample] = []
    seen_episode_ids: set[str] = set()
    for episode in episodes:
        if isinstance(episode, Mapping):
            reject_hidden_ledger(episode)
        episode_id = _episode_id(episode)
        if episode_id in seen_episode_ids:
            raise ValueError(f"Duplicate episode_id in R4 transition index: {episode_id!r}")
        seen_episode_ids.add(episode_id)
        turns = _turns(episode)
        for index, turn in enumerate(turns):
            _validate_turn(turn, episode_id=episode_id, index=index)

        updater_indices = tuple(index for index, turn in enumerate(turns) if _event_kind(turn) is not None)
        for ordinal, target_index in enumerate(updater_indices):
            target_turn = turns[target_index]
            event_kind = _event_kind(target_turn)
            assert event_kind is not None  # established by updater_indices
            next_updater_index = updater_indices[ordinal + 1] if ordinal + 1 < len(updater_indices) else None
            stop = next_updater_index if next_updater_index is not None else len(turns)
            local_query_indices: list[int] = []
            if _query(target_turn) is not None:
                local_query_indices.append(target_index)
            local_query_indices.extend(
                index for index in range(target_index + 1, stop) if _query(turns[index]) is not None
            )
            prefix_turns = tuple(turns[:target_index])
            prefix_updater_indices = tuple(index for index in updater_indices if index < target_index)
            objective = _objective(
                event_kind,
                has_local_query=bool(local_query_indices),
                next_updater_index=next_updater_index,
            )
            examples.append(
                TransitionExample(
                    transition_id=f"{episode_id}:turn-{target_index}",
                    episode_id=episode_id,
                    episode=episode,
                    target_turn_index=target_index,
                    target_event_ordinal=ordinal,
                    event_kind=event_kind,
                    target_turn=target_turn,
                    prefix_turns=prefix_turns,
                    prefix_updater_indices=prefix_updater_indices,
                    local_query_turns=tuple(turns[index] for index in local_query_indices),
                    local_query_indices=tuple(local_query_indices),
                    next_updater_index=next_updater_index,
                    objective=objective,
                )
            )
    if not examples:
        raise ValueError("R4 transition index is empty; no event_kind!=null turns were found.")
    return tuple(examples)


def trainable_transition_examples(
    examples: Iterable[TransitionExample],
) -> tuple[TransitionExample, ...]:
    """Drop prefix-only mutations while retaining every NOOP identity target."""

    result = tuple(example for example in examples if example.is_trainable)
    if not result:
        raise ValueError("R4 transition index contains no teacher-free trainable targets.")
    return result


def transition_index_audit(examples: Iterable[TransitionExample]) -> dict[str, Any]:
    """Return JSON-ready counts so exclusions are explicit rather than silent."""

    values = tuple(examples)
    by_kind = {kind: 0 for kind in R4_EVENT_KINDS}
    trainable_by_kind = {kind: 0 for kind in R4_EVENT_KINDS}
    by_objective = {objective.value: 0 for objective in TransitionObjective}
    local_query_count = 0
    for example in values:
        if example.event_kind not in by_kind:
            raise ValueError(f"Unexpected R4 event kind in audit: {example.event_kind!r}")
        by_kind[example.event_kind] += 1
        if example.is_trainable:
            trainable_by_kind[example.event_kind] += 1
        by_objective[example.objective.value] += 1
        local_query_count += len(example.local_query_indices)
    return {
        "schema": "vision_memory.r4-transition-index-audit.v1",
        "total_transitions": len(values),
        "trainable_transitions": sum(trainable_by_kind.values()),
        "prefix_only_transitions": by_objective[TransitionObjective.PREFIX_ONLY.value],
        "local_query_count": local_query_count,
        "by_event_kind": by_kind,
        "trainable_by_event_kind": trainable_by_kind,
        "by_objective": by_objective,
    }


def _stable_digest(*parts: object) -> bytes:
    rendered = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(rendered.encode("utf-8")).digest()


def _stable_permutation(
    values: Sequence[TransitionExample],
    *,
    seed: int,
    event_kind: str,
    epoch: int,
) -> tuple[TransitionExample, ...]:
    return tuple(
        sorted(
            values,
            key=lambda example: (
                _stable_digest(R4_SCHEDULE_SCHEMA, seed, event_kind, epoch, example.transition_id),
                example.transition_id,
            ),
        )
    )


def _example_for_kind_draw(
    bucket: Sequence[TransitionExample],
    *,
    seed: int,
    event_kind: str,
    draw_index: int,
) -> TransitionExample:
    if not bucket:
        raise ValueError(f"R4 schedule has no trainable {event_kind!r} transitions.")
    epoch, offset = divmod(draw_index, len(bucket))
    return _stable_permutation(bucket, seed=seed, event_kind=event_kind, epoch=epoch)[offset]


def build_balanced_schedule_block(
    examples: Iterable[TransitionExample],
    *,
    seed: int,
    block_index: int,
) -> tuple[ScheduledTransition, ...]:
    """Build one resumable 16-unit event-kind x four-step block.

    Each block contains every ``(event_kind, selected_step_index)`` pair exactly
    once.  Examples within each event-kind bucket cycle through stable SHA256
    permutations, so a checkpoint needs only ``seed`` and ``next_unit_index``.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("R4 schedule seed must be an integer, not bool.")
    if isinstance(block_index, bool) or not isinstance(block_index, int) or block_index < 0:
        raise ValueError("R4 block_index must be a non-negative integer.")
    buckets: dict[str, list[TransitionExample]] = {kind: [] for kind in R4_EVENT_KINDS}
    seen_transition_ids: set[str] = set()
    for example in examples:
        if example.transition_id in seen_transition_ids:
            raise ValueError(f"Duplicate transition_id in R4 schedule: {example.transition_id!r}")
        seen_transition_ids.add(example.transition_id)
        if example.event_kind not in buckets:
            raise ValueError(f"Unsupported event kind in R4 schedule: {example.event_kind!r}")
        if example.is_trainable:
            buckets[example.event_kind].append(example)
    missing = [kind for kind, bucket in buckets.items() if not bucket]
    if missing:
        raise ValueError(f"R4 balanced schedule requires trainable examples for every event kind: {missing}")

    pairs = [(kind, step) for kind in R4_EVENT_KINDS for step in R4_DIFFUSION_STEPS]
    pairs.sort(key=lambda pair: (_stable_digest(R4_SCHEDULE_SCHEMA, seed, block_index, *pair), pair))
    kind_occurrence = {kind: 0 for kind in R4_EVENT_KINDS}
    scheduled: list[ScheduledTransition] = []
    for block_offset, (event_kind, step) in enumerate(pairs):
        occurrence = kind_occurrence[event_kind]
        kind_occurrence[event_kind] += 1
        draw_index = block_index * len(R4_DIFFUSION_STEPS) + occurrence
        transition = _example_for_kind_draw(
            buckets[event_kind],
            seed=seed,
            event_kind=event_kind,
            draw_index=draw_index,
        )
        scheduled.append(
            ScheduledTransition(
                unit_index=block_index * R4_BALANCED_BLOCK_SIZE + block_offset,
                block_index=block_index,
                block_offset=block_offset,
                selected_step_index=step,
                transition=transition,
            )
        )
    return tuple(scheduled)


def iter_balanced_schedule(
    examples: Iterable[TransitionExample],
    *,
    seed: int,
    start_unit_index: int = 0,
    num_units: int,
) -> Iterator[ScheduledTransition]:
    """Yield an exact schedule suffix; suitable for checkpoint resume."""

    if isinstance(start_unit_index, bool) or not isinstance(start_unit_index, int) or start_unit_index < 0:
        raise ValueError("start_unit_index must be a non-negative integer.")
    if isinstance(num_units, bool) or not isinstance(num_units, int) or num_units < 0:
        raise ValueError("num_units must be a non-negative integer.")
    values = tuple(examples)
    end = start_unit_index + num_units
    block_cache: dict[int, tuple[ScheduledTransition, ...]] = {}
    for unit_index in range(start_unit_index, end):
        block_index, block_offset = divmod(unit_index, R4_BALANCED_BLOCK_SIZE)
        block = block_cache.get(block_index)
        if block is None:
            block = build_balanced_schedule_block(values, seed=seed, block_index=block_index)
            block_cache = {block_index: block}
        yield block[block_offset]


def make_r4_manifest_contract(*, schedule_seed: int) -> dict[str, Any]:
    """Create the immutable teacher-free protocol block for a run manifest."""

    if isinstance(schedule_seed, bool) or not isinstance(schedule_seed, int):
        raise TypeError("R4 schedule seed must be an integer, not bool.")
    return {
        "schema": R4_MANIFEST_SCHEMA,
        "protocol": R4_PROTOCOL,
        "persistent_state": "float_rgb",
        "state_target_policy": "none",
        "canonical_teacher_artifacts_loaded": False,
        "pixel_or_latent_targets_forbidden": True,
        "codebook_forbidden": True,
        "learn_initial_state": False,
        "updater_input_contract": "previous_rgb+visible_event_text+deterministic_sampling_noise",
        "updater_learned_conditioning": "previous_rgb+visible_event_text",
        "sampling_noise_contract": ("prng(global_seed,episode_id,turn_id,presentation_index)"),
        "sampling_noise_keys_answer_agnostic": True,
        "query_or_target_in_noise_seed_forbidden": True,
        "updater_query_inputs_forbidden": True,
        "reader_parameters_frozen": True,
        "event_index_rule": "event_kind_non_null",
        "query_attribution_rule": "target_mixed_or_before_next_updater",
        "intermediate_noop_objective": "identity_only",
        "event_tbptt_horizon": 1,
        "diffusion_gradient_mode": "drtune",
        "diffusion_step_count": len(R4_DIFFUSION_STEPS),
        "selected_steps_per_transition": 1,
        "balanced_schedule_schema": R4_SCHEDULE_SCHEMA,
        "balanced_schedule_seed": schedule_seed,
        "balanced_block_size": R4_BALANCED_BLOCK_SIZE,
    }


def validate_r4_manifest_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an R4 contract exactly and return a detached plain dictionary."""

    if not isinstance(value, Mapping):
        raise TypeError("R4 manifest contract must be a mapping.")
    if "balanced_schedule_seed" not in value:
        raise ValueError("R4 manifest contract is missing balanced_schedule_seed.")
    seed = value["balanced_schedule_seed"]
    expected = make_r4_manifest_contract(schedule_seed=seed)
    unknown = set(value) - set(expected)
    missing = set(expected) - set(value)
    if unknown or missing:
        raise ValueError(f"Invalid R4 manifest contract fields; missing={sorted(missing)}, unknown={sorted(unknown)}")
    mismatches = {
        key: {"expected": expected_item, "observed": value[key]}
        for key, expected_item in expected.items()
        if value[key] != expected_item or type(value[key]) is not type(expected_item)
    }
    if mismatches:
        raise ValueError(f"R4 manifest contract mismatch: {mismatches}")
    return dict(value)


def validate_teacher_free_bindings(**bindings: Any) -> None:
    """Fail closed if a caller tries to attach any teacher/target binding.

    Callers should pass every relevant optional artifact slot, for example
    ``teacher_manifest=args.teacher_manifest`` and ``codebook=args.codebook``.
    The function intentionally treats even false-y non-None objects as bindings.
    """

    attached = sorted(name for name, value in bindings.items() if value is not None)
    if attached:
        raise ValueError(f"R4-FreePixel forbids teacher, canonical target, and codebook bindings: {attached}")


def target_update_kwargs(scheduled: ScheduledTransition) -> dict[str, Any]:
    """Return the locked kwargs expected by an R4-compatible DreamLite updater."""

    if scheduled.selected_step_index not in R4_DIFFUSION_STEPS:
        raise ValueError(f"Invalid R4 selected diffusion step: {scheduled.selected_step_index}")
    return {
        "gradient_mode": "drtune",
        "selected_step_indices": scheduled.selected_step_indices,
        "persistent_state": "float_rgb",
    }


__all__ = [
    "PrefixReplayFn",
    "R4_BALANCED_BLOCK_SIZE",
    "R4_DIFFUSION_STEPS",
    "R4_EVENT_KINDS",
    "R4_MANIFEST_SCHEMA",
    "R4_PROTOCOL",
    "R4_SCHEDULE_SCHEMA",
    "R4ScheduleCursor",
    "ScheduledTransition",
    "TargetUpdateFn",
    "TransitionExample",
    "TransitionObjective",
    "build_balanced_schedule_block",
    "build_transition_index",
    "iter_balanced_schedule",
    "make_r4_manifest_contract",
    "target_update_kwargs",
    "trainable_transition_examples",
    "transition_index_audit",
    "validate_r4_manifest_contract",
    "validate_teacher_free_bindings",
]
