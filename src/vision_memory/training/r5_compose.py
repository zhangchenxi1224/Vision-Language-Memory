"""Model-free data and schedule contracts for R5-Compose.

R5 deliberately composes visible turns from the existing formal synthetic corpus.
No semantic ledger, teacher image, target latent, or answer text is introduced.  A
segment carries only source event text and an existing MCQ query.  Cross-slot
families pair episodes with different entity IDs, making their intervening updates
real state changes rather than NOOP distractors.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from vision_memory.data.schema import EventKind, QuerySpec, TurnType, reject_hidden_ledger


R5_PROTOCOL = "R5-Compose"
R5_MANIFEST_SCHEMA = "vision_memory.r5-compose-contract.v1"
R5_SEGMENT_SCHEMA = "vision_memory.r5-compose-segments.v1"
R5_SCHEDULE_SCHEMA = "vision_memory.r5-compose-curriculum.v1"
R5_FAMILIES = ("F1", "F2", "F3", "F4", "F5", "F6")
R5_DIFFUSION_STEPS = (0, 1, 2, 3)


def _field(value: Mapping[str, Any] | Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _enum_text(value: Any) -> str | None:
    value = getattr(value, "value", value)
    return None if value is None else str(value)


def _required_text(value: Mapping[str, Any] | Any, name: str) -> str:
    item = _field(value, name)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"R5 requires a non-empty {name}.")
    return item.strip()


def _stable_digest(*parts: Any) -> bytes:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).digest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class R5Event:
    source_episode_id: str
    source_turn_index: int
    entity_id: str
    event_kind: str
    event_text: str

    def __post_init__(self) -> None:
        if self.event_kind not in {kind.value for kind in EventKind}:
            raise ValueError(f"Unsupported R5 event kind: {self.event_kind!r}")
        if self.source_turn_index < 0:
            raise ValueError("R5 source_turn_index must be non-negative.")
        for value in (self.source_episode_id, self.entity_id, self.event_text):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("R5 event identifiers and visible text must be non-empty.")

    @property
    def noise_turn_id(self) -> int:
        return self.source_turn_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_episode_id": self.source_episode_id,
            "source_turn_index": self.source_turn_index,
            "entity_id": self.entity_id,
            "event_kind": self.event_kind,
            "event_text": self.event_text,
        }


@dataclass(frozen=True)
class R5Segment:
    segment_id: str
    family: str
    events: tuple[R5Event, ...]
    query: QuerySpec
    query_source_episode_id: str
    query_turn_index: int
    query_entity_id: str
    target_event_position: int
    cross_slot_interference: bool
    stale_target_text: str | None = None

    def __post_init__(self) -> None:
        if self.family not in R5_FAMILIES:
            raise ValueError(f"Unsupported R5 family: {self.family!r}")
        if not self.segment_id.strip() or not self.events:
            raise ValueError("R5 segment requires an ID and at least one event.")
        if not 0 <= self.target_event_position < len(self.events):
            raise ValueError("R5 target_event_position is outside the event sequence.")
        if self.query_turn_index < 0:
            raise ValueError("R5 query_turn_index must be non-negative.")
        if self.events[self.target_event_position].entity_id != self.query_entity_id:
            raise ValueError("R5 target event and query must address the same entity.")
        reject_hidden_ledger(self.to_dict())

    @property
    def updater_count(self) -> int:
        return len(self.events)

    @property
    def query_gap(self) -> int:
        """Number of updater calls in the delayed-query prefix."""

        return len(self.events)

    @property
    def target_event_kind(self) -> str:
        return self.events[self.target_event_position].event_kind

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": R5_SEGMENT_SCHEMA,
            "segment_id": self.segment_id,
            "family": self.family,
            "events": [event.to_dict() for event in self.events],
            "query": self.query.to_dict(),
            "query_source_episode_id": self.query_source_episode_id,
            "query_turn_index": self.query_turn_index,
            "query_entity_id": self.query_entity_id,
            "target_event_position": self.target_event_position,
            "target_event_kind": self.target_event_kind,
            "query_gap": self.query_gap,
            "cross_slot_interference": self.cross_slot_interference,
            "stale_target_text": self.stale_target_text,
        }


@dataclass(frozen=True)
class ScheduledR5Segment:
    global_micro_index: int
    optimizer_step_zero: int
    micro_in_step: int
    phase: str
    family_draw_index: int
    segment: R5Segment
    selected_step_indices: tuple[int, ...] | None

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": R5_SCHEDULE_SCHEMA,
            "global_micro_index": self.global_micro_index,
            "optimizer_step_zero": self.optimizer_step_zero,
            "micro_in_step": self.micro_in_step,
            "phase": self.phase,
            "family": self.segment.family,
            "family_draw_index": self.family_draw_index,
            "segment_id": self.segment.segment_id,
            "selected_step_indices": (
                list(self.selected_step_indices) if self.selected_step_indices is not None else None
            ),
        }


@dataclass(frozen=True)
class _Anchor:
    episode_id: str
    entity_id: str
    set_event: R5Event
    initial_query: QuerySpec
    initial_query_index: int


@dataclass(frozen=True)
class _Mutation:
    anchor: _Anchor
    mutation_event: R5Event
    final_query: QuerySpec
    final_query_index: int
    stale_target_text: str | None


def _turns(episode: Mapping[str, Any] | Any) -> Sequence[Any]:
    turns = _field(episode, "turns")
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)) or not turns:
        raise ValueError("R5 requires a non-empty episode turns sequence.")
    return turns


def _event(episode: Mapping[str, Any] | Any, index: int) -> R5Event:
    turn = _turns(episode)[index]
    return R5Event(
        source_episode_id=_required_text(episode, "episode_id"),
        source_turn_index=index,
        entity_id=_required_text(episode, "entity_id"),
        event_kind=_enum_text(_field(turn, "event_kind")) or "",
        event_text=_required_text(turn, "event_text"),
    )


def _query(turn: Mapping[str, Any] | Any) -> QuerySpec | None:
    value = _field(turn, "query")
    if value is None:
        return None
    return QuerySpec.from_dict(value) if isinstance(value, Mapping) else value


def _first_query_before_next_change(
    turns: Sequence[Any],
    *,
    event_index: int,
) -> tuple[int, QuerySpec] | None:
    for index in range(event_index, len(turns)):
        turn = turns[index]
        if index > event_index:
            kind = _enum_text(_field(turn, "event_kind"))
            if kind is not None and kind != EventKind.NOOP.value:
                return None
        query = _query(turn)
        if query is not None:
            return index, query
    return None


def _extract_single_slot_records(
    episodes: Iterable[Mapping[str, Any] | Any],
) -> tuple[tuple[_Anchor, ...], tuple[_Mutation, ...], tuple[_Mutation, ...]]:
    anchors: list[_Anchor] = []
    overwrites: list[_Mutation] = []
    clears: list[_Mutation] = []
    seen_anchor: set[tuple[str, int]] = set()
    for episode in episodes:
        reject_hidden_ledger(episode)
        turns = _turns(episode)
        episode_id = _required_text(episode, "episode_id")
        entity_id = _required_text(episode, "entity_id")
        event_indices = [
            index
            for index, turn in enumerate(turns)
            if _enum_text(_field(turn, "type", _field(turn, "kind"))) in {
                TurnType.EVENT.value,
                TurnType.MIXED.value,
            }
        ]
        set_indices = [
            index
            for index in event_indices
            if _enum_text(_field(turns[index], "event_kind")) == EventKind.SET.value
        ]
        for set_index in set_indices:
            found = _first_query_before_next_change(turns, event_index=set_index)
            if found is None:
                continue
            query_index, query = found
            anchor = _Anchor(episode_id, entity_id, _event(episode, set_index), query, query_index)
            key = (episode_id, set_index)
            if key not in seen_anchor:
                anchors.append(anchor)
                seen_anchor.add(key)

        # Mutation families use the nearest preceding SET and the first query
        # before another state-changing event.  NOOPs are intentionally ignored:
        # R5 routes them through exact identity.
        for mutation_index in event_indices:
            kind = _enum_text(_field(turns[mutation_index], "event_kind"))
            if kind not in {EventKind.OVERWRITE.value, EventKind.CLEAR.value}:
                continue
            preceding_sets = [index for index in set_indices if index < mutation_index]
            if not preceding_sets:
                continue
            found = _first_query_before_next_change(turns, event_index=mutation_index)
            if found is None:
                continue
            query_index, query = found
            set_index = preceding_sets[-1]
            initial_read = _first_query_before_next_change(turns, event_index=set_index)
            stale_target_text = (
                initial_read[1].target
                if initial_read is not None and initial_read[0] < mutation_index
                else None
            )
            anchor = _Anchor(
                episode_id,
                entity_id,
                _event(episode, set_index),
                initial_read[1] if initial_read is not None else query,
                initial_read[0] if initial_read is not None else query_index,
            )
            mutation = _Mutation(
                anchor,
                _event(episode, mutation_index),
                query,
                query_index,
                stale_target_text,
            )
            (overwrites if kind == EventKind.OVERWRITE.value else clears).append(mutation)

    key = lambda item: (item.episode_id, item.set_event.source_turn_index)  # noqa: E731
    mutation_key = lambda item: (  # noqa: E731
        item.anchor.episode_id,
        item.anchor.set_event.source_turn_index,
        item.mutation_event.source_turn_index,
    )
    return (
        tuple(sorted(anchors, key=key)),
        tuple(sorted(overwrites, key=mutation_key)),
        tuple(sorted(clears, key=mutation_key)),
    )


def _segment_id(family: str, events: Sequence[R5Event], query_episode: str, query_index: int) -> str:
    parts: list[Any] = [R5_SEGMENT_SCHEMA, family]
    for event in events:
        parts.extend((event.source_episode_id, event.source_turn_index))
    parts.extend((query_episode, query_index))
    return f"r5-{family.casefold()}-{_stable_digest(*parts).hex()[:24]}"


def _make_segment(
    family: str,
    events: Sequence[R5Event],
    *,
    query: QuerySpec,
    query_episode_id: str,
    query_turn_index: int,
    query_entity_id: str,
    target_event_position: int,
    cross_slot_interference: bool,
    stale_target_text: str | None = None,
) -> R5Segment:
    values = tuple(events)
    return R5Segment(
        segment_id=_segment_id(family, values, query_episode_id, query_turn_index),
        family=family,
        events=values,
        query=query,
        query_source_episode_id=query_episode_id,
        query_turn_index=query_turn_index,
        query_entity_id=query_entity_id,
        target_event_position=target_event_position,
        cross_slot_interference=cross_slot_interference,
        stale_target_text=stale_target_text,
    )


def _different_entity_partner(
    left_id: str,
    values: Sequence[_Anchor | _Mutation],
    *,
    start: int,
) -> _Anchor | _Mutation:
    if not values:
        raise ValueError("R5 cross-slot construction received an empty partner pool.")
    for offset in range(len(values)):
        candidate = values[(start + offset) % len(values)]
        entity_id = candidate.entity_id if isinstance(candidate, _Anchor) else candidate.anchor.entity_id
        if entity_id != left_id:
            return candidate
    raise ValueError("R5 cross-slot construction requires at least two different entity IDs.")


def build_r5_family_pools(
    episodes: Iterable[Mapping[str, Any] | Any],
    *,
    pairing_seed: int = 0,
) -> dict[str, tuple[R5Segment, ...]]:
    """Build deterministic F1--F6 pools from visible formal-corpus turns."""

    records = tuple(episodes)
    anchors, overwrites, clears = _extract_single_slot_records(records)
    if not anchors or not overwrites or not clears:
        raise ValueError(
            "R5 requires train/dev records containing SET anchors, OVERWRITE transitions, and CLEAR transitions."
        )

    pools: dict[str, list[R5Segment]] = {family: [] for family in R5_FAMILIES}
    for anchor in anchors:
        pools["F1"].append(
            _make_segment(
                "F1",
                (anchor.set_event,),
                query=anchor.initial_query,
                query_episode_id=anchor.episode_id,
                query_turn_index=anchor.initial_query_index,
                query_entity_id=anchor.entity_id,
                target_event_position=0,
                cross_slot_interference=False,
            )
        )
    for mutation in overwrites:
        pools["F3"].append(
            _make_segment(
                "F3",
                (mutation.anchor.set_event, mutation.mutation_event),
                query=mutation.final_query,
                query_episode_id=mutation.anchor.episode_id,
                query_turn_index=mutation.final_query_index,
                query_entity_id=mutation.anchor.entity_id,
                target_event_position=1,
                cross_slot_interference=False,
                stale_target_text=mutation.stale_target_text,
            )
        )
    for mutation in clears:
        pools["F4"].append(
            _make_segment(
                "F4",
                (mutation.anchor.set_event, mutation.mutation_event),
                query=mutation.final_query,
                query_episode_id=mutation.anchor.episode_id,
                query_turn_index=mutation.final_query_index,
                query_entity_id=mutation.anchor.entity_id,
                target_event_position=1,
                cross_slot_interference=False,
                stale_target_text=mutation.stale_target_text,
            )
        )

    partner_anchors = tuple(
        sorted(anchors, key=lambda item: (_stable_digest(R5_SEGMENT_SCHEMA, pairing_seed, "anchor", item.episode_id), item.episode_id))
    )
    partner_overwrites = tuple(
        sorted(
            overwrites,
            key=lambda item: (
                _stable_digest(R5_SEGMENT_SCHEMA, pairing_seed, "overwrite", item.anchor.episode_id),
                item.anchor.episode_id,
            ),
        )
    )
    for index, anchor_a in enumerate(anchors):
        anchor_b = _different_entity_partner(
            anchor_a.entity_id,
            partner_anchors,
            start=int.from_bytes(_stable_digest(pairing_seed, "F2", anchor_a.episode_id)[:8], "big") % len(partner_anchors),
        )
        assert isinstance(anchor_b, _Anchor)
        pools["F2"].append(
            _make_segment(
                "F2",
                (anchor_a.set_event, anchor_b.set_event),
                query=anchor_a.initial_query,
                query_episode_id=anchor_a.episode_id,
                query_turn_index=anchor_a.initial_query_index,
                query_entity_id=anchor_a.entity_id,
                target_event_position=0,
                cross_slot_interference=True,
            )
        )

        overwrite_b = _different_entity_partner(
            anchor_a.entity_id,
            partner_overwrites,
            start=int.from_bytes(_stable_digest(pairing_seed, "F5", anchor_a.episode_id)[:8], "big") % len(partner_overwrites),
        )
        assert isinstance(overwrite_b, _Mutation)
        pools["F5"].append(
            _make_segment(
                "F5",
                (anchor_a.set_event, overwrite_b.anchor.set_event, overwrite_b.mutation_event),
                query=anchor_a.initial_query,
                query_episode_id=anchor_a.episode_id,
                query_turn_index=anchor_a.initial_query_index,
                query_entity_id=anchor_a.entity_id,
                target_event_position=0,
                cross_slot_interference=True,
            )
        )

        overwrite_a = overwrites[index % len(overwrites)]
        anchor_b_for_f6 = _different_entity_partner(
            overwrite_a.anchor.entity_id,
            partner_anchors,
            start=int.from_bytes(_stable_digest(pairing_seed, "F6", overwrite_a.anchor.episode_id, index)[:8], "big") % len(partner_anchors),
        )
        assert isinstance(anchor_b_for_f6, _Anchor)
        pools["F6"].append(
            _make_segment(
                "F6",
                (overwrite_a.anchor.set_event, anchor_b_for_f6.set_event, overwrite_a.mutation_event),
                query=anchor_b_for_f6.initial_query,
                query_episode_id=anchor_b_for_f6.episode_id,
                query_turn_index=anchor_b_for_f6.initial_query_index,
                query_entity_id=anchor_b_for_f6.entity_id,
                target_event_position=1,
                cross_slot_interference=True,
            )
        )

    result: dict[str, tuple[R5Segment, ...]] = {}
    for family, values in pools.items():
        deduplicated = {segment.segment_id: segment for segment in values}
        if not deduplicated:
            raise ValueError(f"R5 family {family} is empty.")
        result[family] = tuple(sorted(deduplicated.values(), key=lambda item: item.segment_id))
    return result


def family_pool_audit(pools: Mapping[str, Sequence[R5Segment]]) -> dict[str, Any]:
    if set(pools) != set(R5_FAMILIES):
        raise ValueError("R5 family pools must contain F1--F6 exactly.")
    flat = [segment for family in R5_FAMILIES for segment in pools[family]]
    if len({segment.segment_id for segment in flat}) != len(flat):
        raise ValueError("R5 segment IDs must be globally unique across family pools.")
    payload = [segment.to_dict() for segment in flat]
    return {
        "schema": R5_SEGMENT_SCHEMA,
        "family_counts": {family: len(pools[family]) for family in R5_FAMILIES},
        "event_count_distribution": dict(sorted(Counter(segment.updater_count for segment in flat).items())),
        "query_gap_distribution": dict(sorted(Counter(segment.query_gap for segment in flat).items())),
        "cross_slot_count": sum(segment.cross_slot_interference for segment in flat),
        "segment_payload_sha256": canonical_sha256(payload),
    }


def curriculum_phase(step_zero: int) -> str:
    if isinstance(step_zero, bool) or not isinstance(step_zero, int) or step_zero < 0:
        raise ValueError("R5 optimizer step must be a non-negative integer.")
    if step_zero < 64:
        return "retention_guidance"
    if step_zero < 256:
        return "state_algebra"
    return "full_composition"


def family_composition_for_step(step_zero: int) -> tuple[str, ...]:
    phase = curriculum_phase(step_zero)
    if phase == "retention_guidance":
        return ("F1",) * 4 + ("F2",) * 4
    if phase == "state_algebra":
        # A five-step block realizes 25/25/30/20 exactly.  The 192-step
        # preregistered phase ends within 0.08 percentage points of that target.
        if step_zero % 5 in {0, 1}:
            return ("F1",) * 2 + ("F2",) * 2 + ("F3",) * 3 + ("F4",)
        return ("F1",) * 2 + ("F2",) * 2 + ("F3",) * 2 + ("F4",) * 2
    long_family = "F5" if (step_zero - 256) % 2 == 0 else "F6"
    return ("F1",) * 2 + ("F2",) * 2 + ("F3",) * 2 + ("F4", long_family)


def selected_step_indices(global_micro_index: int, selected_step_count: int) -> tuple[int, ...]:
    if selected_step_count == 1:
        return (R5_DIFFUSION_STEPS[global_micro_index % len(R5_DIFFUSION_STEPS)],)
    if selected_step_count == 2:
        return ((0, 2), (1, 3))[global_micro_index % 2]
    raise ValueError("R5 selected_step_count must be 1 or 2.")


def _permuted_pool(
    pool: Sequence[R5Segment],
    *,
    family: str,
    schedule_seed: int,
    epoch: int,
) -> tuple[R5Segment, ...]:
    return tuple(
        sorted(
            pool,
            key=lambda segment: (
                _stable_digest(R5_SCHEDULE_SCHEMA, schedule_seed, family, epoch, segment.segment_id),
                segment.segment_id,
            ),
        )
    )


def build_r5_schedule(
    pools: Mapping[str, Sequence[R5Segment]],
    *,
    optimizer_steps: int,
    schedule_seed: int,
    selected_step_count: int,
) -> tuple[ScheduledR5Segment, ...]:
    if optimizer_steps <= 0:
        raise ValueError("R5 optimizer_steps must be positive.")
    if selected_step_count not in {0, 1, 2}:
        raise ValueError("R5 selected_step_count must be 0, 1, or 2.")
    if set(pools) != set(R5_FAMILIES) or any(not pools[family] for family in R5_FAMILIES):
        raise ValueError("R5 schedule requires non-empty F1--F6 pools.")
    draws: defaultdict[str, int] = defaultdict(int)
    epoch_cache: dict[tuple[str, int], tuple[R5Segment, ...]] = {}
    scheduled: list[ScheduledR5Segment] = []
    global_micro = 0
    for step_zero in range(optimizer_steps):
        composition = family_composition_for_step(step_zero)
        ordered = tuple(
            sorted(
                enumerate(composition),
                key=lambda item: (
                    _stable_digest(R5_SCHEDULE_SCHEMA, schedule_seed, step_zero, item[0], item[1]),
                    item,
                ),
            )
        )
        for micro_in_step, (_source_position, family) in enumerate(ordered):
            draw = draws[family]
            pool = pools[family]
            epoch, offset = divmod(draw, len(pool))
            key = (family, epoch)
            if key not in epoch_cache:
                epoch_cache[key] = _permuted_pool(
                    pool,
                    family=family,
                    schedule_seed=schedule_seed,
                    epoch=epoch,
                )
            segment = epoch_cache[key][offset]
            scheduled.append(
                ScheduledR5Segment(
                    global_micro_index=global_micro,
                    optimizer_step_zero=step_zero,
                    micro_in_step=micro_in_step,
                    phase=curriculum_phase(step_zero),
                    family_draw_index=draw,
                    segment=segment,
                    selected_step_indices=(
                        None if selected_step_count == 0 else selected_step_indices(global_micro, selected_step_count)
                    ),
                )
            )
            draws[family] += 1
            global_micro += 1
    return tuple(scheduled)


def schedule_audit(schedule: Sequence[ScheduledR5Segment]) -> dict[str, Any]:
    receipts = [unit.receipt() for unit in schedule]
    return {
        "schema": R5_SCHEDULE_SCHEMA,
        "micro_segments": len(schedule),
        "optimizer_steps": len(schedule) // 8,
        "family_counts": dict(sorted(Counter(unit.segment.family for unit in schedule).items())),
        "phase_counts": dict(sorted(Counter(unit.phase for unit in schedule).items())),
        "selected_step_set_counts": dict(
            sorted(Counter(str(unit.selected_step_indices) for unit in schedule).items())
        ),
        "receipts_sha256": canonical_sha256(receipts),
    }


def stable_record_split(
    records: Sequence[Any],
    *,
    seed: int,
    select_count: int = 32,
    final_count: int = 128,
) -> dict[str, tuple[Any, ...]]:
    if len(records) < select_count + final_count:
        raise ValueError("R5 dev pool is too small for disjoint select/final subsets.")
    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                _stable_digest(R5_SEGMENT_SCHEMA, seed, "dev-split", _required_text(record, "episode_id")),
                _required_text(record, "episode_id"),
            ),
        )
    )
    return {
        "select": ordered[:select_count],
        "final": ordered[select_count : select_count + final_count],
        "reserve": ordered[select_count + final_count :],
    }


def mechanism_subsets(
    pools: Mapping[str, Sequence[R5Segment]],
    *,
    seed: int,
    select_per_category: int = 8,
    final_per_category: int = 32,
) -> dict[str, Any]:
    categories = {
        "delayed_retention": "F5",
        "overwrite": "F3",
        "clear": "F4",
        "cross_slot_interference": "F6",
    }
    selected: list[R5Segment] = []
    final: list[R5Segment] = []
    category_counts: dict[str, dict[str, int]] = {}
    for category, family in categories.items():
        ordered = sorted(
            pools[family],
            key=lambda segment: (
                _stable_digest(R5_SEGMENT_SCHEMA, seed, "mechanism", category, segment.segment_id),
                segment.segment_id,
            ),
        )
        required = select_per_category + final_per_category
        if len(ordered) < required:
            raise ValueError(f"R5 mechanism category {category} has {len(ordered)} < {required} segments.")
        selected.extend(ordered[:select_per_category])
        final.extend(ordered[select_per_category:required])
        category_counts[category] = {"select": select_per_category, "final": final_per_category}
    return {
        "select": tuple(selected),
        "final": tuple(final),
        "category_counts": category_counts,  # type: ignore[dict-item]
    }


def make_r5_manifest_contract(
    *,
    persistent_state: str,
    tbptt_horizon: int,
    selected_step_count: int,
    schedule_seed: int,
    gradient_mode: str = "drtune_stateful",
) -> dict[str, Any]:
    if persistent_state not in {"float_rgb", "latent"}:
        raise ValueError("R5 persistent_state must be float_rgb or latent.")
    if tbptt_horizon not in {2, 4}:
        raise ValueError("R5 tbptt_horizon must be 2 or 4.")
    if gradient_mode not in {"drtune_stateful", "full"}:
        raise ValueError("R5 gradient_mode must be drtune_stateful or full.")
    expected_counts = {1, 2} if gradient_mode == "drtune_stateful" else {0}
    if selected_step_count not in expected_counts:
        raise ValueError(
            f"R5 selected_step_count must be {sorted(expected_counts)} for gradient_mode={gradient_mode}."
        )
    return {
        "schema": R5_MANIFEST_SCHEMA,
        "protocol": R5_PROTOCOL,
        "persistent_state": persistent_state,
        "tbptt_horizon": tbptt_horizon,
        "gradient_mode": gradient_mode,
        "selected_steps_per_transition": selected_step_count,
        "noise_key": ["global_seed", "source_episode_id", "source_turn_id"],
        "presentation_index_in_noise": False,
        "noop_policy": "hard_identity_same_tensor",
        "noop_dreamlite_forward": False,
        "noop_identity_loss": False,
        "state_target_policy": "none",
        "teacher_or_canonical_target_loaded": False,
        "reader_parameters_frozen": True,
        "dreamlite_base_frozen": True,
        "trainable_parameters": "dreamlite_unet_lora_only",
        "final_query_weight": 1.0,
        "aux_query_total_weight": 0.0,
        "schedule_schema": R5_SCHEDULE_SCHEMA,
        "schedule_seed": schedule_seed,
    }


__all__ = [
    "R5_DIFFUSION_STEPS",
    "R5_FAMILIES",
    "R5_MANIFEST_SCHEMA",
    "R5_PROTOCOL",
    "R5_SCHEDULE_SCHEMA",
    "R5_SEGMENT_SCHEMA",
    "R5Event",
    "R5Segment",
    "ScheduledR5Segment",
    "build_r5_family_pools",
    "build_r5_schedule",
    "canonical_sha256",
    "curriculum_phase",
    "family_composition_for_step",
    "family_pool_audit",
    "make_r5_manifest_contract",
    "mechanism_subsets",
    "schedule_audit",
    "selected_step_indices",
    "stable_record_split",
]
