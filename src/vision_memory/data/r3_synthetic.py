"""Leakage-resistant deterministic synthetic episodes for the R3 pilot and main study.

The important construction order is explicit in this module: semantic groups are
allocated to a split first, and only then expanded into counterfactual and
clean/no-op members.  Consequently no related member can cross a split.  The
model-visible JSONL contains only routed events, questions, choices, and labels.
Canonical state ledgers are written to a separate train-only analysis sidecar.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

from .schema import (
    DistractorVariant,
    Episode,
    EventKind,
    QuerySpec,
    Turn,
    TurnType,
    read_jsonl,
    reject_hidden_ledger,
    write_jsonl,
)


R3_SYNTHETIC_SCHEMA = "vlm.r3.synthetic.v1"
R3_ANALYSIS_SIDECAR_SCHEMA = "vlm.r3.teacher_transition.v1"
R3_SEMANTIC_STATE_SCHEMA = "vlm.semantic_state.v1"
R3_SYNTHETIC_SEED = 2026
NO_ACTIVE_PREFERENCE = "no active preference"
OOD_GROUPS = ("heldout_entity", "heldout_topic", "heldout_paraphrase", "heldout_length")
TERMINAL_PROFILES = ("set", "overwrite", "clear", "noop")
READ_FORMS = ("separate", "mixed")

TOPIC_VALUES: dict[str, tuple[str, ...]] = {
    "color": ("red", "blue", "green", "yellow", "purple", "orange"),
    "material": ("wood", "glass", "steel", "ceramic", "linen", "leather"),
    "drink": ("tea", "coffee", "water", "juice", "cocoa", "milk"),
    "style": ("minimal", "vintage", "modern", "rustic", "formal", "playful"),
    "meal": ("pasta", "salad", "curry", "soup", "tacos", "rice"),
    "music": ("jazz", "classical", "folk", "rock", "ambient", "blues"),
}
HELDOUT_TOPIC_VALUES: dict[str, tuple[str, ...]] = {
    "fragrance": ("citrus", "cedar", "vanilla", "mint", "lavender", "rose"),
    "lighting": ("warm", "cool", "dim", "bright", "soft", "focused"),
}
BASE_NOUNS = ("mug", "lamp", "notebook", "backpack", "chair", "device", "desk", "room")
HELDOUT_NOUNS = ("telescope", "violin", "greenhouse", "statue", "drone", "kayak")
ADJECTIVES = ("amber", "canvas", "copper", "linen", "marble", "silver", "willow", "indigo")


def _stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_int(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True)
class R3SyntheticSizes:
    """Episode counts after semantic-group expansion."""

    train: int = 1_000
    dev: int = 500
    test_id: int = 1_000
    test_ood: int = 1_000

    @classmethod
    def pilot(cls) -> "R3SyntheticSizes":
        return cls()

    @classmethod
    def formal(cls) -> "R3SyntheticSizes":
        return cls(train=5_000, dev=500, test_id=1_000, test_ood=1_000)

    def as_dict(self) -> dict[str, int]:
        return {
            "train": self.train,
            "dev": self.dev,
            "test_id": self.test_id,
            "test_ood": self.test_ood,
        }

    def validate(self) -> None:
        for split, count in self.as_dict().items():
            if count <= 0 or count % 2:
                raise ValueError(f"R3 {split} size must be a positive even number, got {count}.")
        if self.test_ood % len(OOD_GROUPS):
            raise ValueError("R3 test_ood must divide evenly across the four preregistered OOD groups.")
        per_ood_group = self.test_ood // len(OOD_GROUPS)
        if per_ood_group % 2:
            raise ValueError("Each R3 OOD group must contain an even number of episodes.")


@dataclass(frozen=True)
class _SemanticGroup:
    semantic_group_id: str
    split: str
    ordinal: int
    expansion_size: int
    ood_group: str | None = None


@dataclass(frozen=True)
class R3SyntheticValidationReport:
    valid: bool
    total_episodes: int
    split_statistics: dict[str, dict[str, Any]]
    split_group_sha256: dict[str, str]
    train_sidecar_records: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "total_episodes": self.total_episodes,
            "split_statistics": self.split_statistics,
            "split_group_sha256": self.split_group_sha256,
            "train_sidecar_records": self.train_sidecar_records,
        }


def _stage_counted_groups(
    *,
    split: str,
    episode_count: int,
    ordinal_start: int,
    ood_group: str | None = None,
) -> tuple[list[_SemanticGroup], int]:
    """Allocate a split before constructing any related episode member."""

    full_groups, residue = divmod(episode_count, 4)
    if residue not in {0, 2}:
        raise ValueError(f"R3 split stratum {split}/{ood_group} cannot be grouped: residue={residue}.")
    groups: list[_SemanticGroup] = []
    ordinal = ordinal_start
    label = split if ood_group is None else f"{split}-{ood_group.replace('_', '-')}"
    for local_index in range(full_groups + int(residue == 2)):
        expansion_size = 2 if residue == 2 and local_index == full_groups else 4
        groups.append(
            _SemanticGroup(
                semantic_group_id=f"r3-{label}-semantic-{local_index:06d}",
                split=split,
                ordinal=ordinal,
                expansion_size=expansion_size,
                ood_group=ood_group,
            )
        )
        ordinal += 1
    return groups, ordinal


def stage_semantic_groups(sizes: R3SyntheticSizes) -> dict[str, tuple[_SemanticGroup, ...]]:
    """Return the immutable split assignment used before all expansion."""

    sizes.validate()
    staged: dict[str, tuple[_SemanticGroup, ...]] = {}
    ordinal = 0
    for split in ("train", "dev", "test_id"):
        groups, ordinal = _stage_counted_groups(
            split=split,
            episode_count=sizes.as_dict()[split],
            ordinal_start=ordinal,
        )
        staged[split] = tuple(groups)
    ood_groups: list[_SemanticGroup] = []
    per_group = sizes.test_ood // len(OOD_GROUPS)
    for ood_group in OOD_GROUPS:
        groups, ordinal = _stage_counted_groups(
            split="test_ood",
            episode_count=per_group,
            ordinal_start=ordinal,
            ood_group=ood_group,
        )
        ood_groups.extend(groups)
    staged["test_ood"] = tuple(ood_groups)
    return staged


def _rotate(values: Sequence[str], phase: int) -> tuple[str, ...]:
    phase %= len(values)
    return tuple(values[phase:]) + tuple(values[:phase])


def _group_semantics(group: _SemanticGroup, *, seed: int) -> dict[str, Any]:
    terminal_profile = TERMINAL_PROFILES[(group.ordinal // 4) % len(TERMINAL_PROFILES)]
    read_form = READ_FORMS[(group.ordinal // 16) % len(READ_FORMS)]
    if group.ood_group == "heldout_topic":
        topics = tuple(HELDOUT_TOPIC_VALUES)
        topic = topics[(group.ordinal // 32) % len(topics)]
        value_bank = HELDOUT_TOPIC_VALUES[topic]
    else:
        topics = tuple(TOPIC_VALUES)
        topic = topics[(group.ordinal // 32) % len(topics)]
        value_bank = TOPIC_VALUES[topic]
    offset = _stable_int(seed, group.semantic_group_id, "values") % len(value_bank)
    rotated_values = _rotate(value_bank, offset)
    candidates = (rotated_values[0], rotated_values[1], rotated_values[2], NO_ACTIVE_PREFERENCE)
    stale = rotated_values[2] if terminal_profile in {"set", "noop"} else rotated_values[0]
    if terminal_profile == "set":
        targets = (rotated_values[0], rotated_values[1])
    elif terminal_profile == "overwrite":
        targets = (rotated_values[1], rotated_values[2])
    elif terminal_profile == "clear":
        targets = (NO_ACTIVE_PREFERENCE, rotated_values[2])
    else:
        targets = (rotated_values[0], rotated_values[1])
    phase = group.ordinal % 4
    choices = _rotate(candidates, phase)
    return {
        "terminal_profile": terminal_profile,
        "read_form": read_form,
        "topic": topic,
        "choices": choices,
        "stale": stale,
        "targets": targets,
        "choice_phase": phase,
    }


def _entity(group: _SemanticGroup, *, seed: int) -> tuple[str, str]:
    nouns = HELDOUT_NOUNS if group.ood_group == "heldout_entity" else BASE_NOUNS
    noun = nouns[_stable_int(seed, group.semantic_group_id, "noun") % len(nouns)]
    adjective = ADJECTIVES[_stable_int(seed, group.semantic_group_id, "adjective") % len(ADJECTIVES)]
    entity_id = f"{group.semantic_group_id}-entity"
    surface = f"the {adjective} {noun} {group.split.replace('_', '-')} {group.ordinal:06d}"
    return entity_id, surface


def _template_family(group: _SemanticGroup) -> str:
    suffix = group.ood_group or "standard"
    return f"r3-{group.split.replace('_', '-')}-{suffix.replace('_', '-')}-templates-{group.ordinal % 32:02d}"


def _event_text(
    *,
    kind: EventKind,
    entity: str,
    topic: str,
    value: str | None,
    style: int,
    paraphrase: bool,
    template_family: str,
) -> str:
    marker = template_family.replace("-", " ").title()
    if kind is EventKind.NOOP:
        facts = (
            "an unrelated hallway clock was repaired yesterday",
            "an unrelated parcel arrived at noon",
            "an unrelated meeting moved to another room",
            "rain is expected in a distant city",
        )
        return f"{marker}: {facts[style % len(facts)]}."
    if kind is EventKind.CLEAR:
        forms = (
            (
                f"Withdraw every current {topic} inclination associated with {entity}.",
                f"Treat {entity} as having no active {topic} preference from now on.",
            )
            if paraphrase
            else (
                f"Clear the saved {topic} preference for {entity}.",
                f"Forget the current {topic} choice for {entity}; none is active now.",
            )
        )
        return f"{marker}: {forms[style % len(forms)]}"
    if value is None:
        raise ValueError(f"{kind.value} requires a value.")
    if kind is EventKind.OVERWRITE:
        forms = (
            (
                f"Supersede the earlier {topic} inclination for {entity} with {value}.",
                f"Resolve the revised {topic} selection for {entity} in favor of {value}.",
            )
            if paraphrase
            else (
                f"Replace the earlier {topic} preference for {entity} with {value}.",
                f"The {topic} preference for {entity} is now {value}, not the previous value.",
            )
        )
    else:
        forms = (
            (
                f"Associate {entity}'s current {topic} inclination with {value}.",
                f"When {entity} comes up later, favor {value} in the {topic} category.",
            )
            if paraphrase
            else (
                f"For {entity}, remember that the preferred {topic} is {value}.",
                f"Save {value} as the current {topic} preference for {entity}.",
            )
        )
    return f"{marker}: {forms[style % len(forms)]}"


def _query(
    *,
    entity: str,
    topic: str,
    choices: tuple[str, str, str, str],
    target: str,
    role: str,
    comparison_id: str,
    paraphrase: bool,
    template_family: str,
) -> QuerySpec:
    if target not in choices:
        raise ValueError(f"Target {target!r} is absent from the four choices.")
    marker = template_family.replace("-", " ").title()
    if paraphrase:
        prompts = {
            "initial": f"Before any later revision, which {topic} inclination applies to {entity}?",
            "immediate": f"Using the update in this same message, which {topic} option now applies to {entity}?",
            "delayed": f"At this later check, which current {topic} inclination applies to {entity}?",
        }
    else:
        prompts = {
            "initial": f"Before any later update, what is the current {topic} preference for {entity}?",
            "immediate": f"After applying this update first, what is the current {topic} preference for {entity}?",
            "delayed": f"At this later check, what is the current {topic} preference for {entity}?",
        }
    return QuerySpec(
        text=f"{marker}: {prompts[role]} Choose exactly one option.",
        choices=choices,
        target_index=choices.index(target),
        comparison_id=comparison_id,
    )


def _state(
    *,
    entity_id: str,
    entity_text: str,
    topic: str,
    status: str,
    value: str | None,
) -> dict[str, Any]:
    if status not in {"unset", "active", "cleared"}:
        raise ValueError(f"Unsupported state status: {status}")
    if (status == "active") != (value is not None):
        raise ValueError("Only an active semantic state may carry a value.")
    return {
        "schema": R3_SEMANTIC_STATE_SCHEMA,
        "entries": [
            {
                "entity_id": entity_id,
                "entity_text": entity_text,
                "slot_id": topic,
                "slot_text": topic,
                "status": status,
                "value_id": value,
                "value_text": value,
            }
        ],
    }


def _transition(status: str, current: str | None, kind: EventKind, value: str | None) -> tuple[str, str | None]:
    if kind in {EventKind.SET, EventKind.OVERWRITE}:
        if value is None:
            raise ValueError(f"{kind.value} requires a value.")
        return "active", value
    if kind is EventKind.CLEAR:
        return "cleared", None
    return status, current


def _expand_episode(
    group: _SemanticGroup,
    *,
    semantic_variant: int,
    stream: str,
    seed: int,
    semantics: Mapping[str, Any],
) -> tuple[Episode, list[dict[str, Any]]]:
    if stream not in {"clean", "noop"}:
        raise ValueError(f"Unsupported stream: {stream}")
    entity_id, entity_text = _entity(group, seed=seed)
    topic = str(semantics["topic"])
    raw_choices = tuple(semantics["choices"])
    if len(raw_choices) != 4:
        raise ValueError("Internal R3 candidate construction failed.")
    choices = cast(tuple[str, str, str, str], tuple(str(item) for item in raw_choices))
    target = str(semantics["targets"][semantic_variant])
    stale = str(semantics["stale"])
    terminal_profile = str(semantics["terminal_profile"])
    read_form = str(semantics["read_form"])
    template_family = _template_family(group)
    paraphrase = group.ood_group == "heldout_paraphrase"
    style = _stable_int(seed, group.semantic_group_id, semantic_variant, stream, "style") % 32
    episode_id = f"{group.semantic_group_id}-s{semantic_variant}-{stream}"
    mate_id = f"{group.semantic_group_id}-s{1 - semantic_variant}-{stream}"
    pair_id = f"{group.semantic_group_id}-counterfactual-{stream}"
    distractor_pair_id = f"{group.semantic_group_id}-noop-pair-s{semantic_variant}"
    distractor_episode_id = f"{group.semantic_group_id}-s{semantic_variant}-{'noop' if stream == 'clean' else 'clean'}"
    turns: list[Turn] = []
    sidecar: list[dict[str, Any]] = []
    status, current = "unset", None

    def append_event(kind: EventKind, value: str | None, *, query: QuerySpec | None = None) -> None:
        nonlocal status, current
        turn_index = len(turns)
        before = _state(
            entity_id=entity_id,
            entity_text=entity_text,
            topic=topic,
            status=status,
            value=current,
        )
        text = _event_text(
            kind=kind,
            entity=entity_text,
            topic=topic,
            value=value,
            style=style + turn_index,
            paraphrase=paraphrase,
            template_family=template_family,
        )
        turns.append(
            Turn(
                TurnType.MIXED if query is not None else TurnType.EVENT,
                kind,
                text,
                query,
            )
        )
        status, current = _transition(status, current, kind, value)
        after = _state(
            entity_id=entity_id,
            entity_text=entity_text,
            topic=topic,
            status=status,
            value=current,
        )
        if group.split == "train":
            sidecar.append(
                {
                    "schema_version": R3_ANALYSIS_SIDECAR_SCHEMA,
                    "split": group.split,
                    "episode_id": episode_id,
                    "turn_id": turn_index,
                    "event_kind": kind.value,
                    "before_state": before,
                    "after_state": after,
                }
            )

    comparison_prefix = f"{group.semantic_group_id}:s{semantic_variant}"
    append_event(EventKind.SET, stale)
    turns.append(
        Turn(
            TurnType.QUERY,
            query=_query(
                entity=entity_text,
                topic=topic,
                choices=choices,
                target=stale,
                role="initial",
                comparison_id=f"{comparison_prefix}:initial",
                paraphrase=paraphrase,
                template_family=template_family,
            ),
        )
    )

    if group.ood_group == "heldout_length":
        for _ in range(5):
            append_event(EventKind.SET, stale)

    final_kind: EventKind
    final_value: str | None
    if terminal_profile == "overwrite":
        final_kind, final_value = EventKind.OVERWRITE, target
    elif terminal_profile == "clear":
        if semantic_variant == 0:
            final_kind, final_value = EventKind.CLEAR, None
        else:
            final_kind, final_value = EventKind.OVERWRITE, target
    else:
        final_kind, final_value = EventKind.SET, target

    if terminal_profile == "noop" and stream == "noop":
        append_event(EventKind.SET, target)
        final_kind, final_value = EventKind.NOOP, None
    elif stream == "noop":
        append_event(EventKind.NOOP, None)

    immediate_query = _query(
        entity=entity_text,
        topic=topic,
        choices=choices,
        target=target,
        role="immediate",
        comparison_id=f"{comparison_prefix}:immediate",
        paraphrase=paraphrase,
        template_family=template_family,
    )
    if read_form == "mixed":
        append_event(final_kind, final_value, query=immediate_query)
    else:
        append_event(final_kind, final_value)

    delayed_query = _query(
        entity=entity_text,
        topic=topic,
        choices=choices,
        target=target,
        role="delayed",
        comparison_id=f"{comparison_prefix}:delayed",
        paraphrase=paraphrase,
        template_family=template_family,
    )
    turns.append(Turn(TurnType.QUERY, query=delayed_query))

    if target == NO_ACTIVE_PREFERENCE:
        if status != "cleared" or current is not None:
            raise RuntimeError("Clear episode did not reach the preregistered cleared state.")
    elif status != "active" or current != target:
        raise RuntimeError("Episode terminal ledger disagrees with the delayed-query target.")

    if group.expansion_size == 4:
        distractor_variant = DistractorVariant.CLEAN if stream == "clean" else DistractorVariant.DISTRACTOR
        episode_distractor_pair_id: str | None = distractor_pair_id
        episode_distractor_episode_id: str | None = distractor_episode_id
    else:
        distractor_variant = DistractorVariant.UNPAIRED
        episode_distractor_pair_id = None
        episode_distractor_episode_id = None
    episode = Episode(
        episode_id=episode_id,
        split=group.split,
        seed=seed,
        entity_id=entity_id,
        entity_surface=entity_text,
        template_id=f"{template_family}-{terminal_profile}-{read_form}",
        template_family=template_family,
        turns=tuple(turns),
        pair_id=pair_id,
        counterfactual_episode_id=mate_id,
        topic=topic,
        semantic_group_id=group.semantic_group_id,
        ood_group=group.ood_group,
        distractor_variant=distractor_variant,
        distractor_pair_id=episode_distractor_pair_id,
        distractor_episode_id=episode_distractor_episode_id,
    )
    return episode, sidecar


def _expand_group(group: _SemanticGroup, *, seed: int) -> tuple[list[Episode], list[dict[str, Any]]]:
    semantics = _group_semantics(group, seed=seed)
    streams = ("clean", "noop") if group.expansion_size == 4 else ("clean",)
    episodes: list[Episode] = []
    sidecar: list[dict[str, Any]] = []
    for stream in streams:
        for semantic_variant in (0, 1):
            episode, records = _expand_episode(
                group,
                semantic_variant=semantic_variant,
                stream=stream,
                seed=seed,
                semantics=semantics,
            )
            episodes.append(episode)
            sidecar.extend(records)
    return episodes, sidecar


def build_r3_synthetic(
    *,
    sizes: R3SyntheticSizes = R3SyntheticSizes(),
    seed: int = R3_SYNTHETIC_SEED,
) -> tuple[dict[str, tuple[Episode, ...]], tuple[dict[str, Any], ...], dict[str, Any]]:
    """Build episodes in memory while preserving the split-before-expansion proof."""

    staged = stage_semantic_groups(sizes)
    split_episodes: dict[str, tuple[Episode, ...]] = {}
    train_sidecar: list[dict[str, Any]] = []
    group_stage: dict[str, Any] = {}
    for split, groups in staged.items():
        episodes: list[Episode] = []
        for group in groups:
            members, records = _expand_group(group, seed=seed)
            episodes.extend(members)
            train_sidecar.extend(records)
        episodes.sort(key=lambda item: item.episode_id)
        split_episodes[split] = tuple(episodes)
        group_ids = sorted(group.semantic_group_id for group in groups)
        choice_phase_assignments = [
            {"semantic_group_id": group.semantic_group_id, "phase": group.ordinal % 4}
            for group in sorted(groups, key=lambda item: item.semantic_group_id)
        ]
        group_stage[split] = {
            "semantic_group_count": len(groups),
            "semantic_group_ids_sha256": _sha256_value(group_ids),
            "choice_phase_assignment_sha256": _sha256_value(choice_phase_assignments),
            "choice_phase_counts": {
                str(phase): sum(group.ordinal % 4 == phase for group in groups) for phase in range(4)
            },
            "full_expansion_groups": sum(group.expansion_size == 4 for group in groups),
            "counterfactual_only_residual_groups": sum(group.expansion_size == 2 for group in groups),
            "expanded_episode_count": len(episodes),
        }
    train_sidecar.sort(key=lambda row: (str(row["episode_id"]), int(row["turn_id"])))
    blueprint = {
        "schema_version": R3_SYNTHETIC_SCHEMA,
        "seed": seed,
        "sizes": sizes.as_dict(),
        "split_before_expansion": True,
        "expansion_order": ["semantic_group_split", "counterfactual", "clean_noop", "choice_permutation"],
        "choice_permutation_contract": (
            "A deterministic four-phase candidate rotation is assigned only after group splitting; "
            "training cyclic4 and evaluation reverse-cyclic4 views remain runtime Reader views."
        ),
        "pair_metadata_contract": {
            "semantic_group_id": "contains every counterfactual and clean/no-op expansion member",
            "pair_id": "reciprocal two-member semantic counterfactual pair",
            "counterfactual_episode_id": "reciprocal in-split member with a different delayed target",
            "distractor_pair_id": "reciprocal equal-target clean versus forced no-op pair",
            "residual_policy": "two-member counterfactual-only residue is marked distractor_variant=unpaired",
        },
        "group_stage": group_stage,
        "analysis_sidecar_policy": {
            "analysis_only": True,
            "train_only": True,
            "embedded_in_episode_jsonl": False,
            "dev_test_teacher_references_forbidden": True,
        },
    }
    return split_episodes, tuple(train_sidecar), blueprint


def _split_statistics(episodes: Sequence[Episode]) -> dict[str, Any]:
    positions: Counter[int] = Counter()
    event_kinds: Counter[str] = Counter()
    ood_groups: Counter[str] = Counter()
    mixed_count = 0
    delayed_after_mixed = 0
    for episode in episodes:
        if episode.ood_group is not None:
            ood_groups[episode.ood_group] += 1
        for index, turn in enumerate(episode.turns):
            if turn.event_kind is not None:
                event_kinds[turn.event_kind.value] += 1
            if turn.query is not None:
                positions[turn.query.target_index] += 1
            if turn.type is TurnType.MIXED:
                mixed_count += 1
                if index + 1 < len(episode.turns) and episode.turns[index + 1].type is TurnType.QUERY:
                    delayed_after_mixed += 1
    total_queries = sum(positions.values())
    shares = {str(position): positions[position] / total_queries for position in range(4)}
    return {
        "episode_count": len(episodes),
        "semantic_group_count": len({episode.semantic_group_id for episode in episodes}),
        "query_count": total_queries,
        "target_position_counts": {str(position): positions[position] for position in range(4)},
        "target_position_shares": shares,
        "max_target_position_deviation": max(abs(value - 0.25) for value in shares.values()),
        "event_kind_counts": dict(sorted(event_kinds.items())),
        "mixed_turn_count": mixed_count,
        "mixed_with_immediate_delayed_probe": delayed_after_mixed,
        "counterfactual_pair_count": len({episode.pair_id for episode in episodes}),
        "clean_noop_pair_count": len(
            {episode.distractor_pair_id for episode in episodes if episode.distractor_pair_id is not None}
        ),
        "ood_group_counts": dict(sorted(ood_groups.items())),
    }


def _write_jsonl_records(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def generate_r3_synthetic(
    output_dir: Path,
    *,
    sizes: R3SyntheticSizes = R3SyntheticSizes(),
    seed: int = R3_SYNTHETIC_SEED,
    profile: str = "pilot",
) -> dict[str, Any]:
    """Write the R3 corpus, independent train sidecar, and reproducibility manifest."""

    if profile not in {"pilot", "formal", "custom"}:
        raise ValueError("R3 profile must be pilot, formal, or custom.")
    if profile == "pilot" and sizes != R3SyntheticSizes.pilot():
        raise ValueError("The named R3 pilot profile has fixed 1000/500/1000/1000 episode counts.")
    if profile == "formal" and sizes != R3SyntheticSizes.formal():
        raise ValueError("The named R3 formal profile has fixed 5000/500/1000/1000 episode counts.")
    if profile in {"pilot", "formal"} and seed != R3_SYNTHETIC_SEED:
        raise ValueError(f"The named R3 {profile} profile is preregistered with seed {R3_SYNTHETIC_SEED}.")
    split_episodes, sidecar, blueprint = build_r3_synthetic(sizes=sizes, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    statistics: dict[str, Any] = {}
    for split in ("train", "dev", "test_id", "test_ood"):
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, split_episodes[split])
        artifacts[path.name] = {
            "sha256": _sha256_file(path),
            "count": len(split_episodes[split]),
            "model_visible": True,
        }
        statistics[split] = _split_statistics(split_episodes[split])
    sidecar_path = output_dir / "train_analysis_teacher_sidecar.jsonl"
    _write_jsonl_records(sidecar_path, sidecar)
    artifacts[sidecar_path.name] = {
        "sha256": _sha256_file(sidecar_path),
        "count": len(sidecar),
        "model_visible": False,
        "analysis_only": True,
        "split": "train",
    }
    manifest = {
        **blueprint,
        "profile": profile,
        "statistics": statistics,
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _last_query(episode: Episode) -> QuerySpec:
    for turn in reversed(episode.turns):
        if turn.query is not None:
            return turn.query
    raise ValueError(f"Episode {episode.episode_id!r} has no query.")


def validate_r3_synthetic(
    dataset_dir: Path,
    *,
    expected_sizes: Mapping[str, int] | None = None,
    balance_tolerance: float = 0.02,
    verify_manifest_hashes: bool = True,
) -> R3SyntheticValidationReport:
    """Fail closed on split leakage, mixed ordering, pairs, balance, and sidecar isolation."""

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing R3 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != R3_SYNTHETIC_SCHEMA:
        raise ValueError("Unexpected R3 synthetic manifest schema.")
    if manifest.get("split_before_expansion") is not True:
        raise ValueError("R3 manifest does not certify split-before-expansion construction.")
    expected = dict(expected_sizes or manifest.get("sizes", {}))
    split_episodes: dict[str, list[Episode]] = {}
    seen_episode_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    entity_splits: dict[str, set[str]] = defaultdict(set)
    template_splits: dict[str, set[str]] = defaultdict(set)
    split_group_sha: dict[str, str] = {}

    for split in ("train", "dev", "test_id", "test_ood"):
        path = dataset_dir / f"{split}.jsonl"
        raw_lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for line_number, value in enumerate(raw_lines, start=1):
            try:
                reject_hidden_ledger(value, path=f"{path.name}:{line_number}")
            except ValueError as exc:
                raise ValueError(f"Model-visible R3 JSONL contains a forbidden ledger: {exc}") from exc
            if "semantic_group_id" not in value:
                raise ValueError(f"{path.name}:{line_number} lacks semantic_group_id.")
        episodes = read_jsonl(path)
        split_episodes[split] = episodes
        if len(episodes) != int(expected[split]):
            raise ValueError(f"R3 {split} has {len(episodes)} episodes, expected {expected[split]}.")
        for episode in episodes:
            if episode.split != split:
                raise ValueError(f"Episode {episode.episode_id!r} is serialized under the wrong split.")
            if episode.episode_id in seen_episode_ids:
                raise ValueError(f"Duplicate cross-split episode_id: {episode.episode_id}")
            seen_episode_ids.add(episode.episode_id)
            if episode.semantic_group_id is None:
                raise ValueError(f"Episode {episode.episode_id!r} lacks semantic_group_id.")
            group_splits[episode.semantic_group_id].add(split)
            entity_splits[episode.entity_id].add(split)
            if episode.template_family is None:
                raise ValueError(f"Episode {episode.episode_id!r} lacks template_family.")
            template_splits[episode.template_family].add(split)
            if split != "test_ood" and not 4 <= len(episode.turns) <= 8:
                raise ValueError(f"ID episode {episode.episode_id!r} must contain 4--8 turns.")
            if split == "test_ood" and episode.ood_group == "heldout_length":
                if not 9 <= len(episode.turns) <= 16:
                    raise ValueError(f"Length-OOD episode {episode.episode_id!r} must contain 9--16 turns.")
            elif split == "test_ood" and not 4 <= len(episode.turns) <= 8:
                raise ValueError(f"Non-length OOD episode {episode.episode_id!r} must contain 4--8 turns.")
            for index, turn in enumerate(episode.turns):
                if turn.type is not TurnType.MIXED:
                    continue
                if index + 1 >= len(episode.turns) or episode.turns[index + 1].type is not TurnType.QUERY:
                    raise ValueError(
                        f"Mixed turn in {episode.episode_id!r} lacks an immediate delayed pure-query probe."
                    )
                assert turn.query is not None
                delayed = episode.turns[index + 1].query
                if delayed is None or delayed.target != turn.query.target:
                    raise ValueError(f"Mixed/delayed targets disagree in {episode.episode_id!r}.")

        group_ids = sorted({episode.semantic_group_id for episode in episodes if episode.semantic_group_id})
        split_group_sha[split] = _sha256_value(group_ids)
        expected_group_sha = manifest["group_stage"][split]["semantic_group_ids_sha256"]
        if split_group_sha[split] != expected_group_sha:
            raise ValueError(f"R3 {split} semantic-group stage SHA mismatch.")

    for label, mapping in (
        ("semantic group", group_splits),
        ("entity", entity_splits),
        ("template family", template_splits),
    ):
        leaked = {key: sorted(value) for key, value in mapping.items() if len(value) != 1}
        if leaked:
            raise ValueError(f"R3 split leakage through {label}: {next(iter(leaked.items()))!r}")

    for split, episodes in split_episodes.items():
        by_id = {episode.episode_id: episode for episode in episodes}
        for episode in episodes:
            mate = by_id.get(episode.counterfactual_episode_id)
            if mate is None:
                raise ValueError(f"Counterfactual link for {episode.episode_id!r} leaves split {split}.")
            if mate.counterfactual_episode_id != episode.episode_id or mate.pair_id != episode.pair_id:
                raise ValueError(f"Counterfactual link for {episode.episode_id!r} is not reciprocal.")
            if mate.semantic_group_id != episode.semantic_group_id:
                raise ValueError("Counterfactual members crossed a semantic group.")
            left_final, right_final = _last_query(episode), _last_query(mate)
            if left_final.target == right_final.target:
                raise ValueError(f"Counterfactual pair {episode.pair_id!r} has the same delayed target.")
            if set(left_final.choices) != set(right_final.choices):
                raise ValueError(f"Counterfactual pair {episode.pair_id!r} has different candidate sets.")
            if episode.distractor_variant in {DistractorVariant.CLEAN, DistractorVariant.DISTRACTOR}:
                distractor = by_id.get(str(episode.distractor_episode_id))
                if distractor is None or distractor.distractor_episode_id != episode.episode_id:
                    raise ValueError(f"Clean/no-op link for {episode.episode_id!r} is not reciprocal.")
                if distractor.semantic_group_id != episode.semantic_group_id:
                    raise ValueError("Clean/no-op members crossed a semantic group.")
                if distractor.distractor_pair_id != episode.distractor_pair_id:
                    raise ValueError("Clean/no-op pair_id mismatch.")
                if _last_query(distractor).target != left_final.target:
                    raise ValueError("Clean/no-op pair changed its semantic delayed target.")

        stats = _split_statistics(episodes)
        if stats["max_target_position_deviation"] > balance_tolerance:
            raise ValueError(
                f"R3 {split} target-position deviation {stats['max_target_position_deviation']:.6f} "
                f"exceeds {balance_tolerance:.6f}."
            )
        if stats["mixed_turn_count"] != stats["mixed_with_immediate_delayed_probe"]:
            raise ValueError(f"R3 {split} contains a mixed turn without delayed persistence evidence.")
        if set(stats["event_kind_counts"]) != {kind.value for kind in EventKind}:
            raise ValueError(f"R3 {split} does not cover set/overwrite/clear/noop.")
        if split == "test_ood":
            expected_per_group = expected[split] // len(OOD_GROUPS)
            if stats["ood_group_counts"] != {group: expected_per_group for group in OOD_GROUPS}:
                raise ValueError("R3 test_ood is not evenly stratified across the four OOD groups.")

    sidecar_path = dataset_dir / "train_analysis_teacher_sidecar.jsonl"
    sidecar_rows = [json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_by_id = {episode.episode_id: episode for episode in split_episodes["train"]}
    expected_sidecar_keys = {
        (episode.episode_id, turn_id)
        for episode in split_episodes["train"]
        for turn_id, turn in enumerate(episode.turns)
        if turn.calls_updater
    }
    observed_sidecar_keys: set[tuple[str, int]] = set()
    sidecar_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_sidecar_fields = {
        "schema_version",
        "split",
        "episode_id",
        "turn_id",
        "event_kind",
        "before_state",
        "after_state",
    }
    for row in sidecar_rows:
        if set(row) != expected_sidecar_fields:
            raise ValueError("R3 state sidecar differs from the locked teacher-cache input schema.")
        if row.get("schema_version") != R3_ANALYSIS_SIDECAR_SCHEMA or row.get("split") != "train":
            raise ValueError("R3 state sidecar must use the locked schema and remain train-only.")
        episode = train_by_id.get(str(row.get("episode_id")))
        if episode is None:
            raise ValueError("R3 sidecar references a non-train or unknown episode.")
        turn_id = int(row["turn_id"])
        key = (episode.episode_id, turn_id)
        if key in observed_sidecar_keys:
            raise ValueError(f"Duplicate R3 sidecar transition: {key!r}")
        observed_sidecar_keys.add(key)
        sidecar_by_episode[episode.episode_id].append(row)
        if not 0 <= turn_id < len(episode.turns) or not episode.turns[turn_id].calls_updater:
            raise ValueError("R3 sidecar references a non-updater turn.")
        turn_kind = episode.turns[turn_id].event_kind
        if turn_kind is None or row.get("event_kind") != turn_kind.value:
            raise ValueError("R3 sidecar event_kind disagrees with the routed updater turn.")
        if episode.turns[turn_id].event_kind is EventKind.NOOP:
            if row.get("before_state") != row.get("after_state"):
                raise ValueError("R3 no-op transition changed the analysis-only semantic state.")
    if observed_sidecar_keys != expected_sidecar_keys:
        missing = sorted(expected_sidecar_keys - observed_sidecar_keys)
        extra = sorted(observed_sidecar_keys - expected_sidecar_keys)
        raise ValueError(f"R3 train sidecar coverage mismatch: missing={missing[:1]}, extra={extra[:1]}.")
    for episode in split_episodes["train"]:
        records = sorted(sidecar_by_episode[episode.episode_id], key=lambda row: int(row["turn_id"]))
        for previous, current_record in zip(records, records[1:]):
            if previous["after_state"] != current_record["before_state"]:
                raise ValueError("R3 sidecar transitions are not a continuous semantic-state path.")
        final_entry = records[-1]["after_state"]["entries"][0]
        final_target = _last_query(episode).target
        if final_target == NO_ACTIVE_PREFERENCE:
            if final_entry.get("status") != "cleared" or final_entry.get("value_text") is not None:
                raise ValueError("R3 sidecar clear state disagrees with its delayed target.")
        elif final_entry.get("status") != "active" or final_entry.get("value_text") != final_target:
            raise ValueError("R3 sidecar active state disagrees with its delayed target.")
        if episode.distractor_variant is DistractorVariant.CLEAN:
            paired_records = sorted(
                sidecar_by_episode[str(episode.distractor_episode_id)],
                key=lambda row: int(row["turn_id"]),
            )
            if records[-1]["after_state"] != paired_records[-1]["after_state"]:
                raise ValueError("R3 clean/no-op pair does not end in an identical semantic state.")
    if any(
        "teacher" in episode.to_dict() or "sidecar" in episode.to_dict()
        for episodes in split_episodes.values()
        for episode in episodes
    ):
        raise ValueError("A model-visible episode contains a teacher/sidecar reference.")

    if verify_manifest_hashes:
        for name, artifact in manifest["artifacts"].items():
            path = dataset_dir / name
            if _sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"R3 artifact SHA mismatch: {name}")

    statistics = {split: _split_statistics(episodes) for split, episodes in split_episodes.items()}
    return R3SyntheticValidationReport(
        valid=True,
        total_episodes=sum(len(episodes) for episodes in split_episodes.values()),
        split_statistics=statistics,
        split_group_sha256=split_group_sha,
        train_sidecar_records=len(sidecar_rows),
    )


__all__ = [
    "NO_ACTIVE_PREFERENCE",
    "OOD_GROUPS",
    "R3SyntheticSizes",
    "R3SyntheticValidationReport",
    "build_r3_synthetic",
    "generate_r3_synthetic",
    "stage_semantic_groups",
    "validate_r3_synthetic",
]
