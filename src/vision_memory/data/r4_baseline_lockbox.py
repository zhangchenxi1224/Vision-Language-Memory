"""Prospective, answer-safe lockbox data for the R4 Qwen text baselines.

R4 data are generated once with a fixed seed and written fail-closed.  Formal
episodes reuse the audited R3 semantic construction, then receive a
deterministic R4 namespace and natural-language surface remap.  The discarded
privileged return value from the R3 builder is never serialized or referenced
by any R4 artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .r3_synthetic import (
    HELDOUT_TOPIC_VALUES,
    NO_ACTIVE_PREFERENCE,
    TOPIC_VALUES,
    R3SyntheticSizes,
    build_r3_synthetic,
)
from .schema import (
    DistractorVariant,
    Episode,
    EventKind,
    QuerySpec,
    Turn,
    TurnType,
    reject_hidden_ledger,
    write_jsonl,
)


R4_BASELINE_LOCKBOX_SCHEMA = "vlm.r4.qwen-baseline-lockbox.v1"
R4_BASELINE_SEED = 20260722
R4_FORMAL_SIZES = R3SyntheticSizes.formal()
R4_ARTIFACT_NAMES = (
    "smoke4.jsonl",
    "transition32.jsonl",
    "formal_train.jsonl",
    "formal_dev.jsonl",
    "formal_test_id.jsonl",
    "formal_test_ood.jsonl",
)
R4_TERMINAL_KINDS = (
    EventKind.SET,
    EventKind.OVERWRITE,
    EventKind.CLEAR,
    EventKind.NOOP,
)
R4_READ_FORMS = ("separate", "mixed")
R4_HISTORY_LENGTHS = ("short", "long")
R4_MICRO_VALUES = ("teal", "burgundy", "ivory", NO_ACTIVE_PREFERENCE)


_R4_TOPIC_NAMES = {
    "color": "accent",
    "material": "finish",
    "drink": "beverage",
    "style": "aesthetic",
    "meal": "entree",
    "music": "soundtrack",
    "fragrance": "aroma",
    "lighting": "illumination",
}
_R4_VALUE_BANKS: dict[str, tuple[str, ...]] = {
    "color": ("teal", "burgundy", "ivory", "charcoal", "coral", "turquoise"),
    "material": ("bamboo", "acrylic", "titanium", "porcelain", "denim", "suede"),
    "drink": ("kombucha", "seltzer", "lemonade", "smoothie", "chicory", "oatmilk"),
    "style": ("geometric", "antique", "contemporary", "coastal", "ceremonial", "whimsical"),
    "meal": ("risotto", "gazpacho", "tagine", "ramen", "burrito", "pilaf"),
    "music": ("bebop", "baroque", "bluegrass", "metal", "drone", "soul"),
    "fragrance": ("bergamot", "sandalwood", "tonka", "eucalyptus", "jasmine", "peony"),
    "lighting": ("amber", "daylight", "low", "radiant", "diffuse", "spotlight"),
}
_R4_BASE_NOUNS = ("carafe", "sconce", "journal", "satchel", "stool", "console", "workbench", "studio")
_R4_HELDOUT_NOUNS = ("sextant", "cello", "conservatory", "bust", "rover", "canoe")
_R4_ADJECTIVES = ("azure", "burlap", "brass", "velvet", "granite", "platinum", "cedar", "umber")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_value(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_int(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).digest()[:8],
        "big",
    )


def _r4_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    if "r3-" not in value:
        raise ValueError(f"Expected an R3 namespace identifier, got {value!r}.")
    return value.replace("r3-", "r4-")


def _replace_surface(text: str, replacements: Sequence[tuple[str, str]]) -> str:
    result = text
    for source, target in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        if not source or source == target:
            continue
        result = re.sub(rf"(?<![\w-]){re.escape(source)}(?![\w-])", target, result)
    return result


def _formal_entity_surface(episode: Episode) -> str:
    group_id = cast(str, _r4_identifier(episode.semantic_group_id))
    nouns = _R4_HELDOUT_NOUNS if episode.ood_group == "heldout_entity" else _R4_BASE_NOUNS
    adjective = _R4_ADJECTIVES[_stable_int(R4_BASELINE_SEED, group_id, "adjective") % len(_R4_ADJECTIVES)]
    noun = nouns[_stable_int(R4_BASELINE_SEED, group_id, "noun") % len(nouns)]
    suffix = hashlib.sha256(group_id.encode("utf-8")).hexdigest()[:8]
    return f"the {adjective} {noun} r4 {episode.split.replace('_', '-')} {suffix}"


def _value_remap(topic: str) -> dict[str, str]:
    source = TOPIC_VALUES.get(topic) or HELDOUT_TOPIC_VALUES.get(topic)
    target = _R4_VALUE_BANKS.get(topic)
    if source is None or target is None or len(source) != len(target):
        raise ValueError(f"R4 has no complete surface remap for topic {topic!r}.")
    return {**dict(zip(source, target, strict=True)), NO_ACTIVE_PREFERENCE: NO_ACTIVE_PREFERENCE}


def remap_r3_episode_to_r4(episode: Episode, *, seed: int = R4_BASELINE_SEED) -> Episode:
    """Create an R4-only episode without consulting any privileged state."""

    if seed != R4_BASELINE_SEED:
        raise ValueError(f"The R4 baseline lockbox is fixed to seed {R4_BASELINE_SEED}.")
    if episode.entity_surface is None or episode.template_family is None:
        raise ValueError("Formal source episodes must expose their non-privileged text surfaces.")
    new_entity = _formal_entity_surface(episode)
    new_topic = _R4_TOPIC_NAMES[episode.topic]
    values = _value_remap(episode.topic)
    old_marker = episode.template_family.replace("-", " ").title()
    new_template_family = cast(str, _r4_identifier(episode.template_family))
    new_marker = new_template_family.replace("-", " ").title()
    replacements = [
        (old_marker, new_marker),
        (episode.entity_surface, new_entity),
        (episode.topic, new_topic),
        *values.items(),
    ]

    turns: list[Turn] = []
    for turn in episode.turns:
        event_text = (
            _replace_surface(turn.event_text, replacements) if turn.event_text is not None else None
        )
        query: QuerySpec | None = None
        if turn.query is not None:
            remapped_choices = tuple(values.get(choice, choice) for choice in turn.query.choices)
            if len(set(remapped_choices)) != 4:
                raise ValueError("R4 value remap collapsed distinct choices.")
            query = QuerySpec(
                text=_replace_surface(turn.query.text, replacements),
                choices=cast(tuple[str, str, str, str], remapped_choices),
                target_index=turn.query.target_index,
                target_token_count=turn.query.target_token_count,
                comparison_id=_r4_identifier(turn.query.comparison_id),
            )
        turns.append(Turn(turn.type, turn.event_kind, event_text, query))

    remapped = Episode(
        episode_id=cast(str, _r4_identifier(episode.episode_id)),
        split=episode.split,
        seed=seed,
        entity_id=cast(str, _r4_identifier(episode.entity_id)),
        template_id=cast(str, _r4_identifier(episode.template_id)),
        turns=tuple(turns),
        pair_id=cast(str, _r4_identifier(episode.pair_id)),
        counterfactual_episode_id=cast(str, _r4_identifier(episode.counterfactual_episode_id)),
        topic=new_topic,
        semantic_group_id=_r4_identifier(episode.semantic_group_id),
        ood_group=episode.ood_group,
        entity_surface=new_entity,
        template_family=new_template_family,
        distractor_variant=episode.distractor_variant,
        distractor_pair_id=_r4_identifier(episode.distractor_pair_id),
        distractor_episode_id=_r4_identifier(episode.distractor_episode_id),
    )
    payload = remapped.to_dict()
    reject_hidden_ledger(payload)
    if "r3-" in json.dumps(payload, ensure_ascii=False).casefold():
        raise ValueError("R3 namespace survived the deterministic R4 remap.")
    return remapped


def build_r4_formal_episodes(
    *,
    sizes: R3SyntheticSizes = R4_FORMAL_SIZES,
    seed: int = R4_BASELINE_SEED,
) -> tuple[dict[str, tuple[Episode, ...]], str]:
    """Build formal episodes and return the hash of the non-privileged source contract."""

    if seed != R4_BASELINE_SEED:
        raise ValueError(f"The R4 baseline lockbox is fixed to seed {R4_BASELINE_SEED}.")
    source_splits, _discarded_privileged_records, source_blueprint = build_r3_synthetic(
        sizes=sizes,
        seed=seed,
    )
    source_contract_sha256 = _sha256_value(source_blueprint)
    remapped = {
        split: tuple(remap_r3_episode_to_r4(episode, seed=seed) for episode in episodes)
        for split, episodes in source_splits.items()
    }
    return remapped, source_contract_sha256


def _micro_event_text(
    entity: str,
    kind: EventKind,
    value: str | None,
    *,
    lexical_replica: int,
) -> str:
    if kind is EventKind.SET:
        forms = (
            f"For {entity}, store {value} as the current accent preference.",
            f"Remember that {entity} currently favors the {value} accent.",
        )
    elif kind is EventKind.OVERWRITE:
        forms = (
            f"Replace the earlier accent preference for {entity} with {value}.",
            f"The accent preference for {entity} is now {value}, superseding the prior value.",
        )
    elif kind is EventKind.CLEAR:
        forms = (
            f"Clear the saved accent preference for {entity}.",
            f"Treat {entity} as having no active accent preference now.",
        )
    else:
        forms = (
            "An unrelated observatory window was cleaned this morning.",
            "An unrelated delivery cart was moved to the west corridor.",
        )
    return f"R4 Lockbox Event: {forms[lexical_replica % 2]}"


def _micro_query(
    entity: str,
    *,
    target: str,
    choices: tuple[str, str, str, str],
    comparison_id: str,
    immediate: bool,
) -> QuerySpec:
    timing = "After applying the event in this same message" if immediate else "At this later check"
    return QuerySpec(
        text=(
            f"R4 Lockbox Query: {timing}, what is the current accent preference for {entity}? "
            "Choose exactly one option."
        ),
        choices=choices,
        target_index=choices.index(target),
        comparison_id=comparison_id,
    )


def _rotate(values: Sequence[str], phase: int) -> tuple[str, ...]:
    offset = phase % len(values)
    return tuple(values[offset:]) + tuple(values[:offset])


def _transition_event_specs(
    kind: EventKind,
    history_length: str,
    *,
    target: str,
    replica: int,
) -> tuple[tuple[EventKind, str | None], ...]:
    stale_a = ("ivory", "teal")[replica]
    stale_b = ("burgundy", "ivory")[replica]
    if history_length not in R4_HISTORY_LENGTHS:
        raise ValueError(f"Unsupported R4 history length: {history_length!r}.")
    if history_length == "short":
        if kind is EventKind.SET:
            return ((EventKind.SET, target),)
        if kind is EventKind.NOOP:
            return ((EventKind.SET, target), (EventKind.NOOP, None))
        return ((EventKind.SET, stale_a), (kind, None if kind is EventKind.CLEAR else target))

    clean_prefix = (
        (EventKind.SET, stale_a),
        (EventKind.OVERWRITE, stale_b),
    )
    if kind is EventKind.SET:
        return (*clean_prefix, (EventKind.SET, target))
    if kind is EventKind.NOOP:
        return (*clean_prefix, (EventKind.SET, target), (EventKind.NOOP, None))
    return (*clean_prefix, (EventKind.NOOP, None), (kind, None if kind is EventKind.CLEAR else target))


def _build_transition_episode(
    *,
    suite: str,
    terminal_kind: EventKind,
    read_form: str,
    history_length: str,
    lexical_replica: int,
) -> Episode:
    if read_form not in R4_READ_FORMS:
        raise ValueError(f"Unsupported R4 read form: {read_form!r}.")
    target = NO_ACTIVE_PREFERENCE if terminal_kind is EventKind.CLEAR else ("teal", "burgundy")[lexical_replica]
    group = f"r4-{suite}-{read_form}-{history_length}-r{lexical_replica}"
    entity_id = f"{group}-entity"
    entity = (
        f"the {read_form} brass carafe {history_length} r4",
        f"the {read_form} velvet journal {history_length} r4",
    )[lexical_replica]
    episode_id = f"{group}-{terminal_kind.value}"
    mate_kind = {
        EventKind.SET: EventKind.OVERWRITE,
        EventKind.OVERWRITE: EventKind.SET,
        EventKind.CLEAR: EventKind.NOOP,
        EventKind.NOOP: EventKind.CLEAR,
    }[terminal_kind]
    # Every state-swap donor must have a different answer. SET/OVERWRITE
    # therefore cross lexical replicas; CLEAR/NOOP pair within a replica.
    mate_replica = (
        1 - lexical_replica
        if terminal_kind in {EventKind.SET, EventKind.OVERWRITE}
        else lexical_replica
    )
    mate_group = f"r4-{suite}-{read_form}-{history_length}-r{mate_replica}"
    mate_episode_id = f"{mate_group}-{mate_kind.value}"
    phase = _stable_int(R4_BASELINE_SEED, suite, read_form, history_length, lexical_replica) % 4
    choices = cast(tuple[str, str, str, str], _rotate(R4_MICRO_VALUES, phase))
    comparison = f"{group}:delayed"
    specs = _transition_event_specs(
        terminal_kind,
        history_length,
        target=target,
        replica=lexical_replica,
    )
    turns: list[Turn] = []
    for index, (kind, value) in enumerate(specs):
        text = _micro_event_text(entity, kind, value, lexical_replica=lexical_replica)
        is_final = index == len(specs) - 1
        if is_final and read_form == "mixed":
            turns.append(
                Turn(
                    TurnType.MIXED,
                    kind,
                    text,
                    _micro_query(
                        entity,
                        target=target,
                        choices=choices,
                        comparison_id=f"{group}:immediate",
                        immediate=True,
                    ),
                )
            )
        else:
            turns.append(Turn(TurnType.EVENT, kind, text))
    turns.append(
        Turn(
            TurnType.QUERY,
            query=_micro_query(
                entity,
                target=target,
                choices=choices,
                comparison_id=comparison,
                immediate=False,
            ),
        )
    )
    common = {
        "distractor_variant": None,
        "distractor_pair_id": None,
        "distractor_episode_id": None,
    }
    if terminal_kind in {EventKind.SET, EventKind.NOOP}:
        donor_kind = EventKind.NOOP if terminal_kind is EventKind.SET else EventKind.SET
        common = {
            "distractor_variant": (
                DistractorVariant.CLEAN if terminal_kind is EventKind.SET else DistractorVariant.DISTRACTOR
            ),
            "distractor_pair_id": f"{group}-clean-noop-pair",
            "distractor_episode_id": f"{group}-{donor_kind.value}",
        }
    return Episode(
        episode_id=episode_id,
        split="lockbox",
        seed=R4_BASELINE_SEED,
        entity_id=entity_id,
        entity_surface=entity,
        template_id=f"r4-{suite}-{read_form}-{history_length}-query-b",
        template_family=f"r4-{suite}-templates-b",
        turns=tuple(turns),
        pair_id="r4-counterfactual:" + ":".join(sorted((episode_id, mate_episode_id))),
        counterfactual_episode_id=mate_episode_id,
        topic="accent",
        semantic_group_id=group,
        distractor_variant=common["distractor_variant"],
        distractor_pair_id=common["distractor_pair_id"],
        distractor_episode_id=common["distractor_episode_id"],
    )


def build_smoke4(*, seed: int = R4_BASELINE_SEED) -> tuple[Episode, ...]:
    if seed != R4_BASELINE_SEED:
        raise ValueError(f"R4 Smoke4 is fixed to seed {R4_BASELINE_SEED}.")
    return tuple(
        _build_transition_episode(
            suite="smoke4",
            terminal_kind=kind,
            read_form="separate",
            history_length="short",
            lexical_replica=0,
        )
        for index, kind in enumerate(R4_TERMINAL_KINDS)
    )


def build_transition32(*, seed: int = R4_BASELINE_SEED) -> tuple[Episode, ...]:
    if seed != R4_BASELINE_SEED:
        raise ValueError(f"R4 Transition32 is fixed to seed {R4_BASELINE_SEED}.")
    return tuple(
        _build_transition_episode(
            suite="transition32",
            terminal_kind=kind,
            read_form=read_form,
            history_length=history_length,
            lexical_replica=replica,
        )
        for kind in R4_TERMINAL_KINDS
        for read_form in R4_READ_FORMS
        for history_length in R4_HISTORY_LENGTHS
        for replica in range(2)
    )


def _artifact_statistics(episodes: Sequence[Episode]) -> dict[str, Any]:
    return {
        "count": len(episodes),
        "query_count": sum(episode.query_count for episode in episodes),
        "update_count": sum(episode.update_count for episode in episodes),
        "event_kind_counts": dict(
            sorted(
                Counter(
                    turn.event_kind.value
                    for episode in episodes
                    for turn in episode.turns
                    if turn.event_kind is not None
                ).items()
            )
        ),
        "mixed_turn_count": sum(episode.mixed_query_count for episode in episodes),
    }


@dataclass(frozen=True)
class R4BaselineLockbox:
    artifacts: Mapping[str, tuple[Episode, ...]]
    source_contract_sha256: str


def build_r4_baseline_lockbox(*, seed: int = R4_BASELINE_SEED) -> R4BaselineLockbox:
    if seed != R4_BASELINE_SEED:
        raise ValueError(f"The R4 baseline lockbox is fixed to seed {R4_BASELINE_SEED}.")
    formal, source_contract_sha256 = build_r4_formal_episodes(seed=seed)
    artifacts = {
        "smoke4.jsonl": build_smoke4(seed=seed),
        "transition32.jsonl": build_transition32(seed=seed),
        **{f"formal_{split}.jsonl": episodes for split, episodes in formal.items()},
    }
    if tuple(artifacts) != R4_ARTIFACT_NAMES:
        raise RuntimeError("R4 artifact inventory drifted from its locked order.")
    return R4BaselineLockbox(artifacts=artifacts, source_contract_sha256=source_contract_sha256)


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"R4 output path already exists and is not a directory: {output_dir}")
        existing = sorted(path.name for path in output_dir.iterdir())
        if existing:
            raise FileExistsError(
                f"R4 lockbox generation refuses to overwrite non-empty directory {output_dir}: {existing[:3]}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def generate_r4_baseline_lockbox(
    output_dir: Path,
    *,
    seed: int = R4_BASELINE_SEED,
) -> dict[str, Any]:
    """Materialize the fixed R4 lockbox exactly once and bind every artifact."""

    _prepare_output_dir(output_dir)
    suite = build_r4_baseline_lockbox(seed=seed)
    artifact_manifest: dict[str, Any] = {}
    for name, episodes in suite.artifacts.items():
        path = output_dir / name
        write_jsonl(path, episodes)
        raw = path.read_text(encoding="utf-8")
        forbidden = tuple(token for token in ("ledger", "teacher", "sidecar") if token in raw.casefold())
        if forbidden:
            raise RuntimeError(f"Forbidden privileged token(s) in {name}: {forbidden}")
        artifact_manifest[name] = {
            "sha256": _sha256_file(path),
            **_artifact_statistics(episodes),
        }

    contract = {
        "schema_version": R4_BASELINE_LOCKBOX_SCHEMA,
        "seed": seed,
        "artifact_order": list(R4_ARTIFACT_NAMES),
        "formal_sizes": R4_FORMAL_SIZES.as_dict(),
        "micro_contract": {
            "smoke4": "set/overwrite/clear/noop x 1",
            "transition32": "4 terminal kinds x 2 read forms x 2 history lengths x 2 lexical replicas",
            "mixed_order": "event is visible before immediate query; delayed pure-query probe follows",
        },
        "formal_contract": {
            "source_builder": "vision_memory.data.r3_synthetic.build_r3_synthetic",
            "source_seed": seed,
            "deterministic_r4_namespace_and_text_surface_remap": True,
            "privileged_artifacts_emitted": False,
            "source_contract_sha256": suite.source_contract_sha256,
        },
        "model_visible_episode_schema": "vision_memory.data.schema.Episode",
        "artifacts": artifact_manifest,
    }
    manifest = {
        **contract,
        "lockbox_contract_sha256": _sha256_value(contract),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return manifest
