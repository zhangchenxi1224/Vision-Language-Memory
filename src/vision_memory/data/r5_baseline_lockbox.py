"""Prospective same-entity lockbox for the R5 Qwen history baseline.

R5 changes exactly one scientific factor relative to R4: every explicit
state-swap counterfactual is query-compatible and remains inside the same
entity scope.  The formal R4 files were never evaluated, so their exact bytes
are inherited after SHA256 verification instead of being regenerated after an
observed micro-gate failure.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .r3_synthetic import NO_ACTIVE_PREFERENCE
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


R5_BASELINE_LOCKBOX_SCHEMA = "vlm.r5.qwen-baseline-lockbox.same-entity.v1"
R5_BASELINE_SEED = 20260723
R5_ARTIFACT_NAMES = (
    "smoke4.jsonl",
    "transition32.jsonl",
    "formal_train.jsonl",
    "formal_dev.jsonl",
    "formal_test_id.jsonl",
    "formal_test_ood.jsonl",
)
R5_TERMINAL_KINDS = (
    EventKind.SET,
    EventKind.OVERWRITE,
    EventKind.CLEAR,
    EventKind.NOOP,
)
R5_READ_FORMS = ("separate", "mixed")
R5_HISTORY_LENGTHS = ("short", "long")
R5_FORMAL_SIZES = {"train": 5000, "dev": 500, "test_id": 1000, "test_ood": 1000}

R4_PARENT_MANIFEST_SHA256 = "f3c5f235df9a3f026e3671ff2d330167fa7fd7ea39d520666f4624874209b321"
R4_INHERITED_FORMAL_SHA256 = {
    "formal_train.jsonl": "493a8668422a17bd94666b4a07198eaa7d38192b5e1e7a42f5d6db98496dddc7",
    "formal_dev.jsonl": "541da388a0261d119db8b77f7903ba1082b07c7ff34f9fc60acd6c2b3e09bde0",
    "formal_test_id.jsonl": "d1722dba096d1da8fee4ea4a994b026add8e0e704cb6cf71d132f83453e27079",
    "formal_test_ood.jsonl": "b193dadd116ecd1f112682c1d574da91eca55ae48601cc8f5b716a743e7c6f9f",
}

_PRIMARY_VALUES = ("cobalt", "plum")
_ALTERNATE_VALUES = ("saffron", "jade")
_TERTIARY_VALUES = ("linen", "copper")
_CHOICE_BANKS = tuple(
    (primary, alternate, tertiary, NO_ACTIVE_PREFERENCE)
    for primary, alternate, tertiary in zip(_PRIMARY_VALUES, _ALTERNATE_VALUES, _TERTIARY_VALUES, strict=True)
)


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
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _rotate(values: Sequence[str], phase: int) -> tuple[str, ...]:
    offset = phase % len(values)
    return tuple(values[offset:]) + tuple(values[:offset])


def r5_target_for(kind: EventKind, lexical_replica: int) -> str:
    if lexical_replica not in (0, 1):
        raise ValueError("R5 lexical_replica must be 0 or 1.")
    if kind is EventKind.CLEAR:
        return NO_ACTIVE_PREFERENCE
    if kind is EventKind.OVERWRITE:
        return _ALTERNATE_VALUES[lexical_replica]
    if kind in {EventKind.SET, EventKind.NOOP}:
        return _PRIMARY_VALUES[lexical_replica]
    raise ValueError(f"Unsupported R5 terminal kind: {kind!r}.")


def _micro_event_text(
    entity: str,
    kind: EventKind,
    value: str | None,
    *,
    lexical_replica: int,
) -> str:
    if kind is EventKind.SET:
        forms = (
            f"For {entity}, store {value} as the current trim preference.",
            f"Remember that {entity} currently favors the {value} trim.",
        )
    elif kind is EventKind.OVERWRITE:
        forms = (
            f"Replace the earlier trim preference for {entity} with {value}.",
            f"The trim preference for {entity} is now {value}, superseding the prior value.",
        )
    elif kind is EventKind.CLEAR:
        forms = (
            f"Clear the saved trim preference for {entity}.",
            f"Treat {entity} as having no active trim preference now.",
        )
    else:
        forms = (
            "An unrelated archive window was inspected this morning.",
            "An unrelated supply cart was moved to the north corridor.",
        )
    return f"R5 Same-Entity Lockbox Event: {forms[lexical_replica]}"


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
            f"R5 Same-Entity Lockbox Query: {timing}, what is the current trim preference "
            f"for {entity}? Choose exactly one option."
        ),
        choices=choices,
        target_index=choices.index(target),
        comparison_id=comparison_id,
    )


def _transition_event_specs(
    kind: EventKind,
    history_length: str,
    *,
    target: str,
    lexical_replica: int,
) -> tuple[tuple[EventKind, str | None], ...]:
    if history_length not in R5_HISTORY_LENGTHS:
        raise ValueError(f"Unsupported R5 history length: {history_length!r}.")
    stale_a = _TERTIARY_VALUES[lexical_replica]
    stale_b = _PRIMARY_VALUES[lexical_replica]
    if history_length == "short":
        if kind is EventKind.SET:
            return ((EventKind.SET, target),)
        if kind is EventKind.NOOP:
            return ((EventKind.SET, target), (EventKind.NOOP, None))
        return ((EventKind.SET, stale_a), (kind, None if kind is EventKind.CLEAR else target))

    clean_prefix = ((EventKind.SET, stale_a), (EventKind.OVERWRITE, stale_b))
    if kind is EventKind.SET:
        return (*clean_prefix, (EventKind.SET, target))
    if kind is EventKind.NOOP:
        return (*clean_prefix, (EventKind.SET, target), (EventKind.NOOP, None))
    return (*clean_prefix, (EventKind.NOOP, None), (kind, None if kind is EventKind.CLEAR else target))


def _mate_kind(kind: EventKind) -> EventKind:
    return {
        EventKind.SET: EventKind.OVERWRITE,
        EventKind.OVERWRITE: EventKind.SET,
        EventKind.CLEAR: EventKind.NOOP,
        EventKind.NOOP: EventKind.CLEAR,
    }[kind]


def _build_transition_episode(
    *,
    suite: str,
    terminal_kind: EventKind,
    read_form: str,
    history_length: str,
    lexical_replica: int,
) -> Episode:
    if read_form not in R5_READ_FORMS:
        raise ValueError(f"Unsupported R5 read form: {read_form!r}.")
    target = r5_target_for(terminal_kind, lexical_replica)
    group = f"r5-{suite}-{read_form}-{history_length}-r{lexical_replica}"
    entity_id = f"{group}-entity"
    entity = (
        f"the {read_form} cobalt astrolabe {history_length} r5",
        f"the {read_form} plum drafting case {history_length} r5",
    )[lexical_replica]
    episode_id = f"{group}-{terminal_kind.value}"
    mate_episode_id = f"{group}-{_mate_kind(terminal_kind).value}"
    phase = _stable_int(R5_BASELINE_SEED, suite, read_form, history_length, lexical_replica) % 4
    choices = cast(
        tuple[str, str, str, str],
        _rotate(_CHOICE_BANKS[lexical_replica], phase),
    )
    specs = _transition_event_specs(
        terminal_kind,
        history_length,
        target=target,
        lexical_replica=lexical_replica,
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
                comparison_id=f"{group}:delayed",
                immediate=False,
            ),
        )
    )

    distractor: dict[str, Any] = {
        "distractor_variant": None,
        "distractor_pair_id": None,
        "distractor_episode_id": None,
    }
    if terminal_kind in {EventKind.SET, EventKind.NOOP}:
        donor_kind = EventKind.NOOP if terminal_kind is EventKind.SET else EventKind.SET
        distractor = {
            "distractor_variant": (
                DistractorVariant.CLEAN if terminal_kind is EventKind.SET else DistractorVariant.DISTRACTOR
            ),
            "distractor_pair_id": f"{group}-clean-noop-pair",
            "distractor_episode_id": f"{group}-{donor_kind.value}",
        }
    return Episode(
        episode_id=episode_id,
        split="lockbox",
        seed=R5_BASELINE_SEED,
        entity_id=entity_id,
        entity_surface=entity,
        template_id=f"r5-{suite}-{read_form}-{history_length}-query-b-r{lexical_replica}",
        template_family=f"r5-{suite}-templates-b",
        turns=tuple(turns),
        pair_id="r5-same-entity-counterfactual:" + ":".join(sorted((episode_id, mate_episode_id))),
        counterfactual_episode_id=mate_episode_id,
        topic="trim",
        semantic_group_id=group,
        distractor_variant=distractor["distractor_variant"],
        distractor_pair_id=distractor["distractor_pair_id"],
        distractor_episode_id=distractor["distractor_episode_id"],
    )


def build_smoke4(*, seed: int = R5_BASELINE_SEED) -> tuple[Episode, ...]:
    if seed != R5_BASELINE_SEED:
        raise ValueError(f"R5 Smoke4 is fixed to seed {R5_BASELINE_SEED}.")
    episodes = tuple(
        _build_transition_episode(
            suite="smoke4",
            terminal_kind=kind,
            read_form="separate",
            history_length="short",
            lexical_replica=0,
        )
        for kind in R5_TERMINAL_KINDS
    )
    validate_same_entity_pair_contract(episodes, expected_delayed_states=4)
    return episodes


def build_transition32(*, seed: int = R5_BASELINE_SEED) -> tuple[Episode, ...]:
    if seed != R5_BASELINE_SEED:
        raise ValueError(f"R5 Transition32 is fixed to seed {R5_BASELINE_SEED}.")
    episodes = tuple(
        _build_transition_episode(
            suite="transition32",
            terminal_kind=kind,
            read_form=read_form,
            history_length=history_length,
            lexical_replica=replica,
        )
        for kind in R5_TERMINAL_KINDS
        for read_form in R5_READ_FORMS
        for history_length in R5_HISTORY_LENGTHS
        for replica in range(2)
    )
    validate_same_entity_pair_contract(episodes, expected_delayed_states=32)
    return episodes


def _query_snapshots(episode: Episode) -> tuple[dict[str, Any], ...]:
    events: list[tuple[str, str]] = []
    snapshots: list[dict[str, Any]] = []
    query_ordinal = 0
    for turn in episode.turns:
        if turn.calls_updater:
            if turn.event_kind is None or turn.event_text is None:
                raise ValueError("Updater turn lacks event kind/text.")
            events.append((turn.event_kind.value, turn.event_text))
        if not turn.calls_reader:
            continue
        if turn.query is None:
            raise ValueError("Reader turn lacks query payload.")
        query = turn.query
        probe_role = "immediate" if turn.calls_updater else "delayed"
        scope = {
            "entity_id": episode.entity_id,
            "entity_surface": episode.entity_surface,
            "topic": episode.topic,
            "query_text": query.text,
            "choices": list(query.choices),
            "probe_role": probe_role,
            "query_ordinal": query_ordinal,
        }
        snapshots.append(
            {
                "query_scope_sha256": _sha256_value(scope),
                "query_text": query.text,
                "choices": query.choices,
                "target": query.target,
                "probe_role": probe_role,
                "events": tuple(events),
            }
        )
        query_ordinal += 1
    return tuple(snapshots)


def validate_same_entity_pair_contract(
    episodes: Sequence[Episode],
    *,
    expected_delayed_states: int,
) -> dict[str, Any]:
    """Fail closed unless every donor changes only the visible event history."""

    by_id = {episode.episode_id: episode for episode in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("R5 pair contract requires unique episode IDs.")
    delayed_states = 0
    query_pairs = 0
    pair_map: list[dict[str, Any]] = []
    for episode in episodes:
        donor_id = episode.counterfactual_episode_id
        donor = by_id.get(donor_id or "")
        if donor is None or donor is episode:
            raise ValueError(f"R5 donor is missing or self-referential for {episode.episode_id}.")
        if donor.counterfactual_episode_id != episode.episode_id or donor.pair_id != episode.pair_id:
            raise ValueError(f"R5 donor relation is not reciprocal for {episode.episode_id}.")
        same_scope = (
            episode.entity_id == donor.entity_id
            and episode.entity_surface == donor.entity_surface
            and episode.topic == donor.topic
            and episode.semantic_group_id == donor.semantic_group_id
            and episode.template_id == donor.template_id
            and episode.template_family == donor.template_family
        )
        if not same_scope:
            raise ValueError(f"R5 donor crossed entity/query scope for {episode.episode_id}.")
        recipient_queries = _query_snapshots(episode)
        donor_queries = _query_snapshots(donor)
        if len(recipient_queries) != len(donor_queries):
            raise ValueError(f"R5 donor query count mismatch for {episode.episode_id}.")
        for ordinal, (recipient_query, donor_query) in enumerate(zip(recipient_queries, donor_queries, strict=True)):
            if recipient_query["query_scope_sha256"] != donor_query["query_scope_sha256"]:
                raise ValueError(f"R5 donor changed query scope for {episode.episode_id}:q{ordinal}.")
            if recipient_query["target"] == donor_query["target"]:
                raise ValueError(f"R5 donor target must differ for {episode.episode_id}:q{ordinal}.")
            if tuple(recipient_query["choices"]).count(donor_query["target"]) != 1:
                raise ValueError(
                    f"R5 donor target must map uniquely into recipient choices for {episode.episode_id}:q{ordinal}."
                )
            if recipient_query["events"] == donor_query["events"]:
                raise ValueError(f"R5 donor event stream must differ for {episode.episode_id}:q{ordinal}.")
            query_pairs += 1
            delayed_states += int(recipient_query["probe_role"] == "delayed")
        pair_map.append(
            {
                "recipient_episode_id": episode.episode_id,
                "donor_episode_id": donor.episode_id,
                "entity_id": episode.entity_id,
                "pair_id": episode.pair_id,
                "query_scope_sha256": [query["query_scope_sha256"] for query in recipient_queries],
            }
        )
    if delayed_states != expected_delayed_states:
        raise ValueError(f"R5 expected {expected_delayed_states} delayed states, found {delayed_states}.")
    return {
        "schema": "vlm.r5.same-entity-pair-audit.v1",
        "episode_count": len(episodes),
        "query_pair_count": query_pairs,
        "delayed_state_count": delayed_states,
        "reciprocal_pairs_valid": True,
        "same_entity_query_scope_valid": True,
        "different_target_valid": True,
        "donor_event_stream_differs": True,
        "ordered_pair_map_sha256": _sha256_value(pair_map),
    }


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


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"R5 output path is not a directory: {output_dir}")
        existing = sorted(path.name for path in output_dir.iterdir())
        if existing:
            raise FileExistsError(
                f"R5 lockbox generation refuses to overwrite non-empty directory {output_dir}: {existing[:3]}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)


def _verified_r4_formal_source(formal_source_dir: Path) -> tuple[dict[str, Any], str]:
    manifest_path = formal_source_dir / "manifest.json"
    if _sha256_file(manifest_path) != R4_PARENT_MANIFEST_SHA256:
        raise ValueError("R4 parent manifest SHA256 does not match the sealed inheritance contract.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("R4 parent manifest must be a JSON object.")
    for name, expected_sha256 in R4_INHERITED_FORMAL_SHA256.items():
        path = formal_source_dir / name
        if _sha256_file(path) != expected_sha256:
            raise ValueError(f"Inherited formal artifact SHA256 mismatch: {name}")
        recorded = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        if recorded != expected_sha256:
            raise ValueError(f"R4 parent manifest does not bind inherited artifact: {name}")
    source_contract_sha256 = manifest.get("formal_contract", {}).get("source_contract_sha256")
    if not isinstance(source_contract_sha256, str):
        raise ValueError("R4 parent manifest lacks its formal source contract SHA256.")
    return dict(manifest), source_contract_sha256


def generate_r5_baseline_lockbox(
    output_dir: Path,
    *,
    formal_source_dir: Path,
    seed: int = R5_BASELINE_SEED,
) -> dict[str, Any]:
    """Generate new R5 micro data and byte-inherit the still-unseen formal files."""

    if seed != R5_BASELINE_SEED:
        raise ValueError(f"The R5 lockbox is fixed to seed {R5_BASELINE_SEED}.")
    _prepare_output_dir(output_dir)
    parent_manifest, source_contract_sha256 = _verified_r4_formal_source(formal_source_dir)
    smoke = build_smoke4(seed=seed)
    transition = build_transition32(seed=seed)
    audits = {
        "smoke4": validate_same_entity_pair_contract(smoke, expected_delayed_states=4),
        "transition32": validate_same_entity_pair_contract(transition, expected_delayed_states=32),
    }
    artifact_manifest: dict[str, Any] = {}
    for name, episodes in (("smoke4.jsonl", smoke), ("transition32.jsonl", transition)):
        path = output_dir / name
        write_jsonl(path, episodes)
        raw = path.read_text(encoding="utf-8")
        forbidden = tuple(token for token in ("ledger", "teacher", "sidecar") if token in raw.casefold())
        if forbidden:
            raise RuntimeError(f"Forbidden privileged token(s) in {name}: {forbidden}")
        if "r4 lockbox" in raw.casefold() or '"r4-' in raw.casefold():
            raise RuntimeError(f"R4 micro namespace leaked into prospective R5 artifact: {name}")
        artifact_manifest[name] = {
            "sha256": _sha256_file(path),
            **_artifact_statistics(episodes),
        }
    for name, expected_sha256 in R4_INHERITED_FORMAL_SHA256.items():
        source = formal_source_dir / name
        destination = output_dir / name
        shutil.copyfile(source, destination)
        if _sha256_file(destination) != expected_sha256:
            raise RuntimeError(f"Byte-preserving formal inheritance failed for {name}.")
        artifact_manifest[name] = dict(parent_manifest["artifacts"][name])

    contract = {
        "schema_version": R5_BASELINE_LOCKBOX_SCHEMA,
        "seed": seed,
        "artifact_order": list(R5_ARTIFACT_NAMES),
        "formal_sizes": dict(R5_FORMAL_SIZES),
        "micro_contract": {
            "origin": "new_prospective_r5_same_entity",
            "smoke4": "set/overwrite/clear/noop x 1",
            "transition32": ("4 terminal kinds x 2 read forms x 2 history lengths x 2 lexical replicas"),
            "state_swap_pairing": "set<->overwrite and clear<->noop within identical entity/query scope",
            "mixed_order": "event is visible before immediate query; delayed pure-query probe follows",
            "pair_audits": audits,
        },
        "formal_contract": {
            "origin": "r4_carried_forward_sealed_unseen_holdout",
            "parent_manifest_sha256": R4_PARENT_MANIFEST_SHA256,
            "byte_preserving_copy": True,
            "semantic_json_not_parsed_by_r5_generator": True,
            "source_contract_sha256": source_contract_sha256,
        },
        "model_visible_episode_schema": "vision_memory.data.schema.Episode",
        "artifacts": artifact_manifest,
    }
    if tuple(artifact_manifest) != R5_ARTIFACT_NAMES:
        raise RuntimeError("R5 artifact inventory drifted from its locked order.")
    manifest = {**contract, "lockbox_contract_sha256": _sha256_value(contract)}
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    reject_hidden_ledger(manifest)
    return manifest


__all__ = [
    "R4_INHERITED_FORMAL_SHA256",
    "R4_PARENT_MANIFEST_SHA256",
    "R5_ARTIFACT_NAMES",
    "R5_BASELINE_LOCKBOX_SCHEMA",
    "R5_BASELINE_SEED",
    "R5_FORMAL_SIZES",
    "R5_HISTORY_LENGTHS",
    "R5_READ_FORMS",
    "R5_TERMINAL_KINDS",
    "build_smoke4",
    "build_transition32",
    "generate_r5_baseline_lockbox",
    "r5_target_for",
    "validate_same_entity_pair_contract",
]
