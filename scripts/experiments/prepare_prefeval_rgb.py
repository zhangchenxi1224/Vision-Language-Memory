"""Register a finite PrefEval RGB pilot before any model-dependent selection."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_memory.prefeval.adapter import PrefEvalAdapter
from vision_memory.prefeval.manifest import adaptation_topic_split
from vision_memory.prefeval.rgb_protocol import (SEED, digest, normalize, state_id,
    transition, event_wordings, queries, MAX_RECOVERY_TOKENS)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare(root, output, review):
    adapter = PrefEvalAdapter(root)
    records, alignment_differences = {}, []
    for ep in adapter.iter_episodes(forms=("explicit",)):
        raw = adapter._topic_rows[ep.topic]
        # Validate the alignment beyond the legacy adapter's question check.
        explicit = raw["explicit"][ep.row_index]["preference"]
        for form in ("mcq", "implicit_choice", "implicit_persona"):
            other = normalize(raw[form][ep.row_index]["preference"])
            if other != normalize(explicit):
                # Reviewed official revision: 30 rows add only the emphatic adverb
                # 'absolutely' in the other three forms; no constraint changes.
                if re.sub(r"\babsolutely\s+", "", other) != normalize(explicit):
                    raise ValueError(f"Unreviewed preference mismatch: {ep.base_pair_id}/{form}")
                alignment_differences.append([ep.base_pair_id, form, "added absolutely only"])
        records[ep.base_pair_id] = {"id": ep.base_pair_id, "topic": ep.topic,
            "preference": ep.turns[0].text, "question": ep.turns[-1].text,
            "options": list(ep.turns[-1].options), "target_index": ep.target_index,
            "option_permutation": list(ep.option_permutation),
            "distractors": adapter._sample_distractors(ep.base_pair_id, "explicit", 10)}
    distractor_texts = {}
    for record in records.values():
        keys = []
        for text in record.pop("distractors"):
            key = digest(text)[:20]
            distractor_texts[key] = text
            keys.append(key)
        record["distractor_ids"] = keys
    parent = {key: key for key in records}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        x, y = sorted((find(a), find(b)))
        parent[y] = x
    exact = {}
    for key, record in records.items():
        norm = normalize(record["preference"])
        if norm in exact:
            union(key, exact[norm])
        exact[norm] = key
    reviewed = json.loads(review.read_text(encoding="utf-8"))
    for pair in reviewed["pairs"]:
        if pair["decision"] == "same_split_family":
            union(pair["a"], pair["b"])
    for family in reviewed.get("additional_reviewed_families", []):
        for member in family["members"][1:]:
            union(family["members"][0], member)
    components = defaultdict(list)
    for key in records:
        components[find(key)].append(key)
    topic_split = adaptation_topic_split()
    ood = set(topic_split.ood_topics)
    groups, by_topic = {}, defaultdict(list)
    historical = json.loads((ROOT / "reports/official-alignment-results-20260913/broader151-bank-manifest.json").read_text())
    historical_text = "\n".join(g["event_text"] for g in historical["groups"])
    # Conservative concept-overlap screen, retaining rows in inventory but not
    # allocating these components to prospective unseen-content evaluation.
    inherited_atoms = {"ambient", "jazz", "green", "purple", "yellow", "juice", "coffee",
                       "cocoa", "leather", "linen", "ceramic", "glass", "rice", "pasta"}
    for representative, members in sorted(components.items()):
        topics = sorted({records[key]["topic"] for key in members})
        gid = "semantic-" + digest(sorted(members))[:16]
        matched = sorted({word for key in members for word in inherited_atoms
                          if re.search(r"\b" + re.escape(word) + r"\b", normalize(records[key]["preference"]))})
        group = {"id": gid, "members": sorted(members), "topics": topics,
                 "representative": representative, "inherited_concept_overlap": matched}
        if any(topic in ood for topic in topics) and any(topic not in ood for topic in topics):
            group["split"] = "quarantine_boundary"
        elif matched:
            group["split"] = "quarantine_inherited_concept"
        elif all(topic in ood for topic in topics):
            group["split"] = "ood_test"
        else:
            by_topic[topics[0]].append(gid)
        groups[gid] = group
        for key in members:
            records[key]["semantic_group"] = gid
    pilot_train, pilot_dev, small = [], [], []
    for topic in topic_split.adaptation_topics:
        ids = sorted(by_topic[topic], key=lambda x: digest([SEED, topic, x]))
        n_eval = max(2, round(len(ids) * .125))
        dev, test, train = ids[:n_eval], ids[n_eval:2*n_eval], ids[2*n_eval:]
        if len(train) < 4:
            raise ValueError(f"Insufficient training groups for {topic}")
        for name, subset in (("train", train), ("dev", dev), ("id_test", test)):
            for gid in subset:
                groups[gid]["split"] = name
        pilot_train.extend(train[:4])
        pilot_dev.extend(dev[:2])
        small.extend(train[:max(4, round(len(train) * .25))])
    for record in records.values():
        record["split"] = groups[record["semantic_group"]]["split"]

    targets, episodes = {}, []
    def register(state, previous, changed, split):
        sid = state_id(state)
        if split == "train" and state:
            if sid not in targets or previous is None:
                targets[sid] = {"id": sid, "state": dict(state), "predecessor": previous,
                    "changed_scope": changed, "queries": queries(state, training=True),
                    "qualification_queries": queries(state), "split": "train"}
        return sid
    def row(gid):
        return records[groups[gid]["representative"]]
    def make_episode(gids, index, panel):
        selected = [row(gid) for gid in gids]
        split = groups[gids[0]]["split"]
        topics = [r["topic"] for r in selected]
        if len(topics) != len(set(topics)):
            raise ValueError("Multiple active units must have distinct explicit scopes")
        state, transitions = {}, []
        first_topic = topics[0]
        available = pilot_train if split == "train" else pilot_dev
        partners = [gid for gid in available if row(gid)["topic"] == first_topic and gid != gids[0]]
        partner = partners[0]
        operations = [("set", r["topic"], r["preference"]) for r in selected]
        if len(gids) == 1:
            operations.append(("retain", None, None))
        operations.extend([("overwrite", first_topic, row(partner)["preference"]),
                           ("clear", first_topic, None), ("retain", None, None)])
        for ordinal, (op, scope, value) in enumerate(operations):
            previous = state_id(state) if state else None
            before = dict(state)
            state = transition(state, scope, value, op)
            sid = register(state, previous, scope, split)
            transitions.append({"ordinal": ordinal, "operation": op, "scope": scope,
                "event_wordings": event_wordings(scope, value, op),
                "before": before, "state": dict(state), "source_state_id": previous,
                "target_state_id": sid, "active_k": sum(v is not None for v in state.values()),
                "changed_scope": scope, "official_mcq_allowed": op == "set",
                "semantic_groups": sorted(set(gids + [partner]))})
        episode = {"id": f"{panel}-{index:03d}", "panel": panel, "split": split,
            "capacity": len(gids), "semantic_groups": sorted(set(gids + [partner])),
            "base_pairs": [r["id"] for r in selected], "transitions": transitions}
        episodes.append(episode)
        return episode
    # Install all train singleton roots first, preventing cyclic target initializers.
    for gid in pilot_train:
        r = row(gid)
        register({r["topic"]: r["preference"]}, None, r["topic"], "train")
    for label, selected_ids in (("train", pilot_train), ("dev", pilot_dev)):
        ordered = sorted(selected_ids, key=lambda gid: (row(gid)["topic"], gid))
        for i, gid in enumerate(ordered):
            make_episode([gid], i, label + "-k1")
        # Transpose per-topic selections: neighboring entries are different topics.
        per_topic = defaultdict(list)
        for gid in ordered:
            per_topic[row(gid)["topic"]].append(gid)
        interleaved = [values[j] for j in range(min(map(len, per_topic.values())))
                       for _, values in sorted(per_topic.items())]
        for k in (2, 4):
            for i in range(8):
                make_episode(interleaved[i*k:(i+1)*k], i, label + f"-k{k}")
        if label == "train":
            # New combinations, same seen content, not teacher-construction cases.
            held = interleaved[::2] + interleaved[1::2]
            for i in range(4):
                ep = make_episode(held[i*4:(i+1)*4], i, "seen-new-combination")
                ep["evaluation_only"] = True
    # Remove targets introduced only by held-out-combination episodes.
    train_target_ids = {t["target_state_id"] for ep in episodes if ep["split"] == "train"
                        and not ep.get("evaluation_only") for t in ep["transitions"]}
    targets = {sid: value for sid, value in targets.items() if sid in train_target_ids}
    sentinel = set()
    for topic in topic_split.adaptation_topics:
        r = row(next(gid for gid in pilot_train if row(gid)["topic"] == topic))
        sentinel.add(state_id({topic: r["preference"]}))
    for ep in [e for e in episodes if e["panel"] == "train-k4"][:4]:
        sentinel.update(t["target_state_id"] for t in ep["transitions"])
    # Include deterministic initialization ancestors, never silently initialize a
    # sentinel successor from an unavailable/nonmatching state.
    pending = list(sentinel)
    while pending:
        sid = pending.pop()
        pred = targets[sid]["predecessor"]
        if pred and pred != sid and pred not in sentinel:
            sentinel.add(pred)
            pending.append(pred)
    manifest = {"schema": "prefeval-rgb-pilot/v1", "seed": SEED,
        "upstream_commit": "50795054b5ff5f418d2b768a331d71e480f93331",
        "alignment_differences": alignment_differences,
        "source_hashes": {k.replace('\\', '/'): v for k, v in adapter.manifest()["source_sha256"].items()},
        "groups": groups, "records": records, "distractor_texts": distractor_texts, "pilot_train": pilot_train,
        "pilot_dev": pilot_dev, "small": sorted(small),
        "full": sorted(gid for gid, g in groups.items() if g["split"] == "train"),
        "episodes": episodes, "targets": targets, "sentinel_targets": sorted(sentinel),
        "max_recovery_tokens": MAX_RECOVERY_TOKENS,
        "test_policy": "Membership sealed before model results; no test teacher optimization or tuning.",
        "exposure": {"source": "broader151-bank-manifest.json", "historical_event_text_sha": digest(historical_text),
            "screened_atoms": sorted(inherited_atoms),
            "claim": "Prospective adaptation-content holdout, not foundation-pretraining novelty; lexical screening is conservative, not exhaustive."}}
    manifest["membership_sha"] = digest({gid: (g["members"], g["split"]) for gid, g in groups.items()})
    summary = {"semantic_groups": len(groups), "base_pairs": len(records),
        "split_groups": dict(Counter(g["split"] for g in groups.values())),
        "pilot_train": len(pilot_train), "pilot_dev": len(pilot_dev),
        "small": len(small), "full": len(manifest["full"]),
        "target_states": len(targets), "sentinel_states": len(sentinel),
        "episodes": dict(Counter(ep["panel"] for ep in episodes)),
        "membership_sha": manifest["membership_sha"], "manifest_sha": digest(manifest)}
    write(output / "manifest.json", manifest)
    write(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefeval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review", type=Path, default=ROOT / "reports/prefeval-rgb-20260917/near-duplicate-review.json")
    args = parser.parse_args()
    prepare(args.prefeval_root, args.output, args.review)
