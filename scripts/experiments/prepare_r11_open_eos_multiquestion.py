"""Prospectively select real, source-bound questions without inspecting outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SHA256 = "24327edc39e0d133df5150dc1aab4f55c6cf5b05ccfca9025ad90c5accc6d184"
PROTOCOL = "r11-open-eos-multiquestion-20260908"
STRATA = [("color", "green"), ("drink", "juice"), ("music", "jazz"),
          ("material", "linen"), ("meal", "pasta"),
          ("color", "no active preference"), ("drink", "no active preference"),
          ("material", "no active preference")]


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":")).encode("utf-8"))


def make_prompts(query: str) -> dict[str, str]:
    suffix = " Choose exactly one option."
    if not query.endswith(suffix):
        raise ValueError("Unexpected source MCQ suffix; cannot silently rewrite task")
    question = query[:-len(suffix)]
    match = re.search(r"what is the current ([a-z]+) preference for (.+)\?", question)
    if not match:
        raise ValueError("Unexpected preference-question syntax")
    topic, entity = match.groups()
    before = question[:match.start()]
    paraphrase = before + f"which {topic} does {entity} currently prefer?"
    rewrite = before + f"name the currently stored {topic} preference for {entity}."
    instruction = "\nUse the memory image to answer. Answer with a short phrase only."
    return {"original_open": question + instruction,
            "paraphrase_open": paraphrase + instruction,
            "new_rewrite_open": rewrite + "\nReturn only a short phrase."}


def select_questions(episodes, excluded_groups: set[str]) -> list[dict]:
    candidates = []
    record_indices = {e["episode_id"]: i for i, e in enumerate(episodes)}
    for episode in episodes:
        if episode["split"] != "train" or episode["distractor_variant"] != "clean":
            continue
        if episode["semantic_group_id"] in excluded_groups:
            continue
        for turn_id, turn in enumerate(episode["turns"]):
            if "query" not in turn:
                continue
            query = turn["query"]
            gold = query["choices"][query["target_index"]]
            if (episode["topic"], gold) not in STRATA:
                continue
            key = sha_bytes(f"{PROTOCOL}\x1f{episode['episode_id']}\x1f{turn_id}".encode())
            candidates.append((key, episode, turn_id, gold))
    used = set(excluded_groups)
    selected = []
    for topic, gold in STRATA:
        count = 0
        for key, episode, turn_id, answer in sorted(candidates, key=lambda x: x[0]):
            group = episode["semantic_group_id"]
            if episode["topic"] != topic or answer != gold or group in used:
                continue
            query = episode["turns"][turn_id]["query"]
            prefix = episode["turns"][:turn_id + 1]
            selected.append({
                "target_index": len(selected), "segment_id": f"open-mq-{key[:24]}",
                "topic": topic, "stratum": f"{topic}:{gold}",
                "source_episode_id": episode["episode_id"],
                "source_record_index": record_indices[episode["episode_id"]],
                "semantic_group_id": group, "source_turn_id": turn_id,
                "source_episode_sha256": canonical_sha(episode),
                "source_query_sha256": canonical_sha(query),
                "source_prefix": prefix,
                "selection_key_sha256": key,
                "original": {"query": query["text"], "choices": query["choices"],
                             "answer_index": query["target_index"]},
                "inputs": make_prompts(query["text"]),
                "scorer_metadata": {"gold": gold, "aliases": []},
                "scope": "independent_question_vae_latent_oracle_not_shared_writer",
                "source_prefix_is_audit_only_not_reader_input": True,
            })
            used.add(group)
            count += 1
            if count == 2:
                break
        if count != 2:
            raise ValueError(f"Not enough independent source groups for {topic}:{gold}")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-targets", type=Path,
                        default=ROOT / "configs/experiments/r11_open_answer_replay.json")
    args = parser.parse_args()
    payload = args.train.read_bytes()
    if sha_bytes(payload) != TRAIN_SHA256:
        raise ValueError("Locked original train dataset hash mismatch")
    episodes = [json.loads(line) for line in payload.splitlines() if line.strip()]
    old = json.loads(args.old_targets.read_text(encoding="utf-8"))
    numbers = [re.search(r"train (\d{6})", t["original"]["query"]).group(1)
               for t in old["targets"]]
    excluded = {f"r3-train-semantic-{n}" for n in numbers}
    targets = select_questions(episodes, excluded)
    result = {
        "schema": "vision_memory.r11-open-eos-multiquestion-targets.v1",
        "protocol": PROTOCOL, "source_train_sha256": TRAIN_SHA256,
        "source_train_path": str(args.train), "source_split": "train",
        "selection": {"source_dataset_path": str(args.train), "source_dataset_sha256": TRAIN_SHA256,
                      "split": "train",
                      "method": "fixed eight strata; two unique semantic groups per stratum; SHA256 ascending",
                      "outcomes_used": False, "excluded_old_semantic_groups": sorted(excluded),
                      "strata": [{"topic": t, "gold": g} for t, g in STRATA],
                      "failed_target_replacement": False},
        "targets": targets, "targets_payload_sha256": canonical_sha(targets),
        "arms": ["A", "B"], "seeds": [0, 1, 2, 3], "optimizer_steps": 256,
        "learning_rate": 0.05, "lambda_eos": 1.0,
        "training_prompts": ["original_open"],
        "evaluation_only_prompts": ["paraphrase_open", "new_rewrite_open"],
        "donor_spec": {
            "kind": "frozen_completed_ambient_eos_B_endpoint",
            "gold": "ambient", "source_run": "seed-00-B", "step": 256,
            "path": "/inspire/ssd/project/exploration-topic/czxs26210936/runs/vision-language-memory-r11-open-eos/all-e6e6c80-20260908/runs/seed-00-B/step-256.pt",
            "parent_training_commit": "e6e6c8071a6c1311a966bda1aa636748fbb016f6",
            "all_target_golds_must_differ": True,
        },
        "lanes": [{"gpu_pair": [0, 1], "target_indices": list(range(0, 16, 2))},
                  {"gpu_pair": [2, 3], "target_indices": list(range(1, 16, 2))}],
        "planned_questions": 16, "planned_question_seed_pairs": 64,
        "planned_optimization_runs": 128, "planned_optimizer_updates": 32768,
        "token_length_source": "actual contextual Qwen tokenizer; dataset target_token_count is MCQ label metadata",
        "limitations": ["all 16 questions are individually optimized, not held-out question generalization",
                        "no shared Writer is trained", "answer-length groups differ in semantic content",
                        "natural train vocabulary has no light blue/orange juice; no synthetic replacement"],
    }
    donor = Path(result["donor_spec"]["path"])
    donor_raw = donor.parents[2] / "raw_generations.jsonl"
    result["control_spec"] = {
        "donor_checkpoint": str(donor), "donor_checkpoint_sha256": sha_bytes(donor.read_bytes()),
        "donor_gold": "ambient", "donor_source_run_id": "seed-00-B",
        "donor_source_raw_generations": str(donor_raw),
        "donor_source_raw_generations_sha256": sha_bytes(donor_raw.read_bytes()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": sha_bytes(args.output.read_bytes()),
                      "target_count": len(targets), "topics": dict(Counter(t["topic"] for t in targets))}))


if __name__ == "__main__":
    main()
