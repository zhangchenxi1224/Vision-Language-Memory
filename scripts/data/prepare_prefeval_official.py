#!/usr/bin/env python3
"""Export the released PrefEval SFT recipe and unabridged benchmark disclosures."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_memory.prefeval.official_pipeline import (  # noqa: E402
    EVAL_TOPICS, TRAIN_TOPICS, TOPIC_ORDER, SFT_INTER_TURNS, UPSTREAM_REVISION,
    context_messages, disclosure_messages, sft_record,
)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_records(path: Path, records: list[dict]):
    # Stable compression permits byte-for-byte regeneration.
    with path.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
        for record in records:
            stream.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))


def prepare(upstream: Path, output: Path) -> dict:
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected registered upstream revision {UPSTREAM_REVISION}, got {revision}")
    if subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain"], text=True).strip():
        raise ValueError("Upstream checkout has modifications; use the pinned source")
    output.mkdir(parents=True, exist_ok=True)
    pool = read(upstream / "benchmark_dataset/filtered_inter_turns.json")
    contexts = context_messages(pool, for_sft=True)
    write_json(output / "context-pools.json", {
        "sft": contexts, "benchmark": context_messages(pool, for_sft=False),
        "unit": "one turn = one user/assistant exchange",
    })
    exports = {(split, n): [] for split in ("train", "eval_topic") for n in SFT_INTER_TURNS}
    benchmark, differences, provenance = [], [], []
    counts = Counter()
    for topic in TOPIC_ORDER:
        sft_path = f"SFT/single_pref_remind/{topic}/mistral8x7b_{topic}_2turn.json"
        paths = {
            "explicit": f"benchmark_dataset/explicit_preference/{topic}.json",
            "implicit_choice": f"benchmark_dataset/implicit_preference/choice-based/{topic}.json",
            "implicit_persona": f"benchmark_dataset/implicit_preference/persona-driven/{topic}.json",
            "mcq": f"benchmark_dataset/mcq_options/{topic}.json",
        }
        rows = {form: read(upstream / path) for form, path in paths.items()}
        teachers = read(upstream / sft_path)
        if len({len(teachers), *(len(r) for r in rows.values())}) != 1:
            raise ValueError(f"Mismatched row counts for {topic}")
        split = "train" if topic in TRAIN_TOPICS else "eval_topic"
        for index, teacher in enumerate(teachers):
            base_id = f"{topic}:{index:04d}"
            if any(r[index]["question"] != teacher["question"] for r in rows.values()):
                raise ValueError(f"Question mismatch at {base_id}")
            explicit = rows["explicit"][index]
            if explicit["preference"] != teacher["preference"]:
                # The pinned release differs by 'absolutely ' in 30 records.
                # Preserve both exact strings; reject any unreviewed join.
                if explicit["preference"].replace("absolutely ", "") != teacher["preference"].replace("absolutely ", ""):
                    raise ValueError(f"Unreviewed preference mismatch at {base_id}")
                differences.append({"base_pair_id": base_id, "benchmark": explicit["preference"],
                                    "sft": teacher["preference"], "action": "preserve_both_verbatim"})
            for n in SFT_INTER_TURNS:
                exports[split, n].append(sft_record(teacher, topic=topic, row_index=index,
                                                  contexts=contexts, inter_turns=n))
            for form in ("explicit", "implicit_choice", "implicit_persona"):
                row = rows[form][index]
                benchmark.append({
                    "base_pair_id": base_id, "topic": topic, "split": split, "form": form,
                    "input": {"disclosure": disclosure_messages(row, form),
                              "needs_model_acknowledgment": form == "explicit",
                              "query": {"role": "user", "content": row["question"]}},
                    "evaluation_only": {
                        "preference": row["preference"], "explanation": row.get("explanation"),
                        "classification_task_options": rows["mcq"][index]["classification_task_options"],
                        "unshuffled_correct_option_index": 0,
                    },
                    "source": {"path": paths[form], "row_index": index, "revision": revision},
                })
            counts[split] += 1
        for path in [sft_path, *paths.values()]:
            provenance.append({"path": path, "sha256": hashlib.sha256((upstream / path).read_bytes()).hexdigest()})
    for (split, n), records in exports.items():
        topic_order = TRAIN_TOPICS if split == "train" else EVAL_TOPICS
        records.sort(key=lambda r: (topic_order.index(r["topic"]), r["source"]["row_index"]))
        write_records(output / f"sft-{split}-{n}interturn.jsonl.gz", records)
    write_records(output / "benchmark-disclosures.jsonl.gz", benchmark)
    write_json(output / "wording-differences.json", differences)
    for path in ("benchmark_dataset/filtered_inter_turns.json", "SFT/train_sft.py", "SFT/sft_utils.py",
                 "utils/common_utils.py", "utils/implicit_utils.py"):
        provenance.append({"path": path, "sha256": hashlib.sha256((upstream / path).read_bytes()).hexdigest()})
    manifest = {
        "schema": "prefeval.official-alignment.v1", "upstream_revision": revision,
        "topic_split_seed": 42, "train_topics": TRAIN_TOPICS, "eval_topics": EVAL_TOPICS,
        "base_pairs": dict(counts), "benchmark_disclosures": len(benchmark),
        "sft_context_conditions": SFT_INTER_TURNS,
        "context_condition_policy": "separate datasets/experiments, not one 3x mixed training set",
        "sft_target": "verbatim released response_to_q; final-answer-only token CE",
        "teacher_identity_caveat": "paper says Mistral-7B; released filename/README say mistral8x7b",
        "wording_differences": len(differences), "source_files": provenance,
        "status": "DATA_PREPARED; no SFT or shared Writer training performed",
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                  for p in sorted(output.iterdir()) if p.name != "manifest.json" and p.is_file()},
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefeval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.prefeval_root, args.output)
    print(json.dumps({k: result[k] for k in ("base_pairs", "benchmark_disclosures", "wording_differences", "status")}, indent=2))
