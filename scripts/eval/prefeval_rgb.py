"""Frozen Reader references and PNG readback for the registered RGB experiment."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from vision_memory.prefeval.rgb_protocol import queries, digest, scope_name, ABSENT
from vision_memory.reader.open_answer import generate_short_answer, score_short_answer
from vision_memory.reader.qwen3vl import R3_QWEN_READER_RESIZE_CONTRACT


def load_reader(path, device):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=True,
        min_pixels=256*256, max_pixels=256*256)
    reader = Qwen3VLForConditionalGeneration.from_pretrained(path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(device)
    reader.eval().requires_grad_(False)
    return processor, reader


def read_png(path):
    with Image.open(path) as im:
        if im.mode != "RGB" or im.size != (1024, 1024):
            raise ValueError("Persistent state must be a 1024-square RGB PNG")
        a = np.array(im, copy=True)
    return torch.from_numpy(a).permute(2, 0, 1).float()/255


def text_prefix(state):
    entries = [f"{scope_name(scope)}: {value if value is not None else ABSENT}"
               for scope, value in sorted(state.items())]
    return "The current stored preference statements are:\n" + "\n".join(entries) + "\n\n"


def mcq_query(record):
    options = "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", record["options"]))
    return (record["question"] + "\nI'm trying to decide on this and here are 4 options for my query:\n" + options +
        "\nNow, pick one as your top recommendation matching my preferences. "
        "Choose only A, B, C, or D. Give your answer in exactly this format, "
        "with no additional explanation: <choice>[A/B/C/D]</choice>")


def mcq_score(raw, index):
    match = re.fullmatch(r"\s*<choice>\s*([ABCD])\s*</choice>\s*", raw)
    predicted = "ABCD".index(match.group(1)) if match else None
    return {"predicted_index": predicted, "strict_correct": predicted == index,
            "format_valid": match is not None}


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_references(args):
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {"manifest_sha": digest(manifest), "reader": str(args.reader),
                "stage": "references", "shard": args.shard, "shards": args.shards,
                "max_recovery_tokens": manifest["max_recovery_tokens"]}
    idpath = args.output / "identity.json"
    if idpath.exists() and json.loads(idpath.read_text()) != identity:
        raise ValueError("Reference output belongs to a different registration")
    idpath.write_text(json.dumps(identity, indent=2))
    processor, reader = load_reader(args.reader, args.device)
    tokenizer = processor.tokenizer
    targets = [r["preference"] for r in manifest["records"].values()
               if r["semantic_group"] in set(manifest["pilot_train"] + manifest["pilot_dev"])]
    lengths = [len(tokenizer.encode(target, add_special_tokens=False)) for target in targets]
    if max(lengths) + 8 >= manifest["max_recovery_tokens"]:
        raise ValueError("Registered generation budget does not fit full preference targets")
    (args.output / "token-lengths.json").write_text(json.dumps({"min": min(lengths), "max": max(lengths), "count": len(lengths)}))
    gray_path = args.output / "blank.png"
    if not gray_path.exists():
        Image.new("RGB", (1024,1024), (128,128,128)).save(gray_path)
    image = read_png(gray_path)
    jobs = []
    for i, gid in enumerate(manifest["pilot_train"] + manifest["pilot_dev"]):
        record = manifest["records"][manifest["groups"][gid]["representative"]]
        state = {record["topic"]: record["preference"]}
        jobs.append({"id": "mcq:"+gid, "groups": [gid], "split": record["split"],
            "state": state, "query": mcq_query(record), "target_index": record["target_index"], "kind": "official_mcq"})
    # Deduplicate equal state/scope/query checks before measurement, preserving
    # the semantic group membership needed to reconstruct episode metrics.
    seen = set()
    for ep in manifest["episodes"]:
        if ep.get("evaluation_only"):
            continue
        for tr in ep["transitions"]:
            for q in queries(tr["state"]):
                key = digest([tr["state"], q["query"]])
                if key in seen:
                    continue
                seen.add(key)
                jobs.append({"id": key, "groups": ep["semantic_groups"], "split": ep["split"],
                    "state": tr["state"], **q, "kind": "derived_recovery"})
    rows_path = args.output / "reads.jsonl"
    completed = set()
    if rows_path.exists():
        completed = {r["key"] for r in map(json.loads, rows_path.read_text(encoding="utf-8").splitlines())}
    started = time.monotonic()
    for index, job in enumerate(jobs):
        if index % args.shards != args.shard:
            continue
        for condition in ("blank", "text"):
            key = condition + ":" + job["id"]
            if key in completed:
                continue
            query = (text_prefix(job["state"]) if condition == "text" else "") + job["query"]
            result = generate_short_answer(model=reader, processor=processor, image=image,
                query=query, device=args.device, max_new_tokens=manifest["max_recovery_tokens"],
                reader_resize_contract=R3_QWEN_READER_RESIZE_CONTRACT)
            scored = (mcq_score(result["raw"], job["target_index"]) if job["kind"] == "official_mcq"
                      else score_short_answer(result["raw"], job["target"]))
            # EOS and token-limit outcomes remain explicit; no truncation exclusion.
            scored["strict_correct"] = bool(scored["strict_correct"] and result["eos_reached"])
            append(rows_path, {"key": key, "job": job, "condition": condition,
                "result": result, "score": scored, "elapsed_s": time.monotonic()-started})
            print(json.dumps({"index":index,"total":len(jobs),"condition":condition,
                              "correct":scored["strict_correct"]}), flush=True)
    rows = [json.loads(s) for s in rows_path.read_text(encoding="utf-8").splitlines()]
    counts = defaultdict(lambda: [0,0])
    for row in rows:
        key = "/".join((row["condition"],row["job"]["split"],row["job"]["kind"]))
        counts[key][0] += int(row["score"]["strict_correct"])
        counts[key][1] += 1
    summary = {"status":"completed", "counts": dict(counts), "seconds": time.monotonic()-started,
               "identity": identity, "total_registered_jobs": len(jobs)}
    (args.output/"result.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    run_references(parser.parse_args())
