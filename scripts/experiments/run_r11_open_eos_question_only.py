"""Evaluate existing EOS endpoints with fixed instructions and changed questions.

No optimizer is created. The original three prompts must reproduce all 216 old
endpoint generations exactly before the supplemental result is marked complete.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
import traceback

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.experiments import run_r11_open_eos_paired as eos
from scripts.experiments import run_r11_open_answer_replay as replay
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.repro import canonical_tensor_sha256

SCHEMA = "vision_memory.r11-open-eos-question-only.v1"
PREFIX = "R3 Train Standard Templates 03: "
SUFFIX = "\nUse the memory image to answer. Answer with a short phrase only."
ANCHORS = ["original_open", "paraphrase_open", "new_rewrite_open"]
QUESTION_IDS = ["question_only_restored", *[f"question_only_{i:02d}" for i in range(2, 6)]]


def locked_prompts(config, target, parent_config):
    if (config["schema"] != SCHEMA or config["endpoint_step"] != 256
            or config["arms"] != ["A", "B", "C"] or config["seeds"] != list(range(8))
            or config["conditions"] != ["matched", "blank", "fixed_donor"]
            or config["fixed_prefix"] != PREFIX or config["fixed_suffix"] != SUFFIX
            or config["anchor_prompts"] != ANCHORS
            or list(config["question_sentences"]) != QUESTION_IDS
            or config["generation"] != replay.GENERATION
            or config["training_updates"] != 0
            or config["expected_generation_count"] != 576
            or config["expected_anchor_parity_count"] != 216):
        raise ValueError("Locked single-variable protocol changed.")
    prompts = {**target["inputs"], "new_rewrite_open": parent_config["new_rewrite_open"]}
    for name in ANCHORS[:2]:
        if not prompts[name].startswith(PREFIX) or not prompts[name].endswith(SUFFIX):
            raise ValueError("Old fixed instruction contract changed.")
    gold = target["scorer_metadata"]["gold"].casefold()
    for name, question in config["question_sentences"].items():
        if ("\n" in question or gold in question.casefold()
                or "indigo desk train 001123" not in question or "later check" not in question):
            raise ValueError("Question changes task identity, timing, or leaks the answer.")
        prompts[name] = PREFIX + question + SUFFIX
    if len(set(prompts.values())) != 8:
        raise ValueError("Duplicate prompt.")
    if prompts["question_only_restored"].split("\n")[0] != prompts["new_rewrite_open"].split("\n")[0]:
        raise ValueError("Restored case must retain the exact old rewrite question.")
    return prompts


def key(row):
    return row["arm"], row["seed"], row["condition"], row["prompt_id"]


def check_anchor_parity(rows, parent_rows):
    records = []
    for row in rows:
        if row["prompt_id"] not in ANCHORS:
            continue
        parent = parent_rows[key(row)]
        fields = ["generated_token_ids", "input_token_ids", "chat_prompt", "raw", "strict_correct"]
        same = all(row[field] == parent[field] for field in fields)
        records.append({"key": list(key(row)), "exact_parity": same, "compared_fields": fields})
        if not same:
            raise RuntimeError(f"Old endpoint generation mismatch: {key(row)}")
    return records


def summarize(rows):
    expected = {(a, s, c, p) for a in ("A", "B", "C") for s in range(8)
                for c in ("matched", "blank", "fixed_donor") for p in ANCHORS + QUESTION_IDS}
    if len(rows) != len(expected) or {key(r) for r in rows} != expected:
        raise ValueError("Missing or duplicate endpoint evaluations.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["arm"], row["condition"], row["prompt_id"])].append(row)
    cells = []
    for (arm, condition, prompt), values in sorted(grouped.items()):
        cells.append({"arm": arm, "condition": condition, "prompt_id": prompt, "n": 8,
                      **{field: sum(bool(r[field]) for r in values) for field in
                         ("strict_correct", "answer_prefix_token_exact", "overgeneration")}})
    lookup = {key(r): r for r in rows}
    consistency = []
    transitions = []
    for arm in ("A", "B", "C"):
        for seed in range(8):
            views = [lookup[(arm, seed, "matched", p)] for p in QUESTION_IDS]
            consistency.append({"arm": arm, "seed": seed,
                                "correct_out_of_five": sum(r["strict_correct"] for r in views),
                                "all_five_correct": all(r["strict_correct"] for r in views)})
            old = lookup[(arm, seed, "matched", "new_rewrite_open")]
            restored = lookup[(arm, seed, "matched", "question_only_restored")]
            transitions.append({"arm": arm, "seed": seed, "old_raw": old["raw"],
                                "restored_raw": restored["raw"], "old_correct": old["strict_correct"],
                                "restored_correct": restored["strict_correct"]})
    return {"schema": SCHEMA + ".summary", "cells": cells, "per_seed_consistency": consistency,
            "restored_instruction_transitions": transitions, "generation_count": len(rows),
            "formal_shared_memory_success": False,
            "scope": "one_question_post_hoc_diagnostic_not_new_task_generalization",
            "metric_unit": "seed; five prompts are repeated measurements"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt-config", type=Path, default=ROOT / "configs/experiments/r11_open_eos_question_only.json")
    p.add_argument("--config", type=Path, default=ROOT / "configs/experiments/r11_open_eos_paired.json")
    p.add_argument("--parent-root", type=Path, required=True)
    p.add_argument("--old-multistart-root", type=Path, required=True)
    p.add_argument("--dreamlite", type=Path, required=True)
    p.add_argument("--reader", type=Path, required=True)
    p.add_argument("--vae-device", default="cuda:0")
    p.add_argument("--reader-device", default="cuda:1")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args(argv)
    args.arms, args.seeds, args.mode = ["A", "B", "C"], list(range(8)), "question_only_endpoint_replay"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        config = json.loads(args.prompt_config.read_text(encoding="utf-8"))
        parent_config = eos.load_config(args.config)
        parent_manifest = json.loads((args.parent_root / "manifest.json").read_text())
        published = ROOT / "reports/r11-open-eos-results-20260908/raw"
        for name in ("manifest.json", "raw_generations.jsonl", "terminal.json"):
            if replay.sha256_file(args.parent_root / name) != replay.sha256_file(published / name):
                raise ValueError(f"Original published result hash differs: {name}")
        if parent_manifest["git_commit"] != config["parent_training_commit"] or parent_manifest["config"] != parent_config:
            raise ValueError("Wrong parent training run.")
        parent_rows = {key(r): r for r in eos.read_rows(args.parent_root / "raw_generations.jsonl")
                       if r["arm"] in args.arms and r["step"] == 256}
        if len(parent_rows) != 216:
            raise ValueError("Incomplete original endpoint generation ledger.")
        endpoints = []
        for seed in args.seeds:
            for arm in args.arms:
                path = args.parent_root / "runs" / f"seed-{seed:02d}-{arm}" / "step-256.pt"
                endpoints.append({"arm": arm, "seed": seed, "path": str(path),
                                  "file_sha256": replay.sha256_file(path)})
        runtime = eos.load_runtime(args, parent_config)
        prompts = locked_prompts(config, runtime["target"], parent_config)
        runtime["target"]["inputs"] = prompts
        eval_config = {**parent_config, "evaluation_prompts": list(prompts)}
        replay.write_json(args.output_dir / "question_only_manifest.json", {
            "schema": SCHEMA + ".manifest", "config": config, "actual_prompts": prompts,
            "prompt_config_sha256": replay.sha256_file(args.prompt_config),
            "git_commit": replay.command_output(["git", "rev-parse", "HEAD"]),
            "parent_root": str(args.parent_root), "endpoint_bindings_before_evaluation": endpoints,
            "primary_question_ids": QUESTION_IDS, "anchor_ids": ANCHORS,
            "training_updates": 0, "best_checkpoint_selection": False})
        rows, parity = [], []
        for endpoint in endpoints:
            path = Path(endpoint["path"])
            if replay.sha256_file(path) != endpoint["file_sha256"]:
                raise ValueError("Endpoint changed during evaluation.")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            latent = payload["latent_fp32"]
            if (payload["step"] != 256 or latent.dtype != torch.float32
                    or tuple(latent.shape) != (1, 4, 128, 128) or not torch.isfinite(latent).all()):
                raise ValueError("Invalid raw endpoint.")
            tensor_sha = canonical_tensor_sha256(latent)
            with torch.no_grad():
                image = decode_model_latents_unit_interval(runtime["vae"],
                    latent.to(runtime["vae_device"], dtype=torch.bfloat16)).cpu()
            current = eos.evaluate(runtime, eval_config, image, run_id=f"seed-{endpoint['seed']:02d}-{endpoint['arm']}",
                arm=endpoint["arm"], seed=endpoint["seed"], step=256, output_dir=args.output_dir,
                conditions=config["conditions"], deployment=False)
            checks = check_anchor_parity(current, parent_rows)
            for check in checks:
                replay.append_jsonl(args.output_dir / "anchor_parity.jsonl", check)
            parity.extend(checks)
            rows.extend(current)
            if canonical_tensor_sha256(latent) != tensor_sha or replay.sha256_file(path) != endpoint["file_sha256"]:
                raise RuntimeError("Evaluation mutated the endpoint.")
            replay.append_jsonl(args.output_dir / "endpoint_audit.jsonl", {
                **endpoint, "latent_tensor_sha256": tensor_sha,
                "image_tensor_sha256": canonical_tensor_sha256(image), "unchanged_after_evaluation": True})
            del payload, latent, image
        if len(parity) != config["expected_anchor_parity_count"]:
            raise ValueError("Incomplete original-prompt parity coverage.")
        replay.frozen_audit(runtime["vae"], runtime["reader"])
        summary = summarize(rows)
        replay.write_json(args.output_dir / "summary.json", summary)
        replay.write_json(args.output_dir / "terminal.json", {
            "status": "completed", "elapsed_seconds": time.monotonic() - started,
            "raw_generation_count": len(rows), "exact_anchor_parity_count": len(parity),
            "endpoint_count": len(endpoints), "training_updates": 0,
            "formal_shared_memory_success": False})
        print(json.dumps({"status": "completed", "generations": len(rows)}), flush=True)
        return 0
    except BaseException as error:
        replay.write_json(args.output_dir / "terminal.json", {
            "status": "failed", "error": str(error), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
