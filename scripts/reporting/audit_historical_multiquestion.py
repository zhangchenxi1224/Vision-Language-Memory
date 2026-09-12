"""CPU-only verification of the completed, fixed 16-question EOS campaign.

Execute the pinned historical inventory/scoring functions via AST extraction,
without importing Torch or loading a CUDA model. This does not rerun inference.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace
from typing import Any
import unicodedata

COMMIT = "2c0e41c899910bf0641f16ee724f85bbe3491a7e"
PANEL_SHA = "d356238fd5c267812dcf28d214ab062fd43388bb6b53b78602f0c1e8f5b36672"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            if node.decorator_list:
                raise ValueError("CPU audit cannot strip function decorators")
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                ast.literal_eval(node.value)  # Only literal constants are allowed.
                selected.append(node)
                found.add(node.targets[0].id)
    if found != set(names):
        raise ValueError(f"Missing historical functions: {set(names)-found}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.repo, text=True).strip() != COMMIT:
        raise ValueError("Wrong historical source")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=a.repo, text=True).strip():
        raise ValueError("Historical source is dirty")
    panel_path = a.repo / "configs/experiments/r11_open_eos_multiquestion_targets.json"
    if sha(panel_path) != PANEL_SHA:
        raise ValueError("Historical panel differs from preregistration")
    panel = load(panel_path)
    targets = {x["target_index"]: x for x in panel["targets"]}
    if len(targets) != 16 or len({t["semantic_group_id"] for t in targets.values()}) != 16:
        raise ValueError("Expected 16 independent semantic groups")
    ns = dict(json=json, hashlib=hashlib, Path=Path, Any=Any, re=re, unicodedata=unicodedata,
              replay=SimpleNamespace(sha256_file=sha))
    extract(a.repo / "scripts/experiments/run_r11_open_eos_multiquestion.py",
        ["SCHEMA", "STEPS", "PROMPTS", "ARMS", "CONDITIONS", "OPTIMIZER", "require", "canonical_hash",
         "read_rows", "run_binding", "verify_completed_run", "validate_run_rows"], ns)
    extract(a.repo / "src/vision_memory/reader/open_answer.py", ["normalize_short_answer", "score_short_answer"], ns)
    extract(a.repo / "src/vision_memory/reader/open_eos.py", ["generation_diagnostics"], ns)
    require = ns["require"]
    terminal = load(a.run / "terminal.json")
    require(terminal["status"] == "completed" and terminal["planned_runs"] == 128, "Campaign incomplete")
    rows, runs, seen_targets, initial_pairs = [], [], set(), {}
    for lane in (0, 1):
        root = a.run / f"lane-{lane}"
        plan = load(root / "plan.json")
        end = load(root / "end_bindings.json")
        require(plan["training_commit"] == COMMIT and plan["targets_json_sha256"] == PANEL_SHA, "Lane source drift")
        require(plan["source_hashes"] == end["source_hashes"] and all(sha(a.repo/k) == v for k,v in plan["source_hashes"].items()), "Source hash drift")
        require(all(end[k] is True for k in ("sources_unchanged", "targets_unchanged", "models_unchanged")), "Failed end audit")
        require(plan["seeds"] == [0,1,2,3] and plan["arms"] == ["A","B"], "Campaign factorial changed")
        require(not seen_targets.intersection(plan["target_indices"]), "Repeated target across lanes")
        seen_targets.update(plan["target_indices"])
        for i in plan["target_indices"]:
            target = targets[i]
            for seed in plan["seeds"]:
                for arm in plan["arms"]:
                    run = root / "runs" / f"target-{i:03d}-seed-{seed:02d}-{arm}"
                    manifest = load(run / "manifest.json")
                    key = (i, seed)
                    initial = manifest["initial_latent_sha256"]
                    require(initial_pairs.setdefault(key, initial) == initial, "A/B initial latents differ")
                    binding = ns["run_binding"](target, seed, arm, ns["canonical_hash"](plan), initial)
                    values = ns["verify_completed_run"](run, binding)
                    for r in values:
                        scores = ns["generation_diagnostics"](r, target["scorer_metadata"]["gold"], r["gold_token_ids"])
                        require(all(r[k] == v for k,v in scores.items()), "Raw score/token mismatch")
                    rows.extend(values)
                    runs.append({"run": str(run), "inventory_sha256": sha(run/"artifact_inventory.json")})
                    print(f"verified {len(runs)}/128", flush=True)
    require(seen_targets == set(targets) and len(runs) == 128 and len(rows) == 3456, "Missing planned evidence")
    cells = {}
    for r in rows:
        if r["step"] != 256:
            continue
        key = "/".join([r["arm"], r["condition"], r["prompt_id"]])
        cell = cells.setdefault(key, {"n":0, "exact_match":0, "answer_eos":0})
        cell["n"] += 1
        cell["exact_match"] += int(r["strict_correct"])
        cell["answer_eos"] += int(r["strict_correct"] and r["answer_followed_immediately_by_eos"])
    result = {"source_commit":COMMIT, "panel_sha256":PANEL_SHA, "verified_runs":len(runs),
        "verified_updates":len(runs)*256, "raw_generation_count":len(rows), "cells":cells,
        "independent_questions":16, "repeated_seeds_per_question":4,
        "scope":"Historical raw outputs and artifact hashes reverified; independent BF16 VAE latent oracles, no shared Writer; no new model inference",
        "model_reuse_limit":"Current official FM uses FP32 VAE. These old BF16-decoded endpoints require fresh readback before supervising it.",
        "runs":runs}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()
