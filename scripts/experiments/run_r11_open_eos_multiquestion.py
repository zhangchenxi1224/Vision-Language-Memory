"""Fresh paired A/B Open-EOS oracles for a preregistered real-question panel.

Each question/seed receives an independent optimized VAE latent. This runner does
not train a shared writer and does not claim cross-question generalization.
Existing completed runs are reused only after full artifact SHA verification;
partial runs are retained and stop execution rather than being overwritten.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.experiments import run_r11_open_eos_paired as paired
from scripts.experiments import run_r11_mcq_open_multistart as old
from scripts.experiments import run_r11_open_answer_replay as replay
from scripts.train.latent_r11_vae_oracle import VAELatentOracle
from vision_memory.dreamlite.latent_codec import decode_model_latents_unit_interval
from vision_memory.reader.open_answer import generate_short_answer, normalize_short_answer
from vision_memory.reader.open_eos import generation_diagnostics
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism

SCHEMA = "vision_memory.r11-open-eos-multiquestion.v1"
STEPS = [0, 16, 32, 64, 128, 192, 256]
PROMPTS = ["original_open", "paraphrase_open", "new_rewrite_open"]
ARMS = ["A", "B"]
CONDITIONS = ["matched", "blank", "fixed_donor"]
OPTIMIZER = {"name": "Adam", "steps": 256, "lr": .05, "betas": [.9, .999], "eps": 1e-8,
             "weight_decay": 0., "amsgrad": False}


@contextmanager
def lane_lock(directory):
    """Linux advisory lock lives for the entire lane, including completed-run reuse."""
    import fcntl
    path = directory / ".lane.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process owns this output lane: {directory}") from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"pid": os.getpid(), "acquired_unix": time.time()}))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_panel(panel):
    """Validate generation/scoring boundaries before loading any GPU model."""
    require(isinstance(panel, dict) and isinstance(panel.get("selection"), dict), "Panel must record real-data selection provenance.")
    selection = panel["selection"]
    require(selection.get("split") == "train" and bool(selection.get("method")), "Use preregistered real train selection.")
    require(bool(selection.get("source_dataset_path")) and re.fullmatch("[0-9a-f]{64}", selection.get("source_dataset_sha256", "")),
            "Selection must bind source dataset path and SHA256.")
    targets = panel.get("targets")
    require(isinstance(targets, list) and len(targets) > 0, "Empty target panel.")
    indices, ids, semantic = set(), set(), set()
    for target in targets:
        index = target.get("target_index")
        require(isinstance(index, int) and not isinstance(index, bool) and index >= 0, "Invalid target_index.")
        require(index not in indices and target.get("segment_id") not in ids, "Duplicate target identity.")
        require(isinstance(target.get("segment_id"), str) and target["segment_id"], "Missing real segment ID.")
        group = target.get("semantic_group_id")
        require(isinstance(group, str) and group and group not in semantic, "Missing or duplicate semantic_group_id.")
        require(isinstance(target.get("source_record_index"), int) and target["source_record_index"] >= 0
                and isinstance(target.get("source_turn_id"), int) and target["source_turn_id"] >= 0
                and bool(target.get("source_episode_id")), "Missing real source record/turn identity.")
        indices.add(index); ids.add(target["segment_id"]); semantic.add(group)
        original = target.get("original", {})
        choices = original.get("choices", [])
        answer_index = original.get("answer_index")
        require(len(choices) == 4 and len(set(choices)) == 4 and all(isinstance(x, str) and x.strip() for x in choices),
                "Original real question must have four distinct nonempty choices.")
        require(isinstance(answer_index, int) and not isinstance(answer_index, bool) and 0 <= answer_index < 4,
                "Invalid original answer index.")
        gold = target.get("scorer_metadata", {}).get("gold")
        require(isinstance(gold, str) and gold == choices[answer_index] and bool(normalize_short_answer(gold)),
                "Gold must exactly equal original choices[answer_index].")
        require(target.get("scorer_metadata", {}).get("aliases", []) == [], "Primary EM cannot add aliases.")
        inputs = target.get("inputs", {})
        require(set(inputs) == set(PROMPTS), "Exactly original and two fixed untrained rewrites are required.")
        require(len(set(inputs.values())) == 3, "Original and rewrites must be distinct.")
        for name, query in inputs.items():
            require(isinstance(query, str) and query.strip(), "Empty generation query.")
            for answer in choices:
                require(not re.search(r"(?<!\w)" + re.escape(normalize_short_answer(answer)) + r"(?!\w)",
                                      normalize_short_answer(query)), f"Answer/choice text leaked into {name} for target {index}.")
    control = panel.get("control_spec", {})
    for name in ["donor_checkpoint", "donor_gold", "donor_source_run_id", "donor_source_raw_generations"]:
        require(isinstance(control.get(name), str) and control[name], f"Missing donor provenance: {name}.")
    for name in ["donor_checkpoint_sha256", "donor_source_raw_generations_sha256"]:
        require(bool(re.fullmatch("[0-9a-f]{64}", control.get(name, ""))), f"Missing donor SHA256: {name}.")
    require(all(normalize_short_answer(x["scorer_metadata"]["gold"]) != normalize_short_answer(control["donor_gold"])
                for x in targets), "Donor gold must differ from every target gold.")
    return panel


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_source(panel):
    selection = panel["selection"]
    source = Path(selection["source_dataset_path"])
    require(source.is_file() and replay.sha256_file(source) == selection["source_dataset_sha256"], "Real source dataset hash changed.")
    wanted = {x["source_record_index"]: x for x in panel["targets"]}
    require(len(wanted) == len(panel["targets"]), "Repeated source episode selected.")
    found = set()
    with source.open(encoding="utf-8") as handle:
        for index, line in enumerate(x for x in handle if x.strip()):
            if index not in wanted:
                continue
            episode, target = json.loads(line), wanted[index]
            require(episode["episode_id"] == target["source_episode_id"]
                    and episode["semantic_group_id"] == target["semantic_group_id"]
                    and episode["split"] == "train", "Source episode identity/split changed.")
            require(canonical_hash(episode) == target["source_episode_sha256"], "Real source episode contents changed.")
            query = episode["turns"][target["source_turn_id"]]["query"]
            require(canonical_hash(query) == target["source_query_sha256"], "Source query hash changed.")
            require(target["original"] == {"query": query["text"], "choices": query["choices"],
                                            "answer_index": query["target_index"]}, "Panel question/choices do not match real source.")
            require(target.get("source_prefix_is_audit_only_not_reader_input") is True,
                    "Source history must be provenance-only, never a Reader input.")
            require(target.get("source_prefix") == episode["turns"][:target["source_turn_id"]+1], "Source prefix provenance changed.")
            found.add(index)
    require(found == set(wanted), "Source record not found.")
    control = panel["control_spec"]
    checkpoint = Path(control["donor_checkpoint"])
    raw = Path(control["donor_source_raw_generations"])
    require(replay.sha256_file(checkpoint) == control["donor_checkpoint_sha256"], "Donor checkpoint bytes changed.")
    require(replay.sha256_file(raw) == control["donor_source_raw_generations_sha256"], "Donor behavioral evidence changed.")
    records = [r for r in read_rows(raw) if r.get("run_id") == control["donor_source_run_id"]
               and r.get("step") == 256 and r.get("condition") == "matched"
               and r.get("prompt_id") in ["original_open", "paraphrase_open"]]
    require(len(records) == 2 and {r["prompt_id"] for r in records} == {"original_open", "paraphrase_open"}, "Missing donor success evidence.")
    require(all(normalize_short_answer(r["raw"]) == normalize_short_answer(control["donor_gold"])
                and r["strict_correct"] for r in records), "Donor image is not behaviorally verified for its declared gold.")
    donor = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(donor["step"] == 256 and tuple(donor["latent_fp32"].shape) == (1, 4, 128, 128)
            and donor["latent_fp32"].dtype == torch.float32 and bool(torch.isfinite(donor["latent_fp32"]).all()), "Invalid donor latent.")
    return donor["latent_fp32"], {**control, "donor_latent_sha256": canonical_tensor_sha256(donor["latent_fp32"]),
                                   "success_evidence_count": len(records), "source_dataset_sha256": selection["source_dataset_sha256"]}


def run_id(target_index, seed, arm):
    return f"target-{target_index:03d}-seed-{seed:02d}-{arm}"


def run_binding(target, seed, arm, plan_hash, initial_hash):
    return {"schema": SCHEMA + ".run", "target_index": target["target_index"], "segment_id": target["segment_id"],
            "semantic_group_id": target["semantic_group_id"], "source_episode_id": target["source_episode_id"],
            "topic": target.get("topic"), "stratum": target.get("stratum"), "seed": seed, "arm": arm,
            "target_canonical_sha256": canonical_hash(target), "plan_sha256": plan_hash,
            "initial_latent_sha256": initial_hash, "optimizer": OPTIMIZER,
            "training_prompt": "original_open", "rewrites_used_in_training": False,
            "append_eos": arm == "B", "eos_lambda": 1.0 if arm == "B" else None,
            "baseline": "fresh paired A; no old ambient trajectory parity claim",
            "formal_shared_memory_success": False}


def write_inventory(directory):
    files = sorted(x for x in directory.rglob("*") if x.is_file() and x.name != "artifact_inventory.json")
    inventory = {"schema": SCHEMA + ".inventory", "files": [
        {"path": str(x.relative_to(directory)).replace("\\", "/"), "bytes": x.stat().st_size,
         "sha256": replay.sha256_file(x)} for x in files]}
    replay.write_json(directory / "artifact_inventory.json", inventory)
    return inventory


def verify_completed_run(directory, expected_binding):
    require((directory / "terminal.json").is_file() and (directory / "artifact_inventory.json").is_file(),
            f"Partial run retained; refusing overwrite: {directory}")
    terminal = json.loads((directory / "terminal.json").read_text(encoding="utf-8"))
    require(terminal.get("status") == "completed" and terminal.get("optimizer_steps") == 256,
            f"Incomplete/failed run retained: {directory}")
    inventory = json.loads((directory / "artifact_inventory.json").read_text(encoding="utf-8"))
    listed = set()
    for item in inventory["files"]:
        path = (directory / item["path"]).resolve(strict=True)
        require(directory.resolve() in path.parents and item["path"] not in listed, "Unsafe/duplicate artifact entry.")
        listed.add(item["path"])
        require(path.stat().st_size == item["bytes"] and replay.sha256_file(path) == item["sha256"], "Completed artifact SHA mismatch.")
    actual = {str(x.relative_to(directory)).replace("\\", "/") for x in directory.rglob("*")
              if x.is_file() and x.name != "artifact_inventory.json"}
    require(listed == actual, "Completed artifact file set changed.")
    required = {"manifest.json", "metrics.jsonl", "raw_generations.jsonl", "terminal.json", "checkpoint_index.jsonl"}
    required.update(f"checkpoints/step-{step:03d}.pt" for step in STEPS)
    require(required <= listed, "Completed run lacks required raw checkpoint artifacts.")
    checkpoints = read_rows(directory / "checkpoint_index.jsonl")
    require([x["step"] for x in checkpoints] == STEPS, "Completed checkpoint schedule changed.")
    for item in checkpoints:
        require(item["path"].replace("\\", "/") == f"checkpoints/step-{item['step']:03d}.pt"
                and replay.sha256_file(directory/item["path"]) == item["file_sha256"], "Completed checkpoint index hash changed.")
    require(json.loads((directory / "manifest.json").read_text(encoding="utf-8")) == expected_binding, "Completed run binding changed.")
    metrics = read_rows(directory / "metrics.jsonl")
    require([r["step"] for r in metrics] == list(range(1, 257)), "Completed run has missing/duplicate updates.")
    rows = read_rows(directory / "raw_generations.jsonl")
    validate_run_rows(rows, expected_binding)
    return rows


def validate_run_rows(rows, binding):
    expected = {(step, condition, prompt) for step in STEPS
                for condition in (CONDITIONS if step == 256 else ["matched"]) for prompt in PROMPTS}
    keys = [(r["step"], r["condition"], r["prompt_id"]) for r in rows]
    require(len(keys) == len(expected) and set(keys) == expected, "Incomplete/duplicate generation grid.")
    for row in rows:
        require(all(row[k] == binding[k] for k in ["target_index", "segment_id", "seed", "arm"]), "Generation identity drift.")
        require(row["prompt_exposed_in_training"] == (row["prompt_id"] == "original_open"), "Heldout rewrite exposure drift.")
        require(row["decoding"] == "raw_greedy_32_original_eos", "Generation policy drift.")


@torch.no_grad()
def evaluate(runtime, target, binding, image, step, directory):
    records = []
    with old.preserve_rng():
        for condition in (CONDITIONS if step == 256 else ["matched"]):
            selected = image if condition == "matched" else runtime["controls"][condition]
            for prompt in PROMPTS:
                output = paired.teacher(runtime, selected, prompt, eos=True, require_grad=False)
                gold_ids = output.target_ids[0, :output.answer_token_count].cpu().tolist()
                generated = generate_short_answer(model=runtime["reader"], processor=runtime["processor"],
                    image=selected.to(runtime["reader_device"]), query=target["inputs"][prompt],
                    device=runtime["reader_device"], max_new_tokens=32)
                diagnostics = generation_diagnostics(generated, target["scorer_metadata"]["gold"], gold_ids)
                row = {"schema": SCHEMA + ".generation", **{k: binding[k] for k in ["target_index", "segment_id", "semantic_group_id", "source_episode_id", "topic", "stratum", "seed", "arm"]},
                       "run_id": run_id(binding["target_index"], binding["seed"], binding["arm"]), "step": step,
                       "condition": condition, "prompt_id": prompt, "prompt_exposed_in_training": prompt == "original_open",
                       "decoding": "raw_greedy_32_original_eos", **generated, **diagnostics,
                       "teacher_forced_answer_ce": float(output.answer_loss), "teacher_forced_eos_ce": float(output.eos_loss),
                       "teacher_forced_answer_token_accuracy": float(output.teacher_forced_answer_accuracy),
                       "teacher_forced_eos_token_accuracy": float(output.teacher_forced_eos_accuracy)}
                replay.append_jsonl(directory / "raw_generations.jsonl", row)
                records.append(row)
    print(json.dumps({"stage": "checkpoint", "run_id": run_id(binding["target_index"], binding["seed"], binding["arm"]),
                      "step": step, "records": len(records)}), flush=True)
    return records


def train_one(runtime, target, seed, arm, initial, directory, plan_hash):
    binding = run_binding(target, seed, arm, plan_hash, initial["latent_sha256"])
    if directory.exists():
        rows = verify_completed_run(directory, binding)
        print(json.dumps({"stage": "verified_completed_run_reused", "run_id": directory.name}), flush=True)
        return rows
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "checkpoints").mkdir()
    replay.write_json(directory / "manifest.json", binding)
    started = time.monotonic()
    try:
        configure_strict_cuda_determinism(0)
        oracle = VAELatentOracle(vae=runtime["vae"], initial_latent=initial["latent_fp32"].to(runtime["vae_device"]),
                                compute_dtype=torch.bfloat16)
        require(canonical_tensor_sha256(oracle.latent_fp32) == initial["latent_sha256"], "Paired device-copy initialization changed.")
        optimizer = torch.optim.Adam([oracle.latent_fp32], lr=.05, betas=(.9, .999), eps=1e-8,
                                     weight_decay=0., amsgrad=False)
        records = []
        for step in range(257):
            if step in STEPS:
                path = directory / "checkpoints" / f"step-{step:03d}.pt"
                digest = old.save_tensor_payload(path, {"step": step, "latent_fp32": oracle.latent_fp32,
                    "optimizer": optimizer.state_dict(), "rng": old.capture_rng(), "run_binding": binding})
                replay.append_jsonl(directory / "checkpoint_index.jsonl", {"step": step,
                    "path": str(path.relative_to(directory)), "file_sha256": digest,
                    "latent_sha256": canonical_tensor_sha256(oracle.latent_fp32)})
                with torch.no_grad():
                    image = oracle.image().cpu()
                records += evaluate(runtime, target, binding, image, step, directory)
            if step == 256:
                break
            optimizer.zero_grad(set_to_none=True)
            output = paired.teacher(runtime, oracle.image(), "original_open", eos=arm == "B", require_grad=True)
            output.loss.backward()
            grad = oracle.latent_fp32.grad
            require(grad is not None and bool(torch.isfinite(grad).all()) and bool(torch.isfinite(output.loss)), "Invalid connected loss/gradient.")
            metric = {"step": step + 1, "loss_before_step": float(output.loss.detach()),
                      "gradient_l2": float(grad.double().norm()), "training_prompt": "original_open",
                      "answer_token_count": int(output.answer_token_count if arm == "B" else output.target_ids.numel())}
            if arm == "B":
                metric.update(answer_ce=float(output.answer_loss.detach()), eos_ce=float(output.eos_loss.detach()))
            optimizer.step()
            require(bool(torch.isfinite(oracle.latent_fp32).all()), "Optimizer produced nonfinite latent.")
            optimizer.zero_grad(set_to_none=True)
            replay.append_jsonl(directory / "metrics.jsonl", metric)
            if (step + 1) % 64 == 0:
                print(json.dumps({"stage": "training", "run_id": directory.name, **metric}), flush=True)
            del output
        replay.frozen_audit(runtime["vae"], runtime["reader"])
        validate_run_rows(records, binding)
        replay.write_json(directory / "terminal.json", {"status": "completed", "optimizer_steps": 256,
            "elapsed_seconds": time.monotonic() - started, "raw_generation_count": len(records),
            "formal_shared_memory_success": False})
        write_inventory(directory)
        return records
    except BaseException as error:
        replay.write_json(directory / "terminal.json", {"status": "failed", "error": str(error),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic()-started,
            "formal_shared_memory_success": False})
        raise


def summarize(records, target_ids, seeds, arms):
    expected_runs = {(t, s, a) for t in target_ids for s in seeds for a in arms}
    require({(r["target_index"], r["seed"], r["arm"]) for r in records} == expected_runs, "Missing question-seed-arm runs.")
    groups = defaultdict(list)
    for row in records:
        if row["step"] == 256:
            groups[(row["arm"], row["condition"], row["prompt_id"])].append(row)
    cells = []
    for (arm, condition, prompt), group in sorted(groups.items()):
        require(len(group) == len(target_ids)*len(seeds), "Wrong question-seed denominator.")
        by_target = defaultdict(list)
        for row in group:
            by_target[row["target_index"]].append(row)
        cells.append({"arm": arm, "condition": condition, "prompt_id": prompt, "question_count": len(target_ids),
            "seeds_per_question": len(seeds), "question_seed_count": len(group),
            "exact_match_count": sum(r["strict_correct"] for r in group),
            "macro_question_EM": sum(sum(r["strict_correct"] for r in v)/len(v) for v in by_target.values())/len(by_target),
            "answer_prefix_correct_count": sum(r["answer_prefix_token_exact"] for r in group),
            "overgeneration_count": sum(r["overgeneration"] for r in group),
            "generated_answer_token_accuracy_macro": sum(r["generated_answer_token_accuracy"] for r in group)/len(group),
            "teacher_forced_answer_token_accuracy_macro": sum(r["teacher_forced_answer_token_accuracy"] for r in group)/len(group),
            "per_question_EM_counts": {str(t): sum(r["strict_correct"] for r in v) for t,v in sorted(by_target.items())}})
    return {"schema": SCHEMA + ".summary", "target_indices": target_ids, "seeds": seeds, "arms": arms,
        "completed_run_count": len(expected_runs), "optimizer_updates": 256*len(expected_runs),
        "raw_generation_count": len(records), "endpoint_cells": cells,
        "denominator": "question × seed; prompts/checkpoints/controls are repeated measures, not new independent questions",
        "scientific_scope": "multi-question independent VAE-latent oracles; no shared writer or cross-question generalization claim",
        "formal_shared_memory_success": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets-json", type=Path, required=True)
    parser.add_argument("--target-indices", type=int, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3])
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-multistart-root", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--vae-device", default="cuda:0")
    parser.add_argument("--reader-device", default="cuda:1")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with lane_lock(args.output_dir):
        return execute(args)


def execute(args):
    for values, name in [(args.target_indices,"targets"),(args.seeds,"seeds"),(args.arms,"arms")]:
        require(len(set(values)) == len(values) and len(values)>0, f"Duplicate/empty {name}.")
    require(all(isinstance(x,int) and x>=0 for x in args.seeds), "Seeds must be nonnegative.")
    panel = validate_panel(json.loads(args.targets_json.read_text(encoding="utf-8")))
    targets = {x["target_index"]:x for x in panel["targets"]}
    require(set(args.target_indices) <= set(targets), "Unknown selected target index.")
    donor, donor_audit = verify_source(panel)
    plan = {"schema": SCHEMA + ".plan", "targets_json_sha256": replay.sha256_file(args.targets_json),
            "target_indices": args.target_indices, "seeds": args.seeds, "arms": args.arms,
            "optimizer": OPTIMIZER, "checkpoint_steps": STEPS, "donor": donor_audit,
            "training_commit": replay.command_output(["git","rev-parse","HEAD"]),
            "source_hashes": {p: replay.sha256_file(ROOT/p) for p in ["scripts/experiments/run_r11_open_eos_multiquestion.py",
                "scripts/experiments/run_r11_open_eos_paired.py", "scripts/experiments/run_r11_mcq_open_multistart.py",
                "scripts/experiments/run_r11_open_answer_replay.py", "scripts/train/latent_r11_vae_oracle.py",
                "src/vision_memory/reader/open_eos.py", "src/vision_memory/reader/open_answer.py",
                "src/vision_memory/reader/qwen3vl.py", "src/vision_memory/reader/deterministic_resize.py",
                "src/vision_memory/dreamlite/latent_codec.py"]}}
    plan_hash = canonical_hash(plan)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "plan.json"
    if plan_path.exists():
        require(json.loads(plan_path.read_text(encoding="utf-8")) == plan, "Existing output plan differs; choose a new output directory.")
    else:
        require(not any(x.name != ".lane.lock" for x in args.output_dir.iterdir()), "Existing nonempty output has no matching plan; refusing overwrite.")
        replay.write_json(plan_path, plan)
    for target_index in args.target_indices:
        for seed in args.seeds:
            for arm in args.arms:
                directory=args.output_dir/"runs"/run_id(target_index,seed,arm)
                if directory.exists():
                    require((directory/"terminal.json").exists() and json.loads((directory/"terminal.json").read_text()).get("status")=="completed",
                            f"Partial/failed run retained; refusing overwrite: {directory}")
    started = time.monotonic()
    try:
        legacy_config_path = ROOT/"configs/experiments/r11_open_eos_paired.json"
        legacy_config = paired.load_config(legacy_config_path)
        legacy_args = argparse.Namespace(**vars(args))
        legacy_args.config = legacy_config_path
        attempt_root = args.output_dir / "runtime_attempts"
        attempt_root.mkdir(exist_ok=True)
        attempt_index = len(list(attempt_root.iterdir()))
        legacy_args.output_dir = attempt_root / f"attempt-{attempt_index:03d}"
        legacy_args.output_dir.mkdir(exist_ok=False)
        legacy_args.mode = "audit"
        runtime = paired.load_runtime(legacy_args, legacy_config)
        with torch.no_grad():
            donor_image = decode_model_latents_unit_interval(runtime["vae"], donor.to(runtime["vae_device"],dtype=torch.bfloat16)).cpu()
        runtime["controls"] = {"blank":runtime["controls"]["blank"], "fixed_donor":donor_image}
        control_record = {**donor_audit,
            "reference_sha256":canonical_tensor_sha256(runtime["reference"]),
            "donor_image_sha256":canonical_tensor_sha256(donor_image)}
        control_path = args.output_dir/"control_audit.json"
        if control_path.exists():
            require(json.loads(control_path.read_text(encoding="utf-8")) == control_record, "Resumed control images changed.")
        else:
            replay.write_json(control_path, control_record)
        records=[]
        for target_index in args.target_indices:
            target=targets[target_index]
            runtime["target"]=target
            for seed in args.seeds:
                initial=old.make_initial_latent(runtime["reference"],seed=seed,rho=.1)
                for arm in args.arms:
                    directory=args.output_dir/"runs"/run_id(target_index,seed,arm)
                    records+=train_one(runtime,target,seed,arm,initial,directory,plan_hash)
        end_sources = {p: replay.sha256_file(ROOT/p) for p in plan["source_hashes"]}
        require(end_sources == plan["source_hashes"], "Executed source files changed during the lane.")
        require(replay.sha256_file(args.targets_json) == plan["targets_json_sha256"], "Preregistered target file changed during the lane.")
        initial_models = json.loads((legacy_args.output_dir/"manifest.json").read_text(encoding="utf-8"))["model_snapshot_payloads"]
        end_models = replay.snapshot_bindings(args)
        require(end_models == initial_models, "Model snapshot payloads changed during the lane.")
        replay.frozen_audit(runtime["vae"], runtime["reader"])
        replay.write_json(args.output_dir/"end_bindings.json", {"sources_unchanged":True,"targets_unchanged":True,
            "models_unchanged":True,"source_hashes":end_sources,"model_snapshot_payloads":end_models})
        summary=summarize(records,args.target_indices,args.seeds,args.arms)
        summary["elapsed_seconds"]=time.monotonic()-started
        replay.write_json(args.output_dir/"summary.json",summary)
        replay.write_json(args.output_dir/"terminal.json",{"status":"completed", "elapsed_seconds":summary["elapsed_seconds"],
            "completed_run_count":summary["completed_run_count"],"optimizer_updates":summary["optimizer_updates"],
            "formal_shared_memory_success":False})
        return 0
    except BaseException as error:
        replay.write_json(args.output_dir/"terminal.json",{"status":"failed","error":str(error),
            "traceback":traceback.format_exc(),"elapsed_seconds":time.monotonic()-started,"formal_shared_memory_success":False})
        traceback.print_exc()
        return 1


if __name__=="__main__":
    raise SystemExit(main())
