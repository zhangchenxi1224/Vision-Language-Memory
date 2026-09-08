"""Recompute the frozen 16-question A/B experiment with Python standard library.

Run from any directory: python /path/to/this/audit_results.py
Only writes derived JSON/CSV beside this script. No model or training imports.
Raw tensor/model/data bytes remain on Inspire; their receipts are not a local replay.
"""
import csv
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
RAW = ROOT / "raw"
COMMIT = "2c0e41c899910bf0641f16ee724f85bbe3491a7e"
PANEL_SHA = "d356238fd5c267812dcf28d214ab062fd43388bb6b53b78602f0c1e8f5b36672"
STEPS = [0, 16, 32, 64, 128, 192, 256]
PROMPTS = ["original_open", "paraphrase_open", "new_rewrite_open"]
CONDITIONS = ["matched", "blank", "fixed_donor"]


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":")).encode())


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(value):
    value = " ".join(value.casefold().split())
    while value and (value[-1].isspace() or unicodedata.category(value[-1]).startswith("P")):
        value = value[:-1]
    return value


def diagnostics(row):
    """Recompute full-string EM and positional token diagnostics from raw output."""
    gold, tokens, eos = row["gold_token_ids"], row["generated_token_ids"], row["eos_token_ids"]
    prefix = tokens[:len(gold)] == gold
    after = tokens[len(gold):] if prefix else []
    return {"normalized": normalized(row["raw"]),
            "strict_correct": normalized(row["raw"]) == row["normalized_expected"],
            "answer_prefix_token_exact": prefix,
            "generated_answer_token_accuracy": sum(i < len(tokens) and tokens[i] == g for i, g in enumerate(gold))/len(gold),
            "overgeneration": prefix and any(t not in eos for t in after),
            "answer_followed_immediately_by_eos": bool(prefix and after and after[0] in eos)}


def write_csv(name, values):
    require(bool(values), "Empty derived table: " + name)
    with (ROOT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)


def cell(group):
    return {"question_count": len({r["target_index"] for r in group}),
            "question_seed_count": len(group),
            "exact_match_count": sum(r["strict_correct"] for r in group),
            "answer_prefix_correct_count": sum(r["answer_prefix_token_exact"] for r in group),
            "overgeneration_count": sum(r["overgeneration"] for r in group),
            "generated_answer_token_accuracy_macro": sum(r["generated_answer_token_accuracy"] for r in group)/len(group),
            "teacher_forced_answer_token_accuracy_macro": sum(r["teacher_forced_answer_token_accuracy"] for r in group)/len(group),
            "mean_answer_ce": sum(r["teacher_forced_answer_ce"] for r in group)/len(group),
            "mean_eos_ce": sum(r["teacher_forced_eos_ce"] for r in group)/len(group)}


def aggregate(records, fields):
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[f] for f in fields)].append(row)
    return [{**dict(zip(fields, key)), **cell(group)} for key, group in sorted(groups.items())]


def main():
    inventory = load(RAW / "download_manifest.json")
    actual_files = {p.relative_to(RAW).as_posix() for p in RAW.rglob("*") if p.is_file()}
    expected_files = {f["path"] for f in inventory["files"]}
    require(len(expected_files) == len(inventory["files"]) == 784, "Wrong archive file count")
    require(actual_files == expected_files | {"download_manifest.json"}, "Unexpected/missing raw file")
    for item in inventory["files"]:
        require(not Path(item["path"]).is_absolute() and ".." not in Path(item["path"]).parts, "Unsafe path")
        data = (RAW / item["path"]).read_bytes()
        require(len(data) == item["bytes"] and digest(data) == item["sha256"], "Downloaded bytes changed: " + item["path"])
    panel_path = REPO / "configs/experiments/r11_open_eos_multiquestion_targets.json"
    require(digest(panel_path.read_bytes()) == PANEL_SHA, "Frozen panel changed")
    panel = load(panel_path)
    targets = {t["target_index"]: t for t in panel["targets"]}
    require(set(targets) == set(range(16)), "Wrong target set")
    require(len({t["semantic_group_id"] for t in targets.values()}) == 16, "Repeated semantic group")
    require(not ({t["semantic_group_id"] for t in targets.values()} & set(panel["selection"]["excluded_old_semantic_groups"])), "Old question reused")
    require(panel["selection"]["outcomes_used"] is False and panel["source_split"] == "train", "Wrong selection scope")
    for target in targets.values():
        choices = target["original"]["choices"]
        require(target["scorer_metadata"]["gold"] == choices[target["original"]["answer_index"]], "Gold differs from original label")
        require(target["source_prefix_is_audit_only_not_reader_input"] is True, "History was not audit-only")
        require(set(target["inputs"]) == set(PROMPTS), "Missing fixed prompt")
        for query in target["inputs"].values():
            require(all(not re.search(r"(?<!\w)" + re.escape(normalized(c)) + r"(?!\w)", normalized(query)) for c in choices), "Choice/gold leaked into query")
        require(normalized(target["scorer_metadata"]["gold"]) != normalized(panel["control_spec"]["donor_gold"]), "Donor label equals target")

    records, run_manifests, source_receipts, original_json_hashes = [], {}, {}, 0
    for lane in range(2):
        base = RAW / f"lane-{lane}"
        plan, end = load(base / "plan.json"), load(base / "end_bindings.json")
        runtime = load(base / "runtime_attempts/attempt-000/manifest.json")
        require(plan["training_commit"] == runtime["git_commit"] == COMMIT, "Training commit mismatch")
        require(plan["targets_json_sha256"] == PANEL_SHA and plan["target_indices"] == list(range(lane,16,2)), "Wrong lane plan")
        require(plan["arms"] == ["A","B"] and plan["seeds"] == list(range(4)) and plan["checkpoint_steps"] == STEPS, "Wrong arms/seeds/checkpoints")
        require(all(end[k] is True for k in ["models_unchanged", "sources_unchanged", "targets_unchanged"]), "End binding failed")
        require(end["source_hashes"] == plan["source_hashes"], "Source bindings disagree")
        require(end["model_snapshot_payloads"] == runtime["model_snapshot_payloads"], "Model start/end bindings differ")
        require(all(v["passed"] is True for v in end["model_snapshot_payloads"].values()), "Model receipt failed")
        source_receipts.update(plan["source_hashes"])
        termination = runtime["termination_contract"]
        require(termination["assistant_end_token_id"] == 151645 and termination["generation_eos_token_ids"] == [151645,151643], "EOS contract differs")
        control = load(base / "control_audit.json")
        require(control["success_evidence_count"] == 2 and control["source_dataset_sha256"] == panel["source_train_sha256"], "Donor/source receipt failed")
        require(all(control[k] == v for k,v in panel["control_spec"].items()), "Frozen donor specification differs")
        lt = load(base / "terminal.json")
        require(lt["status"] == "completed" and lt["completed_run_count"] == 64 and lt["optimizer_updates"] == 16384, "Incomplete lane")
        directories = sorted(p for p in (base / "runs").iterdir() if p.is_dir())
        require(len(directories) == 64, "Wrong lane run count")
        for directory in directories:
            manifest = load(directory / "manifest.json")
            t,s,a = manifest["target_index"],manifest["seed"],manifest["arm"]
            require(t in plan["target_indices"] and s in range(4) and a in ["A","B"], "Run outside plan")
            require((t,s,a) not in run_manifests and directory.name == f"target-{t:03d}-seed-{s:02d}-{a}", "Duplicate/misnamed run")
            require(manifest["plan_sha256"] == canonical(plan) and manifest["target_canonical_sha256"] == canonical(targets[t]), "Run binding mismatch")
            require(manifest["optimizer"] == plan["optimizer"] and manifest["optimizer"]["steps"] == 256 and manifest["optimizer"]["lr"] == .05, "Optimizer mismatch")
            require(manifest["append_eos"] == (a == "B") and manifest["eos_lambda"] == (1.0 if a == "B" else None), "Loss arm mismatch")
            require(manifest["rewrites_used_in_training"] is False and manifest["training_prompt"] == "original_open", "Rewrite trained")
            run_manifests[(t,s,a)] = manifest
            for item in load(directory / "artifact_inventory.json")["files"]:
                if not item["path"].endswith(".pt"):
                    data = (directory / item["path"]).read_bytes()
                    require(digest(data) == item["sha256"] and len(data) == item["bytes"], "Original run inventory failed")
                    original_json_hashes += 1
            terminal = load(directory / "terminal.json")
            require(terminal["status"] == "completed" and terminal["optimizer_steps"] == 256 and terminal["raw_generation_count"] == 27, "Incomplete run")
            metrics = rows(directory / "metrics.jsonl")
            require([m["step"] for m in metrics] == list(range(1,257)), "Missing/repeated optimizer step")
            require(all(m["training_prompt"] == "original_open" and math.isfinite(m["loss_before_step"]) and math.isfinite(m["gradient_l2"]) for m in metrics), "Invalid training metrics")
            if a == "B":
                require(all(math.isclose(m["loss_before_step"],m["answer_ce"]+m["eos_ce"],rel_tol=2e-6,abs_tol=2e-6) for m in metrics), "Split EOS loss does not sum")
            checkpoints = rows(directory / "checkpoint_index.jsonl")
            require([c["step"] for c in checkpoints] == STEPS, "Checkpoint receipt grid incomplete")
            require(checkpoints[0]["latent_sha256"] == manifest["initial_latent_sha256"], "Initial latent receipt mismatch")
            run_rows = rows(directory / "raw_generations.jsonl")
            require(len(run_rows) == 27 and all((r["target_index"],r["seed"],r["arm"]) == (t,s,a) for r in run_rows), "Run rows mismatch")
            records.extend(run_rows)

    # Verify training-source receipts against the exact historical Git blobs, not current HEAD.
    for name, sha in source_receipts.items():
        blob = subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=REPO)
        require(digest(blob) == sha, "Historical source hash differs: " + name)
    require(set(run_manifests) == {(t,s,a) for t in range(16) for s in range(4) for a in ["A","B"]}, "Incomplete run Cartesian grid")
    for seed in range(4):
        require(len({m["initial_latent_sha256"] for (t,s,a),m in run_manifests.items() if s == seed}) == 1, "Seed initial latent not paired")
    require(len({m["initial_latent_sha256"] for m in run_manifests.values()}) == 4, "Distinct seeds share initialization")

    key_fields = ["target_index","seed","arm","step","condition","prompt_id"]
    keys = [tuple(r[f] for f in key_fields) for r in records]
    expected = {(t,s,a,step,c,p) for t in range(16) for s in range(4) for a in ["A","B"] for step in STEPS
                for c in (CONDITIONS if step == 256 else ["matched"]) for p in PROMPTS}
    require(len(records) == len(set(keys)) == 3456 and set(keys) == expected, "Missing/repeated raw cell")
    for row in records:
        target = targets[row["target_index"]]
        require(row["normalized_expected"] == normalized(target["scorer_metadata"]["gold"]), "Raw scorer gold mismatch")
        require(row["answer_token_count"] == len(row["gold_token_ids"]) > 0, "Invalid answer token count")
        require(row["eos_token_ids"] == [151645,151643] and row["decoding"] == "raw_greedy_32_original_eos", "Scientific generation policy changed")
        require(len(row["generated_token_ids"]) <= 32, "Generation limit changed")
        require(row["prompt_exposed_in_training"] == (row["prompt_id"] == "original_open"), "Wrong rewrite exposure label")
        expected_chat = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + target["inputs"][row["prompt_id"]] + "<|im_end|>\n<|im_start|>assistant\n"
        require(row["chat_prompt"] == expected_chat, "Reader prompt contains unexpected content")
        for field,value in diagnostics(row).items():
            require(row[field] == value, "Raw metric mismatch: " + field)
        require(all(math.isfinite(row[f]) for f in ["teacher_forced_answer_ce","teacher_forced_eos_ce","teacher_forced_answer_token_accuracy"]), "Nonfinite evaluation")
    index = dict(zip(keys,records))
    for t in range(16):
        tokenizations = {tuple(r["gold_token_ids"]) for r in records if r["target_index"] == t}
        require(len(tokenizations) == 1, "Answer tokenization varies unexpectedly")
        for seed in range(4):
            for prompt in PROMPTS:
                require(index[(t,seed,"A",0,"matched",prompt)]["generated_token_ids"] == index[(t,seed,"B",0,"matched",prompt)]["generated_token_ids"], "Paired step-zero generation differs")
        for condition in ["blank","fixed_donor"]:
            for prompt in PROMPTS:
                require(len({tuple(index[(t,s,a,256,condition,prompt)]["generated_token_ids"]) for s in range(4) for a in ["A","B"]}) == 1, "Fixed-image control changes across seeds/arms")

    endpoints = [r for r in records if r["step"] == 256]
    cells = aggregate(endpoints,["arm","condition","prompt_id"])
    remote_summary = load(RAW / "summary.json")
    for remote in remote_summary["endpoint_cells"]:
        local = next(c for c in cells if all(c[f] == remote[f] for f in ["arm","condition","prompt_id"]))
        require(all(local[k] == remote[k] for k in local if k in remote), "Remote summary disagrees with raw")
    write_csv("endpoint_metrics.csv",cells)
    write_csv("trajectory_metrics.csv",aggregate([r for r in records if r["condition"] == "matched"],["arm","step","prompt_id"]))
    matched = [r for r in endpoints if r["condition"] == "matched"]
    write_csv("per_question_metrics.csv",aggregate(matched,["target_index","topic","stratum","normalized_expected","answer_token_count","arm","prompt_id"]))
    write_csv("answer_length_metrics.csv",aggregate(matched,["answer_token_count","arm","prompt_id"]))
    paired = []
    consistency = []
    for t in range(16):
        for s in range(4):
            for p in PROMPTS:
                aa,bb = [index[(t,s,a,256,"matched",p)] for a in ["A","B"]]
                paired.append({"target_index":t,"seed":s,"prompt_id":p,"gold":aa["normalized_expected"],
                               "A_raw":aa["raw"],"B_raw":bb["raw"],"A_EM":aa["strict_correct"],"B_EM":bb["strict_correct"],
                               "B_minus_A_EM":int(bb["strict_correct"])-int(aa["strict_correct"]),
                               "A_prefix":aa["answer_prefix_token_exact"],"B_prefix":bb["answer_prefix_token_exact"],
                               "A_overgeneration":aa["overgeneration"],"B_overgeneration":bb["overgeneration"]})
            for a in ["A","B"]:
                group = [index[(t,s,a,256,"matched",p)] for p in PROMPTS]
                consistency.append({"target_index":t,"seed":s,"arm":a,
                                    "all_three_same_normalized_answer":len({r["normalized"] for r in group}) == 1,
                                    "all_three_correct":all(r["strict_correct"] for r in group)})
    write_csv("paired_endpoints.csv",paired)
    write_csv("rewrite_consistency.csv",consistency)
    write_csv("endpoint_raw_answers.csv",[{k:r[k] for k in ["target_index","seed","arm","condition","prompt_id","normalized_expected","answer_token_count","raw","strict_correct","answer_prefix_token_exact","overgeneration"]} for r in endpoints])
    paired_totals = [{"prompt_id":p,"both_correct":sum(r["A_EM"] and r["B_EM"] for r in paired if r["prompt_id"] == p),
                     "B_only_correct":sum(r["B_minus_A_EM"] == 1 for r in paired if r["prompt_id"] == p),
                     "A_only_correct":sum(r["B_minus_A_EM"] == -1 for r in paired if r["prompt_id"] == p),
                     "both_wrong":sum(not r["A_EM"] and not r["B_EM"] for r in paired if r["prompt_id"] == p)} for p in PROMPTS]
    terminal = load(RAW / "terminal.json")
    require(terminal["status"] == "completed" and terminal["source_commit"] == COMMIT and terminal["raw_generation_count"] == 3456, "Root completion mismatch")
    report = {"all_checks_passed":True,"downloaded_original_files_hash_verified":784,
              "original_run_JSON_inventory_hashes_verified":original_json_hashes,"historical_source_Git_blobs_verified":len(source_receipts),
              "completed_runs":128,"independent_questions":16,"seeds_per_question":4,"optimizer_updates_verified":32768,
              "raw_records_recomputed":3456,"exact_expected_raw_grid":True,"checkpoint_receipts":896,
              "paired_initial_latent_hashes_match":64,"paired_step_zero_generation_matches":192,
              "end_binding_receipts_verified":2,"training_commit":COMMIT,"targets_sha256":PANEL_SHA,
              "elapsed_seconds":terminal["elapsed_seconds"],"paired_EM_counts":paired_totals,
              "rewrite_consistency":[{"arm":a,"n":64,"all_three_same":sum(r["all_three_same_normalized_answer"] for r in consistency if r["arm"] == a),
                                      "all_three_correct":sum(r["all_three_correct"] for r in consistency if r["arm"] == a)} for a in ["A","B"]],
              "formal_shared_memory_success":False,
              "verification_boundary":"Raw strings/tokens and completeness recomputed locally. Teacher-forced metrics, source dataset, checkpoint tensors and model payloads are receipt-checked; no logits, tensors or GPU models loaded locally."}
    (ROOT / "audit.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))


if __name__ == "__main__":
    main()
