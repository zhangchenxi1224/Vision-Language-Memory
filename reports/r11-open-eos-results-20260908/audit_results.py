"""Read-only scientific audit of downloaded JSON artifacts; writes derived reports.

Run: python reports/r11-open-eos-results-20260908/audit_results.py
No model loading, training, checkpoint selection or remote resource operations.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import unicodedata

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
STEPS = [0, 16, 32, 64, 128, 192, 256]
PROMPTS = ["original_open", "paraphrase_open", "new_rewrite_open"]
CONDITIONS = ["matched", "blank", "fixed_donor"]


def load(name):
    return json.loads((RAW / name).read_text(encoding="utf-8"))


def rows(name):
    return [json.loads(x) for x in (RAW / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def normalized(text):
    text = " ".join(text.casefold().split())
    while text and (text[-1].isspace() or unicodedata.category(text[-1]).startswith("P")):
        text = text[:-1]
    return text


def csv_write(name, values):
    with (ROOT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    inventory = load("download_manifest.json")
    for item in inventory["files"]:
        data = (RAW / item["path"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    records = rows("raw_generations.jsonl")
    assert len(records) == 864
    expected = {(arm, seed, step, condition, prompt)
                for arm in ["old_open", "A", "B", "C"] for seed in range(8) for step in STEPS
                for condition in (CONDITIONS if step == 256 else ["matched"]) for prompt in PROMPTS}
    keys = [(r["arm"], r["seed"], r["step"], r["condition"], r["prompt_id"]) for r in records]
    assert len(set(keys)) == len(keys) and set(keys) == expected
    for row in records:
        assert row["decoding"] == "raw_greedy_32_original_eos"
        assert row["gold_token_ids"] == [59614] and row["normalized_expected"] == "ambient"
        assert row["eos_token_ids"] == [151645, 151643]
        assert len(row["generated_token_ids"]) <= 32
        assert row["strict_correct"] == (normalized(row["raw"]) == "ambient")
        prefix = row["generated_token_ids"][:1] == [59614]
        assert prefix == row["answer_prefix_token_exact"]
        assert row["generated_answer_token_accuracy"] == float(prefix)
        assert row["overgeneration"] == (prefix and any(x not in row["eos_token_ids"] for x in row["generated_token_ids"][1:]))
        expected_exposure = row["prompt_id"] == "original_open" or (row["arm"] == "C" and row["prompt_id"] == "paraphrase_open")
        assert row["prompt_exposed_in_training"] == expected_exposure
        for field in ["teacher_forced_answer_ce", "teacher_forced_eos_ce", "teacher_forced_answer_token_accuracy"]:
            assert math.isfinite(row[field])
    run_checks = []
    for seed in range(8):
        for arm in ["A", "B", "C"]:
            prefix = f"runs/seed-{seed:02d}-{arm}/"
            assert load(prefix + "terminal.json") == {"status": "completed", "steps": 256}
            metrics = rows(prefix + "metrics.jsonl")
            assert [x["step"] for x in metrics] == list(range(1, 257))
            counts = Counter(x["training_prompt"] for x in metrics)
            assert counts == ({"original_open": 128, "paraphrase_open": 128} if arm == "C" else {"original_open": 256})
            assert all(math.isfinite(x["loss_before_step"]) and math.isfinite(x["gradient_l2"]) for x in metrics)
            if arm != "A":
                assert all(math.isclose(x["loss_before_step"], x["answer_ce"] + x["eos_ce"], rel_tol=2e-6, abs_tol=2e-6) for x in metrics)
            if arm == "A":
                parity = rows(prefix + "legacy_parity.jsonl")
                assert [x["step"] for x in parity] == STEPS
                assert all(x["exact_tensor_parity"] and x["max_abs_difference"] == 0. for x in parity)
            run_checks.append({"seed": seed, "arm": arm, "optimizer_steps": len(metrics), "finite_metrics": True})
    groups = defaultdict(list)
    for row in records:
        groups[(row["arm"], row["step"], row["condition"], row["prompt_id"])].append(row)
    cells = []
    for (arm, step, condition, prompt), group in sorted(groups.items()):
        cells.append({"arm": arm, "step": step, "condition": condition, "prompt": prompt, "n": len(group),
            "exact_match_count": sum(x["strict_correct"] for x in group),
            "correct_answer_prefix_count": sum(x["answer_prefix_token_exact"] for x in group),
            "overgeneration_count": sum(x["overgeneration"] for x in group),
            "generated_answer_token_accuracy": sum(x["generated_answer_token_accuracy"] for x in group)/len(group),
            "teacher_forced_answer_token_accuracy": sum(x["teacher_forced_answer_token_accuracy"] for x in group)/len(group),
            "mean_answer_ce": sum(x["teacher_forced_answer_ce"] for x in group)/len(group),
            "mean_eos_ce": sum(x["teacher_forced_eos_ce"] for x in group)/len(group),
            "prompt_exposed_in_training": group[0]["prompt_exposed_in_training"]})
    csv_write("endpoint_metrics.csv", [x for x in cells if x["step"] == 256 and x["arm"] in ["A", "B", "C"]])
    csv_write("trajectory_metrics.csv", [x for x in cells if x["condition"] == "matched"])
    csv_write("endpoint_raw_answers.csv", [{k: x[k] for k in ["arm", "seed", "condition", "prompt_id", "raw", "strict_correct", "answer_prefix_token_exact", "overgeneration"]}
              for x in records if x["step"] == 256 and x["arm"] in ["A", "B", "C"]])
    deployed = rows("deployment_generations.jsonl")
    assert len(deployed) == 288
    assert all(x["included_in_raw_scientific_metrics"] is False and x["max_new_tokens"] == 8 for x in deployed)
    terminal = load("terminal.json")
    assert terminal["status"] == "completed" and terminal["raw_generation_count"] == len(records)
    manifest = load("manifest.json")
    assert manifest["git_commit"] == "e6e6c8071a6c1311a966bda1aa636748fbb016f6"
    index = dict(zip(keys, records))
    assert all(index[("A", s, t, c, p)]["generated_token_ids"] == index[("old_open", s, t, c, p)]["generated_token_ids"]
               for s in range(8) for t in STEPS for c in (CONDITIONS if t == 256 else ["matched"]) for p in PROMPTS)
    report = {"all_checks_passed": True, "downloaded_file_hashes_verified": len(inventory["files"]),
              "raw_records_recomputed": len(records), "exact_expected_raw_grid": True,
              "deployment_records_separate": len(deployed), "completed_runs": len(run_checks),
              "optimizer_updates_verified": 24*256, "A_checkpoint_parity_receipts_verified": 56,
              "A_and_old_replay_generated_sequences_exact_equal": 216,
              "training_commit": manifest["git_commit"], "elapsed_seconds": terminal["elapsed_seconds"],
              "formal_shared_memory_success": False,
              "tensor_verification_boundary": "56 parity receipts checked locally; raw tensors remain on shared storage"}
    (ROOT / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
