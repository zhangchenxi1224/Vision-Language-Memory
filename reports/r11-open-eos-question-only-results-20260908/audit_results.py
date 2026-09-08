"""Independently verify supplemental JSON bytes, coverage, prompts and raw scores.

Uses only Python's standard library. Does not require or modify model tensors.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = HERE / "raw"
ANCHORS = ["original_open", "paraphrase_open", "new_rewrite_open"]
QUESTIONS = ["question_only_restored", *[f"question_only_{i:02d}" for i in range(2, 6)]]


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def normalized(text):
    text = " ".join(text.casefold().split())
    while text and (text[-1].isspace() or unicodedata.category(text[-1]).startswith("P")):
        text = text[:-1]
    return text


def key(r):
    return r["arm"], r["seed"], r["condition"], r["prompt_id"]


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    inventory = load(RAW / "download_manifest.json")
    for item in inventory["files"]:
        data = (RAW / item["path"]).read_bytes()
        assert len(data) == item["bytes"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    terminal = load(RAW / "terminal.json")
    assert terminal["status"] == "completed" and terminal["training_updates"] == 0
    assert terminal["raw_generation_count"] == 576 and terminal["exact_anchor_parity_count"] == 216
    manifest = load(RAW / "question_only_manifest.json")
    config = load(ROOT / "configs/experiments/r11_open_eos_question_only.json")
    assert manifest["config"] == config
    for name in QUESTIONS:
        expected_prompt = config["fixed_prefix"] + config["question_sentences"][name] + config["fixed_suffix"]
        assert manifest["actual_prompts"][name] == expected_prompt
    original = rows(ROOT / "reports/r11-open-eos-results-20260908/raw/raw_generations.jsonl")
    parent = {key(r): r for r in original if r["arm"] in ["A", "B", "C"] and r["step"] == 256}
    records = rows(RAW / "raw_generations.jsonl")
    expected = {(a, s, c, p) for a in ["A", "B", "C"] for s in range(8)
                for c in ["matched", "blank", "fixed_donor"] for p in ANCHORS + QUESTIONS}
    assert len(records) == 576 and {key(r) for r in records} == expected
    parity_count = 0
    grouped = defaultdict(list)
    for r in records:
        assert r["step"] == 256 and r["decoding"] == "raw_greedy_32_original_eos"
        assert r["eos_token_ids"] == [151645, 151643] and r["gold_token_ids"] == [59614]
        assert r["normalized_expected"] == "ambient" and len(r["generated_token_ids"]) <= 32
        assert r["strict_correct"] == (normalized(r["raw"]) == "ambient")
        prefix = r["generated_token_ids"][:1] == [59614]
        assert r["answer_prefix_token_exact"] == prefix
        assert r["overgeneration"] == (prefix and any(t not in r["eos_token_ids"] for t in r["generated_token_ids"][1:]))
        if r["prompt_id"] in QUESTIONS:
            assert not r["prompt_exposed_in_training"]
            query = manifest["actual_prompts"][r["prompt_id"]]
            chat = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + query + "<|im_end|>\n<|im_start|>assistant\n"
            assert r["chat_prompt"] == chat
        else:
            old = parent[key(r)]
            for field in ["generated_token_ids", "input_token_ids", "chat_prompt", "raw", "strict_correct"]:
                assert r[field] == old[field]
            parity_count += 1
        grouped[(r["arm"], r["condition"], r["prompt_id"])].append(r)
    assert parity_count == 216
    recorded_parity = rows(RAW / "anchor_parity.jsonl")
    assert len(recorded_parity) == 216 and all(r["exact_parity"] for r in recorded_parity)
    endpoints = rows(RAW / "endpoint_audit.jsonl")
    before = manifest["endpoint_bindings_before_evaluation"]
    assert len(endpoints) == 24
    assert {(r["arm"], r["seed"]) for r in endpoints} == {(a, s) for a in ["A", "B", "C"] for s in range(8)}
    for r, binding in zip(endpoints, before):
        assert all(r[k] == binding[k] for k in binding) and r["unchanged_after_evaluation"]
    summary = load(RAW / "summary.json")
    for cell in summary["cells"]:
        values = grouped[(cell["arm"], cell["condition"], cell["prompt_id"])]
        assert len(values) == cell["n"] == 8
        for field in ["strict_correct", "answer_prefix_token_exact", "overgeneration"]:
            assert cell[field] == sum(bool(r[field]) for r in values)
    lookup = {key(r): r for r in records}
    for row in summary["per_seed_consistency"]:
        values = [lookup[(row["arm"], row["seed"], "matched", p)] for p in QUESTIONS]
        assert row["correct_out_of_five"] == sum(r["strict_correct"] for r in values)
        assert row["all_five_correct"] == all(r["strict_correct"] for r in values)
    write("endpoint_answers.json", [{k: r[k] for k in ["arm", "seed", "prompt_id", "raw", "strict_correct", "overgeneration"]}
                                     for r in records if r["condition"] == "matched"])
    result = {"passed": True, "file_sha256_checks": len(inventory["files"]), "generation_records": 576,
              "independently_verified_anchor_parity": parity_count, "endpoint_records": 24,
              "new_question_matched_generations": 120, "training_updates": 0,
              "endpoint_tensor_hash_scope": "remote_recorded; local audit does not load remote tensors",
              "checks": ["complete_coverage", "identical_instructions", "unexposed_questions", "full_output_scoring",
                         "answer_prefix_and_overgeneration", "original_input_and_output_token_parity", "summary_recalculation"]}
    write("audit.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
