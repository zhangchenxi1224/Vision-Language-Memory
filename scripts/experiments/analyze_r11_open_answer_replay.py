"""Analyze a technically complete old-R11 open replay using only Python stdlib.

Preserve raw answers and the preregistered exact-match metric. Auxiliary string
matches never award semantic credit. All outputs go to a fresh analysis directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "vision_memory.r11-open-answer-replay-analysis.v1"
CONDITIONS = ("matched", "blank", "donor")
PROMPTS = ("original_open", "paraphrase_open")
REVERSE_CYCLIC4 = ((3, 2, 1, 0), (2, 1, 0, 3), (1, 0, 3, 2), (0, 3, 2, 1))
INITIAL_SHA256 = "719e92867b60546b21b281cfc633ab782c8ce2274bfb41c6b3cee6d673e74eaa"
INPUT_FILES = (
    "config.json", "manifest.json", "summary.json", "terminal.json",
    "generations.jsonl", "mcq_anchor.jsonl", "initial_latent_verification.json",
    "image_reproduction.json", "model_snapshot_verification_end.json", "endpoint_integrity.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def normalize(text: str) -> str:
    """The stage-1 frozen scorer, duplicated to avoid importing Torch for analysis."""
    require(isinstance(text, str), "Answers must be strings.")
    value = " ".join(text.casefold().split())
    while value and (value[-1].isspace() or unicodedata.category(value[-1]).startswith("P")):
        value = value[:-1]
    return value


def score(raw: str, gold: str) -> dict[str, Any]:
    normalized, expected = normalize(raw), normalize(gold)
    require(bool(expected), "Gold must be nonempty after normalization.")
    correct = normalized == expected
    present = bool(re.search(r"(?<!\w)" + re.escape(expected) + r"(?!\w)", normalized))
    extra = present and not correct
    status = ("empty" if not normalized else "exact_short_answer" if correct else
              "extra_text_or_explanation" if extra else "different_answer")
    return {
        "raw": raw, "normalized": normalized, "normalized_expected": expected,
        "strict_correct": correct, "format_status": status,
        "has_extra_text": extra, "expected_text_present": present,
    }


def read_inputs(run_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    values, hashes = {}, {}
    for name in INPUT_FILES:
        raw = (run_dir / name).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8-sig")
        values[name] = ([json.loads(line) for line in text.splitlines() if line.strip()]
                        if name.endswith(".jsonl") else json.loads(text))
    return values, hashes


def validate(data: Mapping[str, Any]) -> None:
    config, manifest = data["config.json"], data["manifest.json"]
    summary, terminal = data["summary.json"], data["terminal.json"]
    require(config.get("schema") == "vision_memory.r11-open-answer-replay-config.v1", "Wrong config schema.")
    require(config.get("conditions") == list(CONDITIONS) and config.get("prompts") == list(PROMPTS),
            "Expected the frozen three conditions and two prompts.")
    require(config.get("generation") == {"do_sample": False, "max_new_tokens": 32, "eos_policy": "original_model"},
            "Generation config drifted.")
    targets = config["targets"]
    require(len(targets) == 8 and [t["target_index"] for t in targets] == list(range(8)), "Need eight ordered targets.")
    require(manifest.get("config") == config, "Saved config differs from run manifest.")
    require(terminal.get("status") == "completed" and terminal.get("technical_passed") is True,
            "Replay terminal is not technically completed.")
    require(terminal.get("generation_count") == 48 and terminal.get("mcq_anchor_count") == 32,
            "Terminal record counts are incomplete.")
    for key in ("technical_passed", "complete_records", "all_images_exactly_reproduced", "snapshots_unchanged"):
        require(summary.get(key) is True, f"Technical summary gate failed: {key}.")
    initial = data["initial_latent_verification.json"]
    require(initial.get("passed") is True and initial.get("observed_sha256") == INITIAL_SHA256
            and initial.get("expected_sha256") == INITIAL_SHA256, "Blank latent verification failed.")
    snapshots = data["model_snapshot_verification_end.json"]
    require(snapshots.get("passed") is True and snapshots.get("bindings") == manifest.get("model_snapshot_payloads_start"),
            "Model snapshots did not remain unchanged.")
    image_rows = data["image_reproduction.json"]["targets"]
    require(len(image_rows) == 8 and {r["target_index"] for r in image_rows} == set(range(8)),
            "Need eight image reproduction records.")
    images = {r["target_index"]: r for r in image_rows}
    require(all(r.get("exactly_equal") is True and r["decoded_sha256"] == r["saved_sha256"] for r in image_rows),
            "An endpoint image was not exactly reproduced.")
    integrity = data["endpoint_integrity.json"]["endpoints"]
    require(len(integrity) == 8 and {r["target_index"] for r in integrity} == set(range(8)),
            "Need eight endpoint integrity records.")
    for row in integrity:
        artifact = targets[row["target_index"]]["artifact"]
        require(row["file_sha256"] == artifact["endpoint_file_sha256"]
                and row["latent_sha256"] == artifact["endpoint_tensor_sha256"], "Endpoint provenance mismatch.")

    rows = data["generations.jsonl"]
    expected = {(i, c, p) for i in range(8) for c in CONDITIONS for p in PROMPTS}
    keys = [(r["target_index"], r["condition"], r["prompt_id"]) for r in rows]
    require(len(rows) == 48 and len(set(keys)) == 48 and set(keys) == expected,
            "Need exactly 48 unique and complete generation records.")
    blank_image_hashes = set()
    for row in rows:
        target = targets[row["target_index"]]
        gold = target["scorer_metadata"]["gold"]
        require(target["scorer_metadata"].get("aliases") == [], "Aliases must remain empty.")
        require(target["original"]["choices"][target["original"]["answer_index"]] == gold, "Gold/source disagreement.")
        require(row["segment_id"] == target["segment_id"] and row["query"] == target["inputs"][row["prompt_id"]],
                "Question or target ID differs from frozen input.")
        require(row["scorer_metadata"] == target["scorer_metadata"] and row["scorer"] == score(row["raw"], gold),
                "Saved scoring disagrees with independently recomputed exact match.")
        require(isinstance(row["raw_with_special_tokens"], str) and isinstance(row["chat_prompt"], str),
                "Missing preserved raw text or serialized prompt.")
        generated, input_ids = row["generated_token_ids"], row["input_token_ids"]
        require(isinstance(generated, list) and isinstance(input_ids, list) and bool(input_ids), "Missing token records.")
        require(all(isinstance(x, int) and not isinstance(x, bool) for x in generated + input_ids), "Invalid token IDs.")
        require(len(generated) <= 32 and row["generated_token_count"] == len(generated)
                and row["prompt_token_count"] == len(input_ids), "Token count drift.")
        eos_reached = any(token in row["eos_token_ids"] for token in generated)
        reason = "eos" if eos_reached else "token_limit" if len(generated) >= 32 else "other_stop"
        require(row["eos_reached"] is eos_reached and row["truncated"] is (not eos_reached)
                and row["finish_reason"] == reason, "EOS/truncation metadata is inconsistent.")
        if row["condition"] == "blank":
            require(row["latent_sha256"] == INITIAL_SHA256, "Blank condition used the wrong latent.")
            blank_image_hashes.add(row["image_sha256"])
        else:
            image_index = target["donor_target_index"] if row["condition"] == "donor" else row["target_index"]
            donor = targets[target["donor_target_index"]]
            require(donor["scorer_metadata"]["gold"] != gold, "Donor must have a different gold string.")
            if row["condition"] == "donor":
                require(row["donor_target_index"] == donor["target_index"]
                        and row["donor_segment_id"] == donor["segment_id"], "Donor assignment drift.")
            require(row["latent_sha256"] == targets[image_index]["artifact"]["endpoint_tensor_sha256"]
                    and row["image_sha256"] == images[image_index]["decoded_sha256"], "Condition image provenance mismatch.")
    require(len(blank_image_hashes) == 1, "Blank image changed between questions.")

    anchors = data["mcq_anchor.jsonl"]
    anchor_keys = [(r["target_index"], r["view_index"]) for r in anchors]
    require(len(anchors) == 32 and len(set(anchor_keys)) == 32
            and set(anchor_keys) == {(i, v) for i in range(8) for v in range(4)}, "Need all 32 unique MCQ anchors.")
    for row in anchors:
        target = targets[row["target_index"]]
        permutation = REVERSE_CYCLIC4[row["view_index"]]
        answer_index = permutation.index(target["original"]["answer_index"])
        logits = row["choice_logits"]
        require(row["permutation"] == list(permutation) and row["answer_index"] == answer_index,
                "MCQ anchor changed its original reverse-cyclic view.")
        require(row["choices"] == [target["original"]["choices"][i] for i in permutation], "MCQ choices drifted.")
        require(len(logits) == 4 and all(math.isfinite(x) for x in logits) and math.isfinite(row["ce"]),
                "Nonfinite or malformed anchor scores.")
        predicted = max(range(4), key=lambda i: logits[i])
        require(row["correct"] is True and row["predicted_index"] == predicted == answer_index,
                "Original MCQ anchor was not reproduced on all 32 views.")
        require(row["image_sha256"] == images[row["target_index"]]["decoded_sha256"], "MCQ image provenance mismatch.")
    require(summary["mcq_anchor"].get("count") == 32 and summary["mcq_anchor"].get("correct") == 32
            and summary["mcq_anchor"].get("all_correct") is True, "MCQ summary disagrees with rows.")
    for condition in CONDITIONS:
        for prompt in PROMPTS:
            selected = [r for r in rows if r["condition"] == condition and r["prompt_id"] == prompt]
            observed = summary["by_condition_and_prompt"][condition][prompt]
            correct = sum(r["scorer"]["strict_correct"] for r in selected)
            require(observed["count"] == 8 and observed["strict_correct"] == correct
                    and observed["strict_accuracy"] == correct / 8, "Summary accuracy disagrees with raw answers.")


def analyze(data: Mapping[str, Any], input_hashes: Mapping[str, str], run_dir: Path) -> dict[str, Any]:
    validate(data)
    config = data["config.json"]
    rows = sorted(data["generations.jsonl"], key=lambda r: (r["target_index"], CONDITIONS.index(r["condition"]), PROMPTS.index(r["prompt_id"])))
    indexed = {(r["target_index"], r["condition"], r["prompt_id"]): r for r in rows}
    flat = []
    for row in rows:
        target = config["targets"][row["target_index"]]
        donor = config["targets"][target["donor_target_index"]]
        flat.append({
            "target_index": row["target_index"], "segment_id": row["segment_id"],
            "condition": row["condition"], "prompt_id": row["prompt_id"], "query": row["query"],
            "gold": target["scorer_metadata"]["gold"], "raw": row["raw"],
            "raw_json": json.dumps(row["raw"], ensure_ascii=False),
            "raw_with_special_tokens": row["raw_with_special_tokens"],
            "normalized": row["scorer"]["normalized"], "strict_correct": row["scorer"]["strict_correct"],
            "format_status": row["scorer"]["format_status"], "has_extra_text": row["scorer"]["has_extra_text"],
            "expected_text_present": row["scorer"]["expected_text_present"], "truncated": row["truncated"],
            "finish_reason": row["finish_reason"], "generated_token_count": row["generated_token_count"],
            "donor_target_index": donor["target_index"] if row["condition"] == "donor" else None,
            "donor_gold": donor["scorer_metadata"]["gold"] if row["condition"] == "donor" else None,
            "donor_gold_string_match": (normalize(row["raw"]) == normalize(donor["scorer_metadata"]["gold"]))
                if row["condition"] == "donor" else None,
        })
    metrics = []
    for condition in CONDITIONS:
        for prompt in PROMPTS:
            selected = [r for r in flat if r["condition"] == condition and r["prompt_id"] == prompt]
            correct = sum(r["strict_correct"] for r in selected)
            metrics.append({
                "condition": condition, "prompt_id": prompt, "count": 8, "strict_correct": correct,
                "strict_accuracy": correct / 8, "truncated_count": sum(r["truncated"] for r in selected),
                "extra_text_count": sum(r["has_extra_text"] for r in selected),
                "empty_count": sum(r["format_status"] == "empty" for r in selected),
                "different_answer_count": sum(r["format_status"] == "different_answer" for r in selected),
            })
    paired = []
    for prompt in PROMPTS:
        for control in ("blank", "donor"):
            categories: dict[str, list[int]] = {k: [] for k in (
                "both_correct", "matched_only_correct", "control_only_correct", "neither_correct")}
            for i in range(8):
                m, c = (indexed[(i, cond, prompt)]["scorer"]["strict_correct"] for cond in ("matched", control))
                key = "both_correct" if m and c else "matched_only_correct" if m else "control_only_correct" if c else "neither_correct"
                categories[key].append(i)
            paired.append({"prompt_id": prompt, "control": control, "count": 8,
                           **{k: len(v) for k, v in categories.items()}, "target_indices": categories,
                           "paired_accuracy_difference": (len(categories["matched_only_correct"]) - len(categories["control_only_correct"])) / 8})
    agreement = []
    for condition in CONDITIONS:
        raw_equal, normalized_equal = [], []
        for i in range(8):
            a, b = (indexed[(i, condition, p)] for p in PROMPTS)
            if a["raw"] == b["raw"]:
                raw_equal.append(i)
            if normalize(a["raw"]) == normalize(b["raw"]):
                normalized_equal.append(i)
        agreement.append({"condition": condition, "count": 8, "raw_equal_count": len(raw_equal),
                          "normalized_equal_count": len(normalized_equal), "normalized_agreement": len(normalized_equal) / 8,
                          "raw_equal_target_indices": raw_equal, "normalized_equal_target_indices": normalized_equal})
    donor_summary = []
    for prompt in PROMPTS:
        selected = [r for r in flat if r["condition"] == "donor" and r["prompt_id"] == prompt]
        hits = [r["target_index"] for r in selected if r["donor_gold_string_match"]]
        donor_summary.append({"prompt_id": prompt, "count": 8, "donor_gold_string_match_count": len(hits),
                              "donor_gold_string_match_rate": len(hits) / 8, "target_indices": hits})
    anchors = []
    for row in sorted(data["mcq_anchor.jsonl"], key=lambda r: (r["target_index"], r["view_index"])):
        gold_logit = row["choice_logits"][row["answer_index"]]
        margin = gold_logit - max(v for i, v in enumerate(row["choice_logits"]) if i != row["answer_index"])
        anchors.append({"target_index": row["target_index"], "view_index": row["view_index"],
                        "correct": True, "ce": row["ce"], "gold_minus_best_wrong_margin": margin,
                        "ce_difference_from_old": row.get("ce_difference_from_old"),
                        "choice_logits": row["choice_logits"], "permutation": row["permutation"]})
    return {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
        "input_file_sha256": dict(input_hashes), "source_config_sha256": data["manifest.json"]["config_sha256"],
        "technical_passed": True, "generation_count": 48, "mcq_anchor_count": 32,
        "condition_prompt_metrics": metrics, "paired_controls": paired, "cross_prompt_agreement": agreement,
        "donor_string_matches_exploratory": donor_summary, "answers": flat, "mcq_anchors": anchors,
        "semantic_review": {"status": "pending_root_raw_text_review", "automatic_semantic_credit": False},
        "diagnostic_definitions": {
            "extra_text_count": "Only non-exact answers containing the current gold string. Zero does not mean there are no unrelated sentences or extra words.",
            "truncated_count": "Output lacks a recorded EOS; independent of exact match and extra-text detection.",
            "donor_gold_string_match": "Normalized equality to the donor question's gold text; exploratory lexical diagnostic only.",
        },
        "interpretation_limits": [
            "Eight fixed questions; conditions and prompts are repeated measures, not independent replications.",
            "Donor-gold matches and extra-text detection are string diagnostics, not semantic correctness.",
            "Prompt agreement may reflect the same wrong answer; it is not an accuracy metric.",
            "Truncation and extra text are separate flags and may overlap.",
            "No endpoint optimization, multistart convergence claim, semantic relabeling, or significance test is performed.",
        ],
    }


def csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    require(bool(rows), f"No rows for {path.name}.")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def md_raw(raw: str) -> str:
    """JSON string notation preserves whitespace while keeping a single table cell."""
    return "<code>" + html.escape(json.dumps(raw, ensure_ascii=False), quote=False).replace("|", "&#124;") + "</code>"


def markdown(result: Mapping[str, Any]) -> str:
    lines = ["# Old R11 open-answer replay", "", "The replay is technically complete: 48 open generations and 32/32 correct original MCQ anchor views.",
             "No semantic credit has been added. Raw answers await manual review; the primary metric remains the preregistered normalized exact match.", "",
             "## Exact-match accuracy", "", "| Image condition | Prompt | Correct / 8 | Accuracy | Truncated | Gold-containing extra text | Empty |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["condition_prompt_metrics"]:
        lines.append(f"| {row['condition']} | {row['prompt_id']} | {row['strict_correct']}/8 | {row['strict_accuracy']:.1%} | {row['truncated_count']} | {row['extra_text_count']} | {row['empty_count']} |")
    lines += ["", "Gold-containing extra text means the expected string appears with additional content, including possible negation. A zero count does not mean the responses contain no unrelated sentences or extra words. This is not semantic correctness, and truncation is an independent flag.", "",
              "## Paired matched-versus-control outcomes", "", "| Prompt | Control | Both correct | Matched only | Control only | Neither | Matched − control |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in result["paired_controls"]:
        lines.append(f"| {row['prompt_id']} | {row['control']} | {row['both_correct']} | {row['matched_only_correct']} | {row['control_only_correct']} | {row['neither_correct']} | {row['paired_accuracy_difference']:+.1%} |")
    lines += ["", "## Cross-prompt agreement", "", "| Image condition | Raw strings identical / 8 | Normalized strings identical / 8 |",
              "|---|---:|---:|"]
    for row in result["cross_prompt_agreement"]:
        lines.append(f"| {row['condition']} | {row['raw_equal_count']}/8 | {row['normalized_equal_count']}/8 |")
    lines += ["", "Agreement can be agreement on an incorrect answer.", "", "## Donor-string diagnostic", "",
              "This only checks whether the answer under the donor image equals the donor question's gold string. Entities and attributes differ; a hit is not a semantic counterfactual score.", "",
              "| Prompt | Donor-gold string matches / 8 | Query target indices |", "|---|---:|---|"]
    for row in result["donor_string_matches_exploratory"]:
        lines.append(f"| {row['prompt_id']} | {row['donor_gold_string_match_count']}/8 | {row['target_indices']} |")
    lines += ["", "## Preserved raw answers", "",
              "Cells show the entire original string in JSON notation: `\\n` and spaces preserve whitespace; no text is shortened. `EM` is exact match, `T` is truncation, and `E` is extra wording containing the expected string. The CSV/JSON retain the raw strings directly.", "",
              "| Target | Gold | Prompt | Matched answer | Blank answer | Donor answer |", "|---:|---|---|---|---|---|"]
    by_key = {(r["target_index"], r["condition"], r["prompt_id"]): r for r in result["answers"]}
    for i in range(8):
        for prompt in PROMPTS:
            cells = []
            for condition in CONDITIONS:
                row = by_key[(i, condition, prompt)]
                flags = f"EM={int(row['strict_correct'])}, T={int(row['truncated'])}, E={int(row['has_extra_text'])}"
                cells.append(f"{md_raw(row['raw'])}<br>{flags}")
            gold = by_key[(i, "matched", prompt)]["gold"]
            lines.append(f"| {i:02d} | {gold} | {prompt} | " + " | ".join(cells) + " |")
    lines += ["", "## Technical and interpretation notes", ""]
    lines.extend(f"- {text}" for text in result["interpretation_limits"])
    margins = [r["gold_minus_best_wrong_margin"] for r in result["mcq_anchors"]]
    lines += [f"- MCQ gold-minus-best-wrong margin: min {min(margins):.6g}, mean {sum(margins)/32:.6g}, max {max(margins):.6g}.",
              f"- Frozen source config SHA-256: `{result['source_config_sha256']}`.",
              "- The analysis independently rechecks row coverage, strict scores, image/latent provenance, EOS metadata, the saved technical gates and all 32 MCQ predictions. It does not reload models or deserialize tensors.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Completed replay output directory.")
    parser.add_argument("--output-dir", type=Path, help="Fresh output directory; default: RUN_DIR/analysis.")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / "analysis"
    data, hashes = read_inputs(run_dir)
    result = analyze(data, hashes, run_dir)
    # Validate all inputs before creating outputs; never overwrite an earlier analysis.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(markdown(result), encoding="utf-8")
    for name, key in (
        ("condition_prompt_metrics.csv", "condition_prompt_metrics"), ("generations.csv", "answers"),
        ("paired_controls.csv", "paired_controls"), ("cross_prompt_agreement.csv", "cross_prompt_agreement"),
        ("donor_string_matches.csv", "donor_string_matches_exploratory"), ("mcq_anchors.csv", "mcq_anchors"),
    ):
        csv_write(output_dir / name, result[key])
    print(json.dumps({"technical_passed": True, "generation_count": 48, "mcq_anchor_count": 32,
                      "output_dir": str(output_dir), "semantic_review": "pending_root_raw_text_review"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"Analysis refused: {error}", file=sys.stderr)
        raise SystemExit(1)
