"""Fail-closed, offline recomputation for the fixed canonical teacher 4/7 audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROTOCOL = "R11-New-Teacher47-Swap-Diagnostic"
PREFIX = "vision_memory.r11-new-teacher-swap"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/experiments/r11_new_teacher47_swap_audit.json"
CONFIG_BYTES_SHA256 = "3d11e602d8ac70cc3716ad375db92fe7c83fc8ba2c13d692d3e063d3f477ef1f"
CONFIG_CANONICAL_SHA256 = "3f65185444e7c711571cc7cf19c56ee0e52937912e83bbb1403c319ee8a7f536"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_config() -> dict[str, Any]:
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "Config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(canonical_sha(value) == CONFIG_CANONICAL_SHA256, "Config canonical hash drift.")
    require(value["protocol"] == PROTOCOL and value["schema"] == f"{PREFIX}-config.v1", "Config identity drift.")
    require([t["target_index"] for t in value["targets"]] == [4, 7], "Teacher selection drift.")
    return value


def recompute(logits: Sequence[float], ordered_target: int, permutation: Sequence[int]) -> dict[str, Any]:
    require(len(logits) == 4 and all(type(x) in (int, float) and math.isfinite(x) for x in logits),
            "Exactly four finite raw logits required.")
    require(sorted(permutation) == [0, 1, 2, 3] and type(ordered_target) is int and 0 <= ordered_target < 4,
            "Invalid candidate order or target.")
    values = torch.tensor(logits, dtype=torch.float32)
    require(bool(torch.isfinite(values).all()), "FP32 logits overflow.")
    ce = float(torch.logsumexp(values, dim=0) - values[ordered_target])
    predicted = permutation[int(values.argmax())]
    return {"ce": ce, "predicted_index": predicted, "correct": predicted == permutation[ordered_target]}


def expected_image(target: int, condition: str) -> str:
    require(target in (4, 7) and condition in ("own", "donor", "reset"), "Invalid audit cell.")
    return "reset" if condition == "reset" else str(target if condition == "own" else 11 - target)


def validate_rows(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
                  image_hashes: Mapping[str, str]) -> dict[tuple[int, str, int], dict[str, Any]]:
    require(len(rows) == config["expected_rows"] == 24, "Missing or extra rows.")
    require(set(image_hashes) == {"4", "7", "reset"} and len(set(image_hashes.values())) == 3,
            "All three actual images must be distinct and bound.")
    require(all(isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                for v in image_hashes.values()), "Invalid image hash.")
    targets = {t["target_index"]: t["target_segment"] for t in config["targets"]}
    cells = {}
    for row in rows:
        index, condition, view = row["target_index"], row["condition"], row["view_index"]
        require(type(index) is int and index in targets and condition in config["conditions"]
                and type(view) is int and 0 <= view < 4, "Unexpected audit row.")
        key = (index, condition, view)
        require(key not in cells, "Duplicate row.")
        segment = targets[index]
        permutation = config["permutations"][view]
        source = expected_image(index, condition)
        query = segment["query"]
        require(row["schema"] == f"{PREFIX}-row.v1", "Wrong row schema.")
        require(row["segment_id"] == segment["segment_id"] and row["query_sha256"] == canonical_sha(query),
                "Query/target provenance drift.")
        require(row["permutation"] == permutation and row["original_answer_index"] == query["target_index"],
                "Answer/permutation drift.")
        require(row["image_key"] == source and row["image_sha256"] == image_hashes[source],
                "Actual image/donor binding drift.")
        derived = recompute(row["choice_logits_ordered"], permutation.index(query["target_index"]), permutation)
        require(type(row["ce"]) in (int, float) and math.isfinite(row["ce"])
                and abs(row["ce"] - derived["ce"]) <= 1e-6, "Reported CE differs from raw logits.")
        require(type(row["correct"]) is bool and row["correct"] == derived["correct"]
                and type(row["predicted_index"]) is int and row["predicted_index"] == derived["predicted_index"],
                "Reported prediction differs from raw logits.")
        cells[key] = derived
    require(set(cells) == {(t, c, v) for t in (4, 7) for c in config["conditions"] for v in range(4)},
            "Incomplete Cartesian audit grid.")
    return cells


def aggregate(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
              image_hashes: Mapping[str, str]) -> dict[str, Any]:
    cells = validate_rows(rows, config, image_hashes)
    gate = config["gate"]
    results = []
    for target in (4, 7):
        scores = {c: {"mean_ce": sum(cells[target, c, v]["ce"] for v in range(4)) / 4,
                      "accuracy": sum(cells[target, c, v]["correct"] for v in range(4)) / 4}
                  for c in config["conditions"]}
        replay = scores["own"]["accuracy"] == 1 and scores["own"]["mean_ce"] <= gate["own_mean_ce_max"]
        contrasts = {}
        for alternative in ("donor", "reset"):
            own, other = scores["own"], scores[alternative]
            improved = sum(cells[target, "own", v]["ce"] < cells[target, alternative, v]["ce"] for v in range(4))
            relative = None if other["mean_ce"] <= 0 else (other["mean_ce"] - own["mean_ce"]) / other["mean_ce"]
            gap = own["accuracy"] - other["accuracy"]
            contrasts[alternative] = {"accuracy_gap": gap, "relative_ce_improvement": relative,
                "improved_views": improved, "passed": bool(relative is not None
                    and relative >= gate["minimum_relative_ce_improvement"]
                    and gap >= gate["minimum_accuracy_gap"] and improved == gate["required_improved_views"])}
        results.append({"target_index": target, "scores": scores, "teacher_replay_gate": replay,
                        "contrasts": contrasts, "distinguishability_gate": all(c["passed"] for c in contrasts.values())})
    replay = all(t["teacher_replay_gate"] for t in results)
    distinguishable = all(t["distinguishability_gate"] for t in results)
    return {"schema": f"{PREFIX}-comparison.v1", "protocol": PROTOCOL,
        "raw_row_recomputation_passed": True, "row_count": len(rows), "targets": results,
        "teacher_replay_gate": replay, "distinguishability_gate": distinguishable,
        "d0_diagnostic_gate": replay and distinguishable,
        "decision": "eligible_for_d1_implementation_and_preflight" if replay and distinguishable
                    else "hold_d1_teacher_replay_failed" if not replay else "hold_d1_teacher_specificity_failed",
        "formal_success": False, "phase2_allowed": False, "shared_training_allowed": False}


def audit_delivery(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    inventory = json.loads((root / "artifact_inventory.json").read_text(encoding="utf-8"))
    require(inventory["schema"] == f"{PREFIX}-inventory.v1", "Inventory schema drift.")
    listed = set()
    for item in inventory["artifacts"]:
        rel = item["path"]
        require(isinstance(rel, str) and "\\" not in rel and not Path(rel).is_absolute()
                and all(p not in (".", "..", "") for p in rel.split("/")) and rel not in listed,
                "Unsafe or duplicate artifact path.")
        path = root / rel
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), "Escaping artifact.")
        require(path.is_file() and path.stat().st_size == item["bytes"] and sha256_file(path) == item["sha256"],
                "Artifact bytes/hash mismatch.")
        listed.add(rel)
    observed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
                and p.name != "artifact_inventory.json"}
    require(observed == listed and inventory["artifact_count"] == len(listed), "Incomplete inventory.")
    require({"manifest.json", "result.json", "terminal.json", "receipts.jsonl", "REPORT.md", "images.pt",
             "environment.txt", "runtime.json", "config.json", "snapshots-end.json",
             "execution-counts.json"}.issubset(listed),
             "Required artifacts missing.")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    require(json.loads((root / "config.json").read_text(encoding="utf-8")) == config, "Saved config drift.")
    require(manifest["protocol"] == PROTOCOL and manifest["config_sha256"] == canonical_sha(config), "Manifest drift.")
    require(terminal["status"] == "completed" and terminal["exit_code"] == 0
            and terminal["engineering_gate"] is True and result["engineering_gate"] is True, "Technical run invalid.")
    require(manifest["git_commit"] == terminal["git_commit"] == result["git_commit"], "Code commit mismatch.")
    require(terminal["manifest_sha256"] == sha256_file(root / "manifest.json")
            and terminal["result_sha256"] == sha256_file(root / "result.json"), "Terminal artifact mismatch.")
    require(result["snapshots_unchanged"] is True and result["all_parameters_frozen"] is True, "Frozen evidence missing.")
    expected_counters = {"unet_forward_calls": config["guardrails"]["unet_forward_calls"],
        "reader_forward_calls": config["expected_reader_forward_calls"],
        "optimizer_steps": config["guardrails"]["optimizer_steps"]}
    execution_counts = json.loads((root / "execution-counts.json").read_text(encoding="utf-8"))
    require(execution_counts == {"observed": expected_counters, "expected": expected_counters}
            and result["counters"] == expected_counters, "Execution counts invalid.")
    from vision_memory.repro import canonical_tensor_sha256

    images = torch.load(root / "images.pt", map_location="cpu", weights_only=True)
    hashes = {key: canonical_tensor_sha256(value) for key, value in images.items()}
    require(hashes == manifest["image_hashes"], "Raw image hash mismatch.")
    lines = (root / "receipts.jsonl").read_text(encoding="utf-8").splitlines()
    require(all(lines), "Blank receipt.")
    comparison = aggregate([json.loads(line) for line in lines], config, hashes)
    require(result["comparison"] == comparison, "Claimed comparison disagrees with independent recomputation.")
    return {"passed": True, "artifact_count": len(listed), "comparison": comparison}
