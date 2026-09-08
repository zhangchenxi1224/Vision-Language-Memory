"""Own two disjoint question lanes on an already allocated four-H200 notebook."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path("/inspire/ssd/project/exploration-topic/czxs26210936")
MODELS = Path("/inspire/qb-ilm/project/exploration-topic/czxs26210936/models/vision-language-memory")
PANEL_SHA = "d356238fd5c267812dcf28d214ab062fd43388bb6b53b78602f0c1e8f5b36672"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def lane_commands(panel, panel_path, root, python=sys.executable):
    all_indices = [t["target_index"] for t in panel["targets"]]
    lanes = panel["lanes"]
    assigned = [i for lane in lanes for i in lane["target_indices"]]
    if len(lanes) != 2 or sorted(assigned) != sorted(all_indices) or len(set(assigned)) != len(assigned):
        raise ValueError("The two lanes must cover each question exactly once")
    if [lane["gpu_pair"] for lane in lanes] != [[0, 1], [2, 3]]:
        raise ValueError("Expected two disjoint physical GPU pairs")
    commands = []
    for index, lane in enumerate(lanes):
        command = [python, "-u", str(ROOT / "scripts/experiments/run_r11_open_eos_multiquestion.py"),
                   "--targets-json", str(panel_path), "--target-indices", *map(str, lane["target_indices"]),
                   "--seeds", *map(str, panel["seeds"]), "--arms", *panel["arms"],
                   "--output-dir", str(root / f"lane-{index}"),
                   "--old-multistart-root", str(PROJECT / "runs/vision-language-memory-r11-open/multistart-5d06b76-20260907-round02"),
                   "--dreamlite", str(MODELS / "DreamLite-mobile"),
                   "--reader", str(MODELS / "Qwen3-VL-4B-Instruct"),
                   "--vae-device", "cuda:0", "--reader-device", "cuda:1"]
        commands.append((lane, command))
    return commands


def stop_children(children):
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--max-hours", type=float, default=2.5)
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != args.expected_commit or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("An exact, clean committed checkout is required")
    panel_path = ROOT / "configs/experiments/r11_open_eos_multiquestion_targets.json"
    if hashlib.sha256(panel_path.read_bytes()).hexdigest() != PANEL_SHA:
        raise RuntimeError("Prospective target panel bytes changed")
    panel = json.loads(panel_path.read_text())
    commands = lane_commands(panel, panel_path, args.output_root)
    gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"], text=True)
    gpu_rows = [line.split(",") for line in gpu_text.splitlines() if line.strip()]
    if len(gpu_rows) != 4 or any("H200" not in row[1] or int(row[2].strip()) < 140000 for row in gpu_rows):
        raise RuntimeError("Actual four full-memory H200 devices are required")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    if processes.strip():
        raise RuntimeError("Existing GPU process detected; inspect ownership before launch")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    children = []
    logs = []
    identity = {"instance": "dl-base-h200x4-20260907", "source_commit": commit,
                "panel_sha256": PANEL_SHA, "planned_runs": 128, "planned_questions": 16,
                "started_at_utc": now(), "pid": os.getpid(), "max_hours": args.max_hours}
    write_json(args.output_root / "launch.json", {**identity, "gpu_inventory": gpu_text,
               "lanes": [{"spec": lane, "command": command} for lane, command in commands]})
    interrupted = []
    def receive_signal(signum, frame):
        interrupted.append(signum)
    signal.signal(signal.SIGTERM, receive_signal)
    signal.signal(signal.SIGINT, receive_signal)
    try:
        for index, (lane, command) in enumerate(commands):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, lane["gpu_pair"]))
            log = (args.output_root / f"lane-{index}.log").open("x")
            logs.append(log)
            children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT))
        while True:
            codes = [child.poll() for child in children]
            write_json(args.output_root / "status.json", {**identity, "state": "running",
                       "checked_at_utc": now(), "lane_pids": [c.pid for c in children], "lane_returncodes": codes})
            if interrupted:
                raise RuntimeError(f"Supervisor interrupted by signal {interrupted[0]}")
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f"A lane failed: {codes}; inspect lane logs and retained artifacts")
            if all(code == 0 for code in codes):
                break
            if time.monotonic() - started > args.max_hours * 3600:
                raise TimeoutError("Bounded runtime exhausted; completed runs retained, partial runs not overwritten")
            time.sleep(5)
        sys.path.insert(0, str(ROOT))
        from scripts.experiments.run_r11_open_eos_multiquestion import summarize
        records = []
        for index in range(2):
            lane_root = args.output_root / f"lane-{index}"
            terminal = json.loads((lane_root / "terminal.json").read_text())
            if terminal.get("status") != "completed" or terminal.get("completed_run_count") != 64:
                raise RuntimeError("Lane terminal is not a complete 64-run panel")
            for path in sorted((lane_root / "runs").glob("*/raw_generations.jsonl")):
                records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        summary = summarize(records, list(range(16)), [0, 1, 2, 3], ["A", "B"])
        lengths = {str(r["target_index"]): r["answer_token_count"] for r in records
                   if r["step"] == 256 and r["arm"] == "A" and r["seed"] == 0
                   and r["condition"] == "matched" and r["prompt_id"] == "original_open"}
        summary["actual_answer_token_lengths_by_question"] = lengths
        summary["question_counts_by_answer_token_length"] = dict(Counter(map(str, lengths.values())))
        length_groups = defaultdict(list)
        for record in records:
            if record["step"] == 256 and record["condition"] == "matched":
                length_groups[(record["arm"], record["prompt_id"], record["answer_token_count"])].append(record)
        summary["answer_length_descriptive_cells"] = [
            {"arm": key[0], "prompt_id": key[1], "answer_token_count": key[2],
             "questions": len({r["target_index"] for r in group}), "question_seed_pairs": len(group),
             "exact_match_count": sum(r["strict_correct"] for r in group),
             "prefix_correct_count": sum(r["answer_prefix_token_exact"] for r in group),
             "overgeneration_count": sum(r["overgeneration"] for r in group)}
            for key, group in sorted(length_groups.items())]
        cells = {(c["arm"], c["condition"], c["prompt_id"]): c for c in summary["endpoint_cells"]}
        summary["image_dependence"] = [
            {"arm": arm, "prompt_id": prompt,
             "matched_minus_blank_EM": cells[(arm, "matched", prompt)]["macro_question_EM"] - cells[(arm, "blank", prompt)]["macro_question_EM"],
             "matched_minus_donor_EM": cells[(arm, "matched", prompt)]["macro_question_EM"] - cells[(arm, "fixed_donor", prompt)]["macro_question_EM"]}
            for arm in ["A", "B"] for prompt in ["original_open", "paraphrase_open", "new_rewrite_open"]]
        write_json(args.output_root / "summary.json", summary)
        write_json(args.output_root / "terminal.json", {**identity, "status": "completed",
                   "elapsed_seconds": time.monotonic() - started,
                   "raw_generation_count": len(records), "formal_shared_memory_success": False})
        write_json(args.output_root / "status.json", {**identity, "state": "completed", "completed_at_utc": now()})
    except BaseException as exc:
        stop_children(children)
        write_json(args.output_root / "status.json", {**identity, "state": "failed", "error": str(exc), "checked_at_utc": now()})
        write_json(args.output_root / "terminal.json", {**identity, "status": "failed", "error": str(exc),
                   "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
