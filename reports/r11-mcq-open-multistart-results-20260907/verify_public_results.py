"""Verify the published text records; does not load tensors or run a model."""
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "multistart-5d06b76-20260907-round02"


def check(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    value = " ".join(text.casefold().split())
    while value and (value[-1].isspace() or unicodedata.category(value[-1]).startswith("P")):
        value = value[:-1]
    return value


def main():
    checksums = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = (ROOT / relative).resolve()
        check(path.is_relative_to(ROOT), "Checksum path escapes result directory")
        check(path.is_file() and digest(path) == expected, f"File hash mismatch: {relative}")
        checksums[relative] = expected
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and p.name != "SHA256SUMS" and "__pycache__" not in p.parts}
    check(actual == set(checksums), "Checksum inventory differs from published files")
    for item in read_json(ROOT / "publication_sources.json")["copied_files"]:
        if item["publication_transform"] == "none":
            check(digest(ROOT / item["path"]) == item["source_sha256"], f"Source bytes changed: {item['path']}")
    for path in ROOT.rglob("*.json"):
        read_json(path)
    for path in ROOT.rglob("*.jsonl"):
        rows(path)
    # Public report links must resolve inside the repository.
    for path in ROOT.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#")):
                continue
            check(not re.match(r"^[A-Za-z]:", target), f"Local-only link in {path.name}")
            check((path.parent / target.split("#", 1)[0]).exists(), f"Broken link: {path.name}: {target}")

    summary = read_json(RUN / "summary.json")
    audit = read_json(ROOT / "multistart-full-audit.json")
    check(summary["technical_passed"] and audit["passed"], "Recorded technical audit failed")
    check(summary["formal_success"] is False and audit["formal_success"] is False, "Scientific status changed")
    check(digest(RUN / "manifest.json") == audit["provenance"]["manifest_sha256"], "Audited manifest differs")
    all_gen, all_mcq = rows(RUN / "generations.jsonl"), rows(RUN / "mcq_endpoint.jsonl")
    check(len(all_gen) == summary["generation_count"] == 108, "Generation count")
    check(len(all_mcq) == summary["mcq_count"] == 72, "MCQ count")
    directories = sorted(p for p in (RUN / "runs").iterdir() if p.is_dir())
    check(len(directories) == summary["run_count"] == 18, "Run count")
    totals = dict(metrics=0, latent_index_entries=0, checkpoint_index_entries=0, fixed_probes=0)
    gen_by_run, mcq_by_run = [], []
    for run in directories:
        metrics = rows(run / "metrics.jsonl")
        latent = rows(run / "latent_index.jsonl")
        checkpoints = rows(run / "checkpoint_index.jsonl")
        check([x["optimizer_step"] for x in metrics] == list(range(1, 257)), f"Steps: {run.name}")
        check([x["optimizer_step"] for x in latent] == list(range(257)), f"Latent index: {run.name}")
        check([x["optimizer_step"] for x in checkpoints] == [0, 64, 128, 192, 256], f"Checkpoint index: {run.name}")
        for key, value in zip(totals, [len(metrics), len(latent), len(checkpoints), len(rows(run / "fixed_probes.jsonl"))]):
            totals[key] += value
        gen_by_run.extend(rows(run / "generations.jsonl"))
        mcq_by_run.extend(rows(run / "mcq_endpoint.jsonl"))
    canonical = lambda data: sorted(json.dumps(x, sort_keys=True) for x in data)
    check(canonical(gen_by_run) == canonical(all_gen), "Aggregate generation records differ")
    check(canonical(mcq_by_run) == canonical(all_mcq), "Aggregate MCQ records differ")
    check(totals == dict(metrics=4608, latent_index_entries=4626, checkpoint_index_entries=90, fixed_probes=450), "Record totals")
    for row in all_gen:
        check((normalize(row["raw"]) == "ambient") == row["scorer"]["strict_correct"], "Strict answer score differs")
    for row in all_mcq:
        predicted = max(range(4), key=lambda i: row["choice_logits"][i])
        check((predicted == row["target_index"]) == row["correct"], "MCQ score differs")
    behavior = {}
    for arm in ["mcq", "open"]:
        rebuilt = dict(open_original=0, open_robust=0, mcq_all4=0)
        for seed in range(8):
            rid = f"noise-seed-{seed:02d}-{arm}"
            answers = {r["prompt_id"]: normalize(r["raw"]) == "ambient" for r in all_gen if r["run_id"] == rid and r["condition"] == "matched"}
            check(set(answers) == {"original_open", "paraphrase_open"}, "Paired prompts missing")
            rebuilt["open_original"] += answers["original_open"]
            rebuilt["open_robust"] += all(answers.values())
            scores = [r["correct"] for r in all_mcq if r["run_id"] == rid]
            check(len(scores) == 4, "MCQ views missing")
            rebuilt["mcq_all4"] += all(scores)
            a = rows(RUN / "runs" / f"noise-seed-{seed:02d}-mcq" / "latent_index.jsonl")[0]
            b = rows(RUN / "runs" / f"noise-seed-{seed:02d}-open" / "latent_index.jsonl")[0]
            check(a["latent_sha256"] == b["latent_sha256"], "Paired initial tensor hashes differ")
        for key, count in rebuilt.items():
            check(count / 8 == summary["by_arm"][arm][key], f"Summary disagreement: {arm}/{key}")
        behavior[arm] = rebuilt

    projection = read_json(ROOT / "trajectories/projection.json")
    check(projection["shape"] == [8, 257, 65536], "Projection shape")
    check(projection["fileSHA256Verified"] == 2056, "Recorded projection source audit count")
    for path in projection["paths"]:
        seed = path["seed"]
        run = RUN / "runs" / f"noise-seed-{seed:02d}-mcq"
        check(digest(run / "latent_index.jsonl") == projection["sourceIndexHashes"][seed], "Projection source index differs")
        check(len(path["points"]) == 257, "Projection points missing")
        for step, row in enumerate(rows(run / "metrics.jsonl"), 1):
            check(math.isclose(path["fromStart"][step], row["delta_from_own_start_rms"], abs_tol=6e-9), "Projection distance differs")
            check(math.isclose(path["update"][step], row["actual_update_rms"], abs_tol=6e-11), "Projection update differs")
    geometry = rows(RUN / "geometry_mcq.jsonl")
    for key, index in [("initialDistances", 0), ("endpointDistances", -1)]:
        matrix = geometry[index]["raw_latent"]["rmse_matrix"]
        check(max(abs(projection[key][i][j] - matrix[i][j]) for i in range(8) for j in range(8)) < 1e-6, "Projection distance matrix differs")
    replay = ROOT / "replay-9f86bc9-20260907-round01"
    check(len(rows(replay / "generations.jsonl")) == 48, "Replay generation count")
    check(len(rows(replay / "mcq_anchor.jsonl")) == 32, "Replay MCQ anchor count")
    print(json.dumps({"published_text_verification_passed": True, "files_hashed": len(checksums), "runs": 18, **totals, "generations": 108, "mcq_views": 72, "random_start_success_counts_out_of_8": behavior, "formal_success": False, "raw_tensor_files_reaudited_in_this_check": False}, indent=2))


if __name__ == "__main__":
    main()
