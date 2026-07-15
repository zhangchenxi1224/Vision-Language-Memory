from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_state(raw: str) -> str:
    return raw.strip().split(maxsplit=1)[0].rstrip("+")


def parse_sacct(text: str, job_id: str) -> dict[str, Any]:
    matches = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("|")
        if len(fields) < 5:
            raise ValueError(f"Malformed sacct row for {job_id}: {line!r}")
        if fields[0] == job_id:
            matches.append(fields)
    if len(matches) != 1:
        raise ValueError(f"Expected one parent sacct row for {job_id}, found {len(matches)}.")
    job, state, elapsed, exit_code, allocated_tres = matches[0][:5]
    return {
        "job_id": job,
        "state": normalized_state(state),
        "state_raw": state,
        "elapsed_seconds": None if not elapsed else int(elapsed),
        "exit_code": exit_code,
        "allocated_tres": allocated_tres,
    }


def query_sacct(job_id: str) -> str:
    command = [
        "sacct",
        "-n",
        "-P",
        "-j",
        job_id,
        "--format=JobIDRaw,State,ElapsedRaw,ExitCode,AllocTRES",
    ]
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_report(
    manifest_path: Path,
    *,
    sacct_query: Callable[[str], str] = query_sacct,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise ValueError("Submission manifest has no jobs.")
    run_root = Path(str(manifest["run_root"])).resolve()

    records = []
    by_stage: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"jobs": 0, "terminal_jobs": 0, "failures": 0, "gpu_hours": 0.0}
    )
    for name, submission in jobs.items():
        job_id = submission.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise ValueError(f"Job {name!r} was not submitted and has no Slurm ID.")
        status = parse_sacct(sacct_query(job_id), job_id)
        gpus = int(submission["resources"]["gpus"])
        elapsed = status["elapsed_seconds"]
        gpu_hours = None if elapsed is None else elapsed * gpus / 3600.0
        terminal = status["state"] in TERMINAL_STATES
        failed = terminal and status["state"] != "COMPLETED"
        stdout = run_root / "logs" / f"{name}_{job_id}.out"
        stderr = run_root / "logs" / f"{name}_{job_id}.err"
        record = {
            "name": name,
            "stage": submission["stage"],
            **status,
            "gpus": gpus,
            "gpu_hours": gpu_hours,
            "terminal": terminal,
            "failed": failed,
            "stdout": artifact_record(stdout) if stdout.is_file() else None,
            "stderr": artifact_record(stderr) if stderr.is_file() else None,
        }
        records.append(record)
        stage = by_stage[str(submission["stage"])]
        stage["jobs"] += 1
        stage["terminal_jobs"] += int(terminal)
        stage["failures"] += int(failed)
        stage["gpu_hours"] += float(gpu_hours or 0.0)

    result_files = sorted(path for path in (run_root / "results").rglob("*") if path.is_file())
    state_counts = Counter(record["state"] for record in records)
    terminal_jobs = sum(int(record["terminal"]) for record in records)
    failures = sum(int(record["failed"]) for record in records)
    return {
        "schema_version": "vision_memory.slurm_ledger.v1",
        "submission_manifest": artifact_record(manifest_path),
        "run_root": str(run_root),
        "commit": manifest.get("commit"),
        "jobs": records,
        "summary": {
            "jobs": len(records),
            "terminal_jobs": terminal_jobs,
            "nonterminal_jobs": len(records) - terminal_jobs,
            "failures": failures,
            "all_completed": terminal_jobs == len(records) and failures == 0,
            "state_counts": dict(sorted(state_counts.items())),
            "gpu_hours": sum(float(record["gpu_hours"] or 0.0) for record in records),
            "by_stage": dict(sorted(by_stage.items())),
        },
        "result_artifacts": [artifact_record(path) for path in result_files],
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest final Slurm status, GPU-hours, logs, and result hashes")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-terminal", action="store_true")
    args = parser.parse_args()
    report = build_report(args.submission.resolve(strict=True))
    output = args.output or args.submission.with_name("slurm_ledger.json")
    write_json_atomic(output, report)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if args.require_terminal and report["summary"]["nonterminal_jobs"]:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
