from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vision_memory.repro import assert_determinism_environment, compare_bitwise_repro_reports  # noqa: E402


PROBE = ROOT / "scripts" / "probes" / "lightweight_determinism.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run two fresh lightweight-determinism processes serially in one Slurm allocation"
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, choices=(1, 100), required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def read_report(path: Path, *, returncode: int) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "failed",
            "error": "child produced no report.json",
            "returncode": returncode,
        }
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "status": "failed",
            "error": f"child report is unreadable: {error}",
            "returncode": returncode,
        }
    report["wrapper_observed_returncode"] = returncode
    if returncode != 0 and report.get("status") == "complete":
        report["status"] = "failed"
        report["error"] = "child returned non-zero despite a complete report"
    return report


def main() -> int:
    args = parse_args()
    assert_determinism_environment()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("The paired reproducibility wrapper must run inside one Slurm allocation.")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise SystemExit("CUDA_VISIBLE_DEVICES must identify the allocation's physical GPU.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("The paired reproducibility wrapper refuses a non-empty --output-dir.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    child_results: dict[str, dict[str, Any]] = {}
    for replica in ("a", "b"):
        replica_dir = args.output_dir / replica
        command = [
            sys.executable,
            str(PROBE),
            "--train",
            str(args.train),
            "--reader",
            str(args.reader),
            "--output-dir",
            str(replica_dir),
            "--steps",
            str(args.steps),
            "--device",
            args.device,
        ]
        stdout_path = args.output_dir / f"replica_{replica}.stdout.log"
        stderr_path = args.output_dir / f"replica_{replica}.stderr.log"
        with (
            stdout_path.open("w", encoding="utf-8", newline="\n") as stdout_handle,
            stderr_path.open("w", encoding="utf-8", newline="\n") as stderr_handle,
        ):
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=os.environ.copy(),
            )
        child_results[replica] = {
            "returncode": completed.returncode,
            "report": read_report(replica_dir / "report.json", returncode=completed.returncode),
            "stdout": str(stdout_path.resolve()),
            "stderr": str(stderr_path.resolve()),
        }

    comparison = compare_bitwise_repro_reports(
        child_results["a"]["report"],
        child_results["b"]["report"],
    )
    pair_report = {
        "schema_version": "vision_memory.lightweight_determinism_pair.v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "steps": args.steps,
        "children": child_results,
        "comparison": comparison,
        "valid": comparison["valid"]
        and child_results["a"]["returncode"] == 0
        and child_results["b"]["returncode"] == 0,
    }
    pair_path = args.output_dir / "pair_report.json"
    pair_path.write_text(
        json.dumps(pair_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(pair_report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if pair_report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
