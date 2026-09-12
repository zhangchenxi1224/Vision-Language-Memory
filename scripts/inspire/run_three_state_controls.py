"""Run two fixed frozen controls only if the registered endpoint fails its gate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
PROBE_COMMIT = "1e4acd2bcd9cbf9f0b2878e51694c17bebe06110"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent-run", type=Path, required=True)
    p.add_argument("--probe-source", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--deadline-unix", type=float, required=True)
    a = p.parse_args()
    if os.name != "posix":
        raise RuntimeError("This supervisor runs on the Linux training instance")
    from scripts.train import train_latent_bank_unet as train
    from scripts.probes.official_three_state_confirmation import BANK_SHA, PARENT_COMMIT, validate_development_rows
    from vision_memory.training.latent_bank_unet import file_sha256, load_teacher_bank
    status_path = Path(str(a.output_prefix) + "-queue.json")
    if status_path.exists():
        raise RuntimeError("Do not duplicate the existing control queue")
    status = {"parent": str(a.parent_run), "probe_commit": PROBE_COMMIT, "state": "waiting", "arms": []}
    train.write_json(status_path, status)
    while not (a.parent_run / "terminal.json").exists():
        if time.time() >= a.deadline_unix - 900:
            status.update(state="parent_wait_deadline")
            train.write_json(status_path, status)
            return 2
        time.sleep(20)
    terminal = json.loads((a.parent_run / "terminal.json").read_text())
    if terminal.get("state") != "completed":
        status.update(state="parent_not_completed", terminal=terminal)
        train.write_json(status_path, status)
        return 2
    if file_sha256(a.parent_run / "train/result.json") != terminal["training_result_sha256"]:
        raise RuntimeError("Parent result changed")
    command = json.loads((a.parent_run / "commands.json").read_text())["commands"][-1]
    args = train.parser().parse_args(command[3:])
    if (args.expected_commit != PARENT_COMMIT or args.steps != 1536 or args.seed != 20260913
            or file_sha256(args.bank_manifest) != BANK_SHA):
        raise ValueError("Unexpected registered parent")
    bank, _ = load_teacher_bank(args.bank_manifest)
    phase = a.parent_run / "train/trained"
    complete = json.loads((phase / "complete.json").read_text())
    for name, sha in complete["artifact_hashes"].items():
        if file_sha256(phase / name) != sha:
            raise RuntimeError("Parent evaluation artifact changed")
    rows = [json.loads(line) for line in (phase / "generations.jsonl").read_text().splitlines()]
    try:
        validate_development_rows(rows, bank)
    except ValueError as error:
        if str(error) != "Development endpoint did not pass every state with immediate EOS":
            raise
        status.update(state="parent_functional_gate_failed", reason=str(error))
    else:
        status.update(state="not_needed_parent_passed")
        train.write_json(status_path, status)
        return 0
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a.probe_source, text=True).strip() != PROBE_COMMIT:
        raise RuntimeError("Wrong immutable control source")
    for style in ("native", "training_raw"):
        remaining = a.deadline_unix - time.time()
        if remaining < 300:
            status.update(state="control_lease_insufficient")
            train.write_json(status_path, status)
            return 2
        output = Path(str(a.output_prefix) + "-" + style)
        if output.exists():
            raise RuntimeError("Control output already exists")
        arm = {"condition_style": style, "output": str(output), "state": "running"}
        status["arms"].append(arm)
        train.write_json(status_path, status)
        cmd = [sys.executable, "-u", str(a.probe_source / "scripts/probes/official_base_guidance.py"),
               "--run", str(a.parent_run), "--output", str(output), "--condition-style", style]
        with Path(str(output) + ".log").open("x") as log:
            child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                result = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                # Only this newly created process group, never the parent run.
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                arm.update(state="deadline_terminated", exit_code=child.returncode)
                status.update(state="control_deadline")
                train.write_json(status_path, status)
                return 2
        arm.update(state="completed" if result == 0 and (output / "complete.json").exists() else "failed", exit_code=result)
        train.write_json(status_path, status)
        if arm["state"] == "failed":
            return result or 2
    status.update(state="completed_controls_require_raw_review")
    train.write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
