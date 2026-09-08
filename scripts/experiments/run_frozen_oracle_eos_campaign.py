"""A bounded, fail-closed two-pair H200 batch campaign.

The optional Open/EOS subexperiment owns GPU pair 2/3 until it exits. No DDP
averaging is used: independent oracle starts must remain independent.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vision_memory.training.frozen_oracle_geometry import canonical_hash, determinism_gate


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_summary(summary, out, mode, run):
    if not summary.get("technical_pass") or not summary.get("model_snapshot_end_verified"):
        raise RuntimeError("summary lacks verified technical/model completion")
    if summary.get("manifest_sha256") != file_hash(out / "manifest.json"):
        raise RuntimeError("worker manifest hash mismatch")
    worker_manifest = load(out / "manifest.json")
    if worker_manifest.get("run_spec") != run or worker_manifest.get("mode") != mode:
        raise RuntimeError("worker summary is for another run or mode")
    if mode == "optimize":
        if summary.get("optimizer_steps") != 256:
            raise RuntimeError("optimizer budget incomplete")
        index = load(summary["trajectory_index_path"])
        if [r["step"] for r in index] != list(range(257)):
            raise RuntimeError("completed trajectory incomplete")
        for record in index:
            for space in ("xT", "z"):
                if file_hash(record[space]["path"]) != record[space]["file_sha256"]:
                    raise RuntimeError("completed trajectory array changed")
        checkpoints = load(summary["checkpoint_index_path"])
        if [r["step"] for r in checkpoints] != [0, 1, 2, 4, 8, 16, 32, 64, 128, 192, 256]:
            raise RuntimeError("checkpoint index does not cover the eleven locked steps")
        for checkpoint in checkpoints:
            if file_hash(checkpoint["path"]) != checkpoint["file_sha256"]:
                raise RuntimeError("checkpoint changed")
            if file_hash(checkpoint["png_path"]) != checkpoint["png_sha256"]:
                raise RuntimeError("checkpoint PNG changed")


def run_campaign(args):
    config, manifest = load(args.config), load(args.manifest)
    if canonical_hash(manifest) != config["manifest_sha256"]:
        raise ValueError("prospective manifest hash mismatch")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.expected_commit and commit != args.expected_commit:
        raise ValueError("source commit mismatch")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("campaign requires a clean committed source tree")
    args.output_root.mkdir(parents=True, exist_ok=True)
    # An existing live controller is never replaced, including after a platform retry.
    import fcntl
    lock = (args.output_root / ".campaign.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {"commit": commit, "config_sha256": file_hash(args.config),
                "manifest_sha256": file_hash(args.manifest)}
    identity_path = args.output_root / "identity.json"
    if identity_path.exists() and load(identity_path) != identity:
        raise ValueError("output root belongs to a different campaign")
    atomic_json(identity_path, identity)
    deadline = time.monotonic() + args.max_hours * 3600
    error_event = threading.Event()
    live: dict[str, subprocess.Popen] = {}
    mutex = threading.Lock()

    def stop_live():
        error_event.set()
        with mutex:
            for proc in live.values():
                if proc.poll() is None:
                    proc.terminate()

    def wait_process(proc):
        while True:
            if error_event.is_set() or time.monotonic() >= deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise RuntimeError("campaign stopped or walltime exhausted")
            try:
                return proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _command_run(run, pair, mode="optimize"):
        if error_event.is_set():
            raise RuntimeError("another campaign worker failed")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("campaign walltime budget exhausted")
        run_id = run["run_id"]
        out = args.output_root / ("probes" if mode == "probe" else "runs") / run_id
        sentinel = out / "campaign_terminal.json"
        if sentinel.exists():
            terminal = load(sentinel)
            summary_path = out / "summary.json"
            if terminal.get("returncode") == 0 and summary_path.is_file():
                if terminal.get("summary_sha256") != file_hash(summary_path):
                    raise RuntimeError(f"completed summary changed: {run_id}")
                summary = load(summary_path)
                validate_summary(summary, out, mode, run)
                if mode == "probe" or summary.get("technical_pass"):
                    print(f"REUSE verified completed {run_id}", flush=True)
                    return summary
            raise RuntimeError(f"previous failed/incomplete terminal requires explicit audit: {run_id}")
        if out.exists() and any(out.iterdir()):
            raise RuntimeError(f"incomplete run is preserved and not overwritten: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        # Logs live outside the worker's fresh artifact root.
        logpath = args.output_root / "logs" / f"{run_id}-{mode}.log"
        logpath.parent.mkdir(exist_ok=True)
        cmd = [sys.executable, str(ROOT / "scripts/experiments/run_frozen_oracle_eos_worker.py"),
               "--config", str(args.config), "--run-spec", json.dumps(run), "--mode", mode,
               "--train", str(args.train), "--dev", str(args.dev),
               "--dreamlite", str(args.dreamlite), "--reader", str(args.reader),
               "--output-dir", str(out), "--dreamlite-device", f"cuda:{pair[0]}",
               "--reader-device", f"cuda:{pair[1]}"]
        if mode == "evaluate":
            cmd += ["--evaluation-spec", str(run["evaluation_spec_path"])]
        started = time.time()
        print(f"START {run_id} mode={mode} pair={pair}", flush=True)
        with logpath.open("w") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            with mutex:
                live[run_id] = proc
            try:
                code = wait_process(proc)
            except BaseException:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise
            finally:
                with mutex:
                    live.pop(run_id, None)
        summary_path = out / "summary.json"
        terminal = {"returncode": code, "elapsed_seconds": time.time() - started,
                    "summary_sha256": file_hash(summary_path) if summary_path.exists() else None,
                    "command": cmd, **identity}
        atomic_json(sentinel, terminal)
        if code or not summary_path.exists():
            raise RuntimeError(f"worker failed {run_id}: exit={code}; log={logpath}")
        summary = load(summary_path)
        validate_summary(summary, out, mode, run)
        print(f"DONE {run_id} QA={summary.get('qa_pass')} seconds={terminal['elapsed_seconds']:.1f}", flush=True)
        return summary

    def command_run(run, pair, mode="optimize"):
        try:
            return _command_run(run, pair, mode)
        except BaseException:
            stop_live()
            raise

    def parallel_runs(runs, *, auxiliary=False):
        pending = queue.Queue()
        for run in runs:
            pending.put(run)
        results = []

        def lane(index):
            pair = ((0, 1), (2, 3))[index]
            if auxiliary and index == 1 and args.aux_command_json:
                aux = load(args.aux_command_json)
                cmd = aux["command"]
                observed_aux_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                               cwd=aux["cwd"], text=True).strip()
                if observed_aux_commit != aux["expected_commit"]:
                    raise RuntimeError("Open/EOS source commit mismatch")
                env = os.environ.copy()
                env.update(aux.get("env", {}))
                print("START Open/EOS auxiliary on pair 2,3", flush=True)
                with (args.output_root / "open_eos_auxiliary.log").open("w") as log:
                    proc = subprocess.Popen(cmd, cwd=aux.get("cwd", str(ROOT)), env=env,
                                            stdout=log, stderr=subprocess.STDOUT)
                    with mutex:
                        live["open-eos-auxiliary"] = proc
                    try:
                        returncode = wait_process(proc)
                    finally:
                        with mutex:
                            live.pop("open-eos-auxiliary", None)
                atomic_json(args.output_root / "open_eos_auxiliary_terminal.json",
                            {"returncode": returncode, "command": cmd})
                if returncode:
                    # Its distinct scientific branch cannot invalidate geometry data.
                    print("Open/EOS auxiliary failed; see its log. Pair released to geometry.", flush=True)
            while not error_event.is_set():
                try:
                    run = pending.get_nowait()
                except queue.Empty:
                    break
                try:
                    row = command_run(run, pair)
                    with mutex:
                        results.append(row)
                except BaseException:
                    stop_live()
                    raise
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(lane, i) for i in range(2)]
            for future in futures:
                future.result()
        return results

    try:
        probe = dict(manifest["runs"][0], run_id="probe-frozen-full-path")
        command_run(probe, (0, 1), "probe")
        atomic_json(args.output_root / "status.json", {"stage": "A1", "state": "running", **identity})
        a1 = [r for r in manifest["runs"] if r["stage"] == "A1"]
        # Group exact replicas by target/device pair for the primary A1 test.
        def reproducibility_lane(target):
            try:
                return [command_run(r, ((0, 1), (2, 3))[target]) for r in a1 if r["target_index"] == target]
            except BaseException:
                stop_live()
                raise
        if args.aux_command_json:
            # Open/EOS gets pair 2/3 immediately; all exact A1 repeats use pair 0/1.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as aux_pool:
                aux_future = aux_pool.submit(parallel_runs, [], auxiliary=True)
                try:
                    a1_results = [command_run(r, (0, 1)) for r in a1]
                    gate = determinism_gate(a1_results)
                    atomic_json(args.output_root / "A1_determinism_gate.json", gate)
                    if not gate["passed"]:
                        raise RuntimeError(f"A1 determinism gate failed: {gate['errors']}")
                    aux_future.result()
                except BaseException:
                    stop_live()
                    raise
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                a1_results = sum(list(pool.map(reproducibility_lane, (0, 1))), [])
            gate = determinism_gate(a1_results)
            atomic_json(args.output_root / "A1_determinism_gate.json", gate)
            if not gate["passed"]:
                raise RuntimeError(f"A1 determinism gate failed: {gate['errors']}")
        atomic_json(args.output_root / "status.json", {"stage": "A2_A3_A4_A9", "state": "running", **identity})
        parallel_runs([r for r in manifest["runs"] if r["stage"] != "A1"])
        completed = []
        for run in manifest["runs"]:
            out = args.output_root / "runs" / run["run_id"]
            terminal, summary = load(out / "campaign_terminal.json"), load(out / "summary.json")
            if terminal["summary_sha256"] != file_hash(out / "summary.json"):
                raise RuntimeError("Final summary changed")
            validate_summary(summary, out, "optimize", run)
            completed.append(summary)
        atomic_json(args.output_root / "open_eos_results.json", {
            "completed": len(completed), "planned": len(manifest["runs"]),
            "qa_pass": sum(bool(r["qa_pass"]) for r in completed),
            "results": completed, "protocol": "fresh_frozen_xT_open_answer_plus_eos",
            "unet_training_started": False, "geometry_analysis_pending": True})
        atomic_json(args.output_root / "status.json", {
            "stage": "oracle_bank_complete", "state": "completed",
            "next": "analyze successful endpoint geometry before separate U-Net supervision", **identity})
    except BaseException as exc:
        error_event.set()
        with mutex:
            for proc in live.values():
                proc.terminate()
        atomic_json(args.output_root / "status.json", {
            "state": "failed", "error": str(exc), "traceback": traceback.format_exc(), **identity})
        raise
    finally:
        lock.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "train", "dev", "dreamlite", "reader", "output-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-commit")
    p.add_argument("--aux-command-json", type=Path)
    p.add_argument("--max-hours", type=float, default=72)
    run_campaign(p.parse_args())


if __name__ == "__main__":
    main()
