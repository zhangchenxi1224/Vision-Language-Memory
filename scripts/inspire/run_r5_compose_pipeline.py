"""Resumable on-instance controller for the preregistered R5-Compose sequence."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "vision_memory.r5-compose-pipeline-state.v1"
TRAIN_SCRIPT = Path("scripts/train/dreamlite_r5_compose.py")
TOPOLOGY_SCRIPT = Path("scripts/experiments/assess_r5_topology.py")
RESUME_SCRIPT = Path("scripts/experiments/assess_r5_resume.py")
PILOT_SELECTOR = Path("scripts/experiments/select_r5_pilot.py")
REPORT_SCRIPT = Path("scripts/reporting/render_r5_compose_report.py")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


@dataclass
class Job:
    name: str
    command: list[str]
    output_dir: Path
    completion_file: Path
    timeout_seconds: float
    visible_devices: str
    process: subprocess.Popen[str] | None = None
    log_handle: Any = None
    started_at: float | None = None


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = args.repo.resolve()
        self.root = args.output_root.resolve()
        self.runs = self.root / "runs"
        self.logs = self.root / "logs"
        self.state_path = self.root / "pipeline_state.json"
        self.events_path = self.root / "pipeline_events.jsonl"
        self.runs.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self._status("startup", "running")

    def _status(self, phase: str, status: str, **extra: Any) -> None:
        event = {
            "schema": SCHEMA,
            "timestamp": time.time(),
            "phase": phase,
            "status": status,
            **extra,
        }
        self.events.append(event)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
        _atomic_json(
            self.state_path,
            {
                "schema": SCHEMA,
                "status": status,
                "phase": phase,
                "updated_at": event["timestamp"],
                "repo": str(self.repo),
                "output_root": str(self.root),
                "events": self.events,
                **extra,
            },
        )

    def environment(self, visible_devices: str) -> dict[str, str]:
        value = dict(os.environ)
        value.update(
            {
                "CUDA_VISIBLE_DEVICES": visible_devices,
                "PYTHONHASHSEED": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "VLM_DREAMLITE_SNAPSHOT_MANIFEST_SHA256": self.args.dreamlite_manifest_sha256,
                "VLM_READER_SNAPSHOT_MANIFEST_SHA256": self.args.reader_manifest_sha256,
            }
        )
        return value

    def base_train_command(
        self,
        *,
        profile: str,
        output_dir: Path,
        persistent_state: str,
        horizon: int,
        gradient_mode: str,
        selected_step_count: int,
        seed: int,
        same_device: bool,
        checkpoint_every: int,
        residual_blend: float = 1.0,
    ) -> list[str]:
        command = [
            str(self.args.python),
            str(TRAIN_SCRIPT),
            "--profile",
            profile,
            "--train",
            str(self.args.train),
            "--dev",
            str(self.args.dev),
            "--dreamlite",
            str(self.args.dreamlite),
            "--reader",
            str(self.args.reader),
            "--output-dir",
            str(output_dir),
            "--persistent-state",
            persistent_state,
            "--tbptt-horizon",
            str(horizon),
            "--gradient-mode",
            gradient_mode,
            "--selected-step-count",
            str(selected_step_count),
            "--seed",
            str(seed),
            "--adapter-seed",
            str(seed),
            "--schedule-seed",
            "0",
            "--pairing-seed",
            "0",
            "--split-seed",
            "20260730",
            "--checkpoint-every",
            str(checkpoint_every),
            "--residual-blend",
            str(residual_blend),
            "--strict-determinism",
            "--audit-state-gradients",
        ]
        command.extend(("--dreamlite-device", "cuda:0", "--reader-device", "cuda:0" if same_device else "cuda:1"))
        return command

    @staticmethod
    def _latest_checkpoint(output_dir: Path) -> Path | None:
        checkpoints = sorted((output_dir / "checkpoints").glob("step-*.pt"))
        return checkpoints[-1] if checkpoints else None

    def _prepare_job(self, job: Job) -> Job | None:
        if job.completion_file.is_file():
            self._status(job.name, "skipped_completed", completion_file=str(job.completion_file))
            return None
        if job.output_dir.exists() and any(job.output_dir.iterdir()):
            checkpoint = self._latest_checkpoint(job.output_dir)
            if checkpoint is not None and "--resume" not in job.command and job.command[1].endswith("dreamlite_r5_compose.py"):
                job.command.extend(("--resume", str(checkpoint)))
                self._status(job.name, "resume_selected", checkpoint=str(checkpoint))
            elif checkpoint is None:
                quarantine = job.output_dir.with_name(f"{job.output_dir.name}.partial-{int(time.time())}")
                job.output_dir.rename(quarantine)
                self._status(job.name, "partial_quarantined", path=str(quarantine))
        return job

    def _start(self, job: Job) -> None:
        log_path = self.logs / f"{job.name}.log"
        job.log_handle = log_path.open("a", encoding="utf-8")
        job.log_handle.write("\nCOMMAND " + shlex.join(job.command) + "\n")
        job.log_handle.flush()
        job.process = subprocess.Popen(
            job.command,
            cwd=self.repo,
            env=self.environment(job.visible_devices),
            stdout=job.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        job.started_at = time.monotonic()
        self._status(job.name, "started", pid=job.process.pid, log=str(log_path), command=job.command)

    def _stop(self, job: Job, *, reason: str) -> None:
        if job.process is not None and job.process.poll() is None:
            os.killpg(job.process.pid, signal.SIGTERM)
            try:
                job.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(job.process.pid, signal.SIGKILL)
                job.process.wait(timeout=30)
        self._status(job.name, "terminated", reason=reason)

    def run_jobs(self, jobs: Sequence[Job]) -> None:
        active: list[Job] = []
        for original in jobs:
            job = self._prepare_job(original)
            if job is None:
                continue
            self._start(job)
            active.append(job)
        while active:
            for job in tuple(active):
                assert job.process is not None and job.started_at is not None
                return_code = job.process.poll()
                if return_code is not None:
                    job.log_handle.close()
                    active.remove(job)
                    if return_code != 0 or not job.completion_file.is_file():
                        for sibling in active:
                            self._stop(sibling, reason=f"peer {job.name} failed")
                        raise RuntimeError(
                            f"R5 job {job.name} failed with code {return_code}; completion={job.completion_file.is_file()}"
                        )
                    self._status(
                        job.name,
                        "completed",
                        elapsed_seconds=time.monotonic() - job.started_at,
                        completion_file=str(job.completion_file),
                    )
                elif time.monotonic() - job.started_at > job.timeout_seconds:
                    self._stop(job, reason=f"hard timeout {job.timeout_seconds}s")
                    for sibling in active:
                        if sibling is not job:
                            self._stop(sibling, reason=f"peer {job.name} timed out")
                    raise TimeoutError(f"R5 job {job.name} exceeded its hard timeout.")
            if active:
                time.sleep(10)

    def script_job(
        self,
        *,
        name: str,
        command: list[str],
        output_dir: Path,
        completion_name: str,
        timeout_hours: float,
        visible_devices: str,
    ) -> Job:
        return Job(
            name=name,
            command=command,
            output_dir=output_dir,
            completion_file=output_dir / completion_name,
            timeout_seconds=timeout_hours * 3600,
            visible_devices=visible_devices,
        )

    def topology(self) -> dict[str, Any]:
        self._status("topology", "running")
        for profile, steps, timeout_hours in (("smoke", 2, 1.0), ("topology", 10, 1.5)):
            same_dir = self.runs / f"{profile}-same-latent-h4"
            split_dir = self.runs / f"{profile}-split-latent-h4"
            same_command = self.base_train_command(
                profile=profile,
                output_dir=same_dir,
                persistent_state="latent",
                horizon=4,
                gradient_mode="drtune_stateful",
                selected_step_count=2,
                seed=0,
                same_device=True,
                checkpoint_every=1 if steps == 2 else 5,
            )
            split_command = self.base_train_command(
                profile=profile,
                output_dir=split_dir,
                persistent_state="latent",
                horizon=4,
                gradient_mode="drtune_stateful",
                selected_step_count=2,
                seed=0,
                same_device=False,
                checkpoint_every=1 if steps == 2 else 5,
            )
            self.run_jobs(
                [
                    self.script_job(
                        name=f"{profile}-same",
                        command=same_command,
                        output_dir=same_dir,
                        completion_name="summary.json",
                        timeout_hours=timeout_hours,
                        visible_devices="0",
                    )
                ]
            )
            if profile == "smoke":
                resume_dir = self.runs / "smoke-resume-same-latent-h4"
                resume_command = self.base_train_command(
                    profile="smoke",
                    output_dir=resume_dir,
                    persistent_state="latent",
                    horizon=4,
                    gradient_mode="drtune_stateful",
                    selected_step_count=2,
                    seed=0,
                    same_device=True,
                    checkpoint_every=1,
                )
                resume_command.extend(("--resume", str(same_dir / "checkpoints" / "step-000001.pt")))
                self.run_jobs(
                    [
                        self.script_job(
                            name="smoke-resume-same",
                            command=resume_command,
                            output_dir=resume_dir,
                            completion_name="summary.json",
                            timeout_hours=1.0,
                            visible_devices="0",
                        )
                    ]
                )
                subprocess.run(
                    [
                        str(self.args.python),
                        str(RESUME_SCRIPT),
                        "--direct-dir",
                        str(same_dir),
                        "--resumed-dir",
                        str(resume_dir),
                        "--output",
                        str(self.root / "checkpoint_resume_audit.json"),
                    ],
                    cwd=self.repo,
                    check=True,
                    env=self.environment("0"),
                )
            self.run_jobs(
                [
                    self.script_job(
                        name=f"{profile}-split",
                        command=split_command,
                        output_dir=split_dir,
                        completion_name="summary.json",
                        timeout_hours=timeout_hours,
                        visible_devices="0,1",
                    )
                ]
            )
            decision_path = self.root / f"{profile}_topology_decision.json"
            subprocess.run(
                [
                    str(self.args.python),
                    str(TOPOLOGY_SCRIPT),
                    "--same-dir",
                    str(same_dir),
                    "--split-dir",
                    str(split_dir),
                    "--output",
                    str(decision_path),
                ],
                cwd=self.repo,
                check=True,
                env=self.environment("0,1"),
            )
        decision = _load_json(self.root / "topology_topology_decision.json")
        smoke = _load_json(self.root / "smoke_topology_decision.json")
        resume = _load_json(self.root / "checkpoint_resume_audit.json")
        decision["checkpoint_resume_audit"] = resume
        decision["checks"]["checkpoint_resume_exact"] = bool(resume["passed"])
        if not smoke["passed"] or not resume["passed"]:
            decision["passed"] = False
            decision["decision"] = "dual_h200_serial_latent_only"
            decision["smoke_parity_override"] = smoke
        _atomic_json(self.root / "topology_topology_decision.json", decision)
        self._status("topology", "completed", decision=decision["decision"])
        return decision

    def gradient_audit(self) -> dict[str, Any]:
        output = self.runs / "gradient-audit-latent-h4"
        command = self.base_train_command(
            profile="gradient-audit",
            output_dir=output,
            persistent_state="latent",
            horizon=4,
            gradient_mode="drtune_stateful",
            selected_step_count=2,
            seed=0,
            same_device=False,
            checkpoint_every=64,
        )
        self.run_jobs(
            [
                self.script_job(
                    name="gradient-audit",
                    command=command,
                    output_dir=output,
                    completion_name="gradient_fidelity.json",
                    timeout_hours=2.0,
                    visible_devices="0,1",
                )
            ]
        )
        report = _load_json(output / "gradient_fidelity.json")
        self._status("gradient-audit", "completed", selection=report["selection"])
        return report

    def _pilot_job(
        self,
        *,
        name: str,
        persistent_state: str,
        horizon: int,
        gradient_mode: str,
        selected_count: int,
        visible_devices: str,
        same_device: bool,
    ) -> Job:
        output = self.runs / f"pilot-{name}"
        command = self.base_train_command(
            profile="pilot",
            output_dir=output,
            persistent_state=persistent_state,
            horizon=horizon,
            gradient_mode=gradient_mode,
            selected_step_count=selected_count,
            seed=0,
            same_device=same_device,
            checkpoint_every=64,
        )
        return self.script_job(
            name=f"pilot-{name}",
            command=command,
            output_dir=output,
            completion_name="summary.json",
            timeout_hours=2.5,
            visible_devices=visible_devices,
        )

    def pilots(self, topology: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
        selection = audit["selection"]
        gradient_mode = str(selection["gradient_mode"])
        selected_count = int(selection["selected_step_count"])
        pilot_dirs: list[Path] = []
        if gradient_mode == "full":
            job = self._pilot_job(
                name="latent-h2-full",
                persistent_state="latent",
                horizon=2,
                gradient_mode="full",
                selected_count=0,
                visible_devices="0,1",
                same_device=False,
            )
            self.run_jobs([job])
            pilot_dirs.append(job.output_dir)
        elif topology["decision"] == "single_h200_parallel_arms":
            first = [
                self._pilot_job(
                    name="rgb-h2",
                    persistent_state="float_rgb",
                    horizon=2,
                    gradient_mode=gradient_mode,
                    selected_count=selected_count,
                    visible_devices="0",
                    same_device=True,
                ),
                self._pilot_job(
                    name="latent-h2",
                    persistent_state="latent",
                    horizon=2,
                    gradient_mode=gradient_mode,
                    selected_count=selected_count,
                    visible_devices="1",
                    same_device=True,
                ),
            ]
            second = [
                self._pilot_job(
                    name="rgb-h4",
                    persistent_state="float_rgb",
                    horizon=4,
                    gradient_mode=gradient_mode,
                    selected_count=selected_count,
                    visible_devices="0",
                    same_device=True,
                ),
                self._pilot_job(
                    name="latent-h4",
                    persistent_state="latent",
                    horizon=4,
                    gradient_mode=gradient_mode,
                    selected_count=selected_count,
                    visible_devices="1",
                    same_device=True,
                ),
            ]
            self.run_jobs(first)
            self.run_jobs(second)
            pilot_dirs.extend(job.output_dir for job in first + second)
        else:
            for horizon in (2, 4):
                job = self._pilot_job(
                    name=f"latent-h{horizon}",
                    persistent_state="latent",
                    horizon=horizon,
                    gradient_mode=gradient_mode,
                    selected_count=selected_count,
                    visible_devices="0,1",
                    same_device=False,
                )
                self.run_jobs([job])
                pilot_dirs.append(job.output_dir)
        selection_path = self.root / "pilot_selection.json"
        command = [str(self.args.python), str(PILOT_SELECTOR)]
        for directory in pilot_dirs:
            command.extend(("--pilot-dir", str(directory)))
        command.extend(("--output", str(selection_path)))
        subprocess.run(command, cwd=self.repo, check=True, env=self.environment("0,1"))
        result = _load_json(selection_path)
        self._status("pilots", "completed", decision=result["decision"])
        return result

    def rescue(self, *, selected_count: int, reason: str) -> dict[str, Any]:
        output = self.runs / f"rescue-latent-h4-k{selected_count}-tau05"
        gradient_mode = "full" if selected_count == 0 else "drtune_stateful"
        command = self.base_train_command(
            profile="rescue",
            output_dir=output,
            persistent_state="latent",
            horizon=4,
            gradient_mode=gradient_mode,
            selected_step_count=selected_count,
            seed=0,
            same_device=False,
            checkpoint_every=64,
            residual_blend=0.5,
        )
        job = self.script_job(
            name="conditional-residual-rescue",
            command=command,
            output_dir=output,
            completion_name="summary.json",
            timeout_hours=2.5,
            visible_devices="0,1",
        )
        self.run_jobs([job])
        summary = _load_json(output / "summary.json")
        _atomic_json(
            self.root / "rescue_decision.json",
            {"schema": SCHEMA, "reason": reason, "run": str(output), "summary": summary},
        )
        return summary

    def _main_job(
        self,
        *,
        seed: int,
        winner: Mapping[str, Any],
        visible_devices: str,
        same_device: bool,
    ) -> Job:
        output = self.runs / f"main-seed{seed}"
        command = self.base_train_command(
            profile="main",
            output_dir=output,
            persistent_state=str(winner["persistent_state"]),
            horizon=int(winner["tbptt_horizon"]),
            gradient_mode=str(winner["gradient_mode"]),
            selected_step_count=int(winner["selected_step_count"]),
            seed=seed,
            same_device=same_device,
            checkpoint_every=64,
        )
        return self.script_job(
            name=f"main-seed{seed}",
            command=command,
            output_dir=output,
            completion_name="summary.json",
            timeout_hours=10.0,
            visible_devices=visible_devices,
        )

    def _eval_job(
        self,
        *,
        seed: int,
        winner: Mapping[str, Any],
        visible_devices: str,
        same_device: bool,
    ) -> Job:
        main_dir = self.runs / f"main-seed{seed}"
        output = self.runs / f"final-eval-seed{seed}"
        command = self.base_train_command(
            profile="final-eval",
            output_dir=output,
            persistent_state=str(winner["persistent_state"]),
            horizon=int(winner["tbptt_horizon"]),
            gradient_mode=str(winner["gradient_mode"]),
            selected_step_count=int(winner["selected_step_count"]),
            seed=seed,
            same_device=same_device,
            checkpoint_every=64,
        )
        for checkpoint in (
            main_dir / "checkpoints" / "step-000000.pt",
            main_dir / "endpoint_raw.pt",
            main_dir / "endpoint_ema.pt",
        ):
            command.extend(("--checkpoint", str(checkpoint)))
        return self.script_job(
            name=f"final-eval-seed{seed}",
            command=command,
            output_dir=output,
            completion_name="evaluation_summary.json",
            timeout_hours=6.0,
            visible_devices=visible_devices,
        )

    @staticmethod
    def _same_device_parallel_allowed(
        topology: Mapping[str, Any], winner: Mapping[str, Any]
    ) -> bool:
        return (
            topology["decision"] == "single_h200_parallel_arms"
            and str(winner["gradient_mode"]) != "full"
        )

    def main_and_evaluate(self, topology: Mapping[str, Any], winner: Mapping[str, Any]) -> list[int]:
        same = self._same_device_parallel_allowed(topology, winner)
        if same:
            self.run_jobs(
                [
                    self._main_job(seed=0, winner=winner, visible_devices="0", same_device=True),
                    self._main_job(seed=1, winner=winner, visible_devices="1", same_device=True),
                ]
            )
            self.run_jobs(
                [
                    self._eval_job(seed=0, winner=winner, visible_devices="0", same_device=True),
                    self._eval_job(seed=1, winner=winner, visible_devices="1", same_device=True),
                ]
            )
        else:
            for seed in (0, 1):
                self.run_jobs([self._main_job(seed=seed, winner=winner, visible_devices="0,1", same_device=False)])
                self.run_jobs([self._eval_job(seed=seed, winner=winner, visible_devices="0,1", same_device=False)])
        completed = [0, 1]
        passed = [self._seed_passed(seed) for seed in completed]
        durations = [float(_load_json(self.runs / f"main-seed{seed}" / "summary.json")["training_interval_seconds"]) for seed in completed]
        if passed.count(True) == 1 or (all(passed) and max(durations) < 3 * 3600):
            seed = 2
            visible = "0" if same else "0,1"
            self.run_jobs([self._main_job(seed=seed, winner=winner, visible_devices=visible, same_device=same)])
            self.run_jobs([self._eval_job(seed=seed, winner=winner, visible_devices=visible, same_device=same)])
            completed.append(seed)
        self._status("main-and-eval", "completed", seeds=completed, passed=[self._seed_passed(seed) for seed in completed])
        return completed

    def _seed_passed(self, seed: int) -> bool:
        value = _load_json(self.runs / f"final-eval-seed{seed}" / "evaluation_summary.json")
        comparisons = value.get("endpoint_comparisons", {})
        if not all(
            float(comparisons.get(suite, {}).get("primary_minus_m0_ce", float("inf"))) < 0
            for suite in ("formal_final_128", "mechanism_final_128")
        ):
            return False
        key = "ema_step640|mechanism_final_128|normal_vs_reset"
        bootstrap = value.get("paired_bootstrap", {}).get(key)
        return isinstance(bootstrap, Mapping) and float(bootstrap["estimate"]) < 0

    def render_report(self) -> None:
        command = [
            str(self.args.report_python),
            str(REPORT_SCRIPT),
            "--experiment-root",
            str(self.root),
            "--output-dir",
            str(self.root / "delivery"),
        ]
        subprocess.run(command, cwd=self.repo, check=True, env=self.environment("0,1"))
        if not (self.root / "delivery" / "FINAL_REPORT.md").is_file():
            raise RuntimeError("R5 report renderer did not produce FINAL_REPORT.md.")

    def run(self) -> None:
        try:
            topology = self.topology()
            audit = self.gradient_audit()
            pilots = self.pilots(topology, audit)
            selected_count = int(audit["selection"]["selected_step_count"])
            if pilots["decision"] != "winner_selected":
                self.rescue(selected_count=selected_count, reason="no pilot passed both technical and mechanism gates")
                self.render_report()
                self._status("complete", "completed_negative_with_rescue")
                return
            winner = pilots["winner"]
            seeds = self.main_and_evaluate(topology, winner)
            if not any(self._seed_passed(seed) for seed in seeds):
                self.rescue(selected_count=selected_count, reason="all completed main seeds failed endpoint/causal gates")
            self.render_report()
            self._status("complete", "completed", seeds=seeds, passed=[self._seed_passed(seed) for seed in seeds])
        except BaseException as exc:
            self._status("failed", "failed", error=repr(exc))
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--report-python", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dreamlite", type=Path, required=True)
    parser.add_argument("--reader", type=Path, required=True)
    parser.add_argument("--dreamlite-manifest-sha256", required=True)
    parser.add_argument("--reader-manifest-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Controller(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
