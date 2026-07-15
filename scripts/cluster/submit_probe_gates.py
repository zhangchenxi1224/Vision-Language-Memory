from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Gate:
    name: str
    gpus: int
    cpus: int
    memory: str
    walltime: str
    command: str


def shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def run_text(*arguments: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def write_manifest_atomic(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gate_specs(*, project: Path, model_root: Path, results: Path) -> list[Gate]:
    dreamlite = model_root / "DreamLite-mobile"
    reader = model_root / "Qwen3-VL-4B-Instruct"
    python = "python"

    def probe(script: str, *arguments: str | Path) -> str:
        return shell_join([python, project / "scripts" / "probes" / script, *arguments])

    return [
        Gate(
            "G1_vae_reader",
            1,
            12,
            "48G",
            "00:45:00",
            probe(
                "vae_reader_grad.py",
                "--dreamlite",
                dreamlite,
                "--reader",
                reader,
                "--state-resolution",
                "1024",
                "--seed",
                "0",
                "--max-out-of-range-fraction",
                "0.99",
                "--max-zero-gradient-fraction",
                "0.99",
                "--output-json",
                results / "G1_vae_reader.json",
            ),
        ),
        Gate(
            "G2_parity",
            1,
            12,
            "48G",
            "01:00:00",
            probe(
                "dreamlite_parity.py",
                "--model",
                dreamlite,
                "--event",
                "the background is a quiet blue room",
                "--resolution",
                "1024",
                "--seed",
                "0",
                "--atol",
                "1e-3",
                "--rtol",
                "1e-3",
                "--output-json",
                results / "G2_parity.json",
            ),
        ),
        Gate(
            "G3_sampler_grad",
            1,
            12,
            "48G",
            "02:00:00",
            probe(
                "dreamlite_sampler_grad.py",
                "--model",
                dreamlite,
                "--event",
                "the background is a quiet blue room",
                "--resolution",
                "1024",
                "--adapter-seed",
                "0",
                "--noise-seed",
                "0",
                "--lora-rank",
                "4",
                "--checkpoint-unet",
                "--output-json",
                results / "G3_sampler_grad.json",
            ),
        ),
        Gate(
            "G4_e2e_one",
            2,
            24,
            "96G",
            "02:00:00",
            probe(
                "e2e_episode_grad.py",
                "--dreamlite",
                dreamlite,
                "--reader",
                reader,
                "--event",
                "the preferred mug is red",
                "--query",
                "What color mug is preferred?",
                "--target",
                "red",
                "--resolution",
                "1024",
                "--adapter-seed",
                "0",
                "--noise-seed",
                "0",
                "--lora-rank",
                "4",
                "--checkpoint-unet",
                "--dreamlite-device",
                "cuda:0",
                "--reader-device",
                "cuda:1",
                "--output-json",
                results / "G4_e2e_one.json",
            ),
        ),
        Gate(
            "G5_e2e_two",
            2,
            24,
            "96G",
            "03:00:00",
            probe(
                "e2e_episode_grad.py",
                "--dreamlite",
                dreamlite,
                "--reader",
                reader,
                "--event",
                "the preferred mug is red",
                "--event",
                "the room has a wooden table",
                "--query",
                "What color mug is preferred?",
                "--target",
                "red",
                "--resolution",
                "1024",
                "--adapter-seed",
                "0",
                "--noise-seed",
                "0",
                "--lora-rank",
                "4",
                "--checkpoint-unet",
                "--dreamlite-device",
                "cuda:0",
                "--reader-device",
                "cuda:1",
                "--output-json",
                results / "G5_e2e_two.json",
            ),
        ),
        Gate(
            "G6_detach_pair",
            2,
            24,
            "96G",
            "03:00:00",
            " && ".join(
                [
                    probe(
                        "e2e_episode_grad.py",
                        "--dreamlite",
                        dreamlite,
                        "--reader",
                        reader,
                        "--event",
                        "the preferred mug is red",
                        "--event",
                        "the room has a wooden table",
                        "--query",
                        "What color mug is preferred?",
                        "--target",
                        "red",
                        "--resolution",
                        "1024",
                        "--adapter-seed",
                        "0",
                        "--noise-seed",
                        "0",
                        "--lora-rank",
                        "4",
                        "--checkpoint-unet",
                        "--dreamlite-device",
                        "cuda:0",
                        "--reader-device",
                        "cuda:1",
                        "--detach-between-events",
                        "--output-json",
                        results / "G6_e2e_detach.json",
                    ),
                    probe(
                        "validate_e2e_pair.py",
                        "--positive",
                        results / "G5_e2e_two.json",
                        "--detached",
                        results / "G6_e2e_detach.json",
                        "--atol",
                        "1e-5",
                        "--rtol",
                        "1e-4",
                        "--output-json",
                        results / "G6_pair_validation.json",
                    ),
                ]
            ),
        ),
    ]


def render_sbatch(
    gate: Gate,
    *,
    project: Path,
    environment: Path,
    model_root: Path,
    run_root: Path,
    expected_commit: str,
    expected_torch: str,
) -> str:
    preflight = shell_join(
        [
            "python",
            project / "scripts" / "bootstrap" / "preflight.py",
            "--mode",
            "cluster",
            "--model-root",
            model_root,
            "--expected-torch",
            expected_torch,
            "--min-gpus",
            str(gate.gpus),
            "--min-gpu-memory-gib",
            "40",
            "--output",
            run_root / "results" / f"{gate.name}_preflight.json",
        ]
    )
    environment_lock = ""
    if gate.name == "G1_vae_reader":
        environment_lock = shell_join(
            [
                "python",
                project / "scripts" / "bootstrap" / "freeze_environment.py",
                "--output",
                run_root / "results" / "environment-lock.json",
            ]
        )
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=vlm_{gate.name}
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:{gate.gpus}
#SBATCH --cpus-per-task={gate.cpus}
#SBATCH --mem={gate.memory}
#SBATCH --time={gate.walltime}
#SBATCH --output={run_root}/logs/{gate.name}_%j.out
#SBATCH --error={run_root}/logs/{gate.name}_%j.err

set -euo pipefail
source /etc/profile.d/modules.sh
module purge
module load cuda/11.8
source {shlex.quote(str(environment / 'bin' / 'activate'))}
cd {shlex.quote(str(project))}

test "$(git rev-parse HEAD)" = {shlex.quote(expected_commit)}
test -z "$(git status --porcelain)"
{preflight}
{environment_lock}
{gate.command}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or submit strict afterok G1-G6 probe gates")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_runs/vision-language-memory"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_models/vision-language-memory"),
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_envs/vision_memory_py310_cu118_torch271"),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_runs/vision-language-memory-runs"),
    )
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-torch", default="2.7.1+cu118")
    parser.add_argument("--through", choices=[f"G{index}" for index in range(1, 7)], default="G6")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve(strict=True)
    commit = run_text("git", "rev-parse", "HEAD", cwd=project)
    if args.expected_commit and commit != args.expected_commit:
        raise SystemExit(f"Commit mismatch: expected {args.expected_commit}, found {commit}")
    status = run_text("git", "status", "--porcelain", cwd=project)
    if status:
        raise SystemExit("Probe gates refuse a dirty project checkout.")

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_root = args.runs_root / f"probes-{stamp}-{commit[:8]}"
    scripts = run_root / "sbatch"
    logs = run_root / "logs"
    results = run_root / "results"
    for directory in (scripts, logs, results):
        directory.mkdir(parents=True, exist_ok=False if directory == scripts else True)

    gates = gate_specs(project=project, model_root=args.model_root, results=results)
    through = int(args.through[1:])
    gates = gates[:through]
    manifest = {
        "schema_version": 1,
        "dry_run": args.dry_run,
        "project": str(project),
        "commit": commit,
        "model_root": str(args.model_root),
        "environment": str(args.environment),
        "run_root": str(run_root),
        "status": "generating",
        "jobs": {},
    }
    manifest_path = run_root / "submission.json"
    write_manifest_atomic(manifest_path, manifest)
    dependency: str | None = None
    previous_gate: str | None = None
    for gate in gates:
        sbatch_path = scripts / f"{gate.name}.sbatch"
        sbatch_path.write_text(
            render_sbatch(
                gate,
                project=project,
                environment=args.environment,
                model_root=args.model_root,
                run_root=run_root,
                expected_commit=commit,
                expected_torch=args.expected_torch,
            ),
            encoding="utf-8",
        )
        job_record = {
            "job_id": None,
            "depends_on_gate": previous_gate,
            "depends_on_job_id": dependency,
            "sbatch": str(sbatch_path),
            "status": "generated",
        }
        manifest["jobs"][gate.name] = job_record
        write_manifest_atomic(manifest_path, manifest)
        if not args.dry_run:
            command = ["sbatch", "--parsable"]
            if dependency is not None:
                command.append(f"--dependency=afterok:{dependency}")
            command.append(str(sbatch_path))
            try:
                job_id = run_text(*command).split(";", 1)[0]
            except Exception as exc:
                job_record["status"] = "submission_failed"
                job_record["error"] = repr(exc)
                manifest["status"] = "partial_failure"
                write_manifest_atomic(manifest_path, manifest)
                raise
            job_record["job_id"] = job_id
            job_record["status"] = "submitted"
            dependency = job_id
            write_manifest_atomic(manifest_path, manifest)
        previous_gate = gate.name

    manifest["status"] = "dry_run_complete" if args.dry_run else "submitted"
    write_manifest_atomic(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
