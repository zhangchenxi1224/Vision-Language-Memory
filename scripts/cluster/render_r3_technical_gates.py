from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "probes"))

from validate_r3_technical_gates import (  # noqa: E402
    CHOICES,
    GATE_ORDER,
    OVERWRITE_EVENT,
    QUERY,
    SET_EVENT,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
RENDER_GATE_ORDER = ("R3-R0", "R3-S0", *GATE_ORDER[1:])


@dataclass(frozen=True)
class R3Paths:
    project: Path
    environment: Path
    model_root: Path
    train: Path
    dev: Path
    run_root: Path

    @property
    def results(self) -> Path:
        return self.run_root / "results"

    @property
    def logs(self) -> Path:
        return self.run_root / "logs"

    @property
    def sbatch(self) -> Path:
        return self.run_root / "sbatch"


@dataclass(frozen=True)
class R3Gate:
    name: str
    dependency: str | None
    walltime: str
    commands: tuple[str, ...]


def shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _probe_command(
    paths: R3Paths,
    *,
    events: tuple[str, ...],
    target_index: int,
    output: Path,
    detach: bool = False,
) -> str:
    parts: list[str | Path] = [
        "python",
        paths.project / "scripts" / "probes" / "e2e_episode_grad.py",
        "--dreamlite",
        paths.model_root / "DreamLite-mobile",
        "--reader",
        paths.model_root / "Qwen3-VL-4B-Instruct",
    ]
    for event in events:
        parts.extend(("--event", event))
    parts.extend(
        (
            "--query",
            QUERY,
            "--reader-loss-mode",
            "listwise-choice",
        )
    )
    for choice in CHOICES:
        parts.extend(("--choice", choice))
    parts.extend(
        (
            "--target-index",
            str(target_index),
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
        )
    )
    if detach:
        parts.append("--detach-between-events")
    parts.extend(("--output-json", output))
    return shell_join(parts)


def _technical_validation_command(paths: R3Paths, *, through: str, output: Path) -> str:
    parts: list[str | Path] = [
        "python",
        paths.project / "scripts" / "probes" / "validate_r3_technical_gates.py",
        "--through",
        through,
        "--resize-contract",
        paths.results / "R3_R0_qwen_resize_contract.json",
    ]
    if through != "R3-R0":
        parts.extend(("--g4", paths.results / "G4_L.json"))
    if through in {"G5-L", "G6-L", "DL-S"}:
        parts.extend(("--g5", paths.results / "G5_L.json"))
    if through in {"G6-L", "DL-S"}:
        parts.extend(("--g6", paths.results / "G6_L_detached.json"))
    if through == "DL-S":
        parts.extend(("--resume-report", paths.results / "DL_S_resume_equivalence.json"))
    parts.extend(
        (
            "--pair-atol",
            "1e-5",
            "--pair-rtol",
            "1e-4",
            "--output",
            output,
        )
    )
    return shell_join(parts)


def _training_command(paths: R3Paths, *, output_dir: Path, resume: Path | None = None) -> str:
    parts: list[str | Path] = [
        "python",
        paths.project / "scripts" / "train" / "dreamlite_episode.py",
        "--train",
        paths.train,
        "--dev",
        paths.dev,
        "--dataset-format",
        "synthetic",
        "--dreamlite",
        paths.model_root / "DreamLite-mobile",
        "--reader",
        paths.model_root / "Qwen3-VL-4B-Instruct",
        "--reader-loss-mode",
        "listwise-choice",
        "--choice-view-schedule",
        "cyclic4",
        "--training-regime",
        "qa_only",
        "--objective-stage",
        "qa",
        "--initial-state-mode",
        "blank",
        "--output-dir",
        output_dir,
        "--resolution",
        "1024",
        "--seed",
        "0",
        "--adapter-seed",
        "0",
        "--learning-rate",
        "1e-4",
        "--weight-decay",
        "0.01",
        "--lora-rank",
        "4",
        "--epochs",
        "2",
        "--gradient-accumulation",
        "1",
        "--gradient-clip",
        "1.0",
        "--checkpoint-every",
        "8",
        "--eval-every",
        "100000",
        "--eval-limit",
        "1",
        "--early-stopping-patience",
        "3",
        "--max-train-episodes",
        "16",
        "--max-optimizer-steps",
        "17",
        "--audit-gradient-sha",
        "--strict-determinism",
        "--require-mixed-delayed-probe",
        "--recurrence-mode",
        "direct_latent",
        "--noop-policy",
        "update",
        "--curriculum",
        "full",
        "--checkpoint-unet",
        "--dreamlite-device",
        "cuda:0",
        "--reader-device",
        "cuda:1",
    ]
    if resume is not None:
        parts.extend(("--resume", resume))
    return shell_join(parts)


def build_gates(
    paths: R3Paths,
    *,
    expected_train_sha256: str,
    expected_dev_sha256: str,
) -> list[R3Gate]:
    """Build the locked R3 sequence without performing any submission."""

    for label, digest in (
        ("expected_train_sha256", expected_train_sha256),
        ("expected_dev_sha256", expected_dev_sha256),
    ):
        if _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{label} must be a lowercase 64-character SHA256 digest.")

    g4 = paths.results / "G4_L.json"
    g5 = paths.results / "G5_L.json"
    g6 = paths.results / "G6_L_detached.json"
    reference_dir = paths.run_root / "dl_s" / "reference_16"
    resumed_dir = paths.run_root / "dl_s" / "resumed_from_8"
    prefix_checkpoint = reference_dir / "checkpoint-000008.pt"
    reference_checkpoint = reference_dir / "checkpoint-000016.pt"
    resumed_checkpoint = resumed_dir / "checkpoint-000016.pt"
    resume_report = paths.results / "DL_S_resume_equivalence.json"

    r0_commands = (
        shell_join(
            [
                "python",
                paths.project / "scripts" / "probes" / "qwen_resize_contract.py",
                "--reader",
                paths.model_root / "Qwen3-VL-4B-Instruct",
                "--device",
                "cuda:0",
                "--seed",
                "0",
                "--output-json",
                paths.results / "R3_R0_qwen_resize_contract.json",
            ]
        ),
        _technical_validation_command(
            paths,
            through="R3-R0",
            output=paths.results / "R3_R0_validation.json",
        ),
    )

    s0_commands = (
        shell_join(
            [
                "python",
                paths.project / "scripts" / "probes" / "qwen_scorer_contract.py",
                "--reader",
                paths.model_root / "Qwen3-VL-4B-Instruct",
                "--device",
                "cuda:0",
                "--output-json",
                paths.results / "R3_S0_qwen_scorer_contract.json",
            ]
        ),
    )
    g4_commands = (
        _probe_command(paths, events=(SET_EVENT,), target_index=0, output=g4),
        _technical_validation_command(
            paths,
            through="G4-L",
            output=paths.results / "G4_L_validation.json",
        ),
    )
    g5_commands = (
        _probe_command(
            paths,
            events=(SET_EVENT, OVERWRITE_EVENT),
            target_index=1,
            output=g5,
        ),
        _technical_validation_command(
            paths,
            through="G5-L",
            output=paths.results / "G5_L_validation.json",
        ),
    )
    g6_commands = (
        _probe_command(
            paths,
            events=(SET_EVENT, OVERWRITE_EVENT),
            target_index=1,
            output=g6,
            detach=True,
        ),
        _technical_validation_command(
            paths,
            through="G6-L",
            output=paths.results / "G6_L_validation.json",
        ),
    )
    dataset_contract = (
        f"test \"$(sha256sum {shlex.quote(str(paths.train))} | awk '{{print $1}}')\" = "
        f"{shlex.quote(expected_train_sha256)}",
        f"test \"$(sha256sum {shlex.quote(str(paths.dev))} | awk '{{print $1}}')\" = "
        f"{shlex.quote(expected_dev_sha256)}",
    )
    dl_commands = (
        *dataset_contract,
        _training_command(paths, output_dir=reference_dir),
        _training_command(paths, output_dir=resumed_dir, resume=prefix_checkpoint),
        shell_join(
            [
                "python",
                paths.project / "scripts" / "probes" / "validate_r3_resume_equivalence.py",
                "--prefix",
                prefix_checkpoint,
                "--reference",
                reference_checkpoint,
                "--resumed",
                resumed_checkpoint,
                "--reference-next",
                reference_dir / "last.pt",
                "--resumed-next",
                resumed_dir / "last.pt",
                "--reference-metrics",
                reference_dir / "metrics.jsonl",
                "--resumed-metrics",
                resumed_dir / "metrics.jsonl",
                "--output",
                resume_report,
            ]
        ),
        _technical_validation_command(
            paths,
            through="DL-S",
            output=paths.results / "R3_technical_gates_final.json",
        ),
    )

    return [
        R3Gate("R3-R0", None, "01:00:00", r0_commands),
        R3Gate("R3-S0", "R3-R0", "02:00:00", s0_commands),
        R3Gate("G4-L", "R3-S0", "02:00:00", g4_commands),
        R3Gate("G5-L", "G4-L", "03:00:00", g5_commands),
        R3Gate("G6-L", "G5-L", "03:00:00", g6_commands),
        R3Gate("DL-S", "G6-L", "08:00:00", dl_commands),
    ]


def _preflight(paths: R3Paths, *, expected_torch: str, output_name: str) -> str:
    return shell_join(
        [
            "python",
            paths.project / "scripts" / "bootstrap" / "preflight.py",
            "--mode",
            "cluster",
            "--model-root",
            paths.model_root,
            "--expected-torch",
            expected_torch,
            "--min-gpus",
            "2",
            "--min-gpu-memory-gib",
            "70",
            "--output",
            paths.results / output_name,
        ]
    )


def _gpu_contract() -> str:
    code = (
        "import torch; "
        "names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]; "
        "assert len(names)==2, names; "
        "assert all('A800' in name for name in names), names"
    )
    return shell_join(["python", "-c", code])


def _body_preamble(
    paths: R3Paths,
    *,
    expected_commit: str,
    expected_torch: str,
    preflight_output: str,
) -> str:
    return f"""set -euo pipefail
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
source /etc/profile.d/modules.sh
module purge
module load cuda/11.8
source {shlex.quote(str(paths.environment / "bin" / "activate"))}
cd {shlex.quote(str(paths.project))}

test \"$(git rev-parse HEAD)\" = {shlex.quote(expected_commit)}
test -z \"$(git status --porcelain --untracked-files=all)\"
{_preflight(paths, expected_torch=expected_torch, output_name=preflight_output)}
{_gpu_contract()}"""


def render_gate_sbatch(
    gate: R3Gate,
    *,
    paths: R3Paths,
    expected_commit: str,
    expected_torch: str,
) -> str:
    dependency = "NONE" if gate.dependency is None else gate.dependency
    commands = "\n".join(gate.commands)
    return f"""#!/usr/bin/env bash
# R3_DRY_RUN_TEMPLATE=1
# R3_DEPENDS_ON={dependency}
# R3_REQUIRED_DEPENDENCY_MODE=afterok
#SBATCH --job-name=r3_{gate.name.replace("-", "_")}
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time={gate.walltime}
#SBATCH --output={paths.logs}/{gate.name}_%j.out
#SBATCH --error={paths.logs}/{gate.name}_%j.err

{_body_preamble(paths, expected_commit=expected_commit, expected_torch=expected_torch, preflight_output=f"{gate.name}_preflight.json")}

{commands}
"""


def render_strict_chain_sbatch(
    gates: list[R3Gate],
    *,
    paths: R3Paths,
    expected_commit: str,
    expected_torch: str,
) -> str:
    sections = []
    for gate in gates:
        sections.append(f"# BEGIN {gate.name}\n" + "\n".join(gate.commands) + f"\n# END {gate.name}")
    commands = "\n\n".join(sections)
    return f"""#!/usr/bin/env bash
# R3_DRY_RUN_TEMPLATE=1
# R3_STRICT_SERIAL_FAIL_STOP=1
# This single allocation is the authoritative fail-stop path; no command after a failed gate can run.
#SBATCH --job-name=r3_strict_chain
#SBATCH --partition=a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output={paths.logs}/R3_strict_chain_%j.out
#SBATCH --error={paths.logs}/R3_strict_chain_%j.err

{_body_preamble(paths, expected_commit=expected_commit, expected_torch=expected_torch, preflight_output="R3_strict_chain_preflight.json")}

{commands}
"""


def _git(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def materialize_dry_run(
    *,
    paths: R3Paths,
    gates: list[R3Gate],
    expected_commit: str,
    expected_torch: str,
    expected_train_sha256: str,
    expected_dev_sha256: str,
) -> dict[str, Any]:
    for directory in (paths.run_root, paths.results, paths.logs, paths.sbatch):
        directory.mkdir(parents=True, exist_ok=directory != paths.sbatch)

    jobs: dict[str, Any] = {}
    for gate in gates:
        output = paths.sbatch / f"{gate.name}.sbatch"
        output.write_text(
            render_gate_sbatch(
                gate,
                paths=paths,
                expected_commit=expected_commit,
                expected_torch=expected_torch,
            ),
            encoding="utf-8",
            newline="\n",
        )
        jobs[gate.name] = {
            "job_id": None,
            "status": "template_only",
            "depends_on_gate": gate.dependency,
            "required_dependency_mode": None if gate.dependency is None else "afterok",
            "nodes": 1,
            "gpus_per_node": 2,
            "partition": "a800",
            "sbatch": str(output),
        }

    chain = paths.sbatch / "R3_strict_chain.sbatch"
    chain.write_text(
        render_strict_chain_sbatch(
            gates,
            paths=paths,
            expected_commit=expected_commit,
            expected_torch=expected_torch,
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": 2,
        "protocol": "R3-technical-listwise-resize-v2",
        "dry_run": True,
        "submission_supported": False,
        "status": "dry_run_complete",
        "project": str(paths.project),
        "run_root": str(paths.run_root),
        "commit": expected_commit,
        "expected_torch": expected_torch,
        "dataset": {
            "train": str(paths.train),
            "train_sha256": expected_train_sha256,
            "dev": str(paths.dev),
            "dev_sha256": expected_dev_sha256,
        },
        "strict_order": [gate.name for gate in gates],
        "failure_policy": "set -euo pipefail; recommended strict-chain template stops at first failed gate",
        "recommended_template": str(chain),
        "jobs": jobs,
    }
    _atomic_json(paths.run_root / "dry_run_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render R3-R0/R3-S0/G4-L/G5-L/G6-L/DL-S sbatch templates; "
            "this command can never submit jobs"
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_runs/vision-language-memory"),
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_envs/vision_memory_py310_cu118_torch271"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_models/vision-language-memory"),
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-dev-sha256", required=True)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("/remote-home1/cxzhang/codex_runs/vision-language-memory-runs"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-torch", default="2.7.1+cu118")
    parser.add_argument("--through", choices=RENDER_GATE_ORDER, default="DL-S")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = args.project_root.resolve(strict=True)
    commit = _git("rev-parse", "HEAD", cwd=project)
    if args.expected_commit is not None and commit != args.expected_commit:
        raise SystemExit(f"Commit mismatch: expected {args.expected_commit}, found {commit}.")
    if _git("status", "--porcelain", "--untracked-files=all", cwd=project):
        raise SystemExit("R3 dry-run rendering refuses a dirty project checkout.")

    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    run_name = args.run_name or f"r3-technical-{stamp}-{commit[:8]}"
    paths = R3Paths(
        project=project,
        environment=args.environment,
        model_root=args.model_root,
        train=args.train,
        dev=args.dev,
        run_root=args.runs_root / run_name,
    )
    gates = build_gates(
        paths,
        expected_train_sha256=args.expected_train_sha256,
        expected_dev_sha256=args.expected_dev_sha256,
    )
    gates = gates[: RENDER_GATE_ORDER.index(args.through) + 1]
    manifest = materialize_dry_run(
        paths=paths,
        gates=gates,
        expected_commit=commit,
        expected_torch=args.expected_torch,
        expected_train_sha256=args.expected_train_sha256,
        expected_dev_sha256=args.expected_dev_sha256,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
