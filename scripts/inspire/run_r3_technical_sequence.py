from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_r3_dag import (  # noqa: E402
    _load_verified_plan,
    _verify_completed_stage,
    authorize_stage,
    initialize_technical_dag,
)
from poll_stage import stage_status  # noqa: E402
from r3_dag_contract import (  # noqa: E402
    LAUNCH_COMMAND_PROTOCOL,
    require_absolute_executable,
    verify_sha_sidecar,
)


SCHEMA = "vision_memory.r3-poststart-technical-sequence.v1"
PLATFORM_STATUS_PROTOCOL = "vision-memory-inspire-platform-status.v1"
JOB_STATUS_PROTOCOL = "vision-memory-inspire-job-status.v1"
STAGES = ("R3-R0", "R3-S0", "G4-L", "G5-L", "G6-L", "DL-S")
DEFAULT_STAGE_TIMEOUTS = {
    "R3-R0": 60 * 60,
    "R3-S0": 60 * 60,
    "G4-L": 2 * 60 * 60,
    "G5-L": 3 * 60 * 60,
    "G6-L": 3 * 60 * 60,
    "DL-S": 6 * 60 * 60,
}
DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
}


class StageOrchestrationTimeout(TimeoutError):
    def __init__(self, stage: str, run_dir: Path, snapshot: Mapping[str, Any]) -> None:
        self.stage = stage
        self.run_dir = run_dir
        self.snapshot = dict(snapshot)
        self.worker_may_still_be_running = self.snapshot.get("status") == "running"
        super().__init__(f"{stage} did not write terminal.json before its orchestration timeout")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def append_event(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verify_platform_status(args: argparse.Namespace) -> dict[str, Any]:
    status_path = args.platform_status.resolve()
    actual_sha256 = verify_sha_sidecar(status_path, expected_sha256=args.platform_status_sha256)
    receipt = load_json(status_path)
    protocol = receipt.get("protocol")
    expected_protocol = JOB_STATUS_PROTOCOL if args.expected_workload_kind == "job" else PLATFORM_STATUS_PROTOCOL
    expected = {
        "schema_version": 1,
        "protocol": expected_protocol,
        "workload_kind": args.expected_workload_kind,
        "instance": args.expected_instance,
        "status": "RUNNING",
        "node": args.expected_node,
        "node_status": "READY",
        "image": args.expected_image,
        "workspace": args.expected_workspace,
        "project": args.expected_project,
        "project_priority": "10",
        "gpu_product": "H200",
        "gpu_count": 2,
        "cpu_count": 40,
        "memory_gib": 400,
    }
    if args.expected_workload_kind == "notebook":
        expected.update({"image_source": "SOURCE_OFFICIAL", "shared_memory_gib": 128, "auto_stop": False})
    else:
        expected.update({"node_count": 1, "compute_group": args.expected_compute_group})
    for key, expected_value in expected.items():
        if receipt.get(key) != expected_value:
            raise ValueError(f"Platform status receipt requires {key}={expected_value!r}, got {receipt.get(key)!r}")
    captured_at = receipt.get("captured_at")
    if not isinstance(captured_at, str):
        raise ValueError("Platform status receipt has no captured_at timestamp")
    try:
        captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Platform status receipt captured_at is malformed") from exc
    if captured.tzinfo is None:
        raise ValueError("Platform status receipt captured_at must include a timezone")
    age_seconds = (datetime.now(UTC) - captured.astimezone(UTC)).total_seconds()
    if age_seconds < -60 or age_seconds > args.max_platform_status_age_seconds:
        raise ValueError(
            "Platform status receipt is outside the allowed freshness window: "
            f"age={age_seconds:.1f}s, maximum={args.max_platform_status_age_seconds}s"
        )
    return {
        "path": str(status_path),
        "sha256": actual_sha256,
        "age_seconds": age_seconds,
        "protocol": protocol,
        "workload_kind": args.expected_workload_kind,
    }


def verify_launch_binding(
    *,
    run_root: Path,
    plan: Mapping[str, Any],
    stage: str,
    materialized: Mapping[str, Any],
) -> list[str]:
    definition = plan["stages"][stage]
    expected_path = run_root / "launch_commands" / f"{definition['slug']}.json"
    declared_path = materialized.get("launch_path")
    declared_sha256 = materialized.get("launch_sha256")
    in_memory = materialized.get("launch")
    if declared_path != str(expected_path):
        raise ValueError(f"{stage} materializer returned the wrong launch path")
    if not isinstance(declared_sha256, str):
        raise ValueError(f"{stage} materializer omitted its launch SHA256")
    verify_sha_sidecar(expected_path, expected_sha256=declared_sha256)
    on_disk = load_json(expected_path)
    if not isinstance(in_memory, Mapping) or on_disk != dict(in_memory):
        raise ValueError(f"{stage} launch command drifted from the materializer return value")
    expected_values = {
        "schema_version": 1,
        "protocol": LAUNCH_COMMAND_PROTOCOL,
        "stage": stage,
        "run_dir": definition["run_dir"],
        "executed": False,
    }
    for key, expected_value in expected_values.items():
        if on_disk.get(key) != expected_value:
            raise ValueError(f"{stage} launch command requires {key}={expected_value!r}")
    argv = on_disk.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(value, str) for value in argv):
        raise ValueError(f"{stage} launch command is malformed")
    return list(argv)


def write_duplicate_attempt(
    *,
    control_parent: Path,
    control_dir: Path,
    run_root: Path,
    preflight: Path,
    args: argparse.Namespace,
) -> Path:
    attempts = control_parent / "_attempts" / args.run_name
    attempts.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S.%fZ')}-{os.getpid()}-{uuid.uuid4().hex}.json"
    path = attempts / name
    payload = {
        "schema": SCHEMA,
        "state": "duplicate_control_refused",
        "at": utc_now(),
        "pid": os.getpid(),
        "control_dir": str(control_dir),
        "run_root": str(run_root),
        "preflight": str(preflight),
        "expected_commit": args.expected_commit,
        "existing_control_terminal": (control_dir / "terminal.json").is_file(),
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def git(repo: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def require_clean_commit(repo: Path, expected_commit: str) -> None:
    if git(repo, "rev-parse", "HEAD") != expected_commit:
        raise ValueError("Repository commit differs from the post-start sequence contract")
    if git(repo, "status", "--porcelain=v1", "--untracked-files=all") != "":
        raise ValueError("Post-start technical sequence requires a clean repository")


def command_record(command: list[str]) -> list[str]:
    forbidden = ("token", "password", "secret", "credential", "api-key", "api_key")
    if any(any(word in argument.casefold() for word in forbidden) for argument in command):
        raise ValueError("Refusing to record a command that may contain credentials")
    return command


def run_checked(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    events_path: Path,
    label: str,
) -> None:
    command_record(command)
    append_event(events_path, {"event": "command_started", "label": label, "at": utc_now(), "argv": command})
    result = subprocess.run(
        command,
        cwd=cwd,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    append_event(
        events_path,
        {
            "event": "command_finished",
            "label": label,
            "at": utc_now(),
            "exit_code": result.returncode,
            "stdout": result.stdout[-12000:],
            "stderr": result.stderr[-12000:],
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def progress_payload(
    *,
    run_name: str,
    run_root: Path,
    preflight: Path,
    expected_commit: str,
    state: str,
    current_stage: str | None,
    completed_stages: list[str],
    detail: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "updated_at": utc_now(),
        "state": state,
        "current_stage": current_stage,
        "completed_stages": list(completed_stages),
        "detail": detail,
        "run_name": run_name,
        "run_root": str(run_root),
        "preflight": str(preflight),
        "expected_commit": expected_commit,
        "technical_order": list(STAGES),
        "micro_or_pilot_authorized": False,
    }


def wait_for_stage(
    *,
    plan: Mapping[str, Any],
    stage: str,
    timeout_seconds: int,
    poll_seconds: float,
    progress_path: Path,
    progress_base: dict[str, Any],
) -> dict[str, Any]:
    run_dir = Path(plan["stages"][stage]["run_dir"])
    terminal_path = run_dir / "terminal.json"
    deadline = time.monotonic() + timeout_seconds
    next_heartbeat = 0.0
    while time.monotonic() < deadline:
        if terminal_path.is_file():
            terminal = load_json(terminal_path)
            if terminal.get("passed") is not True or terminal.get("exit_code") != 0:
                raise RuntimeError(f"{stage} terminal is not passing: {terminal}")
            return terminal
        now = time.monotonic()
        if now >= next_heartbeat:
            atomic_json(
                progress_path,
                {
                    **progress_base,
                    "updated_at": utc_now(),
                    "state": "waiting_for_stage_terminal",
                    "current_stage": stage,
                    "stage_run_dir": str(run_dir),
                    "stage_timeout_seconds": timeout_seconds,
                },
            )
            next_heartbeat = now + 60.0
        time.sleep(poll_seconds)
    if terminal_path.is_file():
        terminal = load_json(terminal_path)
        if terminal.get("passed") is True and terminal.get("exit_code") == 0:
            return terminal
        raise RuntimeError(f"{stage} terminal is not passing: {terminal}")
    snapshot, _ = stage_status(run_dir)
    raise StageOrchestrationTimeout(stage, run_dir, snapshot)


def build_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(DETERMINISTIC_ENVIRONMENT)
    environment.update(
        {
            "HF_HOME": str(args.hf_home),
            "TORCH_HOME": str(args.torch_home),
            "VLM_INSPIRE_IMAGE": args.expected_image,
            "VLM_INSPIRE_INSTANCE": args.expected_instance,
            "VLM_INSPIRE_NODE": args.expected_node,
            "VLM_INSPIRE_PROJECT": args.expected_project,
            "VLM_INSPIRE_WORKSPACE": args.expected_workspace,
            "VLM_MODEL_ROOT": str(args.model_root),
            "VLM_RUN_ROOT": str(args.runs_root),
        }
    )
    return environment


def run_sequence(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    python = require_absolute_executable(args.python, "--python")
    model_root = args.model_root.resolve()
    runs_root = args.runs_root.resolve()
    run_root = runs_root / args.run_name
    preflight = args.preflight.resolve()
    control_parent = runs_root / "control"
    control_dir = control_parent / args.run_name
    progress_path = control_dir / "progress.json"
    events_path = control_dir / "events.jsonl"
    terminal_path = control_dir / "terminal.json"
    completed_stages: list[str] = []
    started_at = utc_now()

    control_parent.mkdir(parents=True, exist_ok=True)
    try:
        control_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        write_duplicate_attempt(
            control_parent=control_parent,
            control_dir=control_dir,
            run_root=run_root,
            preflight=preflight,
            args=args,
        )
        return 2
    environment = build_environment(args)
    manifest = {
        "schema": SCHEMA,
        "started_at": started_at,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "repo": str(repo),
        "python": str(python),
        "model_root": str(model_root),
        "runs_root": str(runs_root),
        "run_root": str(run_root),
        "run_name": args.run_name,
        "preflight": str(preflight),
        "expected_commit": args.expected_commit,
        "train": str(args.train.resolve()),
        "train_sha256": args.train_sha256,
        "dev": str(args.dev.resolve()),
        "dev_sha256": args.dev_sha256,
        "expected_instance": args.expected_instance,
        "expected_node": args.expected_node,
        "expected_image": args.expected_image,
        "expected_workspace": args.expected_workspace,
        "expected_project": args.expected_project,
        "platform_status": {
            "path": str(args.platform_status.resolve()),
            "sha256": args.platform_status_sha256,
            "max_age_seconds": args.max_platform_status_age_seconds,
        },
        "technical_order": list(STAGES),
        "stage_timeouts": {stage: DEFAULT_STAGE_TIMEOUTS[stage] for stage in STAGES},
        "micro_or_pilot_authorized": False,
    }
    atomic_json(control_dir / "manifest.json", manifest)
    atomic_json(
        progress_path,
        progress_payload(
            run_name=args.run_name,
            run_root=run_root,
            preflight=preflight,
            expected_commit=args.expected_commit,
            state="starting",
            current_stage=None,
            completed_stages=completed_stages,
        ),
    )

    try:
        if run_root.exists() or preflight.exists() or preflight.with_suffix(preflight.suffix + ".sha256").exists():
            raise ValueError("Post-start sequence refuses any pre-existing run or preflight output")
        platform_binding = verify_platform_status(args)
        append_event(events_path, {"event": "platform_status_verified", "at": utc_now(), **platform_binding})
        require_clean_commit(repo, args.expected_commit)
        preflight_command = [
            str(python),
            str(repo / "scripts" / "inspire" / "preflight_r3_h200.py"),
            "--repo",
            str(repo),
            "--model-root",
            str(model_root),
            "--expected-commit",
            args.expected_commit,
            "--expected-instance",
            args.expected_instance,
            "--expected-image",
            args.expected_image,
            "--expected-node",
            args.expected_node,
            "--expected-workspace",
            args.expected_workspace,
            "--expected-project",
            args.expected_project,
            "--expected-driver",
            args.expected_driver,
            "--require-models",
            "--output",
            str(preflight),
        ]
        run_checked(
            preflight_command,
            cwd=repo,
            environment=environment,
            events_path=events_path,
            label="formal_preflight",
        )
        preflight_report = load_json(preflight)
        if preflight_report.get("passed") is not True or preflight_report.get("formal_ready") is not True:
            raise RuntimeError("Formal H200 preflight did not pass")

        initialization = initialize_technical_dag(
            repo=repo,
            python=python,
            model_root=model_root,
            train=args.train.resolve(),
            train_sha256=args.train_sha256,
            dev=args.dev.resolve(),
            dev_sha256=args.dev_sha256,
            run_root=run_root,
            preflight=preflight,
            expected_commit=args.expected_commit,
            through="DL-S",
            dry_run=False,
        )
        locked_plan_sha256 = str(initialization["plan_sha256"])
        pending_launch = initialization["first_stage"]
        append_event(events_path, {"event": "technical_dag_initialized", "at": utc_now(), "run_root": str(run_root)})

        for index, stage in enumerate(STAGES):
            plan, _plan_path, plan_sha256 = _load_verified_plan(run_root)
            if plan_sha256 != locked_plan_sha256:
                raise ValueError("Technical DAG plan SHA256 drifted after initialization")
            argv = verify_launch_binding(
                run_root=run_root,
                plan=plan,
                stage=stage,
                materialized=pending_launch,
            )
            base = progress_payload(
                run_name=args.run_name,
                run_root=run_root,
                preflight=preflight,
                expected_commit=args.expected_commit,
                state="launching_stage",
                current_stage=stage,
                completed_stages=completed_stages,
            )
            atomic_json(progress_path, base)
            run_checked(
                list(argv),
                cwd=repo,
                environment=environment,
                events_path=events_path,
                label=f"launch:{stage}",
            )
            wait_for_stage(
                plan=plan,
                stage=stage,
                timeout_seconds=DEFAULT_STAGE_TIMEOUTS[stage],
                poll_seconds=args.poll_seconds,
                progress_path=progress_path,
                progress_base=base,
            )
            verified_bindings = _verify_completed_stage(plan, stage)
            completed_stages.append(stage)
            append_event(
                events_path,
                {
                    "event": "stage_verified",
                    "at": utc_now(),
                    "stage": stage,
                    "bindings": verified_bindings,
                },
            )
            if index + 1 < len(STAGES):
                pending_launch = authorize_stage(run_root, stage=STAGES[index + 1], dry_run=False)

        final_plan, _final_plan_path, final_plan_sha256 = _load_verified_plan(run_root)
        if final_plan_sha256 != locked_plan_sha256:
            raise ValueError("Technical DAG plan SHA256 drifted before final sequence closure")
        require_clean_commit(repo, args.expected_commit)
        verify_sha_sidecar(preflight, expected_sha256=final_plan["formal_preflight"]["sha256"])
        final_stage_bindings = {stage: _verify_completed_stage(final_plan, stage) for stage in STAGES}

        terminal = {
            **progress_payload(
                run_name=args.run_name,
                run_root=run_root,
                preflight=preflight,
                expected_commit=args.expected_commit,
                state="succeeded",
                current_stage=None,
                completed_stages=completed_stages,
            ),
            "passed": True,
            "exit_code": 0,
            "started_at": started_at,
            "finished_at": utc_now(),
            "preflight_sha256": sha256_file(preflight),
            "platform_status_sha256": platform_binding["sha256"],
            "dag_plan_sha256": locked_plan_sha256,
            "final_stage_bindings": final_stage_bindings,
        }
        atomic_json(terminal_path, terminal)
        atomic_json(progress_path, terminal)
        return 0
    except StageOrchestrationTimeout as exc:
        terminal = {
            **progress_payload(
                run_name=args.run_name,
                run_root=run_root,
                preflight=preflight,
                expected_commit=args.expected_commit,
                state=("orchestration_timeout_unknown" if exc.worker_may_still_be_running else "failed"),
                current_stage=exc.stage,
                completed_stages=completed_stages,
                detail=f"{type(exc).__name__}: {exc}",
            ),
            "passed": False,
            "exit_code": 124,
            "started_at": started_at,
            "finished_at": utc_now(),
            "stage_run_dir": str(exc.run_dir),
            "worker_may_still_be_running": exc.worker_may_still_be_running,
            "stage_status_snapshot": exc.snapshot,
            "traceback": traceback.format_exc(),
        }
        atomic_json(terminal_path, terminal)
        atomic_json(progress_path, terminal)
        append_event(events_path, {"event": "sequence_timeout", "at": utc_now(), "detail": terminal["detail"]})
        return 124
    except BaseException as exc:
        terminal = {
            **progress_payload(
                run_name=args.run_name,
                run_root=run_root,
                preflight=preflight,
                expected_commit=args.expected_commit,
                state="failed",
                current_stage=None,
                completed_stages=completed_stages,
                detail=f"{type(exc).__name__}: {exc}",
            ),
            "passed": False,
            "exit_code": 1,
            "started_at": started_at,
            "finished_at": utc_now(),
            "traceback": traceback.format_exc(),
        }
        atomic_json(terminal_path, terminal)
        atomic_json(progress_path, terminal)
        append_event(
            events_path,
            {"event": "sequence_failed", "at": utc_now(), "error": terminal["detail"]},
        )
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run only the fail-stop R3 technical sequence from an Inspire notebook post-start command"
    )
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--dev-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-instance", required=True)
    parser.add_argument("--expected-node", required=True)
    parser.add_argument("--expected-workload-kind", choices=("notebook", "job"), default="notebook")
    parser.add_argument("--expected-compute-group", default="开发区-H200-3号机房")
    parser.add_argument("--expected-image", required=True)
    parser.add_argument("--expected-workspace", required=True)
    parser.add_argument("--expected-project", required=True)
    parser.add_argument("--platform-status", type=Path, required=True)
    parser.add_argument("--platform-status-sha256", required=True)
    parser.add_argument("--max-platform-status-age-seconds", type=int, default=1800)
    parser.add_argument("--expected-driver", default="570.124.06")
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if len(args.expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in args.expected_commit
    ):
        parser.error("--expected-commit must be a lowercase 40-character commit")
    for name in ("train_sha256", "dev_sha256", "platform_status_sha256"):
        value = getattr(args, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"--{name.replace('_', '-')} must be a lowercase SHA256")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.max_platform_status_age_seconds <= 0:
        parser.error("--max-platform-status-age-seconds must be positive")
    if not args.run_name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in args.run_name
    ):
        parser.error("--run-name must contain only lowercase letters, digits, dot, underscore, or dash")
    return args


def main() -> int:
    args = parse_args()
    try:
        return run_sequence(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
