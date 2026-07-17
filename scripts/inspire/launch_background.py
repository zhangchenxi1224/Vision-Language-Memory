from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAGE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
INFRASTRUCTURE_STAGES = {
    "data-build",
    "environment-smoke",
    "model-fetch",
    "source-fetch",
    "teacher-cache-build",
}
STRICT_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_VISIBLE_DEVICES": "0,1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "TOKENIZERS_PARALLELISM": "false",
}
SENSITIVE_OPTION = re.compile(r"^--(?:hf-?)?(?:token|password|secret|credential|api[-_]?key)(?:=|$)", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload.encode("utf-8"))
    temporary.replace(path)


def git(repo: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def verify_clean_commit(repo: Path, expected_commit: str) -> None:
    actual = git(repo, "rev-parse", "HEAD")
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if actual != expected_commit:
        raise ValueError(f"Commit mismatch: expected {expected_commit}, got {actual}")
    if status != "":
        raise ValueError("Formal Inspire stages require a clean checkout")


def verify_preflight(path: Path, *, expected_commit: str, infrastructure_stage: bool) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ValueError(f"Preflight report is missing: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"Preflight SHA sidecar is missing: {sidecar}")
    expected_sha = sidecar.read_text(encoding="utf-8").split()[0]
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(f"Preflight SHA mismatch: expected {expected_sha}, got {actual_sha}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise ValueError("Preflight report did not pass")
    if report.get("git", {}).get("commit") != expected_commit:
        raise ValueError("Preflight report was produced from a different commit")
    if not infrastructure_stage and report.get("formal_ready") is not True:
        raise ValueError("Scientific stages require a model-complete formal preflight")
    return report, actual_sha


def validate_command(command: list[str]) -> None:
    if not command:
        raise ValueError("A command is required after --")
    for argument in command:
        if SENSITIVE_OPTION.match(argument) or argument.lower().startswith("hf_"):
            raise ValueError("Secrets must not be passed on a recorded stage command line")


def worker(config_path: Path, expected_config_sha: str) -> int:
    run_dir = config_path.parent
    try:
        if sha256_file(config_path) != expected_config_sha:
            raise ValueError("Worker configuration SHA mismatch")
        configuration = json.loads(config_path.read_text(encoding="utf-8"))
        repo = Path(configuration["repo"])
        expected_commit = configuration["expected_commit"]
        verify_clean_commit(repo, expected_commit)
        command = [str(value) for value in configuration["command"]]
        validate_command(command)
        started_at = utc_now()
        running = {
            "schema_version": 1,
            "status": "running",
            "stage": configuration["stage"],
            "pid": os.getpid(),
            "started_at": started_at,
            "expected_commit": expected_commit,
            "configuration_sha256": expected_config_sha,
        }
        atomic_json(run_dir / "running.json", running)

        environment = os.environ.copy()
        environment.update(STRICT_ENVIRONMENT)
        environment.update(
            {
                "VLM_STAGE_WORKER_INPUT": str(config_path),
                "VLM_STAGE_CONFIGURATION_SHA256": expected_config_sha,
                "VLM_STAGE_PREFLIGHT": str(configuration["preflight"]),
                "VLM_STAGE_PREFLIGHT_SHA256": str(configuration["preflight_sha256"]),
            }
        )
        with (run_dir / "stdout.log").open("wb") as stdout, (run_dir / "stderr.log").open("wb") as stderr:
            process = subprocess.run(
                command,
                cwd=repo,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        exit_code = process.returncode
        terminal = {
            **running,
            "status": "succeeded" if exit_code == 0 else "failed",
            "passed": exit_code == 0,
            "exit_code": exit_code,
            "finished_at": utc_now(),
            "stdout_sha256": sha256_file(run_dir / "stdout.log"),
            "stderr_sha256": sha256_file(run_dir / "stderr.log"),
        }
        atomic_json(run_dir / "terminal.json", terminal)
        return exit_code
    except BaseException as exc:  # pragma: no cover - defensive detached-worker path
        terminal = {
            "schema_version": 1,
            "status": "launcher_failed",
            "passed": False,
            "exit_code": 125,
            "finished_at": utc_now(),
            "error": f"{type(exc).__name__}: {exc}",
            "configuration_sha256": expected_config_sha,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(run_dir / "terminal.json", terminal)
        return 125


def launch(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    run_root = args.run_root.resolve()
    run_dir = args.run_dir.resolve()
    if not args.run_root.is_absolute() or not args.run_dir.is_absolute():
        raise ValueError("--run-root and --run-dir must be absolute paths")
    if not is_within(run_dir, run_root) or run_dir == run_root:
        raise ValueError("--run-dir must be a child of --run-root")
    if not STAGE_PATTERN.fullmatch(args.stage):
        raise ValueError("--stage must use lowercase letters, digits, dot, underscore, or dash")
    if args.infrastructure_stage and args.stage not in INFRASTRUCTURE_STAGES:
        raise ValueError(f"Infrastructure bypass is restricted to {sorted(INFRASTRUCTURE_STAGES)}")
    if len(args.expected_commit) != 40 or any(character not in "0123456789abcdef" for character in args.expected_commit.lower()):
        raise ValueError("--expected-commit must be a full hexadecimal commit")
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    validate_command(command)
    verify_clean_commit(repo, args.expected_commit.lower())
    _, preflight_sha = verify_preflight(
        args.preflight.resolve(),
        expected_commit=args.expected_commit.lower(),
        infrastructure_stage=args.infrastructure_stage,
    )

    run_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=False, exist_ok=False)
    configuration = {
        "schema_version": 1,
        "stage": args.stage,
        "repo": str(repo),
        "run_root": str(run_root),
        "run_dir": str(run_dir),
        "expected_commit": args.expected_commit.lower(),
        "preflight": str(args.preflight.resolve()),
        "preflight_sha256": preflight_sha,
        "infrastructure_stage": args.infrastructure_stage,
        "strict_environment": STRICT_ENVIRONMENT,
        "command": command,
        "created_at": utc_now(),
    }
    config_path = run_dir / "worker_input.json"
    atomic_json(config_path, configuration)
    config_sha = sha256_file(config_path)
    atomic_json(
        run_dir / "launch.json",
        {
            "schema_version": 1,
            "stage": args.stage,
            "status": "launching",
            "expected_commit": args.expected_commit.lower(),
            "preflight_sha256": preflight_sha,
            "configuration_sha256": config_sha,
            "created_at": configuration["created_at"],
        },
    )

    with open(os.devnull, "rb") as stdin, open(os.devnull, "ab") as output:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path), config_sha],
            cwd=repo,
            env=os.environ.copy(),
            stdin=stdin,
            stdout=output,
            stderr=output,
            start_new_session=True,
            close_fds=True,
        )
    deadline = time.monotonic() + args.start_timeout_seconds
    while time.monotonic() < deadline:
        if (run_dir / "running.json").is_file() or (run_dir / "terminal.json").is_file():
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Detached worker did not write a sentinel before the launch timeout")

    print(
        json.dumps(
            {
                "status": "started",
                "stage": args.stage,
                "pid": process.pid,
                "run_dir": str(run_dir),
                "sentinel": str(run_dir / "terminal.json"),
                "configuration_sha256": config_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        if len(sys.argv) != 4:
            raise SystemExit("internal worker requires CONFIG_PATH CONFIG_SHA256")
        return worker(Path(sys.argv[2]).resolve(), sys.argv[3])

    parser = argparse.ArgumentParser(description="Launch one long Inspire stage as a detached process with audit sentinels")
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=os.environ.get("VLM_RUN_ROOT"), required=os.environ.get("VLM_RUN_ROOT") is None)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--infrastructure-stage", action="store_true")
    parser.add_argument("--start-timeout-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        return launch(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
