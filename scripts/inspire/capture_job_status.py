from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


PROTOCOL = "vision-memory-inspire-job-status.v1"
SENSITIVE_WORDS = ("token", "password", "secret", "credential", "api-key", "api_key")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Inspire output did not contain a JSON object")


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Inspire payload field {label} must be an object")
    return value


def unwrap(payload: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if payload.get("success") is True:
        return require_mapping(payload.get("data"), f"{label}.data")
    return payload


def option_value(command: list[str], option: str) -> str | None:
    for index, argument in enumerate(command):
        if argument == option and index + 1 < len(command):
            return command[index + 1]
        prefix = f"{option}="
        if argument.startswith(prefix):
            return argument[len(prefix) :]
    return None


def normalize_job_payloads(
    status_payload: Mapping[str, Any],
    instances_payload: Mapping[str, Any],
    *,
    status_stdout_sha256: str,
    status_stderr_sha256: str,
    status_exit_code: int,
    status_command: list[str],
    instances_stdout_sha256: str,
    instances_stderr_sha256: str,
    instances_exit_code: int,
    instances_command: list[str],
) -> dict[str, Any]:
    status_data = unwrap(status_payload, "status")
    instances_data = unwrap(instances_payload, "instances")
    job = require_mapping(status_data.get("job"), "status.data.job")
    rows = instances_data.get("instances")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError("Inspire job must expose exactly one runtime instance")
    instance = rows[0]
    if instances_data.get("total") != 1:
        raise ValueError("Inspire job instance total must be exactly one")
    status_job_id = job.get("job_id")
    instances_job_id = instances_data.get("job_id")
    if not isinstance(status_job_id, str) or status_job_id != instances_job_id:
        raise ValueError("Inspire status and instances payloads identify different jobs")
    framework = job.get("framework_config")
    if not isinstance(framework, list) or len(framework) != 1 or not isinstance(framework[0], Mapping):
        raise ValueError("Inspire job must expose exactly one framework configuration")
    spec = framework[0]
    price = require_mapping(spec.get("instance_spec_price_info"), "framework_config[0].instance_spec_price_info")
    gpu = require_mapping(price.get("gpu_info"), "instance_spec_price_info.gpu_info")
    job_name = job.get("name")
    command = job.get("command")
    node = instance.get("node")
    instance_name = instance.get("name")
    if not all(isinstance(value, str) and value for value in (job_name, command, node, instance_name)):
        raise ValueError("Inspire job identity, command, instance, or node is malformed")
    status_workspace = option_value(status_command, "--workspace")
    instances_workspace = option_value(instances_command, "--workspace")
    if not status_workspace or status_workspace != instances_workspace:
        raise ValueError("Inspire job queries must bind the same explicit workspace")
    if job.get("workspace_name") not in (None, status_workspace):
        raise ValueError("Inspire job detail disagrees with the queried workspace")
    normalized_gpu_product = gpu.get("gpu_product_simple") or gpu.get("gpu_type_display")
    if isinstance(normalized_gpu_product, str) and "H200" in normalized_gpu_product.upper():
        normalized_gpu_product = "H200"
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "workload_kind": "job",
        "captured_at": utc_now(),
        "source": "inspire --json job status + job instances",
        "status_source_command": status_command,
        "status_source_exit_code": status_exit_code,
        "status_source_stdout_sha256": status_stdout_sha256,
        "status_source_stderr_sha256": status_stderr_sha256,
        "instances_source_command": instances_command,
        "instances_source_exit_code": instances_exit_code,
        "instances_source_stdout_sha256": instances_stdout_sha256,
        "instances_source_stderr_sha256": instances_stderr_sha256,
        "accepted_nonzero_after_valid_payload": status_exit_code != 0 or instances_exit_code != 0,
        "instance": job_name,
        "runtime_instance": instance_name,
        "status": job.get("status"),
        "node": node,
        "node_status": "READY" if instance.get("instance_status") == "instance_running" else instance.get("instance_status"),
        "runtime_instance_status": instance.get("instance_status"),
        "runtime_instance_type": instance.get("instance_type"),
        "image": spec.get("image"),
        "image_source": job.get("image_source"),
        "workspace": status_workspace,
        "project": job.get("project_name"),
        "project_priority": str(job.get("priority_level") or job.get("priority") or ""),
        "gpu_product": normalized_gpu_product,
        "gpu_type": gpu.get("gpu_type") or gpu.get("gpu_type_display"),
        "gpu_count": spec.get("gpu_count"),
        "cpu_count": spec.get("cpu") or price.get("cpu_count"),
        "memory_gib": spec.get("mem_gi"),
        "shared_memory_gib": spec.get("shm_gi"),
        "node_count": spec.get("instance_count") or job.get("node_count"),
        "compute_group": job.get("logic_compute_group_name"),
        "framework": job.get("framework"),
        "command_sha256": sha256_bytes(command.encode("utf-8")),
    }


def run_command(command: list[str], timeout_seconds: float) -> tuple[bytes, bytes, int]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=timeout_seconds)
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        return exc.stdout or b"", exc.stderr or b"", 124


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def split_commands(arguments: list[str]) -> tuple[list[str], list[str]]:
    values = list(arguments)
    if values and values[0] == "--":
        values = values[1:]
    if "---" not in values:
        raise ValueError("separate the status and instances commands with ---")
    index = values.index("---")
    status = values[:index]
    instances = values[index + 1 :]
    if not status or not instances:
        raise ValueError("both Inspire job commands are required")
    return status, instances


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a SHA-bound Inspire GPU job runtime receipt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("commands", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        status_command, instances_command = split_commands(args.commands)
        if args.timeout_seconds <= 0:
            raise ValueError("--timeout-seconds must be positive")
        for command in (status_command, instances_command):
            if any(any(word in argument.casefold() for word in SENSITIVE_WORDS) for argument in command):
                raise ValueError("refusing to record a command that may contain credentials")
        status_stdout, status_stderr, status_exit = run_command(status_command, args.timeout_seconds)
        instances_stdout, instances_stderr, instances_exit = run_command(instances_command, args.timeout_seconds)
        try:
            status_payload = extract_json_object(status_stdout.decode("utf-8", errors="replace"))
        except ValueError:
            status_payload = extract_json_object(status_stderr.decode("utf-8", errors="replace"))
        try:
            instances_payload = extract_json_object(instances_stdout.decode("utf-8", errors="replace"))
        except ValueError:
            instances_payload = extract_json_object(instances_stderr.decode("utf-8", errors="replace"))
        receipt = normalize_job_payloads(
            status_payload,
            instances_payload,
            status_stdout_sha256=sha256_bytes(status_stdout),
            status_stderr_sha256=sha256_bytes(status_stderr),
            status_exit_code=status_exit,
            status_command=status_command,
            instances_stdout_sha256=sha256_bytes(instances_stdout),
            instances_stderr_sha256=sha256_bytes(instances_stderr),
            instances_exit_code=instances_exit,
            instances_command=instances_command,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    output = args.output.resolve()
    encoded = (json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_write(output, encoded)
    digest = sha256_bytes(encoded)
    atomic_write(output.with_suffix(output.suffix + ".sha256"), f"{digest}  {output.name}\n".encode("utf-8"))
    print(json.dumps({"output": str(output), "sha256": digest, "receipt": receipt}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
