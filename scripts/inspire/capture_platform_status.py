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


PROTOCOL = "vision-memory-inspire-platform-status.v1"
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
    raise ValueError("Inspire status output did not contain a JSON object")


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Inspire status payload field {label} must be an object")
    return value


def normalize_status_payload(
    payload: Mapping[str, Any],
    *,
    source_stdout_sha256: str,
    source_stderr_sha256: str,
    source_exit_code: int,
    source_command: list[str],
) -> dict[str, Any]:
    if payload.get("success") is True:
        data = require_mapping(payload.get("data"), "data")
    else:
        data = payload
    extra = require_mapping(data.get("extra_info"), "extra_info")
    node = require_mapping(data.get("node"), "node")
    image = require_mapping(data.get("image"), "image")
    workspace = require_mapping(data.get("workspace"), "workspace")
    project = require_mapping(data.get("project"), "project")
    quota = require_mapping(data.get("quota"), "quota")
    start = require_mapping(data.get("start_config"), "start_config")
    resource = require_mapping(data.get("resource_spec_price"), "resource_spec_price")
    gpu = require_mapping(resource.get("gpu_info"), "resource_spec_price.gpu_info")
    extra_node = extra.get("NodeName")
    node_name = node.get("name")
    if not isinstance(extra_node, str) or extra_node != node_name:
        raise ValueError("Inspire status payload has inconsistent node identities")
    if quota.get("gpu_count") != start.get("gpu_count") or quota.get("cpu_count") != start.get("cpu_count"):
        raise ValueError("Inspire status payload has inconsistent quota and start_config resources")
    if quota.get("memory_size") != start.get("memory_size"):
        raise ValueError("Inspire status payload has inconsistent memory resources")
    image_name = image.get("name")
    image_version = image.get("version")
    if not isinstance(image_name, str) or not isinstance(image_version, str):
        raise ValueError("Inspire status payload has a malformed image identity")
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "captured_at": utc_now(),
        "source": "inspire --json notebook status",
        "source_command": source_command,
        "source_exit_code": source_exit_code,
        "source_stdout_sha256": source_stdout_sha256,
        "source_stderr_sha256": source_stderr_sha256,
        "accepted_nonzero_after_valid_payload": source_exit_code != 0,
        "instance": data.get("name"),
        "status": data.get("status"),
        "node": node_name,
        "node_status": node.get("status"),
        "image": f"{image_name}:{image_version}",
        "image_source": image.get("source"),
        "workspace": workspace.get("name"),
        "project": project.get("name"),
        "project_priority": project.get("priority_name"),
        "gpu_product": gpu.get("gpu_product_simple"),
        "gpu_type": gpu.get("gpu_type"),
        "gpu_count": quota.get("gpu_count"),
        "gpu_ram_gib": quota.get("gpu_ram"),
        "cpu_count": quota.get("cpu_count"),
        "memory_gib": quota.get("memory_size"),
        "shared_memory_gib": start.get("shared_memory_size"),
        "auto_stop": start.get("auto_stop"),
        "runtime": data.get("runtime"),
        "compute_group": require_mapping(data.get("logic_compute_group"), "logic_compute_group").get("name"),
    }


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture and normalize an Inspire notebook status JSON response into a SHA-bound audit receipt"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("an Inspire status command is required after --")
    if any(any(word in argument.casefold() for word in SENSITIVE_WORDS) for argument in command):
        parser.error("refusing to record a command that may contain credentials")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=args.timeout_seconds,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or b""
        stderr = exc.stderr or b""
        exit_code = 124
    try:
        decoded_stdout = stdout.decode("utf-8", errors="replace")
        decoded_stderr = stderr.decode("utf-8", errors="replace")
        try:
            payload = extract_json_object(decoded_stdout)
        except ValueError:
            payload = extract_json_object(decoded_stderr)
        receipt = normalize_status_payload(
            payload,
            source_stdout_sha256=sha256_bytes(stdout),
            source_stderr_sha256=sha256_bytes(stderr),
            source_exit_code=exit_code,
            source_command=command,
        )
    except (TypeError, ValueError) as exc:
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
