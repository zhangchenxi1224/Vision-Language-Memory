from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.inspire import start_r3_sequence_after_status as waiter
from scripts.inspire.r3_dag_contract import atomic_json


def running_receipt(path: Path) -> str:
    return atomic_json(
        path,
        {
            "schema_version": 1,
            "protocol": waiter.PLATFORM_STATUS_PROTOCOL,
            "status": "RUNNING",
            "node_status": "READY",
            "node": "qb-prod-gpu2059",
        },
    )


def test_load_running_receipt_is_sha_bound(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    digest = running_receipt(path)
    value, actual = waiter.load_running_receipt(path)
    assert actual == digest
    assert value["node"] == "qb-prod-gpu2059"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar mismatch"):
        waiter.load_running_receipt(path)


def test_refuses_forwarded_derived_platform_options() -> None:
    waiter.validate_forwarded(["--repo", "/repo"])
    for option in waiter.FORBIDDEN_FORWARDED_OPTIONS:
        with pytest.raises(ValueError, match="derived exclusively"):
            waiter.validate_forwarded(["--repo", "/repo", f"{option}=tampered"])


def test_stopped_receipt_cannot_launch(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    running_receipt(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    value["status"] = "STOPPED"
    atomic_json(path, value)
    with pytest.raises(ValueError, match="running ready"):
        waiter.load_running_receipt(path)


def test_job_receipt_derives_job_workload_kind(tmp_path: Path) -> None:
    path = tmp_path / "job-status.json"
    digest = atomic_json(
        path,
        {
            "schema_version": 1,
            "protocol": waiter.JOB_STATUS_PROTOCOL,
            "workload_kind": "job",
            "status": "RUNNING",
            "node_status": "READY",
            "node": "qb-prod-gpu2007",
        },
    )
    value, actual = waiter.load_running_receipt(path)
    assert actual == digest
    assert value["workload_kind"] == "job"
