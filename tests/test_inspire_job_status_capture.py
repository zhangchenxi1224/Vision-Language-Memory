from __future__ import annotations

import pytest

from scripts.inspire.capture_job_status import normalize_job_payloads


def payloads() -> tuple[dict, dict]:
    status = {
        "success": True,
        "data": {
            "source": "web",
            "job": {
                "job_id": "job-fixture",
                "name": "vlm-r3-tech-fixture",
                "status": "RUNNING",
                "workspace_name": "分布式训练空间",
                "project_name": "前沿课题探索",
                "logic_compute_group_name": "开发区-H200-3号机房",
                "priority_level": 10,
                "framework": "pytorch",
                "command": "bash /shared/launch.sh",
                "framework_config": [
                    {
                        "instance_count": 1,
                        "gpu_count": 2,
                        "cpu": 40,
                        "mem_gi": 400,
                        "shm_gi": 128,
                        "image": "ngc-pytorch:25.02-cuda12.8.0-py3",
                        "instance_spec_price_info": {
                            "cpu_count": 40,
                            "gpu_info": {"gpu_product_simple": "H200", "gpu_type": "NVIDIA_H200_SXM_141G"},
                        },
                    }
                ],
            },
        },
    }
    instances = {
        "success": True,
        "data": {
            "source": "web",
            "job_id": "job-fixture",
            "instances": [
                {
                    "name": "worker-0",
                    "instance_status": "instance_running",
                    "instance_type": "worker",
                    "node": "qb-prod-gpu2007",
                }
            ],
            "total": 1,
        },
    }
    return status, instances


def normalize(status: dict, instances: dict) -> dict:
    return normalize_job_payloads(
        status,
        instances,
        status_stdout_sha256="a" * 64,
        status_stderr_sha256="b" * 64,
        status_exit_code=0,
        status_command=["inspire", "--json", "job", "status", "vlm-r3-tech-fixture", "--workspace", "分布式训练空间"],
        instances_stdout_sha256="c" * 64,
        instances_stderr_sha256="d" * 64,
        instances_exit_code=0,
        instances_command=[
            "inspire",
            "--json",
            "job",
            "instances",
            "vlm-r3-tech-fixture",
            "--workspace",
            "分布式训练空间",
        ],
    )


def test_normalizes_single_node_two_h200_job() -> None:
    status, instances = payloads()
    receipt = normalize(status, instances)
    assert receipt["protocol"] == "vision-memory-inspire-job-status.v1"
    assert receipt["workload_kind"] == "job"
    assert receipt["instance"] == "vlm-r3-tech-fixture"
    assert receipt["node"] == "qb-prod-gpu2007"
    assert receipt["node_status"] == "READY"
    assert receipt["gpu_count"] == 2
    assert receipt["node_count"] == 1
    assert len(receipt["command_sha256"]) == 64


def test_rejects_cross_job_status_and_instances() -> None:
    status, instances = payloads()
    instances["data"]["job_id"] = "job-other"
    with pytest.raises(ValueError, match="different jobs"):
        normalize(status, instances)


def test_rejects_more_than_one_runtime_instance() -> None:
    status, instances = payloads()
    instances["data"]["instances"].append(dict(instances["data"]["instances"][0]))
    instances["data"]["total"] = 2
    with pytest.raises(ValueError, match="exactly one runtime instance"):
        normalize(status, instances)
