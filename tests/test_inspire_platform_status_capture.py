from __future__ import annotations

import json

import pytest

from scripts.inspire.capture_platform_status import extract_json_object, normalize_status_payload


def payload() -> dict:
    return {
        "success": True,
        "data": {
            "name": "vlm-r3-h200x2-live-20260717",
            "status": "RUNNING",
            "runtime": "standard",
            "extra_info": {"NodeName": "qb-prod-gpu2059"},
            "node": {"name": "qb-prod-gpu2059", "status": "READY"},
            "image": {
                "name": "ngc-pytorch",
                "version": "25.02-cuda12.8.0-py3",
                "source": "SOURCE_OFFICIAL",
            },
            "workspace": {"name": "分布式训练空间"},
            "project": {"name": "前沿课题探索", "priority_name": "10"},
            "quota": {"cpu_count": 40, "gpu_count": 2, "gpu_ram": 141, "memory_size": 400},
            "start_config": {
                "cpu_count": 40,
                "gpu_count": 2,
                "memory_size": 400,
                "shared_memory_size": 128,
                "auto_stop": False,
            },
            "resource_spec_price": {"gpu_info": {"gpu_product_simple": "H200", "gpu_type": "NVIDIA_H200_SXM_141G"}},
            "logic_compute_group": {"name": "开发区-H200-3号机房"},
        },
    }


def test_extracts_json_after_cli_warning_and_normalizes_status() -> None:
    value = extract_json_object("warning before JSON\n" + json.dumps(payload()) + "\ntraceback")
    receipt = normalize_status_payload(
        value,
        source_stdout_sha256="a" * 64,
        source_stderr_sha256="b" * 64,
        source_exit_code=1,
        source_command=["inspire", "--json", "notebook", "status"],
    )
    assert receipt["node"] == "qb-prod-gpu2059"
    assert receipt["gpu_count"] == 2
    assert receipt["shared_memory_gib"] == 128
    assert receipt["accepted_nonzero_after_valid_payload"] is True


def test_rejects_inconsistent_control_plane_node() -> None:
    value = payload()
    value["data"]["node"]["name"] = "different-node"
    with pytest.raises(ValueError, match="inconsistent node"):
        normalize_status_payload(
            value,
            source_stdout_sha256="a" * 64,
            source_stderr_sha256="b" * 64,
            source_exit_code=0,
            source_command=["inspire"],
        )
