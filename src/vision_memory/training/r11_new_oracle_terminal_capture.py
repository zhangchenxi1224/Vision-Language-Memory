"""Locked contracts and audit for the R11_new oracle terminal-capture scan."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vision_memory.repro import canonical_tensor_sha256
from vision_memory.training import r11_new_direction_fidelity as direction_core


PROTOCOL = "R11-New-Oracle-Terminal-Capture-Target01"
PREFIX = "vision_memory.r11-new-oracle-terminal-capture"
CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs/experiments/r11_new_oracle_terminal_capture_target01.json"
)
CONFIG_BYTES_SHA256 = "cbf92ece9c2b64a28a10d0d7c637a05caa208396e9717ccbb88d6671a4094a7f"
CONFIG_CANONICAL_SHA256 = "3e0a8b2f912d75fd302d310403f8783fac455495989ba04fadc35fc9d84f0b01"
PASSES = ("teacher-outward", "plateau-inward")
GRID = (
    0.0,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    5e-2,
    2.0512112034227754,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def load_config() -> dict[str, Any]:
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "Terminal config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(
        canonical_sha(value) == CONFIG_CANONICAL_SHA256,
        "Terminal config canonical hash drift.",
    )
    require(
        value.get("schema") == f"{PREFIX}-config.v1" and value.get("protocol") == PROTOCOL,
        "Terminal config identity drift.",
    )
    contract = value["terminal_path_contract"]
    require(
        tuple(contract["requested_remaining_l2_grid"]) == GRID
        and tuple(contract["formal_passes"]) == PASSES
        and len(contract["point_bindings"]) == len(GRID) == 17,
        "Terminal path grid drift.",
    )
    require(
        value["formal_technical_gate"]["total_full_chain_forward_calls"] == 35
        and value["formal_technical_gate"]["scan_rows"] == 34
        and value["interpretation_boundaries"]["formal_success_always_false"] is True
        and value["interpretation_boundaries"]["phase2_always_false"] is True,
        "Terminal execution/boundary drift.",
    )
    return value


def construct_points(
    teacher_x_t: torch.Tensor, plateau_x_t: torch.Tensor, config: Mapping[str, Any]
) -> list[torch.Tensor]:
    require(
        isinstance(teacher_x_t, torch.Tensor)
        and isinstance(plateau_x_t, torch.Tensor)
        and teacher_x_t.dtype == plateau_x_t.dtype == torch.float32
        and teacher_x_t.shape == plateau_x_t.shape == (1, 4, 128, 128)
        and bool(torch.isfinite(teacher_x_t).all())
        and bool(torch.isfinite(plateau_x_t).all()),
        "Invalid terminal endpoint tensors.",
    )
    contract = config["terminal_path_contract"]
    distance = float(contract["full_distance_D"])
    observed_distance = float((plateau_x_t - teacher_x_t).double().norm())
    require(
        math.isclose(observed_distance, distance, rel_tol=0.0, abs_tol=1e-12),
        "Parent plateau distance drift.",
    )
    unit = (plateau_x_t.double() - teacher_x_t.double()) / distance
    points: list[torch.Tensor] = []
    for index, binding in enumerate(contract["point_bindings"]):
        requested = float(binding["requested_remaining_l2"])
        require(requested == GRID[index], "Terminal point order drift.")
        if index == 0:
            point = teacher_x_t.clone()
        elif index == len(GRID) - 1:
            point = plateau_x_t.clone()
        else:
            point = (teacher_x_t.double() + requested * unit).float()
        statistics = direction_core.quantization_statistics(point, teacher_x_t)
        require(
            canonical_tensor_sha256(point) == binding["x_T_fp32_sha256"],
            f"Terminal point tensor drift at index {index}.",
        )
        expected = {
            "fp32_l2_to_teacher": binding["actual_fp32_l2"],
            "bf16_equal_fraction_to_teacher": binding["bf16_equal_fraction_to_teacher"],
            "bf16_l2_to_teacher": binding["bf16_l2_to_teacher"],
        }
        require(
            all(
                math.isclose(statistics[name], float(value), rel_tol=1e-12, abs_tol=1e-15)
                for name, value in expected.items()
            ),
            f"Terminal point quantization drift at index {index}.",
        )
        points.append(point)
    return points


def scan_row_id(pass_name: str, point_index: int) -> str:
    require(
        pass_name in PASSES and type(point_index) is int and 0 <= point_index < len(GRID),
        "Invalid terminal scan-row identity.",
    )
    return f"{pass_name}__p{point_index:02d}"


def _validate_row(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, int]:
    pass_name = row.get("pass")
    point_index = row.get("point_index")
    require(
        row.get("schema") == f"{PREFIX}-scan-row.v1"
        and row.get("protocol") == PROTOCOL
        and pass_name in PASSES
        and type(point_index) is int
        and 0 <= point_index < len(GRID)
        and row.get("row_id") == scan_row_id(pass_name, point_index),
        "Invalid terminal scan row.",
    )
    binding = config["terminal_path_contract"]["point_bindings"][point_index]
    require(
        row.get("requested_remaining_l2") == binding["requested_remaining_l2"]
        and row.get("x_T_fp32_sha256") == binding["x_T_fp32_sha256"],
        "Terminal row point binding drift.",
    )
    numeric = (
        row.get("loss"),
        row.get("loss_ratio_to_plateau"),
        row.get("actual_fp32_l2_to_teacher"),
        row.get("bf16_l2_to_teacher"),
        row.get("bf16_equal_fraction_to_teacher"),
    )
    require(
        all(type(value) in (int, float) and math.isfinite(value) for value in numeric)
        and all(value >= 0.0 for value in numeric[:4])
        and 0.0 <= numeric[4] <= 1.0
        and math.isclose(numeric[2], binding["actual_fp32_l2"], rel_tol=1e-12, abs_tol=1e-15)
        and math.isclose(
            numeric[3], binding["bf16_l2_to_teacher"], rel_tol=1e-12, abs_tol=1e-15
        )
        and numeric[4] == binding["bf16_equal_fraction_to_teacher"],
        "Non-finite or drifted terminal row.",
    )
    endpoint_hash = row.get("endpoint_fp32_sha256")
    require(
        isinstance(endpoint_hash, str) and len(endpoint_hash) == 64,
        "Invalid terminal endpoint hash.",
    )
    return pass_name, point_index


def summarize_scan(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    require(len(rows) == 34, "Terminal scan row count drift.")
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = _validate_row(row, config)
        require(key not in by_key, "Duplicate terminal scan row.")
        by_key[key] = row
    duplicates = []
    canonical_rows = []
    for index, requested in enumerate(GRID):
        outward = by_key[("teacher-outward", index)]
        inward = by_key[("plateau-inward", index)]
        equal = (
            outward["endpoint_fp32_sha256"] == inward["endpoint_fp32_sha256"]
            and outward["loss"] == inward["loss"]
            and outward["loss_ratio_to_plateau"] == inward["loss_ratio_to_plateau"]
        )
        require(equal, f"Duplicate terminal endpoint mismatch at index {index}.")
        duplicates.append(
            {
                "point_index": index,
                "requested_remaining_l2": requested,
                "bitwise_equal": True,
            }
        )
        canonical_rows.append(outward)
    contract = config["terminal_path_contract"]
    fixed = config["fixed_parent_artifacts"]
    require(
        canonical_rows[0]["loss"] == 0.0
        and canonical_rows[0]["loss_ratio_to_plateau"] == 0.0
        and canonical_rows[0]["endpoint_bitwise_equal_to_teacher"] is True
        and canonical_rows[0]["endpoint_fp32_sha256"] == fixed["teacher_endpoint_fp32_sha256"]
        and canonical_rows[-1]["endpoint_fp32_sha256"] == fixed["plateau_endpoint_fp32_sha256"]
        and math.isclose(
            canonical_rows[-1]["loss_ratio_to_plateau"], 1.0, rel_tol=1e-6, abs_tol=1e-12
        ),
        "Terminal teacher/plateau endpoint boundary drift.",
    )
    strong_threshold = float(contract["strong_capture_loss_ratio_lte"])
    meaningful_threshold = float(contract["meaningful_capture_loss_ratio_lte"])
    strong = [row["loss_ratio_to_plateau"] <= strong_threshold for row in canonical_rows]
    meaningful = [
        row["loss_ratio_to_plateau"] <= meaningful_threshold for row in canonical_rows
    ]
    prefix_end = 0
    for index, passed in enumerate(strong):
        if not passed:
            break
        prefix_end = index
    disconnected = any(strong[index] for index in range(prefix_end + 1, len(strong)))
    significant_monotonicity_violations = []
    for index in range(1, len(canonical_rows)):
        previous = float(canonical_rows[index - 1]["loss_ratio_to_plateau"])
        current = float(canonical_rows[index]["loss_ratio_to_plateau"])
        if current < previous - 0.1:
            significant_monotonicity_violations.append(
                {"from_index": index - 1, "to_index": index, "absolute_drop": previous - current}
            )
    return {
        "duplicate_replays": duplicates,
        "all_duplicate_endpoints_bitwise_equal": True,
        "strong_capture_point_indices": [index for index, passed in enumerate(strong) if passed],
        "meaningful_capture_point_indices": [
            index for index, passed in enumerate(meaningful) if passed
        ],
        "strong_capture_prefix_end_index": prefix_end,
        "strong_capture_prefix_max_requested_remaining_l2": GRID[prefix_end],
        "strong_capture_disconnected": disconnected,
        "largest_bitwise_teacher_endpoint_index": max(
            index
            for index, row in enumerate(canonical_rows)
            if row["endpoint_bitwise_equal_to_teacher"]
        ),
        "largest_all_bf16_codes_equal_index": max(
            index
            for index, binding in enumerate(contract["point_bindings"])
            if binding["bf16_equal_fraction_to_teacher"] == 1.0
        ),
        "significant_monotonicity_violations": significant_monotonicity_violations,
        "canonical_teacher_outward_rows": canonical_rows,
    }


def classify_outcome(summary: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    if summary["strong_capture_disconnected"]:
        return "nonmonotone_capture_anomaly"
    prefix_end = int(summary["strong_capture_prefix_end_index"])
    remaining = float(summary["strong_capture_prefix_max_requested_remaining_l2"])
    contract = config["terminal_path_contract"]
    if remaining >= float(contract["macroscopic_capture_remaining_l2_gte"]):
        return "macroscopic_terminal_capture"
    if remaining >= float(contract["narrow_capture_remaining_l2_gte"]):
        return "narrow_terminal_capture"
    if prefix_end == 0:
        return "exact_teacher_only"
    binding = contract["point_bindings"][prefix_end]
    if binding["bf16_equal_fraction_to_teacher"] < 1.0:
        return "microscopic_partial_code_capture"
    return "bf16_exact_cell_only"


def validate_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory = json.loads((root / "artifact_inventory.json").read_text(encoding="utf-8"))
    require(inventory.get("schema") == f"{PREFIX}-inventory.v1", "Inventory schema drift.")
    listed: set[str] = set()
    for item in inventory.get("artifacts", []):
        relative = item.get("path")
        require(
            isinstance(relative, str)
            and relative not in listed
            and "\\" not in relative
            and not Path(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            "Unsafe or duplicate inventory path.",
        )
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root) and path.is_file() and not path.is_symlink(),
            "Escaping/missing terminal artifact.",
        )
        require(
            path.stat().st_size == item.get("bytes") and sha256_file(path) == item.get("sha256"),
            "Terminal artifact bytes/hash mismatch.",
        )
        listed.add(relative)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    require(
        observed == listed and inventory.get("artifact_count") == len(listed),
        "Incomplete terminal inventory.",
    )
    return {"artifact_count": len(listed), "listed": listed}


def _valid_fp32_tensor(value: Any) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        and tuple(value.shape) == (1, 4, 128, 128)
        and bool(torch.isfinite(value).all())
    )


def audit_delivery(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    inventory = validate_inventory(root)
    required = {
        "config.json",
        "manifest.json",
        "result.json",
        "terminal.json",
        "runtime.json",
        "target/fixed_parent_target.pt",
        "parent/lr001-step256.pt",
        "terminal_path_tensors.pt",
        "model_snapshot_verification_start.json",
        "model_snapshot_verification_end.json",
    }
    require(required.issubset(inventory["listed"]), "Required terminal artifacts missing.")
    saved_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    require(saved_config == config and manifest.get("config_sha256") == canonical_sha(config), "Config drift.")
    require(
        manifest.get("protocol") == result.get("protocol") == terminal.get("protocol") == PROTOCOL
        and manifest.get("git_commit") == result.get("git_commit") == terminal.get("git_commit"),
        "Terminal protocol/Git drift.",
    )
    require(
        terminal.get("status") == "technical_completed"
        and terminal.get("engineering_gate") is True
        and terminal.get("exit_code") == 0
        and terminal.get("formal_success") is False
        and terminal.get("phase2_allowed") is False
        and result.get("engineering_gate") is True
        and result.get("formal_success") is False
        and result.get("phase2_allowed") is False,
        "Terminal result boundary drift.",
    )
    start_snapshot = json.loads(
        (root / "model_snapshot_verification_start.json").read_text(encoding="utf-8")
    )
    end_snapshot = json.loads(
        (root / "model_snapshot_verification_end.json").read_text(encoding="utf-8")
    )
    require(
        start_snapshot == end_snapshot
        and result.get("snapshots_unchanged") is True
        and result.get("models_frozen") is True
        and result.get("only_student_x_T_trainable") is True,
        "Terminal frozen/snapshot evidence drift.",
    )
    parent = config["fixed_parent_artifacts"]
    target_path = root / "target/fixed_parent_target.pt"
    plateau_path = root / "parent/lr001-step256.pt"
    require(
        sha256_file(target_path) == parent["source_target_artifact_sha256"]
        and sha256_file(plateau_path) == parent["source_plateau_checkpoint_sha256"],
        "Copied terminal parent artifact drift.",
    )
    target = torch.load(target_path, map_location="cpu", weights_only=True)
    plateau = torch.load(plateau_path, map_location="cpu", weights_only=True)
    teacher_x_t = target["teacher_x_T_fp32"]
    teacher_endpoint = target["teacher_endpoint_fp32"]
    plateau_x_t = plateau["student_x_T_fp32"]
    plateau_endpoint = plateau["endpoint_fp32"]
    require(
        canonical_tensor_sha256(teacher_x_t) == parent["teacher_x_T_fp32_sha256"]
        and canonical_tensor_sha256(teacher_endpoint) == parent["teacher_endpoint_fp32_sha256"]
        and canonical_tensor_sha256(plateau_x_t) == parent["plateau_x_T_fp32_sha256"]
        and canonical_tensor_sha256(plateau_endpoint) == parent["plateau_endpoint_fp32_sha256"],
        "Terminal parent tensor hash drift.",
    )
    points = construct_points(teacher_x_t, plateau_x_t, config)
    bundle = torch.load(root / "terminal_path_tensors.pt", map_location="cpu", weights_only=True)
    require(
        bundle.get("schema") == f"{PREFIX}-tensor-bundle.v1"
        and bundle.get("protocol") == PROTOCOL
        and result.get("tensor_bundle_sha256") == sha256_file(root / "terminal_path_tensors.pt")
        and len(bundle.get("candidates", [])) == 17
        and all(torch.equal(left, right) for left, right in zip(bundle["candidates"], points, strict=True)),
        "Terminal tensor bundle drift.",
    )
    baseline_mse = float((plateau_endpoint - teacher_endpoint).square().mean())
    require(
        math.isclose(baseline_mse, parent["plateau_endpoint_mse"], rel_tol=1e-6, abs_tol=1e-10)
        and math.isclose(result["plateau_endpoint_mse"], baseline_mse, rel_tol=1e-6, abs_tol=1e-10),
        "Terminal plateau baseline drift.",
    )
    mode = result.get("mode")
    require(mode in ("technical-preflight", "formal") and terminal.get("mode") == mode, "Mode drift.")
    if mode == "technical-preflight":
        endpoints = bundle.get("scan_endpoints", {})
        require(
            result.get("counters")
            == {
                "full_chain_forward_calls": 2,
                "reader_forward_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
            }
            and result.get("preflight_gate") is True
            and set(endpoints) == {"teacher-replay", "plateau-replay"}
            and torch.equal(endpoints["teacher-replay"], teacher_endpoint)
            and torch.equal(endpoints["plateau-replay"], plateau_endpoint),
            "Terminal preflight execution drift.",
        )
        summary = None
    else:
        require(
            result.get("counters")
            == {
                "full_chain_forward_calls": 35,
                "reader_forward_calls": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
            }
            and "scan_metrics.jsonl" in inventory["listed"],
            "Terminal formal execution drift.",
        )
        rows = [
            json.loads(line)
            for line in (root / "scan_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        endpoints = bundle.get("scan_endpoints", {})
        require(
            len(endpoints) == 34 and set(endpoints) == {row.get("row_id") for row in rows},
            "Terminal endpoint coverage drift.",
        )
        for row in rows:
            pass_name, index = _validate_row(row, config)
            endpoint = endpoints.get(scan_row_id(pass_name, index))
            require(
                _valid_fp32_tensor(endpoint)
                and canonical_tensor_sha256(endpoint) == row["endpoint_fp32_sha256"],
                "Terminal endpoint tensor/hash drift.",
            )
            loss = float((endpoint - teacher_endpoint).square().mean())
            statistics = direction_core.quantization_statistics(points[index], teacher_x_t)
            require(
                math.isclose(loss, row["loss"], rel_tol=1e-6, abs_tol=1e-12)
                and math.isclose(
                    loss / baseline_mse,
                    row["loss_ratio_to_plateau"],
                    rel_tol=1e-6,
                    abs_tol=1e-12,
                )
                and row["endpoint_bitwise_equal_to_teacher"]
                == torch.equal(endpoint, teacher_endpoint)
                and row["x_T_fp32_sha256"] == canonical_tensor_sha256(points[index])
                and math.isclose(
                    row["actual_fp32_l2_to_teacher"],
                    statistics["fp32_l2_to_teacher"],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                and math.isclose(
                    row["bf16_l2_to_teacher"],
                    statistics["bf16_l2_to_teacher"],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                and row["bf16_equal_fraction_to_teacher"]
                == statistics["bf16_equal_fraction_to_teacher"],
                "Terminal row recomputation drift.",
            )
        summary = summarize_scan(rows, config)
        classification = classify_outcome(summary, config)
        require(
            result.get("scan_summary") == summary and result.get("classification") == classification,
            "Terminal summary/classification drift.",
        )
        prerequisite = manifest.get("validation", {}).get("preflight_prerequisite", {})
        require(
            prerequisite.get("passed") is True
            and prerequisite.get("mode") == "technical-preflight"
            and prerequisite.get("git_commit") == result["git_commit"],
            "Terminal preflight chain drift.",
        )
    return {
        "passed": True,
        "mode": mode,
        "artifact_count": inventory["artifact_count"],
        "git_commit": result["git_commit"],
        "classification": result.get("classification"),
        "scan_summary": summary,
        "formal_success": False,
        "phase2_allowed": False,
    }
