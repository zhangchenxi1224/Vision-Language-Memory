"""Contracts and independent audit for the R11_new activation-precision control."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vision_memory.repro import canonical_tensor_sha256
from vision_memory.training import r11_new_oracle_terminal_capture as terminal_core


PROTOCOL = "R11-New-Activation-Precision-Control-Target01"
PREFIX = "vision_memory.r11-new-activation-precision-control"
CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs/experiments/r11_new_activation_precision_control_target01.json"
)
CONFIG_BYTES_SHA256 = "6fc1caf6e9760c295b25582a804584d6fa326137d48d1b6fcbd6e4146cf89cc5"
CONFIG_CANONICAL_SHA256 = "7d11a0f1bcc2b96bc71ba500410b9a60a092a84204584078558f98451f31c1c3"
CONDITIONS = ("bf16-baseline", "fp32-lifted")
PASSES = ("teacher-outward", "plateau-inward")
PATH_POINT_INDICES = (0, 8, 9, 11, 13, 15, 16)
GRADIENT_RADII = (0.0001, 0.001, 0.01, 0.1, 0.3, 1.0)


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
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_config() -> dict[str, Any]:
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "Precision config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(
        canonical_sha(value) == CONFIG_CANONICAL_SHA256,
        "Precision config canonical hash drift.",
    )
    require(
        value.get("schema") == f"{PREFIX}-config.v1" and value.get("protocol") == PROTOCOL,
        "Precision config identity drift.",
    )
    require(
        tuple(value["precision_conditions"]["order"]) == CONDITIONS
        and tuple(value["path_control"]["source_point_indices_from_terminal_contract"]) == PATH_POINT_INDICES
        and tuple(value["path_control"]["passes"]) == PASSES
        and tuple(value["gradient_control"]["signed_radii_l2"]) == GRADIENT_RADII,
        "Precision condition/path/radius drift.",
    )
    require(
        value["formal_technical_gate"]["total_full_chain_forward_calls"] == 54
        and value["formal_technical_gate"]["backward_calls"] == 2
        and value["formal_technical_gate"]["optimizer_steps"] == 0
        and value["interpretation_boundaries"]["formal_success_always_false"] is True
        and value["interpretation_boundaries"]["phase2_always_false"] is True,
        "Precision execution/boundary drift.",
    )
    return value


def unit_vector(value: torch.Tensor) -> torch.Tensor:
    require(
        isinstance(value, torch.Tensor)
        and value.dtype == torch.float32
        and value.numel() > 0
        and bool(torch.isfinite(value).all()),
        "Invalid precision direction tensor.",
    )
    norm = value.double().norm()
    require(float(norm) > 0.0 and bool(torch.isfinite(norm)), "Zero precision direction.")
    result = (value.double() / norm).float()
    return (result.double() / result.double().norm()).float()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    require(
        left.shape == right.shape and left.dtype == right.dtype == torch.float32,
        "Invalid precision cosine tensors.",
    )
    left_d = left.double().flatten()
    right_d = right.double().flatten()
    denominator = left_d.norm() * right_d.norm()
    require(float(denominator) > 0.0, "Zero precision cosine denominator.")
    return float(torch.dot(left_d, right_d) / denominator)


def path_row_id(condition: str, pass_name: str, source_point_index: int) -> str:
    require(
        condition in CONDITIONS and pass_name in PASSES and source_point_index in PATH_POINT_INDICES,
        "Invalid precision path-row identity.",
    )
    return f"{condition}__path__{pass_name}__p{source_point_index:02d}"


def gradient_row_id(condition: str, radius_index: int, sign: str) -> str:
    require(
        condition in CONDITIONS
        and type(radius_index) is int
        and 0 <= radius_index < len(GRADIENT_RADII)
        and sign in ("plus", "minus"),
        "Invalid precision gradient-row identity.",
    )
    return f"{condition}__negative-gradient__r{radius_index:02d}__{sign}"


def _validate_path_row(row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, str, int]:
    condition = row.get("condition")
    pass_name = row.get("pass")
    point_index = row.get("source_point_index")
    require(
        row.get("schema") == f"{PREFIX}-path-row.v1"
        and row.get("protocol") == PROTOCOL
        and condition in CONDITIONS
        and pass_name in PASSES
        and point_index in PATH_POINT_INDICES
        and row.get("row_id") == path_row_id(condition, pass_name, point_index),
        "Invalid precision path row.",
    )
    terminal_config = terminal_core.load_config()
    binding = terminal_config["terminal_path_contract"]["point_bindings"][point_index]
    numeric = (row.get("loss"), row.get("loss_ratio_to_plateau"))
    require(
        row.get("requested_remaining_l2") == binding["requested_remaining_l2"]
        and row.get("x_T_fp32_sha256") == binding["x_T_fp32_sha256"]
        and all(type(value) in (int, float) and math.isfinite(value) and value >= 0.0 for value in numeric)
        and isinstance(row.get("endpoint_fp32_sha256"), str)
        and len(row["endpoint_fp32_sha256"]) == 64
        and isinstance(row.get("endpoint_bitwise_equal_to_condition_teacher"), bool),
        "Precision path-row value drift.",
    )
    return condition, pass_name, point_index


def _validate_gradient_row(row: Mapping[str, Any]) -> tuple[str, int, str]:
    condition = row.get("condition")
    radius_index = row.get("radius_index")
    sign = row.get("sign")
    require(
        row.get("schema") == f"{PREFIX}-gradient-row.v1"
        and row.get("protocol") == PROTOCOL
        and condition in CONDITIONS
        and type(radius_index) is int
        and 0 <= radius_index < len(GRADIENT_RADII)
        and sign in ("plus", "minus")
        and row.get("row_id") == gradient_row_id(condition, radius_index, sign)
        and row.get("radius_l2") == GRADIENT_RADII[radius_index],
        "Invalid precision gradient row.",
    )
    numeric = (
        row.get("loss"),
        row.get("loss_ratio_to_plateau"),
        row.get("analytic_directional_derivative"),
    )
    require(
        all(type(value) in (int, float) and math.isfinite(value) for value in numeric)
        and numeric[0] >= 0.0
        and numeric[1] >= 0.0
        and isinstance(row.get("endpoint_fp32_sha256"), str)
        and len(row["endpoint_fp32_sha256"]) == 64
        and isinstance(row.get("scanned_x_T_fp32_sha256"), str)
        and len(row["scanned_x_T_fp32_sha256"]) == 64,
        "Precision gradient-row value drift.",
    )
    return condition, radius_index, sign


def summarize_condition(
    condition: str,
    path_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    require(condition in CONDITIONS and len(path_rows) == 14, "Precision path row count drift.")
    path_by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in path_rows:
        observed_condition, pass_name, point_index = _validate_path_row(row, config)
        key = (pass_name, point_index)
        require(observed_condition == condition and key not in path_by_key, "Precision path grouping drift.")
        path_by_key[key] = row
    require(
        set(path_by_key) == {(pass_name, index) for pass_name in PASSES for index in PATH_POINT_INDICES},
        "Precision path coverage drift.",
    )
    canonical_rows = []
    for point_index in PATH_POINT_INDICES:
        outward = path_by_key[("teacher-outward", point_index)]
        inward = path_by_key[("plateau-inward", point_index)]
        require(
            outward["endpoint_fp32_sha256"] == inward["endpoint_fp32_sha256"]
            and outward["loss"] == inward["loss"]
            and outward["loss_ratio_to_plateau"] == inward["loss_ratio_to_plateau"],
            "Precision duplicate path endpoint drift.",
        )
        canonical_rows.append(outward)
    require(
        canonical_rows[0]["loss"] == 0.0
        and canonical_rows[0]["loss_ratio_to_plateau"] == 0.0
        and canonical_rows[0]["endpoint_bitwise_equal_to_condition_teacher"] is True
        and math.isclose(canonical_rows[-1]["loss_ratio_to_plateau"], 1.0, rel_tol=1e-6, abs_tol=1e-12),
        "Precision teacher/plateau path boundary drift.",
    )
    strong_threshold = float(config["path_control"]["strong_capture_loss_ratio_lte"])
    strong = [row["loss_ratio_to_plateau"] <= strong_threshold for row in canonical_rows]
    prefix_end = 0
    for index, passed in enumerate(strong):
        if not passed:
            break
        prefix_end = index
    disconnected = any(strong[index] for index in range(prefix_end + 1, len(strong)))

    require(len(gradient_rows) == 12, "Precision gradient row count drift.")
    gradient_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for row in gradient_rows:
        observed_condition, radius_index, sign = _validate_gradient_row(row)
        key = (radius_index, sign)
        require(
            observed_condition == condition and key not in gradient_by_key,
            "Precision gradient grouping drift.",
        )
        gradient_by_key[key] = row
    require(
        set(gradient_by_key) == {(index, sign) for index in range(len(GRADIENT_RADII)) for sign in ("plus", "minus")},
        "Precision gradient coverage drift.",
    )
    plus_rows = [gradient_by_key[(index, "plus")] for index in range(len(GRADIENT_RADII))]
    best = min(plus_rows, key=lambda row: (row["loss_ratio_to_plateau"], row["radius_index"]))
    analytic = float(plus_rows[0]["analytic_directional_derivative"])
    require(
        analytic < 0.0 and all(row["analytic_directional_derivative"] == analytic for row in gradient_rows),
        "Precision analytic gradient drift.",
    )
    finite = []
    sign_agreements = 0
    for index, radius in enumerate(GRADIENT_RADII):
        plus = gradient_by_key[(index, "plus")]
        minus = gradient_by_key[(index, "minus")]
        derivative = (float(plus["loss"]) - float(minus["loss"])) / (2.0 * radius)
        finite.append(derivative)
        sign_agreements += int(derivative < 0.0)
    return {
        "condition": condition,
        "path": {
            "canonical_teacher_outward_rows": canonical_rows,
            "strong_capture_prefix_end_source_point_index": PATH_POINT_INDICES[prefix_end],
            "strong_capture_prefix_max_requested_remaining_l2": canonical_rows[prefix_end]["requested_remaining_l2"],
            "strong_capture_disconnected": disconnected,
            "all_duplicate_endpoints_bitwise_equal": True,
        },
        "gradient": {
            "analytic_directional_derivative": analytic,
            "best_positive_loss": float(best["loss"]),
            "best_positive_loss_ratio": float(best["loss_ratio_to_plateau"]),
            "best_positive_radius_l2": float(best["radius_l2"]),
            "actionable_descent": bool(
                best["loss_ratio_to_plateau"] <= config["gradient_control"]["actionable_descent_loss_ratio_lte"]
            ),
            "central_finite_difference": finite,
            "finite_difference_descent_sign_count": sign_agreements,
        },
    }


def summarize_experiment(
    path_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    require(len(path_rows) == 28 and len(gradient_rows) == 24, "Precision total row count drift.")
    conditions = {
        condition: summarize_condition(
            condition,
            [row for row in path_rows if row.get("condition") == condition],
            [row for row in gradient_rows if row.get("condition") == condition],
            config,
        )
        for condition in CONDITIONS
    }
    bf16_rows = conditions["bf16-baseline"]["path"]["canonical_teacher_outward_rows"]
    expected = config["path_control"]["bf16_parent_expected"]
    require(len(bf16_rows) == len(expected), "BF16 parent control row-count drift.")
    bf16_reproduced = all(
        row["source_point_index"] == binding["source_point_index"]
        and row["endpoint_fp32_sha256"] == binding["endpoint_fp32_sha256"]
        and math.isclose(
            row["loss_ratio_to_plateau"],
            binding["loss_ratio_to_plateau"],
            rel_tol=1e-6,
            abs_tol=1e-12,
        )
        for row, binding in zip(bf16_rows, expected, strict=True)
    )
    bf16_width = float(conditions["bf16-baseline"]["path"]["strong_capture_prefix_max_requested_remaining_l2"])
    fp32_width = float(conditions["fp32-lifted"]["path"]["strong_capture_prefix_max_requested_remaining_l2"])
    require(bf16_width > 0.0, "BF16 capture width drifted below the bound parent control.")
    widening = fp32_width / bf16_width
    capture_restored = bool(
        fp32_width >= config["path_control"]["fp32_capture_restored_min_remaining_l2"]
        and widening >= config["path_control"]["fp32_capture_widening_factor_gte"]
    )
    gradient_actionable = bool(conditions["fp32-lifted"]["gradient"]["actionable_descent"])
    return {
        "conditions": conditions,
        "bf16_parent_reproduced": bf16_reproduced,
        "bf16_capture_width": bf16_width,
        "fp32_capture_width": fp32_width,
        "fp32_to_bf16_capture_widening_factor": widening,
        "fp32_capture_restored": capture_restored,
        "fp32_gradient_actionable": gradient_actionable,
    }


def classify_outcome(summary: Mapping[str, Any]) -> str:
    if not summary["bf16_parent_reproduced"]:
        return "bf16_reproduction_failure"
    capture = bool(summary["fp32_capture_restored"])
    gradient = bool(summary["fp32_gradient_actionable"])
    if capture and gradient:
        return "fp32_restores_capture_and_gradient"
    if capture:
        return "fp32_widens_capture_without_actionable_gradient"
    if gradient:
        return "fp32_gradient_only"
    return "precision_lift_insufficient"


def validate_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory = json.loads((root / "artifact_inventory.json").read_text(encoding="utf-8"))
    require(inventory.get("schema") == f"{PREFIX}-inventory.v1", "Precision inventory schema drift.")
    listed: set[str] = set()
    for item in inventory.get("artifacts", []):
        relative = item.get("path")
        require(
            isinstance(relative, str)
            and relative not in listed
            and "\\" not in relative
            and not Path(relative).is_absolute()
            and all(part not in ("", ".", "..") for part in relative.split("/")),
            "Unsafe precision inventory path.",
        )
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root) and path.is_file() and not path.is_symlink(),
            "Escaping/missing precision artifact.",
        )
        require(
            path.stat().st_size == item.get("bytes") and sha256_file(path) == item.get("sha256"),
            "Precision artifact bytes/hash mismatch.",
        )
        listed.add(relative)
    observed = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "artifact_inventory.json"
    }
    require(
        observed == listed and inventory.get("artifact_count") == len(listed),
        "Incomplete precision inventory.",
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
        "precision_tensors.pt",
        "model_snapshot_verification_start.json",
        "model_snapshot_verification_end.json",
    }
    require(required.issubset(inventory["listed"]), "Required precision artifacts missing.")
    saved_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    require(saved_config == config and manifest.get("config_sha256") == canonical_sha(config), "Config drift.")
    require(
        manifest.get("protocol") == result.get("protocol") == terminal.get("protocol") == PROTOCOL
        and manifest.get("git_commit") == result.get("git_commit") == terminal.get("git_commit"),
        "Precision protocol/Git drift.",
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
        "Precision result boundary drift.",
    )
    start_snapshot = json.loads((root / "model_snapshot_verification_start.json").read_text(encoding="utf-8"))
    end_snapshot = json.loads((root / "model_snapshot_verification_end.json").read_text(encoding="utf-8"))
    require(
        start_snapshot == end_snapshot
        and result.get("snapshots_unchanged") is True
        and result.get("models_frozen") is True
        and result.get("only_student_x_T_trainable") is True,
        "Precision frozen/snapshot evidence drift.",
    )
    terminal_config = terminal_core.load_config()
    fixed = terminal_config["fixed_parent_artifacts"]
    target_path = root / "target/fixed_parent_target.pt"
    plateau_path = root / "parent/lr001-step256.pt"
    require(
        sha256_file(target_path) == fixed["source_target_artifact_sha256"]
        and sha256_file(plateau_path) == fixed["source_plateau_checkpoint_sha256"],
        "Copied precision parent bytes drift.",
    )
    target = torch.load(target_path, map_location="cpu", weights_only=True)
    plateau = torch.load(plateau_path, map_location="cpu", weights_only=True)
    all_points = terminal_core.construct_points(
        target["teacher_x_T_fp32"], plateau["student_x_T_fp32"], terminal_config
    )
    points = {index: all_points[index] for index in PATH_POINT_INDICES}
    bundle_path = root / "precision_tensors.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    require(
        bundle.get("schema") == f"{PREFIX}-tensor-bundle.v1"
        and bundle.get("protocol") == PROTOCOL
        and result.get("tensor_bundle_sha256") == sha256_file(bundle_path)
        and set(bundle.get("candidates", {})) == set(PATH_POINT_INDICES)
        and all(torch.equal(bundle["candidates"][index], point) for index, point in points.items()),
        "Precision tensor-bundle candidate drift.",
    )
    mode = result.get("mode")
    require(mode in ("technical-preflight", "formal") and terminal.get("mode") == mode, "Mode drift.")
    condition_tensors = bundle.get("conditions", {})
    require(set(condition_tensors) == set(CONDITIONS), "Precision condition tensor coverage drift.")
    require(set(result.get("conditions", {})) == set(CONDITIONS), "Precision condition result coverage drift.")
    for condition in CONDITIONS:
        record = condition_tensors[condition]
        teacher_endpoint = record.get("teacher_endpoint")
        plateau_endpoint = record.get("plateau_endpoint")
        gradient = record.get("gradient")
        direction = record.get("negative_gradient_direction")
        require(
            all(_valid_fp32_tensor(value) for value in (teacher_endpoint, plateau_endpoint, gradient, direction))
            and cosine(direction, -gradient) >= 1.0 - 1e-6
            and abs(float(direction.double().norm()) - 1.0) <= 1e-6,
            "Precision condition target/gradient tensor drift.",
        )
        if condition == "bf16-baseline":
            require(
                torch.equal(teacher_endpoint, target["teacher_endpoint_fp32"])
                and torch.equal(plateau_endpoint, plateau["endpoint_fp32"]),
                "BF16 precision control parent replay drift.",
            )
        baseline = float((plateau_endpoint - teacher_endpoint).square().mean())
        saved = result["conditions"][condition]
        nonzero_fraction = float((gradient != 0).double().mean())
        require(
            math.isclose(saved["plateau_mse"], baseline, rel_tol=1e-6, abs_tol=1e-12)
            and saved["teacher_endpoint_fp32_sha256"] == canonical_tensor_sha256(teacher_endpoint)
            and saved["plateau_endpoint_fp32_sha256"] == canonical_tensor_sha256(plateau_endpoint)
            and saved["gradient_fp32_sha256"] == canonical_tensor_sha256(gradient)
            and saved["direction_fp32_sha256"] == canonical_tensor_sha256(direction)
            and math.isclose(saved["gradient_norm"], float(gradient.double().norm()), rel_tol=1e-6, abs_tol=1e-12)
            and math.isclose(saved["gradient_nonzero_fraction"], nonzero_fraction, rel_tol=0.0, abs_tol=0.0)
            and nonzero_fraction >= config["gradient_control"]["gradient_nonzero_fraction_gte"],
            "Precision condition result binding drift.",
        )
    if mode == "technical-preflight":
        require(
            result.get("counters")
            == {
                "full_chain_forward_calls": 4,
                "reader_forward_calls": 0,
                "backward_calls": 2,
                "optimizer_steps": 0,
            }
            and result.get("preflight_gate") is True
            and all(
                not condition_tensors[condition].get("path_endpoints")
                and not condition_tensors[condition].get("gradient_scan_endpoints")
                for condition in CONDITIONS
            ),
            "Precision preflight execution drift.",
        )
        summary = None
    else:
        require(
            result.get("counters")
            == {
                "full_chain_forward_calls": 54,
                "reader_forward_calls": 0,
                "backward_calls": 2,
                "optimizer_steps": 0,
            }
            and {"path_metrics.jsonl", "gradient_metrics.jsonl"}.issubset(inventory["listed"]),
            "Precision formal execution drift.",
        )
        path_rows = [
            json.loads(line) for line in (root / "path_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        gradient_rows = [
            json.loads(line) for line in (root / "gradient_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        expected_path_ids = {
            path_row_id(condition, pass_name, point_index)
            for condition in CONDITIONS
            for pass_name in PASSES
            for point_index in PATH_POINT_INDICES
        }
        expected_gradient_ids = {
            gradient_row_id(condition, radius_index, sign)
            for condition in CONDITIONS
            for radius_index in range(len(GRADIENT_RADII))
            for sign in ("plus", "minus")
        }
        require(
            all(
                set(condition_tensors[condition].get("path_endpoints", {}))
                == {row_id for row_id in expected_path_ids if row_id.startswith(f"{condition}__")}
                for condition in CONDITIONS
            )
            and all(
                set(condition_tensors[condition].get("gradient_scan_endpoints", {}))
                == {row_id for row_id in expected_gradient_ids if row_id.startswith(f"{condition}__")}
                for condition in CONDITIONS
            ),
            "Precision formal endpoint coverage drift.",
        )
        for row in path_rows:
            condition, pass_name, point_index = _validate_path_row(row, config)
            record = condition_tensors[condition]
            endpoint = record["path_endpoints"].get(path_row_id(condition, pass_name, point_index))
            teacher_endpoint = record["teacher_endpoint"]
            baseline = float((record["plateau_endpoint"] - teacher_endpoint).square().mean())
            loss = float((endpoint - teacher_endpoint).square().mean())
            require(
                _valid_fp32_tensor(endpoint)
                and canonical_tensor_sha256(endpoint) == row["endpoint_fp32_sha256"]
                and math.isclose(loss, row["loss"], rel_tol=1e-6, abs_tol=1e-12)
                and math.isclose(loss / baseline, row["loss_ratio_to_plateau"], rel_tol=1e-6, abs_tol=1e-12)
                and row["endpoint_bitwise_equal_to_condition_teacher"] == torch.equal(endpoint, teacher_endpoint)
                and row["x_T_fp32_sha256"] == canonical_tensor_sha256(points[point_index]),
                "Precision path-row recomputation drift.",
            )
        for row in gradient_rows:
            condition, radius_index, sign_name = _validate_gradient_row(row)
            record = condition_tensors[condition]
            endpoint = record["gradient_scan_endpoints"].get(gradient_row_id(condition, radius_index, sign_name))
            teacher_endpoint = record["teacher_endpoint"]
            baseline = float((record["plateau_endpoint"] - teacher_endpoint).square().mean())
            sign = 1.0 if sign_name == "plus" else -1.0
            scanned = plateau["student_x_T_fp32"] + (
                sign * GRADIENT_RADII[radius_index] * record["negative_gradient_direction"]
            )
            loss = float((endpoint - teacher_endpoint).square().mean())
            analytic = float((record["gradient"].double() * record["negative_gradient_direction"].double()).sum())
            require(
                _valid_fp32_tensor(endpoint)
                and canonical_tensor_sha256(endpoint) == row["endpoint_fp32_sha256"]
                and canonical_tensor_sha256(scanned) == row["scanned_x_T_fp32_sha256"]
                and math.isclose(loss, row["loss"], rel_tol=1e-6, abs_tol=1e-12)
                and math.isclose(loss / baseline, row["loss_ratio_to_plateau"], rel_tol=1e-6, abs_tol=1e-12)
                and math.isclose(analytic, row["analytic_directional_derivative"], rel_tol=1e-6, abs_tol=1e-12),
                "Precision gradient-row recomputation drift.",
            )
        summary = summarize_experiment(path_rows, gradient_rows, config)
        classification = classify_outcome(summary)
        require(
            result.get("scan_summary") == summary and result.get("classification") == classification,
            "Precision summary/classification drift.",
        )
        prerequisite = manifest.get("validation", {}).get("preflight_prerequisite", {})
        require(
            prerequisite.get("passed") is True
            and prerequisite.get("mode") == "technical-preflight"
            and prerequisite.get("git_commit") == result["git_commit"],
            "Precision preflight chain drift.",
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
