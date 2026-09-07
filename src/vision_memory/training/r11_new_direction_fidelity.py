"""Locked contracts and independent audit for the R11_new local direction scan."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


PROTOCOL = "R11-New-Local-Direction-Fidelity-Target01"
PREFIX = "vision_memory.r11-new-local-direction-fidelity"
CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs/experiments/r11_new_local_direction_fidelity_target01.json"
CONFIG_BYTES_SHA256 = "1dd314776247683487df7ba17c1a04d86f079c19c20ce0d01db6a9ed363e0511"
CONFIG_CANONICAL_SHA256 = "5e42cd51bd1a35a59b51cbff8dc696539e33023d2dda53240eb47b3b6f243bbd"
ANCHORS = ("alpha099-start", "lr001-raw256")
DIRECTIONS = (
    "negative-autograd",
    "negative-adam-preconditioned",
    "teacher-residual",
    "deterministic-orthogonal-control",
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
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_config() -> dict[str, Any]:
    require(sha256_file(CONFIG_PATH) == CONFIG_BYTES_SHA256, "Direction-scan config byte hash drift.")
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    require(canonical_sha(value) == CONFIG_CANONICAL_SHA256, "Direction-scan config canonical hash drift.")
    require(value.get("protocol") == PROTOCOL and value.get("schema") == f"{PREFIX}-config.v1",
            "Direction-scan config identity drift.")
    fixed = value["fixed_target_and_anchor_contract"]
    scan = value["direction_scan_contract"]
    require(tuple(fixed["anchor_order"]) == ANCHORS
            and tuple(item["name"] for item in fixed["anchors"]) == ANCHORS,
            "Anchor order drift.")
    require(tuple(scan["directions"]) == DIRECTIONS
            and scan["signed_radii_l2"] == [0.0001, 0.001, 0.01, 0.1, 0.3, 1.0, 2.0],
            "Direction/radius grid drift.")
    require(value["formal_technical_gate"]["scan_rows"] == 112
            and value["formal_technical_gate"]["total_full_chain_forward_calls"] == 115,
            "Formal scan count drift.")
    require(value["interpretation_boundaries"]["formal_success_always_false"] is True
            and value["interpretation_boundaries"]["phase2_always_false"] is True,
            "Diagnostic boundary drift.")
    return value


def unit_vector(value: torch.Tensor) -> torch.Tensor:
    require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
            and value.numel() > 0 and bool(torch.isfinite(value).all()), "Invalid direction tensor.")
    norm = value.double().norm()
    require(bool(torch.isfinite(norm)) and float(norm) > 0.0, "Zero/non-finite direction norm.")
    result = (value.double() / norm).float()
    result = (result.double() / result.double().norm()).float()
    return result


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    require(left.shape == right.shape and left.dtype == right.dtype == torch.float32,
            "Invalid cosine tensors.")
    left_d = left.double().flatten()
    right_d = right.double().flatten()
    denominator = left_d.norm() * right_d.norm()
    require(float(denominator) > 0.0, "Zero cosine denominator.")
    return float(torch.dot(left_d, right_d) / denominator)


def orthogonal_control(gradient: torch.Tensor, residual: torch.Tensor, *, seed: int) -> torch.Tensor:
    require(gradient.shape == residual.shape and type(seed) is int, "Invalid orthogonal-control inputs.")
    gradient_unit = unit_vector(gradient).double().flatten()
    residual_d = residual.double().flatten()
    residual_orthogonal = residual_d - torch.dot(residual_d, gradient_unit) * gradient_unit
    require(float(residual_orthogonal.norm()) > 0.0, "Gradient and residual are collinear.")
    residual_unit = residual_orthogonal / residual_orthogonal.norm()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    random = torch.randn(gradient.numel(), generator=generator, dtype=torch.float64)
    random = random - torch.dot(random, gradient_unit) * gradient_unit
    random = random - torch.dot(random, residual_unit) * residual_unit
    require(float(random.norm()) > 0.0, "Degenerate orthogonal control.")
    return unit_vector((random / random.norm()).reshape_as(gradient).float())


def adam_preconditioned_direction(gradient: torch.Tensor, optimizer: Mapping[str, Any]) -> torch.Tensor:
    require(gradient.dtype == torch.float32 and bool(torch.isfinite(gradient).all()), "Invalid Adam gradient.")
    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    require(isinstance(groups, list) and len(groups) == 1 and isinstance(state, dict),
            "Invalid Adam state dictionary.")
    group = groups[0]
    betas = group.get("betas")
    require(isinstance(betas, (tuple, list)) and len(betas) == 2
            and group.get("amsgrad") is False and group.get("maximize") is False,
            "Unsupported Adam contract.")
    beta1, beta2 = (float(value) for value in betas)
    epsilon = float(group["eps"])
    if state:
        require(len(state) == 1, "Unexpected multi-parameter Adam state.")
        record = next(iter(state.values()))
        step_value = record["step"]
        step = int(step_value.item() if isinstance(step_value, torch.Tensor) else step_value)
        exp_avg = record["exp_avg"].detach().float().cpu()
        exp_avg_sq = record["exp_avg_sq"].detach().float().cpu()
        require(exp_avg.shape == gradient.shape == exp_avg_sq.shape and step == 256,
                "Plateau Adam state drift.")
    else:
        step = 0
        exp_avg = torch.zeros_like(gradient)
        exp_avg_sq = torch.zeros_like(gradient)
    next_step = step + 1
    next_avg = exp_avg.mul(beta1).add(gradient, alpha=1.0 - beta1)
    next_avg_sq = exp_avg_sq.mul(beta2).addcmul(gradient, gradient, value=1.0 - beta2)
    corrected_avg = next_avg / (1.0 - beta1**next_step)
    corrected_avg_sq = next_avg_sq / (1.0 - beta2**next_step)
    proposal = -corrected_avg / (corrected_avg_sq.sqrt() + epsilon)
    return unit_vector(proposal)


def quantization_statistics(value: torch.Tensor, teacher: torch.Tensor) -> dict[str, float]:
    require(value.shape == teacher.shape and value.dtype == teacher.dtype == torch.float32,
            "Invalid quantization tensors.")
    value_bf16 = value.to(torch.bfloat16)
    teacher_bf16 = teacher.to(torch.bfloat16)
    difference = value_bf16.float() - teacher_bf16.float()
    return {
        "bf16_equal_fraction_to_teacher": float((value_bf16 == teacher_bf16).double().mean()),
        "bf16_l2_to_teacher": float(difference.double().norm()),
        "fp32_l2_to_teacher": float((value - teacher).double().norm()),
    }


def scan_row_id(anchor: str, direction: str, radius_index: int, sign: str) -> str:
    require(anchor in ANCHORS and direction in DIRECTIONS and type(radius_index) is int
            and sign in ("plus", "minus"), "Invalid scan-row identity.")
    return f"{anchor}__{direction}__r{radius_index:02d}__{sign}"


def summarize_scan(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    radii = config["direction_scan_contract"]["signed_radii_l2"]
    expected = len(ANCHORS) * len(DIRECTIONS) * len(radii) * 2
    require(len(rows) == expected == 112, "Scan row count drift.")
    by_key: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (row.get("anchor"), row.get("direction"), row.get("radius_index"), row.get("sign"))
        require(row.get("schema") == f"{PREFIX}-scan-row.v1" and row.get("protocol") == PROTOCOL
                and key not in by_key
                and key[0] in ANCHORS and key[1] in DIRECTIONS
                and type(key[2]) is int and 0 <= key[2] < len(radii) and key[3] in ("plus", "minus")
                and row.get("row_id") == scan_row_id(*key)
                and row.get("radius_l2") == radii[key[2]], "Invalid or duplicate scan row.")
        numbers = (row.get("loss"), row.get("loss_ratio_to_anchor"), row.get("fp32_l2_to_teacher"),
                   row.get("bf16_l2_to_teacher"), row.get("bf16_equal_fraction_to_teacher"),
                   row.get("analytic_directional_derivative"))
        require(all(type(value) in (int, float) and math.isfinite(value) for value in numbers)
                and numbers[0] >= 0.0 and numbers[1] >= 0.0 and numbers[2] >= 0.0
                and numbers[3] >= 0.0 and 0.0 <= numbers[4] <= 1.0, "Non-finite scan row.")
        by_key[key] = row
    summary: dict[str, Any] = {}
    threshold = float(config["direction_scan_contract"]["meaningful_descent_loss_ratio_lte"])
    for anchor in ANCHORS:
        summary[anchor] = {}
        for direction in DIRECTIONS:
            plus_rows = [by_key[(anchor, direction, index, "plus")] for index in range(len(radii))]
            best = min(plus_rows, key=lambda row: (row["loss_ratio_to_anchor"], row["radius_index"]))
            derivatives = []
            sign_agreements = 0
            nonzero = 0
            analytic = float(plus_rows[0]["analytic_directional_derivative"])
            for index, radius in enumerate(radii):
                plus = by_key[(anchor, direction, index, "plus")]
                minus = by_key[(anchor, direction, index, "minus")]
                derivative = (float(plus["loss"]) - float(minus["loss"])) / (2.0 * float(radius))
                derivatives.append(derivative)
                if derivative != 0.0:
                    nonzero += 1
                    sign_agreements += int((derivative < 0.0) == (analytic < 0.0))
            summary[anchor][direction] = {
                "analytic_directional_derivative": analytic,
                "best_positive_loss": float(best["loss"]),
                "best_positive_loss_ratio": float(best["loss_ratio_to_anchor"]),
                "best_positive_radius_l2": float(best["radius_l2"]),
                "meaningful_descent": bool(best["loss_ratio_to_anchor"] <= threshold),
                "central_finite_difference": derivatives,
                "nonzero_finite_difference_count": nonzero,
                "finite_difference_sign_agreement_count": sign_agreements,
            }
    return summary


def classify_outcome(summary: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    plateau = summary["lr001-raw256"]
    negative_gradient = bool(plateau["negative-autograd"]["meaningful_descent"])
    adam = bool(plateau["negative-adam-preconditioned"]["meaningful_descent"])
    oracle_ratio = float(plateau["teacher-residual"]["best_positive_loss_ratio"])
    oracle_strong = oracle_ratio <= config["direction_scan_contract"]["strong_oracle_descent_loss_ratio_lte"]
    if not oracle_strong:
        return "plateau_oracle_not_strongly_descending"
    if negative_gradient and adam:
        return "plateau_negative_gradient_and_adam_descent"
    if negative_gradient:
        return "plateau_negative_gradient_only"
    if not adam:
        return "plateau_teacher_residual_only"
    return "adam_only_or_random_control_anomaly"


def validate_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    inventory = json.loads((root / "artifact_inventory.json").read_text(encoding="utf-8"))
    require(inventory.get("schema") == f"{PREFIX}-inventory.v1", "Inventory schema drift.")
    listed: set[str] = set()
    for item in inventory.get("artifacts", []):
        relative = item.get("path")
        require(isinstance(relative, str) and relative not in listed and "\\" not in relative
                and not Path(relative).is_absolute()
                and all(part not in ("", ".", "..") for part in relative.split("/")),
                "Unsafe or duplicate inventory path.")
        path = (root / relative).resolve()
        require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(), "Escaping/missing artifact.")
        require(path.stat().st_size == item.get("bytes") and sha256_file(path) == item.get("sha256"),
                "Artifact bytes/hash mismatch.")
        listed.add(relative)
    observed = {path.relative_to(root).as_posix() for path in root.rglob("*")
                if path.is_file() and path.name != "artifact_inventory.json"}
    require(observed == listed and inventory.get("artifact_count") == len(listed), "Incomplete inventory.")
    return {"artifact_count": len(listed), "listed": listed}


def _valid_fp32_tensor(value: Any) -> bool:
    return (isinstance(value, torch.Tensor) and value.dtype == torch.float32
            and tuple(value.shape) == (1, 4, 128, 128) and bool(torch.isfinite(value).all()))


def audit_delivery(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    from vision_memory.repro import canonical_tensor_sha256

    root = root.resolve()
    inventory = validate_inventory(root)
    required = {
        "config.json", "manifest.json", "result.json", "terminal.json", "runtime.json",
        "target/fixed_parent_target.pt", "parent/alpha099-step000.pt", "parent/lr001-step256.pt",
        "scan_tensors.pt", "model_snapshot_verification_start.json", "model_snapshot_verification_end.json",
    }
    require(required.issubset(inventory["listed"]), "Required direction-scan artifacts missing.")
    saved_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    require(saved_config == config and manifest.get("config_sha256") == canonical_sha(config), "Config drift.")
    require(manifest.get("protocol") == result.get("protocol") == terminal.get("protocol") == PROTOCOL
            and manifest.get("git_commit") == result.get("git_commit") == terminal.get("git_commit"),
            "Protocol/Git drift.")
    require(terminal.get("status") == "technical_completed" and terminal.get("engineering_gate") is True
            and terminal.get("exit_code") == 0 and terminal.get("formal_success") is False
            and terminal.get("phase2_allowed") is False and result.get("engineering_gate") is True
            and result.get("formal_success") is False and result.get("phase2_allowed") is False,
            "Terminal/result boundary drift.")
    require(result.get("models_frozen") is True and result.get("only_student_x_T_trainable") is True,
            "Frozen/trainable evidence missing.")
    start_snapshot = json.loads((root / "model_snapshot_verification_start.json").read_text(encoding="utf-8"))
    end_snapshot = json.loads((root / "model_snapshot_verification_end.json").read_text(encoding="utf-8"))
    require(start_snapshot == end_snapshot and result.get("snapshots_unchanged") is True,
            "Model snapshot drift.")
    parent = config["parent_low_lr_failure"]
    paths = {
        "target": (root / "target/fixed_parent_target.pt", parent["source_target_artifact_sha256"]),
        "alpha": (root / "parent/alpha099-step000.pt", parent["source_alpha099_checkpoint_sha256"]),
        "plateau": (root / "parent/lr001-step256.pt", parent["source_plateau_checkpoint_sha256"]),
    }
    for name, (path, expected) in paths.items():
        require(sha256_file(path) == expected, f"Copied parent {name} artifact drift.")
    target = torch.load(paths["target"][0], map_location="cpu", weights_only=True)
    alpha_checkpoint = torch.load(paths["alpha"][0], map_location="cpu", weights_only=True)
    plateau_checkpoint = torch.load(paths["plateau"][0], map_location="cpu", weights_only=True)
    require(target.get("tensor_sha256") == parent["target_tensor_sha256"], "Target tensor map drift.")
    require(alpha_checkpoint.get("tensor_sha256", {}).get("student_x_T_fp32")
            == parent["alpha099_tensor_sha256"]["student_x_T_fp32"]
            and alpha_checkpoint.get("tensor_sha256", {}).get("endpoint_fp32")
            == parent["alpha099_tensor_sha256"]["endpoint_fp32"], "Alpha anchor binding drift.")
    require(plateau_checkpoint.get("tensor_sha256", {}).get("student_x_T_fp32")
            == parent["plateau_tensor_sha256"]["student_x_T_fp32"]
            and plateau_checkpoint.get("tensor_sha256", {}).get("endpoint_fp32")
            == parent["plateau_tensor_sha256"]["endpoint_fp32"], "Plateau anchor binding drift.")
    bundle = torch.load(root / "scan_tensors.pt", map_location="cpu", weights_only=True)
    require(bundle.get("schema") == f"{PREFIX}-tensor-bundle.v1" and bundle.get("protocol") == PROTOCOL,
            "Tensor-bundle identity drift.")
    require(result.get("tensor_bundle_sha256") == sha256_file(root / "scan_tensors.pt"),
            "Tensor-bundle result binding drift.")
    expected_anchors = {
        "alpha099-start": alpha_checkpoint["student_x_T_fp32"],
        "lr001-raw256": plateau_checkpoint["student_x_T_fp32"],
    }
    expected_baselines = {
        "alpha099-start": alpha_checkpoint["endpoint_fp32"],
        "lr001-raw256": plateau_checkpoint["endpoint_fp32"],
    }
    for anchor in ANCHORS:
        anchor_tensor = bundle["anchors"][anchor]
        gradient = bundle["gradients"][anchor]
        baseline = bundle["baseline_endpoints"][anchor]
        require(_valid_fp32_tensor(anchor_tensor) and _valid_fp32_tensor(gradient)
                and _valid_fp32_tensor(baseline) and torch.equal(anchor_tensor, expected_anchors[anchor])
                and torch.equal(baseline, expected_baselines[anchor]), "Anchor/baseline replay drift.")
        residual = target["teacher_x_T_fp32"] - anchor_tensor
        for direction in DIRECTIONS:
            vector = bundle["directions"][anchor][direction]
            require(_valid_fp32_tensor(vector)
                    and abs(float(vector.double().norm()) - 1.0)
                    <= config["technical_preflight_gate"]["all_direction_norms_abs_error_lte"],
                    "Invalid saved unit direction.")
            if direction == "negative-autograd":
                require(cosine(vector, -gradient) >= 1.0 - 1e-6, "Negative-gradient direction drift.")
            elif direction == "negative-adam-preconditioned":
                checkpoint = alpha_checkpoint if anchor == "alpha099-start" else plateau_checkpoint
                expected = adam_preconditioned_direction(gradient, checkpoint["optimizer"])
                require(cosine(vector, expected) >= 1.0 - 1e-6, "Adam direction drift.")
            elif direction == "teacher-residual":
                require(cosine(vector, residual) >= 1.0 - 1e-6, "Teacher-residual direction drift.")
            elif direction == "deterministic-orthogonal-control":
                require(abs(cosine(vector, gradient))
                        <= config["technical_preflight_gate"]["orthogonal_control_abs_cosine_to_gradient_lte"]
                        and abs(cosine(vector, residual))
                        <= config["technical_preflight_gate"]["orthogonal_control_abs_cosine_to_teacher_residual_lte"],
                        "Orthogonal-control direction drift.")
        record = result["anchors"][anchor]
        baseline_mse = float((baseline - target["teacher_endpoint_fp32"]).square().mean())
        require(math.isclose(record["baseline_mse"], baseline_mse, rel_tol=1e-6, abs_tol=1e-10)
                and record["gradient_fp32_sha256"] == canonical_tensor_sha256(gradient),
                "Anchor result drift.")
    mode = result.get("mode")
    require(mode in ("technical-preflight", "formal") and terminal.get("mode") == mode, "Mode drift.")
    if mode == "technical-preflight":
        require(result.get("counters") == {"full_chain_forward_calls": 3, "reader_forward_calls": 0,
                                           "backward_calls": 2, "optimizer_steps": 0}
                and result.get("preflight_gate") is True and not bundle["scan_endpoints"],
                "Preflight execution drift.")
        summary = None
    else:
        require(result.get("counters") == {"full_chain_forward_calls": 115, "reader_forward_calls": 0,
                                           "backward_calls": 2, "optimizer_steps": 0},
                "Formal execution drift.")
        rows = [json.loads(line) for line in (root / "scan_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        require("scan_metrics.jsonl" in inventory["listed"]
                and set(bundle["scan_endpoints"]) == {row.get("row_id") for row in rows}
                and len(bundle["scan_endpoints"]) == 112,
                "Formal scan metric/endpoint coverage drift.")
        teacher_endpoint = target["teacher_endpoint_fp32"]
        teacher_x_t = target["teacher_x_T_fp32"]
        for row in rows:
            endpoint = bundle["scan_endpoints"].get(row["row_id"])
            require(_valid_fp32_tensor(endpoint)
                    and canonical_tensor_sha256(endpoint) == row["endpoint_fp32_sha256"],
                    "Scan endpoint tensor/hash drift.")
            anchor_tensor = bundle["anchors"][row["anchor"]]
            direction = bundle["directions"][row["anchor"]][row["direction"]]
            sign = 1.0 if row["sign"] == "plus" else -1.0
            scanned_x_t = anchor_tensor + sign * float(row["radius_l2"]) * direction
            loss = float((endpoint - teacher_endpoint).square().mean())
            quantized = quantization_statistics(scanned_x_t, teacher_x_t)
            analytic = float((bundle["gradients"][row["anchor"]].double()
                              * direction.double()).sum())
            require(math.isclose(loss, row["loss"], rel_tol=1e-6, abs_tol=1e-10)
                    and math.isclose(row["loss_ratio_to_anchor"],
                                     loss / result["anchors"][row["anchor"]]["baseline_mse"],
                                     rel_tol=1e-6, abs_tol=1e-10)
                    and math.isclose(row["analytic_directional_derivative"], analytic,
                                     rel_tol=1e-6, abs_tol=1e-10)
                    and all(math.isclose(row[name], quantized[name], rel_tol=1e-6, abs_tol=1e-10)
                            for name in quantized), "Scan-row recomputation drift.")
        summary = summarize_scan(rows, config)
        classification = classify_outcome(summary, config)
        require(result.get("scan_summary") == summary and result.get("classification") == classification,
                "Scan summary/classification drift.")
        prerequisite = manifest.get("validation", {}).get("preflight_prerequisite", {})
        require(prerequisite.get("passed") is True and prerequisite.get("mode") == "technical-preflight"
                and prerequisite.get("git_commit") == result["git_commit"], "Preflight chain drift.")
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
