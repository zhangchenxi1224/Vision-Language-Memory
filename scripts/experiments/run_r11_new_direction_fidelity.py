"""Run or independently audit the preregistered R11_new local direction scan."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.experiments import run_r11_new_reachable_lr_control as low_lr_runner  # noqa: E402
from scripts.train import r11_new_frozen_dreamlite_oracle as phase1a  # noqa: E402
from scripts.train import r11_new_identity_condition_binding as condition_binding  # noqa: E402
from scripts.train import r11_new_identity_condition_bridge as identity  # noqa: E402
from vision_memory.repro import canonical_tensor_sha256, configure_strict_cuda_determinism  # noqa: E402
from vision_memory.training import r11_new_direction_fidelity as core  # noqa: E402
from vision_memory.training import r11_new_reachable_lr_control as low_lr_core  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("technical-preflight", "formal", "audit"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit")
    parser.add_argument("--preflight-root", type=Path)
    return parser.parse_args(argv)


def validate_environment(args: argparse.Namespace, config: Mapping[str, Any]) -> dict[str, Any]:
    base_config = low_lr_core.load_config()
    for key in ("data", "models", "model_snapshots", "environment", "deployment"):
        core.require(config[key] == base_config[key], f"Fixed runtime contract drift: {key}.")
    base_args = argparse.Namespace(**vars(args))
    base_args.mode = "technical-preflight"
    base_args.preflight_root = None
    base_validation = low_lr_runner.validate_environment(base_args, base_config)
    parent = config["parent_low_lr_failure"]
    parent_root = Path(parent["source_root"])
    archive = parent_root.parent / "r11-new-lr-af00d04-20260907-round02.tar.gz"
    bindings = {
        "target": (Path(parent["source_target_path"]), parent["source_target_artifact_sha256"]),
        "alpha_checkpoint": (
            Path(parent["source_alpha099_checkpoint_path"]), parent["source_alpha099_checkpoint_sha256"]
        ),
        "plateau_checkpoint": (
            Path(parent["source_plateau_checkpoint_path"]), parent["source_plateau_checkpoint_sha256"]
        ),
        "result": (parent_root / "result.json", parent["source_result_sha256"]),
        "manifest": (parent_root / "manifest.json", parent["source_manifest_sha256"]),
        "terminal": (parent_root / "terminal.json", parent["source_terminal_sha256"]),
        "inventory": (parent_root / "artifact_inventory.json", parent["source_inventory_sha256"]),
        "archive": (archive, parent["source_archive_sha256"]),
    }
    for name, (path, expected_hash) in bindings.items():
        core.require(path.is_file() and core.sha256_file(path) == expected_hash,
                     f"Parent low-LR {name} binding drift.")
    prerequisite = None
    if args.mode == "technical-preflight":
        core.require(args.preflight_root is None, "Preflight cannot consume another preflight.")
    else:
        core.require(args.preflight_root is not None and args.preflight_root.resolve() != args.output_root.resolve(),
                     "Formal requires a distinct preflight root.")
        prerequisite = core.audit_delivery(args.preflight_root, config)
        core.require(prerequisite["mode"] == "technical-preflight"
                     and prerequisite["git_commit"] == args.expected_commit,
                     "Wrong direction-scan preflight prerequisite.")
        prerequisite = {
            **prerequisite,
            "root": str(args.preflight_root.resolve()),
            "terminal_sha256": core.sha256_file(args.preflight_root / "terminal.json"),
            "inventory_sha256": core.sha256_file(args.preflight_root / "artifact_inventory.json"),
            "tensor_bundle_sha256": core.sha256_file(args.preflight_root / "scan_tensors.pt"),
        }
    return {
        **base_validation,
        "parent_low_lr_bindings": {
            name: {"path": str(path), "sha256": expected_hash}
            for name, (path, expected_hash) in bindings.items()
        },
        "preflight_prerequisite": prerequisite,
    }


def copy_parent_artifacts(output_root: Path, config: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    parent = config["parent_low_lr_failure"]
    paths = {
        "target/fixed_parent_target.pt": (
            Path(parent["source_target_path"]), parent["source_target_artifact_sha256"]
        ),
        "parent/alpha099-step000.pt": (
            Path(parent["source_alpha099_checkpoint_path"]), parent["source_alpha099_checkpoint_sha256"]
        ),
        "parent/lr001-step256.pt": (
            Path(parent["source_plateau_checkpoint_path"]), parent["source_plateau_checkpoint_sha256"]
        ),
    }
    for relative, (source, expected_hash) in paths.items():
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        core.require(core.sha256_file(destination) == expected_hash, f"Copied parent artifact drift: {relative}.")
    target = torch.load(output_root / "target/fixed_parent_target.pt", map_location="cpu", weights_only=True)
    alpha_checkpoint = torch.load(output_root / "parent/alpha099-step000.pt", map_location="cpu", weights_only=True)
    plateau_checkpoint = torch.load(output_root / "parent/lr001-step256.pt", map_location="cpu", weights_only=True)
    core.require(target.get("tensor_sha256") == parent["target_tensor_sha256"], "Target tensor map drift.")
    core.require(alpha_checkpoint["tensor_sha256"]["student_x_T_fp32"]
                 == parent["alpha099_tensor_sha256"]["student_x_T_fp32"]
                 and alpha_checkpoint["tensor_sha256"]["endpoint_fp32"]
                 == parent["alpha099_tensor_sha256"]["endpoint_fp32"], "Alpha checkpoint tensor drift.")
    core.require(plateau_checkpoint["tensor_sha256"]["student_x_T_fp32"]
                 == parent["plateau_tensor_sha256"]["student_x_T_fp32"]
                 and plateau_checkpoint["tensor_sha256"]["endpoint_fp32"]
                 == parent["plateau_tensor_sha256"]["endpoint_fp32"], "Plateau checkpoint tensor drift.")
    return target, alpha_checkpoint, plateau_checkpoint


def load_runtime(args: argparse.Namespace, config: Mapping[str, Any], counters: dict[str, int]):
    identity_config = identity._load_config()
    load_args = low_lr_runner.runtime_args(args.output_root, config)
    processor, pipe, reader, oracle, source_latents, context = condition_binding.load_runtime(
        load_args, identity_config
    )
    del processor, pipe
    source_only_x_t, initialization = identity._source_only_initialization(
        source_latents=source_latents,
        scheduler=oracle.sampler,
        compute_device=oracle.source_latents.device,
        output_dir=args.output_root,
    )
    target, alpha_checkpoint, plateau_checkpoint = copy_parent_artifacts(args.output_root, config)
    core.require(torch.equal(source_only_x_t.detach().float().cpu(), target["source_only_x_T_init_fp32"]),
                 "Current source-only initialization differs from parent target.")

    def forward(*, gradient: bool) -> phase1a.OracleForward:
        counters["full_chain_forward_calls"] += 1
        if gradient:
            return oracle()
        with torch.no_grad():
            return oracle()

    teacher_x_t = target["teacher_x_T_fp32"]
    with torch.no_grad():
        oracle.x_T_fp32.copy_(teacher_x_t.to(oracle.x_T_fp32.device))
        oracle.initial_x_T_fp32.copy_(teacher_x_t.to(oracle.initial_x_T_fp32.device))
    replay = forward(gradient=False)
    replay_endpoint = replay.z_t.detach().float().cpu()
    core.require(torch.equal(replay_endpoint, target["teacher_endpoint_fp32"]),
                 "Teacher endpoint does not replay bitwise.")
    phase1a._save_image(args.output_root / "target/fixed_parent_target.png", replay.image)
    target_record = {
        "artifact_sha256": core.sha256_file(args.output_root / "target/fixed_parent_target.pt"),
        "tensor_sha256": dict(target["tensor_sha256"]),
        "teacher_endpoint_replay_bitwise_equal": True,
        "source_parent_root": config["parent_low_lr_failure"]["source_root"],
    }
    return (reader, oracle, source_latents, context, target, alpha_checkpoint, plateau_checkpoint,
            target_record, initialization, forward)


def prepare_anchor(*, name: str, checkpoint: Mapping[str, Any], target: Mapping[str, torch.Tensor],
                   oracle: phase1a.FrozenDreamLiteOracle, reader: torch.nn.Module,
                   counters: dict[str, int], forward: Any, output_root: Path,
                   config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    anchor = checkpoint["student_x_T_fp32"].detach().float().cpu()
    expected_endpoint = checkpoint["endpoint_fp32"].detach().float().cpu()
    with torch.no_grad():
        oracle.x_T_fp32.copy_(anchor.to(oracle.x_T_fp32.device))
        oracle.initial_x_T_fp32.copy_(anchor.to(oracle.initial_x_T_fp32.device))
    oracle.x_T_fp32.grad = None
    output = forward(gradient=True)
    endpoint = output.z_t.detach().float().cpu()
    core.require(torch.equal(endpoint, expected_endpoint), f"Anchor endpoint replay drift: {name}.")
    teacher_endpoint = target["teacher_endpoint_fp32"].to(output.z_t.device)
    loss = (output.z_t.float() - teacher_endpoint).square().mean()
    core.require(loss.numel() == 1 and bool(torch.isfinite(loss)), "Invalid anchor loss.")
    loss.backward()
    counters["backward_calls"] += 1
    gradient = oracle.x_T_fp32.grad
    core.require(gradient is not None and bool(torch.isfinite(gradient).all()), "Invalid anchor gradient.")
    gradient_cpu = gradient.detach().float().cpu()
    core.require(float(gradient_cpu.double().norm()) > 0.0
                 and float((gradient_cpu != 0).double().mean())
                 >= config["technical_preflight_gate"]["gradient_nonzero_fraction_gte"],
                 "Anchor gradient gate failed.")
    oracle.x_T_fp32.grad = None
    residual = target["teacher_x_T_fp32"] - anchor
    directions = {
        "negative-autograd": core.unit_vector(-gradient_cpu),
        "negative-adam-preconditioned": core.adam_preconditioned_direction(
            gradient_cpu, checkpoint["optimizer"]
        ),
        "teacher-residual": core.unit_vector(residual),
        "deterministic-orthogonal-control": core.orthogonal_control(
            gradient_cpu, residual, seed=config["direction_scan_contract"]["orthogonal_control_seed"]
        ),
    }
    phase1a._save_image(output_root / f"anchors/{name}.png", output.image)
    direction_records = {}
    for direction_name, direction in directions.items():
        direction_records[direction_name] = {
            "l2_norm": float(direction.double().norm()),
            "cosine_to_gradient": core.cosine(direction, gradient_cpu),
            "cosine_to_teacher_residual": core.cosine(direction, residual),
            "tensor_sha256": canonical_tensor_sha256(direction),
        }
    record = {
        "checkpoint_sha256": core.sha256_file(
            output_root / ("parent/alpha099-step000.pt" if name == "alpha099-start"
                           else "parent/lr001-step256.pt")
        ),
        "anchor_x_T_fp32_sha256": canonical_tensor_sha256(anchor),
        "baseline_endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
        "baseline_mse": float(loss.detach()),
        "gradient_norm": float(gradient_cpu.double().norm()),
        "gradient_nonzero_fraction": float((gradient_cpu != 0).double().mean()),
        "gradient_fp32_sha256": canonical_tensor_sha256(gradient_cpu),
        "quantization": core.quantization_statistics(anchor, target["teacher_x_T_fp32"]),
        "directions": direction_records,
    }
    tensors = {"anchor": anchor, "gradient": gradient_cpu, "baseline_endpoint": endpoint,
               "directions": directions}
    core.require(low_lr_runner.models_frozen(oracle, reader)
                 and low_lr_runner.only_student_trainable(oracle, reader),
                 "Frozen/trainable contract failed during anchor preparation.")
    return record, tensors


def execute_scan(*, args: argparse.Namespace, config: Mapping[str, Any],
                 oracle: phase1a.FrozenDreamLiteOracle, target: Mapping[str, torch.Tensor],
                 anchor_tensors: Mapping[str, Mapping[str, Any]], counters: dict[str, int],
                 forward: Any) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor]]:
    rows: list[dict[str, Any]] = []
    endpoints: dict[str, torch.Tensor] = {}
    metrics_path = args.output_root / "scan_metrics.jsonl"
    radii = config["direction_scan_contract"]["signed_radii_l2"]
    teacher_endpoint = target["teacher_endpoint_fp32"]
    teacher_x_t = target["teacher_x_T_fp32"]
    with metrics_path.open("x", encoding="utf-8", newline="\n") as metrics:
        for anchor_name in core.ANCHORS:
            anchor = anchor_tensors[anchor_name]["anchor"]
            baseline_mse = float((anchor_tensors[anchor_name]["baseline_endpoint"] - teacher_endpoint).square().mean())
            gradient = anchor_tensors[anchor_name]["gradient"]
            for direction_name in core.DIRECTIONS:
                direction = anchor_tensors[anchor_name]["directions"][direction_name]
                analytic = float((gradient.double() * direction.double()).sum())
                for radius_index, radius in enumerate(radii):
                    for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                        row_id = core.scan_row_id(anchor_name, direction_name, radius_index, sign_name)
                        scanned_x_t = anchor + sign * float(radius) * direction
                        with torch.no_grad():
                            oracle.x_T_fp32.copy_(scanned_x_t.to(oracle.x_T_fp32.device))
                            oracle.initial_x_T_fp32.copy_(scanned_x_t.to(oracle.initial_x_T_fp32.device))
                        output = forward(gradient=False)
                        endpoint = output.z_t.detach().float().cpu()
                        loss = float((endpoint - teacher_endpoint).square().mean())
                        quantization = core.quantization_statistics(scanned_x_t, teacher_x_t)
                        row = {
                            "schema": f"{core.PREFIX}-scan-row.v1",
                            "protocol": core.PROTOCOL,
                            "row_id": row_id,
                            "anchor": anchor_name,
                            "direction": direction_name,
                            "radius_index": radius_index,
                            "radius_l2": radius,
                            "sign": sign_name,
                            "loss": loss,
                            "loss_ratio_to_anchor": loss / baseline_mse,
                            "analytic_directional_derivative": analytic,
                            "endpoint_fp32_sha256": canonical_tensor_sha256(endpoint),
                            **quantization,
                        }
                        endpoints[row_id] = endpoint
                        rows.append(row)
                        metrics.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                        metrics.flush()
                        os.fsync(metrics.fileno())
                        phase1a._save_image(args.output_root / f"scan_images/{row_id}.png", output.image)
                        print(f"scan {len(rows)}/112 {row_id} ratio={row['loss_ratio_to_anchor']:.10g}", flush=True)
    return rows, endpoints


def write_report(root: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# R11_new local direction/Jacobian fidelity scan",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Engineering gate: `{result['engineering_gate']}`",
        f"- Counters: `{json.dumps(result['counters'], sort_keys=True)}`",
    ]
    if result["mode"] == "formal":
        lines.append(f"- Classification: `{result['classification']}`")
        plateau = result["scan_summary"]["lr001-raw256"]
        for direction in core.DIRECTIONS:
            lines.append(
                f"- plateau {direction} best + loss ratio: "
                f"`{plateau[direction]['best_positive_loss_ratio']:.10g}`"
            )
    lines += ["", "This is a fixed-target geometric diagnostic with oracle information.",
              "Formal Picture Memory success and Phase 2 remain false."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_experiment(args: argparse.Namespace, config: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    phase1a._atomic_json(args.output_root / "config.json", config)
    phase1a._write_environment(args.output_root / "environment.txt")
    runtime = {**phase1a._runtime_versions(), "python_executable": str(Path(sys.executable).resolve())}
    phase1a._atomic_json(args.output_root / "runtime.json", runtime)
    phase1a._atomic_json(args.output_root / "determinism.json", configure_strict_cuda_determinism(0))
    snapshots_start = {name: phase1a.verify_snapshot_binding(value)
                       for name, value in config["model_snapshots"].items()}
    phase1a._atomic_json(args.output_root / "model_snapshot_verification_start.json", snapshots_start)
    counters = {"full_chain_forward_calls": 0, "reader_forward_calls": 0,
                "backward_calls": 0, "optimizer_steps": 0}
    started = time.monotonic()
    (reader, oracle, source_latents, context, target, alpha_checkpoint, plateau_checkpoint,
     target_record, initialization, forward) = load_runtime(args, config, counters)

    def forbidden_reader(_module: Any, _inputs: Any) -> None:
        counters["reader_forward_calls"] += 1
        raise RuntimeError("Reader is forbidden in the local direction scan.")

    reader_hook = reader.model.register_forward_pre_hook(forbidden_reader)
    try:
        core.require(low_lr_runner.models_frozen(oracle, reader)
                     and low_lr_runner.only_student_trainable(oracle, reader),
                     "Initial frozen/trainable contract failed.")
        manifest = {
            "schema": f"{core.PREFIX}-manifest.v1",
            "protocol": core.PROTOCOL,
            "mode": args.mode,
            "git_commit": args.expected_commit,
            "config_sha256": core.canonical_sha(config),
            "validation": dict(validation),
            "target": target_record,
            "initialization": initialization,
            "target_segment_id": context["target"].segment_id,
            "anchor_order": list(core.ANCHORS),
            "direction_order": list(core.DIRECTIONS),
            "information_boundary": {
                "target_source": "exact parent low-LR target artifact",
                "plateau_source": "exact parent lr-001 raw256 checkpoint",
                "teacher_x_T_exposed_to_oracle_direction": True,
                "teacher_residual_deployable": False,
                "optimizer_steps": 0,
                "reader_used": False,
            },
            "formal_success": False,
            "phase2_allowed": False,
        }
        phase1a._atomic_json(args.output_root / "manifest.json", manifest)
        checkpoints = {"alpha099-start": alpha_checkpoint, "lr001-raw256": plateau_checkpoint}
        anchor_records = {}
        anchor_tensors = {}
        for anchor_name in core.ANCHORS:
            anchor_records[anchor_name], anchor_tensors[anchor_name] = prepare_anchor(
                name=anchor_name, checkpoint=checkpoints[anchor_name], target=target,
                oracle=oracle, reader=reader, counters=counters, forward=forward,
                output_root=args.output_root, config=config
            )
        rows: list[dict[str, Any]] = []
        endpoints: dict[str, torch.Tensor] = {}
        if args.mode == "formal":
            rows, endpoints = execute_scan(
                args=args, config=config, oracle=oracle, target=target,
                anchor_tensors=anchor_tensors, counters=counters, forward=forward
            )
        tensor_bundle = {
            "schema": f"{core.PREFIX}-tensor-bundle.v1",
            "protocol": core.PROTOCOL,
            "anchors": {name: values["anchor"] for name, values in anchor_tensors.items()},
            "gradients": {name: values["gradient"] for name, values in anchor_tensors.items()},
            "baseline_endpoints": {
                name: values["baseline_endpoint"] for name, values in anchor_tensors.items()
            },
            "directions": {name: values["directions"] for name, values in anchor_tensors.items()},
            "scan_endpoints": endpoints,
        }
        phase1a._atomic_torch_save(args.output_root / "scan_tensors.pt", tensor_bundle)
        snapshots_end = {name: phase1a.verify_snapshot_binding(value)
                         for name, value in config["model_snapshots"].items()}
        phase1a._atomic_json(args.output_root / "model_snapshot_verification_end.json", snapshots_end)
        if args.mode == "technical-preflight":
            expected_counters = {"full_chain_forward_calls": 3, "reader_forward_calls": 0,
                                 "backward_calls": 2, "optimizer_steps": 0}
            direction_gate = all(
                abs(record["l2_norm"] - 1.0)
                <= config["technical_preflight_gate"]["all_direction_norms_abs_error_lte"]
                for anchor in anchor_records.values() for record in anchor["directions"].values()
            )
            orthogonal_gate = all(
                abs(anchor["directions"]["deterministic-orthogonal-control"][key])
                <= config["technical_preflight_gate"][threshold]
                for anchor in anchor_records.values()
                for key, threshold in (
                    ("cosine_to_gradient", "orthogonal_control_abs_cosine_to_gradient_lte"),
                    ("cosine_to_teacher_residual", "orthogonal_control_abs_cosine_to_teacher_residual_lte"),
                )
            )
            engineering_gate = bool(counters == expected_counters and snapshots_end == snapshots_start
                                    and direction_gate and orthogonal_gate
                                    and low_lr_runner.models_frozen(oracle, reader)
                                    and low_lr_runner.only_student_trainable(oracle, reader))
            core.require(engineering_gate, "Technical preflight gate failed.")
            scan_summary = None
            classification = None
        else:
            expected_counters = {"full_chain_forward_calls": 115, "reader_forward_calls": 0,
                                 "backward_calls": 2, "optimizer_steps": 0}
            engineering_gate = bool(counters == expected_counters and len(rows) == 112
                                    and len(endpoints) == 112 and snapshots_end == snapshots_start
                                    and low_lr_runner.models_frozen(oracle, reader)
                                    and low_lr_runner.only_student_trainable(oracle, reader))
            core.require(engineering_gate, "Formal technical gate failed.")
            scan_summary = core.summarize_scan(rows, config)
            classification = core.classify_outcome(scan_summary, config)
        result = {
            "schema": f"{core.PREFIX}-result.v1",
            "protocol": core.PROTOCOL,
            "mode": args.mode,
            "git_commit": args.expected_commit,
            "engineering_gate": engineering_gate,
            "preflight_gate": engineering_gate if args.mode == "technical-preflight" else None,
            "anchors": anchor_records,
            "scan_summary": scan_summary,
            "classification": classification,
            "counters": counters,
            "target_record": target_record,
            "tensor_bundle_sha256": core.sha256_file(args.output_root / "scan_tensors.pt"),
            "models_frozen": True,
            "only_student_x_T_trainable": True,
            "snapshots_unchanged": snapshots_end == snapshots_start,
            "formal_success": False,
            "phase2_allowed": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        phase1a._atomic_json(args.output_root / "result.json", result)
        write_report(args.output_root, result)
        return result
    finally:
        reader_hook.remove()
        del source_latents


def artifact_inventory(root: Path) -> dict[str, Any]:
    artifacts = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
         "sha256": core.sha256_file(path)}
        for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_inventory.json"
    ]
    return {"schema": f"{core.PREFIX}-inventory.v1", "artifact_count": len(artifacts), "artifacts": artifacts}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = core.load_config()
    if args.mode == "audit":
        print(json.dumps(core.audit_delivery(args.output_root, config), ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    from scripts.inspire import run_r11_new_identity_condition_bridge as safety

    validation = validate_environment(args, config)
    lock_path = Path("/tmp/vision-memory-r11-new-direction-fidelity.lock")
    lock_path.mkdir(exist_ok=False)
    owner = {"owner_token": uuid.uuid4().hex, "pid": os.getpid(), "mode": args.mode,
             "git_commit": args.expected_commit, "output_root": str(args.output_root),
             "started_at_utc": safety._utc_now()}
    phase1a._atomic_json(lock_path / "owner.json", owner)
    lock = {"path": str(lock_path), "owner_path": str(lock_path / "owner.json"), "owner": owner,
            "owner_sha256": core.sha256_file(lock_path / "owner.json")}
    claimed = False
    terminal = {"schema": f"{core.PREFIX}-terminal.v1", "protocol": core.PROTOCOL, "mode": args.mode,
                "status": "technical_failed", "engineering_gate": False, "exit_code": 2,
                "git_commit": args.expected_commit, "formal_success": False, "phase2_allowed": False,
                "started_at_utc": safety._utc_now()}
    try:
        args.output_root.mkdir(parents=True, exist_ok=False)
        claimed = True
        phase1a._atomic_json(args.output_root / "launch.json", {"owner": owner, "validation": validation})
        with (args.output_root / "stdout.log").open("x", encoding="utf-8") as stdout, (
            args.output_root / "stderr.log"
        ).open("x", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                result = run_experiment(args, config, validation)
            except Exception:
                traceback.print_exc()
                raise
        terminal.update(status="technical_completed", engineering_gate=True, exit_code=0,
                        result_sha256=core.sha256_file(args.output_root / "result.json"),
                        classification=result.get("classification"))
    except Exception as error:
        terminal["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            terminal["lock_release"] = safety._release_lock(lock)
        except Exception as error:
            terminal.update(status="technical_failed", engineering_gate=False, exit_code=2,
                            lock_release_error=f"{type(error).__name__}: {error}")
        if claimed:
            terminal["completed_at_utc"] = safety._utc_now()
            phase1a._atomic_json(args.output_root / "terminal.json", terminal)
            phase1a._atomic_json(args.output_root / "artifact_inventory.json", artifact_inventory(args.output_root))
            if terminal["exit_code"] == 0:
                core.audit_delivery(args.output_root, config)
    print(json.dumps(terminal, ensure_ascii=False, sort_keys=True, allow_nan=False), flush=True)
    return int(terminal["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
